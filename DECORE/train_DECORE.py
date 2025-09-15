import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import actor
import tensorflow as tf
import numpy as np
from DECORE import DECORE
from DECORE import Agente
import gym
#Importamos el paquete que contiene el entorno personalizado
import gym_graph
import random
import sys

SEED = 9
MATRICES_VALIDACION = 20 #Nº de matrices que serán usadas en validación

EPOCAS_VALIDACION = 20 

#Hiperparámetros del modelo
hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.01, #Tasa de aprendizaje inicial
    'T': 5,
}

#Factor usado para indicar la importancia de la entropía de la política
ENTROPY_BETA = 0.01
#Cada cuántos episodios se reduce el factor de entropía
ENTROPY_STEP = 20

#Elementos necesarios para aplicar decaimiento de la tasa de aprendizaje
DECAY_STEPS = 20
DECAY_RATE = 0.96
#Método que se encarga de decaer la tasa de aprendizaje inicial un determinado nº de veces, pasando como parámetro 'epoca': índice de la época de entrenamiento que se va a iniciar
def decaer_tasa_aprendizaje(epoca):
    lr = configuracion_compresion['learning_rate']*(DECAY_RATE ** (epoca/DECAY_STEPS))
    #Aseguramos siempre un mínimo tamaño de paso en la dirección del gradiente
    if lr < 10e-5:
        lr = 10e-5
    return lr


'''
Configuración usada para el entrenamiento asociado al proceso de compresión:
    * TamañoLote : nº de trayectorias (de longitud 1) que se desean recorrer antes de actualizar los agentes
    * Episodios : nº total de episodios de entrenamiento
}
'''
configuracion_compresion = {
    'TamañoLote' : 256,
    'Episodios' : 150,
    'FactorRendimiento': 2,
    'FactorCompresion':1,
    'learning_rate': 0.01
}

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

#Inicialización de los parámetros del modelo
hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)

#Clase que contiene el actor entrenado, y una serie de métodos adicionales
class PPOAgent:
    def __init__(self):
        self.actor = actor.myModel(hparams,hidden_init_actor,kernel_init_actor)
        self.actor.build()
        #Optimizador que define cómo se modifican los parámetros entrenables del modelo a partir de uss gradientes : Adam
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])

'''
La siguiente clase contiene:
    * Recompensa que usaremos de base a la hora de obtener la recompensa en cuanto a mantenimiento del rendimiento del modelo comprimido
    * Modelo base modificado, es decir, con los agentes insertados
    * Optimizador usado para actualizar los parámetros de los agentes a partir de sus gradientes
'''
class MPNN_COMPRIMIDA:
    def __init__(self,modelo_base):
        self.decore = DECORE(modelo_base) #Modelo base + agentes
        self.decore.build()
        self.modelo_base = modelo_base
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=configuracion_compresion['learning_rate']) #Optimizador a usar

    def fijar_acciones(self,training=False):
        self.decore.fijar_acciones(training=training)
    
    #Método encargado de devolver las características del grafo con un desvío candidato aplicado
    def get_graph_features(self, env, source, destination):
        self.bw_allocated_feature = env.edge_state[:,2]
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature, #capacidad del enlace en relación a la capacidad del enlace de mayor capacidad
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        '''
        Aquí hacemos lo siguiente, ya sea para el uso, capacidad o ancho de banda del desvío:
            * Convertimos el tensor de una sola dimensión (vector fila) en un tensor de dos dimensiones (la 2º dimension con un solo valor) --> matriz con una sola columna
        '''
        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        #Concatenamos los vectores columna a o alrgo del eje de las columnas, obteniendo una matriz con tres columnas: uso,capacidad y ancho de banda del desvío (para reconocer los enlaces del desvío)
        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]]) #Creamos un tensor de la forma [ [0,0], [0,hparams[...]-3] ], lista con tantos elementos como dimensiones tenga el tensor al que le queremos meter el padding
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {
            'link_state': link_state,
            'first': sample['first'][0:sample['length']],
            'second': sample['second'][0:sample['length']], 
            'num_edges': sample['num_edges']
        }

        return inputs

    '''
    Méotodo que recibe el entorno y la demanda a reenrutar,  devolviendo las probabilidades de aplicar cada uno de los posibles
    desvíos que se pueden seguir a la hora de dirigir dicha demanda
    '''
    def pred_action_distrib_sp(self, env, source, destination, usar_modelo_base = False):
        # List of graph features that are used in the cummax() call
        list_k_features = list()

        # We get the K-middlepoints between source-destination
        middlePointList = env.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        itMidd = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while itMidd < len(middlePointList):
            env.mark_action_sp(source, middlePointList[itMidd], source, destination)
            # If we allocated to a middlepoint that is not the final destination
            if middlePointList[itMidd]!=destination:
                env.mark_action_sp(middlePointList[itMidd], destination, source, destination)

            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            env.edge_state[:,2] = 0
            itMidd = itMidd + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict qvalues for all graphs within tensors
        if usar_modelo_base:
            r = self.modelo_base(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'])
        else:
            r = self.decore(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'])

        self.listQValues = tf.reshape(r, (1, len(r)))
        self.softMaxQValues = tf.nn.softmax(self.listQValues)
        
        #Devuelvo un array de Numpy de 1 sola dimensión : [_ , _, _, ...] contiene las probabilidades de aplicar cada posible desvío
        return self.softMaxQValues.numpy()[0]
    

    def train_step(self,lista_entornos_compresion, lista_entornos_base):
        #Nº de trayectorias que se van a recorrer antes de actualizar
        trayectorias = configuracion_compresion['TamañoLote']
        #Repartimos las trayectorias entre las 3 topologías
        acciones_por_topologia = int(trayectorias/len(lista_entornos_compresion))
        
        #Lista que almacena el error total cometido en cada episodio/trayectoria recorrida
        errores = [] 

        training_tm_ids =set(range(100))
        acciones_aplicadas = 0
        seguir_topologia = True #Nos indica si podemos seguir trabajando con la misma topología

        with tf.GradientTape() as tape:
            for env_comp, env_base in zip(lista_entornos_compresion,lista_entornos_base):
                #Fijamos la matriz de tráfico con la que vamos a trabajar en dicha topología
                tm_id = random.sample(training_tm_ids,1)[0]
                #print(f'\n\nSe ha escogido como matriz de tráfico aquella de índice {tm_id}')
                while seguir_topologia:
                    demand, source, destination = env_comp.reset(tm_id)
                    demand_b, source_b, destination_b = env_base.reset(tm_id)
                    #Reenruto las demandas de tráfico críticas seleccionadas según dicha matriz de tráfico
                    while 1:
                        #Reenruto la demanda usando el modelo decore
                        self.fijar_acciones(training=True)
                        action_dist = self.pred_action_distrib_sp(env_comp, source, destination, usar_modelo_base=False)
                        action = np.random.choice(len(action_dist),p=action_dist)
                        # uso_enlace_max es una tupla (nodo origen del enlace, nodo destino del enlace, uso actual) asociado al nuevo enlace de mayor uso en la red
                        reward, done, _, demand, source, destination, uso_enlace_max, _, _ = env_comp.step(action,demand, source, destination)
                        uso_actual = float(uso_enlace_max[2])
                        acciones_aplicadas += 1

                        #Reenruto la misma demanda crítica pero usando el modelo base que tengo de referencia
                        action_dist= self.pred_action_distrib_sp(env_base, source_b, destination_b,usar_modelo_base=True)
                        action = np.random.choice(len(action_dist),p=action_dist)
                        _, _, _, demand_b, source_b, destination_b, uso_enlace_max_base, _, _ = env_base.step(action,demand_b, source_b, destination_b)
                        uso_base = float(uso_enlace_max_base[2])

                        #Para calcular la recompensa por rendimiento, puedo tomar la diferencia de usos de los enlaces de mayor uso, y multiplicarlo por un cierto factor 'alpha'
                        #print(f'Uso base : {uso_base} - Uso comprimido : {uso_actual}')
                        diff = uso_base - uso_actual
                        if abs(diff) < 1e-6:
                            #El rendimiento se ha mantenido. Asignamos una recompensa mínima de 0.1
                            recompensa_rendimiento = tf.constant(0.1,dtype=tf.float32)
                        else:
                            diff = tf.convert_to_tensor(diff,dtype=tf.float32)
                            recompensa_rendimiento  = tf.sigmoid(diff) * configuracion_compresion['FactorRendimiento']
                        #print(f'Obteniendo tipos: para diff --> {type(diff)}, {diff}, para recompensa_rendimiento --> {type(recompensa_rendimiento)},{recompensa_rendimiento}')
                        

                        #Trayectoria de longitud 1 recorrida --> Calculamos la pérdida asociada
                        errores_individuales = []
                        #print("\n------ Tras aplicar forward propagation para dicha demanda --------")
                        for agente in self.decore.agentes:
                            #Calculo la recompensa final de dicho agente
                            #recompensa_final = (agente.R) * recompensa_rendimiento
                            #tf.size devuelve por defecto un tensor escalar de dtype int32
                            #agente.R es un tensor escalar de tipo float32
                            num_params_agente = tf.size(agente.params,tf.float32)
                            #print(f'Los parametros del agente son {num_params_agente}, y la compresion es {agente.R}')
                            #La recompensa por compresión será el porcentaje de compresión conseguido
                            recompensa_compresion = (agente.R / num_params_agente) * configuracion_compresion['FactorCompresion']
                            #print(f'La recompensa por compresión es {recompensa_compresion}. La recompensa por rendimiento es {recompensa_rendimiento}')
                            #recompensa_final = recompensa_compresion * recompensa_rendimiento
                            recompensa_final = recompensa_compresion + recompensa_rendimiento
                            recompensa_final = tf.stop_gradient(recompensa_final)
                            #print(f'La recompensa final es {recompensa_final}')
                            #Calculo el log de la política que sigue el agente
                            log_probs =  agente.A * tf.math.log(agente.Pi + 1e-8) + (1 - agente.A) * tf.math.log(1 - agente.Pi + 1e-8)
                            #print(f'Acciones aplicadas por el agente: {agente.A}')
                            log_prob_sum = tf.math.reduce_sum(log_probs) # tf.reduce_sum con axis=None nos devuelve un tensor escalar (tf.float32), sin dimensiones : ()
                            #print(f'El log de la política del agente es {log_prob_sum}')
                            #Cálculo de la ENTROPÍA de la política que sigue el agente para cada neurona de la capa
                            entropia_por_neurona = -agente.Pi * tf.math.log(agente.Pi + 1e-8) - (1 - agente.Pi) * tf.math.log(1 - agente.Pi + 1e-8)
                            entropia_media = tf.reduce_mean(entropia_por_neurona)
                            #Pérdida de dicho agente
                            errores_individuales.append(-log_prob_sum*recompensa_final - entropia_media*ENTROPY_BETA)  #Lista con tensores escalares de tipo tf.float32
                        #Tras calcular el error cometido por cada agente, lo sumo para obtener el erorr total de dicha trayectoria
                        errores.append(tf.math.reduce_sum(errores_individuales)) #Esto almacena un tensor escalar 
                        
                        if acciones_aplicadas == acciones_por_topologia:
                            #Cambiamos la topología de trabajo
                            #print("Hemos terminado de trabajar con esta topología")
                            seguir_topologia = False
                            break

                        if done:
                            break #He terminado de reenrutar todas las demandas críticas seleccionadas

                #Hemos salido del while principal --> Pasamos a la siguiente topología
                seguir_topologia = True
                acciones_aplicadas = 0

            #Hemos recorrido las trayectorias deseadaas. Calculamos error medio
            #print(tf.convert_to_tensor(errores))
            error_medio = tf.math.reduce_mean(errores) #Recibe errores : lista de tensores escalares.Internamente, en el reduce_mean, se usa 'convert_to_tensor' transformando dicha lista de tensores escalares en un tensor unidimensional
            #print(f'\nNº de trayectorias recorridas es {len(errores)}')
            #print(f'\nEl error medio cometido es de {error_medio}')

        #Cálculo de los gradientes de la función de pérdida sobre los parámetros de los agentes
        all_params = [agent.params for agent in self.decore.agentes]
        #print("\n Calculando los gradientes sobre los parámetros de los agentes insertados")
        gradientes = tape.gradient(error_medio, sources = all_params)
        #Aplicación de los gradientes usando el optimizador deseado
        #print("Aplicando gradientes...")
        self.optimizer.apply_gradients(zip(gradientes,all_params))

        return error_medio


    #Método usado para validar el modelo tras entrenarlo/actualizarlo
    def val_step(self,lista_entornos):
        uso_enlace_mayor_uso = []
        recompensa_futura_acumulada = []

        for env in lista_entornos:
            for tm_id in range(MATRICES_VALIDACION):
                demand, source, destination = env.reset(tm_id)
                done = False
                recompensa_agregada = 0 #Guardamos la recompensa futura acumulada obtenida tras aplicar el agente sobre dicha matriz de dicha topología (suma de las recompensas obtenidas tras aplicar cada acción)
                #Reenrutamos las demandas críticas seleccionadas para dicha matriz de tráfico
                while 1:
                    #Aquí no hace falta fijar las acciones que aplica cada agente porque ya venimos de entrenar dichas políticas
                    action_dist = self.pred_action_distrib_sp(env, source, destination, usar_modelo_base=False)
                    action = np.argmax(action_dist)
                    # uso_enlace_max es una tupla (nodo origen del enlace, nodo destino del enlace, uso actual) asociado al nuevo enlace de mayor uso en la red
                    reward, done, _, demand, source, destination, uso_enlace_max, _, _ = env.step(action,demand, source, destination)
                    recompensa_agregada += reward
                    if done:
                        break

                #Tras reenrutar todas las demandas críticas, guardamos el uso del enlace de mayor uso sobre el estado final de la red
                uso_enlace_mayor_uso.append(uso_enlace_max[2])
                #También guardamos la recompensa futura acumulada obtenida tras aplicar el agente en dicha matriz de tráfico
                recompensa_futura_acumulada.append(recompensa_agregada)
            
        #Tras validar usando las diferentes TM's en las diferentes topologías, tomamos el valor de la métrica en el peor de los casos
        valor_metrica = np.max(uso_enlace_mayor_uso)
        #Recompensa futura acumulada media
        recompensa_media = np.mean(recompensa_futura_acumulada)

        return valor_metrica, recompensa_media

#Método encargado de generar un entorno de trabajo, pasando el directorio de la topología a usar, su nombre, y si queremos usar las matrices de entrenamiento o validación
def generar_entorno(ruta_topologia,nombre_topologia,entrenamiento=True):
    entorno = gym.make(ENV_NAME)
    entorno.seed(SEED)
    if entrenamiento:
        #Se quieren usar las matrices de tráfico reservadas para entrenamiento
        ruta_topologia = ruta_topologia + "/TRAIN"
    else:
        #Se quieren usar las matrices de validación
        ruta_topologia = ruta_topologia + "/EVALUATE"

    entorno.generate_environment(ruta_topologia, nombre_topologia , NUM_ACTIONS, percentage_demands)
    entorno.top_K_critical_demands = take_critic_demands
    return entorno

'''
Para ejecutar este script será necesario pasar dos parámetros:
    * -d -> Ruta del fichero que contiene el registro del proceso de entrenamiento del modelo base a comprimir
    * -c --> Ruta del directorio de checkpoints realizados sobre el modelo base
En este caso, ejecutar el siguiente comando:
python train_DECORE.py -d ../ENERO/Logs/expEnero_3top_15_B_NEWLogs.txt -c ../ENERO/modelsEnero_3top_15_B_NEW
'''
#Comenzamos buscando en el archivo de Logs la versión del modelo para el que se ha obtenido una mejor recompensa tra validación
if __name__ == "__main__":
    #Definimos los argumentos que se pueden pasar a la hora de ejecutar este script, incluido su parseo
    parser = argparse.ArgumentParser(description="Parsear argumentos para entrenamiento")
    parser.add_argument('-d', help='Archivo de logs',type=str, required=True,nargs='+')
    parser.add_argument('-c',help="Directorio de checkpoints",type=str,required=True,nargs='+')
    #Tomamos los argumentos pasados al iniciar la ejecución
    args = parser.parse_args()

    path_logs = args.d[0]
    path_checkpoints = args.c[0]

    print("Archivo de logs es " + path_logs + " - Directorio de checkpoints es " + path_checkpoints)

    max_recompensa_localizada = False
    model_id = -1
    with open(path_logs) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0] == 'MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                print(f'El mejor modelo encontrado tras el entrenamiento se corresponde con la versión {model_id}')
                break

    #Restauramos el estado del actor usando la versión localizada previamente
    AgenteDRL = PPOAgent()
    saved_model = tf.train.Checkpoint(model=AgenteDRL.actor, optimizer=AgenteDRL.optimizer)
    saved_model.restore(path_checkpoints+"/ckpt_ACT-"+str(model_id))
    print("Se ha restaurado el estado del modelo...")

    '''
    Pasamos como parámetros:
        * Modelo entrenado que se desea comprimir
        * Máxima recompensa obtenida tras el proceso de entrenamiento de dicho modelo
    '''
    input("\nPresione ENTER para generar el modelo DECORE")
    nuevo_modelo = MPNN_COMPRIMIDA(AgenteDRL.actor)

    input("\nPresione ENTER para generar los entorno de trabajo...")
    #Definimos el nombre de las 3 topologías usadas para el entrenamiento de los agentes insertados
    dataset_name1 = "NEW_BtAsiaPac"
    dataset_name2 = "NEW_Garr199905"
    dataset_name3 = "NEW_Goodnet"
    #Se construye la ruta relativa al directorio base con los datos, para cada una de las topologías previas
    ruta_base = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"
    dataset_folder_name1 = ruta_base + dataset_name1
    dataset_folder_name2 = ruta_base + dataset_name2
    dataset_folder_name3 = ruta_base + dataset_name3

    #Entrenamiento del modelo a comprimir (tras insertar agentes)
    #Comenzamos instanciando los entornos de trabajo: un entorno para cada topología, asociando a cada entorno las matrices de tráfico de entrenamiento
    ENV_NAME = "GraphEnv-v16"
    NUM_ACTIONS = 100 
    take_critic_demands = True #Para indicar que se desea seleccionar las demandas críticas
    percentage_demands = 15 #Porcentaje del total de demandas críticas que serán seleccionadas para reenrutar

    # Entornos de entrenamiento (modelo comprimido)
    env_training1 = generar_entorno(dataset_folder_name1, "BtAsiaPac", entrenamiento=True)
    env_training2 = generar_entorno(dataset_folder_name2, "Garr199905", entrenamiento=True)
    env_training3 = generar_entorno(dataset_folder_name3, "Goodnet", entrenamiento=True)

    # Entornos base (modelo no comprimido)
    env_trainining_base1 = generar_entorno(dataset_folder_name1, "BtAsiaPac", entrenamiento=True)
    env_training_base2 = generar_entorno(dataset_folder_name2, "Garr199905", entrenamiento=True)
    env_training_base3 = generar_entorno(dataset_folder_name3, "Goodnet", entrenamiento=True)

    print("Entornos de entrenamiento creados e inicializados con éxito")

    # Entornos de evaluación
    env_eval1 = generar_entorno(dataset_folder_name1, "BtAsiaPac", entrenamiento=False)
    env_eval2 = generar_entorno(dataset_folder_name2, "Garr199905", entrenamiento=False)
    env_eval3 = generar_entorno(dataset_folder_name3, "Goodnet", entrenamiento=False)

    print("Entornos de validación creados e inicializados con éxito")

    #Configuramos el archivo que almacena un registro del proceso de entrenamiento
    ruta_log = "./Logs/decore_2.2.txt"

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    fileLogs = open(ruta_log,"w")
    print("\nSe ha creado y abierto sin problemas el archivo de logs")
    
    #Configuramos el objeto Checkpoint que nos permitirá guardar el estado del modelo, cada vez que se obtenga un nuevo mejor modelo
    max_reward = -1000
    checkpoint = tf.train.Checkpoint(model=nuevo_modelo.decore) #Traqueamos el estado de la red (modelo base con agentes insertados)
    
    if not os.path.exists("./DECORE"):
        os.makedirs("./DECORE")
    ruta_checkpoint_agente = "./DECORE/decore_2.2"
    print("Se ha creado con éxito el directorio donde realizar los checkpoints")
    

    #Entrenamos durante un determinado número de episodios
    print("\nComenzando proceso de entrenamiento...")
    
    for episodio in range(configuracion_compresion['Episodios']):
        print(f'\nComenzando episodio {episodio}...')
        
        if episodio%DECAY_STEPS==0 and episodio>0:
            #Cada 20 episodios, la tasa de aprendizaje se reduce al 96% del último valor asociado
            nuevo_modelo.optimizer.learning_rate.assign(decaer_tasa_aprendizaje(episodio))
            print(f'Al inicio del episodio {episodio} la tasa de aprendizaje ha decaído a {nuevo_modelo.optimizer.learning_rate.numpy()}')
        if episodio%ENTROPY_STEP==0 and episodio>0:
            #Reducimos el grado de implicación de la entropía de la política actual sobre el cálculo de la pérdida total
            ENTROPY_BETA=ENTROPY_BETA*0.1
            print(f'Al inicio del episodio {episodio} se ha reducido el grado de la entropía a un 10% : {ENTROPY_BETA}')

        #Entrenamos
        perdida_train = nuevo_modelo.train_step([env_training1, env_training2, env_training3], [env_trainining_base1, env_training_base2, env_training_base3]) #Obtengo la perdida como un tensor escalar de tipo float
        fileLogs.write(f'{perdida_train.numpy()},\n')
        #Volcamos los datos del buffer en memoria a disco(fichero de logs) y vacíamos el buffer
        fileLogs.flush()
    
    '''
    Tras finalizar el entrenamiento, pasamos con la ejecución de la segunda fase, tal que para cada época:
        * Selecciono las acciones que aplican los agentes, teniendo en cuenta el valor final de sus parámetros tras finalizar el entrenamiento, y usando una política estocástica
        * Valido usando las mismas topologías empleadas en el entrenamiento previo, y usando X matrices de validación
        * Si la recompensa futura acumulada media supera a la mejor actual, entonces me guardo el estado del modelo, con especial interés en el vector de acciones que han decidido aplicar los agentes
    '''
    max_recompensa = -100000
    fileLogs.write("VALIDACION,\n")
    for epoca in range(EPOCAS_VALIDACION):
        print(f'\nComenzado la época {epoca} de validación...')
        nuevo_modelo.fijar_acciones(training=True) #Los agentes usarán las políticas estocásticas, pero teniendo en cuenta la prob final de preservar cada neurona
        #Valido manteniendo fijas las acciones que aplican cada uno de los agentes insertados
        uso, recompensa = nuevo_modelo.val_step([env_eval1,env_eval2,env_eval3])
        if recompensa > max_recompensa:
            max_recompensa = recompensa
            #Me guardo en el fichero los valores obtenidos tras la validación, y el vector de acciones que aplica cada agente
            fileLogs.write(f'MAX REWD,{uso},{recompensa},\n')
            for agente in nuevo_modelo.decore.agentes:
                fileLogs.write(f'{agente.name},{agente.A.numpy()},\n')
            fileLogs.flush()
            #Guardo el estado del modelo
            checkpoint.write(ruta_checkpoint_agente)
