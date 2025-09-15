import tensorflow as tf
from tensorflow import keras
#import keras
import tensorflow_probability as tfp
import copy
import numpy as np
import gym
import gym_graph
import actor #Modelo del agente a comprimir
import os
from apagado_aleatorio import Mascara
from apagado_aleatorio import ApagadoAleatorio

#Porcentaje de neuronas que se quiere apagar sobre cada una de las capas con una máscara asociada
porcentaje_apagado = 10
porcentaje_apagado /= 100

NUM_EPOCAS = 20
EVALUATION_EPISODES = 20 #Número de episodios de validación, por cada topología

#Parámetros usados para generar el entorno
ENV_NAME = 'GraphEnv-v16'
SEED = 9
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100
take_critic_demands = True

#Definimos el nombre de las 3 topologías de trabajo (usadas para tomar el .graph y las matrices de tráfico)
name1 = "BtAsiaPac"
name2 = "Garr199905"
name3 = "Goodnet"
#Definimos el nombre de las carpetas asociadas a las 3 topologías de trabajo
dataset_name1 = "NEW_BtAsiaPac"
dataset_name2 = "NEW_Garr199905"
dataset_name3 = "NEW_Goodnet"
#Se construye la ruta relativa al directorio base con los datos, para cada una de las topologías previas
ruta_base = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"
dataset_folder_name1 = ruta_base + dataset_name1
dataset_folder_name2 = ruta_base + dataset_name2
dataset_folder_name3 = ruta_base + dataset_name3

#Método encargado de generar un entorno con una cierta topología para validación
def generar_entorno(ruta_topologia,nombre_topologia):
    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(ruta_topologia+"/EVALUATE", nombre_topologia , NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands
    return env_eval

#Parámetros necesarios para inicializar el modelo base
hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.01, 
    'T': 5,
}


path_checkpoints = "../ENERO/modelsEnero_3top_15_B_NEW" #Ruta donde se encuentran los diferentes estados guardados del actor tras el entrenamiento
log = "../ENERO/Logs/expEnero_3top_15_B_NEWLogs.txt" #Registro del proceso de entrenamiento del actor

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class Modelo_Compresion:
    def __init__(self,agente_base):
        self.actor = ApagadoAleatorio(agente_base)
        self.actor.build()

    def total_mascaras(self):
        return len(self.actor.mascaras)
    
    def obtener_acciones_mascara(self,ind_mascara):
        return (self.actor.mascaras[ind_mascara].A).numpy()
    
    def obtener_nombre_mascara(self,ind_mascara):
        return self.actor.mascaras[ind_mascara].name

    #Método encargado de apagar de forma aleatoria el porcentaje de neuronas del total de la capa con dicha máscara
    def apagar_neuronas(self,ind_mascara,porcentaje):
        #Comprobamos que sea un índice de máscara válido
        if 0 <= ind_mascara < self.total_mascaras():
            #print(f'Modificando máscara {ind_mascara}')
            self.actor.mascaras[ind_mascara].fijar_acciones(porcentaje)
        else:
            raise ValueError("El índice de máscara no es válido")

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

    def pred_action_distrib_sp(self, env, source, destination, training=False):
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
        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'], training=training)
        self.listQValues = tf.reshape(r, (1, len(r)))
        self.softMaxQValues = tf.nn.softmax(self.listQValues)
        
        #Devuelvo un array de Numpy de 1 sola dimensión : [_ , _, _, ...] contiene las probabilidades de aplicar cada posible desvío
        return self.softMaxQValues.numpy()[0]

if __name__ == "__main__":
    #Comencemos instanciado y recuperando el estado del agente base a comprimir
    agente_base = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
    agente_base.build()
    print("Se ha instanciado el actor base")
    #Buscamos la versión del mejor actor encontrado a lo largo del proceso de entrenamiento
    model_id = 0
    with open(log) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    print(f'La mejor versión encontrada es {model_id}')
    #Restablecemos el estado del mejor actor localizado
    checkpoint = tf.train.Checkpoint(model=agente_base, optimizer=tf.keras.optimizers.Adam())
    checkpoint.restore(path_checkpoints+"/ckpt_ACT-"+str(model_id)).expect_partial()
    print("Se ha restablecido el estado del mejor modelo localizado")
    #Insertamos las máscaras sobre el modelo base para iniciar el apagado aleatorio
    input("\nPresione ENTER para asignar las máscaras sobre las capas del modelo base")
    modelo = Modelo_Compresion(agente_base)
    num_mascaras = modelo.total_mascaras()
    print(f'Se han insertado un total de {num_mascaras}')
    print("Listo!!")

    checkpoint = tf.train.Checkpoint(model = modelo.actor)

    print("\nGenerando los entornos de validación")
    env_eval1 = generar_entorno(dataset_folder_name1,name1)
    env_eval2 = generar_entorno(dataset_folder_name2,name2)
    env_eval3 = generar_entorno(dataset_folder_name3, name3)
    print("Entornos creados con éxito")

    max_reward = -10000

    print("\nCreando el directorio de checkpoints")
    if not os.path.exists("./ApagadoAleatorio"):
        os.makedirs("./ApagadoAleatorio")
    ruta_checkpoint = "./ApagadoAleatorio/apagado_" + str(int(porcentaje_apagado*100)) + "%"
    print("Se ha creado con éxito el directorio donde realizar los checkpoints")

    ruta_log = "./Logs/apagado_aleatorio_" + str(int(porcentaje_apagado*100)) + "%.txt"
    print(f'\nCreando y abriendo el fichero log en la ruta: {ruta_log}')
    os.makedirs("./Logs",exist_ok=True) #En caso de que exista, no se crea y no lanza ningún error
    fileLogs = open(ruta_log,"w")

    for epoca in range(NUM_EPOCAS):

        print(f'\nComenzando episodio {epoca}...\n')
        #Al inicio de cada época, podo de forma aleatoria un 20% de las neuronas de cada capa con máscara asociada
        for ind in range(num_mascaras):
            modelo.apagar_neuronas(ind,porcentaje_apagado)
            print(f'El vector de acciones de la máscara será {modelo.obtener_acciones_mascara(ind)}')
        print("Porcentaje de neuronas apagadas sobre las capas con máscaras asociadas. Validando...")

        #Validamos usando las 20 primeras matrices de validación, para obtener la recompensa futura acumulada media
        max_link_uti = []
        rewards_test = []
        for env in [env_eval1,env_eval2,env_eval3]:
            for eps in range(EVALUATION_EPISODES):
                demand, source, destination = env.reset(eps)
                done = False 
                recompensa_acumulada = 0
                #Reenruto todas las demandas críticas para dicha matriz de tráfico y dicha topología
                while 1:
                    action_dist = modelo.pred_action_distrib_sp(env, source, destination)
                    action = np.argmax(action_dist)
                    reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env.step(action, demand, source, destination)
                    recompensa_acumulada += reward
                    if done:
                        break
                #Me guardo la recompensa futura acumulada obtenida, y el uso del enlace de mayor uso sobre el estado final en el que queda la red
                rewards_test.append(recompensa_acumulada)
                max_link_uti.append(maxLinkUti[2])

        #Una vez hayamos terminado de validar el agente, comprobamos si ha mejorado
        evalMeanReward = np.mean(rewards_test)
        if evalMeanReward > max_reward:
            #Ha mejorado
            max_reward = evalMeanReward
            #Guardamos el modelo
            checkpoint.write(ruta_checkpoint)
            print(f'Se ha obtenido un nuevo mejor modelo en la época {epoca}')
            fileLogs.write(f'MAX REWD,{np.max(max_link_uti)},{evalMeanReward},\n')
            #Escribimos también el vector de acciones que aplica cada máscara insertada
            for mask in range(num_mascaras):
                fileLogs.write(f'{modelo.obtener_nombre_mascara(mask)},{modelo.obtener_acciones_mascara(mask)},\n')
            fileLogs.flush()



