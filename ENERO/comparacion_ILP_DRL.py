import ILP
import actorPPOmiddR as actor
import numpy as np
import tensorflow as tf
import gym
import gym_graph
import os
import sys
import time as tt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Añado la localización de la carpeta DECORE, la cual contiene los mmódulos de Python necesarios en este script: 'compresion_final' y 'apagado_aleatorio'
ruta_DECORE = os.path.abspath("../DECORE")
sys.path.append(ruta_DECORE)
from compresion_final import ModeloBaseComprimido
from apagado_aleatorio import ApagadoAleatorio

#Parámetros usados para generar el entorno
ENV_NAME = 'GraphEnv-v16'
SEED = 9
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100
take_critic_demands = True

#Variables usadas para inicializar los hiperparámetros y parámetros del actor base
hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

hparams = {
    'l2': 0.005,
    'dropout_rate': 0.1,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5,
}


#Ruta del directorio que contiene las carpetas con las diferentes topologías que pueden ser usadas
ruta_directorio_base = "../ENERO_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1"
#Nombre de las carpetas de las topologías
nombre_carpetas_topologias = ["NEW_BtAsiaPac","NEW_Garr199905","NEW_Goodnet"]
#Lista con la ruta de las carpetas de trabajo
ruta_carpetas_topologias = []
for nombre in nombre_carpetas_topologias:
    ruta_carpetas_topologias.append(ruta_directorio_base+"/"+nombre)
#Nombre de los archivos .graph de cada una de las topologías
nombre_grafos_topologias = ["BtAsiaPac","Garr199905","Goodnet"]

#Método encargado de generar un entorno de trabajo con las matrices de tráfico de validación
def generar_entorno_validacion(carpeta_topologia,grafo_topologia):
    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(carpeta_topologia+"/EVALUATE", grafo_topologia, EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands
    return env_eval


#Método que nos devuelve el actor inicial pero comprimido, preparado para su uso
def obtener_modelo_base_comprimido():
    actor_base = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
    actor_base.build()
    #Añado las máscaras
    actor_mascaras = ApagadoAleatorio(actor_base)
    actor_mascaras.build()
    #Restauro el estado del modelo con las mascaras que se tiene guardado en cierta carpeta
    checkpoint = tf.train.Checkpoint(model=actor_mascaras)
    checkpoint.restore("../DECORE/ApagadoAleatorio/apagado_40%").expect_partial()
    #Aplico la compresion final
    actor_comprimido = ModeloBaseComprimido(actor_mascaras,tiene_agentes=False)
    actor_comprimido.build()

    #Restauro el estado guardado tras el ajuste fino realizado
    checkpoint = tf.train.Checkpoint(model=actor_comprimido)
    #Buscamos la ruta del mejor modelo obtenido tras el ajuste
    log_compresion = "./Logs/expEneroComprimidoLogs.txt"
    checkpoints_compresion = "./modelsEneroComprimido/"
    with open(log_compresion) as fp:
            for line in reversed(list(fp)):
                arrayLine = line.split(":")
                if arrayLine[0]=='MAX REWD':
                    model_id = int(arrayLine[2].split(",")[0])
                    break
    print(f'La mejor versión encontrada para el modelo comprimido es {model_id}')
    checkpoint.restore(checkpoints_compresion+"ckpt_ACT-"+str(model_id)).expect_partial()

    return actor_comprimido


#Clase que contiene internamente como atributos el agente que va a ser empleado, y los métodos necesarios para poder hacer uso de él
class PPOMIDDROUTING_SP:
    def __init__(self):
        self.listQValues = None
        self.softMaxQValues = None

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None
        #self.K = env_training.K

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.actor = obtener_modelo_base_comprimido()

    def pred_action_node_distrib_sp(self, env, source, destination):
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
            tensor['num_edges'], training=False)
        self.listQValues = tf.reshape(r, (1, len(r)))
        self.softMaxQValues = tf.nn.softmax(self.listQValues)

        # Return action distribution
        return self.softMaxQValues.numpy()[0], tensor
    
    def get_graph_features(self, env, source, destination):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.bw_allocated_feature = env.edge_state[:,2]
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

'''
Método encargado de aplicar el agente sobre un determinado entorno (con una topología de red concreta) y una matriz de tráfico específica. Los parámetros a especificar son los siguientes:
    * tm_id: id de la matriz de tráfico que se quiere asociar
    * env_middRout_sp: entorno con la topología de red de trabajo
    * agent: instancia del agente que se quiere emplear para reenrutar las demandas críticas

'''
def aplicar_agente_DRL(tm_id, env_middRout_sp, agent):
    demand, source, destination = env_middRout_sp.reset(tm_id)

    initMaxUti = env_middRout_sp.edgeMaxUti[2]
    #Guardamos el uso del enlace de mayor congestión sobre el estado inicial de la red, antes de aplicar el agente
    OSPF_init = initMaxUti

    #Inicialmente, el middlepoint que usa cada demanda es el propio nodo destino (es decir, inicialmente no se realiza ningún desvío)
    best_routing = env_middRout_sp.sp_middlepoints_step.copy()

    #Tomamos la lista de demandas críticas seleccionadas para re-enrutar
    list_of_demands_to_change = env_middRout_sp.list_eligible_demands

    start = tt.time()
    #Guardamos el instante en el que comenzamos a aplicar el agente DRL
    time_start_DRL = start

    while 1:
        action_dist, tensor = agent.pred_action_node_distrib_sp(env_middRout_sp, source, destination)
        action = np.argmax(action_dist)
        #Avanzamos en el entorno llevando a cabo la accíon (desvío) seleccionado para dicha demanda 
        reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_middRout_sp.step(action, demand, source, destination)
        #Si mejora el uso del enlace de mayor congestión, guardamos el enrutamiento actual de todas las demandas críticas
        if maxLinkUti[2]<initMaxUti:
            initMaxUti = maxLinkUti[2] 
            best_routing = env_middRout_sp.sp_middlepoints_step.copy()
        if done:
            break

    end = tt.time()
    return initMaxUti, end-start, OSPF_init, best_routing, list_of_demands_to_change, time_start_DRL


#Elementos necesarios para trabajar con el LS seleccionado: HILL CLIMBING
ENV_LS = 'GraphEnv-v15'
EPISODE_LENGTH_MIDDROUT = 100

class HILL_CLIMBING:
    def __init__(self, env):
        self.num_actions = env.K 

    '''
    Método que devuelve la puntuación que obtendríamos en caso de aplicar una acción específica (desvío) para una cierta demanda crítica. Parámetros:
        * env: entorno con la topología de red de trabajo
        * source-destination: nodo emisor y receptor de dicha demanda
        * action: desvío a aplicar
    '''
    def get_value_sp(self, env, source, destination, action):
        #Obtenemos el nodo de desvío que nos indica dicha acción
        middlePointList = list(env.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]


        #Alojamos el ancho de banda de la demanda source:destination en los enlaces que definen el camino source-middlepoint y middlepoint-destination

        #Primero alojamos el tráfico en los enlaces del camino emisor-middlepoint
        env.allocate_to_destination_sp(source, middlePoint, source, destination)
        #Comprobamos si el desvío a aplicar es el propio nodo receptor
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination
            env.allocate_to_destination_sp(middlePoint, destination, source, destination)
            #Guardamos el desvío que sigue dicha demanda (siempre que el desvío sea un nodo intermedio diferente al propio nodo receptor)
            env.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        #Procedemos a calcular el uso del nuevo enlace de mayor uso en la red, tras aplicar dicho desvío 
        currentValue = -1000000 
        position = 0
        # Get the maximum loaded link and it's value after allocating to the corresponding middlepoint
        for i in env.graph:
            for j in env.graph[i]:
                link_capacity = env.links_bw[i][j]
                if env.edge_state[position][0]/link_capacity>currentValue:
                    currentValue = env.edge_state[position][0]/link_capacity
                position = position + 1
        
        #Ahora volvemos a desalojar el tráfico de la demanda enrutado siguiendo el camino emisor -> middlepoint -> receptor, para poder probar con otras acciones para esa misma demanda
        #Comprobamos si el desvío que se ha probado es el propio nodo receptor o no
        if str(source)+':'+str(destination) in env.sp_middlepoints:
            middlepoint = env.sp_middlepoints[str(source)+':'+str(destination)]
            env.decrease_links_utilization_sp(source, middlepoint, source, destination)
            env.decrease_links_utilization_sp(middlepoint, destination, source, destination)
            del env.sp_middlepoints[str(source)+':'+str(destination)] 
        else: # Remove the bandwidth allocated from the src to the destination
            env.decrease_links_utilization_sp(source, destination, source, destination)
        
        #Devolvemos el uso del enlace de mayor congestión, pero en negativo, de forma que las acciones que conduzcan a un uso más alto del enlace más congestionado tendrán una menor puntuación
        return -currentValue
    

    def explore_neighbourhood_DRL_sp(self, env):
        dem_iter = 0
        nextVal = -1000000
        next_state = None

        '''
        Procedemos a recorrer, para cada demanda seleccionada, cada uno de los posibles desvíos a usar, dada dicha demanda.

        Al final del for, y tras recorrer todas las acciones de todas las demandas de trabajo, obtendremos una tupla next_state : (action,source,dest)
            * Esta tupla almacena información sobre la demanda para la que se ha localizado el mejor uso del enlace de mayor uso aplicando cierta acción
        
        De esta forma, en cada iteración del algoritmo de búsqueda local --> se recorren todas las demandas y todas las acciones, y se aplica (al final de la iteración) un cambio local, escogiendo una demanda concreta, y aplicando un desvío concreto.
        '''

        #Recorremos las diferentes demandas seleccionadas
        for elem in env.list_eligible_demands:
            source = elem[0]
            dest = elem[1]
            #Obtenemos la puntuación asociada a las diferentes acciones (desvío) que se pueden aplicar dada dicha demanda, con el objetivo de aplicar la mejor
            for action in range(len(env.src_dst_k_middlepoints[str(source)+':'+str(dest)])):
                middlepoint = -1
                #Desalojamos el tráfico de la demanda de los enlaces que definen la ruta inicial
                if str(source)+':'+str(dest) in env.sp_middlepoints: 
                    middlepoint = env.sp_middlepoints[str(source)+':'+str(dest)] 
                    env.decrease_links_utilization_sp(source, middlepoint, source, dest)
                    env.decrease_links_utilization_sp(middlepoint, dest, source, dest)
                    del env.sp_middlepoints[str(source)+':'+str(dest)] 
                else: 
                    env.decrease_links_utilization_sp(source, dest, source, dest)

                #Obtenemos la puntuación de aplicar dicha acción (desvío) para dicha demanda
                evalState = self.get_value_sp(env, source, dest, action)
                #La puntuación que obtendríamos al aplicar dicha acción sobre dicha demanda es mejor que la que tenemos hasta el momento
                if evalState > nextVal:
                    nextVal = evalState
                    next_state = (action, source, dest) 
                
                #Volvemos a alojar el tráfico sobre la ruta asociada a la demanda inicialmente, antes de probar las diferentes acciones
                if middlepoint>=0:
                    # First we allocate until the middlepoint
                    env.allocate_to_destination_sp(source, middlepoint, source, dest)
                    # Then we allocate from the middlepoint to the destination
                    env.allocate_to_destination_sp(middlepoint, dest, source, dest)
                    # We store that the pair source,destination has a middlepoint
                    env.sp_middlepoints[str(source)+':'+str(dest)] = middlepoint
                else:
                    # Then we allocate from the middlepoint to the destination
                    env.allocate_to_destination_sp(source, dest, source, dest)
        return nextVal, next_state

'''
Método encargado de aplicar el algoritmo de búsqueda local HILL CLIMBING para mejorar la solución propuesta por el agente DRL
Será necesario especificar los siguientes parámetros:
    * tm_id: matriz de tráfico de trabajo
    * best_routing: el mejor enrutamiento conseguido usando el agente DRL, teniendo en cuenta las rutas asignadas a las demandas críticas a lo largo del período temporal durante el que se ha estado aplicando el agente
                    DRL sobre esa topología y matriz de tráfico concretas (no tiene por qué ser las rutas que siguen las demandas tras terminar de aplicar el agente sobre la topología/matriz)
    * list_of_demands_to_change: lista de demandas a reenrutar (demandas críticas seleccionadas)
    * time_start_DRL: instante inicial en el que se comenzó a aplicar el agente DRL sobre esa topología y matriz concretas
'''
def aplicar_hill_climbing(tm_id, best_routing, list_of_demands_to_change, nombre_grafo_topologia, ruta_carpeta_topologia):
    # Here we use sp in hill climbing to select the middlepoint and to evaluate
    env_hill_climb = gym.make(ENV_LS)
    env_hill_climb.seed(SEED)
    env_hill_climb.generate_environment(ruta_carpeta_topologia, nombre_grafo_topologia, EPISODE_LENGTH_MIDDROUT, NUM_ACTIONS, percentage_demands)

    #Uso del enlace de mayor uso actual para el estado del entorno, pero en NEGATIVO
    currentVal = env_hill_climb.reset_DRL_hill_sp(tm_id, best_routing, list_of_demands_to_change)
    hill_climb_agent = HILL_CLIMBING(env_hill_climb)
    start = tt.time()
    while 1:
        #Seleccionamos la demanda cuya ruta se va a cambiar y la acción (desvío) a aplicar
        #Obtenemos nextVal: uso del enlace de mayor uso aplicando la acción indicada en next_state a la demanda asociada a next_state, dicho uso en NEGATIVO
        nextVal, next_state = hill_climb_agent.explore_neighbourhood_DRL_sp(env_hill_climb)
        # If the difference between the two edges is super small but non-zero, we break (this is because of precision reasons)
        if nextVal<=currentVal or (abs((-1)*nextVal-(-1)*currentVal)<1e-4):
            break
        
        # Before we apply the new action, we need to remove the current allocation of the chosen demand
        action = next_state[0]
        source = next_state[1]
        dest = next_state[2]
       
        # Remove bandwidth allocated until the middlepoint and then from the middlepoint on
        if str(source)+':'+str(dest) in env_hill_climb.sp_middlepoints:
            middlepoint = env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)]
            env_hill_climb.decrease_links_utilization_sp(source, middlepoint, source, dest)
            env_hill_climb.decrease_links_utilization_sp(middlepoint, dest, source, dest)
            del env_hill_climb.sp_middlepoints[str(source)+':'+str(dest)] 
        # If there is no middlepoint assigned to the src,dst pair
        else:
            # Remove the bandwidth allocated from the src to the destination using sp
            env_hill_climb.decrease_links_utilization_sp(source, dest, source, dest)
        
        # We apply the new chosen action to the selected demand
        #Aplicamos la acción y demanda seleccionadas en esta iteración del algoritmo de búsqueda local
        currentVal = env_hill_climb.step_hill_sp(action, source, dest)
    end = tt.time()

    #Devolvemos el uso del enlace de mayor congestión, sobre el estado final de la red tras aplicar LS, y el tiempo de ejecución de dicho algoritmo
    return currentVal*(-1), end-start


if __name__ == "__main__":
    print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
    print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

    with tf.device('/device:GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a,b)
    print("Matrix multiplication result on GPU:\n", c.numpy())
    print("\nGPU seems to be working!")
    #Comienzo generando los entornos con las topologías de red de validación
    input("Presione ENTER para comenzar generando los entornos de validación...")
    entornos = []
    for indice, (ruta_top, grafo_top) in enumerate(zip(ruta_carpetas_topologias,nombre_grafos_topologias)):
        entornos.append(generar_entorno_validacion(ruta_top,grafo_top))
        print(f'Entorno para {entornos[indice].graph_topology_name} creado con éxito.')
    
    agenteDRL = PPOMIDDROUTING_SP()

    #nombre_modelos = ["DRL","DRL+LS","ILP"]
    datos = {"Modelo": [], "Topologia": [], "Uso": [], "Tiempo": []}

    input("\nPresione ENTER para comenzar el proceso de comparación...")

    for env in entornos:
        problema = ILP.ProblemaProgramacionLineal(env.dataset_folder_name)
        for tm_id in range(50):
            #Comenzamos aplicando el agente DRL
            uso_final, tiempo_DRL, uso_inicial , mejor_enrutamiento, lista_demandas_a_cambiar, instante_inicio_agenteDRL= aplicar_agente_DRL(tm_id,env,agenteDRL)
            for (clave,valor) in zip(datos,["DRL",env.graph_topology_name,uso_final,tiempo_DRL]):
                datos[clave].append(valor)
            #Posteriormente, aplicamos el algoritmo LS Hill Climbing
            uso_final, tiempo_LS = aplicar_hill_climbing(tm_id,mejor_enrutamiento,lista_demandas_a_cambiar,env.graph_topology_name,env.dataset_folder_name)
            for (clave,valor) in zip(datos,["DRL+LS",env.graph_topology_name,uso_final,tiempo_DRL+tiempo_LS]):
                datos[clave].append(valor)
            
            #Ahora resuelvo usando programación lineal
            uso, tiempo = problema.resolver_matriz_trafico(tm_id)
            for (clave,valor) in zip(datos,["ILP",env.graph_topology_name,uso,tiempo]):
                datos[clave].append(valor)
            input("Pulse ENTER para asignar la siguiente matriz de tráfico...")
    
    print("\nCreando los archivos CSV correspondientes...")
    #Una vez obtenidos los datos necesarios, creamos el directorio donde se almacenarán los resultados de esta comparación
    ruta_dir_DRL_ILP = "./Imagenes/DRLvsILP"
    if not os.path.exists(ruta_dir_DRL_ILP):
        os.makedirs(ruta_dir_DRL_ILP)
    

    df = pd.DataFrame(datos)
    df.to_csv(ruta_dir_DRL_ILP+"/resultados_DRL_vs_ILP.csv",index=False)

    media_por_modelo = df.groupby("Modelo")["Tiempo"].mean()
    media_por_modelo.to_csv(ruta_dir_DRL_ILP+"/tiempo_medio_DRL_vs_ILP.csv")

    print("\nCreando los boxplots correspondientes...")
    valores_graficar = ["Uso","Tiempo"]
    nombre_graficos = ["comparativa_usos.png","comparativa_tiempos.png"]
    etiquetas_ejeY = ["Uso del enlace de mayor uso", "Tiempo de ejecución"]
    for valor, nombre, etiqueta in zip(valores_graficar,nombre_graficos,etiquetas_ejeY):
        #Generamos los boxplots
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 5))

        # Crear boxplots: uno por topología, con un boxplot por modelo
        ax = sns.boxplot(x="Topologia", y=valor, hue="Modelo", data=df, palette="Set2")

        # Opcional: etiquetas y leyenda
        plt.title("Comparativa por Topologia")
        plt.ylabel(etiqueta)
        plt.xlabel("Topologia")
        plt.legend(title="Modelo", loc="upper right")

        plt.tight_layout()
        path_imagen_boxplots = ruta_dir_DRL_ILP + "/" + nombre
        plt.savefig(path_imagen_boxplots)
        plt.close()




    
    
    

