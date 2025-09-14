import numpy as np
import gym
import gc
import os
import gym_graph
import random
import criticPPO as critic
import actorPPOmiddR as actor
import actorPPO_v2 as newactor
import criticPPO_v2 as newcritic
import tensorflow as tf
from collections import deque
#import time as tt
import argparse
import pickle
import heapq
from keras import backend as K
import csv
import math

# Use BtAsiaPac, EliBackbone and Goodnet for training

'''
Este script se ejecuta al inicio de cada episodio de entrenamiento, realizando un determinado nº de iteraciones por episodio
'''

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' Esto supondría ignorar la GPU para usar únicamente la CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# In this experiment we learn how to pick the best action(middlepoint) by marking for each middlepoint
# the action in the topology edges. Rewards are given per time-step.
# We also remove the SP paths that can create a loop with the source node!

ENV_NAME = 'GraphEnv-v16'

# Indicates how many time-steps has an episode
EPISODE_LENGTH = 100 # We are not using it now
SEED = 9
MINI_BATCH_SIZE = 55
experiment_letter = "_B_NEW"
take_critic_demands = True # True if we want to take the demands from the most critical links, True if we want to take the largest
#Porcentaje del total de demandas críticas que se van a considerar
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

EVALUATION_EPISODES = 20 # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM
PPO_EPOCHS = 8
num_samples_top1 = int(np.ceil(percentage_demands*380))*5
num_samples_top2 = int(np.ceil(percentage_demands*506))*4
num_samples_top3 = int(np.ceil(percentage_demands*272))*6

BUFF_SIZE = num_samples_top1+num_samples_top2+num_samples_top3 # Experience buffer size. Careful to don't have more samples from one TM!

'''
Este script se ejecuta para cada episodio (conformado a su vez por 'episode_iters = 20' iteraciones, que son las que se realiza justamente en este script). 
Se usa una tasa de aprendizaje que va decayendo a lo largo de las épocas de entrenamiento. En este caso, al inicio de cada época, decae dicha tasa.
'''
# The DECAY_STEPS must be a multiple of args.e (episode_iters)
DECAY_STEPS = 60 # The second value is to indicate every how many PPO EPISODES we decay the lr
DECAY_RATE = 0.96

CRITIC_DISCOUNT = 0.8

# if agent struggles to explore the environment, increase BETA
# if the agent instead is very random in its actions, not allowing it to take good decisions, you should lower it
ENTROPY_BETA = 0.01
ENTROPY_STEP = 60 #Cada tres épocas de entrenamiento, se usa esta variable para reducir la libertad de exploración (y que el agente DRL se vaya centrando más en buscar en un espacio concreto)

clipping_val = 0.1
gamma = 0.99
lmbda = 0.95

max_grad_norm = 0.5

'''
Especificamos cuál es el directorio donde se va a guardar cada versión del agente (actor+crítico)
    /modelsSP_3top_15_B_NEW/    
            * 3: las 3 topologías usadas para train/val
            * 15: se seleccionan el 15% del total de demandas críticas (aquellas que pasan por los 5 enlaces de mayor uso)
'''
#differentiation_str = "Enero_3top_"+str_perctg_demands+experiment_letter
differentiation_str = "Probando"
checkpoint_dir = "./models"+differentiation_str

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)

#train_dir = "./TensorBoard/"+differentiation_str
#summary_writer = tf.summary.create_file_writer(train_dir)
global_step = 0
NUM_ACTIONS = 100 # For now we have dynamic action space. This means that we consider all nodes as actions but removing the repeated paths

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)

hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5, #Nº de iteraciones que se realiza en la fase de envío de mensajes-actualización
}

'''
Función que recibe una lista de diccionarios (cada uno con la info de un cierto desvío candidato), y el 'extractor' o función encargada de extraer el valor de una cierta
clave para cada uno de los diccionarios. En este caso, dicho extractor extrae los enlaces mensajeros que contiene cada diccionario

Se usa 'reduce_max' que toma el máximo sobre las dimensiones especificadas

Devuelve una lista con tantos elementos como desvíos candidatos se puedan realizar para la demanda de trabajo, tal que
cada elemento refleja la suma de:
    * Posición del enlace mensajero de mayor posición, sobre el grafo con el desvío actual
    * Posición del enlace mensajero de mayor posición, sobre cada uno de los grafos asociados a los desvíos previos

(suma acumulativa)
'''
def old_cummax(alist, extractor):
    with tf.name_scope('cummax'): #Define un espacio de trabajo

        #Lista que almacena, para cada desvío candidato, la posición del enlace mensajero de mayor posición (sumándole 1, es decir, 1-based) sobre el grafo con dicho desvío
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        #Se realiza la suma acumulada de los elementos de maxes
        #Primer elemento
        cummaxes = [tf.zeros_like(maxes[0])]
        #Resto de elementos
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

#Nos permite, al inicio de cada época de entrenamiento, decaer la tasa de aprendizaje usando un decaimiento exponencial (más acelerado cuanto más épocas llevaos)
def decayed_learning_rate(step):
    lr = hparams['learning_rate']*(DECAY_RATE ** (step / DECAY_STEPS))
    #Para asegurar siempre un mínimo aprendizaje, se limita la tasa mínima
    if lr<10e-5:
        lr = 10e-5
    return lr

class PPOActorCritic:
    def __init__(self):
        self.memory = deque(maxlen=BUFF_SIZE)
        #índices de los estados que conforman la trayectoria que recorre el agente para recoger experiencia
        self.inds = np.arange(BUFF_SIZE)
        self.listQValues = None
        self.softMaxQValues = None
        self.global_step = global_step

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        self.critic = critic.myModel(hparams, hidden_init_critic, kernel_init_critic)
        self.critic.build()
    
    '''
    Este método de forma interna se encargará de diseñar los grafos resultantes de aplicar cada uno de los posibles desvíos que se pueden seguir
    Devolverá:
        * Probabilidad de llevar a cabo cada desvío (acción)
        * Info del estado en el que se quedaría el grafo tras aplicar cada desvío
    '''
    def pred_action_distrib_sp(self, env, source, destination):
        # List of graph features that are used in the cummax() call
        '''
        Esta lista almacena tantos elementos como desvíos podamos realizar, tal que para cada desvío posible:
            * Se ajusta el grafo (de forma abstracta) suponiendo que se realiza ese desvío a la hora de enviar los datos de origen a destino
            * Se obtiene un diccionario que contiene la información del grafo previo
            * Se almacena dicho diccionario (objeto) en list_k_features
        '''
        list_k_features = list()

        # We get the K-middlepoints between source-destination
        #Esto es, tomamos una lista de los nodos intermedios por los que podemos pasar para realizar el desvío
        middlePointList = env.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        itMidd = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        #Ahora calculamos el estado en el que quedaría el grafo si aplicásemos cada uno de los middlepoints/desvíos posibles
        while itMidd < len(middlePointList):
            #Marcamos el camino a seguir en el caso de realizar el desvío al nodo de índice 'itMidd'
            env.mark_action_sp(source, middlePointList[itMidd], source, destination)
            # If we allocated to a middlepoint that is not the final destination
            if middlePointList[itMidd]!=destination:
                env.mark_action_sp(middlePointList[itMidd], destination, source, destination)

            #Obtenemos las características del grafo con el desvío aplicado, indicando la demanda origen-destino (aunque realmente estos dos últimos params no se usan), en forma de diccionario
            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            #Esto contiene, para todos aquellos enlaces del camino origen-middlepoint-destino, el uso considerando únicamente la demanda origen-destino
            env.edge_state[:,2] = 0
            itMidd = itMidd + 1

        vs = [v for v in list_k_features] #lista que almacena las características del grafo (en forma de diccionario) para cada desvío candidato (dado dicha demanda de trabajo)

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.

        '''
        graph_ids: nos permite generar una lista con tantos elementos como desvíos candidatos se puedan aplicar, tal que para cada elemento(desvío candidato) se almacena:
            tensor unidimensional (vector) con tantas componentes como enlaces tenga el grafo, usando como valor constante el índice de dicho desvío

        EJEMPLO: si tuviesemos 3 desvíos candidatos a aplicar dada una cierta demanda de trabajo, y el grafo tuviese 5 enlaces, 'graph_ids' sería así:
        [ [0,0,0,0,0], [1,1,1,1,1], [2,2,2,2,2] ]
        '''
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]

        #Se calcula la suma acumulativa de las posiciones de los mensajeros de mayor posición, de los grafos que modelan los diferentes desvíos
        first_offset = old_cummax(vs, lambda v: v['first'])
        #Lo mismo pero trabajando ahora sobre los receptores
        second_offset = old_cummax(vs, lambda v: v['second'])

        '''
        Se modela el supergrafo que contiene los diferentes grafos que se van a pasar como entrada a la GNN
        Se modela mediante un diccionario:

            * graph_id: tensor unidimensional resultado de concatenar los tensores unidimensionales de los id de los desvíos
                - Ejemplo: si fuesen 3 desvíos, y un grafo de 2 enlaces: [0,0,1,1,2,2]

            * linkstate: concatena las matrices de estados ocultos (tensores con 2 dimensiones (nºenlaces x link_state_dim) ) a lo largo del eje de las filas

            * first: concatena los enlaces mensajeros de cada grafo con desvío candidato aplicado, a lo largo de las filas. Se usa 'old_cummax' para realizar los desplazamientos
            La razón de esto es muy básica, simplemente porque no vamos a trabajar sobre cada matriz de vectores de estados ocultos de cada grafo con un desvío candidato aplicado,
            sino que se usa una sola matriz fruto de concatenar las matrices de vectores de estados asociadas a los diferentes desvíos candidatos.
            Por tanto, debemos de realizar el desplazamiento correspondiente para trabajar sobre el trozo de la matriz global que refleja la matriz de vectores de estados ocultos
            del grafo con el desvío candidato correspondiente.
            Es decir, si quisiésemos obtener el vector de estados oculto del mensajero situado en la posición 0 del grafo con el segundo desvío candidato aplicado,
            podríamos:
                * Tomar directamente la primera fila del 'link_state' de dicho grafo candidato
                * En caso de trabajar con el 'link_state' global (concatena las matrices de vectores de estados de todos los desvíos candidatos), tendríamos primero
                que recorrer las filas que reflejan el 'link_state' del grafo con el primer desvío, y luego, ya sí tomar la primera fila tras dicho desplazamiento.
            Por esto, se actualiza el 'first' de cada grafo con desvío candidato, realizando el desplazamiento correspondiente. Lo mismo para 'second'
            
            * second:: igual pero para enlaces receptores

            * num_edges: nº total de enlaces, tomando el nº de enlaces del grafo asociado a cada desvío candidato
        '''
        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0), #Se concatenan tensores unidimensionales a lo largo de esa única dimensión
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]), #math.add_n nos permite sumar rápidamente tensores
            }
        )        

        # Predict qvalues for all graphs within tensors
        #Evaluamos el actor sobre los diferentes grafos asociados a cada desvío, obteniendo un valor para cada desvío posible
        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'], training=False)
        self.listQValues = tf.reshape(r, (1, len(r)))
        #Aplicamos softmax para obtener las probabilidades de llevar a cabo cada uno de los posibles desvíos al envíar los datos de origen-destino
        self.softMaxQValues = tf.nn.softmax(self.listQValues)
        
        # Return action distribution
        #Devolvemos las distribuciones de probabilidad de las acciones y el tensor con la info de todos los desvíos posibles a realizar
        return self.softMaxQValues.numpy()[0], tensor
    
    '''
    Este método nos permite obtener las características del grafo en el estado actual del entorno

    Se usará sobre todo tras marcar de forma abstracta un desvío candidato a aplicar (= marcar el camino que seguiríamos en caso de realizar ese desvío (edge_state[2]), sin tocar nada mas)-
    Nota: no se recalcula el uso de los enlaces aplicando esa nueva ruta. Simplemente se devuelve el uso inicial de cada enlace de la topología, y aparte, edge_state[2] para saber qué enlaces determinan el camino
    a seguir para realizar un cierto desvío
    '''
    def get_graph_features(self, env, source, destination):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        #uso del enlace considerando únicamente la demanda sobre la que estamos trabajando en un estado concreto del entorno
        self.bw_allocated_feature = env.edge_state[:,2]
        #tráfico que pasa por dicho enlace
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

        #Definimos el padding mediante un tensor de dimensiones 2x2 de la forma 
        # (  0       0          )  --> estos dos valores reflejan los 0 a añadir antes-después de los valores de la primera dimension de hiddenstates (filas o nº de enlaces)
        # (  0   hparams[...]-3 ) ----> igual pero para la dimensión que contiene las características de estudio (uso,capacidad,ancho de banda)
        
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]]) #Creamos un tensor de la forma [ [0,0], [0,hparams[...]-3] ], lista con tantos elementos como dimensiones tenga el tensor al que le queremos meter el padding
        #Extendemos el vector de estados oculto de cada enlace para que tenga dimensión 'link_state_dim', que es justamente la dimensión que debe tener la entrada al modelo
        #Esto es, un tensor que se asemeja a una matriz con tantas filas como enlaces, y para cada fila(enlace), un vector de 'link_state_dim' componentes
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        '''
        Devolvemos un diccionario que contiene:
            * link_state: tensor (matriz) con el vector de estados oculto de cada enlace, adaptado a la dimensión de entrada del modelo 
            * first y second: mensajeros y  receptores para la fase de envío de mensajes
            * nº enlaces del grafo
        '''
        inputs = {
            'link_state': link_state,
            'first': sample['first'][0:sample['length']],
            'second': sample['second'][0:sample['length']], 
            'num_edges': sample['num_edges']
        }

        return inputs

    #También nos permite obtener las características del estado actual del grafo
    #A diferencia del método previo, no tenemos en cuenta el camino marcado.
    #Se usará sobre todo para guardar el estado actual del grafo (sin tener en cuenta los posibles desvíos a realizar para la demanda de trabajo), 
    #antes de aplicar finalmente el desvío seleccionado, lo que supondría perdernos el estado inicial del grafo.
    def critic_get_graph_features(self, env):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        #Tomamos el ancho de banda que lleva actualmente cada enlace
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            #guardamos el uso (ancho de banda que aloja / capacidad maxima) de cada enlace
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    

    '''
    En resumen, la diferencia entre los dos métodos previos es la siguiente:
        * get_graph_features: estado inicial del grafo + camino a seguir para un cierto desvío
        * critical_get_graph_features: estado inicial del grafo (sin considerar ninguno de los desvíos candidatos)
    '''
    
    def _write_tf_summary(self, actor_loss, critic_loss, final_entropy):
        with summary_writer.as_default():
            tf.summary.scalar(name="actor_loss", data=actor_loss, step=self.global_step)
            tf.summary.scalar(name="critic_loss", data=critic_loss, step=self.global_step)  
            tf.summary.scalar(name="entropy", data=-final_entropy, step=self.global_step)                      

            tf.summary.histogram(name='ACTOR/FirstLayer/kernel:0', data=self.actor.variables[0], step=self.global_step)
            tf.summary.histogram(name='ACTOR/FirstLayer/bias:0', data=self.actor.variables[1], step=self.global_step)
            tf.summary.histogram(name='ACTOR/kernel:0', data=self.actor.variables[2], step=self.global_step)
            tf.summary.histogram(name='ACTOR/recurrent_kernel:0', data=self.actor.variables[3], step=self.global_step)
            tf.summary.histogram(name='ACTOR/bias:0', data=self.actor.variables[4], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/kernel:0', data=self.actor.variables[5], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/bias:0', data=self.actor.variables[6], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/kernel:0', data=self.actor.variables[7], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/bias:0', data=self.actor.variables[8], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/kernel:0', data=self.actor.variables[9], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/bias:0', data=self.actor.variables[10], step=self.global_step)
            
            tf.summary.histogram(name='CRITIC/FirstLayer/kernel:0', data=self.critic.variables[0], step=self.global_step)
            tf.summary.histogram(name='CRITIC/FirstLayer/bias:0', data=self.critic.variables[1], step=self.global_step)
            tf.summary.histogram(name='CRITIC/kernel:0', data=self.critic.variables[2], step=self.global_step)
            tf.summary.histogram(name='CRITIC/recurrent_kernel:0', data=self.critic.variables[3], step=self.global_step)
            tf.summary.histogram(name='CRITIC/bias:0', data=self.critic.variables[4], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/kernel:0', data=self.critic.variables[5], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/bias:0', data=self.critic.variables[6], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/kernel:0', data=self.critic.variables[7], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/bias:0', data=self.critic.variables[8], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/kernel:0', data=self.critic.variables[9], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/bias:0', data=self.critic.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1
    
    @tf.function(experimental_relax_shapes=True)
    def _critic_step(self, ret, link_state_critic, first_critic, second_critic, num_edges_critic):
        ret = tf.stop_gradient(ret)

        value = self.critic(link_state_critic, first_critic, second_critic,
                    num_edges_critic=num_edges_critic, training=True)[0]
        #Calculamos la pérdida o error producido : cuadrado de la diferencia
        critic_sample_loss = tf.math.squared_difference(ret,value)
        return critic_sample_loss
    
    @tf.function(experimental_relax_shapes=True)

    #Simula dar un paso con el actor. De esta forma, tras recoger experiencia a lo largo de una trayectoria, se calculará la pérdida para cada estado de la trayectoria
    def _actor_step(self, advantage, old_act, old_policy_probs, link_state, graph_id, \
                    first, second, num_edges):
        #Variables que no queremos que se actualicen cuando calculemos y usemos el gradiente
        adv = tf.stop_gradient(advantage)
        old_act = tf.stop_gradient(old_act) #Vector de booleanos que tiene activado únicamente el desvío a aplicar
        old_policy_probs = tf.stop_gradient(old_policy_probs)

        #Evaluamos el actor pasándole los grafos con los desvíos candidatos
        r = self.actor(link_state, graph_id, first, second, num_edges, training=True)
        qvalues = tf.reshape(r, (1, len(r))) # Dimensiones (len(r),1) a (1,len(r))
        #Nuevas probabilidades de llevar a cabo cada desvío
        newpolicy_probs = tf.nn.softmax(qvalues)
        #De esto último, tomamos la nueva probabilidad de aplicar el desvío que seleccióno el agente al recoger experiencia, dado este estado inicial
        #Esto es, newpolicy_probs2 --> prob de aplicar la misma acción dado el mismo estado inicial, pero usando la política nueva.
        newpolicy_probs2 = tf.math.reduce_sum(old_act * newpolicy_probs[0])
        '''
        De esta forma, la probabilidad de aplicar la acción dado el estado inicial será:
            * Para la política antigua --> tf.math.reduce_sum(old_act * old_policy_probs)
            * Para la nueva política --> newpolicy_probs2
        '''

        '''
        Evaluamos la función de pérdida sobre los parámetros actuales del modelo, para saber la pérdida producida al usar esa nueva política, considerando una determinada acción aplicada sobre un determinado estado
        Viendo la fórmula de la función de pérdida para PPO, podemos o seguir la fórmula inicial o usar 'logs', como aquí se hace
        Para ello:
            * Calcular el ratio: dividir la probabilidad de llevar a cabo la misma acción seleccionada sobre un estado,
            considerando tanto la política antigua como la nueva --> 'ratio'
            * Multiplicar el ratio por la ventaja conseguida dado ese estado inicial --> surr1
        '''
        ratio = tf.exp(tf.math.log(newpolicy_probs2) - tf.math.log(tf.math.reduce_sum(old_act*old_policy_probs)))
        surr1 = -ratio*adv
        surr2 = -tf.clip_by_value(ratio, clip_value_min=1 - clipping_val, clip_value_max=1 + clipping_val) * adv
        '''
        La pérdida producida es --> mínimo( ratio*ventaja // clip(...)*ventaja ), o máximo de (-ratio*ventaja, -clip(...)*ventaja)
        '''
        loss_sample = tf.maximum(surr1, surr2)


        entropy_sample = -tf.math.reduce_sum(tf.math.log(newpolicy_probs) * newpolicy_probs[0])
        return loss_sample, entropy_sample

    '''
    inds: conjunto de indices de los estados considerados a la hora de actualizar el agente y el crítico
    
    Este método nos permite actualizar el agente usando un lote de tamaño length(inds). De esta forma:
        * Se calcula la pérdida asociada a cada uno de los estados que conforman el lote de trabajo, tanto para el actor como para el crítico
        * Se toma la pérdida media tanto del actor como del crítico (media en todo el lote) y se suma, obteniendo la pérdida media total que se produce para la politica actual
        * Aplicamos descenso de gradiente considerando 
    '''
    def _train_step_combined(self, inds):
        entropies = []
        actor_losses = []
        critic_losses = []
        # Optimize weights
        with tf.GradientTape() as tape:
            #Volvemos a aplicar el actor/crítico sobre los estados del lote actual 
            for minibatch_ind in inds:
                #Para cada uno de los estados seleccionados, tomamos toda la experiencia que ha recogido el gaente para dicho estado
                sample = self.memory[minibatch_ind]

                # ACTOR
                '''
                Para actualizar el actor usando un determinado estado, pasar:
                    * Ventaja conseguida
                    * Acción aplicada (vector de booleanos)
                    * Probabilidad que había de aplicar cada posible desvío según la política que seguía el agente en dicho estado
                    * Vector de estados de los enlaces del grafo, para cada uno de los grafos asociados a cada posible desvío a seguir (camino marcado)
                    * Id de los diferentes grafos o desvíos a aplicar
                    * first y second: para conocer, para cada uno de los enlaces, cuáles son los enlaces que le envían mensajes
                '''
                loss_sample, entropy_sample = self._actor_step(sample["advantage"], sample["old_act"], sample["old_policy_probs"], \
                            sample["link_state"], sample["graph_id"], sample["first"], sample["second"], sample["num_edges"])
                actor_losses.append(loss_sample) #Error cometido en dicho elemento del lote
                entropies.append(entropy_sample)
                #Acabamos de calcular la pérdida asociada a dicho estado
                
                #Mismo proceso pero para el crítico
                '''
                Para actualizar el crítico, se debe pasar al método encargado de actualizar, los siguientes parámetros:
                    * Valor/calidad de la política seguida en cada instante de tiempo (con la ventaja agregada para cada instante)
                    * Vector de estado de cada enlace del grafo
                    * 'first_critic' y 'second_critic' : enlaces emisores/receptores del grafo inicial de trabajo, para cada instante de la trayectoria
                    * nª enlaces del grafo inicial, para cada instante
                Este método nos devuelve el error obtenido para un estado concreto de la trayectoria
                '''
                critic_sample_loss = self._critic_step(sample["return"], sample["link_state_critic"], sample["first_critic"], sample["second_critic"], sample["num_edges_critic"])
                critic_losses.append(critic_sample_loss)
        
            #Tras calcular la pérdida del actor/crítico para cada estado del lote de trabajo, calculamos la pérdida media del actor y crítico
            #Esto es, error medio del actor/crítico para dicho lote
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = tf.math.reduce_mean(actor_losses) - ENTROPY_BETA * final_entropy
            #La pérdida media total, dado dicho lote de estados, será la suma de las pérdidas media tanto del actor como del crítico (el actor y el crítico usará cada uno su correspondiente función de pérdida)
            total_loss = actor_loss + critic_loss

        '''
        Calculamos el vector gradiente de la función de pérdida, considerando únicamente los siguientes parámetros de la función de pérdida (sobre los cuales se calcula la derivada parcial):
            * Parámetros entrenables del actor
            * Parámetros entrenables del crítico
        Para ello, se usa tape.gradient(), que trabaja con dos parámetros:
            * Pérdida total para dicho lote de estados, sobre el cual calculamos el gradiente
            * Variables de entrada sobre las que queremos calcular las derivadas parciales
        Obtenemos como salida 'grad' ---> lista de derivadas parciales sobre dichas variables indicadas
        '''
        grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)
        #gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        grad, _grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
        #Se actualizan los parámetros entrenables tanto del actor como del crítico (y sobre los cuales se ha calculado el gradiente) usando el vector gradiente previo calculado, y usando 
        #como algoritmo de optimización 'Adam'
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
        entropies.clear()
        actor_losses.clear()
        critic_losses.clear()
        #Devolvemos la pérdida media del actor y del crítico para dicho lote de estados, y la entropía final
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, tensors, critic_features, returns, advantages):

        #Longitud de la trayectoria que seguimos para recopilar experiencia 
        for pos in range(0, int(BUFF_SIZE)): #Para cada uno de los estados por los que ha pasado el agente...

            tensor = tensors[pos] #info de cada grafo con desvío candidato aplicado
            critic_feature = critic_features[pos] #estado inicial del grafo (sin aplicar ningún desvío)
            action = actions[pos] #vector de booleanos, activando la acción o desvío seleccionado
            ret_value = returns[pos] #valor del crítico actualizado usando la ventaja
            adv_value = advantages[pos] #ventaja obtenida
            action_dist = actions_probs[pos] #distribuciones de prob de cada desvío candidato
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'], #estado oculto (uso,capacidad,ancho de banda de la ruta del desvío) para cada enlace, por columna
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      
            #Guardamos en la memoria del agente la experiencia obtenida en dicho estado
            #Este proceso se repite para todos los estados de la trayectoria recorrida
            self.memory.append(final_tensors)  

        
        #Hemos guardado toda la experiencia que ha obtenido el agente a lo largo de la trayectoria recorrida
        #Ahora toca actualizar el actor y crítico.

        #Para actualizar el agente, se recorre la trayectoria completa 'PPO_EPOCHS' veces
        '''
        Dada dicha iteración de trabajo, y tras haber recopilado experiencia a lo largo de esa trayectoria de longitud BUFFSIZE, se procede a actualizar
        tanto el actor/crítico:
        Se recorre un determinado nº de veces la trayectoria completa, tal que, cada vez que la recorramos:
            * Se divide dicha trayectoria en lotes (conjunto de instantes), y se actualiza una vez el agente por
        '''

        for i in range(PPO_EPOCHS):
            #En cada PPO_EPOCH, permutamos los estados de la trayectoria para que se actualice el agente considerando diferentes lotes de estado en función del PPO_EPOCH
            np.random.shuffle(self.inds)

            #Actualizamos por lotes en dicho episodio
            for indice_lote, start in enumerate(range(0, BUFF_SIZE, MINI_BATCH_SIZE)):
                end = start + MINI_BATCH_SIZE
                #Obtenemos la pérdida media del actor/crítico sobre dicho lote de estados
                actor_loss, critic_loss, final_entropy = self._train_step_combined(self.inds[start:end])

        
        
        self.memory.clear()
        # self._write_tf_summary(actor_loss, critic_loss, final_entropy)
        gc.collect()
        #devuuelvo la pérdida media del actor/crítico del último lote de la trayectoria asociada a la última PPO_EPOCHS
        return actor_loss, critic_loss

'''
Este método calcula la ventaja de cada uno de los estados que conforman la trayectoria a lo largo de la cual hemos recogido experiencia.
Para calcular la ventaja asociada al instante 'i', tendremos que usar las recompensas obtenidas en los instantes posteriores a 'i' (sin salirnos de la trayectoria)

 0   1   2   3   4   5
A_0 A_1 A_2 A_3 A_4 A_5

tal que

A_0 : trabaja con los estados i= {0,1,2,3,4,5}
A_1 : trabaja con los estados i= {1,2,3,4,5}
...
A__5: trabaja solo con i = {5}

'mask' se activará cuando se haya reenrutado la última demanda crítica de una cierta matriz de tráfico, para evitar así mezclar acciones del agente sobre diferentes matrices de tráfico.
Es decir, cuando calculemos la ventaja del estado asociado a la última demanda crítica, no se tendrá en cuenta el siguiente estado 't+1' ya que es el final del episodio para dicha matriz de tráfico.

'''
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        #Calculamos el delta que interviene en la ventaja
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        #Para calcular la ventaja del estado 'i' --> delta del estado 'i' + la ventana 'i+1' (multiplicada por gamma*lambda)
        gae = delta + gamma * lmbda * masks[i] * gae
        #Almacenamos como primer elemento de la lista, empujando hacia delante todos los demas, para tener las ventajas ordenadas desde el primer instante hasta el último
        #Sumamos el valor en el estado 'i' con la ventaja obtenida, para actualizar la función valor.
        returns.insert(0, gae + values[i])

    #TTambién tomaremos las ventajas asociadas a cada instante, sin sumarle el valor correspondiente.
    adv = np.array(returns) - values[:-1]
    # Normalize advantages to reduce variance
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-i', help='iters', type=int, required=True)
    parser.add_argument('-c', help='counter model', type=int, required=True)
    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    parser.add_argument('-f1', help='dataset folder name topology 1', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='dataset folder name topology 2', type=str, required=True, nargs='+')
    parser.add_argument('-f3', help='dataset folder name topology 3', type=str, required=True, nargs='+')
    args = parser.parse_args()

    dataset_folder_name1 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f1[0]
    dataset_folder_name2 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f2[0]
    dataset_folder_name3 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f3[0]

    print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))

    # Get the environment and extract the number of actions.
    env_training1 = gym.make(ENV_NAME)
    env_training1.seed(SEED)
    env_training1.generate_environment(dataset_folder_name1+"/TRAIN", "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training1.top_K_critical_demands = take_critic_demands

    env_training2 = gym.make(ENV_NAME)
    env_training2.seed(SEED)
    env_training2.generate_environment(dataset_folder_name2+"/TRAIN", "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training2.top_K_critical_demands = take_critic_demands

    env_training3 = gym.make(ENV_NAME)
    env_training3.seed(SEED)
    env_training3.generate_environment(dataset_folder_name3+"/TRAIN", "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training3.top_K_critical_demands = take_critic_demands

    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name1+"/EVALUATE", "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval2.seed(SEED)
    env_eval2.generate_environment(dataset_folder_name2+"/EVALUATE", "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval2.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval3.seed(SEED)
    env_eval3.generate_environment(dataset_folder_name3+"/EVALUATE", "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval3.top_K_critical_demands = take_critic_demands

    #Si no existe el directorio donde se guarda la versión del agente DLR al final de episodio (para realizar los chekcpoints del modelo)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    #Abrimos (o creamos) en modo 'añadir al final' el archivo de logs, que mantiene un registro del entrenamiento a lo largo del episodio actual
    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    # Load maximum reward from previous iterations and the current lr
    '''
    Al inicio de esta nueva época de entrenamiento, se abrirá el archivo temporal para conocer la máxima recompensa obtenida hasta el momento (teniendo en cuenta todas las épocas pasadas)
    y la tasa de aprendizaje que se empleó a lo largo de la última época.
    '''
    if os.path.exists("./tmp/" + differentiation_str + "tmp.pckl"):
        f = open("./tmp/" + differentiation_str + "tmp.pckl", 'rb')
        max_reward, hparams['learning_rate'] = pickle.load(f)
        f.close()
    else:
        #No existe el archivo temporal ---> Estamos en la primera época de entrenamiento
        max_reward = -1000

    #Cada vez que se procese un nuevo lote de episodios, la tasa de aprendizaje empleada decae
    if args.i%DECAY_STEPS==0:
        hparams['learning_rate'] = decayed_learning_rate(args.i)

    if args.i>=ENTROPY_STEP:
        ENTROPY_BETA = ENTROPY_BETA/10 #Limitamos la exploración

    #Se inicializa el agente
    agent = PPOActorCritic()

    #Definimos las zonas de guardado (lugares donde realizar los checkpoints) tanto del actor como del crítico que lleva implícito el agente DRL
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    #Construimos los objetos que nos permiten guardar/restaurar el estado de los modelos y optimizadores que emplean
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    #Si no es el primer episodio, es decir, ya se ha entrenado el agente DRL al menos durante un episodio completo, cargamos el estado del modelo al final del último episodio, previo al actual
    if args.i>0:
        # -1 because the current value is to store the model that we train in this iteration
        '''
        Restauramos el último estado guardado del actor/crítico y de sus correspondientes optimizadores (que usan una tasa de aprendizaje previamente cambiada)
        Se usarán los '.index' (índices de los parámetros del modelo) y '.data' (valores reales de los parámetros indexados según .index)

        IMPORTANTE: se restaura la última versión guardada al final del episodio previo, comenzando a usar al iniciar las iteraciones de esta nueva época esta versión restaurada
        '''
        checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
        checkpoint_actor.restore(checkpoint_dir + "/ckpt_ACT-" + str(args.c-1)) #..../ckpt_ACT-<counter_store_model-1>, ya que el argumento '-c' que se pasa al ejecutar este script es la nueva versión del modelo que debemos comenzar usando
        checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)
        checkpoint_critic.restore(checkpoint_dir + "/ckpt_CRT-" + str(args.c-1))
    #Ya habremos cargado el último estado guardado del actor/crítico al inicio de este nuevo episodio

    reward_id = 0
    evalMeanReward = 0
    counter_store_model = args.c  #Nueva versión del modelo, que se irá incrementando a lo largo de las iteraciones de esta nueva época

    rewards_test = np.zeros(EVALUATION_EPISODES*3)
    error_links = np.zeros(EVALUATION_EPISODES*3)
    max_link_uti = np.zeros(EVALUATION_EPISODES*3) #Gaurda el uso del enlace de mayor uso, para cada matriz de validación de cada topología (es decir, tras reenrutar todas las demandas críticas)
    min_link_uti = np.zeros(EVALUATION_EPISODES*3)
    uti_std = np.zeros(EVALUATION_EPISODES*3)

    training_tm_ids = set(range(100))

    for iters in range(args.e): #Entrenamos args.e episodios (procesamos un lote de episodios)
        states = []
        critic_features = []
        tensors = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []

        print("MIDDLEPOINT ROUTING(3 TOP Topologies Enero "+experiment_letter+") PPO EPISODE: ", args.i+iters)
        number_samples_reached = False

        '''
        A posteriori, se realiza lo siguiente: Se recorrerá una trayectoria de longitud num_samples1 + num_samples2 + num_samples3 (el numero hace referencia a la topología), tal que para cada topología:
            * Selecciono la matriz de tráfico con la que voy a trabajar en el episodio actual
            * Reenruto todas las demandas críticas dada dicha matriz de tráfico. Además, esto lo repito (sobre la misma matriz de tráfico) un determinado nº de veces, hasta que el nº de acciones aplicada sea num_samples<numero>
            
        '''

        #Selecciono la matriz de tráfico con la que voy a trabajar en la primera topología
        tm_id = random.sample(training_tm_ids, 1)[0]

        #La longitud de la trayectoria que se va a recorrer para recoger experencia = 'num_samples_top1'
        #IMPORTANTE!! --> Se repite el agente un determinado nº de veces sobre la misma matriz de tráfico en dicha topología, ya que las acciones son probabilísticas
        print("Comenzado entrenamiento con la 1ª topología...")
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 1
            ######

            #Se realizarán tantas iteraciones del siguiente bucle 'while' como demandas críticas se hayan seleccionado reenrutar, dada dicha matriz de tráfico
            demand, source, destination = env_training1.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)

                # Predict probabilities over middlepoints
                '''
                Le pasamos al agente:
                    - Entorno de trabajo (asociado a la primera topología)
                    - Origen y Destino de la demanda que queremos reenrutar
                El agente será una MPNN que tendrá que aplicarse sobre tantos grafos como desvíos podamos realizar,
                tal que cada grafo tendrá marcada la ruta a seguir para realizar un cierto desvío.

                Salida: probabilidades de llevar a cabo cada acción(=desvío) y la info de cada acción (grafo con el desvío marcado)
                '''
                action_dist, tensor = agent.pred_action_distrib_sp(env_training1, source, destination)
                #Antes de aplicar finalmente un desvío sobre el grafo en el estado actual, y pasar al siguiente estado, perdiéndonos
                #el estado inicial del grafo, invocamos la función siguiente para guardarnos las características del estado inicial del grafo.
                features = agent.critic_get_graph_features(env_training1)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        num_edges_critic=features['num_edges_critic'], training=False)[0].numpy()[0]

                #Tomamos una acción teniendo en cuenta sus probabilidades. Esto se hace únicamente en el proceso de entrenamiento, para ofrecer una mayor exploración, y no tirar siempre por ell "mejor camino" aparentemente
                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                '''
                Plasmamos la acción escogida (índice del middlepoint a usar) sobre el entorno
                'done' es una variable booleana que se activa cuando todas las demandas críticas seleccionadas ya han sido enrutadas
                Al dar un paso, y siempre que queden demandas críticas aun no reenrutadas, se fija en el entorno la siguiente demanda crítica usando 'pathMaxwdth'
                '''
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training1.step(action, demand, source, destination)
                mask = not done

                states.append((env_training1.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                #Cuando el agente haya pasado por un nº de estados del entorno = num_samples_top1 (nº de decisiones tomadas), salimos del segundo while y tb del primero (ya que number_samples_reached se activa),
                #pasando a trabajar con la siguiente topología.
                if len(states) == num_samples_top1:
                    number_samples_reached = True
                    break

                #Una vez reenrutadas todas las demandas críticas seleccionadas, para dicha matriz de tráfico, salimos del segundo 'while', y volvemos a realizar otra iteración del primer 'while', volviendo a restablecer la ruta OSPF
                #de las demandas asociadas a la misma matriz de tráfico con la que estamos trabajando en la topología 1
                if done:
                    break

        number_samples_reached = False
        tm_id = random.sample(training_tm_ids, 1)[0] #Matriz de tráfico con la que vamos a trabajar en la topología 2
        print("Comenzado entrenamiento con la 2ª topología...")
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 2
            ######

            demand, source, destination = env_training2.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training2, source, destination)
                features = agent.critic_get_graph_features(env_training2)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        num_edges_critic=features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training2.step(action, demand, source, destination)
                mask = not done

                states.append((env_training2.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2:
                    number_samples_reached = True
                    break

                if done:
                    break

        number_samples_reached = False
        tm_id = random.sample(training_tm_ids, 1)[0]
        print("Comenzado entrenamiento con la 3ª topología...")
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 3
            ######

            demand, source, destination = env_training3.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training3, source, destination)
                features = agent.critic_get_graph_features(env_training3)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        num_edges_critic=features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(len(action_dist), p=action_dist)
                #Vector de booleanos de longitud 'nº de middlepoints posibles para la demanda de trabajo', activando únicamente la componente/middlepoint que finalmente se va a usar como desvío.
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training3.step(action, demand, source, destination)
                mask = not done

                states.append((env_training3.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2+num_samples_top3:
                    number_samples_reached = True
                    break

                if done:
                    break

        features = agent.critic_get_graph_features(env_training3)
        #Pasamos al crítico la información del grafo inicial (sin aplicar un desvío candidato) + enlaces mensajeros + enlaces receptores
        #Esto equivale a preguntar al crítico cómo de bueno es el resultado futuro que podemos obtener aplicando la política actual dado el estado inicial del grafo
        q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                num_edges_critic=features['num_edges_critic'], training=False)[0].numpy()[0]       
        values.append(q_value)

        #Después de aplicar el agente varias veces sobre cada topología (y considerando diferentes matrices de tráfico de entrenamiento):
        #Obtenemos los valores actualizados para cada instante de la trayectoria de trabajo, y la ventaja obtenida en cada instante
        returns, advantages = get_advantages(values, masks, rewards)
        '''
        Actualizo el agente al final de la iteración dentro del episodio (tras aplicarse sobre cada una de las 3 topologías, y considerando diferentes TM's sobre cada topología)
        Le paso como parámetros:
            * actions --> Para cada estado: vector de booleanos (tantos como middlepoints se puedan escoger para dicha demanda) activando la acción/middlepoint a aplicar
            * action_probs --> Para cada estado a lo largo de la trayectoria --> distribuciones de probabilidad de las diferentes acciones a aplicar
            * tensors --> para cada estado: los grafos asociados a los diferentes desvíos candidatos a realizar
            * critic_features --> características del grafo en cada uno de los estados iniciales, antes de aplicar el desvío seleccionado
            * returns --> valores del crítico actualizados
            * advantages --> ventaja obtenida en cada uno de los estados

        Resumen: recogemos, para cada uno de los estados de la trayectoria que hemos recorrido para recoger experiencia antes de actualizar nada:
        desvío seleccionado de entre los desvios posibles, distribuciones de prob de llevar a cabo cada desvío, estado inicial del grafo marcando cada una de las rutas de los desvíos candidatos,
        estado inicial del grafo (sin considerar ningún desvío candidato), valor del crítico actualizado (usando la ventaja), ventaja obtenida
        '''

        print("\n ---- Actualizando los parámetros del actor/crítico ----")
        actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, tensors, critic_features, returns, advantages)
        #Aquí se está almacenando la pérdida media del actor/crítico en dicho episodio 
        fileLogs.write("a," + str(actor_loss.numpy()) + ",\n")
        fileLogs.write("c," + str(critic_loss.numpy()) + ",\n")
        fileLogs.flush()

        '''
        Tras actualizar el agente al final del episodio, evaluamos el agente seleccionando 20 de las 50 matrices de tráfico que se usan para validación.
        Suponemos que dichas matrices de tráfico de validación se distribuyen de forma uniforme.
        '''

        print("\nValidando...")
        # Evaluate on FIRST TOPOLOGY, en este caso 20 episodios (las 20 TM seleccionadas del conj de validación)
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            demand, source, destination = env_eval.reset(tm_id)
            done = False
            rewardAddTest = 0
            #Aplicamos el agente tantas veces como demandas críticas hayan sido seleccionadas, dada esa TM
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval, source, destination)
                #Ahora, en lugar de seleccionar el middlepoint de desvío llevando a cabo una selección probabílistica, se selecciona
                #aquel de mayor probabilidad (selección determinista)
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            #La recompensa obtenida en el episodio 'eps' será la suma de las recompensas obtenidas tras reenrutar cada una de las demandas críticas seleccionadas
            #Esto equivale a tomar la recompensa futura acumulada real obtenida tras seguir una trayectoria de longitud concreta.
            rewards_test[eps] = rewardAddTest
            error_links[eps] = error_eval_links
            max_link_uti[eps] = maxLinkUti[2] #Uso del enlace de mayor uso tras reenrutar todas las demandas criticas para dicha matriz de tráfico (métrica)
            min_link_uti[eps] = minLinkUti
            uti_std[eps] = utiStd
        
        # Evaluate on SECOND TOPOLOGy
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES+eps
            demand, source, destination = env_eval2.reset(tm_id)
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval2, source, destination)
                
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval2.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2]
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd

        # Evaluate on THIRD TOPOLOGY
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES*2+eps
            demand, source, destination = env_eval3.reset(tm_id)
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval3, source, destination)
                
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval3.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            #Hemos terminado de reenrutar 
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2] #Uso del enlace de mayor uso después de haber reenrutado todas las demandas críticas seleccionadas para dicha matriz
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd 

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(";," + str(np.mean(uti_std)) + ",\n")
        fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
        fileLogs.write("<," + str(np.amax(max_link_uti)) + ",\n")
        fileLogs.write(">," + str(np.amax(min_link_uti)) + ",\n")
        fileLogs.write("ENTR," + str(ENTROPY_BETA) + ",\n")
        #fileLogs.write("-," + str(agent.epsilon) + ",\n")
        fileLogs.write("REW," + str(evalMeanReward) + ",\n")
        fileLogs.write("lr," + str(hparams['learning_rate']) + ",\n")
  
        #Si al final del episodio de entrenamiento, y tras validar, obtenemos una recompensa media que supera a la mejor obtenida hasta el mometo,
        #escribimos en el archivo de Logs que hemos encontrado un nuevo modelo con mejor recompensa.
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            #Escribimos en el archivo de Logs la recompensa del nuevo mejor modelo y su id, para poder tomar la versión del nuevo mejor modelo en cualquier momento
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        
        # Store trained model
        # Storing the model and the tape.gradient make the memory increase
        # .save guarda tanto las variables internas de Checkpoint (incluyendo save_counter) como los objetos trackeables
        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        gc.collect()
        # ---------------- FIN EPISODIO ---------

    '''
    Al finalizar la época de entrenamiento, se abre el archivo temporal en modo binario para escritura, de forma que cada vez que se abra en este modo, se sobreescribe la información
    que ya contenía por la nueva que estamos introduciendo

    Este archivo temporal se usa a lo largo del entrenamiento del modelo (durante las épocas de entrenamiento) para guardar al final de cada época:
        * Máxima recompensa obtenida hasta el momento, teniendo en cuenta todas las épocas pasadas
        * Última tasa de aprendizaje empleada (en la última época antes de ésta nueva)
    '''
    f = open("./tmp/" + differentiation_str + "tmp.pckl", 'wb')
    pickle.dump((max_reward, hparams['learning_rate']), f)
    f.close()

