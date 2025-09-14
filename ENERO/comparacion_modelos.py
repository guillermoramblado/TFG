import importlib
import numpy as np
import tensorflow as tf
import os
import sys
import gym
import gym_graph
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import actorPPOmiddR as actor

ruta_DECORE = os.path.abspath("../DECORE")
sys.path.append(ruta_DECORE)
from compresion_final import ModeloBaseComprimido
from apagado_aleatorio import ApagadoAleatorio

'''
Este archivo nos permitirá generar boxplots para comparar el rendimiento del modelo base y de la variante implementada:
    * Se usarán las 3 topologías indicadas al ejecutar el script
    * Se usarán las 50 matrices de tráfico reservadas para validación, para cada topología
'''

lista_modelos = ["actorPPOmiddR","actorPPO_v2"] #La clase interna de cada módulo debe llamarse igual: myModel
carpetas_checkpoints = ["./modelsEnero_3top_15_B_NEW/","./modelsVariante_Enero/"] #Ruta de las carpetas donde se realizan los chekcpoints de cada modelo
ruta_log_modelos = ["./Logs/expEnero_3top_15_B_NEWLogs.txt","./Logs/expVariante_EneroLogs.txt"] #Ruta del fichero LOG de cada modelo
lista_checkpoints = [] #En el main añadimos la ruta de la versión a restaurar para cada modelo


#Ruta del directorio que contiene las carpetas con las diferentes topologías que pueden ser usadas
ruta_directorio_base = "../ENERO_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"
#Nombre de las topologías que se quieren usar para validación
nombre_topologias = ["NEW_BtAsiaPac","NEW_Garr199905","NEW_Goodnet"]
nombre_base_topologias = ["BtAsiaPac","Garr199905","Goodnet"]


#Parámetros usados para generar el entorno
ENV_NAME = 'GraphEnv-v16'
SEED = 9
EPISODE_LENGTH = 100
NUM_ACTIONS = 100
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100
take_critic_demands = True

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)


hparams = {
    'l2': 0.005,
    'dropout_rate': 0.1,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5,
}

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


class Modelos:
    def __init__(self):
        self.action = None
        self.softMaxQValues = None
        self.listQValues = None
        #self.K = env_training.K

        self.utilization_feature = None
        self.bw_allocated_feature = None

        #Aquí el actor será uno de los diferentes modelos que se quiere comparar. Este actor será el que se utilice en los métodos usados en validación
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = None

        #Diccionario con los diferentes modelos a comparar
        self.modelos = {}
        self._asignar_modelos()

    #Método empleado para asociar los modelos que se van a comparar
    def _asignar_modelos(self):
        for path_model,path_checkpoint in zip(lista_modelos,lista_checkpoints):
            modulo = importlib.import_module(path_model)
            #Instancio el modelo definido dentro de dicho módulo, y construyo sus parámetros
            modelo = modulo.myModel(hparams, hidden_init_actor, kernel_init_actor)
            modelo.build()
            print(f'\nNúmero total de parámetros del modelo {path_model}:{modelo.count_params()}')
            self.modelos[path_model] = modelo
            #Restauro su estado
            checkpoint = tf.train.Checkpoint(model=modelo)
            checkpoint.restore(path_checkpoint).expect_partial()
            print(f'Se ha restaurado el modelo {path_model} desde checkpoint  {path_checkpoint}')     
         
    #Método que puede ser invocado para asignar un modelo comprimido
    def asignar_metodo_comprimido(self,nuevo_modelo,nombre):
        #Asociamos este modelo a la lista de modelos que se quieren comparar
        
        if nombre in self.modelos:
            print(f'Error: Ya existe un modelo con el nombre {nombre}')
        else:
            self.modelos[nombre] = nuevo_modelo
            lista_modelos.append(nombre)
            print(f'Se ha añadido un nuevo modelo a comparar, de nombre {nombre}')

    #Método que me permite obtener uno de los modelos asociados a la instancia
    def obtener_modelo(self,nombre):
        if nombre in self.modelos:
            return self.modelos[nombre]
        else:
            print(f'No existe un modelo de nombre {nombre}')

    #Método que permite definir el modelo que va a ser utilizado en un cierto momento
    def fijar_modelo(self,nombre_modelo):
        if nombre_modelo not in self.modelos:
            raise ValueError(f'No está disponible el modelo {nombre_modelo}')
        self.actor = self.modelos[nombre_modelo]

    def pred_action_distrib_sp(self, env, source, destination):
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

def generar_entorno(ruta_topologia,nombre_topologia):
    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(ruta_topologia+"/EVALUATE", nombre_topologia, EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands
    return env_eval

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

if __name__ == "__main__":
    #Se localiza, para cada agente a comparar, la mejor versión encontrada (la que conseguió la máxima recompensa futura acumulada)
    
    for pos, (modelo,log) in enumerate(zip(lista_modelos,ruta_log_modelos)):
        print(f'\nBuscando la mejor versión del modelo {modelo} en el fichero de log {log}')
        model_id = 0
        with open(log) as fp:
            for line in reversed(list(fp)):
                arrayLine = line.split(":")
                if arrayLine[0]=='MAX REWD':
                    model_id = int(arrayLine[2].split(",")[0])
                    break
        print(f'La mejor versión encontrada es {model_id}')
        #Añadimos ruta del checkpoint
        ruta_checkpoint = carpetas_checkpoints[pos]+"ckpt_ACT-"+str(model_id)
        #print(tf.train.list_variables(ruta_checkpoint))
        print(f'La ruta del checkpoint es {ruta_checkpoint}')
        lista_checkpoints.append(ruta_checkpoint)
    
    input("Presione Enter para seguir el flujo de ejecución\n")

    #Obtenemos la ruta de las diferentes topologías de trabajo
    #Lista con la ruta completa de cada una de las topologías que van a ser usadas
    lista_topologias = []
    for top in nombre_topologias:
        path_top = ruta_directorio_base+top
        lista_topologias.append(path_top)
        print(f"Se ha añadido la topología situada en {path_top}")
        #Comprobamos que existe la carpeta EVALUATE, la cual incluye las 50 matrices de validación
        if not os.path.exists(path_top+"/EVALUATE"):
            sys.exit("No se ha encontrado la carpeta EVALUATE")

    #Instancio la clase que contiene internamente todos los modelos a comparar
    instancia = Modelos()

    instancia.asignar_metodo_comprimido(obtener_modelo_base_comprimido(),"actorPPOmiddR_Comprimido")

    print("\nLista de modelos insertados:",lista_modelos)

    input("\nPresione Enter para iniciar el proceso de validación")
     #Creo para cada modelo, tantas listas como topologías use para validar, cada lista almacenando las metricas para las diferentes matrices
    resultados = {modelo: {topologia: [] for topologia in nombre_topologias} for modelo in lista_modelos}
    print("Comenzando proceso de validación")
    for model in lista_modelos:
        print(f'\nComenzando validación del modelo {model}')
        #print(f'Número de parametros del modelo es {inst}')
        instancia.fijar_modelo(model) #Para que el actor sea ese modelo
        for indice, top in enumerate(nombre_topologias):
            #Creo el entorno de trabajo con la topología a validar
            entorno = generar_entorno(lista_topologias[indice],nombre_base_topologias[indice])
            print(f'Entorno creado para la topologia {nombre_base_topologias[indice]}')
            #Aplico el agente probando las 50 matrices de validación disponibles
            for tm_id in range(50):
                #Reenruto todas las demandas críticas seleccionadas para dicha matriz
                demand, source, destination = entorno.reset(tm_id)
                done = False
                start = time.time()
                while 1:
                    action_dist, _ = instancia.pred_action_distrib_sp(entorno, source, destination)
                    action = np.argmax(action_dist)
                    reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = entorno.step(action, demand, source, destination)
                    if done:
                        break
                #Calculo el tiempo que ha demorado el agente en reenrutar todas las demandas críticas
                end = time.time()
                resultados[model][top].append((maxLinkUti[2],end-start))
    
    #input("\nPresione Enter para generar el archivo CSV y el dataframe correspondiente")
    #Convierto el diccionario en un dataframe, y lo guardo en un archivo .csv
    data = []
    for modelo, topologias in resultados.items(): #Lista de tuplas (modelo,topologia)
        for topologia, valores in topologias.items(): #Lista de tuplas (topologia, lista de tuplas (uso,tiempo))
            for uso,tiempo_total in valores:
                data.append({"Modelo": modelo, "Topologia": topologia, "Uso": uso, "Tiempo": tiempo_total})

    df = pd.DataFrame(data)
    df.to_csv("resultados_modelos_por_topologia.csv", index=False)
    #Guardo también el tiempo medio de ejecución de cada uno de los agentes
    media_por_modelo = df.groupby("Modelo")["Tiempo"].mean()
    media_por_modelo.to_csv("tiempo_medio_por_modelo.csv")

    input("\nEsperando a generar los boxplots solicitados...")


    valores_graficar = ["Uso","Tiempo"]
    nombre_graficos = ["comparativa_usos.png","comparativa_tiempos.png"]
    etiquetas_ejeY = ["Uso del enlace de mayor uso", "Tiempo de ejecución"]
    #Comprobamos si existe el directorio donde guardar las imágenes con la comparación.
    if not os.path.exists("./Imagenes/ComparacionModelos"):
        os.makedirs("./Imagenes/ComparacionModelos")

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
        path_imagen_boxplots = "./Imagenes/ComparacionModelos/" + nombre
        plt.savefig(path_imagen_boxplots)
        plt.close








    