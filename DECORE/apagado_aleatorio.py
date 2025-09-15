import tensorflow as tf
from tensorflow import keras
#import keras
import tensorflow_probability as tfp
import copy
import numpy as np
import gym
import gym_graph
import actor 

class Mascara(tf.keras.layers.Layer):
    def __init__(self,id,dimension_estado):
        super().__init__(name=f'Mascara_{id}')
        #self.A = tf.ones((dimension_estado,), dtype=tf.float32) #Tensor unidimensional con la acción que aplica el agente para cada neurona (mantener o preservar)
        self.A = tf.Variable(tf.ones((dimension_estado,), dtype=tf.float32), trainable=False)
        self.dimension_estado = dimension_estado

    def fijar_acciones(self, porcentaje):
        neuronas_capa = self.dimension_estado
        num_neuronas_apagar = int(neuronas_capa * porcentaje)
        # Empezamos con todos 1 (mantener todas las neuronas)
        acciones = np.ones(self.dimension_estado, dtype=np.float32)
        # Elegimos índices aleatorios para apagar (poner a 0)
        indices_a_apagar = np.random.choice(self.dimension_estado, size=num_neuronas_apagar, replace=False)
        acciones[indices_a_apagar] = 0.0
        # Guardamos el tensor
        #self.A = tf.convert_to_tensor(acciones)
        self.A.assign(acciones)

        #print(f'Se ha decidido apagar las neuronas de índices {indices_a_apagar} del total ({self.dimension_estado})')
        #print(f'Las acciones aplicadas serán {self.A}')


    def call(self, inputs):
        #Aplicamos sobre las activaciones de la capa previa la máscara obtenida
        mascara = tf.reshape(self.A,[1,-1])
        return(inputs*mascara) 

#Tupla con las capas a localizar dentro del modelo base
capas_buscadas = (tf.keras.layers.Dense,)
#Se ha modificado para que no se añada ninguna máscara en la fase de Message, sólo en las dos primeras capas de Readout
class ApagadoAleatorio(tf.keras.Model):

    def __init__(self,modelo_base):
        super(ApagadoAleatorio,self).__init__() #Nombre del modelo
        self.hparams = modelo_base.hparams
        self.Message = None
        self.Update = None
        self.Readout = None
        self.mascaras = [] #Lista de los agentes insertados

        self._input_shape = []
        self._parametros_modelo_base = []
        self._modificar_modelo(modelo_base)
    
    #Método privado, encargado de modificar las etapas de la MPNN base, introduciendo los agentes
    def _modificar_modelo(self, modelo_base):
        fases = ['Message','Update','Readout']
        next_id = 0
        for fase in fases:
            if hasattr(modelo_base, fase):
                atributo = getattr(modelo_base, fase)
                #Modificamos dicha fase del modelo base, insertando los agentes correspondientes
                if hasattr(atributo,'layers'):
                    #print('\nConfigurando la etapa con un modelo secuencial asociado')
                    #Dicha fase es una secuencia de capas. Recorremos las capas
                    lista_capas = []
                    #Introduzco todas las capas del modelo base en mi nuevo Sequential
                    for posicion, capa in enumerate(atributo.layers):
                        #Es una capa buscada? ---> Insertamos Agente
                        config = capa.get_config()
                        nueva_capa = type(capa).from_config(config)
                        nueva_capa.trainable = False
                        nueva_capa.mascara_asociada = False
                        lista_capas.append(nueva_capa)
                        #Compruebo si es una de las capas buscadas
                        #La única excepción es que no se puede introducir un agente en la cabecera del modelo, es decir, en la última capa de la MPNN (situada en Readout)
                        if isinstance(capa,capas_buscadas) and not(fase=='Readout' and posicion==(len(atributo.layers)-1)) and not fase=='Message':
                            nuevo_agente = Mascara(next_id,capa.units)
                            self.mascaras.append(nuevo_agente)
                            lista_capas.append(nuevo_agente)
                            next_id+=1
                            #Indicamos en la capa previa que se le ha asignado un agente a dicha capa
                            nueva_capa.mascara_asociada = True
                            
                    #Secuencia del modelo base ya recorrida --> Instancio y construyo los parámetros
                    fase_modificada = tf.keras.Sequential(lista_capas)
                    self._input_shape.append(atributo.get_build_config()['input_shape'])
                    
                    for capa in atributo.layers:
                        self._parametros_modelo_base.append(capa.get_weights())
                    
                else:
                    #Dicha fase no es una secuencia, es simplemente una capa sin más
                    #Tomo la info para inicializar de la misma forma la nueva instancia...
                    #print(f'\nConfigurando la etapa con una sola capa base')
                    config = atributo.get_config()
                    nueva_capa = type(atributo).from_config(config)
                    nueva_capa.trainable = False
                    nueva_capa.mascara_asociada = False
                    #Me guardo los parámetros de la capa empleada en dicha fase
                    self._input_shape.append(atributo.get_build_config()['input_shape'])
                    self._parametros_modelo_base.append(atributo.get_weights())
                    #Comprobemos que es una de las capas buscadas, y en caso afirmatico, que estemos en Readout (no tiene sentido añadir un agente a la última capa del modelo)
                    if isinstance(atributo, capas_buscadas) and fase!='Readout' and fase!='Message':
                        #Me creo un Sequential con la capa base y la capa 'Agente'
                        nuevo_agente = Mascara(next_id,atributo.units)
                        self.mascaras.append(nuevo_agente)
                        #Indicamos en la capa que tiene asignado un agente
                        nueva_capa.mascara_asociada = True
                        fase_modificada = tf.keras.Sequential([nueva_capa, nuevo_agente])
                        next_id+=1
                    else:
                        #No es una capa buscada. Dicha fase por tanto se mantendrá igual
                        fase_modificada = nueva_capa
                #Asignamos la capa usada en la etapa actual
                setattr(self,fase,fase_modificada)
        #print(f'Longitud total de inputs guardados: {len(self._input_shape)}')
        #print(f'Longitud de la lista con los parámetros guardados: {len(self._parametros_modelo_base)}')
    
    
    def build(self, input_shape=None):
        #Comenzamos construyendo los parámetros de las capas del modelo DECORE
        #print("\nConstruyendo las capas del modelo de apagado aleatorio..")

        lista_fases = ['Message','Update','Readout']
        ind_input = 0 #índice para movernos por la lista de input_shape
        ind_params = 0 #índice para mvoernos por la lista de parámetros guardados
        for nombre_fase in lista_fases:
            fase = getattr(self,nombre_fase)
            if isinstance(fase,tf.keras.Sequential):
                fase.build(input_shape=self._input_shape[ind_input])
                ind_input += 1
                #Una vez construido el modelo secuencial, asigno los valores a las capas del modelo secuencial
                for capa in fase.layers:
                    if not isinstance(capa,Mascara):
                        capa.set_weights(self._parametros_modelo_base[ind_params])
                        ind_params += 1
            else:
                #La fase está formada por una sola capa, así que no puede ser un agente
                fase.build(input_shape=self._input_shape[ind_input])
                ind_input += 1
                #Tras construir la capa, copio los valores a los parámetros de la capa
                fase.set_weights(self._parametros_modelo_base[ind_params])
                ind_params += 1

        #Indicamos que se ha construido el modelo DECORE
        self.built = True
        #print("Modelo construido con éxito")

        del self._input_shape
        del self._parametros_modelo_base
    
    #@tf.function
    def call(self, link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            outputs = self.Message(edgesConcat, training=training)
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)


            outputs, links_state_list = self.Update(edges_inputs, [link_state], training=training)

            link_state = links_state_list[0]

        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)
        
        r = self.Readout(edges_combi_outputs,training=training)
        return r






