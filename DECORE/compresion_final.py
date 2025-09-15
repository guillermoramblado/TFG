
import tensorflow as tf
from DECORE import Agente
from apagado_aleatorio import Mascara
import numpy as np

'''
Esta clase refleja el modelo final que se obtiene a partir de un modelo en el que se han insertado capas o de tipo 'Agente' o de tipo 'Mascara'.
Para instanciar el modelo será necesario especificar los siguientes parámetros:
    * modelo: modelo base con los agentes (o máscaras) insertados.
    * tiene_agentes: variable booleana que nos permite indicar si se han insertado agentes o se han insertado máscaras.
'''
class ModeloBaseComprimido(tf.keras.Model):
    def __init__(self,modelo,tiene_agentes):
        super().__init__()
        self.hparams = modelo.hparams
        self.Message = None
        self.Update = None
        self.Readout = None

        self._input_shape = []
        self._parametros_a_copiar = []
        self._obtener_modelo_comprimido(modelo,tiene_agentes)

    def _obtener_modelo_comprimido(self,modelo,tiene_agentes):
            indice_mascara= 0
            ultima_capa_tiene_mascara = False
            tam_dim_entrada = None
            #Definimos el nombre del tipo de capa que se ha insertado en el modelo base, y el atributo añadido a las capas del modelo base en función de esto 
            if tiene_agentes:
                tipo_selector = Agente
                atributo_especial = "agente_asociado"
            else:
                tipo_selector = Mascara
                atributo_especial = "mascara_asociada"
            
            fases = ['Message','Update','Readout']
            for nombre_fase in fases:
                fase = getattr(modelo,nombre_fase,None)
                if fase is not None:
                    #print(f'\nProcesando la fase {fase}')
                    #Compruebo si esta etapa se modela con un modelo secuencial o con una sola capa
                    if isinstance(fase,tf.keras.Sequential):
                        #print("Esta fase es un modelo secuencial")
                        lista_capas = []
                        #Recorro las capas del modelo secuecial
                        if ultima_capa_tiene_mascara:
                            #Tomo el tamaño de la dimension de entrada al modelo secuencial
                            input_shape = tf.TensorShape([None,int(np.sum(tam_dim_entrada))])
                        else:
                            input_shape = fase.get_build_config()['input_shape']
                        self._input_shape.append(input_shape)
                        #Procesamos ahora las capas del modelo secuencial antiguo
                        for posicion, capa in enumerate(fase.layers):
                            #Compruebo si dicha capa del modelo secuencial es un agente
                            if isinstance(capa,tipo_selector): 
                                continue #Pasamos a la siguiente iteracion (siguiente capa)
                            #Si llegamos aquí es porque esta capa no es un agente. 
                            #Tomamos el tipo de capa a instanciar y sus parámetros
                            tipo_capa = type(capa)
                            pesos, sesgos = capa.get_weights()
                            #Comprobamos si la ultima capa procesada previa a esta tenía un agente
                            #print(f'¿La capa previa a esta tenía una máscara asociada? --> {ultima_capa_tiene_mascara}')
                            #print(f'¿La capa tiene una máscara/agente asociado?: {getattr(capa,atributo_especial)}')
                            if ultima_capa_tiene_mascara:
                                pesos = pesos[tam_dim_entrada]
                            #Comprobamos si se le ha asociado un agente
                            if getattr(capa,atributo_especial):
                                #Tengo que tomar las neuronas que ha decidido mantener el agente, que es la siguiente capa del modelo secuencial
                                mascara = fase.layers[posicion+1]
                                #print(mascara.A)
                                neuronas = tf.cast(mascara.A,tf.bool).numpy()
                                pesos = pesos[:, neuronas]
                                sesgos = sesgos[neuronas]
                                #Actualizamos la última capa con agente asociado
                                ultima_capa_tiene_mascara = True
                                tam_dim_entrada = neuronas
                                #Incrementamos para usar proximamente el siguiente agente insertado
                                indice_mascara = indice_mascara + 1
                                num_neuronas = np.sum(neuronas)
                            else:
                                #Si no tiene un agente asociado...
                                ultima_capa_tiene_mascara = False
                                num_neuronas = capa.units
                            #Ahora instanciamos la capa
                            nueva_capa = tipo_capa(units=num_neuronas,activation=capa.activation,use_bias=capa.use_bias)
                            lista_capas.append(nueva_capa)
                            #Guardamos los pesos/bias que se asignarán a dicha capa
                            self._parametros_a_copiar.append([pesos,sesgos])
                        #Tras procesar todas las capas del modelo secuencia, construimos el modelo secuencial nuevo
                        fase_modificada = tf.keras.Sequential(lista_capas)
                    else:
                        #Dicha fase no es un modelo secuencial, sino una sola capa (no tendrá asociado ningun agente)
                        #print("\nEsta fase es una sola capa")
                        #Tenemos que comprobar si la ultima capa procesada en la fase previa tenia un agente asociado
                        tipo_capa = type(fase)
                        config = fase.get_config()
                        parametros = fase.get_weights()
                        kernel, recurrent_kernel, bias = parametros
                        #print(fase.get_weights())
                        nueva_capa = tipo_capa.from_config(config)
                        #self._input_shape.append(fase.get_build_config()['input_shape'])
                        #Construimos los parámetros de la capa (.build) pero viendo si la ultima capa tenía un agente
                        #print(f'¿La capa previa a esta tenía una máscara asociada? --> {ultima_capa_tiene_mascara}')
                        #print(f'¿La capa tiene una máscara/agente asociado?: {getattr(capa,atributo_especial)}')
                        if ultima_capa_tiene_mascara:
                            #Debemos de modificar el kernel que usará la nueva GRUCell
                            kernel = kernel[tam_dim_entrada,:]
                            input_shape = tf.TensorShape([None,int(np.sum(tam_dim_entrada))])
                        else:
                            input_shape = fase.get_build_config()['input_shape']
                        self._input_shape.append(input_shape)
                        #Tras finalizar...
                        ultima_capa_tiene_mascara = False

                        #Asignamos los valores de los pesos a la capa ya inicializada y construida
                        self._parametros_a_copiar.append([kernel,recurrent_kernel,bias])
                        fase_modificada = nueva_capa
                    #Tras procesar toda la fase...
                    setattr(self,nombre_fase,fase_modificada)
                else:
                    print(f'No se ha localizado la fase {fase}')
            #Fin del procesamiento de las diferentes fases
            #print("\nSe ha reconstruido el modelo base pero comprimiéndolo")
    
    def build(self, input_shape = None):
        #print("\nConstruyendo las capas del modelo base ahora comprimido")

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
                        capa.set_weights(self._parametros_a_copiar[ind_params])
                        ind_params += 1
            else:
                #La fase está formada por una sola capa
                fase.build(input_shape=self._input_shape[ind_input])
                ind_input += 1
                #Tras construir la capa, copio los valores a los parámetros de la capa
                fase.set_weights(self._parametros_a_copiar[ind_params])
                ind_params += 1

        #Indicamos que se ha construido el modelo DECORE
        self.built = True
        #print("Modelo comprimido reconstruido con éxito")

        del self._input_shape
        del self._parametros_a_copiar

    
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