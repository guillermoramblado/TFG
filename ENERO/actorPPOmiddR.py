# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

#Esta clase modela el actor
class myModel(tf.keras.Model):
    '''
    Método que nos permite comenzar definiendo la arquitectura del actor, sin especificar la dimensión de entrada de cada capa (eso se hace en el método build())
        *  hparams: diccionario con parámetros de configuración
        * hidden_init_actor y kernel_init_actor: valores iniciales de los parámetros del modelo del actor
    '''
    def __init__(self, hparams, hidden_init_actor, kernel_init_actor):
        super(myModel, self).__init__()
        self.hparams = hparams

        # --- Primera parte del modelo usada para el ENVÍO DE MENSAJES
        self.Message = tf.keras.models.Sequential()
        #Se usa esa dimensión como salida de la capa ya que esta capa devuelve el mensaje que envía un determinado enlace emisor a un determinado enlace receptor. Dicho mensaje se modela como un vector de
        # 'link_state_dim' componentes, para que luego se pueda utilizar junto con el vector de estados oculto del enlace receptor a la hora de actualizar su estado.
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_actor,
                                            activation=tf.nn.selu, name="FirstLayer"))
        #Segunda parte: actualizar el vector de estados oculto de cada enlace a partir de los mensajes recibidos. Esta capa devuelve justamente el vector de estados oculto actualizado del enlace receptor.
        #Se comienza indicando el tamaño del estado oculto 'h'
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        #La fase de interpretación de la información acumulada en dicho grafo se realiza aquí, mediante tres capas densas:
        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        #self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        #self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        #La última capa densa tendrá una sola neurona, devolviendo el valor con el que se representa cada grafo candidato
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3"))
        '''
        Dado un estado concreto del entorno (y la demanda a enrutar), el actor se aplicará tantas veces como desvíos candidatos se puedan realizar.
        De esta forma, para cada desvío (modelado mediante el grafo inicial marcando el camino del desvío), el agente representará dicho grafo mediante un valor.
        '''

    '''
    El método 'init' se encarga de definir la arquitectura del modelo, indicando la dimensión de salida de cada capa.
    El método 'build' nos permite modificar/especificar la dimensión de entrada para cada una de las fases
    '''
    def build(self, input_shape=None):
        # Create the weights of the layer
        #Se especifica la dimensión de entrada tanto para la fase de envío de mensajes, actualización y representación del grafo con un único valor
        
        '''
        Para el envío de mensajes se requiere trabajar sobre el vector de estados oculto tanto del enlace receptor como de los emisores.
        Esto es, se debe pasar como entrada un tensor bidimensional:
            * Primera dimensión: 'None', esto es, no se fija el tamaño o nº de indices concretos que se emplean en esta primera dimensión. 
              Seguramente la primera dimensión refleje los diferentes enlaces emisores, dado un enlace receptor concreto.
            * Segunda dimensión: se usan ['link_state_dim' * 2] índices. Seguramente, esta dimensión sea fruto de concatenar el vector de estados oculto de un enlace emisor y el enlace receptor fijo.

        Esto es, la entrada será una matriz, con tantas filas como enlaces emisores haya (indeterminado), y cada fila contiene el vector de estados oculto del enlace emisor y receptor, que es lo que se necesita
        para modelar el  mensaje que se envía.
        '''
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))

        '''
        Como es una GRUCell, a diferencia de una GRU, ésta GRUCell procesa la información de un solo token de la secuencia, a diferencia de una GRU, que procesa de golpe una secuencia de tokens
            * En este caso, se hace así porque para cada enlace, se pasará como entrada una secuencia de un solo token: la agregación de los mensajes recibidios por su vecino.
            * Se usará como estado oculto inicial --> vector de estado del enlace (antes de actualizar)
        
            - Se obtiene como salida, un tensor de dimensiones [None, 'link_state_dim'], obteniendo para cada elemento del lote (para cada enlace) su vector de estados actualizado
        '''
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))


        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    '''
    Cada vez que queramos procesar el supergrafo con los grafos asociados a los diferentes desvíos candidatos usando nuestro actor (GNN), se realizará el siguiente proceso: FORWARD PROPAGATION
    Se debera pasar al actor los siguientes parámetros de entrada:
    self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'],tensor['num_edges'])

    El proceso es el siguiente:
    A lo largo de un periodo temporal de longitud T:
        1. Se generan y suman los mensajes que recibe cada enlace de sus vecinos
        2. Se actualiza usando GRUCell que trabaja, para cada enlace, con una secuencia temporal de longitud 1 (mensaje agregado) y con el estado de dicho enlace como estado oculto inicial, obteniendo así,
        para cada enlace, el estado oculto de salida, que es el nuevo estado de dicho enlace
        3. Se vuelve a repetir el PASO 1, generando los nuevos mensajes que recibe cada enlace de sus vecinos, considerando su nuevo estado tras el paso 2
    '''
    # training= ... --> Indicar si el modelo está usandose para entrenar o para inferir, y hacer uso así de las capas de Dropout que se hayan insertado
    def call(self, link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours

            #Se toma el vector de estados oculto de los enlaces emisores de mensajes, para cada grafo con desvío candidato aplicado (y trabajando sobre la matriz de vectores de estados oculto global)
            #Esto se puede hacer debido a que preprocesamos el tensor que recibe como entrada el modelo, desplazando las posiciones de los emisores para poder trabajar sobre 'link_state_dim' global
            mainEdges = tf.gather(link_state, states_first)
            #Se toma el vector de estados oculto de los enlaces receptores
            neighEdges = tf.gather(link_state, states_second)

            #Para generar el mensaje, se deberá pasar como entrada el vector de estados oculto tanto del emisor como del receptor (se concatenan, tal que cada fila contiene lo indicado previamente)
            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat)
            #Obtenemos los mensajes generados para cada pareja emisor-receptor, obteniendo un tensor de salida de dimensiones (pares emisor-receptor, mensaje generado)

            '''
            Ahora toca sumar, para cada enlace, los mensajes que ha recibido...
            Lo que haremos será sumar los mensajes (vectores) que recibe un mismo enlace receptor. Esto es, habrá que ir sumando todas las filas de 'outputs' que reflejen los mensajes enviados a un mismo enlace
            recpetor. Se realizará por tanto una suma segmentada, sumando aquellas filas que peretenezcan a un mismo segmento (enlace receptor)
            '''
            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)


            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            '''
            Para actualizar, se le pasa como entrada a la capa de actualización (GRUCell):
                * Tensor bidimensional con la suma de los mensajes que recibe cada enlace, de cada grafo con desvío candidato
                * Lista con un solo elemento: matriz global de vectores de estados oculto inicial de cada enlace de cada grafo
            Obtenemos como salida:
                * outputs: salida de la GRUCell, que en este caso coincide con el nuevo estado oculto 'h'
                * link_state_list: lista de un solo elemento (tensor), que contiene el nuevo estado oculto 'h' (nuevo estado) de cada uno de los enlaces
            '''
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            #Nuevo vector de estados oculto de cada enlace del grafo
            link_state = links_state_list[0]

        #Tras la fase de envío de mensajes, se procede a sumar el vector de estados de los enlaces, para cada grafo candidato.
        #Sumamos los estados de cada uno de los enlaces (por cada grafo candidato) tras el período de envío de mensajes, e intentamos obtener una valoración de dicho grafo candidato
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)
        
        #Se intenta representar mediante un único valor numérico cada grafo candidato (Desvío)
        r = self.Readout(edges_combi_outputs,training=training)
        return r
