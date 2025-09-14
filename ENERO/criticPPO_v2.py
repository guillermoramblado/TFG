# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams, hidden_init_critic, kernel_init_critic):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_critic,
                                            activation=tf.nn.selu, name="FirstLayer"))

        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        self.ReadoutGRU = tf.keras.layers.GRU(self.hparams['readout_units'],dtype=tf.float32)
        self.Readout = keras.layers.Dense(1, kernel_initializer=kernel_init_critic, name="Readout")

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        #self.ReadoutGRU.build(input_shape=tf.TensorShape([None, self.hparams['T'], self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['readout_units']])
        self.built = True

    #@tf.function
    def call(self, link_state, first_critic, second_critic, num_edges_critic, training=False):

        # Execute T times
        secuencia_estados = []
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main nodes with the neighbours
            mainNodes = tf.gather(link_state, first_critic)
            neighNodes = tf.gather(link_state, second_critic)

            nodesConcat = tf.concat([mainNodes, neighNodes], axis=1)

            ### 1.a Message passing for node link with all it's neighbours
            outputs = self.Message(nodesConcat)

            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=second_critic, num_segments=num_edges_critic)

            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]
            #Añadimos los nuevos estados de los enlaces del grafo en el instante actual
            secuencia_estados.append(link_state)
        
        #Fuera del for.... ---> Apilamos los tensores de la lista a lo largo del eje 0 (filas). 
        #Tendríamos un tensor de dimensiones (T,enlaces, link_state_dim)
        secuencia_estados = tf.stack(secuencia_estados, axis=0)
        #print(f'------------El tamaño despues del STACK es {secuencia_estados.shape}---------------')
        #Trasponemos las dos primeras dimensiones, obteniendo (enlaces,T,link_state_dim)
        secuencia_estados = tf.transpose(secuencia_estados,perm=[1,0,2])
        #Evaluamos la GRU, obteniendo para cada enlace --> vector de dimensión 'readut_units', es decir, tensor de salida (enlaces,readout_units)
        #Por defecto, el estado inicial de cada secuencia es un estado de ceros
        historial = self.ReadoutGRU(secuencia_estados,training=True)
        #Sumamos la información recopilada (tras la GRU) de los enlaces del único grafo de trabajo
        edges_combi_outputs = tf.math.reduce_sum(historial, axis=0)
        #Reshape del tensor de salida (unidimensional) para que sea un tensor bidimensional, y poder pasarselo como entrada a Readout
        edges_combi_expanded = tf.expand_dims(edges_combi_outputs,axis=0)
        #Se intenta representar mediante un único valor numérico cada grafo candidato (Desvío)
        r = self.Readout(edges_combi_expanded,training=training)
        return r