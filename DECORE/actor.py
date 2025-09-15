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

        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            kernel_initializer=hidden_init_actor,
                                            activation=tf.nn.selu, name="FirstLayer"))
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_initializer=hidden_init_actor,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dense(1, kernel_initializer=kernel_init_actor, name="Readout3"))

    def build(self, input_shape=None):
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    def call(self, link_state, states_graph_ids, states_first, states_second, sates_num_edges, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            outputs = self.Message(edgesConcat)
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)


            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]

        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)
        
        r = self.Readout(edges_combi_outputs,training=training)
        return r
