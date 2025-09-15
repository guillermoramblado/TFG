import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import copy
from keras import regularizers
#Modelo del agente que se inserta tras capa Densa, GRU's...
class Agente(tf.keras.layers.Layer):
    def __init__(self,id,dimension_estado):
        super().__init__(name=f'Agente_{id}')
        #Parámetros que conforman dicha capa, que serán los pesos que asigna el agente a las activaciones de la capa de trabajo
        #self.params = self.add_weight(name=f'params_{id}',shape=(dimension_estado,),initializer=keras.initializers.Constant(0.0),trainable=True)
        self.params = self.add_weight(name=f'params_{id}',shape=(dimension_estado,),initializer=keras.initializers.Constant(2.2),trainable=True,regularizer=regularizers.l2(0.0001))
        self.Pi = None #Tensor unidimensional que almacena la probabilidad de mantener cada neurona
        #self.A = tf.ones((dimension_estado,), dtype=tf.float32) #Tensor unidimensional con la acción que aplica el agente para cada neurona (mantener o preservar)
        self.A = tf.Variable(tf.ones((dimension_estado,), dtype=tf.float32), trainable=False) #Se define como un tensor variable para que se guarde automáticamente al hacer un checkpoint.write()
        self.R = None #Recompensa que consigue el agente en cuanto a la compresión alcanzada

    #Método encargado de definir las acciones que aplicará el agente en el próximo paso que de.
    #El parámetro 'training' se activará para indicar si queremos usar políticas estocásticas
    def fijar_acciones(self,training):
        self.Pi = tf.sigmoid(self.params)
        if training:
            #Las políticas aplicadas serán estocásticas (sigue una política para cada neurona)
            bernoullis = tfp.distributions.Bernoulli(probs=self.Pi, dtype=tf.float32)
            acciones = bernoullis.sample()
        else:
            #Modo inferencia --> políticas deterministas
            acciones = tf.cast(self.Pi >= 0.5, tf.float32)
        acciones = tf.stop_gradient(acciones)
        self.A.assign(acciones)
        self.R = tf.reduce_sum(1.0-self.A)

    #El agente recibirá como entrada las activaciones de la capa previa: [ enlaces(juntando todos los grafos candidatos), dimension_estado ]
    def call(self, inputs):
        #Aplicamos sobre las activaciones de la capa previa la máscara obtenida
        mascara = tf.reshape(self.A,[1,-1])
        return(inputs*mascara) #Producto elemento a  elemento usando broadcasting explícito



#Tupla con las capas a localizar dentro del modelo base
capas_buscadas = (tf.keras.layers.Dense,)
class DECORE(tf.keras.Model):
    def __init__(self,modelo_base):
        super(DECORE,self).__init__() #Nombre del modelo
        self.hparams = modelo_base.hparams
        self.Message = None
        self.Update = None
        self.Readout = None
        self.agentes = [] #Lista de los agentes insertados

        #Valores que son necesarios guardar para construir posteriormente el modelo DECORE
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
                        nueva_capa.agente_asociado = False
                        lista_capas.append(nueva_capa)
                        #Compruebo si es una de las capas buscadas
                        #La única excepción es que no se puede introducir un agente en la cabecera del modelo, es decir, en la última capa de la MPNN (situada en Readout)
                        if isinstance(capa,capas_buscadas) and not(fase=='Readout' and posicion==(len(atributo.layers)-1)):
                            nuevo_agente = Agente(next_id,capa.units)
                            self.agentes.append(nuevo_agente)
                            lista_capas.append(nuevo_agente)
                            next_id+=1
                            #Indicamos en la capa previa que se le ha asignado un agente a dicha capa
                            nueva_capa.agente_asociado = True
                            
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
                    nueva_capa.agente_asociado = False
                    #Me guardo los parámetros de la capa empleada en dicha fase
                    self._input_shape.append(atributo.get_build_config()['input_shape'])
                    self._parametros_modelo_base.append(atributo.get_weights())
                    #Comprobemos que es una de las capas buscadas, y en caso afirmatico, que estemos en Readout (no tiene sentido añadir un agente a la última capa del modelo)
                    if isinstance(atributo, capas_buscadas) and fase!='Readout':
                        #Me creo un Sequential con la capa base y la capa 'Agente'
                        nuevo_agente = Agente(next_id,atributo.units)
                        self.agentes.append(nuevo_agente)
                        #Indicamos en la capa que tiene asignado un agente
                        nueva_capa.agente_asociado = True
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
        print("\nConstruyendo las capas del modelo DECORE...")

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
                    if not isinstance(capa,Agente):
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
        print("Modelo DECORE construido con éxito")

        del self._input_shape
        del self._parametros_modelo_base
    
    def fijar_acciones(self,training):
        for agente in self.agentes:
            agente.fijar_acciones(training=training)
            
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


if __name__ == "__main__":
    
    #Probemos que el agente se ha modelado sin problemas
    agente = Agente(1,8)
    entrada = tf.random.normal([2,8])
    salida = agente(entrada)
    print(f'Entrada del modelo : {entrada.numpy()}')
    print(f'Salida del modelo es {salida.numpy()}')
    print(f'Parámetros entrenables: {agente.trainable_weights}')
    print("\nProbabilidades (Pi):", agente.Pi.numpy())
    print("Acción (A):", agente.A.numpy())
    print("Recompensa (R):", agente.R)
    






        