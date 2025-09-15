- Terminar de ver por qué no restaura el modelo
- Definir la clase DECORE
- Definir función de pérdida

--- Sequential() ---- 
* Hereda de la clase base 'Layer' y tb 'Model'
* Como hereda de Model ---> Nos ofrece el atributo 'layers', que contiene las capas más externas de dicho modelo o sequential

------  Nombres de las capas (tf.keras.Layer) -----

Imaginemos que nuestro modelo contiene lo siguiente:
 1. Sequential() con tres capas Denses 
 2. Sequential() con dos capas Dense
Todo ello sin especificar en ningún momento un nombre

Sus nombres internos serían:

sequentials
    dense
    dense_1
    dense_2

sequential_1
    dense_3
    dense_4

Para mi ejemplo, mi modelo consta de:
    1. Sequential() sin especificar un nombre concreto :
        Dense(name='FirstLayer')
    2. GRUCell sin especificar nombre a la capa
    3. Sequence() sin especificar nombre:
        Tres capas densas con nombres Readout1, Readout2, Readout3

Esto devolvería los nombres:

sequential
    FirstLayer
gru_cell
    Readout1
    Readout2
    Readout3


--------- tf.keras.Layer ----------

* get_config(self) : que devuelve un diccionario con la configuración usada para inicializar la capa (__init__)
* get_build_config(): diccionario con info usada para construir la capa (método build()), con valores como input_shape


https://www.tensorflow.org/api_docs/python/tf/keras/Layer
