import actor
import tensorflow as tf
import numpy as np
from DECORE import DECORE 
from apagado_aleatorio import ApagadoAleatorio
from compresion_final import ModeloBaseComprimido
import os
import json

#Parámetros necesarios para inicializar el modelo base
SEED = 9
hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.01, 
    'T': 5,
}

# === Configuración de modelos DECORE ===
dir_check_decore = "./DECORE/"
archivos_check_decore = ["decore_0.0","decore_2.2"]
logs_decore = ["decore_0.0.txt","decore_2.2.txt"]
nombre_modelos_decore = ["DECORE_0.0","DECORE_2.2"]

# === Configuración de modelos Apagado Aleatorio ===
dir_check_apagado_aleatorio = "./ApagadoAleatorio/"
archivos_check_apagado_aleatorio = ["apagado_10%","apagado_20%","apagado_30%","apagado_40%","apagado_50%"]
logs_apagado_aleatorio = ["apagado_aleatorio_10%.txt","apagado_aleatorio_20%.txt","apagado_aleatorio_30%.txt","apagado_aleatorio_40%.txt","apagado_aleatorio_50%.txt"]
nombre_modelos_apagado_aleatorio = ["APAGADO_ALEATORIO_10%","APAGADO_ALEATORIO_20%","APAGADO_ALEATORIO_30%","APAGADO_ALEATORIO_40%","APAGADO_ALEATORIO_50%"]

# === Directorio de logs de los modelos a comparar ===
dir_logs = "./Logs/"


lista_checkpoints = []
lista_logs = []
#Añadimos primero la ruta de los checkpoints y logs de los diferentes modelos DECORE
for check,log in zip(archivos_check_decore,logs_decore):
    lista_checkpoints.append(dir_check_decore+check)
    lista_logs.append(dir_logs+log)
#Ahora lo mismo pero para los diferentes porcentajes de apagados aleatorios aplicados
for check,log in zip(archivos_check_apagado_aleatorio,logs_apagado_aleatorio):
    lista_checkpoints.append(dir_check_apagado_aleatorio+check)
    lista_logs.append(dir_logs+log)

nombre_modelos = nombre_modelos_decore+nombre_modelos_apagado_aleatorio



#Ruta del directorio donde se guardarán todos los modelos comprimidos usando DECORE, Apagado Aleatorio...
dir_modelos_finales = "./ModelosComprimidos"
os.makedirs(dir_modelos_finales,exist_ok=True)

#Comparamos los resultados obtenidos en validación usando decore-0.0, decore-2.2, apagado_aleatorio_10%, apagado_aleatorio_20%
if __name__ == "__main__":
    print("Imprimiendo la lista con las rutas de las versiones comprimidas a comparar")
    print(lista_checkpoints)

    print("\nImprimiendo la lista de los correspondientes logs")
    print(lista_logs)

    print("\nImprimiendo el nombre de los modelos a comparar")
    print(nombre_modelos)

    #Primero comenzaré construyendo el modelo base que he comprimido
    print("\nConstruyendo el modelo base")
    modelo_base = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
    modelo_base.build()
    num_params_base = modelo_base.count_params()
    print(f'Construcción finalizada. Número de parámetros: {num_params_base}')

    #Creamos el diccionario de las diferentes versiones comprimidas a comparar
    modelos_comprimidos = {}
    for nombre, check, log in zip(nombre_modelos,lista_checkpoints,lista_logs):
        print(f'\nProcesando el modelo {nombre}')
        #Comenzamos tomando el peor uso obtenido y la recompensa futura acumulada media
        with open(log,"r") as fileLog:
            for line in reversed(list(fileLog)):
                valores = line.split(",")
                if valores[0] == "MAX REWD":
                    uso = round(float(valores[1]),4)
                    recompensa = round(float(valores[2]),4)
                    break
        print(f'Uso: {uso} - Recompensa: {recompensa}')

        #input("\nPresione ENTER para visualizar el ckpt...")
        #print(tf.train.list_variables(check))
        #Ahora restauramos el estado del modelo con los agentes o mascaras insertadas, y removemos las neuronas
        if nombre.startswith("DECORE"):
            #Estamos ante un modelo DECORE
            es_decore = True
            modelo_intermedio = DECORE(modelo_base=modelo_base)
            modelo_intermedio.build()
            print("Se ha construido el modelo DECORE")
        else:
            #Estamos ante un modelo APAGADO_ALEATORIO
            es_decore = False
            modelo_intermedio = ApagadoAleatorio(modelo_base=modelo_base)
            modelo_intermedio.build()
            print("Se ha construido el modelo Apagado Aleatorio")

        #Restauramos el estado del mejor modelo decore encontrado
        checkpoint = tf.train.Checkpoint(model=modelo_intermedio)
        checkpoint.restore(check)
        print("Se ha restablecido el estado del modelo")
        
        input("Presione ENTER para deconstruir el modelo")
        #Eliminamos las neuronas que se han decidido eliminar
        modelo_final = ModeloBaseComprimido(modelo_intermedio,es_decore)
        modelo_final.build()
        print(f'Se ha reconstruido el modelo base pero comprimido. Número de parámetros: {modelo_final.count_params()}')
        input("Presione ENTER para guardar el estado final del modelo comprimido")
        #Guardo el estado del modelo base comprimido
        checkpoint = tf.train.Checkpoint(model=modelo_final)
        checkpoint.write(os.path.join(dir_modelos_finales,nombre))
        print("Modelo guardado con éxito")

        num_params_comprimido = modelo_final.count_params()
        porcentaje_compresion = np.round((1-(num_params_comprimido/num_params_base))*100,2)

        modelos_comprimidos[nombre] = {
            "PorcentajeCompresion" : porcentaje_compresion,
            "Uso" : uso,
            "Recompensa": recompensa
        }
    #Escribimos el diccionario en un archivo .json
    with open("resultados_compresion.json", "w") as f:
        json.dump(modelos_comprimidos, f, indent=4)