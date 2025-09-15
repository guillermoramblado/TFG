import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    #Ruta donde se almacenarán los gráficos asociados al proceso de entrenamiento
    dir_img_train = "./Imagenes/Entrenamiento"
    #Ruta del archivo de log correspondiente a partir del cual se desean generar las gráficas
    ruta_log = "./Logs/training.txt"
    #Nombre del modelo entrenado
    nombre_modelo = "DECORE-2.2"


    errores_entrenamiento = []
    with open(ruta_log) as fp:
        for linea in list(fp):
            palabras = linea.split(",")
            if palabras[0] == "VAL":
                break
            else:
                errores_entrenamiento.append(float(palabras[0]))

    print("Se han extraído los datos del fichero de logs correctamente...")

    print("\nComenzando a generar las gráficas del proceso de entrenamiento del modelo...")
    
    #Creamos las gráficas con los errores de entrenamiento del actor/crítico

    nombre_imagen = nombre_modelo + "_errores.png"
    plt.figure(figsize=(10,5))
    plt.plot(errores_entrenamiento,label="Error Entrenamiento " + nombre_modelo)
    plt.xlabel("Episodio")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_img_train,nombre_imagen))
    plt.close()

    print("Gráficas creadas correctamente")
