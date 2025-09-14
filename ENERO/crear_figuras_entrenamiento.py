import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

#Ejemplo: python crear_figuras_entrenamiento.py -d ./Logs/expEnero_3top_15_B_NEWLogs.txt -n ModeloBase
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsear archivo y crear gráficas")
    parser.add_argument('-d', help='registro del proceso de entrenamiento',type=str,required=True)
    parser.add_argument('-n',help='nombre del modelo asociado',type=str,required=True)
    args = parser.parse_args()
    '''
    Acceso al registro del proceso de entrenamiento, y tomo:
        * Error medio del actor y crítico en entrenamiento
        * Valor de la métrica al validar al final de cada episodio
    '''

    ruta_log = args.d
    nombre_modelo = args.n
    print(f'El nombre de la imagen generada será {nombre_modelo}')
    print(f'La ruta del archivo de logs es {ruta_log}')

    if not os.path.exists(ruta_log):
        print("No existe ningún fichero log en la ruta especificada...")
        sys.exit(-1)

    #Comprobamos si existe la carpeta de imágenes donde se almacenerán las gráficas del proceso de entrenamiento de cualquier modelo
    dir_img_train = "./Imagenes/Entrenamiento/" + nombre_modelo
    if not os.path.exists(dir_img_train):
        os.makedirs(dir_img_train)
    
    
    errores_actor = []
    errores_critico = []
    uso_enlace_mayor_uso = []
    recompensa_futura_acumulada = []

    with open(ruta_log) as fp:
        for linea in list(fp):
            palabras = linea.split(",")
            etiqueta = palabras[0]
            if etiqueta == 'a':
                #Pérdida del actor al final de cada episodio
                errores_actor.append(float(palabras[1]))
            elif etiqueta == 'c':
                #Pérdida media del crítico en un episodio
                errores_critico.append(float(palabras[1]))
            elif etiqueta == '<':
                #Peor valor de métrica obtenido tras validar
                uso_enlace_mayor_uso.append(float(palabras[1]))
            elif etiqueta == 'REW':
                #Recompensa furua acumulada media obtenida tras validar
                recompensa_futura_acumulada.append(float(palabras[1]))
            else:
                continue #No hago nada

    print("\nComenzando a generar las gráficas del proceso de entrenamiento del modelo...")
    
    #Creamos las gráficas con los errores de entrenamiento del actor/crítico
    nombre_imagen = "errores_actor_critico.png"
    plt.figure(figsize=(10,5))
    plt.plot(errores_actor,label="Pérdida del actor")
    plt.plot(errores_critico,label="Pérdida del crítico")
    plt.xlabel("Episodio Entrenamiento")
    plt.ylabel("Error cometido")
    plt.title(f'Errores Entrenamiento Actor/Crítico')
    plt.legend() #Para hacer visible la legenda en dicha figura
    plt.tight_layout()
    plt.savefig(os.path.join(dir_img_train,nombre_imagen))
    plt.close()

    #Graficamos el peor uso del enlace de mayor congestión obtenido tras validar al final de cada episodio
    nombre_imagen = "metrica.png"
    plt.figure(figsize=(10,5))
    plt.plot(uso_enlace_mayor_uso)
    plt.xlabel("Episodio")
    plt.ylabel("Uso obtenido")
    plt.title("Uso del enlace de mayor congestión al entrenar")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_img_train,nombre_imagen))
    plt.close()

    #Recompensa futura acumulada media obtenida tras aplicar el agente sobre las diferentes matrices de validación, al final de cada episodio
    nombre_imagen = "recompensa.png"
    plt.figure(figsize=(10,5))
    plt.plot(recompensa_futura_acumulada)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa obtenida")
    plt.title("Recompensa futura acumulada media")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_img_train,nombre_imagen))
    plt.close()

    print("\nGráficas creadas correctamente")
