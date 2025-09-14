import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import time as tt
#import resource
import subprocess
import csv
import pandas as pd
import matplotlib.pyplot as plt

max_iters = 3000 # Total number of training episodes
episode_iters = 20 # How many training episodes to execute before the training script is called again

# NOTICE: The training script trains and stores the models every 20 episode_iters. When the batch of episode_iters
# finishes, the model is stored and in the next itreation is loaded again to start the training were it was left before.
# This is to avoid some memory leak issue existing with TF

'''
Este archivo es el que permite entrenar el agente DLR, realizando un total de 'max_iters' epocas. En este caso, se entrena usando
lotes de épocas de tamaño 'episode_iters' por tema memoria

'''


if __name__ == "__main__":


    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    
    #Directorio temporal creado al comenzar la ejecución de este archivo, y justo antes de comenzar a entrenar el modelo.
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    iters = 0
    counter_store_model = 1
    #Nombre de las topologías usadas en el proceso de entrenamiento
    dataset_folder_name1 = "NEW_BtAsiaPac"
    dataset_folder_name2 = "NEW_Garr199905"
    dataset_folder_name3 = "NEW_Goodnet"

    while iters < max_iters:
        processes = []
        '''
        Pasamos como argumentos:
            * iters --> nuevo episodio por el que vamos en el entrenamiento
            * counter_store_model --> la nueva versión del modelo que se va a obtener
            * episode_iters ---> tamaño de lote de episodios (cuántos episodios procesar en el script train_Enero_3top_script.py)
        '''
        
        #subprocess.call(['python train_Enero_3top_script.py -i '+str(iters)+ ' -c '+str(counter_store_model)+' -e '+str(episode_iters)+ ' -f1 '+dataset_folder_name1+' -f2 '+dataset_folder_name2+' -f3 '+dataset_folder_name3], shell=True)
        #cmd = f"python train_Enero_3top_script_compresion.py -i {iters} -c {counter_store_model} -e {episode_iters} -f1 {dataset_folder_name1} -f2 {dataset_folder_name2} -f3 {dataset_folder_name3}"
        cmd = f"python train_Enero_3top_script.py -i {iters} -c {counter_store_model} -e {episode_iters} -f1 {dataset_folder_name1} -f2 {dataset_folder_name2} -f3 {dataset_folder_name3}"
        subprocess.call(cmd, shell=True)
        #usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        counter_store_model = counter_store_model + episode_iters
        iters = iters + episode_iters






