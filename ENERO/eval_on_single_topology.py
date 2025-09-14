import os
import subprocess
import argparse
from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
import multiprocessing

'''
Este script nos permite evaluar el rendimiento del modelo entrenado sobre una topología concreta, considerando todas las matrices de tráfico de evaluación. 
Para ello, será necesario
previamente ejecutar 'convert_dataset.py' para separar las matrices de tráfico de dicha topología en TRAIN y VALIDATION.
'''

'''
Método que recibe una serie de parámetros, como:
    * Versión del modelo entrenado que se quiera usar para evaluar (nos encargamos de pasarle siempre la versión del mejor modelo, es decir, aquella con mejor rendimiento encontrado)
    * Id de la matriz de tráfico que se quiere usar en el grafo de evaluación
    ... Entre otros
y que se encarga de evaluar el modelo especificado sobre una red y matriz de tráfico concreta
'''

def worker_execute(args):
    tm_id = args[0]
    model_id = args[1]
    drl_eval_res_folder = args[2]
    differentiation_str = args[3]
    graph_topology_name = args[4]
    general_dataset_folder = args[5]
    specific_dataset_folder = args[6]
    #Invocamos ese .py para evaluar el modelo sobre una cierta topología, y considerando una matriz de tráfico
    #subprocess.call(["python script_eval_on_single_topology.py -t "+str(tm_id)+" -m "+str(model_id)+" -g "+graph_topology_name+" -o "+drl_eval_res_folder+" -d "+differentiation_str+ ' -f ' + general_dataset_folder + ' -f2 '+specific_dataset_folder], shell=True)
    cmd = f"python script_eval_on_single_topology.py -t {tm_id} -m {model_id} -g {graph_topology_name} -o {drl_eval_res_folder} -d {differentiation_str} -f {general_dataset_folder} -f2 {specific_dataset_folder}"
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    # First we execute this script to evaluate our drl agent over different topologies from the folder (argument -f2)
    # python eval_on_single_topology.py -max_edge 100 -min_edge 5 -max_nodes 30 -min_nodes 1 -n 2 -f1 results_my_3_tops_unif_05-1 -f2 NEW_Garr199905/EVALUATE -d ./Logs/expSP_3top_15_B_NEWLogs.txt
    # To parse the results of this script, we must then execute the parse_middrouting_files.py file
    
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    #Archivo de logs del modelo entrenado, donde se registró el proceso de entrenamiento del modelo y donde se tomará el id del mejor modelo 
    parser.add_argument('-d', help='logs data file', type=str, required=True, nargs='+')
    #Directorio padre donde se localizan las carpetas de las diferentes topologías de trabajo
    parser.add_argument('-f1', help='Dataset name within dataset_sing_top', type=str, required=True, nargs='+')
    #Carpeta específica de la topología sobre la que queremos validar
    parser.add_argument('-f2', help='specific dataset folder name of the topology to evaluate on', type=str, required=True, nargs='+') #NEW_Garr199905/EVALUATE
    parser.add_argument('-max_edge', help='maximum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_edge', help='minimum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-max_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-n', help='number of processes to use for the pool (number of DEFO instances running at the same time)', type=int, required=True, nargs='+')

    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp") 
    differentiation_str = str(aux[1].split("Logs")[0]) #Esto almacenaria por tanto SP_3top_15_B_NEW

    # Point to the folder were the datasets of argument f2 are located
    #EJEMPLO --> ../ENERO_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/NEW_Garr199905/EVALUATE/
    general_dataset_folder = "../Enero_datasets/dataset_sing_top/data/"+args.f1[0]+"/"+args.f2[0]+"/"

    # In this folder we store the rewards that later will be parsed for plotting
    #Ruta donde se almacenarán  los resultados de la evaluación de dicha topología
    #EJEMPLO --> ../ENERO_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/evalRes_NEW_Garr199905/EVALUATE/
    drl_eval_res_folder = "../Enero_datasets/dataset_sing_top/data/"+args.f1[0]+"/evalRes_"+args.f2[0]+"/"

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    if not os.path.exists(drl_eval_res_folder):
        os.makedirs(drl_eval_res_folder)

    #Ruta destino: ../ENERO_datasetss/dataset_sing_top/data/results_my_3_tops_unif_05-1/evalRes_NEW_Garr199905/EVALUATE/SP_3top_15_B_NEW
    if not os.path.exists(drl_eval_res_folder+differentiation_str): 
        os.makedirs(drl_eval_res_folder+differentiation_str)
    else:
        os.system("rm -rf %s" % (drl_eval_res_folder+differentiation_str))
        os.makedirs(drl_eval_res_folder+differentiation_str)

    model_id = 0
    # Load best model
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break

    # Iterate over all topologies and evaluate our DRL agent on all TMs
    #Recorremos de forma recursiva el directorio con la info de dicha topología (que contiene internamente los directorios ALL, EVALUATION y TRAIN)
    '''
    Cada iteración nos devuelve una tupla (directorio donde nos encontramos hasta el momento, directorios que contiene el directorio, archivos que contiene el directorio)
    '''
    for subdir, dirs, files in os.walk(general_dataset_folder):
        for file in files:
            if file.endswith((".graph")):
                topology_num_nodes = 0
                with open(general_dataset_folder+file) as fd:
                    # Loop to read the Number of NODES and EDGES
                    while (True):
                        line = fd.readline()
                        if (line == ""):
                            break
                        if (line.startswith("NODES")):
                            topology_num_nodes = int(line.split(' ')[1])

                        # If we are inside the range of number of nodes
                        if topology_num_nodes>=args.min_nodes[0] and topology_num_nodes<=args.max_nodes[0]:
                            if (line.startswith("EDGES")):
                                topology_num_edges = int(line.split(' ')[1])
                                # If we are inside the range of number of edges
                                if topology_num_edges<=args.max_edge[0] and topology_num_edges>=args.min_edge[0]:
                                    topology_Name = file.split('.')[0]
                                    print("*****")
                                    print("***** Evaluating on file: "+file+" with number of edges "+str(topology_num_edges))
                                    print("*****")
                                    '''
                                    Lista de tuplas, una tupla por cada una de las 50 matrices de validación disponibles para cualquier topología  
                                    Cada tupla contiene:
                                        * tm_id --> Id de la matriz de tráfico de validación (de 0 a 49)
                                        * model_id --> Versión del agente empleado (tomada anteriormente, en este caso, la mejor versión encontrada a lo largo del proceso de entrenamiento) 
                                        * drl_eval_res_folder --> Ruta donde se almacenarán los resultados de la evaluación del agente en la topología concreta
                                        * differentiation_str --> SP_3top_15_B_NEW
                                        * topology_Name --> Nombre de la toplogía
                                        * general_dataset_folder --> Ruta de la carpeta de la topología sobre la que queremos evaluar (con matrices de validacion)
                                        * args.f2[0] --> NEW_Garr199905/EVALUATE
                                    '''               
                                    argums = [(tm_id, model_id, drl_eval_res_folder, differentiation_str, topology_Name, general_dataset_folder, args.f2[0]) for tm_id in range(50)]
                                    with Pool(processes=args.n[0]) as pool:
                                        pool.map(worker_execute, argums)
                        else:
                            break

    