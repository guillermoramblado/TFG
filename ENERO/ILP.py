import os
import networkx as nx
from pulp import *
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ProblemaProgramacionLineal:
    #ruta_topologia: ruta de la carpeta que contiene el .graph y el subdirectorio con las matrices de validación (o entrenamiento)
    def __init__(self,ruta_topologia):
        #Comenzamos buscando el .graph
        self.ruta_graph = None #Ruta relativa del archivo .graph con el diseño de la topología
        self.dir_matrices = None #Ruta relativa del directorio con las matrices de tráfico
        self.nombre_topologia = None
        #Busco entre los diferentes archivos del directorio --> Carpeta TM y el .graph
        for file in os.listdir(ruta_topologia):
            ruta_completa = os.path.join(ruta_topologia,file)
            if os.path.isfile(ruta_completa) and file.endswith(".graph"):
                self.ruta_graph = ruta_completa
                self.nombre_topologia = file.split(".")[0]
                #print(f'\nArchivo .graph localizado en {ruta_completa}')
                #print(f'Nombre de la topología de trabajo: {self.nombre_topologia}')
            elif os.path.isdir(ruta_completa) and file.startswith("TM"):
                #print(f'Carpeta de matrices localizada en {ruta_completa}')
                self.dir_matrices = ruta_completa

        #Procedemos a crear un multidigraph
        self.grafo = nx.MultiDiGraph()
        self.__procesar_archivo_graph()

    #Método encargado de generar la topología de red a partir del .graph
    def __procesar_archivo_graph(self):
        with open(self.ruta_graph) as fd:
            line = fd.readline()
            self.num_nodos = int(line.split(" ")[1])
            #print(f'Número de nodos: {self.num_nodos}')
            #Nos saltamos la línea 'label x y'
            line = fd.readline() 
            while not line.startswith("EDGES"):
                line = fd.readline()
            #Guardamos el nº de enlaces que conforman la red
            self.num_enlaces = int(line.split(" ")[1])
            #print(f'Número de enlaces: {self.num_enlaces}')
            #Por último, tomo todos los enlaces 
            for line in fd:
                if not line.startswith("Link_"):
                    continue
                campos = line.split(" ")
                nodo_origen = int(campos[1])
                nodo_destino = int(campos[2])
                coste_enlace = int(campos[3])
                capacidad_enlace = int(campos[4])
                self.grafo.add_edge(nodo_origen,nodo_destino,capacidad=capacidad_enlace,coste=coste_enlace)
            #input("Presione ENTER para comprobar que se haya creado el grafo correctamente")
            #print(self.grafo.number_of_edges())

    '''
    Método encargado de devolver la característica especificada de un determinado enlace dirigido
    Los parámetros que se deben especificar son los siguientes:
        * origen: nodo origen del enlace
        * destino:: nodo destino del enlace
        * caracteristica: clave que identifica la característica que se desea obtener
    No es necesario especificar el índice del enlace, dada una pareja de nodos concreta, ya que en este problema se trabaja
    con un único enlace dirigido como máximo para cada pareja de nodos vecinos
    '''
    def obtener_caracteristica_enlace(self,origen,destino,caracteristica):
        if caracteristica not in self.grafo[origen][destino][0]:
            print("No existe la característica especificada")
            return
        else:
            return self.grafo[origen][destino][0][caracteristica]
            

    #Método encargado de resolver el problema de programacion lineal usando una determinada matriz de tráfico
    def resolver_matriz_trafico(self,ind):

        #Comprobamos primero que sea un índice válido
        num_matrices_trafico = len(os.listdir(self.dir_matrices))
        if ind not in range(num_matrices_trafico):
            print("El índice de la mátriz de tráfico no es válido. Debe ser un índice en el rango [0,49]")
            return
        
        #Creamos un diccionario que mantiene el tráfico a enviar para cada demanda de tráfico existente
        demandas = {}
        ruta_matriz = self.dir_matrices + "/" + self.nombre_topologia + f'.{int(ind)}.demands'
        if not os.path.exists(ruta_matriz):
            print(f'No existe una matriz de tráfico en la ruta {ruta_matriz}')
            return
        
        with open(ruta_matriz) as fd:
            fd.readline()
            fd.readline()
            for line in fd:
                campos = line.split(" ")
                emisor = int(campos[1])
                receptor = int(campos[2])
                trafico = float(campos[3])
                demandas[(emisor,receptor)] = trafico

        #Modelo ahora el problema de programación lineal usando Pulp
        problema = LpProblem(name='ProblemaMinMaxUso',sense=LpMinimize)
        #Defino la variable que mide el uso del enlace de mayor uso
        uso_maximo = LpVariable(name="UsoMaximo",lowBound=0)
        problema += uso_maximo
        #Definimos, para cada demanda, una variable de decisión para cada uno de los enlaces de la red
        #Esto define un diccionario con claves (s,t,u,v) y valor x_(s,t,u,v)
        lista_enlaces = list(self.grafo.edges(keys=False))
        x = LpVariable.dicts("x",[(s,t,u,v) for (s,t) in demandas for (u,v) in lista_enlaces],cat=const.LpBinary)
        
        #RESTRICCIONES DE CAPACIDAD
        for i,j in lista_enlaces:
            #Para cada enlace, tomamos su capacidad
            capacidad_enlace = self.obtener_caracteristica_enlace(i,j,'capacidad')
            #Calculamos el flujo total de dicho enlace, teniendo en cuenta todas las demandas que crucen por dicho enlace
            flujo_total_enlace = lpSum(x[(s,t,i,j)]*demandas[(s,t)] for (s,t) in demandas)
            problema += flujo_total_enlace <= uso_maximo * capacidad_enlace
        
        #RESTRICCIONES DE CONSERVACIÓN DE FLUJO
        for emisor,receptor in demandas:
            #Conservación del flujo de dicha demanda para cada uno de los nodos de la red
            for nodo in list(self.grafo.nodes):
                #De todos los enlaces dirigidos que terminan en 'nodo', contabilizo los que llevan tráfico de dicha demanda
                flujo_entrante = lpSum(x[(emisor,receptor,origen,destino)] for (origen,destino) in lista_enlaces if destino==nodo)
                #Lo mismo pero trabajando sobre los enlaces dirigidos que salen de 'nodo'
                flujo_saliente = lpSum(x[(emisor,receptor,origen,destino)] for (origen,destino) in lista_enlaces if origen==nodo)
                
                if nodo==emisor:
                    problema += flujo_entrante == 0
                    problema += flujo_saliente == 1
                elif nodo==receptor:
                    problema += flujo_entrante == 1
                    problema += flujo_saliente == 0
                else:
                    #Ese nodo es un nodo intermedio. Si recibe tráfico de la demanda, tiene que sacarlo completamente
                    problema += flujo_entrante - flujo_saliente ==0
                    problema += flujo_entrante <=1
        
        solver = PULP_CBC_CMD(msg=False)
        inicio = time.time()
        problema.solve(solver)
        fin = time.time()
        print(f'\nEstado del problema: {LpStatus[problema.status]}')
        print(f'Uso del enlace de mayor congestión: {value(uso_maximo)}')
        print(f'Tiempo total requerido: {round(fin-inicio,4)} segundos')

        #self.__pintar_rutas(demandas,x)

        return value(uso_maximo), round(fin-inicio,4)
    
    #Método encargado de pintar la ruta asignada a cada una de las demandas
    def __pintar_rutas(self, demandas, x_vars):
        #Genero el diccionario con parejas (clave,valor): ("indice de nodo","posicion en el layout")
        G = self.grafo
        pos = nx.spring_layout(G,seed=42)
        #Pintamos inicialmente todos los nodos usando el color por defecto (azul) y todos los enlaces usando el color gris
        nx.draw_networkx_nodes(G,pos)
        nx.draw_networkx_labels(G,pos)
        nx.draw_networkx_edges(G,pos,edge_color="#cccccc")

        #Pinto la ruta asignada a cada una de las demandas
        for (emisor,receptor) in demandas:
            pos = nx.spring_layout(G,seed=42)
            #Pintamos inicialmente todos los nodos usando el color por defecto (azul) y todos los enlaces usando el color gris
            nx.draw_networkx_nodes(G,pos)
            nx.draw_networkx_labels(G,pos)
            nx.draw_networkx_edges(G,pos,edge_color="#cccccc")
            print(f'\nTomando enlaces que definen la ruta asociada a la demanda {emisor}->{receptor}')
            #Tomo los enlaces que llevan tráfico de dicha demanda
            enlaces_uso = [(nodo_salida,nodo_llegada) for (origen,destino,nodo_salida,nodo_llegada), var in x_vars.items() if origen==emisor and destino==receptor and var.value() == 1]
            print(enlaces_uso)
            #Ordeno los enlaces que llevan tráfico de la demanda
            enlaces_ordenados = []
            for ind, (nodo_salida,nodo_llegada) in enumerate(enlaces_uso):
                if nodo_salida == emisor:
                    enlaces_ordenados.append((nodo_salida,nodo_llegada))
                    del enlaces_uso[ind]
                    break
            
            # nodo_llegada es el nodo al que llega el primer enlace que lleva tráfico de la demanda
            nodo_actual = nodo_llegada
            nodos_intermedios = []
            while nodo_llegada != receptor:
                #Buscamos el enlace que sale del último nodo alcanzado (nodo_llegada)
                for ind, (nodo_salida,nodo_llegada) in enumerate(enlaces_uso):
                    if nodo_salida == nodo_actual:
                        enlaces_ordenados.append((nodo_salida,nodo_llegada))
                        del enlaces_uso[ind]
                        nodo_actual = nodo_llegada
                        nodos_intermedios.append(nodo_salida)
                        break 
            print(f'Lista de enlaces ordenada: {enlaces_ordenados}')
            #Pintamos los nodos emisor y receptor de la demanda específica
            nx.draw_networkx_nodes(G,pos,nodelist=[emisor,receptor],node_color="red")
            #Pintamos los enlaces que definen la ruta
            nx.draw_networkx_edges(G,pos,edgelist=enlaces_ordenados,edge_color="red")
            #Pinto los nodos intermedios de otro color
            nx.draw_networkx_nodes(G,pos,nodelist=nodos_intermedios,node_color="salmon")
            
            plt.title(f'Ruta demanda {emisor}->{receptor}')
            plt.axis("off")
            plt.show()
            input("Procesada la demanda actual.Presione ENTER")



     

'''
Previamente, será necesario ejecutar el script crear_figuras_entrenamiento.py, que almacena los resultados de validar los diferentes agentes a comparar sobre las 50 matrices de validación

Este script nos permite generar los boxplots asociados a la resolución del problema de MinMaxLinkUtilization usando Deep Learning (seleccionando uno de los agentes) vs Programación Lineal
'''

#Lista con el nombre de las carpetas que contienen toda la información asociada a las tres topologías empleadas en el proceso de entrenamiento/validación
lista_topologias = ['NEW_BtAsiaPac','NEW_Garr199905','NEW_Goodnet']
#Ruta del directorio padre que contiene todas las carpetas asociadas a las diferentes topologías
dir_padre_topologias = "../ENERO_datasets\dataset_sing_top/data/results_my_3_tops_unif_05-1"
nombre_modelo = "MLP"

if __name__ == "__main__":

    #Comienzo indicando el nombre del modelo (con resultados guardados en resultados_modelos_por_topologia.csv) que quiero finalmente usar para compararlo con respecto al uso de programación lineal
    ruta_csv_original = "./resultados_modelos_por_topologia.csv"
    comparativa_modelos = pd.read_csv(ruta_csv_original)
    resultados_modelos = comparativa_modelos[comparativa_modelos['Modelo']=='actorPPOmiddR']

    #Generamos nuevas filas que almacenan los resultados de resolver mediante programación lineal el problema MinMaxLinkUtilization usando las 3 topologías de siempre, empleando las 50 matrices de validación
    resultados_MLP = {"Modelo":[],"Topologia":[],"Uso":[],"Tiempo":[]}
    for topologia in lista_topologias:
        print(f'\nTopología {topologia}')
        ruta = dir_padre_topologias + "/" + topologia + "/EVALUATE"
        problema = ProblemaProgramacionLineal(ruta)
        for ind_matriz in range(50):
            uso, tiempo = problema.resolver_matriz_trafico(ind_matriz)
            #Añadimos la fila con los resultados obtenidos para esa matriz y topología concretas
            for clave, valor in zip(resultados_MLP,[nombre_modelo,topologia,uso,tiempo]):
                resultados_MLP[clave].append(valor)
            print(f'Resuelto problema MLP para la matriz de tráfico {ind_matriz}')
    
    #Combino los resultados obtenidos usando Deep Learning y MLP
    resultados_modelos = pd.concat([resultados_modelos,pd.DataFrame(resultados_MLP)],ignore_index=True)
    
    resultados_modelos.to_csv("comparacion_Agente_MLP.csv")
    media_por_modelo = resultados_modelos.groupby("Modelo")["Tiempo"].mean()
    media_por_modelo.to_csv("tiempo_medio_Agente_MLP.csv")

    #Generamos los boxplots para la métrica (uso enlace mayor uso) y tiempos de ejecución
    valores_graficar = ["Uso","Tiempo"]
    nombre_graficos = ["comparativa_usos.png","comparativa_tiempos.png"]
    etiquetas_ejeY = ["Uso del enlace de mayor uso", "Tiempo de ejecución"]

    if not os.path.exists("./Imagenes/Comparacion_Agente_MLP"):
        os.makedirs("./Imagenes/Comparacion_Agente_MLP")

    for valor, nombre, etiqueta in zip(valores_graficar,nombre_graficos,etiquetas_ejeY):
        #Generamos los boxplots
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 5))

        # Crear boxplots: uno por topología, con un boxplot por modelo
        ax = sns.boxplot(x="Topologia", y=valor, hue="Modelo", data=resultados_modelos, palette="Set2")

        # Opcional: etiquetas y leyenda
        plt.title("Comparativa por Topologia")
        plt.ylabel(etiqueta)
        plt.xlabel("Topologia")
        plt.legend(title="Modelo", loc="upper right")

        plt.tight_layout()
        path_imagen_boxplots = "./Imagenes/Comparacion_Agente_MLP/" + nombre
        plt.savefig(path_imagen_boxplots)
        plt.close


        



    
        