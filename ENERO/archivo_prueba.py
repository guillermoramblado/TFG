import tensorflow as tf
import time

N = 9000 #Tamaño de cada dimensión del tensor
#Método encargado de devolver el tiempo de procesamiento del dispositivo indicado por parámetro
def benchmark(dispositivo):
    a = tf.random.normal([N, N])
    b = tf.random.normal([N, N])
    with tf.device(dispositivo):
        inicio = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        fin = time.time()
    return fin-inicio #Tiempo de procesamiento

if __name__ == "__main__":
    #gpus = tf.config.list_physical_devices('GPU')
    #print(f"GPU's detectadas: {gpus}")

    dispositivos = tf.config.list_physical_devices()
    print(f'Dispositivos localizados: {dispositivos}')

    for dis in ['/GPU:0','/GPU:1','CPU:0']:
        tiempo = benchmark(dis)
        print(f'\n{dis}--> {round(tiempo,3)}')

    #Con el siguiente código, podemos afirmar que por defecto usa la GPU de menor índice (GPU:0)
    a = tf.random.normal([N, N])
    b = tf.random.normal([N, N])
    inicio = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()
    fin = time.time()

    print(f'\nSupuestamente usando la GPU 0: {round(fin-inicio,3)}')