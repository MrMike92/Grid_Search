import os
import time
import shutil
import kagglehub
import itertools
import numpy as np
import pandas as pd
import multiprocess
import multiprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Método para hacer nivelación de cargas
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

def cargar_datos(ruta_archivo):
    data = pd.read_csv(ruta_archivo)
    data = data.dropna()  # Elimina filas con valores NaN
    X = data.drop(columns=['Potability']).values # Caracteristicas
    y = data['Potability'].values # Etiquetas
    feature_names = data.drop(columns=['Potability']).columns.tolist()  # Nombres de las características
    return X, y, feature_names

# Función para descargar el dataset si no está disponible localmente
def descargar_dataset():
    print("Verificando y descargando dataset de Kaggle si es necesario...")
    path = kagglehub.dataset_download("adityakadiwal/water-potability")
    ruta_archivo_original = os.path.join(path, "water_potability.csv")
    ruta_destino = "./water_potability.csv"

    if not os.path.exists(ruta_destino):
        shutil.copy(ruta_archivo_original, ruta_destino)
        print(f"Dataset movido a: {ruta_destino}")
    else:
        print("Dataset ya disponible en la ubicación actual.")
    return ruta_destino

# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, ruta_archivo):
    X, y, feature_names = cargar_datos(ruta_archivo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    
    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        nombre_carpeta = 'resultsFilesSVM'

        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)

        nombre_archivo = f"{nombre_carpeta}/{str(s['C']).replace('.', '')}_{s['kernel']}_{s['gamma']}.txt"
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(f"Parámetros: C={s['C']}, Kernel={s['kernel']}, Gamma={s['gamma']}\n")
            f.write(f"Características utilizadas: {', '.join(feature_names)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
        
        # Exclusión mutua
        lock.acquire()
        print(f'Resultados guardados en: {nombre_archivo}')
        lock.release()

if __name__ == '__main__':
    print("Verificando dataset...")
    ruta_archivo = descargar_dataset() # Intentar descargar el dataset si no está disponible

    # Evaluar con múltiples procesos
    threads = []
    n_threads = multiprocessing.cpu_count()
    splits = nivelacion_cargas(combinations_svm, n_threads)
    lock = multiprocess.Lock()

    for i in range(n_threads):
        threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock, ruta_archivo)))

    start_time = time.perf_counter()
    
    # Lanzar procesos
    for thread in threads:
        thread.start()

    # Esperar a que todos terminen
    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Programa finalizado en {finish_time - start_time:.2f} segundos")
    print(f"Numero de hilos = {n_threads}")