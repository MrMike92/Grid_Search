import os
import time
import shutil
import kagglehub
import itertools
import numpy as np
import pandas as pd
import multiprocess
import multiprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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

param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

keys_xgb, values_xgb = zip(*param_grid_xgb.items())
combinations_xgb = [dict(zip(keys_xgb, v)) for v in itertools.product(*values_xgb)]

def cargar_datos(ruta_archivo):
    data = pd.read_csv(ruta_archivo)
    data = data.dropna()  # Elimina filas con valores NaN
    X = data.drop(columns=['Potability']).values  # Características
    y = data['Potability'].values  # Etiquetas
    feature_names = data.drop(columns=['Potability']).columns.tolist()  # Nombres de las características
    
    return X, y, feature_names

# Función para descargar el dataset si no está disponible localmente
def descargar_dataset():
    print("Descargando dataset de Kaggle...")
    path = kagglehub.dataset_download("adityakadiwal/water-potability")
    ruta_archivo_original = os.path.join(path, "water_potability.csv")
    ruta_destino = "./water_potability.csv"

    if not os.path.exists(ruta_destino):
        shutil.copy(ruta_archivo_original, ruta_destino)
        print(f"Dataset movido a: {ruta_destino}")
    else:
        print("Dataset ya disponible en la ubicación actual.")
    
    return ruta_destino

class XGBoostManual:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        self.pred = np.full(y.shape, np.mean(y))  # Predicción inicial (media de las etiquetas)
        for _ in range(self.n_estimators):
            residuals = y - self.pred  # Calcular residuos

            # Ajustar un árbol a los residuos
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            self.pred += self.learning_rate * tree.predict(X)  # Actualizar predicciones

    def predict(self, X):
        pred = np.full(X.shape[0], np.mean(self.pred))
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        return np.round(pred)  # Redondear para obtener predicciones binarias (0 o 1)

def evaluate_set(hyperparameter_set, lock, ruta_archivo, nombre_archivo):
    X, y, feature_names = cargar_datos(ruta_archivo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    local_accuracy = []
    
    for s in hyperparameter_set:
        clf = XGBoostManual(max_depth=s['max_depth'], learning_rate=s['learning_rate'], n_estimators=s['n_estimators'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        local_accuracy.append(accuracy)
        lock.acquire()

        with open(nombre_archivo, 'a', encoding='utf-8') as f:
            f.write(f"Parámetros: max_depth={s['max_depth']}, learning_rate={s['learning_rate']}, n_estimators={s['n_estimators']}\n")
            f.write(f"Características utilizadas: {', '.join(feature_names)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("=" * 80 + "\n")
        
        print(f'Resultados guardados en: {nombre_archivo}')
        lock.release()

    return local_accuracy

if __name__ == '__main__':
    print("Verificando dataset...")
    ruta_archivo = descargar_dataset()
    nombre_carpeta = 'resultsFilesXGBoost'

    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    else:
        shutil.rmtree(nombre_carpeta)
        os.makedirs(nombre_carpeta)
    
    results_summary = f"{nombre_carpeta}/resumen_resultados.log"
    n_threads_max = multiprocessing.cpu_count()
    accuracies_per_thread = {}

    for n_threads in range(1, n_threads_max + 1):
        print(f"\nEjecutando con {n_threads} hilos...")

        nombre_archivo_unico = f"{nombre_carpeta}/combinaciones_resultados_{n_threads}_hilos.log"
        threads = []
        splits = nivelacion_cargas(combinations_xgb, n_threads)
        lock = multiprocess.Lock()
        start_time = time.perf_counter()

        for i in range(n_threads):
            threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock, ruta_archivo, nombre_archivo_unico)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        finish_time = time.perf_counter()
        elapsed_time = finish_time - start_time
        accuracies = []
        params_list = []

        print(f"Ejecutado en {elapsed_time:.2f} segundos")

        with open(nombre_archivo_unico, 'r', encoding='utf-8') as f:
            current_params = None
            
            for line in f:
                if "Parámetros:" in line:
                    current_params = line.strip()

                if "Accuracy:" in line and current_params:
                    accuracy = float(line.split('Accuracy: ')[-1].strip())
                    accuracies.append(accuracy)
                    params_list.append((accuracy, current_params))
                    current_params = None

        best_accuracy, best_params = max(params_list, key=lambda x: x[0])
        worst_accuracy, worst_params = min(params_list, key=lambda x: x[0])
        accuracies_per_thread[n_threads] = (best_accuracy, worst_accuracy, elapsed_time)

        with open(results_summary, 'a', encoding='utf-8') as f:
            f.write(f"Hilos utilizados: {n_threads}\n")
            f.write(f"Mejor precisión: {best_accuracy:.4f}; Parámetros: {best_params.split(': ')[1]}\n")
            f.write(f"Peor precisión: {worst_accuracy:.4f}; Parámetros: {worst_params.split(': ')[1]}\n")
            f.write(f"Ejecutado en {elapsed_time:.2f} segundos\n")
            f.write("=" * 80 + "\n")

    mejor_hilos, mejor_tiempo = min(accuracies_per_thread.items(), key=lambda x: x[1][2])
    peor_hilos, peor_tiempo = max(accuracies_per_thread.items(), key=lambda x: x[1][2])

    with open(results_summary, 'a', encoding='utf-8') as f:
        f.write("Resumen final de tiempos de ejecución:\n")
        f.write(f"Mejor tiempo de ejecución: {mejor_tiempo[2]:.2f} segundos con {mejor_hilos} hilos\n")
        f.write(f"Peor tiempo de ejecución: {peor_tiempo[2]:.2f} segundos con {peor_hilos} hilos\n")