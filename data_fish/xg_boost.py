import os
import time
import shutil
import kagglehub
import itertools
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import multiprocess
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

keys_xgb, values_xgb = zip(*param_grid_xgb.items())
combinations_xgb = [dict(zip(keys_xgb, v)) for v in itertools.product(*values_xgb)]

def cargar_datos(ruta_archivo):
    data = pd.read_csv(ruta_archivo)
    
    # Convertir las columnas a formato numérico, forzando errores a NaN
    data['length'] = pd.to_numeric(data['length'], errors='coerce')
    data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
    data['w_l_ratio'] = pd.to_numeric(data['w_l_ratio'], errors='coerce')
    data = data.dropna() # Eliminar filas con valores NaN
    le = LabelEncoder() # Para convertir las etiquetas de especie a numéricas
    data['species'] = le.fit_transform(data['species'])
    X = data.drop(columns=['species']).values # Caracteristicas
    y = data['species'].values # Etiquetas
    feature_names = data.drop(columns=['species']).columns.tolist()  # Nombres de las características
    # print(le.classes_) # Ver las clases
    return X, y, feature_names

# Función para descargar el dataset si no está disponible localmente
def descargar_dataset():
    print("Descargando dataset de Kaggle...")
    path = kagglehub.dataset_download("taweilo/fish-species-sampling-weight-and-height-data")
    ruta_archivo_original = os.path.join(path, "fish_data.csv")
    ruta_destino = "./fish_data.csv"
    
    if not os.path.exists(ruta_destino):
        shutil.copy(ruta_archivo_original, ruta_destino)
        print(f"Dataset movido a: {ruta_destino}")
    else:
        print("Dataset disponible en la ubicación actual.")
    return ruta_destino

class XGBoostManual:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        self.pred = np.full(y.shape, np.mean(y)) # Predicción inicial (media de las etiquetas)
        for _ in range(self.n_estimators):
            residuals = y - self.pred # Calcular residuos
            
            # Ajustar un árbol a los residuos
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            self.pred += self.learning_rate * tree.predict(X) # Actualizar predicciones

    def predict(self, X):
        pred = np.full(X.shape[0], np.mean(self.pred))
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        return np.round(pred)  # Redondear para obtener predicciones binarias (0 o 1)

# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, ruta_archivo):
    X, y, feature_names = cargar_datos(ruta_archivo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    for s in hyperparameter_set:
        clf = XGBoostManual(max_depth=s['max_depth'], learning_rate=s['learning_rate'], n_estimators=s['n_estimators'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        nombre_carpeta = 'resultsFilesXGBoost'
        if not os.path.exists(nombre_carpeta):
            os.makedirs(nombre_carpeta)

        nombre_archivo = f"{nombre_carpeta}/{s['max_depth']}_{s['learning_rate']}_{s['n_estimators']}.txt"

        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write(f"Parámetros: max_depth={s['max_depth']}, learning_rate={s['learning_rate']}, n_estimators={s['n_estimators']}\n")
            f.write(f"Características utilizadas: {', '.join(feature_names)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")

        # Exclusión mutua
        lock.acquire()
        print(f'Resultados guardados en: {nombre_archivo}')
        lock.release()

if __name__ == '__main__':
    print("Verificando dataset...")
    ruta_archivo = descargar_dataset()  # Intentar descargar el dataset si no está disponible

    # Evaluar con múltiples procesos
    threads = []
    n_threads = multiprocessing.cpu_count()
    splits = nivelacion_cargas(combinations_xgb, n_threads)
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
    print(f"Número de hilos = {n_threads}")