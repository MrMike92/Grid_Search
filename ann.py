import os
import time
import torch
import shutil
import itertools
import kagglehub
import numpy as np
import multiprocess
import pandas as pd
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
torch.set_num_interop_threads(1) # Se establece la cantidad de subprocesos utilizados para el paralelismo de interoperabilidad en la CPU.

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

param_grid_nn = {
    'hidden_layers': [2, 3],
    'hidden_neurons': [32, 64],
    'learning_rate': [0.01, 0.1],
    'regularization': ["L2", "L1"],
    'regularization_rate': [0.01, 0.1],
    'activation': ["ReLU"]
}

keys_nn, values_nn = zip(*param_grid_nn.items())
combinations_nn = [dict(zip(keys_nn, v)) for v in itertools.product(*values_nn)]

def cargar_datos(ruta_archivo):
    data = pd.read_csv(ruta_archivo)
    data = data.dropna() # Elimina filas con valores NaN
    X = data.drop(columns=['Potability']).values # Características
    y = data['Potability'].values # Etiquetas
    feature_names = data.drop(columns=['Potability']).columns.tolist() # Nombres de las características
    
    return X, y, feature_names

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

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=1, hidden_neurons=32, activation='ReLU'):
        super(ArtificialNeuralNetwork, self).__init__()

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoideo':
            self.activation = nn.Sigmoid()
        elif activation == 'Lineal':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Función de activación no soportada: {activation}")

        # Lista de capas
        layers = []
        
        # Capa de entrada y primera capa oculta
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)
        
        # Capas ocultas adicionales
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        # Capa de salida
        layers.append(nn.Linear(hidden_neurons, 1))
        layers.append(nn.Sigmoid())  # Sigmoid para la salida binaria

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_ann(model, X_train, y_train, learning_rate, regularization_type, regularization_rate, epochs=1000):
    criterion = nn.BCELoss()  # Clasificación binaria
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs.squeeze(), torch.FloatTensor(y_train))

        # Aplicar regularización
        if regularization_type == "L2":
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += regularization_rate * l2_norm
        elif regularization_type == "L1":
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += regularization_rate * l1_norm

        loss.backward()
        optimizer.step()

def evaluate_ann(model, X_test):
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        return (outputs.squeeze().numpy() > 0.5).astype(int)

def evaluate_set(hyperparameter_set, lock, ruta_archivo, nombre_archivo, results_summary):
    X, y, feature_names = cargar_datos(ruta_archivo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
    local_accuracy = []

    for s in hyperparameter_set:
        model = ArtificialNeuralNetwork(input_size=X_train.shape[1], 
                                        hidden_layers=s['hidden_layers'], 
                                        hidden_neurons=s['hidden_neurons'], 
                                        activation=s['activation'])
        
        train_ann(model, X_train, y_train, s['learning_rate'], s['regularization'], s['regularization_rate'])
        y_pred = evaluate_ann(model, X_test)
        accuracy = np.mean(y_pred == y_test)
        local_accuracy.append(accuracy)
        
        lock.acquire()

        with open(nombre_archivo, 'a', encoding='utf-8') as f:
            f.write(f"Parámetros: hidden_layers={s['hidden_layers']}, hidden_neurons={s['hidden_neurons']}, learning_rate={s['learning_rate']}, activation={s['activation']}, regularization={s['regularization']}, regularization_rate={s['regularization_rate']}\n")
            f.write(f"Características utilizadas: {', '.join(feature_names)}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("=" * 80 + "\n")
        
        print(f'Resultados guardados en: {nombre_archivo}')
        lock.release()

    return local_accuracy

if __name__ == '__main__':
    print("Verificando dataset...")
    ruta_archivo = descargar_dataset()
    nombre_carpeta = 'resultsFilesANN'

    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    else:
        shutil.rmtree(nombre_carpeta)
        os.makedirs(nombre_carpeta)

    results_summary = f"{nombre_carpeta}/resumen_resultados.log"
    n_threads_max = multiprocessing.cpu_count()
    accuracies_per_thread = {}

    for n_threads in range(1, n_threads_max + 1):
        # print("Número de hilos de interop:", torch.get_num_interop_threads())
        # print("Número de hilos:", torch.get_num_threads())
        print(f"\nEjecutando con {n_threads} hilos...")

        nombre_archivo_unico = f"{nombre_carpeta}/combinaciones_resultados_{n_threads}_hilos.log"
        threads = []
        splits = nivelacion_cargas(combinations_nn, n_threads)
        lock = multiprocess.Lock()
        start_time = time.perf_counter()

        for i in range(n_threads):
            threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock, ruta_archivo, nombre_archivo_unico, results_summary)))

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