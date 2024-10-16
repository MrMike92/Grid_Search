import itertools
import multiprocess
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score

# Cargar dataset
data = pd.read_csv('water_potability.csv')
data = data.dropna()

X = data.drop('Potability', axis=1)
y = data['Potability']

# Parámetros para Naive Bayes
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Combinaciones de hiperparámetros
keys_nb, values_nb = zip(*param_grid_nb.items())
combinations_nb = [dict(zip(keys_nb, v)) for v in itertools.product(*values_nb)]

# Nivelación de cargas para paralelizar combinaciones
def nivelacion_cargas(D, n_p):
    s = len(D)%n_p
    n_D = D[:s]
    t = int((len(D)-s)/n_p)
    out=[]
    temp=[]
    for i in D[s:]:
        temp.append(i)
        if len(temp)==t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Evaluar un conjunto de hiperparámetros
def evaluate_nb(hyperparameter_set, results_list, lock):
    thread_results = []
    for params in hyperparameter_set:
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        # Entrenamiento de Naive Bayes
        clf = GaussianNB(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        elapsed_time = time.time() - start_time
        result = {
            'params': params,
            'accuracy': accuracy,
            'recall': recall,
            'time': elapsed_time
        }
        thread_results.append(result)

    # Guardar resultados globales
    lock.acquire()
    results_list.append(thread_results)
    lock.release()

# Función para calcular el resumen de mejores, peores y promedios
def calcular_resumen(results_list):
    all_results = [item for sublist in results_list for item in sublist]
    best = max(all_results, key=lambda x: x['accuracy'])
    worst = min(all_results, key=lambda x: x['accuracy'])
    
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_time = np.mean([r['time'] for r in all_results])
    
    return best, worst, avg_accuracy, avg_recall, avg_time

# Ejecución con diferentes hilos y resumen final
if __name__ == '__main__':
    lock = multiprocess.Lock()
    
    # Crear archivo para resultados generales
    with open('results_nb.log', 'w') as f:
        f.write('Resultados GaussianNB\n')

    resumen_general = []

    # Ejecución de 1 a 10 hilos
    for num_threads in range(1, 11):
        results_list = multiprocess.Manager().list()
        splits_nb = nivelacion_cargas(combinations_nb, num_threads)
        
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            t = multiprocess.Process(target=evaluate_nb, args=(splits_nb[i], results_list, lock))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

        end_time = time.time()

        # Calcular mejores, peores y promedios para este número de hilos
        best, worst, avg_accuracy, avg_recall, avg_time = calcular_resumen(results_list)

        # Guardar en el archivo de resultados generales con separación por hilos
        with open('results_nb.log', 'a') as f:
            f.write(f"\n--- Resultados con {num_threads} hilos ---\n")
            for result in [item for sublist in results_list for item in sublist]:
                f.write(f"{result}\n")
        
        # Guardar resumen para este número de hilos
        resumen_general.append({
            'hilos': num_threads,
            'mejor': best,
            'peor': worst,
            'avg_accuracy': avg_accuracy,
            'avg_recall': avg_recall,
            'avg_time': avg_time
        })

    # Guardar resumen general
    with open('summary_nb.log', 'w') as f:
        f.write("Resumen General de Naive Bayes\n\n")
        for resumen in resumen_general:
            f.write(f"--- Resultados con {resumen['hilos']} hilos ---\n")
            f.write(f"Mejor resultado: {resumen['mejor']}\n")
            f.write(f"Peor resultado: {resumen['peor']}\n")
            f.write(f"Promedio - Accuracy: {resumen['avg_accuracy']}, Recall: {resumen['avg_recall']}, Tiempo: {resumen['avg_time']} segundos\n\n")
        
        # Mejor, peor y promedio general
        mejor_general = max(resumen_general, key=lambda x: x['mejor']['accuracy'])
        peor_general = min(resumen_general, key=lambda x: x['peor']['accuracy'])
        promedio_general_accuracy = np.mean([r['avg_accuracy'] for r in resumen_general])
        promedio_general_recall = np.mean([r['avg_recall'] for r in resumen_general])
        promedio_general_time = np.mean([r['avg_time'] for r in resumen_general])

        f.write(f"\n--- Resumen general de todos los hilos ---\n")
        f.write(f"Mejor resultado general: {mejor_general['mejor']}\n")
        f.write(f"Peor resultado general: {peor_general['peor']}\n")
        f.write(f"Promedio general - Accuracy: {promedio_general_accuracy}, Recall: {promedio_general_recall}, Tiempo: {promedio_general_time} segundos\n")

    print(f"Proceso GaussianNB completado.")
