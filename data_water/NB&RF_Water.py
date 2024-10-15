import pandas as pd
import numpy as np
import itertools
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score
import multiprocessing

# Cargar dataset
def cargar_datos(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Eliminar filas con valores faltantes
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    return X, y

# Función para dividir el trabajo equitativamente entre los hilos
def dividir_carga_trabajo(data, n_hilos):
    tamano_division = len(data) // n_hilos
    resto = len(data) % n_hilos
    cargas = [data[i * tamano_division + min(i, resto):(i + 1) * tamano_division + min(i + 1, resto)] for i in range(n_hilos)]
    return cargas

# Función para evaluar los modelos
def evaluar_modelo(hiperparametros, nombre_modelo, X_train, X_test, y_train, y_test):
    inicio = time.time()
    
    if nombre_modelo == 'RandomForest':
        modelo = RandomForestClassifier(
            n_estimators=hiperparametros['n_estimators'], 
            max_depth=hiperparametros['max_depth'], 
            min_samples_split=hiperparametros['min_samples_split'], 
            min_samples_leaf=hiperparametros['min_samples_leaf']
        )
    elif nombre_modelo == 'NaiveBayes':
        modelo = GaussianNB()
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    precision = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    fin = time.time()
    duracion = fin - inicio
    
    resultado = {
        'modelo': nombre_modelo,
        'hiperparametros': hiperparametros,
        'precision': precision,
        'recall': recall,
        'tiempo': duracion
    }
    
    return resultado

# Función de envoltura para pasar a starmap (sin conflicto)
def ejecutar_grid_search(conjunto_hiperparametros, X_train, X_test, y_train, y_test, nombre_modelo):
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    y_train = np.copy(y_train)
    y_test = np.copy(y_test)
    
    resultados = []
    for hiperparametros in conjunto_hiperparametros:
        resultado = evaluar_modelo(hiperparametros, nombre_modelo, X_train, X_test, y_train, y_test)
        resultados.append(resultado)
    return resultados

# Función principal para ejecutar el experimento
if __name__ == '__main__':
    # Cargar datos
    file_path = 'water_potability.csv'
    X, y = cargar_datos(file_path)
    
    # Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Rejilla de hiperparámetros para Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Convertir la rejilla de hiperparámetros en combinaciones
    llaves_rf, valores_rf = zip(*param_grid_rf.items())
    combinaciones_rf = [dict(zip(llaves_rf, v)) for v in itertools.product(*valores_rf)]
    
    # Para Naive Bayes, no hay hiperparámetros que ajustar, pero necesitamos incluirlo en las comparaciones
    combinaciones_nb = [{}]  # Diccionario vacío para Naive Bayes

    # Probar con diferentes cantidades de hilos
    for n_hilos in range(1, 11):
        print(f"\nEjecutando con {n_hilos} hilos...")
        
        # Dividir el trabajo entre los hilos
        rf_carga = dividir_carga_trabajo(combinaciones_rf, n_hilos)
        nb_carga = dividir_carga_trabajo(combinaciones_nb, n_hilos)
        
        # Crear pools de multiprocesamiento
        with multiprocessing.Pool(n_hilos) as pool:
            rf_resultados = pool.starmap(ejecutar_grid_search, [(trabajo, X_train, X_test, y_train, y_test, 'RandomForest') for trabajo in rf_carga])
            nb_resultados = pool.starmap(ejecutar_grid_search, [(trabajo, X_train, X_test, y_train, y_test, 'NaiveBayes') for trabajo in nb_carga])
        
        # Aplanar los resultados de todos los hilos
        rf_resultados = [item for sublista in rf_resultados for item in sublista]
        nb_resultados = [item for sublista in nb_resultados for item in sublista]
        
        # Combinar todos los resultados
        todos_rf_resultados = rf_resultados
        todos_nb_resultados = nb_resultados
        
        # Encontrar el mejor, peor y promedio para Random Forest
        mejor_rf = max(todos_rf_resultados, key=lambda x: x['precision'])
        peor_rf = min(todos_rf_resultados, key=lambda x: x['precision'])
        promedio_rf_precision = np.mean([res['precision'] for res in todos_rf_resultados])
        promedio_rf_recall = np.mean([res['recall'] for res in todos_rf_resultados])
        promedio_rf_tiempo = np.mean([res['tiempo'] for res in todos_rf_resultados])
        
        # Encontrar el mejor, peor y promedio para Naive Bayes
        mejor_nb = max(todos_nb_resultados, key=lambda x: x['precision'])
        peor_nb = min(todos_nb_resultados, key=lambda x: x['precision'])
        promedio_nb_precision = np.mean([res['precision'] for res in todos_nb_resultados])
        promedio_nb_recall = np.mean([res['recall'] for res in todos_nb_resultados])
        promedio_nb_tiempo = np.mean([res['tiempo'] for res in todos_nb_resultados])
        
        # Imprimir resultados para Random Forest
        print("\n--- Resultados de Random Forest ---")
        print(f"Mejor resultado: Hiperparámetros: {mejor_rf['hiperparametros']}, Precisión: {mejor_rf['precision']:.6f}, Recall: {mejor_rf['recall']:.6f}, Tiempo: {mejor_rf['tiempo']:.6f}s")
        print(f"Peor resultado: Hiperparámetros: {peor_rf['hiperparametros']}, Precisión: {peor_rf['precision']:.6f}, Recall: {peor_rf['recall']:.6f}, Tiempo: {peor_rf['tiempo']:.6f}s")
        print(f"Precisión promedio: {promedio_rf_precision:.6f}, Recall promedio: {promedio_rf_recall:.6f}, Tiempo promedio: {promedio_rf_tiempo:.6f}s")
        
        # Imprimir resultados para Naive Bayes
        print("\n--- Resultados de Naive Bayes ---")
        print(f"Mejor resultado: Precisión: {mejor_nb['precision']:.6f}, Recall: {mejor_nb['recall']:.6f}, Tiempo: {mejor_nb['tiempo']:.6f}s")
        print(f"Peor resultado: Precisión: {peor_nb['precision']:.6f}, Recall: {peor_nb['recall']:.6f}, Tiempo: {peor_nb['tiempo']:.6f}s")
        print(f"Precisión promedio: {promedio_nb_precision:.6f}, Recall promedio: {promedio_nb_recall:.6f}, Tiempo promedio: {promedio_nb_tiempo:.6f}s")
