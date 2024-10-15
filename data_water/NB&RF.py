import pandas as pd
import numpy as np
import itertools
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score
import multiprocessing

# Cargar conjunto de datos
def cargar_datos(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    datos = datos.dropna()  # Eliminar filas con valores faltantes
    X = datos.drop('Potability', axis=1)
    y = datos['Potability']
    return X, y

# Función para dividir el trabajo equitativamente entre hilos
def dividir_carga_trabajo(datos, n_hilos):
    tamano_division = len(datos) // n_hilos
    resto = len(datos) % n_hilos
    cargas_trabajo = [datos[i * tamano_division + min(i, resto):(i + 1) * tamano_division + min(i + 1, resto)] for i in range(n_hilos)]
    return cargas_trabajo

# Función para evaluar los modelos
def evaluar_modelo(hyperparametros, nombre_modelo, X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    inicio = time.time()
    
    if nombre_modelo == 'RandomForest':
        modelo = RandomForestClassifier(
            n_estimators=hyperparametros['n_estimators'], 
            max_depth=hyperparametros['max_depth'], 
            min_samples_split=hyperparametros['min_samples_split'], 
            min_samples_leaf=hyperparametros['min_samples_leaf']
        )
    elif nombre_modelo == 'NaiveBayes':
        modelo = GaussianNB()
    
    modelo.fit(X_entrenamiento, y_entrenamiento)
    y_pred = modelo.predict(X_prueba)
    
    exactitud = accuracy_score(y_prueba, y_pred)
    recall = recall_score(y_prueba, y_pred)
    
    fin = time.time()
    duracion = fin - inicio
    
    resultado = {
        'modelo': nombre_modelo,
        'hyperparametros': hyperparametros,
        'exactitud': exactitud,
        'recall': recall,
        'tiempo': duracion
    }
    
    return resultado

# Función envoltura para pasar a starmap (sin conflictos)
def ejecutar_busqueda_rejilla(conjunto_hyperparametros, X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, nombre_modelo):
    # Asegurarse de que se pasen copias de los arreglos
    X_entrenamiento = np.copy(X_entrenamiento)
    X_prueba = np.copy(X_prueba)
    y_entrenamiento = np.copy(y_entrenamiento)
    y_prueba = np.copy(y_prueba)
    
    resultados = []
    for hyperparams in conjunto_hyperparametros:
        resultado = evaluar_modelo(hyperparams, nombre_modelo, X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
        resultados.append(resultado)
    return resultados

# Función principal para ejecutar el experimento
if __name__ == '__main__':
    # Cargar datos
    ruta_archivo = 'water_potability.csv'
    X, y = cargar_datos(ruta_archivo)
    
    # Dividir el conjunto de datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Rejilla de hiperparámetros para Random Forest
    rejilla_parametros_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Convertir rejilla de hiperparámetros en combinaciones
    claves_rf, valores_rf = zip(*rejilla_parametros_rf.items())
    combinaciones_rf = [dict(zip(claves_rf, v)) for v in itertools.product(*valores_rf)]
    
    # Para Naive Bayes, no hay hiperparámetros para ajustar, pero necesitamos incluirlo en las comparaciones
    combinaciones_nb = [{}]  # Diccionario vacío para Naive Bayes

    # Probar diferentes cantidades de hilos
    for n_hilos in range(1, 11):
        print(f"\nEjecutando con {n_hilos} hilos...")
        
        # Dividir el trabajo entre hilos
        carga_trabajo_rf = dividir_carga_trabajo(combinaciones_rf, n_hilos)
        carga_trabajo_nb = dividir_carga_trabajo(combinaciones_nb, n_hilos)
        
        # Crear grupos de multiprocesamiento
        with multiprocessing.Pool(n_hilos) as pool:
            resultados_rf = pool.starmap(ejecutar_busqueda_rejilla, [(trabajo, X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, 'RandomForest') for trabajo in carga_trabajo_rf])
            resultados_nb = pool.starmap(ejecutar_busqueda_rejilla, [(trabajo, X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, 'NaiveBayes') for trabajo in carga_trabajo_nb])
        
        # Aplanar los resultados de todos los hilos
        resultados_rf = [item for sublista in resultados_rf for item in sublista]
        resultados_nb = [item for sublista in resultados_nb for item in sublista]
        
        # Combinar todos los resultados
        todos_los_resultados_rf = resultados_rf
        todos_los_resultados_nb = resultados_nb
        
        # Encontrar mejor, peor y promedio para Random Forest
        mejor_rf = max(todos_los_resultados_rf, key=lambda x: x['exactitud'])
        peor_rf = min(todos_los_resultados_rf, key=lambda x: x['exactitud'])
        promedio_rf_exactitud = np.mean([res['exactitud'] for res in todos_los_resultados_rf])
        promedio_rf_recall = np.mean([res['recall'] for res in todos_los_resultados_rf])
        promedio_rf_tiempo = np.mean([res['tiempo'] for res in todos_los_resultados_rf])
        
        # Encontrar mejor, peor y promedio para Naive Bayes
        mejor_nb = max(todos_los_resultados_nb, key=lambda x: x['exactitud'])
        peor_nb = min(todos_los_resultados_nb, key=lambda x: x['exactitud'])
        promedio_nb_exactitud = np.mean([res['exactitud'] for res in todos_los_resultados_nb])
        promedio_nb_recall = np.mean([res['recall'] for res in todos_los_resultados_nb])
        promedio_nb_tiempo = np.mean([res['tiempo'] for res in todos_los_resultados_nb])
        
        # Imprimir resultados para Random Forest
        print("\n--- Resultados de Random Forest ---")
        print(f"Mejor resultado: Hiperparámetros: {mejor_rf['hyperparametros']}, Exactitud: {mejor_rf['exactitud']:.4f}, Recall: {mejor_rf['recall']:.4f}, Tiempo: {mejor_rf['tiempo']:.2f}s")
        print(f"Peor resultado: Hiperparámetros: {peor_rf['hyperparametros']}, Exactitud: {peor_rf['exactitud']:.4f}, Recall: {peor_rf['recall']:.4f}, Tiempo: {peor_rf['tiempo']:.2f}s")
        print(f"Exactitud Promedio: {promedio_rf_exactitud:.4f}, Recall Promedio: {promedio_rf_recall:.4f}, Tiempo Promedio: {promedio_rf_tiempo:.2f}s")
        
        # Imprimir resultados para Naive Bayes
        print("\n--- Resultados de Naive Bayes ---")
        print(f"Mejor resultado: Exactitud: {mejor_nb['exactitud']:.4f}, Recall: {mejor_nb['recall']:.4f}, Tiempo: {mejor_nb['tiempo']:.2f}s")
        print(f"Peor resultado: Exactitud: {peor_nb['exactitud']:.4f}, Recall: {peor_nb['recall']:.4f}, Tiempo: {peor_nb['tiempo']:.2f}s")
        print(f"Exactitud Promedio: {promedio_nb_exactitud:.4f}, Recall Promedio: {promedio_nb_recall:.4f}, Tiempo Promedio: {promedio_nb_tiempo:.2f}s")
