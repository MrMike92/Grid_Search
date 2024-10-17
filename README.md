# Grid Search

> [!WARNING]
> En esta rama solo se encuentra las versiones pruebas de los códigos con sus archivos resutlantes de las diferentes combinaciones de los hiperparámetros de los modelos, los códigos finales estan en la rama principal ***main***.

Grid Search es un método de optimización que se utiliza en el aprendizaje automático para ajustar hiperparámetros, esta consiste en probar todas las combinaciones posibles de los valores de hiperparámetros para encontrar el modelo que mejor se adapte a los datos.

Los hiperparámetros controlan la estructura, las funciones y el rendimiento de los modelos. Ajustar los hiperparámetros permite modificar el rendimiento del modelo para obtener resultados óptimos.

El grid search funciona de la siguiente manera:
- Se entrena un modelo para cada combinación de hiperparámetros.
- Se selecciona el modelo que mejor se desempeña.

## Funcionamiento

1. Se cargan el conjunto de datos desde un archivo CSV y se eliminan cualquier fila que contenga valores faltantes (valores nulos/NaN) para evitar errores durante el entrenamiento, a continuación, se separan las características (variables predictoras) de la variable objetivo, que indica la potabilidad del agua, luego, el conjunto de datos se divide en dos partes: datos de entrenamiento y datos de prueba. Los primeros se utilizan para entrenar los modelos, mientras que los segundos permiten evaluar el rendimiento de estos (relación 80% de entrenamiento y 20% de pruebas).

2. Con el usa del procesamiento paralelo, se reparte el trabajo de probar distintas combinaciones de hiperparámetros, que, dependiendo de la cantidad de hilos seleccionados, se divide la carga de trabajo entre los procesos disponibles de forma ***equitativa***. Pero los procesos internos del modelo se hacen de forma secuencial.

3. Se realizan las pruebas de forma simultáneo en varios hilos en lugar de forma secuencial con diferentes combinaciones de parámetros, que se evalúan en términos de precisión en la clasificación y el tiempo que toma entrenar para predecir con cada configuración.

4. Cada modelo es evaluado con base en la precisión de la clasificación, que mide el porcentaje de predicciones correctas en el conjunto de prueba y el tiempo de ejecución también es un factor clave que se toma en cuenta, ya que ayuda a identificar qué modelos y configuraciones son más eficientes en términos computacionales.

5. Tras la ejecución de las pruebas, se analizan los resultados para encontrar la mejor y la peor combinación de parámetros en la precisión de la clasificación para cada modelo, esto para dar una visión global del rendimiento del modelo bajo distintas configuraciones.

6. Al final se imprimen los resultados detallados, incluyendo la mejor y peor precisión obtenida, así como el tiempo de ejecución para cada número de hilos utilizados de cada uno de los modelos utilizados: ***una red neuronal artificial (ANN), Random Forest, XGBoost y Naive Bayes***.

Esto proporciona una visión clara del balance entre precisión, sensibilidad y costo computacional, esto es fundamental para la toma de decisiones sobre qué modelo utilizar en la práctica.

## Instrucciones de uso.

- Clona este repositorio en tu máquina local o descargue el repositorio en zip, asegure que el respositorio se haya descargado correctamente.
- Descomprimir el zip (en caso que se haya descargado el zip).
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python.
- Descargue las siguentes bibliotecas para poder ejecutar los programas:
  - pip install kagglehub
  - pip install pandas
  - pip install multiprocess
  - pip install scikit-learn

> [!CAUTION]
> Para evitar cualquier problema de compatibilidad, utiliza la versión de Python a partir de 3.8.x hasta 3.11.x
> * kagglehub:
>   * Versión mínima: Python 3.7.x
>   * Versión máxima: Python 3.11.x
> * pandas:
>   * Versión mínima: Python 3.7.x
>   * Versión máxima: Python 3.11.x
> * multiprocess:
>   * Versión mínima: Python 3.6.x
>   * Versión máxima: Python 3.11.x
> * scikit-learn:
>   * Versión mínima: Python 3.8.x
>   * Versión máxima: Python 3.11.x

Si deseas contribuir a este repositorio, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92, en un futuro planeo abrir un correo para poder contactarme. 🐢
