# Grid_Search

> [!WARNING]
> En esta rama solo se encuentra las versiones finales de los códigos, para ver más detalles, vaya a la rama test_config_programs.

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

## Resultados.

El rendimiento general de Random Forest fue bastante estable, con precisiones y recalls promedios rondando los 0.932, sin embargo, los mejores resultados variaron ligeramente según el número de hilos, con un desempeño óptimo en 3 y 7 hilos (precisión y recall de 0.941176). Este rendimiento óptimo se logró con configuraciones que implicaban un mayor número de estimadores (n\_estimators=100), mayor profundidad (max_depth=30) y un tamaño mínimo de división de muestras de 5, combinado con un mínimo de 2 hojas por división.

Estas configuraciones más complejas en cuanto a profundidad y número de estimadores parecen haber proporcionado mejor rendimiento, aunque el tiempo de ejecución aumentó proporcionalmente a medida que el número de hilos aumentaba, en general, mientras que una mayor complejidad en los hiperparámetros mejoró ligeramente el rendimiento en precisión y recall, también incrementó considerablemente el tiempo de ejecución.

Con Naive Bayes, los resultados fueron constantes e invariables a través de todos los experimentos y tanto la precisión como el recall se mantuvieron en 0.928922, y el tiempo de ejecución fue extremadamente bajo (entre 0.002 y 0.007 segundos). Este resultado es esperable, dado que Naive Bayes no depende de hiperparámetros complejos como Random Forest y es menos costoso en términos de cómputo.

El rendimiento general de la red neuronal (ANN) mostró variabilidad, con precisiones que variaron entre 40.45% y 59.80%, donde los mejores resultados se lograron con una configuración que incluía 3 capas ocultas, 32 neuronas por capa, una tasa de aprendizaje de 0.01, activación ReLU, y regularización L2 con una tasa de 0.01. A pesar de estos ajustes, la precisión promedio no superó los 0.60, y el tiempo de ejecución fue relativamente alto, alcanzando alrededor de 86 segundos, esto sugiere que aunque ANN puede adaptarse a configuraciones más complejas, el aumento en la profundidad de la red y la regularización no siempre se tradujo en mejoras significativas de rendimiento.

En comparación con XGBoost, tuvo un rendimiento superior y más consistente, con precisiones que variaron de 59.06% a 68.98%, donde el mejor desempeño se obtuvo con una profundidad máxima de 5, una tasa de aprendizaje baja de 0.01, y 200 árboles estimadores, además, el tiempo de ejecución fue más eficiente, promediando 44 segundos, a diferencia de ANN, el ajuste de hiperparámetros de XGBoost permitió encontrar configuraciones que mejoraron la precisión sin aumentar significativamente el tiempo de cómputo.

## Conclusión.

Para este conjunto de pruebas, Random Forest mostró el mejor equilibrio entre precisión y capacidad de ajuste, logrando resultados óptimos con configuraciones complejas, mientras que XGBoost destacó por ser el modelo más eficiente entre los analizados en términos de tiempo y consistencia de rendimiento; y Naive Bayes fue la opción más rápida y sencilla, adecuada para aplicaciones donde la velocidad es importante, pero su precisión, aunque ligeramente menor al resto, es aceptable. Para ANN, aunque flexible, no pudo igualar la eficiencia y precisión de los demás modelos, sugiriendo que puede no ser la opción más adecuada.

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
