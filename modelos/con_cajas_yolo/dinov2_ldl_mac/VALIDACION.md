# Validación técnica

Antes de subir esta carpeta corrimos estas pruebas en macOS ARM64 con Python 3.12, TensorFlow 2.18.1, Keras 3.8.0, tensorflow-metal 1.2.0, PyTorch 2.9.1 y NumPy 2.0.2.

Con datos ficticios probamos la comparación pareada, la fórmula LDL de TensorFlow contra NumPy, los gradientes, guardar y cargar el modelo con su optimizador, el ETL, las dos vistas, el entrenamiento, la reanudación, la evaluación y la oclusión. También probamos el caso en que el modelo se abstiene en todas las fotos y que el código rechace evaluar test si no se marca como exploratorio.

Además hicimos una prueba de integración en CPU con DINOv2 real y una imagen sintética. Da las 1536 características esperadas, completa un paso de TensorFlow con gradientes y genera el mapa de oclusión. Por último revisamos la sintaxis de Python y bash, la ayuda de los comandos y que el commit no llevara datos privados.

Estas pruebas corrieron sin GPU, así que la GPU se comprueba con el instalador en cada Mac. Sirven para saber que el código funciona, pero no miden la precisión del modelo ni sustituyen una evaluación con vacas que el modelo no vio.

No incluimos datos, etiquetas, identidades, predicciones por vaca ni pesos entrenados. Los CSV de comparación los pone cada integrante. Esta receta cambia el backend y la inicialización respecto al modelo anterior, y si se entrena otro detector también cambia el sistema completo, así que una diferencia en resultados no se puede atribuir solo al clasificador.
