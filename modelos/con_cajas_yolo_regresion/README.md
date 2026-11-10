# YOLO + regresión de BCS

Esta carpeta contiene el notebook de evaluación de un modelo que detecta la vaca y su grupa con YOLO26n y estima un BCS continuo a partir del recorte.

## Estado del código

`kaggle_yolo_regresion.ipynb` carga el detector y el regresor ya entrenados, verifica los datos y calcula resultados por fotografía. No contiene el entrenamiento que produjo esos pesos. Por tanto, esta carpeta permite reproducir la evaluación **solo si se proporcionan los datos y artefactos privados correspondientes**; todavía no permite reproducir el entrenamiento desde cero.

El conjunto de prueba de esta versión ya fue consultado. Sus resultados no deben usarse para elegir nuevos parámetros ni presentarse como una nueva evaluación independiente.

## Entradas privadas

El notebook requiere el dataset autorizado con particiones `train`, `valid` y `test`, además de los artefactos de entrenamiento v2: `configuracion.json`, `seleccion_regresor.json`, `detector/weights/best.pt` y el `regresor/<fase>/best.keras` seleccionado. Estos archivos permanecen fuera de GitHub.

## Ejecución

El entorno previsto es Kaggle con GPU. El notebook utiliza Ultralytics 8.4.0 y las bibliotecas disponibles en ese entorno. Antes de ejecutarlo, se deben revisar `ORIGEN_ENTRENAMIENTO`, `DATOS`, `SALIDA` y `PARTICION_EVALUACION` en su celda de configuración.

Para comprobaciones de desarrollo se utiliza validación. La prueba ya observada no debe repetirse como evidencia independiente. El notebook genera métricas, predicciones y diagnósticos que pueden contener información del dataset; sus salidas no deben guardarse dentro del archivo versionado.

## Pendiente para completar el estándar

- Incorporar o localizar el código de entrenamiento v2 y sus instrucciones.
- Registrar sus dependencias y configuración efectiva.
- Añadir pruebas con datos ficticios.
- Vincular cada resultado publicado con el commit, los datos y los checkpoints utilizados.
