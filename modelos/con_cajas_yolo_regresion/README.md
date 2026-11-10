# YOLO + regresión de BCS

Esta carpeta contiene la versión v2 de un modelo que detecta vaca y grupa con YOLO26n y estima un BCS continuo por fotografía mediante EfficientNet-B0.

## Archivos y estado

- `entrenar_yolo_regresion_v2.ipynb`: prepara los datos, entrena el detector y el regresor, selecciona la fase por MAE de validación y evalúa `val` por defecto.
- `kaggle_yolo_regresion.ipynb`: carga los pesos ya entrenados y reproduce la evaluación. No entrena modelos.

El código de entrenamiento ya está disponible, pero no se han incluido datos, pesos ni resultados individuales. Todavía falta comprobar una ejecución completa a partir de un entorno limpio.

## Datos y artefactos privados

El entrenamiento requiere el dataset autorizado con particiones `train`, `valid` y `test` y sus anotaciones. El notebook guarda configuración, huellas de datos, checkpoints y resultados en `/kaggle/working/bcs_secuencial_v2_regresion`.

El notebook de evaluación requiere esos resultados v2, incluidos `configuracion.json`, `seleccion_regresor.json`, `detector/weights/best.pt` y el `regresor/<fase>/best.keras` elegido. Todo ello permanece fuera de GitHub.

## Ejecución

El entorno previsto es Kaggle con GPU. El código exige Ultralytics 8.4.0 y utiliza las bibliotecas del entorno Kaggle; aún no existe un archivo de dependencias completo y probado.

Antes de ejecutar el entrenamiento, revisa `DATOS`, `SALIDA` y los parámetros de la celda de configuración. Para la evaluación, revisa `ORIGEN_ENTRENAMIENTO`, `DATOS`, `SALIDA` y `PARTICION_EVALUACION`.

El test de esta versión ya fue observado. Una nueva configuración debe seleccionarse con entrenamiento y validación; repetir test no produce una nueva evaluación independiente. Nunca guardes salidas con fotografías, nombres de imagen o predicciones dentro de los notebooks versionados.

## Pendiente para completar el estándar

- Registrar y probar las dependencias efectivas del entorno.
- Añadir pruebas automatizadas con datos ficticios.
- Vincular los resultados históricos con el commit, los datos y los checkpoints exactos.
