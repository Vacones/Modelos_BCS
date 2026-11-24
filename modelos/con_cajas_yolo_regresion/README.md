# YOLO + regresión continua de BCS

Este modelo detecta la vaca y la grupa con YOLO26n. EfficientNet-B0 estima un BCS continuo **por fotografía** a partir del recorte de la grupa. La evaluación del sistema completo usa cajas predichas y la del regresor solo usa cajas anotadas. Se informa la cobertura cuando el sistema se abstiene.

## Código disponible

- `entrenar_yolo_regresion_v2.ipynb`: notebook histórico preparado para Kaggle.
- `entrenar_yolo_regresion_v2_mac.ipynb`: entrenamiento y validación en Apple Silicon. Empieza con una prueba corta de una época por fase.
- `kaggle_yolo_regresion.ipynb`: evaluación de pesos ya entrenados, sin entrenar.
- `validar_datos.py`: comprueba que las identidades no crucen entre particiones COCO. Cuenta todas las identidades registradas, incluidas las de imágenes ambiguas.
- `evaluar_por_vaca.py`: calcula métricas agregadas por vaca a partir de las predicciones privadas de validación.
- `tests/`: pruebas con datos ficticios.

## Datos privados

El export aprobado de Roboflow contiene `train`, `valid` y `test`, cada uno con imágenes y `_annotations.coco.json`. La versión revisada tiene 243, 36 y 56 imágenes respectivamente. Cuatro imágenes de entrenamiento tienen más de una identidad. Sirven para entrenar el detector, pero no entran al regresor BCS. El notebook comprueba imágenes, cajas, etiquetas, huellas de archivo y separación entre particiones antes de entrenar.

El COCO, las fotografías, el manifiesto, las predicciones individuales y los checkpoints se guardan fuera de GitHub. El notebook Mac toma la ruta del COCO de la variable `VACONES_COCO_DIR`.

## Entorno Mac

Usa Python 3.12 en macOS arm64 y el entorno aislado del modelo:

```bash
uv venv --python 3.12 .venv-regresion-mac
source .venv-regresion-mac/bin/activate
uv pip install -r modelos/con_cajas_yolo_regresion/requirements-mac.txt
python -m unittest discover -s modelos/con_cajas_yolo_regresion/tests -p 'test_*.py' -v
```

Selecciona `.venv-regresion-mac` como kernel del notebook en VS Code. YOLO usa PyTorch MPS y el regresor usa TensorFlow Metal. `requirements-mac.txt` fija las dependencias directas que usamos y `configuracion.json` guarda las versiones y huellas del experimento. Algunas operaciones MPS avisan que no son deterministas, así que la misma semilla puede dar resultados un poco distintos.

## Entrenamiento y validación

1. Ejecuta `python -m modelos.con_cajas_yolo_regresion.validar_datos RUTA_DEL_COCO` para comprobar identidades entre particiones.
2. Abre `entrenar_yolo_regresion_v2_mac.ipynb`. Con `MODO_PRUEBA=True`, cada fase entrena una época. Con `False`, el detector entrena 80 épocas, la cabeza del regresor 60 y el ajuste fino 80. Los dos modos usan carpetas de resultados distintas.
3. Mantén `PARTICION_EVALUACION="val"` mientras seleccionas modelo y umbrales. Ejecuta las celdas en orden. La carpeta de salida se indica con `VACONES_RESULTADOS_DIR` y debe quedar fuera del repositorio.
4. Revisa `best.pt`, `best.keras`, `seleccion_regresor.json`, las métricas y la cobertura. El notebook muestra métricas por fotografía y compara la base y el regresor en las mismas fotos cubiertas por el pipeline.

Para obtener un resumen **por vaca** a partir de la ejecución local, usa los archivos privados de validación:

```bash
python -m modelos.con_cajas_yolo_regresion.evaluar_por_vaca \
  --datos RUTA_PRIVADA/datos.json \
  --predicciones RUTA_PRIVADA/predicciones_val.csv \
  --salida RUTA_PRIVADA/metricas_vaca_val.json
```

La agregación por vaca usa la media de las predicciones disponibles por fotografía y da el mismo peso a cada vaca. Reporta cobertura por vaca y por foto. Para comparar el sistema completo con la base y el regresor con cajas anotadas, usa **las mismas fotos donde el pipeline produjo una predicción**. El resumen no contiene identificadores ni predicciones individuales, pero debe revisarse antes de publicarlo.

## Límites de interpretación

El conjunto `test` ya se usó en una ejecución anterior, así que una nueva medición ahí ya no sería una prueba independiente. No se usa para elegir fase, umbrales ni configuración. Las cifras del notebook Mac proceden de un entorno distinto del histórico de Kaggle. La validación tiene solo 10 vacas, así que sus resultados no bastan para usar el modelo en el rancho.

Los notebooks se suben sin salidas, adjuntos ni contadores de ejecución. Los archivos de resultados pueden contener identificadores, BCS y recortes, así que no se agregan al repositorio.
