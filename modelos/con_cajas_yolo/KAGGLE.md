# BCS por regresión en Kaggle

El entregable es [bcs_kaggle.ipynb](bcs_kaggle.ipynb). Es autónomo: sus celdas generan los módulos en working; no necesita los archivos Python del repositorio para ejecutarse en Kaggle.

1. Importar el notebook en Kaggle y adjuntar el export COCO con **imágenes** como dataset privado. Debe contener `train`, `valid` y `test`, cada uno con `_annotations.coco.json` y sus fotos. Puede existir una carpeta contenedora.
2. Ejecutar de arriba abajo con las banderas iniciales para auditar y preparar, incluso con acelerador desactivado. La configuración enumera las entradas; si hay varios exports, seleccionar `DATA_ROOT` explícitamente.
3. Revisar distribución BCS, proporciones después de exclusiones y muestras de recortes. Se conservan las particiones originales; las fotos ambiguas solo se excluyen de regresión.
4. Consultar los límites actuales mostrados por Kaggle. Activar GPU para las etapas de redes; comenzar con una GPU aunque la sesión disponga de dos T4. Cada etapa usa un proceso y libera su memoria al terminar.
5. Habilitar primero las pruebas cortas y revisar sus estimaciones. Habilitar los entrenamientos largos después. Desactivar las etapas ya completadas; los artefactos compatibles se reutilizan.
6. Evaluar primero con `EVALUATION_SPLIT="val"`. Cerrar decisiones antes de habilitar test mediante `FINAL_TEST_CONFIRMED=True`.
7. Exportar y guardar una versión **con outputs**, comprobar sus archivos y descargar el ZIP. Adjuntar esos outputs junto con el dataset original en otra sesión; configurar `RESUME_ROOT` y mantener los hiperparámetros.

El notebook explica las opciones de persistencia. Guardar código o dejar archivos en working no garantiza conservarlos después de perder la sesión. Una parada por presupuesto es preventiva por época; una terminación externa puede perder la época en curso.

## Etapas sugeridas

| Ejecución | Banderas a activar |
|---|---|
| Preparación | `RUN_AUDIT`, `RUN_PREPARE_DATA`, `RUN_EXPORT` |
| Detector | `RUN_DETECTOR_SMOKE`, después `RUN_TRAIN_DETECTOR`, `RUN_EXPORT` |
| Regresor | `RUN_REGRESSOR_SMOKE`, después `RUN_TRAIN_REGRESSOR`, `RUN_EXPORT` |
| Evaluación | `RUN_EVALUATION`, `RUN_EXPORT` |
| Foto nueva | `RUN_INFERENCE`, `RUN_EXPORT`, con `INFERENCE_IMAGE` configurada |

Los smoke tests fuerzan una pausa y recuperación. El smoke de regresión usa pesos aleatorios para verificar técnicamente el guardado/carga y el optimizador; sus métricas no evalúan la solución final. Si falta presupuesto, continuar en otra sesión; no se recortan automáticamente las épocas, resolución ni particiones.

## Artefactos y reanudación

Todo se guarda en `/kaggle/working/bcs_project/artifacts/`:

- `manifests/`: inventario, exclusiones, particiones y muestras, con rutas relativas.
- `configs/`: configuración central, entorno instalado y YAML de detección regenerado por sesión.
- `detector/<EXPERIMENT_ID>/`: `best.pt`, `last.pt`, `resume.pt`, snapshots recuperables y salida nativa Ultralytics.
- `regressor/<EXPERIMENT_ID>/`: selección global y fases `head/` y `fine_tuning/`, cada una con `best.keras`, `last.keras`, snapshots, historial y estado.
- `state/`, `histories/`, `reports/`: registro de etapas, historiales, auditoría, sondas de entorno, estimaciones y evaluación.
- `inventory.json`: hashes para comprobar el export antes de importarlo en otra sesión.

Reanudar significa recuperar el optimizador y la época del mismo experimento. En YOLO se usa el snapshot recuperable anterior a la limpieza final de Ultralytics; no se presupone que cualquier `best.pt` o `last.pt` mantenga el estado. Cambiar épocas o configuración del detector exige otro experimento desde pesos. En Keras se restauran paciencia, mejor MAE, optimizador y época; el cambio entre fases crea deliberadamente otro optimizador. Ninguno promete continuidad idéntica a nivel de bits.

Cambiar los datos, la tolerancia de auditoría o el tamaño de los recortes requiere otro `WORK_ROOT` para preparar artefactos independientes. Cambiar los hiperparámetros de entrenamiento o la semilla requiere otro `EXPERIMENT_ID` y repetir su prueba corta.

Usar únicamente checkpoints de procedencia conocida: los checkpoints PyTorch pueden contener objetos serializados con pickle. Mantener privados los outputs, que incluyen imágenes preparadas, metadatos y resultados por animal. El notebook versionado debe permanecer sin salidas ejecutadas.

## Mantener y comprobar el notebook

Editar los módulos en `kaggle_src/` y regenerar desde la raíz:

```bash
python3 modelos/con_cajas_yolo/build_kaggle_notebook.py
python3 -m unittest discover -s modelos/con_cajas_yolo/tests -p test_kaggle.py -v
```

Las pruebas locales necesitan Python y Pillow. Cubren COCO, fugas, conversiones, hashes, traslado entre sesiones, métricas con abstenciones, presupuesto y recuperación simulada de checkpoints. El notebook contiene las verificaciones GPU reales; estas deben pasar en Kaggle antes de considerar verificados los entrenamientos y su reanudación sobre T4.

Ultralytics se fija a `8.4.0`, que incluye YOLO26. No se reemplazan automáticamente PyTorch, TensorFlow ni CUDA de Kaggle. Si la sesión trae otra versión de Ultralytics, la comprobación se detiene con indicaciones. Las sondas guardan versiones y código de la API de reanudación de la instalación real. La configuración de una GPU es la única admitida por esta entrega; DDP y MirroredStrategy necesitan una validación y comparación independientes antes de habilitarlos.
