# BCS con DINOv2 + LDL para Mac Apple Silicon

Esta es una variante del área `con_cajas_yolo`. YOLO localiza la vaca y la grupa, DINOv2 congelado procesa dos vistas de la grupa y una cabeza LDL estima el BCS continuo y la categoría. La carpeta incluye preparación de datos, entrenamiento, checkpoints, evaluación y mapas de sensibilidad por oclusión.

Aquí solo hay código, pruebas con datos ficticios y documentación. Los datos, las anotaciones, las regiones marcadas a mano, los checkpoints y los resultados se guardan en cada equipo y no se suben a commits, Pull Requests ni Releases.

## Arquitectura

| Componente | Función | Dispositivo |
|---|---|---|
| YOLO | Detectar `cow` y `ass` y exigir un par único | PyTorch/MPS, GPU Apple |
| Preparación | Recortar la grupa y rellenar a 224 × 224 sin estirar | CPU |
| DINOv2 ViT-S/14 congelado | Extraer el CLS y la media de parches de cada vista | PyTorch/MPS, GPU Apple |
| Cabeza LDL | LayerNorm, Dense 128, GELU, Dropout 0.30 y Dense K | TensorFlow/Metal, GPU Apple al entrenar |
| Evaluación de la cabeza | Obtener distribución, esperanza y categoría | CPU, para no competir con DINOv2 por Metal |

La primera vista es la grupa completa. La segunda es por defecto el 65 % superior del recorte (`[0,0,1,0.65]`), que es una aproximación geométrica de la zona de cadera y sacro y no un detector de huesos. Con el modo manual se marcan las regiones a mano en cada imagen, y con el modo `full` se apaga la segunda vista para medir cuánto aporta.

Cada vista da el token CLS (384 valores) y el promedio de parches (384 valores), así que la cabeza recibe 1536 características, normalizadas con las estadísticas de las fotos originales de train. DINOv2 queda congelado en modo evaluación y se usa en PyTorch con su implementación oficial. La parte entrenable que estima el BCS está en TensorFlow.

## 1. Instalación

Se necesita macOS con Apple Silicon, Terminal sin Rosetta y Python 3.11 o 3.12 ARM64. Si faltan las Xcode Command Line Tools, se instalan con `xcode-select --install`. Desde la raíz del repositorio:

```bash
cd modelos/con_cajas_yolo/dinov2_ldl_mac
bash instalar_mac.sh
```

El instalador crea `.venv-mac`, fija PyTorch 2.9.1, torchvision 0.24.1, TensorFlow 2.18.1, tensorflow-metal 1.2.0, Keras 3.8.0 y NumPy 2.0.2, corre `pip check` y comprueba que MPS y Metal funcionen con operaciones reales y con DINOv2. Para usar otro Python compatible:

```bash
PYTHON_BIN=/ruta/a/python3.12 bash instalar_mac.sh
```

Los pesos públicos de DINOv2 se descargan la primera vez. No hace falta CUDA ni WSL. El instalador usa los certificados de certifi para las descargas y no desactiva la verificación HTTPS. Si el Python de python.org da errores de certificados con pip, hay que correr el `Install Certificates.command` de esa instalación.

Si el equipo no tiene MPS o Metal, el ejecutor se detiene con un error. Para pruebas en CPU, los comandos individuales aceptan `--device cpu`. Con redes y lotes pequeños la GPU no siempre es más rápida.

## 2. Datos

El dataset va fuera del repositorio. En los ejemplos hay que cambiar `/ruta/datos_autorizados/dataset_yolo` por su ubicación. Las particiones se dejan como están.

| Ruta dentro del dataset | Contenido |
|---|---|
| `images/train`, `images/val`, `images/test` | Fotografías |
| `labels/train`, `labels/val`, `labels/test` | Cajas YOLO, 0 = cow y 1 = ass |
| `metadata.csv` | Identidad, BCS, estado y tags |
| `data.yaml` | Clases del detector |
| `coco_original/train/_annotations.coco.json` | COCO original de train |
| `coco_original/valid/_annotations.coco.json` | COCO original de validación |
| `coco_original/test/_annotations.coco.json` | COCO original de test |

También acepta el COCO original con fotos y JSON en `train/valid/test`. Los tags de cada imagen deben tener la identidad numérica y `BCS_3_0` o un formato equivalente. Las fotos con BCS ambiguo no entran al modelo BCS pero sí al detector. El catálogo de clases sale de train y debe tener categorías consecutivas cada 0.25 BCS.

El ETL revisa la geometría y que COCO, YOLO y metadatos coincidan, y rechaza identidades o fotos idénticas repetidas entre particiones. No detecta IDs mal anotados ni todas las fotos casi duplicadas. Las identidades no entran como características al modelo.

## 3. Ejecutar el flujo completo

Antes de entrenar con el COCO original se corre una comprobación desde esta carpeta, después de `instalar_mac.sh`:

```bash
mkdir -p /ruta/privada/resultados_dino
RUN_DESTINO="/ruta/privada/resultados_dino/corrida_01"
.venv-mac/bin/python preflight.py \
  --datos /ruta/datos_autorizados/coco_cajas_v2_aprobado \
  --salida "$RUN_DESTINO"
```

Revisa macOS ARM64, Python, las dependencias, que existan las imágenes de `train` y `valid`, las clases `cow` y `ass`, y que la carpeta de salida todavía no exista. No crea nada, no lee `test` y no entrena. Si falla por dependencias, se vuelve a correr `bash instalar_mac.sh` en lugar de instalar paquetes sueltos.

Si la comprobación pasa, la corrida usa esa misma carpeta:

```bash
env RUN_DESTINO="$RUN_DESTINO" caffeinate -i bash ejecutar_mac.sh \
  /ruta/datos_autorizados/coco_cajas_v2_aprobado
```

El ejecutor repite la comprobación y se detiene si `RUN_DESTINO` ya existe, para no sobrescribir otra corrida. Luego comprueba la GPU, prepara los datos, entrena la cabeza LDL y entrena un detector nuevo si no se le pasa uno como segundo argumento. Si no se define `RUN_DESTINO`, crea una carpeta nueva dentro de `runs/` y guarda su ruta en `ULTIMA_EJECUCION.txt`. Todo lo que está en `runs/` es privado y Git lo ignora.

Al terminar hay que revisar dentro de `RUN_DESTINO` los archivos `FINALIZADO.txt`, `val_gt/resumen.md`, `val_pipeline/resumen.md`, `val_pipeline/metricas.json`, `val_pipeline/predicciones.csv` y `val_pipeline/por_vaca.csv`, confirmar que hay 36 fotos de validación elegibles y anotar la cobertura y las abstenciones. Si falta `FINALIZADO.txt`, la corrida no terminó y el error está en `ejecucion.log`. El ejecutor nunca evalúa `test`.

Sin `RUN_DESTINO` y sin detector previo, basta con:

```bash
caffeinate -i bash ejecutar_mac.sh /ruta/datos_autorizados/dataset_yolo
```

Para reutilizar un detector ya entrenado:

```bash
caffeinate -i bash ejecutar_mac.sh \
  /ruta/datos_autorizados/dataset_yolo \
  /ruta/modelos_privados/detector/weights/best.pt
```

Como tercer argumento opcional se puede pasar un CSV del notebook anterior con `clave,real,estado,caja_predicha` para reconstruir exactamente sus recortes. Las cajas anotadas del dataset no sustituyen al `best.pt` entrenado.

Las etapas van en este orden: comprobar la GPU, ETL y EDA, características de train y val, entrenar LDL, evaluar con cajas anotadas, entrenar YOLO si hace falta, recortes predichos, evaluación del sistema completo, mapas de oclusión y comparaciones opcionales. Después del ETL no se deben mover los datos ni la carpeta de trabajo, porque el manifiesto guarda rutas absolutas.

## 4. Entrenamiento LDL

- La distribución objetivo es una gaussiana discreta centrada en la etiqueta, con sigma 0.125 BCS. Es un suavizado que elegimos y no una medida de la diferencia entre evaluadores.
- La pérdida es entropía cruzada suave más 0.5 × el promedio de las diferencias cuadráticas entre las CDF, sobre los primeros K−1 límites.
- AdamW con lr 0.001, weight decay 0.0001, lote de 32, máximo 100 épocas y paciencia de 15.
- Cada foto pesa el inverso del número de fotos de su vaca, para que cada vaca aporte lo mismo a la pérdida. No hay sobremuestreo por clase.
- El checkpoint se elige por el MAE continuo de validación, nunca por test. El MAE de train y val se reporta por foto.
- La primera corrida usa una vista por muestra y sin aumentos. `features --views 4` permite probar aumentos solo en train, y al voltear la foto también voltea la ROI.

La salida continua es la esperanza de la distribución BCS. La categoría es el valor del catálogo más cercano a esa esperanza (en un empate, el menor) y no el argmax. Los scores no están calibrados. Si se cambia el backend, la inicialización o el detector, una diferencia en los resultados ya no se puede atribuir a una sola parte.

## 5. Métricas y comparación con otro modelo

```bash
RUN=$(cat ULTIMA_EJECUCION.txt)
cat "$RUN/val_gt/resumen.md"
cat "$RUN/val_pipeline/resumen.md"
```

En `metricas.json`, la clave `continuous_bcs` tiene las métricas continuas y el nivel superior las de categoría. Se reportan la cobertura, las abstenciones, el MAE y el RMSE de las fotos clasificadas, las tolerancias ±0.25 y ±0.50 sobre las clasificadas y sobre todas las elegibles, F1 y kappa por categoría, y la mediana de train como referencia constante. En las métricas globales una abstención cuenta como fallo. Las proporciones se multiplican por 100.

Las comparaciones pareadas usan categorías en los dos modelos, porque comparar el MAE continuo de uno con el MAE redondeado del otro no sería justo. Que dos modelos se abstengan en las mismas fotos no garantiza que vean los mismos píxeles, así que para aislar el clasificador hay que conservar el detector y las coordenadas. Si se entrena un detector nuevo, se indica junto a la comparación del sistema completo y las cajas anotadas se revisan por separado.

El repositorio no trae predicciones reales. Para comparar, cada integrante usa su propio CSV con `split,file_name,true_bcs,pred_bcs,status`, y los IDs se toman de su manifiesto privado. Por ejemplo:

```bash
.venv-mac/bin/python comparar.py \
  --ordinal /ruta/resultados_privados/baseline_val_gt.csv \
  --dinov2 "$RUN/val_gt/predicciones.csv" \
  --work "$RUN/datos" --out "$RUN/comparacion_gt_manual"
```

Para que el ejecutor haga la comparación solo, se definen `BASELINE_GT_CSV` o `BASELINE_PIPELINE_CSV` con rutas fuera del repositorio antes de correr `ejecutar_mac.sh`. Si no se definen, se omite la comparación y se guardan solo las métricas de este modelo.

El bootstrap por vaca es exploratorio y no corrige que la validación se haya usado para elegir el modelo. Si test ya se usó para decidir algo, sus cifras dejan de ser una prueba independiente.

## 6. Regiones anatómicas y diagnóstico visual

El 65 % superior es solo una aproximación y puede incluir relleno o perder parte de la anatomía si cambia el ángulo. Para revisar o marcar regiones a mano:

```bash
PY=.venv-mac/bin/python
$PY pipeline.py etl --data /ruta/datos_autorizados/dataset_yolo --work runs/manual/datos
$PY revisar_regiones.py --work runs/manual/datos --out runs/manual/rois.csv --interactive
$PY pipeline.py features --work runs/manual/datos --out runs/manual/features \
  --roi-mode manual --roi-csv runs/manual/rois.csv --splits train val --device mps
```

En el editor se arrastra la región visible de pelvis, sacro y base de la cola. Enter guarda, y Q o cerrar la ventana termina. El editor no muestra el BCS. El CSV tiene `key,x1,y1,x2,y2` con coordenadas normalizadas sobre el recorte de 224 × 224 con relleno. Para continuar un CSV a medias se usa otro `--out` y `--from-csv` con el anterior.

Las claves de las cajas anotadas son `annotated/train/archivo.jpg` y las de los recortes de YOLO `predicted/val/archivo.jpg`. Cada geometría se marca por separado, porque si cambian las cajas las coordenadas anteriores ya no sirven. El modo manual necesita todas las anotaciones y también una ROI marcada para cada foto nueva, porque no entrena un localizador automático.

Para medir cuánto aporta la segunda vista se crea otro experimento con `ROI_MODE=full` y los mismos datos, detector y ajustes. En ese caso la rama adicional queda en cero. Las regiones y los hiperparámetros nunca se eligen con test.

El ejecutor genera `explicaciones_val/*.png` con la foto original, la grupa con su ROI, el mapa de sensibilidad y las probabilidades. Para cada celda de una cuadrícula de 7 × 7 tapa esa zona con el color promedio, vuelve a calcular las dos vistas y mide cuánto cambia el BCS continuo. Los JSON guardan el signo, la escala y pruebas tapando la ROI y el exterior. Los mapas no se normalizan a un máximo de 1, así que hay que leer sus barras y no solo el brillo.

Estos mapas no son atención ni Grad-CAM y no detectan huesos. Tapar una zona crea una imagen artificial, así que una zona sensible no prueba que esa parte del cuerpo sea la causa, y un cambio pequeño tampoco prueba que la zona no importe. Las fotos donde el modelo se abstuvo no tienen mapa.

`EXPLAIN_LIMIT=6` limita cuántas fotos se explican. `pipeline.py explain --grid 4 --limit 6` reduce la resolución y la cantidad sin cambiar las métricas de evaluación.

## 7. Checkpoints e interrupciones

```bash
RUN=$(cat ULTIMA_EJECUCION.txt)
.venv-mac/bin/python pipeline.py train --features "$RUN/features_gt" \
  --run "$RUN/modelo_ldl_tf" --epochs 100 --patience 15 --device mps --resume
```

`estado.json` apunta al último y al mejor modelo `.keras` completos, con la historia y el estado de la paciencia. El optimizador se recupera del último modelo y la normalización se guarda en `normalizacion.npz`. Al reanudar hay que usar los mismos parámetros y la misma caché, aunque se puede subir el total de épocas. La paciencia alcanzada se respeta. El resultado puede no ser idéntico al de una corrida sin interrupciones, ni entre equipos distintos.

Cada modelo se escribe antes de actualizar `estado.json`, y solo se guardan el último y el mejor. Si se interrumpe, puede quedar un archivo suelto sin que se dañe el estado anterior. No hay que editar `estado.json` ni borrar la carpeta al reanudar.

La extracción de características no se puede reanudar a medias y se repite con otra salida. YOLO se reanuda con `detector-train --resume /ruta/last.pt` y el mismo trabajo y dispositivo. Si falla una etapa posterior, se repite solo esa etapa, porque volver a lanzar el ejecutor crea otra corrida. Si falta memoria, se bajan los lotes a 1 o 2 en una corrida nueva. Las dos vistas se procesan una después de la otra.

## 8. Foto nueva

```bash
.venv-mac/bin/python pipeline.py predict \
  --image /ruta/imagenes_privadas/foto.jpg \
  --detector /ruta/modelos_privados/best.pt \
  --model /ruta/resultados_privados/modelo_ldl_tf \
  --out runs/foto_nueva --device mps
```

Guarda las detecciones, el recorte, la ROI, el BCS continuo y la categoría. No identifica al animal y no estima el BCS si no encuentra un único par de vaca y grupa. Si se entrenó con ROI manual, hay que agregar `--roi x1 y1 x2 y2`. Es un prototipo académico y no sustituye la evaluación del veterinario.

## 9. Pruebas

Desde esta carpeta, con las dependencias instaladas:

```bash
.venv-mac/bin/python -m unittest discover -s tests -v
.venv-mac/bin/python tests/smoke_dino_tf.py
bash -n instalar_mac.sh ejecutar_mac.sh
```

Las pruebas unitarias usan datos ficticios temporales y cubren la fórmula LDL contra NumPy, los gradientes, guardar y cargar el modelo con su optimizador, el ETL, la reanudación, las métricas, las abstenciones, la oclusión y la comparación pareada. `smoke_dino_tf.py` descarga DINOv2 y conecta el extractor real, la cabeza de TensorFlow y la oclusión en CPU. Estas pruebas revisan que el código funcione, pero no miden la precisión del modelo ni comprueban Metal. La GPU la comprueba el instalador en cada Mac.

Más detalle en [VALIDACION.md](VALIDACION.md). Para contribuir se usa una rama del área y un Pull Request, como indica [CONTRIBUTING.md](../../../CONTRIBUTING.md). Además de los datos, no se suben archivos `.keras`, `.npz`, `.npy` ni la carpeta `runs/`, porque las salidas pueden tener identificadores reales.

## Fuentes técnicas

- https://developer.apple.com/metal/tensorflow-plugin/
- https://docs.pytorch.org/docs/main/notes/mps.html
- https://github.com/facebookresearch/dinov2
- https://arxiv.org/abs/1611.01731
- https://arxiv.org/abs/1611.05916
- https://arxiv.org/abs/1311.2901
