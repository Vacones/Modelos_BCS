# BCS con DINOv2 + LDL para Mac Apple Silicon

Implementación alternativa del área `con_cajas_yolo`: YOLO localiza vaca y grupa; DINOv2 congelado procesa dos vistas de la grupa; una cabeza LDL estima BCS continuo y categoría. Incluye preparación de datos, entrenamiento, checkpoints, evaluación y diagnóstico visual por oclusión.

Esta carpeta publica código, pruebas con datos ficticios y documentación. Los datos autorizados, anotaciones, regiones manuales, checkpoints y resultados se proporcionan y conservan localmente. No adjuntarlos a commits, Pull Requests ni Releases.

## Arquitectura

| Componente | Función | Dispositivo previsto |
|---|---|---|
| YOLO | Detectar `cow` y `ass` y exigir un par único | PyTorch/MPS, GPU Apple |
| Preparación | Recortar la grupa y rellenar a 224×224 sin estirar | CPU |
| DINOv2 ViT-S/14 congelado | Extraer CLS y media de parches de cada vista | PyTorch/MPS, GPU Apple |
| Cabeza LDL | LayerNorm → Dense128 → GELU → Dropout0.30 → DenseK | TensorFlow/Metal, GPU Apple durante entrenamiento |
| Evaluación de la cabeza | Obtener distribución, esperanza y categoría | CPU; evita competir con DINOv2 por Metal |

La primera vista conserva toda la grupa. La segunda usa por defecto el 65% superior del recorte preparado (`[0,0,1,0.65]`). Es una aproximación geométrica revisable, no un detector de huesos. El modo manual exige ROIs por imagen; el modo `full` desactiva la segunda rama para estudiar su aporte.

Cada vista produce CLS384 + promedio de parches384. La cabeza recibe1536 características, normalizadas con estadísticas de originales de train. DINOv2 permanece congelado y en modo evaluación. No se convierte su implementación oficial a TensorFlow: el componente entrenable que estima BCS sí se implementa en TensorFlow.

## 1. Instalación

Requiere macOS/Apple Silicon, Terminal sin Rosetta y Python3.11o3.12 ARM64. Instalar Xcode Command Line Tools si no están disponibles (`xcode-select --install`). Desde la raíz del repositorio:

```bash
cd modelos/con_cajas_yolo/dinov2_ldl_mac
bash instalar_mac.sh
```

El instalador crea `.venv-mac`, fija PyTorch2.9.1/torchvision0.24.1, TensorFlow2.18.1/tensorflow-metal1.2.0/Keras3.8.0 y NumPy2.0.2, ejecuta `pip check` y comprueba ambas rutas GPU con operaciones reales y DINOv2. Para otra ubicación de Python compatible:

```bash
PYTHON_BIN=/ruta/a/python3.12 bash instalar_mac.sh
```

Los pesos públicos se descargan al primer uso. No se instala CUDA ni se necesita WSL. El instalador y ejecutor configuran un bundle CA de certifi para `urllib`/`torch.hub` y Requests; no desactivan la verificación HTTPS. Si Python.org presenta problemas de certificados durante pip, ejecutar el `Install Certificates.command` de esa instalación de Python.

La detección efectiva de MPS/Metal se comprueba en el equipo. Si no hay GPU se produce un error visible; para pruebas CPU explícitas existen comandos individuales con `--device cpu`. No se promete un factor de aceleración: redes pequeñas y lotes pequeños pueden no amortizar el sobrecosto de GPU.

## 2. Datos externos

Proporcionar un dataset autorizado FUERA del checkout y sustituir `/ruta/datos_autorizados/dataset_yolo` en los ejemplos. No cambiar las particiones existentes para mejorar métricas.

| Ruta dentro del dataset externo | Contenido |
|---|---|
| `images/train`, `images/val`, `images/test` | Fotografías |
| `labels/train`, `labels/val`, `labels/test` | Cajas YOLO, 0=cow y 1=ass |
| `metadata.csv` | Identidad, BCS, estado y tags |
| `data.yaml` | Clases del detector |
| `coco_original/train/_annotations.coco.json` | COCO original train |
| `coco_original/valid/_annotations.coco.json` | COCO original validación |
| `coco_original/test/_annotations.coco.json` | COCO original test |

También admite el COCO original con fotografías y JSON en `train/valid/test`. Los tags de imagen esperados contienen identidad numérica y `BCS_3_0` o formato equivalente. Las muestras ambiguas se excluyen de BCS y se conservan para detección. El catálogo se deriva de train y requiere categorías consecutivas cada0.25BCS.

El ETL comprueba geometría y correspondencia COCO/YOLO/metadatos; rechaza identidades y hashes exactos compartidos entre particiones. No detecta por sí solo IDs mal anotados ni todas las fotos casi duplicadas. Las identidades no se entregan como características al modelo.

## 3. Ejecutar el flujo completo

Sin detector privado disponible, entrena uno nuevo a partir de los pesos públicos y las cajas externas:

```bash
caffeinate -i bash ejecutar_mac.sh /ruta/datos_autorizados/dataset_yolo
```

Si existe un detector revisado, reutilizarlo:

```bash
caffeinate -i bash ejecutar_mac.sh \
  /ruta/datos_autorizados/dataset_yolo \
  /ruta/modelos_privados/detector/weights/best.pt
```

Opcionalmente añadir como tercer argumento un CSV privado del notebook previo con `clave,real,estado,caja_predicha`, para reconstruir sus recortes exactos. Las cajas anotadas del dataset son etiquetas, no sustituyen el `best.pt` entrenado.

Orden: comprobar GPU → ETL/EDA → características train/val → entrenar LDL → evaluar con cajas anotadas → entrenar YOLO si hace falta → recortes predichos → evaluación end-to-end → mapas → comparaciones opcionales.

Cada ejecución crea un directorio nuevo en `runs/` y un puntero local `ULTIMA_EJECUCION.txt`. Las salidas son privadas y están ignoradas por Git. No se evalúa test en el ejecutor; EDA describe las particiones existentes. No mover datos/carpeta de trabajo tras ETL, porque el manifiesto conserva rutas absolutas.

## 4. Entrenamiento LDL

- Distribución objetivo gaussiana discreta, centrada en la anotación; sigma inicial0.125BCS. Es una hipótesis de suavizado, no incertidumbre medida de anotadores.
- Pérdida: entropía cruzada suave +0.5× promedio de diferencias cuadráticas entre CDF, sobre los primeros K−1 límites.
- AdamW lr0.001, decay0.0001, lote32, máximo100épocas, paciencia15.
- Ponderación inversa al número de fotos/vistas por animal: cada vaca aporta el mismo peso total a la pérdida. No hay sobremuestreo por clase.
- Selección por MAE CONTINUO de validación; no por test. MAE train/val se reporta por foto.
- Una vista por muestra, sin aumentos, en la primera ejecución. `features --views 4` permite estudiar aumentos solamente en train; al voltear transforma también la ROI.

La salida continua es la esperanza de la distribución BCS. La categoría es el valor del catálogo más cercano a esa esperanza (empates al menor), no argmax. Los scores no están calibrados. Cambiar backend, inicialización o detector impide atribuir toda diferencia a una sola parte de la arquitectura.

## 5. Métricas y comparación con un baseline externo

```bash
RUN=$(cat ULTIMA_EJECUCION.txt)
cat "$RUN/val_gt/resumen.md"
cat "$RUN/val_pipeline/resumen.md"
```

`metricas.json → continuous_bcs` contiene las métricas continuas; el nivel superior contiene métricas por categoría. Se informa cobertura, abstenciones, MAE/RMSE entre clasificadas, tolerancias±0.25/±0.50 sobre clasificadas y sobre todas las elegibles, F1/kappa por categoría y mediana constante de train como baseline. En métricas globales las abstenciones cuentan como no aciertos. Las proporciones se multiplican por100 para porcentajes.

Las comparaciones pareadas usan categorías en ambos modelos. No comparar MAE continuo de uno con MAE redondeado del otro como si fueran el mismo criterio. La coincidencia de estados no demuestra coincidencia de píxeles; conservar detector y coordenadas para aislar el clasificador. Con detector nuevo, informar ese cambio junto a la comparación end-to-end y revisar por separado las cajas anotadas.

El repositorio NO incorpora referencias con predicciones reales por imagen. El usuario proporciona localmente los CSV de baseline con `split,file_name,true_bcs,pred_bcs,status`; los IDs se vinculan desde su manifiesto privado. Por ejemplo:

```bash
.venv-mac/bin/python comparar.py \
  --ordinal /ruta/resultados_privados/baseline_val_gt.csv \
  --dinov2 "$RUN/val_gt/predicciones.csv" \
  --work "$RUN/datos" --out "$RUN/comparacion_gt_manual"
```

Para activar comparación automática en una nueva ejecución, definir `BASELINE_GT_CSV` y/o `BASELINE_PIPELINE_CSV` con rutas privadas externas antes de `ejecutar_mac.sh`. Si no se proporcionan, se omite la comparación y se conservan las métricas del modelo nuevo.

Bootstrap por vaca: exploratorio, sin corregir selección repetida por validación. Si test ya informó decisiones, una confirmación independiente requiere nuevos animales. No se publican resultados científicos privados ni se afirma superioridad general de esta arquitectura.

## 6. Regiones anatómicas y diagnóstico visual

El 65% superior es solo una aproximación: puede incluir relleno o perder anatomía con un ángulo distinto. Para revisión/marcado local:

```bash
PY=.venv-mac/bin/python
$PY pipeline.py etl --data /ruta/datos_autorizados/dataset_yolo --work runs/manual/datos
$PY revisar_regiones.py --work runs/manual/datos --out runs/manual/rois.csv --interactive
$PY pipeline.py features --work runs/manual/datos --out runs/manual/features \
  --roi-mode manual --roi-csv runs/manual/rois.csv --splits train val --device mps
```

Arrastrar región visible de pelvis/sacro/base de cola; Enter guarda, Q/cerrar termina. El editor no muestra BCS. CSV: `key,x1,y1,x2,y2`, coordenadas normalizadas sobre el recorte224×224 con relleno. Para continuar un CSV parcial usar otro `--out` y `--from-csv` con el previo.

Claves anotadas: `annotated/train/archivo.jpg`; para recortes de YOLO: `predicted/val/archivo.jpg`. Marcar cada geometría por separado: no reutilizar ciegamente las coordenadas si cambiaron las cajas. El modo manual exige anotaciones completas y también una ROI humana para fotos nuevas; no entrena un localizador anatómico automático.

Para comparar la vista adicional, crear otro experimento con `ROI_MODE=full`, mismos datos, detector y ajustes. La rama adicional queda a cero. No escoger regiones ni hiperparámetros por test.

El ejecutor genera `explicaciones_val/*.png`: original, grupa/ROI, mapa de sensibilidad y probabilidades. Oculta cada celda7×7 con el promedio RGB, recalcula ambas vistas y mide el cambio absoluto de BCS continuo. Los JSON conservan signo, escala y pruebas ocultando ROI/exterior. Los mapas no se normalizan a máximo1; leer sus barras, no solo brillo.

No son atención, Grad-CAM ni detección de huesos. Las oclusiones crean imágenes artificiales y sus áreas difieren: sensibilidad no demuestra causalidad anatómica, y un cambio pequeño no prueba irrelevancia por redundancia o saturación. Las abstenciones no tienen una estimación que explicar.

`EXPLAIN_LIMIT=6` limita fotos del diagnóstico inicial. `pipeline.py explain --grid 4 --limit 6` reduce resolución/cantidad; no modifica las métricas de evaluación. Entrenar un localizador anatómico futuro requiere etiquetas adicionales y otra validación.

## 7. Checkpoints e interrupciones

```bash
RUN=$(cat ULTIMA_EJECUCION.txt)
.venv-mac/bin/python pipeline.py train --features "$RUN/features_gt" \
  --run "$RUN/modelo_ldl_tf" --epochs 100 --patience 15 --device mps --resume
```

`estado.json` apunta a último y mejor modelo completos `.keras`; incluye historia y estado de paciencia. El optimizador se recupera del último modelo. La normalización se conserva en `normalizacion.npz`. Mantener los mismos parámetros/caché; se permite ampliar el total de épocas. La paciencia alcanzada se respeta; no se garantiza identidad bit a bit del RNG con una corrida ininterrumpida ni entre dispositivos.

El modelo de cada generación se escribe antes del puntero atómico. Solo se conservan último y mejor; una interrupción puede dejar un archivo huérfano sin invalidar el estado previo. No editar estado.json ni borrar el directorio al reanudar.

Extracción: no hay reanudación parcial; repetir con otra salida. YOLO: usar `detector-train --resume /ruta/last.pt` conservando trabajo y dispositivo. Para fallos posteriores repetir solo la etapa pendiente; relanzar el ejecutor crea otra corrida. Para memoria insuficiente reducir lotes a1o2 en una corrida nueva. Las dos vistas se procesan secuencialmente.

## 8. Foto nueva

```bash
.venv-mac/bin/python pipeline.py predict \
  --image /ruta/imagenes_privadas/foto.jpg \
  --detector /ruta/modelos_privados/best.pt \
  --model /ruta/resultados_privados/modelo_ldl_tf \
  --out runs/foto_nueva --device mps
```

Guarda detecciones, recorte, ROI, BCS continuo y categoría. No identifica al animal ni estima BCS sin un par único. Si se entrenó con ROI manual, añadir `--roi x1 y1 x2 y2`. Los modelos son prototipos académicos y no sustituyen evaluación veterinaria.

## 9. Pruebas y publicación

Desde esta carpeta, con las dependencias instaladas:

```bash
.venv-mac/bin/python -m unittest discover -s tests -v
.venv-mac/bin/python tests/smoke_dino_tf.py
bash -n instalar_mac.sh ejecutar_mac.sh
```

Las pruebas unitarias usan datos ficticios temporales: fórmula LDL contrastada con NumPy, gradientes, serialización del modelo/optimizador, ETL, reanudación, métricas, abstenciones, oclusión y comparación pareada. El smoke test descarga DINOv2 público y conecta extractor real, cabeza TensorFlow y oclusión en CPU. Ninguna prueba constituye precisión científica de BCS ni valida Metal por sí sola. El comprobador GPU del instalador se ejecuta separadamente en el equipo destino.

Ver [VALIDACION.md](VALIDACION.md). Contribuir mediante rama del área y PR, conforme a [CONTRIBUTING.md](../../../CONTRIBUTING.md). Excluir también `.keras`, `.npz`, `.npy`, `runs/` y referencias privadas al preparar commits: las salidas pueden contener identificadores reales.

## Fuentes técnicas

- https://developer.apple.com/metal/tensorflow-plugin/
- https://docs.pytorch.org/docs/main/notes/mps.html
- https://github.com/facebookresearch/dinov2
- https://arxiv.org/abs/1611.01731
- https://arxiv.org/abs/1611.05916
- https://arxiv.org/abs/1311.2901
