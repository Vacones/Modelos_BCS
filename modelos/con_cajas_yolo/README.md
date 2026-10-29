# Vacas Holstein: pipeline local con YOLO26 y EfficientNet-B0

Para **Kaggle y regresión continua de BCS con MAE**, utiliza el notebook autónomo
[bcs_kaggle.ipynb](bcs_kaggle.ipynb) y la [guía de Kaggle](KAGGLE.md).
La documentación siguiente corresponde al ejemplo local de clasificación.

Este proyecto utiliza la carpeta descargada del notebook de conversión: fotografías, cajas YOLO, metadatos BCS y JSON COCO originales. Incluye ETL, EDA, entrenamiento, checkpoints, evaluación y predicción sobre imágenes nuevas. No incluye los pesos entrenados ni las fotografías.

**Entorno previsto:** Windows con WSL2/Ubuntu y GPU NVIDIA compatible con CUDA. Los comandos de Python se ejecutan dentro de Ubuntu, no en PowerShell. No hay interfaz web ni archivos de configuración que debas escribir a mano; las opciones se pasan desde la terminal. El YAML que necesita YOLO se genera durante el ETL.

**Arquitectura:** YOLO26 detecta `cow` y `ass`; un recorte de `ass` se clasifica mediante EfficientNet-B0 en TensorFlow/Keras. El entrenamiento oficial de YOLO26 usa PyTorch. Por ello hay dos entornos virtuales separados, y las etapas se ejecutan en procesos distintos para evitar mezclar sus bibliotecas CUDA. El clasificador BCS, su ajuste fino y la inferencia BCS son TensorFlow.

## 1. Preparar WSL2 y comprobar la tarjeta

Actualiza el controlador NVIDIA **en Windows**. No instales un controlador de pantalla Linux/NVIDIA dentro de WSL. El controlador de Windows proporciona el acceso de WSL a la GPU. No necesitas instalar manualmente todo CUDA/cuDNN: las ruedas de los entornos incluyen sus dependencias de ejecución.

En **PowerShell de Windows**:

```powershell
wsl --status
wsl --list --verbose
wsl --update
```

Tu distribución Ubuntu debe aparecer con versión **2**. Si aún no está instalada:

```powershell
wsl --install -d Ubuntu
```

Reinicia si Windows lo solicita y abre Ubuntu. Dentro de **Ubuntu**:

```bash
nvidia-smi
python3 --version
sudo apt update
sudo apt install -y python3-venv python3-pip unzip libgl1 libglib2.0-0
```

Utiliza Python **3.10, 3.11 o 3.12**, por ejemplo el Python 3.12 de Ubuntu 24.04. No uses Python 3.14 para este paquete. La instalación proporciona una base de versiones fijadas, pero la compatibilidad final depende de la tarjeta, el controlador y WSL: las pruebas de GPU deben pasar antes de entrenar.

## 2. Ubicar el código y proporcionar datos externos

Desde la raíz del repositorio, entra a este modelo:

```bash
cd modelos/con_cajas_yolo
```

Todos los comandos siguientes se ejecutan desde esta carpeta. Proporciona el
dataset autorizado fuera del repositorio y sustituye `/ruta/al/dataset_yolo`
por su ubicación al ejecutar ETL. Los ejemplos son rutas ficticias.

Dentro de tu carpeta de datos deben existir:

| Ruta | Contenido |
|---|---|
| `images/train`, `images/val`, `images/test` | Fotografías reales descargadas de Drive. |
| `labels/train`, `labels/val`, `labels/test` | Cajas convertidas: 0=cow, 1=ass. |
| `metadata.csv` | Relación imagen, vaca, BCS y estado de las etiquetas. |
| `coco_original/train`, `coco_original/valid`, `coco_original/test` | Cada una con `_annotations.coco.json`. |
| `data.yaml` | Clases del detector. |

No utilices solamente el ZIP anterior de etiquetas: ese ZIP no contenía las fotografías. Necesitas la carpeta completa generada por Colab. Si Windows añade una carpeta exterior al descomprimir, `--data` debe apuntar al nivel que contiene directamente `images`, `labels` y `metadata.csv`.

Si ya entraste a `modelos/con_cajas_yolo`, permanece en esa carpeta.

## 3. Instalar los entornos

Primero el entorno TensorFlow:

```bash
bash instalar_wsl.sh tf
```

Instala TensorFlow 2.20.0 con el extra `and-cuda`, las bibliotecas de procesamiento y métricas, y ejecuta una operación de entrenamiento convolucional en GPU. Guarda las versiones instaladas en `entorno_tf_instalado.txt`.

Después el entorno YOLO:

```bash
bash instalar_wsl.sh yolo
```

Utiliza Ultralytics 8.4.0 y PyTorch 2.9.1/torchvision 0.24.1 con ruedas CUDA 12.6. No se ejecutan ambas redes a la vez en memoria. Si tu tarjeta requiere otra rueda CUDA, sigue las instrucciones oficiales de PyTorch; el instalador permite cambiar el índice mediante `TORCH_WHL_INDEX`. No cambies de versión al azar si aparece un error de arquitectura CUDA.

Para volver a comprobar cada entorno:

```bash
.venv-tf/bin/python comprobar_gpu.py --framework tf
.venv-yolo/bin/python comprobar_gpu.py --framework torch
```

**No continúes si falla la GPU.** Que `nvidia-smi` funcione no basta: las operaciones de entrenamiento anteriores también deben pasar. Un error `no kernel image` o `unsupported compute capability` indica incompatibilidad entre tarjeta y binario; no demuestra un error en tus datos.

## 4. Pruebas cortas antes del entrenamiento

```bash
.venv-tf/bin/python -m unittest tests.test_core -v
.venv-tf/bin/python -m tests.smoke_tensorflow
```

La segunda prueba crea EfficientNet-B0 con pesos aleatorios, ejecuta un paso de entrenamiento, guarda y carga un `.keras` y comprueba el ajuste fino. Es una prueba técnica: sus predicciones no tienen utilidad BCS y no se guardan como un modelo del proyecto. No descarga ImageNet. Si produce falta de memoria, cierra otros programas que utilicen la GPU antes de continuar.

## 5. ETL y EDA

Adapta únicamente la ruta de entrada a tu equipo. `salida` es una carpeta de trabajo nueva que se creará dentro del proyecto:

```bash
.venv-tf/bin/python pipeline.py etl --data "/ruta/al/dataset_yolo" --work salida --size 224
.venv-tf/bin/python pipeline.py eda --work salida
```

El ETL comprueba fotos y dimensiones, verifica la correspondencia entre metadatos, COCO y cajas YOLO, rechaza identidades compartidas entre particiones y detecta fotografías idénticas entre particiones mediante SHA-256. No detecta por sí solo todos los casos de imágenes visualmente similares o IDs mal anotados.

Mantiene las particiones originales. Los BCS ambiguos se excluyen de clasificación, pero sus fotos y cajas siguen disponibles para el detector. Extrae recortes posteriores anotados, los convierte a RGB y aplica relleno para llegar a 224 × 224 sin estirar la forma del animal. No divide los valores entre 255: EfficientNet de Keras incluye su escalado interno.

Los IDs solo se utilizan para trazabilidad y revisión de particiones: no se entregan como características al clasificador. El catálogo BCS se deriva de las categorías presentes en entrenamiento; no se ajustan clases con imágenes de prueba.

Los conteos se calculan desde los datos externos proporcionados. Las imágenes con etiquetas ambiguas no se usan para entrenar el clasificador.

Archivos principales:

| Salida | Uso |
|---|---|
| `salida/manifest.csv` | Fotos, identidad, BCS, recortes y estado. |
| `salida/excluidos_bcs.csv` | Registros sin supervisión BCS inequívoca. |
| `salida/crops_gt/` | Recortes a partir de cajas anotadas. |
| `salida/classes.json` | Catálogo ordenado de categorías BCS. |
| `salida/etl.json` | Parámetros, conteos y huella del manifiesto. |
| `salida/detector.yaml` | Rutas absolutas y clases para YOLO, generadas automáticamente. |
| `salida/eda/` | Distribución BCS, fotos por vaca y ejemplos de recortes. |

El EDA identifica clases poco representadas. No se inventan etiquetas ni se fusionan categorías automáticamente.

No muevas la carpeta de trabajo ni las fotos después del ETL: el manifiesto contiene rutas absolutas. Si necesitas moverlas o corregir anotaciones, repite ETL con una carpeta de trabajo nueva.

## 6. Entrenar EfficientNet-B0 en TensorFlow

Primero, entrenar la cabeza de clasificación con la base congelada:

```bash
.venv-tf/bin/python pipeline.py train --work salida --run runs/bcs_head --phase head --epochs 30 --batch 4 --lr 0.001
```

Se descargarán los pesos ImageNet la primera vez. El aumento de datos se aplica solo durante entrenamiento: inversión horizontal, rotación moderada y zoom moderado. La salida es `softmax`; la pérdida es entropía cruzada categórica dispersa. Los pesos de clase se calculan únicamente con entrenamiento para moderar el desbalance.

Se utilizan `train` para aprender y `val` para seleccionar el mejor checkpoint según `val_loss`. **El entrenamiento no consulta las fotografías de `test`.** Los valores iniciales son manuales y revisables; no constituyen hiperparámetros optimizados.

Después, ajuste fino de las últimas 20 capas, manteniendo congeladas las capas BatchNormalization:

```bash
.venv-tf/bin/python pipeline.py train --work salida --run runs/bcs_finetune --phase finetune --init runs/bcs_head/best.keras --epochs 20 --batch 4 --lr 0.00001 --fine-layers 20
```

La base se invoca con `training=False` para mantener el comportamiento de BatchNormalization. Esto no impide actualizar las capas convolucionales declaradas entrenables.

Para comparar fases, evalúa ambas **en validación**, con carpetas de salida distintas. Elige una antes de revisar los resultados de prueba:

```bash
.venv-tf/bin/python pipeline.py evaluate --work salida --model runs/bcs_head/best.keras --split val --out evaluacion_val_head
.venv-tf/bin/python pipeline.py evaluate --work salida --model runs/bcs_finetune/best.keras --split val --out evaluacion_val_finetune
```

A continuación se usa `runs/bcs_finetune/best.keras` como ejemplo. Si validación favorece la fase head, cambia esa ruta por `runs/bcs_head/best.keras` en los comandos posteriores.

### Checkpoints y reanudación

Cada época completa guarda un `.keras` con el optimizador y un registro de finalización. Se conservan los dos últimos checkpoints completos y una copia independiente `best.keras`; se mantienen los registros de todas las épocas. También se generan `historial.csv`, `curvas_entrenamiento.png` y logs TensorBoard.

Si se interrumpe la fase head:

```bash
.venv-tf/bin/python pipeline.py train --work salida --run runs/bcs_head --phase head --epochs 30 --batch 4 --lr 0.001 --resume
```

Si se interrumpe el ajuste fino:

```bash
.venv-tf/bin/python pipeline.py train --work salida --run runs/bcs_finetune --phase finetune --init runs/bcs_head/best.keras --epochs 20 --batch 4 --lr 0.00001 --fine-layers 20 --resume
```

Mantén los mismos parámetros. `--epochs` representa el total deseado, no épocas adicionales. Si ya completaste ese total y quieres continuar, auméntalo. Se recupera la última época completa, no un lote a mitad de época. El contador de paciencia se reinicia al reanudar y el flujo aleatorio del aumento puede variar; no se promete reproducción bit a bit. No modifiques el modelo de inicialización ni los datos entre reanudaciones.

## 7. Entrenar el detector YOLO26

Este entrenamiento usa PyTorch en el entorno YOLO:

```bash
.venv-yolo/bin/python pipeline.py detector-train --work salida --run runs/detector --weights yolo26n.pt --epochs 80 --batch 4 --imgsz 640 --device 0
```

Se utiliza inicialmente la variante nano para limitar recursos. El modelo se adapta a tus dos clases y guarda `best.pt`, `last.pt`, curvas y gráficas de detección. No uses `yolo26n.pt` general para generar las cajas finales: debes usar el `best.pt` entrenado con `cow` y `ass`.

Para reanudar:

```bash
.venv-yolo/bin/python pipeline.py detector-train --work salida --resume runs/detector/weights/last.pt --device 0
```

Para medir el detector en prueba, después de cerrar las decisiones usando validación:

```bash
.venv-yolo/bin/python pipeline.py detector-test --work salida --weights runs/detector/weights/best.pt --split test --out evaluacion_detector_test --device 0
```

Se generan las métricas y gráficas soportadas por Ultralytics: precisión, recall, mAP y matriz de confusión, según la evaluación. Las cifras mAP del detector no son la precisión BCS.

## 8. Evaluar el clasificador y el sistema completo

### A. Clasificador con recortes anotados

```bash
.venv-tf/bin/python pipeline.py evaluate --work salida --model runs/bcs_finetune/best.keras --split test --out evaluacion_bcs_gt_test
```

Esta evaluación mide clasificación con una región correcta proporcionada por la anotación. No demuestra la calidad del sistema automático.

### B. Fotos completas → cajas predichas → clasificación

Primero genera las cajas y recortes automáticos en un proceso YOLO:

```bash
.venv-yolo/bin/python pipeline.py detector-crops --work salida --weights runs/detector/weights/best.pt --out recortes_predichos --splits val test --device 0 --conf 0.25
```

La asociación exige un único par plausible: centro de la caja posterior dentro de la vaca y al menos 80 % de su área contenida en ella. Si falta el par o hay varias asociaciones posibles, el sistema se abstiene. No utiliza la caja real, el ID ni el BCS para escoger la predicción.

Los umbrales son puntos de partida. Cualquier ajuste debe hacerse con validación y quedar cerrado antes de analizar prueba.

Ahora evalúa con TensorFlow:

```bash
.venv-tf/bin/python pipeline.py evaluate --work salida --model runs/bcs_finetune/best.keras --split test --boxes recortes_predichos --out evaluacion_end_to_end_test
```

Las fotos con supervisión BCS válida pero sin detección/asociación aceptable cuentan como fallos en los porcentajes calculados sobre todas las imágenes. Se informa además la cobertura, evitando ocultar las abstenciones.

### Métricas y gráficas

| Archivo/métrica | Interpretación |
|---|---|
| `metricas.json` | Conteos, cobertura, exactitud y exactitud dentro de ±0.25/±0.50. |
| `accuracy_all_tolerance_0` | Acierto exacto sobre todas las imágenes elegibles, incluidos fallos del detector. |
| `accuracy_all_tolerance_0.25` / `0.5` | Proporción dentro de cada tolerancia; los casos no clasificados cuentan como no aciertos. |
| `mae_classified`, `rmse_classified` | Error BCS únicamente en imágenes clasificadas; leer junto con cobertura. |
| `macro_f1_all_catalog` | F1 macro sobre el catálogo completo; clases sin soporte reciben 0 y aparecen en el reporte. |
| `quadratic_kappa_classified` | Concordancia ordinal entre clases, cuando es calculable. |
| `predicciones.csv` | Referencia, predicción, score y estado de cada imagen elegible. |
| `por_vaca.csv` | Cobertura y acierto dentro de ±0.50 por vaca. |
| `classification_report.json` | Precisión, recall, F1 y soporte por categoría. |
| `matriz_confusion.png` | Confusiones entre categorías BCS. |
| `errores_bcs.png` | Predicción frente a referencia e histograma del error. |

Las métricas entre 0 y 1 se multiplican por 100 para expresarlas como porcentaje. Si no se clasifica ninguna imagen, las métricas condicionadas a clasificación quedan en `null`, mientras que cobertura y acierto global son 0.

Las salidas softmax son scores sin calibrar. Una exactitud elevada con pocos animales o clases sin soporte requiere una interpretación cuidadosa. Los resultados no sustituyen una validación independiente sobre nuevas vacas y condiciones de captura.

## 9. Predecir una fotografía nueva

En el entorno YOLO:

```bash
.venv-yolo/bin/python pipeline.py detect-image --weights runs/detector/weights/best.pt --image "/ruta/a/imagen_nueva.jpg" --out prediccion_nueva --device 0 --size 224
```

Genera la vista `detecciones.jpg`, `prediction.json` y, si existe un par único, `crop.png`.

En el entorno TensorFlow:

```bash
.venv-tf/bin/python pipeline.py predict --model runs/bcs_finetune/best.keras --bundle prediccion_nueva
```

La salida se guarda en `bcs_prediction.json`. El sistema no identifica automáticamente al animal; una foto con múltiples candidatos puede quedar sin clasificación.

## 10. Problemas frecuentes

- **Memoria GPU insuficiente:** en una ejecución nueva reduce `--batch 4` a `--batch 2` o `1`. Para detector también puedes reducir `--imgsz`. No cambies el batch en una reanudación: sus parámetros están protegidos. Cierra juegos y procesos GPU. No se asume un modelo concreto de tarjeta ni una cantidad fija de VRAM.
- **TensorFlow no ve GPU:** ejecuta primero `nvidia-smi` dentro de Ubuntu y el comprobador del entorno TF. Verifica que estás usando `.venv-tf/bin/python`, WSL2 y un controlador Windows reciente.
- **Conflictos CUDA:** no instales TensorFlow y PyTorch en un único entorno ni actives simultáneamente ambos. Usa los ejecutables explícitos de esta guía.
- **No se encuentran fotos:** revisa `--data` y la carpeta exterior creada al descomprimir. El paquete de conversión sin imágenes no basta.
- **Carpeta existente:** ETL, detector y evaluaciones rechazan mezclar ejecuciones. Usa otra carpeta o el mecanismo de reanudación correspondiente.
- **Entrenamiento interrumpido:** utiliza los comandos `--resume`. Desactiva la suspensión automática de Windows durante el entrenamiento.
- **Cambiaste etiquetas o moviste fotos:** repite ETL en una carpeta nueva y comienza una nueva ejecución de entrenamiento.
- **Dos vacas en la foto:** las cajas y los tags de imagen no resuelven la identidad por instancia. El flujo automático exige un par único y marca ambigüedades.

Las instalaciones CUDA pueden ocupar bastantes GB; reserva margen para entornos, caché de pip, pesos y checkpoints. Los valores exactos dependen del equipo y las descargas. El tamaño del dataset por sí solo no garantiza un porcentaje de precisión.

## Pruebas y límites

Pruebas sin TensorFlow ni GPU, desde esta carpeta:

```bash
python3 -m unittest tests.test_core -v
python3 pipeline.py --help
bash -n instalar_wsl.sh
```

Requieren las dependencias de `requirements-common.txt`. Cubren geometría,
asociaciones ambiguas, formato YOLO y métricas con abstenciones mediante datos
ficticios creados temporalmente. No requieren fotografías ni etiquetas privadas.

`tests.smoke_tensorflow` se ejecuta por separado en el entorno TensorFlow.
No se afirma que las pruebas unitarias validen CUDA, el entrenamiento completo
ni el desempeño científico. No se publican métricas de experimentos privados.

## Publicación y colaboración

Esta carpeta contiene únicamente código, pruebas y documentación técnica.
Las salidas del programa contienen datos sensibles y deben mantenerse fuera de
GitHub, incluso en Releases. No adjuntar datasets, recortes, manifiestos,
identificadores, checkpoints ni predicciones a los Pull Requests.

Trabajar en `modelo/con-cajas-yolo` y abrir un Pull Request hacia `main`,
conforme a [CONTRIBUTING.md](../../CONTRIBUTING.md). La integración requiere revisión.

## Fuentes técnicas de instalación y arquitectura

- TensorFlow en WSL2: https://www.tensorflow.org/install/pip
- CUDA en WSL: https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl
- TensorFlow 2.20.0: https://pypi.org/project/tensorflow/2.20.0/
- EfficientNet y transferencia: https://www.tensorflow.org/guide/keras/transfer_learning
- Ultralytics 8.4.0: https://pypi.org/project/ultralytics/8.4.0/
- Versiones PyTorch/CUDA: https://pytorch.org/get-started/previous-versions/
