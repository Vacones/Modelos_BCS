# Modelo ordinal de BCS con recorte del trasero

Estimamos la condición corporal (BCS) de una vaca a partir de una foto trasera. El modelo recorta el trasero con la caja `ass`, lo pasa por EfficientNet-B0 preentrenada en ImageNet y usa una salida ordinal CORAL. El código compara esa salida contra softmax, CORN, regresión y una constante, siempre evaluando por vaca y con vacas que el modelo no vio.

## Resultados

| Métrica por vaca | Modelo, validación cruzada (56 vacas) | Constante 3.00, validación cruzada | Modelo, prueba final (10 vacas) | Constante 3.00, prueba final |
| --- | --- | --- | --- | --- |
| MAE | 0.250 | 0.317 | 0.300 | 0.325 |
| Dentro de ±0.25 | 73% | 59% | 80% | 60% |
| Dentro de ±0.50 | 96% | 89% | 90% | 90% |
| Acierto en tres niveles | 55% | 41% | 50% | 40% |

- En validación cruzada, recorte + CORAL es la única combinación que mejora a la constante con un intervalo de 95% que no cruza cero (mejora de 0.067, intervalo de 0.013 a 0.125).
- En la prueba final la mejora es de 0.025, con intervalo de −0.075 a 0.125: con 10 vacas no se distingue del azar.
- Con la imagen completa ninguna salida supera a la constante. CORAL empata con softmax.
- El modelo comprime sus predicciones hacia 3.00, y el error crece en los extremos (0.45 en vacas de 3.75).
- Las 10 vacas de prueba ya se usaron. `EVALUAR_PRUEBA` debe quedarse en `False`.

Configuración final: recorte con 10% de margen, 224 × 224 px, EfficientNet-B0 con Dropout 0.3 y salida CORAL; AdamW con weight decay 1e-4 y batch 16. La etapa 1 entrena solo la salida con tasa 1e-3; la etapa 2 descongela `block7` y `top` (BatchNorm congelada) con tasa 1e-5. La parada temprana usa paciencia 8 sobre el MAE por vaca de una validación interna.

## Archivos

| Archivo | Para qué |
| --- | --- |
| `bcs_ordinal.py` | Todo el código: datos, recortes, salidas, entrenamiento, validación cruzada, métricas, Grad-CAM y prueba final. |
| `BCS_ordinal_local.ipynb` | Notebook para correr en una PC con los datos en su carpeta. |
| `BCS_ordinal_colab.ipynb` | Notebook para Colab, lee los datos desde `MyDrive/BCS_ordinal`. |
| `requirements.txt` | Dependencias. |
| `resultados/` | Resúmenes y gráficas del experimento A (base congelada), del experimento B (ajuste fino, tasa 1e-5) y la comparación de tasas de la etapa 2. |

## Datos

Las fotos y etiquetas son del CAETEC y no se suben al repositorio. Están en la carpeta compartida del equipo. Para correr el notebook hacen falta, en su misma carpeta o en la de arriba:

- El export de Roboflow `Vacones BCS - Cajas v2 Aprobado.yolo26` (carpeta o zip), con cajas `cow` y `ass`.
- `metadata_bcs_validos.csv`: 331 fotos de 66 vacas con su BCS y la partición por vaca (46 entrenamiento, 10 validación, 10 prueba).

## Cómo correrlo

### Comprobación previa del COCO local (Mac)

Desde la raíz de `Modelos_BCS`, con el export COCO aprobado ya descargado:

```bash
python3 -m unittest discover -s modelos/con_cajas_ordinal/tests -p 'test_*.py' -v
python3 -m modelos.con_cajas_ordinal.preflight_coco ~/Desktop/vacones_reg/datos/coco_cajas_v2_aprobado
```

`preflight_coco.py` solo lee el COCO: comprueba identidad y BCS por foto, fugas de vacas entre particiones, BCS contradictorios por vaca, archivos presentes y una caja `ass` por foto con BCS válido. Informa conteos por partición, sin publicar identificadores. Inspecciona la estructura de `test` para detectar fugas, pero **no entrena ni evalúa el modelo en `test`**. En el COCO local revisado se esperan 239 fotos válidas de 46 vacas en `train`, 36 de 10 en `valid` y 56 de 10 en `test`; cuatro fotos de `train` quedan excluidas por etiquetas BCS ambiguas.

Este preflight no convierte COCO ni crea un checkpoint. El adaptador y la nueva ruta de entrenamiento se describen abajo.

Para preparar en disco privado una vista compatible con la carga de datos del notebook (fotos enlazadas, sin copiarlas), ejecuta desde la raíz del repositorio:

```bash
python3 -m modelos.con_cajas_ordinal.preparar_coco ~/Desktop/vacones_reg/datos/coco_cajas_v2_aprobado --solo-validar
python3 -m modelos.con_cajas_ordinal.preparar_coco ~/Desktop/vacones_reg/datos/coco_cajas_v2_aprobado \
  --salida ~/Desktop/vacones_reg/datos/ordinal_coco_preparado
```

El destino debe ser una carpeta nueva y privada. Contendrá `metadata_bcs_validos.csv` y `Vacones BCS - Cajas v2 Aprobado.yolo26/`; los archivos `images/` serán enlaces a las fotos originales, mientras que `labels/` contendrá las cajas convertidas de COCO a YOLO. No añadas ese destino a Git. El notebook local admite `BCS_CARPETA` para encontrar esa carpeta, por ejemplo `BCS_CARPETA=~/Desktop/vacones_reg/datos/ordinal_coco_preparado jupyter notebook modelos/con_cajas_ordinal/BCS_ordinal_local.ipynb`.

**No ejecutes el entrenamiento del notebook histórico para la comparación maestra.** Su experimento combina `train` y `valid` en validación cruzada. La nueva ruta de abajo ajusta pesos solo con `train` y usa `valid` para seleccionar el checkpoint; por eso sus métricas de validación son de desarrollo.

### Nueva corrida CORAL con train y valid separados

El script `entrenar_coral_holdout.py` es una ruta nueva y separada del experimento histórico. Usa las cajas **anotadas** del COCO v2, entrena solo con `train` y escoge pesos por MAE continuo **por vaca** en `valid`. El catálogo conserva los siete niveles reales de 2.25 a 3.75, con seis umbrales CORAL; el experimento histórico fusionó 2.25 con 2.50, por lo que sus métricas no son directamente la misma variante. No se calculan predicciones de `test`.

Primero comprueba las particiones, sin cargar TensorFlow ni entrenar:

```bash
python3 -m modelos.con_cajas_ordinal.entrenar_coral_holdout \
  --datos ~/Desktop/vacones_reg/datos/coco_cajas_v2_aprobado --solo-validar
```

Después, cuando corresponda ejecutar el entrenamiento en tu Mac, desde la raíz del repositorio y con el entorno que tenga TensorFlow y Keras:

```bash
python -m modelos.con_cajas_ordinal.entrenar_coral_holdout \
  --datos ~/Desktop/vacones_reg/datos/coco_cajas_v2_aprobado \
  --salida ~/Desktop/vacones_reg/resultados_ordinal/coral_holdout_v1 \
  --epochs-head 30 --epochs-finetune 30 --patience 8 --batch 16
```

La carpeta de salida debe ser nueva y privada. Allí quedarán los pesos `*.weights.h5`, `predicciones_val_foto.csv`, `predicciones_val_vaca.csv`, `metricas_val.json` y `ejecucion.json` con huellas del código y COCO. No subas esos artefactos individuales a Git. Como usa cajas anotadas, este resultado mide el predictor ordinal con recortes ideales; para la comparación del sistema completo con otros modelos hará falta una evaluación adicional con cajas predichas y abstenciones.

### Windows con GPU NVIDIA

En Windows, Keras usa PyTorch como motor, porque TensorFlow ya no usa la GPU en Windows nativo. En PowerShell, dentro de esta carpeta:

```
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
jupyter notebook BCS_ordinal_local.ipynb
```

Si PowerShell no deja activar el entorno, corre antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. En VS Code hay que elegir como kernel el Python de `venv`. La primera celda debe mostrar `motor PyTorch ... | GPU: <tu tarjeta>`.

### Colab

Sube el zip del export y `metadata_bcs_validos.csv` a `MyDrive/BCS_ordinal`, elige un entorno con GPU y ejecuta `BCS_ordinal_colab.ipynb`.

### Linux o Mac

`pip install tensorflow -r requirements.txt` y abre `BCS_ordinal_local.ipynb` desde una carpeta con los datos.

En una RTX 4060 de laptop, cada configuración del experimento B tarda alrededor de 30 minutos. Lo que ya se corrió no se repite: cada fila de resultados guarda en la columna `config` los hiperparámetros, y `comparar_configuraciones` pone lado a lado las configuraciones corridas.

## Referencias

- Cao, W., Mirjalili, V. y Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*, 140, 325–331.
- Shi, X., Cao, W. y Raschka, S. (2023). Deep neural networks for rank-consistent ordinal regression based on conditional probabilities. *Pattern Analysis and Applications*, 26, 941–955.
- Tan, M. y Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning*, 6105–6114.
