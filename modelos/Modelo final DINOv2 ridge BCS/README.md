# Modelo final BCS: DINOv2-S y ridge (versión 1.0)

Este modelo estima la condición corporal (BCS) de una vaca Holstein a partir de fotos traseras tomadas a la altura del animal. La unidad de evaluación es la vaca: el modelo califica cada foto y la calificación de la vaca es la media de sus fotos. Es el modelo que aprobamos en la fase de Evaluation del proyecto (TC3009C.601, Tec de Monterrey, noviembre de 2026).

## Qué hace

Primero, el detector YOLO26n encuentra la vaca y la grupa en la foto. Tomamos la grupa de mayor confianza que queda dentro de la vaca de mayor confianza y la recortamos con 10 % de margen, porque así el recorte conserva el contorno de la cadera y la base de la cola, que es donde el veterinario mira la condición corporal.

El recorte pasa por DINOv2 ViT-S/14 congelado, con los pesos oficiales de Meta cargados con timm. De la red tomamos el token CLS y la media de los tokens de parche (768 valores) y los promediamos con los de la foto espejada. No ajustamos la red porque con 66 vacas cualquier ajuste fino sobreajustaba, y una red congelada con una cabeza lineal fue lo más estable en la validación cruzada.

Sobre esas características entrenamos una regresión ridge con alfa 100. Promediamos las predicciones de las fotos de la vaca y aplicamos la calibración lineal `bcs = 1.2027 * media - 0.6043`, que corrige la tendencia del ridge a acercar todas las vacas a la media. El valor final se redondea a pasos de 0.25 entre 2.25 y 3.75, y si el valor continuo es 2.90 o menos se activa la alerta de vaca flaca (BCS de 2.75 o menos).

Todos los parámetros están en [`modelo/config.json`](modelo/config.json).

## Uso previsto y límites

Es un prototipo académico para apoyar la revisión de condición corporal en el CAETEC. No sustituye la calificación del veterinario ni debe usarse como única base para decisiones de manejo.

El modelo solo vio vacas con BCS de 2.25 a 3.75, así que fuera de ese rango no hay garantía. El error es mayor en los extremos porque hubo una ligera falta de vacas de BCS 2.50 o menos y de 3.50 o más. Recomendamos tomar 3 fotos por vaca, porque así medimos el error.

## Pesos

Los pesos no están en GitHub. Están en el Drive del equipo, con acceso solo para el equipo y los profesores:

https://drive.google.com/drive/folders/1HmdvLhVme6qmUo3Iepow8bICiPGXPDZU

Hay que descargarlos y colocarlos dentro de `modelo/` con estas rutas:

| Archivo | Tamaño | SHA-256 |
|---|---|---|
| `modelo/cabeza.joblib` | 22.0 KB | `4478d1f79b88c92154e6e3aecb508c447ff4feac4f945f19e9e229b4824da34b` |
| `modelo/detector/best.pt` | 5.2 MB | `93acef2a8311d4048162ec0af0f0031ec1a8d9b32ce5353a94e6de02421a3f3b` |
| `modelo/pesos/dinov2_vits14.safetensors` | 84.2 MB | `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081` |

`cabeza.joblib` tiene el escalador y el ridge entrenados con las 331 fotos de las 66 vacas. `best.pt` es el detector, entrenado solo con las fotos del split train del export de Roboflow. `dinov2_vits14.safetensors` son los pesos públicos de DINOv2 ViT-S/14 de Meta (licencia Apache 2.0) en la versión de timm `vit_small_patch14_dinov2.lvd142m`, sin cambios.

Para comprobar que los archivos son los mismos que usamos, desde la carpeta `modelo/`:

```bash
sha256sum -c HUELLAS.txt
```

En Windows, con PowerShell:

```powershell
Get-FileHash .\cabeza.joblib, .\detector\best.pt, .\pesos\dinov2_vits14.safetensors -Algorithm SHA256
```

## Instalación y uso

Usamos Python 3.12. Para predecir basta con la versión CPU de PyTorch:

```bash
python -m venv .venv
# Activar con .venv\Scripts\activate en Windows o con source .venv/bin/activate en Linux y Mac
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r codigo/requirements.txt
```

Desde la carpeta `codigo/`, con tres fotos de la misma vaca:

```bash
python predecir.py foto1.jpg foto2.jpg foto3.jpg --vaca V01
```

El resultado es un JSON con el BCS redondeado, el valor continuo, el rango de ±0.25, la alerta de vaca flaca y las fotos rechazadas con su motivo. Una foto se rechaza si no se encuentra la grupa dentro de la vaca o si la confianza de la grupa es menor que 0.25. Si ya se tienen las cajas en formato YOLO, el detector se puede saltar con `--caja foto1.txt foto2.txt foto3.txt`.

## Datos

Las fotos y las etiquetas son del CAETEC y no se publican. Para entrenar o evaluar se necesitan dos cosas que cada integrante tiene en su equipo:

1. El export de Roboflow "Vacones BCS - Cajas v2 Aprobado" en formato YOLO, con las carpetas `train`, `valid` y `test`. En ese export la clase 0 es `ass` (grupa) y la clase 1 es `cow`.
2. Un CSV de metadatos con una fila por foto y las columnas `split`, `image_id`, `file_name`, `cow_id`, `bcs` y `metadata_status`. Solo se usan las filas con `metadata_status = ok`. El archivo [`examples/metadata_bcs.example.csv`](examples/metadata_bcs.example.csv) muestra el formato con datos ficticios. Si dos identificadores corresponden al mismo animal, hay que unificarlos en este CSV antes de correr, para que sus fotos queden siempre en el mismo pliegue.

En la copia "VERIFICADO" del Drive las clases están al revés (0 = cow y 1 = ass). El detector y `predecir.py` leen las clases por nombre, pero las etiquetas en texto se leen por número, así que antes de usar otras etiquetas hay que revisar su mapa de clases.

## Cómo reproducir

Todos los scripts reciben las rutas como argumentos y no escriben fotos ni recortes. Los que guardan características o predicciones por vaca lo hacen en la carpeta que se les indica, que debe quedar fuera del repositorio.

```bash
# Detector de vaca y grupa (GPU recomendada)
python entrenar_detector.py RUTA_EXPORT_YOLO

# Cabeza, calibración y umbral de la alerta con las 66 vacas
python entrenar_final.py RUTA_EXPORT_YOLO RUTA_METADATA.csv ../modelo/pesos/dinov2_vits14.safetensors

# Evaluación del sistema completo con el detector, validación cruzada por vaca 5x10
python evaluar_sistema.py RUTA_EXPORT_YOLO RUTA_METADATA.csv ../modelo/pesos/dinov2_vits14.safetensors ../modelo/detector/best.pt ../resultados/sistema_metricas.json

# Comparación con la receta DINOv2 + LDL del equipo en los mismos pliegues
python comparar_ldl.py RUTA_EXPORT_YOLO RUTA_METADATA.csv ../modelo/pesos/dinov2_vits14.safetensors ../resultados/resultados_cierre.json

# Figuras y resumen del modelo final
python figuras.py RUTA_EXPORT_YOLO RUTA_METADATA.csv ../modelo/pesos/dinov2_vits14.safetensors ../resultados
```

`extraer_features.py` y `evaluar_cabezas.py` son la comparación de redes base y cabezas con la que elegimos DINOv2-S y ridge. `comparar_b0.py` y `comparar_prueba_10.py` comparan contra el modelo ordinal EfficientNet-B0 + CORAL. `extraer_externos.py` y `evaluar_externos.py` prueban los datasets externos, que no usamos porque empeoraban el error en las vacas del CAETEC.

## Resultados

Medimos con validación cruzada por vaca: 5 pliegues agrupados por vaca, repetidos 10 veces con distinta semilla, sobre las 331 fotos de 66 vacas. La calibración se ajusta dentro de cada pliegue de entrenamiento, así que las vacas evaluadas nunca se usan para ajustar nada.

Con las cajas anotadas, el MAE por vaca es 0.199 y el RMSE 0.251. El 87 % de las vacas queda dentro de ±0.25 y el 98 % dentro de ±0.50, con un kappa ponderado de 0.73. Con las cajas del detector, que es como funciona el sistema completo, el MAE es 0.196 y el 86 % queda dentro de ±0.25. En las 20 vacas cuyas fotos el detector nunca vio, el MAE es 0.206. La alerta de vaca flaca encuentra al 79 % de las vacas de 2.75 o menos con una precisión de 83 %. Como referencia, predecir siempre 3.00 da un MAE de 0.318, y la receta DINOv2 + LDL del equipo medida en los mismos pliegues da 0.226.

El alfa de 100 lo fijamos después de ver la validación cruzada, por eso el 0.199 puede ser un poco optimista. Con el alfa elegido dentro de cada pliegue el MAE fue 0.205. Las 10 vacas de prueba del modelo ordinal también entraron en esta validación cruzada, así que no las presentamos como prueba independiente.

El detector tiene un mAP50 de 0.995 en valid y en test, y un mAP50-95 de 0.81 en valid y 0.79 en test. En las 331 fotos encontró la grupa dentro de la vaca en todas.

La carpeta [`resultados/`](resultados/) tiene los resultados agregados de cada comparación y las figuras. No incluye identificadores de vacas ni nombres de fotos.

## Contenido de la carpeta

`codigo/` tiene los scripts y `requirements.txt` con las versiones que usamos. `modelo/` tiene `config.json`, las métricas del detector y `HUELLAS.txt` con el SHA-256 de cada peso. `resultados/` tiene los resultados agregados. `examples/` tiene el formato del CSV de metadatos con datos ficticios.

