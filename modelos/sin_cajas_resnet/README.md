# Modelo BCS sin cajas

Baseline para estimar el **Body Condition Score (BCS)** de una vaca a partir de fotografías completas, sin *bounding boxes*. El modelo usa aprendizaje por transferencia con ResNet18 y produce una predicción conjunta por vaca al promediar las probabilidades de todas sus fotografías.

Esta implementación forma parte del repositorio multimodelo de Vacones. Los comandos de este documento deben ejecutarse desde `modelos/sin_cajas_resnet`.

Desde la raíz del repositorio:

```bash
cd modelos/sin_cajas_resnet
```

> Este repositorio contiene código y pruebas, pero no incluye fotografías, etiquetas reales, manifiestos privados, resultados por animal ni checkpoints.

## Alcance

- Clasificación multiclase para BCS: `2.25`, `2.50`, `2.75`, `3.00`, `3.25`, `3.50` y `3.75`.
- Separación de entrenamiento, validación y prueba por vaca.
- Preprocesamiento con *letterbox* a 224 × 224, sin recorte central.
- Selección del checkpoint mediante MAE de validación por vaca.
- Ejecución en Apple MPS cuando está disponible; en otro caso utiliza CPU.

El proyecto es un baseline académico. Sus resultados no deben utilizarse para decisiones clínicas o productivas.

## Requisitos

- Python 3.12
- Acceso autorizado al dataset privado
- Espacio suficiente para las imágenes y los resultados

La primera ejecución con pesos preentrenados requiere acceso a internet para descargar los pesos de ResNet18.

## Instalación

Con `uv`:

```bash
uv venv --python 3.12 .venv-modelo
source .venv-modelo/bin/activate
uv pip install -r requirements.txt
```

Con las herramientas estándar de Python:

```bash
python3.12 -m venv .venv-modelo
source .venv-modelo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows, la activación equivalente es:

```powershell
.venv-modelo\Scripts\Activate.ps1
```

## Preparación privada del dataset

El dataset no debe copiarse al repositorio. Por defecto, el programa espera una carpeta local llamada `dataset_sin_cajas_v2_aprobado` con esta estructura:

```text
dataset_sin_cajas_v2_aprobado/
├── _control/
│   └── dataset_manifest.csv
├── vaca-001/
│   ├── imagen-001.jpg
│   └── imagen-002.jpg
└── vaca-002/
    └── imagen-003.jpg
```

El manifiesto necesita las columnas:

| Columna | Descripción |
|---|---|
| `imagen_id` | Identificador estable de la imagen. |
| `vaca_id` | Identificador usado para agrupar y separar animales. |
| `bcs` | Etiqueta BCS autorizada. |
| `conjunto` | `ENTRENAMIENTO`, `VALIDACION` o `PRUEBA`. |
| `ruta_dataset` | Ruta relativa de la imagen dentro del dataset. |

Hay un ejemplo ficticio en `examples/dataset_manifest.example.csv`. No lo use como dataset real.

También puede mantener el dataset fuera de la carpeta del repositorio y proporcionar su ruta explícitamente:

```bash
python entrenar_sin_cajas.py --solo-validar --dataset-root /ruta/privada/al/dataset
```

## Verificaciones

Ejecute todas las pruebas:

```bash
python -m pytest -v
```

Sin el dataset privado, tres pruebas de integración se omiten automáticamente. Las pruebas unitarias restantes no necesitan fotografías reales. Cuando el dataset autorizado está presente, también se verifican los conteos y las particiones completas.

Compruebe el manifiesto, las imágenes y las particiones sin iniciar entrenamiento:

```bash
python entrenar_sin_cajas.py --solo-validar
```

## Entrenamiento

Prueba rápida de una época:

```bash
python entrenar_sin_cajas.py \
  --prueba-rapida \
  --run-name prueba-local
```

Entrenamiento completo:

```bash
python entrenar_sin_cajas.py \
  --entrenar \
  --epochs 30 \
  --patience 7 \
  --run-name modelo-sin-cajas-local
```

Los resultados se escriben en `resultados_sin_cajas/`, que está excluida de Git. Cada nombre de ejecución debe ser nuevo porque el programa evita sobrescribir resultados existentes.

## Preparación inicial de fotografías

`preparar_datos.py` convierte imágenes compatibles a JPEG RGB, corrige la orientación EXIF, conserva la proporción, limita el lado mayor a 1024 píxeles y genera reportes locales.

Coloque los originales autorizados con esta forma:

```text
datos/
└── originales/
    ├── vaca-001/
    │   └── foto.heic
    └── vaca-002/
        └── foto.jpg
```

Después ejecute:

```bash
python preparar_datos.py
```

Las imágenes, el inventario y los reportes generados permanecen excluidos del repositorio.

## Privacidad y publicación

Antes de contribuir, confirme que el cambio no incluya:

- fotografías o datasets;
- identificadores reales de animales;
- etiquetas, predicciones o particiones reales;
- rutas locales, documentos académicos o datos personales;
- tokens, credenciales o archivos `.env`;
- checkpoints o resultados de experimentos.

Revise siempre `git status` y `git diff --cached` antes de crear un commit.

Para contribuir al repositorio común, consulte también [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).
