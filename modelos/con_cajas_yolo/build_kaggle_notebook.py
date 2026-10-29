"""Reconstruye el notebook autónomo desde kaggle_src; no necesita Jupyter."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip()+'\n'})

    def code(text):
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                      'source': text.strip()+'\n'})

    md('''# BCS continuo: YOLO26 → grupa → EfficientNet-B0 / Keras → MAE

Notebook autónomo para **Kaggle Notebooks**. Sube `Datasetfotos` como dataset privado con
`train`, `valid` y `test`, cada uno con fotos y `_annotations.coco.json`. No necesita clonar el repositorio.
El código se genera en working; las entradas permanecen de solo lectura.

**Primeras sesiones:** (1) auditoría y preparación; (2) prueba corta del detector, estimación y entrenamiento;
(3) inspección de recortes, prueba corta y entrenamiento del regresor; (4) validación, cierre de decisiones y test.
Puedes combinar etapas si los tiempos medidos lo permiten. No hay garantía de terminar en una sesión.
Consulta en tu cuenta los límites y cuota actuales de Kaggle; no se presupone una duración fija.

Los entrenamientos largos están desactivados. No hay resultados de modelos precalculados en este archivo.
Las pruebas cortas son técnicas, con sus propios checkpoints, y no sustituyen una evaluación.
''')
    md('''## 1. Configuración y selección de etapas

Edita solamente esta configuración. El primer recorrido prepara datos sin entrenar.
Para una etapa posterior, desactiva las anteriores y adjunta los outputs persistidos mediante `RESUME_ROOT`.
Elige un nuevo `EXPERIMENT_ID` para cambiar hiperparámetros; un experimento existente protege su configuración.
Al reanudar mantén épocas y parámetros. Los objetivos representan épocas totales por fase.
''')
    code('''from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time

DATA_ROOT = "/kaggle/input/X"
WORK_ROOT = "/kaggle/working/bcs_project"
RESUME_ROOT = None  # "/kaggle/input/<outputs-previos>/"
EXPERIMENT_ID = "bcs_v1"
SEED = 42

RUN_AUDIT = True
RUN_PREPARE_DATA = True
RUN_TRAIN_DETECTOR = False
RUN_TRAIN_REGRESSOR = False
RUN_EVALUATION = False
RUN_DETECTOR_SMOKE = False
RUN_REGRESSOR_SMOKE = False
RUN_INFERENCE = False
RUN_EXPORT = True

SESSION_BUDGET_MINUTES = None  # Consulta el tiempo disponible mostrado por Kaggle.
SAFETY_MARGIN_MINUTES = 20
SMOKE_EPOCHS = 2
EVALUATION_SPLIT = "val"
FINAL_TEST_CONFIRMED = False  # True solo después de cerrar decisiones usando validación.
REGRESSOR_PHASE = "all"  # "head", "fine_tuning" o "all"
INFERENCE_IMAGE = None  # Ruta a una foto nueva adjunta como entrada.

# Mantener los frameworks de Kaggle. Instalar solo si falta el paquete; sin reemplazar torch/TF.
INSTALL_MISSING_DEPENDENCIES = True
ULTRALYTICS_VERSION = "8.4.0"
BOX_ROUNDING_TOLERANCE_PX = 0.02
DETECTOR = dict(weights="yolo26n.pt", epochs=80, imgsz=640, batch=4,
                device="0", patience=15, save_period=5)
REGRESSOR = dict(weights="imagenet", size=224, batch=4, device="0",
                 head_epochs=30, head_lr=1e-3, fine_tuning_epochs=20,
                 fine_tuning_lr=1e-5, fine_layers=20, patience=8)
INFERENCE = dict(confidence=0.25, association_coverage=0.8)

# También puedes apuntar DETECTOR['weights'] y REGRESSOR['weights'] a pesos locales
# adjuntos a Kaggle para trabajar sin Internet. No usar pesos aleatorios para el experimento final.
assert Path(EXPERIMENT_ID).name == EXPERIMENT_ID and EXPERIMENT_ID not in ("", ".", "..")
assert SESSION_BUDGET_MINUTES is None or SESSION_BUDGET_MINUTES > SAFETY_MARGIN_MINUTES
assert SAFETY_MARGIN_MINUTES >= 0 and SMOKE_EPOCHS >= 2
assert str(DETECTOR['device']).isdigit() and str(REGRESSOR['device']).isdigit()
assert 0 <= INFERENCE['confidence'] <= 1 and 0 < INFERENCE['association_coverage'] <= 1
assert BOX_ROUNDING_TOLERANCE_PX >= 0
assert all(DETECTOR[k] > 0 for k in ('epochs', 'imgsz', 'batch', 'patience', 'save_period'))
assert all(REGRESSOR[k] > 0 for k in ('size', 'batch', 'head_epochs', 'head_lr',
                                    'fine_tuning_epochs', 'fine_tuning_lr', 'fine_layers', 'patience'))
assert REGRESSOR['weights'] is not None, 'Los pesos aleatorios se reservan a la prueba técnica.'
if Path('/kaggle').exists():
    assert Path(WORK_ROOT).resolve().is_relative_to(Path('/kaggle/working')), 'WORK_ROOT debe estar en working.'

# El reloj conserva el inicio al repetir esta celda en el mismo kernel.
_SESSION_START_UNIX = globals().get('_SESSION_START_UNIX', time.time())
DEADLINE_UNIX = (None if SESSION_BUDGET_MINUTES is None else
                 _SESSION_START_UNIX + (SESSION_BUDGET_MINUTES-SAFETY_MARGIN_MINUTES)*60)
Path(WORK_ROOT, 'src').mkdir(parents=True, exist_ok=True)
SOURCE_FILES = {}
''')
    md('''### Módulos generados

Estas celdas contienen el código fuente del entregable. Se escriben antes de iniciar procesos;
no cargan frameworks en el kernel. Para mantener el notebook desde el repositorio, editar `kaggle_src`
y ejecutar `python3 build_kaggle_notebook.py`.
''')
    for path in sorted((ROOT / 'kaggle_src').glob('*.py')):
        source = path.read_text(encoding='utf-8')
        md(f'**Módulo `{path.name}`**')
        assert "'''" not in source, 'El delimitador del literal debe cambiar para este módulo.'
        code(f"SOURCE_FILES[{path.name!r}] = r'''{source}'''\n"
             f"Path(WORK_ROOT, 'src', {path.name!r}).write_text(SOURCE_FILES[{path.name!r}], encoding='utf-8')")
    code('''# Selección de dataset sin importar módulos de procesamiento ni frameworks.
input_root = Path('/kaggle/input')
print('Entradas disponibles:', sorted(p.name for p in input_root.iterdir()) if input_root.exists() else [])
if DATA_ROOT == '/kaggle/input/X':
    candidates = sorted({p.parent.parent for p in input_root.rglob('_annotations.coco.json')
                         if p.parent.name == 'train'
                         and (p.parent.parent/'valid/_annotations.coco.json').is_file()
                         and (p.parent.parent/'test/_annotations.coco.json').is_file()})
    print('Exports COCO candidatos:', [str(p) for p in candidates])
    if len(candidates) != 1:
        raise RuntimeError('Selecciona DATA_ROOT en la configuración. Hay varios exports o no se encontró ninguno; '
                           'adjunta el export COCO con imágenes. No se elegirá una entrada arbitraria.')
    DATA_ROOT = str(candidates[0])
if not Path(DATA_ROOT).is_dir():
    raise FileNotFoundError('DATA_ROOT no existe. Adjunta el export COCO con sus imágenes.')

CODE_HASH = hashlib.sha256(json.dumps(SOURCE_FILES, sort_keys=True).encode()).hexdigest()
keys = ['DATA_ROOT', 'WORK_ROOT', 'RESUME_ROOT', 'EXPERIMENT_ID', 'SEED',
        'SESSION_BUDGET_MINUTES', 'SAFETY_MARGIN_MINUTES', 'DEADLINE_UNIX',
        'SMOKE_EPOCHS', 'EVALUATION_SPLIT', 'FINAL_TEST_CONFIRMED', 'REGRESSOR_PHASE',
        'INFERENCE_IMAGE', 'ULTRALYTICS_VERSION', 'BOX_ROUNDING_TOLERANCE_PX',
        'DETECTOR', 'REGRESSOR', 'INFERENCE', 'CODE_HASH',
        'RUN_AUDIT', 'RUN_PREPARE_DATA', 'RUN_TRAIN_DETECTOR', 'RUN_TRAIN_REGRESSOR',
        'RUN_EVALUATION', 'RUN_DETECTOR_SMOKE', 'RUN_REGRESSOR_SMOKE', 'RUN_INFERENCE', 'RUN_EXPORT']
CFG = {key: globals()[key] for key in keys}
CONFIG_PATH = Path(WORK_ROOT, 'runtime_config.json')
CONFIG_PATH.write_text(json.dumps(CFG, indent=2), encoding='utf-8')

def run_stage(action, *extra):
    command = [sys.executable, '-u', str(Path(WORK_ROOT, 'src/run.py')), action,
               '--config', str(CONFIG_PATH), *extra]
    # Salida en vivo; cada proceso termina antes de iniciar el siguiente.
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2', WANDB_MODE='disabled')
    subprocess.run(command, check=True, env=env)

if SESSION_BUDGET_MINUTES is None:
    print('Presupuesto desconocido: consulta los límites actuales de tu cuenta; trabaja por etapas y persiste outputs.')
''')
    md('''## 2. Comprobación del entorno y dependencias

Se conservan PyTorch y TensorFlow de Kaggle. No se instalan controladores CUDA ni se actualizan
globalmente frameworks. Si hay una versión diferente de Ultralytics, la celda se detiene para evitar cambiarla
silenciosamente: instala la versión fijada en una sesión limpia o valida explícitamente otra versión.
Sin Internet, adjunta los pesos y las ruedas necesarias; no se cambia a pesos aleatorios como alternativa.

Las sondas GPU se ejecutan al solicitar entrenamiento, pruebas cortas, evaluación o inferencia.
La auditoría y preparación pueden hacerse con acelerador desactivado.
La VRAM se reporta por dispositivo: dos T4 no forman una GPU con memoria compartida.
Esta entrega usa una GPU por etapa. DDP (`device=[0,1]`) y MirroredStrategy requieren un benchmark separado,
pruebas de reanudación y mejora medida antes de incorporarlos; no se activan por tener dos dispositivos.
''')
    code('''requirements = {'Pillow': 'Pillow>=10,<13', 'numpy': 'numpy>=1.26,<3',
                'matplotlib': 'matplotlib>=3.7,<4', 'psutil': 'psutil>=5.9,<8',
                'PyYAML': 'PyYAML>=6,<7', 'pandas': 'pandas>=2,<4'}
need_torch = RUN_DETECTOR_SMOKE or RUN_TRAIN_DETECTOR or RUN_EVALUATION or RUN_INFERENCE
if need_torch:
    requirements.update({'ultralytics': 'ultralytics=='+ULTRALYTICS_VERSION,
                         'ultralytics-thop': 'ultralytics-thop>=2,<3',
                         'scipy': 'scipy>=1.11,<2', 'requests': 'requests>=2,<3',
                         'polars': 'polars>=0.20,<2'})
missing = []
for package, spec in requirements.items():
    try:
        version = importlib.metadata.version(package)
        if package == 'ultralytics' and version != ULTRALYTICS_VERSION:
            raise RuntimeError(f'Ultralytics {version} ya instalado; se requiere {ULTRALYTICS_VERSION}. '
                               'Usa una sesión limpia y fija la versión antes de importar frameworks.')
    except importlib.metadata.PackageNotFoundError:
        missing.append(spec)
if missing:
    if not INSTALL_MISSING_DEPENDENCIES:
        raise RuntimeError('Faltan dependencias: '+', '.join(missing))
    # --no-deps evita que instalar Ultralytics reemplace torch, CUDA o numpy de Kaggle.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--no-deps', *missing], check=True)
if need_torch:
    # Estos paquetes los aporta Kaggle; no reemplazarlos con una instalación automática.
    subprocess.run([sys.executable, '-c', 'import torch, torchvision, cv2, ultralytics'], check=True)
run_stage('restore')
artifact_root = Path(WORK_ROOT, 'artifacts')
(artifact_root/'configs').mkdir(parents=True, exist_ok=True)
(artifact_root/'configs/config.json').write_text(json.dumps(CFG, indent=2), encoding='utf-8')
# Registrar exactamente lo instalado; útil al preparar una sesión de reanudación.
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(artifact_root/'configs/packages.txt').write_text(freeze)
run_stage('environment', '--framework', 'system')
if RUN_DETECTOR_SMOKE or RUN_TRAIN_DETECTOR or RUN_EVALUATION or RUN_INFERENCE:
    run_stage('environment', '--framework', 'torch')
if RUN_REGRESSOR_SMOKE or RUN_TRAIN_REGRESSOR or RUN_EVALUATION or RUN_INFERENCE:
    run_stage('environment', '--framework', 'tensorflow')
''')
    md('''## 3. Inspección del dataset COCO

Inspeccionar las categorías originales antes de exportar. El mapeo del detector es `0=cow`, `1=ass`;
el ID de categoría contenedora de Roboflow no se transforma en una clase entrenable.
Los IDs de imagen COCO son locales a cada partición; la clave de trazabilidad incluye el nombre de partición.
''')
    code("run_stage('inspect')")
    md('''## 4. Auditoría y manifiesto

Comprobar imágenes, dimensiones, cajas, etiquetas BCS, identidades y SHA-256 entre particiones.
Las fotos con ID/BCS ambiguos conservan supervisión de detección y se excluyen de regresión.
No se adivina cuál BCS corresponde a una grupa. Los desbordamientos de redondeo de hasta 0.02 píxeles
se limitan al borde y quedan documentados; errores mayores bloquean la preparación.
Las huellas se vuelven a verificar al ejecutar etapas posteriores, aunque `RUN_AUDIT=False`.
''')
    code('''if RUN_AUDIT:
    run_stage('audit')
summary_path = artifact_root/'reports/audit_summary.json'
if summary_path.exists():
    import pandas as pd
    display(pd.DataFrame(json.loads(summary_path.read_text())))
    print('Las proporciones se calculan por separado para detector y regresor después de las exclusiones.')
''')
    md('''## 5. Particiones por vaca

Conservar `train/valid/test` originales después de verificar aislamiento de todas las identidades,
incluidas las etiquetas ambiguas. La validación no se reemplaza por test para reducir tiempos.
Guardar asignaciones y hash. Las correcciones futuras de datos requieren una nueva versión de trabajo.
SHA-256 no detecta todos los duplicados visualmente similares ni identidades mal anotadas: revisar muestras.
''')
    code("if RUN_PREPARE_DATA:\n    run_stage('partition')")
    md('''## 6. Exportación a formato YOLO

Convertir `xywh` COCO a cajas normalizadas y guardar YAML de rutas de esta sesión.
Las fotos se copian a working para conservar una exportación portable; no se escriben etiquetas en input.
La exportación se reutiliza únicamente si sus archivos y huellas coinciden.
''')
    code("if RUN_PREPARE_DATA:\n    run_stage('export-yolo')")
    md('''## 7. Entrenamiento o reanudación de YOLO26

Primero activa `RUN_DETECTOR_SMOKE`: dos épocas separadas del experimento final, con parada después
de la primera y recuperación del optimizador para la segunda. Se guarda tiempo por época y memoria pico.
Antes del entrenamiento largo revisa la estimación y consulta el tiempo restante de Kaggle.
Con presupuesto configurado se reserva el margen indicado y se pausa al completar una época.
Una interrupción abrupta puede perder la época en curso; ningún callback puede impedir que Kaggle termine la sesión.

Guardar `best.pt`, `last.pt`, periódicos y `resume.pt` antes de la limpieza del optimizador de Ultralytics.
El checkpoint recuperable preserva la época y estado de entrenamiento; `best.pt` se usa para inferencia.
Cambiar épocas después de finalizar no es una reanudación compatible en esta versión:
elige otro `EXPERIMENT_ID` y configura `DETECTOR['weights']` con los pesos anteriores para un nuevo experimento.
La paciencia de Ultralytics y el flujo aleatorio no se prometen idénticos al reanudar.

Referencias: [entrenamiento](https://docs.ultralytics.com/modes/train/),
[implementación fijada del trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/engine/trainer.py).
La sonda guarda el código de reanudación y finalización de la versión realmente instalada.
''')
    code('''if RUN_DETECTOR_SMOKE:
    run_stage('detector-smoke')
estimate_path = artifact_root/'reports/detector_smoke.json'
if estimate_path.exists():
    print(estimate_path.read_text())
if RUN_TRAIN_DETECTOR:
    run_stage('estimate')
    run_stage('detector-train')
''')
    md('''## 8. Generación de muestras de regresión

Usar cajas anotadas de grupa para entrenar. Crear recortes RGB con relleno 224×224, sin deformar.
El preprocesamiento será el mismo en entrenamiento, evaluación e inferencia. Los IDs solo sirven para
trazabilidad, particiones y reportes; no son entradas de la red.
''')
    code('''if RUN_PREPARE_DATA:
    run_stage('samples')
samples_path = artifact_root/'manifests/regression_samples.json'
if samples_path.exists():
    from IPython.display import display, Image as DisplayImage
    sample_rows = json.loads(samples_path.read_text())
    for row in sample_rows[:6]:
        print(row['split'], 'BCS:', row['bcs'])
        display(DisplayImage(filename=str(artifact_root/row['crop']), width=180))
''')
    md('''## 9. Entrenamiento o reanudación de EfficientNet-B0 con MAE

Salida escalar lineal y pérdida MAE; no hay softmax, pesos de clase ni redondeo a categorías.
La red recibe RGB en 0..255 porque EfficientNet integra su escalado.
Solo entrenamiento usa aumentos; BatchNormalization permanece congelada durante ajuste fino.

Activar primero `RUN_REGRESSOR_SMOKE`: usa pesos aleatorios únicamente para comprobar las dos fases,
guardar/cargar modelos completos y verificar que el contador del optimizador continúa tras una pausa.
Sus métricas no son resultados científicos ni seleccionan el modelo final.
El entrenamiento final usa ImageNet o los pesos locales configurados.

Cada época guarda un modelo completo, estado JSON e historial CSV. `best.keras` se selecciona por
MAE de validación; `last.keras` refleja la última época completa. El estado restaura mejor métrica,
paciencia y optimizador; se conservan checkpoints inmutables para recuperación ante escrituras interrumpidas.
Entre cabeza y ajuste fino se recompila con un optimizador nuevo: es un cambio de fase deliberado.
La mejor fase se selecciona por validación. No se garantiza continuidad bit a bit de RNG o del cargador.

Referencia: [EfficientNet de Keras](https://keras.io/api/applications/efficientnet/).
''')
    code('''if RUN_REGRESSOR_SMOKE:
    run_stage('regressor-smoke')
estimate_path = artifact_root/'reports/regressor_smoke.json'
if estimate_path.exists():
    print(estimate_path.read_text())
if RUN_TRAIN_REGRESSOR:
    run_stage('estimate')
    run_stage('regressor-train')
''')
    md('''## 10. Evaluación

Por defecto evaluar validación. Cerrar hiperparámetros, fase y umbrales antes de configurar
`EVALUATION_SPLIT="test"` y `FINAL_TEST_CONFIRMED=True`. Se guarda una selección con hashes para
detectar cambios después de abrir test. No es un mecanismo para impedir que alguien cree otro proyecto;
la independencia de test también requiere disciplina experimental.

Reportar detección (precisión, recall, mAP), regresión con cajas anotadas y pipeline con cajas predichas.
La asociación automática exige un único par vaca/grupa con centro contenido y cobertura mínima configurada.
La ausencia o ambigüedad produce abstención sin consultar ID, BCS ni cajas reales.
MAE y RMSE del sistema se calculan sobre predicciones disponibles y se muestran junto con cobertura;
las tolerancias globales incluyen abstenciones como fallos. Agregar reportes por vaca y promedio macro.
Comparar también con la mediana BCS de entrenamiento. No recortar predicciones al rango ni redondearlas.
''')
    code('''if RUN_EVALUATION:
    run_stage('detector-evaluate')
    run_stage('detector-crops')
    run_stage('evaluate')
    for tag in ('regressor_gt_', 'end_to_end_', 'baseline_'):
        print(tag+EVALUATION_SPLIT)
        print((artifact_root/'reports'/(tag+EVALUATION_SPLIT)/'metrics.json').read_text())
''')
    md('''## 11. Inferencia y visualización

Configurar `INFERENCE_IMAGE` con una foto nueva. Ejecutar detector y regresor en procesos sucesivos.
La salida es un BCS continuo o una abstención con motivo; los scores de detección no son confianza BCS.
También mostrar ejemplos de la última evaluación cuando estén disponibles.
''')
    code('''if RUN_INFERENCE:
    run_stage('infer-detector')
    run_stage('infer-regressor')
    print((artifact_root/'reports/inference_external/prediction.json').read_text())
from IPython.display import display, Image as DisplayImage
preview_tag = 'external' if RUN_INFERENCE else EVALUATION_SPLIT
for path in sorted((artifact_root/'predicted'/preview_tag).glob('*_boxes.jpg'))[:4]:
    display(DisplayImage(filename=str(path), width=420))
predictions_path = artifact_root/'predicted'/preview_tag/'predictions.json'
if predictions_path.exists():
    for row in json.loads(predictions_path.read_text())['rows'][:4]:
        print('Asociación:', row['status'])
        if row['crop']:
            display(DisplayImage(filename=str(artifact_root/row['crop']), width=180))
for tag in ('regressor_gt_', 'end_to_end_'):
    path = artifact_root/'reports'/(tag+EVALUATION_SPLIT)/'errors.png'
    if path.exists():
        display(DisplayImage(filename=str(path), width=720))
''')
    md('''## 12. Exportación de artefactos y continuación entre sesiones

1. Ejecutar esta sección antes de terminar. Revisar estados; una etapa pausada no está completada.
2. En **Save Version**, usar una modalidad que guarde outputs y comprobar la opción correspondiente.
   Para guardar el estado interactivo sin repetir entrenamiento, revisar las opciones avanzadas de Quick Save
   y activar el guardado de outputs si está disponible. **Save & Run All** vuelve a ejecutar el notebook:
   revisar banderas antes de usarlo. La interfaz puede cambiar; verificar el resultado guardado.
3. Abrir los outputs de la versión y comprobar que aparecen `artifacts/state`, checkpoints e inventario.
   Descargar también `bcs_project_artifacts.zip` desde el panel de archivos de Kaggle.
4. En una sesión nueva, adjuntar los outputs de esa versión como entrada, además del dataset original.
   Apuntar `RESUME_ROOT` a la carpeta que contiene `artifacts` o a `artifacts` directamente.
   Si solo se conserva el ZIP, extraerlo a working y usar esa carpeta como `RESUME_ROOT`.
5. Mantener `EXPERIMENT_ID` y parámetros; configurar únicamente las etapas pendientes y el presupuesto
   disponible de la nueva sesión. El notebook valida el inventario y copia a working sin alterar input.
   No basta adjuntar los pesos: la reanudación requiere estados, manifiestos y configuración compatibles.

Guardar en `/kaggle/working` o el autoguardado del código **no garantiza** conservar los outputs tras perder
la sesión. Verificar la versión persistida antes de cerrar. Sin presupuesto conocido, exportar después de cada etapa.
El ZIP incluye datos preparados y reportes privados: mantener privados el notebook, dataset y outputs.
Los archivos fuente publicados en Git deben permanecer sin outputs ni datos de animales.

Referencia: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).
''')
    code('''run_stage('status')
if RUN_EXPORT:
    run_stage('export')
    print('Descargar desde el panel de archivos:', Path(WORK_ROOT).parent/(Path(WORK_ROOT).name+'_artifacts.zip'))
''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'language': 'python', 'isInternetEnabled': True, 'isGpuEnabled': False},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    for n, cell in enumerate(cells):
        cell['id'] = hashlib.sha256(f'{n}:{cell["source"]}'.encode()).hexdigest()[:12]
        cell['source'] = cell['source'].splitlines(keepends=True)
    output = ROOT / 'bcs_kaggle.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    build()
