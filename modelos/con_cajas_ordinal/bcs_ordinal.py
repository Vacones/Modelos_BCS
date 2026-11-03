# %% [md]
# # Modelo ordinal de BCS: código del experimento
#
# Compara cuatro salidas sobre la misma red (EfficientNet-B0) con validación cruzada por vaca:
# softmax, ordinal CORAL, ordinal CORN y regresión, más la constante como referencia.
# También compara recorte del trasero contra imagen completa.
#
# Hay dos experimentos:
# - **A, rápido:** la base queda congelada y solo se entrena la salida sobre características guardadas. Tarda minutos.
# - **B, completo:** ajuste fino en dos etapas, con aumentos, como en la tabla del punto 5. Necesita GPU.
#
# Las 10 vacas de prueba no se usan hasta la sección final, y solo si `EVALUAR_PRUEBA = True`.

# %% [code] Librerías
import os, json, math, time, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# Motor de Keras: en Windows se usa PyTorch, porque TensorFlow ya no usa la GPU en Windows nativo.
# En Colab o Linux se usa TensorFlow. Para forzar uno, define KERAS_BACKEND antes de correr.
os.environ.setdefault("KERAS_BACKEND", "torch" if os.name == "nt" else "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers, ops

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def describir_equipo():
    if keras.backend.backend() == "torch":
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no hay, corre en CPU"
        return f"motor PyTorch {torch.__version__} | GPU: {gpu}"
    import tensorflow as tf
    return f"motor TensorFlow {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU') or 'no hay, corre en CPU'}"


print("Keras", keras.__version__, "|", describir_equipo())

# %% [md]
# ## 1. Configuración
#
# Cambia solo esta celda. En `CARPETA` deben estar dos archivos:
# - `Vacones BCS - Cajas v2 Aprobado.yolo26.zip`, el export de Roboflow (o la carpeta ya descomprimida).
# - `metadata_bcs_validos.csv` de Jacob, que liga cada imagen con su vaca y su BCS.
#
# Los resultados se guardan en `CARPETA/resultados_bcs_ordinal`, así sobreviven si Colab se desconecta.

# %% [code] Configuración
try:
    from google.colab import drive
    drive.mount("/content/drive")
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

CARPETA = Path(os.environ.get("BCS_CARPETA", "/content/drive/MyDrive/BCS_ordinal" if EN_COLAB else ".")).resolve()
NOMBRE_EXPORT = "Vacones BCS - Cajas v2 Aprobado.yolo26"


def buscar_export():
    """Busca el export ya descomprimido en CARPETA o en la carpeta de arriba; si no, descomprime el zip."""
    destino = (Path("/content/datos") if EN_COLAB else CARPETA) / NOMBRE_EXPORT
    for ruta in (destino, CARPETA / NOMBRE_EXPORT, CARPETA.parent / NOMBRE_EXPORT):
        if (ruta / "data.yaml").exists():
            return ruta
    for ruta in (CARPETA / f"{NOMBRE_EXPORT}.zip", CARPETA.parent / f"{NOMBRE_EXPORT}.zip"):
        if ruta.exists():
            import zipfile
            zipfile.ZipFile(ruta).extractall(destino)
            return destino
    raise FileNotFoundError(f"No encuentro '{NOMBRE_EXPORT}' (carpeta o .zip) en {CARPETA} ni en {CARPETA.parent}")


RUTA_DATASET = buscar_export()
print("Datos:", RUTA_DATASET)
RUTA_METADATA = CARPETA / "metadata_bcs_validos.csv"
RUTA_SALIDA = CARPETA / "resultados_bcs_ordinal"

CABEZAS = ["softmax", "coral", "corn", "regresion"]   # salidas a comparar
ENTRADAS = ["recorte", "completa"]                      # recorte del trasero o imagen completa
SEMILLAS = [20261026, 20261027, 20261028]
N_PLIEGUES = 5

# Experimento B (ajuste fino). Por tiempo, por defecto solo recorte y una semilla.
ENTRADAS_B = ["recorte"]
SEMILLAS_B = [20261026]
LR_ETAPA2 = 1e-5          # explorar 1e-5, 3e-5 y 1e-4

TAM = 224                 # entrada de EfficientNet-B0
MARGEN = 0.10             # margen del recorte por lado
BATCH = 16
EPOCAS_MAX = 30
PACIENCIA = 8
FRACCION_VAL_INTERNA = 0.2  # vacas del pliegue de entrenamiento reservadas para la parada temprana

EVALUAR_PRUEBA = False    # True solo al final, una vez
RUTA_CAJAS_YOLO_PRUEBA = None  # carpeta con los .txt de `yolo predict save_txt=True save_conf=True`; None usa las cajas de Roboflow

RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

# Niveles: 2.25 se junta con 2.50 porque hay una sola vaca con ese valor.
NIVELES = np.array([2.50, 2.75, 3.00, 3.25, 3.50, 3.75])
K = len(NIVELES)          # 6 niveles, 5 umbrales


def bcs_a_nivel(bcs):
    return np.clip(np.rint((np.asarray(bcs, dtype=float) - 2.50) / 0.25), 0, K - 1).astype(int)


def categoria3(bcs):
    """0 = baja (<= 2.75), 1 = adecuada (3.00 a 3.25), 2 = alta (>= 3.50)."""
    bcs = np.asarray(bcs, dtype=float)
    return np.where(bcs <= 2.75, 0, np.where(bcs <= 3.25, 1, 2))


# %% [md]
# ## 2. Datos
#
# Lee las cajas de Roboflow por nombre de clase (`cow`, `ass`), así funciona igual con el export directo
# (`ass` = 0) y con el VERIFICADO de Jacob (`cow` = 0).

# %% [code] Carga de datos
def leer_clases(ruta_dataset):
    texto = (Path(ruta_dataset) / "data.yaml").read_text(encoding="utf-8")
    try:
        import yaml
        nombres = yaml.safe_load(texto)["names"]
    except Exception:
        linea = [l for l in texto.splitlines() if l.strip().startswith("names")][0]
        nombres = [n.strip(" '\"") for n in linea.split("[", 1)[1].rsplit("]", 1)[0].split(",")]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]
    return {n: i for i, n in enumerate(nombres)}


def leer_cajas(ruta_txt, clases, umbral_conf=0.0):
    """Devuelve la caja `ass` asociada a la caja `cow` de mayor confianza, en coordenadas relativas (cx, cy, w, h)."""
    if not Path(ruta_txt).exists():
        return None
    vacas, traseros = [], []
    for linea in Path(ruta_txt).read_text().splitlines():
        p = linea.split()
        if len(p) < 5:
            continue
        c, caja = int(p[0]), [float(v) for v in p[1:5]]
        conf = float(p[5]) if len(p) > 5 else 1.0
        if c == clases["cow"]:
            vacas.append((conf, caja))
        elif c == clases["ass"] and conf >= umbral_conf:
            traseros.append((conf, caja))
    if not traseros:
        return None
    if not vacas:
        return max(traseros)[1] if umbral_conf == 0.0 else None
    _, (vx, vy, vw, vh) = max(vacas)
    dentro = [(conf, (cx, cy, w, h)) for conf, (cx, cy, w, h) in traseros
              if abs(cx - vx) <= vw / 2 and abs(cy - vy) <= vh / 2]
    return max(dentro)[1] if dentro else None


def cargar_tabla(ruta_dataset=RUTA_DATASET, ruta_metadata=RUTA_METADATA):
    meta = pd.read_csv(ruta_metadata)
    meta = meta[meta["metadata_status"].eq("ok")].copy()
    clases = leer_clases(ruta_dataset)
    carpetas = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    filas = []
    for r in meta.itertuples():
        img = Path(ruta_dataset) / carpetas[r.split] / "images" / r.file_name
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        filas.append({"archivo": str(img), "etiqueta": str(lbl), "existe": img.exists(),
                      "conjunto": "val" if r.split == "valid" else r.split,
                      "vaca": int(r.cow_id), "bcs": float(r.bcs),
                      "caja_ass": leer_cajas(lbl, clases)})
    tabla = pd.DataFrame(filas)
    tabla["nivel"] = bcs_a_nivel(tabla["bcs"])
    return tabla, clases


def verificar_tabla(tabla):
    print("Imágenes:", len(tabla), "| faltan en disco:", (~tabla.existe).sum(),
          "| sin caja ass:", tabla.caja_ass.isna().sum())
    por_vaca = tabla.groupby("vaca").agg(bcs=("bcs", "nunique"), conjunto=("conjunto", "nunique"))
    assert (por_vaca.bcs == 1).all(), "Hay vacas con más de un BCS"
    assert (por_vaca.conjunto == 1).all(), "Hay vacas repartidas en más de un conjunto"
    vacas = tabla.groupby("vaca").agg(bcs=("bcs", "first"), conjunto=("conjunto", "first"), fotos=("archivo", "size"))
    print(pd.crosstab(vacas.bcs, vacas.conjunto, margins=True, margins_name="total"))
    return vacas


# %% [md]
# ## 3. Imágenes y recortes
#
# Cada foto se carga una vez. Para el recorte se guarda una región con 25% de margen alrededor de la caja `ass`;
# al entrenar se corta de ahí con margen aleatorio de 5% a 15% y desplazamiento de ±5%.
# Para evaluar, el margen es fijo de 10%. Todo se lleva a 224 × 224 con relleno, sin deformar.

# %% [code] Preparación de imágenes
def abrir(ruta):
    return ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")


def letterbox(im, tam=TAM):
    w, h = im.size
    r = tam / max(w, h)
    im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)
    lienzo = Image.new("RGB", (tam, tam), (0, 0, 0))
    lienzo.paste(im, ((tam - im.size[0]) // 2, (tam - im.size[1]) // 2))
    return lienzo


def preparar_fuentes(tabla, entrada):
    """Lista de (imagen PIL, caja en píxeles dentro de esa imagen o None)."""
    fuentes = []
    for r in tabla.itertuples():
        im = abrir(r.archivo)
        W, H = im.size
        if entrada == "completa" or r.caja_ass is None:
            im.thumbnail((2 * TAM, 2 * TAM))
            fuentes.append((im, None))
            continue
        cx, cy, w, h = r.caja_ass
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        x0, y0 = max(0, cx - 0.75 * w), max(0, cy - 0.75 * h)
        x1, y1 = min(W, cx + 0.75 * w), min(H, cy + 0.75 * h)
        region = im.crop((round(x0), round(y0), round(x1), round(y1)))
        escala = min(1.0, 3 * TAM / max(region.size))
        if escala < 1:
            region = region.resize((round(region.size[0] * escala), round(region.size[1] * escala)), Image.BICUBIC)
        caja = ((cx - x0) * escala, (cy - y0) * escala, w * escala, h * escala)
        fuentes.append((region, caja))
    return fuentes


def vista(fuente, aumentar=False, rng=None):
    im, caja = fuente
    if caja is not None:
        cx, cy, w, h = caja
        m = rng.uniform(0.05, 0.15) if aumentar else MARGEN
        if aumentar:
            cx += rng.uniform(-0.05, 0.05) * w
            cy += rng.uniform(-0.05, 0.05) * h
        W, H = im.size
        im = im.crop((max(0, round(cx - w * (0.5 + m))), max(0, round(cy - h * (0.5 + m))),
                      min(W, round(cx + w * (0.5 + m))), min(H, round(cy + h * (0.5 + m)))))
    im = letterbox(im)
    if aumentar:
        if rng.random() < 0.5:
            im = ImageOps.mirror(im)
        im = im.rotate(rng.uniform(-5, 5), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.9, 1.1))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.9, 1.1))
    return np.asarray(im, dtype=np.float32)  # 0 a 255; EfficientNetB0 de Keras normaliza por dentro


class Lotes(keras.utils.PyDataset):
    """Lotes de imágenes con aumentos; y = nivel (0 a 5) o BCS − 3.00 para regresión; pesos 1 / fotos de la vaca."""

    def __init__(self, fuentes, y, pesos, aumentar, semilla, **kw):
        super().__init__(**kw)
        self.fuentes, self.y, self.pesos, self.aumentar = fuentes, y, pesos, aumentar
        self.rng = np.random.default_rng(semilla)
        self.orden = np.arange(len(fuentes))
        if aumentar:
            self.rng.shuffle(self.orden)

    def __len__(self):
        return math.ceil(len(self.fuentes) / BATCH)

    def __getitem__(self, i):
        idx = self.orden[i * BATCH:(i + 1) * BATCH]
        x = np.stack([vista(self.fuentes[j], self.aumentar, self.rng) for j in idx])
        return x, self.y[idx], self.pesos[idx]

    def on_epoch_end(self):
        if self.aumentar:
            self.rng.shuffle(self.orden)


def matriz(fuentes, espejo=False):
    x = np.stack([vista(f) for f in fuentes])
    return x[:, :, ::-1, :].copy() if espejo else x


def pesos_por_vaca(vacas):
    vacas = pd.Series(vacas)
    p = 1.0 / vacas.map(vacas.value_counts()).to_numpy()
    return (p * len(p) / p.sum()).astype("float32")


# %% [md]
# ## 4. Salidas y pérdidas
#
# - **softmax:** 6 niveles, entropía cruzada (como el reporte, con 2.25 junto a 2.50).
# - **coral:** un peso compartido y 5 sesgos; P(BCS > umbral) = σ(w·h + b_k). Cao, Mirjalili y Raschka (2020).
# - **corn:** 5 salidas independientes entrenadas de forma condicional; P(BCS > u_k) = Π σ(z_j). Shi, Cao y Raschka (2023).
# - **regresion:** una salida, objetivo BCS − 3.00, pérdida MAE.

# %% [code] Salidas y pérdidas
@keras.saving.register_keras_serializable(package="bcs")
class SesgosCoral(layers.Layer):
    def __init__(self, n_umbrales, inicial=None, **kw):
        super().__init__(**kw)
        self.n_umbrales = n_umbrales
        self.inicial = None if inicial is None else [float(v) for v in inicial]

    def build(self, forma):
        ini = keras.initializers.Constant(self.inicial) if self.inicial else "zeros"
        self.b = self.add_weight(shape=(self.n_umbrales,), initializer=ini, name="sesgos")

    def call(self, z):
        return z + self.b

    def get_config(self):
        return {**super().get_config(), "n_umbrales": self.n_umbrales, "inicial": self.inicial}


def logit_previo(niveles_entrenamiento):
    """Sesgos iniciales de CORAL: logit de la proporción de fotos por encima de cada umbral."""
    p = np.array([(niveles_entrenamiento > k).mean() for k in range(K - 1)]).clip(0.02, 0.98)
    return np.log(p / (1 - p))


def capa_salida(cabeza, h, niveles_entrenamiento):
    if cabeza == "softmax":
        return layers.Dense(K, activation="softmax", name="salida")(h)
    if cabeza == "regresion":
        return layers.Dense(1, name="salida")(h)
    if cabeza == "coral":
        z = layers.Dense(1, use_bias=False, name="coral_peso")(h)
        return SesgosCoral(K - 1, logit_previo(niveles_entrenamiento), name="salida")(z)
    if cabeza == "corn":
        return layers.Dense(K - 1, name="salida")(h)
    raise ValueError(cabeza)


def _niveles(y_true):
    return ops.cast(ops.reshape(y_true, (-1, 1)), "float32")


@keras.saving.register_keras_serializable(package="bcs")
def perdida_coral(y_true, logits):
    y = _niveles(y_true)
    umbrales = ops.cast(ops.arange(K - 1), "float32")[None, :]
    objetivo = ops.cast(y > umbrales, "float32")
    ls = ops.log_sigmoid(logits)
    return -ops.mean(objetivo * ls + (1 - objetivo) * (ls - logits), axis=1)


@keras.saving.register_keras_serializable(package="bcs")
def perdida_corn(y_true, logits):
    y = _niveles(y_true)
    umbrales = ops.cast(ops.arange(K - 1), "float32")[None, :]
    objetivo = ops.cast(y > umbrales, "float32")
    activo = ops.cast(y >= umbrales, "float32")  # tarea k solo con vacas que ya superaron el umbral k − 1
    ls = ops.log_sigmoid(logits)
    bce = -(objetivo * ls + (1 - objetivo) * (ls - logits))
    return ops.sum(bce * activo, axis=1) / ops.maximum(ops.sum(activo, axis=1), 1.0)


PERDIDAS = {"softmax": "sparse_categorical_crossentropy", "coral": perdida_coral,
            "corn": perdida_corn, "regresion": "mean_absolute_error"}


def objetivo(cabeza, tabla):
    if cabeza == "regresion":
        return (tabla["bcs"].to_numpy() - 3.0).astype("float32").reshape(-1, 1)
    return tabla["nivel"].to_numpy().astype("int32")


def acumuladas(cabeza, salida):
    """Probabilidades P(BCS > umbral_k) por foto, forma (n, 5). None para regresión."""
    salida = np.asarray(salida, dtype=float)
    if cabeza == "softmax":
        return 1 - np.cumsum(salida, axis=1)[:, :-1]
    if cabeza == "coral":
        return 1 / (1 + np.exp(-salida))
    if cabeza == "corn":
        return np.cumprod(1 / (1 + np.exp(-salida)), axis=1)
    return None


def resumir_por_vaca(cabeza, salida, vacas):
    """Agrega las fotos de cada vaca y devuelve BCS predicho, valor continuo y P(BCS <= 2.75)."""
    salida = np.asarray(salida, dtype=float)
    df = pd.DataFrame({"vaca": np.asarray(vacas)})
    if cabeza == "regresion":
        df["cont"] = 3.0 + salida[:, 0]
        g = df.groupby("vaca").cont.mean()
        pred = NIVELES[bcs_a_nivel(g.to_numpy())]
        return pd.DataFrame({"vaca": g.index, "bcs_pred": pred, "bcs_cont": g.to_numpy(), "p_baja": np.nan})
    if cabeza == "softmax":
        P = pd.DataFrame(salida).groupby(df.vaca.to_numpy()).mean()
        clase = P.to_numpy().argmax(1)
        cont = 2.50 + 0.25 * (P.to_numpy() @ np.arange(K))
        p_baja = P.to_numpy()[:, :2].sum(1)
        return pd.DataFrame({"vaca": P.index, "bcs_pred": NIVELES[clase], "bcs_cont": cont, "p_baja": p_baja})
    A = pd.DataFrame(acumuladas(cabeza, salida)).groupby(df.vaca.to_numpy()).mean()
    a = A.to_numpy()
    return pd.DataFrame({"vaca": A.index, "bcs_pred": 2.50 + 0.25 * (a > 0.5).sum(1),
                         "bcs_cont": 2.50 + 0.25 * a.sum(1), "p_baja": 1 - a[:, 1]})


# %% [md]
# ## 5. Modelo y entrenamiento
#
# La parada temprana usa un 20% de las vacas del pliegue de entrenamiento, nunca las del pliegue que se evalúa.
# Se vigila el MAE por vaca del valor continuo, que cambia más suave que el redondeado.

# %% [code] Modelo
def base_efficientnet(entrada):
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=entrada)
    for capa in base.layers:
        capa.trainable = False
    return base


def modelo_completo(cabeza, niveles_entrenamiento):
    entrada = keras.Input((TAM, TAM, 3), name="imagen")
    base = base_efficientnet(entrada)
    h = layers.GlobalAveragePooling2D(name="promedio_global")(base.output)
    h = layers.Dropout(0.3, name="dropout")(h)
    return keras.Model(entrada, capa_salida(cabeza, h, niveles_entrenamiento), name=f"b0_{cabeza}")


def modelo_sobre_caracteristicas(cabeza, niveles_entrenamiento, dim=1280):
    entrada = keras.Input((dim,), name="caracteristicas")
    h = layers.Dropout(0.3, name="dropout")(entrada)
    return keras.Model(entrada, capa_salida(cabeza, h, niveles_entrenamiento), name=f"cabeza_{cabeza}")


def descongelar_etapa2(modelo):
    for capa in modelo.layers:
        nombre = capa.name
        if (nombre.startswith("block7") or nombre.startswith("top_")) and not isinstance(capa, layers.BatchNormalization):
            capa.trainable = True


def compilar(modelo, cabeza, lr):
    modelo.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4), loss=PERDIDAS[cabeza])


class MAEPorVaca(keras.callbacks.Callback):
    def __init__(self, cabeza, x_val, vacas_val, bcs_por_vaca):
        super().__init__()
        self.cabeza, self.x_val, self.vacas_val, self.real = cabeza, x_val, vacas_val, bcs_por_vaca

    def on_epoch_end(self, epoca, logs=None):
        r = resumir_por_vaca(self.cabeza, self.model.predict(self.x_val, verbose=0, batch_size=64), self.vacas_val)
        error = np.abs(r.bcs_cont.to_numpy() - self.real.loc[r.vaca].to_numpy())
        logs["val_mae_vaca"] = float(error.mean())


def entrenar(modelo, cabeza, datos_fit, x_val, vacas_val, bcs_val, lr, etapa):
    compilar(modelo, cabeza, lr)
    cb = [MAEPorVaca(cabeza, x_val, vacas_val, bcs_val),
          keras.callbacks.ReduceLROnPlateau(monitor="val_mae_vaca", mode="min", factor=0.5, patience=3, verbose=0),
          keras.callbacks.EarlyStopping(monitor="val_mae_vaca", mode="min", patience=PACIENCIA,
                                        restore_best_weights=True, verbose=0)]
    if isinstance(datos_fit, tuple):
        x, y, pesos = datos_fit
        h = modelo.fit(x, y, sample_weight=pesos, batch_size=BATCH, shuffle=True,
                       epochs=EPOCAS_MAX, callbacks=cb, verbose=0)
    else:
        h = modelo.fit(datos_fit, epochs=EPOCAS_MAX, callbacks=cb, verbose=0)
    hist = pd.DataFrame(h.history)
    hist["etapa"] = etapa
    return hist


# %% [md]
# ## 6. Validación cruzada por vaca
#
# Pliegues estratificados por nivel sobre las vacas de entrenamiento y validación (56). Cada combinación
# guarda sus predicciones por vaca en `predicciones_*.csv`; si Colab se desconecta, al volver a correr
# se saltan las combinaciones ya hechas. Cada fila lleva la columna `config` con los hiperparámetros:
# si cambias uno (por ejemplo `LR_ETAPA2`), se corre como configuración nueva y las anteriores se conservan.

# %% [code] Pliegues
def pliegues(vacas_cv, semilla):
    """Devuelve lista de (vacas_entrenamiento, vacas_evaluacion)."""
    ids = vacas_cv.index.to_numpy()
    estrato = bcs_a_nivel(vacas_cv.bcs.to_numpy())
    skf = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=semilla)
    return [(ids[a], ids[b]) for a, b in skf.split(ids, estrato)]


def separar_val_interna(vacas_ent, vacas_cv, semilla):
    estrato = bcs_a_nivel(vacas_cv.loc[vacas_ent].bcs.to_numpy())
    sss = StratifiedShuffleSplit(1, test_size=FRACCION_VAL_INTERNA, random_state=semilla)
    a, b = next(sss.split(vacas_ent, estrato))
    return vacas_ent[a], vacas_ent[b]


def fila_constante(vacas_ent, vacas_ev, vacas_cv):
    mediana = NIVELES[bcs_a_nivel(np.median(vacas_cv.loc[vacas_ent].bcs))]
    return pd.DataFrame({"vaca": vacas_ev, "bcs_pred": mediana, "bcs_cont": mediana, "p_baja": np.nan})


def ya_hecho(ruta, **clave):
    if not Path(ruta).exists():
        return False
    df = pd.read_csv(ruta)
    m = np.ones(len(df), bool)
    for k, v in clave.items():
        m &= df[k].astype(str).eq(str(v)).to_numpy()
    return m.any()


def config_a():
    return f"pliegues{N_PLIEGUES}_epocas{EPOCAS_MAX}_paciencia{PACIENCIA}_margen{MARGEN:g}"


def config_b():
    return (f"pliegues{N_PLIEGUES}_lr2-{LR_ETAPA2:g}_margen{MARGEN:g}_batch{BATCH}"
            f"_epocas{EPOCAS_MAX}_paciencia{PACIENCIA}")


def guardar(ruta, df):
    if Path(ruta).exists():
        df = df.reindex(columns=pd.read_csv(ruta, nrows=0).columns)
    df.to_csv(ruta, mode="a", header=not Path(ruta).exists(), index=False)


# %% [code] Experimento A: base congelada
def caracteristicas(tabla, entrada):
    """Características de EfficientNet-B0 congelada, promedio de la foto y su espejo."""
    ruta = RUTA_SALIDA / f"caracteristicas_{entrada}_margen{MARGEN:g}.npy"
    if ruta.exists():
        return np.load(ruta)
    entrada_k = keras.Input((TAM, TAM, 3))
    base = base_efficientnet(entrada_k)
    extractor = keras.Model(entrada_k, layers.GlobalAveragePooling2D()(base.output))
    fuentes = preparar_fuentes(tabla, entrada)
    F = []
    for i in range(0, len(fuentes), 32):
        bloque = fuentes[i:i + 32]
        F.append((extractor.predict(matriz(bloque), verbose=0) + extractor.predict(matriz(bloque, True), verbose=0)) / 2)
    F = np.concatenate(F).astype("float32")
    np.save(ruta, F)
    return F


def experimento_a(tabla, vacas_cv):
    ruta = RUTA_SALIDA / "predicciones_A.csv"
    cfg = config_a()
    tcv = tabla[tabla.vaca.isin(vacas_cv.index)].reset_index(drop=True)
    for entrada in ENTRADAS:
        F_todo = caracteristicas(tabla, entrada)
        F = F_todo[tabla.vaca.isin(vacas_cv.index).to_numpy()]
        for semilla in SEMILLAS:
            for p, (v_ent, v_ev) in enumerate(pliegues(vacas_cv, semilla)):
                base_fila = dict(experimento="A", config=cfg, entrada=entrada, semilla=semilla, pliegue=p)
                if entrada == ENTRADAS[0] and not ya_hecho(ruta, config=cfg, cabeza="constante", semilla=semilla, pliegue=p):
                    guardar(ruta, fila_constante(v_ent, v_ev, vacas_cv).assign(**{**base_fila, "entrada": "-", "cabeza": "constante"}))
                v_fit, v_val = separar_val_interna(v_ent, vacas_cv, semilla)
                i_fit, i_val, i_ev = (tcv.vaca.isin(v).to_numpy() for v in (v_fit, v_val, v_ev))
                for cabeza in CABEZAS:
                    if ya_hecho(ruta, config=cfg, entrada=entrada, cabeza=cabeza, semilla=semilla, pliegue=p):
                        continue
                    keras.utils.set_random_seed(semilla + p)
                    m = modelo_sobre_caracteristicas(cabeza, tcv.nivel.to_numpy()[i_fit])
                    datos = (F[i_fit], objetivo(cabeza, tcv[i_fit]), pesos_por_vaca(tcv.vaca[i_fit]))
                    entrenar(m, cabeza, datos, F[i_val], tcv.vaca[i_val].to_numpy(),
                             vacas_cv.bcs, lr=1e-3, etapa=1)
                    r = resumir_por_vaca(cabeza, m.predict(F[i_ev], verbose=0), tcv.vaca[i_ev].to_numpy())
                    guardar(ruta, r.assign(**base_fila, cabeza=cabeza))
                    keras.backend.clear_session()
            print(f"A | {entrada} | semilla {semilla} listo")
    todo = pd.read_csv(ruta)
    return todo[todo.config == cfg].reset_index(drop=True)


# %% [code] Experimento B: ajuste fino en dos etapas
def experimento_b(tabla, vacas_cv):
    ruta = RUTA_SALIDA / "predicciones_B.csv"
    ruta_hist = RUTA_SALIDA / "historial_B.csv"
    cfg = config_b()
    tcv = tabla[tabla.vaca.isin(vacas_cv.index)].reset_index(drop=True)
    for entrada in ENTRADAS_B:
        fuentes = preparar_fuentes(tcv, entrada)
        for semilla in SEMILLAS_B:
            for p, (v_ent, v_ev) in enumerate(pliegues(vacas_cv, semilla)):
                base_fila = dict(experimento="B", config=cfg, entrada=entrada, semilla=semilla, pliegue=p)
                if entrada == ENTRADAS_B[0] and not ya_hecho(ruta, config=cfg, cabeza="constante", semilla=semilla, pliegue=p):
                    guardar(ruta, fila_constante(v_ent, v_ev, vacas_cv).assign(**{**base_fila, "entrada": "-", "cabeza": "constante"}))
                v_fit, v_val = separar_val_interna(v_ent, vacas_cv, semilla)
                i_fit, i_val, i_ev = (np.where(tcv.vaca.isin(v))[0] for v in (v_fit, v_val, v_ev))
                x_val = matriz([fuentes[i] for i in i_val])
                x_ev = matriz([fuentes[i] for i in i_ev])
                x_ev_espejo = x_ev[:, :, ::-1, :].copy()
                for cabeza in CABEZAS:
                    if ya_hecho(ruta, config=cfg, entrada=entrada, cabeza=cabeza, semilla=semilla, pliegue=p):
                        continue
                    t0 = time.time()
                    keras.utils.set_random_seed(semilla + p)
                    t_fit = tcv.iloc[i_fit]
                    lotes = Lotes([fuentes[i] for i in i_fit], objetivo(cabeza, t_fit), pesos_por_vaca(t_fit.vaca),
                                  aumentar=True, semilla=semilla + p)
                    m = modelo_completo(cabeza, t_fit.nivel.to_numpy())
                    args = (x_val, tcv.vaca.iloc[i_val].to_numpy(), vacas_cv.bcs)
                    h1 = entrenar(m, cabeza, lotes, *args, lr=1e-3, etapa=1)
                    descongelar_etapa2(m)
                    h2 = entrenar(m, cabeza, lotes, *args, lr=LR_ETAPA2, etapa=2)
                    salida = (m.predict(x_ev, verbose=0, batch_size=32) + m.predict(x_ev_espejo, verbose=0, batch_size=32)) / 2 \
                        if cabeza in ("softmax", "regresion") else \
                        _promedio_logits(m, x_ev, x_ev_espejo)
                    r = resumir_por_vaca(cabeza, salida, tcv.vaca.iloc[i_ev].to_numpy())
                    guardar(ruta, r.assign(**base_fila, cabeza=cabeza))
                    guardar(ruta_hist, pd.concat([h1, h2]).reset_index(names="epoca").assign(**base_fila, cabeza=cabeza))
                    print(f"B | {entrada} | {cabeza} | semilla {semilla} | pliegue {p} | {time.time() - t0:.0f} s")
                    keras.backend.clear_session()
    todo = pd.read_csv(ruta)
    return todo[todo.config == cfg].reset_index(drop=True)


def _promedio_logits(m, x, x_espejo):
    """Para CORAL y CORN se promedian las probabilidades y se regresan como logits."""
    a = 1 / (1 + np.exp(-m.predict(x, verbose=0, batch_size=32)))
    b = 1 / (1 + np.exp(-m.predict(x_espejo, verbose=0, batch_size=32)))
    p = np.clip((a + b) / 2, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# %% [md]
# ## 7. Análisis
#
# Todas las métricas son por vaca. Para cada combinación se promedia sobre semillas.
# `mejora_vs_constante` es cuánto baja el MAE frente a la constante, con intervalo bootstrap pareado sobre vacas.

# %% [code] Métricas
def agregar_real(pred, vacas_cv):
    pred = pred.copy()
    pred["bcs_real"] = vacas_cv.bcs.loc[pred.vaca].to_numpy()
    pred["error"] = (pred.bcs_pred - pred.bcs_real).abs()
    pred["error_cont"] = (pred.bcs_cont - pred.bcs_real).abs()
    return pred


def metricas(g):
    real, pred = g.bcs_real.to_numpy(), g.bcs_pred.to_numpy()
    ext = (real <= 2.50) | (real >= 3.50)
    baja = real <= 2.75
    auc = roc_auc_score(baja, g.p_baja) if g.p_baja.notna().all() and 0 < baja.sum() < len(baja) else np.nan
    return pd.Series({
        "MAE": g.error.mean(),
        "exacto": (g.error < 1e-9).mean(),
        "dentro_025": (g.error <= 0.25 + 1e-9).mean(),
        "dentro_050": (g.error <= 0.50 + 1e-9).mean(),
        "acierto_3_niveles": (categoria3(pred) == categoria3(real)).mean(),
        "MAE_extremos": g.error[ext].mean(),
        "spearman": spearmanr(g.bcs_cont, real)[0] if g.bcs_cont.nunique() > 1 else 0.0,
        "AUC_baja": auc,
    })


def tabla_resumen(pred):
    por_semilla = pred.groupby(["entrada", "cabeza", "semilla"]).apply(metricas, include_groups=False)
    media = por_semilla.groupby(["entrada", "cabeza"]).mean()
    de = por_semilla.groupby(["entrada", "cabeza"]).std()
    return media.round(3), de.round(3)


def bootstrap_pareado(pred, a, b, n=5000, semilla=0):
    """Diferencia de MAE por vaca (b − a): positiva = a es mejor. a y b son tuplas (entrada, cabeza)."""
    e = pred.groupby(["entrada", "cabeza", "vaca"]).error.mean()
    d = (e.loc[b] - e.loc[a]).dropna().to_numpy()
    rng = np.random.default_rng(semilla)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return round(d.mean(), 3), np.round(np.percentile(boot, [2.5, 97.5]), 3)


def comparaciones(pred, referencia=("-", "constante")):
    filas = []
    for (entrada, cabeza) in pred.groupby(["entrada", "cabeza"]).groups:
        if (entrada, cabeza) == referencia:
            continue
        mejora, ic = bootstrap_pareado(pred, (entrada, cabeza), referencia)
        filas.append({"entrada": entrada, "cabeza": cabeza, "mejora_MAE": mejora, "IC95_bajo": ic[0], "IC95_alto": ic[1]})
    return pd.DataFrame(filas).sort_values("mejora_MAE", ascending=False)


def error_por_nivel(pred):
    return pred.groupby(["entrada", "cabeza", "bcs_real"]).error.mean().unstack().round(2)


def reparto_predicciones(pred, vacas_cv):
    r = pred.groupby(["entrada", "cabeza"]).bcs_pred.value_counts(normalize=True).unstack().fillna(0).round(2)
    r.loc[("real", "vacas"), :] = vacas_cv.bcs.clip(lower=2.5).value_counts(normalize=True).reindex(r.columns).fillna(0).round(2).to_numpy()
    return r


def comparar_configuraciones(ruta, vacas_cv):
    """MAE y dentro de ±0.25 por configuración, para comparar hiperparámetros ya corridos."""
    todo = agregar_real(pd.read_csv(ruta), vacas_cv)
    g = todo.groupby(["config", "entrada", "cabeza", "semilla"]).agg(
        MAE=("error", "mean"), dentro_025=("error", lambda e: (e <= 0.25 + 1e-9).mean()), vacas=("vaca", "nunique"))
    return g.groupby(["config", "entrada", "cabeza"]).agg(
        MAE=("MAE", "mean"), dentro_025=("dentro_025", "mean"), semillas=("vacas", "size")).round(3)


def matriz_confusion(pred, entrada, cabeza):
    g = pred[(pred.entrada == entrada) & (pred.cabeza == cabeza)]
    return pd.crosstab(g.bcs_real, g.bcs_pred, rownames=["real"], colnames=["predicho"])


# %% [code] Gráficas
def graficar(pred, vacas_cv, ruta=RUTA_SALIDA, prefijo="A"):
    import matplotlib.pyplot as plt
    media, de = tabla_resumen(pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    etiquetas = [f"{e} | {c}" for e, c in media.index]
    ax[0].barh(etiquetas, media.MAE, xerr=de.MAE, color="#4c78a8")
    ax[0].axvline(media.loc[("-", "constante"), "MAE"] if ("-", "constante") in media.index else np.nan, color="gray", ls="--")
    ax[0].set_xlabel("MAE por vaca (BCS)")
    ax[0].set_title("Error por vaca, media ± desviación entre semillas")
    ax[1].barh(etiquetas, media.dentro_025, xerr=de.dentro_025, color="#59a14f")
    ax[1].set_xlabel("Vacas dentro de ±0.25")
    ax[1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(ruta / f"{prefijo}_resumen.png", dpi=150)
    plt.show()

    t = error_por_nivel(pred)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    im = ax.imshow(t.to_numpy(), cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(t.shape[1]), [f"{v:.2f}" for v in t.columns])
    ax.set_yticks(range(len(t)), [f"{e} | {c}" for e, c in t.index])
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if not np.isnan(t.iat[i, j]):
                ax.text(j, i, f"{t.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("BCS real")
    ax.set_title("Error medio por nivel real")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(ruta / f"{prefijo}_error_por_nivel.png", dpi=150)
    plt.show()


# %% [md]
# ## 8. Grad-CAM
#
# Muestra qué zona del recorte pesa en la predicción. Sirve para revisar que el modelo mira isquiones,
# base de la cola y ganchos, y no el fondo o marcas de pintura.

# %% [code] Grad-CAM
def grad_cam(modelo, x, cabeza, capa="top_conv"):
    """Mapa de calor para una imagen (224, 224, 3). Para salidas ordinales usa la suma de logits (BCS esperado)."""
    sub = keras.Model(modelo.inputs, [modelo.get_layer(capa).output, modelo.output])
    niveles = np.arange(K, dtype="float32")
    if keras.backend.backend() == "torch":
        import torch
        xt = ops.convert_to_tensor(x[None]).requires_grad_(True)
        mapa, salida = sub(xt, training=False)
        puntaje = (salida * torch.as_tensor(niveles, device=salida.device)).sum() if cabeza == "softmax" else salida.sum()
        grad = torch.autograd.grad(puntaje, mapa)[0]
        mapa, grad = mapa.detach().cpu().numpy(), grad.detach().cpu().numpy()
    else:
        import tensorflow as tf
        xt = tf.convert_to_tensor(x[None])
        with tf.GradientTape() as cinta:
            cinta.watch(xt)
            mapa, salida = sub(xt, training=False)
            puntaje = tf.reduce_sum(salida * niveles) if cabeza == "softmax" else tf.reduce_sum(salida)
        grad = cinta.gradient(puntaje, mapa)
        mapa, grad = mapa.numpy(), grad.numpy()
    pesos = grad.mean(axis=(1, 2))
    cam = np.maximum((mapa * pesos[:, None, None, :]).sum(-1), 0)[0]
    cam = cam / (cam.max() + 1e-8)
    return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((TAM, TAM), Image.BILINEAR)) / 255.0


def modelo_para_inspeccion(tabla, vacas_cv, cabeza="coral", entrada="recorte", semilla=SEMILLAS[0]):
    """Entrena con el pliegue 0 y regresa el modelo y las fotos de las vacas que no vio (nunca las de prueba)."""
    tcv = tabla[tabla.vaca.isin(vacas_cv.index)].reset_index(drop=True)
    fuentes = preparar_fuentes(tcv, entrada)
    v_ent, v_ev = pliegues(vacas_cv, semilla)[0]
    v_fit, v_val = separar_val_interna(v_ent, vacas_cv, semilla)
    i_fit, i_val, i_ev = (np.where(tcv.vaca.isin(v))[0] for v in (v_fit, v_val, v_ev))
    keras.utils.set_random_seed(semilla)
    t_fit = tcv.iloc[i_fit]
    lotes = Lotes([fuentes[i] for i in i_fit], objetivo(cabeza, t_fit), pesos_por_vaca(t_fit.vaca), True, semilla)
    m = modelo_completo(cabeza, t_fit.nivel.to_numpy())
    args = (matriz([fuentes[i] for i in i_val]), tcv.vaca.iloc[i_val].to_numpy(), vacas_cv.bcs)
    entrenar(m, cabeza, lotes, *args, lr=1e-3, etapa=1)
    descongelar_etapa2(m)
    entrenar(m, cabeza, lotes, *args, lr=LR_ETAPA2, etapa=2)
    ev = tcv.iloc[i_ev].reset_index(drop=True)
    return m, [fuentes[i] for i in i_ev], ev


def mostrar_grad_cam(modelo, cabeza, fuentes, titulos, ruta=None):
    import matplotlib.pyplot as plt
    n = len(fuentes)
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3.2))
    for i, (f, t) in enumerate(zip(fuentes, titulos)):
        x = vista(f)
        cam = grad_cam(modelo, x, cabeza)
        ax[i].imshow(x.astype("uint8"))
        ax[i].imshow(cam, cmap="jet", alpha=0.4)
        ax[i].set_title(t, fontsize=9)
        ax[i].axis("off")
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150)
    plt.show()


# %% [md]
# ## 9. Prueba final (una sola vez)
#
# Reentrena con las 56 vacas y evalúa en las 10 de prueba. Con `RUTA_CAJAS_YOLO_PRUEBA` usa las cajas que
# predice YOLO26 (regla: `cow` de mayor confianza, `ass` de mayor confianza con centro dentro, confianza ≥ 0.50).
# Las fotos sin asociación válida se reportan y no entran.

# %% [code] Prueba final
def entrenar_final(tabla, vacas_cv, cabeza, entrada="recorte", semilla=SEMILLAS[0]):
    tcv = tabla[tabla.vaca.isin(vacas_cv.index)].reset_index(drop=True)
    fuentes = preparar_fuentes(tcv, entrada)
    v_fit, v_val = separar_val_interna(vacas_cv.index.to_numpy(), vacas_cv, semilla)
    i_fit, i_val = (np.where(tcv.vaca.isin(v))[0] for v in (v_fit, v_val))
    keras.utils.set_random_seed(semilla)
    t_fit = tcv.iloc[i_fit]
    lotes = Lotes([fuentes[i] for i in i_fit], objetivo(cabeza, t_fit), pesos_por_vaca(t_fit.vaca), True, semilla)
    m = modelo_completo(cabeza, t_fit.nivel.to_numpy())
    args = (matriz([fuentes[i] for i in i_val]), tcv.vaca.iloc[i_val].to_numpy(), vacas_cv.bcs)
    entrenar(m, cabeza, lotes, *args, lr=1e-3, etapa=1)
    descongelar_etapa2(m)
    entrenar(m, cabeza, lotes, *args, lr=LR_ETAPA2, etapa=2)
    return m


def evaluar_prueba(modelo, cabeza, tabla, entrada="recorte"):
    assert EVALUAR_PRUEBA, "Activa EVALUAR_PRUEBA solo para la evaluación final"
    tp = tabla[tabla.conjunto == "test"].reset_index(drop=True).copy()
    if RUTA_CAJAS_YOLO_PRUEBA is not None:
        clases = leer_clases(RUTA_DATASET)
        tp["caja_ass"] = [leer_cajas(Path(RUTA_CAJAS_YOLO_PRUEBA) / (Path(a).stem + ".txt"), clases, umbral_conf=0.50)
                          for a in tp.archivo]
        rechazadas = tp.caja_ass.isna().sum()
        print(f"Fotos rechazadas por falta de asociación válida: {rechazadas} de {len(tp)}")
        tp = tp[tp.caja_ass.notna()].reset_index(drop=True)
    fuentes = preparar_fuentes(tp, entrada)
    x = matriz(fuentes)
    salida = (modelo.predict(x, verbose=0) + modelo.predict(x[:, :, ::-1, :].copy(), verbose=0)) / 2 \
        if cabeza in ("softmax", "regresion") else _promedio_logits(modelo, x, x[:, :, ::-1, :].copy())
    r = resumir_por_vaca(cabeza, salida, tp.vaca.to_numpy())
    r["bcs_real"] = tp.groupby("vaca").bcs.first().loc[r.vaca].to_numpy()
    r["error"] = (r.bcs_pred - r.bcs_real).abs()
    return r
