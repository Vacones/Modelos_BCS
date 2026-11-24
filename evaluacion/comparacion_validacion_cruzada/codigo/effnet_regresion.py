"""YOLO + EfficientNet-B0 en regresión (modelos/con_cajas_yolo_regresion) con el protocolo común de 66 vacas.

Reproduce las celdas 23 a 38 del notebook entrenar_yolo_regresion_v2_mac.ipynb sin cambiar la receta: recorte de la
grupa anotada (recortar_grupa, letterbox a 224), tf.data con barajado de semilla 42, lote 8, aumentos RandomFlip,
RandomRotation 0.03 y RandomZoom 0.05, Dropout 0.2, salida lineal y pérdida MAE. Cabeza de 60 épocas con lr 3e-4 y
ajuste fino de 80 épocas con lr 5e-6 en las últimas 40 capas sin BatchNorm, sin parada temprana, con
ReduceLROnPlateau(val_mae, 0.5, paciencia 8, min_delta 1e-4, mínimo 1e-7). Se guarda best.keras por val_mae y se
elige la fase con menor val_mae. 'val' es el 20 % de las vacas de entrenamiento del pliegue (protocolo.separar_val).
La única diferencia con el notebook es el entorno: TensorFlow 2.20 con CUDA en lugar de 2.18 con Metal.
La predicción por foto es el valor continuo y la de la vaca es la media de sus fotos. Necesita GPU.

Uso: python effnet_regresion.py [REPETICIONES...]   (por defecto de 0 a 9)
"""
import os, sys, time, json, math
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import protocolo as PR  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

NOMBRE = "effnet_regresion"
SEMILLA = 42
TAM_RECORTE = 224
LOTE_REGRESOR = 8
EPOCAS_CABEZA, EPOCAS_AJUSTE = 60, 80
LR_CABEZA, LR_AJUSTE = 3e-4, 5e-6
CAPAS_AJUSTE = 40
FACTOR_REDUCCION_LR, PACIENCIA_LR, LR_MINIMA = 0.5, 8, 1e-7
PESOS_EFFICIENTNET = "imagenet"
BASE = os.path.join(PR.PRIV, "tmp_effnet_regresion")


def recortar_grupa(imagen, caja):
    x1, y1, x2, y2 = caja
    limites = (max(0, math.floor(x1)), max(0, math.floor(y1)),
               min(imagen.width, math.ceil(x2)), min(imagen.height, math.ceil(y2)))
    assert limites[2] > limites[0] and limites[3] > limites[1], "Recorte vacío."
    region = imagen.convert("RGB").crop(limites)
    return ImageOps.pad(region, (TAM_RECORTE, TAM_RECORTE), method=Image.Resampling.BILINEAR, color=(0, 0, 0))


def recortes(df):
    d = os.path.join(BASE, "recortes_gt"); os.makedirs(d, exist_ok=True); out = []
    for i, r in enumerate(df.itertuples()):
        dst = os.path.join(d, f"{i:04d}.png"); out.append(dst)
        if os.path.exists(dst):
            continue
        ass = [b for k, _, b in r.cajas if k == PR.ID_ASS]
        assert len(ass) == 1
        with Image.open(r.imagen) as foto:
            W, H = foto.size; cx, cy, w, h = ass[0]
            recortar_grupa(foto, [max(0, (cx - w / 2) * W), max(0, (cy - h / 2) * H),
                                  min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)]).save(dst)
    return out



def correr(rep):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    gpus = tf.config.list_physical_devices('GPU')
    assert gpus, 'TensorFlow no detecta la GPU.'
    t0 = time.time()
    df = PR.tabla(rep); rec = np.asarray(recortes(df))
    pred = np.full(len(df), np.nan); info = []

    def leer_recorte(ruta, bcs):
        imagen = tf.io.decode_png(tf.io.read_file(ruta), channels=3)
        imagen = tf.ensure_shape(imagen, (TAM_RECORTE, TAM_RECORTE, 3))
        return tf.cast(imagen, tf.float32), tf.cast(bcs, tf.float32)

    def callbacks_fase(carpeta):
        return [keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=FACTOR_REDUCCION_LR,
                                                  patience=PACIENCIA_LR, min_delta=1e-4, min_lr=LR_MINIMA, verbose=1),
                keras.callbacks.ModelCheckpoint(os.path.join(carpeta, "best.keras"), monitor="val_mae", mode="min", save_best_only=True),
                keras.callbacks.CSVLogger(os.path.join(carpeta, "history.csv"), append=False),
                keras.callbacks.TerminateOnNaN()]

    for f in range(PR.N_PLIEGUES):
        tp = time.time()
        te = np.where(df.pliegue.values == f)[0]; tr = np.where(df.pliegue.values != f)[0]
        fit, val = PR.separar_val(df, tr, rep)
        keras.utils.set_random_seed(SEMILLA)
        conjuntos = {}
        for nombre, idx in (("train", fit), ("val", val)):
            c = tf.data.Dataset.from_tensor_slices((rec[idx], df.bcs.values[idx].astype(np.float32).reshape(-1, 1)))
            if nombre == "train":
                c = c.shuffle(len(idx), seed=SEMILLA)
            conjuntos[nombre] = c.map(leer_recorte, num_parallel_calls=tf.data.AUTOTUNE).batch(LOTE_REGRESOR).prefetch(tf.data.AUTOTUNE)
        entrada = keras.Input(shape=(TAM_RECORTE, TAM_RECORTE, 3), name="rgb_0_255")
        aumentos = keras.Sequential([layers.RandomFlip("horizontal", seed=SEMILLA),
                                     layers.RandomRotation(0.03, seed=SEMILLA + 1),
                                     layers.RandomZoom(0.05, seed=SEMILLA + 2)], name="aumentos_entrenamiento")
        base = keras.applications.EfficientNetB0(include_top=False, weights=PESOS_EFFICIENTNET,
                                                 input_shape=(TAM_RECORTE, TAM_RECORTE, 3))
        base.trainable = False
        x = base(aumentos(entrada), training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2, seed=SEMILLA + 3)(x)
        salida_bcs = layers.Dense(1, activation="linear", dtype="float32", name="bcs")(x)
        regresor = keras.Model(entrada, salida_bcs, name="regresor_bcs")
        regresor.compile(optimizer=keras.optimizers.Adam(LR_CABEZA), loss="mae", metrics=["mae"])
        CABEZA = os.path.join(BASE, f"rep{rep}", f"pliegue{f}", "cabeza"); AJUSTE = os.path.join(BASE, f"rep{rep}", f"pliegue{f}", "ajuste_fino")
        os.makedirs(CABEZA, exist_ok=True); os.makedirs(AJUSTE, exist_ok=True)
        regresor.fit(conjuntos["train"], validation_data=conjuntos["val"], epochs=EPOCAS_CABEZA,
                     callbacks=callbacks_fase(CABEZA), verbose=2)
        regresor = keras.models.load_model(os.path.join(CABEZA, "best.keras"))
        base = regresor.get_layer("efficientnetb0"); base.trainable = True
        for indice, capa in enumerate(base.layers):
            capa.trainable = (indice >= len(base.layers) - CAPAS_AJUSTE and not isinstance(capa, layers.BatchNormalization))
        regresor.compile(optimizer=keras.optimizers.Adam(LR_AJUSTE), loss="mae", metrics=["mae"])
        regresor.fit(conjuntos["train"], validation_data=conjuntos["val"], epochs=EPOCAS_AJUSTE,
                     callbacks=callbacks_fase(AJUSTE), verbose=2)
        hist = {n: pd.read_csv(os.path.join(c, "history.csv")) for n, c in (("cabeza", CABEZA), ("ajuste_fino", AJUSTE))}
        mejores = {n: float(h.val_mae.min()) for n, h in hist.items()}
        fase = min(mejores, key=mejores.get)
        regresor = keras.models.load_model(os.path.join(CABEZA if fase == "cabeza" else AJUSTE, "best.keras"))
        lote = np.stack([np.asarray(Image.open(p), dtype=np.float32) for p in rec[te]])
        pred[te] = regresor.predict(lote, verbose=0, batch_size=1).reshape(-1)
        info.append({"pliegue": f, "fase_elegida": fase, "val_mae": mejores,
                     "mejor_epoca": {n: int(h.epoch[h.val_mae.idxmin()]) + 1 for n, h in hist.items()},
                     "gpu": [g.name for g in gpus], "segundos": time.time() - tp})
        print(json.dumps(info[-1]), flush=True)
        keras.backend.clear_session()
    assert not np.isnan(pred).any()
    P = PR.guardar_rep(NOMBRE, rep, df, pred)
    mae = float(np.abs(P - PR.por_vaca(df, pred)[1]).mean())
    json.dump({"rep": rep, "tensorflow": tf.__version__, "pliegues": info, "segundos_total": time.time() - t0, "mae_vaca": mae},
              open(os.path.join(PR.PARC, f"{NOMBRE}_rep{rep}_log.json"), "w"), indent=1)
    print("repetición", rep, "MAE por vaca", round(mae, 4), "segundos", round(time.time() - t0))


if __name__ == "__main__":
    PR.correr_repeticiones(NOMBRE, correr, [int(a) for a in sys.argv[1:]] or list(range(10)))
