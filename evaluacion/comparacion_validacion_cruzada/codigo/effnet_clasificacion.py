"""YOLO + EfficientNet-B0 en clasificación (modelos/con_cajas_yolo, TensorFlow) con el protocolo común de 66 vacas.

Llama a vacas.tf_model.train de esa carpeta sin cambios: GPU obligatoria, semilla 42, aumentos, pesos de clase,
checkpoint por val_loss y EarlyStopping con paciencia 8. La receta es la de su README: cabeza 30 épocas con lr 1e-3
y lote 4, luego ajuste fino de 20 épocas con lr 1e-5 en las últimas 20 capas con BatchNorm congelada. Siempre se
usa el best.keras del ajuste fino.
El recorte es vacas.common.crop_image con la caja anotada de la grupa, sin margen y con letterbox a 224.
El manifiesto se arma aquí y no con el ETL de esa carpeta, para usar el reparto común. Su 'train' son las vacas de
entrenamiento del pliegue menos el 20 % que va a 'val' (protocolo.separar_val).
El catálogo de BCS son las clases presentes en 'train', que es la regla de esa carpeta. Las fotos de 'val' con un
BCS fuera del catálogo no se pueden etiquetar y se quitan de 'val'. Esto pasa con la única vaca de 2.25.
La predicción por foto es la clase de mayor probabilidad y la de la vaca es la media de sus fotos.

Uso: python effnet_clasificacion.py [REPETICIONES...]   (por defecto de 0 a 9)
"""
import os, sys, time, json, argparse
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import protocolo as PR  # noqa: E402
sys.path.insert(0, os.path.join(PR.RAIZ_REPO, "modelos", "con_cajas_yolo"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from vacas.common import crop_image, save_csv, save_json, sha256  # noqa: E402
from vacas import tf_model as TM  # noqa: E402

NOMBRE = "effnet_clasificacion"
BASE = os.path.join(PR.PRIV, "tmp_effnet_clasificacion")
CAMPOS = ["split", "image_id", "file_name", "cow_id", "bcs", "status", "crop_path"]


def recortes(df):
    out = []
    d = os.path.join(BASE, "crops"); os.makedirs(d, exist_ok=True)
    for i, r in enumerate(df.itertuples()):
        dst = os.path.join(d, f"{i:04d}.png"); out.append(dst)
        if os.path.exists(dst):
            continue
        ass = [b for k, _, b in r.cajas if k == PR.ID_ASS]
        assert len(ass) == 1, r.file_name
        with Image.open(r.imagen) as im:
            W, H = im.size; cx, cy, w, h = ass[0]
            crop_image(im, [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 224).save(dst)
    return out


def correr(rep):
    t0 = time.time()
    df = PR.tabla(rep); crops = recortes(df)
    pred = np.full(len(df), np.nan); info = []
    for f in range(PR.N_PLIEGUES):
        tf_ = time.time()
        te = np.where(df.pliegue.values == f)[0]; tr = np.where(df.pliegue.values != f)[0]
        fit, val = PR.separar_val(df, tr, rep)
        clases = sorted({float(v) for v in df.bcs.values[fit]})
        val_ok = [i for i in val if float(df.bcs.values[i]) in clases]
        work = os.path.join(BASE, f"rep{rep}", f"pliegue{f}", "work"); os.makedirs(work, exist_ok=True)
        filas = [{"split": s, "image_id": str(i), "file_name": df.file_name.values[i], "cow_id": df.cow.values[i],
                  "bcs": float(df.bcs.values[i]), "status": "ok", "crop_path": crops[i]}
                 for s, idx in (("train", fit), ("val", val_ok)) for i in idx]
        save_csv(os.path.join(work, "manifest.csv"), filas, CAMPOS)
        save_json(os.path.join(work, "etl.json"), {"size": 224, "classes": clases,
                                                    "manifest_sha256": sha256(os.path.join(work, "manifest.csv"))})
        comun = dict(work=work, weights="imagenet", batch=4, fine_layers=20, patience=8, resume=False, mixed=False, allow_cpu=False)
        head = os.path.join(BASE, f"rep{rep}", f"pliegue{f}", "head"); fine = os.path.join(BASE, f"rep{rep}", f"pliegue{f}", "finetune")
        TM.train(argparse.Namespace(run=head, phase="head", init=None, epochs=30, lr=1e-3, **comun))
        TM.train(argparse.Namespace(run=fine, phase="finetune", init=os.path.join(head, "best.keras"), epochs=20, lr=1e-5, **comun))
        tf = TM.tensorflow_ready(False)
        model = tf.keras.models.load_model(os.path.join(fine, "best.keras"))
        sel = [{"crop_path": crops[i], "bcs": clases[0]} for i in te]  # etiqueta de relleno, no se usa al predecir
        prob = model.predict(TM.dataset(tf, sel, clases, 224, 4).map(lambda x, y: x), verbose=0)
        pred[te] = np.asarray(clases)[prob.argmax(1)]
        bh = json.load(open(os.path.join(head, "best_epoch.json"))); bf = json.load(open(os.path.join(fine, "best_epoch.json")))
        info.append({"pliegue": f, "clases": clases, "fotos_val_excluidas": int(len(val) - len(val_ok)),
                     "mejor_epoca_head": bh["epoch"], "mejor_epoca_finetune": bf["epoch"],
                     "epocas_head": len([p for p in os.listdir(os.path.join(head, "checkpoints")) if p.endswith(".done.json")]),
                     "epocas_finetune": len([p for p in os.listdir(os.path.join(fine, "checkpoints")) if p.endswith(".done.json")]),
                     "gpu": [d.name for d in tf.config.list_physical_devices("GPU")], "segundos": time.time() - tf_})
        print(json.dumps(info[-1]), flush=True)
        tf.keras.backend.clear_session()
    assert not np.isnan(pred).any()
    P = PR.guardar_rep(NOMBRE, rep, df, pred)
    mae = float(np.abs(P - PR.por_vaca(df, pred)[1]).mean())
    json.dump({"rep": rep, "pliegues": info, "segundos_total": time.time() - t0, "mae_vaca": mae},
              open(os.path.join(PR.PARC, f"{NOMBRE}_rep{rep}_log.json"), "w"), indent=1)
    print("repetición", rep, "MAE por vaca", round(mae, 4), "segundos", round(time.time() - t0))


if __name__ == "__main__":
    PR.correr_repeticiones(NOMBRE, correr, [int(a) for a in sys.argv[1:]] or list(range(10)))
