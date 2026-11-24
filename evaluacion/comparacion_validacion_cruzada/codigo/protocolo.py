"""Protocolo común para reentrenar los modelos del equipo con las mismas fotos, vacas y pliegues.

Usa las mismas filas y el mismo orden que evaluar_sistema.py del modelo final: fotos con
metadata_status = ok y con caja de grupa anotada, elegida con la misma regla de bcs_comun.py.
La repetición r usa GroupKFold(5, shuffle=True, random_state=r) por vaca, igual que el modelo final.

Rutas, por variables de entorno:
  BCS_DATOS     carpeta del export YOLO de Roboflow "Vacones BCS - Cajas v2 Aprobado"
  BCS_METADATA  CSV de metadatos (split, file_name, cow_id, bcs, metadata_status)
  BCS_PRIVADO   carpeta para predicciones por vaca y archivos temporales. Por defecto ../privado,
                que está en .gitignore. Todo lo que se escribe ahí tiene identificadores reales.
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPO = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))
PRIV = os.environ.get("BCS_PRIVADO", os.path.join(AQUI, "..", "privado"))
PARC = os.path.join(PRIV, "parciales")
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
ID_ASS, ID_COW = 0, 1  # export de Roboflow "Cajas v2": names ['ass', 'cow']
N_PLIEGUES = 5


def ruta_entorno(nombre):
    valor = os.environ.get(nombre)
    if not valor:
        raise SystemExit(f"Falta la variable de entorno {nombre}. Revisa el README de esta carpeta.")
    return valor


def leer_txt_yolo(ruta):
    out = []
    for ln in open(ruta):
        p = ln.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), float(p[5]) if len(p) > 5 else 1.0, tuple(map(float, p[1:5]))))
    return out


def elegir_grupa(cajas):
    """Copia de elegir_grupa de bcs_comun.py. Está aquí para no depender de PyTorch en el entorno de TensorFlow."""
    cows = [(c, b) for k, c, b in cajas if k == ID_COW]
    asses = [(c, b) for k, c, b in cajas if k == ID_ASS]
    if not asses:
        return None
    if not cows:
        return max(asses, key=lambda t: (t[0], t[1][2] * t[1][3]))
    _, (vx, vy, vw, vh) = max(cows, key=lambda t: (t[0], t[1][2] * t[1][3]))
    dentro = [a for a in asses if abs(a[1][0] - vx) <= vw / 2 and abs(a[1][1] - vy) <= vh / 2]
    return max(dentro, key=lambda t: (t[0], t[1][2] * t[1][3])) if dentro else None


def tabla(rep):
    """Una fila por foto, con su vaca, su BCS, sus cajas y el pliegue de prueba en la repetición rep."""
    datos = ruta_entorno("BCS_DATOS")
    meta = pd.read_csv(ruta_entorno("BCS_METADATA"), dtype={"cow_id": str})
    meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    filas = []
    for _, r in meta.iterrows():
        d = os.path.join(datos, SPLIT_DIR[r.split])
        lab = os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")
        cajas = leer_txt_yolo(lab)
        if elegir_grupa(cajas) is None:
            continue
        filas.append({"file_name": r.file_name, "split": r.split, "imagen": os.path.join(d, "images", r.file_name),
                      "label": lab, "cow": r.cow_id, "bcs": float(r.bcs), "cajas": cajas})
    df = pd.DataFrame(filas)
    df["pliegue"] = -1
    grupos = GroupKFold(N_PLIEGUES, shuffle=True, random_state=rep).split(np.zeros(len(df)), df.bcs.values, df.cow.values)
    for f, (_, te) in enumerate(grupos):
        df.loc[te, "pliegue"] = f
    return df


def separar_val(df, idx_tr, rep):
    """Separa el 20 % de las vacas de entrenamiento para la parada temprana o la elección del checkpoint."""
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rep).split(idx_tr, groups=df.cow.values[idx_tr]))
    return np.asarray(idx_tr)[a], np.asarray(idx_tr)[b]


def por_vaca(df, pred_foto):
    """Media de las predicciones de las fotos de cada vaca, con las vacas en orden."""
    cows = np.unique(df.cow.values)
    ycow = df.groupby("cow").bcs.first().reindex(cows).values
    P = pd.Series(np.asarray(pred_foto, float)).groupby(df.cow.values).mean().reindex(cows).values
    return P[None, :], ycow, cows


def guardar_rep(nombre, rep, df, pred_foto):
    """Guarda la predicción por vaca de una repetición en BCS_PRIVADO/parciales/<nombre>_rep<r>.npy."""
    os.makedirs(PARC, exist_ok=True)
    P, ycow, cows = por_vaca(df, pred_foto)
    ref = np.load(os.path.join(PRIV, "final_cajas_anotadas.npz"), allow_pickle=True)
    assert np.array_equal(ref["cows"].astype(str), cows.astype(str)), "orden de vacas distinto al del modelo final"
    assert np.allclose(ref["ycow"], ycow), "BCS por vaca distinto al del modelo final"
    np.save(os.path.join(PARC, f"{nombre}_rep{rep}.npy"), P[0])
    return P[0]


def ensamblar(nombre, n_rep=10):
    """Une las repeticiones en BCS_PRIVADO/<nombre>.npz con P (n_rep, vacas), en el orden de vacas del modelo final."""
    ref = np.load(os.path.join(PRIV, "final_cajas_anotadas.npz"), allow_pickle=True)
    P = np.vstack([np.load(os.path.join(PARC, f"{nombre}_rep{r}.npy")) for r in range(n_rep)])
    np.savez(os.path.join(PRIV, nombre + ".npz"), P=P, ycow=ref["ycow"], cows=ref["cows"])
    return P


def correr_repeticiones(nombre, correr, reps):
    """Corre las repeticiones que falten, una por una, y al final arma el .npz si ya están todas."""
    for rep in reps:
        if os.path.exists(os.path.join(PARC, f"{nombre}_rep{rep}.npy")):
            print("ya existe la repetición", rep, flush=True)
            continue
        correr(rep)
    try:
        P = ensamblar(nombre)
        print("ensamblado", nombre, P.shape, flush=True)
    except FileNotFoundError as e:
        print("aún faltan repeticiones:", e, flush=True)
