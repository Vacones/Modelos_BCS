"""Entrena el modelo final con todas las vacas: características de DINOv2-S y ridge con alfa fijo.
La calibración lineal y el umbral de la alerta de vaca flaca salen de las predicciones por vaca
fuera de pliegue de GroupKFold 5x10.

Uso: python entrenar_final.py DATOS_YOLO METADATA_CSV PESOS_DINOV2 [ALFA]
Escribe en ../modelo: cabeza.joblib, config.json y oof_metricas.json.
"""
import sys, os, json, hashlib, platform, datetime
import numpy as np, pandas as pd, joblib, torch, timm, sklearn
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import bcs_comun as C

AQUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelo")  # las salidas van a ../modelo
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
N_SPLITS, N_REP = 5, 10


def metricas(pc, yc):
    e = np.abs(pc - yc); es = np.abs(np.clip(np.round(pc * 4) / 4, 2.25, 3.75) - yc)
    return {"mae": float(e.mean()), "mae_redondeado": float(es.mean()), "dentro_025": float((es <= .25 + 1e-9).mean()),
            "dentro_050": float((es <= .5 + 1e-9).mean()), "pendiente": float(np.polyfit(yc, pc, 1)[0]),
            "mae_vacas_ge350": float(e[yc >= 3.5].mean()), "n_ge350": int((yc >= 3.5).sum()),
            "mae_vacas_le250": float(e[yc <= 2.5].mean()), "n_le250": int((yc <= 2.5).sum()),
            "sesgo": float((pc - yc).mean())}


def main():
    data, meta_csv, pesos = sys.argv[1:4]; alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    torch.set_num_threads(os.cpu_count())
    meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    crops, keep = [], []
    for i, r in meta.iterrows():
        d = os.path.join(data, SPLIT_DIR[r.split]); lab = os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")
        sel = C.elegir_grupa(C.leer_txt_yolo(lab)) if os.path.exists(lab) else None
        if sel is not None:
            crops.append(C.recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), sel[1])); keep.append(i)
    meta = meta.loc[keep].reset_index(drop=True)
    X = C.features(C.cargar_dinov2(pesos), crops); y = meta.bcs.values.astype(float)
    g = meta.cow_id.values
    cows = np.unique(g); yc = pd.Series(y).groupby(g).first().reindex(cows).values
    nueva = lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    # predicciones por vaca fuera de pliegue en cada una de las 10 repeticiones
    P = np.zeros((N_REP, len(cows)))
    for r in range(N_REP):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, g):
            p[te] = nueva().fit(X[tr], y[tr]).predict(X[te])
        P[r] = pd.Series(p).groupby(g).mean().reindex(cows).values
    pc = P.mean(0)
    # calibración y = a * pred + b sobre las predicciones de cada repetición apiladas. El modelo que se usa
    # es uno solo y no un promedio de 10, y promediar repeticiones encogería la dispersión e inflaría a
    a, b = np.polyfit(P.ravel(), np.tile(yc, N_REP), 1)
    pcal_rep = a * P + b  # para el umbral de la alerta, con todas las repeticiones juntas
    # métricas por repetición, cada una es una pasada completa fuera de pliegue, y luego su promedio
    rep_raw = pd.DataFrame([metricas(P[r], yc) for r in range(N_REP)]).mean().to_dict()
    rep_cal = pd.DataFrame([metricas(a * P[r] + b, yc) for r in range(N_REP)]).mean().to_dict()

    # alerta de vaca flaca: pred_cal <= T. Se elige el T con mejor especificidad entre los de sensibilidad >= 0.8
    flaca = np.tile(yc <= 2.75, N_REP); cand = []
    for T in np.arange(2.60, 3.21, 0.01):
        al = pcal_rep.ravel() <= T + 1e-9
        sens, spec = (al & flaca).sum() / flaca.sum(), (~al & ~flaca).sum() / (~flaca).sum()
        cand.append((round(float(T), 2), float(sens), float(spec)))
    ok = [c for c in cand if c[1] >= 0.8]
    T, sens, spec = max(ok, key=lambda c: (c[2], c[1])) if ok else max(cand, key=lambda c: c[1] + c[2])

    final = nueva().fit(X, y)
    joblib.dump(final, os.path.join(AQUI, "cabeza.joblib"))
    cfg = {"creado": datetime.date.today().isoformat(),
           "backbone": "timm vit_small_patch14_dinov2.lvd142m (DINOv2 ViT-S/14), congelado",
           "pesos_backbone": "pesos/dinov2_vits14.safetensors",
           "entrada": {"recorte": "caja 'ass' (grupa) dentro de la caja 'cow' de mayor confianza, margen 10%", "margen": C.MARGEN,
                       "tamano": C.SIZE, "normalizacion": {"mean": C.MEAN, "std": C.STD}, "redimension": "bicubica a 224x224"},
           "features": "concat(token CLS, media de tokens de parche) = 768, promedio con la imagen espejada (flip TTA)",
           "cabeza": {"tipo": "StandardScaler + Ridge", "alpha": alpha, "peso_por_foto": "1 (igual que en la CV)",
                      "entrenada_con": {"fotos": int(len(y)), "vacas": int(len(cows))}},
           "agregacion": "media de las predicciones de las fotos de la vaca",
           "calibracion": {"a": float(a), "b": float(b), "formula": "bcs_cal = a * media_fotos + b"},
           "redondeo": {"paso": 0.25, "min": 2.25, "max": 3.75},
           "alerta_flaca": {"umbral": T, "regla": "alerta si bcs_continuo calibrado <= umbral", "objetivo": "BCS <= 2.75",
                            "sensibilidad_oof": sens, "especificidad_oof": spec},
           "rechazo": {"conf_min_grupa": 0.25},
           "versiones": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__}}
    json.dump(cfg, open(os.path.join(AQUI, "config.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    res = {"protocolo": "GroupKFold por vaca 5x10, alpha fijo; metricas por vaca promediadas sobre las 10 repeticiones",
           "sin_calibrar": rep_raw, "calibrado": rep_cal,
           "nota_calibrado": "a y b se ajustaron sobre estas mismas predicciones fuera de pliegue, por eso las métricas calibradas "
                             "son algo optimistas. La estimación anidada (evaluar_externos.py, config e_ridge_calib) fue MAE 0.205.",
           "alerta_flaca": {"umbral": T, "sensibilidad": sens, "especificidad": spec, "n_flacas": int((yc <= 2.75).sum()),
                            "curva": [c for c in cand if c[0] in np.round(np.arange(2.7, 3.11, 0.05), 2)]}}
    json.dump(res, open(os.path.join(AQUI, "oof_metricas.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)); print(f"a={a:.4f} b={b:.4f} T={T}")


if __name__ == "__main__":
    main()
