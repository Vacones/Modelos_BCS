"""Apples-to-apples vs the old EfficientNet-B0 CORAL model: train DINOv2-S + ridge (+linear
calibration) on the 56 train+valid cows, predict the 10 test cows of metadata_bcs_validos.csv.
Usage: python comparar_prueba_10.py DATA_YOLO META WEIGHTS PRUEBA_FINAL_OLD_CSV
"""
import sys, os
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from extraer_features import leer_cajas, recortar, cargar_pesos, SPLIT_DIR
from extraer_externos import extraer

data, meta_csv, w, old_csv = sys.argv[1:5]
torch.set_num_threads(os.cpu_count())
m = cargar_pesos(timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=224), w).eval()
meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
crops, keep = [], []
for i, r in meta.iterrows():
    d = os.path.join(data, SPLIT_DIR[r.split]); b = leer_cajas(os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt"))
    if b is not None:
        crops.append(recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), b)); keep.append(i)
meta = meta.loc[keep].reset_index(drop=True); X = extraer(m, crops); y = meta.bcs.values.astype(float); g = meta.cow_id.values
tr, te = (meta.split != "test").values, (meta.split == "test").values
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]

def fit(Xa, ya, a):
    s = StandardScaler().fit(Xa); r = Ridge(alpha=a).fit(s.transform(Xa), ya); return lambda Z: r.predict(s.transform(Z))

def cow_mae(p, yy, gg):
    d = pd.DataFrame({"g": gg, "p": p, "y": yy}).groupby("g").mean(); return np.abs(d.p - d.y).mean(), d

# choose alpha + calibration with grouped 5-fold OOF on the 56 cows only
best = None
for a in ALPHAS:
    oof = np.zeros(tr.sum())
    for itr, ite in GroupKFold(5).split(X[tr], y[tr], g[tr]):
        oof[ite] = fit(X[tr][itr], y[tr][itr], a)(X[tr][ite])
    e, d = cow_mae(oof, y[tr], g[tr])
    if best is None or e < best[0]:
        best = (e, a, d)
_, a, d = best; ca, cb = np.polyfit(d.p.values, d.y.values, 1)
p = fit(X[tr], y[tr], a)(X[te])
snap = lambda v: np.clip(np.round(np.asarray(v) * 4) / 4, 2.25, 3.75)
_, raw = cow_mae(p, y[te], g[te]); _, cal = cow_mae(ca * p + cb, y[te], g[te])
old = pd.read_csv(old_csv, dtype={"vaca": str}).set_index("vaca")
out = pd.DataFrame({"bcs_real": raw.y, "dino_cont": raw.p.round(3), "dino": snap(raw.p), "dino_cal_cont": cal.p.round(3),
                    "dino_cal": snap(cal.p), "b0_old": old.reindex(raw.index).bcs_pred})
print(f"alpha={a} calib a={ca:.3f} b={cb:.3f}  n_test_cows={len(out)}")
print(out.to_string())
for c in ("dino_cont", "dino", "dino_cal_cont", "dino_cal", "b0_old"):
    e = np.abs(out[c] - out.bcs_real)
    print(f"{c:14s} MAE={e.mean():.3f}  within0.25={(e <= .25 + 1e-9).mean():.2f}  within0.50={(e <= .5 + 1e-9).mean():.2f}  range=[{out[c].min():.2f},{out[c].max():.2f}]")
