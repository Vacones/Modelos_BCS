"""(1) Check that in the Roboflow export class 0 = ass (rump, small, inside) and 1 = cow (big).
(2) Paired per-cow comparison old EfficientNet-B0 CORAL (BCS_ordinal CV predictions, 56 cows)
    vs DINOv2-S + ridge (+calibration fitted inside the fold) with a comparable protocol
    (5-fold GroupKFold on the same 56 train+valid cows, 10 repeats).
Usage: python verificar_y_comparar.py DATA_YOLO META PESOS PRED_B_CSV PRED_A_CSV
"""
import sys, os, glob, json
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcs_comun as C  # noqa: E402

data, meta_csv, pesos, pb, pa = sys.argv[1:6]
# ---- (1) class geometry
n, ok_small, ok_inside = 0, 0, 0
for f in glob.glob(os.path.join(data, "*", "labels", "*.txt")):
    bx = C.leer_txt_yolo(f); a = [b for k, _, b in bx if k == 0]; c = [b for k, _, b in bx if k == 1]
    if a and c:
        n += 1; ok_small += a[0][2] * a[0][3] < c[0][2] * c[0][3]
        ok_inside += abs(a[0][0] - c[0][0]) <= c[0][2] / 2 and abs(a[0][1] - c[0][1]) <= c[0][3] / 2
print(f"[clases] archivos con ambas cajas: {n}; clase0 mas chica que clase1: {ok_small}; centro clase0 dentro de clase1: {ok_inside}")

# ---- (2) paired comparison
meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
SPLIT = {"train": "train", "val": "valid", "test": "test"}
crops = []
for _, r in meta.iterrows():
    d = os.path.join(data, SPLIT[r.split]); sel = C.elegir_grupa(C.leer_txt_yolo(os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")))
    crops.append(C.recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), sel[1]))
torch.set_num_threads(os.cpu_count())
X = C.features(C.cargar_dinov2(pesos), crops); y = meta.bcs.values.astype(float); g = meta.cow_id.values
m56 = (meta.split != "test").values
X, y, g = X[m56], y[m56], g[m56]
cows = np.unique(g); yc = pd.Series(y).groupby(g).first().reindex(cows).values
nueva = lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))
R = 10; Praw = np.zeros((R, len(cows))); Pcal = np.zeros((R, len(cows)))
for r in range(R):
    for tr, te in GroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
        mod = nueva().fit(X[tr], y[tr]); oof = np.zeros(len(tr))
        for itr, ite in GroupKFold(5).split(X[tr], y[tr], g[tr]):
            oof[ite] = nueva().fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite])
        d = pd.DataFrame({"g": g[tr], "p": oof, "y": y[tr]}).groupby("g").mean(); a, b = np.polyfit(d.p, d.y, 1)
        p = pd.Series(mod.predict(X[te])).groupby(g[te]).mean(); idx = np.searchsorted(cows, p.index.values)
        Praw[r, idx] = p.values; Pcal[r, idx] = a * p.values + b
snap = lambda v: np.clip(np.round(v * 4) / 4, 2.25, 3.75)
dino = {"dino_raw_cont": np.abs(Praw - yc).mean(0), "dino_cal_cont": np.abs(Pcal - yc).mean(0),
        "dino_raw_snap": np.abs(snap(Praw) - yc).mean(0), "dino_cal_snap": np.abs(snap(Pcal) - yc).mean(0)}

old = {}
for path, exp in ((pb, "B"), (pa, "A")):
    df = pd.read_csv(path, dtype={"vaca": str})
    df = df[df.cabeza != "constante"]
    print(f"[{exp}] configs:", df.groupby(["config", "entrada", "cabeza"]).size().to_dict() if exp == "B" else df.groupby(["entrada", "cabeza"]).size().to_dict())
    for keys, dd in df.groupby(["config", "entrada", "cabeza"]):
        per = dd.groupby("vaca").apply(lambda t: pd.Series({"snap": np.abs(t.bcs_pred - t.bcs_pred * 0 + 0).mean(), "pred": t.bcs_pred.mean(), "cont": t.bcs_cont.mean()}))
        e_snap = dd.assign(e=np.abs(dd.bcs_pred - pd.Series(yc, cows).reindex(dd.vaca).values)).groupby("vaca").e.mean()
        e_cont = dd.assign(e=np.abs(dd.bcs_cont - pd.Series(yc, cows).reindex(dd.vaca).values)).groupby("vaca").e.mean()
        old[f"{exp}|{keys[1]}|{keys[2]}|{keys[0][:40]}"] = (e_snap.reindex(cows).values, e_cont.reindex(cows).values)

rng = np.random.default_rng(0)
def boot(v):
    v = v[~np.isnan(v)]; bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
const = np.abs(yc - 3.0)
out = {"n_vacas": int(len(cows)), "constante_3.00": boot(const)}
out.update({k: boot(v) for k, v in dino.items()})
for k, (es, ec) in old.items():
    if np.isnan(es).all():
        continue
    out[k] = {"mae_snap": boot(es), "mae_cont": boot(ec), "n": int((~np.isnan(es)).sum()),
              "delta_dino_cal_snap_menos_viejo_snap": boot(dino["dino_cal_snap"] - es),
              "delta_dino_raw_cont_menos_viejo_cont": boot(dino["dino_raw_cont"] - ec),
              "vacas_dino_mejor_igual_peor": [int(((dino["dino_cal_snap"] - es) < -1e-9).sum()), int((np.abs(dino["dino_cal_snap"] - es) <= 1e-9).sum()), int(((dino["dino_cal_snap"] - es) > 1e-9).sum())]}
print(json.dumps(out, indent=1))
json.dump(out, open("comparacion_b0_vs_dino.json", "w"), indent=1)
