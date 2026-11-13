"""CHECK 1: our final recipe vs the team's "DINOv2 + LDL" head, same folds, same cows.

Configs (all GroupKFold by cow 5x10, identical splits, annotated rump boxes, per-cow metrics):
  a  ours: 1 view (rump +10% margin, square resize, flip TTA), ridge alpha 100
  b  ridge on the team's 2-view features (1536), alpha by inner grouped 3-fold CV
  c  team LDL MLP head on the team's 2-view features, 3 seeds
  d  team LDL MLP head on our 1-view features, 3 seeds
  e  ridge alpha 100 on the team's full-crop view only (768), isolates crop/preprocessing
Team recipe (repo Vacones/Modelos_BCS, modelos/con_cajas_yolo/dinov2_ldl_mac): exact rump box, no margin,
letterbox (ImageOps.pad, bilinear) to 224; second view = top 65% of that 224 image padded to 224 (DEFAULT_ROI);
CLS + mean patch token (final norm) per view; standardize with train mean/std; LayerNorm-Dense128-GELU-
Dropout0.3-Dense(7); loss CE with Gaussian soft labels (sigma .125) + 0.5 * mean squared CDF difference;
sample weight 1/photos per cow (normalized to mean 1); AdamW lr 1e-3 wd 1e-4, batch 32, <=100 epochs,
patience 15; prediction = expected value. Their checkpoint was chosen on validation cows, so here the
epoch is chosen honestly: 20% of the TRAINING cows (grouped) are held out for early stopping on per-cow MAE.

Usage: python comparar_ldl.py DATA_YOLO META PESOS OUT_JSON
"""
import sys, os, json, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import bcs_comun as C  # noqa: E402
import evaluar_cabezas as EC  # noqa: E402

ALIAS = {"2107": "1548"}; SPLIT = {"train": "train", "val": "valid", "test": "test"}
CLASSES = np.array([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75], np.float32)
N_SPLITS, N_REP, SEEDS = 5, 10, (0, 1, 2)
ROI = (0.0, 0.0, 1.0, 0.65)


# ------------------------------------------------------------------ team features
def team_crop(img, b):
    W, H = img.size; cx, cy, w, h = b
    x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(W, math.ceil(x2)), min(H, math.ceil(y2)))
    return ImageOps.pad(img.crop(box), (224, 224), method=Image.Resampling.BILINEAR, color=(0, 0, 0), centering=(.5, .5))


def team_roi(im224):
    w, h = im224.size; x1, y1, x2, y2 = ROI
    box = (math.floor(x1 * w), math.floor(y1 * h), min(w, math.ceil(x2 * w)), min(h, math.ceil(y2 * h)))
    return ImageOps.pad(im224.crop(box), (224, 224), method=Image.Resampling.BILINEAR, color=(0, 0, 0))


@torch.no_grad()
def feats_no_flip(model, ims, bs=16):
    def tens(im):
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
        return (x - torch.tensor(C.MEAN)[:, None, None]) / torch.tensor(C.STD)[:, None, None]
    out = []
    for j in range(0, len(ims), bs):
        t = model.forward_features(torch.stack([tens(i) for i in ims[j:j + bs]]))
        out.append(torch.cat([t[:, 0], t[:, model.num_prefix_tokens:].mean(1)], 1).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ LDL head (PyTorch port of tf_model.py)
class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim, eps=1e-5), nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.30), nn.Linear(128, len(CLASSES)))

    def forward(self, x):
        return self.net(x)


def ldl_loss(logits, idx, sigma=0.125, emd=0.5):
    vals = torch.tensor(CLASSES)
    q = torch.softmax(-0.5 * ((vals[None, :] - vals[idx][:, None]) / sigma) ** 2, -1)
    ce = -(q * torch.log_softmax(logits, -1)).sum(-1)
    p = torch.softmax(logits, -1)
    dist = ((torch.cumsum(p, -1)[:, :-1] - torch.cumsum(q, -1)[:, :-1]) ** 2).mean(-1)
    return ce + emd * dist


def expected(model, X):
    model.eval()
    with torch.no_grad():
        return (torch.softmax(model(torch.from_numpy(X)), -1).numpy() @ CLASSES)


def train_ldl(Xtr, ytr, gtr, seed, max_epochs=100, patience=15, lr=1e-3, wd=1e-4, batch=32):
    """Early stopping on per-cow MAE of 20% held-out TRAINING cows; returns predictor fn."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    itr, iva = next(gss.split(Xtr, ytr, gtr))
    mu = Xtr[itr].mean(0); sd = np.maximum(Xtr[itr].std(0), 1e-6)
    Z = ((Xtr - mu) / sd).astype(np.float32)
    idx = np.abs(ytr[:, None] - CLASSES[None, :]).argmin(1)
    cnt = pd.Series(gtr[itr]).value_counts(); w = np.array([1 / cnt[c] for c in gtr[itr]], np.float32); w /= w.mean()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Head(Z.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt, wt = torch.from_numpy(Z[itr]), torch.from_numpy(idx[itr]), torch.from_numpy(w)
    best, best_state, best_ep, stale = np.inf, None, 0, 0
    for ep in range(max_epochs):
        m.train(); order = rng.permutation(len(itr))
        for o in range(0, len(order), batch):
            b = order[o:o + batch]
            loss = (ldl_loss(m(Xt[b]), yt[b]) * wt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        p = expected(m, Z[iva])
        d = pd.DataFrame({"g": gtr[iva], "p": p, "y": ytr[iva]}).groupby("g").mean(); mae = float(np.abs(d.p - d.y).mean())
        if mae < best - 1e-8:
            best, best_ep, stale = mae, ep + 1, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    m.load_state_dict(best_state)
    return (lambda X: expected(m, ((X - mu) / sd).astype(np.float32))), best_ep


# ------------------------------------------------------------------ ridge helpers
def ridge_fixed(Xtr, ytr, Xte, alpha=100.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr, ytr).predict(Xte)


def ridge_cv(Xtr, ytr, gtr, Xte):
    best = None
    for a in [1e-1, 1, 10, 100, 1e3, 1e4, 1e5]:
        e = []
        for i, j in GroupKFold(3).split(Xtr, ytr, gtr):
            p = ridge_fixed(Xtr[i], ytr[i], Xtr[j], a)
            d = pd.DataFrame({"g": gtr[j], "p": p, "y": ytr[j]}).groupby("g").mean(); e.append(np.abs(d.p - d.y).mean())
        if best is None or np.mean(e) < best[0]:
            best = (np.mean(e), a)
    return ridge_fixed(Xtr, ytr, Xte, best[1]), best[1]


def summarize(pc, ycow, rng, ref_ae=None):
    st = EC.cow_stats(pc, ycow); n = len(ycow)
    r = {k: EC.boot(lambda i, k=k: st[k][i].mean(), n, rng) for k in ("ae", "w25", "w50")}
    r["slope"] = EC.boot(lambda i: EC.slope(st["pred"][i], ycow[i]), n, rng)
    r["auc_le275"] = EC.boot(lambda i: EC.auc(st["pred"][i], ycow[i]), n, rng)
    r["mae_ge350"] = float(st["ae"][ycow >= 3.5].mean()); r["mae_le250"] = float(st["ae"][ycow <= 2.5].mean())
    if ref_ae is not None:
        r["delta_vs_a"] = EC.boot(lambda i: (st["ae"][i] - ref_ae[i]).mean(), n, rng)
    return r, st["ae"]


def main():
    data, meta_csv, pesos, out_json = sys.argv[1:5]
    torch.set_num_threads(os.cpu_count())
    meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    ours, full, roi = [], [], []
    for _, r in meta.iterrows():
        d = os.path.join(data, SPLIT[r.split]); img = Image.open(os.path.join(d, "images", r.file_name)).convert("RGB")
        sel = C.elegir_grupa(C.leer_txt_yolo(os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")))
        ours.append(C.recortar(img, sel[1])); f = team_crop(img, sel[1]); full.append(f); roi.append(team_roi(f))
    m = C.cargar_dinov2(pesos)
    t0 = time.time()
    X1 = C.features(m, ours).astype(np.float32)
    Xf = feats_no_flip(m, full).astype(np.float32); Xr = feats_no_flip(m, roi).astype(np.float32)
    X2 = np.hstack([Xf, Xr])
    print(f"features {time.time() - t0:.0f}s  X1{X1.shape} X2{X2.shape}", flush=True)
    y = meta.bcs.values.astype(np.float32); g = meta.cow_id.map(lambda c: ALIAS.get(c, c)).values
    cows = np.unique(g); ycow = pd.Series(y).groupby(g).first().reindex(cows).values

    keys = ["a_ours_1view_ridge100", "b_ridge_2view_cv", "e_ridge100_teamfull_1view"] + \
           [f"c_ldl_2view_s{s}" for s in SEEDS] + [f"d_ldl_1view_s{s}" for s in SEEDS]
    P = {k: np.zeros((N_REP, len(y))) for k in keys}; info = {"alpha_b": [], "epochs_c": [], "epochs_d": []}
    t0 = time.time()
    for r in range(N_REP):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X1, y, g):
            P["a_ours_1view_ridge100"][r, te] = ridge_fixed(X1[tr], y[tr], X1[te])
            P["e_ridge100_teamfull_1view"][r, te] = ridge_fixed(Xf[tr], y[tr], Xf[te])
            P["b_ridge_2view_cv"][r, te], a = ridge_cv(X2[tr], y[tr], g[tr], X2[te]); info["alpha_b"].append(a)
            for s in SEEDS:
                f, ep = train_ldl(X2[tr], y[tr], g[tr], seed=1000 * r + s); P[f"c_ldl_2view_s{s}"][r, te] = f(X2[te]); info["epochs_c"].append(ep)
                f, ep = train_ldl(X1[tr], y[tr], g[tr], seed=1000 * r + s); P[f"d_ldl_1view_s{s}"][r, te] = f(X1[te]); info["epochs_d"].append(ep)
        print(f"rep {r + 1}/{N_REP} {time.time() - t0:.0f}s", flush=True)
    # seed-averaged LDL predictions (mean over 3 seeds) as the main c / d entries
    P["c_ldl_2view_media3"] = np.mean([P[f"c_ldl_2view_s{s}"] for s in SEEDS], 0)
    P["d_ldl_1view_media3"] = np.mean([P[f"d_ldl_1view_s{s}"] for s in SEEDS], 0)

    rng = np.random.default_rng(0); PC = {k: EC.per_cow(v, g, cows) for k, v in P.items()}
    R = {"n_vacas": int(len(cows)), "n_fotos": int(len(y)), "configs": {}}
    const = EC.per_cow(EC.constant_baseline(y, g, cows, ycow), g, cows)
    R["configs"]["const_mediana"], _ = summarize(const, ycow, rng)
    R["configs"]["a_ours_1view_ridge100"], ref = summarize(PC["a_ours_1view_ridge100"], ycow, rng)
    for k in [k for k in PC if k != "a_ours_1view_ridge100"]:
        R["configs"][k], _ = summarize(PC[k], ycow, rng, ref)
    for tag in ("c_ldl_2view", "d_ldl_1view"):
        maes = [R["configs"][f"{tag}_s{s}"]["ae"][0] for s in SEEDS]
        R[f"spread_semillas_{tag}"] = {"mae_por_semilla": maes, "min": min(maes), "max": max(maes)}
    R["alpha_b_elegido"] = pd.Series(info["alpha_b"]).value_counts().to_dict()
    R["epocas_c"] = {"mediana": float(np.median(info["epochs_c"])), "p10": float(np.percentile(info["epochs_c"], 10)), "p90": float(np.percentile(info["epochs_c"], 90))}
    R["epocas_d"] = {"mediana": float(np.median(info["epochs_d"])), "p10": float(np.percentile(info["epochs_d"], 10)), "p90": float(np.percentile(info["epochs_d"], 90))}
    R["segundos"] = time.time() - t0
    json.dump(R, open(out_json, "w"), indent=1, default=float)
    f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
    print(f"{'config':28s} {'MAE':>20s} {'w25':>6s} {'w50':>6s} {'slope':>6s} {'AUC':>6s} {'>=3.5':>6s} {'<=2.5':>6s} {'dMAE vs a':>22s}")
    for k, c in R["configs"].items():
        print(f"{k:28s} {f3(c['ae']):>20s} {c['w25'][0]:6.3f} {c['w50'][0]:6.3f} {c['slope'][0]:6.3f} {c['auc_le275'][0]:6.3f} "
              f"{c['mae_ge350']:6.3f} {c['mae_le250']:6.3f} {f3(c['delta_vs_a']) if 'delta_vs_a' in c else '':>22s}")
    print(json.dumps({k: R[k] for k in R if k.startswith(("spread", "alpha", "epocas", "segundos"))}, indent=1, default=float))


if __name__ == "__main__":
    main()
