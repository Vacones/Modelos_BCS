"""Pilot: frozen features + sklearn heads, repeated GroupKFold by cow (5x10),
hyper-parameters chosen ONLY by inner GroupKFold(3) on the training fold.
Metrics are per cow (each cow weighs once), 95% bootstrap CIs over cows.

Usage: python evaluar_cabezas.py FEATS_DIR [OUT_JSON]
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

LEV = np.array([2.5, 2.75, 3.0, 3.25, 3.5, 3.75])  # 2.25 merged into 2.50 for class heads (2 photos)
GRID_MIN, GRID_MAX = 2.25, 3.75
ALIAS = {"2107": "1548"}  # possibly the same animal -> same CV group
N_SPLITS, N_REP, N_BOOT = 5, 10, 2000
PCA_DIM = 128


def snap(v):
    return np.clip(np.round(np.asarray(v) * 4) / 4, GRID_MIN, GRID_MAX)


def prep(Xtr, Xte, pca):
    s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
    if pca:
        p = PCA(min(PCA_DIM, Xtr.shape[0] - 1), random_state=0).fit(Xtr); Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        sc = Xtr.std(0).mean(); Xtr, Xte = Xtr / sc, Xte / sc
    return Xtr, Xte


def cls_idx(y):
    return np.clip(np.round((np.maximum(y, 2.5) - 2.5) / 0.25).astype(int), 0, len(LEV) - 1)


def fit_predict(head, hp, Xtr, ytr, Xte):
    if head == "ridge":
        return Ridge(alpha=hp).fit(Xtr, ytr).predict(Xte)
    if head in ("mlog", "mlog_soft"):
        k = cls_idx(ytr)
        if head == "mlog":
            m = LogisticRegression(C=hp, max_iter=3000).fit(Xtr, k)
            P = np.zeros((len(Xte), len(LEV))); P[:, m.classes_] = m.predict_proba(Xte)
        else:  # Gaussian soft labels (sigma 0.125) via replicated, weighted samples
            W = np.exp(-0.5 * ((LEV[None, :] - np.maximum(ytr, 2.5)[:, None]) / 0.125) ** 2); W /= W.sum(1, keepdims=True)
            rr, cc = np.nonzero(W > 1e-3)
            m = LogisticRegression(C=hp, max_iter=3000).fit(Xtr[rr], cc, sample_weight=W[rr, cc])
            P = np.zeros((len(Xte), len(LEV))); P[:, m.classes_] = m.predict_proba(Xte)
        return P @ LEV
    if head == "coral":  # shared weights, one bias per threshold (K-1 binary tasks)
        K1 = len(LEV) - 1; k = cls_idx(ytr); B = 10.0  # bias columns scaled so the L2 penalty barely touches them
        Xe = np.vstack([np.hstack([Xtr, B * np.eye(K1)[[t] * len(Xtr)]]) for t in range(K1)])
        ye = np.concatenate([(k > t).astype(int) for t in range(K1)])
        m = LogisticRegression(C=hp, max_iter=3000, fit_intercept=False).fit(Xe, ye)
        w, b = m.coef_[0, :-K1], B * m.coef_[0, -K1:]
        Pgt = 1 / (1 + np.exp(-(Xte @ w)[:, None] - b[None, :]))
        Pgt = np.minimum.accumulate(Pgt, axis=1)  # enforce monotone cumulative probs
        return LEV[0] + 0.25 * Pgt.sum(1)
    raise ValueError(head)


C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
GRIDS = {"ridge": [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5], "mlog": C_GRID, "mlog_soft": C_GRID, "coral": C_GRID}


def nested_fold(X, y, g, tr, te, head, pca):
    gi = GroupKFold(3)
    scores = []
    for hp in GRIDS[head]:
        err = []
        for itr, ite in gi.split(X[tr], y[tr], g[tr]):
            a, b = prep(X[tr][itr], X[tr][ite], pca)
            p = fit_predict(head, hp, a, y[tr][itr], b)
            d = pd.DataFrame({"g": g[tr][ite], "p": p, "y": y[tr][ite]}).groupby("g").mean()
            err.append(np.abs(d.p - d.y).mean())  # inner criterion: per-cow MAE
        scores.append(np.mean(err))
    hp = GRIDS[head][int(np.argmin(scores))]
    a, b = prep(X[tr], X[te], pca)
    return fit_predict(head, hp, a, y[tr], b), hp


def run_config(X, y, g, head, pca):
    out = np.zeros((N_REP, len(y))); hps = []
    for r in range(N_REP):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, g):
            out[r, te], hp = nested_fold(X, y, g, tr, te, head, pca); hps.append(hp)
    return out, hps


def constant_baseline(y, g, cows, ycow):
    out = np.zeros((N_REP, len(y)))
    for r in range(N_REP):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(y, y, g):
            out[r, te] = np.median(ycow[np.isin(cows, np.unique(g[tr]))])  # median over training COWS
    return out


def per_cow(pred_img, g, cows):
    """(N_REP, n_img) -> (N_REP, n_cows): mean of the cow's photo predictions."""
    return np.stack([pd.Series(p).groupby(g).mean().reindex(cows).values for p in pred_img])


def cow_stats(pc, ycow):
    """Per-cow quantities averaged over repeats (each cow weighs once)."""
    e = np.abs(pc - ycow); es = np.abs(snap(pc) - ycow)
    return {"ae": e.mean(0), "ae_snap": es.mean(0), "w25": (es <= 0.25 + 1e-9).mean(0),
            "w50": (es <= 0.50 + 1e-9).mean(0), "pred": pc.mean(0)}


def boot(fn, n, rng):
    v = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(v), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def slope(pred, yc):
    return np.polyfit(yc, pred, 1)[0] if np.ptp(yc) > 0 else np.nan


def auc(pred, yc):
    t = yc <= 2.75
    return roc_auc_score(t, -pred) if 0 < t.sum() < len(t) else np.nan


def summarize(st, ycow, rng):
    n = len(ycow); res = {}
    for k in ("ae", "ae_snap", "w25", "w50"):
        res[k] = boot(lambda i: st[k][i].mean(), n, rng)
    res["slope"] = boot(lambda i: slope(st["pred"][i], ycow[i]), n, rng)
    res["auc_le275"] = boot(lambda i: auc(st["pred"][i], ycow[i]), n, rng)
    res["mae_by_level"] = {f"{l:.2f}": [float(st["ae"][ycow == l].mean()), int((ycow == l).sum())] for l in np.unique(ycow)}
    return res


def main():
    fdir = sys.argv[1]; out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(fdir, "resultados.json")
    idx = pd.read_csv(os.path.join(fdir, "index.csv"), dtype={"cow_id": str})
    g = idx.cow_id.map(lambda c: ALIAS.get(c, c)).values; y = idx.bcs.values.astype(float)
    cows = np.unique(g); ycow = pd.Series(y).groupby(g).first().reindex(cows).values
    assert (pd.Series(y).groupby(g).nunique() == 1).all(), "a cow has >1 BCS label"
    feats = sorted(f[:-4] for f in os.listdir(fdir) if f.endswith(".npy"))
    cfgs = [(f, h, p) for f in feats for h, p in (("ridge", False), ("ridge", True), ("mlog", True), ("mlog_soft", True), ("coral", True))]
    t0 = time.time()
    outs = Parallel(n_jobs=-1)(delayed(run_config)(np.load(os.path.join(fdir, f + ".npy")), y, g, h, p) for f, h, p in cfgs)
    print(f"CV done in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    PC = {"const_median": per_cow(constant_baseline(y, g, cows, ycow), g, cows)}
    HP = {}
    for (f, h, p), (o, hps) in zip(cfgs, outs):
        name = f"{f}|{h}{'+pca' if p else ''}"; PC[name] = per_cow(o, g, cows)
        HP[name] = pd.Series(hps).value_counts().to_dict()
    ST = {k: cow_stats(v, ycow) for k, v in PC.items()}
    R = {"n_cows": int(len(cows)), "n_images": int(len(y)), "cow_bcs_counts": pd.Series(ycow).value_counts().sort_index().to_dict(),
         "configs": {k: summarize(ST[k], ycow, rng) for k in PC}, "hp_chosen": {k: {str(a): int(b) for a, b in v.items()} for k, v in HP.items()}}
    # paired differences in per-cow absolute error (continuous mean prediction); negative = better than reference
    model_keys = [k for k in PC if k != "const_median"]
    best = min(model_keys, key=lambda k: R["configs"][k]["ae"][0])
    R["best_config"] = best
    R["paired_vs_const"] = {k: boot(lambda i, k=k: (ST[k]["ae"][i] - ST["const_median"]["ae"][i]).mean(), len(cows), rng) for k in model_keys}
    R["paired_vs_const_snap"] = {k: boot(lambda i, k=k: (ST[k]["ae_snap"][i] - ST["const_median"]["ae_snap"][i]).mean(), len(cows), rng) for k in model_keys}
    dino = [k for k in model_keys if k.startswith("dino")]; b0 = [k for k in model_keys if k.startswith("b0")]
    if dino and b0:
        bd = min(dino, key=lambda k: R["configs"][k]["ae"][0]); bb = min(b0, key=lambda k: R["configs"][k]["ae"][0])
        R["paired_best_dino_vs_best_b0"] = [bd, bb, boot(lambda i: (ST[bd]["ae"][i] - ST[bb]["ae"][i]).mean(), len(cows), rng)]
        ens = cow_stats((PC[bd] + PC[bb]) / 2, ycow); R["configs"]["ensemble_dino_b0"] = summarize(ens, ycow, rng)
    else:
        R["paired_best_dino_vs_best_b0"] = "not run: no DINOv2 features (weights unavailable)"
    maes = [R["configs"][k]["ae"][0] for k in model_keys]
    R["spread_mae_across_configs"] = [float(min(maes)), float(np.median(maes)), float(max(maes))]
    R["cv_seconds"] = time.time() - t0
    json.dump(R, open(out_json, "w"), indent=1)
    print(f"{'config':28s} {'MAE':>18s} {'MAEsnap':>6s} {'w25':>18s} {'w50':>6s} {'slope':>18s} {'AUC<=2.75':>18s} {'dMAE vs const':>22s}")
    f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
    for k, c in R["configs"].items():
        d = R["paired_vs_const"].get(k)
        print(f"{k:28s} {f3(c['ae'])} {c['ae_snap'][0]:.3f} {f3(c['w25'])} {c['w50'][0]:.3f} {f3(c['slope'])} {f3(c['auc_le275'])} {f3(d) if d else '':>22s}")
    print("best:", best, "by level:", R["configs"][best]["mae_by_level"])
    print("const by level:", R["configs"]["const_median"]["mae_by_level"])
    print("spread:", R["spread_mae_across_configs"], "hp:", json.dumps(R["hp_chosen"])[:800])


if __name__ == "__main__":
    main()
