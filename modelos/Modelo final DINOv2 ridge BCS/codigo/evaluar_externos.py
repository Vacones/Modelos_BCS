"""Does Huang (ScienceDB) help the CAETEC BCS model? Evaluation ONLY on CAETEC cows,
same protocol as evaluar_cabezas.py: repeated GroupKFold by cow (5x10), hyper-parameters
(and optionally the Huang weight w) chosen by inner GroupKFold(3) over CAETEC training cows,
per-cow mean of photo predictions, bootstrap CIs over cows.

Usage: python evaluar_externos.py FEATS_DIR OUT_JSON [N_REP]
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluar_cabezas as EC  # noqa: E402
warnings.filterwarnings("ignore")

N_SPLITS, N_BOOT, PCA_DIM, BDOM = 5, 2000, 128, 10.0
GRIDS = {"ridge": [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5], "coral": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
PCA_FOR = {"ridge": False, "coral": True}  # best variants of the reference run


# ---------------------------------------------------------------- model
def prep(Xtr, Xte, pca, dom_tr=None):
    s = StandardScaler().fit(Xtr); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
    if pca:
        p = PCA(min(PCA_DIM, Xtr.shape[0] - 1), random_state=0).fit(Xtr); Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        sc = Xtr.std(0).mean(); Xtr, Xte = Xtr / sc, Xte / sc
    if dom_tr is not None:  # domain indicator (Huang=1); test rows are CAETEC -> 0
        Xtr = np.hstack([Xtr, BDOM * dom_tr[:, None]]); Xte = np.hstack([Xte, np.zeros((len(Xte), 1))])
    return Xtr, Xte


def fit_predict(head, hp, Xtr, ytr, Xte, w):
    if head == "ridge":
        return Ridge(alpha=hp).fit(Xtr, ytr, sample_weight=w).predict(Xte)
    lev = np.arange(2.5, max(3.75, ytr.max()) + 1e-9, 0.25)  # CAETEC only -> 2.50..3.75 (= reference)
    K1 = len(lev) - 1; k = np.clip(np.round((np.maximum(ytr, 2.5) - 2.5) / 0.25).astype(int), 0, K1); B = 10.0
    Xe = np.vstack([np.hstack([Xtr, B * np.eye(K1)[[t] * len(Xtr)]]) for t in range(K1)])
    ye = np.concatenate([(k > t).astype(int) for t in range(K1)]); we = np.tile(w, K1)
    m = LogisticRegression(C=hp, max_iter=3000, fit_intercept=False).fit(Xe, ye, sample_weight=we)
    wv, b = m.coef_[0, :-K1], B * m.coef_[0, -K1:]
    P = np.minimum.accumulate(1 / (1 + np.exp(-(Xte @ wv)[:, None] - b[None, :])), axis=1)
    return lev[0] + 0.25 * P.sum(1)


def train_predict(cfg, hp, w, Xc, yc, Xh, yh, dh, Xte):
    """Train on CAETEC rows (weight 1) + Huang rows (weight w) and predict Xte (CAETEC)."""
    if Xh is None or w == 0:
        Xtr, ytr, wt, dom = Xc, yc, np.ones(len(yc)), None
    else:
        Xtr, ytr = np.vstack([Xc, Xh]), np.concatenate([yc, yh]); wt = np.concatenate([np.ones(len(yc)), np.full(len(yh), w)])
        dom = np.concatenate([np.zeros(len(yc)), dh]) if cfg["dom"] else None
    a, b = prep(Xtr, Xte, PCA_FOR[cfg["head"]], dom)
    return fit_predict(cfg["head"], hp, a, ytr, b, wt)


def outer_fold(cfg, X, y, g, Xh, yh, tr, te):
    head = cfg["head"]; ws = cfg["ws"]; Xh_ = Xh if cfg["ext"] else None
    dh = np.ones(len(yh)) if Xh is not None else None
    inner = list(GroupKFold(3).split(X[tr], y[tr], g[tr]))
    best, oof_best = None, None
    for w in ws:
        for hp in GRIDS[head]:
            err, oof = [], np.zeros(len(tr))
            for itr, ite in inner:
                p = train_predict(cfg, hp, w, X[tr][itr], y[tr][itr], Xh_, yh, dh, X[tr][ite]); oof[ite] = p
                d = pd.DataFrame({"g": g[tr][ite], "p": p, "y": y[tr][ite]}).groupby("g").mean()
                err.append(np.abs(d.p - d.y).mean())
            if best is None or np.mean(err) < best[0]:
                best, oof_best = (np.mean(err), w, hp), oof
    _, w, hp = best
    pred = train_predict(cfg, hp, w, X[tr], y[tr], Xh_, yh, dh, X[te])
    if cfg.get("calib"):  # post-hoc linear calibration fitted on inner OOF predictions (per cow)
        d = pd.DataFrame({"g": g[tr], "p": oof_best, "y": y[tr]}).groupby("g").mean()
        a1, b1 = np.polyfit(d.p.values, d.y.values, 1); pred = a1 * pred + b1
    return te, pred, (w, hp)


# ---------------------------------------------------------------- metrics
def summarize(pc, ycow, rng, ref_ae=None):
    st = EC.cow_stats(pc, ycow); n = len(ycow)
    r = {k: EC.boot(lambda i, k=k: st[k][i].mean(), n, rng) for k in ("ae", "ae_snap", "w25", "w50")}
    r["slope"] = EC.boot(lambda i: EC.slope(st["pred"][i], ycow[i]), n, rng)
    r["auc_le275"] = EC.boot(lambda i: EC.auc(st["pred"][i], ycow[i]), n, rng)
    hi, lo = ycow >= 3.5, ycow <= 2.5
    r["mae_ge350"] = [float(st["ae"][hi].mean()), int(hi.sum())]; r["mae_le250"] = [float(st["ae"][lo].mean()), int(lo.sum())]
    r["mean_bias"] = float((st["pred"] - ycow).mean())
    if ref_ae is not None:
        r["delta_vs_ref"] = EC.boot(lambda i: (st["ae"][i] - ref_ae[i]).mean(), n, rng)
    return r, st["ae"]


# ---------------------------------------------------------------- transfer (f)
def choose_alpha(X, y, g):
    best = None
    for a in GRIDS["ridge"]:
        e = []
        for itr, ite in GroupKFold(3).split(X, y, g):
            s = StandardScaler().fit(X[itr]); p = Ridge(alpha=a).fit(s.transform(X[itr]), y[itr]).predict(s.transform(X[ite]))
            d = pd.DataFrame({"g": g[ite], "p": p, "y": y[ite]}).groupby("g").mean(); e.append(np.abs(d.p - d.y).mean())
        if best is None or np.mean(e) < best[0]:
            best = (np.mean(e), a)
    return best[1]


def ridge_full(Xtr, ytr, gtr, Xte):
    a = choose_alpha(Xtr, ytr, gtr); s = StandardScaler().fit(Xtr)
    return Ridge(alpha=a).fit(s.transform(Xtr), ytr).predict(s.transform(Xte)), a


def transfer(X, y, g, Xh, yh, gh, vh):
    out = {}
    cows = np.unique(g); ycow = pd.Series(y).groupby(g).first().reindex(cows).values
    for name, m in (("huang_all", np.ones(len(yh), bool)), *((f"huang_{v}", vh == v) for v in np.unique(vh))):
        p, a = ridge_full(Xh[m], yh[m], gh[m], X)
        pc = pd.Series(p).groupby(g).mean().reindex(cows).values
        ge = ycow >= 3.25
        out[f"{name}->caetec"] = {"alpha": a, "mae": float(np.abs(pc - ycow).mean()), "spearman": float(spearmanr(pc, ycow)[0]),
                                  "slope": float(EC.slope(pc, ycow)), "auc_le275": float(EC.auc(pc, ycow)),
                                  "mae_cows_ge325": [float(np.abs(pc - ycow)[ge].mean()), int(ge.sum())],
                                  "mean_bias": float((pc - ycow).mean()),
                                  "mae_after_offset": float(np.abs(pc - pc.mean() + ycow.mean() - ycow).mean())}
    p, a = ridge_full(X, y, g, Xh)
    d = pd.DataFrame({"g": gh, "v": vh, "p": p, "y": yh}).groupby(["v", "g"]).mean().reset_index()
    res = {"alpha": a}
    for v, dv in [("all", d), *d.groupby("v")]:
        ov = dv[dv.y <= 3.75]
        res[v] = {"n_groups": int(len(dv)), "spearman_all_levels": float(spearmanr(dv.p, dv.y)[0]),
                  "spearman_325_375": float(spearmanr(ov.p, ov.y)[0]), "mae_325_375": float(np.abs(ov.p - ov.y).mean()),
                  "mean_pred_by_level": dv.groupby("y").p.mean().round(3).to_dict()}
    out["caetec->huang"] = res
    return out


# ---------------------------------------------------------------- main
def main():
    fdir, out_json = sys.argv[1], sys.argv[2]; n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    idx = pd.read_csv(os.path.join(fdir, "caetec_index.csv"), dtype={"cow_id": str})
    g = idx.cow_id.map(lambda c: EC.ALIAS.get(c, c)).values; y = idx.bcs.values.astype(float)
    X = np.load(os.path.join(fdir, "caetec_dinoS.npy"))
    hi = pd.read_csv(os.path.join(fdir, "huang_index.csv")); XH = np.load(os.path.join(fdir, "huang_dinoS.npy"))
    yh_all, gh_all, vh_all = hi.bcs.values.astype(float), hi.group.values, hi.view.values
    cows = np.unique(g); ycow = pd.Series(y).groupby(g).first().reindex(cows).values
    SUB = os.environ.get("SUBSET_VIEWS", "YM").split(",")
    sel = {"all": np.ones(len(hi), bool), "sub": np.isin(vh_all, SUB)}

    cfgs = {}
    for head in ("ridge", "coral"):
        cfgs[f"a_{head}_caetec"] = dict(head=head, ext=None, ws=[0], dom=False)
        cfgs[f"e_{head}_calib"] = dict(head=head, ext=None, ws=[0], dom=False, calib=True)
        for s, tag in (("all", "b"), ("sub", "c")):
            for w in (0.1, 0.3, 1.0):
                cfgs[f"{tag}_{head}_{s}_w{w}"] = dict(head=head, ext=s, ws=[w], dom=False)
            cfgs[f"{tag}_{head}_{s}_wCV"] = dict(head=head, ext=s, ws=[0, 0.1, 0.3, 1.0], dom=False)
            for w in (0.3, 1.0):
                cfgs[f"d_{head}_{s}_dom_w{w}"] = dict(head=head, ext=s, ws=[w], dom=True)
            cfgs[f"d_{head}_{s}_dom_wCV"] = dict(head=head, ext=s, ws=[0, 0.1, 0.3, 1.0], dom=True)
    only = os.environ.get("ONLY")
    if only:
        cfgs = {k: v for k, v in cfgs.items() if any(k.startswith(o) for o in only.split(","))}

    splits = [(r, tr, te) for r in range(n_rep) for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, g)]
    def job(name, r, tr, te):
        c = cfgs[name]; m = sel[c["ext"]] if c["ext"] else sel["all"]
        return name, r, outer_fold(c, X, y, g, XH[m] if c["ext"] else None, yh_all[m] if c["ext"] else yh_all, tr, te)
    t0 = time.time()
    # costliest first for load balancing
    order = sorted(cfgs, key=lambda k: -(len(cfgs[k]["ws"]) * (5 if cfgs[k]["head"] == "coral" else 1) * (2 if cfgs[k]["ext"] else 1)))
    res = Parallel(n_jobs=-1, verbose=5)(delayed(job)(k, r, tr, te) for k in order for r, tr, te in splits)
    print(f"CV done in {time.time() - t0:.0f}s", flush=True)
    P = {k: np.zeros((n_rep, len(y))) for k in cfgs}; HP = {k: [] for k in cfgs}
    for name, r, (te, pred, hp) in res:
        P[name][r, te] = pred; HP[name].append(hp)

    rng = np.random.default_rng(0)
    PCc = {k: EC.per_cow(v, g, cows) for k, v in P.items()}
    const = EC.per_cow(EC.constant_baseline(y, g, cows, ycow)[:n_rep], g, cows)
    R = {"n_cows": int(len(cows)), "n_images": int(len(y)), "n_rep": n_rep, "subset_views": SUB,
         "huang_n": {k: int(v.sum()) for k, v in sel.items()}, "configs": {}}
    R["configs"]["const_median"], _ = summarize(const, ycow, rng)
    ref = {}
    for head in ("ridge", "coral"):
        k = f"a_{head}_caetec"
        if k in PCc:
            R["configs"][k], ref[head] = summarize(PCc[k], ycow, rng)
    for k in sorted(PCc):
        if not k.startswith("a_"):
            R["configs"][k], _ = summarize(PCc[k], ycow, rng, ref.get(cfgs[k]["head"]))
    R["hp_chosen"] = {k: pd.Series([f"w={a}|hp={b}" for a, b in v]).value_counts().head(6).to_dict() for k, v in HP.items()}
    R["transfer"] = transfer(X, y, g, XH, yh_all, gh_all, vh_all)
    R["seconds"] = time.time() - t0
    json.dump(R, open(out_json, "w"), indent=1)

    f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
    print(f"{'config':26s} {'MAE':>20s} {'w25':>6s} {'w50':>6s} {'slope':>6s} {'AUC':>6s} {'>=3.5':>6s} {'<=2.5':>6s} {'bias':>6s} {'dMAE vs a (same head)':>24s}")
    for k, c in R["configs"].items():
        d = f3(c["delta_vs_ref"]) if "delta_vs_ref" in c else ""
        print(f"{k:26s} {f3(c['ae']):>20s} {c['w25'][0]:6.3f} {c['w50'][0]:6.3f} {c['slope'][0]:6.3f} {c['auc_le275'][0]:6.3f} "
              f"{c['mae_ge350'][0]:6.3f} {c['mae_le250'][0]:6.3f} {c['mean_bias']:+6.3f} {d:>24s}")
    print("hp:", json.dumps(R["hp_chosen"]))
    print("transfer:", json.dumps(R["transfer"], indent=1))


if __name__ == "__main__":
    main()
