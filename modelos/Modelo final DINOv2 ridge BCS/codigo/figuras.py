"""Figures and summary numbers of the final model (out-of-fold, annotated boxes).

Final recipe: rump crop +10%, DINOv2-S (CLS + mean patch token, flip TTA), ridge alpha 100, linear calibration
fitted inside each training fold (inner GroupKFold 5 on per-cow OOF predictions). GroupKFold by cow 5x10.
Per-cow prediction = mean over the 10 repeats of the mean of its photos.

Usage: python figuras.py DATA_YOLO META PESOS RESULTADOS_DIR
Writes fig_real_vs_predicho.png, fig_error_por_nivel.png, fig_comparacion_modelos.png and
resumen_modelo_final.json into RESULTADOS_DIR. No per-cow ids are written.
"""
import sys, os, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
import bcs_comun as C  # noqa: E402

ALIAS = {"2107": "1548"}; SPLIT = {"train": "train", "val": "valid", "test": "test"}
N_SPLITS, N_REP, ALPHA, UMBRAL = 5, 10, 100.0, 2.90
AZUL, NARANJA, GRIS = "#0072B2", "#E69F00", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def nueva():
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))


def metricas(pc, yc):
    e = np.abs(pc - yc); es = np.abs(np.clip(np.round(pc * 4) / 4, 2.25, 3.75) - yc)
    fl = yc <= 2.75; al = pc <= UMBRAL + 1e-9
    return {"mae": e.mean(), "mae_redondeado": es.mean(), "dentro_025": (es <= .25 + 1e-9).mean(), "dentro_050": (es <= .5 + 1e-9).mean(),
            "pendiente": np.polyfit(yc, pc, 1)[0], "sens_flaca": (al & fl).sum() / fl.sum(), "espec_flaca": (~al & ~fl).sum() / (~fl).sum()}


def main():
    data, meta_csv, pesos, out = sys.argv[1:5]
    torch.set_num_threads(os.cpu_count())
    meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    crops = []
    for _, r in meta.iterrows():
        d = os.path.join(data, SPLIT[r.split])
        sel = C.elegir_grupa(C.leer_txt_yolo(os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")))
        crops.append(C.recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), sel[1]))
    X = C.features(C.cargar_dinov2(pesos), crops); y = meta.bcs.values.astype(float)
    g = meta.cow_id.map(lambda c: ALIAS.get(c, c)).values
    cows = np.unique(g); yc = pd.Series(y).groupby(g).first().reindex(cows).values
    Praw = np.zeros((N_REP, len(cows))); Pcal = np.zeros((N_REP, len(cows)))
    for r in range(N_REP):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, g):
            mod = nueva().fit(X[tr], y[tr]); oof = np.zeros(len(tr))
            for i, j in GroupKFold(5).split(X[tr], y[tr], g[tr]):
                oof[j] = nueva().fit(X[tr][i], y[tr][i]).predict(X[tr][j])
            d = pd.DataFrame({"g": g[tr], "p": oof, "y": y[tr]}).groupby("g").mean(); a, b = np.polyfit(d.p, d.y, 1)
            p = pd.Series(mod.predict(X[te])).groupby(g[te]).mean(); idx = np.searchsorted(cows, p.index.values)
            Praw[r, idx] = p.values; Pcal[r, idx] = a * p.values + b
    raw = pd.DataFrame([metricas(Praw[r], yc) for r in range(N_REP)]).mean().to_dict()
    cal = pd.DataFrame([metricas(Pcal[r], yc) for r in range(N_REP)]).mean().to_dict()
    pc = Pcal.mean(0)  # per-cow calibrated prediction averaged over repeats (for the figures)
    ae = np.abs(Pcal - yc).mean(0); const_ae = np.abs(yc - 3.0)
    niveles = np.unique(yc)
    por_nivel = {f"{l:.2f}": {"n_vacas": int((yc == l).sum()), "mae_modelo": float(ae[yc == l].mean()), "mae_constante": float(const_ae[yc == l].mean())} for l in niveles}
    R = {"protocolo": "GroupKFold por vaca 5x10, ridge alfa 100, calibracion dentro del fold, cajas anotadas, 66 vacas",
         "sin_calibrar": {k: float(v) for k, v in raw.items()}, "calibrado_en_fold": {k: float(v) for k, v in cal.items()},
         "umbral_alerta": UMBRAL, "por_nivel": por_nivel,
         "rango_prediccion_por_vaca": {"min": float(pc.min()), "max": float(pc.max()), "p05": float(np.percentile(pc, 5)), "p95": float(np.percentile(pc, 95))},
         "mae_por_vaca_promedio_repeticiones": float(ae.mean())}
    json.dump(R, open(os.path.join(out, "resumen_modelo_final.json"), "w"), indent=1)
    print(json.dumps(R, indent=1))

    # ---- figure 1: real vs predicted
    rng = np.random.default_rng(0); jit = rng.uniform(-0.06, 0.06, len(yc))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([2.1, 3.9], [2.1, 3.9], color=GRIS, lw=1, ls="--", label="Predicción perfecta")
    ax.axhline(3.0, color=NARANJA, lw=1.5, label="Constante 3.00")
    ax.scatter(yc + jit, pc, s=28, color=AZUL, alpha=.8, edgecolor="white", lw=.5, label="Vaca (fuera de pliegue)")
    ax.set_xlim(2.1, 3.9); ax.set_ylim(2.1, 3.9); ax.set_xticks(niveles); ax.set_yticks(np.arange(2.25, 3.80, .25))
    ax.set_xlabel("BCS del veterinario"); ax.set_ylabel("BCS predicho (calibrado)"); ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=.25, lw=.5); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_real_vs_predicho.png"), dpi=150); plt.close(fig)

    # ---- figure 2: MAE per level
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); x = np.arange(len(niveles)); w = .38
    mm = [por_nivel[f"{l:.2f}"]["mae_modelo"] for l in niveles]; mc = [por_nivel[f"{l:.2f}"]["mae_constante"] for l in niveles]
    ax.bar(x - w / 2, mm, w, color=AZUL, label="Modelo final")
    ax.bar(x + w / 2, mc, w, color=NARANJA, label="Constante 3.00")
    ax.set_xticks(x, [f"{l:.2f}\nn = {por_nivel[f'{l:.2f}']['n_vacas']}" for l in niveles])
    ax.set_xlabel("BCS del veterinario (n = vacas)"); ax.set_ylabel("MAE por vaca"); ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25, lw=.5); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_error_por_nivel.png"), dpi=150); plt.close(fig)

    # ---- figure 3: model comparison (same 66-cow folds)
    rc = json.load(open(os.path.join(out, "resultados_cierre.json"))); r1 = json.load(open(os.path.join(out, "resultados.json")))
    filas = [("Constante 3.00", rc["configs"]["const_mediana"]["ae"]),
             ("EfficientNet-B0 congelado + CORAL", r1["configs"]["b0_224|coral+pca"]["ae"]),
             ("DINOv2-B + ridge", r1["configs"]["dinoB_224|ridge"]["ae"]),
             ("LDL del equipo, 2 vistas", rc["configs"]["c_ldl_2view_media3"]["ae"]),
             ("Ridge con 2 vistas del equipo", rc["configs"]["b_ridge_2view_cv"]["ae"]),
             ("Modelo final DINOv2-S + ridge", rc["configs"]["a_ours_1view_ridge100"]["ae"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); yy = np.arange(len(filas))[::-1]
    for yi, (nom, (v, lo, hi)) in zip(yy, filas):
        col = AZUL if nom.startswith("Modelo final") else (NARANJA if nom.startswith("Constante") else GRIS)
        ax.plot([lo, hi], [yi, yi], color=col, lw=2); ax.plot(v, yi, "o", color=col, ms=7)
        ax.text(hi + 0.004, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(yy, [f[0] for f in filas]); ax.set_xlabel("MAE por vaca (IC 95 %), 66 vacas, mismos pliegues")
    ax.set_xlim(0.15, 0.40); ax.grid(axis="x", alpha=.25, lw=.5); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_comparacion_modelos.png"), dpi=150); plt.close(fig)
    json.dump({"filas": [[n, v] for n, v in filas]}, open(os.path.join(out, "fig_comparacion_modelos_valores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
