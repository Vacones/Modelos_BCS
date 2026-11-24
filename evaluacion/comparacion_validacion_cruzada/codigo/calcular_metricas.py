"""Tabla de métricas por modelo y gráficas de predicho contra real, con el BCS por vaca.

Entrada: carpeta privada con un .npz por modelo, cada uno con
  P     (R, n_vacas)  predicción continua por vaca (media de sus fotos) en cada repetición de la validación
                      cruzada, NaN si el modelo no calificó a la vaca en esa repetición
  ycow  (n_vacas,)    BCS del veterinario por vaca
  cows  (n_vacas,)    identificadores ordenados, solo para verificar que todos los modelos usan el mismo orden
La repetición r usa GroupKFold(5, shuffle=True, random_state=r) por vaca, igual en todos los modelos.

Salida sin identificadores ni predicciones individuales: metricas_modelos.json, metricas_modelos.csv,
matriz_confusion_final.json y las figuras PNG.
Uso: python calcular_metricas.py CARPETA_PRIVADA modelos.json CARPETA_SALIDA
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix

NIVELES = np.round(np.arange(2.25, 3.76, 0.25), 2)
UMBRAL_ALERTA = 2.90
GRIS, AZUL_C, AZUL, TINTA = "#6b6b6b", "#56b4e9", "#0072b2", "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def snap(p):
    return np.clip(np.round(np.asarray(p) * 4) / 4, NIVELES[0], NIVELES[-1])


def clase(v):
    return np.round((np.asarray(v) - NIVELES[0]) / 0.25).astype(int)


def metricas(p, y, alerta_umbral=None):
    v = ~np.isnan(p); p, y = p[v], y[v]
    e = p - y; s = snap(p)
    yl, pl = clase(y), clase(s)
    presentes = np.unique(yl)
    pm, rm, fm, _ = precision_recall_fscore_support(yl, pl, labels=presentes, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(yl, pl, labels=presentes, average="weighted", zero_division=0)
    flaca = y <= 2.75 + 1e-9; al = s <= 2.75 + 1e-9
    pf, rf, ff, _ = precision_recall_fscore_support(flaca, al, average="binary", zero_division=0)
    out = {"cobertura": float(v.mean()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
           "exactitud": float((np.abs(s - y) < 1e-9).mean()),
           "dentro_025": float((np.abs(s - y) <= 0.25 + 1e-9).mean()), "dentro_050": float((np.abs(s - y) <= 0.50 + 1e-9).mean()),
           "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm), "f1_ponderado": float(fw),
           "qwk": float(cohen_kappa_score(yl, pl, weights="quadratic", labels=list(range(len(NIVELES))))),
           "flaca_precision": float(pf), "flaca_sensibilidad": float(rf), "flaca_f1": float(ff),
           "flaca_auc": float(roc_auc_score(flaca, -p)) if np.ptp(p) > 0 else 0.5,
           "pendiente": float(np.polyfit(y, p, 1)[0]) if np.ptp(p) > 0 else 0.0}
    if alerta_umbral is not None:
        a2 = p <= alerta_umbral + 1e-9
        pa, ra, fa, _ = precision_recall_fscore_support(flaca, a2, average="binary", zero_division=0)
        out.update({"alerta290_precision": float(pa), "alerta290_sensibilidad": float(ra), "alerta290_f1": float(fa)})
    return out


def promedio(lista):
    return {k: float(np.mean([d[k] for d in lista])) for k in lista[0]}


def main():
    priv, cfg_json, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(cfg_json, encoding="utf-8"))  # [{"clave", "nombre", "alerta": bool}, ...] en orden
    D, cows_ref, y_ref = {}, None, None
    for m in cfg:
        f = os.path.join(priv, m["clave"] + ".npz")
        if not os.path.exists(f):
            print("falta", m["clave"]); continue
        z = np.load(f, allow_pickle=True)
        cows = np.array([str(c) for c in z["cows"]]); y = z["ycow"].astype(float); P = np.atleast_2d(z["P"].astype(float))
        if cows_ref is None:
            cows_ref, y_ref = cows, y
        assert (cows == cows_ref).all() and np.allclose(y, y_ref), f"{m['clave']}: orden de vacas o etiquetas distintas"
        D[m["clave"]] = P
    comunes = min(P.shape[0] for P in D.values())
    R = {"n_vacas": int(len(y_ref)), "repeticiones_comunes": int(comunes), "modelos": {}}
    filas = []
    for m in cfg:
        if m["clave"] not in D:
            continue
        P = D[m["clave"]]; u = UMBRAL_ALERTA if m.get("alerta") else None
        tabla = promedio([metricas(P[r], y_ref, u) for r in range(comunes)])
        todas = promedio([metricas(P[r], y_ref, u) for r in range(P.shape[0])])
        por_rep = [metricas(P[r], y_ref, u) for r in range(P.shape[0])]
        R["modelos"][m["clave"]] = {"nombre": m["nombre"], "repeticiones_disponibles": int(P.shape[0]),
                                    "metricas_repeticiones_comunes": tabla, "metricas_todas_las_repeticiones": todas,
                                    "por_repeticion": por_rep}
        filas.append({"modelo": m["nombre"], **{k: round(v, 3) + 0.0 for k, v in tabla.items()}})
    json.dump(R, open(os.path.join(out, "metricas_modelos.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pd.DataFrame(filas).to_csv(os.path.join(out, "metricas_modelos.csv"), index=False, encoding="utf-8-sig")
    print(pd.DataFrame(filas).to_string(index=False))

    # ---------- gráficas de predicho contra real, con la predicción por vaca promediada sobre las repeticiones
    rng = np.random.default_rng(0); jit = rng.uniform(-0.07, 0.07, len(y_ref))

    def panel(ax, clave, nombre, color, etiquetas=True):
        p = np.nanmean(D[clave][:comunes], 0)
        ax.fill_between([2.0, 4.0], [1.75, 3.75], [2.25, 4.25], color="#e8f1f8", zorder=0, lw=0)
        ax.plot([2.0, 4.0], [2.0, 4.0], color=GRIS, lw=1.2, ls="--", zorder=1)
        ax.scatter(y_ref + jit, p, s=30, color=color, edgecolor="white", lw=0.6, zorder=3)
        mt = R["modelos"][clave]["metricas_repeticiones_comunes"]
        ax.text(0.03, 0.97, f"MAE {mt['mae']:.3f}\nRMSE {mt['rmse']:.3f}\nR² {round(mt['r2'], 2) + 0.0:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5, color=TINTA)
        ax.set_title(nombre, fontsize=10.5, color=TINTA)
        ax.set_xlim(2.05, 3.95); ax.set_ylim(2.05, 3.95)
        ax.set_xticks(NIVELES[::2]); ax.set_yticks(NIVELES[::2])
        for s_ in ["top", "right"]:
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors="#555555", labelsize=9)
        if etiquetas:
            ax.set_xlabel("BCS veterinario", color=TINTA); ax.set_ylabel("BCS predicho", color=TINTA)

    principal = [m for m in cfg if m.get("principal") and m["clave"] in D]
    if principal:
        m = principal[0]
        fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=200)
        panel(ax, m["clave"], m["nombre"], AZUL)
        ax.legend(handles=[plt.Line2D([], [], color=GRIS, ls="--", label="Predicción perfecta"),
                           plt.Rectangle((0, 0), 1, 1, color="#e8f1f8", label="Dentro de ±0.25"),
                           plt.Line2D([], [], marker="o", ls="", color=AZUL, label="Una vaca")],
                  frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_predicho_vs_real_final.png")); plt.close(fig)

    claves = [m for m in cfg if m["clave"] in D and not m.get("solo_tabla")]
    n = len(claves); cols = 4 if n > 6 else 3; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows), dpi=200, squeeze=False)
    for i, m in enumerate(claves):
        ax = axs[i // cols][i % cols]
        col = GRIS if m["clave"] == "constante" else (AZUL if m.get("principal") else AZUL_C)
        panel(ax, m["clave"], m.get("corto", m["nombre"]), col, etiquetas=False)
        if i % cols == 0: ax.set_ylabel("BCS predicho", color=TINTA)
        if i // cols == rows - 1 or i + cols >= n: ax.set_xlabel("BCS veterinario", color=TINTA)
    for j in range(n, rows * cols):
        axs[j // cols][j % cols].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_predicho_vs_real_modelos.png")); plt.close(fig)

    if principal:
        m = principal[0]; p = np.nanmean(D[m["clave"]][:comunes], 0); v = ~np.isnan(p)
        cm = confusion_matrix(clase(y_ref[v]), clase(snap(p[v])), labels=list(range(len(NIVELES))))
        fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=200)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(NIVELES)):
            for j in range(len(NIVELES)):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else TINTA, fontsize=10)
        ax.set_xticks(range(len(NIVELES)), [f"{l:.2f}" for l in NIVELES]); ax.set_yticks(range(len(NIVELES)), [f"{l:.2f}" for l in NIVELES])
        ax.set_xlabel("BCS predicho (redondeado a 0.25)", color=TINTA); ax.set_ylabel("BCS veterinario", color=TINTA)
        ax.tick_params(colors="#555555", labelsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_matriz_confusion_final.png")); plt.close(fig)
        json.dump({"niveles": NIVELES.tolist(), "matriz": cm.tolist()}, open(os.path.join(out, "matriz_confusion_final.json"), "w"))


if __name__ == "__main__":
    main()
