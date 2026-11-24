"""Figura de RMSE por modelo con la variación entre repeticiones y la referencia de 0.25.

Lee solo metricas_modelos.json (métricas agregadas por repetición, sin IDs ni predicciones por vaca).
Uso: python figura_rmse.py metricas_modelos.json modelos.json CARPETA_SALIDA
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1], encoding="utf-8"))
cfg = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
GRIS, AZUL_C, AZUL, TINTA = "#6b6b6b", "#56b4e9", "#0072b2", "#333333"


def estilo(ax):
    ax.xaxis.grid(True, color="#e3e3e3"); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#555555")


filas = []
for m in cfg:
    d = R["modelos"].get(m["clave"])
    if d is None:
        continue
    reps = [r["rmse"] for r in d["por_repeticion"]]
    filas.append((m.get("corto", m["nombre"]), m["clave"], d["metricas_repeticiones_comunes"]["rmse"], min(reps), max(reps)))
filas.sort(key=lambda f: -f[2])  # el mejor arriba

fig, ax = plt.subplots(figsize=(8, 0.55 * len(filas) + 1.4), dpi=200)
for y, (nom, clave, v, lo, hi) in enumerate(filas):
    col = GRIS if clave == "constante" else (AZUL if clave.startswith("final") else AZUL_C)
    if hi - lo > 1e-6:
        ax.plot([lo, hi], [y, y], color=col, lw=2.5, solid_capstyle="round", alpha=0.55, zorder=2)
    ax.plot(v, y, "o", ms=9, color=col, mec="white", mew=1, zorder=3)
    ax.text(max(hi, v) + 0.006, y, f"{v:.3f}", va="center", ha="left", color=TINTA, fontsize=10.5)
ax.axvline(0.25, color=GRIS, ls="--", lw=1.3, zorder=0)
ax.text(0.253, len(filas) - 0.45, "Referencia 0.25", ha="left", va="bottom", color=GRIS, fontsize=10)
ax.set_yticks(range(len(filas)))
ax.set_yticklabels([f[0] for f in filas], color=TINTA)
xmax = max(f[4] for f in filas)
ax.set_xlim(0.20, max(0.45, xmax + 0.05))
ax.set_ylim(-0.6, len(filas) - 0.05)
ax.set_xlabel("RMSE por vaca", color=TINTA)
estilo(ax)
ax.xaxis.grid(False)
for xg in np.arange(0.20, ax.get_xlim()[1], 0.05):
    if abs(xg - 0.25) > 1e-6:
        ax.axvline(xg, color="#e3e3e3", lw=0.9, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_rmse_modelos.png"))
plt.close(fig)
print([(f[0], round(f[2], 3), round(f[3], 3), round(f[4], 3)) for f in filas])
