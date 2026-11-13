"""System-level evaluation: detector crops -> DINOv2-S -> ridge -> calibration, per cow.

1. Detector coverage on the 331 labeled photos and IoU of the chosen rump vs the annotated rump
   (train split vs valid+test split, which the detector never saw).
2. GroupKFold by cow 5x10: ridge (alpha 100) trained on ANNOTATED-box features of the training cows,
   calibration a*pred+b fitted inside the training fold (inner GroupKFold 5 OOF per cow), held-out
   cows predicted from DETECTOR-box features. A cow with no usable photo is an abstention.

Usage: python evaluar_sistema.py DATA_YOLO META_CSV PESOS_DINOV2 DETECTOR_PT OUT_JSON
"""
import sys, os, json
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcs_comun as C  # noqa: E402
from predecir import cajas_detector  # noqa: E402

ALIAS = {"2107": "1548"}
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
ALPHA, N_SPLITS, N_REP, UMBRAL, CONF_MIN = 100.0, 5, 10, 2.90, 0.25


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def nueva():
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))


def main():
    data, meta_csv, pesos, det_pt, out_json = sys.argv[1:6]
    torch.set_num_threads(os.cpu_count())
    meta = pd.read_csv(meta_csv, dtype={"cow_id": str}); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    det = YOLO(det_pt)
    rows, crops_ann, crops_det = [], [], []
    for _, r in meta.iterrows():
        d = os.path.join(data, SPLIT_DIR[r.split])
        img = Image.open(os.path.join(d, "images", r.file_name)).convert("RGB")
        ann = C.elegir_grupa(C.leer_txt_yolo(os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")))
        cajas = cajas_detector(det, img); sel = C.elegir_grupa(cajas)
        n_cow = sum(k == C.ID_COW and c >= CONF_MIN for k, c, _ in cajas); n_ass = sum(k == C.ID_ASS and c >= CONF_MIN for k, c, _ in cajas)
        if sel is None:
            motivo = "sin grupa detectada" if not any(k == C.ID_ASS for k, _, _ in cajas) else "grupa fuera de la vaca"
        elif sel[0] < CONF_MIN:
            motivo = "confianza grupa < 0.25"
        else:
            motivo = "ok"
        rows.append({"cow": ALIAS.get(r.cow_id, r.cow_id), "bcs": float(r.bcs), "split": r.split, "ann_ok": ann is not None,
                     "motivo": motivo, "conf": sel[0] if sel else np.nan, "iou": iou(sel[1], ann[1]) if (sel and ann) else np.nan,
                     "n_cow_det": n_cow, "n_ass_det": n_ass})
        crops_ann.append(C.recortar(img, ann[1]) if ann else None)
        crops_det.append(C.recortar(img, sel[1]) if motivo == "ok" else None)
    df = pd.DataFrame(rows)
    df = df[df.ann_ok].reset_index(drop=True)
    crops_ann = [c for c in crops_ann if c is not None]
    crops_det_k = [c for c in crops_det]  # aligned with meta rows; filter same as df
    keep_mask = [r["ann_ok"] for r in rows]; crops_det_k = [c for c, k in zip(crops_det_k, keep_mask) if k]
    m = C.cargar_dinov2(pesos)
    Xa = C.features(m, crops_ann)
    ok = df.motivo.eq("ok").values
    Xd = np.full_like(Xa, np.nan); Xd[ok] = C.features(m, [c for c, o in zip(crops_det_k, ok) if o])

    # ---- coverage / IoU
    df["grupo"] = np.where(df.split == "train", "train (visto por el detector)", "valid+test (no visto)")
    cob = {}
    for gname, dg in [("todas", df), *df.groupby("grupo")]:
        okg = dg.motivo.eq("ok")
        cob[gname] = {"fotos": int(len(dg)), "vacas": int(dg.cow.nunique()), "pct_fotos_utiles": float(okg.mean()),
                      "rechazos": dg.motivo[~okg].value_counts().to_dict(),
                      "iou_media": float(dg.iou[okg].mean()), "iou_mediana": float(dg.iou[okg].median()),
                      "pct_iou_gt_05": float((dg.iou[okg] > 0.5).mean()), "pct_iou_gt_07": float((dg.iou[okg] > 0.7).mean()),
                      "vacas_sin_foto_util": int((dg.groupby("cow").motivo.apply(lambda s: (s == "ok").sum()) == 0).sum())}

    # ---- system CV
    y = df.bcs.values; g = df.cow.values
    cows = np.unique(g); yc = pd.Series(y).groupby(g).first().reindex(cows).values
    vt = np.isin(cows, df[df.split != "train"].cow.unique())
    keys = ["ann_raw", "ann_cal", "det_raw", "det_cal"]
    P = {k: np.full((N_REP, len(cows)), np.nan) for k in keys}
    for r in range(N_REP):
        for tr, te in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(Xa, y, g):
            mod = nueva().fit(Xa[tr], y[tr])
            # calibration inside the training fold
            oof = np.zeros(len(tr))
            for itr, ite in GroupKFold(5).split(Xa[tr], y[tr], g[tr]):
                oof[ite] = nueva().fit(Xa[tr][itr], y[tr][itr]).predict(Xa[tr][ite])
            d = pd.DataFrame({"g": g[tr], "p": oof, "y": y[tr]}).groupby("g").mean()
            a, b = np.polyfit(d.p.values, d.y.values, 1)
            pa = pd.Series(mod.predict(Xa[te])).groupby(g[te]).mean()
            okt = ok[te]
            pdd = pd.Series(mod.predict(Xd[te][okt]) if okt.any() else [], dtype=float).groupby(g[te][okt]).mean()
            idx_a = np.searchsorted(cows, pa.index.values)
            P["ann_raw"][r, idx_a] = pa.values; P["ann_cal"][r, idx_a] = a * pa.values + b
            if len(pdd):
                idx_d = np.searchsorted(cows, pdd.index.values)
                P["det_raw"][r, idx_d] = pdd.values; P["det_cal"][r, idx_d] = a * pdd.values + b

    def met(Pk, mask):
        yy = yc[mask]; out = []
        for r in range(N_REP):
            p = Pk[r, mask]; v = ~np.isnan(p); e = np.abs(p[v] - yy[v])
            es = np.abs(np.clip(np.round(p[v] * 4) / 4, 2.25, 3.75) - yy[v])
            fl = yy[v] <= 2.75; al = p[v] <= UMBRAL + 1e-9
            out.append({"mae": e.mean(), "mae_redondeado": es.mean(), "dentro_025": (es <= .25 + 1e-9).mean(),
                        "dentro_050": (es <= .5 + 1e-9).mean(), "cobertura_vacas": v.mean(), "abstenciones": int((~v).sum()),
                        "mae_ge350": e[yy[v] >= 3.5].mean(), "mae_le250": e[yy[v] <= 2.5].mean(),
                        "sens_flaca_2.90": (al & fl).sum() / max(fl.sum(), 1), "espec_flaca_2.90": (~al & ~fl).sum() / max((~fl).sum(), 1)})
        return {k: float(v) for k, v in pd.DataFrame(out).mean().items()} | {"n_vacas": int(mask.sum())}

    const = {}
    for name, mask in (("todas_66", np.ones(len(cows), bool)), ("valid_test_20", vt)):
        const[name] = float(np.abs(yc[mask] - 3.0).mean())
    sis = {name: {k: met(P[k], mask) for k in keys} for name, mask in (("todas_66", np.ones(len(cows), bool)), ("valid_test_20", vt))}
    R = {"cobertura": cob, "sistema": sis, "constante_3.00_mae": const,
         "nota": "ann = cajas anotadas (Roboflow); det = cajas del detector. raw = sin calibrar, cal = calibrado dentro del fold. "
                 "Umbral alerta 2.90 se aplica a la prediccion (cal o raw segun la columna)."}
    json.dump(R, open(out_json, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
