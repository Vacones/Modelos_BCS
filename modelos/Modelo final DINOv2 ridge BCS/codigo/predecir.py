"""Calcula el BCS de UNA vaca a partir de una o varias fotos traseras.

Ejemplos:
  python predecir.py foto1.jpg foto2.jpg foto3.jpg --vaca V01      (usa ../modelo/detector/best.pt)
  python predecir.py foto1.jpg foto2.jpg --caja foto1.txt foto2.txt
Las cajas salen del detector YOLO (.pt de Ultralytics con clases 'ass' y 'cow') o de archivos txt en
formato YOLO (--caja, en el mismo orden que las fotos, clase 0 = ass, 1 = cow y sexta columna opcional
con la confianza).
Imprime un JSON con bcs, bcs_continuo, rango, alerta_flaca, n_fotos_usadas y fotos_rechazadas.
"""
import argparse, json, os, sys, warnings
import numpy as np, joblib, torch
from PIL import Image, ImageOps

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "..", "modelo")  # cabeza.joblib, config.json, detector/best.pt y pesos/
sys.path.insert(0, AQUI)
import bcs_comun as C  # noqa: E402
warnings.filterwarnings("ignore")


def cajas_detector(det, ruta):
    """Corre el detector y devuelve las cajas con la misma convención de clases que bcs_comun."""
    names = {int(k): str(v).lower() for k, v in det.names.items()}
    cls_map = {k: (C.ID_ASS if v in ("ass", "grupa", "rump") else C.ID_COW if v in ("cow", "vaca") else -1) for k, v in names.items()}
    r = det.predict(ruta, verbose=False, conf=0.05)[0]
    out = []
    for (cx, cy, w, h), k, c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append((cls_map.get(int(k), -1), float(c), (cx, cy, w, h)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--vaca", default=None, help="identificador de la vaca, solo se copia a la salida")
    ap.add_argument("--caja", nargs="*", default=None, help="txt YOLO por foto, en el mismo orden")
    ap.add_argument("--detector", default=os.path.join(MODELO, "detector", "best.pt"),
                    help="pesos .pt del detector YOLO (Ultralytics). Por defecto ../modelo/detector/best.pt")
    ap.add_argument("--pesos", default=os.path.join(MODELO, "pesos", "dinov2_vits14.safetensors"),
                    help="pesos de DINOv2 ViT-S/14 en formato safetensors")
    a = ap.parse_args()
    for ruta in (a.pesos, os.path.join(MODELO, "cabeza.joblib")):
        if not os.path.exists(ruta):
            sys.exit(f"no existe {ruta}. Descarga los pesos como indica el README de esta carpeta")
    cfg = json.load(open(os.path.join(MODELO, "config.json"), encoding="utf-8"))
    if a.caja is not None and len(a.caja) != len(a.fotos):
        sys.exit("--caja necesita un txt por foto, en el mismo orden")
    if a.caja is None and not os.path.exists(a.detector):
        sys.exit(f"no existe el detector {a.detector}. Usa --detector o --caja")
    det = None
    if a.caja is None:
        from ultralytics import YOLO
        det = YOLO(a.detector)
    conf_min = cfg["rechazo"]["conf_min_grupa"]
    crops, usadas, rechazadas = [], [], []
    for i, f in enumerate(a.fotos):
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            rechazadas.append({"foto": f, "motivo": f"no se pudo abrir: {e}"}); continue
        cajas = C.leer_txt_yolo(a.caja[i]) if a.caja is not None else cajas_detector(det, img)
        sel = C.elegir_grupa(cajas)
        if sel is None:
            rechazadas.append({"foto": f, "motivo": "no se detectó la grupa o no está dentro de la vaca"}); continue
        if sel[0] < conf_min:
            rechazadas.append({"foto": f, "motivo": f"confianza de la grupa {sel[0]:.2f} < {conf_min}"}); continue
        crops.append(C.recortar(img, sel[1])); usadas.append(f)
    res = {"vaca": a.vaca, "n_fotos_usadas": len(usadas), "fotos_usadas": usadas, "fotos_rechazadas": rechazadas}
    if not crops:
        res.update({"bcs": None, "bcs_continuo": None, "rango": None, "alerta_flaca": None, "error": "ninguna foto válida"})
        print(json.dumps(res, ensure_ascii=False, indent=1)); return
    torch.set_num_threads(os.cpu_count())
    X = C.features(C.cargar_dinov2(a.pesos), crops)
    p_fotos = joblib.load(os.path.join(MODELO, "cabeza.joblib")).predict(X)
    cal = cfg["calibracion"]; rd = cfg["redondeo"]
    cont = cal["a"] * float(p_fotos.mean()) + cal["b"]
    bcs = float(np.clip(np.round(cont / rd["paso"]) * rd["paso"], rd["min"], rd["max"]))
    res.update({"bcs": bcs, "bcs_continuo": round(cont, 3), "rango": [bcs - 0.25, bcs + 0.25],
                "alerta_flaca": bool(cont <= cfg["alerta_flaca"]["umbral"] + 1e-9),
                "pred_por_foto_sin_calibrar": [round(float(v), 3) for v in p_fotos]})
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
