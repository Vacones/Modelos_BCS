"""Entrena el detector de vaca y grupa (YOLO26n) con el export de Roboflow 'Vacones BCS - Cajas v2 Aprobado'.
Usa solo el split train, la parada temprana se decide con valid y al final mide el mAP en valid y en test.

Uso: python entrenar_detector.py DATOS_YOLO [pesos_base]
  DATOS_YOLO: carpeta del export YOLO de Roboflow (train, valid y test con images/ y labels/,
              clases del export: 0 = ass, 1 = cow)
  pesos_base: yolo26n.pt por defecto (Ultralytics lo descarga). Si falla, usa yolo11n.pt
Guarda la corrida en ./runs y copia best.pt y metricas_detector.json a ../modelo/detector.
Necesita ultralytics y, para entrenar en un tiempo razonable, torch con CUDA (ver requirements.txt).
"""
import os, sys, json, shutil, tempfile
from ultralytics import YOLO
import ultralytics.data.dataset as _ds
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))


def _no_cache(prefix, path, x, version):
    x["version"] = version  # no se escribe labels.cache dentro de la carpeta del dataset


_ds.save_dataset_cache_file = _no_cache


def main():
    data = os.path.abspath(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else "yolo26n.pt"
    runs = os.path.join(AQUI, "runs"); os.makedirs(runs, exist_ok=True)
    yaml_path = os.path.join(runs, "cajas_v2.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {data.replace(os.sep, '/')}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                "nc: 2\nnames: ['ass', 'cow']\n")
    try:
        m = YOLO(base)
    except Exception as e:
        print("no se pudo cargar", base, e, "-> yolo11n.pt"); base = "yolo11n.pt"; m = YOLO(base)
    dev = 0 if torch.cuda.is_available() else "cpu"
    m.train(data=yaml_path, imgsz=640, epochs=200, patience=30, batch=16, device=dev,
            workers=2, seed=0, deterministic=True, project=runs, name="cajas_v2", exist_ok=True, plots=False)
    best = os.path.join(runs, "cajas_v2", "weights", "best.pt")
    res = {"base": base, "device": str(dev)}
    names = YOLO(best).names
    for split in ("val", "test"):
        r = YOLO(best).val(data=yaml_path, split=split, imgsz=640, device=dev, plots=False,
                           project=runs, name=f"eval_{split}", exist_ok=True)
        res[split] = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                      "por_clase_mAP50": {names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)},
                      "por_clase_mAP50_95": {names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}}
    dst = os.path.join(AQUI, "..", "modelo", "detector"); os.makedirs(dst, exist_ok=True)
    shutil.copy(best, os.path.join(dst, "best.pt"))
    res["names"] = names
    json.dump(res, open(os.path.join(dst, "metricas_detector.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
