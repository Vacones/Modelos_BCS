"""Train the rump/cow detector (YOLO26n) on the Roboflow 'Vacones BCS - Cajas v2 Aprobado' export:
train split only, early stopping on valid, then mAP on valid and test.

Usage: python entrenar_detector.py DATA_YOLO [base_weights]
  DATA_YOLO: folder of the Roboflow YOLO export (train/valid/test with images/ and labels/,
             classes in the export: 0 = ass, 1 = cow)
  base_weights: yolo26n.pt by default (Ultralytics downloads it), falls back to yolo11n.pt
Writes the run to .\\runs and copies best.pt + metricas_detector.json to ..\\modelo\\detector.
Needs ultralytics and, to train in reasonable time, a CUDA build of torch (see requirements.txt).
"""
import os, sys, json, shutil, tempfile
from ultralytics import YOLO
import ultralytics.data.dataset as _ds
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))


def _no_cache(prefix, path, x, version):
    x["version"] = version  # keep the in-memory contract, never write labels.cache into the dataset folder


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
