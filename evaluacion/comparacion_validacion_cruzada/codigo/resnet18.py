"""ResNet18 con la foto completa (modelos/sin_cajas_resnet) con el protocolo común de 66 vacas.

Usa las funciones de esa carpeta sin cambios: letterbox a 224, aumentos, muestreo con peso 1 / fotos por vaca,
AdamW con lr 1e-4, hasta 30 épocas, paciencia 1, checkpoint por MAE por vaca de validación, semilla 20261026,
lote 16 y el dispositivo que elige select_device() (MPS o CPU).
La predicción por foto es la clase de mayor probabilidad y la de la vaca es la media de sus fotos.

Uso: python resnet18.py [REPETICIONES...]   (por defecto de 0 a 9)
"""
import os, sys, time, json
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import protocolo as PR  # noqa: E402
sys.path.insert(0, os.path.join(PR.RAIZ_REPO, "modelos", "sin_cajas_resnet"))
from pathlib import Path  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from modelo_sin_cajas.datos import CLASS_VALUES, ImageRecord, validate_records  # noqa: E402
from modelo_sin_cajas.imagenes import CowImageDataset, build_train_sampler, build_transforms  # noqa: E402
from modelo_sin_cajas.entrenar import fit_model, predict_loader, set_seed  # noqa: E402
from modelo_sin_cajas.modelo import build_model, select_device  # noqa: E402

NOMBRE = "resnet18_foto_completa"
SEED, BATCH, SIZE, EPOCHS, LR, PATIENCE = 20261026, 16, 224, 30, 1e-4, 1


def correr(rep):
    t0 = time.time()
    df = PR.tabla(rep)
    tmp = os.path.join(PR.PRIV, "tmp_resnet18", f"rep{rep}"); os.makedirs(tmp, exist_ok=True)
    pred = np.full(len(df), np.nan); info = []
    for f in range(PR.N_PLIEGUES):
        tf = time.time()
        te = np.where(df.pliegue.values == f)[0]; tr = np.where(df.pliegue.values != f)[0]
        fit, val = PR.separar_val(df, tr, rep)
        conj = np.empty(len(df), object); conj[fit] = "ENTRENAMIENTO"; conj[val] = "VALIDACION"; conj[te] = "PRUEBA"
        recs = [ImageRecord(imagen_id=r.file_name, vaca_id=r.cow, bcs=r.bcs, conjunto=c, ruta_imagen=Path(r.imagen))
                for r, c in zip(df.itertuples(), conj)]
        validate_records(recs)
        R = {k: [x for x in recs if x.conjunto == k] for k in ("ENTRENAMIENTO", "VALIDACION", "PRUEBA")}
        set_seed(SEED)
        common = {"batch_size": BATCH, "num_workers": 0}
        tr_l = DataLoader(CowImageDataset(R["ENTRENAMIENTO"], build_transforms(True, SIZE)),
                          sampler=build_train_sampler(R["ENTRENAMIENTO"], seed=SEED), **common)
        va_l = DataLoader(CowImageDataset(R["VALIDACION"], build_transforms(False, SIZE)), shuffle=False, **common)
        te_l = DataLoader(CowImageDataset(R["PRUEBA"], build_transforms(False, SIZE)), shuffle=False, **common)
        device = select_device()
        model = build_model(num_classes=len(CLASS_VALUES), pretrained=True)
        hist = fit_model(model, tr_l, va_l, device, EPOCHS, LR, PATIENCE, Path(tmp) / f"best_f{f}.pt")
        out = predict_loader(model, te_l, device)
        cls = np.asarray(CLASS_VALUES)[out["probabilities"].argmax(1)]
        pos = {n: i for i, n in zip(te, df.file_name.values[te])}
        assert len(pos) == len(te)
        for n, v in zip(out["image_ids"], cls):
            pred[pos[n]] = v
        best = min(hist, key=lambda h: (h["validation_mean_absolute_error"], -h["validation_exact_accuracy"]))["epoch"]
        info.append({"pliegue": f, "epocas": len(hist), "mejor_epoca": best, "segundos": time.time() - tf,
                     "fotos_fit": len(fit), "fotos_val": len(val), "fotos_test": len(te), "device": str(device)})
        print(json.dumps(info[-1]), flush=True)
    assert not np.isnan(pred).any()
    P = PR.guardar_rep(NOMBRE, rep, df, pred)
    mae = float(np.abs(P - PR.por_vaca(df, pred)[1]).mean())
    json.dump({"rep": rep, "pliegues": info, "segundos_total": time.time() - t0, "mae_vaca": mae},
              open(os.path.join(PR.PARC, f"{NOMBRE}_rep{rep}_log.json"), "w"), indent=1)
    print("repetición", rep, "MAE por vaca", round(mae, 4), "segundos", round(time.time() - t0))


if __name__ == "__main__":
    PR.correr_repeticiones(NOMBRE, correr, [int(a) for a in sys.argv[1:]] or list(range(10)))
