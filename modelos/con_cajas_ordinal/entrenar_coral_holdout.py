"""Nueva corrida CORAL: entrena en train y selecciona únicamente en valid.

Usa las cajas anotadas del COCO aprobado. Nunca produce predicciones de test.
Los artefactos individuales se guardan solo en la carpeta privada indicada.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .preparar_coco import planificar_export


NIVELES = (2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75)


def nivel_bcs(valor: float) -> int:
    """Conserva cada nivel aprobado; no fusiona el 2.25 del histórico."""
    for indice, nivel in enumerate(NIVELES):
        if abs(float(valor) - nivel) < 1e-9:
            return indice
    raise ValueError(f"BCS fuera del catálogo aprobado: {valor}")


def preparar_ajuste(modelo, base, ruta_cabeza: Path | str, tipo_bn) -> int:
    """Carga la mejor cabeza y habilita block7/top, dejando BatchNorm fija."""
    modelo.load_weights(ruta_cabeza)
    base.trainable = True
    entrenables = 0
    for capa in base.layers:
        capa.trainable = ((capa.name.startswith("block7") or capa.name.startswith("top_"))
                          and not isinstance(capa, tipo_bn))
        entrenables += int(capa.trainable)
    if not entrenables:
        raise ValueError("No se habilitaron capas para ajuste fino")
    return entrenables


def separar_train_valid(registros: list[dict]) -> tuple[list[dict], list[dict]]:
    """Aísla test y evita que la misma vaca aparezca en train y valid."""
    train = [r for r in registros if r["particion"] == "train"]
    valid = [r for r in registros if r["particion"] == "valid"]
    if not train or not valid:
        raise ValueError("Se requieren fotos de train y valid")
    if {r["vaca"] for r in train} & {r["vaca"] for r in valid}:
        raise ValueError("Una vaca aparece en train y valid")
    for particion, filas in (("train", train), ("valid", valid)):
        bcs_por_vaca = defaultdict(set)
        for fila in filas:
            nivel_bcs(float(fila["bcs"]))
            bcs_por_vaca[fila["vaca"]].add(float(fila["bcs"]))
        if any(len(valores) != 1 for valores in bcs_por_vaca.values()):
            raise ValueError(f"BCS contradictorio en {particion}")
    return train, valid


def _sigmoide(valor: float) -> float:
    if valor >= 0:
        return 1 / (1 + math.exp(-valor))
    exp = math.exp(valor)
    return exp / (1 + exp)


def construir_salidas(valid: list[dict], logits) -> tuple[list[dict], list[dict]]:
    """Predicción CORAL por foto y por vaca, con promedio de probabilidades."""
    if len(valid) != len(logits):
        raise ValueError("Cantidad de logits distinta a fotos")
    fotos = []
    acumuladas = defaultdict(list)
    reales = {}
    for registro, salida in zip(valid, logits):
        if registro["particion"] != "valid":
            raise ValueError("Solo se aceptan fotos de valid")
        if len(salida) != 6:
            raise ValueError("CORAL requiere seis umbrales")
        if any(not math.isfinite(float(v)) for v in salida):
            raise ValueError("Logits no finitos")
        probabilidades = [_sigmoide(float(v)) for v in salida]
        if any(not math.isfinite(p) for p in probabilidades):
            raise ValueError("Predicción no finita")
        vaca = str(registro["vaca"])
        real = float(registro["bcs"])
        nivel_bcs(real)
        if vaca in reales and reales[vaca] != real:
            raise ValueError("BCS contradictorio en valid")
        reales[vaca] = real
        acumuladas[vaca].append(probabilidades)
        fotos.append({
            "particion": "valid", "clave": f"valid/{registro['nombre']}",
            "vaca": vaca, "bcs_real": real,
            "bcs_pred": 2.25 + .25 * sum(p > .5 for p in probabilidades),
            "bcs_cont": 2.25 + .25 * sum(probabilidades), "estado": "ok",
        })
    vacas = []
    for vaca in sorted(acumuladas):
        probabilidades = [statistics.mean(foto[k] for foto in acumuladas[vaca])
                          for k in range(6)]
        vacas.append({
            "particion": "valid", "vaca": vaca, "bcs_real": reales[vaca],
            "bcs_pred": 2.25 + .25 * sum(p > .5 for p in probabilidades),
            "bcs_cont": 2.25 + .25 * sum(probabilidades),
            "n_fotos": len(acumuladas[vaca]), "estado": "ok",
        })
    return fotos, vacas


def _imagenes(registros: list[dict]):
    import numpy as np
    from PIL import Image, ImageOps

    imagenes = []
    for fila in registros:
        with Image.open(fila["origen"]) as archivo:
            foto = ImageOps.exif_transpose(archivo).convert("RGB")
        ancho, alto = foto.size
        if (ancho, alto) != (fila["ancho"], fila["alto"]):
            raise ValueError("Dimensiones de foto distintas del COCO")
        cx, cy, w, h = fila["cajas"]["ass"]
        margen = .10
        x0 = max(0, round((cx - w * (.5 + margen)) * ancho))
        y0 = max(0, round((cy - h * (.5 + margen)) * alto))
        x1 = min(ancho, round((cx + w * (.5 + margen)) * ancho))
        y1 = min(alto, round((cy + h * (.5 + margen)) * alto))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Recorte ass vacío")
        recorte = foto.crop((x0, y0, x1, y1))
        factor = 224 / max(recorte.size)
        recorte = recorte.resize((max(1, round(recorte.width * factor)),
                                  max(1, round(recorte.height * factor))), Image.Resampling.BICUBIC)
        lienzo = Image.new("RGB", (224, 224), (0, 0, 0))
        lienzo.paste(recorte, ((224 - recorte.width) // 2,
                               (224 - recorte.height) // 2))
        imagenes.append(np.asarray(lienzo, dtype=np.float32))
    return np.stack(imagenes)


def _guardar_csv(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        raise ValueError("No hay predicciones para guardar")
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def _metricas(filas: list[dict]) -> dict:
    errores = [abs(f["bcs_pred"] - f["bcs_real"]) for f in filas]
    continuos = [abs(f["bcs_cont"] - f["bcs_real"]) for f in filas]
    return {
        "n": len(filas), "cobertura": 1.0,
        "mae_categoria": statistics.mean(errores),
        "mae_continuo": statistics.mean(continuos),
        "dentro_025": sum(e <= .25 for e in errores) / len(errores),
        "dentro_050": sum(e <= .50 for e in errores) / len(errores),
    }


def _huella(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def entrenar(datos: Path, salida: Path, epocas_cabeza: int, epocas_ajuste: int,
            paciencia: int, semilla: int, lote: int) -> dict:
    """Ajusta pesos solo con train y elige checkpoint mediante MAE de valid."""
    if epocas_cabeza < 1 or epocas_ajuste < 0 or paciencia < 1 or lote < 1:
        raise ValueError("Épocas, paciencia o lote inválidos")
    registros, resumen = planificar_export(datos)
    train, valid = separar_train_valid(registros)
    salida = Path(salida).expanduser().resolve()
    if salida.exists():
        raise FileExistsError(f"El destino ya existe: {salida}")
    if salida.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("La salida privada no debe quedar dentro del repositorio")

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import numpy as np
    import keras
    from keras import layers, ops

    keras.utils.set_random_seed(semilla)
    x_train, x_valid = _imagenes(train), _imagenes(valid)
    y_train = np.array([nivel_bcs(r["bcs"]) for r in train], dtype=np.int32)
    cuentas = Counter(r["vaca"] for r in train)
    pesos = np.array([len(train) / (len(cuentas) * cuentas[r["vaca"]]) for r in train],
                     dtype=np.float32)
    previos = np.array([(y_train > k).mean() for k in range(6)]).clip(.02, .98)
    sesgos = np.log(previos / (1 - previos))

    class SesgosCoral(layers.Layer):
        def build(self, forma):
            self.b = self.add_weight(shape=(6,), initializer=keras.initializers.Constant(sesgos),
                                     name="sesgos")

        def call(self, entrada):
            return entrada + self.b

    def perdida_coral(y_real, logits):
        nivel = ops.cast(ops.reshape(y_real, (-1, 1)), "float32")
        objetivo = ops.cast(nivel > ops.arange(6, dtype="float32")[None, :], "float32")
        log_sig = ops.log_sigmoid(logits)
        return -ops.mean(objetivo * log_sig + (1 - objetivo) * (log_sig - logits), axis=1)

    entrada = keras.Input((224, 224, 3), name="imagen")
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                              input_tensor=entrada)
    base.trainable = False
    h = layers.GlobalAveragePooling2D()(base.output)
    h = layers.Dropout(.3, seed=semilla)(h)
    logits = SesgosCoral(name="salida")(layers.Dense(1, use_bias=False, name="coral_peso")(h))
    modelo = keras.Model(entrada, logits, name="b0_coral_holdout")

    salida.mkdir(parents=True)
    class MejorVaca(keras.callbacks.Callback):
        def __init__(self, ruta: Path):
            super().__init__()
            self.ruta = ruta
            self.mejor = math.inf
            self.epoca = None

        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict(x_valid, batch_size=lote, verbose=0)
            _, vacas = construir_salidas(valid, pred)
            mae = _metricas(vacas)["mae_continuo"]
            logs["val_mae_vaca"] = mae
            if mae < self.mejor - 1e-12:
                self.mejor, self.epoca = mae, epoch + 1
                self.model.save_weights(self.ruta)

    def fase(nombre: str, epocas: int, lr: float):
        ruta = salida / f"{nombre}_best.weights.h5"
        callback = MejorVaca(ruta)
        modelo.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
                       loss=perdida_coral)
        historia = modelo.fit(x_train, y_train, sample_weight=pesos, batch_size=lote,
                              epochs=epocas, shuffle=True, verbose=1,
                              callbacks=[callback, keras.callbacks.EarlyStopping(
                                  monitor="val_mae_vaca", mode="min", patience=paciencia)])
        modelo.load_weights(ruta)
        return {"ruta": ruta, "mae": callback.mejor, "mejor_epoca": callback.epoca,
                "epocas_ejecutadas": len(historia.epoch)}

    cabeza = fase("cabeza", epocas_cabeza, 1e-3)
    ajuste = None
    if epocas_ajuste:
        preparar_ajuste(modelo, base, cabeza["ruta"], layers.BatchNormalization)
        ajuste = fase("ajuste", epocas_ajuste, 1e-5)
    elegido = min((r for r in (cabeza, ajuste) if r is not None), key=lambda r: r["mae"])
    modelo.load_weights(elegido["ruta"])
    fotos, vacas = construir_salidas(valid, modelo.predict(x_valid, batch_size=lote, verbose=0))
    _guardar_csv(salida / "predicciones_val_foto.csv", fotos)
    _guardar_csv(salida / "predicciones_val_vaca.csv", vacas)
    metadatos = {
        "protocolo": "train_holdout_valid", "cajas": "anotadas_coco",
        "catalogo_bcs": list(NIVELES),
        "test_evaluado": False, "semilla": semilla, "lote": lote,
        "epocas_cabeza": epocas_cabeza, "epocas_ajuste": epocas_ajuste,
        "paciencia": paciencia, "seleccion": "MAE continuo por vaca en valid",
        "checkpoint_elegido": elegido["ruta"].name,
        "fases": {"cabeza": {k: v for k, v in cabeza.items() if k != "ruta"},
                  "ajuste": None if ajuste is None else {k: v for k, v in ajuste.items() if k != "ruta"}},
        "conteos": resumen,
        "huellas_coco": {p: _huella(Path(datos) / p / "_annotations.coco.json")
                         for p in ("train", "valid", "test")},
        "huella_codigo": _huella(Path(__file__)),
        "keras": keras.__version__,
    }
    (salida / "ejecucion.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))
    metricas = {"foto": _metricas(fotos), "vaca": _metricas(vacas)}
    (salida / "metricas_val.json").write_text(json.dumps(metricas, indent=2))
    return {"checkpoint": str(elegido["ruta"]), "metricas_val": metricas,
            "test_evaluado": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", required=True, type=Path, help="COCO v2 aprobado")
    parser.add_argument("--salida", type=Path, help="Carpeta privada nueva")
    parser.add_argument("--solo-validar", action="store_true", help="Comprueba datos sin entrenar")
    parser.add_argument("--epochs-head", type=int, default=30)
    parser.add_argument("--epochs-finetune", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    registros, resumen = planificar_export(args.datos)
    train, valid = separar_train_valid(registros)
    if args.solo_validar:
        print(json.dumps({"train": len(train), "valid": len(valid),
                          "vacas_train": len({r['vaca'] for r in train}),
                          "vacas_valid": len({r['vaca'] for r in valid}),
                          "test_evaluado": False, "conteos": resumen}, ensure_ascii=False))
        return
    if args.salida is None:
        parser.error("Indica --salida o usa --solo-validar")
    print(json.dumps(entrenar(args.datos, args.salida, args.epochs_head,
                             args.epochs_finetune, args.patience, args.seed, args.batch),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
