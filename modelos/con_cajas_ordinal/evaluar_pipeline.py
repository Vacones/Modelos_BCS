"""Evalúa CORAL en validación con las cajas predichas del detector compartido.

No vuelve a entrenar, no lee imágenes de test y guarda predicciones solo en
una carpeta privada. Las abstenciones permanecen en el denominador global.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from modelos.con_cajas_yolo.vacas.pairing import pair_boxes
from . import entrenar_coral_holdout
from .entrenar_coral_holdout import (
    _huella, _imagenes, construir_salidas, separar_train_valid,
)
from .preparar_coco import planificar_export


def _modelo_para_pesos(keras, semilla: int):
    """Topología de la corrida registrada, sin descargar pesos ImageNet."""
    from keras import layers

    class SesgosCoral(layers.Layer):
        def build(self, forma):
            self.b = self.add_weight(shape=(6,), initializer="zeros", name="sesgos")

        def call(self, entrada):
            return entrada + self.b

    entrada = keras.Input((224, 224, 3), name="imagen")
    base = keras.applications.EfficientNetB0(include_top=False, weights=None,
                                              input_tensor=entrada)
    base.trainable = False
    h = layers.GlobalAveragePooling2D()(base.output)
    h = layers.Dropout(.3, seed=semilla)(h)
    logits = SesgosCoral(name="salida")(
        layers.Dense(1, use_bias=False, name="coral_peso")(h))
    return keras.Model(entrada, logits, name="b0_coral_holdout")


def preparar_validacion(registros: list[dict], predicciones: list[dict]):
    """Asocia exactamente cada foto valid con su detección, sin leer test."""
    if any(r["particion"] != "valid" for r in registros):
        raise ValueError("Solo se aceptan registros de validación")
    nombres = [r["nombre"] for r in registros]
    if len(nombres) != len(set(nombres)):
        raise ValueError("Nombres duplicados en validación")
    por_nombre = {}
    for fila in predicciones:
        if fila.get("split") != "val":
            raise ValueError("Las predicciones deben pertenecer solo a la partición val")
        nombre = fila.get("file_name")
        if nombre in por_nombre:
            raise ValueError("Predicción duplicada")
        por_nombre[nombre] = fila
    if set(por_nombre) != set(nombres):
        raise ValueError("Faltan predicciones o hay fotos extra en validación")

    cubiertas, todas = [], []
    for registro in registros:
        fila = por_nombre[registro["nombre"]]
        cajas = fila.get("boxes")
        if not isinstance(cajas, list):
            raise ValueError("Faltan cajas de detección")
        for caja in cajas:
            coordenadas = caja.get("xyxy")
            if caja.get("class") not in (0, 1) or not isinstance(coordenadas, list) or len(coordenadas) != 4:
                raise ValueError("Caja predicha inválida")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in coordenadas):
                raise ValueError("Coordenadas predichas inválidas")
            x0, y0, x1, y1 = coordenadas
            if not (0 <= x0 < x1 <= registro["ancho"] and
                    0 <= y0 < y1 <= registro["alto"]):
                raise ValueError("Caja predicha fuera de la foto")
        estado, par = pair_boxes(cajas)
        if fila.get("status") != estado:
            raise ValueError("Estado de asociación incompatible con las cajas")
        salida = {
            "particion": "valid", "clave": f"valid/{registro['nombre']}",
            "vaca": str(registro["vaca"]), "bcs_real": float(registro["bcs"]),
            "bcs_pred": "", "bcs_cont": "", "estado": estado,
        }
        if estado == "ok":
            x0, y0, x1, y1 = par[1]["xyxy"]
            ancho, alto = registro["ancho"], registro["alto"]
            normalizada = ((x0 + x1) / (2 * ancho), (y0 + y1) / (2 * alto),
                           (x1 - x0) / ancho, (y1 - y0) / alto)
            cubiertas.append({**registro, "cajas": {"ass": normalizada}})
        todas.append(salida)
    return cubiertas, todas


def _metricas(fotos: list[dict], vacas_nativas: list[dict]) -> dict:
    cubiertas = [r for r in fotos if r["estado"] == "ok"]
    vacas_total = len({r["vaca"] for r in fotos})
    if not cubiertas or not vacas_nativas:
        raise ValueError("No hay predicciones válidas para calcular MAE")

    def errores(filas, campo):
        return [abs(float(r[campo]) - float(r["bcs_real"])) for r in filas]

    agrupar = defaultdict(list)
    for fila in cubiertas:
        agrupar[fila["vaca"]].append(fila)
    vacas_comunes = []
    for vaca, filas in agrupar.items():
        vacas_comunes.append({
            "vaca": vaca, "bcs_real": filas[0]["bcs_real"],
            "bcs_pred": statistics.mean(float(f["bcs_pred"]) for f in filas),
            "bcs_cont": statistics.mean(float(f["bcs_cont"]) for f in filas),
        })
    cat, cont = errores(cubiertas, "bcs_pred"), errores(cubiertas, "bcs_cont")
    return {
        "fotos_total": len(fotos), "fotos_cubiertas": len(cubiertas),
        "cobertura_fotos": len(cubiertas) / len(fotos),
        "abstenciones": dict((estado, sum(r["estado"] == estado for r in fotos))
                             for estado in ("asociacion_ambigua", "sin_par_valido")),
        "mae_categoria_fotos_cubiertas": statistics.mean(cat),
        "mae_continuo_fotos_cubiertas": statistics.mean(cont),
        "exactos_categoria_global": sum(e < 1e-9 for e in cat),
        "dentro_025_categoria_global": sum(e <= .25 + 1e-9 for e in cat),
        "dentro_050_categoria_global": sum(e <= .50 + 1e-9 for e in cat),
        "vacas_total": vacas_total, "vacas_cubiertas": len(vacas_nativas),
        "cobertura_vacas": len(vacas_nativas) / vacas_total,
        "mae_categoria_vaca_nativa": statistics.mean(errores(vacas_nativas, "bcs_pred")),
        "mae_continuo_vaca_nativa": statistics.mean(errores(vacas_nativas, "bcs_cont")),
        "mae_categoria_vaca_media_fotos": statistics.mean(errores(vacas_comunes, "bcs_pred")),
        "mae_continuo_vaca_media_fotos": statistics.mean(errores(vacas_comunes, "bcs_cont")),
    }


def _csv(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def evaluar(datos: Path, entrenamiento: Path, cajas: Path, salida: Path, lote: int,
            solo_validar: bool) -> dict:
    datos, entrenamiento, cajas, salida = (
        Path(p).expanduser().resolve() for p in (datos, entrenamiento, cajas, salida)
    )
    if lote < 1:
        raise ValueError("El lote debe ser positivo")
    if salida.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("La salida privada debe estar fuera del repositorio")
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    registros, _ = planificar_export(datos)
    _, valid = separar_train_valid(registros)
    ejecucion = json.loads((entrenamiento / "ejecucion.json").read_text(encoding="utf-8"))
    if ejecucion.get("protocolo") != "train_holdout_valid" or ejecucion.get("test_evaluado") is not False:
        raise ValueError("Entrenamiento incompatible con este protocolo")
    if ejecucion.get("huella_codigo") != _huella(Path(entrenar_coral_holdout.__file__)):
        raise ValueError("El código de entrenamiento difiere del registrado en la corrida")
    for particion in ("train", "valid", "test"):
        ruta = datos / particion / "_annotations.coco.json"
        if _huella(ruta) != ejecucion["huellas_coco"][particion]:
            raise ValueError(f"COCO de {particion} distinto al usado en entrenamiento")
    checkpoint = entrenamiento / ejecucion["checkpoint_elegido"]
    if not checkpoint.is_file():
        raise FileNotFoundError("Falta el checkpoint seleccionado")
    ajustes = json.loads((cajas / "settings.json").read_text(encoding="utf-8"))
    if ajustes.get("splits") != ["val"] or ajustes.get("size") != 224:
        raise ValueError("Las cajas deben corresponder solo a val y tamaño 224")
    predicciones = [json.loads(line) for line in
                    (cajas / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    cubiertas, todas = preparar_validacion(valid, predicciones)
    resumen = {
        "estado": "validado", "fotos_valid": len(todas),
        "fotos_con_par": len(cubiertas), "test_evaluado": False,
        "detector_sha256": ajustes["weights_sha256"],
        "checkpoint_sha256": _huella(checkpoint),
    }
    if solo_validar:
        return resumen

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras

    modelo = _modelo_para_pesos(keras, ejecucion["semilla"])
    modelo.load_weights(checkpoint)
    imagenes = _imagenes(cubiertas)
    logits = modelo.predict(imagenes, batch_size=lote, verbose=0)
    fotos_cubiertas, vacas = construir_salidas(cubiertas, logits)
    por_clave = {r["clave"]: r for r in fotos_cubiertas}
    for fila in todas:
        if fila["estado"] == "ok":
            pred = por_clave[fila["clave"]]
            fila["bcs_pred"], fila["bcs_cont"] = pred["bcs_pred"], pred["bcs_cont"]
    metricas = _metricas(todas, vacas)
    salida.mkdir(parents=True)
    _csv(salida / "predicciones_val_foto.csv", todas)
    _csv(salida / "predicciones_val_vaca_cubierta.csv", vacas)
    (salida / "metricas_val.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    (salida / "ejecucion.json").write_text(json.dumps({
        **resumen, "modelo": "CORAL ordinal, cajas predichas",
        "entrenamiento": str(entrenamiento), "cajas": str(cajas),
        "regla_vaca_nativa": "media de probabilidades ordinales y umbral 0.5",
        "regla_vaca_comun": "media de BCS predichos por fotografía",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**resumen, "metricas_val": metricas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", required=True, type=Path)
    parser.add_argument("--entrenamiento", required=True, type=Path)
    parser.add_argument("--cajas", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--solo-validar", action="store_true")
    args = parser.parse_args()
    resultado = evaluar(args.datos, args.entrenamiento, args.cajas,
                        args.salida, args.batch, args.solo_validar)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
