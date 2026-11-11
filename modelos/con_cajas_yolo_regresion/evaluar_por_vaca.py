"""Métricas BCS por vaca a partir de predicciones privadas por fotografía."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Registro:
    vaca: str
    real: float
    base: float
    regresor_gt: float
    pipeline: float | None


def _metricas(grupos: dict[str, list[Registro]], campo: str) -> dict:
    errores = []
    for registros in grupos.values():
        real = registros[0].real
        predicho = mean(getattr(registro, campo) for registro in registros)
        errores.append(abs(predicho - real))
    return {
        "vacas": len(errores),
        "mae": mean(errores) if errores else None,
        "error_hasta_025": (
            sum(error <= 0.25 for error in errores) / len(errores)
            if errores else None
        ),
    }


def resumir_por_vaca(registros: list[Registro]) -> dict:
    if not registros:
        raise ValueError("No hay fotografías evaluables")

    por_vaca: dict[str, list[Registro]] = defaultdict(list)
    for registro in registros:
        if not registro.vaca:
            raise ValueError("Falta identidad de vaca")
        valores = (registro.real, registro.base, registro.regresor_gt)
        if registro.pipeline is not None:
            valores += (registro.pipeline,)
        if not all(math.isfinite(valor) for valor in valores):
            raise ValueError("Hay valores BCS no finitos")
        por_vaca[registro.vaca].append(registro)

    for grupo in por_vaca.values():
        if len({registro.real for registro in grupo}) != 1:
            raise ValueError("Una vaca tiene BCS distintos entre fotografías")

    cubiertas = {
        vaca: [registro for registro in grupo if registro.pipeline is not None]
        for vaca, grupo in por_vaca.items()
    }
    cubiertas = {vaca: grupo for vaca, grupo in cubiertas.items() if grupo}
    fotos_cubiertas = sum(len(grupo) for grupo in cubiertas.values())

    return {
        "unidad": "vaca",
        "regla_agregacion": "media de predicciones por foto; cada vaca pesa igual",
        "vacas_total": len(por_vaca),
        "vacas_cubiertas": len(cubiertas),
        "cobertura_vacas": len(cubiertas) / len(por_vaca),
        "fotos_total": len(registros),
        "fotos_cubiertas": fotos_cubiertas,
        "cobertura_fotos": fotos_cubiertas / len(registros),
        "todas_fotos": {
            "base": _metricas(por_vaca, "base"),
            "regresor_gt": _metricas(por_vaca, "regresor_gt"),
        },
        "mismas_fotos": {
            "base": _metricas(cubiertas, "base"),
            "regresor_gt": _metricas(cubiertas, "regresor_gt"),
            "pipeline": _metricas(cubiertas, "pipeline"),
        },
    }


def cargar_validacion(datos: Path, predicciones: Path) -> list[Registro]:
    metadatos = json.loads(datos.read_text(encoding="utf-8"))
    elegibles = {}
    for fila in metadatos:
        if fila["particion"] != "val" or fila["bcs"] is None:
            continue
        clave = fila["clave"]
        if clave in elegibles:
            raise ValueError("Metadatos duplicados en validación")
        elegibles[clave] = fila
    if not elegibles:
        raise ValueError("No hay metadatos de validación elegibles")

    registros = []
    claves_vistas = set()
    with predicciones.open(newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            clave = fila["clave"]
            if clave not in elegibles or clave in claves_vistas:
                raise ValueError("Predicción duplicada o ajena a validación")
            claves_vistas.add(clave)
            meta = elegibles[clave]
            real = float(fila["real"])
            if not math.isclose(real, float(meta["bcs"]), abs_tol=1e-9):
                raise ValueError("BCS de predicción y metadatos no coinciden")
            if meta["vaca"] is None:
                raise ValueError("Falta identidad única de vaca")
            registros.append(Registro(
                vaca=str(meta["vaca"]),
                real=real,
                base=float(fila["base"]),
                regresor_gt=float(fila["regresor_gt"]),
                pipeline=float(fila["pipeline"]) if fila["pipeline"] else None,
            ))

    if claves_vistas != set(elegibles):
        raise ValueError("Faltan predicciones de fotografías de validación")
    return registros


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume en privado las métricas BCS de validación por vaca."
    )
    parser.add_argument("--datos", required=True, type=Path)
    parser.add_argument("--predicciones", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    args = parser.parse_args()

    resumen = resumir_por_vaca(cargar_validacion(args.datos, args.predicciones))
    resumen["particion"] = "val"
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
