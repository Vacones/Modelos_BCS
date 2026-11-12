"""Comparación privada por vaca; el resultado agregado no contiene identidades."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Registro:
    modelo: str
    foto: str
    vaca: str
    real: float
    predicho: float | None


def _validar(registros: list[Registro]) -> dict[str, Registro]:
    if not registros:
        raise ValueError("No hay fotografías evaluables")
    if len({r.modelo for r in registros}) != 1:
        raise ValueError("Se mezclaron modelos")
    fotos: dict[str, Registro] = {}
    etiquetas: dict[str, float] = {}
    for r in registros:
        if not r.foto or not r.vaca:
            raise ValueError("Falta identidad de fotografía o vaca")
        if r.foto in fotos:
            raise ValueError("Fotografía duplicada")
        if not math.isfinite(r.real) or (r.predicho is not None and not math.isfinite(r.predicho)):
            raise ValueError("BCS no finito")
        if r.vaca in etiquetas and not math.isclose(etiquetas[r.vaca], r.real, abs_tol=1e-9):
            raise ValueError("Una vaca tiene BCS real inconsistente")
        fotos[r.foto] = r
        etiquetas[r.vaca] = r.real
    return fotos


def _agregar(registros: list[Registro]) -> tuple[dict, dict[str, float]]:
    grupos: dict[str, list[Registro]] = defaultdict(list)
    for r in registros:
        grupos[r.vaca].append(r)
    errores = {}
    fotos_cubiertas = 0
    for vaca, filas in grupos.items():
        predichos = [r.predicho for r in filas if r.predicho is not None]
        fotos_cubiertas += len(predichos)
        if predichos:
            errores[vaca] = abs(mean(predichos) - filas[0].real)
    total = len(grupos)
    return {
        "vacas_elegibles": total,
        "vacas_cubiertas": len(errores),
        "cobertura_vacas": len(errores) / total,
        "fotos_elegibles": len(registros),
        "fotos_cubiertas": fotos_cubiertas,
        "cobertura_fotos": fotos_cubiertas / len(registros),
        "mae_vaca": mean(errores.values()) if errores else None,
        "dentro_025_vaca": sum(e <= 0.25 + 1e-9 for e in errores.values()) / len(errores) if errores else None,
        "dentro_050_vaca": sum(e <= 0.50 + 1e-9 for e in errores.values()) / len(errores) if errores else None,
    }, errores


def comparar(a: list[Registro], b: list[Registro]) -> dict:
    """Exige las mismas fotos, vacas y referencias antes de comparar errores."""
    fotos_a, fotos_b = _validar(a), _validar(b)
    if set(fotos_a) != set(fotos_b):
        raise ValueError("Las fotografías elegibles no coinciden")
    for clave in fotos_a:
        x, y = fotos_a[clave], fotos_b[clave]
        if x.vaca != y.vaca:
            raise ValueError("Identidad de vaca no coincide")
        if not math.isclose(x.real, y.real, abs_tol=1e-9):
            raise ValueError("BCS real no coincide")
    resumen_a, errores_a = _agregar(a)
    resumen_b, errores_b = _agregar(b)
    fotos_comunes = [clave for clave in fotos_a
                     if fotos_a[clave].predicho is not None and fotos_b[clave].predicho is not None]
    errores_pares_a = _agregar([fotos_a[k] for k in fotos_comunes])[1] if fotos_comunes else {}
    errores_pares_b = _agregar([fotos_b[k] for k in fotos_comunes])[1] if fotos_comunes else {}
    comunes = set(errores_pares_a) & set(errores_pares_b)
    return {
        "a": resumen_a,
        "b": resumen_b,
        "pareado": {
            "vacas": len(comunes),
            "fotos": len(fotos_comunes),
            "mae_a": mean(errores_pares_a[v] for v in comunes) if comunes else None,
            "mae_b": mean(errores_pares_b[v] for v in comunes) if comunes else None,
            "diferencia_mae_b_menos_a": mean(errores_pares_b[v] - errores_pares_a[v] for v in comunes) if comunes else None,
        },
    }
