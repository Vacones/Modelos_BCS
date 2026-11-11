"""Gráficas descriptivas de agregados públicos de validación, sin datos individuales."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def validar_fila(fila: dict[str, str]) -> dict:
    salida = fila["salida"]
    if salida not in {"categorica", "continua"}:
        raise ValueError("Tipo de salida desconocido")
    if fila["evidencia"] not in {"documentado", "recalculado_local"}:
        raise ValueError("Estado de evidencia desconocido")

    resultado = dict(fila)
    for campo in (
        "elegibles_fotos", "clasificadas_fotos", "aciertos_025_global",
        "aciertos_050_global",
    ):
        resultado[campo] = int(fila[campo])
    exactos = fila["aciertos_exactos_global"]
    resultado["aciertos_exactos_global"] = int(exactos) if exactos else None
    mae = fila["mae_clasificadas_foto"]
    resultado["mae_clasificadas_foto"] = float(mae) if mae else None

    total = resultado["elegibles_fotos"]
    cubiertas = resultado["clasificadas_fotos"]
    if total <= 0 or not 0 <= cubiertas <= total:
        raise ValueError("Denominadores imposibles")
    aciertos = [resultado["aciertos_025_global"], resultado["aciertos_050_global"]]
    if resultado["aciertos_exactos_global"] is not None:
        aciertos.insert(0, resultado["aciertos_exactos_global"])
    if any(x < 0 or x > cubiertas for x in aciertos):
        raise ValueError("Los aciertos superan las predicciones emitidas")
    if aciertos != sorted(aciertos):
        raise ValueError("Las tolerancias no son monótonas")
    if salida == "continua" and resultado["aciertos_exactos_global"] is not None:
        raise ValueError("Acierto exacto no definido para esta salida continua")
    if resultado["mae_clasificadas_foto"] is not None:
        if not math.isfinite(resultado["mae_clasificadas_foto"]) or resultado["mae_clasificadas_foto"] < 0:
            raise ValueError("MAE inválido")
    return resultado


def cargar_agregados(ruta: Path) -> list[dict]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        filas = [validar_fila(fila) for fila in csv.DictReader(archivo)]
    ids = [fila["id"] for fila in filas]
    if len(ids) != len(set(ids)):
        raise ValueError("ID de candidato duplicado")
    if not filas:
        raise ValueError("No hay candidatos")
    return filas


def dibujar(filas: list[dict], destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destino.mkdir(parents=True, exist_ok=True)
    nombres = [fila["modelo"] for fila in filas]
    y = np.arange(len(filas))
    cobertura = [fila["clasificadas_fotos"] / fila["elegibles_fotos"] for fila in filas]
    tolerancia = [fila["aciertos_025_global"] / fila["elegibles_fotos"] for fila in filas]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.barh(y - .19, cobertura, height=.34, color="#287a75", label="Cobertura")
    ax.barh(y + .19, tolerancia, height=.34, color="#c47830", label="Dentro de ±0.25, global")
    for i, fila in enumerate(filas):
        ax.text(cobertura[i] + .01, i - .19,
                f"{fila['clasificadas_fotos']}/{fila['elegibles_fotos']}", va="center", fontsize=8)
        ax.text(tolerancia[i] + .01, i + .19,
                f"{fila['aciertos_025_global']}/{fila['elegibles_fotos']}", va="center", fontsize=8)
    ax.set_yticks(y, nombres)
    ax.set_xlim(0, 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Proporción de fotos elegibles")
    ax.set_title("Validación: cobertura y tolerancia global por fotografía")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    fig.text(.02, .01,
             "Cifras descriptivas; ResNet y regresión recalculados localmente, otros documentados. "
             "La regresión usa salida continua; los demás, categórica.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(destino / "cobertura_y_tolerancia_val.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, fila in enumerate(filas):
        mae = fila["mae_clasificadas_foto"]
        color = "#b56a25" if fila["salida"] == "continua" else "#406ca1"
        if fila["id"] == "constante":
            color = "#888888"
        if mae is None:
            ax.text(.01, i, "sin MAE documentado para esta ejecución", va="center", fontsize=9)
            continue
        ax.barh(i, mae, height=.66, color=color)
        ax.text(mae + .01, i, f"{mae:.3f}  ·  n={fila['clasificadas_fotos']}", va="center", fontsize=9)
    ax.set_yticks(y, nombres)
    ax.set_xlim(0, .7)
    ax.invert_yaxis()
    ax.set_xlabel("MAE BCS entre fotos clasificadas")
    ax.set_title("Validación: error condicionado y denominadores")
    ax.grid(axis="x", alpha=.2)
    fig.text(.02, .01,
             "Salidas continuas y categóricas se muestran juntas solo para inventario; "
             "no constituyen una clasificación de modelos. Leer con cobertura y fuente.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(destino / "mae_condicionado_val.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path,
                        default=Path(__file__).with_name("datos_agregados_validacion.csv"))
    parser.add_argument("--salida", type=Path,
                        default=Path(__file__).with_name("figuras"))
    args = parser.parse_args()
    dibujar(cargar_agregados(args.datos), args.salida)
    print(f"Gráficas guardadas en {args.salida}")


if __name__ == "__main__":
    main()
