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


def validar_pareados(filas: list[dict[str, str]]) -> list[dict]:
    if len(filas) < 2 or len({r["id"] for r in filas}) != len(filas):
        raise ValueError("Se requieren al menos dos modelos pareados distintos")
    resultado = []
    for fila in filas:
        fotos, vacas = int(fila["fotos_pareadas"]), int(fila["vacas_pareadas"])
        mae_foto = float(fila["mae_foto_categoria"])
        mae_vaca = float(fila["mae_vaca_media_fotos"])
        if fotos <= 0 or vacas <= 0 or vacas > fotos:
            raise ValueError("Denominadores pareados inválidos")
        if any(not math.isfinite(v) or v < 0 for v in (mae_foto, mae_vaca)):
            raise ValueError("MAE pareado inválido")
        resultado.append({**fila, "fotos_pareadas": fotos, "vacas_pareadas": vacas,
                          "mae_foto_categoria": mae_foto,
                          "mae_vaca_media_fotos": mae_vaca})
    if len({(r["fotos_pareadas"], r["vacas_pareadas"]) for r in resultado}) != 1:
        raise ValueError("Los denominadores pareados deben ser iguales")
    return resultado


def cargar_pareados(ruta: Path) -> list[dict]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return validar_pareados(list(csv.DictReader(archivo)))


def validar_vacas(filas: list[dict[str, str]]) -> list[dict]:
    if not filas or len({r["id"] for r in filas}) != len(filas):
        raise ValueError("Se requieren modelos por vaca con ID único")
    resultado = []
    for fila in filas:
        total, cubiertas = int(fila["vacas_elegibles"]), int(fila["vacas_cubiertas"])
        mae = float(fila["mae_vaca"])
        if total <= 0 or not 0 < cubiertas <= total:
            raise ValueError("La cobertura por vaca es imposible")
        if not math.isfinite(mae) or mae < 0 or fila["salida"] not in {"categorica", "continua"}:
            raise ValueError("MAE o salida por vaca inválidos")
        if not fila["regla"]:
            raise ValueError("Falta la regla de agregación")
        resultado.append({**fila, "vacas_elegibles": total,
                          "vacas_cubiertas": cubiertas, "mae_vaca": mae})
    return resultado


def cargar_vacas(ruta: Path) -> list[dict]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return validar_vacas(list(csv.DictReader(archivo)))


def dibujar_vacas(filas: list[dict], destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.arange(len(filas))
    fig, (ax_cov, ax_mae) = plt.subplots(
        1, 2, figsize=(12, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})
    cobertura = [r["vacas_cubiertas"] / r["vacas_elegibles"] for r in filas]
    mae = [r["mae_vaca"] for r in filas]
    colores = ["#406ca1" if r["salida"] == "categorica" else "#b56a25" for r in filas]
    ax_cov.barh(y, cobertura, color="#287a75", height=.58)
    ax_mae.barh(y, mae, color=colores, height=.58)
    for i, fila in enumerate(filas):
        ax_cov.text(cobertura[i] + .015, i,
                    f"{fila['vacas_cubiertas']}/{fila['vacas_elegibles']}", va="center")
        ax_mae.text(mae[i] + .008, i, f"{mae[i]:.3f}", va="center")
    ax_cov.set_yticks(y, [r["modelo"] for r in filas])
    ax_cov.set_xlim(0, 1.25)
    ax_mae.set_xlim(0, max(mae) * 1.28)
    ax_cov.invert_yaxis()
    ax_cov.set_xlabel("Cobertura de vacas")
    ax_mae.set_xlabel("MAE BCS entre vacas cubiertas")
    ax_cov.grid(axis="x", alpha=.2)
    ax_mae.grid(axis="x", alpha=.2)
    fig.suptitle("Validación por vaca: cobertura y error condicionado")
    fig.text(.02, .015,
             "Inventario descriptivo: hay reglas de agregación, salidas y subconjuntos de fotos distintos. "
             "Las fuentes constan en el CSV.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(destino / "cobertura_y_mae_vaca_val.png", dpi=180)
    plt.close(fig)


def dibujar_pareados(filas: list[dict], destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    y = np.arange(len(filas))
    fotos = [r["mae_foto_categoria"] for r in filas]
    vacas = [r["mae_vaca_media_fotos"] for r in filas]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y - .19, fotos, height=.34, color="#406ca1", label="MAE por foto")
    ax.barh(y + .19, vacas, height=.34, color="#287a75", label="MAE por vaca")
    for i in range(len(filas)):
        ax.text(fotos[i] + .006, i - .19, f"{fotos[i]:.3f}", va="center")
        ax.text(vacas[i] + .006, i + .19, f"{vacas[i]:.3f}", va="center")
    ax.set_yticks(y, [r["modelo"] for r in filas])
    ax.set_xlim(0, max(fotos + vacas) * 1.25)
    ax.invert_yaxis()
    ax.set_xlabel("Error absoluto medio en BCS (menor es mejor)")
    ax.set_title("Tres clasificadores sobre las mismas fotos y vacas de validación")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    fig.text(.02, .01,
             f"{filas[0]['fotos_pareadas']} fotos y {filas[0]['vacas_pareadas']} vacas cubiertas por los tres; "
             "detector compartido. Validación usada durante desarrollo.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(destino / "clasificadores_pareados_val.png", dpi=180)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(11, max(5.8, .75 * len(filas) + 1.8)))
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
             "Cifras descriptivas: las fuentes constan en el CSV. DINO histórico y DINO Mac nuevo son corridas distintas; "
             "categoría y continuo del nuevo DINO son salidas del mismo modelo.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(destino / "cobertura_y_tolerancia_val.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(5.4, .72 * len(filas) + 1.8)))
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
    parser.add_argument("--pareados", type=Path,
                        default=Path(__file__).with_name("clasificadores_pareados_validacion.csv"))
    parser.add_argument("--vacas", type=Path,
                        default=Path(__file__).with_name("datos_agregados_vaca_validacion.csv"))
    args = parser.parse_args()
    dibujar(cargar_agregados(args.datos), args.salida)
    dibujar_pareados(cargar_pareados(args.pareados), args.salida)
    dibujar_vacas(cargar_vacas(args.vacas), args.salida)
    print(f"Gráficas guardadas en {args.salida}")


if __name__ == "__main__":
    main()
