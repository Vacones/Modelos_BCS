"""Matrices agregadas de tres clasificadores en las mismas fotos de validación.

Lee predicciones privadas, verifica el pareo y escribe únicamente conteos.
Nunca guarda claves, nombres de fotos, vacas ni predicciones individuales.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NIVELES = (2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
NOMBRES = {"DINOv2": "DINOv2/LDL", "CORAL": "CORAL ordinal",
           "EfficientNet": "EfficientNet"}


def matrices_pareadas(datos: dict[str, list[tuple]]) -> dict[str, list[list[int]]]:
    """Cada fila: (clave_foto, vaca, BCS_real, estado, BCS_predicho)."""
    if len(datos) < 2:
        raise ValueError("Se requieren al menos dos modelos")
    referencias = None
    matrices = {}
    for modelo, filas in datos.items():
        if len({fila[0] for fila in filas}) != len(filas):
            raise ValueError("Hay claves de foto duplicadas")
        actual = {fila[0]: (str(fila[1]), float(fila[2]), fila[3]) for fila in filas}
        if referencias is None:
            referencias = actual
        elif set(actual) != set(referencias):
            raise ValueError("Los modelos deben tener las mismas fotos")
        elif actual != referencias:
            raise ValueError("Los modelos tienen etiquetas o estados distintos")
        matriz = [[0] * 8 for _ in NIVELES]
        for _, _, real, estado, pred in filas:
            if float(real) not in NIVELES:
                raise ValueError("BCS real fuera del catálogo")
            if estado == "ok":
                if pred is None or float(pred) not in NIVELES:
                    raise ValueError("BCS predicho fuera del catálogo")
                columna = NIVELES.index(float(pred))
            elif estado in ("asociacion_ambigua", "sin_par_valido"):
                if pred is not None:
                    raise ValueError("Abstención con BCS predicho")
                columna = 7
            else:
                raise ValueError("Estado de predicción desconocido")
            matriz[NIVELES.index(float(real))][columna] += 1
        matrices[modelo] = matriz
    return matrices


def _leer(ruta: Path, ordinal: bool) -> list[tuple]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    salida = []
    for fila in filas:
        if ordinal:
            if fila["particion"] != "valid" or not fila["clave"].startswith("valid/"):
                raise ValueError("Partición ordinal distinta de valid")
            clave = fila["clave"].removeprefix("valid/")
            vaca, real, estado, pred = fila["vaca"], fila["bcs_real"], fila["estado"], fila["bcs_pred"]
        else:
            if fila["split"] != "val":
                raise ValueError("Partición distinta de val")
            clave = fila["file_name"]
            vaca, real, estado, pred = fila["cow_id"], fila["true_bcs"], fila["status"], fila["pred_bcs"]
        salida.append((clave, vaca, float(real), estado, float(pred) if pred else None))
    return salida


def dibujar(matrices: dict[str, list[list[int]]], ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(1, len(matrices), figsize=(5.6 * len(matrices), 7.0),
                             sharex=True, sharey=True)
    if len(matrices) == 1:
        ejes = [ejes]
    for eje, (modelo, valores) in zip(ejes, matrices.items()):
        imagen = eje.imshow(valores, cmap="Blues", vmin=0,
                           vmax=max(max(max(fila) for fila in matriz) for matriz in matrices.values()))
        eje.set_title(NOMBRES.get(modelo, modelo))
        eje.set_xticks(range(8), [*map(str, NIVELES), "Abst."], rotation=55, ha="right")
        eje.set_yticks(range(7), list(map(str, NIVELES)))
        eje.set_xlabel("Predicción BCS")
        for i, fila in enumerate(valores):
            for j, numero in enumerate(fila):
                eje.text(j, i, str(numero), ha="center", va="center", fontsize=8,
                         color="white" if numero > 3 else "black")
    ejes[0].set_ylabel("BCS de referencia")
    fig.suptitle("Validación pareada: matrices de confusión y abstenciones")
    fig.text(.02, .018,
             "Mismas 36 fotos, 24 con predicción y 12 abstenciones en cada modelo. "
             "Detector compartido; validación de desarrollo.", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, .94))
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino", required=True, type=Path)
    parser.add_argument("--ordinal", required=True, type=Path)
    parser.add_argument("--efficientnet", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    args = parser.parse_args()
    datos = {
        "DINOv2": _leer(args.dino, False),
        "CORAL": _leer(args.ordinal, True),
        "EfficientNet": _leer(args.efficientnet, False),
    }
    matrices = matrices_pareadas(datos)
    dibujar(matrices, args.salida)
    print(json.dumps({"estado": "guardado", "modelos": len(matrices),
                      "fotos_por_modelo": sum(matrices["DINOv2"][i][j]
                                              for i in range(7) for j in range(8)),
                      "contiene_filas_privadas": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
