"""Prepara en una carpeta privada el COCO v2 para el notebook ordinal.

Las fotos se enlazan con symlinks: no se copian ni se escriben en el repositorio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .preflight_coco import PARTICIONES, _bcs, revisar_coco_bcs


NOMBRE_EXPORT = "Vacones BCS - Cajas v2 Aprobado.yolo26"
CLASES = {"cow": 0, "ass": 1}


def _caja_yolo(caja: list[float], ancho_imagen: int, alto_imagen: int) -> tuple[float, ...]:
    if len(caja) != 4 or any(not math.isfinite(float(v)) for v in caja):
        raise ValueError("Caja COCO inválida")
    x, y, ancho, alto = map(float, caja)
    if ancho <= 0 or alto <= 0 or ancho_imagen <= 0 or alto_imagen <= 0:
        raise ValueError("Caja o dimensiones inválidas")
    exceso = max(0.0, -x, -y, x + ancho - ancho_imagen, y + alto - alto_imagen)
    if exceso > 0.02:
        raise ValueError("Caja fuera de los límites de la foto")
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(ancho_imagen), x + ancho), min(float(alto_imagen), y + alto)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Caja vacía después de ajustar límites")
    return ((x0 + x1) / (2 * ancho_imagen), (y0 + y1) / (2 * alto_imagen),
            (x1 - x0) / ancho_imagen, (y1 - y0) / alto_imagen)


def planificar_export(raiz: Path) -> tuple[list[dict], dict]:
    """Valida todo y devuelve registros privados en memoria, sin escribir archivos."""
    raiz = Path(raiz).expanduser().resolve()
    resumen = revisar_coco_bcs(raiz)
    registros = []
    for particion in PARTICIONES:
        carpeta = raiz / particion
        coco = json.loads((carpeta / "_annotations.coco.json").read_text(encoding="utf-8"))
        nombres = {c["id"]: c["name"] for c in coco["categories"]}
        if not set(CLASES) <= set(nombres.values()):
            raise ValueError(f"Faltan clases cow y ass en {particion}")
        anotaciones = defaultdict(list)
        for anotacion in coco["annotations"]:
            anotaciones[anotacion["image_id"]].append(anotacion)
        for imagen in coco["images"]:
            etiquetas = [str(t) for t in imagen.get("extra", {}).get("user_tags", [])]
            identidades = {t for t in etiquetas if re.fullmatch(r"\d+", t)}
            etiquetas_bcs = [t for t in etiquetas if t.startswith("BCS_")]
            valor = _bcs(etiquetas_bcs[0]) if len(etiquetas_bcs) == 1 else None
            if len(identidades) != 1 or valor is None:
                continue
            cajas = {}
            for anotacion in anotaciones[imagen["id"]]:
                clase = nombres.get(anotacion["category_id"])
                if clase in CLASES:
                    if clase in cajas:
                        raise ValueError(f"Más de una caja {clase} en {particion}")
                    cajas[clase] = _caja_yolo(anotacion["bbox"],
                                              imagen["width"], imagen["height"])
            if set(cajas) != set(CLASES):
                raise ValueError(f"Se requiere una caja cow y una ass en {particion}")
            registros.append({
                "particion": particion,
                "nombre": imagen["file_name"],
                "origen": carpeta / imagen["file_name"],
                "vaca": next(iter(identidades)),
                "bcs": valor,
                "cajas": cajas,
                "ancho": imagen["width"],
                "alto": imagen["height"],
            })
    for particion in PARTICIONES:
        if sum(r["particion"] == particion for r in registros) != resumen[particion]["fotos_bcs_validas"]:
            raise ValueError(f"Conteo inconsistente en {particion}")
    return registros, resumen


def crear_export_ordinal(raiz: Path, destino: Path) -> dict:
    """Crea export YOLO con enlaces a fotos y metadata, fuera del repositorio."""
    raiz = Path(raiz).expanduser().resolve()
    destino = Path(destino).expanduser().resolve()
    if destino.is_relative_to(raiz):
        raise ValueError("La salida privada no debe quedar dentro del COCO de origen")
    if destino.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("La salida privada no debe quedar dentro del repositorio")
    if destino.exists():
        raise FileExistsError(f"El destino ya existe: {destino}")
    registros, resumen = planificar_export(raiz)
    export = destino / NOMBRE_EXPORT
    for particion in PARTICIONES:
        (export / particion / "images").mkdir(parents=True)
        (export / particion / "labels").mkdir()
    (export / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnames: [cow, ass]\n",
        encoding="utf-8",
    )
    with (destino / "metadata_bcs_validos.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=(
            "metadata_status", "file_name", "split", "cow_id", "bcs"))
        escritor.writeheader()
        for registro in registros:
            carpeta = export / registro["particion"]
            (carpeta / "images" / registro["nombre"]).symlink_to(registro["origen"])
            lineas = []
            for clase in ("cow", "ass"):
                coordenadas = " ".join(f"{v:.6f}" for v in registro["cajas"][clase])
                lineas.append(f"{CLASES[clase]} {coordenadas}")
            (carpeta / "labels" / (Path(registro["nombre"]).stem + ".txt")).write_text(
                "\n".join(lineas) + "\n", encoding="utf-8")
            escritor.writerow({
                "metadata_status": "ok", "file_name": registro["nombre"],
                "split": registro["particion"], "cow_id": registro["vaca"],
                "bcs": f"{registro['bcs']:.2f}",
            })
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Carpeta COCO v2 aprobada")
    parser.add_argument("--salida", type=Path, help="Carpeta privada nueva para el notebook")
    parser.add_argument("--solo-validar", action="store_true", help="No escribe archivos")
    argumentos = parser.parse_args()
    if argumentos.solo_validar:
        _, resumen = planificar_export(argumentos.dataset)
    else:
        if argumentos.salida is None:
            parser.error("Indica --salida o usa --solo-validar")
        resumen = crear_export_ordinal(argumentos.dataset, argumentos.salida)
    print(json.dumps(resumen, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
