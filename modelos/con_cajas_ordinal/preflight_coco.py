"""Comprueba el COCO aprobado antes de preparar el modelo ordinal.

No carga píxeles ni genera recortes, predicciones o archivos de salida.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PARTICIONES = ("train", "valid", "test")
PATRON_BCS = re.compile(r"BCS_(\d+)(?:_(\d+))?")


def _bcs(etiqueta: str) -> float | None:
    coincidencia = PATRON_BCS.fullmatch(etiqueta)
    if not coincidencia:
        return None
    valor = float(coincidencia[1] + ("." + coincidencia[2] if coincidencia[2] else ""))
    return valor if 1 <= valor <= 5 else None


def revisar_coco_bcs(raiz: Path) -> dict[str, dict[str, int]]:
    """Devuelve solo conteos; rechaza fugas por vaca y BCS contradictorios."""
    raiz = Path(raiz)
    resumen = {}
    particiones_por_vaca: dict[str, set[str]] = defaultdict(set)
    bcs_por_vaca: dict[str, set[float]] = defaultdict(set)
    vacas_validas: dict[str, set[str]] = {p: set() for p in PARTICIONES}

    for particion in PARTICIONES:
        carpeta = raiz / particion
        ruta = carpeta / "_annotations.coco.json"
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        categorias = {c["id"]: c["name"] for c in contenido["categories"]}
        if "ass" not in categorias.values():
            raise ValueError(f"Falta la clase ass en {particion}")
        imagenes = contenido["images"]
        ids = [imagen["id"] for imagen in imagenes]
        nombres = [imagen["file_name"] for imagen in imagenes]
        if len(ids) != len(set(ids)) or len(nombres) != len(set(nombres)):
            raise ValueError(f"Imágenes duplicadas en {particion}")
        cajas_ass = defaultdict(int)
        for anotacion in contenido["annotations"]:
            if anotacion["image_id"] not in ids:
                raise ValueError(f"Anotación sin imagen en {particion}")
            if categorias.get(anotacion["category_id"]) == "ass":
                cajas_ass[anotacion["image_id"]] += 1

        validas = 0
        for imagen in imagenes:
            nombre = imagen["file_name"]
            if Path(nombre).name != nombre or "\\" in nombre:
                raise ValueError(f"Nombre de imagen inválido en {particion}")
            if not (carpeta / nombre).is_file():
                raise FileNotFoundError(f"Falta una imagen en {particion}: {nombre}")
            etiquetas = [str(t) for t in imagen.get("extra", {}).get("user_tags", [])]
            identidades = {t for t in etiquetas if re.fullmatch(r"\d+", t)}
            if not identidades:
                raise ValueError(f"Falta identidad de vaca en {particion}: {nombre}")
            for vaca in identidades:
                particiones_por_vaca[vaca].add(particion)
            etiquetas_bcs = [t for t in etiquetas if t.startswith("BCS_")]
            valor = _bcs(etiquetas_bcs[0]) if len(etiquetas_bcs) == 1 else None
            if len(identidades) != 1 or valor is None:
                continue  # No asignar un BCS ambiguo al entrenamiento.
            if cajas_ass[imagen["id"]] != 1:
                raise ValueError(f"Se requiere una caja ass para {particion}: {nombre}")
            vaca = next(iter(identidades))
            bcs_por_vaca[vaca].add(valor)
            vacas_validas[particion].add(vaca)
            validas += 1
        resumen[particion] = {
            "fotos": len(imagenes),
            "fotos_bcs_validas": validas,
            "excluidas_bcs": len(imagenes) - validas,
            "vacas_bcs_validas": len(vacas_validas[particion]),
        }

    if any(len(particiones) > 1 for particiones in particiones_por_vaca.values()):
        raise ValueError("Una o más vacas aparecen en varias particiones")
    if any(len(valores) > 1 for valores in bcs_por_vaca.values()):
        raise ValueError("Una o más vacas tienen BCS distintos")
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Carpeta COCO con train, valid y test")
    argumentos = parser.parse_args()
    print(json.dumps(revisar_coco_bcs(argumentos.dataset), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
