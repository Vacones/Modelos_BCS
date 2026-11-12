"""Revisión de solo lectura del COCO aprobado antes de preparar YOLO + EfficientNet.

Uso: python -m modelos.con_cajas_yolo.preflight_mac RUTA_COCO
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def inspeccionar_coco(raiz: Path) -> dict:
    raiz = Path(raiz).expanduser().resolve()
    if not raiz.is_dir():
        raise ValueError("No existe la carpeta COCO")
    fotos, vacas, utilizables, huellas = {}, {}, {}, {}
    particion_por_vaca = {}
    bcs_por_vaca = {}
    particion_por_foto = {}
    for particion in ("train", "valid", "test"):
        carpeta = raiz / particion
        ruta_json = carpeta / "_annotations.coco.json"
        if not ruta_json.is_file():
            raise ValueError(f"Falta JSON COCO en {particion}")
        contenido = ruta_json.read_bytes()
        coco = json.loads(contenido)
        categorias = {c["id"]: c["name"] for c in coco["categories"]}
        if len(categorias) != len(coco["categories"]) or sum(x == "ass" for x in categorias.values()) != 1 or sum(x == "cow" for x in categorias.values()) != 1:
            raise ValueError(f"Clases COCO incorrectas en {particion}")
        imagenes = coco["images"]
        ids = [im["id"] for im in imagenes]
        nombres = [im["file_name"] for im in imagenes]
        if len(ids) != len(set(ids)) or len(nombres) != len(set(nombres)):
            raise ValueError(f"Identificadores COCO repetidos en {particion}")
        por_imagen = {identificador: Counter() for identificador in ids}
        for anotacion in coco["annotations"]:
            imagen = anotacion["image_id"]
            if imagen not in por_imagen or categorias.get(anotacion["category_id"]) not in {"cow", "ass"}:
                raise ValueError(f"Anotación sin imagen o clase válida en {particion}")
            caja = anotacion["bbox"]
            if len(caja) != 4 or not all(math.isfinite(float(x)) for x in caja) or caja[2] <= 0 or caja[3] <= 0:
                raise ValueError(f"Caja inválida en {particion}")
            por_imagen[imagen][categorias[anotacion["category_id"]]] += 1
        vacas_particion = set()
        total_utilizable = 0
        for imagen in imagenes:
            nombre = imagen["file_name"]
            if Path(nombre).name != nombre or not (carpeta / nombre).is_file():
                raise ValueError(f"Falta fotografía o ruta inválida en {particion}")
            huella_foto = hashlib.sha256()
            with (carpeta / nombre).open("rb") as archivo:
                for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                    huella_foto.update(bloque)
            anterior_foto = particion_por_foto.setdefault(huella_foto.digest(), particion)
            if anterior_foto != particion:
                raise ValueError("Hay una fotografía idéntica entre particiones")
            if por_imagen[imagen["id"]] != Counter({"cow": 1, "ass": 1}):
                raise ValueError(f"Se requiere una caja cow y una ass por fotografía en {particion}")
            etiquetas = imagen.get("extra", {}).get("user_tags", [])
            identificadores = {str(x) for x in etiquetas if re.fullmatch(r"\d+", str(x))}
            bcs_tags = {str(x) for x in etiquetas if str(x).startswith("BCS_")}
            if not identificadores:
                raise ValueError(f"Identidad ausente en {particion}")
            for vaca in identificadores:
                anterior = particion_por_vaca.setdefault(vaca, particion)
                if anterior != particion:
                    raise ValueError("Una vaca aparece en dos particiones")
            vacas_particion.update(identificadores)
            if len(identificadores) == 1 and len(bcs_tags) == 1:
                vaca = next(iter(identificadores))
                etiqueta = next(iter(bcs_tags))
                coincidencia = re.fullmatch(r"BCS_(\d+)(?:_(\d+))?", etiqueta)
                if coincidencia is None:
                    raise ValueError(f"BCS mal formado en {particion}")
                bcs = float(coincidencia[1] + ("." + coincidencia[2] if coincidencia[2] else ""))
                if not 1 <= bcs <= 5:
                    raise ValueError(f"BCS fuera de rango en {particion}")
                previo = bcs_por_vaca.setdefault(vaca, bcs)
                if previo != bcs:
                    raise ValueError(f"BCS contradictorio dentro de una vaca en {particion}")
                total_utilizable += 1
        fotos[particion] = len(imagenes)
        vacas[particion] = len(vacas_particion)
        utilizables[particion] = total_utilizable
        huellas[particion] = hashlib.sha256(contenido).hexdigest()
    return {"fotos": fotos, "vacas": vacas, "fotos_bcs_utilizable": utilizables,
            "sha256_coco": huellas, "solo_lectura": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco", type=Path)
    argumentos = parser.parse_args()
    print(json.dumps(inspeccionar_coco(argumentos.coco), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
