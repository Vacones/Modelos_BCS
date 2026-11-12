"""Comprobaciones de lectura antes de ejecutar DINOv2/LDL en Mac."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


DEPENDENCIAS = (
    "torch", "torchvision", "numpy", "pandas", "matplotlib",
    "scikit-learn", "Pillow", "PyYAML", "tqdm", "ultralytics",
    "tensorflow", "tensorflow-metal", "keras", "certifi",
)


def verificar_entorno(versiones: dict[str, str], sistema: str, maquina: str,
                      python: tuple[int, int]) -> None:
    if sistema != "darwin" or maquina != "arm64" or python not in ((3, 11), (3, 12)):
        raise ValueError("Se requiere macOS ARM64 con Python 3.11 o 3.12")
    faltantes = [nombre for nombre in DEPENDENCIAS if nombre not in versiones]
    if faltantes:
        raise ValueError("Faltan dependencias: " + ", ".join(faltantes))


def verificar_dataset(raiz: Path) -> dict[str, int]:
    """Lee únicamente train y valid del COCO original; nunca inspecciona test."""
    resumen = {}
    for split in ("train", "valid"):
        carpeta = raiz / split
        archivo = carpeta / "_annotations.coco.json"
        if not archivo.is_file():
            raise ValueError(f"Falta el COCO de {split}")
        try:
            coco = json.loads(archivo.read_text(encoding="utf-8"))
            nombres = {categoria["name"] for categoria in coco["categories"]}
            imagenes = coco["images"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"COCO ilegible en {split}: {error}") from error
        if not {"cow", "ass"}.issubset(nombres):
            raise ValueError(f"Faltan clases cow/ass en {split}")
        if not imagenes:
            raise ValueError(f"No hay imágenes en {split}")
        faltantes = 0
        for imagen in imagenes:
            nombre = imagen.get("file_name")
            if not isinstance(nombre, str) or Path(nombre).name != nombre:
                raise ValueError(f"Nombre de imagen inválido en {split}")
            if not (carpeta / nombre).is_file():
                faltantes += 1
        if faltantes:
            raise ValueError(f"En {split} faltan archivos de imagen: {faltantes}")
        resumen[split] = len(imagenes)
    return resumen


def verificar_salida(ruta: Path) -> None:
    if ruta.exists():
        raise ValueError("La carpeta de salida ya existe; usa un nombre nuevo")
    ancestro = ruta.parent
    while not ancestro.exists():
        if ancestro == ancestro.parent:
            raise ValueError("No se encuentra un directorio padre para la salida")
        ancestro = ancestro.parent
    if not ancestro.is_dir():
        raise ValueError("El padre de la salida no es un directorio")
    if not os.access(ancestro, os.W_OK):
        raise ValueError("No hay permiso de escritura para la salida")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    args = parser.parse_args()
    versiones = {}
    for nombre in DEPENDENCIAS:
        try:
            versiones[nombre] = importlib.metadata.version(nombre)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        verificar_entorno(versiones, sys.platform, platform.machine(), sys.version_info[:2])
        resumen = verificar_dataset(args.datos)
        verificar_salida(args.salida)
    except ValueError as error:
        parser.exit(2, f"Preflight no aprobado: {error}\n")
    print(json.dumps({"estado": "listo_para_ejecutar", "imagenes_train": resumen["train"],
                      "imagenes_valid": resumen["valid"], "test_leido": False,
                      "salida_nueva": str(args.salida.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
