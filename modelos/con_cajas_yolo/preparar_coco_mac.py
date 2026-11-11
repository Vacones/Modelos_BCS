"""Normaliza un COCO revisado a la entrada del pipeline, sin modificar el origen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelos.con_cajas_yolo.preflight_mac import inspeccionar_coco


def preparar_entrada(fuente: Path, destino: Path) -> dict:
    fuente = Path(fuente).expanduser().resolve()
    destino = Path(destino).expanduser().resolve()
    if fuente == destino or fuente in destino.parents:
        raise ValueError("La salida debe estar fuera del origen COCO")
    if destino.exists():
        raise ValueError("La salida ya existe; elige una ruta nueva")
    resumen = inspeccionar_coco(fuente)
    from modelos.con_cajas_yolo.dinov2_ldl_mac.bcs.raw_coco import normalize
    normalize(fuente, destino)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--solo-validar", action="store_true")
    argumentos = parser.parse_args()
    if argumentos.solo_validar:
        resultado = inspeccionar_coco(argumentos.coco)
    else:
        if argumentos.salida is None:
            parser.error("--salida es obligatorio para normalizar")
        resultado = preparar_entrada(argumentos.coco, argumentos.salida)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
