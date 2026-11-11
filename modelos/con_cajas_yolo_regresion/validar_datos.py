import argparse
import json
import re
from pathlib import Path


def validar_particiones_coco(raiz):
    """Comprueba que ninguna identidad aparezca en varias particiones COCO."""
    raiz = Path(raiz)
    particion_por_vaca = {}
    vacas_por_particion = {
        particion: set() for particion in ("train", "valid", "test")
    }

    for particion in ("train", "valid", "test"):
        ruta = raiz / particion / "_annotations.coco.json"
        contenido = json.loads(ruta.read_text(encoding="utf-8"))

        for imagen in contenido["images"]:
            etiquetas = imagen.get("extra", {}).get("user_tags", [])
            identidades = {
                str(etiqueta)
                for etiqueta in etiquetas
                if re.fullmatch(r"\d+", str(etiqueta))
            }

            if not identidades:
                raise ValueError(
                    f"Falta identidad de vaca en {particion}/"
                    f"{imagen.get('file_name', '?')}"
                )

            for vaca in sorted(identidades):
                anterior = particion_por_vaca.get(vaca)
                if anterior is not None and anterior != particion:
                    raise ValueError(
                        f"La vaca {vaca} aparece en varias particiones: "
                        f"{anterior} y {particion}"
                    )

                particion_por_vaca[vaca] = particion
                vacas_por_particion[particion].add(vaca)

    return {
        particion: len(vacas)
        for particion, vacas in vacas_por_particion.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprueba la separación de identidades en un dataset COCO."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Carpeta que contiene train, valid y test.",
    )
    argumentos = parser.parse_args()

    if not argumentos.dataset.is_dir():
        parser.error(f"No se encontró el dataset: {argumentos.dataset}")

    resumen = validar_particiones_coco(argumentos.dataset)
    print(json.dumps(resumen, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()