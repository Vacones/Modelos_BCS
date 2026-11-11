import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_yolo_regresion.validar_datos import validar_particiones_coco


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[3]


def escribir_coco(raiz, particion, vacas_por_imagen):
    carpeta = raiz / particion
    carpeta.mkdir()

    imagenes = []
    for indice, vacas in enumerate(vacas_por_imagen, start=1):
        identidades = [vacas] if isinstance(vacas, str) else list(vacas)
        imagenes.append(
            {
                "id": indice,
                "file_name": f"ejemplo_{indice}.jpg",
                "extra": {"user_tags": identidades + ["BCS_3_0"]},
            }
        )

    (carpeta / "_annotations.coco.json").write_text(
        json.dumps({"images": imagenes}), encoding="utf-8"
    )


class ValidarParticionesCocoTest(unittest.TestCase):
    def test_rechaza_vaca_en_dos_particiones(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            escribir_coco(raiz, "train", ["101"])
            escribir_coco(raiz, "valid", ["101"])
            escribir_coco(raiz, "test", ["202"])

            with self.assertRaisesRegex(ValueError, "varias particiones"):
                validar_particiones_coco(raiz)

    def test_acepta_particiones_validas_y_cuenta_identidades(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            escribir_coco(raiz, "train", ["101", "101"])
            escribir_coco(raiz, "valid", ["202"])
            escribir_coco(raiz, "test", ["303"])

            self.assertEqual(
                validar_particiones_coco(raiz),
                {"train": 1, "valid": 1, "test": 1},
            )

    def test_acepta_varias_identidades_en_una_imagen(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            escribir_coco(raiz, "train", [("101", "102")])
            escribir_coco(raiz, "valid", ["202"])
            escribir_coco(raiz, "test", ["303"])

            self.assertEqual(
                validar_particiones_coco(raiz),
                {"train": 2, "valid": 1, "test": 1},
            )

    def test_detecta_cruce_de_una_identidad_ambigua(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            escribir_coco(raiz, "train", [("101", "102")])
            escribir_coco(raiz, "valid", ["102"])
            escribir_coco(raiz, "test", ["303"])

            with self.assertRaisesRegex(ValueError, "varias particiones"):
                validar_particiones_coco(raiz)

    def test_comando_rechaza_ruta_inexistente(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta_inexistente = Path(temporal) / "no_existe"
            resultado = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modelos.con_cajas_yolo_regresion.validar_datos",
                    str(ruta_inexistente),
                ],
                cwd=RAIZ_REPOSITORIO,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(resultado.returncode, 0)
            self.assertIn("No se encontró el dataset", resultado.stderr)

    def test_comando_muestra_resumen(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            escribir_coco(raiz, "train", ["101", "101"])
            escribir_coco(raiz, "valid", ["202"])
            escribir_coco(raiz, "test", ["303"])

            resultado = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modelos.con_cajas_yolo_regresion.validar_datos",
                    str(raiz),
                ],
                cwd=RAIZ_REPOSITORIO,
                capture_output=True,
                text=True,
            )

            self.assertEqual(resultado.returncode, 0, resultado.stderr)
            self.assertEqual(
                json.loads(resultado.stdout),
                {"train": 1, "valid": 1, "test": 1},
            )


if __name__ == "__main__":
    unittest.main()