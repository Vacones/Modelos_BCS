"""Pruebas con COCO sintético; no leen fotografías del proyecto."""

import json
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_yolo.preflight_mac import inspeccionar_coco
from modelos.con_cajas_yolo.preparar_coco_mac import preparar_entrada


class PreflightMacTest(unittest.TestCase):
    def crear_fuente(self, raiz: Path) -> None:
        for indice, particion in enumerate(("train", "valid", "test"), start=1):
            carpeta = raiz / particion
            carpeta.mkdir()
            (carpeta / f"foto{indice}.jpg").write_bytes(f"imagen sintetica {indice}".encode())
            coco = {
                "images": [{"id": indice, "file_name": f"foto{indice}.jpg",
                            "width": 100, "height": 100,
                            "extra": {"user_tags": [str(indice), "BCS_3"]}}],
                "categories": [{"id": 1, "name": "cow"}, {"id": 2, "name": "ass"}],
                "annotations": [
                    {"id": 2 * indice, "image_id": indice, "category_id": 1,
                     "bbox": [0, 0, 100, 100]},
                    {"id": 2 * indice + 1, "image_id": indice, "category_id": 2,
                     "bbox": [20, 30, 40, 30]},
                ],
            }
            (carpeta / "_annotations.coco.json").write_text(json.dumps(coco))

    def test_resume_coco_sin_escribir_archivos(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            antes = sorted(p.relative_to(raiz) for p in raiz.rglob("*"))
            resumen = inspeccionar_coco(raiz)
            despues = sorted(p.relative_to(raiz) for p in raiz.rglob("*"))
            self.assertEqual(resumen["fotos"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(resumen["vacas"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(resumen["fotos_bcs_utilizable"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(antes, despues)

    def test_rechaza_vaca_en_dos_particiones(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            ruta = raiz / "valid/_annotations.coco.json"
            coco = json.loads(ruta.read_text())
            coco["images"][0]["extra"]["user_tags"] = ["1", "BCS_3"]
            ruta.write_text(json.dumps(coco))
            with self.assertRaisesRegex(ValueError, "particiones"):
                inspeccionar_coco(raiz)

    def test_rechaza_fotografia_faltante(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            (raiz / "test/foto3.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "fotografía"):
                inspeccionar_coco(raiz)

    def test_acepta_categoria_fondo_sin_anotaciones(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            for particion in ("train", "valid", "test"):
                ruta = raiz / particion / "_annotations.coco.json"
                coco = json.loads(ruta.read_text())
                coco["categories"].append({"id": 0, "name": "proyecto"})
                ruta.write_text(json.dumps(coco))
            self.assertEqual(inspeccionar_coco(raiz)["fotos"]["valid"], 1)

    def test_cuenta_como_no_utilizable_una_foto_ambigua(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            ruta = raiz / "train/_annotations.coco.json"
            coco = json.loads(ruta.read_text())
            coco["images"][0]["extra"]["user_tags"] = ["1", "4", "BCS_3", "BCS_4"]
            ruta.write_text(json.dumps(coco))
            resumen = inspeccionar_coco(raiz)
            self.assertEqual(resumen["fotos"]["train"], 1)
            self.assertEqual(resumen["fotos_bcs_utilizable"]["train"], 0)

    def test_prepara_entrada_sin_cambiar_origen(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            fuente, destino = raiz / "fuente", raiz / "salida"
            fuente.mkdir()
            self.crear_fuente(fuente)
            antes = (fuente / "valid/_annotations.coco.json").read_bytes()
            resumen = preparar_entrada(fuente, destino)
            self.assertTrue((destino / "metadata.csv").is_file())
            self.assertTrue((destino / "images/val/foto2.jpg").is_file())
            self.assertTrue((destino / "labels/val/foto2.txt").is_file())
            self.assertEqual((fuente / "valid/_annotations.coco.json").read_bytes(), antes)
            self.assertEqual(resumen["fotos_bcs_utilizable"]["valid"], 1)
            self.assertFalse(resumen["solo_lectura"])

    def test_no_crea_salida_si_hay_fuga_entre_particiones(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            fuente, destino = raiz / "fuente", raiz / "salida"
            fuente.mkdir()
            self.crear_fuente(fuente)
            ruta = fuente / "valid/_annotations.coco.json"
            coco = json.loads(ruta.read_text())
            coco["images"][0]["extra"]["user_tags"] = ["1", "BCS_3"]
            ruta.write_text(json.dumps(coco))
            with self.assertRaisesRegex(ValueError, "particiones"):
                preparar_entrada(fuente, destino)
            self.assertFalse(destino.exists())

    def test_rechaza_salida_dentro_del_origen(self):
        with tempfile.TemporaryDirectory() as temporal:
            fuente = Path(temporal) / "fuente"
            fuente.mkdir()
            self.crear_fuente(fuente)
            destino = fuente / "derivado"
            with self.assertRaisesRegex(ValueError, "fuera del origen"):
                preparar_entrada(fuente, destino)
            self.assertFalse(destino.exists())

    def test_rechaza_fotografia_identica_entre_particiones(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_fuente(raiz)
            (raiz / "valid/foto2.jpg").write_bytes((raiz / "train/foto1.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "fotografía idéntica"):
                inspeccionar_coco(raiz)


if __name__ == "__main__":
    unittest.main()
