import json
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_ordinal.preflight_coco import revisar_coco_bcs


class PreflightCocoTest(unittest.TestCase):
    def crear_particion(self, raiz, nombre, fotos):
        carpeta = raiz / nombre
        carpeta.mkdir()
        imagenes = []
        anotaciones = []
        for numero, (archivo, etiquetas) in enumerate(fotos, start=1):
            (carpeta / archivo).write_bytes(b"imagen sintetica")
            imagenes.append({"id": numero, "file_name": archivo, "extra": {"user_tags": etiquetas}})
            anotaciones.append({"id": numero, "image_id": numero, "category_id": 2,
                                 "bbox": [0, 0, 10, 10]})
        (carpeta / "_annotations.coco.json").write_text(json.dumps({
            "categories": [{"id": 2, "name": "ass"}],
            "images": imagenes,
            "annotations": anotaciones,
        }))

    def test_acepta_particiones_y_excluye_bcs_ambiguo(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_particion(raiz, "train", [
                ("uno.jpg", ["101", "BCS_3_00"]),
                ("ambiguo.jpg", ["102", "103", "BCS_3_25", "BCS_3_50"]),
            ])
            self.crear_particion(raiz, "valid", [("dos.jpg", ["201", "BCS_2_75"])])
            self.crear_particion(raiz, "test", [("tres.jpg", ["301", "BCS_3_50"])])
            resumen = revisar_coco_bcs(raiz)
            self.assertEqual(resumen["train"], {"fotos": 2, "fotos_bcs_validas": 1,
                                                 "excluidas_bcs": 1, "vacas_bcs_validas": 1})
            self.assertEqual(resumen["valid"]["fotos_bcs_validas"], 1)

    def test_rechaza_vaca_entre_train_y_valid(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_particion(raiz, "train", [("uno.jpg", ["101", "BCS_3_00"])])
            self.crear_particion(raiz, "valid", [("dos.jpg", ["101", "BCS_3_00"])])
            self.crear_particion(raiz, "test", [("tres.jpg", ["301", "BCS_3_50"])])
            with self.assertRaisesRegex(ValueError, "varias particiones"):
                revisar_coco_bcs(raiz)

    def test_rechaza_bcs_distinto_en_misma_vaca(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_particion(raiz, "train", [
                ("uno.jpg", ["101", "BCS_3_00"]),
                ("otro.jpg", ["101", "BCS_3_25"]),
            ])
            self.crear_particion(raiz, "valid", [])
            self.crear_particion(raiz, "test", [])
            with self.assertRaisesRegex(ValueError, "BCS distintos"):
                revisar_coco_bcs(raiz)

    def test_excluye_etiqueta_bcs_malformada(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_particion(raiz, "train", [("uno.jpg", ["101", "BCS_sin_valor"])])
            self.crear_particion(raiz, "valid", [])
            self.crear_particion(raiz, "test", [])
            self.assertEqual(revisar_coco_bcs(raiz)["train"]["excluidas_bcs"], 1)

    def test_excluye_dos_etiquetas_bcs_aunque_una_sea_valida(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_particion(raiz, "train", [
                ("uno.jpg", ["101", "BCS_3_00", "BCS_sin_valor"]),
            ])
            self.crear_particion(raiz, "valid", [])
            self.crear_particion(raiz, "test", [])
            self.assertEqual(revisar_coco_bcs(raiz)["train"]["excluidas_bcs"], 1)


if __name__ == "__main__":
    unittest.main()
