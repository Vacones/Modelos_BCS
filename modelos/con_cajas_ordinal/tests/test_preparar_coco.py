import csv
import json
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_ordinal.preparar_coco import crear_export_ordinal, planificar_export


class PrepararCocoTest(unittest.TestCase):
    def crear_coco(self, raiz, particion, fotos):
        carpeta = raiz / particion
        carpeta.mkdir()
        imagenes, anotaciones = [], []
        for numero, (nombre, vaca, bcs) in enumerate(fotos, start=1):
            (carpeta / nombre).write_bytes(b"imagen de prueba")
            imagenes.append({"id": numero, "file_name": nombre, "width": 100,
                             "height": 50, "extra": {"user_tags": [vaca, bcs]}})
            anotaciones.extend([
                {"id": numero * 2, "image_id": numero, "category_id": 1,
                 "bbox": [10, 5, 80, 40]},
                {"id": numero * 2 + 1, "image_id": numero, "category_id": 2,
                 "bbox": [40, 20, 20, 15]},
            ])
        (carpeta / "_annotations.coco.json").write_text(json.dumps({
            "categories": [{"id": 1, "name": "cow"}, {"id": 2, "name": "ass"}],
            "images": imagenes, "annotations": anotaciones,
        }))

    def test_exporta_metadatos_y_cajas_sin_copiar_fotos(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "coco"
            raiz.mkdir()
            self.crear_coco(raiz, "train", [("uno.jpg", "101", "BCS_3_00")])
            self.crear_coco(raiz, "valid", [("dos.jpg", "201", "BCS_2_75")])
            self.crear_coco(raiz, "test", [])
            destino = Path(temporal) / "ordinal_privado"
            resumen = crear_export_ordinal(raiz, destino)
            self.assertEqual(resumen["train"]["fotos_bcs_validas"], 1)
            export = destino / "Vacones BCS - Cajas v2 Aprobado.yolo26"
            self.assertTrue((export / "train/images/uno.jpg").is_symlink())
            self.assertEqual((export / "train/labels/uno.txt").read_text().splitlines(),
                             ["0 0.500000 0.500000 0.800000 0.800000",
                              "1 0.500000 0.550000 0.200000 0.300000"])
            with (destino / "metadata_bcs_validos.csv").open() as archivo:
                filas = list(csv.DictReader(archivo))
            self.assertEqual(len(filas), 2)
            self.assertEqual(filas[0]["metadata_status"], "ok")
            self.assertEqual(filas[1]["split"], "valid")

    def test_planificacion_no_escribe_archivos_y_excluye_ambiguedad(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "coco"
            raiz.mkdir()
            self.crear_coco(raiz, "train", [("uno.jpg", "101", "BCS_3_00"),
                                            ("dos.jpg", "102", "BCS_desconocido")])
            self.crear_coco(raiz, "valid", [])
            self.crear_coco(raiz, "test", [])
            registros, resumen = planificar_export(raiz)
            self.assertEqual(len(registros), 1)
            self.assertEqual((registros[0]["ancho"], registros[0]["alto"]), (100, 50))
            self.assertEqual(resumen["train"]["excluidas_bcs"], 1)
            self.assertFalse((Path(temporal) / "ordinal_privado").exists())

    def test_rechaza_salida_dentro_del_coco_antes_de_escribir(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "coco"
            raiz.mkdir()
            self.crear_coco(raiz, "train", [("uno.jpg", "101", "BCS_3_00")])
            self.crear_coco(raiz, "valid", [("dos.jpg", "201", "BCS_2_75")])
            self.crear_coco(raiz, "test", [])
            destino = raiz / "salida_privada"
            with self.assertRaisesRegex(ValueError, "COCO"):
                crear_export_ordinal(raiz, destino)
            self.assertFalse(destino.exists())

    def test_rechaza_salida_dentro_del_repositorio_antes_de_escribir(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / "coco"
            raiz.mkdir()
            self.crear_coco(raiz, "train", [("uno.jpg", "101", "BCS_3_00")])
            self.crear_coco(raiz, "valid", [("dos.jpg", "201", "BCS_2_75")])
            self.crear_coco(raiz, "test", [])
            repo = Path(__file__).resolve().parents[3]
            destino = repo / "salida_privada_prueba_ordinal"
            self.assertFalse(destino.exists())
            with self.assertRaisesRegex(ValueError, "repositorio"):
                crear_export_ordinal(raiz, destino)
            self.assertFalse(destino.exists())


if __name__ == "__main__":
    unittest.main()
