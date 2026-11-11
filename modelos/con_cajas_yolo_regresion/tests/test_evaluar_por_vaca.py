import csv
import json
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_yolo_regresion.evaluar_por_vaca import (
    Registro,
    cargar_validacion,
    resumir_por_vaca,
)


class EvaluacionPorVacaTest(unittest.TestCase):
    def test_cada_vaca_pesa_igual_aunque_tenga_distintas_fotos(self):
        registros = [
            Registro("ficticia_a", 3.0, 3.0, 3.0, 2.5),
            Registro("ficticia_a", 3.0, 3.0, 3.0, 3.0),
            Registro("ficticia_a", 3.0, 3.0, 3.0, 3.5),
            Registro("ficticia_b", 4.0, 3.0, 4.0, 3.0),
        ]

        resumen = resumir_por_vaca(registros)

        self.assertEqual(resumen["vacas_total"], 2)
        self.assertEqual(resumen["vacas_cubiertas"], 2)
        self.assertEqual(resumen["fotos_cubiertas"], 4)
        self.assertAlmostEqual(resumen["mismas_fotos"]["pipeline"]["mae"], 0.5)

    def test_compara_solo_fotos_donde_el_pipeline_predijo(self):
        registros = [
            Registro("ficticia_a", 3.0, 3.0, 2.5, 2.75),
            Registro("ficticia_a", 3.0, 3.0, 4.0, None),
            Registro("ficticia_b", 4.0, 3.0, 4.0, None),
        ]

        resumen = resumir_por_vaca(registros)

        self.assertEqual(resumen["vacas_total"], 2)
        self.assertEqual(resumen["vacas_cubiertas"], 1)
        self.assertEqual(resumen["fotos_total"], 3)
        self.assertEqual(resumen["fotos_cubiertas"], 1)
        self.assertAlmostEqual(resumen["cobertura_vacas"], 0.5)
        self.assertAlmostEqual(resumen["cobertura_fotos"], 1 / 3)
        self.assertAlmostEqual(resumen["todas_fotos"]["regresor_gt"]["mae"], 0.125)
        self.assertAlmostEqual(resumen["mismas_fotos"]["regresor_gt"]["mae"], 0.5)
        self.assertAlmostEqual(resumen["mismas_fotos"]["pipeline"]["mae"], 0.25)

    def test_rechaza_bcs_distinto_para_la_misma_vaca(self):
        registros = [
            Registro("ficticia_a", 3.0, 3.0, 3.0, 3.0),
            Registro("ficticia_a", 3.25, 3.0, 3.0, 3.0),
        ]

        with self.assertRaisesRegex(ValueError, "BCS distintos"):
            resumir_por_vaca(registros)

    def test_cobertura_cero_no_inventa_mae(self):
        registros = [Registro("ficticia_a", 3.0, 3.0, 3.0, None)]

        resumen = resumir_por_vaca(registros)

        self.assertEqual(resumen["vacas_cubiertas"], 0)
        self.assertEqual(resumen["mismas_fotos"]["pipeline"]["vacas"], 0)
        self.assertIsNone(resumen["mismas_fotos"]["pipeline"]["mae"])

    def test_rechaza_claves_duplicadas_en_metadatos(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            datos = raiz / "datos.json"
            predicciones = raiz / "predicciones_val.csv"
            fila = {
                "clave": "valid/ejemplo_ficticio.jpg",
                "particion": "val",
                "vaca": "ficticia_a",
                "bcs": 3.0,
            }
            datos.write_text(json.dumps([fila, fila]), encoding="utf-8")
            with predicciones.open("w", newline="", encoding="utf-8") as archivo:
                escritor = csv.DictWriter(
                    archivo,
                    fieldnames=["clave", "real", "base", "regresor_gt", "pipeline"],
                )
                escritor.writeheader()
                escritor.writerow({
                    "clave": fila["clave"], "real": 3.0, "base": 3.0,
                    "regresor_gt": 3.0, "pipeline": 3.0,
                })

            with self.assertRaisesRegex(ValueError, "Metadatos duplicados"):
                cargar_validacion(datos, predicciones)


if __name__ == "__main__":
    unittest.main()
