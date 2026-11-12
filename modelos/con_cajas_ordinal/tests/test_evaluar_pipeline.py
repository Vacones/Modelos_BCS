import unittest

from modelos.con_cajas_ordinal.evaluar_pipeline import _metricas, preparar_validacion


class PrepararValidacionTest(unittest.TestCase):
    def setUp(self):
        self.registros = [
            {"particion": "valid", "nombre": "a.jpg", "vaca": "1", "bcs": 3.0,
             "ancho": 100, "alto": 80, "cajas": {"ass": (.5, .5, .2, .2)}},
            {"particion": "valid", "nombre": "b.jpg", "vaca": "2", "bcs": 3.25,
             "ancho": 100, "alto": 80, "cajas": {"ass": (.5, .5, .2, .2)}},
        ]
        self.filas = [
            {"split": "val", "file_name": "a.jpg", "status": "ok", "boxes": [
                {"class": 0, "xyxy": [0, 0, 100, 80]},
                {"class": 1, "xyxy": [40, 24, 60, 56]},
            ]},
            {"split": "val", "file_name": "b.jpg", "status": "sin_par_valido", "boxes": []},
        ]

    def test_usa_caja_predicha_y_conserva_abstencion(self):
        cubiertas, todas = preparar_validacion(self.registros, self.filas)
        self.assertEqual(len(cubiertas), 1)
        self.assertEqual(cubiertas[0]["cajas"]["ass"], (.5, .5, .2, .4))
        self.assertEqual([r["estado"] for r in todas], ["ok", "sin_par_valido"])

    def test_rechaza_foto_faltante_o_particion_test(self):
        with self.assertRaisesRegex(ValueError, "Faltan"):
            preparar_validacion(self.registros, self.filas[:1])
        with self.assertRaisesRegex(ValueError, "partición"):
            preparar_validacion(self.registros, self.filas + [{**self.filas[0], "split": "test"}])

    def test_rechaza_estado_incompatible_con_cajas(self):
        filas = [dict(self.filas[0], status="sin_par_valido"), self.filas[1]]
        with self.assertRaisesRegex(ValueError, "Estado"):
            preparar_validacion(self.registros, filas)

    def test_abstencion_cuenta_en_cobertura_y_acierto_global(self):
        fotos = [
            {"vaca": "1", "bcs_real": 3.0, "bcs_pred": 3.0,
             "bcs_cont": 3.1, "estado": "ok"},
            {"vaca": "2", "bcs_real": 3.25, "bcs_pred": "",
             "bcs_cont": "", "estado": "sin_par_valido"},
        ]
        vacas = [{"vaca": "1", "bcs_real": 3.0, "bcs_pred": 3.0,
                  "bcs_cont": 3.1, "estado": "ok"}]
        resultado = _metricas(fotos, vacas)
        self.assertEqual(resultado["cobertura_fotos"], 0.5)
        self.assertEqual(resultado["cobertura_vacas"], 0.5)
        self.assertEqual(resultado["exactos_categoria_global"], 1)
        self.assertEqual(resultado["abstenciones"]["sin_par_valido"], 1)


if __name__ == "__main__":
    unittest.main()
