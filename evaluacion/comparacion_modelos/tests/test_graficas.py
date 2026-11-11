import unittest

from evaluacion.comparacion_modelos.generar_graficas import validar_fila


class ValidarFilaTest(unittest.TestCase):
    def test_rechaza_aciertos_mayores_a_predicciones_emitidas(self):
        fila = {
            "id": "ficticio", "salida": "categorica", "elegibles_fotos": "10",
            "clasificadas_fotos": "6", "aciertos_exactos_global": "4",
            "aciertos_025_global": "7", "aciertos_050_global": "8",
            "mae_clasificadas_foto": "0.25", "evidencia": "documentado",
        }
        with self.assertRaisesRegex(ValueError, "superan las predicciones"):
            validar_fila(fila)

    def test_admite_salida_continua_sin_acierto_exacto(self):
        fila = {
            "id": "ficticio", "salida": "continua", "elegibles_fotos": "10",
            "clasificadas_fotos": "6", "aciertos_exactos_global": "",
            "aciertos_025_global": "3", "aciertos_050_global": "5",
            "mae_clasificadas_foto": "0.25", "evidencia": "recalculado_local",
        }
        self.assertEqual(validar_fila(fila)["aciertos_025_global"], 3)


if __name__ == "__main__":
    unittest.main()
