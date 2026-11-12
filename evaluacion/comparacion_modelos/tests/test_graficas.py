import unittest

from evaluacion.comparacion_modelos.generar_graficas import validar_fila, validar_pareados, validar_vacas


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

    def test_comparacion_pareada_exige_mismos_denominadores(self):
        filas = [
            {"id": "uno", "modelo": "Uno", "fotos_pareadas": "24", "vacas_pareadas": "8",
             "mae_foto_categoria": "0.2", "mae_vaca_media_fotos": "0.18"},
            {"id": "dos", "modelo": "Dos", "fotos_pareadas": "23", "vacas_pareadas": "8",
             "mae_foto_categoria": "0.3", "mae_vaca_media_fotos": "0.25"},
        ]
        with self.assertRaisesRegex(ValueError, "denominadores"):
            validar_pareados(filas)

    def test_resumen_por_vaca_rechaza_cobertura_imposible(self):
        fila = {"id": "modelo", "modelo": "Modelo", "salida": "categorica",
                "vacas_elegibles": "10", "vacas_cubiertas": "11", "mae_vaca": "0.2",
                "regla": "media"}
        with self.assertRaisesRegex(ValueError, "cobertura"):
            validar_vacas([fila])


if __name__ == "__main__":
    unittest.main()
