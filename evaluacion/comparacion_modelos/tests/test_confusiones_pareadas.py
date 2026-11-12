import unittest

from evaluacion.comparacion_modelos.confusiones_pareadas import matrices_pareadas


class MatricesPareadasTest(unittest.TestCase):
    def test_rechaza_fotos_o_estados_distintos(self):
        base = [("foto-a", "vaca-a", 3.0, "ok", 3.0)]
        distinta = [("foto-b", "vaca-a", 3.0, "ok", 3.0)]
        with self.assertRaisesRegex(ValueError, "mismas fotos"):
            matrices_pareadas({"uno": base, "dos": distinta})
        distinta = [("foto-a", "vaca-a", 3.0, "sin_par_valido", None)]
        with self.assertRaisesRegex(ValueError, "estados"):
            matrices_pareadas({"uno": base, "dos": distinta})

    def test_abstencion_ocupa_columna_propia(self):
        filas = [("a", "vaca-a", 3.0, "ok", 3.25),
                 ("b", "vaca-b", 2.5, "sin_par_valido", None)]
        matrices = matrices_pareadas({"uno": filas, "dos": filas})
        self.assertEqual(matrices["uno"][3][4], 1)
        self.assertEqual(matrices["uno"][1][7], 1)


if __name__ == "__main__":
    unittest.main()
