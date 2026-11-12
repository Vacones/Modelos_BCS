import unittest

from evaluacion.comparacion_modelos.evaluador_comun import Registro, comparar


class EvaluadorComunTest(unittest.TestCase):
    def test_pesa_cada_vaca_igual_y_conserva_abstenciones(self):
        a = [
            Registro('a', 'f1', 'v1', 3.0, 3.0),
            Registro('a', 'f2', 'v1', 3.0, 3.0),
            Registro('a', 'f3', 'v2', 2.5, None),
        ]
        b = [
            Registro('b', 'f1', 'v1', 3.0, 3.25),
            Registro('b', 'f2', 'v1', 3.0, 3.25),
            Registro('b', 'f3', 'v2', 2.5, 3.0),
        ]
        resultado = comparar(a, b)
        self.assertEqual(resultado['a']['vacas_cubiertas'], 1)
        self.assertEqual(resultado['a']['fotos_cubiertas'], 2)
        self.assertEqual(resultado['b']['mae_vaca'], 0.375)
        self.assertEqual(resultado['pareado']['vacas'], 1)
        self.assertEqual(resultado['pareado']['mae_a'], 0.0)
        self.assertEqual(resultado['pareado']['mae_b'], 0.25)

    def test_rechaza_diferente_referencia_en_la_misma_foto(self):
        a = [Registro('a', 'f1', 'v1', 3.0, 3.0)]
        b = [Registro('b', 'f1', 'v1', 2.75, 2.75)]
        with self.assertRaisesRegex(ValueError, 'BCS'):
            comparar(a, b)

    def test_rechaza_fotografias_no_coincidentes(self):
        a = [Registro('a', 'f1', 'v1', 3.0, 3.0)]
        b = [Registro('b', 'f2', 'v1', 3.0, 3.0)]
        with self.assertRaisesRegex(ValueError, 'fotografías'):
            comparar(a, b)

    def test_pareado_usa_solo_fotos_con_prediccion_en_ambos(self):
        a = [Registro('a', 'f1', 'v1', 3.0, 3.0),
             Registro('a', 'f2', 'v1', 3.0, 4.0)]
        b = [Registro('b', 'f1', 'v1', 3.0, 3.25),
             Registro('b', 'f2', 'v1', 3.0, None)]
        pareado = comparar(a, b)['pareado']
        self.assertEqual(pareado['fotos'], 1)
        self.assertEqual(pareado['mae_a'], 0.0)
        self.assertEqual(pareado['mae_b'], 0.25)


if __name__ == '__main__':
    unittest.main()
