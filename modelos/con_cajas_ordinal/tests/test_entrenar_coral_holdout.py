import unittest

from modelos.con_cajas_ordinal.entrenar_coral_holdout import (
    construir_salidas,
    nivel_bcs,
    preparar_ajuste,
    separar_train_valid,
)


def registro(particion, vaca, nombre, bcs):
    return {"particion": particion, "vaca": vaca, "nombre": nombre, "bcs": bcs}


class HoldoutCoralTest(unittest.TestCase):
    def test_separa_train_valid_y_no_devuelve_test(self):
        registros = [registro("train", "101", "a.jpg", 3.0),
                     registro("valid", "201", "b.jpg", 2.75),
                     registro("test", "301", "c.jpg", 3.5)]
        train, valid = separar_train_valid(registros)
        self.assertEqual([r["nombre"] for r in train], ["a.jpg"])
        self.assertEqual([r["nombre"] for r in valid], ["b.jpg"])

    def test_rechaza_vaca_en_train_y_valid(self):
        registros = [registro("train", "101", "a.jpg", 3.0),
                     registro("valid", "101", "b.jpg", 3.0)]
        with self.assertRaisesRegex(ValueError, "train y valid"):
            separar_train_valid(registros)

    def test_predicciones_por_foto_y_promedio_por_vaca(self):
        valid = [registro("valid", "201", "a.jpg", 3.0),
                 registro("valid", "201", "b.jpg", 3.0),
                 registro("valid", "202", "c.jpg", 2.5)]
        fotos, vacas = construir_salidas(valid, [[10, 10, -10, -10, -10, -10],
                                                [0, 0, -10, -10, -10, -10],
                                                [-10, -10, -10, -10, -10, -10]])
        self.assertEqual(len(fotos), 3)
        self.assertEqual(len(vacas), 2)
        self.assertEqual(vacas[0]["n_fotos"], 2)
        self.assertEqual(vacas[0]["bcs_pred"], 2.75)
        self.assertEqual(vacas[0]["estado"], "ok")
        self.assertEqual(fotos[0]["particion"], "valid")

    def test_rechaza_salida_sin_seis_umbrales(self):
        valid = [registro("valid", "201", "a.jpg", 3.0)]
        with self.assertRaisesRegex(ValueError, "seis umbrales"):
            construir_salidas(valid, [[1.0, 2.0]])

    def test_catalogo_conserva_225_y_rechaza_valor_desconocido(self):
        self.assertEqual(nivel_bcs(2.25), 0)
        self.assertEqual(nivel_bcs(3.75), 6)
        with self.assertRaisesRegex(ValueError, "catálogo"):
            nivel_bcs(2.60)

    def test_ajuste_recarga_cabeza_y_descongela_solo_block7_y_top(self):
        eventos = []

        class Capa:
            def __init__(self, name):
                self.name, self.trainable = name, False

        class BatchNormalization(Capa):
            pass

        class Base:
            def __init__(self):
                self.trainable = False
                self.layers = [Capa("block6a_conv"), Capa("block7a_conv"),
                               BatchNormalization("top_bn"), Capa("top_conv")]

        class Modelo:
            def load_weights(self, ruta):
                eventos.append(("load", ruta))

        base = Base()
        entrenables = preparar_ajuste(Modelo(), base, "mejor_cabeza.weights.h5", BatchNormalization)
        self.assertEqual(eventos, [("load", "mejor_cabeza.weights.h5")])
        self.assertTrue(base.trainable)
        self.assertEqual(entrenables, 2)
        self.assertEqual([c.trainable for c in base.layers], [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
