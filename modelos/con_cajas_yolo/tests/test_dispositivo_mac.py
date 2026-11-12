import unittest

from modelos.con_cajas_yolo.vacas.detection import seleccionar_dispositivo


class DispositivoMacTest(unittest.TestCase):
    def test_acepta_mps_cuando_disponible(self):
        self.assertEqual(seleccionar_dispositivo("mps", cuda=False, mps=True), "mps")

    def test_rechaza_mps_ausente(self):
        with self.assertRaisesRegex(RuntimeError, "MPS"):
            seleccionar_dispositivo("mps", cuda=False, mps=False)

    def test_conserva_cuda_y_cpu(self):
        self.assertEqual(seleccionar_dispositivo("0", cuda=True, mps=False), "0")
        self.assertEqual(seleccionar_dispositivo("cpu", cuda=False, mps=False), "cpu")


if __name__ == "__main__":
    unittest.main()
