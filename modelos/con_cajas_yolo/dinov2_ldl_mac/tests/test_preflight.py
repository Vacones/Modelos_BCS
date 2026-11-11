import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from modelos.con_cajas_yolo.dinov2_ldl_mac.preflight import (
    verificar_dataset,
    verificar_entorno,
    verificar_salida,
)


class PreflightTest(unittest.TestCase):
    def test_acepta_train_y_valid_sin_leer_test(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            for split in ("train", "valid"):
                carpeta = raiz / split
                carpeta.mkdir()
                (carpeta / "foto.jpg").write_bytes(b"imagen ficticia")
                (carpeta / "_annotations.coco.json").write_text(json.dumps({
                    "categories": [{"id": 1, "name": "cow"}, {"id": 2, "name": "ass"}],
                    "images": [{"id": 1, "file_name": "foto.jpg"}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
                }))
            self.assertEqual(verificar_dataset(raiz), {"train": 1, "valid": 1})

    def test_rechaza_foto_faltante(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            for split in ("train", "valid"):
                carpeta = raiz / split
                carpeta.mkdir()
                (carpeta / "_annotations.coco.json").write_text(json.dumps({
                    "categories": [{"id": 1, "name": "cow"}, {"id": 2, "name": "ass"}],
                    "images": [{"id": 1, "file_name": "ausente.jpg"}],
                    "annotations": [],
                }))
            with self.assertRaisesRegex(ValueError, "faltan archivos de imagen"):
                verificar_dataset(raiz)

    def test_rechaza_salida_que_ya_existe(self):
        with tempfile.TemporaryDirectory() as temporal:
            with self.assertRaisesRegex(ValueError, "ya existe"):
                verificar_salida(Path(temporal))

    def test_rechaza_entorno_sin_dependencias(self):
        with self.assertRaisesRegex(ValueError, "Faltan dependencias"):
            verificar_entorno({}, "darwin", "arm64", (3, 12))

    def test_ejecutor_comprueba_destino_explicito_antes_de_crearlo(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            script_original = Path(__file__).resolve().parents[1] / "ejecutar_mac.sh"
            shutil.copy2(script_original, raiz / "ejecutar_mac.sh")
            binario = raiz / ".venv-mac" / "bin" / "python"
            binario.parent.mkdir(parents=True)
            registro = raiz / "llamadas.txt"
            binario.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$REGISTRO_PRUEBA\"\n"
                "if [ \"$1\" = '-m' ]; then echo /dev/null; exit 0; fi\n"
                "if [ \"$1\" = 'preflight.py' ]; then "
                "[ ! -e \"$RUN_DESTINO\" ] || exit 31; exit 0; fi\n"
                "exit 17\n"
            )
            binario.chmod(0o755)
            datos = raiz / "datos"
            datos.mkdir()
            salida = raiz / "resultados_privados" / "corrida_fija"
            salida.parent.mkdir()
            entorno = os.environ.copy()
            entorno.update({"RUN_DESTINO": str(salida), "REGISTRO_PRUEBA": str(registro)})
            proceso = subprocess.run(
                ["bash", str(raiz / "ejecutar_mac.sh"), str(datos)],
                env=entorno, capture_output=True, text=True,
            )
            self.assertNotEqual(proceso.returncode, 0)
            llamadas = registro.read_text().splitlines()
            self.assertIn(f"preflight.py --datos {datos} --salida {salida}", llamadas)
            self.assertTrue(salida.is_dir(), msg=(llamadas, proceso.stdout, proceso.stderr))


if __name__ == "__main__":
    unittest.main()
