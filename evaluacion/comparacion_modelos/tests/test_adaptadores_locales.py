import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluacion.comparacion_modelos.adaptadores_locales import normalizar_locales


def guardar_csv(ruta, columnas, filas):
    with ruta.open('w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


class AdaptadoresLocalesTest(unittest.TestCase):
    def test_rechaza_identificador_duplicado_en_manifiesto(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            manifest = raiz / 'manifest.csv'
            guardar_csv(manifest, ['imagen_id', 'vaca_id', 'bcs', 'conjunto', 'sha256_procesada'], [
                {'imagen_id': 'i1', 'vaca_id': 'v1', 'bcs': '3', 'conjunto': 'VALIDACION', 'sha256_procesada': 'hash1'},
                {'imagen_id': 'i1', 'vaca_id': 'v1', 'bcs': '3', 'conjunto': 'VALIDACION', 'sha256_procesada': 'hash2'},
            ])
            with self.assertRaisesRegex(ValueError, 'duplicad'):
                normalizar_locales(manifest, raiz / 'sin_predicciones.csv',
                                   raiz / 'sin_datos.json', raiz / 'sin_regresion.csv')

    def test_empareja_por_hash_y_separa_fotos_distintas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            (raiz / 'dataset' / '_control').mkdir(parents=True)
            manifest = raiz / 'dataset' / '_control' / 'manifest.csv'
            resnet = raiz / 'resnet.csv'
            datos = raiz / 'datos.json'
            regresion = raiz / 'regresion.csv'
            foto1 = raiz / 'dataset' / 'foto1.jpg'
            foto2 = raiz / 'dataset' / 'foto2.jpg'
            coco1 = raiz / 'coco1.jpg'
            coco2 = raiz / 'coco2.jpg'
            foto1.write_bytes(b'foto uno')
            foto2.write_bytes(b'foto dos')
            coco1.write_bytes(b'foto uno')
            coco2.write_bytes(b'otra codificacion')
            h1 = hashlib.sha256(foto1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(foto2.read_bytes()).hexdigest()
            h2_coco = hashlib.sha256(coco2.read_bytes()).hexdigest()
            guardar_csv(manifest,
                        ['imagen_id', 'vaca_id', 'bcs', 'conjunto', 'sha256_procesada', 'ruta_dataset'], [
                            {'imagen_id': 'i1', 'vaca_id': 'v1', 'bcs': '3', 'conjunto': 'VALIDACION', 'sha256_procesada': h1, 'ruta_dataset': 'foto1.jpg'},
                            {'imagen_id': 'i2', 'vaca_id': 'v2', 'bcs': '2.5', 'conjunto': 'VALIDACION', 'sha256_procesada': h2, 'ruta_dataset': 'foto2.jpg'},
                        ])
            guardar_csv(resnet, ['imagen_id', 'vaca_id', 'bcs_real', 'bcs_predicho'], [
                {'imagen_id': 'i1', 'vaca_id': 'v1', 'bcs_real': '3', 'bcs_predicho': '3'},
                {'imagen_id': 'i2', 'vaca_id': 'v2', 'bcs_real': '2.5', 'bcs_predicho': '2.75'},
            ])
            datos.write_text(json.dumps([
                {'clave': 'k1', 'particion': 'val', 'vaca': 'v1', 'bcs': 3, 'sha256': h1, 'ruta': str(coco1)},
                {'clave': 'k2', 'particion': 'val', 'vaca': 'v2', 'bcs': 2.5, 'sha256': h2_coco, 'ruta': str(coco2)},
            ]))
            guardar_csv(regresion, ['clave', 'real', 'pipeline'], [
                {'clave': 'k1', 'real': '3', 'pipeline': '3.25'},
                {'clave': 'k2', 'real': '2.5', 'pipeline': ''},
            ])
            a, b, resumen = normalizar_locales(manifest, resnet, datos, regresion)
            self.assertEqual(len(a), 1)
            self.assertEqual(len(b), 1)
            self.assertEqual(a[0].foto, h1)
            self.assertEqual(b[0].predicho, 3.25)
            self.assertEqual(resumen, {'resnet_fotos': 2, 'regresion_fotos': 2, 'fotos_hash_iguales': 1, 'resnet_sin_par': 1, 'regresion_sin_par': 1})
            coco1.write_bytes(b'archivo modificado')
            with self.assertRaisesRegex(ValueError, 'huella'):
                normalizar_locales(manifest, resnet, datos, regresion)


if __name__ == '__main__':
    unittest.main()
