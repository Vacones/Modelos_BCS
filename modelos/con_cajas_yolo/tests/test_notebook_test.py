"""Comprobar la evaluación de pesos existentes sin necesitar frameworks GPU."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebookc40da8f492-test.ipynb').read_text())
SOURCES = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


def source_containing(text):
    return next(s for s in SOURCES if text in s)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestFinalNotebook(unittest.TestCase):
    def test_no_training_and_clean_outputs(self):
        for source in SOURCES:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('train', 'fit', 'compile'))
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def make_outputs(self, folder):
        cfg = {'version': 'secuencial_2_regresion', 'semilla': 42,
               'yolo': {'tam': 640, 'lote': 4, 'version': '8.4.0'},
               'regresor': {'tam': 224},
               'inferencia': {'confianza': .25, 'cobertura': .8}}
        selection = {'fase': 'cabeza', 'mae_validacion': {'cabeza': .23, 'ajuste_fino': .24},
                     'checkpoint': 'regresor/cabeza/best.keras'}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'configuracion.json').write_text(json.dumps(cfg))
        (folder / 'seleccion_regresor.json').write_text(json.dumps(selection))
        for rel, content in [('detector/weights/best.pt', b'detector selected'),
                             ('regresor/cabeza/best.keras', b'head selected'),
                             ('regresor/ajuste_fino/best.keras', b'wrong phase')]:
            p = folder / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)

    def environment(self, origin, output):
        # Supply only the CPU dependencies used by the actual restoration cell.
        class Seeds:
            @staticmethod
            def seed(value):
                pass
        class Numpy:
            random = Seeds()
        class Torch:
            manual_seed = staticmethod(Seeds.seed)
        return dict(Path=Path, json=json, zipfile=zipfile, shutil=shutil,
                    ORIGEN_ENTRENAMIENTO=str(origin), SALIDA=output,
                    ULTRALYTICS_VERSION='8.4.0', huella_archivo=sha,
                    np=Numpy(), torch=Torch())

    def test_restore_uses_selected_head_and_rejects_different_existing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_outputs(root / 'origin')
            env = self.environment(root / 'origin', root / 'evaluation')
            exec(source_containing('configuracion_candidata'), env)
            self.assertEqual(env['MEJOR_REGRESOR'].read_bytes(), b'head selected')
            self.assertEqual(env['CONFIANZA'], .25)
            self.assertEqual(env['COBERTURA_GRUPA'], .8)
            self.assertFalse((root / 'evaluation/regresor/ajuste_fino').exists())
            env['MEJOR_REGRESOR'].write_bytes(b'weights from another experiment')
            with self.assertRaisesRegex(AssertionError, 'otro modelo'):
                exec(source_containing('configuracion_candidata'), env)

    def test_rejects_selection_that_did_not_win_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_outputs(root / 'origin')
            p = root / 'origin/seleccion_regresor.json'
            selection = json.loads(p.read_text())
            selection.update(fase='ajuste_fino', checkpoint='regresor/ajuste_fino/best.keras')
            p.write_text(json.dumps(selection))
            with self.assertRaises(AssertionError):
                exec(source_containing('configuracion_candidata'),
                     self.environment(root / 'origin', root / 'evaluation'))

    def test_test_selection_is_locked_before_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['detector.pt', 'regresor.keras', 'configuracion.json']:
                (root / name).write_bytes(name.encode())
            env = dict(SALIDA=root, PARTICION_EVALUACION='test', TEST_FINAL_CONFIRMADO=True,
                       MEJOR_YOLO=root / 'detector.pt', MEJOR_REGRESOR=root / 'regresor.keras',
                       ruta_configuracion=root / 'configuracion.json', huella_archivo=sha, json=json)
            lock_source = source_containing('seleccion_final =')
            exec(lock_source, env)
            self.assertTrue((root / 'seleccion_test.json').is_file())
            (root / 'regresor.keras').write_bytes(b'another regressor')
            with self.assertRaisesRegex(AssertionError, 'selección cambió'):
                exec(lock_source, env)

    def test_association_implementation_matches_training_notebook(self):
        training = json.loads((ROOT / 'notebookc40da8f492.ipynb').read_text())
        reference = next(''.join(c['source']) for c in training['cells']
                         if ''.join(c['source']).startswith('def asociar_grupa'))
        # Structural comparison tolerates whitespace, but forbids an altered decision rule.
        self.assertEqual(ast.dump(ast.parse(reference)),
                         ast.dump(ast.parse(source_containing('def asociar_grupa'))))


if __name__ == '__main__':
    unittest.main()
