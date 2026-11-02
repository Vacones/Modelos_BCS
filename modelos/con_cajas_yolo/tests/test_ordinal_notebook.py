"""Pruebas de la cabeza real del notebook, sin descargar ImageNet ni usar fotos."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'notebookc40da8f492-ordinal.ipynb').read_text())
CODE = [''.join(c['source']) for c in NB['cells'] if c['cell_type'] == 'code']
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None


class NotebookStructure(unittest.TestCase):
    def test_syntax_and_no_old_regression_loss(self):
        for source in CODE:
            compile(source, '<ordinal notebook>', 'exec')
            self.assertNotIn('loss="mae"', source)
            self.assertNotIn('callbacks.EarlyStopping(', source)
        base = json.loads((ROOT / 'notebookc40da8f492.ipynb').read_text())
        association = lambda nb: next(''.join(c['source']) for c in nb['cells']
                                      if ''.join(c['source']).startswith('def asociar_grupa'))
        self.assertEqual(ast.dump(ast.parse(association(base))), ast.dump(ast.parse(association(NB))))
        self.assertIn('REUTILIZAR_YOLO = False', '\n'.join(CODE))
        self.assertTrue(all(not c.get('outputs') for c in NB['cells']))


@unittest.skipIf(tf is None, 'Necesita TensorFlow CPU/GPU y NumPy')
class OrdinalHead(unittest.TestCase):
    notebook_path = ROOT / "notebookc40da8f492-ordinal.ipynb"
    @classmethod
    def setUpClass(cls):
        nb = json.loads(cls.notebook_path.read_text())
        cls.code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        cls.classes = [2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75]
        env = dict(tf=tf, keras=keras, layers=keras.layers, np=np, CLASES_BCS=cls.classes,
                   TAM_RECORTE=32, WEIGHT_DECAY=1e-4, DROPOUT=.35)
        definitions = next(s for s in cls.code if 'class UmbralesOrdenados' in s).split('entrada = keras.Input')[0]
        exec(definitions, env)
        decode = next(s for s in cls.code if s.startswith('def leer_recorte')).split('conjuntos =')[0]
        exec(decode, env)
        cls.env = env

    def test_labels_include_extremes_and_middle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'crop.png'
            tf.io.write_file(str(path), tf.io.encode_png(tf.zeros((32, 32, 3), tf.uint8)))
            for bcs, expected in [(2.25, [0]*6), (3., [1,1,1,0,0,0]), (3.75, [1]*6)]:
                _, target = self.env['leer_recorte'](str(path), tf.constant(bcs))
                np.testing.assert_array_equal(target.numpy(), expected)

    def test_monotonic_probabilities_and_decode(self):
        layer = self.env['UmbralesOrdenados'](6)
        logits = layer(tf.constant([[-100.], [0.], [100.]]))
        layer.incrementos.assign([-20., -3., 0., 3., 20.])
        logits = layer(tf.constant([[-100.], [0.], [100.]]))
        self.assertTrue(np.all(np.diff(logits.numpy(), axis=1) < 0))
        categories, expected, probabilities = self.env['decodificar_ordinal'](logits.numpy())
        self.assertEqual(categories[0], 2.25)
        self.assertEqual(categories[-1], 3.75)
        self.assertTrue(np.all(probabilities >= -1e-6))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.all((expected >= 2.25) & (expected <= 3.75)))

    def test_metric_in_bcs_units(self):
        metric = self.env['MAEOrdinal'](self.classes)
        metric.update_state(tf.constant([[1.,1.,1.,0.,0.,0.], [0.,0.,0.,0.,0.,0.]]),
                            tf.constant([[2.,1.,-1.,-2.,-3.,-4.], [-1.,-2.,-3.,-4.,-5.,-6.]]))
        self.assertAlmostEqual(float(metric.result()), .125)

    def test_train_save_reload_optimizer_and_custom_metric(self):
        keras.utils.set_random_seed(42)
        inputs = keras.Input((4,))
        score = keras.layers.Dense(1, use_bias=False)(inputs)
        model = keras.Model(inputs, self.env['UmbralesOrdenados'](6)(score))
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[self.env['MAEOrdinal'](self.classes)])
        x = np.ones((2, 4), np.float32)
        y = np.array([[1,1,1,0,0,0], [1,1,1,1,0,0]], np.float32)
        result = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        before = model(x).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ordinal.keras'
            model.save(path)
            loaded = keras.models.load_model(path)
            np.testing.assert_allclose(before, loaded(x).numpy(), atol=1e-6)
            self.assertEqual(int(loaded.optimizer.iterations), int(model.optimizer.iterations))
            result = loaded.train_on_batch(x, y, return_dict=True)
            self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_actual_efficientnet_graph_and_fine_tuning(self):
        # Build the notebook's real graph at the smallest supported resolution.
        # Random weights are only for this smoke test; the notebook uses ImageNet.
        env = dict(self.env, TAM_RECORTE=32, PESOS_EFFICIENTNET=None, NUM_UMBRALES=6,
                   SEMILLA=42, LR_CABEZA=3e-4, LR_AJUSTE=5e-6, CAPAS_AJUSTE=20,
                   DROPOUT=.35, WEIGHT_DECAY=1e-4)
        source = next(s for s in self.code if 'class UmbralesOrdenados' in s)
        graph = source[source.index('entrada = keras.Input'):]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(graph, env)
        model = env['clasificador']
        x = np.zeros((2, 32, 32, 3), np.float32)
        y = np.array([[1,1,1,0,0,0], [1,1,1,1,0,0]], np.float32)
        self.assertEqual(model.output_shape, (None, 6))
        logs = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(all(np.isfinite(v) for v in logs.values()))
        phase = next(s for s in self.code if s.startswith('AJUSTE ='))
        begin = phase.index('        base = clasificador.get_layer')
        end = phase.index('\n    if epoca_inicial <', begin)
        import textwrap
        exec(textwrap.dedent(phase[begin:end]), env)
        base = env['base']
        self.assertTrue(any(layer.trainable for layer in base.layers))
        self.assertTrue(all(not layer.trainable for layer in base.layers
                            if isinstance(layer, keras.layers.BatchNormalization)))
        logs = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(all(np.isfinite(v) for v in logs.values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'efficientnet_ordinal.keras'
            model.save(path)
            loaded = keras.models.load_model(path)
            np.testing.assert_allclose(model(x).numpy(), loaded(x).numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
