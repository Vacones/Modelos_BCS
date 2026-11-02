import ast
import json
import unittest
import test_ordinal_notebook as base

class RegularizedHead(base.OrdinalHead):
    notebook_path = base.ROOT / 'notebookc40da8f492-ordinal-regularizado.ipynb'

class OrdinalWorkflow(unittest.TestCase):
    def test_test_notebook_does_not_train_and_versions_are_supported(self):
        nb = json.loads((base.ROOT / 'notebookc40da8f492-ordinal-test.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type'] != 'code':
                continue
            s = ''.join(c['source'])
            tree = ast.parse(s)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('fit', 'train'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('secuencial_4_ordinal_regularizado', code)
        self.assertIn('class UmbralesOrdenados', code)
        self.assertIn('decodificar_ordinal(clasificador.predict', code)

    def test_notebooks_are_clean_and_compile(self):
        for filename in ['notebookc40da8f492-ordinal-test.ipynb',
                         'notebookc40da8f492-ordinal-regularizado.ipynb']:
            nb = json.loads((base.ROOT / filename).read_text())
            for i, c in enumerate(nb['cells']):
                if c['cell_type'] == 'code':
                    compile(''.join(c['source']), f'{filename}:{i}', 'exec')
                    self.assertEqual(c['outputs'], [])
