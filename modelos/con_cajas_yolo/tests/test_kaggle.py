"""Pruebas CPU del notebook y contratos; no requieren NumPy, TensorFlow ni PyTorch."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kaggle_src'))
import common
import data
import detector
import evaluation
import regressor
import run


def code_hash():
    sources = {p.name: p.read_text() for p in (ROOT / 'kaggle_src').glob('*.py')}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'input' / 'nested' / 'coco'
        for split, number in [('train', 3), ('valid', 1), ('test', 1)]:
            folder = self.source / split
            folder.mkdir(parents=True)
            images, annotations = [], []
            for i in range(number):
                name = f'photo_{i}.png'
                Image.new('RGB', (30, 40), (i*30, len(split)*20, i*10+len(split))).save(folder / name)
                tags = [str(1000+100*len(split)+i), 'BCS_2_75']
                if split == 'train' and i == 2:
                    tags += ['999', 'BCS_3']
                images.append({'id': i, 'file_name': name, 'width': 30, 'height': 40,
                               'extra': {'user_tags': tags}})
                annotations += [{'id': 2*i, 'image_id': i, 'category_id': 2,
                                 'bbox': [0, 0, 30, 40.01], 'iscrowd': 0},
                                {'id': 2*i+1, 'image_id': i, 'category_id': 1,
                                 'bbox': [5, 5, 20, 20], 'iscrowd': 0}]
            common.save(folder / '_annotations.coco.json', {
                'images': images, 'annotations': annotations,
                'categories': [{'id': 0, 'name': 'container'}, {'id': 1, 'name': 'ass'}, {'id': 2, 'name': 'cow'}]})
        # valid y train tienen igual longitud de nombre: distinguir identidades y fotos.
        valid = self.source / 'valid/_annotations.coco.json'
        d = common.load(valid)
        d['images'][0]['extra']['user_tags'] = ['777', 'BCS_3_5']
        common.save(valid, d)
        Image.new('RGB', (30, 40), (222, 11, 43)).save(valid.parent / 'photo_0.png')
        self.cfg = {'DATA_ROOT': str(self.source), 'WORK_ROOT': str(self.base/'work'),
                    'CODE_HASH': code_hash(), 'BOX_ROUNDING_TOLERANCE_PX': .02,
                    'EXPERIMENT_ID': 'test', 'SEED': 42, 'RESUME_ROOT': None,
                    'DEADLINE_UNIX': None, 'SMOKE_EPOCHS': 2,
                    'EVALUATION_SPLIT': 'val', 'ULTRALYTICS_VERSION': '8.4.0',
                    'DETECTOR': dict(epochs=80, weights='yolo26n.pt', imgsz=640, batch=4,
                                     device='0', patience=15, save_period=5), 'REGRESSOR_PHASE': 'all',
                    'REGRESSOR': dict(size=32, batch=2, device='0', weights='imagenet',
                                      head_epochs=3, fine_tuning_epochs=2, head_lr=.001,
                                      fine_tuning_lr=.00001, fine_layers=2, patience=8)}

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data.audit(self.cfg)
            data.partition(self.cfg)
            data.export_yolo(self.cfg)
            data.samples(self.cfg)


class DatasetCase(DatasetFixture):
    def test_discovery_and_coco_export(self):
        self.assertEqual(data.discover(self.base/'input'), [self.source])
        self.prepare()
        root = common.artifacts(self.cfg)
        summary = common.load(root/'reports/audit_summary.json')
        self.assertEqual([s['detector_images'] for s in summary], [3, 1, 1])
        self.assertEqual([s['regressor_images'] for s in summary], [2, 1, 1])
        self.assertAlmostEqual(summary[0]['regressor_pct'], 50)
        self.assertEqual(len(common.load(root/'reports/audit_issues.json')['box_corrections']), 5)
        labels = list((root/'prepared/yolo/labels').rglob('*.txt'))
        self.assertEqual(len(labels), 5)
        for label in labels:
            lines = [line.split() for line in label.read_text().splitlines()]
            self.assertEqual([line[0] for line in lines], ['0', '1'])
            self.assertTrue(all(0 <= float(v) <= 1 for line in lines for v in line[1:]))
        samples = common.load(root/'manifests/regression_samples.json')
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]['bcs'], 2.75)
        for row in samples:
            with Image.open(root/row['crop']) as picture:
                self.assertEqual(picture.size, (32, 32))
        before = (root/'manifests/manifest.json').stat().st_mtime_ns
        self.prepare()
        self.assertEqual(before, (root/'manifests/manifest.json').stat().st_mtime_ns)

    def test_rejects_identity_leak_even_from_ambiguous_photo(self):
        path = self.source/'test/_annotations.coco.json'
        d = common.load(path)
        d['images'][0]['extra']['user_tags'] = ['999', 'BCS_3']
        common.save(path, d)
        with self.assertRaisesRegex(RuntimeError, 'Fuga'):
            data.audit(self.cfg)

    def test_rejects_image_leak_and_large_overflow(self):
        source = self.source/'train/photo_0.png'
        common.copy_atomic(source, self.source/'test/photo_0.png')
        with self.assertRaisesRegex(RuntimeError, 'Fuga'):
            data.audit(self.cfg)
        p = self.source/'train/_annotations.coco.json'
        d = common.load(p)
        d['annotations'][0]['bbox'][3] = 42
        common.save(p, d)
        with self.assertRaisesRegex(RuntimeError, 'fuera de imagen'):
            data.audit(self.cfg)

    def test_detects_modified_inputs_and_partitions(self):
        self.prepare()
        run.validate_inputs(self.cfg)
        p = self.source/'train/photo_0.png'
        original = p.read_bytes()
        Image.new('RGB', (30, 40), 'white').save(p)
        with self.assertRaisesRegex(RuntimeError, 'Cambió una imagen'):
            run.validate_inputs(self.cfg)
        p.write_bytes(original)
        split = common.artifacts(self.cfg)/'manifests/splits.json'
        s = common.load(split)
        key = next(iter(s['assignments']))
        s['assignments'][key] = 'val' if s['assignments'][key] != 'val' else 'train'
        common.save(split, s)
        with self.assertRaisesRegex(RuntimeError, 'Particiones alteradas'):
            run.validate_inputs(self.cfg)

    def test_restore_portability_and_integrity(self):
        self.prepare()
        root = common.artifacts(self.cfg)
        common.save(root/'configs/config.json', self.cfg)
        run.export(self.cfg)
        restored = {**self.cfg, 'WORK_ROOT': str(self.base/'second_session'), 'RESUME_ROOT': str(root)}
        common.restore(restored)
        run.validate_inputs(restored)
        run.verify_prepared(restored, 'detector-train')
        yaml = common.load(common.artifacts(restored)/'configs/detector.yaml')
        self.assertTrue(yaml['path'].startswith(restored['WORK_ROOT']))
        common.save(common.artifacts(restored)/'configs/config.json', {'continued': True})
        common.restore(restored)  # No sobrescribir el progreso al repetir Run All.
        self.assertEqual(common.load(common.artifacts(restored)/'configs/config.json'), {'continued': True})
        common.save(root/'manifests/excluded_bcs.csv', {'tampered': True})
        other = {**restored, 'WORK_ROOT': str(self.base/'third_session')}
        with self.assertRaisesRegex(RuntimeError, 'hashes incorrectos'):
            common.restore(other)

    def test_incompatible_preparation_is_rejected(self):
        self.prepare()
        other = copy.deepcopy(self.cfg)
        other['REGRESSOR']['size'] = 64
        with self.assertRaisesRegex(RuntimeError, 'incompatible'):
            data.samples(other)

    def test_generated_runner_works_as_a_separate_process(self):
        notebook = common.load(ROOT/'bcs_kaggle.ipynb')
        namespace = {'Path': Path, 'WORK_ROOT': self.cfg['WORK_ROOT'], 'SOURCE_FILES': {}}
        Path(self.cfg['WORK_ROOT'], 'src').mkdir(parents=True)
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and source.startswith('SOURCE_FILES['):
                exec(source, namespace)
        config = Path(self.cfg['WORK_ROOT'])/'runtime.json'
        common.save(config, self.cfg)
        runner = Path(self.cfg['WORK_ROOT'])/'src/run.py'
        for action in ('audit', 'partition', 'export-yolo', 'samples'):
            result = subprocess.run([sys.executable, str(runner), action, '--config', str(config)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(common.load(common.artifacts(self.cfg)/'manifests/regression_samples.json')), 4)


class Scalar:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class FakeOptimizer:
    def __init__(self, lr, iterations=0):
        self.learning_rate = lr
        self.iterations = Scalar(iterations)


class FakeModel:
    """Simula fit/serialización para inyectar una interrupción y una validación peor."""
    output_shape = (None, 1)

    def __init__(self, iterations=0, lr=.001):
        self.optimizer = FakeOptimizer(lr, iterations)
        self.layers = [types.SimpleNamespace(name='head', trainable=True)]
        self.base = types.SimpleNamespace(layers=[types.SimpleNamespace(name='conv', trainable=False)])

    def compile(self, optimizer, **kwargs):
        self.optimizer = optimizer

    def get_layer(self, name):
        return self.base

    def save(self, path):
        common.save(path, {'iterations': self.optimizer.iterations.numpy(), 'lr': self.optimizer.learning_rate})

    def fit(self, train, validation_data, initial_epoch, epochs, callbacks, verbose):
        self.stop_training = False
        for callback in callbacks:
            callback.model = self
        for epoch in range(initial_epoch, epochs):
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(epoch)
            self.optimizer.iterations.value += 1
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, {'loss': .5, 'mae': .5, 'val_loss': [.4, .8, .6][epoch],
                                                  'val_mae': [.4, .8, .6][epoch]})
            if self.stop_training:
                break


class Callback:
    pass


def fake_tf():
    def load_model(path):
        state = common.load(path)
        return FakeModel(state['iterations'], state['lr'])
    return types.SimpleNamespace(__version__='test',
        config=types.SimpleNamespace(experimental=types.SimpleNamespace(
            get_memory_info=lambda device: {'current': 100, 'peak': 200})),
        keras=types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=load_model),
        backend=types.SimpleNamespace(get_value=lambda v: v, clear_session=lambda: None),
        callbacks=types.SimpleNamespace(Callback=Callback, TerminateOnNaN=Callback),
        optimizers=types.SimpleNamespace(Adam=FakeOptimizer),
        losses=types.SimpleNamespace(MeanAbsoluteError=lambda: 'mae'),
        metrics=types.SimpleNamespace(MeanAbsoluteError=lambda **kw: 'mae')))


class CheckpointCase(DatasetFixture):
    def test_resume_restores_optimizer_and_preserves_previous_best(self):
        self.prepare()
        tf = fake_tf()
        patches = [patch.object(regressor, 'dataset', return_value='dataset'),
                   patch.object(regressor, 'build', return_value=FakeModel()),
                   patch.object(regressor.importlib.metadata, 'version', return_value='3.0')]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            first = regressor.phase_train(tf, self.cfg, 'head', smoke=True, interrupt_after=1)
            self.assertEqual(first['status'], 'paused')
            runpath = regressor.root_run(self.cfg, True)/'head'
            best_hash = common.sha(runpath/first['best_checkpoint'])
            # Simular un alias dañado por caída entre escritura y commit.
            (runpath/'last.keras').write_text('incomplete')
            final = regressor.phase_train(tf, self.cfg, 'head', smoke=True)
            self.assertEqual(final['completed_epochs'], 2)
            self.assertEqual(final['optimizer_iterations'], 2)
            self.assertEqual(final['best_val_mae'], .4)
            self.assertEqual(final['wait'], 1)
            self.assertEqual(common.sha(runpath/'best.keras'), best_hash)
            self.assertEqual(len(final['history']), 2)
            self.assertEqual(final['status'], 'complete')
            other = copy.deepcopy(self.cfg)
            other['REGRESSOR']['head_lr'] = .01
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                regressor.phase_train(tf, other, 'head', smoke=True)

    def test_yolo_recovers_raw_optimizer_after_framework_strips_last(self):
        self.prepare()
        fake_torch = types.SimpleNamespace(__version__='test',
            cuda=types.SimpleNamespace(max_memory_allocated=lambda: 123),
            load=lambda path, **kw: common.load(path), save=lambda value, path: common.save(path, value))
        package = types.ModuleType('ultralytics')
        package.YOLO = FakeYOLO
        trainer_module = types.ModuleType('ultralytics.engine.trainer')
        trainer_module.BaseTrainer = FakeTrainer
        with patch.object(detector, 'runtime', return_value=fake_torch), \
                patch.object(detector.importlib.metadata, 'version', return_value='8.4.0'), \
                patch.dict(sys.modules, {'ultralytics': package, 'ultralytics.engine.trainer': trainer_module}):
            detector.train(self.cfg, smoke=True, interrupt_after=1)
            runpath = detector.run_dir(self.cfg, True)
            state = common.load(runpath/'state.json')
            self.assertEqual(state['status'], 'paused')
            self.assertIsNone(common.load(runpath/'ultralytics/weights/last.pt')['optimizer'])
            self.assertIsNotNone(common.load(runpath/'resume.pt')['optimizer'])
            (runpath/'resume.pt').write_text('incomplete alias')
            detector.train(self.cfg, smoke=True)
            state = common.load(runpath/'state.json')
            self.assertEqual(state['completed_epochs'], 2)
            self.assertEqual(state['status'], 'complete')
            self.assertTrue(state['resume_verified'])
            self.assertEqual(common.load(runpath/'resume.pt')['optimizer']['iterations'], 2)
            self.assertTrue(common.load(common.artifacts(self.cfg)/'reports/detector_smoke.json')['resume_verified'])


class FakeTrainer:
    def _do_train(self):
        return ('on_model_save', 'on_fit_epoch_end')


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks[event] = callback

    def train(self, **kwargs):
        runpath = Path(kwargs['project'])/kwargs['name']
        folder = runpath/'weights'
        folder.mkdir(parents=True, exist_ok=True)
        start = 0
        if kwargs.get('resume'):
            ckpt = common.load(self.weights)
            assert ckpt['optimizer'] is not None
            start = ckpt['epoch']+1
        trainer = types.SimpleNamespace(last=folder/'last.pt', best=folder/'best.pt',
                                        best_fitness=.8, stop=False)
        for epoch in range(start, kwargs['epochs']):
            trainer.epoch = epoch
            trainer.stop = epoch+1 >= kwargs['epochs']
            self.callbacks['on_train_epoch_start'](trainer)
            common.save(trainer.last, {'epoch': epoch, 'optimizer': {'iterations': epoch+1},
                                       'train_args': kwargs})
            common.copy_atomic(trainer.last, trainer.best)
            self.callbacks['on_model_save'](trainer)
            if trainer.stop:
                break
        common.save(trainer.last, {'epoch': -1, 'optimizer': None})
        common.save(trainer.best, {'epoch': -1, 'optimizer': None})


class NotebookCase(unittest.TestCase):
    def test_notebook_is_standalone_and_matches_sources(self):
        notebook = json.loads((ROOT/'bcs_kaggle.ipynb').read_text())
        self.assertEqual(notebook['nbformat'], 4)
        namespace = {'Path': Path, 'SOURCE_FILES': {}}
        with tempfile.TemporaryDirectory() as temp:
            namespace['WORK_ROOT'] = temp
            Path(temp, 'src').mkdir()
            for cell in notebook['cells']:
                source = ''.join(cell['source'])
                if cell['cell_type'] == 'code':
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                    compile(source, '<notebook-cell>', 'exec')
                    if source.startswith('SOURCE_FILES['):
                        exec(source, namespace)
            for name, source in namespace['SOURCE_FILES'].items():
                self.assertEqual(source, (ROOT/'kaggle_src'/name).read_text())
                compile(source, name, 'exec')
            self.assertEqual(len(namespace['SOURCE_FILES']), len(list((ROOT/'kaggle_src').glob('*.py'))))
        headings = [c for c in notebook['cells'] if c['cell_type'] == 'markdown'
                    and ''.join(c['source']).startswith('## ')]
        self.assertEqual(len(headings), 12)

    def test_metrics_count_abstentions_and_do_not_round(self):
        records = [{'true_bcs': 2.5, 'pred_bcs': 2.6}, {'true_bcs': 3., 'pred_bcs': None}]
        m = evaluation.metrics(records)
        self.assertAlmostEqual(m['mae_predicted'], .1)
        self.assertEqual(m['coverage'], .5)
        self.assertEqual(m['within_0.25_all'], .5)
        self.assertEqual(m['within_0.25_predicted'], 1.)
        self.assertIsNone(evaluation.metrics([{'true_bcs': 3., 'pred_bcs': None}])['mae_predicted'])

    def test_budget_and_ambiguous_association(self):
        with patch.object(common.time, 'time', return_value=100):
            self.assertFalse(common.can_start({'DEADLINE_UNIX': 110}, 10))
            self.assertTrue(common.can_start({'DEADLINE_UNIX': None}, 100000))
        cow = {'class': 0, 'xyxy': [0, 0, 100, 100]}
        rump = {'class': 1, 'xyxy': [10, 10, 50, 50]}
        self.assertEqual(common.pair_boxes([cow, rump])[0], 'ok')
        self.assertEqual(common.pair_boxes([cow, cow, rump])[0], 'asociacion_ambigua')


if __name__ == '__main__':
    unittest.main()
