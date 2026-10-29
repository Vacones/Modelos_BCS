"""Entrypoint subprocess: python src/run.py ACCION --config config.json."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from common import artifacts, contract, digest, load, require, restore, save, sha


def validate_inputs(cfg):
    root = artifacts(cfg)
    manifest = root / 'manifests/manifest.json'
    require(manifest.exists(), 'No hay manifiesto. Activa RUN_AUDIT.')
    m = load(manifest)
    state = load(root / 'state/audit.json')
    require(state['outputs']['manifests/manifest.json'] == sha(manifest), 'Manifiesto alterado.')
    for split, h in m['coco_hashes'].items():
        require(sha(Path(cfg['DATA_ROOT']) / split / '_annotations.coco.json') == h,
                'Cambió COCO: repite la auditoría en otro WORK_ROOT.')
    for row in m['rows']:
        require(sha(Path(cfg['DATA_ROOT']) / row['image']) == row['sha256'],
                'Cambió una imagen: repite auditoría en otro WORK_ROOT.')
    if (root / 'manifests/splits.json').exists():
        s = load(root / 'manifests/splits.json')
        require(s['split_hash'] == digest(s['assignments']) and
                s['assignments'] == {r['key']: r['split'] for r in m['rows']}, 'Particiones alteradas.')


def verify_prepared(cfg, action):
    names = ['partitions', 'yolo_export'] if action.startswith('detector') else ['partitions', 'regression_samples']
    if action in ('evaluate', 'lock-test', 'infer-regressor'):
        names = ['partitions', 'yolo_export', 'regression_samples']
    for name in names:
        p = artifacts(cfg) / 'state' / (name + '.json')
        require(p.is_file(), f'Falta etapa {name}; activa RUN_PREPARE_DATA.')
        state = load(p)
        require(all((artifacts(cfg) / f).is_file() and sha(artifacts(cfg) / f) == h
                    for f, h in state['outputs'].items()), f'Artefactos alterados en {name}.')
    if 'yolo_export' in names:
        from data import export_yolo
        export_yolo(cfg)  # Reconstruir YAML con rutas de working de esta sesión.


def lock_test(cfg):
    from detector import trained as detector_trained
    from regressor import trained as regressor_trained
    root = artifacts(cfg)
    require(cfg['FINAL_TEST_CONFIRMED'],
            'Cierra decisiones usando validación y configura FINAL_TEST_CONFIRMED=True para abrir test.')
    selection = {'contract': contract(cfg), 'detector': sha(detector_trained(cfg)),
                 'regressor': sha(regressor_trained(cfg)), 'inference': cfg['INFERENCE']}
    lock = root / 'state/test_selection.json'
    if lock.exists():
        require(load(lock) == selection, 'La selección cambió después de abrir test: no reutilices test para ajustar.')
    else:
        save(lock, selection)


def export(cfg):
    root = artifacts(cfg)
    root.mkdir(parents=True, exist_ok=True)
    inventory = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                 if p.is_file() and p.name != 'inventory.json' and '.tmp' not in p.name}
    save(root / 'inventory.json', inventory)
    target = Path(cfg['WORK_ROOT']).parent / (Path(cfg['WORK_ROOT']).name + '_artifacts.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in (root, Path(cfg['WORK_ROOT']) / 'src'):
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and '.tmp' not in p.name:
                    archive.write(p, str(p.relative_to(Path(cfg['WORK_ROOT']))))
    print('Export descargable:', target, f'({target.stat().st_size/1024**2:.1f} MiB)')


def status(cfg):
    root = artifacts(cfg)
    names = {'audit': 'Auditoría completada y manifiesto creado', 'partitions': 'Particiones verificadas',
             'yolo_export': 'Exportación YOLO creada', 'regression_samples': 'Muestras de regresión creadas',
             'detector_trained': 'Detector entrenado', 'regressor_trained': 'Regresor entrenado',
             'evaluation_val': 'Evaluación de validación completada', 'evaluation_test': 'Evaluación de test completada'}
    for key, label in names.items():
        p = root / 'state' / (key + '.json')
        if key == 'detector_trained':
            p = root / 'detector' / cfg['EXPERIMENT_ID'] / 'state.json'
        elif key == 'regressor_trained':
            p = root / 'regressor' / cfg['EXPERIMENT_ID'] / 'selected.json'
            if p.exists():
                print('✓ ' + label + ' (selección por validación disponible)')
                continue
        print(('✓ ' if p.exists() and load(p).get('status') == 'complete' else '· ') + label)


def estimate(cfg):
    root = artifacts(cfg)
    stages, unknown = {}, []
    detector = root / 'reports/detector_smoke.json'
    state_path = root / 'detector' / cfg['EXPERIMENT_ID'] / 'state.json'
    state = load(state_path) if state_path.exists() else {}
    if state.get('status') == 'complete':
        stages['detector'] = 0
    elif detector.exists():
        stages['detector'] = load(detector)['seconds_per_epoch'] * max(0, cfg['DETECTOR']['epochs']-state.get('completed_epochs', 0))/60
    else:
        unknown.append('detector: falta prueba corta')
    reg = root / 'reports/regressor_smoke.json'
    for phase in ('head', 'fine_tuning'):
        p = root / 'regressor' / cfg['EXPERIMENT_ID'] / phase / 'state.json'
        s = load(p) if p.exists() else {}
        if s.get('status') == 'complete':
            stages[phase] = 0
        elif reg.exists():
            stages[phase] = load(reg)['phases'][phase]['seconds_per_epoch'] * max(0, cfg['REGRESSOR'][phase+'_epochs']-s.get('completed_epochs', 0))/60
        else:
            unknown.append(phase+': falta prueba corta')
    from common import remaining
    left = remaining(cfg)
    result = {'estimated_training_minutes': stages, 'unknown': unknown,
              'known_total_minutes': sum(stages.values()),
              'remaining_useful_minutes': None if left is None else max(0, left/60),
              'reserved_evaluation_and_export_minutes': cfg.get('SAFETY_MARGIN_MINUTES', 20),
              'note': 'Estimación orientativa; no incluye descarga/inicialización. No se reducen objetivos automáticamente.'}
    save(root / 'reports/time_projection.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action')
    p.add_argument('--config', required=True)
    p.add_argument('--framework', choices=['system', 'torch', 'tensorflow'])
    args = p.parse_args()
    cfg = load(args.config)
    source_files = {p.name: p.read_text(encoding='utf-8') for p in Path(__file__).parent.glob('*.py')}
    require(hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest() == cfg['CODE_HASH'],
            'Cambió el código generado después de configurar el notebook. Reejecuta la configuración.')
    require(cfg['SMOKE_EPOCHS'] >= 2, 'SMOKE_EPOCHS debe ser al menos 2.')
    require(cfg['REGRESSOR_PHASE'] in ('all', 'head', 'fine_tuning'), 'Fase de regresor inválida.')
    require(cfg['EVALUATION_SPLIT'] in ('val', 'test'), 'Partición de evaluación inválida.')
    require(str(cfg['DETECTOR']['device']).isdigit() and str(cfg['REGRESSOR']['device']).isdigit(),
            'Selecciona una GPU por etapa; DDP/MirroredStrategy requieren benchmark independiente.')
    root = artifacts(cfg)
    for folder in ('configs', 'manifests', 'state', 'reports', 'histories'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    action = args.action
    if action not in ('restore', 'environment', 'inspect', 'audit', 'status', 'export', 'estimate'):
        validate_inputs(cfg)
    if action.startswith(('detector-', 'regressor-')) or action in ('evaluate', 'lock-test', 'infer-regressor'):
        verify_prepared(cfg, action)
    if action == 'restore':
        restore(cfg)
    elif action == 'environment':
        from environment import probe
        probe(cfg, args.framework)
    elif action in ('inspect', 'audit', 'partition', 'export-yolo', 'samples'):
        import data
        functions = {'inspect': data.inspect, 'audit': data.audit, 'partition': data.partition,
                     'export-yolo': data.export_yolo, 'samples': data.samples}
        functions[action](cfg)
    elif action in ('detector-smoke', 'detector-train'):
        from detector import train, run_dir
        smoke = action.endswith('smoke')
        if smoke:
            train(cfg, True, interrupt_after=1)
            s = load(run_dir(cfg, True) / 'state.json')
            if s['status'] != 'complete':
                train(cfg, True)
            s = load(run_dir(cfg, True) / 'state.json')
            require(s['status'] == 'complete' and s['completed_epochs'] >= 2,
                    'Prueba detector incompleta por presupuesto.')
        else:
            train(cfg)
    elif action in ('regressor-smoke', 'regressor-train'):
        from regressor import train
        train(cfg, action.endswith('smoke'))
    elif action == 'lock-test':
        lock_test(cfg)
    elif action in ('detector-evaluate', 'detector-crops'):
        import detector
        if cfg['EVALUATION_SPLIT'] == 'test':
            lock_test(cfg)
        if action == 'detector-evaluate':
            detector.evaluate(cfg, cfg['EVALUATION_SPLIT'])
        else:
            detector.detect(cfg, cfg['EVALUATION_SPLIT'])
    elif action == 'evaluate':
        if cfg['EVALUATION_SPLIT'] == 'test':
            lock_test(cfg)
        from evaluation import predict
        predict(cfg, cfg['EVALUATION_SPLIT'])
    elif action == 'infer-detector':
        from detector import detect
        require(cfg.get('INFERENCE_IMAGE'), 'Configura INFERENCE_IMAGE con una foto nueva.')
        detect(cfg, external=cfg['INFERENCE_IMAGE'])
    elif action == 'infer-regressor':
        from evaluation import predict
        predict(cfg, external=True)
    elif action == 'export':
        export(cfg)
    elif action == 'status':
        status(cfg)
    elif action == 'estimate':
        estimate(cfg)
    else:
        raise ValueError('Acción desconocida: ' + action)


if __name__ == '__main__':
    main()
