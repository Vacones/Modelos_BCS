"""YOLO26: checkpoints recuperables antes de strip_optimizer y pausa por época."""
import importlib.metadata
import json
import os
import time
from pathlib import Path
from PIL import Image
from common import (artifacts, can_start, contract, copy_atomic, crop, digest, image_path, load,
                    pair_boxes, require, rows, save, sha, stage)


def runtime(cfg):
    import torch
    import ultralytics
    require(ultralytics.__version__ == cfg['ULTRALYTICS_VERSION'], 'Versión Ultralytics incompatible.')
    require(torch.cuda.is_available(), 'Se requiere GPU CUDA para esta etapa.')
    require(str(cfg['DETECTOR']['device']).isdigit(),
            'Este notebook valida una GPU. Multi-GPU requiere verificar DDP, reanudación y mejora de tiempo.')
    return torch


def run_dir(cfg, smoke=False):
    return artifacts(cfg) / ('smoke/detector' if smoke else 'detector') / cfg['EXPERIMENT_ID']


def publish(cfg, run, state, smoke):
    root = artifacts(cfg)
    history = run / 'ultralytics/results.csv'
    if history.is_file():
        copy_atomic(history, root / 'histories' / ('detector_smoke.csv' if smoke else 'detector.csv'))
    if smoke:
        save(root / 'reports/detector_smoke.json', {
            'contract': contract(cfg), 'settings': cfg['DETECTOR'], 'seed': cfg['SEED'],
            'seconds_per_epoch': state['seconds_per_epoch'],
            'peak_gpu_allocated_bytes': state['peak_gpu_allocated_bytes'],
            'completed_epochs': state['completed_epochs'], 'resume_verified': state.get('resume_verified', False),
            'estimated_full_minutes': state['seconds_per_epoch']*cfg['DETECTOR']['epochs']/60})
    elif state['status'] == 'complete':
        stage(cfg, 'detector_trained', state['fingerprint'],
              [run / 'best.pt', run / 'last.pt', run / 'resume.pt', run / 'state.json'])


def train(cfg, smoke=False, interrupt_after=None):
    torch = runtime(cfg)
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    import inspect
    root = artifacts(cfg)
    settings = dict(cfg['DETECTOR'])
    settings['epochs'] = cfg['SMOKE_EPOCHS'] if smoke else settings['epochs']
    settings['save_period'] = 1 if smoke else settings['save_period']
    run = run_dir(cfg, smoke)
    signature = {**contract(cfg), 'settings': settings, 'ultralytics': importlib.metadata.version('ultralytics'),
                 'torch': torch.__version__, 'smoke': smoke, 'seed': cfg['SEED']}
    fp = digest(signature)
    status_file = run / 'state.json'
    previous = load(status_file) if status_file.exists() else None
    if previous:
        require(previous['fingerprint'] == fp, 'Detector incompatible: usa otro EXPERIMENT_ID para cambiar épocas o parámetros.')
        require(sha(run / previous['resume_checkpoint']) == previous['resume_sha256'], 'Checkpoint recuperable incompleto.')
        require(sha(run / previous['best_checkpoint']) == previous['best_sha256'], 'Mejor checkpoint incompleto.')
        copy_atomic(run / previous['resume_checkpoint'], run / 'resume.pt')
        copy_atomic(run / previous['resume_checkpoint'], run / 'last.pt')
        copy_atomic(run / previous['best_checkpoint'], run / 'best.pt')
        if previous['status'] == 'complete':
            require(sha(run / 'best.pt') == previous['best_sha256'], 'best.pt alterado.')
            publish(cfg, run, previous, smoke)
            print('Detector ya completado:', run)
            return
    # Verificar orden de callbacks sobre la versión fijada, además de guardar el código en environment.
    source = inspect.getsource(BaseTrainer._do_train)
    require('on_model_save' in source and 'on_fit_epoch_end' in source, 'Callbacks YOLO incompatibles.')
    estimate_file = root / 'reports/detector_smoke.json'
    if not smoke:
        require(estimate_file.is_file(), 'Ejecuta primero la prueba corta del detector.')
        estimate = load(estimate_file)
        require(estimate['contract'] == contract(cfg) and estimate['settings'] == cfg['DETECTOR']
                and estimate['seed'] == cfg['SEED'],
                'La prueba corta no corresponde a estos datos/configuración.')
        if not can_start(cfg, estimate['seconds_per_epoch']):
            print('Presupuesto insuficiente para iniciar otra época del detector.')
            return
    require(can_start(cfg, 0), 'Se agotó el presupuesto útil de sesión.')
    run.mkdir(parents=True, exist_ok=True)
    save(run / 'signature.json', signature)
    weights = settings['weights']
    if previous:
        checkpoint = run / 'resume.pt'
        require(checkpoint.is_file() and sha(checkpoint) == previous['resume_sha256'], 'Checkpoint recuperable incompleto.')
        # Solo checkpoints del propio proyecto; torch.load usa pickle.
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(ckpt.get('optimizer') is not None and ckpt.get('epoch', -1) >= 0,
                'Pesos sin estado de entrenamiento: inicia un experimento nuevo desde pesos.')
        require(ckpt['epoch'] + 1 == previous['completed_epochs'], 'Época de checkpoint y estado no coincide.')
        require(ckpt['epoch'] + 1 < settings['epochs'], 'Entrenamiento terminado: usa un experimento nuevo.')
        # Reubicar exclusivamente rutas, sin modificar épocas ni optimizador.
        ckpt['train_args'].update(data=str(root / 'configs/detector.yaml'),
                                  project=str(run), name='ultralytics', save_dir=str(run / 'ultralytics'))
        relocated = run / 'resume_session.pt'
        torch.save(ckpt, relocated)
        del ckpt
        model = YOLO(str(relocated))
    else:
        require(not (run / 'ultralytics').exists(), 'Run incompleto sin estado: usa otro EXPERIMENT_ID.')
        model = YOLO(weights)
    epoch_start = [None]
    durations = list(previous.get('epoch_durations', [])) if previous else []
    paused = [False]
    previous_epochs = previous['completed_epochs'] if previous else 0
    state = dict(previous or {})

    def begin(trainer):
        epoch_start[0] = time.monotonic()

    def checkpoint(trainer):
        if int(os.environ.get('RANK', '-1')) not in (-1, 0):
            return
        elapsed = time.monotonic() - epoch_start[0]
        durations.append(elapsed)
        resume_name = f'resume_epoch_{trainer.epoch+1:04d}.pt'
        best_name = f'best_epoch_{trainer.epoch+1:04d}.pt'
        copy_atomic(trainer.last, run / resume_name)
        copy_atomic(trainer.last, run / 'resume.pt')
        copy_atomic(trainer.last, run / 'last.pt')
        if trainer.best.is_file():
            copy_atomic(trainer.best, run / best_name)
            copy_atomic(trainer.best, run / 'best.pt')
        state.update(fingerprint=fp, stage='detector', status='complete' if trainer.stop else 'running',
                     completed_epochs=trainer.epoch+1, target_epochs=settings['epochs'],
                     resume_checkpoint=resume_name, best_checkpoint=best_name,
                     resume_sha256=sha(run / resume_name), best_sha256=sha(run / best_name),
                     seconds_per_epoch=sum(durations)/len(durations),
                     epoch_durations=durations,
                     resume_verified=previous is not None,
                     best_fitness=float(trainer.best_fitness),
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                     callback_resume_note='Ultralytics restaura estado del checkpoint; paciencia y RNG no se garantizan idénticos.')
        save(status_file, state)
        for pattern in ('resume_epoch_*.pt', 'best_epoch_*.pt'):
            for old in sorted(run.glob(pattern))[:-2]:
                old.unlink()
        forced = interrupt_after is not None and trainer.epoch+1 >= interrupt_after
        if trainer.epoch+1 < settings['epochs'] and (forced or not can_start(cfg, max(durations))):
            trainer.stop = True
            paused[0] = True

    model.add_callback('on_train_epoch_start', begin)
    model.add_callback('on_model_save', checkpoint)
    options = dict(data=str(root / 'configs/detector.yaml'), epochs=settings['epochs'],
                   imgsz=settings['imgsz'], batch=settings['batch'], device=settings['device'],
                   workers=2, seed=cfg['SEED'], deterministic=True, save=True,
                   save_period=settings['save_period'], patience=settings['patience'],
                   project=str(run), name='ultralytics', exist_ok=True, cache=False, plots=True,
                   amp=True, val=True)
    if previous:
        options['resume'] = True
    model.train(**options)
    require(state.get('completed_epochs', 0) > previous_epochs, 'No se guardó una época completa.')
    state['status'] = 'paused' if paused[0] else 'complete'
    state['best_sha256'] = sha(run / 'best.pt')
    save(status_file, state)
    publish(cfg, run, state, smoke)
    print(json.dumps(state, indent=2))


def trained(cfg):
    run = run_dir(cfg)
    state = load(run / 'state.json')
    require(state['status'] == 'complete', 'Detector pendiente; termina el entrenamiento antes de evaluar.')
    signature = load(run / 'signature.json')
    require(all(signature[k] == v for k, v in contract(cfg).items()), 'Detector de otros datos/código.')
    require(signature['settings'] == cfg['DETECTOR'], 'Configuración detector distinta del entrenamiento.')
    require(signature['seed'] == cfg['SEED'], 'Semilla detector distinta del entrenamiento.')
    require(sha(run / 'best.pt') == state['best_sha256'], 'Checkpoint detector alterado.')
    return run / 'best.pt'


def detect(cfg, split=None, external=None):
    runtime(cfg)
    from ultralytics import YOLO
    root = artifacts(cfg)
    weights = trained(cfg)
    model = YOLO(str(weights))
    require(model.names == {0: 'cow', 1: 'ass'}, 'Clases del detector incorrectas.')
    sources = rows(cfg, split, supervised=True) if external is None else [
        {'key': Path(external).name, 'image': external, 'bcs': None, 'cow_id': None}]
    tag = split if external is None else 'external'
    dest = root / 'predicted' / tag
    settings = {'detector_hash': sha(weights), **contract(cfg), 'inference': cfg['INFERENCE'],
                'size': cfg['REGRESSOR']['size']}
    predictions = []
    for row in sources:
        source = image_path(cfg, row) if external is None else Path(external)
        result = model.predict(source=str(source), imgsz=cfg['DETECTOR']['imgsz'],
                               conf=cfg['INFERENCE']['confidence'], device=cfg['DETECTOR']['device'], verbose=False)[0]
        boxes = [{'class': int(c), 'xyxy': [float(v) for v in xy], 'confidence': float(score)}
                 for xy, c, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                                         result.boxes.conf.cpu().tolist())]
        status, pair = pair_boxes(boxes, cfg['INFERENCE']['association_coverage'])
        dest.mkdir(parents=True, exist_ok=True)
        crop_path = None
        if status == 'ok':
            p = dest / (digest(row['key']) + '.png')
            with Image.open(source) as im:
                crop(im, pair[1]['xyxy'], cfg['REGRESSOR']['size']).save(p)
            crop_path = str(p.relative_to(root))
        # Vista de las primeras nueve imágenes y toda inferencia externa.
        if len(predictions) < 9:
            result.save(filename=str(dest / (digest(row['key']) + '_boxes.jpg')))
        predictions.append({'key': row['key'], 'cow_id': row['cow_id'], 'true_bcs': row['bcs'],
                            'status': status, 'boxes': boxes, 'crop': crop_path})
    save(dest / 'predictions.json', {'settings': settings, 'rows': predictions})


def evaluate(cfg, split):
    runtime(cfg)
    from ultralytics import YOLO
    weights = trained(cfg)
    result = YOLO(str(weights)).val(data=str(artifacts(cfg) / 'configs/detector.yaml'), split=split,
                                   imgsz=cfg['DETECTOR']['imgsz'], device=cfg['DETECTOR']['device'],
                                   project=str(artifacts(cfg) / 'reports'), name='detector_'+split,
                                   exist_ok=True, plots=True)
    save(artifacts(cfg) / 'reports' / ('detector_'+split+'.json'),
         {k: float(v) for k, v in result.results_dict.items()})
