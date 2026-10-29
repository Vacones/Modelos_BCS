"""Regresión continua MAE, dos fases y checkpoints transaccionales por época."""
import importlib.metadata
import json
import time
from pathlib import Path
from common import (artifacts, can_start, contract, copy_atomic, digest, load, require,
                    save, sha, stage, write_csv)


def runtime(cfg):
    import tensorflow as tf
    require(int(tf.__version__.split('.')[0]) >= 2, 'Se requiere TensorFlow 2 con Keras 3.')
    require(int(importlib.metadata.version('keras').split('.')[0]) >= 3, 'Se requiere Keras 3.')
    gpu = tf.config.list_physical_devices('GPU')
    require(gpu, 'TensorFlow no detecta GPU.')
    index = int(cfg['REGRESSOR']['device'])
    require(index < len(gpu), 'GPU del regresor inexistente.')
    # Una GPU: la sobrecarga de MirroredStrategy no se justifica sin benchmark.
    tf.config.set_visible_devices(gpu[index], 'GPU')
    tf.config.experimental.set_memory_growth(gpu[index], True)
    tf.keras.utils.set_random_seed(cfg['SEED'])
    return tf


def dataset(tf, cfg, selected, training=False):
    root = artifacts(cfg)
    size, batch = cfg['REGRESSOR']['size'], cfg['REGRESSOR']['batch']
    require(selected, 'Partición de regresión vacía.')
    paths = [str(root / r['crop']) for r in selected]
    values = [[float(r['bcs'])] for r in selected]
    ds = tf.data.Dataset.from_tensor_slices((paths, values))
    if training:
        ds = ds.shuffle(len(paths), seed=cfg['SEED'], reshuffle_each_iteration=True)

    def decode(path, value):
        im = tf.io.decode_png(tf.io.read_file(path), channels=3)
        im = tf.ensure_shape(im, [size, size, 3])
        return tf.cast(im, tf.float32), tf.cast(value, tf.float32)

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build(tf, cfg, weights=None):
    size = cfg['REGRESSOR']['size']
    inputs = tf.keras.Input((size, size, 3), name='rgb_0_255')
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal', seed=cfg['SEED']),
        tf.keras.layers.RandomRotation(.03, seed=cfg['SEED']+1),
        tf.keras.layers.RandomZoom(.05, seed=cfg['SEED']+2),
    ], name='train_augmentation')
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                               input_shape=(size, size, 3))
    base.trainable = False
    features = base(augmentation(inputs), training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dropout(.2, seed=cfg['SEED']+3)(x)
    output = tf.keras.layers.Dense(1, activation='linear', dtype='float32', name='bcs')(x)
    return tf.keras.Model(inputs, output, name='bcs_regressor')


def root_run(cfg, smoke=False):
    return artifacts(cfg) / ('smoke/regressor' if smoke else 'regressor') / cfg['EXPERIMENT_ID']


def phase_train(tf, cfg, phase, smoke=False, interrupt_after=None):
    root = artifacts(cfg)
    runroot = root_run(cfg, smoke)
    run = runroot / phase
    settings = cfg['REGRESSOR']
    epochs = cfg['SMOKE_EPOCHS'] if smoke else settings[phase+'_epochs']
    fp = digest({**contract(cfg), 'regressor': settings, 'phase': phase, 'epochs': epochs,
                 'smoke': smoke, 'seed': cfg['SEED'], 'tensorflow': tf.__version__,
                 'keras': importlib.metadata.version('keras')})
    state_path = run / 'state.json'
    state = load(state_path) if state_path.exists() else None
    if state:
        require(state['fingerprint'] == fp, 'Regresor incompatible: crea otro EXPERIMENT_ID.')
        require(sha(run / state['checkpoint']) == state['checkpoint_sha256'], 'Checkpoint Keras incompleto.')
        require(sha(run / state['best_checkpoint']) == state['best_checkpoint_sha256'], 'Mejor checkpoint Keras incompleto.')
        # Reconstruir aliases incluso si el proceso murió entre copias y commit del estado.
        copy_atomic(run / state['checkpoint'], run / 'last.keras')
        copy_atomic(run / state['best_checkpoint'], run / 'best.keras')
        write_csv(run / 'history.csv', state['history'])
        copy_atomic(run / 'history.csv', root / 'histories' / ('smoke_' if smoke else '') / ('regressor_'+phase+'.csv'))
        if state['status'] == 'complete':
            print('Fase ya completada:', phase)
            return state
    if not smoke:
        estimate_path = root / 'reports/regressor_smoke.json'
        require(estimate_path.is_file(), 'Ejecuta primero la prueba corta del regresor.')
        estimate = load(estimate_path)
        require(estimate['contract'] == contract(cfg) and estimate['settings'] == settings
                and estimate['seed'] == cfg['SEED'],
                'Prueba corta de regresión incompatible.')
        if not can_start(cfg, estimate['phases'][phase]['seconds_per_epoch']):
            print('Presupuesto insuficiente para otra época de', phase)
            return None
    require(can_start(cfg, 0), 'Se agotó el presupuesto útil de sesión.')
    run.mkdir(parents=True, exist_ok=True)
    selected = load(root / 'manifests/regression_samples.json')
    train = dataset(tf, cfg, [r for r in selected if r['split'] == 'train'], True)
    val = dataset(tf, cfg, [r for r in selected if r['split'] == 'val'])
    if state:
        model = tf.keras.models.load_model(run / state['checkpoint'])
        require(model.optimizer is not None, 'Modelo sin optimizador: no es una reanudación.')
        require(int(model.optimizer.iterations.numpy()) == state['optimizer_iterations'], 'Optimizador no restaurado.')
        require(model.output_shape[-1] == 1, 'Se esperaba regresión escalar.')
    else:
        if phase == 'head':
            model = build(tf, cfg, None if smoke else settings['weights'])
        else:
            head = runroot / 'head'
            headstate = load(head / 'state.json')
            require(headstate['status'] == 'complete', 'Finaliza head antes del ajuste fino.')
            require(headstate['contract'] == contract(cfg) and headstate['settings'] == settings
                    and headstate['seed'] == cfg['SEED'],
                    'La cabeza congelada pertenece a otra configuración o datos.')
            require(sha(head / headstate['best_checkpoint']) == headstate['best_checkpoint_sha256'],
                    'Checkpoint de la cabeza alterado.')
            model = tf.keras.models.load_model(head / headstate['best_checkpoint'])
            base = model.get_layer('efficientnetb0')
            base.trainable = True
            n = settings['fine_layers']
            require(0 < n <= len(base.layers), 'Número de capas de ajuste fino inválido.')
            for i, layer in enumerate(base.layers):
                layer.trainable = i >= len(base.layers)-n and not isinstance(layer, tf.keras.layers.BatchNormalization)
        model.compile(optimizer=tf.keras.optimizers.Adam(settings[phase+'_lr']),
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')], jit_compile=False)
        state = {'fingerprint': fp, 'stage': 'regressor_'+phase, 'phase': phase,
                 'completed_epochs': 0, 'best_val_mae': None, 'wait': 0, 'history': [],
                 'status': 'running', 'contract': contract(cfg),
                 'settings': settings,
                 'seed': cfg['SEED'],
                 'config_path': 'configs/config.json', 'target_epochs': epochs}
    initial_epoch = state['completed_epochs']
    require(initial_epoch < epochs, 'La fase ya alcanzó su objetivo de épocas.')
    started = [None]

    class DurableEpoch(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            started[0] = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            require(all(__import__('math').isfinite(v) for v in logs.values()),
                    'Pérdida no finita: se conserva el checkpoint anterior.')
            number = epoch+1
            elapsed = time.monotonic()-started[0]
            filename = f'epoch_{number:04d}.keras'
            temp = run / f'epoch_{number:04d}.tmp.keras'
            self.model.save(temp)
            temp.replace(run / filename)
            improved = state['best_val_mae'] is None or logs['val_mae'] < state['best_val_mae']
            state['wait'] = 0 if improved else state['wait']+1
            if improved:
                state.update(best_val_mae=logs['val_mae'], best_epoch=number,
                             best_checkpoint=filename, best_checkpoint_sha256=sha(run / filename))
            state['history'].append({'epoch': number, **logs, 'seconds': elapsed})
            state.update(completed_epochs=number, checkpoint=filename, checkpoint_sha256=sha(run / filename),
                         optimizer_iterations=int(self.model.optimizer.iterations.numpy()),
                         learning_rate=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
                         trainable_layers=[l.name for l in self.model.layers if l.trainable],
                         base_trainable_layers=[l.name for l in self.model.get_layer('efficientnetb0').layers if l.trainable],
                         seconds_per_epoch=sum(h['seconds'] for h in state['history'])/len(state['history']))
            # Incluir coste de serialización al estimar la siguiente época.
            elapsed = time.monotonic()-started[0]
            state['history'][-1]['seconds'] = elapsed
            state['seconds_per_epoch'] = sum(h['seconds'] for h in state['history'])/len(state['history'])
            try:
                state['gpu_memory'] = tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError, RuntimeError):
                state['gpu_memory'] = None
            done = number >= epochs or state['wait'] >= settings['patience']
            forced = interrupt_after is not None and number >= interrupt_after
            pause = not done and (forced or not can_start(cfg, max(h['seconds'] for h in state['history'])))
            state['status'] = 'complete' if done else 'paused' if pause else 'running'
            copy_atomic(run / filename, run / 'last.keras')
            copy_atomic(run / state['best_checkpoint'], run / 'best.keras')
            # Estado es el commit: apunta a archivos inmutables, nunca a un alias parcialmente actualizado.
            save(state_path, state)
            write_csv(run / 'history.csv', state['history'])
            copy_atomic(run / 'history.csv', root / 'histories' / ('smoke_' if smoke else '') / ('regressor_'+phase+'.csv'))
            keep = {filename, state['best_checkpoint'], f'epoch_{number-1:04d}.keras'}
            for path in run.glob('epoch_*.keras'):
                if path.name not in keep and '.tmp.' not in path.name:
                    path.unlink()
            if done or pause:
                self.model.stop_training = True

    model.fit(train, validation_data=val, initial_epoch=initial_epoch, epochs=epochs,
              callbacks=[DurableEpoch(), tf.keras.callbacks.TerminateOnNaN()], verbose=2)
    del model
    tf.keras.backend.clear_session()
    return load(state_path)


def train(cfg, smoke=False):
    tf = runtime(cfg)
    if smoke:
        require(cfg['SMOKE_EPOCHS'] >= 2, 'La prueba de reanudación necesita dos épocas.')
        phases = {}
        for phase in ('head', 'fine_tuning'):
            state = phase_train(tf, cfg, phase, True, interrupt_after=1)
            if state['status'] != 'complete':
                require(state['completed_epochs'] == 1, 'La prueba corta no pausó después de una época.')
                iterations = state['optimizer_iterations']
                state = phase_train(tf, cfg, phase, True)
                require(state['optimizer_iterations'] > iterations, 'No avanzó el optimizador al reanudar.')
            require(state['status'] == 'complete', 'Prueba corta pausada por presupuesto; repítela en otra sesión.')
            phases[phase] = {'seconds_per_epoch': state['seconds_per_epoch'],
                             'best_val_mae': state['best_val_mae'], 'gpu_memory': state.get('gpu_memory'),
                             'optimizer_restored': True}
        save(artifacts(cfg) / 'reports/regressor_smoke.json', {
            'contract': contract(cfg), 'settings': cfg['REGRESSOR'], 'phases': phases, 'seed': cfg['SEED'],
            'weights': 'random_for_technical_test',
            'estimated_full_minutes': sum(phases[p]['seconds_per_epoch']*cfg['REGRESSOR'][p+'_epochs']/60 for p in phases)})
        return
    phases = ('head', 'fine_tuning') if cfg['REGRESSOR_PHASE'] == 'all' else (cfg['REGRESSOR_PHASE'],)
    for phase in phases:
        state = phase_train(tf, cfg, phase)
        if state is None or state['status'] != 'complete':
            return
    select(cfg)


def select(cfg):
    root = root_run(cfg)
    candidates = []
    for phase in ('head', 'fine_tuning'):
        p = root / phase / 'state.json'
        if p.exists():
            s = load(p)
            require(s['contract'] == contract(cfg) and s['settings'] == cfg['REGRESSOR']
                    and s['seed'] == cfg['SEED'],
                    'Fase de otros datos/código/configuración.')
            if s['status'] == 'complete':
                candidates.append((s['best_val_mae'], phase, s))
    require(candidates, 'No hay fases completas para seleccionar.')
    metric, phase, state = min(candidates, key=lambda v: v[0])
    source = root / phase / state['best_checkpoint']
    copy_atomic(source, root / 'best.keras')
    lastphase = 'fine_tuning' if any(p == 'fine_tuning' for _, p, _ in candidates) else 'head'
    copy_atomic(root / lastphase / 'last.keras', root / 'last.keras')
    save(root / 'selected.json', {'contract': contract(cfg), 'settings': cfg['REGRESSOR'],
                                 'seed': cfg['SEED'],
                                 'phase': phase, 'best_val_mae': metric, 'sha256': sha(root / 'best.keras')})
    stage(cfg, 'regressor_trained', digest({**contract(cfg), 'settings': cfg['REGRESSOR']}),
          [root / 'selected.json', root / 'best.keras', root / 'last.keras'])


def trained(cfg):
    root = root_run(cfg)
    selected = load(root / 'selected.json')
    require(selected['contract'] == contract(cfg) and selected['settings'] == cfg['REGRESSOR']
            and selected['seed'] == cfg['SEED'],
            'Regresor de otros datos o configuración.')
    require(sha(root / 'best.keras') == selected['sha256'], 'Mejor modelo alterado.')
    # Bloquear una selección antigua mientras una fase posterior está a medio entrenar.
    for phase in ('head', 'fine_tuning'):
        p = root / phase / 'state.json'
        if p.exists():
            require(load(p)['status'] == 'complete', 'Existe una fase de regresión pendiente.')
    return root / 'best.keras'
