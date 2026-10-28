import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import json
import shutil
import sys
import platform
import numpy as np
from .common import require, save_json, load_json, save_csv, sha256
from .data import manifest


def tensorflow_ready(allow_cpu=False, mixed=False):
    import tensorflow as tf
    gpu=tf.config.list_physical_devices('GPU')
    if not gpu and not allow_cpu:
        raise RuntimeError('TensorFlow no detecta GPU. Revisa WSL2/nvidia-smi. Usa --allow-cpu solo para pruebas pequeñas.')
    for device in gpu:tf.config.experimental.set_memory_growth(device,True)
    if mixed:tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.keras.utils.set_random_seed(42)
    return tf


def dataset(tf, selected, classes, size, batch, training=False):
    require(bool(selected),'Partición sin imágenes BCS utilizables')
    paths=[r['crop_path'] for r in selected]; labels=[classes.index(float(r['bcs'])) for r in selected]
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if training:ds=ds.shuffle(len(paths),seed=42,reshuffle_each_iteration=True)
    def decode(path,label):
        image=tf.io.decode_png(tf.io.read_file(path),channels=3)
        image=tf.ensure_shape(image,[size,size,3])
        return tf.cast(image,tf.float32),label  # EfficientNet incluye Rescaling: valores 0..255.
    return ds.map(decode,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(tf,classes,size,weights='imagenet'):
    inputs=tf.keras.Input((size,size,3),name='crop_rgb_0_255')
    augment=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal',seed=42),
        tf.keras.layers.RandomRotation(.03,fill_mode='reflect',seed=43),
        tf.keras.layers.RandomZoom(.05,fill_mode='reflect',seed=44)],name='aumento_solo_train')
    base=tf.keras.applications.EfficientNetB0(include_top=False,weights=weights,input_shape=(size,size,3))
    base.trainable=False
    features=base(augment(inputs),training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(features)
    x=tf.keras.layers.Dropout(.2,seed=45)(x)
    output=tf.keras.layers.Dense(len(classes),activation='softmax',dtype='float32',name='bcs')(x)
    return tf.keras.Model(inputs,output,name='bcs_efficientnet_b0')


def history_plots(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    done=sorted((run/'checkpoints').glob('epoch_*.done.json'))
    data=[load_json(p) for p in done]
    if not data:return
    save_csv(run/'historial.csv',[{'epoch':d['epoch'],**d['logs']} for d in data])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,['loss','accuracy']):
        for prefix in ('','val_'):
            ax.plot([d['epoch'] for d in data],[d['logs'].get(prefix+key,np.nan) for d in data],label=prefix+key)
        ax.set_xlabel('Época');ax.set_title(key);ax.legend()
    fig.tight_layout();fig.savefig(run/'curvas_entrenamiento.png',dpi=160);plt.close(fig)


def train(args):
    tf=tensorflow_ready(args.allow_cpu,args.mixed)
    work,cfg,all_rows=manifest(args.work)
    run=Path(args.run).resolve()
    require(args.batch>0 and args.epochs>0 and args.lr>0 and args.patience>0,'Hiperparámetros inválidos')
    train_rows=[r for r in all_rows if r['split']=='train' and r['status']=='ok']
    val_rows=[r for r in all_rows if r['split']=='val' and r['status']=='ok']
    settings={'classes':cfg['classes'],'size':cfg['size'],'manifest_sha256':cfg['manifest_sha256'],
              'phase':args.phase,'lr':args.lr,'batch':args.batch,'mixed':args.mixed,
              'fine_layers':args.fine_layers,'weights':args.weights,
              'init':str(Path(args.init).resolve()) if args.init else None,
              'init_sha256':sha256(args.init) if args.init else None,'seed':42}
    initial_epoch=0
    if args.resume:
        require((run/'settings.json').is_file(),'No existe la ejecución para reanudar')
        require(load_json(run/'settings.json')==settings,'Cambió el dataset o la configuración; usa los mismos argumentos para reanudar')
        candidates=[p for p in sorted((run/'checkpoints').glob('epoch_*.done.json')) if p.with_name(p.name.replace('.done.json','.keras')).is_file()]
        require(bool(candidates),'No hay checkpoint completo; usa otra carpeta de ejecución')
        state=load_json(candidates[-1]); initial_epoch=state['epoch']
        model=tf.keras.models.load_model(candidates[-1].with_name(candidates[-1].name.replace('.done.json','.keras')))
    else:
        require(not run.exists(),'La ejecución ya existe. Usa --resume o una carpeta nueva.')
        run.mkdir(parents=True);(run/'checkpoints').mkdir()
        save_json(run/'settings.json',settings)
        if args.phase=='finetune':
            require(args.init is not None,'Finetune requiere --init ruta/al/best.keras de la fase head')
            source_settings=load_json(Path(args.init).parent/'settings.json')
            require(source_settings['classes']==cfg['classes'] and source_settings['size']==cfg['size'] and source_settings['manifest_sha256']==cfg['manifest_sha256'],'Modelo inicial incompatible')
            model=tf.keras.models.load_model(args.init)
            base=model.get_layer('efficientnetb0');base.trainable=True
            require(0<args.fine_layers<=len(base.layers),'--fine-layers inválido')
            for layer in base.layers[:-args.fine_layers]:layer.trainable=False
            for layer in base.layers:
                if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
        else:
            model=build_model(tf,cfg['classes'],cfg['size'],None if args.weights=='none' else 'imagenet')
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'],jit_compile=False)
    require(initial_epoch<args.epochs,'Ya completaste esas épocas. Aumenta --epochs si deseas continuar.')
    save_json(run/'environment.json',{'python':sys.version,'platform':platform.platform(),'tensorflow':tf.__version__,
        'gpu':[d.name for d in tf.config.list_physical_devices('GPU')], 'train_images':len(train_rows),'val_images':len(val_rows),
        'trainable_params':int(sum(np.prod(v.shape) for v in model.trainable_weights))})
    with (run/'modelo.txt').open('w') as f:model.summary(print_fn=lambda line,**kwargs:f.write(line+'\n'))
    counts=np.bincount([cfg['classes'].index(float(r['bcs'])) for r in train_rows],minlength=len(cfg['classes']))
    require(bool(np.all(counts>0)),'Clase sin imágenes de entrenamiento')
    weights={i:float(len(train_rows)/(len(counts)*count)) for i,count in enumerate(counts)}
    save_json(run/'class_weights.json',weights)
    class DurableCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            number=epoch+1;logs={k:float(v) for k,v in (logs or {}).items()}
            require(all(np.isfinite(v) for v in logs.values()),'Métrica no finita; no se guarda como checkpoint válido')
            prefix=run/'checkpoints'/f'epoch_{number:04d}'
            temporary=prefix.with_suffix('.tmp.keras');final=prefix.with_suffix('.keras')
            self.model.save(temporary);temporary.replace(final)
            previous=[load_json(p) for p in (run/'checkpoints').glob('epoch_*.done.json')]
            best=min([p['logs']['val_loss'] for p in previous],default=float('inf'))
            if logs['val_loss']<best or not (run/'best.keras').exists():
                shutil.copyfile(final,run/'best.tmp.keras');(run/'best.tmp.keras').replace(run/'best.keras')
                save_json(run/'best_epoch.json',{'epoch':number,'val_loss':logs['val_loss']})
            save_json(prefix.with_suffix('.done.json'),{'epoch':number,'logs':logs})
            # Dos últimos estados completos + best.keras. Se conservan todos los registros de métricas.
            checkpoints=sorted(p for p in (run/'checkpoints').glob('epoch_*.keras') if not p.name.endswith('.tmp.keras'))
            for old in checkpoints[:-2]:old.unlink()
            history_plots(run)
    callbacks=[DurableCheckpoint(),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.patience),
               tf.keras.callbacks.TerminateOnNaN(),tf.keras.callbacks.TensorBoard(log_dir=str(run/'tensorboard'))]
    model.fit(dataset(tf,train_rows,cfg['classes'],cfg['size'],args.batch,True),
        validation_data=dataset(tf,val_rows,cfg['classes'],cfg['size'],args.batch),
        initial_epoch=initial_epoch,epochs=args.epochs,class_weight=weights,callbacks=callbacks)
    history_plots(run)
    print('Mejor modelo por val_loss:',run/'best.keras')
    print('La prueba no se consultó durante el entrenamiento.')
