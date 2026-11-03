"""Entrenamiento real de la cabeza LDL en TensorFlow/Metal; checkpoints por generación."""
from pathlib import Path
from collections import Counter
import numpy as np
from .common import require,save_json,load_json,save_csv,sha256
from .features import cache,DIM,DINO_NAME,DINO_REPO


def load_model(path,device='cpu'):
    # La inferencia de esta cabeza pequeña usa CPU; evita dos gestores Metal simultáneos en explain.
    from .tf_model import tf,configure_tf
    configure_tf('cpu')
    root=Path(path).resolve();root=root.parent if root.is_file() else root
    saved=load_json(root/'estado.json');cfg=saved['config']
    require(cfg['output']=='tf_ldl_expected_then_nearest' and cfg['repo']==DINO_REPO,'Modelo incompatible: usa el directorio TensorFlow de esta entrega')
    with np.load(root/'normalizacion.npz',allow_pickle=False) as z:
        saved['mean']=z['mean'];saved['std']=z['std']
    with tf.device('/CPU:0'):
        model=tf.keras.models.load_model(root/saved['best_checkpoint'],compile=False)
    return saved,model


def predict_features(model,x,saved,device=None):
    from .tf_model import decode_numpy
    x=np.asarray(x,dtype=np.float32)
    if not len(x):
        return np.empty(0,np.float32),np.empty(0,np.float32),np.empty((0,len(saved['config']['classes'])),np.float32)
    z=(x-saved['mean'])/saved['std']
    logits=model(z,training=False).numpy()
    return decode_numpy(logits,saved['config']['classes'])


def curves(run,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['loss','mae']):
        ax.plot([r['epoch'] for r in history],[r[key] for r in history],label='train')
        ax.plot([r['epoch'] for r in history],[r['val_'+key] for r in history],label='val')
        ax.set(xlabel='Época',ylabel=key);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(run/'curvas.png',dpi=150);plt.close(fig)


def train(args):
    from .tf_model import tf,configure_tf,build_head,LDLLoss,decode_numpy
    fcfg,data=cache(args.features);run=Path(args.run).resolve()
    require(fcfg['kind']=='annotated','Entrenar solo con cajas anotadas')
    require(args.epochs>0 and args.batch>0 and args.lr>0 and args.patience>=0,'Parámetros no válidos')
    require(0<=args.dropout<1 and args.weight_decay>=0 and args.sigma>0 and args.emd_weight>=0,'Regularización/pérdida no válida')
    dev=configure_tf('cpu' if args.device=='cpu' else 'gpu')
    tf.keras.utils.set_random_seed(args.seed)
    classes=fcfg['classes'];trainmask=data['splits']=='train';valmask=data['splits']=='val'
    require(trainmask.any() and valmask.any(),'Faltan train/val')
    require(set(data['y'][trainmask].tolist())==set(classes),'Catálogo train incompleto')
    require(len(set(data['keys'][valmask].tolist()))==int(valmask.sum()),'Validación duplicada')
    original=trainmask & (data['views']==0);require(original.any(),'Faltan vistas originales')
    mean=data['x'][original].mean(0);std=np.maximum(data['x'][original].std(0),1e-6)
    x=(data['x']-mean)/std
    indices=np.asarray([classes.index(float(v)) for v in data['y']],np.int32)
    xtrain,ytrain=x[trainmask],indices[trainmask];xval,yval=x[valmask],indices[valmask]
    cows=data['cows'][trainmask].tolist();counts=Counter(cows)
    weights=np.asarray([1/counts[c] for c in cows],np.float32);weights/=weights.mean()
    config={'encoder':DINO_NAME,'repo':DINO_REPO,'classes':classes,'dim':DIM,'size':224,
        'output':'tf_ldl_expected_then_nearest','manifest_sha256':fcfg['manifest_sha256'],
        'features_sha256':fcfg['features_sha256'],'seed':args.seed,'lr':args.lr,'batch':args.batch,
        'dropout':args.dropout,'weight_decay':args.weight_decay,'patience':args.patience,
        'sigma':args.sigma,'emd_weight':args.emd_weight,'roi_mode':fcfg['roi_mode'],
        'default_roi':fcfg['default_roi'],'device_requested':args.device,'cow_weighting':'inverse_image_count'}
    history=[];stale=0;start=0;best=float('inf');best_epoch=0;best_path=None
    with tf.device(dev):
        loss_fn=LDLLoss(classes,args.sigma,args.emd_weight)
        if args.resume:
            require((run/'estado.json').is_file(),'No hay estado.json para reanudar')
            saved=load_json(run/'estado.json');require(saved['config']==config,'Cambió configuración/caché: inicia otra ejecución')
            with np.load(run/'normalizacion.npz') as z:require(np.array_equal(mean,z['mean']) and np.array_equal(std,z['std']),'Normalización distinta')
            model=tf.keras.models.load_model(run/saved['last_checkpoint'])
            optimizer=model.optimizer;history=saved['history'];stale=saved['stale'];start=saved['epoch']
            best=saved['best_metric'];best_epoch=saved['best_epoch'];best_path=saved['best_checkpoint']
        else:
            require(not run.exists(),'--run ya existe: usa --resume u otra carpeta')
            run.mkdir(parents=True);(run/'checkpoints').mkdir()
            np.savez_compressed(run/'normalizacion.npz',mean=mean,std=std)
            model=build_head(DIM,classes,args.dropout,args.seed)
            optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr,weight_decay=args.weight_decay)
            model.compile(optimizer=optimizer,loss=loss_fn)
            save_json(run/'settings.json',config)
        save_json(run/'entorno.json',{'tensorflow':tf.__version__,'keras':tf.keras.__version__,
            'device_requested':args.device,'device_scope':dev,'gpus':[g.name for g in tf.config.list_physical_devices('GPU')],
            'note':'Operaciones auxiliares sin kernel Metal pueden ejecutarse en CPU. No se promete identidad numérica entre plataformas.'})
        @tf.function(reduce_retracing=True)
        def step(xb,yb,wb):
            with tf.GradientTape() as tape:
                logits=model(xb,training=True);loss=tf.reduce_mean(loss_fn(yb,logits)*wb)
            gradients=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            return loss,logits
        for epoch in range(start,args.epochs):
            if args.patience and stale>=args.patience:
                print('Parada temprana alcanzada.');break
            order=np.random.default_rng(args.seed+epoch).permutation(len(ytrain))
            loss_sum=0.;err_sum=0.;n=len(ytrain)
            for offset in range(0,n,args.batch):
                ix=order[offset:offset+args.batch]
                loss,logits=step(xtrain[ix],ytrain[ix],weights[ix])
                loss=float(loss.numpy());require(np.isfinite(loss),'Pérdida no finita')
                _,pred,_=decode_numpy(logits.numpy(),classes)
                loss_sum+=loss*len(ix);err_sum+=float(np.abs(pred-np.asarray(classes)[ytrain[ix]]).sum())
            logits=model(xval,training=False);val_loss=float(tf.reduce_mean(loss_fn(yval,logits)).numpy())
            _,pred,_=decode_numpy(logits.numpy(),classes)
            val_mae=float(np.abs(pred-np.asarray(classes)[yval]).mean())
            h={'epoch':epoch+1,'loss':loss_sum/n,'mae':err_sum/n,'val_loss':val_loss,'val_mae':val_mae};history.append(h)
            filename=f'checkpoints/epoch_{epoch+1:04}.keras'
            if val_mae<best-1e-8:best=val_mae;best_epoch=epoch+1;stale=0;best_path=filename
            else:stale+=1
            # Guardar un modelo completo antes de actualizar el puntero atómico estado.json.
            temp=run/'checkpoints/guardando.tmp.keras';model.save(temp);temp.replace(run/filename)
            save_json(run/'estado.json',{'config':config,'epoch':epoch+1,'history':history,'stale':stale,
                'best_metric':best,'best_epoch':best_epoch,'best_checkpoint':best_path,'last_checkpoint':filename})
            # Mantener solo último y mejor; una interrupción antes del puntero conserva el estado previo.
            for p in (run/'checkpoints').glob('epoch_*.keras'):
                if p.relative_to(run).as_posix() not in (filename,best_path):p.unlink()
            save_csv(run/'historial.csv',history)
            print(f'Época {epoch+1}: train MAE={err_sum/n:.4f}; val MAE={val_mae:.4f}; mejor={best:.4f}',flush=True)
    require(bool(history),'No hay épocas guardadas')
    curves(run,history)
    save_json(run/'seleccion.json',{'best_epoch':best_epoch,'val_mae_continuous':best,
        'checkpoint':best_path,'note':'Selección con validación; no hay resultados test.'})


def model_fingerprint(path):
    root=Path(path).resolve();root=root.parent if root.is_file() else root
    saved=load_json(root/'estado.json')
    return sha256(root/saved['best_checkpoint'])
