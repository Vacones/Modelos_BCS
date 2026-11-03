#!/usr/bin/env python3
"""Comprobación explícita de las dos rutas GPU; no sustituye por CPU silenciosamente."""
import argparse,platform,time
p=argparse.ArgumentParser();p.add_argument('--dino',action='store_true');a=p.parse_args()
import torch
print('Sistema:',platform.platform(),'arquitectura:',platform.machine())
print('PyTorch:',torch.__version__,'MPS compilado:',torch.backends.mps.is_built(),'MPS disponible:',torch.backends.mps.is_available())
if not torch.backends.mps.is_available():raise SystemExit('MPS no disponible: revisa macOS, Python ARM64 y Terminal sin Rosetta. No se inicia entrenamiento CPU.')
x=torch.randn(256,256,device='mps');y=x@x;torch.mps.synchronize()
assert y.device.type=='mps' and torch.isfinite(y).all().item()
print('Multiplicación GPU MPS: correcta')
if a.dino:
    from bcs.features import encoder,embed_images,DIM
    from bcs.regions import DEFAULT_ROI
    from PIL import Image
    net=encoder(torch.device('mps'));z=embed_images(net,[Image.new('RGB',(224,224),'gray')],[DEFAULT_ROI],torch.device('mps'))
    assert z.shape==(1,DIM)
    del net,z
    print('DINOv2 real en GPU MPS: correcto')
del x,y;torch.mps.empty_cache()
from bcs.tf_model import tf,configure_tf,build_head,LDLLoss
configure_tf('gpu')
print('TensorFlow:',tf.__version__,'GPUs:',tf.config.list_physical_devices('GPU'))
with tf.device('/GPU:0'):
    x=tf.random.normal((256,256));out=tf.matmul(x,x)
    assert 'GPU:0' in out.device and bool(tf.reduce_all(tf.math.is_finite(out)).numpy())
    model=build_head(1536,[2.25,2.5,2.75,3.,3.25,3.5,3.75])
    optimizer=tf.keras.optimizers.AdamW(learning_rate=.001)
    with tf.GradientTape() as tape:
        logits=model(tf.zeros((8,1536)),training=True)
        loss=tf.reduce_mean(LDLLoss([2.25,2.5,2.75,3.,3.25,3.5,3.75])(tf.constant([3]*8),logits))
    optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
    assert 'GPU:0' in logits.device and bool(tf.math.is_finite(loss).numpy())
print('TensorFlow/Metal: matrices y paso de entrenamiento LDL en GPU correctos.')
print('Esto comprueba compatibilidad, no mide velocidad frente a CPU ni precisión BCS.')
