"""Cabeza LDL en TensorFlow/Keras; DINOv2 se conserva en su implementación oficial."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import tensorflow as tf
from .common import require


def configure_tf(device='gpu'):
    """GPU obligatoria para entrenar; CPU explícita para pruebas o inferencia pequeña."""
    if str(device)=='cpu':
        try:tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            require(not tf.config.get_visible_devices('GPU'),'TensorFlow ya inicializó GPU; usa otro proceso para CPU')
        return '/CPU:0'
    gpus=tf.config.list_physical_devices('GPU')
    require(bool(gpus),'TensorFlow no detecta GPU Metal. Ejecuta comprobar_gpu_mac.py; no se cambia a CPU automáticamente.')
    # Permite operaciones auxiliares en CPU si Metal no tiene kernel; las matrices/variables usan GPU.
    tf.config.set_soft_device_placement(True)
    return '/GPU:0'


def soft_targets(indices,classes,sigma=.125):
    values=tf.constant(classes,dtype=tf.float32)
    real=tf.gather(values,tf.cast(tf.reshape(indices,[-1]),tf.int32))
    return tf.nn.softmax(-.5*tf.square((values[None,:]-real[:,None])/sigma),axis=-1)


def point_losses(indices,logits,classes,sigma=.125,emd_weight=.5):
    q=soft_targets(indices,classes,sigma)
    ce=-tf.reduce_sum(q*tf.nn.log_softmax(logits,axis=-1),axis=-1)
    p=tf.nn.softmax(logits,axis=-1)
    distance=tf.reduce_mean(tf.square(tf.cumsum(p,axis=-1)[:,:-1]-tf.cumsum(q,axis=-1)[:,:-1]),axis=-1)
    return ce+emd_weight*distance


@tf.keras.utils.register_keras_serializable(package='BCSMac')
class LDLLoss(tf.keras.losses.Loss):
    def __init__(self,classes,sigma=.125,emd_weight=.5,reduction='none',name='ldl_loss',**kwargs):
        super().__init__(name=name,reduction=reduction,**kwargs)
        self.classes=list(classes);self.sigma=float(sigma);self.emd_weight=float(emd_weight)

    def call(self,y_true,y_pred):
        return point_losses(y_true,y_pred,self.classes,self.sigma,self.emd_weight)

    def get_config(self):
        return {**super().get_config(),'classes':self.classes,'sigma':self.sigma,'emd_weight':self.emd_weight}


def build_head(dim,classes,dropout=.30,seed=42):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(dim,)),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.Dense(128),tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dropout(dropout,seed=seed),tf.keras.layers.Dense(len(classes))
    ],name='BCS_LDL_Mac')


def decode_numpy(logits,classes):
    logits=np.asarray(logits,dtype=np.float32)
    exp=np.exp(logits-logits.max(-1,keepdims=True));probs=exp/exp.sum(-1,keepdims=True)
    values=np.asarray(classes,dtype=np.float32);expected=probs@values
    category=values[np.abs(expected[:,None]-values[None,:]).argmin(-1)]
    return category,expected,probs
