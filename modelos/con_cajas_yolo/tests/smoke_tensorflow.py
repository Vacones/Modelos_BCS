"""Prueba técnica pequeña sin Internet, sin entrenar ni evaluar el dataset real.
Ejecutar desde la raíz: .venv-tf/bin/python -m tests.smoke_tensorflow
"""
import tempfile
from pathlib import Path
import numpy as np
from vacas.tf_model import tensorflow_ready,build_model


def main():
    tf=tensorflow_ready()
    # Arquitectura real, pesos aleatorios: no descarga ImageNet ni produce resultados BCS útiles.
    model=build_model(tf,[2.5,3.0],224,weights=None)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',jit_compile=False)
    x=np.random.default_rng(42).uniform(0,255,(2,224,224,3)).astype('float32');y=np.array([0,1])
    model.train_on_batch(x,y)
    before=model(x,training=False).numpy()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'test.keras';model.save(path)
        restored=tf.keras.models.load_model(path)
        np.testing.assert_allclose(before,restored(x,training=False).numpy(),rtol=1e-5,atol=1e-6)
        base=restored.get_layer('efficientnetb0');base.trainable=True
        for layer in base.layers[:-20]:layer.trainable=False
        for layer in base.layers:
            if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
        restored.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',jit_compile=False)
        restored.train_on_batch(x,y)
    print('SMOKE CORRECTO: EfficientNet, paso de entrenamiento, guardado/carga y ajuste fino.')


if __name__=='__main__':main()
