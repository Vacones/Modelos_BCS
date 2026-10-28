import argparse
import json
import platform
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--framework',choices=['tf','torch'],required=True);args=parser.parse_args()
    print('Python:',sys.version,'Sistema:',platform.platform())
    try:subprocess.run(['nvidia-smi'],check=True)
    except (OSError,subprocess.CalledProcessError) as e:
        raise RuntimeError('No funciona nvidia-smi dentro de WSL. Actualiza el controlador NVIDIA de Windows y WSL2 antes de continuar.') from e
    if args.framework=='tf':
        from vacas.tf_model import tensorflow_ready
        tf=tensorflow_ready()
        with tf.device('/GPU:0'):
            x=tf.random.normal((8,64,64,3));layer=tf.keras.layers.Conv2D(8,3)
            with tf.GradientTape() as tape:loss=tf.reduce_mean(layer(x)**2)
            gradients=tape.gradient(loss,layer.trainable_weights)
            tf.keras.optimizers.Adam().apply_gradients(zip(gradients,layer.trainable_weights))
        if 'GPU' not in loss.device:raise RuntimeError('La operación se ejecutó fuera de GPU')
        print('TensorFlow:',tf.__version__,'Operación:',loss.device,'Pérdida:',float(loss))
        print(json.dumps(tf.sysconfig.get_build_info(),default=str,indent=2))
    else:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('PyTorch no ve la GPU')
        layer=torch.nn.Conv2d(3,8,3).cuda();x=torch.randn(8,3,64,64,device='cuda')
        optimizer=torch.optim.Adam(layer.parameters());loss=layer(x).square().mean();loss.backward();optimizer.step();torch.cuda.synchronize()
        print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),'Pérdida:',float(loss.detach().cpu()))
    print('GPU comprobada con una operación de entrenamiento. Aún falta probar EfficientNet/YOLO con tus datos.')


if __name__=='__main__':main()
