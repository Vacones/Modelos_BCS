"""Sondas en procesos separados; sin modificar el entorno instalado."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from common import artifacts, require, save


def probe(cfg, framework):
    report = {'python': sys.version, 'platform': platform.platform(), 'framework': framework}
    if framework == 'system':
        import psutil
        report['ram'] = dict(psutil.virtual_memory()._asdict())
        report['disk'] = dict(zip(('total', 'used', 'free'), shutil.disk_usage(cfg['WORK_ROOT'])))
        command = ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', '--format=csv']
        try:
            report['nvidia_smi'] = subprocess.check_output(command, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            report['nvidia_smi'] = str(exc)
        report['internet'] = {}
        for host in ('https://pypi.org/simple/ultralytics/', 'https://github.com/ultralytics/assets',
                     'https://storage.googleapis.com/keras-applications/'):
            try:
                with urllib.request.urlopen(host, timeout=10) as response:
                    report['internet'][host] = response.status
            except Exception as exc:
                report['internet'][host] = str(exc)
        report['packages'] = {p: importlib.metadata.version(p) for p in
                              ('numpy', 'pillow', 'matplotlib', 'psutil', 'PyYAML')}
    elif framework == 'torch':
        import torch
        import ultralytics
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.engine.trainer import BaseTrainer
        import inspect
        require(ultralytics.__version__ == cfg['ULTRALYTICS_VERSION'],
                'Ultralytics no coincide con la versión fijada. Instala la versión de configuración.')
        required = {'resume', 'save_period', 'time', 'device', 'epochs'}
        require(required <= DEFAULT_CFG_DICT.keys(), 'API Ultralytics incompatible.')
        report.update(torch=torch.__version__, ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                      gpu_count=torch.cuda.device_count(), api_defaults={k: DEFAULT_CFG_DICT[k] for k in required},
                      trainer_source_sha=__import__('hashlib').sha256(inspect.getsource(BaseTrainer).encode()).hexdigest())
        report['resume_implementation'] = inspect.getsource(BaseTrainer.check_resume)
        report['resume_training_implementation'] = inspect.getsource(BaseTrainer.resume_training)
        report['final_eval_implementation'] = inspect.getsource(BaseTrainer.final_eval)
        report['multi_gpu_implementation'] = inspect.getsource(BaseTrainer.train)
        require(torch.cuda.is_available(), 'PyTorch no detecta CUDA. Activa GPU en Kaggle.')
        report['gpus'] = [dict(index=i, name=torch.cuda.get_device_name(i),
                               memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                          for i in range(torch.cuda.device_count())]
        device = int(cfg['DETECTOR']['device'])
        require(device < torch.cuda.device_count(), 'GPU configurada inexistente.')
        x = torch.randn((32, 32), device=f'cuda:{device}', requires_grad=True)
        x.square().mean().backward()
        torch.cuda.synchronize(device)
        report['gpu_operation_ok'] = True
    elif framework == 'tensorflow':
        import tensorflow as tf
        import keras
        gpu = tf.config.list_physical_devices('GPU')
        require(gpu, 'TensorFlow no detecta GPU. Revisa el entorno antes de entrenar.')
        tf.config.set_soft_device_placement(False)
        tf.config.set_visible_devices(gpu[int(cfg['REGRESSOR']['device'])], 'GPU')
        tf.config.experimental.set_memory_growth(tf.config.get_visible_devices('GPU')[0], True)
        with tf.device('/GPU:0'):
            x = tf.Variable(tf.ones((16, 16)))
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.matmul(x, x))
            require(tape.gradient(loss, x) is not None, 'Falló operación GPU TensorFlow.')
        report.update(tensorflow=tf.__version__, keras=keras.__version__,
                      gpu_count=len(gpu), build=tf.sysconfig.get_build_info(), gpu_operation_ok=True)
    save(artifacts(cfg) / 'reports' / ('environment_' + framework + '.json'), report)
    print(json.dumps(report, indent=2, default=str))
