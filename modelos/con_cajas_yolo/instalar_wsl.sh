#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  tf)
    python3 -m venv .venv-tf
    .venv-tf/bin/python -m pip install --upgrade pip
    .venv-tf/bin/python -m pip install -r requirements-tf.txt
    .venv-tf/bin/python -m pip check
    .venv-tf/bin/python comprobar_gpu.py --framework tf
    .venv-tf/bin/python -m pip freeze > entorno_tf_instalado.txt
    ;;
  yolo)
    python3 -m venv .venv-yolo
    .venv-yolo/bin/python -m pip install --upgrade pip
    .venv-yolo/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "${TORCH_WHL_INDEX:-https://download.pytorch.org/whl/cu126}"
    .venv-yolo/bin/python -m pip install -r requirements-yolo.txt
    .venv-yolo/bin/python -m pip check
    .venv-yolo/bin/python comprobar_gpu.py --framework torch
    .venv-yolo/bin/python -m pip freeze > entorno_yolo_instalado.txt
    ;;
  *) echo "Uso: bash instalar_wsl.sh tf | bash instalar_wsl.sh yolo"; exit 2 ;;
esac
