#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
command -v "$PYTHON_BIN" >/dev/null || { echo 'Necesitas Python 3.11 o 3.12 ARM64. Lee README.md'; exit 2; }
"$PYTHON_BIN" - <<'PY'
import platform,sys
assert sys.platform=='darwin' and platform.machine()=='arm64','Ejecuta Terminal nativa en Apple Silicon (sin Rosetta)'
assert (3,11)<=sys.version_info[:2]<=(3,12),'Usa Python 3.11 o 3.12: tensorflow-metal 1.2.0 no ofrece wheel para 3.13+'
print('Python compatible:',sys.version,platform.machine())
PY
"$PYTHON_BIN" -m venv .venv-mac
.venv-mac/bin/python -m pip install --upgrade pip
.venv-mac/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 -r requirements.txt
.venv-mac/bin/python -m pip check
export SSL_CERT_FILE="$(.venv-mac/bin/python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
.venv-mac/bin/python comprobar_gpu_mac.py --dino
.venv-mac/bin/python -m pip freeze > entorno_instalado.txt
echo 'Instalación terminada. MPS y TensorFlow/Metal comprobados en este equipo.'
