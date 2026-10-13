#!/bin/bash
# Local venv (system torch 2.8 + wan env pins) at /opt/venv_od. Idempotent.
set -e
[ -f /opt/venv_od/.ok ] && { echo "venv ok"; exit 0; }
/usr/bin/python3 -m venv --system-site-packages /opt/venv_od
/opt/venv_od/bin/pip install -q --upgrade pip
/opt/venv_od/bin/pip install -q -r /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/wan_reqs.txt
/opt/venv_od/bin/pip install -q "numpy<2.3" opencv-python-headless decord av pynvml
/opt/venv_od/bin/pip install -q torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
/opt/venv_od/bin/python -c "import torch,transformers,accelerate,peft,librosa,diffsynth" 2>/dev/null || true
cd /opt && PYTHONPATH=/apdcephfs_gy4/share_302533218/cedricnie/diff /opt/venv_od/bin/python -c "import torch,transformers,accelerate,peft,librosa; import diffsynth; print('import ok', torch.__version__, transformers.__version__)"
touch /opt/venv_od/.ok
