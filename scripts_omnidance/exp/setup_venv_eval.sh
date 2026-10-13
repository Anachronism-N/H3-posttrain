#!/bin/bash
# Eval venv: /opt/venv_eval = system torch 2.8 + transformers>=4.57 (Qwen3-VL) + "onnxruntime-gpu==1.22.0" (DWPose) + librosa etc.
set -e
export http_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128
[ -f /opt/venv_eval/.ok ] && { echo ok; exit 0; }
/usr/bin/python3 -m venv --system-site-packages /opt/venv_eval
P=/opt/venv_eval/bin/pip
$P install -q --upgrade pip
$P install -q --ignore-installed certifi idna tqdm typing_extensions huggingface_hub hf_xet
$P install -q "transformers>=4.57" accelerate qwen-vl-utils "onnxruntime-gpu==1.22.0" librosa soundfile scipy opencv-python-headless imageio imageio-ffmpeg decord pandas "numpy<2.3" safetensors einops
cd /opt && /opt/venv_eval/bin/python -c "import transformers,onnxruntime,librosa,cv2,decord;from transformers import Qwen3VLForConditionalGeneration;print('eval ok',transformers.__version__,onnxruntime.get_available_providers())"
touch /opt/venv_eval/.ok
