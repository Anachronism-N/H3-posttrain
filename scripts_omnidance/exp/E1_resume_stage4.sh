#!/bin/bash
# E1: stage4 resume from step-11000 on N1 (8 GPUs). Idempotent.
set -e
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
export PATH=/opt/venv_od/bin:$PATH
export PYTHONPATH=/apdcephfs_gy4/share_302533218/cedricnie/diff
cd "$REPO"
CKPT=/dev/shm/od/ckpt/S4-11000/step-11000.safetensors
[ -f /dev/shm/od/.staged ] || CKPT=$REPO/models/train/OSS_fullSFT_stage4_omni257k_b32_2ep/step-11000.safetensors
if pgrep -f "[m]odel_training/train.py" > /dev/null; then echo "[E1] already running"; exit 0; fi
OUT=$REPO/models/train/OSS_fullSFT_stage4_resume_from11k
if ls $OUT/step-*.safetensors >/dev/null 2>&1; then echo "[E1] FATAL: output dir already has ckpts, refusing to overwrite"; exit 1; fi
bash $REPO/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
mkdir -p $REPO/logs_omnidance/exp/E1
LOG=$REPO/logs_omnidance/exp/E1/train.log
CFG=$REPO/examples/wanvideo/model_training/full/reproduce
DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI \
DIFFSYNTH_SKIP_DOWNLOAD=true \
WANDB_MODE=offline OD_ZERO_OPT=1 OD_GRAD_BUCKET_VIEW=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
setsid nohup accelerate launch --num_processes 8 --num_machines 1 \
  examples/wanvideo/model_training/train.py \
  --config $CFG/omnidance_fullSFT_base.yaml \
  --config $CFG/fullSFT_stage4_omni2ep.yaml \
  --config $CFG/fullSFT_stage4_resume_from11k.yaml \
  --find_unused_parameters \
  --trainable_models dit,dit.audio_injector.injector_global_mod_layers \
  --mert_encoder_path /dev/shm/od/MERT-v1-330M \
  --tokenizer_path /dev/shm/od/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl \
  --train_checkpoint_path "$CKPT" \
  > "$LOG" 2>&1 < /dev/null &
disown
echo "[E1] launched ckpt=$CKPT log=$LOG $(date)"
