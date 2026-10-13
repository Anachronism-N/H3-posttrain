#!/bin/bash
# 从 S3 step-5000 恢复 stage3 训练(修正版)
# - 新输出目录 OSS_fullSFT_stage3_resume_from5k,不覆盖原 stage3 产出
# - 本地步数 1..6800 映射原始 stage3 步数 5204..12000(含 LR 衰减对齐)
# - 优化器状态无法恢复(权重 warm-start),300 步短 warmup
set -e

REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate wan
cd "$REPO"

RESUME_CKPT=$REPO/models/train/OSS_fullSFT_stage3_10k_joint30_a30_t30_u10/step-5000.safetensors
if [ ! -f "$RESUME_CKPT" ]; then
  echo "FATAL: resume ckpt not found: $RESUME_CKPT"
  exit 1
fi

# 幂等:已在跑则不重启
if pgrep -f "model_training/train.py" > /dev/null 2>&1; then
  echo "[resume] training already running, skip. $(date)"
  exit 0
fi

LOG=$REPO/logs_omnidance/s3_resume.log

DIFFSYNTH_MODEL_BASE_PATH=$REPO/models/Wan-AI \
DIFFSYNTH_SKIP_DOWNLOAD=true \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DATA_DIR=$REPO/data/omnidance \
setsid nohup accelerate launch --num_processes 8 --num_machines 1 \
  examples/wanvideo/model_training/train.py \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/omnidance_fullSFT_base.yaml \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/fullSFT_stage3.yaml \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/fullSFT_stage3_resume.yaml \
  --find_unused_parameters \
  --trainable_models dit,dit.audio_injector.injector_global_mod_layers \
  --train_checkpoint_path "$RESUME_CKPT" \
  > "$LOG" 2>&1 < /dev/null &

disown
echo "[resume] launched, log=$LOG  $(date)"
