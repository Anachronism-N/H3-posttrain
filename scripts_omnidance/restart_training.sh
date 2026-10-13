#!/bin/bash
# OmniDance fullSFT 一键重启脚本（由会话监控使用，也可手动执行）
# 特性：幂等（若训练已在跑则不动）、setsid 脱离会话防 SIGHUP、日志滚动保留
set -u

REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
LOGDIR=$REPO/logs_omnidance

# 已在跑则不重启
if pgrep -f "train.py --config" > /dev/null 2>&1; then
  echo "[restart] training already running, skip. $(date)"
  exit 0
fi

source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate wan
cd "$REPO"

N=$(ls "$LOGDIR"/s1_full.log* 2>/dev/null | wc -l)
LOG="$LOGDIR/s1_full.restart${N}.log"

DIFFSYNTH_MODEL_BASE_PATH=$REPO/models/Wan-AI \
DIFFSYNTH_SKIP_DOWNLOAD=true \
WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
RESUME_STEP=6000 \
DATA_DIR=$REPO/data/omnidance \
BASE_CONFIG=examples/wanvideo/model_training/full/reproduce/omnidance_fullSFT_base.yaml \
setsid nohup bash examples/wanvideo/model_training/full/reproduce/run_fullSFT_full_stages.sh \
  > "$LOG" 2>&1 < /dev/null &
disown

echo "[restart] relaunched training, log=$LOG  $(date)"
