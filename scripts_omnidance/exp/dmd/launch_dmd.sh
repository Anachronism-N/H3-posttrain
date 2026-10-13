#!/bin/bash
# Usage: launch_dmd.sh <exp_name> <gpus e.g. 0,1,..,7> [extra train_dmd.py args...]
NAME=$1; GPUS=$2; shift 2
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
OUT=$REPO/models/train/$NAME
LOG=$REPO/logs_omnidance/exp/E5/$NAME.log
mkdir -p $REPO/logs_omnidance/exp/E5 $OUT
bash $REPO/scripts_omnidance/exp/hold.sh stop $GPUS
N=$(echo $GPUS | tr ',' '\n' | wc -l)
PORT=$((29500 + RANDOM % 1000))
cd $REPO
PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPUS \
DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true WANDB_MODE=offline \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup /opt/venv_od/bin/python -m torch.distributed.run --nproc_per_node $N --master_port $PORT \
  scripts_omnidance/exp/dmd/train_dmd.py --out $OUT "$@" >> $LOG 2>&1 < /dev/null &
echo "[E5] $NAME launched on $(hostname -I | awk '{print $1}') gpus=$GPUS log=$LOG"
