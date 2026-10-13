#!/bin/bash
# launch_e6.sh <script.py> <exp_name> <gpus> [args...]
SC=$1; NAME=$2; GPUS=$3; shift 3
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
OUT=$REPO/models/train/$NAME; LOG=$REPO/logs_omnidance/exp/E6/$NAME.log
mkdir -p $REPO/logs_omnidance/exp/E6 $OUT
bash $REPO/scripts_omnidance/exp/hold.sh stop $GPUS >/dev/null
N=$(echo $GPUS | tr ',' '\n' | wc -l); PORT=$((29500 + RANDOM % 1000))
cd $REPO
PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPUS \
DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
setsid nohup /opt/venv_od/bin/python -m torch.distributed.run --nproc_per_node $N --master_port $PORT \
  scripts_omnidance/exp/e6/$SC --out $OUT "$@" >> $LOG 2>&1 < /dev/null &
echo "[E6] $NAME on $(hostname -I | awk '{print $1}') gpus=$GPUS log=$LOG"
