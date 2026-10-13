#!/bin/bash
# chain_train_val.sh <gen_dir> <n_groups> <init_lora> <tag> [e7_train args...]: wait rewards -> train (8 GPU) -> val sample -> val eval
G=$1; NG=$2; INIT=$3; TAG=$4; shift 4
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; E7=$R/logs_omnidance/exp/E7; cd $R
while [ "$(cat $G/rewards_*.jsonl 2>/dev/null | wc -l)" -lt "$NG" ]; do sleep 60; done
while pgrep -f "[e]7_reward.py|[e]7_gen.py" >/dev/null; do sleep 30; done
bash scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
mkdir -p $E7/${TAG}_train
PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$R DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /opt/venv_od/bin/python -m torch.distributed.run --nproc_per_node 8 --master_port 29741 scripts_omnidance/exp/e7/e7_train.py \
  --gen $G --init $INIT --out $E7/${TAG}_train "$@" > $E7/${TAG}_train.log 2>&1
cp $E7/${TAG}_train/gen_lora_ema.pt /dev/shm/od/init/E7_${TAG}.pt
SIG=1.0,0.9405940771102905,0.8333333134651184,0.6372548937797546
bash scripts_omnidance/exp/dmd/run_student_node.sh $E7/val/E7_$TAG 0,1,2,3,4,5,6,7 --lora /dev/shm/od/init/E7_${TAG}.pt \
  --ckpt /dev/shm/od/ckpt/S4-final/step-final.safetensors --height 1280 --width 704 --num_frames 81 --sigmas $SIG --conds joint,tswap --ar_block 4 --csv_dir $E7/val_full
bash scripts_omnidance/exp/e7/post_val_eval.sh $E7/val/E7_$TAG E7_$TAG
echo "[chain] $TAG done $(date)"
