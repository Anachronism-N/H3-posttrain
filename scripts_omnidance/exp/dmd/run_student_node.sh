#!/bin/bash
# run_student_node.sh <out_root> <gpus csv> [sample_student.py args...]   (holders restored per GPU after)
O=$1; GPUS=$2; shift 2
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O
cd $REPO
N=$(echo $GPUS | tr ',' '\n' | wc -l); i=0
for g in ${GPUS//,/ }; do
  (
    bash scripts_omnidance/exp/hold.sh stop $g
    PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$g \
    DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true \
    python scripts_omnidance/exp/dmd/sample_student.py --out $O --shard $i --nshard $N "$@" > $O/shard_$i.log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g
  ) &
  sleep 15; i=$((i+1))
done
wait
touch $O/.done
echo "[student] node done $(date)"
