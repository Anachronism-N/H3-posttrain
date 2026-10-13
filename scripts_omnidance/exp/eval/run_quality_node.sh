#!/bin/bash
# run_quality_node.sh <manifest> <out_dir> <gpus csv>: quality_metrics.py, one shard per GPU, holders restored per GPU
M=$1; O=$2; GPUS=$3; REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O; cd $REPO
N=$(echo $GPUS | tr ',' '\n' | wc -l); i=0
for g in ${GPUS//,/ }; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 /opt/venv_eval/bin/python scripts_omnidance/exp/eval/quality_metrics.py --manifest $M --out $O/shard_$i.jsonl \
      --shard $i --nshard $N > $O/shard_$i.log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  i=$((i+1))
done
wait
echo "[quality] node done $(date)"
