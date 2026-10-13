#!/bin/bash
# run_eval_node.sh <manifest> <out_dir> <gpus csv> [extra eval args]
# One shard per GPU; results out_dir/shard_<i>.jsonl. Holders restored per-GPU when done.
M=$1; O=$2; GPUS=$3; shift 3
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O
cd $REPO
N=$(echo $GPUS | tr ',' '\n' | wc -l)
i=0
for g in ${GPUS//,/ }; do
  (
    bash scripts_omnidance/exp/hold.sh stop $g
    NUMBA_CACHE_DIR=/tmp/numba_cache_$g CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HOME=/apdcephfs_gy4/share_302533218/cedricnie/models/hf_cache \
    http_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128 \
    /opt/venv_eval/bin/python scripts_omnidance/exp/eval/eval_metrics.py --manifest $M --out $O/shard_$i.jsonl \
      --shard $i --nshard $N "$@" > $O/shard_$i.log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g
  ) &
  i=$((i+1))
done
wait
echo "[eval] node done $(date)"
