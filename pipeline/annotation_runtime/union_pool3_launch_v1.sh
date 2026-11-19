#!/bin/bash
# usage: union_pool3_launch_v1.sh <host> <worker_id> <cuda_visible_devices>
set -e
R=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime; B=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1
host=$1; wid=$2; cvd=$3
cmd="mkdir -p $B/union_pool3_logs; test ! -e $B/union_pool3_logs/$wid.log; cd $R && env CUDA_VISIBLE_DEVICES=$cvd HF_HUB_OFFLINE=1 MS_VISUAL_CPU_SITEPATH=/opt/venv_eval/lib/python3.11/site-packages OMP_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 setsid nohup /opt/venv_vllm/bin/python -B $R/caption_union_pool_worker_v3.py --config $R/${wid}_config.json > $B/union_pool3_logs/$wid.log 2>&1 < /dev/null &"
timeout 40 ssh -f -n -o BatchMode=yes $host "$cmd"
