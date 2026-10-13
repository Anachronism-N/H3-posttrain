#!/bin/bash
# run_caption_node.sh <csv> <out_dir> <node_idx> <n_nodes>
M=$1; O=$2; NI=$3; NN=$4; R=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O; cd $R
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    NUMBA_CACHE_DIR=/tmp/nbc_$g CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 /opt/venv_eval/bin/python scripts_omnidance/exp/e10/caption_motion.py --csv $M --out $O/cap_$((NI*8+g)).csv --shard $((NI*8+g)) --nshard $((NN*8)) > $O/cap_$((NI*8+g)).log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 5
done
wait; touch $O/.done_$NI
