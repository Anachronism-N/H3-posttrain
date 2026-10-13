#!/bin/bash
# run_gen_node.sh <out> <node_idx> <n_nodes> [e7_gen.py args...]
O=$1; NI=$2; NN=$3; shift 3
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O; cd $R
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$R CUDA_VISIBLE_DEVICES=$g DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true \
      python scripts_omnidance/exp/e7/e7_gen.py --out $O --shard $((NI*8+g)) --nshard $((NN*8)) "$@" > $O/gen_$((NI*8+g)).log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 12
done
wait; touch $O/.gen_done_$NI
