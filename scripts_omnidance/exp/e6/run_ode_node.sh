#!/bin/bash
# run_ode_node.sh <node_idx> <n_nodes> <out> [num]
NI=$1; NN=$2; O=$3; NUM=${4:-1200}
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O/logs; cd $R
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$R CUDA_VISIBLE_DEVICES=$g DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true \
      python scripts_omnidance/exp/e6/gen_ode_pairs.py --out $O --num $NUM --start ${START:-0} --shard $((NI*8+g)) --nshard $((NN*8)) > $O/logs/n${NI}_g$g.log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 15
done
wait; echo "[ode] node $NI done $(date)"
