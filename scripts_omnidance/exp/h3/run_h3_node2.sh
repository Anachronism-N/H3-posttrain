#!/bin/bash
# run_h3_node.sh <node_idx> <n_nodes> <out>   8 GPUs/node, global shard = node*8+gpu
NI=$1; NN=$2; O=$3
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930
mkdir -p $O/logs
for g in 0 1 2 3 4 5 6 7; do
  (
    bash $R/scripts_omnidance/exp/hold.sh stop $g
    cd /opt && NUMBA_CACHE_DIR=/tmp/nbh_$g CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$U \
      /opt/venv_h3/bin/python $R/scripts_omnidance/exp/h3/e3_h3_zeroshot.py --out $O \
      --shard $((NI*8+g)) --nshard $((NN*8)) $H3ARGS > $O/logs/n${NI}_g$g.log 2>&1
    bash $R/scripts_omnidance/exp/hold.sh start $g
  ) &
  sleep 45
done
wait
echo "[h3] node $NI done $(date)"
