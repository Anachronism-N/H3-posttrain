#!/bin/bash
# run_h3ar_node.sh <node_idx> <n_nodes> <out> [sample_h3_ar.py args]
NI=$1; NN=$2; O=$3; shift 3
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930
# GPUS="0 1 2 3 4" limits processes per node: each H3 sampler holds ~122 GB CPU RAM and the cgroup limit is 998 GB incl. /dev/shm
G=(${GPUS:-0 1 2 3 4}); NG=${#G[@]}
mkdir -p $O/logs
pids=()
for i in "${!G[@]}"; do
  g=${G[$i]}
  ( bash $R/scripts_omnidance/exp/hold.sh stop $g
    cd /opt && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$U /opt/venv_h3/bin/python $R/scripts_omnidance/exp/h3ar/sample_h3_ar.py --out $O --shard $((NI*NG+i)) --nshard $((NN*NG)) "$@" > $O/logs/n${NI}_g$g.log 2>&1
    rc=$?
    bash $R/scripts_omnidance/exp/hold.sh start $g
    exit "$rc" ) &
  pids+=("$!")
  sleep 3m
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
[ "$rc" = 0 ] && touch "$O/.done_$NI"
exit "$rc"
