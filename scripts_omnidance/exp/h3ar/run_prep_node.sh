#!/bin/bash
# run_prep_node.sh <node_idx> <n_nodes>: 8 GPUs -> /dev/shm/h3data
NI=$1; NN=$2; R=/apdcephfs_gy4/share_302533218/cedricnie/diff; L=$R/logs_omnidance/exp/H3AR; cd $R
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    CUDA_VISIBLE_DEVICES=$g /opt/venv_eval/bin/python scripts_omnidance/exp/h3ar/prep_h3_data.py --csv $L/h3_train36k.csv --out /dev/shm/h3data --shard $((NI*8+g)) --nshard $((NN*8)) > $L/prep_$((NI*8+g)).log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 3
done
wait; touch /dev/shm/h3data/.done_$NI; echo "[prep] node $NI done $(date)"
