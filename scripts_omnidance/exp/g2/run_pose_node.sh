#!/bin/bash
# run_pose_node.sh <cap_node_idx> <pose_node_idx> <n_pose_nodes>: wait for this node's caption job, then DWPose on 8 GPUs
CI=$1; NI=$2; NN=$3; R=/apdcephfs_gy4/share_302533218/cedricnie/diff; G=$R/logs_omnidance/exp/G2; cd $R
while [ ! -f $R/logs_omnidance/exp/E10/train_caps/.done_$CI ]; do sleep 60; done
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    CUDA_VISIBLE_DEVICES=$g /opt/venv_eval/bin/python scripts_omnidance/exp/g2/extract_pose.py --csv $G/pose20k.csv --out $G/pose --shard $((NI*8+g)) --nshard $((NN*8)) > $G/pose_$((NI*8+g)).log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 5
done
wait; touch $G/.pose_done_$NI
