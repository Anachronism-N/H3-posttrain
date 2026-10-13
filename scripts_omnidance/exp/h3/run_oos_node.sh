#!/bin/bash
# run_oos_node.sh <out>: wait for FL2VA copy, then 5 GPU processes x 2 OOS prompts (cgroup RAM limit -> max 5 H3 procs/node)
O=$1; R=/apdcephfs_gy4/share_302533218/cedricnie; U=$R/DiffSynth-Studio-upstream-20260930
while ! grep -q copied /tmp/cp_fl2va.log 2>/dev/null; do sleep 30; done
mkdir -p $O/logs
for g in 0 1 2 3 4; do
  ( bash $R/diff/scripts_omnidance/exp/hold.sh stop $g
    cd /opt && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$U /opt/venv_h3/bin/python $R/diff/scripts_omnidance/exp/h3/oos_t2va.py \
      --prompts $O/oos_prompts.json --out $O --shard $g --nshard 5 > $O/logs/g$g.log 2>&1
    bash $R/diff/scripts_omnidance/exp/hold.sh start $g ) &
  sleep 60
done
wait; touch $O/.done
