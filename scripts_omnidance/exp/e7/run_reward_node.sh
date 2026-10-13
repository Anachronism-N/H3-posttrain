#!/bin/bash
# run_reward_node.sh <gen_dir> <node_idx> <n_nodes> <n_groups_expected>
O=$1; NI=$2; NN=$3; NG=$4; shift 4
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; cd $R
while [ "$(cat $O/groups_*.jsonl 2>/dev/null | wc -l)" -lt "$NG" ]; do sleep 30; done
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    NUMBA_CACHE_DIR=/tmp/nbr_$g CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 HF_HOME=/apdcephfs_gy4/share_302533218/cedricnie/models/hf_cache \
      /opt/venv_eval/bin/python scripts_omnidance/exp/e7/e7_reward.py --gen $O --out $O/rewards_$((NI*8+g)).jsonl --shard $((NI*8+g)) --nshard $((NN*8)) "$@" > $O/rew_$((NI*8+g)).log 2>&1
    bash scripts_omnidance/exp/hold.sh start $g ) &
  sleep 6
done
wait; touch $O/.rew_done_$NI
