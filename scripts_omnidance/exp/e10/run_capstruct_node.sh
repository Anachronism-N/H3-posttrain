#!/bin/bash
# run_capstruct_node.sh <csv> <out_dir> <node_idx> <n_nodes> : caption_struct.py on 8 GPUs of this node (frozen copy in /tmp)
M=$1; O=$2; NI=$3; NN=$4; R=/apdcephfs_gy4/share_302533218/cedricnie/diff
mkdir -p $O; cd $R
pids=()
for g in 0 1 2 3 4 5 6 7; do
  ( bash scripts_omnidance/exp/hold.sh stop $g
    NUMBA_CACHE_DIR=/tmp/nbc_$g CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 /opt/venv_eval/bin/python scripts_omnidance/exp/e10/caption_struct.py --csv $M --out $O/cap_$((NI*8+g)).jsonl --shard $((NI*8+g)) --nshard $((NN*8)) > $O/cap_$((NI*8+g)).log 2>&1
    rc=$?; bash scripts_omnidance/exp/hold.sh start $g; exit $rc ) &
  pids+=("$!"); sleep 5
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc = 0 ] && touch $O/.done_$NI; exit $rc
