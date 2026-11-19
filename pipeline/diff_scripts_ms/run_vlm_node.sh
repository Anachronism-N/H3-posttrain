#!/bin/bash
# 用法: run_vlm_node.sh SHARD NSHARD   (占满本节点 8 卡, TP=8; 结束后归还占卡)
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; E=$R/logs_omnidance/exp/MS
bash $R/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
cd /tmp && /opt/venv_vllm/bin/python $R/scripts_omnidance/exp/ms/vlm_label.py --ids $E/ids_all.txt --out $E/vlm/labels_s$1.jsonl --shard $1 --nshard $2 --bs 256 ${3:+--model $3} > $E/logs/vlm_s$1.log 2>&1
rc=$?; bash $R/scripts_omnidance/exp/hold.sh idle; exit $rc
