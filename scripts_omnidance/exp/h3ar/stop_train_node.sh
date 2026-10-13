#!/bin/bash
# stop_train_node.sh <script-substring>: kill bash wrappers running <substring> (argv[1] of bash is a script path) + all H3 train.py/accelerate python; restart holders
P=$1
for pid in $(ps -eo pid=); do
  [ "$pid" = "$$" ] && continue
  a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
  case "$a" in "bash /"*"$P"*) kill $pid ;; esac
done
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/killpat.sh model_training/train.py accelerate
sleep 20
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5,6,7 >/dev/null 2>&1
echo "gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
