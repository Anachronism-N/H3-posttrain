#!/bin/bash
# posi_node.sh: rich-prompt text sidecars for full12k on 8 GPUs (text encoder resident) -> /opt/h3cache/posi/full12k
for g in 0 1 2 3 4 5 6 7; do
  ( bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; cd /opt && CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/text_recache.py \
      --list /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/full12k_posi_list.tsv --meta /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/full12k_cache_meta.json --prompts /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/full12k_rich_prompts.json --out /opt/h3cache/posi --shard $g --nshard 8 > /tmp/posi_$g.log 2>&1 ) &
  sleep 20
done; wait; echo "[posi_node] done $(ls /opt/h3cache/posi/full12k | wc -l) $(date)"
