#!/bin/bash
# posi_after_prep.sh <tag> <W> <H>: after prep_variant.sh finished encoding /opt/h3cache/<tag>, build rich-prompt text sidecars on this node
T=$1; W=$2; H=$3; D=/opt/h3raw_$T
while ! grep -q 'pv\] cached' /tmp/pv_$T.log 2>/dev/null; do sleep 120; done
cd /opt && /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/cache_meta_list.py /opt/h3cache/$T $D/metadata.json /opt/h3cache/meta_$T.json /opt/h3cache/posi_list_$T.tsv
for g in 0 1 2 3 4 5 6 7; do
  ( bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/text_recache.py --list /opt/h3cache/posi_list_$T.tsv \
      --meta /opt/h3cache/meta_$T.json --prompts /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/full12k_rich_prompts.json --out /opt/h3cache/posi --ref_dirs $D/ref --height $H --width $W --shard $g --nshard 8 > /tmp/posi_${T}_$g.log 2>&1;
    bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start $g >/dev/null 2>&1 ) &
  sleep 20
done; wait; echo "[posi_after_prep] $T done $(find /opt/h3cache/posi/$T -name '*.posi.pt' | wc -l) $(date)"
