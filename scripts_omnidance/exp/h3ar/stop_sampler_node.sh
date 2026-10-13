#!/bin/bash
# stop_sampler_node.sh: kill run_h3ar_node.sh wrappers and sample_h3_ar.py python on this node, restart holders
for pid in $(ps -eo pid=); do
  [ "$pid" = "$$" ] && continue
  a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
  case "$a" in "bash /"*run_h3ar_node.sh*) kill $pid ;; esac
done
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/killpat.sh sample_h3_ar.py
sleep 15; bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5,6,7 >/dev/null 2>&1
