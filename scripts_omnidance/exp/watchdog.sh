#!/bin/bash
# Every 10 min: on every node, put holders on GPUs with no compute process; log monitor snapshot.
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
while true; do
  for h in $(awk '{print $1}' /etc/taiji/hostfile); do
    timeout 60 ssh -o BatchMode=yes $h "bash $R/scripts_omnidance/exp/hold.sh idle" 2>&1 | grep started | sed "s/^/[$h] /"
  done
  bash $R/scripts_omnidance/exp/monitor.sh > $R/logs_omnidance/exp/monitor_latest.txt 2>&1
  cat $R/logs_omnidance/exp/monitor_latest.txt >> $R/logs_omnidance/exp/monitor_history.txt
  sleep 600
done
