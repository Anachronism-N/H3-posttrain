#!/bin/bash
# ship_full12k.sh <src_cache> <metadata> <tag>: stage selected samples on this node and stream them to N7 and N8 /opt/h3cache/full12k
L=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR; S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar
ST=/opt/h3cache/stage_$3; rm -rf $ST
python3 $S/stage_sel.py $1 $2 $L/full12k_ids.txt $3 $ST
ME=$(hostname -I | awk '{print $1}')
cd $ST
for D in 28.216.18.144 28.216.18.136; do
  if [ "$D" = "$ME" ]; then mkdir -p /opt/h3cache/full12k && cp -L ./*.pth /opt/h3cache/full12k/
  else nice -n 10 tar -chf - . | ssh $D "mkdir -p /opt/h3cache/full12k && tar -C /opt/h3cache/full12k -xf -"; fi
  echo "[ship] $3 -> $D exit $? $(date)"
done
