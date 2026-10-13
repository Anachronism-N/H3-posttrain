#!/bin/bash
O=/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/m2_feats; L=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/m2_list.txt; cd /opt
for i in $(seq 0 31); do OMP_NUM_THREADS=1 /opt/venv_od/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/m2_feats.py --phase beat --list $L --out $O --shard $i --nshard 32 > /tmp/m2b_$i.log 2>&1 & done
wait
while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*m2_feats.py --phase clap'; do sleep 30; done
nb=$(ls $O/beat | wc -l); nc=$(ls $O/clap | wc -l); n=$(wc -l < $L)
echo "[m2feat2] beat $nb clap $nc of $n $(date)" >> /tmp/m2prep.log
[ $nb -ge $((n-20)) ] && [ $nc -ge $((n-20)) ] && echo "[m2prep] done $(date)" >> /tmp/m2prep.log
