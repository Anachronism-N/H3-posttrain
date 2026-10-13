#!/bin/bash
# M2 features on N2: beat phase on CPU (32 procs, venv_od: librosa beat_track segfaults in venv_h3), CLAP on 7 GPUs, then mark done
O=/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/m2_feats; L=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/m2_list.txt; cd /opt
for i in $(seq 0 31); do OMP_NUM_THREADS=1 /opt/venv_od/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/m2_feats.py --phase beat --list $L --out $O --shard $i --nshard 32 > /tmp/m2b_$i.log 2>&1 & done
for g in 0 1 2 3 4 5 6; do CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/m2_feats.py --phase clap --list $L --out $O --shard $g --nshard 7 > /tmp/m2c_$g.log 2>&1 & done
wait
nb=$(ls $O/beat | wc -l); nc=$(ls $O/clap | wc -l); n=$(wc -l < $L)
echo "[m2feat] beat $nb clap $nc of $n $(date)" >> /tmp/m2prep.log
[ $nb -ge $((n-20)) ] && [ $nc -ge $((n-20)) ] && echo "[m2prep] done $(date)" >> /tmp/m2prep.log
