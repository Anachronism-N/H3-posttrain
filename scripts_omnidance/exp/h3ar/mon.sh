#!/bin/bash
# mon.sh: one-line-per-job status of the H3 campaign
E=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR; M=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train
echo "[$(date '+%m-%d %H:%M')]"
for d in $(ls -d $E/val_* | xargs -n1 basename); do echo "eval $d: $(find $E/$d -name '*.mp4' | wc -l) mp4 $(ls -a $E/$d | grep -c '^.done')done"; done
for h in 28.216.19.69 28.216.19.225 28.216.17.70; do echo "enc $h: $(ssh $h "tail -c 2000 /tmp/enc_cur.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]|\[enc\] done.*' | tail -1")"; done
echo "H-A  : $(ssh 28.216.18.144 "tail -c 300 /tmp/h3a_node0.log | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]|\[ha\] exit.*' | tail -1") ckpts=$(ls $M/H3A_lora64_cleanmusic | grep -c safetensors)"
echo "H-B  : $(ssh 28.216.18.144 "tail -c 300 /tmp/chain_hb_0.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1; grep -o '\[hb\].*' /tmp/chain_hb_0.log | tail -1") ckpts=$(ls $M/H3B_mix_arteacher_b6_cur 2>/dev/null | grep -c safetensors)"
echo "H-B0 : $(ssh 28.216.19.143 "tail -c 300 /tmp/hb0_mix.log | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]|exit.*' | tail -1") ckpts=$(ls $M/H3B0_mix_fromBase_b6_cur | grep -c safetensors)"
for h in 28.216.19.69 28.216.17.70 28.216.19.213 28.216.19.143 28.216.19.137 28.216.19.225 28.216.18.144 28.216.18.136; do printf "%s:%s " ${h##*.} "$(ssh $h "dmesg -T 2>/dev/null | grep -c 'Killed process'")"; done; echo "(oom-kill counts)"
