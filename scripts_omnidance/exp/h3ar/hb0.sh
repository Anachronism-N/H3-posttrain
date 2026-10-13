#!/bin/bash
# hb0.sh <tf|mix|df>: early H-B on ONE node, AR teacher (teacher forcing, block-causal b6) LoRA r64 from the ORIGINAL H3
# (no H-A), to compare history modes on H3: tf = clean context; mix = 30% samples context sigma~U(0,0.2); df = per-block sigma~U(0,1).
# Waits for /opt/h3cache/.copied (cache copy) unless the cache was already there.
H=$1; S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar
O=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3B0_${H}_fromBase_b6_cur
while [ ! -f /opt/h3cache/.copied ]; do sleep 60; done
IP=$(hostname -I | awk '{print $1}')
echo "[hb0] $H on $IP start $(date)"
OD_H3_HIST=$H OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=/opt/h3cache/f0_cur TF=1 REPEAT=${REPEAT:-4} SAVE=100 LR=1e-4 bash $S/train_ha.sh $O 0 1 $IP
echo "[hb0] exit $(date)"
