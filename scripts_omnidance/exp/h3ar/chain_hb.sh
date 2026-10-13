#!/bin/bash
# chain_hb.sh <node_rank>: wait for H-A exit on this node, then H-B (TF70/DF30 mixed AR teacher, curated cache) on N7+N8 from the last H-A LoRA
NR=$1; S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar
A=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3A_lora64_cleanmusic
B=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3B_mix_arteacher_b6_cur
while ! grep -q "\[ha\] exit" /tmp/h3a_node$NR.log 2>/dev/null; do sleep 120; done
sleep 180
L=$(ls -t $A/*.safetensors | head -1); echo "[hb] init from $L"
OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=/opt/h3cache/f0_cur TF=1 REPEAT=4 SAVE=100 LR=5e-5 bash $S/train_ha.sh $B $NR 2 28.216.18.144 --lora_checkpoint $L
