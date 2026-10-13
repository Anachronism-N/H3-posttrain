#!/bin/bash
# Copy base models + teacher ckpt to local /dev/shm/od (idempotent). Run on each node.
set -u
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
D=/dev/shm/od
mkdir -p $D/Wan-AI/Wan2.2-TI2V-5B $D/ckpt
t0=$(date +%s)
cp -rLu $R/models/Wan-AI/Wan2.2-TI2V-5B/. $D/Wan-AI/Wan2.2-TI2V-5B/
cp -ru $R/models/MERT-v1-330M $D/
cp -u $R/models/train/OSS_fullSFT_stage4_omni257k_b32_2ep/step-11000.safetensors $D/ckpt/S4-11000.safetensors
cp -u $R/models/train/OSS_fullSFT_stage4_omni257k_b32_2ep/args.json $D/ckpt/S4-11000.args.json
du -sh $D
echo "STAGE_DONE $(hostname -I | awk '{print $1}') $(( $(date +%s)-t0 ))s"
touch $D/.staged
