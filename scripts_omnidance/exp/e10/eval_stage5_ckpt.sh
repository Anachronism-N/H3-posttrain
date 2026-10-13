#!/bin/bash
# eval_stage5_ckpt.sh <step> <pose_node_idx>: wait ckpt + this node's pose job, then teacher_val on it
ST=$1; PI=$2; R=/apdcephfs_gy4/share_302533218/cedricnie/diff; O=$R/models/train/OSS_fullSFT_stage5_motioncap
while [ ! -f $O/step-$ST.safetensors ]; do sleep 120; done
sleep 120  # let the write finish
while [ ! -f $R/logs_omnidance/exp/G2/.pose_done_$PI ]; do sleep 60; done
bash $R/scripts_omnidance/exp/e10/teacher_val.sh $O/step-$ST.safetensors S5_step$ST
