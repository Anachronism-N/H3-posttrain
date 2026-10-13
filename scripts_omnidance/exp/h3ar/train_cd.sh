#!/bin/bash
# train_cd.sh <out> <node_rank> <n_nodes> <master_ip> <teacher_lora>: Causal Forcing++ stage 2 (causal consistency distillation)
# student LoRA (r64) initialised from the AR teacher; frozen teacher + EMA adapters (h3_cd.py). env: CACHE REPEAT LR(2e-5) SAVE(100) OD_H3_CD_EMA
O=$1; NR=$2; NN=$3; M=$4; T=$5
OD_H3_CD=1 OD_H3_CD_TEACHER=$T TF=1 LR=${LR:-2e-5} SAVE=${SAVE:-100} REPEAT=${REPEAT:-4} CACHE=${CACHE:-/opt/h3cache/f0_cur} \
  bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/train_ha2.sh $O $NR $NN $M --lora_checkpoint $T
echo "[cd] exit $(date)"
