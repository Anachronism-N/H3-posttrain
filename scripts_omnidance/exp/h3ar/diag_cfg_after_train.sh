#!/bin/bash
# diag_cfg_after_train.sh <train log to wait for> <out dir name> [sample_h3_ar.py args]
# 10-13 diagnosis after "every FM fine-tune blurs, even on self-generated targets": real CFG at inference
# (optionally keeping the de-distill base). Waits for this node's training to exit, then samples the fixed 3 cases (joint+aswap) on 6 GPUs.
L=$1; OUT=$2; shift 2
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; E=$R/logs_omnidance/exp/H3AR
while ! grep -q '\[ha\] exit=' "$L" 2>/dev/null; do sleep 60; done
sleep 60
[ -e "$E/$OUT/.done_0" ] && { echo "[diag] $OUT already done"; exit 0; }
GPUS="0 1 2 3 4 5" bash $R/scripts_omnidance/exp/h3ar/run_h3ar_node.sh 0 1 "$E/$OUT" --mode bi --steps 50 --prompt_mode orig \
  --shift 12 --audio_shift 3 --conds joint,aswap --rows 3 "$@"
echo "[diag] $OUT exit=$? $(date)"
