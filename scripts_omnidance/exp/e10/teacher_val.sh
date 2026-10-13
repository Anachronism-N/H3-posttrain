#!/bin/bash
# teacher_val.sh <ckpt> <tag>: 50-step teacher on intended prompts (joint,tswap), then motion-caption metrics. one node.
CK=$1; TAG=$2; R=/apdcephfs_gy4/share_302533218/cedricnie/diff; D=$R/logs_omnidance/exp/E10/val/$TAG
mkdir -p $D; CONDS=${CONDS:-joint,tswap,aswap}; SRC=${SRC:-$R/logs_omnidance/exp/E7/val_full}; for c in ${CONDS//,/ }; do cp $SRC/full_$c.csv $D/; done
cd $R; python3 scripts_omnidance/exp/e10/make_shards_full.py $D 1 $CONDS
mkdir -p /dev/shm/od/ckpt/$TAG; [ -f /dev/shm/od/ckpt/$TAG/model.safetensors ] || cp $CK /dev/shm/od/ckpt/$TAG/model.safetensors
cp /dev/shm/od/ckpt/S4-final/args.json /dev/shm/od/ckpt/$TAG/args.json
CKPT=/dev/shm/od/ckpt/$TAG/model.safetensors E2_DIR=$D bash scripts_omnidance/exp/E2_run_node.sh 0 $TAG
O=$D/out/$TAG; touch $O/.done
bash scripts_omnidance/exp/e7/post_val_eval.sh $O $TAG
echo "[tval] $TAG done $(date)"
