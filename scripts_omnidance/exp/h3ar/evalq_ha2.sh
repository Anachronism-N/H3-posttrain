#!/bin/bash
# evalq_ha2.sh (env MODE=bi|ar) <model dir> <tag> <steps "100 300 500 ..."> [ROWS]: as checkpoints appear, sample bidirectional demos with the ORIGINAL
# base (/dev/shm/h3/Ref2VA, adapter discarded) + new LoRA, official settings (shift 12/3, 50 steps, cfg 1), joint+aswap first ROWS val rows.
MD=$1; TAG=$2; STEPS=$3; ROWS=${ROWS:-6}
R=/apdcephfs_gy4/share_302533218/cedricnie; S=$R/diff/scripts_omnidance/exp/h3ar; E=$R/diff/logs_omnidance/exp/H3AR
for st in $STEPS; do
  f=$MD/step-$st.safetensors
  while [ ! -e $f ]; do sleep 120; done; sleep 180
  echo "[evalq] $TAG step-$st $(date)"
  if [ "${MODE:-bi}" = ar ]; then  # block AR, 20 steps per block
    bash $S/run_h3ar_node.sh 0 1 $E/val_${TAG}${st}_ar --mode ar --shift 12 --audio_shift 3 --steps 20 --lora $f --conds joint,aswap --rows $ROWS
  else
    bash $S/run_h3ar_node.sh 0 1 $E/val_${TAG}${st}_bi --mode bi --shift 12 --audio_shift 3 --steps 50 --lora $f --conds joint,aswap --rows $ROWS
  fi
done
echo "[evalq] done $(date)"
