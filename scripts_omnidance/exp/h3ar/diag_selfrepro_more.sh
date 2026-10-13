#!/bin/bash
# diag_selfrepro_more.sh <cache root>: self-reproduction test (same cached condition + same seed as SG target) for more LoRAs, 6 GPUs
ROOT=$1; D=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/DIAG_selfrepro; U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5
cd /opt; g=0
for x in "sg100 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_sg512_fm/step-100.safetensors" "gt100 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm/step-100.safetensors" "hi300 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_hinoise/step-300.safetensors" \
         "lo300 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_lonoise/step-300.safetensors" "lr300 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_lr1e-5/step-300.safetensors" "r16300 /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_r16/step-300.safetensors"; do
  set -- $x
  DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH=$U CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/sg_gen.py --root $ROOT --list $D/list3.txt \
    --decode_all --out $D/$1 --lora $2 > $D/logs/$1.log 2>&1 &
  g=$((g+1)); sleep 60
done
wait
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5
echo "[selfrepro_more] done $(date)"
