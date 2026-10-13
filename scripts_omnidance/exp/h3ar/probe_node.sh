#!/bin/bash
# probe_node.sh: probe_fm.py on 8 GPUs (1 training item each), node-local cache /opt/h3cache/gt512
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
cd /opt
for g in 0 1 2 3 4 5 6 7; do
  DIFFSYNTH_SKIP_DOWNLOAD=True CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/probe_fm.py --root /opt/h3cache/gt512 \
    --list /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/DIAG_probe/list_g$g.txt --n 1 --sg_lat /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/SG512/lat --lora sg100=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_sg512_fm/step-100.safetensors sg300=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_sg512_fm/step-300.safetensors gt100=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm/step-100.safetensors gt300=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm/step-300.safetensors hi300=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_hinoise/step-300.safetensors lo300=/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3REF_gt512_fm_lonoise/step-300.safetensors --out /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/DIAG_probe/probe_g$g.json > /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/DIAG_probe/logs/g$g.log 2>&1 &
  sleep 30
done
wait
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5,6,7
echo "[probe] done $(date)"
