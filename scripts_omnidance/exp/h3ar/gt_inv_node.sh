#!/bin/bash
# gt_inv_node.sh : run gt_invert.py on all 8 GPUs of this node (staggered 60 s) over SG512/sg512.txt with the local
# /opt/h3cache/gt512 cache; several nodes share H3AR/INV512 and claim items by mkdir. Run a frozen copy in /tmp.
S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp; E=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR
O=$E/INV512; mkdir -p $O/logs; H=$(hostname -I | awk '{print $1}')
bash $S/hold.sh stop 0,1,2,3,4,5,6,7
cd /opt
pids=()
for g in 0 1 2 3 4 5 6 7; do
  DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930 CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 \
    /opt/venv_h3/bin/python $S/h3ar/gt_invert.py --root /opt/h3cache/gt512 --list $E/SG512/sg512.txt --out $O --check $([ $g = 0 ] && echo 1 || echo 0) > $O/logs/${H}_g$g.log 2>&1 &
  pids+=("$!")
  sleep 60
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
bash $S/hold.sh start 0,1,2,3,4,5,6,7
echo "[inv_node] $H done rc=$rc $(date) items=$(ls $O/inv | wc -l)"
exit $rc
