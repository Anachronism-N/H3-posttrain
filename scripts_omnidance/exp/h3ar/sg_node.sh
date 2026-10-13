#!/bin/bash
# sg_node.sh <cache root> : run sg_gen.py on all 8 GPUs of this node (staggered 60 s), wait for all, restore holders.
# Workers on several nodes share SG512/sg512.txt and claim items by mkdir. Run a frozen copy: cp sg_node.sh /tmp/sg_node_run.sh
ROOT=$1
S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp; O=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/SG512
mkdir -p $O/logs; H=$(hostname -I | awk '{print $1}')
bash $S/hold.sh stop 0,1,2,3,4,5,6,7
cd /opt
for g in 0 1 2 3 4 5 6 7; do
  DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930 CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 \
    /opt/venv_h3/bin/python $S/h3ar/sg_gen.py --root $ROOT --list $O/sg512.txt --out $O > $O/logs/${H}_g$g.log 2>&1 &
  sleep 60
done
wait
bash $S/hold.sh start 0,1,2,3,4,5,6,7
echo "[sg_node] $H done $(date) items=$(ls $O/lat | wc -l)"
