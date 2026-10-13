#!/bin/bash
# dp_node_all.sh <n_clips>: mux av, build metadata (f0 ref, first N), data_process on 8 GPUs -> /opt/h3cache/f0
N=$1; S=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar
bash $S/mux_av.sh /dev/shm/h3data
/opt/venv_h3/bin/python $S/build_h3_meta.py /dev/shm/h3data f0 $N
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
mkdir -p /opt/h3cache
bash $S/data_process_node.sh /dev/shm/h3data/metadata_f0_$N.json /opt/h3cache/f0 0,1,2,3,4,5,6,7
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5,6,7
echo "[dp] done $(find /opt/h3cache/f0 -name '*.pth' | wc -l) files $(date)"
