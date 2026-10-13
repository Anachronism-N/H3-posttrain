#!/bin/bash
# enc_cur_node.sh <metadata.json> <out_cache>: encode curated clips on all 8 GPUs of this node (stops/restarts GPU holders)
M=$1; O=$2; H=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh
bash $H stop 0,1,2,3,4,5,6,7; mkdir -p /opt/h3cache
echo "[enc] start $(date) $M"
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/data_process_node.sh $M $O 0,1,2,3,4,5,6,7
bash $H start 0,1,2,3,4,5,6,7
echo "[enc] done $(find $O -name '*.pth' | wc -l) files $(date)"
