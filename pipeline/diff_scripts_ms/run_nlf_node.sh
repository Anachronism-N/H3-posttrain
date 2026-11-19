#!/bin/bash
# 在本节点指定 GPU 上跑 NLF 3D 提取。用法: run_nlf_node.sh NODE_IDX "0 1 2 ..." [NSHARD]
# 分片号 = NODE_IDX*8 + gpu；可断点续跑（已存在的 npz 跳过）
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
NI=$1; GPUS=$2; NS=${3:-64}
OUT=$R/logs_omnidance/exp/MS/nlf
mkdir -p $OUT /dev/shm/od $R/logs_omnidance/exp/MS/logs
if [ ! -f /dev/shm/od/nlf.pt ]; then cp /apdcephfs_gy4/share_302533218/cedricnie/models/nlf/nlf_l_multi_0.3.2.torchscript /dev/shm/od/nlf.pt.tmp.$$ && mv -f /dev/shm/od/nlf.pt.tmp.$$ /dev/shm/od/nlf.pt; fi
bash $R/scripts_omnidance/exp/hold.sh stop "${GPUS// /,}"
for g in $GPUS; do
  SH=$((NI*8+g))
  CUDA_VISIBLE_DEVICES=$g NLF_PATH=/dev/shm/od/nlf.pt /opt/venv_eval/bin/python $R/scripts_omnidance/exp/ms/nlf_extract.py \
    --ids $R/logs_omnidance/exp/MS/ids_all.txt --shard $SH --nshard $NS --out $OUT \
    > $R/logs_omnidance/exp/MS/logs/nlf_$SH.log 2>&1 &
done
wait
