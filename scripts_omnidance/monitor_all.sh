#!/bin/bash
# 双机训练 + HF 上传守护的周期监控(单次快照输出)
echo "========== $(date '+%F %T') =========="

echo "--- [.25] warm2brch ---"
ALIVE25=$(ps aux | awk '/[t]rain.py --config/ && $8 !~ /^D/ {n++} END {print n+0}')
ZOMBIE25=$(ps aux | awk '/[t]rain.py --config/ && $8 ~ /^D/ {n++} END {print n+0}')
echo "procs: alive=$ALIVE25 zombie=$ZOMBIE25"
tail -c 400 /root/warm2brch.log 2>/dev/null | grep -a "step=" | tail -1
grep -a "step=" /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/warm2brch_fullSFT_stage1_aud5e5_xattn5e6_warm1500/train.log 2>/dev/null | tail -1 | grep -oE "step=[0-9]+ .*loss=[0-9.]+" | head -c 120
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -2
ERR25=$(grep -ac "Cannot detect\|Traceback" /root/warm2brch.log 2>/dev/null)
echo "log errors: $ERR25"

echo "--- [.248] s3_resume ---"
timeout 30 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no 28.7.187.248 "
ALIVE=\$(ps aux | awk '/[t]rain.py --config/ && \$8 !~ /^D/ {n++} END {print n+0}')
echo \"procs: alive=\$ALIVE\"
grep -a 'step=' /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/OSS_fullSFT_stage3_resume2_from8k/train.log 2>/dev/null | tail -1 | grep -oE 'step=[0-9]+ .*loss=[0-9.]+' | head -c 120
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -2
grep -ac 'Cannot detect\|Traceback' /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/s3_resume_248.log 2>/dev/null | xargs echo 'log errors:'
" 2>&1

echo "--- [HF watchdog] ---"
WD=$(ps aux | grep -c "[u]pload_ckpts_hf")
echo "watchdog procs: $WD"
tail -1 /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_upload.log 2>/dev/null
