#!/bin/bash
# 在本机 8×H20 上用 OmniDance 数据跑 warm2brch 两阶段训练
# 数据适配:通过 symlink 把 metadata_train/val.csv 挂到 warm2brch base yaml 期望的
#   $DATA_DIR/inbox/7-10_3000_6600_lowQ1100.csv 和 $DATA_DIR/val_mace16good/_run_global.csv
set -e

REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
source /apdcephfs_gy2/share_303214315/cedricnie/miniconda3/etc/profile.d/conda.sh
conda activate wan
cd "$REPO"

# 数据目录结构自检(缺失则自动补 symlink)
D=$REPO/data/omnidance
[ -f "$D/metadata_train.csv" ] || { echo "FATAL: $D/metadata_train.csv 不存在"; exit 1; }
mkdir -p "$D/inbox" "$D/val_mace16good"
ln -sf "$D/metadata_train.csv" "$D/inbox/7-10_3000_6600_lowQ1100.csv"
ln -sf "$D/metadata_val.csv" "$D/val_mace16good/_run_global.csv"

# 幂等:已在跑则不重启。只统计非 D 状态的进程(D 状态 = 页锁死锁僵尸,杀不死
# 且不占关键资源,不能作为"在跑"依据;启动命令行勿含此模式字符串,防 pgrep 自匹配)
RUNNING=$(ps aux | awk '/model_training\/train\.py --config/ && $8 !~ /^D/ {n++} END {print n+0}')
if [ "$RUNNING" -gt 0 ]; then
  echo "[warm2brch] training already running ($RUNNING procs), skip. $(date)"
  exit 0
fi

# 日志写本地盘:cephfs 上的日志文件曾因页锁死锁卡死全部 worker(master 被 kill 时
# 未释放 cephfs log 文件的页锁 → 8 worker 永久卡在 __lock_page_killable)
LOG=/root/warm2brch.log

# 注意:沙箱设置 BASH_ENV=/root/.bashrc,新 bash 会 source 它导致 PATH 被重置、
# conda 环境被切回 torch-base。必须用 env -u 剥离 BASH_ENV/ENV 再 setsid。
env -u BASH_ENV -u ENV \
DIFFSYNTH_MODEL_BASE_PATH=$REPO/models/Wan-AI \
DIFFSYNTH_SKIP_DOWNLOAD=true \
WANDB_MODE=offline \
WARM2BRCH_NO_BASHRC=1 \
WARM2BRCH_EXTRA_CONFIG=$REPO/examples/wanvideo/model_training/full/warm2brch/local_infra_override.yaml \
REPRO_S1_CKPT=/root/models_local/step-5000.safetensors \
DATA_DIR=$D \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
REPRO_S1_STEP=5000 \
setsid nohup bash examples/wanvideo/model_training/full/warm2brch/run_warm2brch_stages.sh \
  > "$LOG" 2>&1 < /dev/null &

disown
echo "[warm2brch] launched, log=$LOG  $(date)"
