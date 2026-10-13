#!/bin/bash
# 在 28.7.187.248 上运行:S3 完成段续训(原始 stage3 step ~8000 → 12000)
# 前置:
#   - gy4 已挂载(taiji_client mount ...)
#   - wan 环境已解包到 /dockerdata/wan_env(conda-pack)
#   - 音频缓存已 rsync 到 /dockerdata/audio_cache
#   - vLLM 已停(GPU 已释放)
set -e

REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
ENVBIN=/dockerdata/wan_env/bin
# 本地盘拷贝(ceph-fuse 慢且曾页锁死锁)
RESUME_CKPT=/dockerdata/models_local/step-3000.safetensors

[ -x "$ENVBIN/python" ] || { echo "FATAL: $ENVBIN/python 不存在(先解包 wan_env)"; exit 1; }
[ -f "$RESUME_CKPT" ] || { echo "FATAL: $RESUME_CKPT 不存在(等模型拷贝完成)"; exit 1; }
grep -q COPY_DONE /dockerdata/copy_models.log 2>/dev/null || { echo "FATAL: 模型拷贝未完成(见 /dockerdata/copy_models.log)"; exit 1; }
grep -q DIRS_DONE /dockerdata/copy_models.log 2>/dev/null || { echo "FATAL: tokenizer/MERT 目录拷贝未完成"; exit 1; }
[ -d /dockerdata/audio_cache_248 ] || { echo "FATAL: /dockerdata/audio_cache_248 不存在"; exit 1; }

export PATH="$ENVBIN:$PATH"
export PYTHONPATH="$REPO"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
cd "$REPO"

# 幂等(排除 D 状态僵尸:页锁死锁进程杀不死且不代表在跑)
RUNNING=$(ps aux | awk '/model_training\/train\.py --config/ && $8 !~ /^D/ {n++} END {print n+0}')
if [ "$RUNNING" -gt 0 ]; then
  echo "[resume248] training already running ($RUNNING procs), skip. $(date)"
  exit 0
fi

LOG=$REPO/logs_omnidance/s3_resume_248.log
setsid nohup accelerate launch --num_processes 8 --num_machines 1 \
  examples/wanvideo/model_training/train.py \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/omnidance_fullSFT_base.yaml \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/fullSFT_stage3.yaml \
  --config $REPO/examples/wanvideo/model_training/full/reproduce/fullSFT_stage3_resume2.yaml \
  --find_unused_parameters \
  --trainable_models dit,dit.audio_injector.injector_global_mod_layers \
  --train_checkpoint_path "$RESUME_CKPT" \
  > "$LOG" 2>&1 < /dev/null &

disown
echo "[resume248] launched, log=$LOG  $(date)"
