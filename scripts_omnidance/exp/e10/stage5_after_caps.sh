#!/bin/bash
# wait for all 6 caption nodes, build metadata, launch stage5 full SFT on this node (8 GPUs)
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff; E=$REPO/logs_omnidance/exp/E10/train_caps
while [ $(ls $E/.done_* 2>/dev/null | wc -l) -lt 6 ]; do sleep 120; done
cd $REPO; export PATH=/opt/venv_od/bin:$PATH PYTHONPATH=$REPO
python3 scripts_omnidance/exp/e10/build_stage5_meta.py
OUT=$REPO/models/train/OSS_fullSFT_stage5_motioncap
if ls $OUT/step-*.safetensors >/dev/null 2>&1; then echo "refusing to overwrite"; exit 1; fi
while pgrep -f "[c]aption_motion.py" >/dev/null; do sleep 30; done
bash scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
mkdir -p $REPO/logs_omnidance/exp/E10; CFG=$REPO/examples/wanvideo/model_training/full/reproduce
DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true WANDB_MODE=offline OD_ZERO_OPT=1 OD_GRAD_BUCKET_VIEW=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes 8 --num_machines 1 examples/wanvideo/model_training/train.py \
  --config $CFG/omnidance_fullSFT_base.yaml --config $CFG/fullSFT_stage4_omni2ep.yaml --config $CFG/fullSFT_stage5_motioncap.yaml \
  --find_unused_parameters --trainable_models dit,dit.audio_injector.injector_global_mod_layers \
  --mert_encoder_path /dev/shm/od/MERT-v1-330M --tokenizer_path /dev/shm/od/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl \
  --train_checkpoint_path /dev/shm/od/ckpt/S4-final/step-final.safetensors > $REPO/logs_omnidance/exp/E10/stage5_train.log 2>&1
echo "[stage5] train exit $? $(date)"
