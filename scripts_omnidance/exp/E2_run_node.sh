#!/bin/bash
# Run E2 shards for one node. Usage: E2_run_node.sh <node_idx> [ckpt_dir_tag]
# Each GPU processes its shard csvs sequentially; holder restarted on GPU when done.
NODE=$1
TAG=${2:-S4-11000}
CKPT=${CKPT:-/dev/shm/od/ckpt/S4-11000/step-11000.safetensors}
REPO=/apdcephfs_gy4/share_302533218/cedricnie/diff
E2=${E2_DIR:-$REPO/logs_omnidance/exp/E2}
export PATH=/opt/venv_od/bin:$PATH
cd $REPO
run_gpu() {
  g=$1
  bash $REPO/scripts_omnidance/exp/hold.sh stop $g
  for csvf in $E2/shards/n${NODE}_g${g}_*.csv; do
    [ -f "$csvf" ] || continue
    c=$(basename $csvf .csv); c=${c#n${NODE}_g${g}_}
    AM=real; TM=original
    [ "$c" = "audio_only" ] && TM=no_text
    [ "$c" = "text_only" ] && AM=none
    mkdir -p $E2/out/$TAG/$c $E2/logs
    echo "[E2] node=$NODE gpu=$g cond=$c start $(date)"
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$REPO WANDB_MODE=offline \
    DIFFSYNTH_MODEL_BASE_PATH=/dev/shm/od/Wan-AI DIFFSYNTH_SKIP_DOWNLOAD=true \
    python3 inference_bundle/toolkit/run_batch.py \
      --csv_path $csvf \
      --inference_script examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py \
      --output_dir $E2/out/$TAG/$c --base_label $TAG \
      --model_path /dev/shm/od/Wan-AI/Wan2.2-TI2V-5B --mert_path /dev/shm/od/MERT-v1-330M \
      --ckpt_path $CKPT --ckpt_type onlyCKPT --no-use_lora \
      --cfg_mode text_first --audio_mode $AM --text_mode $TM \
      --num_frames ${NF:-81} --num_inference_steps 50 --cfg_scale ${CFG:-2.0} --audio_scale ${AS:-2.0} \
      --seed ${SEED:-42} --fps 16 --height ${H:-1280} --width ${W:-704} --device cuda --save_with_audio \
      >> $E2/logs/n${NODE}_g${g}.log 2>&1
    echo "[E2] node=$NODE gpu=$g cond=$c rc=$? end $(date)"
  done
  bash $REPO/scripts_omnidance/exp/hold.sh start $g
}
for g in 0 1 2 3 4 5 6 7; do run_gpu $g & sleep 20; done
wait
echo "[E2] node $NODE all done $(date)"
