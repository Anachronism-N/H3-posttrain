#!/bin/bash
# data_process_node.sh <metadata.json> <out_cache_dir> <gpus csv>   (env H / W_: resolution, default 832x480)
M=$1; O=$2; G=$3; N=$(echo $G | tr ',' '\n' | wc -l)
U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930; W=/dev/shm/h3/Ref2VA
TE=$(ls $W/text_encoder/model*.safetensors | python3 -c "import sys,json;print(json.dumps([l.strip() for l in sys.stdin]))")
MP=$(python3 -c "import json;print(json.dumps([json.loads('$TE'), '$W/video_vae/source/model.safetensors', '$W/audio_vae/model.safetensors']))")
cd $U
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$U /opt/venv_h3/bin/accelerate launch --num_processes $N --num_machines 1 --main_process_port $((29600 + RANDOM % 300)) \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path $(dirname $M) --dataset_metadata_path $M \
  --data_file_keys "video,input_audio,references" --extra_inputs "input_audio,references" \
  --height ${H:-832} --width ${W_:-480} --num_frames 124 --dataset_repeat 1 \
  --model_paths "$MP" --processor_path "$W/processor/" \
  --learning_rate 1e-5 --num_epochs 1 --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $O --trainable_models "dit" --use_gradient_checkpointing --task "sft:data_process"
