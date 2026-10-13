#!/bin/bash
# train_full.sh <out> <node_rank> <n_nodes> <master_ip> [extra args]: FULL-parameter H3 DiT SFT (ZeRO-3, optimizer offload to CPU by default)
# env: CACHE (default /opt/h3cache/full12k) REPEAT LR(1e-5) SAVE(200) OFFLOAD(cpu|none) TF/OD_H3_HIST as in train_ha.sh
# music = clean time-aligned target audio (OD_H3_CLEAN_AUDIO=1) + reference audio anchor; video-only loss.
O=$1; NR=$2; NN=$3; M=$4; shift 4
U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930; W=/dev/shm/h3/Ref2VA
MP=$(ls $W/transformer/model*.safetensors | python3 -c "import sys,json;print(json.dumps([[l.strip() for l in sys.stdin]]))")
cat > /tmp/acc_h3full_$NN.yaml <<YAML
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: ${OFFLOAD:-cpu}
  offload_param_device: none
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
  deepspeed_multinode_launcher: standard
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: $NN
num_processes: $((NN*8))
machine_rank: $NR
main_process_ip: $M
main_process_port: 29881
rdzv_backend: static
same_network: true
use_cpu: false
YAML
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop 0,1,2,3,4,5,6,7
cd $U
PATH=/opt/venv_h3/bin:$PATH OD_H3_TF=${TF:-0} OD_DS_ACTCKPT=1 OD_H3_CLEAN_AUDIO=1 PYTHONPATH=$U NCCL_IB_DISABLE=0 GLOO_SOCKET_IFNAME=${IFN:-} /opt/venv_h3/bin/accelerate launch --config_file /tmp/acc_h3full_$NN.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ${CACHE:-/opt/h3cache/full12k} \
  --data_file_keys "video,input_audio,references" --extra_inputs "input_audio,references" \
  --height 832 --width 480 --num_frames 124 --dataset_repeat ${REPEAT:-4} \
  --model_paths "$MP" --processor_path "$W/processor/" \
  --learning_rate ${LR:-1e-5} --num_epochs 1 --save_steps ${SAVE:-200} --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $O --trainable_models dit \
  --use_gradient_checkpointing --find_unused_parameters --task "sft:train" "$@"
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start 0,1,2,3,4,5,6,7
echo "[full] exit $(date)"
