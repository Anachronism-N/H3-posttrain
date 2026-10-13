#!/bin/bash
# train_ha2.sh (= train_ha.sh + env W: base weights dir, e.g. adapter-fused DiT for training) <out> <node_rank> <n_nodes> <master_ip> [extra args]: H-A bidirectional H3 Ref2VA LoRA SFT on /opt/h3cache/f0
# music = clean time-aligned target audio (OD_H3_CLEAN_AUDIO=1) + reference audio anchor; video-only loss.
O=$1; NR=$2; NN=$3; M=$4; shift 4
U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930; W=${W:-/dev/shm/h3/Ref2VA}
MP=$(ls $W/transformer/model*.safetensors | python3 -c "import sys,json;print(json.dumps([[l.strip() for l in sys.stdin]]))")
NG=${NGPU:-8}; ACC=${ACCUM:-1}; HOLD_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
CONF=/tmp/acc_h3_${NN}_${NG}_${ACC}.yaml
cat > "$CONF" <<YAML
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: $ACC
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero3_save_16bit_model: true
  zero_stage: 3
  deepspeed_multinode_launcher: standard
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: $NN
num_processes: $((NN*NG))
machine_rank: $NR
main_process_ip: $M
main_process_port: 29871
rdzv_backend: static
same_network: true
use_cpu: false
YAML
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop "$HOLD_GPUS"
cd $U
PATH=/opt/venv_h3/bin:$PATH OD_H3_TF=${TF:-0} OD_DS_ACTCKPT=1 OD_H3_CLEAN_AUDIO=1 PYTHONPATH=$U NCCL_IB_DISABLE=0 GLOO_SOCKET_IFNAME=${IFN:-} /opt/venv_h3/bin/accelerate launch --config_file "$CONF" \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ${CACHE:-/opt/h3cache/f0} \
  --data_file_keys "video,input_audio,references" --extra_inputs "input_audio,references" \
  --height ${H:-832} --width ${W_:-480} --num_frames 124 --dataset_repeat ${REPEAT:-4} \
  --model_paths "$MP" --processor_path "$W/processor/" \
  ${ADAPTER:+--preset_lora_path /apdcephfs_gy4/share_302533218/cedricnie/models/H3-TrainingAdapter/model_ref2va.safetensors --preset_lora_model dit} \
  --learning_rate ${LR:-1e-4} --num_epochs 1 --save_steps ${SAVE:-250} --gradient_accumulation_steps "$ACC" --remove_prefix_in_ckpt "pipe.dit." \
  --output_path $O --lora_base_model dit --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" --lora_rank ${RANK:-64} \
  --use_gradient_checkpointing --find_unused_parameters --task "sft:train" "$@"
rc=$?
bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start "$HOLD_GPUS"
echo "[ha] exit=$rc $(date)"
exit "$rc"
