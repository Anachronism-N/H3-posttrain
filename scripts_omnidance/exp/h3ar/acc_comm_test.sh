#!/bin/bash
# acc_comm_test.sh <node_rank> <n_nodes> <master> <rdzv>
NR=$1; NN=$2; M=$3; RZ=$4
cat > /tmp/acc_test.yaml <<YAML
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  zero3_init_flag: false
  zero_stage: 3
  deepspeed_multinode_launcher: standard
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: $NN
num_processes: $((NN*8))
machine_rank: $NR
main_process_ip: $M
main_process_port: 29881
rdzv_backend: $RZ
same_network: true
use_cpu: false
YAML
PATH=/opt/venv_h3/bin:$PATH timeout 300 /opt/venv_h3/bin/accelerate launch --config_file /tmp/acc_test.yaml /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/acc_comm_test.py
