#!/bin/bash
# wait W5a DONE -> causal CD from iter 3000 on this node
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; M=$R/models/train/W5a_arteacher_b4_lr1e-4
while [ ! -f $M/DONE ]; do sleep 120; done; sleep 60
mkdir -p /dev/shm/od/init; cp $M/iter_003000/gen_lora.pt /dev/shm/od/init/W5a_it3000.pt
bash $R/scripts_omnidance/exp/e6/launch_e6.sh train_causal_cd.py W5d_causalCD_from5a3000 0,1,2,3,4,5,6,7 --teacher_lora /dev/shm/od/init/W5a_it3000.pt --max_iters 2000 --save_every 250 --sample_every 250
