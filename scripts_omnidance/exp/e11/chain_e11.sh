#!/bin/bash
# chain_e11.sh <run_name>: wait DONE -> val sampling (intended prompts; joint,tswap,aswap seed42 + joint seed43) -> metrics (TA, ID, kps) -> G2 discriminator
N=$1; R=/apdcephfs_gy4/share_302533218/cedricnie/diff; cd $R
M=$R/models/train/$N; V=$R/logs_omnidance/exp/E11/val; mkdir -p $V
while [ ! -f $M/DONE ]; do sleep 120; done
sleep 60
L=/dev/shm/od/init/${N}.pt; cp $M/iter_002500/gen_lora.pt $L
bash scripts_omnidance/exp/e11/run_e11_node.sh $V/$N 0,1,2,3,4,5,6,7 --lora $L --conds joint,tswap,aswap
bash scripts_omnidance/exp/e7/post_val_eval.sh $V/$N $N
bash scripts_omnidance/exp/e11/run_e11_node.sh $V/${N}_s43 0,1,2,3,4,5,6,7 --lora $L --conds joint --seed 43
bash scripts_omnidance/exp/e7/post_val_eval.sh $V/${N}_s43 ${N}_s43
bash scripts_omnidance/exp/g2/disc_dir.sh $V/$N > $V/$N.disc.md 2>&1
echo "[chain_e11] $N done $(date)"
