#!/bin/bash
# queue_20261010.sh <job>: wait for the previous job on this node to finish, then launch the next experiment.
# Launch on the target node: ssh <host> "setsid nohup bash queue_20261010.sh <job> > /tmp/q_<job>.log 2>&1 < /dev/null &"
J=$1
R=/apdcephfs_gy4/share_302533218/cedricnie; S=$R/diff/scripts_omnidance/exp/h3ar; E=$R/diff/logs_omnidance/exp/H3AR; M=$R/diff/models/train
IP=$(hostname -I | awk '{print $1}')
waitf() { while [ ! -e "$1" ]; do sleep 60; done; }
waitlog() { while ! grep -q "$2" "$1" 2>/dev/null; do sleep 60; done; }
latest() { ls -t $1/step-*.safetensors | head -1; }
echo "[q] $J on $IP start $(date)"
case $J in
  n2_music)   # M1b: original H3, bidirectional, BPM + beat times written into the prompt (vs val_zs_bi_motion)
    waitf $E/val_HBmix100_ar/.done_0
    bash $S/run_h3ar_node.sh 0 1 $E/val_zs_bi_musicprompt --mode bi --prompt_mode music --conds joint,aswap --rows 24 ;;
  n1_hb0b)    # quality ablation: AR teacher from original H3 with lr 2e-5 (vs 1e-4) on all curated clips of N1
    waitf $E/val_HBmix300_ar/.done_0
    mkdir -p /opt/h3cache/n1_cur; ln -sfn /opt/h3cache/f0_cur /opt/h3cache/n1_cur/a; ln -sfn /opt/h3cache/f1_cur /opt/h3cache/n1_cur/b
    OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=/opt/h3cache/n1_cur TF=1 REPEAT=1 SAVE=100 LR=2e-5 \
      bash $S/train_ha.sh $M/H3B0b_mix_fromBase_lr2e-5_cur 0 1 $IP ;;
  n35_eval)   # $2 = node index (0 for N3, 1 for N5): H-B final, then H-B0 final, block AR sampling
    waitf $E/val_HB0mix700_ar/.done_$2
    true
    bash $S/run_h3ar_node.sh $2 2 $E/val_HBmix500_ar --mode ar --lora $M/H3B_mix_arteacher_b6_cur/step-500.safetensors --conds joint,aswap --rows 24
    waitf $M/H3B0_mix_fromBase_b6_cur/step-1046.safetensors; sleep 300
    bash $S/run_h3ar_node.sh $2 2 $E/val_HB0mix1046_ar --mode ar --lora $M/H3B0_mix_fromBase_b6_cur/step-1046.safetensors --conds joint,aswap --rows 24 ;;
  n78_hc)     # $2 = node rank: H-C main run (CF++ stage 2) from the final H-B teacher, curated 12k set on both nodes
    waitlog /tmp/chain_hb_$2.log "\[ha\] exit"; sleep 240
    T=$(latest $M/H3B_mix_arteacher_b6_cur); echo "[q] teacher $T"
    CACHE=/opt/h3cache/full12k REPEAT=1 SAVE=100 bash $S/train_cd.sh $M/H3C_cd_fromHB_full12k $2 2 28.216.18.144 $T ;;
  n4_hc0)     # H-C branch from the final H-B0 teacher (is H-A needed?)
    waitlog /tmp/hb0_mix.log "\[hb0\] exit"; sleep 240
    T=$(latest $M/H3B0_mix_fromBase_b6_cur); echo "[q] teacher $T"
    CACHE=/opt/h3cache/f0_cur REPEAT=2 SAVE=100 bash $S/train_cd.sh $M/H3C0_cd_fromHB0_cur 0 1 $IP $T ;;
esac
echo "[q] $J done $(date)"
