#!/bin/bash
# q2_20261010.sh <job> [node_rank]: CFG-preserving retrain (2026-10-10 night). Base DiT for TRAINING = de-distill adapter fused in
# NOTE: never edit while jobs run from this file (bash reads scripts lazily). Launch via: cp q2_20261010.sh /tmp/q2_<job>.sh && bash /tmp/q2_<job>.sh <job>
# (adapter discarded at inference: sample with the original /dev/shm/h3/Ref2VA + the new LoRA).
#   n78_ha2  : H-A2 bidirectional, DiffSynth DeCFG adapter fused, lr 5e-5, curated full12k (N7+N8, 16 GPUs)
#   n1_ha2lr : lr ablation (1e-4) of n78_ha2 on N1
#   n4_ha2os : H-A2 bidirectional, ostris ai-toolkit Ref2VA adapter v3 fused, lr 5e-5, curated f0_cur (N4, 8 GPUs) -> adapter comparison
J=$1; NR=${2:-0}
R=/apdcephfs_gy4/share_302533218/cedricnie; S=$R/diff/scripts_omnidance/exp/h3ar; M=$R/diff/models/train
IP=$(hostname -I | awk '{print $1}')
stage() {  # base fused locally by fuse_adapter.py (log /tmp/fuse_local.log); wait until done
  while ! grep -q "\[fuse\] done" /tmp/fuse_local.log 2>/dev/null; do sleep 30; done
  [ "$(ls $2/transformer/*.safetensors | wc -l)" = 13 ] || { echo "[q2] base incomplete $2"; exit 1; }; echo "[q2] base ok $2"
}
echo "[q2] $J nr=$NR on $IP start $(date)"
case $J in
  hyb_rich_g35|os1_rich_g35|ref_rich_g35|os1_rich_fm|ds_rich_fm4|ds_rich_fm|ds_rich_g35|hyb_rich_fm|ref_rich_fm|ref_rich_inverse35)
    (
      set -e
      [ -e "${CFG_READY:-/opt/h3cache/.cfg_rich_cur_ready}" ] || exit 1
      G=3.5; CFG_MODE=direct
      case "$J" in
        hyb_rich_g35) BASE=/opt/h3w/Hybrid25_bf16; OUT=H3HYB25_rich_g3p5_cur ;;
        os1_rich_g35) BASE=/opt/h3w/Ref2VA_decfgOstrisV1; OUT=H3OS1_rich_g3p5_cur ;;
        ref_rich_g35) BASE=/dev/shm/h3/Ref2VA; OUT=H3REF_rich_g3p5_cur ;;
        os1_rich_fm) BASE=/opt/h3w/Ref2VA_decfgOstrisV1; OUT=H3OS1_rich_fm_cur; G=1 ;;
        ds_rich_fm4) BASE=/opt/h3w/Ref2VA_decfgDS; OUT=H3DS_rich_fm_cur_4gpu; G=1 ;;
        ds_rich_fm) BASE=/opt/h3w/Ref2VA_decfgDS; OUT=H3DS_rich_fm_cur_8gpu; G=1 ;;
        ds_rich_g35) BASE=/opt/h3w/Ref2VA_decfgDS; OUT=H3DS_rich_g3p5_cur ;;
        hyb_rich_fm) BASE=/opt/h3w/Hybrid25_bf16; OUT=H3HYB25_rich_fm_cur; G=1 ;;
        ref_rich_fm) BASE=/dev/shm/h3/Ref2VA; OUT=H3REF_rich_fm_cur; G=1 ;;
        ref_rich_inverse35) BASE=/dev/shm/h3/Ref2VA; OUT=H3REF_rich_inverse3p5_cur; CFG_MODE=inverse ;;
      esac
      case "$BASE" in
        /dev/shm/h3/Ref2VA|/opt/h3w/Ref2VA_decfgDS) ;;
        *) [ -e "$BASE/merge_manifest.json" ] || exit 1 ;;
      esac
      [ "$(ls "$BASE"/transformer/model*.safetensors | wc -l)" = 13 ] || exit 1
      [ ! -e "$M/${CFG_OUT:-$OUT}" ] || exit 1
      OUT=${CFG_OUT:-$OUT}
      cp "$S/train_ha2.sh" "/tmp/train_cfg_${J}.sh"
      env -u OD_H3_MUSIC -u OD_H3_MUSIC_MODE -u OD_H3_MUSIC_LR -u OD_H3_DPO -u OD_H3_CD_TEACHER -u ADAPTER \
        W="$BASE" TF=0 OD_H3_DPO=${CFG_DPO:-0} OD_H3_POSI_DIR=/opt/h3cache/posi OD_H3_NEGA_DIR=${CFG_NEGA-/opt/h3cache/nega} \
        CACHE=${CFG_CACHE:-/opt/h3cache/cfg_rich_cur} REPEAT=${CFG_REPEAT:-2} SAVE=100 LR=${CFG_LR:-5e-5} RANK=${CFG_RANK:-64} \
        bash "/tmp/train_cfg_${J}.sh" "$M/$OUT" 0 1 "$IP" \
          --training_cfg_scale "$G" --training_cfg_mode "$CFG_MODE" --seed 42 --max_train_steps 300 --enable_csv_log ${CFG_EXTRA:-}
    )
    rc=$?; [ "$rc" = 0 ] || exit "$rc" ;;
  n78_ha2)
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS CACHE=/opt/h3cache/full12k REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3A2_decfgDS_lr5e-5_full12k $NR 2 28.216.18.144 ;;
  n4_ha2os)
    stage $R/models/H3_Ref2VA_decfgOstrisV3 /opt/h3w/Ref2VA_decfgOstrisV3
    W=/opt/h3w/Ref2VA_decfgOstrisV3 CACHE=/opt/h3cache/f0_cur REPEAT=2 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3A2_decfgOstrisV3_lr5e-5_cur 0 1 $IP ;;
  n1_ha2lr)   # lr ablation: same DiffSynth adapter, lr 1e-4, N1 curated f0_cur+f1_cur (5.6k)
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 LR=1e-4 \
      bash $S/train_ha2.sh $M/H3A2_decfgDS_lr1e-4_n1cur 0 1 $IP ;;
  n1_hb2pre)  # early AR teacher (Causal Forcing stage 1, TF70/DF30 mix) from H-A2 step-200, fused base, lr 5e-5, N1 curated 5.6k.
              # purpose: AR signal ~10 h before H-A2 finishes; the main H-B2 restarts from the final H-A2 on N7+N8.
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    T=$M/H3A2_decfgDS_lr5e-5_full12k/step-200.safetensors
    while [ ! -e $T ]; do sleep 120; done; sleep 180
    W=/opt/h3w/Ref2VA_decfgDS TF=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3B2pre_mix_fromHA2s200_n1cur 0 1 $IP --lora_checkpoint $T ;;
  n4_m1a)     # M1a (notes/20): H-A2 recipe + zero-init MERT injector on target-audio tokens (h3_music.py); compare with H-A2 at equal steps
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_MAP=/opt/h3cache/music_map_f0cur.json CACHE=/opt/h3cache/f0_cur REPEAT=3 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3M1a_mert_decfgDS_lr5e-5_cur 0 1 $IP ;;
  hb2base)    # AR teacher (CF stage 1, TF70/DF30 mix) directly from the ORIGINAL H3 with the CFG fix (fused DS adapter); env CACHE OUT
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS TF=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=$CACHE REPEAT=${REPEAT:-1} SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/$OUT 0 1 $IP ;;
  hb2pre)     # AR teacher from an early H-A2 checkpoint (env FROM=<step>, CACHE, OUT); waits for the checkpoint
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    T=$M/H3A2_decfgDS_lr5e-5_full12k/step-$FROM.safetensors
    while [ ! -e $T ]; do sleep 60; done; sleep 120
    W=/opt/h3w/Ref2VA_decfgDS TF=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=$CACHE REPEAT=${REPEAT:-1} SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/$OUT 0 1 $IP --lora_checkpoint $T ;;
  dpo)        # W1b music-contrastive Diffusion-DPO + SFT, bidirectional, fresh LoRA on the CFG-fixed base (h3_dpo.py); env CACHE OUT
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_DPO=1 OD_H3_DPO_BETA=${BETA:-1000} CACHE=$CACHE REPEAT=${REPEAT:-1} SAVE=${SAVE:-100} LR=5e-5 \
      bash $S/train_ha2.sh $M/$OUT 0 1 $IP ;;
  n3_hc)      # Causal Forcing++ stage 2 (H-C, 4-step consistency distillation) on N3 after the OOS run, teacher = H-B2base step-$FROM
    while [ ! -e $R/diff/logs_omnidance/exp/OOS_H3/.done ]; do sleep 120; done
    while ! grep -q "ship\] done" /tmp/ship_n3_done.log 2>/dev/null; do sleep 60; done
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    T=$M/H3B2base_mix_decfgDS_n1cur/step-${FROM:-300}.safetensors
    while [ ! -e $T ]; do sleep 60; done; sleep 120
    W=/opt/h3w/Ref2VA_decfgDS CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 bash $S/train_cd.sh $M/H3C2_cd_fromHB2base${FROM:-300}_n1cur 0 1 $IP $T ;;
  hc)         # H-C2 (CF++ stage 2, 4-step causal consistency distillation), teacher = H-B2base step-$FROM; waits for /tmp/ship_done
              # and the teacher, then stops any eval worker on this node and trains (8 GPUs)
    while [ ! -e /tmp/ship_done ]; do sleep 60; done
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    T=$M/H3B2base_mix_decfgDS_n1cur/step-${FROM:-200}.safetensors
    while [ ! -e $T ]; do sleep 60; done; sleep 120
    bash $S/kill_prefix.sh "bash $S/eval_sched.sh"; bash $S/stop_sampler_node.sh; sleep 30
    W=/opt/h3w/Ref2VA_decfgDS CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 bash $S/train_cd.sh $M/H3C2_cd_fromHB2base${FROM:-200}_n1cur 0 1 $IP $T ;;
  evalw)      # become an eval worker after the OOS run finishes
    while [ ! -e $R/diff/logs_omnidance/exp/OOS_H3/.done ]; do sleep 120; done
    bash $S/eval_sched.sh ;;
  m2)         # M2 (notes/20): multi-encoder MoE music fusion (MERT lo/hi, beat, H3 audio-VAE, CLAP), H-A2 recipe, N2 f1_cur
    while ! grep -q "m2prep\] done" /tmp/m2prep.log 2>/dev/null; do sleep 60; done
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_MODE=moe OD_H3_MUSIC_MAP=/opt/h3cache/music_idmap_f1cur.json \
      CACHE=/opt/h3cache/f1_cur REPEAT=1 SAVE=100 LR=5e-5 bash $S/train_ha2.sh $M/H3M2_moe5_decfgDS_lr5e-5_f1cur 0 1 $IP ;;
  n78_hb2)    # mainline H-B2 AR teacher (CF stage 1, TF70/DF30) from FINAL H-A2 step-750, full12k, N7+N8
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    T=$M/H3A2_decfgDS_lr5e-5_full12k/step-750.safetensors
    W=/opt/h3w/Ref2VA_decfgDS TF=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 CACHE=/opt/h3cache/full12k REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3B2_mix_fromHA2s750_full12k $NR 2 28.216.18.144 --lora_checkpoint $T ;;
  n1_hb3cfg)  # AR-fix: H-B2base recipe (TF70/DF30, n1_cur, from ORIGINAL H3) but on the ORIGINAL distilled base (no de-distill
              # adapter) + DiffSynth inverse-CFG guidance loss (--training_cfg_scale ${G:-4}); uncond inputs from nega_cache.py sidecars.
              # 1) let the running eval job finish, 2) build sidecars on 6 GPUs, 3) train. Compare with H-B2base at equal steps.
    bash $S/kill_prefix.sh "bash /tmp/eval_sched_run.sh" >/dev/null
    while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*sample_h3_ar'; do sleep 60; done; sleep 20
    ND=/opt/h3cache/nega
    for g in 0 1 2 3 4 5; do
      ( bash $R/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; cd /opt && CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python $S/nega_cache.py \
          --list /opt/h3cache/nega_list_n1cur.tsv --out $ND --shard $g --nshard 6 > /tmp/nega_$g.log 2>&1 ) &
    done; wait
    nn=$(find $ND -name '*.nega.pt' | wc -l); echo "[q2] nega sidecars $nn / $(wc -l < /opt/h3cache/nega_list_n1cur.tsv)"
    [ $nn -ge $(wc -l < /opt/h3cache/nega_list_n1cur.tsv) ] || { echo "[q2] nega incomplete"; exit 1; }
    TF=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 OD_H3_NEGA_DIR=$ND CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3B3_mix_cfgloss${G:-4}_n1cur 0 1 $IP --training_cfg_scale ${G:-4} ;;
  m2b)        # M2 rerun: same as m2 but the music MoE gets its own lr (OD_H3_MUSIC_LR, default 1e-3); 10-11 m2 router stayed uniform at lr 5e-5
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_MODE=moe OD_H3_MUSIC_LR=${MLR:-1e-3} OD_H3_MUSIC_MAP=/opt/h3cache/music_idmap_f1cur.json \
      CACHE=/opt/h3cache/f1_cur REPEAT=1 SAVE=100 LR=5e-5 bash $S/train_ha2.sh $M/H3M2b_moe5_mlr${MLR:-1e-3}_decfgDS_f1cur 0 1 $IP ;;
  n78_ha3)    # H-A3 bidirectional (user 10-11: bidirectional first): ORIGINAL distilled base (no de-distill adapter) + inverse-CFG
              # guidance loss --training_cfg_scale ${G:-4}; full12k, lr 5e-5, 750 steps; compare with H-A2 (adapter recipe) at equal steps.
              # step 1: this node builds its half of the uncond sidecars (6 GPUs), pushes them to the peer, waits for the peer's half.
    ND=/opt/h3cache/nega; PEER=$([ "$NR" = 0 ] && echo 28.216.18.136 || echo 28.216.18.144); rm -f /tmp/nega_peer_done
    for g in 0 1 2 3 4 5; do
      ( bash $R/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; cd /opt && CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python $S/nega_cache.py --idmode \
          --list /opt/h3cache/nega_list_full12k.tsv --out $ND --shard $((NR*6+g)) --nshard 12 > /tmp/nega_$g.log 2>&1 ) &
    done; wait
    cd $ND && tar -cf - full12k | ssh $PEER "mkdir -p $ND && tar -C $ND -xf - && touch /tmp/nega_peer_done"; echo "[q2] pushed half to $PEER exit $?"
    while [ ! -e /tmp/nega_peer_done ]; do sleep 30; done
    nn=$(ls $ND/full12k | wc -l); echo "[q2] nega sidecars $nn / 11992"; [ $nn -ge 11992 ] || { echo "[q2] nega incomplete"; exit 1; }
    OD_H3_NEGA_DIR=$ND CACHE=/opt/h3cache/full12k REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3A3_cfgloss${G:-4}_full12k $NR 2 28.216.18.144 --training_cfg_scale ${G:-4} ;;
  m1ab)       # M1a rerun: MERT injector with its own lr (OD_H3_MUSIC_LR, default 1e-3); 10-11 M1a at lr 5e-5 barely moved (fc2 |w| 1e-3)
              # waits for the eval sample job still running on this node
    while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*sample_h3_ar'; do sleep 60; done; sleep 30
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_LR=${MLR:-1e-3} OD_H3_MUSIC_MAP=/opt/h3cache/music_map_f0cur.json CACHE=/opt/h3cache/f0_cur REPEAT=3 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3M1ab_mert_mlr${MLR:-1e-3}_decfgDS_cur 0 1 $IP ;;
  n1_hb4)     # H-B3 recipe (original base + guidance loss g=4, TF70/DF30, n1_cur) but VAE-group-aligned blocks: [0,5),[5,10),...
              # (H3 VAE: 5 latents = 17 frames (1,4,4,4,4); the 1+6 layout cut every group). Test whether TF smearing comes from misaligned blocks.
    TF=1 OD_H3_FIRST=5 OD_H3_BLOCK=5 OD_H3_MERGE_TAIL=1 OD_H3_HIST=mix OD_H3_DF_P=0.3 OD_H3_DF_MAX=0.2 OD_H3_NEGA_DIR=/opt/h3cache/nega CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3B4m_align5tail_cfgloss4_n1cur 0 1 $IP --training_cfg_scale 4 ;;
  n78_ha4)    # H-A4 (10-12): H-A2 recipe (DiffSynth DeCFG adapter fused base, lr 5e-5, full12k, 750 steps) but trained on the RICH VLM
              # prompt (E3 style) instead of the step-by-step motion caption: text sidecars from N1 (text_recache.py, OD_H3_POSI_DIR).
              # waits for H-A3 to finish on this node and for N1's sidecars, pulls them, then trains.
    while ps -eo args | grep -q '^/opt/venv_h3/bin/python.*model_training/train.py'; do sleep 60; done
    while ! ssh 28.216.19.69 "grep -q 'posi_node\] done 11992' /tmp/posi_node.log" 2>/dev/null; do sleep 60; done
    mkdir -p /opt/h3cache/posi; ssh 28.216.19.69 "cd /opt/h3cache/posi && tar -cf - full12k" | tar -C /opt/h3cache/posi -xf -
    nn=$(ls /opt/h3cache/posi/full12k | wc -l); echo "[q2] posi sidecars $nn"; [ $nn -ge 11992 ] || { echo "[q2] posi incomplete"; exit 1; }
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_POSI_DIR=/opt/h3cache/posi CACHE=/opt/h3cache/full12k REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3A4_richprompt_decfgDS_full12k $NR 2 28.216.18.144 ;;
  ha5hr|ha6dup)  # 10-12: H-A4 recipe (DeCFG adapter base, rich prompt, lr 5e-5) on the new full12k variants, single node (data local):
              # ha5hr = native 704x1280 + RIFE (hands); ha6dup = 480x832, no synthesized frames (RIFE ghosting). Compare with H-A4 at equal steps.
    T=$([ $J = ha5hr ] && echo hr || echo dup); HH=$([ $J = ha5hr ] && echo 1280 || echo 832); WW=$([ $J = ha5hr ] && echo 704 || echo 480)
    while ! grep -q "posi_after_prep\] $T done" /tmp/posi_$T.log 2>/dev/null; do sleep 120; done
    n=$(find /opt/h3cache/posi/$T -name '*.posi.pt' | wc -l); echo "[q2] $T sidecars $n"; [ $n -ge 11900 ] || { echo "[q2] sidecars incomplete"; exit 1; }
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS H=$HH W_=$WW OD_H3_POSI_DIR=/opt/h3cache/posi CACHE=/opt/h3cache/$T REPEAT=1 SAVE=100 LR=5e-5 \
      bash $S/train_ha2.sh $M/H3A$([ $J = ha5hr ] && echo 5hr || echo 6dup)_richprompt_decfgDS_full12k 0 1 $IP ;;
  m1ab_rich)  # 10-12: M1ab (MERT injector, music lr 1e-3) re-done on the RICH prompt (old music runs used the list caption -> invalid)
    bash $S/kill_prefix.sh "bash /tmp/eval_sched_run" "bash -c NOWAIT" >/dev/null
    while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*sample_h3_ar'; do sleep 60; done; sleep 30
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_LR=1e-3 OD_H3_MUSIC_MAP=/opt/h3cache/music_map_n1cur.json OD_H3_POSI_DIR=/opt/h3cache/posi \
      CACHE=/opt/h3cache/n1_cur REPEAT=1 SAVE=100 LR=5e-5 bash $S/train_ha2.sh $M/H3M1ab_rich_mert_mlr1e-3_n1cur 0 1 $IP ;;
  m2b_rich)   # 10-12: M2b (5-expert MoE, music lr 1e-3) re-done on the RICH prompt; first builds rich-prompt sidecars for N2's f1 cache
    bash $S/kill_prefix.sh "bash /tmp/eval_sched_run" "bash -c NOWAIT" >/dev/null
    while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*sample_h3_ar'; do sleep 60; done; sleep 30
    cd /opt && /opt/venv_h3/bin/python $S/cache_meta_list.py /opt/h3cache/f1 /dev/shm/h3data/metadata_cur_n7_p0.json /opt/h3cache/meta_f1.json /opt/h3cache/posi_list_f1.tsv
    for g in 0 1 2 3 4 5 6 7; do
      ( bash $R/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python $S/text_recache.py --list /opt/h3cache/posi_list_f1.tsv \
          --meta /opt/h3cache/meta_f1.json --prompts $R/diff/logs_omnidance/exp/H3AR/all_rich_prompts.json --out /opt/h3cache/posi --shard $g --nshard 8 > /tmp/posi2_$g.log 2>&1 ) &
      sleep 20
    done; wait
    n=$(find /opt/h3cache/posi/f1 -name '*.posi.pt' | wc -l); echo "[q2] f1 sidecars $n"; [ $n -ge 3500 ] || { echo "[q2] sidecars incomplete"; exit 1; }
    stage $R/models/H3_Ref2VA_decfgDS /opt/h3w/Ref2VA_decfgDS
    W=/opt/h3w/Ref2VA_decfgDS OD_H3_MUSIC=1 OD_H3_MUSIC_MODE=moe OD_H3_MUSIC_LR=1e-3 OD_H3_MUSIC_MAP=/opt/h3cache/music_idmap_f1cur.json OD_H3_POSI_DIR=/opt/h3cache/posi \
      CACHE=/opt/h3cache/f1_cur REPEAT=1 SAVE=100 LR=5e-5       bash $S/train_ha2.sh $M/H3M2b_rich_moe5_mlr1e-3_f1cur 0 1 $IP ;;
  *) echo "[q2] unknown job $J"; exit 2 ;;
esac
echo "[q2] $J done $(date)"
