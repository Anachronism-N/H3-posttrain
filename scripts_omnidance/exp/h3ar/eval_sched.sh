#!/bin/bash
# eval_sched.sh: shared eval worker (run on any eval node; several workers coordinate via atomic mkdir claims on cephfs).
# Jobs: lines "<model_dir_name> <tag> <mode> <rows> <steps...>" in eval_jobs.txt (re-read every loop, edit freely).
# Sampling = ORIGINAL base (/dev/shm/h3/Ref2VA) + LoRA (adapter discarded), official shift 12/3; bi 50 steps, ar 20 steps/block.
# rows 0 = all 47 val cases (joint + aswap). BASE lines = original H3 without LoRA.
# modes: bi | ar | cm4 | arW (keep de-distill adapter at inference) | arcfg3 (real CFG 3) | bifal (fal realism trigger word)
# NOTE: never edit this file while a worker runs it (bash reads lazily). Run a frozen copy: cp eval_sched.sh /tmp/eval_sched_run.sh
R=/apdcephfs_gy4/share_302533218/cedricnie; S=$R/diff/scripts_omnidance/exp/h3ar; E=$R/diff/logs_omnidance/exp/H3AR; M=$R/diff/models/train
J=${JOBS:-$S/eval_jobs.txt}; C=$E/.claims; mkdir -p $C
echo "[sched] worker $(hostname -I | awk '{print $1}') start $(date)"
[ -n "$NOWAIT" ] || while ps -eo args | grep -q '^/opt/venv_h3/bin/python /.*sample_h3_ar'; do sleep 30; done   # let a previous worker's job finish
# second worker on the same node: NOWAIT=1 GPUS="5 6 7" (RAM: ~60 GB/sampler steady, ~122 GB while loading; node cgroup 1071 GB)
while true; do
  ran=0
  while read -r md tag mode rows steps; do
    [ -z "$md" ] || [ "${md:0:1}" = "#" ] && continue
    for st in $steps; do
      if [ "$md" = BASE ]; then  # original H3 without LoRA: "BASE <out dir name> <mode> <rows> <claim suffix>"
        out=$E/$tag; LA=""; cl=${tag}_r${rows}_$st
      else
        f=$M/$md/step-$st.safetensors; out=$E/val_${tag}${st}_$mode; LA="--lora $f"; cl=${tag}${st}_$mode
        [ -e $f ] || continue
        [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 180 ] || continue
        [ "$rows" = 0 ] && cl=${cl}_all
      fi
      if [ "$mode" = biorighyb ] && [ ! -e /opt/h3w/Hybrid25_bf16/merge_manifest.json ]; then
        continue
      fi
      mkdir $C/$cl 2>/dev/null || continue          # claimed by another worker (or done)
      echo "[sched] $tag step-$st $mode rows=$rows $(date)"
      case $mode in
        ar) A="--mode ar --steps 20" ;;
        cm4) A="--mode ar --cm_steps 4" ;;
        arW) A="--mode ar --steps 20 --w /opt/h3w/Ref2VA_decfgDS" ;;
        arcfg3) A="--mode ar --steps 20 --cfg 3" ;;
        bifal) A="--mode bi --steps 50 --prompt_prefix r34l1sm" ;;
        ar5) A="--mode ar --steps 20 --block 5" ;;   # VAE-group-aligned blocks (OD_H3_FIRST=5 set below)
        bihr) A="--mode bi --steps 50 --height 1344 --width 768" ;;   # official resolution (hands: 480p has too few tokens)
        bihrorig) A="--mode bi --steps 50 --height 1280 --width 704 --prompt_mode orig" ;;   # hr-trained model at its training res
        biorigcfg3) A="--mode bi --steps 50 --prompt_mode orig --cfg 3" ;;   # diag: residual CFG-distillation loss?
        biospeech) A="--mode bi --steps 50 --prompt_mode orig --audio_override /apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_probe/speech_librispeech.wav" ;;
        bionoise) A="--mode bi --steps 50 --prompt_mode orig --audio_override /apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_probe/white_noise.wav" ;;
        biosilence) A="--mode bi --steps 50 --prompt_mode orig --audio_override /apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_probe/silence.wav" ;;
        biorighyb) A="--mode bi --steps 50 --prompt_mode orig --w /opt/h3w/Hybrid25_bf16" ;;
        biorig) A="--mode bi --steps 50 --prompt_mode orig" ;;   # rich VLM prompt (what E3 used)
        *) A="--mode bi --steps 50" ;;
      esac
      FIRST=1; [ $mode = ar5 ] && FIRST=5
      OD_H3_FIRST=$FIRST OD_H3_MERGE_TAIL=$([ $mode = ar5 ] && echo 1 || echo 0) bash $S/run_h3ar_node.sh 0 1 $out $A --shift 12 --audio_shift 3 $LA --conds $([ "${mode#bio}" != "$mode" ] && [ "$mode" != biorig ] && [ "$mode" != biorigcfg3 ] && [ "$mode" != biorighyb ] && echo joint || echo joint,aswap) --rows $rows < /dev/null
      ran=1; break 2
    done
  done < $J
  [ $ran = 0 ] && sleep 120
done
