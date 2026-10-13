#!/bin/bash
# Runs on N8. Loop: for every E5 run checkpoint iter_{1000,2000,...} (EMA lora if present) not yet evaluated:
#   sample OD-Val-47 x 5 conds with the 4-step student (8 GPUs) -> metrics (DWPose/TA/ID) -> summary.
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
EV=$R/logs_omnidance/exp/E5/eval
mkdir -p $EV
cd $R
while true; do
  for run in $R/models/train/E5*dmd*; do
    name=$(basename $run)
    H=832; W=480; NF=49
    [[ $name == *704* ]] && { H=1280; W=704; NF=81; }
    for ck in $(ls -d $run/iter_* 2>/dev/null | sort); do
      it=$((10#$(basename $ck | cut -d_ -f2)))
      [ $((it % 1000)) -eq 0 ] || [ -f $run/DONE -a "$ck" = "$(ls -d $run/iter_* | sort | tail -1)" ] || continue
      tag=${name}_it${it}
      out=$EV/$tag
      [ -f $out/.evaluated ] && continue
      lora=$ck/gen_lora_ema.pt; [ -f $lora ] || lora=$ck/gen_lora.pt
      [ -f $lora ] || continue
      echo "[auto] $(date) sampling $tag"
      [ -f $out/.done ] || bash scripts_omnidance/exp/dmd/run_student_node.sh $out 0,1,2,3,4,5,6,7 --lora $lora --height $H --width $W --num_frames $NF
      python3 scripts_omnidance/exp/eval/build_manifest.py $out $tag $out/manifest.csv
      bash scripts_omnidance/exp/eval/run_eval_node.sh $out/manifest.csv $out/metrics 0,1,2,3,4,5,6,7 --vlm /dev/shm/od/vlm/Qwen3-VL-8B-Instruct
      /opt/venv_od/bin/python scripts_omnidance/exp/eval/summarize.py $out/summary.md $out/metrics > /dev/null 2>&1
      touch $out/.evaluated
      echo "[auto] $(date) done $tag"
    done
  done
  sleep 300
done
