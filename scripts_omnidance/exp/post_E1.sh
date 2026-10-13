#!/bin/bash
# Runs on N1. Waits for E1 to finish, then: E2 protocol on S4-final (8 GPUs, 704p) -> metrics on N1 -> review set.
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
S=$R/models/train/OSS_fullSFT_stage4_resume_from11k
D=$R/logs_omnidance/exp/E2_S4final
cd $R
while pgrep -f "[m]odel_training/train.py" > /dev/null; do sleep 300; done
FINAL=$(ls $S/step-*.safetensors | sort -t- -k2 -n | tail -1)
echo "[post_E1] $(date) E1 finished, final=$FINAL"
mkdir -p /dev/shm/od/ckpt/S4-final $D
cp $S/args.json /dev/shm/od/ckpt/S4-final/
cp $FINAL /dev/shm/od/ckpt/S4-final/step-final.safetensors
E2_DIR=$D python3 scripts_omnidance/exp/E2_make_shards.py 1
CKPT=/dev/shm/od/ckpt/S4-final/step-final.safetensors E2_DIR=$D bash scripts_omnidance/exp/E2_run_node.sh 0 S4-final
python3 scripts_omnidance/exp/eval/build_manifest.py $D/out/S4-final S4-final $D/manifest.csv
# eval venv only exists on N7/N8 -> run metrics on N8 (shares GPUs with auto-eval; run sequentially after it is idle)
ssh 28.216.18.136 "cd $R; while pgrep -f '[s]ample_student.py|[e]val_metrics.py' >/dev/null; do sleep 120; done; bash scripts_omnidance/exp/eval/run_eval_node.sh $D/manifest.csv $D/metrics_TA 0,1,2,3,4,5,6,7 --vlm /dev/shm/od/vlm/Qwen3-VL-8B-Instruct"
/opt/venv_od/bin/python scripts_omnidance/exp/eval/summarize.py $D/summary_TA.md $D/metrics_TA $R/logs_omnidance/exp/E2/metrics_TA
E=$R/logs_omnidance/exp
/opt/venv_od/bin/python scripts_omnidance/exp/make_review.py --out $E/review/S4final_vs_S4-11000 --cases 6 --gt \
  --col "S4-11000_joint=$E/E2/out/S4-11000:joint" --col "S4final_joint=$D/out/S4-final:joint" \
  --col "S4final_textonly=$D/out/S4-final:text_only" --col "S4final_audioonly=$D/out/S4-final:audio_only" \
  --col "S4final_tswap=$D/out/S4-final:tswap" --col "S4final_aswap=$D/out/S4-final:aswap"
bash scripts_omnidance/exp/hold.sh idle
echo "[post_E1] $(date) all done"
