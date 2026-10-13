#!/bin/bash
# post_sample_eval.sh <dir with samples + .done> <tag>  (run on a node with VLM+venv_eval)
D=$1; T=$2; R=/apdcephfs_gy4/share_302533218/cedricnie/diff
cd $R
while [ ! -f $D/.done ]; do sleep 30; done
/opt/venv_od/bin/python scripts_omnidance/exp/eval/build_manifest.py $D $T $D/manifest.csv
/opt/venv_od/bin/python scripts_omnidance/exp/eval/beats_cache.py logs_omnidance/exp/beats.json $D/manifest.csv > /dev/null 2>&1
bash scripts_omnidance/exp/eval/run_eval_node.sh $D/manifest.csv $D/metrics 0,1,2,3,4,5,6,7 --vlm /dev/shm/od/vlm/Qwen3-VL-8B-Instruct
/opt/venv_od/bin/python scripts_omnidance/exp/eval/summarize.py $D/summary.md $D/metrics > /dev/null 2>&1
touch $D/.evaluated; echo "[post] $T evaluated $(date)"
