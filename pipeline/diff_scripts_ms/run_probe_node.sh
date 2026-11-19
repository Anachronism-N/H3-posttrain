#!/bin/bash
# 冻结 H3 读新格式 caption 探针。用法: run_probe_node.sh BODY_JSON OUT_ROOT "key1 key2 ..." "gpus" [NSPLIT]
# 每个 key 用 NSPLIT 张卡（3 case × joint+aswap 共 6 条视频按 shard 切分），进程间错开 2 分钟加载（CPU 内存）
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930
BJ=$1; OR=$2; KEYS=($3); G=($4); NS=${5:-2}
B="--mode bi --steps 50 --height 1280 --width 704 --shift 12 --audio_shift 3 --prompt_mode orig --rows 3 --conds joint,aswap"
i=0; pids=()
for k in "${KEYS[@]}"; do
  for s in $(seq 0 $((NS-1))); do
    g=${G[$i]}; i=$((i+1)); O=$OR/val_MS_$k; mkdir -p $O/logs
    ( bash $R/scripts_omnidance/exp/hold.sh stop $g
      cd /opt && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$U /opt/venv_h3/bin/python $R/scripts_omnidance/exp/h3ar/sample_h3_ar.py \
        --out $O --shard $s --nshard $NS $B --body_json $BJ --body_key $k > $O/logs/s${s}_g$g.log 2>&1
      rc=$?; bash $R/scripts_omnidance/exp/hold.sh start $g; exit $rc ) &
    pids+=($!); sleep 2m
  done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
