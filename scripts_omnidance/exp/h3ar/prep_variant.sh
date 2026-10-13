#!/bin/bash
# prep_variant.sh <tag> <W> <H> <interp rife|dup>: full12k curated clips -> converted clips (/opt/h3raw_<tag>) -> H3 cache (/opt/h3cache/<tag>), all 8 GPUs.
# 10-11: hand blur/interpenetration diagnosis -> hr (704x1280 native, RIFE) and dup (480x832, no synthesized frames).
T=$1; W=$2; H=$3; I=$4
R=/apdcephfs_gy4/share_302533218/cedricnie; S=$R/diff/scripts_omnidance/exp/h3ar; L=$R/diff/logs_omnidance/exp/H3AR
D=/opt/h3raw_$T; O=/opt/h3cache/$T; mkdir -p $D $O
python3 - <<PY
import csv; csv.field_size_limit(10**9)
ids=set(open("$L/full12k_ids.txt").read().split()); rs=[r for r in csv.DictReader(open("$L/h3_train36k.csv")) if r["sample_id"] in ids]
w=csv.DictWriter(open("$D/list.csv","w"),fieldnames=["sample_id","video","audio","caption"]); w.writeheader(); [w.writerow(r) for r in rs]; print("[pv] clips",len(rs))
PY
for g in 0 1 2 3 4 5 6 7; do
  ( bash $R/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; cd $R
    CUDA_VISIBLE_DEVICES=$g /opt/venv_eval/bin/python $S/prep_h3_data.py --csv $D/list.csv --out $D --shard $g --nshard 8 --W $W --H $H --interp $I > /tmp/pv_${T}_$g.log 2>&1 ) &
done; wait
echo "[pv] converted $(ls $D/video | grep -vc tmp) $(date)"
mkdir -p $D/av; bash $S/mux_av.sh $D; echo "[pv] muxed $(ls $D/av | grep -vc tmp) $(date)"
python3 $S/build_h3_meta.py $D f0
ln -sfn $D/metadata_f0.json $D/metadata.json
H=$H W_=$W bash $S/data_process_node.sh $D/metadata.json $O 0,1,2,3,4,5,6,7 > /tmp/pv_${T}_dp.log 2>&1
echo "[pv] cached $(find $O -name '*.pth' | wc -l) $(date)"
