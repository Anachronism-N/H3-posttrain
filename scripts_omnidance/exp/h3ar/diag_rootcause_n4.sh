#!/bin/bash
# diag_rootcause_n4.sh (10-14): why does every FM fine-tune of H3 blur? inference-only probes on one 8-GPU node, cfg 1.
#  g0-g2  self-reproduction: cached training condition + the SAME seed as the SG512 target; base / SG512-300 / GT512-300
#         (base must reproduce the SG target; a model trained on that target should too - if it blurs, the objective/pipeline damages it)
#  g3-g5  noise-range switch (val 3 joint cases): SG512-300 LoRA only on steps [0,42) (sigma>0.7) / only [42,50); GT512-300 only [42,50)
#  g6-g7  sampling steps 100 (trajectory-curvature hypothesis): SG512-300, GT512-300
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; S=$R/scripts_omnidance/exp; E=$R/logs_omnidance/exp/H3AR; M=$R/models/train
U=/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930
SG=$M/H3REF_sg512_fm/step-300.safetensors; GT=$M/H3REF_gt512_fm/step-300.safetensors
D=$E/DIAG_selfrepro; mkdir -p $D; head -3 $E/SG512/sg512.txt > $D/list3.txt
bash $S/hold.sh stop 0,1,2,3,4,5,6,7
cd /opt
run() { g=$1; shift; DIFFSYNTH_SKIP_DOWNLOAD=True PYTHONPATH=$U CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=4 /opt/venv_h3/bin/python "$@"; }
SR="$S/h3ar/sg_gen.py --root /opt/h3cache/cfg_rich_cur --list $D/list3.txt --decode_all"
SA="$S/h3ar/sample_h3_ar.py --mode bi --steps 50 --prompt_mode orig --shift 12 --audio_shift 3 --conds joint --rows 3"
mkdir -p $D/logs
run 0 $SR --out $D/base > $D/logs/base.log 2>&1 &
sleep 60; run 1 $SR --out $D/sg300 --lora $SG > $D/logs/sg300.log 2>&1 &
sleep 60; run 2 $SR --out $D/gt300 --lora $GT > $D/logs/gt300.log 2>&1 &
for x in "3 SG512300_L0-42 $SG 0:42" "4 SG512300_L42-50 $SG 42:50" "5 GT512300_L42-50 $GT 42:50"; do set -- $x
  sleep 60; O=$E/val_RC_$2_biorig; mkdir -p $O/logs; run $1 $SA --out $O --lora $3 --lora_steps $4 > $O/logs/g$1.log 2>&1 &
done
for x in "6 SG512300_n100 $SG" "7 GT512300_n100 $GT"; do set -- $x
  sleep 60; O=$E/val_RC_$2_biorig; mkdir -p $O/logs; run $1 ${SA/--steps 50/--steps 100} --out $O --lora $3 > $O/logs/g$1.log 2>&1 &
done
wait
bash $S/hold.sh start 0,1,2,3,4,5,6,7
echo "[rc] done $(date)"
