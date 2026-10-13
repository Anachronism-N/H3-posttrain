#!/bin/bash
# One-shot status of all experiments + GPU usage on 8 nodes.
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
echo "=== $(date)"
echo "--- GPU (mem GB / util%) per node; H=holder procs"
i=1
for h in $(awk '{print $1}' /etc/taiji/hostfile); do
  s=$(timeout 15 ssh -o BatchMode=yes $h "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf \"%d/%d \", \$1/1024, \$2}'; echo -n ' H='; pgrep -fc '[h]old_gpu.py'")
  echo "N$i $h: $s"; i=$((i+1))
done
echo "--- E1 stage4 resume"
tail -1 $R/models/train/OSS_fullSFT_stage4_resume_from11k/train.log | grep -o "step=[0-9]* .*loss=[0-9.]*\|grad_norm=[0-9.]*" | tr '\n' ' '; echo
ls $R/models/train/OSS_fullSFT_stage4_resume_from11k/ | grep safetensors | tr '\n' ' '; echo
echo "--- E5 DMD"
for d in $R/models/train/E5*dmd*; do
  n=$(basename $d); [ -f $d/train_log.jsonl ] || continue
  /opt/venv_od/bin/python - "$d" <<'EOF' 2>/dev/null
import json,sys,os
d=sys.argv[1]; L=[json.loads(l) for l in open(d+'/train_log.jsonl')]
g=[x for x in L[-200:] if 'gen_loss' in x]; f=L[-50:]
sec=sum(x['sec'] for x in f)/len(f)
print(f"{os.path.basename(d):32s} iter={L[-1]['iter']:5d} s/it={sec:5.1f} fake={sum(x['fake_loss'] for x in f)/len(f):.4f} gen={sum(x['gen_loss'] for x in g)/max(1,len(g)):.4f} dmdgrad={sum(x['dmd_grad'] for x in g)/max(1,len(g)):.3f} mem={L[-1]['mem_gb']:.0f}G ckpts={len([x for x in os.listdir(d) if x.startswith('iter_')])} {'DONE' if os.path.exists(d+'/DONE') else ''}")
EOF
  grep -q Traceback $R/logs_omnidance/exp/E5/$n.log 2>/dev/null && echo "   !!! Traceback in $n.log"
done
echo "--- evals"
for d in $R/logs_omnidance/exp/E2/metrics_* $R/logs_omnidance/exp/E5/metrics/*; do [ -d $d ] && echo "$(basename $d): $(cat $d/shard_*.jsonl 2>/dev/null | wc -l) rows"; done
for d in $R/logs_omnidance/exp/E2_teacher480/out/* $R/logs_omnidance/exp/E5/eval/*; do [ -d $d ] && echo "$(basename $d): $(find $d -name '*.mp4' | wc -l) videos $( [ -f $d/.done ] && echo done)"; done
echo "--- E3 H3 zero-shot: $(find $R/logs_omnidance/exp/E3_h3/out -name '*.mp4' 2>/dev/null | wc -l)/188 videos"
echo "--- E6-1 ODE pairs: $(ls $R/data/ode_pairs/S4final_704p/*.pt 2>/dev/null | wc -l)/1200"
echo "--- E6/E9 runs"
for d in $R/models/train/E6_2* $R/models/train/E6_3[a-z]* $R/models/train/E9[a-z]*; do
  [ -f $d/train_log.jsonl ] || continue
  echo "$(basename $d) it=$(tail -1 $d/train_log.jsonl | grep -o '"iter": [0-9]*' | cut -d' ' -f2) $([ -f $d/DONE ] && echo DONE) tb=$(grep -c Traceback $R/logs_omnidance/exp/E6/$(basename $d).log 2>/dev/null) h=$(tail -1 $d/health.jsonl 2>/dev/null | cut -c1-150)"
done
echo "--- ODE part2: $(ls $R/data/ode_pairs/S4final_704p_part2/*.pt 2>/dev/null | wc -l)/2400"
