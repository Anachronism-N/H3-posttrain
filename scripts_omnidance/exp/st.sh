#!/bin/bash
# compact status of active runs
R=/apdcephfs_gy4/share_302533218/cedricnie/diff
date
for d in $R/models/train/E6_2* $R/models/train/E6_3[a-z]* $R/models/train/E9a* $R/models/train/E5b5*; do
  [ -f $d/train_log.jsonl ] || continue
  n=$(basename $d); lg=$(ls $R/logs_omnidance/exp/E6/$n.log $R/logs_omnidance/exp/E5/$n.log 2>/dev/null | head -1)
  echo "$n it=$(tail -1 $d/train_log.jsonl | grep -o '"iter": [0-9]*' | cut -d' ' -f2) $([ -f $d/DONE ] && echo DONE) tb=$(grep -c Traceback $lg) h=$(tail -1 $d/health.jsonl 2>/dev/null | sed 's/"hf_ratio": [0-9.]*, //;s/"ar_mse_vs_teacher"/mse/' | cut -c1-130)"
done
echo "ODE part2: $(ls $R/data/ode_pairs/S4final_704p_part2/*.pt 2>/dev/null | wc -l)/2400"
