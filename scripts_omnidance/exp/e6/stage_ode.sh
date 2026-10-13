#!/bin/bash
# stage ODE pairs to /dev/shm/od/ode (32 parallel copies)
S=/apdcephfs_gy4/share_302533218/cedricnie/diff/data/ode_pairs/S4final_704p; D=/dev/shm/od/ode; mkdir -p $D
t0=$(date +%s)
ls $S | grep '\.pt$' | xargs -P 32 -I{} sh -c "[ -f $D/{} ] && [ \$(stat -c %s $S/{}) -eq \$(stat -c %s $D/{}) ] || cp $S/{} $D/{}.tmp && mv $D/{}.tmp $D/{} 2>/dev/null"
echo "ODE_STAGED $(ls $D/*.pt | wc -l) $(( $(date +%s)-t0 ))s"; touch $D/.staged
