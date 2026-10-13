#!/bin/bash
# ship_cache.sh <src view dir> <dst host> <dst dir>: copy a (symlink) cache view to another node as real files, 2 parallel tar streams
SRC=$1; H=$2; DST=$3
ssh $H "mkdir -p $DST"
for half in 0 1; do
  ( cd $SRC && find -L . -name '*.pth' | awk -v h=$half 'NR%2==h' | tar -chf - -T - | ssh $H "cd $DST && tar -xf -" ) &
done
wait; echo "[ship] done $(ssh $H "find $DST -name '*.pth' | wc -l") files $(date)"
