#!/bin/bash
# stage_h3.sh <shard> <nshard>: copy files (sorted, idx%nshard==shard) of H3 Ref2VA into /dev/shm/h3/Ref2VA, 16 parallel.
SH=${1:-0}; NS=${2:-1}
W=/apdcephfs_gy4/share_302533218/cedricnie/hyperflow_exp/weights/minimax-h3/Ref2VA
D=/dev/shm/h3/Ref2VA
t0=$(date +%s)
cd $W && find . -type d | while read d; do mkdir -p $D/$d; done
find . -type f | sort | awk -v s=$SH -v n=$NS 'NR%n==s' | xargs -P 16 -I{} sh -c '[ -f "'$D'/{}" ] && [ $(stat -c %s "{}") -eq $(stat -c %s "'$D'/{}") ] || cp "{}" "'$D'/{}"'
du -sh $D; echo "H3_STAGED shard $SH/$NS $(( $(date +%s)-t0 ))s"; touch /dev/shm/h3/.staged_$SH
