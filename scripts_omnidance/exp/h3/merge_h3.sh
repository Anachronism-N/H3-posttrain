#!/bin/bash
# Run on the controller after stage_h3 shards finish. Owner of shard i = NODES[i].
# Pushes each shard's files from its owner to every other node (tar over ssh), then verifies sizes vs cephfs.
NODES=(28.216.17.70 28.216.19.213 28.216.19.225)
EXTRA=("$@")          # extra destination nodes that hold no shard
W=/apdcephfs_gy4/share_302533218/cedricnie/hyperflow_exp/weights/minimax-h3/Ref2VA
cd $W
n=${#NODES[@]}
for i in $(seq 0 $((n-1))); do
  find . -type f | sort | awk -v s=$i -v n=$n 'NR%n==s' > /tmp/h3_shard_$i.txt
done
for i in $(seq 0 $((n-1))); do
  src=${NODES[$i]}
  ssh $src "cat > /tmp/h3_shard_$i.txt" < /tmp/h3_shard_$i.txt
  for dst in "${NODES[@]}" "${EXTRA[@]}"; do
    [ "$dst" = "$src" ] && continue
    ( ssh $src "cd /dev/shm/h3/Ref2VA && tar cf - -T /tmp/h3_shard_$i.txt" | ssh $dst "mkdir -p /dev/shm/h3/Ref2VA && cd /dev/shm/h3/Ref2VA && tar xf -" && echo "shard $i $src -> $dst ok" ) &
  done
  wait
done
find . -type f -printf "%s %p\n" | sort -k2 > /tmp/h3_sizes.txt
for dst in "${NODES[@]}" "${EXTRA[@]}"; do
  ssh $dst "cat > /tmp/h3_sizes.txt" < /tmp/h3_sizes.txt
  ssh $dst 'cd /dev/shm/h3/Ref2VA && bad=0; while read s f; do [ "$(stat -c %s "$f" 2>/dev/null)" = "$s" ] || bad=$((bad+1)); done < /tmp/h3_sizes.txt; echo "$(hostname -I | cut -d" " -f1) mismatches=$bad"; [ -s /tmp/h3_sizes.txt ] && [ $bad -eq 0 ] && touch /dev/shm/h3/.complete'
done
