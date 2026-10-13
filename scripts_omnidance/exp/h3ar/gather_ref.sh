#!/bin/bash
# gather_ref.sh: send this node's full12k reference first frames (/dev/shm/h3data/ref/<id>_f0.png) to N7 and N8 /opt/h3raw/ref
ME=$(hostname -I | awk '{print $1}'); cd /dev/shm/h3data/ref
python3 -c "
import os
ids=open('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/full12k_ids.txt').read().split()
open('/tmp/ref_send.txt','w').write(''.join(f'{i}_f0.png\n' for i in ids if os.path.exists(f'{i}_f0.png')))"
for D in 28.216.18.144 28.216.18.136; do
  [ "$D" = "$ME" ] && continue
  tar -cf - -T /tmp/ref_send.txt | ssh $D "mkdir -p /opt/h3raw/ref && tar -C /opt/h3raw/ref -xf -"; echo "[gather] $ME -> $D $(wc -l < /tmp/ref_send.txt) exit $?"
done
