#!/bin/bash
# ode_node.sh: 8 GPU-resident teacher processes (H-A2 step-750, 25 steps, shift 12) -> /opt/h3ode/ha2_750 ; list shuffled n1_cur
python3 -c "import random;L=open('/opt/h3cache/ode_list_n1cur.txt').read().split();random.Random(0).shuffle(L);open('/opt/h3cache/ode_list_n1cur_shuf.txt','w').write('\n'.join(L)+'\n')"
for g in 0 1 2 3 4 5 6 7; do
  ( bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; cd /opt && CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/ode_gen.py --resident \
      --list /opt/h3cache/ode_list_n1cur_shuf.txt --out /opt/h3ode/ha2_750 --lora /apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/H3A2_decfgDS_lr5e-5_full12k/step-750.safetensors \
      --steps 25 --limit ${LIMIT:-2400} --shard $g --nshard 8 > /tmp/ode_$g.log 2>&1 ) &
  sleep 45
done; wait; echo "[ode_node] done $(ls /opt/h3ode/ha2_750 | wc -l) $(date)"
