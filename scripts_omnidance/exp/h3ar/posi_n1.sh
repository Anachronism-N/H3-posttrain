#!/bin/bash
# posi_n1.sh: rich-prompt sidecars for N1's raw caches f0 (metadata_f0_2700) and f1 (metadata_cur_p0) -> /opt/h3cache/posi/{f0,f1}
cd /opt
/opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/cache_meta_list.py /opt/h3cache/f0 /dev/shm/h3data/metadata_f0_2700.json /opt/h3cache/meta_f0.json /opt/h3cache/posi_list_f0.tsv
/opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/cache_meta_list.py /opt/h3cache/f1 /dev/shm/h3data/metadata_cur_p0.json /opt/h3cache/meta_f1.json /opt/h3cache/posi_list_f1.tsv
python3 -c "import json;m={**json.load(open('/opt/h3cache/meta_f0.json')),**json.load(open('/opt/h3cache/meta_f1.json'))};json.dump(m,open('/opt/h3cache/meta_f01.json','w'))"
P=$(python3 -c "import json;P=json.load(open('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/all_rich_prompts.json'));print(sum(1 for l in open('/opt/h3cache/posi_list_f0.tsv').read().split('\n')+open('/opt/h3cache/posi_list_f1.tsv').read().split('\n') if l and l.split('\t')[1] in P))")
echo "[posi_n1] rich prompts available for $P"
cat /opt/h3cache/posi_list_f0.tsv /opt/h3cache/posi_list_f1.tsv > /opt/h3cache/posi_list_f01.tsv
for g in 0 1 2 3 4 5 6 7; do
  ( bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop $g >/dev/null 2>&1; CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/text_recache.py --list /opt/h3cache/posi_list_f01.tsv \
      --meta /opt/h3cache/meta_f01.json --prompts /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/all_rich_prompts.json --out /opt/h3cache/posi --ref_dirs /dev/shm/h3data/ref,/opt/h3raw/ref --shard $g --nshard 8 > /tmp/posin1_$g.log 2>&1;
    bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh start $g >/dev/null 2>&1 ) &
  sleep 20
done; wait; echo "[posi_n1] done $(find /opt/h3cache/posi/f0 /opt/h3cache/posi/f1 -name '*.posi.pt' | wc -l) $(date)"
