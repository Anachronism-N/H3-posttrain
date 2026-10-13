#!/bin/bash
# M2 data prep on N2: cache->id map, feature list (train clips + val audios), then beat/CLAP features on 8 GPUs
cd /opt
MAP_ID=1 python3 /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/build_music_map.py /opt/h3cache/f1_cur /dev/shm/h3data/metadata_cur_n7_p0.json /opt/h3cache/music_idmap_f1cur.json
python3 - <<'PY'
import json,os,csv
csv.field_size_limit(10**9)
A='/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio'
ids=set(json.load(open('/opt/h3cache/music_idmap_f1cur.json')).values())
L=[(i,f'{A}/{i}.wav') for i in sorted(ids)]
E='/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2'
for c in ['joint','aswap']:
    for r in csv.DictReader(open(f'{E}/full_{c}.csv')): L.append((os.path.basename(r['audio_path'])[:-4], r['audio_path']))
L=sorted(set(L))
open('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/m2_list.txt','w').write(''.join(f'{i}\t{w}\n' for i,w in L)); print('list',len(L))
PY
O=/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/m2_feats
for g in 0 1 2 3 4 5 6; do
  ( bash /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold.sh stop $g; CUDA_VISIBLE_DEVICES=$g /opt/venv_h3/bin/python /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/m2_feats.py --list /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/m2_list.txt --out $O --shard $g --nshard 7 > /tmp/m2f_$g.log 2>&1 ) &
done
wait; echo "[m2prep] done $(ls $O | wc -l) $(date)"
