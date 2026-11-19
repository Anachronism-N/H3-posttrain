import json,hashlib
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');EXP=RT.parent
labels={}
for p in sorted((EXP.parent/'vlm').glob('labels_s*.jsonl')):
 fsha=hashlib.sha256(p.read_bytes()).hexdigest()
 for n,line in enumerate(p.open(),1):
  v=json.loads(line);labels[v['sid']]={'label':v['label'],'ref':{'path':str(p),'sha256':fsha,'line':n,'raw_line_sha256':hashlib.sha256(line.encode()).hexdigest()}}
paths=sorted(p for s in range(4) for p in (EXP/'full12000_stable_roi_v1'/('shard'+str(s))/'completed').glob('*.json'));rows=[]
for p in paths:
 r=json.loads(p.read_text());sid=r['id'];m=r['manifest_record'];w=r.get('final',{}).get('json',{}).get('windows',[])
 if m['split']!='train' or r['status']!='candidate' or len(w)!=len(r['original_windows']) or not all(x['caption'].strip() for x in w) or sid not in labels:continue
 if not (EXP/'hand_roi_full12000_v1'/sid/'roi_manifest.json').exists():continue
 label=labels[sid]['label'];genre=label.get('genre');motion=r['final_timed_caption']
 if not genre or not motion.strip():continue
 rows.append({'id':sid,'source_id':sid,'split':'train','batch_id':'independent-pilot25-v1','video_path':m['video_path'],'audio_path':m['audio_path'],'dance_style_label':genre,'label':{'dance_style':genre,'timed_motion_script':motion},'target_text':'Dance style: '+genre+chr(10)+motion,'label_status':'completed','style_status':'resolved','annotator_model_sha256':r['model_identity_ref']['sha256'] if False else json.loads(Path(r['model_identity_ref']['path']).read_text())['sha256'],'source_refs':{'source_label':labels[sid]['ref'],'motion_completed':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}},'semantic_approval':False,'human_review_completed':False})
 if len(rows)==25:break
assert len(rows)==25
p=RT/'independent_review_pilot25_rows_v1.jsonl';raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+chr(10) for r in rows).encode()
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(p),'count':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'selected_full_window_motion_only':True,'historical_style_not_human_verified':True}))
