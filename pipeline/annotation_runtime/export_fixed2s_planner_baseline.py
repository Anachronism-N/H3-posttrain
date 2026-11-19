import collections,hashlib,json
from pathlib import Path
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie');MS=ROOT/'diff/logs_omnidance/exp/MS'
OUT=MS/'music_planner_fixed2s_baseline_20261024_v1'
INPUT=MS/'chunk_caption12k_input_v1/manifest.jsonl'
SELECTION=INPUT.with_name('selection.json');LINEAGE=MS/'ms_chunk12k_final_export_v1/lineage.jsonl'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def save(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=raw:raise ValueError('Different existing baseline artifact: '+str(p))
  return
 with p.open('xb') as f:f.write(raw)
input_raw=INPUT.read_bytes()
if sha(input_raw)!='4b2efc8233fed8dd7cdb46acf8da318db76cb575faa1177e9e2bc113d453d283':raise ValueError('Authoritative input changed')
rows=[json.loads(s) for s in input_raw.decode().splitlines() if s]
if len(rows)!=12000 or len({r['id'] for r in rows})!=12000:raise ValueError('Expected current12k unique source pool')
selection=json.loads(SELECTION.read_bytes());oldval=set(selection['authority_planner_validation_overlap_ids'])
metadata={}
for line in LINEAGE.open():
 v=json.loads(line);metadata[v['id']]={'audio_sha256':v['audio_sha256'],'video_sha256':v['video_sha256']}
if set(metadata)!={r['id'] for r in rows}:raise ValueError('Missing prior source-bound lineage')
parent={r['id']:r['id'] for r in rows}
def find(x):
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[max(a,b)]=min(a,b)
group_seen={};audio_seen={}
for r in rows:
 sid=r['id'];sg=r['source_group'];audio=metadata[sid]['audio_sha256']
 if sg in group_seen:union(sid,group_seen[sg])
 else:group_seen[sg]=sid
 if audio in audio_seen:union(sid,audio_seen[audio])
 else:audio_seen[audio]=sid
members=collections.defaultdict(list)
for sid in parent:members[find(sid)].append(sid)
valgroups={find(sid) for sid in oldval};index={};refs=[]
for i in range(8):
 p=MS/f'visual12k_local32b_full_v1/shard{i}/shards/shard-{i:05d}-of-00008/index.jsonl';raw=p.read_bytes();refs.append({'path':str(p),'sha256':sha(raw)})
 for l in raw.decode().splitlines():
  if not l:continue
  v=json.loads(l)
  if v['id'] in index:raise ValueError('Duplicate indexed source')
  index[v['id']]=v
if set(index)!=set(parent):raise ValueError('Incomplete indexed observation pool')
eligible={};excluded=[]
for r in rows:
 sid=r['id'];ref=index[sid];path=Path(ref['final_path']);raw=path.read_bytes();f=json.loads(raw)
 if sha(raw)!=ref['final_sha256'] or f['id']!=sid or f['status']!=ref['status']:raise ValueError('Final source/digest mismatch '+sid)
 windows=f['segments']
 if [(w['start'],w['end']) for w in windows]!=[(0.,2.),(2.,5.0625)]:raise ValueError('Unexpected source windows '+sid)
 expected='\n'.join(f"[{w['start']:.3f}-{w['end']:.3f}s] {w['caption']}" for w in windows if w['caption'])
 if f['timed_caption']!=expected:raise ValueError('Final caption differs from original candidate windows')
 full=all(w['status']=='candidate' and w['caption'].strip() for w in windows)
 if not full:excluded.append({'id':sid,'reason':'incomplete_observed_window_caption','source_ref':{'path':str(path),'sha256':sha(raw)}});continue
 group=find(sid)
 eligible[sid]={'id':sid,'source_group_id':sha(dump(sorted(members[group])).encode()),
  'audio_path':r['audio_path'],'audio_sha256':metadata[sid]['audio_sha256'],
  'video_path':r['video_path'],'video_sha256':metadata[sid]['video_sha256'],
  'audio_window_seconds':[0.,5.0625],'target_text':f['timed_caption'],
  'target_windows':[{'start':w['start'],'end':w['end'],'caption':w['caption']} for w in windows],
  'caption_version':'qwen3vl32b_fixed2s_video_first_weak_v1',
  'source_label_ref':{'path':str(path),'sha256':sha(raw)},
  'audio_video_alignment_verified':False,'original_source_window_mapping':'unknown',
  'semantic_approval':False,'human_review_completed':False,'weak_supervision':True}
 print(dump({'event':'checked','count':len(eligible)+len(excluded)}),flush=True) if (len(eligible)+len(excluded))%1000==0 else None
if len(eligible)!=11710:raise ValueError('Complete usable source count differs from verified11710')
eligiblegroups=collections.defaultdict(list)
for sid in eligible:eligiblegroups[find(sid)].append(sid)
ranked=sorted((g for g in eligiblegroups if g not in valgroups),key=lambda g:sha(('fixed2s-test-seed42:'+dump(sorted(members[g]))).encode()))
test=set()
for g in ranked:
 ids=eligiblegroups[g]
 if len(test)+len(ids)>500:continue
 test.update(ids)
 if len(test)==500:break
if len(test)!=500:raise ValueError('Unable to select500 test clips without splitting source/audio group')
splits={'train':[],'val':[],'test':[]}
for r in rows:
 sid=r['id']
 if sid not in eligible:continue
 split='val' if find(sid) in valgroups else 'test' if sid in test else 'train'
 splits[split].append(dict(eligible[sid],split=split))
for key in ('source_group_id','audio_sha256'):
 values={s:{r[key] for r in v} for s,v in splits.items()}
 if any(values[a]&values[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Split source/audio overlap')
if oldval & {r['id'] for r in splits['train']}:raise ValueError('Old validation leaked into training')
for s,rs in splits.items():save(OUT/(s+'.jsonl'),(''.join(dump(r)+'\n' for r in rs)).encode())
save(OUT/'excluded.jsonl',(''.join(dump(r)+'\n' for r in excluded)).encode())
summary={'schema':'music_planner_fixed2s_weak_baseline_v1','source_input_count':12000,'complete_sources':len(eligible),'excluded_incomplete_sources':len(excluded),
 'split_counts':{s:len(v) for s,v in splits.items()},'source_and_exact_audio_split_overlap':0,'old_validation_in_train':0,
 'song_and_near_duplicate_isolation_verified':False,'group_basis':'original video source and exact audio content SHA only',
 'source_index_refs':refs,'input_manifest_ref':{'path':str(INPUT),'sha256':sha(input_raw)},
 'split_artifacts':{s:{'path':str(OUT/(s+'.jsonl')),'sha256':sha((OUT/(s+'.jsonl')).read_bytes())} for s in splits},
 'source_audio_sampling_rate':'original normalized24k mono; each native model adapter resamples full window as required',
 'model_prompt_bound_in_data':False,'captioner_input_policy':'native audio-only chat tokens; no added text instruction/system',
 'MOSS_Flamingo_input_policy':'native valid audio plus task instruction; same complete audio and target',
 'training_started':False,'semantic_approval':False,'human_review_completed':False,'output':str(OUT)}
save(OUT/'summary.json',(json.dumps(summary,indent=2)+'\n').encode());print(json.dumps(summary),flush=True)
