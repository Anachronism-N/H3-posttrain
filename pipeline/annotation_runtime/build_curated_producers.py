import hashlib,json
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');EXP=RT.parent;FROZEN=RT/'source_curated_producer_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 b=(json.dumps(v,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==b,'Unexpected config '+str(p)
 else:
  with p.open('xb') as f:f.write(b)
receipt=json.loads((FROZEN/'freeze.receipt.json').read_text());summary=json.loads(Path(receipt['manifest_summary']['path']).read_text());assert summary['total']==222939
bindings={p:h for p,h in receipt['files'].items() if p.endswith('.py')};bindings[str(RT/'curated_resource_manager.py')]=sha(RT/'curated_resource_manager.py')
bindings[receipt['manifest_summary']['path']]=receipt['manifest_summary']['sha256'];bindings[receipt['full_manifest']['path']]=receipt['full_manifest']['sha256']
model=RT/'curated_roi_modelref_n8_v1.json';m=json.loads(model.read_text());bindings[str(model)]=sha(model)
beat=Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/models/BeatThis/source_manifest.json');bindings[str(beat)]=sha(beat)
for s in range(4):
 cap=RT/('curated_p%d_fresh.json'%s);identity=json.loads(cap.read_text());assert identity['node']=='N8' and identity['mode']=='producer' and identity['gpus']==[s]
 out=EXP/'curated_production_v1'
 argv=['/opt/venv_eval/bin/python','-B','-u',receipt['producer']['path'],'producer','--manifest',receipt['manifest_summary']['path'],'--out',str(out),'--repo',receipt['repo'],'--shard',str(s),'--nshard','4','--limit','0','--stage','run','--roi-model-reference',str(model),'--beat-sitepath','/opt/curated_music_dependencies_v1','--beat-audio-sitepath','/opt/curated_music_dependencies_v1','--beat-device','cuda','--beat-checkpoint-source',str(beat),'--device','0','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe']
 cfg={'node':'N8','mode':'producer','gpus':[s],'holders':identity['holders'],'release_authorized':True,'phase':'curated_producer_s%d_v1'%s,'queue':str(EXP/'queues'/('curated_producer_s%d_v1'%s)),'out':str(out),'cwd':str(FROZEN),'argv':argv,'producer':receipt['producer'],'plan':receipt['full_manifest'],'modelref':{'path':str(model),'sha256':sha(model),'model_identity_sha256':m['sha256']},'source_sha256':bindings,'env':{'CUDA_VISIBLE_DEVICES':str(s),'OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1'},'input_count':222939,'targetcount':len(range(s,222939,4)),'source_selection':{'shard':s,'nshard':4,'limit':0},'reuse_existing_results':True,'semantic_approval':False}
 p=RT/('curated_producer_s%d_v1_launch.json'%s);save(p,cfg);print(json.dumps({'shard':s,'path':str(p),'sha256':sha(p),'targetcount':cfg['targetcount']}))
