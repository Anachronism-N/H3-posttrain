import json,hashlib
from pathlib import Path
RT=Path(__file__).resolve().parent;EXP=RT.parent
r=json.loads((RT/'source_curated_producer_v1/freeze.receipt.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bindings=dict(r['files'])
# Models and already-verified immutable batch manifests are bound separately.
bindings[str(RT/'curated_resource_manager.py')]=sha(RT/'curated_resource_manager.py')
ref=RT/'curated_roi_modelref_n8_v1.json';model=json.loads(ref.read_text());bindings[str(ref)]=sha(ref)
for v in model['files'].values():bindings[v['path']]=v['sha256']
source=r['producer']['path'];summary=r['manifest_summary']['path'];root=EXP/'curated222939_production_v1';root.mkdir(exist_ok=True)
for i in range(4):
 cfg={'node':'N8','mode':'producer','phase':'curated_producer_p%d_v1'%i,'gpus':[i],'holders':[],'release_authorized':True,
 'producer':r['producer'],'plan':r['full_manifest'],'source_sha256':bindings,'modelref':{'path':str(ref),'sha256':sha(ref),'model_identity_sha256':model['sha256']},
 'input_count':222939,'targetcount':len(range(i,222939,4)),'source_selection':{'shard':i,'nshard':4,'limit':0},
 'cwd':str(Path(source).parent),'out':str(root),'queue':str(EXP/'queues'/('curated_producer_p%d_v1'%i)),
 'reuse_existing_results':True,'semantic_approval':False,
 'env':{'CUDA_VISIBLE_DEVICES':str(i),'OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
 'argv':['/opt/venv_eval/bin/python','-B','-u',source,'producer','--manifest',summary,'--out',str(root),'--repo',r['repo'],
 '--shard',str(i),'--nshard','4','--limit','0','--stage','run','--device','0','--roi-model-reference',str(ref),
 '--beat-sitepath','/opt/curated_music_dependencies_v1','--beat-audio-sitepath','/opt/curated_music_dependencies_v1',
 '--beat-device','cuda','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe']}
 p=RT/('curated_producer_p%d_v1_template.json'%i)
 with p.open('x') as f:json.dump(cfg,f,indent=2)
 print(json.dumps({'template':str(p),'selected':cfg['targetcount'],'sha256':sha(p)}))
