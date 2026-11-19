import hashlib,json
from pathlib import Path
RT=Path(__file__).resolve().parent;EXP=RT.parent
r=json.loads((RT/'source_curated_consumer_v1/freeze.receipt.json').read_text());summary=json.loads(Path(r['manifest_summary']['path']).read_text());plan=RT/'curated222939_admission_ids_v1.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=EXP.parent/'32b_replica_stage_receipts/source_reference_v1.json';model=json.loads(ref.read_text())['model_identity'];old=json.loads((RT/'stable_roi3_v3_launch.json').read_text())
for i,(node,gpus) in enumerate([('N8',[4,5]),('N8',[6,7]),('N1',[0,1]),('N1',[2,3])]):
 files=dict(r['files'])
 for p in [RT/'curated_resource_manager.py',RT/'curated_consumer_resource_manager.py',RT/'resource_manager.py',ref]:files[str(p)]=sha(p)
 cfg={'node':node,'mode':'consumer','phase':'curated_consumer_s%d_v1'%i,'gpus':gpus,'holders':[],'release_authorized':True,'producer':r['producer'],'source_sha256':files,
 'modelref':{'path':str(ref),'sha256':sha(ref),'model_identity_sha256':model['sha256']},'plan':{'path':str(plan),'sha256':sha(plan)},'original_full_manifest':summary['manifest'],'manifest_summary':r['manifest_summary'],
 'input_count':222939,'targetcount':len(range(i,222939,4)),'source_selection':{'shard':i,'nshard':4,'limit':0},'cwd':r['ms_source'],'out':str(EXP/'curated222939_production_v1'),'queue':str(EXP/'queues'/('curated_consumer_s%d_v1'%i)),
 'reuse_existing_results':True,'env':{**old['env'],'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus))},
 'argv':['/opt/venv_vllm/bin/python','-B','-u',r['producer']['path'],'consumer','--ms-source',r['ms_source'],'--manifest',r['manifest_summary']['path'],'--out',str(EXP/'curated222939_production_v1'),
 '--model','/opt/models/Qwen3-VL-32B-Instruct','--model-reference',str(ref),'--tp','2','--shard',str(i),'--nshard','4','--limit','0','--stage','run','--watch','--style-reuse']+[str(EXP.parent/'vlm'/('labels_s%d.jsonl'%s)) for s in range(4)]}
 cfg['env']['PYTHONDONTWRITEBYTECODE']='1'
 p=RT/('curated_consumer_s%d_v1_template.json'%i)
 with p.open('x') as f:json.dump(cfg,f,indent=2)
 print(json.dumps({'template':str(p),'selected':cfg['targetcount']}))
