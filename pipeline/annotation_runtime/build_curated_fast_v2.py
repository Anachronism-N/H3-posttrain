import json,hashlib
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proofpath=RT/'curated_source_index_v1.receipt.json';proof=json.loads(proofpath.read_text());index=proof['index'];model=RT/'curated_roi_local_n8_v1.json';identity=json.loads(model.read_text());manager=RT/'curated_resource_manager_fast_v2.py'
for s in range(4):
 cfg=json.loads((RT/('curated_producer_s%d_v1_launch.json'%s)).read_text());cfg['queue']=str(RT.parent/'queues'/('curated_producer_s%d_v2'%s));cfg['phase']='curated_producer_s%d_v2'%s
 cfg['plan']=index;cfg['source_index_proof']={'path':str(proofpath),'sha256':sha(proofpath)}
 cfg['source_sha256'].pop(proof['source_manifest']['path'],None);cfg['source_sha256'][index['path']]=index['sha256'];cfg['source_sha256'][str(proofpath)]=sha(proofpath);cfg['source_sha256'].pop(str(RT/'curated_resource_manager.py'),None);cfg['source_sha256'][str(manager)]=sha(manager)
 cfg['modelref']={'path':str(model),'sha256':sha(model),'model_identity_sha256':identity['sha256']};cfg['source_sha256'].pop(str(RT/'curated_roi_modelref_n8_v1.json'),None);cfg['source_sha256'][str(model)]=sha(model)
 a=cfg['argv'];a[a.index('--roi-model-reference')+1]=str(model)
 path=RT/('curated_producer_s%d_v2_launch.json'%s)
 with path.open('x') as f:json.dump(cfg,f,indent=2)
 print(json.dumps({'shard':s,'config':str(path),'sha256':sha(path),'count':cfg['targetcount']}))
