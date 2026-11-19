import hashlib,json
from pathlib import Path
RT=Path(__file__).resolve().parent;EXP=RT.parent
r=json.loads((RT/'source_curated_producer_v1/freeze.receipt.json').read_text());summary=json.loads(Path(r['manifest_summary']['path']).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Frozen summary contains immutable per-batch hashes; producer verifies each batch on consumption.
bindings={p:h for p,h in r['files'].items() if p.endswith('.py')}
for p in [r['manifest_summary']['path'],str(RT/'curated_resource_manager.py'),str(RT/'resource_manager.py')]:bindings[p]=sha(p)
plan=RT/'curated222939_admission_ids_v1.jsonl';planref={'path':str(plan),'sha256':sha(plan)}
modelpath=RT/'curated_roi_local_n8_v1.json';model=json.loads(modelpath.read_text());bindings[str(modelpath)]=sha(modelpath)
for v in model['files'].values():
 if not v['path'].startswith('/opt/'):bindings[v['path']]=v['sha256']
producer=r['producer'];summaryref=r['manifest_summary'];msha=sha(RT/'curated_resource_manager.py')
for i in range(4):
 cfg=json.loads((RT/('curated_producer_p%d_v1_template.json'%i)).read_text())
 cfg.update(holders=[],phase='curated_producer_p%d_v2'%i,queue=str(EXP/'queues'/('curated_producer_p%d_v2'%i)),source_sha256=bindings,plan=planref,manifest_summary=summaryref,original_full_manifest=summary['manifest'],modelref={'path':str(modelpath),'sha256':sha(modelpath),'model_identity_sha256':model['sha256']})
 at=cfg['argv'].index('--roi-model-reference');cfg['argv'][at+1]=str(modelpath)
 p=RT/('curated_producer_p%d_v2_template.json'%i)
 with p.open('x') as f:json.dump(cfg,f,indent=2)
print(json.dumps({'templates':4,'bound_small_code_files':len(bindings),'per_batch_hashes_checked_on_consumption':True,'base_manager_sha256':sha(RT/'resource_manager.py'),'wrapper_sha256':msha}))
