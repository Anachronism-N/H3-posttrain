import json,hashlib
from pathlib import Path
RT=Path(__file__).resolve().parent;E=RT.parent;F=RT/'source_independent_review_pilot_v1';P=RT/'curated_review_watch_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=E/'curated222939_inputs_v1/manifest.summary.json';index=RT/'curated222939_admission_ids_v1.jsonl';model=RT/'reviewer8b_stage_TENCENT64.site.json';teacher=E.parent/'32b_replica_stage_receipts/source_reference_v1.json';out=E/'curated_independent_reviews_v1';bindings=json.loads((F/'freeze.receipt.json').read_text())['files']
for p in [P,summary,index,model,teacher,RT/'curated_review_resource_manager_v1.py',RT/'resource_manager.py']:bindings[str(p)]=sha(p)
cap=json.loads((RT/'curated_review_watch_fresh_v1.json').read_text())
c={'node':'N5','gpus':[7],'holders':cap['holders'],'release_authorized':True,'phase':'curated_review_watch_v1','queue':str(E/'queues/curated_review_watch_v1'),'out':str(out),'cwd':str(F),'producer':{'path':str(P),'sha256':sha(P)},'plan':{'path':str(index),'sha256':sha(index)},'input_count':222939,'targetcount':222939,'source_selection':{'shard':0,'nshard':1,'limit':0},'modelref':{'path':str(model),'sha256':sha(model),'model_identity_sha256':sha(model)},'source_sha256':bindings,'reuse_existing_results':True,'env':{'CUDA_VISIBLE_DEVICES':'7','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MS_VISUAL_CPU_SITEPATH':'/opt/venv_eval/lib/python3.11/site-packages'},'argv':['/opt/venv_vllm/bin/python','-B','-u',str(P),'--manifest',str(summary),'--results-root',str(E/'curated222939_production_v1'),'--out',str(out),'--model','/opt/models/Qwen3-VL-8B-Instruct','--ms-source',str(F),'--annotator-model-reference',str(teacher),'--poll-seconds','60'],'semantic_approval':False}
p=RT/'curated_review_watch_launch_v1.json'
with p.open('x') as f:json.dump(c,f,indent=2)
print(json.dumps({'path':str(p),'sha256':sha(p),'batch_count':448,'source_count':222939}))
