import json,hashlib
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');F=RT/'source_independent_review_pilot_v1';E=RT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bindings=json.loads((F/'freeze.receipt.json').read_text())['files'];producer=F/'ms_batch_review.py';plan=RT/'independent_review_pilot25_rows_v1.jsonl';model=RT/'reviewer8b_stage_TENCENT64.site.json';teacher=E.parent/'32b_replica_stage_receipts/source_reference_v1.json';out=E/'independent_review_pilot25_v1'
for p in (plan,model,teacher,RT/'independent_review_resource_manager_v1.py',RT/'resource_manager.py'):bindings[str(p)]=sha(p)
cap=json.loads((RT/'independent_review7_fresh_v1.json').read_text())
cfg={'node':'N5','gpus':[7],'holders':cap['holders'],'release_authorized':True,'phase':'independent_review_pilot25_v1','queue':str(E/'queues/independent_review_pilot25_v1'),'out':str(out),'cwd':str(F),'producer':{'path':str(producer),'sha256':sha(producer)},'plan':{'path':str(plan),'sha256':sha(plan)},'modelref':{'path':str(model),'sha256':sha(model)},'source_sha256':bindings,'input_count':25,'targetcount':25,'reuse_existing_results':True,'env':{'CUDA_VISIBLE_DEVICES':'7','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','MS_VISUAL_CPU_SITEPATH':'/opt/venv_eval/lib/python3.11/site-packages','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1'},'argv':['/opt/venv_vllm/bin/python','-B','-u',str(producer),'--batch-list',str(plan),'--out',str(out),'--model','/opt/models/Qwen3-VL-8B-Instruct','--annotator-model-reference',str(teacher),'--sample-count','25','--risk-count','0','--seed','0','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe','--hand-roi-root',str(E/'hand_roi_full12000_v1'),'--run'],'semantic_approval':False}
p=RT/'independent_review_pilot25_launch_v1.json';raw=(json.dumps(cfg,indent=2)+chr(10)).encode()
with p.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}))
