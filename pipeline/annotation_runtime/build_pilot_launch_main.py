import hashlib,json,sys
from pathlib import Path
ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie");REPO=ROOT/'H3-posttrain';EXP=ROOT/'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1';RT=EXP/'runtime'
SOURCE=REPO/'scripts_omnidance/exp/ms/ms_bar_annotation.py';EXPECTED='b63a765fd06720bbe8283fd91d71cb745fbeba4a13fc0b02eb8b98b4ba07701d'
PLAN=EXP/'plan_first3.jsonl';MODELREF=ROOT/'diff/logs_omnidance/exp/MS/32b_replica_stage_receipts/source_reference_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('Different existing frozen file '+str(p))
  return
 with p.open('xb') as f:f.write(b)
if sha(SOURCE)!=EXPECTED:raise ValueError('Validated pipeline changed')
base=SOURCE.parent;archive=RT/'source_pilot_v1'
files=[base/name for name in ['ms_bar_annotation.py','ms_caption_rewrite.py','ms_visual_batch.py','ms_visual_observe.py','ms_optimize.py','ms_landmarks.py']]+list((base/'motionscript_v2').glob('*.py'))
refs={}
for p in files:
 target=archive/p.relative_to(base);raw=p.read_bytes();save(target,raw);refs[str(target)]=hashlib.sha256(raw).hexdigest()
model=json.loads(MODELREF.read_text())['model_identity']
argv=['/opt/venv_vllm/bin/python','-B','-u',str(archive/'ms_bar_annotation.py'),'--plan',str(PLAN),'--model','/opt/models/Qwen3-VL-32B-Instruct','--model-reference',str(MODELREF),'--out',str(EXP/'pilot3_v1'),'--stage','run','--limit','3','--tp','2','--batch-size','2','--max-num-seqs','2','--sample-fps','8','--boundary-review','on','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe']
cfg={'node':'N3','gpus':[2,3],'holders':[],'release_authorized':True,'modelref':{'path':str(MODELREF),'sha256':sha(MODELREF),'model_identity_sha256':model['sha256']},'producer':{'path':str(archive/'ms_bar_annotation.py'),'sha256':EXPECTED},'plan':{'path':str(PLAN),'sha256':sha(PLAN)},'source_sha256':refs,'argv':argv,'env':{'CUDA_VISIBLE_DEVICES':'2,3','OMP_NUM_THREADS':'2','MS_VISUAL_CPU_SITEPATH':'/opt/venv_eval/lib/python3.11/site-packages','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1'},'cwd':str(archive),'out':str(EXP/'pilot3_v1'),'queue':str(EXP/'queues/pilot3_v1'),'phase':'pilot3','targetcount':3,'input_count':3,'source_selection':{'shard':0,'nshard':1,'limit':3},'reuse_existing_results':False,'semantic_approval':False,'formal_training':False}
save(RT/'pilot3_template_v1.json',(json.dumps(cfg,indent=2)+'\n').encode())
save(RT/'source_pilot_v1.receipt.json',(json.dumps({'producer_sha256':EXPECTED,'plan_sha256':sha(PLAN),'archived_source_sha256':refs},indent=2)+'\n').encode())
print(json.dumps({'template':str(RT/'pilot3_template_v1.json'),'source_sha256':EXPECTED,'plan_sha256':sha(PLAN),'archive_files':len(files)}))
