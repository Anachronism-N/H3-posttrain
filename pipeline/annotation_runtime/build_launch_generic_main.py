import argparse,hashlib,json
from pathlib import Path
ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie");EXP=ROOT/'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1';RT=EXP/'runtime';BASE=ROOT/'H3-posttrain/scripts_omnidance/exp/ms'
EXPECTED='3151fc91d2195389648b45462015fcb82f8998326b3d17976f14794b0419170f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('Different existing frozen artifact '+str(p))
  return
 with p.open('xb') as f:f.write(b)
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--node',required=True);ap.add_argument('--gpus',nargs=2,required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshard',type=int,default=1);ap.add_argument('--reuse',action='store_true');ap.add_argument('--out-name');ap.add_argument('--gate-name');ap.add_argument('--source-sha',required=True);ap.add_argument('--snapshot',default='source_validated_v3');args=ap.parse_args();EXPECTED=args.source_sha
source=BASE/'ms_bar_annotation.py'
if sha(source)!=EXPECTED:raise ValueError('Validated source changed')
archive=RT/args.snapshot;refs={}
files=[BASE/f for f in ['ms_bar_annotation.py','ms_caption_rewrite.py','ms_visual_batch.py','ms_visual_observe.py','ms_optimize.py','ms_landmarks.py','ms_hand_roi.py','dwpose_full_extract.py']]+list((BASE/'motionscript_v2').glob('*.py'))
for p in files:
 target=archive/p.relative_to(BASE);raw=p.read_bytes();save(target,raw);refs[str(target)]=hashlib.sha256(raw).hexdigest()
plan=Path(args.plan).resolve();rows=[json.loads(l) for l in plan.read_text().splitlines() if l.strip()];ids=[r['id'] for r in rows][args.shard::args.nshard];ids=ids[:args.limit] if args.limit else ids
if not ids:raise ValueError('No selected sources')
modelref=ROOT/'diff/logs_omnidance/exp/MS/32b_replica_stage_receipts/source_reference_v1.json';model=json.loads(modelref.read_text())['model_identity'];out=EXP/(args.out_name or args.name);queue=EXP/'queues'/args.name;gpus=list(map(int,args.gpus))
argv=['/opt/venv_vllm/bin/python','-B','-u',str(archive/source.name),'--plan',str(plan),'--model','/opt/models/Qwen3-VL-32B-Instruct','--model-reference',str(modelref),'--out',str(out),'--stage','run','--limit',str(args.limit),'--shard',str(args.shard),'--nshard',str(args.nshard),'--tp','2','--batch-size','2','--max-num-seqs','2','--sample-fps','8','--boundary-review','on','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe']
cfg={'node':args.node,'gpus':gpus,'holders':[],'release_authorized':True,'modelref':{'path':str(modelref),'sha256':sha(modelref),'model_identity_sha256':model['sha256']},'producer':{'path':str(archive/source.name),'sha256':EXPECTED},'plan':{'path':str(plan),'sha256':sha(plan)},'source_sha256':refs,'argv':argv,'env':{'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),'OMP_NUM_THREADS':'2','MS_VISUAL_CPU_SITEPATH':'/opt/venv_eval/lib/python3.11/site-packages','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1'},'cwd':str(archive),'out':str(out),'queue':str(queue),'phase':args.name,'targetcount':len(ids),'input_count':len(rows),'source_selection':{'shard':args.shard,'nshard':args.nshard,'limit':args.limit},'reuse_existing_results':args.reuse,'semantic_approval':False,'formal_training':False}
if args.gate_name:cfg['previous_generation_gate']={'queue':str(EXP/'queues'/args.gate_name),'out':str(EXP/args.gate_name)}
template=RT/(args.name+'_template.json');save(template,(json.dumps(cfg,indent=2)+'\n').encode())
print(json.dumps({'template':str(template),'producer_sha256':EXPECTED,'plan_sha256':sha(plan),'targetcount':len(ids),'selected_ids':ids if len(ids)<10 else None,'out':str(out)}))
