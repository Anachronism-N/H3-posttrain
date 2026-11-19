import importlib.util,json,sys
from pathlib import Path
RT=Path(__file__).resolve().parent;EXP=RT.parent;ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie')
source=RT/'source_hand_roi_v1/ms_hand_roi.py';sys.path.insert(0,str(source.parent))
spec=importlib.util.spec_from_file_location('roi_snapshot_launch',source);roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
manager_spec=importlib.util.spec_from_file_location('roi_resourcemanager',RT/'roi_resource_manager.py');m=importlib.util.module_from_spec(manager_spec);manager_spec.loader.exec_module(m);b=m.base
uuid=b.uuid_for(6);running=[p for p in b.apps() if p['uuid']==uuid]
if len(running)!=1:raise RuntimeError('Expected one project holder on authorized GPU6')
identity=b.identity(running[0]['pid']);expected=dict(identity,gpu=6,uuid=uuid);b.check_holder(expected,b.identity(expected['pid']),6,uuid,b.apps());b.lease_snapshot(Path('/tmp/od_reserve_6'))
model=roi.model_identity();modelpath=RT/'hand_roi_modelref_v1.json'
if modelpath.exists():
 if json.loads(modelpath.read_text())!=model:raise RuntimeError('Model identity changed')
else:
 with modelpath.open('x') as f:json.dump(model,f,indent=2)
plan=ROOT/'diff/logs_omnidance/exp/MS/chunk_caption12k_input_v1/3smoke_manifest.jsonl';out=EXP/'hand_roi_gpu_fixed3_v1';queue=EXP/'queues/hand_roi_gpu_fixed3_v1'
refs={str(source):b.sha(source),str(source.with_name('dwpose_full_extract.py')):b.sha(source.with_name('dwpose_full_extract.py')),str(RT/'roi_resource_manager.py'):b.sha(RT/'roi_resource_manager.py')}
argv=['/opt/venv_eval/bin/python','-B','-u',str(source),'extract','--manifest',str(plan),'--out',str(out),'--limit','3','--ffmpeg','/usr/local/bin/ffmpeg','--ffprobe','/opt/ms_tools/ffprobe']
cfg={'node':'N5','gpus':[6],'holders':[expected],'release_authorized':True,'modelref':{'path':str(modelpath),'sha256':b.sha(modelpath),'model_identity_sha256':model['sha256']},'producer':{'path':str(source),'sha256':b.sha(source)},'plan':{'path':str(plan),'sha256':b.sha(plan)},'source_sha256':refs,'argv':argv,'env':{'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1'},'cwd':str(source.parent),'out':str(out),'queue':str(queue),'phase':'hand_roi_fixed3_v1','targetcount':3,'input_count':3,'source_selection':{'shard':0,'nshard':1,'limit':3},'reuse_existing_results':False}
m.validate_roi(cfg);path=RT/'hand_roi_gpu_launch_v1.json'
with path.open('x') as f:json.dump(cfg,f,indent=2)
print(json.dumps({'config':str(path),'config_sha256':b.sha(path),'manager_sha256':b.sha(RT/'resource_manager.py'),'roi_wrapper_sha256':b.sha(RT/'roi_resource_manager.py'),'holder':expected['pid'],'signals_sent':False}))
