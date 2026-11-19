import hashlib,json,shutil
from pathlib import Path
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie');RT=ROOT/'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime';BASE=ROOT/'H3-posttrain/scripts_omnidance/exp/ms';FROZEN=RT/'source_curated_producer_v1';REPO=FROZEN/'repo'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def preserve(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,'Unexpected frozen content '+str(p)
 else:
  with p.open('xb') as f:f.write(b)
files=[RT/'curated_pipeline.py']+[BASE/n for n in ['ms_bar_annotation.py','ms_visual_batch.py','ms_caption_rewrite.py','ms_visual_observe.py','ms_optimize.py','ms_landmarks.py','ms_hand_roi.py','dwpose_full_extract.py']]+list((BASE/'motionscript_v2').glob('*.py'))
bindings={}
for p in files:
 raw=p.read_bytes();compile(raw,str(p),'exec');target=FROZEN/p.name if p.parent==RT else REPO/'scripts_omnidance/exp/ms'/p.relative_to(BASE);preserve(target,raw);assert p.read_bytes()==raw;bindings[str(target)]=hashlib.sha256(raw).hexdigest()
manifest=RT.parent/'curated222939_inputs_v1/manifest.summary.json';summary=json.loads(manifest.read_text());assert summary['total']==222939 and summary['counts']=={'train':200568,'val':11110,'test':11261}
for r in [summary['manifest']]+summary['batches']+[{'path':str(manifest),'sha256':sha(manifest)}]:
 p=Path(r['path']);assert sha(p)==r['sha256'];bindings[str(p)]=r['sha256']
for n in ['curated_resource_manager.py','resource_manager.py']:
 p=RT/n;bindings[str(p)]=sha(p)
value={'files':bindings,'repo':str(REPO),'producer':{'path':str(FROZEN/'curated_pipeline.py'),'sha256':sha(FROZEN/'curated_pipeline.py')},'manifest_summary':{'path':str(manifest),'sha256':sha(manifest)},'full_manifest':summary['manifest'],'source_count':222939,'CPU_only_verified':True,'GPU_producer_already_verified':False}
p=FROZEN/'freeze.receipt.json';preserve(p,(json.dumps(value,indent=2)+'\n').encode());print(json.dumps({'frozen':str(FROZEN),'receipt':str(p),'sha256':sha(p),'bound_files':len(bindings)}))
