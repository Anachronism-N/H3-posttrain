import hashlib,json
from pathlib import Path
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie');RT=Path(__file__).resolve().parent;BASE=ROOT/'H3-posttrain/scripts_omnidance/exp/ms';OUT=RT/'source_curated_consumer_v1';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[RT/'curated_pipeline.py']+[BASE/n for n in ['ms_bar_annotation.py','ms_visual_batch.py','ms_caption_rewrite.py','ms_visual_observe.py','ms_optimize.py','ms_landmarks.py','ms_hand_roi.py','dwpose_full_extract.py','ms_dance_style.py','vlm_label.py']]+list((BASE/'motionscript_v2').glob('*.py'));bindings={}
for p in files:
 raw=p.read_bytes();compile(raw,str(p),'exec');dest=OUT/p.name if p.parent==RT else OUT/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('xb') as f:f.write(raw)
 bindings[str(dest)]=sha(dest)
manifest=RT.parent/'curated222939_inputs_v1/manifest.summary.json';summary=json.loads(manifest.read_text());bindings[str(manifest)]=sha(manifest)
for p in (ROOT/'diff/logs_omnidance/exp/MS/vlm').glob('labels_s*.jsonl'):bindings[str(p)]=sha(p)
r={'producer':{'path':str(OUT/'curated_pipeline.py'),'sha256':sha(OUT/'curated_pipeline.py')},'ms_source':str(OUT),'files':bindings,'manifest_summary':{'path':str(manifest),'sha256':sha(manifest)},'full_manifest':summary['manifest'],'count':summary['total']}
with (OUT/'freeze.receipt.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps({'frozen':str(OUT),'files':len(bindings),'sha256':sha(OUT/'freeze.receipt.json')}))
