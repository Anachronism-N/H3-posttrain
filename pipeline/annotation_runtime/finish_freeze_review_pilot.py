import hashlib,json
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');B=Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms');O=RT/'source_independent_review_pilot_v1'
files=[B/n for n in ['ms_batch_review.py','ms_dance_style.py','vlm_label.py','ms_bar_annotation.py','ms_caption_rewrite.py','ms_visual_batch.py','ms_visual_observe.py','ms_optimize.py','ms_landmarks.py','ms_hand_roi.py','dwpose_full_extract.py']]+list((B/'motionscript_v2').glob('*.py'));refs={}
for p in files:
 raw=p.read_bytes();compile(raw,str(p),'exec');target=O/p.relative_to(B);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert target.read_bytes()==raw,'source changed '+str(target)
 else:
  with target.open('xb') as f:f.write(raw)
 refs[str(target)]=hashlib.sha256(raw).hexdigest()
value={'files':refs,'CPU_tests_passed':26,'GPU_model_loaded':False};p=O/'freeze.receipt.json';raw=(json.dumps(value,indent=2)+chr(10)).encode()
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'frozen':str(O),'files':len(refs),'receipt_sha256':hashlib.sha256(raw).hexdigest()}))
