import hashlib,json
from pathlib import Path
EXP=Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1");source=EXP/'plan_full12000.jsonl'
raw=source.read_bytes()
if hashlib.sha256(raw).hexdigest()!='ab7fd0921021d9b414b017dc06fc7111bc9df35d125bd1ef3aac8b762e392353':raise ValueError('Full plan changed')
lines=raw.splitlines(keepends=True)[:50]
if len(lines)!=50 or len({json.loads(x)['id'] for x in lines})!=50:raise ValueError('Invalid50 selection')
path=EXP/'plan_validation50.jsonl';data=b''.join(lines)
if path.exists():
 if path.read_bytes()!=data:raise ValueError('Refusing changed validation input')
else:
 with path.open('xb') as f:f.write(data)
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'sources':50,'original_window_policy_preserved':True}))
