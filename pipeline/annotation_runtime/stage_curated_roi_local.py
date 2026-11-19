import json,hashlib,os,shutil
from pathlib import Path
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
original=json.loads((RT/'curated_roi_modelref_n8_v1.json').read_text());dest=Path('/opt/curated_roi_models_v1');dest.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for key in ('detector','pose'):
 v=original['files'][key];source=Path(v['path']);p=dest/source.name
 if not p.exists():
  with source.open('rb') as src,p.open('xb') as out:
   shutil.copyfileobj(src,out,8*1024*1024);out.flush();os.fsync(out.fileno())
 assert not p.is_symlink() and sha(p)==v['sha256'];s=p.stat();v['path']=str(p);v['stat']={'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'device':s.st_dev,'inode':s.st_ino}
original['sha256']=hashlib.sha256((json.dumps(original['files'],sort_keys=True,separators=(',',':'),allow_nan=False)+chr(10)).encode()).hexdigest()
p=RT/'curated_roi_local_n8_v1.json';raw=(json.dumps(original,indent=2)+chr(10)).encode()
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'same_weights_fullSHA_verified':True}))
