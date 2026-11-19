import hashlib,json,os,shutil,socket
from pathlib import Path
SOURCE=Path('/apdcephfs_gy4/share_302533218/cedricnie/models/Qwen3-VL-8B-Instruct')
DEST=Path('/opt/models/Qwen3-VL-8B-Instruct')
RECEIPT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/reviewer8b_stage_'+socket.gethostname()+'.json')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return h.hexdigest()
files=sorted(p for p in SOURCE.iterdir() if p.is_file() and not p.name.startswith('.'))
assert len([p for p in files if p.name.endswith('.safetensors')])==4
assert not DEST.is_symlink()
DEST.mkdir(parents=True,exist_ok=True)
assert not ({p.name for p in DEST.iterdir() if p.is_file()}-{p.name for p in files})
proof={}
for p in files:
 target=DEST/p.name
 expected=digest(p)
 if target.exists():
  assert not target.is_symlink() and target.stat().st_size==p.stat().st_size and digest(target)==expected, str(target)
 else:
  with p.open('rb') as src,target.open('xb') as dst:
   shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
   if hasattr(os,'posix_fadvise'):
    os.posix_fadvise(src.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    os.posix_fadvise(dst.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  assert digest(target)==expected,str(target)
 st=target.stat();proof[p.name]={'path':str(target),'sha256':expected,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'device':st.st_dev}
 print(json.dumps({'verified_file':p.name,'bytes':st.st_size}),flush=True)
value={'complete':True,'model':'Qwen3-VL-8B-Instruct','destination':str(DEST),'files':proof,'source_path':str(SOURCE),'full_copy_SHA_verified':True,'license_declaration':'Apache-2.0 from local README','GPU_model_loaded':False}
raw=(json.dumps(value,indent=2)+'\n').encode()
if RECEIPT.exists():assert json.loads(RECEIPT.read_text())==value
else:
 with RECEIPT.open('xb') as f:f.write(raw)
print(json.dumps({'complete':True,'receipt':str(RECEIPT),'sha256':hashlib.sha256(raw).hexdigest()}),flush=True)
