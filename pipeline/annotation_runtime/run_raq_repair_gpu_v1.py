"""Run frozen finite repair with immutable-pixel and stable-read verification."""
import hashlib,importlib.util,json,stat,sys
from pathlib import Path
RT=Path(__file__).resolve().parent;SOURCE=RT/'source_repair_gpu_v1/curated_repair_queue_v1.py'
def stable_sha(path):
 p=Path(path);before=p.stat()
 if not stat.S_ISREG(before.st_mode):raise ValueError('repair requires regular evidence file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 after=p.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('evidence changed while reading: '+str(p))
 return h.hexdigest()
assert stable_sha(SOURCE)=='853cf9e152738a9a47123500226183ce2b9ab803e28b13ac2797ed476e904347'
spec=importlib.util.spec_from_file_location('frozen_curated_finite_repair',SOURCE);repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair);repair.sha=stable_sha
original_verify=repair.verify_job
def verify_job(job):
 original_verify(job)
 reference=job.get('refs',{}).get('demo_manifest')
 if reference:
  manifest=repair.checked(reference)
  for f in manifest['frames']:
   if stable_sha(f['file'])!=f['sha256']:raise repair.SourceChanged('original or added repair frame changed')
  for part in ('left_hand','right_hand'):
   for f in manifest.get('hand_roi_evidence',{}).get(part,{}).get('crops',[]):
    if stable_sha(f['path'])!=f['sha256']:raise repair.SourceChanged('repair hand ROI changed')
repair.verify_job=verify_job
if __name__=='__main__':
 result=repair.main(sys.argv[1:]);print(json.dumps(result,ensure_ascii=False),flush=True)
