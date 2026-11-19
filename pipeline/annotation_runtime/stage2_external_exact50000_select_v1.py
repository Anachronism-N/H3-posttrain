"""Freeze 50000 existing published caption references; no repeated caption/tar/media reads."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1")
RT=ROOT/'runtime'
BASE=ROOT/'external_ms_unified_clusters80k_20261027_v1'
OUT=ROOT/'external_stage2_vlm_50000_20261028_v1'


def encoded(x):return (json.dumps(x,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))+chr(10)).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def reference(p):
 raw=p.read_bytes();return dict(path=str(p),sha256=digest(raw))
def once(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 df=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df);assert p.read_bytes()==raw;return dict(path=str(p),sha256=digest(raw),bytes=len(raw))
def reserve_keys(reg):
 used=set()
 for j in reg['jobs']:
  ids=j.get('selected_source_ids')
  if ids is None:
   cp=Path(j['config_path'])
   if cp.exists():ids=json.loads(cp.read_bytes())['selected_source_ids']
   else:
    assert j.get('expected_running') is False and 'plan_ref' in j
    r=j['plan_ref'];raw=Path(r['path']).read_bytes();assert digest(raw)==r['sha256'];ids=[json.loads(x)['id'] for x in raw.splitlines() if x.strip()]
  used.update(i.casefold() for i in ids)
 for p in (ROOT/'ms_vlm_external_direct_inputs_20261028_v1/manifest.jsonl',ROOT/'ms_vlm_native_new4000_inputs_20261028_v1/manifest.jsonl',ROOT/'ms_vlm_native_next900_inputs_20261028_v1/manifest.jsonl'):
  if p.exists():used.update(json.loads(x)['id'].casefold() for x in p.read_bytes().splitlines() if x.strip())
 return used

def parse_batch(item):
 cp,worker,ticket,assignment=item
 if not cp.is_file():return [],None
 raw=cp.read_bytes();assert len(raw)<=8*2**20,'Bounded commit metadata required';c=json.loads(raw)
 if c.get('schema')!='ms_unified_novlm_batch_commit_v1' or c.get('edition')!='external_ms_unified_clusters80k_20261027_v1':return [],None
 assert c['worker']==worker and c['assignment_sha256']==ticket['assignment_sha256'] and c['prefetch_drained'] is True
 assert all(c.get(k) is False for k in ('semantic_approval','quality_approved','training_eligible'))
 package=c['package'];assert package['path']==str(cp.parent/'batch.tar') and package['sha256']==c['batch_sha256'] and type(package['bytes']) is int and package['bytes']>0
 files={f['path']:f for f in c['files']};assert len(files)==len(c['files']);rows=[];seen=set()
 for receipt in c['receipts']:
  s=receipt['source_row'];id_=s['id'];split=s['split'];assert id_.casefold() not in seen;seen.add(id_.casefold())
  if receipt['status']!='extracted_rule_draft_unapproved':continue
  assert re.fullmatch('[A-Za-z0-9_-]+',id_) and split in ('train','val','test') and receipt.get('vlm_called') is False
  assert all(receipt.get(k) is False for k in ('semantic_approval','quality_approved','training_eligible')) and receipt.get('needs_visual_revision') is True
  index=s['assignment_index'];assert type(index) is int and 0<=index<len(assignment) and {k:v for k,v in s.items() if k!='assignment_index'}==assignment[index]
  d=cp.parent/'cases'/split/id_;refs=[]
  for name in ('draft.json','fullframe_manifest.json'):
   f=files[str(d/name)];assert type(f['bytes']) is int and 0<f['bytes']<=2**20 and re.fullmatch('[a-f0-9]{64}',f['sha256']);refs.append(dict(path=f['path'],sha256=f['sha256']))
  rows.append(dict(id=id_,split=split,video_path=s['video_path'],rule_draft_ref=refs[0],fullframe_manifest_ref=refs[1],rich_text=''))
 return rows,dict(commit_ref=dict(path=str(cp),sha256=digest(raw)),worker=worker,source_ids=[r['id'] for r in rows])

def run():
 assert (OUT/'request.json').exists() and not (OUT/'manifest.jsonl').exists() and not (OUT/'allocation.json').exists(),'Existing exact allocation must not be overwritten or restarted'
 regpath=RT/'ms_vlm_postprocess_jobs_registry_v1.json';before=json.loads(regpath.read_bytes());excluded=reserve_keys(before);candidates=[];assign_refs=[]
 for worker in ('a100-extra-01','a100-extra-02'):
  tp=BASE/'assignments'/(worker+'.ticket.json');ticket=json.loads(tp.read_bytes());assert ticket['worker_id']==worker and ticket['source_count']==40000
  ap=Path(ticket['assignment_path']);raw=ap.read_bytes();assert digest(raw)==ticket['assignment_sha256'];assignment=[json.loads(x) for x in raw.splitlines() if x.strip()];assert len(assignment)==40000 and len({r['id'].casefold() for r in assignment})==40000;assign_refs.append(dict(ticket_ref=reference(tp),assignment_ref=dict(path=str(ap),sha256=digest(raw))))
  dirs=sorted((BASE/'results'/worker/'batches').iterdir(),key=lambda p:p.name)
  candidates.extend((p/'commit.json',worker,ticket,assignment) for p in dirs if p.is_dir())
 rows=[];seen=set();used_commits=[];begin=time.time()
 with ThreadPoolExecutor(max_workers=8) as pool:
  for offset in range(0,len(candidates),64):
   results=list(pool.map(parse_batch,candidates[offset:offset+64]))
   for values,commit in results:
    picked=[]
    for row in values:
     key=row['id'].casefold()
     if key in excluded or key in seen:continue
     seen.add(key);rows.append(row);picked.append(row['id'])
    if picked:commit['selected_source_ids']=picked;used_commits.append(commit)
   print(json.dumps(dict(event='external50k_selection_progress',source_rows=len(rows),scanned_batches=min(offset+64,len(candidates)),metadata_seconds=time.time()-begin,caption_payload_reads=0,archive_payload_reads=0)),flush=True)
   if len(rows)>=50000:break
 assert len(rows)>=50000,'Not enough completed unused published rows; never synthesize or reuse failed/claimed'
 fd=os.open(regpath.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);reg=json.loads(regpath.read_bytes());used=reserve_keys(reg);rows=[r for r in rows if r['id'].casefold() not in used];assert len(rows)>=50000,'Concurrent assignment reduced unused pool, stop safely';rows=rows[:50000];keys=[[r['split'],r['id']] for r in rows];assert len({r['id'].casefold() for r in rows})==50000
  manifest=once(OUT/'manifest.jsonl',b''.join(encoded(r) for r in rows));source_keys=once(OUT/'source_keys.jsonl',b''.join(encoded(k) for k in keys));phase='stage2_external03to06_exact50000_20261028';cp=RT/(phase+'_descriptor.json');descriptor=dict(schema='stage2_exact_external_source_reservation_v1',config_path=str(cp),phase=phase,node='EXTERNAL03to06',gpus=[],out=str(OUT),queue=str(ROOT/'queues'/phase),plan={k:manifest[k] for k in ('path','sha256')},targetcount=50000,input_count=50000,selected_source_ids=[r['id'] for r in rows],not_a_launch_config=True)
  dr=once(cp,encoded(descriptor));reg['jobs'].append(dict(node='EXTERNAL03to06',gpus=[],phase=phase,config_path=str(cp),plan_ref=descriptor['plan'],selected_source_ids=descriptor['selected_source_ids'],sources=50000,expected_running=False,worker_status='permanently_reserved_external_stage2_group',user_stage_authorized=True,task_handle='external03to06-50000-reserved',quality_approved=False,training_eligible=False))
  with tempfile.NamedTemporaryFile(dir=RT,prefix='.vlm-registry-',delete=False) as f:
   temp=Path(f.name);f.write(encoded(reg));f.flush();os.fsync(f.fileno())
  os.replace(temp,regpath);df=os.open(RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
  allocation=once(OUT/'allocation.json',encoded(dict(schema='external_stage2_exact50000_allocation_v1',observed_unix=time.time(),source_count=50000,manifest_ref=manifest,source_keys_ref=source_keys,source_group='EXTERNAL03to06',target_workers=['a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06'],registry_descriptor_ref=dr,assignment_refs=assign_refs,published_commit_refs=used_commits,registered_overlap=0,previous_input_pool_overlap=0,stage='visual_audit_then_single_caption_fusion',source_claims_created=0,caption_payloads_revalidated=False,source_reference_validation_at_model_boundary_required=True,archive_payload_reads=0,source_transfer_started=False,runtime_launch_ready=False,quality_approved=False,training_eligible=False)))
  print(json.dumps(dict(event='external50k_exact_list_frozen',manifest_ref=manifest,source_keys_ref=source_keys,allocation_ref=allocation,source_count=50000,claim_count=0,models_started=0)),flush=True)
if __name__=='__main__':run()
