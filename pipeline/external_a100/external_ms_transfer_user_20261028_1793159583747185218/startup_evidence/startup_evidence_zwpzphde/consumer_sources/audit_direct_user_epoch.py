"""Single-writer read-only audit of this exact new user-authorized epoch."""
import fcntl,hashlib,json,os,time,traceback
from pathlib import Path
from audit_results_checkpointed import check_batch
from storage_io import publish_file
E=Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_transfer_user_20261028_1793159583747185218");R=Path("/dockerdata/ms_a100-extra-01_20261026_HPDixx/external_ms_transfer_user_20261028_1793159583747185218_startfix01");A=R/'transfer_correctness_audit';A.mkdir(exist_ok=True);lock=(A/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
statepath=A/'state.json';old=json.loads(statepath.read_bytes()) if statepath.is_file() else {};seen=old.get('seen',{});done=set(old.get('checked_commits',[]));totals=old.get('totals_by_worker',{});tickets=[]
for w in ['a100-extra-01','a100-extra-02']:
 raw=(E/'assignments'/(w+'.ticket.json')).read_bytes();t=json.loads(raw);assignmentraw=Path(t['assignment_path']).read_bytes()
 if hashlib.sha256(assignmentraw).hexdigest()!=t['assignment_sha256']:raise ValueError('New audit exact assignment SHA')
 rows=[json.loads(l) for l in assignmentraw.splitlines()];assignment={r['split']+'/'+r['id']:r for r in rows};tickets.append((w,t,assignment));totals.setdefault(w,{})
log=(A/'progress.jsonl').open('a');started=time.time();last_publish=0;last_checkpoint=0

def emit(e,**k):
 row=dict(event=e,epoch=E.name,unix=time.time(),**k);raw=json.dumps(row,ensure_ascii=False);log.write(raw+'\n');log.flush();print(raw,flush=True)

def checkpoint():
 raw=json.dumps(dict(seen=seen,checked_commits=sorted(done),totals_by_worker=totals,observed_unix=time.time())).encode();temp=A/('state.'+str(os.getpid())+'.tmp')
 with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(temp,statepath);fd=os.open(A,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def publish_snapshot():
 local=A/('audit_snapshot_'+str(time.time_ns())+'.json');raw=json.dumps(dict(schema='ms_direct_user_transfer_full_content_audit_snapshot_v1',epoch=E.name,observed_unix=time.time(),verified_unique_sources=len(seen),checked_commit_count=len(done),totals_by_worker=totals,local_checkpoint_sha256=hashlib.sha256(statepath.read_bytes()).hexdigest(),actual_checks='package SHA and actual member SHA, exact current+original source fields, new zero-based GPU partition, fullframes/PTS/raw valid/support geometry, full frozen CPU draft recomputation, beat source/clock',diagnostic_not_production=True,semantic_approval=False,quality_approved=False,training_eligible=False,needs_visual_revision=True),sort_keys=True,indent=2).encode()+b'\n'
 with local.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 rec=publish_file(local,E/'technical_audit'/local.name);emit('shared_technical_audit_snapshot',publication=rec)

while time.time()-started<86400:
 try:
  for w,t,assignment in tickets:
   folder=Path(t['output_root'])/'batches'
   if not folder.is_dir():continue
   for d in sorted(folder.iterdir()):
    path=d/'commit.json'
    if str(path) in done or not path.is_file():continue
    summary,added=check_batch(path,t,assignment,seen)
    commit=json.loads(path.read_bytes())
    originalclaims={r['source_key']:r for r in commit['original_claims']}
    for row in commit['receipts']:
     r=row['source_row'];key=r['split']+'/'+r['id'];cp=Path(t['claims_root'])/r['split']/(r['id']+'.json');raw=cp.read_bytes();c=json.loads(raw)
     if hashlib.sha256(raw).hexdigest()!=originalclaims[key]['claim_sha256'] or c['source_row']!=r or c['authorization_sha256']!=t['authorization_sha256'] or c['edition']!=t['edition'] or c['original_worker']!=r['original_worker'] or c['original_assignment_index']!=r['original_assignment_index']:
      raise ValueError('Actual original-domain claim/authority/source binding mismatch')
    seen.update(added);done.add(str(path))
    for k,v in summary.items():totals[w][k]=totals[w].get(k,0)+v
    emit('new_epoch_full_contract_audit_passed',worker=w,commit_path=str(path),summary=summary)
    if time.monotonic()-last_checkpoint>=30:checkpoint();last_checkpoint=time.monotonic()
  checkpoint()
  if time.time()-last_publish>=900:
   publish_snapshot();last_publish=time.time()
  closed=True
  for w,t,assignment in tickets:
   for g in range(4):
    p=Path(t['output_root'])/('gpu'+str(g)+'_queue_summary.json')
    if not p.is_file():closed=False;continue
    s=json.loads(p.read_bytes())
    if s['remaining_preallocated_sources']!=0 or s['counts']['publish_backlog']!=0:closed=False
  if closed:
   expected=sum(t['source_count'] for w,t,a in tickets)
   if len(seen)!=expected:emit('new_epoch_closed_audit_source_gap',audited=len(seen),expected=expected)
   else:publish_snapshot();emit('new_epoch_finite_audit_complete',sources=len(seen),totals=totals);break
  time.sleep(30)
 except OSError as error:emit('new_epoch_audit_storage_wait',error=str(error),unverified_not_counted=True);checkpoint();time.sleep(15)
 except Exception as error:emit('new_epoch_contract_failure',error=str(error),traceback=traceback.format_exc(),no_claim_or_model_retry=True);checkpoint();raise
log.flush();os.fsync(log.fileno());log.close()
