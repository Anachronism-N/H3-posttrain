"""One write-once shared publication index snapshot; never reruns or modifies consumers."""
from pathlib import Path
import concurrent.futures,datetime,hashlib,io,json,os,shlex,subprocess,tarfile,uuid
LOCAL=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
CURRENT=ROOT/'external_ms_unified_clusters80k_20261027_v1';OLD=ROOT/'external_ms_unified_run_20261027_v1'
WORKERS=['a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06']
FLAGS=dict(diagnostic_not_production=True,semantic_approval=False,quality_approved=False,training_eligible=False,needs_visual_revision=True)
CAP=192*2**20

def encode(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+chr(10)).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,cap=64*2**20):
 with path.open('rb') as f:
  size=os.fstat(f.fileno()).st_size
  if size>cap:raise ValueError('snapshot input exceeds bounded cap:'+str(path))
  raw=f.read(cap+1)
 if len(raw)!=size:raise ValueError('snapshot input length differs:'+str(path))
 return raw
def get(path,cap=64*2**20):
 raw=read(path,cap)
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL/partial JSON retained:'+str(path))
 return json.loads(raw),raw
created=datetime.datetime.now(datetime.timezone.utc).isoformat()
name='shared_results_snapshot_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
parent=CURRENT/'results/a100-extra-03'
if not LOCAL.is_dir() or not parent.is_dir() or parent.resolve(strict=True)!=parent:raise ValueError('authorized snapshot parent unavailable')
work=LOCAL/name;work.mkdir(exist_ok=False)
files={};summaries=[];global_ids=set();total=0;batchrecords=[];diagnosticrecords=[];pending_deep=0

def add(name,raw):
 if name in files:raise ValueError('duplicate snapshot member')
 files[name]=raw
 if sum(len(b) for b in files.values())>CAP:raise ValueError('bounded metadata snapshot budget exceeded')
for edition,sname,aname in [(OLD,'unified_verified_commit_cursor.json','unified_payload_audit_ledger.json'),(CURRENT,'rollover_verified_commit_cursor.json','rollover_payload_audit_ledger.json')]:
 state,sraw=get(LOCAL/'logs'/sname);audit,araw=get(LOCAL/'logs'/aname);ledger_capture_time=datetime.datetime.now(datetime.timezone.utc).isoformat();assignments={};tickets={};source_counts={};status_counts={};deep_counts={};sources=[];batchkeys={};pending=[]
 add('evidence/'+sname,sraw);add('evidence/'+aname,araw)
 for worker in WORKERS:
  ticket,traw=get(edition/'assignments'/(worker+'.ticket.json'),2**20);mraw=read(Path(ticket['assignment_path']))
  if bytes([0]) in mraw or not mraw.endswith(bytes([10])) or sha(mraw)!=ticket['assignment_sha256'] or ticket['worker_id']!=worker:raise ValueError('assigned source identity/SHA differs')
  rows=[json.loads(line) for line in mraw.split(bytes([10]))[:-1]]
  if len(rows)!=ticket['source_count']:raise ValueError('assignment source count differs')
  assignments[worker]=rows;tickets[worker]=ticket
  add('assignments/'+edition.name+'/'+worker+'.ticket.json',traw);add('assignments/'+edition.name+'/'+worker+'.jsonl',mraw)
  if edition==CURRENT:
   qraw=read(edition/(worker+'.queue.json'),2**20)
   if bytes([0]) in qraw or not qraw.endswith(bytes([10])):raise ValueError('NUL/partial queue retained')
   queue=json.loads(qraw);ref=next(v for v in queue['tickets'] if v['path']==str(edition/'assignments'/(worker+'.ticket.json')))
   if queue['worker_id']!=worker or ref['sha256']!=sha(traw):raise ValueError('own queue original ticket differs')
   add('assignments/'+edition.name+'/'+worker+'.queue.json',qraw)
  source_counts[worker]=0;status_counts[worker]={};deep_counts[worker]=0
 for commitpath,record in state['commits'].items():
  path=Path(commitpath)
  if not path.is_relative_to(edition/'results') or path.name!='commit.json' or not record['sources']:raise ValueError('commit scope or count invalid')
  key=sha(commitpath.encode())[:24];batchkeys[commitpath]=key
  if commitpath in audit['commits']:
   av=audit['commits'][commitpath];archive_sha=av['archive_SHA256'];deep_time=av['verified_at'];archive_bytes=None
   if record['sources']!=av['sources']:raise ValueError('commit source count differs from audit')
  else:
   c,raw=get(path,2**20)
   if sha(raw)!=record['sha256']:raise ValueError('new unaudited commit changed')
   archive_sha=c['batch_sha256'];pub=next(v for v in c['publications'] if v['path']==str(path.parent/'batch.tar'));archive_bytes=pub['receipt']['bytes']
   if archive_sha!=pub['sha256'] or archive_sha!=pub['receipt']['sha256']:raise ValueError('unaudited archive publication binding differs')
   deep_time=None
  batchrecords.append(dict(batch_id=key,edition=edition.name,commit_path=commitpath,commit_sha256=record['sha256'],archive_path=str(path.parent/'batch.tar'),archive_sha256=archive_sha,archive_bytes=archive_bytes,archive_size_authority='referenced commit.json publications[path=archive_path].receipt.bytes' if archive_bytes is None else 'verified new commit publication receipt',source_count=record['sources'],commit_verified_at=record['verified_at'],deep_payload_verified=commitpath in audit['commits'],deep_verified_at=deep_time))
 for worker in WORKERS:
  for identity,item in state['sources'][worker].items():
   if identity in global_ids:raise ValueError('committed source repeated across workers or editions:'+identity)
   global_ids.add(identity);idx=item['assignment_index'];rows=assignments[worker]
   if type(idx) is not int or not 0<=idx<len(rows):raise ValueError('original assignment index invalid')
   row=rows[idx]
   if identity!=row['split']+'/'+row['id']:raise ValueError('source row/ID differs')
   path=Path(item['commit_path']);gpu=idx%4
   if path.parent.parent.name!='gpu'+str(gpu) or worker not in path.parts:raise ValueError('source worker/GPU publication path differs')
   status=item['status']
   if status not in ['partial','draft_generated','needs_regeneration','failed']:raise ValueError('unrecognized sealed source status')
   av=audit['sources'][worker].get(identity);deep=bool(av and av['status']==status and av['commit']==str(path) and str(path) in audit['commits'])
   if av and not deep:raise ValueError('source audit status/commit mismatch')
   names=['receipt.json'] if status=='failed' else ['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']
   # These exact archive members were verified by the single-pass audit, or required by validated commit receipts.
   locators={n:dict(batch_id=batchkeys[str(path)],archive_member='cases/'+identity+'/'+n) for n in names}
   entry=dict(edition=edition.name,worker=worker,gpu=gpu,assignment_index=idx,assignment_sha256=tickets[worker]['assignment_sha256'],source_row=row,status=status,frames=item.get('frames',0),events=item.get('events',0),deep_payload_verified=deep,artifacts=locators,**FLAGS)
   sources.append(entry);source_counts[worker]+=1;status_counts[worker][status]=status_counts[worker].get(status,0)+1;deep_counts[worker]+=int(deep);total+=1;pending_deep+=int(not deep)
   if status in ['failed','needs_regeneration']:diagnosticrecords.append(entry)
  pending.extend(dict(worker=worker,assignment_index=i,id=row['id'],split=row['split'],status='not_in_reliable_commit_snapshot',not_a_transfer_or_claimable_manifest=True) for i,row in enumerate(assignments[worker]) if row['split']+'/'+row['id'] not in state['sources'][worker])
 add('sources/'+edition.name+'.jsonl',b''.join(encode(v) for v in sources))
 add('reconciliation/'+edition.name+'_not_yet_committed.jsonl',b''.join(encode(v) for v in pending))
 summaries.append(dict(edition=edition.name,ledger_capture_time=ledger_capture_time,authorized_sources=sum(len(v) for v in assignments.values()),reliably_committed_source_results=len(sources),deep_verified_source_results=sum(deep_counts.values()),pending_deep_audit=len(sources)-sum(deep_counts.values()),uncommitted_result_count=len(pending),worker_source_counts=source_counts,worker_status_counts=status_counts,worker_deep_verified_counts=deep_counts,commits=len(state['commits']),snapshot_commit_cursor_sha256=sha(sraw),snapshot_audit_ledger_sha256=sha(araw),is_terminal=len(pending)==0,terminal_quality_or_training_approval=False))
add('batches.jsonl',b''.join(encode(v) for v in batchrecords));add('failed_and_unknown.jsonl',b''.join(encode(v) for v in diagnosticrecords))
PROBE=r'''
from pathlib import Path
import datetime,json,sys,hashlib
root=Path(sys.argv[1]);worker=sys.argv[2];out=[]
for gpu in range(4):
 p=root/'logs'/('rollover_gpu'+str(gpu)+'_progress.json');b=p.read_bytes()
 if bytes([0]) in b or not b.endswith(bytes([10])):raise ValueError('NUL/partial progress retained')
 v=json.loads(b);leasepath=root/'gpu_locks'/('physical_gpu'+str(gpu)+'.newepoch.lease.json');lb=leasepath.read_bytes()
 if bytes([0]) in lb or not lb.endswith(bytes([10])):raise ValueError('NUL/partial lease evidence retained')
 lease=json.loads(lb);pid=v['pid'];proc=Path('/proc')/str(pid);f=(proc/'stat').read_text().rsplit(') ',1)[1].split();identity=(int(f[19])==v['start_ticks']==lease['start_ticks'] and pid==lease['pid'] and v['worker']==lease['worker']==worker and v['assignment_sha256']==lease['assignment_sha256'])
 out.append(dict(gpu=gpu,progress=v,observed_PID_start_ticks=int(f[19]),original_lease_identity_matches=identity,lease_snapshot=lease,lease_sha256=hashlib.sha256(lb).hexdigest(),evidence_only_not_shared_artifact_locators=True,retired=False,no_resume_proven=False))
print(json.dumps(dict(worker=worker,role=root.name,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),lanes=out,consumer_sha256=hashlib.sha256((root/'unified_lane_rollover_v1.py').read_bytes()).hexdigest(),transport_sha256=hashlib.sha256((root/'unified_access_transport_rollover_v1.py').read_bytes()).hexdigest(),claims_not_read_or_changed=True)),flush=True)
'''
NODES=[('a100-extra-03','launcher',None),('a100-extra-04','worker-0','11.254.248.52'),('a100-extra-05','worker-1','11.239.106.11'),('a100-extra-06','worker-2','11.239.84.43')]
def probe(node):
 worker,role,host=node;args=['python3','-B','-c',PROBE,str(LOCAL.parent/role),worker];cmd=args if host is None else ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026','-o','ConnectTimeout=8',host,shlex.join(args)]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 if r.returncode:raise RuntimeError('readonly continuation evidence failed:'+worker+':'+r.stderr[-1000:])
 if bytes([0]) in r.stdout.encode() or not r.stdout.endswith(chr(10)):raise ValueError('partial node evidence retained')
 return json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:nodes=list(ex.map(probe,NODES))
for n in nodes:add('continuation/'+n['worker']+'.json',encode(n))
codefiles=['unified_lane_rollover_v1.py','unified_access_transport_rollover_v1.py','unified_access_transport_v3.py','fullvideo_validated_helpers_ed283d684.py','rollover_boundary_waiter_v1.py','monitor_environment_v6.py','rollover_goal_supervisor_v1.py','audit_rollover_output_packages_once_v2.py','watch_authorized_queue_events_v2.py']+['unified_code_snapshot_v1/'+n for n in ['ms_unified_novlm_v1.py','ms_fullframe_pose2d_draft_v2.py','ms_fullframe_pose2d_draft_v1.py','ms_pose2d_draft.py','dwpose_full_extract.py']]
for rel in codefiles:
 raw=read(LOCAL/rel,2**20);compile(raw,str(LOCAL/rel),'exec');add('consumer_code/'+rel,raw)
if any(n['consumer_sha256']!=sha(files['consumer_code/unified_lane_rollover_v1.py']) or n['transport_sha256']!=sha(files['consumer_code/unified_access_transport_rollover_v1.py']) or not all(l['original_lease_identity_matches'] for l in n['lanes']) for n in nodes):raise ValueError('actual consumer source or lease snapshot differs')
for rel in ['unified4000_artifact_acceptance.json','monitor_identity_v6.json','rollover_goal_supervisor_identity.json','rollover_payload_audit_identity_v2.json','continual_queue_monitor_cursor.json','rollover_goal_supervisor_latest.json','rollover_payload_audit_latest.json']:
 raw=read(LOCAL/'logs'/rel,2**20)
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL/partial evidence snapshot')
 add('evidence/'+rel,raw)
references=[]
for path in [CURRENT/'results/a100-extra-03/integration_handoff_20261027T093320Z_7dbb8474/commit.json']:
 value,raw=get(path,2**20);references.append(dict(path=str(path),sha256=sha(raw),purpose='previous frozen source/resource handoff'))
for label,path in [('batchRPC_candidate',CURRENT/'results/a100-extra-04/publication_experiments/batchrpc20_candidate_v2_topo_role_20261027_once01/frozen_handoff_20261027T1319Z/commit.json'),('BeatThis_refusal_candidate',CURRENT/'results/a100-extra-03/beat_rejection_evidence_handoff_20261027T1341Z/commit.json'),('transfer_drain_blocked',CURRENT/'results/a100-extra-03/transfer_drain_blocked_20261027T1432Z.json')]:
 _,raw=get(path,2**20);references.append(dict(path=str(path),sha256=sha(raw),purpose=label))
add('continuation/shared_handoff_references.json',encode(references))
summary=dict(schema='ms_shared_results_snapshot_v1',created=created,root=str(ROOT),scope='this session 03–06 old4000 plus current80000 committed MotionScript results; no01/02/pilot/diagnostic double-counting',editions=summaries,total_unique_reliably_committed_results=total,total_deep_verified_results=total-pending_deep,pending_deep_audit=pending_deep,failed_or_unknown_records=len(diagnosticrecords),batches=len(batchrecords),members_and_locators='sources index refers by batch_id to batches.jsonl archive_path+archive_sha256 and exact archive_member; paths within provenance are evidence not artifact locators',source_outputs_already_on_shared_root=True,result_packages_not_recopied=True,consumer_active_nodes=[dict(worker=n['worker'],checked_at=n['checked_at'],PID=[l['progress']['pid'] for l in n['lanes']],status=[l['progress']['status'] for l in n['lanes']]) for n in nodes],source_media_or_weights_packaged=False,claims_copied_or_recreated=False,source_reexecuted=False,production_transfers_created=0,atomic_transfer_contract_status='central validation pending; original queue continues; no production transfer',no_resume_or_retirement_proven=False,full80000_goal_complete=False,**FLAGS)
add('SUMMARY.json',encode(summary))
notes='''# Shared results snapshot

All annotation payload packages referenced here already reside below the shared ROOT. This is a write-once metadata/code/evidence snapshot, not a second generation or input-media archive. Current workers continue their original immutable assignments and normal commit-last publication.

Open sources/<edition>.jsonl, then join artifacts[name].batch_id to batches.jsonl. The true artifact locator is the pair archive_path and archive_member; do not treat original local provenance paths as newly exported files. For each successful/partial/needs_regeneration result the verified archive contains pose.npz, fullframe_manifest.json, beats.json, draft.json and receipt.json; sealed failure results contain only their real failure receipt and never synthetic pose/beats/draft placeholders. Each batch references the original shared commit.json SHA and archive SHA. Original commit publications are authoritative for individual archive byte lengths and per-member SHA/length. Already verified archives are not rehashed by this snapshot; the included single-pass audit ledger records the prior actual archive/member/source validation. New commit-only results explicitly have deep_payload_verified=false until audit catches up.

The uncommitted reconciliation is NOT an unclaimed or transferable source list. It includes possible in-flight/claimed/prefetched sources and must not be scheduled. The central atomic transfer contract remains pending receive/recovery/global reconciliation acceptance; no transfer records, retirement or no-resume proof were created. Existing claims remain untouched.

The continuation files include local lease/progress snapshots for PID/start identity evidence only, not as shared annotation file locators. All captured code bytes are inside this archive with SHA and byte sizes in manifest.json; current consumer and relay SHA were rechecked on each node. Old4000 terminal acceptance is included. The current80k queue is explicitly nonterminal; no fabricated final acceptance is created. Failures/unknown remain unapproved and not trainable. Candidate batchRPC and refused-array diagnostics are referenced as test-only, not deployed or proven root-cause fixes for older sources.

Snapshot payload metadata.tar is published and read back completely first. SUMMARY/manifest references are then published write-once, and snapshot commit.json last, with complete SHA/length and local/shared archive-member verification. It does not reclaim running worker caches or change current publisher. New results continue at their original shared paths after the snapshot cutoff; later snapshots must use new unique namespaces, never overwrite this one.
'''
add('README.md',notes.encode())
manifest=dict(schema='ms_shared_results_snapshot_manifest_v1',created=created,members={n:dict(bytes=len(b),sha256=sha(b)) for n,b in files.items()},total_metadata_payload_bytes=sum(map(len,files.values())),scope='no model weights, videos, audio media copies, GPU work cache or claim records',summary_counts=dict(unique_source_results=total,deep_verified=total-pending_deep),flags=FLAGS)
manifest_raw=encode(manifest);files['manifest.json']=manifest_raw
archive=work/'metadata.tar'
with tarfile.open(archive,mode='x',format=tarfile.USTAR_FORMAT) as tar:
 for member,raw in sorted(files.items()):
  item=tarfile.TarInfo(member);item.size=len(raw);item.mode=0o444;item.mtime=0;tar.addfile(item,io.BytesIO(raw))
if archive.stat().st_size>CAP:raise ValueError('snapshot archive budget exceeded')
def hashfile(path):
 h=hashlib.sha256();count=0
 with path.open('rb') as f:
  expected=os.fstat(f.fileno()).st_size
  while b:=f.read(4*2**20):h.update(b);count+=len(b)
  if count!=expected or os.fstat(f.fileno()).st_size!=count:raise ValueError('opened FD read length differs')
 return h.hexdigest(),count
archive_sha,archive_bytes=hashfile(archive)
with tarfile.open(archive,mode='r:') as tar:
 if len(tar.getmembers())!=len(files) or set(tar.getnames())!=set(files):raise ValueError('snapshot tar member mismatch')
 for member in tar:
  if not member.isfile() or member.issym() or member.islnk() or member.name not in files:raise ValueError('unsafe snapshot member')
  raw=tar.extractfile(member).read();expected=files[member.name]
  if len(raw)!=len(expected) or sha(raw)!=sha(expected):raise ValueError('snapshot member content differs')
out=parent/name;out.mkdir(exist_ok=False)
def publish(name,raw=None,local=None):
 path=out/name
 with path.open('xb') as target:
  if local:
   with local.open('rb') as source:
    while chunk:=source.read(4*2**20):target.write(chunk)
  else:target.write(raw)
  target.flush();os.fsync(target.fileno())
 os.chmod(path,0o444);actual,length=hashfile(path);expected_sha,expected_bytes=(archive_sha,archive_bytes) if local else (sha(raw),len(raw))
 if actual!=expected_sha or length!=expected_bytes:raise ValueError('shared publication readback differs')
 fd=os.open(out,os.O_DIRECTORY|os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return dict(path=str(path),sha256=actual,bytes=length)
publications=[publish('metadata.tar',local=archive),publish('SUMMARY.json',raw=files['SUMMARY.json']),publish('manifest.json',raw=manifest_raw)]
# Check the received shared archive members and locators, not nonexistent scattered files.
with tarfile.open(out/'metadata.tar',mode='r:') as tar:
 if set(tar.getnames())!=set(files):raise ValueError('received shared archive member set differs')
 for member in tar:
  raw=tar.extractfile(member).read()
  if sha(raw)!=sha(files[member.name]) or member.size!=len(files[member.name]):raise ValueError('received shared tar member differs')
commit=dict(schema='ms_shared_results_snapshot_commit_v1',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),publications=publications,archive_sha256=archive_sha,archive_bytes=archive_bytes,source_result_count=total,deep_verified_source_count=total-pending_deep,origin_payloads_already_shared=True,artifact_locator_kind='archive_path+archive_member',claims_unchanged=True,weights_media_and_work_cache_packaged=False,source_reexecution=False,full_goal_complete=False,**FLAGS)
receipt=publish('commit.json',raw=encode(commit))
print(json.dumps(dict(shared_snapshot=str(out),archive_path=str(out/'metadata.tar'),archive_sha256=archive_sha,archive_bytes=archive_bytes,commit_path=receipt['path'],commit_sha256=receipt['sha256'],source_result_count=total,deep_verified_source_count=total-pending_deep,pending_deep_audit=pending_deep,editions=summaries,statistical_cutoff=created,archive_and_member_SHA_readback_verified=True,true_shared_tar_member_locators=True,no_new_GPU_work_or_media_copy=True,claims_unchanged=True)),flush=True)
