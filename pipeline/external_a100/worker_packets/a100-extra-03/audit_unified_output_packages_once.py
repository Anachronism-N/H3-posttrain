"""Single-pass read-only output-package audit; no video/model/claim access."""
from pathlib import Path
import datetime,hashlib,io,json,os,sys,tarfile,time
import numpy as np
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
sys.path.insert(0,str(ROOT/'unified_code_snapshot_v1'))
import ms_fullframe_pose2d_draft_v2 as rules
WORKERS=['a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06']
FLAGS=rules.FLAGS
LF=chr(10)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
 tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,allow_nan=False);f.write(LF);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def getjson(raw):
 if bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('NUL or partial JSON;preserve and report')
 return json.loads(raw)
ledger_path=ROOT/'logs/unified_payload_audit_ledger.json'
ledger=getjson(ledger_path.read_bytes()) if ledger_path.exists() else {'commits':{},'sources':{w:{} for w in WORKERS}}
rows={};ticketshas={}
for w in WORKERS:
 t=getjson((P/'assignments'/(w+'.ticket.json')).read_bytes());raw=Path(t['assignment_path']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=t['assignment_sha256'] or bytes([0]) in raw or not raw.endswith(bytes([10])):raise ValueError('assignment bytes mismatch')
 rows[w]=[json.loads(x) for x in raw.split(bytes([10]))[:-1]];ticketshas[w]=t['assignment_sha256']
started=time.monotonic();identity={'pid':os.getpid(),'started':now(),'scope':'each published batch archive read only once;no inputs/models','claims_modified':False}
with (ROOT/'logs/unified_payload_audit_identity.json').open('x') as f:json.dump(identity,f);f.write(LF)
print(json.dumps({'event':'payload_audit_started',**identity}),flush=True)
while time.monotonic()-started<3*86400:
 try:
  state=getjson((ROOT/'logs/unified_verified_commit_cursor.json').read_bytes())
  todo=[key for key in state['commits'] if key not in ledger['commits']]
  for key in todo:
   path=Path(key);raw=path.read_bytes();commit=getjson(raw);w=commit['worker']
   if w not in WORKERS or commit['edition']!=P.name or commit['assignment_sha256']!=ticketshas[w]:raise ValueError('commit scope changed')
   if hashlib.sha256(raw).hexdigest()!=state['commits'][key]['sha256']:raise ValueError('commit bytes changed after validation')
   expected={str(Path(p['path']).relative_to(path.parent)):p['sha256'] for p in commit['publications']}
   archive=path.parent/'batch.tar';expected_archive=expected.get('batch.tar')
   if expected_archive!=commit['batch_sha256']:raise ValueError('archive identity mismatch')
   with archive.open('rb') as stream:
    before=os.fstat(stream.fileno());hasher=hashlib.sha256();chunks=[];count=0
    if before.st_size>8*2**30:raise ValueError('published archive exceeds ticket bounded size')
    while b:=stream.read(4*1024*1024):hasher.update(b);chunks.append(b);count+=len(b)
    if count!=before.st_size or os.fstat(stream.fileno()).st_size!=count or hasher.hexdigest()!=expected_archive:raise ValueError('published archive bytes/SHA differ')
   content=b''.join(chunks);del chunks
   payloads={}
   with tarfile.open(fileobj=io.BytesIO(content),mode='r:') as tar:
    for member in tar:
     if not member.isfile() or member.issym() or member.islnk() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('archive member unsafe')
     if member.name in payloads or member.name not in expected:raise ValueError('duplicate/unexpected archive member')
     if member.size>64*2**20:raise ValueError('individual diagnostic payload exceeds64MiB independent audit budget')
     b=tar.extractfile(member).read()
     if len(b)!=member.size or hashlib.sha256(b).hexdigest()!=expected[member.name]:raise ValueError('archive member bytes differ from verified publication')
     payloads[member.name]=b
   if set(payloads)!=set(expected)-{'batch.tar'}:raise ValueError('archive five-files/case set mismatch')
   staged={};frames=0;events=0;partial=0;failures=0
   for receipt in commit['receipts']:
    if receipt['status']=='existing_claim_not_reassigned':continue
    index=receipt['assignment_index'];row=receipt['source_row']
    if index%4!=commit['gpu'] or row!=rows[w][index]:raise ValueError('source row/lane identity mismatch')
    sourcekey=row['split']+'/'+row['id'];relative='cases/'+sourcekey
    if sourcekey in ledger['sources'][w] or sourcekey in staged:raise ValueError('source audited in two commits')
    if receipt['status']=='failed_sealed_no_retry':failures+=1;staged[sourcekey]={'status':'failed','commit':key};continue
    names=[relative+'/'+n for n in ['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']]
    if any(n not in payloads for n in names):raise ValueError('source lacks five real output payloads')
    pose_raw=payloads[names[0]];manifest=getjson(payloads[names[1]]);beats=getjson(payloads[names[2]]);draft=getjson(payloads[names[3]]);case_receipt=getjson(payloads[names[4]])
    bound={'id':row['id'],'split':row['split'],'npz_path':str(path.parent/names[0]),'fullframe_manifest_path':str(path.parent/names[1])}
    with np.load(io.BytesIO(pose_raw),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    if data['schema'].item()!='ms_fullvideo_pose2d_cache_v1' or data['bbox'].shape!=(len(data['times']),4):raise ValueError('new cache/bbox schema mismatch')
    if hashlib.sha256(pose_raw).hexdigest()!=manifest['npz_sha256']:raise ValueError('manifest pose SHA mismatch')
    mask=rules.validate_observations(bound,manifest,data)
    if rules.fullframe_binding(manifest,data)!=draft['source']['fullframe_binding']:raise ValueError('rule draft not bound to actual new fullframes')
    if case_receipt['source_row']!=row or case_receipt['frame_count']!=len(data['times']):raise ValueError('receipt source/frames mismatch')
    if manifest['video_path']!=row['video_path'] or not manifest['duration_seconds']>0 or manifest['duration_seconds']!=float(manifest['metadata_provenance']['actual_probe']['streams'][0]['duration']):raise ValueError('video duration not bound to actual video probe')
    if any(draft.get(k)!=v or case_receipt.get(k)!=v or manifest.get(k)!=v for k,v in FLAGS.items()):raise ValueError('approval/diagnostic flags mismatch')
    if draft['status'] not in ['draft_generated','partial','needs_regeneration'] or not draft['events'] or not draft['ms_text'].strip():raise ValueError('no real rule events/text or wrong draft status')
    for event in draft['events']:
     support=event['support'];indices=np.array(support['frame_indices']);joints=np.array(support['joint_indices'])
     if not event['text'].strip() or not len(indices) or np.any(indices<0) or np.any(indices>=len(mask)) or np.any(joints<0) or np.any(joints>=134) or not mask[np.ix_(indices,joints)].all():raise ValueError('rule support outside valid native observations')
    audio=beats['actual_audio_metadata'];duration=float(audio['duration_seconds'])
    if beats['audio_path']!=row['audio_path'] or beats['device']!='cuda:0' or beats['float16'] is not False or beats['dbn'] is not False or beats['audio_video_alignment_verified'] is not False:raise ValueError('BeatThis clock/config mismatch')
    for name in ['beats','downbeats']:
     values=np.asarray(beats[name]);
     if not np.isfinite(values).all() or np.any(values<0) or np.any(np.diff(values)<=0) or np.any(values>duration+.02):raise ValueError('invalid actual beat times')
    n=len(data['times']);e=len(draft['events']);frames+=n;events+=e;partial+=int(draft['status']=='partial');staged[sourcekey]={'status':draft['status'],'frames':n,'events':e,'commit':key,'actual_artifacts_deep_verified':True}
   ledger['sources'][w].update(staged);ledger['commits'][key]={'archive_SHA256':expected_archive,'verified_at':now(),'sources':len(staged),'frames':frames,'events':events,'partial':partial,'failed':failures}
   atomic(ledger_path,ledger)
   report={'time':now(),'status':'running','pid':os.getpid(),'verified_unique_sources':sum(len(v) for v in ledger['sources'].values()),'verified_events':sum(s.get('events',0) for v in ledger['sources'].values() for s in v.values()),'remaining_to_deep_verify':4000-sum(len(v) for v in ledger['sources'].values()),'last_commit':key,'GPU_workers_unchanged':True}
   atomic(ROOT/'logs/unified_payload_audit_latest.json',report)
   print(json.dumps({'event':'batch_deep_verified','time':report['time'],'worker':w,'batch_sources':len(staged),'total_verified_sources':report['verified_unique_sources']}),flush=True)
  total=sum(len(v) for v in ledger['sources'].values())
  if total==4000:
   final={'schema':'unified4000_artifact_acceptance_v1','created':now(),'unique_sources_verified':4000,'workers':{w:{'sources':len(ledger['sources'][w]),'partial':sum(s['status']=='partial' for s in ledger['sources'][w].values()),'failed':sum(s['status']=='failed' for s in ledger['sources'][w].values()),'events':sum(s.get('events',0) for s in ledger['sources'][w].values())} for w in WORKERS},'deep_verified_commits':len(ledger['commits']),'input_media_read':False,'GPU_models_loaded':False,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
   for p in [ROOT/'logs/unified4000_artifact_acceptance.json',P/'results/unified4000_artifact_acceptance_03_to_06.json']:
    with p.open('x') as f:json.dump(final,f,indent=2);f.write(LF);f.flush();os.fsync(f.fileno())
   atomic(ROOT/'logs/unified_payload_audit_latest.json',{'time':now(),'status':'completed','verified_unique_sources':4000,'final_acceptance':str(P/'results/unified4000_artifact_acceptance_03_to_06.json')})
   print(json.dumps({'event':'all4000_output_packages_verified','final_report':str(P/'results/unified4000_artifact_acceptance_03_to_06.json')}),flush=True);break
  time.sleep(60)
 except FileNotFoundError as e:
  if 'unified_verified_commit_cursor.json' not in str(e):raise
  time.sleep(10)
 except Exception as e:
  error={'event':'PAYLOAD_AUDIT_FAILURE','time':now(),'error':type(e).__name__+':'+str(e),'last_commit':locals().get('key'),'no_source_retry':True,'GPU_workers_not_stopped':True}
  atomic(ROOT/'logs/unified_payload_audit_alarm.json',error);print(json.dumps(error),flush=True);sys.exit(1)
