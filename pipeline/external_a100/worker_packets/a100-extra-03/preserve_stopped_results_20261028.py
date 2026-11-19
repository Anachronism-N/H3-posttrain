"""Preserve stopped 03-06 local annotation results; never run inference or alter original state."""
import concurrent.futures
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tarfile
import uuid

BASE = Path('/tmp/ms_pose2d_draft_20261026T1427Z')
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
P = ROOT / 'external_ms_unified_clusters80k_20261027_v1'
NODES = [('a100-extra-03', 'launcher', None), ('a100-extra-04', 'worker-0', '11.254.248.52'),
         ('a100-extra-05', 'worker-1', '11.239.106.11'), ('a100-extra-06', 'worker-2', '11.239.84.43')]
CAP = 1024 ** 3
COLLECTOR = r'''
import datetime, hashlib, io, json, os, stat, tarfile
from pathlib import Path
CAP = 1024 ** 3
TOPS = {'unified_runs', 'results_unreviewed', 'pilot_runs', 'pending_writeback', 'logs', 'gpu_locks',
        'publication_experiments', 'unified_code_snapshot_v1', 'cache', 'batchrpc20_candidate_v2',
        'batchrpc20_candidate_v2_role_path_v1', 'beat_rejection_evidence_candidate_v1',
        'recovery03_publication_v1', 'recovery03_unclaimed_v1', 'supervision_recovery03_v1'}
SKIP_DIRS = {'staging', 'inputs', 'rpc_spools', '__pycache__', 'model_cache', 'venv', 'wheelhouse',
             'pretrained_weights', 'assets'}
EXTS = {'.json', '.jsonl', '.npz', '.py', '.md', '.txt', '.log', '.csv', '.tar', '.gz', '.lock'}
MEDIA = {'.mp4', '.wav', '.mp3', '.flac', '.mov', '.avi', '.mkv', '.ckpt', '.onnx', '.pt', '.pth', '.safetensors'}
def encoded(v):
    return (json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
def identity(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def hashfile(p):
    if p.resolve(strict=True)!=p:raise ValueError('redirected input '+str(p))
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):raise ValueError('not regular '+str(p))
        h=hashlib.sha256();n=0
        with os.fdopen(fd,'rb',closefd=False) as f:
            while b:=f.read(1024**2):h.update(b);n+=len(b)
        after=os.fstat(fd)
        keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if n!=before.st_size or any(getattr(before,k)!=getattr(s,k) for k in keys for s in (after,p.stat())):
            raise ValueError('file changed '+str(p))
        return h.hexdigest(),n,tuple(getattr(before,k) for k in keys)
    finally:os.close(fd)
def inventory(root):
    paths=[];excluded=[]
    for p in sorted(root.iterdir()):
        if p.is_symlink():
            excluded.append({'path':str(p),'reason':'symlink not followed'});continue
        if p.is_file():
            if p.suffix in {'.json','.py','.md','.jsonl','.txt'}:paths.append(p)
            else:excluded.append({'path':str(p),'reason':'not result/metadata/source'})
        elif p.is_dir() and (p.name in TOPS or p.name.startswith(('integration_handoff_', 'shared_results_snapshot_'))):
            for folder,dirs,files in os.walk(p,followlinks=False):
                current=Path(folder)
                kept=[]
                for d in sorted(dirs):
                    q=current/d
                    if d in SKIP_DIRS or q.is_symlink():excluded.append({'path':str(q),'reason':'input/media/weight/cache directory or symlink'})
                    else:kept.append(d)
                dirs[:]=kept
                for name in sorted(files):
                    q=current/name
                    if q.is_symlink():excluded.append({'path':str(q),'reason':'symlink not followed'})
                    elif q.suffix in EXTS and not (p.name=='cache' and q.suffix=='.npz'):paths.append(q)
                    else:excluded.append({'path':str(q),'reason':'input/media/weight/spool or unsupported nonresult'})
        else:excluded.append({'path':str(p),'reason':'environment/weights/nonresult or new preservation work'})
    return sorted(paths),excluded
def require_stopped(root,worker):
    lanes=[]
    for gpu in range(4):
        p=root/'logs'/('rollover_gpu'+str(gpu)+'_progress.json');raw=p.read_bytes();v=json.loads(raw)
        if v.get('worker')!=worker or v.get('physical_gpu_index')!=gpu:raise ValueError('worker/GPU mismatch')
        proc=Path('/proc')/str(v['pid'])
        try:
            f=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if int(f[19])==v['start_ticks'] and f[0]!='Z':raise PermissionError('original worker still live')
        except FileNotFoundError:pass
        lanes.append({'gpu':gpu,'pid':v['pid'],'start_ticks':v['start_ticks'],'status_in_original_progress':v['status'],
                      'published_counter':v['published'],'processed_counter':v['unique_processed'],'run_id':v['run_id'],
                      'original_progress_sha256':hashlib.sha256(raw).hexdigest(),'observed_original_process_active':False})
    return lanes
class HashedReader:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
def verify_tar(path,expected):
    actual=set()
    with tarfile.open(path,'r|') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in actual:raise ValueError('unexpected/duplicate tar member')
            actual.add(member.name);h=hashlib.sha256();n=0;f=tar.extractfile(member)
            while b:=f.read(1024**2):h.update(b);n+=len(b)
            e=expected[member.name]
            if n!=member.size or n!=e['bytes'] or h.hexdigest()!=e['sha256']:raise ValueError('tar member SHA/length mismatch '+member.name)
    if actual!=set(expected):raise ValueError('tar member set mismatch')
def collect(root,worker,folder):
    lanes=require_stopped(root,worker);paths,excluded=inventory(root);records=[];sources=[];total=0
    for p in paths:
        sha,n,snapshot=hashfile(p);total+=n
        if total>CAP:raise ValueError('bounded preservation payload exceeds 1GiB')
        rel=str(p.relative_to(root));records.append({'archive_member':rel,'original_path':str(p),'bytes':n,'sha256':sha,'snapshot':list(snapshot)})
        if p.name=='receipt.json' and rel.startswith('unified_runs/'):
            r=json.loads(p.read_bytes());row=r.get('source_row')
            if row:
                parts=p.relative_to(root).parts;edition=parts[1];run_id=parts[2];batch=parts[3]
                if 'cases' in parts:
                    names=[]
                    for q in sorted(p.parent.iterdir()):
                        if q.is_file() and q.suffix in EXTS:names.append(str(q.relative_to(root)))
                    sources.append({'worker':worker,'edition':edition,'run_id':run_id,'original_batch':batch,
                                    'assignment_index':r.get('assignment_index'),'source_row':row,'original_receipt_status':r.get('status'),
                                    'draft_status':r.get('draft_status'),'receipt_sha256':sha,'artifact_members':names,
                                    'proposed_original_target':str(Path(SHARED_P).parent/edition/'results'/worker/'batches'/('gpu'+run_id[1])/batch),
                                    'preservation_only':True,'reliable_batch_commit_established_by_this_archive':False,
                                    'complete_artifact_semantics_reverified':False,'training_eligible':False,'quality_approved':False})
    manifest={'schema':'ms_stopped_local_result_preservation_manifest_v1','worker':worker,'role':root.name,
              'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'excluded':excluded,
              'total_file_bytes':total,'file_count':len(records),'stopped_original_lanes':lanes,
              'scope':'all extant local annotation/failed/diagnostic/result metadata in inventoried scopes; inputs/media/weights excluded',
              'partial_original_archives_preserved_as_evidence_not_trusted_batches':True,'source_reexecution':False,
              'claims_modified':False,'original_leases_or_results_modified':False,'durable_retirement_proven':False,
              'quality_approved':False,'training_eligible':False,'global80000_complete':False}
    source_raw=b''.join(encoded(r) for r in sorted(sources,key=lambda r:(r['edition'],r['run_id'],r.get('assignment_index') or -1)))
    manifest_raw=encoded(manifest);extras={'FILES.manifest.json':manifest_raw,'LOCAL_SOURCE_RESULTS.jsonl':source_raw}
    folder.mkdir(exist_ok=False)
    for name,raw in extras.items():
        with (folder/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    archive=folder/'local_results.tar'
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for e in records:
                p=Path(e['original_path']);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(fd,'rb') as f:
                    s=os.fstat(f.fileno());snapshot=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                    if list(snapshot)!=e['snapshot']:raise ValueError('file changed before packing')
                    info=tarfile.TarInfo(e['archive_member']);info.size=e['bytes'];info.mode=0o444
                    reader=HashedReader(f);tar.addfile(info,reader)
                    if reader.n!=e['bytes'] or reader.h.hexdigest()!=e['sha256'] or identity(p)!=snapshot:raise ValueError('file changed during packing')
            for name,raw in extras.items():
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;tar.addfile(info,io.BytesIO(raw))
        output.flush();os.fsync(output.fileno())
    expected={e['archive_member']:e for e in records}|{name:{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)} for name,raw in extras.items()}
    verify_tar(archive,expected);sha,n,_=hashfile(archive)
    require_stopped(root,worker)
    return {'worker':worker,'role':root.name,'archive_local_path':str(archive),'archive_sha256':sha,'archive_bytes':n,
            'manifest_local_path':str(folder/'FILES.manifest.json'),'source_index_local_path':str(folder/'LOCAL_SOURCE_RESULTS.jsonl'),
            'file_count':len(records),'source_receipts_preserved':len(sources),
            'current80k_source_receipts':sum(r['edition']==Path(SHARED_P).name for r in sources),
            'total_original_file_bytes':total,'excluded_count':len(excluded),'local_archive_all_members_sha_verified':True}
if __name__=='__main__':
    root=Path('/tmp/ms_pose2d_draft_20261026T1427Z')/ROLE
    result=collect(root,WORKER,root/('results_preservation_'+RUN))
    print(json.dumps(result),flush=True)
'''
def encoded(v):
    return (json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
def hashfile(path):
    h=hashlib.sha256();n=0
    with Path(path).open('rb') as f:
        while b:=f.read(1024**2):h.update(b);n+=len(b)
    return h.hexdigest(),n

def ssh_prefix(host):
    return ['ssh','-p','36000','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile=/tmp/ms_pose2d_resource_known_hosts_20261026',
            '-o','ConnectTimeout=8',host]

def remote_collect(node,run):
    worker,role,host=node;py=str(BASE/role/'venv/bin/python')
    cmd=ssh_prefix(host)+[shlex.join([py,'-B','-'])] if host else [py,'-B','-']
    source='ROLE='+repr(role)+'\nWORKER='+repr(worker)+'\nRUN='+repr(run)+'\nSHARED_P='+repr(str(P))+'\n'+COLLECTOR
    r=subprocess.run(cmd,input=source,capture_output=True,text=True,timeout=180,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    if r.returncode:raise RuntimeError(worker+' collector failed: '+r.stderr[-1500:])
    result=json.loads(r.stdout)
    if result.get('worker')!=worker or result.get('role')!=role or not result.get('local_archive_all_members_sha_verified'):raise ValueError('collector binding')
    return result

def read_local_or_remote(node,path,limit):
    _,role,host=node;p=Path(path)
    if not p.is_relative_to(BASE/role) or str(p)!=path or '..' in p.parts:raise ValueError('remote source scope')
    if not host:
        raw=p.read_bytes()
    else:
        r=subprocess.run(ssh_prefix(host)+[shlex.join(['/bin/cat','--',path])],capture_output=True,timeout=60)
        if r.returncode:raise RuntimeError('metadata transfer failed: '+r.stderr.decode(errors='replace')[-800:])
        raw=r.stdout
    if len(raw)>limit:raise ValueError('metadata cap')
    return raw

def syncdir(path):
    fd=os.open(path,os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def publish_bytes(out,name,raw):
    path=out/name
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    syncdir(out);sha,n=hashfile(path)
    if sha!=hashlib.sha256(raw).hexdigest() or n!=len(raw):raise ValueError('shared metadata readback mismatch')
    return {'path':str(path),'sha256':sha,'bytes':n}

def publish_worker(node,result,out):
    worker,role,host=node;local=Path(result['archive_local_path']);expectedroot=BASE/role
    if not local.is_relative_to(expectedroot) or local.name!='local_results.tar':raise ValueError('source archive scope')
    target=out/(worker+'.local_results.tar');hasher=hashlib.sha256();n=0
    process=None
    if host:
        process=subprocess.Popen(ssh_prefix(host)+[shlex.join(['/bin/cat','--',str(local)])],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        source=process.stdout
    else:source=local.open('rb')
    try:
        with target.open('xb') as f:
            while b:=source.read(1024**2):
                n+=len(b)
                if n>result['archive_bytes'] or n>CAP:raise ValueError('archive copy budget')
                hasher.update(b);f.write(b)
            f.flush();os.fsync(f.fileno())
        if process:
            rc=process.wait(timeout=30);error=process.stderr.read()
            if rc:raise RuntimeError('archive SSH stream failed '+error.decode(errors='replace')[-800:])
        if n!=result['archive_bytes'] or hasher.hexdigest()!=result['archive_sha256']:raise ValueError('archive transport SHA/length')
    finally:source.close()
    syncdir(out);sha,n=hashfile(target)
    if sha!=result['archive_sha256'] or n!=result['archive_bytes']:raise ValueError('shared archive full readback SHA/length')
    manifest_raw=read_local_or_remote(node,result['manifest_local_path'],64*1024**2)
    sources_raw=read_local_or_remote(node,result['source_index_local_path'],16*1024**2)
    manifest=json.loads(manifest_raw)
    if manifest['worker']!=worker or manifest['role']!=role:raise ValueError('manifest worker/role mismatch')
    expected={e['archive_member']:e for e in manifest['files']}
    for name,raw in [('FILES.manifest.json',manifest_raw),('LOCAL_SOURCE_RESULTS.jsonl',sources_raw)]:
        expected[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    env={'__name__':'preservation_verify','SHARED_P':str(P)};exec(compile(COLLECTOR,'<collector>','exec'),env)
    env['verify_tar'](target,expected)
    publications=[{'path':str(target),'sha256':sha,'bytes':n},
                  publish_bytes(out,worker+'.FILES.manifest.json',manifest_raw),
                  publish_bytes(out,worker+'.LOCAL_SOURCE_RESULTS.jsonl',sources_raw)]
    preserved_current=[json.loads(line) for line in sources_raw.splitlines() if line]
    return {'worker':worker,'publications':publications,'original_file_count':manifest['file_count'],
            'preserved_current80k_receipts':sum(r['edition']==P.name for r in preserved_current),
            'all_shared_archive_members_sha_length_verified':True,'annotation_batch_commit_created':False,
            'durable_retirement_proven':False,'preservation_only':True}

def main():
    run=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    parent=P/'results'
    if parent.resolve(strict=True)!=parent:raise ValueError('redirected authorized shared parent')
    out=parent/('stopped_03_to_06_results_preservation_'+run);out.mkdir(exist_ok=False);syncdir(parent)
    print(json.dumps({'event':'preservation_started','shared_directory':str(out),'run':run,'no_inference':True}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        collected=list(pool.map(lambda node:remote_collect(node,run),NODES))
    for r in collected:print(json.dumps({'event':'local_archive_verified',**r}),flush=True)
    # Serial publication avoids adding competing writers to the already unstable shared storage.
    workers=[]
    for node,result in zip(NODES,collected):
        entry=publish_worker(node,result,out);workers.append(entry)
        print(json.dumps({'event':'shared_worker_archive_verified',**entry}),flush=True)
    summary={'schema':'ms_stopped03_06_shared_results_preservation_summary_v1',
             'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':workers,
             'current80k_extant_local_receipts_preserved':sum(w['preserved_current80k_receipts'] for w in workers),
             'original_file_count':sum(w['original_file_count'] for w in workers),
             'old_already_published_payloads_not_reread_by_this_copy':True,'global80000_reconciliation_complete':False,
             'previous_commit_file_missing_report_requires_separate_check':True,
             'all_extant_local_results_in_inventoried_scopes_preserved':True,
             'original_inputs_media_weights_and_credential_files_not_packaged':True,
             'original_partial_tars_are_only_raw_evidence':True,'no_annotation_reexecution':True,
             'no_original_claim_lease_progress_or_result_modification':True,
             'preservation_is_not_reliable_annotation_batch_commit':True,'durable_retirement_proven':False,
             'quality_approved':False,'training_eligible':False}
    pub=publish_bytes(out,'SUMMARY.json',encoded(summary))
    commit={'schema':'ms_stopped_results_preservation_commit_v1','created':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'publications':[p for w in workers for p in w['publications']]+[pub],
            'all_archive_files_and_members_sha_length_readback_verified':True,
            'current80k_extant_local_receipts_preserved':summary['current80k_extant_local_receipts_preserved'],
            'original_file_count':summary['original_file_count'],'preservation_only':True,
            'annotation_reliable_batch_commit':False,'global80000_complete':False,
            'quality_approved':False,'training_eligible':False,'claims_modified':False,'durable_retirement_proven':False}
    cp=publish_bytes(out,'commit.json',encoded(commit))
    print(json.dumps({'event':'all_four_nodes_preserved_and_readback_verified','shared_directory':str(out),
                      'commit':cp,'preserved_current80k_receipts':summary['current80k_extant_local_receipts_preserved'],
                      'original_file_count':summary['original_file_count'],'source_reexecution':False}),flush=True)

if __name__=='__main__':main()
