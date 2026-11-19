"""Local resource/assignment glue; all extraction and rules use frozen central functions."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback

B=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
QUEUE_EDITION='external_ms_unified_clusters80k_20261027_v1'
M=Path(__file__).resolve().parent/'frozen_core'
GIB=1024**3
BUDGET=8*GIB
FLAGS=dict(diagnostic_not_production=True,semantic_approval=False,quality_approved=False,training_eligible=False,needs_visual_revision=True)
EXPECTED={'a100-extra-01':'ts-8b1d8922a0d2976301a11b9f541865dc-launcher','a100-extra-02':'ts-8b1d8922a0d2976301a11b9f541865dc-worker-0'}


from storage_io import mkdir_sync, fsync_dir, publish_file, io_wait, read_json_shared, install_shared_io_adapters


class Worker:
    def __init__(self,args):
        self.args=args;self.gpu=args.gpu;self.worker=args.worker
        self.root=Path(args.run_root)/('gpu'+str(self.gpu));self.root.mkdir(parents=True,exist_ok=False)
        self.begin=time.perf_counter();self.start_unix=time.time();self.abort=threading.Event();self.monitor_stop=threading.Event();self.mutex=threading.RLock()
        self.counts=dict(claimed=0,extracted=0,published=0,partial=0,failed=0,skipped=0,emptybeats=0,frames=0,published_frames=0,publish_backlog=0)
        self.times=dict(copy=0.,probe=0.,decode_pose=0.,rules=0.,beat=0.,write=0.,publication=0.,input_wait=0.)
        self.state=dict(source=None,stage='startup',frame=0,assignment_index=None)
        self.peaks=dict(rss_gib=0.,gpu_mib=0,cgroup_used_gib=0.,cache_reserved_bytes=0)
        self.resume_plan=json.loads((Path(args.run_root)/(args.worker+'.resume_plan.json')).read_text())
        retained=0
        for oldrun in self.resume_plan['retained_run_roots']:
            oldroot=Path(oldrun)/('gpu'+str(self.gpu))
            retained+=sum(p.stat().st_size for p in (oldroot/'work').rglob('*') if p.is_file())+sum(p.stat().st_size for p in oldroot.glob('*.tar'))
        self.reserved=retained;self.retained=retained;self.records=[];self.batch=[];self.batch_number=0;self.locks=[];self.created_pause=False;self.phase_total=0;self.queue_done_sources=0;self.phase_summaries=[]
        self.log=(self.root/'progress.jsonl').open('x',encoding='utf-8')
        sys.path.insert(0,str(M));import ms_unified_novlm_v1 as core
        self.core=core
        install_shared_io_adapters(core)

    def emit(self,event,**fields):
        row=dict(event=event,worker=self.worker,gpu=self.gpu,pid=os.getpid(),timestamp=datetime.datetime.now().astimezone().isoformat(),unix=time.time(),elapsed_s=time.perf_counter()-self.begin,**FLAGS)
        row.update(fields)
        with self.mutex:
            self.log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+chr(10));self.log.flush()
            print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)

    def safe(self):
        if self.abort.is_set():raise RuntimeError('Actual resource/ownership alarm; no continuation or retry')

    def memory(self):
        b=Path('/sys/fs/cgroup/memory')
        if b.is_dir() and (b/'memory.limit_in_bytes').is_file():
            limit=int((b/'memory.limit_in_bytes').read_text());used=int((b/'memory.usage_in_bytes').read_text())
        else:
            b=Path('/sys/fs/cgroup');v=(b/'memory.max').read_text().strip();limit=int(v) if v!='max' else 2**63;used=int((b/'memory.current').read_text())
        host=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        rss=next(int(s.split()[1])*1024 for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))
        return dict(direct_headroom_gib=min(limit-used,host)/GIB,limit_gib=limit/GIB,charged_usage_gib=used/GIB,rss_gib=rss/GIB)

    def gpu_snapshot(self):
        r=subprocess.run(['nvidia-smi','-i',str(self.gpu),'--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        index,uuid,name,used,util=next(csv.reader(io.StringIO(r.stdout)))
        r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        pids=[dict(pid=int(a[1]),exe=a[2].strip(),memory_mib=int(a[3])) for a in csv.reader(io.StringIO(r.stdout)) if len(a)==4 and a[0].strip()==uuid.strip()]
        return dict(index=int(index),uuid=uuid.strip(),name=name.strip(),used_mib=int(used),utilization=int(util),processes=pids)

    def check_resources(self,start=False):
        g=self.gpu_snapshot();m=self.memory();free=shutil.disk_usage(self.root).free
        foreign=[p for p in g['processes'] if p['pid']!=os.getpid()]
        if foreign or g['used_mib']>8192:raise PermissionError('GPU foreign PID or small-model GPU budget exceeded: '+str(g))
        if start and (g['processes'] or g['used_mib']>128):raise PermissionError('GPU not freshly empty')
        if m['direct_headroom_gib']<32 or m['rss_gib']>24:raise PermissionError('Independent actual small-model memory gate failed: '+str(m))
        if start and free<BUDGET+8*GIB:raise PermissionError('Local disk requires cache8GiB plus reserve8GiB')
        with self.mutex:
            self.peaks['rss_gib']=max(self.peaks['rss_gib'],m['rss_gib']);self.peaks['cgroup_used_gib']=max(self.peaks['cgroup_used_gib'],m['charged_usage_gib']);self.peaks['gpu_mib']=max(self.peaks['gpu_mib'],g['used_mib'])
        return dict(gpu=g,memory=m,local_free_gib=free/GIB)

    def acquire_resources(self):
        if self.worker=='a100-extra-01':return self.acquire_free_resources()
        if os.uname().nodename!=EXPECTED[self.worker]:raise PermissionError('Actual node identity mismatch')
        initial=self.gpu_snapshot();uuid=initial['uuid']
        mem=self.memory()
        if mem['direct_headroom_gib']<32 or shutil.disk_usage(self.root).free<BUDGET+8*GIB:raise PermissionError('Actual memory/disk gate before reservation release failed')
        hold=Path('/dockerdata/wm_logs/gpu_hold');status=json.loads((hold/'status.json').read_text())
        entry=status.get('gpus',{}).get(str(self.gpu),{})
        supervisor=status.get('pid');holder_pid=entry.get('holder_pid')
        if not status.get('running') or not status.get('query_ok') or entry.get('gpu_uuid')!=uuid or entry.get('external_pids'):raise PermissionError('Reservation ownership evidence invalid or external workload present')
        for pid in [supervisor]+[p['pid'] for p in initial['processes']]:
            args=Path('/proc')/str(pid)/'cmdline'
            values=args.read_bytes().split(bytes([0]))
            if not any(Path(v.decode(errors='replace')).name=='gpu_hold.py' for v in values if v):raise PermissionError('Unknown GPU reservation process; no release')
            if pid!=supervisor and pid!=holder_pid:raise PermissionError('GPU compute PID differs from owned holder')
            owner=next(x.split()[1] for x in (Path('/proc')/str(pid)/'status').read_text().splitlines() if x.startswith('Uid:'))
            if int(owner)!=os.getuid():raise PermissionError('Reservation belongs to another UID')
        paths=[Path('/dockerdata/ms_smallmodel_resources_v1')/(uuid+'.lock'),Path('/dockerdata/wm_logs/gpu_hold')/('experiment.'+str(self.gpu)+'.lock')]
        for path in paths:
            mkdir_sync(path.parent);lock=path.open('a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except Exception:lock.close();raise PermissionError('Existing GPU lease/flock owner: '+str(path))
            self.locks.append(lock)
        pause=hold/('pause.'+str(self.gpu))
        previously_paused=pause.exists()
        # Hold native experiment lock while requesting its verified pause interface.
        self.created_pause=not previously_paused
        subprocess.run(['/opt/conda/envs/torch-base/bin/python','/dockerdata/wm_tmp/gpu_hold.py','--action','pause','--gpus',str(self.gpu),'--state-dir',str(hold)],check=True,timeout=100)
        acknowledgement=json.loads((hold/'status.json').read_text()).get('gpus',{}).get(str(self.gpu),{})
        if acknowledgement.get('paused') is not True or acknowledgement.get('holder_pid') is not None or acknowledgement.get('external_pids'):raise PermissionError('Native pause did not confirm owned reservation released')
        observed=self.check_resources(start=True)
        lease=dict(worker=self.worker,gpu=self.gpu,uuid=uuid,pid=os.getpid(),uid=os.getuid(),start_unix=self.start_unix,hostname=os.uname().nodename,proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],locks=[str(p) for p in paths],actual_resource_evidence=observed,edition=self.ticket['edition'],assignment_sha256=self.ticket['assignment_sha256'],independent_smallmodel_contract=True,no_native_parent_impersonation=True,reservation_release_protocol='held experiment lock plus gpu_hold --action pause acknowledgment',previously_paused=previously_paused,created_pause=self.created_pause,**FLAGS)
        self.core.write_once(self.root/'lease.json',self.core.json_bytes(lease))
        self.core.write_once(self.output/'gpu_leases'/('gpu'+str(self.gpu)+'-'+Path(self.args.run_root).name+'.json'),self.core.json_bytes(lease))
        self.emit('resource_lease_acquired',evidence=lease)

    def acquire_free_resources(self):
        if os.uname().nodename!=EXPECTED[self.worker]:raise PermissionError('Actual node identity mismatch')
        observed=self.check_resources(start=True);uuid=observed['gpu']['uuid']
        paths=[Path('/dockerdata/ms_smallmodel_resources_v1')/(uuid+'.lock'),Path('/dockerdata/wm_logs/gpu_hold')/('experiment.'+str(self.gpu)+'.lock')]
        for path in paths:
            mkdir_sync(path.parent);lock=path.open('a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except Exception:lock.close();raise PermissionError('Existing GPU lease/flock owner: '+str(path))
            self.locks.append(lock)
        observed=self.check_resources(start=True)
        lease=dict(worker=self.worker,gpu=self.gpu,uuid=uuid,pid=os.getpid(),uid=os.getuid(),start_unix=self.start_unix,hostname=os.uname().nodename,proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],locks=[str(p) for p in paths],actual_resource_evidence=observed,edition=self.ticket['edition'],assignment_sha256=self.ticket['assignment_sha256'],independent_smallmodel_contract=True,no_native_parent_impersonation=True,**FLAGS)
        self.core.write_once(self.root/'lease.json',self.core.json_bytes(lease))
        self.core.write_once(self.output/'gpu_leases'/('gpu'+str(self.gpu)+'-'+Path(self.args.run_root).name+'.json'),self.core.json_bytes(lease))
        self.emit('resource_lease_acquired',evidence=lease)

    def monitor(self):
        report_next=throughput_next=0.;window_start=time.time();window_count=0;last_change=time.monotonic();previous=None
        while not self.monitor_stop.is_set():
            try:
                snap=self.check_resources();now=time.monotonic()
                if now>=report_next:
                    with self.mutex:c=dict(self.counts);s=dict(self.state);p=dict(self.peaks);t=dict(self.times);reserved=self.reserved
                    fp=(s['source'],s['stage'],s['frame'],c['claimed'],c['published'],c['failed'],c['skipped'])
                    if fp!=previous:previous=fp;last_change=now
                    elif now-last_change>=180:self.emit('stalled_progress',seconds=now-last_change,no_automatic_retry=True)
                    self.emit('progress',counts=c,state=s,resource=snap,peaks=p,times_s=t,cache_reserved_gib=reserved/GIB,remaining_preallocated_sources=self.phase_total-self.queue_done_sources-c['published']-c['failed']-c['skipped'])
                    report_next=now+60
                    if time.time()-window_start>=900:
                        seconds=time.time()-window_start;new=c['published']-window_count
                        self.emit('throughput_15min',new_unique_published_sources=new,window_seconds=seconds,sources_per_hour=new*3600/seconds,counts=c)
                        window_start=time.time();window_count=c['published']
            except Exception as e:
                self.abort.set();self.emit('resource_alarm',error=str(e),no_retry=True)
                break
            self.monitor_stop.wait(5)

    def validate_assignment(self,ticket_ref):
        ticket_path=Path(ticket_ref['path']);ticket_raw=io_wait(lambda:ticket_path.read_bytes(),'ticket read')
        if hashlib.sha256(ticket_raw).hexdigest()!=ticket_ref['sha256']:raise ValueError('Queue ticket SHA mismatch')
        P=ticket_path.parent.parent
        self.ticket=json.loads(ticket_raw)
        t=self.ticket
        if t['worker_id']!=self.worker or t['edition']!=P.name or t['source_count'] not in (5000,40000) or t['allowed_gpu_indices']!=[0,1,2,3] or t['automatic_retry'] is not False:raise ValueError('Ticket authorization mismatch')
        if Path(t['claims_root'])!=P/'claims' or Path(t['output_root'])!=P/'results'/self.worker:raise ValueError('Central namespace mismatch')
        if self.gpu not in t['allowed_gpu_indices'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(self.gpu):raise ValueError('Physical GPU visibility mismatch')
        raw=io_wait(lambda:Path(t['assignment_path']).read_bytes(),'assignment read')
        if hashlib.sha256(raw).hexdigest()!=t['assignment_sha256']:raise ValueError('Assignment SHA mismatch')
        rows=[];cursor=0
        for index,line in enumerate(raw.splitlines(keepends=True)):
            try:
                if not line.endswith(bytes([10])) or bytes([0]) in line:raise ValueError('NUL or incomplete assignment')
                row=json.loads(line)
            except Exception as e:
                self.emit('cursor_alarm',offset=cursor,path=t['assignment_path'],cursor_retained=True,error=str(e))
                with Path(t['assignment_path']).open('rb') as f:f.seek(cursor);f.readline()
                raise
            cursor+=len(line)
            if any(not isinstance(row.get(k),str) or not re.fullmatch('[A-Za-z0-9_-]+',row[k]) for k in ['id','split']):raise ValueError('Unsafe assignment identity')
            if any(Path(row[k]).stem!=row['id'] for k in ['video_path','audio_path']):raise ValueError('Source path identity mismatch')
            row['assignment_index']=index
            rows.append(row)
        if len(rows)!=t['source_count'] or len({(r['split'],r['id']) for r in rows})!=t['source_count']:raise ValueError('Assignment count/uniqueness mismatch')
        for ref in t['code_refs']:
            if hashlib.sha256((M/Path(ref['path']).name).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Frozen algorithm SHA mismatch '+ref['path'])
        self.output=Path(t['output_root']);self.rows=[r for r in rows if r['assignment_index']%4==self.gpu]
        if len(self.rows)!=t['source_count']//4:raise ValueError('Exact mod4 partition size mismatch')
        if t['edition']=='external_ms_unified_clusters80k_20261027_v1':
            allowed=set(self.resume_plan['pending_indices'])
            self.rows=[r for r in self.rows if r['assignment_index'] in allowed]
            if len(self.rows)!=self.resume_plan['pending_per_gpu'][str(self.gpu)]:raise ValueError('Exact unconsumed plan mismatch')
        self.core.write_once(self.root/(t['edition']+'.ticket.json'),self.core.json_bytes(t))
        self.core.write_once(self.root/(t['edition']+'.partition.json'),self.core.json_bytes(self.rows))

    def reserve(self,row):
        self.safe()
        # Size metadata reserves bytes only; media binding hash is computed once by central copy_source.
        inputs=sum(Path(row[k]).stat().st_size for k in ['video_path','audio_path'])
        if inputs>2*GIB:raise ValueError('Single input exceeds bounded2GiB copy budget')
        outputs=128*1024**2;need=inputs+outputs
        if self.reserved+need>BUDGET:raise ValueError('8GiB cache reservation cannot fit; source sealed, never truncate/retry')
        self.reserved+=need;self.peaks['cache_reserved_bytes']=max(self.peaks['cache_reserved_bytes'],self.reserved)
        return need,outputs

    def enqueue(self,row,executor,batch_number=None):
        self.safe()
        claim=Path(self.ticket['claims_root'])/row['split']/(row['id']+'.json')
        mkdir_sync(claim.parent)
        value=dict(edition=self.ticket['edition'],worker=self.worker,gpu=self.gpu,pid=os.getpid(),source=row,assignment_sha256=self.ticket['assignment_sha256'],claimed_at_unix=time.time(),automatic_retry=False,**FLAGS)
        try:self.core.claim_once(claim,value)
        except FileExistsError:
            self.counts['skipped']+=1;self.records.append(dict(source_row=row,status='existing_claim_not_reassigned',**FLAGS));self.emit('claim_skipped',id=row['id'],assignment_index=row['assignment_index']);return None
        self.counts['claimed']+=1
        batchid=('gpu'+str(self.gpu)+'-'+Path(self.args.run_root).name+'-b'+str(self.batch_number if batch_number is None else batch_number).zfill(4))
        local=self.root/'work'/batchid
        profile={};source_start=time.perf_counter()
        item=dict(row=row,profile=profile,start=source_start,local_root=local,batchid=batchid,claim_path=str(claim),reservation=0,max_output=0)
        try:
            item['reservation'],item['max_output']=self.reserve(row)
            # One copier thread; pending is capped at two source futures.
            item['future']=executor.submit(self.core.copy_source,row,local,2*GIB,profile)
        except Exception as e:item['error']=str(e)
        return item

    def decoder(self,path,w,h):
        for index,frame in enumerate(self.core.decode_bgr(path,w,h)):
            self.safe();self.state['frame']=index+1
            yield frame
        self.safe()

    def process(self,item,models):
        self.safe();r=item['row'];self.state.update(source=r['id'],assignment_index=r['assignment_index'],stage='copy',frame=0)
        case=item['local_root']/'cases'/r['split']/r['id']
        bound=self.output/'batches'/item['batchid']/'cases'/r['split']/r['id']
        copied_ok=False
        try:
            if 'error' in item:raise ValueError(item['error'])
            t=time.perf_counter();copied,hashes=item['future'].result();self.times['input_wait']+=time.perf_counter()-t;copied_ok=True
            self.safe();io_wait(lambda:Path(self.ticket['claims_root']).stat(),'pre-extraction shared admission');self.state['stage']='BGR_extract_and_CPU_rule'
            result=self.core.extract_and_rule(r,copied,hashes,models,case,bound,decoder=self.decoder,max_output_bytes=item['max_output'],profile=item['profile'],source_started=item['start'])
            self.safe()
            draft=json.loads((case/'draft.json').read_text())
            # Bootstrapping requires actual event/support evidence, not merely file existence.
            has_events=bool(draft['events']) and all(e.get('text') and e.get('support',{}).get('frame_indices') for e in draft['events'])
            result['actual_supported_events']=len(draft['events']);result['bootstrap_evidence']=has_events
            self.counts['extracted']+=1;self.counts['frames']+=result['frame_count'];self.counts['partial']+=int(result['draft_status']=='partial')
            beat=json.loads((case/'beats.json').read_text());self.counts['emptybeats']+=int(beat['emptybeats'])
        except Exception as e:
            result=dict(schema='ms_unified_novlm_case_receipt_v1',source_row=r,status='failed_sealed_no_retry',input_copy_completed=copied_ok,error=str(e),traceback=traceback.format_exc(),profile_seconds=item['profile'],assignment_sha256=self.ticket['assignment_sha256'],budget_consumed=True,**FLAGS)
            self.counts['failed']+=1;mkdir_sync(case)
            self.core.write_once(case/'sealed_failure.json',self.core.json_bytes(result))
            # Never overwrite a receipt if extraction produced it before a resource alarm.
            if not (case/'receipt.json').exists():self.core.write_once(case/'receipt.json',self.core.json_bytes(result))
            self.emit('source_error',source=r['id'],error=str(e),stage=self.state['stage'],no_retry=True)
        item.update(result=result,case=case,bound=bound)
        for k in self.times:
            if k not in ['publication','input_wait']:self.times[k]+=item['profile'].get(k,0.)
        self.batch.append(item);self.counts['publish_backlog']+=1
        self.emit('source_staged',id=r['id'],assignment_index=r['assignment_index'],status=result['status'],draft_status=result.get('draft_status'),frames=result.get('frame_count',0),profile=item['profile'])

    def publish_batch(self):
        if not self.batch:return None
        self.safe();self.state['stage']='batch_publish';batch=self.batch;batchid=batch[0]['batchid']
        directory=self.output/'batches'/batchid;archive=self.root/(batchid+'.tar')
        t=time.perf_counter()
        with archive.open('xb') as f:
            with tarfile.open(fileobj=f,mode='w') as pack:
                for item in batch:
                    for name in self.core.FILES+('sealed_failure.json',):
                        p=item['case']/name
                        if p.is_file():pack.add(p,arcname=str(Path('cases')/item['row']['split']/item['row']['id']/name),recursive=False)
            f.flush();os.fsync(f.fileno())
        # Account for transient package bytes before creating a remote commit.
        if self.reserved+archive.stat().st_size>BUDGET:raise RuntimeError('Package plus cache exceeds8GiB; retained uncommitted, no retry')
        mkdir_sync(directory)
        files=[]
        for item in batch:
            for name in self.core.FILES+('sealed_failure.json',):
                p=item['case']/name
                if p.is_file():files.append(publish_file(p,item['bound']/name))
        package=publish_file(archive,directory/'batch.tar');self.safe()
        commit=dict(schema='ms_unified_novlm_batch_commit_v1',edition=self.ticket['edition'],worker=self.worker,gpu=self.gpu,owner_pid=os.getpid(),ticket_id=batchid,assignment_sha256=self.ticket['assignment_sha256'],batch_sha256=package['sha256'],package=package,files=files,receipts=[i['result'] for i in batch],unique_sources_attempted=len(batch),prefetch_drained=True,publication_scope='new pose/manifest/beats/draft/receipt only; no input videos',profile_seconds=dict(pack_publish=time.perf_counter()-t,whole_e2e=time.perf_counter()-self.begin),commit_unix=time.time(),**FLAGS)
        self.core.write_once(directory/'commit.json',self.core.json_bytes(commit));fsync_dir(directory)
        remote=read_json_shared(directory/'commit.json')
        if remote['batch_sha256']!=package['sha256'] or remote['unique_sources_attempted']!=len(batch):raise ValueError('Reliable commit readback mismatch')
        self.times['publication']+=time.perf_counter()-t
        for item in batch:
            success=item['result']['status']=='extracted_rule_draft_unapproved'
            self.counts['published']+=int(success);self.counts['published_frames']+=item['result'].get('frame_count',0) if success else 0
            self.counts['publish_backlog']-=1
            item['result']['published_commit_path']=str(directory/'commit.json')
            self.records.append(item['result'])
        self.emit('batch_commit',commit_path=str(directory/'commit.json'),package_path=package['path'],sources=len(batch),counts=dict(self.counts),seconds=time.perf_counter()-t)
        # All owned inputs and outputs reclaimed only after reliable shared commit.
        for item in batch:
            own=item['local_root']/item['row']['split']/item['row']['id']
            if own.is_dir():shutil.rmtree(own)
            shutil.rmtree(item['case']);self.reserved-=item['reservation']
        archive.unlink();self.batch=[];self.batch_number+=1
        return commit

    def segment(self,rows,models):
        # One copier thread and <=2 source futures. Next batch copies overlap
        # current publication, but keep their own immutable batch bindings.
        with ThreadPoolExecutor(max_workers=1) as executor:
            prefetched={}
            for start in range(0,len(rows),20):
                group=rows[start:start+20]
                pending=deque(item for item in prefetched.values() if item is not None);already=set(prefetched);prefetched={}
                todo=iter([r for r in group if r['assignment_index'] not in already]);done=False
                def fill():
                    nonlocal done
                    while len(pending)<2 and not done:
                        row=next(todo,None)
                        if row is None:done=True;break
                        item=self.enqueue(row,executor)
                        if item is not None:pending.append(item)
                fill()
                while pending:
                    item=pending.popleft();self.process(item,models);fill()
                if self.batch:
                    for nextrow in rows[start+20:start+22]:
                        item=self.enqueue(nextrow,executor,batch_number=self.batch_number+1)
                        prefetched[nextrow['assignment_index']]=item
                    self.publish_batch()

    def run(self):
        queue_path=B/QUEUE_EDITION/(self.worker+'.queue.json')
        queue_raw=io_wait(lambda:queue_path.read_bytes(),'queue read');q=json.loads(queue_raw)
        if q.get('worker_id')!=self.worker or q.get('schema')!='ms_external_authorized_rollover_queue_v1' or not q.get('automatic_rollover_within_preallocated_tickets') or not q.get('no_automatic_retry_sealed_previous_sources'):
            raise PermissionError('Finite central queue authorization mismatch')
        self.queue_sha=hashlib.sha256(queue_raw).hexdigest()
        previous=B/'external_ms_unified_run_20261027_v1'/'results'/self.worker
        if self.worker=='a100-extra-01':
            proof=read_json_shared(previous/'final_accounting_after_recovery.json')
            if proof['unique_reliably_published_sources']!=999 or len(proof['unfinished_sources'])!=1:raise PermissionError('Prior edition accounting not verified')
        else:
            for g in range(4):
                proof=read_json_shared(previous/('gpu'+str(g)+'_summary.json'))
                if proof['counts']['published']!=250:raise PermissionError('Prior node02 partition still active/incomplete')
        refs=[]
        for ref in q['tickets']:
            path=Path(ref['path'])
            if path.parent.parent.name=='external_ms_unified_run_20261027_v1':
                self.emit('consumed_ticket_preserved_not_restarted',ticket_path=str(path));continue
            if path.parent.parent.name=='external_ms_unified_rollover_20261027_v1':
                self.emit('prior_ticket_preserved_not_restarted',ticket_path=str(path));continue
            if path.parent.parent.name!=QUEUE_EDITION:raise PermissionError('Unexpected finite queue scope')
            refs.append(ref)
        if len(refs)!=1:raise PermissionError('Only current authorized clusters ticket for failed GPU0 resume')
        if self.worker!='a100-extra-02' or self.gpu!=0:raise PermissionError('This resume is scoped to node02 GPU0 only')
        self.phase_total=self.resume_plan['pending_per_gpu'][str(self.gpu)]
        self.validate_assignment(refs[0]);self.acquire_resources()
        monitor=threading.Thread(target=self.monitor,daemon=True);monitor.start()
        self.state['stage']='models_load_once';t=time.perf_counter()
        models=self.core.load_models(dict(gpu_index=self.gpu,scail_root=self.args.scail_root,beat_checkpoint=self.args.beat_checkpoint))
        self.models_init=dict(dwpose_count=1,beatthis_count=1,seconds=time.perf_counter()-t,actual_sessions={'detector':models[0].session_det.get_providers(),'pose':models[0].session_pose.get_providers()})
        self.safe();self.emit('models_ready',initialization=self.models_init,resource=self.check_resources(),remaining_preallocated_sources=self.phase_total)
        for phase,ref in enumerate(refs):
            if phase:
                self.validate_assignment(ref)
                self.counts={k:0 for k in self.counts};self.times={k:0. for k in self.times};self.batch_number=0;self.records=[]
            marker=self.output/'resume_sessions'/Path(self.args.run_root).name/('gpu'+str(self.gpu)+'.json') if phase==0 else self.output/'queue_consumption'/('gpu'+str(self.gpu)+'.json')
            mkdir_sync(marker.parent)
            try:self.core.claim_once(marker,dict(worker=self.worker,gpu=self.gpu,pid=os.getpid(),edition=self.ticket['edition'],queue_sha256=self.queue_sha,ticket_sha256=ref['sha256'],assignment_sha256=self.ticket['assignment_sha256'],started_unix=time.time(),no_automatic_retry=True))
            except FileExistsError:
                self.emit('consumed_ticket_not_restarted',ticket=ref['path'],marker=str(marker));self.queue_done_sources+=len(self.rows);continue
            self.emit('queue_ticket_started',ticket_path=ref['path'],edition=self.ticket['edition'],gpu_assigned_sources=len(self.rows),remaining_preallocated_sources=self.phase_total-self.queue_done_sources,models_reloaded=False)
            self.segment(self.rows,models)
            if self.counts['published']+self.counts['failed']+self.counts['skipped']!=len(self.rows):raise ValueError('Finite partition terminal accounting mismatch')
            self.queue_done_sources+=len(self.rows)
            summary=dict(worker=self.worker,gpu=self.gpu,pid=os.getpid(),edition=self.ticket['edition'],assignment_sha256=self.ticket['assignment_sha256'],assigned_sources=len(self.rows),counts=dict(self.counts),model_initialization=self.models_init,resource_peaks=dict(self.peaks),timings_s=dict(self.times),elapsed_queue_s=time.perf_counter()-self.begin,remaining_preallocated_sources=self.phase_total-self.queue_done_sources,automatic_retries=0,vlm_called=False,**FLAGS)
            p=self.root/(self.ticket['edition']+'.summary.json');self.core.write_once(p,self.core.json_bytes(summary));publish_file(p,self.output/('gpu'+str(self.gpu)+'_'+Path(self.args.run_root).name+'_summary.json'))
            self.phase_summaries.append(summary)
            self.emit('queue_ticket_done',summary=summary,next_ticket_already_authorized=phase+1<len(refs))
        self.monitor_stop.set();monitor.join(timeout=15)
        self.peaks['rss_highwater_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB
        final=dict(worker=self.worker,gpu=self.gpu,pid=os.getpid(),preallocated_sources=self.phase_total,remaining_preallocated_sources=0,phase_summaries=self.phase_summaries,model_initialization=self.models_init,queue_sha256=self.queue_sha,elapsed_s=time.perf_counter()-self.begin,**FLAGS)
        self.core.write_once(self.root/'queue_summary.json',self.core.json_bytes(final));publish_file(self.root/'queue_summary.json',B/QUEUE_EDITION/'results'/self.worker/('gpu'+str(self.gpu)+'_'+Path(self.args.run_root).name+'_queue_summary.json'))
        self.emit('queue_exhausted',remaining_preallocated_sources=0,completed_preallocated_sources=self.queue_done_sources,summary_path=str(B/QUEUE_EDITION/'results'/self.worker/('gpu'+str(self.gpu)+'_queue_summary.json')))
        self.log.flush();os.fsync(self.log.fileno());publish_file(self.root/'progress.jsonl',B/QUEUE_EDITION/'results'/self.worker/('gpu'+str(self.gpu)+'_'+Path(self.args.run_root).name+'_queue_progress.jsonl'))
        return 1 if any(x['counts']['failed'] for x in self.phase_summaries) else 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True,choices=EXPECTED);p.add_argument('--gpu',type=int,required=True,choices=range(4));p.add_argument('--run-root',required=True);p.add_argument('--scail-root',required=True);p.add_argument('--beat-checkpoint',required=True);args=p.parse_args()
    w=Worker(args)
    try:code=w.run()
    except Exception as e:
        w.abort.set();w.monitor_stop.set();w.emit('fatal_error',error=str(e),traceback=traceback.format_exc(),counts=w.counts,no_automatic_retry=True)
        code=2
    finally:
        w.monitor_stop.set();w.log.flush();os.fsync(w.log.fileno());w.log.close()
        if w.created_pause:
            try:
                import torch
                if torch.cuda.is_initialized():torch.cuda.empty_cache()
                subprocess.run(['/opt/conda/envs/torch-base/bin/python','/dockerdata/wm_tmp/gpu_hold.py','--action','resume','--gpus',str(w.gpu),'--state-dir','/dockerdata/wm_logs/gpu_hold'],check=True,timeout=15)
            except Exception as cleanup_error:
                print(json.dumps({'event':'owned_pause_cleanup_alarm','gpu':w.gpu,'error':str(cleanup_error),'pause_not_blindly_deleted':True}),flush=True)
        for lock in reversed(w.locks):lock.close()
    return code


if __name__=='__main__':raise SystemExit(main())
