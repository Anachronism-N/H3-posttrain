"""Independent directly user-authorized successor. Never reads/appends old queues."""
import argparse, hashlib, io, json, os, shutil, tarfile, threading, time, traceback
from pathlib import Path
import unified_queue_worker as base
from storage_io import io_wait, mkdir_sync, publish_file, read_json_shared
from transfer_claim_preflight import consume_original_claim, ConsumptionUncertain


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_contract(ticket_path, ticket_sha, authorization_sha):
    raw = Path(ticket_path).read_bytes()
    if digest(raw) != ticket_sha:
        raise PermissionError('Exact new ticket SHA mismatch')
    t = json.loads(raw)
    if t['schema'] != 'ms_direct_user_transfer_epoch_ticket_v1' or t['edition'] == t['predecessor_epoch']:
        raise PermissionError('Independent user-authorized successor required')
    auth_raw = Path(t['authorization_path']).read_bytes()
    if digest(auth_raw) != authorization_sha or t['authorization_sha256'] != authorization_sha:
        raise PermissionError('Explicit user authorization SHA mismatch')
    a = json.loads(auth_raw)
    if a['authority'] != 'current_user_explicit_instruction' or a['next_epoch'] != t['edition'] or not a['wait_for_central_release_ticket_waived_by_user'] or not a['donor_stop_user_confirmed']:
        raise PermissionError('Current user direct authority missing')
    if a['donor_durable_retirement_independently_proven'] is not False or not a['not_central_release_ticket']:
        raise PermissionError('Do not misrepresent user stop assertion as central enforced seal')
    if a['old_queue_append'] is not False or t['automatic_retry'] is not False or t['allowed_gpu_indices'] != [0,1,2,3]:
        raise PermissionError('Scope/retry/old queue restriction')
    original_root = base.B / t['predecessor_epoch'] / 'claims'
    if Path(t['claims_root']) != original_root or a['original_claim_root'] != str(original_root):
        raise PermissionError('Original claim domain must remain unchanged')
    if Path(t['output_root']) != base.B / t['edition'] / 'results' / t['worker_id']:
        raise PermissionError('New output domain binding mismatch')
    if any(t.get(k) != v for k,v in base.FLAGS.items()):
        raise PermissionError('Unreviewed policy mismatch')
    originals = {}
    for ref in t['donor_assignment_refs']:
        b = Path(ref['assignment_path']).read_bytes()
        if digest(b) != ref['assignment_sha256']:
            raise PermissionError('Original donor assignment SHA mismatch')
        originals[ref['worker']] = ([json.loads(l) for l in b.splitlines()], ref['assignment_sha256'])
    raw = Path(t['assignment_path']).read_bytes()
    if digest(raw) != t['assignment_sha256']:
        raise PermissionError('Fresh frozen receiver assignment SHA mismatch')
    rows = [json.loads(l) for l in raw.splitlines()]
    seen = set()
    for i,row in enumerate(rows):
        original, sha = originals[row['original_worker']]
        expected = original[row['original_assignment_index']]
        if row['assignment_index'] != i or any(row.get(k) != v for k,v in expected.items()) or row['original_assignment_sha256'] != sha or row['original_gpu'] != row['original_assignment_index'] % 4:
            raise PermissionError('Exact receiver and original donor row/index binding')
        key = row['split'] + '/' + row['id']
        if key in seen:
            raise PermissionError('Duplicate receiver source key')
        seen.add(key)
    if len(rows) != t['source_count']:
        raise PermissionError('Frozen receiver count mismatch')
    for ref in t['code_refs']:
        if digest((base.M/ref['name']).read_bytes()) != ref['sha256']:
            raise PermissionError('Frozen algorithm mismatch')
    return t, a, rows


class TransferWorker(base.Worker):
    def __init__(self, args):
        t, self.authority, rows = load_contract(args.ticket, args.ticket_sha256, args.authorization_sha256)
        if t['worker_id'] != args.worker or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
            raise PermissionError('Physical GPU/node worker visibility mismatch')
        super().__init__(args)
        self.ticket=t; self.output=Path(t['output_root']); self.full_assignment={r['split']+'/'+r['id']:r for r in rows}
        self.rows=[r for r in rows if r['assignment_index']%4==self.gpu]; self.phase_total=len(self.rows)
        self.claim_records={}; self.bootstrap_phase=False
        self.core.claim_once=self.claim_atomic
        self.core.write_once(self.root/'new_epoch_ticket.json',self.core.json_bytes(t))
        self.core.write_once(self.root/'new_epoch_partition.json',self.core.json_bytes(self.rows))

    def shared_device(self):
        mount='/apdcephfs_gy4/share_302533218'
        if not any(l.split()[1:3]==[mount,'fuse.dop-fuse'] for l in Path('/proc/mounts').read_text().splitlines()):
            raise OSError(107,'No true FUSE; never admit private shadow claim')
        raw=Path(self.ticket['authorization_path']).read_bytes()
        if digest(raw)!=self.args.authorization_sha256:
            raise PermissionError('Live exact authority readback differs')
        return Path(mount).stat().st_dev

    def claim_atomic(self,path,value):
        self.safe();dev=self.shared_device();row=value['source'];expected=Path(self.ticket['claims_root'])/row['split']/(row['id']+'.json')
        if Path(path)!=expected or self.full_assignment.get(row['split']+'/'+row['id'])!=row:
            raise PermissionError('Source outside fresh receiver frozen assignment')
        record=dict(value,source_row=row,original_worker=row['original_worker'],original_assignment_index=row['original_assignment_index'],original_assignment_sha256=row['original_assignment_sha256'],authorization_sha256=self.args.authorization_sha256,direct_user_authority=True,donor_persistent_retirement_not_claimed=True,budget_consumed=True)
        result=consume_original_claim(self.ticket['claims_root'],row,record,expected_device=dev)
        self.claim_records[row['split']+'/'+row['id']]=result

    def enqueue(self,row,executor,batch_number=None):
        try:
            item=super().enqueue(row,executor,batch_number=batch_number)
        except ConsumptionUncertain:
            self.abort.set();raise
        return item

    def reserve(self,row):
        return io_wait(lambda:super(TransferWorker,self).reserve(row),'input size reservation before copy only')

    def guarded_publish(self,src,dst):
        src,dst=Path(src),Path(dst);sha=digest(src.read_bytes());size=src.stat().st_size
        def action():
            dev=self.shared_device();mkdir_sync(dst.parent)
            directory=os.open(dst.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                if os.fstat(directory).st_dev!=dev:
                    raise OSError(107,'Publication directory is not pinned shared filesystem')
                def read(name):
                    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                    with os.fdopen(fd,'rb') as f:return f.read()
                try:
                    existing=read(dst.name)
                except FileNotFoundError:existing=None
                if existing is not None:
                    if len(existing)!=size or digest(existing)!=sha:raise ValueError('Existing result differs; never overwritten')
                else:
                    temp='.pending-'+str(os.getpid())+'-'+str(time.time_ns())
                    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
                    with src.open('rb') as a,os.fdopen(fd,'wb') as b:
                        shutil.copyfileobj(a,b,1024*1024);b.flush();os.fsync(b.fileno())
                    if digest(read(temp))!=sha:raise ValueError('Shared temporary publication readback mismatch')
                    try:os.link(temp,dst.name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
                    except FileExistsError:
                        if digest(read(dst.name))!=sha:raise ValueError('Concurrent write-once result differs')
                    os.unlink(temp,dir_fd=directory)
                os.fsync(directory)
                actual=read(dst.name)
                if digest(actual)!=sha or len(actual)!=size:raise ValueError('Final result SHA/size readback')
                self.shared_device()
                return dict(path=str(dst),bytes=size,sha256=sha)
            finally:os.close(directory)
        return io_wait(action,'guarded shared publication '+str(dst))

    def publish_batch(self):
        if not self.batch:return
        self.safe();self.state['stage']='batch_publish';batch=self.batch;batchid=batch[0]['batchid'];directory=self.output/'batches'/batchid;archive=self.root/(batchid+'.tar');started=time.perf_counter();payloads={};files=[]
        for item in batch:
            key=item['row']['split']+'/'+item['row']['id'];success=item['result']['status']=='extracted_rule_draft_unapproved'
            for name in self.core.FILES+('sealed_failure.json',):
                p=item['case']/name
                if p.is_file():
                    raw=p.read_bytes();rel='cases/'+key+'/'+name;payloads[rel]=raw
                    files.append(dict(path=str(directory/rel),archive_path=str(directory/'batch.tar'),archive_member=rel,path_kind='logical_binding_archived_member_not_scattered_publication',bytes=len(raw),sha256=digest(raw)))
                elif success and name in self.core.FILES:raise ValueError('Successful source missing five artifacts')
            actual=json.loads((item['case']/'receipt.json').read_bytes())
            if actual['status']!=item['result']['status'] or actual['source_row']!=item['row']:
                raise ValueError('Post-extraction failure has conflicting receipt; retained without retry')
        with archive.open('xb') as f:
            with tarfile.open(fileobj=f,mode='w:') as tar:
                for name,raw in sorted(payloads.items()):
                    member=tarfile.TarInfo(name);member.size=len(raw);member.mtime=0;member.mode=0o600;tar.addfile(member,io.BytesIO(raw))
            f.flush();os.fsync(f.fileno())
        if self.reserved+archive.stat().st_size>base.BUDGET:raise ValueError('Bounded cache+archive exceeded; retained')
        package=self.guarded_publish(archive,directory/'batch.tar')
        commit=dict(schema='ms_unified_novlm_batch_commit_v1',edition=self.ticket['edition'],worker=self.worker,gpu=self.gpu,owner_pid=os.getpid(),ticket_id=batchid,assignment_sha256=self.ticket['assignment_sha256'],batch_sha256=package['sha256'],package=package,files=files,receipts=[i['result'] for i in batch],unique_sources_attempted=len(batch),prefetch_drained=True,publication_scope='bundle-only original five artifacts; real archive/member locators; no media or model weights',profile_seconds=dict(pack_publish=time.perf_counter()-started,whole_e2e=time.perf_counter()-self.begin),commit_unix=time.time(),authorization_sha256=self.args.authorization_sha256,authority='direct_user_authorized_new_epoch',original_claim_root=self.ticket['claims_root'],original_claims=[self.claim_records[i['row']['split']+'/'+i['row']['id']] for i in batch],**base.FLAGS)
        cp=self.root/(batchid+'.commit.json');self.core.write_once(cp,self.core.json_bytes(commit));published=self.guarded_publish(cp,directory/'commit.json')
        if self.bootstrap_phase:
            if any(i['result']['status']!='extracted_rule_draft_unapproved' or not i['result']['bootstrap_evidence'] for i in batch):raise ValueError('Fresh three-case bootstrap lacks actual supported actions')
            from audit_results_checkpointed import check_batch
            summary,_=check_batch(directory/'commit.json',self.ticket,self.full_assignment,{})
            if summary['exact_rules_recomputed']!=3:raise ValueError('Fresh3 complete CPU audit did not match')
            proof=dict(worker=self.worker,gpu=self.gpu,pid=os.getpid(),commit_path=str(directory/'commit.json'),commit_sha256=published['sha256'],exact_rules_recomputed=3,new_source_rows=[i['row']['assignment_index'] for i in batch],epoch=self.ticket['edition'],**base.FLAGS)
            bp=Path(self.args.run_root)/'bootstrap_verified.json';self.core.write_once(bp,self.core.json_bytes(proof));self.guarded_publish(bp,self.output/'bootstrap_verified.json');self.emit('bootstrap_new_epoch_verified',proof=proof)
        self.times['publication']+=time.perf_counter()-started
        for item in batch:
            success=item['result']['status']=='extracted_rule_draft_unapproved';self.counts['published']+=int(success);self.counts['published_frames']+=item['result'].get('frame_count',0) if success else 0;self.counts['publish_backlog']-=1;self.records.append(item['result'])
        self.emit('batch_commit',commit_path=str(directory/'commit.json'),package_path=package['path'],sources=len(batch),counts=dict(self.counts),seconds=time.perf_counter()-started)
        for item in batch:
            own=item['local_root']/item['row']['split']/item['row']['id']
            if own.is_dir():shutil.rmtree(own)
            shutil.rmtree(item['case']);self.reserved-=item['reservation']
        archive.unlink();self.batch=[];self.batch_number+=1
        return commit

    def run(self):
        self.shared_device()
        if self.args.require_bootstrap:
            proof=read_json_shared(self.output/'bootstrap_verified.json')
            if proof['epoch']!=self.ticket['edition'] or proof['exact_rules_recomputed']!=3:raise PermissionError('Current independent epoch bootstrap not accepted')
        # Reuse actual native reservation/UUID/FLOCK/memory/disk resource gates unchanged.
        self.acquire_resources()
        monitor=threading.Thread(target=self.monitor,daemon=True);monitor.start();self.state['stage']='models_load_once';started=time.perf_counter()
        models=self.core.load_models(dict(gpu_index=self.gpu,scail_root=self.args.scail_root,beat_checkpoint=self.args.beat_checkpoint));self.models_init=dict(dwpose_count=1,beatthis_count=1,seconds=time.perf_counter()-started,actual_sessions={'detector':models[0].session_det.get_providers(),'pose':models[0].session_pose.get_providers()})
        self.safe();self.emit('models_ready',initialization=self.models_init,resource=self.check_resources(),new_epoch=True,remaining_preallocated_sources=len(self.rows))
        rows=self.rows
        if self.args.bootstrap_three:
            self.bootstrap_phase=True;first=rows[:3];self.segment(first,models);self.bootstrap_phase=False;rows=rows[3:]
        self.segment(rows,models)
        if self.counts['published']+self.counts['failed']+self.counts['skipped']!=self.phase_total or self.counts['publish_backlog']!=0:raise ValueError('Finite transfer partition not terminal')
        self.monitor_stop.set();monitor.join(timeout=15)
        summary=dict(worker=self.worker,gpu=self.gpu,pid=os.getpid(),edition=self.ticket['edition'],assignment_sha256=self.ticket['assignment_sha256'],assigned_sources=self.phase_total,counts=self.counts,models_initialized_once=self.models_init,remaining_preallocated_sources=0,elapsed_s=time.perf_counter()-self.begin,automatic_retries=0,authority='direct_user_authorization',donor_durable_retirement_independently_proven=False,**base.FLAGS)
        p=self.root/'queue_summary.json';self.core.write_once(p,self.core.json_bytes(summary));self.guarded_publish(p,self.output/('gpu'+str(self.gpu)+'_queue_summary.json'));self.emit('queue_exhausted',remaining_preallocated_sources=0,counts=self.counts)
        return 1 if self.counts['failed'] else 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=list(base.EXPECTED),required=True);p.add_argument('--gpu',type=int,choices=range(4),required=True);p.add_argument('--run-root',required=True);p.add_argument('--ticket',required=True);p.add_argument('--ticket-sha256',required=True);p.add_argument('--authorization-sha256',required=True);p.add_argument('--scail-root',required=True);p.add_argument('--beat-checkpoint',required=True);p.add_argument('--bootstrap-three',action='store_true');p.add_argument('--require-bootstrap',action='store_true');p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if args.preflight_only:
        t,a,rows=load_contract(args.ticket,args.ticket_sha256,args.authorization_sha256);print(json.dumps(dict(preflight_passed=True,worker=args.worker,source_count=len(rows),partition_count=sum(r['assignment_index']%4==args.gpu for r in rows),authority=a['authority'],original_claim_domain=t['claims_root'],real_claims_created=0)));return
    w=None
    try:w=TransferWorker(args);exitcode=w.run()
    except Exception as e:
        if w:w.abort.set();w.monitor_stop.set();w.emit('fatal_error',error=str(e),traceback=traceback.format_exc(),no_retry=True)
        else:print(json.dumps(dict(event='startup_denied_no_claims',error=str(e))),flush=True)
        exitcode=2
    finally:
        if w:
            w.monitor_stop.set();w.log.flush();os.fsync(w.log.fileno());w.log.close()
            if w.created_pause:
                try:
                    import subprocess
                    subprocess.run(['/opt/conda/envs/torch-base/bin/python','/dockerdata/wm_tmp/gpu_hold.py','--action','resume','--gpus',str(w.gpu),'--state-dir','/dockerdata/wm_logs/gpu_hold'],check=True,timeout=30)
                except Exception:print(json.dumps(dict(event='owned_pause_cleanup_alarm',gpu=w.gpu,no_blind_pause_removal=True)),flush=True)
            for lock in reversed(w.locks):lock.close()
    raise SystemExit(exitcode)

if __name__=='__main__':main()
