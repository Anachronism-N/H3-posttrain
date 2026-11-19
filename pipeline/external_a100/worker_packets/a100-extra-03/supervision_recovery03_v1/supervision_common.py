"""Read-only contracts for independent supervision; no worker lifecycle operations."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time
import xml.etree.ElementTree as ET
OLD_ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
REPORT_ROOT = OLD_ROOT / 'supervision_recovery03_v1'
P = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
RECOVERY_ROOT = OLD_ROOT / 'recovery03_unclaimed_v1'
WORKERS = ['a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06']
FLAGS = {'diagnostic_not_production':True,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}

def strict_json(raw):
    if not isinstance(raw, bytes) or b'\0' in raw or not raw.endswith(b'\n'):
        raise ValueError('NUL/partial JSON retained; fail closed')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value: raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def sha(raw): return hashlib.sha256(raw).hexdigest()

def assignment_scope(p=P, expected=20000):
    assignments, tickets, global_ids = {}, {}, set()
    for worker in WORKERS:
        ticket = strict_json((p/'assignments'/(worker+'.ticket.json')).read_bytes())
        path = Path(ticket['assignment_path'])
        if not path.resolve().is_relative_to((p/'assignments').resolve()): raise ValueError('assignment path outside frozen scope')
        raw = path.read_bytes()
        if b'\0' in raw or not raw.endswith(b'\n') or sha(raw) != ticket['assignment_sha256'] or ticket['worker_id'] != worker:
            raise ValueError('original assignment identity/SHA mismatch')
        rows = [strict_json(line+b'\n') for line in raw.split(b'\n')[:-1]]
        local = {(r['split'].casefold(),r['id'].casefold()) for r in rows}
        if len(rows) != expected or len(local) != expected or global_ids.intersection(local):
            raise ValueError('assignment count/duplicate/global-disjoint mismatch')
        global_ids.update(local); assignments[worker] = rows; tickets[worker] = ticket
    return assignments, tickets

def validate_commit(path, raw, assignments, tickets, existing, p=P):
    path = Path(path); commit = strict_json(raw); worker = commit['worker']; gpu = commit['gpu']
    if worker not in WORKERS or type(gpu) is not int or gpu not in range(4): raise ValueError('worker/GPU mismatch')
    if not path.resolve().is_relative_to((p/'results'/worker/'batches'/('gpu'+str(gpu))).resolve()) or path.name != 'commit.json':
        raise ValueError('commit path outside original worker/lane scope')
    if commit['edition'] != p.name or commit['assignment_sha256'] != tickets[worker]['assignment_sha256']: raise ValueError('commit edition/assignment mismatch')
    staged, counts = {}, {'partial':0,'draft_generated':0,'needs_regeneration':0,'failed':0}
    frames = events = 0
    publications = commit['publications']; pubpaths = set()
    for item in publications:
        target = Path(item['path'])
        if not target.resolve().is_relative_to(path.parent.resolve()) or item['path'] in pubpaths: raise ValueError('unsafe/duplicate publication path')
        pubpaths.add(item['path'])
        receipt = item['receipt']
        if type(receipt.get('bytes')) is not int or receipt['bytes'] < 0 or item['sha256'] != receipt['sha256'] or receipt.get('path') != item['path']:
            raise ValueError('publication verified receipt bytes/SHA/path mismatch')
        if not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']): raise ValueError('invalid publication SHA')
    for receipt in commit['receipts']:
        index = receipt['assignment_index']; row = receipt['source_row']
        if type(index) is not int or not 0 <= index < len(assignments[worker]) or index%4 != gpu or row != assignments[worker][index]:
            raise ValueError('source original assignment/lane mismatch')
        if receipt['status'] == 'existing_claim_not_reassigned': continue
        key = row['split']+'/'+row['id']; globalkey = (row['split'].casefold(),row['id'].casefold())
        if key in staged or globalkey in existing: raise ValueError('duplicate globally committed source')
        if any(receipt.get(k)!=v for k,v in FLAGS.items()): raise ValueError('receipt approval/diagnostic flags mismatch')
        status = 'failed' if receipt['status']=='failed_sealed_no_retry' else receipt['draft_status']
        if status not in counts: raise ValueError('unknown source terminal status')
        n,e = receipt.get('frame_count',0),receipt.get('event_count',0)
        if type(n) is not int or n<0 or type(e) is not int or e<0: raise ValueError('invalid source counters')
        if status != 'failed':
            if status != 'needs_regeneration' and (not receipt.get('generated_with_action_events') or not e): raise ValueError('draft without action event evidence')
            case = path.parent/'cases'/row['split']/row['id']
            if not {str(case/name) for name in ['pose.npz','fullframe_manifest.json','beats.json','draft.json','receipt.json']}.issubset(pubpaths):
                raise ValueError('committed source lacks published five-file records')
        staged[key] = {'assignment_index':index,'status':status,'frames':n,'events':e,'commit_path':str(path)}
        counts[status]+=1;frames+=n;events+=e
    if len(staged)!=commit['unique_sources_attempted'] or sum(bool(s['events']) for s in staged.values())!=commit['unique_sources_with_action_events']:
        raise ValueError('commit source totals mismatch')
    return commit, staged, {'sha256':sha(raw),'sources':len(staged),'frames':frames,'counts':counts,'events':events}

def validate_cursor(state, assignments, tickets, p=P):
    if set(state['sources']) != set(WORKERS): raise ValueError('cursor worker scope mismatch')
    reconstructed = {w:{} for w in WORKERS}; global_ids=set(); commits={}
    for key, record in state['commits'].items():
        raw=Path(key).read_bytes()
        if sha(raw)!=record['sha256']: raise ValueError('cursor commit SHA changed')
        commit, staged, metadata=validate_commit(key,raw,assignments,tickets,global_ids,p)
        if any(record.get(k)!=v for k,v in metadata.items()): raise ValueError('cursor commit metadata mismatch')
        reconstructed[commit['worker']].update(staged)
        global_ids.update((k.split('/',1)[0].casefold(),k.split('/',1)[1].casefold()) for k in staged)
        commits[key]=commit
    if reconstructed != state['sources']: raise ValueError('cursor source map not exact commit reconstruction')
    if not isinstance(state.get('latest_changes'),dict) or not isinstance(state.get('invalid_commit_alarms'),dict): raise ValueError('invalid cursor schema')
    return commits

def bind_ledger(ledger,state,commits):
    if set(ledger['sources'])!=set(WORKERS): raise ValueError('deep ledger worker scope mismatch')
    result=json.loads(json.dumps(ledger)); expected={w:set() for w in WORKERS}
    for key, record in result['commits'].items():
        if key not in state['commits'] or key not in commits: raise ValueError('deep ledger outside verified cursor')
        commit=commits[key]; w=commit['worker']; selected={k:v for k,v in state['sources'][w].items() if v['commit_path']==key}
        if record['archive_SHA256']!=commit['batch_sha256']: raise ValueError('deep ledger archive hash binding mismatch')
        archives=[p for p in commit['publications'] if p['path']==str(Path(key).parent/'batch.tar')]
        if len(archives)!=1 or archives[0]['sha256']!=record['archive_SHA256']: raise ValueError('deep ledger committed archive mismatch')
        totals={'sources':len(selected),'frames':sum(s['frames'] for s in selected.values() if s['status']!='failed'),'events':sum(s['events'] for s in selected.values() if s['status']!='failed'),'partial':sum(s['status']=='partial' for s in selected.values()),'failed':sum(s['status']=='failed' for s in selected.values())}
        if any(record.get(k)!=v for k,v in totals.items()): raise ValueError('deep ledger source totals mismatch')
        for source, value in selected.items():
            deep=result['sources'][w].get(source)
            if not deep or deep['commit']!=key or deep['status']!=value['status']: raise ValueError('deep ledger source commit/status mismatch')
            if value['status']!='failed' and (deep.get('actual_artifacts_deep_verified') is not True or deep.get('frames')!=value['frames'] or deep.get('events')!=value['events']): raise ValueError('deep ledger artifacts not evidenced')
            expected[w].add(source)
        record['commit_sha256']=state['commits'][key]['sha256']
    if any(set(result['sources'][w])!=expected[w] for w in WORKERS): raise ValueError('deep ledger extra/missing sources')
    return result

def start_ticks(pid, proc=Path('/proc')):
    value=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    if value[0] in ['Z','X']: raise ValueError('zombie/exited PID')
    return int(value[19])

def pid_identity(record, proc=Path('/proc')):
    result={'pid':record.get('pid'),'start_ticks':record.get('start_ticks'),'alive':False,'exact_pid_start_ticks':False,'observed_unix':time.time()}
    try:
        if type(record.get('pid')) is not int or record['pid']<=0 or type(record.get('start_ticks')) is not int: raise ValueError('missing exact PID/startticks')
        observed=start_ticks(record['pid'],proc); result['observed_start_ticks']=observed
        result['alive']=result['exact_pid_start_ticks']=observed==record['start_ticks']
        if not result['alive']: result['alarm']='PID_reuse_identity_mismatch'
    except (OSError,ValueError,IndexError) as e: result['alarm']=str(e)
    return result

def flock_fd(pid,lock_path,proc=Path('/proc')):
    stat=Path(lock_path).stat(); target=(stat.st_dev,stat.st_ino)
    for fd in (proc/str(pid)/'fd').iterdir():
        try:
            actual=fd.stat()
            if (actual.st_dev,actual.st_ino)!=target: continue
            for line in (proc/str(pid)/'fdinfo'/fd.name).read_text().splitlines():
                fields=line.split()
                if fields[:1]!=['lock:'] or 'FLOCK' not in fields or 'WRITE' not in fields or '->' in fields: continue
                i=fields.index('WRITE'); device=fields[i+2].split(':')
                if int(fields[i+1])!=pid or len(device)!=3: continue
                if (int(device[0],16),int(device[1],16),int(device[2]))!=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino): continue
                return {'held':True,'fd':int(fd.name),'device':stat.st_dev,'inode':stat.st_ino,'evidence':line}
        except (OSError,ValueError,IndexError): continue
    return {'held':False,'device':stat.st_dev,'inode':stat.st_ino}

def gpu_evidence(gpu):
    value=subprocess.run(['nvidia-smi','-i',str(gpu),'-q','-x'],check=True,capture_output=True,text=True,timeout=15)
    nodes=ET.fromstring(value.stdout).findall('gpu')
    if len(nodes)!=1: raise ValueError('single GPU XML required')
    node=nodes[0]; uuid=node.findtext('uuid')
    index=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10).stdout.strip().split(',')
    if len(index)!=2 or int(index[0])!=gpu or index[1].strip()!=uuid: raise ValueError('GPU index/UUID mismatch')
    computes=[p for p in node.findall('processes/process_info') if p.findtext('type')=='C']
    return {'gpu_uuid':uuid,'compute_pids':[int(p.findtext('pid')) for p in computes],'all_process_pids':[int(p.findtext('pid')) for p in node.findall('processes/process_info')]}

def lane_identity(record,gpu,root,lease_path,lock_path=None,proc=Path('/proc'),evidence=None):
    result=pid_identity(record,proc); result['resident_contract_ok']=False
    if not result['alive']: return result
    try:
        lease=strict_json(Path(lease_path).read_bytes())
        for key in ['pid','start_ticks','worker','physical_gpu_index','assignment_sha256','queue_sha256']:
            if lease.get(key)!=record.get(key): raise ValueError('lease/progress identity mismatch: '+key)
        if record.get('physical_gpu_index')!=gpu or lease.get('status','running').startswith('released'): raise ValueError('GPU/released lease mismatch')
        lock=flock_fd(record['pid'],lock_path or Path(root)/'gpu_locks'/('physical_gpu'+str(gpu)+'.lock'),proc)
        result['flock']=lock
        if not lock['held']: raise ValueError('missing real FLOCK WRITE FD inode')
        gpu_state=gpu_evidence(gpu) if evidence is None else evidence
        result['gpu_evidence']=gpu_state
        if gpu_state['gpu_uuid']!=lease.get('gpu_uuid') or gpu_state['compute_pids']!=[record['pid']] or gpu_state['all_process_pids']!=[record['pid']]: raise ValueError('single GPU UUID/compute ownership mismatch')
        environ=(proc/str(record['pid'])/'environ').read_bytes().split(b'\0')
        if ('CUDA_VISIBLE_DEVICES='+str(gpu)).encode() not in environ: raise ValueError('original authorized CUDA GPU mapping mismatch')
        if start_ticks(record['pid'],proc)!=record['start_ticks']: raise ValueError('PID changed during probe')
        result['resident_contract_ok']=True
    except (OSError,ValueError,KeyError,subprocess.SubprocessError,ET.ParseError) as e: result['alarm']=str(e)
    return result

def overlay_lane(original,recovery=None):
    # Original uncommitted work is NOT additive completed/published work.
    baseline=original.get('published',0); processed=original.get('unique_processed',0)
    out=dict(original); out['original_progress']=dict(original)
    out['original_published_counter']=baseline; out['original_uncommitted_processed_not_added']=processed-baseline
    out['recovery_progress']=dict(recovery) if recovery else None
    out['recovery_published_increment']=recovery.get('published',0) if recovery else 0
    out['resident_published_counter_overlay']=baseline+out['recovery_published_increment']
    out['counter_scope']='resident progress only; supervisor exact globally deduplicated commits authoritative'
    out['claimed_sources_completion']='unknown; publication-only results may change committed total; do not assume all 20000 finish'
    if recovery:
        for k in ['pid','start_ticks','status','heartbeat_unix','observed_process_identity','cache_actual_bytes','cache_reserved_bytes','current_source','current_total_frames']:
            if k in recovery: out[k]=recovery[k]
        out['unique_processed']=baseline+recovery.get('unique_processed',0); out['published']=out['resident_published_counter_overlay']
        out['failed']=original.get('failed',0)+recovery.get('failed',0); out['needs_regeneration']=original.get('needs_regeneration',0)+recovery.get('needs_regeneration',0)
    else:
        out['unique_processed']=baseline; out['published']=baseline
        out['recovery_state']='not_started'
        if not original.get('observed_process_identity',{}).get('alive'): out['status']='action_required_original_dead_recovery_not_started'
    return out

def require_initialized(root=REPORT_ROOT):
    marker=strict_json((root/'logs/initialization.json').read_bytes())
    if marker['schema']!='independent_supervision_initialization_v1': raise ValueError('invalid independent initialization')
    return marker

def duration_arguments():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--duration-seconds',type=int,default=3*86400); parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 0<args.duration_seconds<=3*86400: parser.error('duration must be positive and <=3 days; expiration never stops workers')
    return args
