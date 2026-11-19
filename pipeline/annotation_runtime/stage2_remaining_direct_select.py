"""Reserve all remaining published direct inputs in this epoch, no caption/media audit."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import stage2_external_exact50000_select_v1 as s

ROOT=s.ROOT; RT=s.RT
OUT=ROOT/'stage2_main_remaining_direct_20261028_v1'


def run():
    if (OUT/'allocation.json').exists(): raise RuntimeError('Existing reservation must not be rerun')
    OUT.mkdir(exist_ok=False)
    registry=RT/'ms_vlm_postprocess_jobs_registry_v1.json';excluded=s.reserve_keys(json.loads(registry.read_bytes()))
    legacy=ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/source_selection_proof.json'
    if legacy.exists(): excluded.update(k[1].casefold() for k in json.loads(legacy.read_bytes())['source_keys'])
    candidates=[];inputs=[]
    for worker in ('a100-extra-01','a100-extra-02'):
        tp=s.BASE/'assignments'/(worker+'.ticket.json');ticket=json.loads(tp.read_bytes());p=Path(ticket['assignment_path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ticket['assignment_sha256']:raise ValueError('Assignment metadata changed')
        assignment=[json.loads(line) for line in raw.splitlines() if line.strip()]
        inputs.append(dict(ticket_ref=s.reference(tp),assignment_ref=dict(path=str(p),sha256=ticket['assignment_sha256'])))
        candidates.extend((d/'commit.json',worker,ticket,assignment) for d in sorted((s.BASE/'results'/worker/'batches').iterdir()) if d.is_dir())
    rows=[];seen=set();commits=[];started=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(candidates),64):
            for values,commit in pool.map(s.parse_batch,candidates[offset:offset+64]):
                selected=[]
                for row in values:
                    k=row['id'].casefold()
                    if k in excluded or k in seen:continue
                    seen.add(k);rows.append(row);selected.append(row['id'])
                if selected:commit['selected_source_ids']=selected;commits.append(commit)
            print(json.dumps(dict(event='remaining_direct_metadata_scan',candidates=len(rows),scanned_batches=min(offset+64,len(candidates)),elapsed_seconds=time.time()-started,caption_reads=0,archive_reads=0,video_reads=0)),flush=True)
    fd=os.open(registry.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reg=json.loads(registry.read_bytes());fresh=s.reserve_keys(reg);rows=[r for r in rows if r['id'].casefold() not in fresh]
        if not rows:raise ValueError('No unallocated published direct input remains')
        plan=s.once(OUT/'manifest.jsonl',b''.join(s.encoded(r) for r in rows));keys=s.once(OUT/'source_keys.jsonl',b''.join(s.encoded([r['split'],r['id']]) for r in rows))
        phase='stage2_main_remaining_direct_source_group_20261028';cp=RT/(phase+'_descriptor.json');d=dict(schema='stage2_remaining_source_reservation_v1',config_path=str(cp),phase=phase,node='MAIN_DYNAMIC',gpus=[],out=str(OUT),queue=str(ROOT/'queues'/phase),plan={k:plan[k] for k in ('path','sha256')},targetcount=len(rows),input_count=len(rows),selected_source_ids=[r['id'] for r in rows],not_a_launch_config=True)
        ref=s.once(cp,s.encoded(d));reg['jobs'].append(dict(node='MAIN_DYNAMIC',gpus=[],phase=phase,config_path=str(cp),plan_ref=d['plan'],selected_source_ids=d['selected_source_ids'],sources=len(rows),expected_running=False,worker_status='permanently_reserved_next_dynamic_group',user_stage_authorized=True,quality_approved=False,training_eligible=False))
        with tempfile.NamedTemporaryFile(dir=RT,prefix='.vlm-registry-',delete=False) as f:
            temporary=Path(f.name);f.write(s.encoded(reg));f.flush();os.fsync(f.fileno())
        os.replace(temporary,registry);df=os.open(RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
        allocation=s.once(OUT/'allocation.json',s.encoded(dict(schema='stage2_remaining_direct_allocation_v1',source_count=len(rows),manifest_ref=plan,source_keys_ref=keys,registry_descriptor_ref=ref,assignment_refs=inputs,published_commit_refs=commits,source_claims_created=0,models_started=0,registered_overlap=0,caption_reads=0,archive_reads=0,video_reads=0,observed_unix=time.time(),quality_approved=False,training_eligible=False)))
    print(json.dumps(dict(event='all_remaining_direct_inputs_reserved',sources=len(rows),manifest_ref=plan,allocation_ref=allocation)),flush=True)

if __name__=='__main__':run()
