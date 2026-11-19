"""Complete exact remaining452 original archive sources, using all old config/pending exclusions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import stage2_external_exact50000_select_v1 as s
ROOT=s.ROOT;RT=s.RT;OUT=ROOT/'stage2_archive_gap452_inputs_20261028_v1'


def run():
    p=ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/source_selection_proof.json';proof=json.loads(p.read_bytes());registry=RT/'ms_vlm_postprocess_jobs_registry_v1.json';fd=os.open(registry.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reg=json.loads(registry.read_bytes());used=s.reserve_keys(reg);keys=[k for k in proof['source_keys'] if k[1].casefold() not in used];assert len(keys)==452,'Unexpected original source gap, never duplicate pending';allowed={k[1].casefold() for k in keys};selected=[];rows=[]
        for entry in proof['selected_batches']:
            ids=[x for x in entry['source_ids'] if x.casefold() in allowed]
            if not ids:continue
            ref=entry['commit_ref'];raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];c=json.loads(raw);assert c['schema']=='ms_unified_novlm_batch_commit_v3';selected.append(dict(commit_ref=ref,source_ids=ids))
            for id_ in ids:
                source=c['archive_index'][id_]['source_row'];r=next(x for x in c['receipts'] if x['source_row']['id']==id_);assert r['source_row']==source and r['status']=='extracted_rule_draft_unapproved' and source['split']=='train';rows.append(dict(id=id_,split=source['split'],video_path=source['video_path'],original_native_commit_ref=ref,metadata_extract_required=True))
        assert len(rows)==len({r['id'].casefold() for r in rows})==452;OUT.mkdir(exist_ok=False);plan=s.once(OUT/'source_index.jsonl',b''.join(s.encoded(r) for r in rows));keyref=s.once(OUT/'source_keys.jsonl',b''.join(s.encoded([r['split'],r['id']]) for r in rows));phase='stage2_archive_gap452_source_group_20261028';cp=RT/(phase+'_descriptor.json');d=dict(schema='stage2_original_archive_gap_reservation_v1',phase=phase,node='MAIN_NATIVE_DYNAMIC',gpus=[],config_path=str(cp),out=str(OUT),queue=str(ROOT/'queues'/phase),plan=dict(path=plan['path'],sha256=plan['sha256']),targetcount=452,input_count=452,selected_source_ids=[r['id'] for r in rows],not_a_launch_config=True);dr=s.once(cp,s.encoded(d));reg['jobs'].append(dict(node=d['node'],gpus=[],phase=phase,config_path=str(cp),plan_ref=d['plan'],selected_source_ids=d['selected_source_ids'],sources=452,expected_running=False,user_stage_authorized=True,worker_status='permanently_reserved_original_archive_gap',quality_approved=False,training_eligible=False))
        with tempfile.NamedTemporaryFile(dir=RT,prefix='.vlm-registry-',delete=False) as f:temp=Path(f.name);f.write(s.encoded(reg));f.flush();os.fsync(f.fileno())
        os.replace(temp,registry);df=os.open(RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df);ar=s.once(OUT/'allocation.json',s.encoded(dict(schema='stage2_original_archive_gap_allocation_v1',source_count=452,source_index_ref=plan,source_keys_ref=keyref,registry_descriptor_ref=dr,selected_batches=selected,original_source_selection_ref=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),old_failed_recovery_not_restarted=True,registered_overlap=0,source_claims_created=0,models_started=0,quality_approved=False,training_eligible=False,observed_unix=time.time())))
    print(json.dumps(dict(original_archive_gap_reserved=452,selected_original_batches=len(selected),registered_overlap=0,allocation_ref=ar,models_called=0)),flush=True)
if __name__=='__main__':run()
