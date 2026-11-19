"""Index all native completed MS sources not reserved for stage2; no payload audits."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import stage2_external_exact50000_select_v1 as s

ROOT=s.ROOT;RT=s.RT;OUT=ROOT/'stage2_native_remaining_corpus_20261028_v1'


def run():
    OUT.mkdir(exist_ok=False);regpath=RT/'ms_vlm_postprocess_jobs_registry_v1.json';excluded=s.reserve_keys(json.loads(regpath.read_bytes()))
    p=ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/source_selection_proof.json'
    if p.exists():excluded.update(k[1].casefold() for k in json.loads(p.read_bytes())['source_keys'])
    names=sorted(e.name for e in os.scandir(RT) if fnmatch.fnmatch(e.name,'ms_unified_native_v*_config.json'));configs=[];seen_outputs=set()
    for name in names:
        cp=RT/name;c=json.loads(cp.read_bytes());out=c['out']
        if out in seen_outputs:continue
        summary=Path(out)/'summary.json'
        if not summary.exists():continue
        v=json.loads(summary.read_bytes())
        if not v.get('completed_batches'):continue
        if v.get('status') not in ('completed','drained_not_complete') or v.get('vlm_called') is not False or v.get('allprefetch_drained') is not True or v.get('uncommitted_prefetch')!=0:raise ValueError('Native source summary is not reliably drained')
        seen_outputs.add(out);configs.append((cp,c,summary,v))
    batches=[];sources=[]
    for cp,c,sp,v in configs:
        if c['phase']!=v['edition']:raise ValueError('Original native edition differs')
        outcomes={(x['batch_index'],x['id']):x for x in v['all_outcomes']}
        sources.append(dict(config_ref=s.reference(cp),summary_ref=s.reference(sp),source_count=v['processed']))
        for b in v['completed_batches']:batches.append((c,b,outcomes))
    rows=[];seen=set();selected=[];started=time.time()
    def parse(item):
        c,b,outcomes=item;p=Path(b['commit_path']);raw=p.read_bytes();ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
        if ref['sha256']!=b['commit_sha256']:raise ValueError('Frozen committed metadata changed')
        commit=json.loads(raw)
        if commit.get('schema')!='ms_unified_novlm_batch_commit_v3' or commit['edition']!=c['phase'] or commit['batch_index']!=b['batch_index']:raise ValueError('Native batch identity differs')
        values=[]
        for r in commit['receipts']:
            row=r['source_row'];id_=row['id'];split=row['split'];x=outcomes[(b['batch_index'],id_)]
            if x['split']!=split or x['status']!=r['status'] or commit['archive_index'][id_]['source_row']!=row:raise ValueError('Original receipt/source identity differs')
            if r['status']!='extracted_rule_draft_unapproved' or id_.casefold() in excluded:continue
            if split not in ('train','val','test') or any(r.get(k) is not False for k in ('quality_approved','semantic_approval','training_eligible')):raise ValueError('Original split/unapproved scope differs')
            values.append(dict(id=id_,split=split,video_path=row['video_path'],original_native_commit_ref=ref,metadata_extract_required=True,original_archive_path=commit['archive_path'],original_archive_sha256=commit['archive_sha256']))
        return values,ref
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(batches),64):
            for values,ref in pool.map(parse,batches[offset:offset+64]):
                ids=[]
                for row in values:
                    k=row['id'].casefold()
                    if k in seen:continue
                    seen.add(k);rows.append(row);ids.append(row['id'])
                if ids:selected.append(dict(commit_ref=ref,source_ids=ids))
            print(json.dumps(dict(event='native_remaining_source_index',rows=len(rows),batches_scanned=min(offset+64,len(batches)),seconds=time.time()-started,caption_reads=0,archive_reads=0,video_reads=0)),flush=True)
    fd=os.open(regpath.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reg=json.loads(regpath.read_bytes());fresh=s.reserve_keys(reg);rows=[r for r in rows if r['id'].casefold() not in fresh]
        if not rows:raise ValueError('No remaining unused native source')
        plan=s.once(OUT/'source_index.jsonl',b''.join(s.encoded(r) for r in rows));keys=s.once(OUT/'source_keys.jsonl',b''.join(s.encoded([r['split'],r['id']]) for r in rows));phase='stage2_native_remaining_corpus_source_group_20261028';cp=RT/(phase+'_descriptor.json')
        d=dict(schema='stage2_native_metadata_source_reservation_v1',phase=phase,node='MAIN_NATIVE_DYNAMIC',gpus=[],config_path=str(cp),out=str(OUT),queue=str(ROOT/'queues'/phase),plan={k:plan[k] for k in ('path','sha256')},targetcount=len(rows),input_count=len(rows),selected_source_ids=[r['id'] for r in rows],not_a_launch_config=True,metadata_materialization_required=True)
        dr=s.once(cp,s.encoded(d));reg['jobs'].append(dict(node=d['node'],gpus=[],phase=phase,config_path=str(cp),plan_ref=d['plan'],selected_source_ids=d['selected_source_ids'],sources=len(rows),expected_running=False,user_stage_authorized=True,worker_status='permanently_reserved_native_metadata_input_group',quality_approved=False,training_eligible=False))
        with tempfile.NamedTemporaryFile(dir=RT,prefix='.vlm-registry-',delete=False) as f:
            temp=Path(f.name);f.write(s.encoded(reg));f.flush();os.fsync(f.fileno())
        os.replace(temp,regpath);df=os.open(RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
        allocation=s.once(OUT/'allocation.json',s.encoded(dict(schema='stage2_native_remaining_source_allocation_v1',source_count=len(rows),split_counts={k:sum(r['split']==k for r in rows) for k in ('train','val','test')},source_index_ref=plan,source_keys_ref=keys,registry_descriptor_ref=dr,selected_batches=selected,original_source_configs=sources,source_claims_created=0,models_started=0,payload_materialization_required=True,caption_reads=0,archive_reads=0,video_reads=0,quality_approved=False,training_eligible=False,observed_unix=time.time())))
    print(json.dumps(dict(event='all_remaining_native_ms_sources_reserved',sources=len(rows),allocation_ref=allocation,source_index_ref=plan,model_launch_ready=False)),flush=True)

if __name__=='__main__':run()
