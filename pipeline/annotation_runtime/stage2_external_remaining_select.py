"""Reserve all published03-06 remaining MS sources from small commit receipts only."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import stage2_external_exact50000_select_v1 as s

ROOT=s.ROOT;RT=s.RT;OUT=ROOT/'stage2_external03to06_remaining_inputs_20261028_v1'


def run():
    OUT.mkdir(exist_ok=False);regpath=RT/'ms_vlm_postprocess_jobs_registry_v1.json';excluded=s.reserve_keys(json.loads(regpath.read_bytes()));legacy=ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/source_selection_proof.json'
    if legacy.exists():excluded.update(k[1].casefold() for k in json.loads(legacy.read_bytes())['source_keys'])
    batches=[];refs=[]
    for i in range(3,7):
        worker='a100-extra-0'+str(i);tp=s.BASE/'assignments'/(worker+'.ticket.json');ticket=json.loads(tp.read_bytes());ap=Path(ticket['assignment_path']);raw=ap.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ticket['assignment_sha256']:raise ValueError('Original external assignment changed')
        assignment=[json.loads(x) for x in raw.splitlines() if x.strip()]
        if ticket['source_count']!=20000 or len(assignment)!=20000 or ticket['worker_id']!=worker:raise ValueError('Exact external assignment scope differs')
        refs.append(dict(ticket_ref=s.reference(tp),assignment_ref=dict(path=str(ap),sha256=ticket['assignment_sha256'])))
        base=Path(ticket['output_root'])/'batches'
        for lane in sorted(base.iterdir()):
            if not lane.is_dir():continue
            for b in sorted(lane.iterdir()):
                if b.is_dir():batches.append((b/'commit.json',worker,ticket,assignment))
    rows=[];seen=set();commits=[];started=time.time()
    def parse(item):
        cp,worker,ticket,assignment=item
        if not cp.is_file():return [],None
        raw=cp.read_bytes()
        if len(raw)>8*2**20:raise ValueError('Commit metadata exceeds bound')
        c=json.loads(raw)
        if c.get('schema')!='ms_unified_novlm_batch_commit_v1' or c.get('edition')!='external_ms_unified_clusters80k_20261027_v1':return [],None
        if c['worker']!=worker or c['assignment_sha256']!=ticket['assignment_sha256'] or any(c.get(k) is not False for k in ('quality_approved','semantic_approval','training_eligible')):raise ValueError('Original published scope differs')
        filemap={x['path']:x for x in c['publications']}
        if len(filemap)!=len(c['publications']):raise ValueError('Repeated published metadata path')
        values=[]
        for receipt in c['receipts']:
            row=receipt['source_row'];id_=row['id'];split=row['split']
            if receipt['status']!='extracted_rule_draft_unapproved' or id_.casefold() in excluded:continue
            index=receipt['assignment_index']
            if type(index) is not int or not 0<=index<len(assignment) or row!=assignment[index] or receipt.get('vlm_called') is not False or any(receipt.get(k) is not False for k in ('quality_approved','semantic_approval','training_eligible')):raise ValueError('Original source/receipt/assignment differs')
            if not re.fullmatch('[A-Za-z0-9_-]+',id_) or split not in ('train','val','test'):raise ValueError('Exact portable source identity required')
            directory=cp.parent/'cases'/split/id_;refs=[]
            for name in ('draft.json','fullframe_manifest.json'):
                p=directory/name;x=filemap[str(p)];published=x['receipt']
                if x['sha256']!=published['sha256'] or published['path']!=str(p) or not re.fullmatch('[a-f0-9]{64}',x['sha256']) or not 0<published['bytes']<=2**20:raise ValueError('Published original JSON reference differs')
                refs.append(dict(path=str(p),sha256=x['sha256']))
            values.append(dict(id=id_,split=split,video_path=row['video_path'],rule_draft_ref=refs[0],fullframe_manifest_ref=refs[1],rich_text=''))
        return values,dict(path=str(cp),sha256=hashlib.sha256(raw).hexdigest())
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(batches),64):
            for values,ref in pool.map(parse,batches[offset:offset+64]):
                ids=[]
                for row in values:
                    k=row['id'].casefold()
                    if k in seen:continue
                    seen.add(k);rows.append(row);ids.append(row['id'])
                if ids:commits.append(dict(commit_ref=ref,selected_source_ids=ids))
            print(json.dumps(dict(event='external03to06_remaining_source_scan',rows=len(rows),scanned_batches=min(offset+64,len(batches)),seconds=time.time()-started,caption_reads=0,archive_reads=0,video_reads=0)),flush=True)
    fd=os.open(regpath.with_suffix('.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reg=json.loads(regpath.read_bytes());fresh=s.reserve_keys(reg);rows=[r for r in rows if r['id'].casefold() not in fresh]
        if not rows:raise ValueError('No unused published external source remains')
        plan=s.once(OUT/'manifest.jsonl',b''.join(s.encoded(r) for r in rows));keys=s.once(OUT/'source_keys.jsonl',b''.join(s.encoded([r['split'],r['id']]) for r in rows));phase='stage2_external03to06_remaining_source_group_20261028';cp=RT/(phase+'_descriptor.json');d=dict(schema='stage2_remaining_source_reservation_v1',phase=phase,node='MAIN_DYNAMIC',gpus=[],config_path=str(cp),out=str(OUT),queue=str(ROOT/'queues'/phase),plan={k:plan[k] for k in ('path','sha256')},targetcount=len(rows),input_count=len(rows),selected_source_ids=[r['id'] for r in rows],not_a_launch_config=True);dr=s.once(cp,s.encoded(d));reg['jobs'].append(dict(node=d['node'],gpus=[],phase=phase,config_path=str(cp),plan_ref=d['plan'],selected_source_ids=d['selected_source_ids'],sources=len(rows),expected_running=False,user_stage_authorized=True,worker_status='permanently_reserved_external_remaining_input_group',quality_approved=False,training_eligible=False))
        with tempfile.NamedTemporaryFile(dir=RT,prefix='.vlm-registry-',delete=False) as f:temp=Path(f.name);f.write(s.encoded(reg));f.flush();os.fsync(f.fileno())
        os.replace(temp,regpath);df=os.open(RT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df);allocation=s.once(OUT/'allocation.json',s.encoded(dict(source_count=len(rows),manifest_ref=plan,source_keys_ref=keys,registry_descriptor_ref=dr,assignment_refs=refs,selected_commits=commits,source_claims_created=0,models_started=0,registered_overlap=0,caption_reads=0,archive_reads=0,video_reads=0,quality_approved=False,training_eligible=False,observed_unix=time.time())))
    print(json.dumps(dict(event='remaining_external03to06_inputs_reserved',sources=len(rows),manifest_ref=plan,allocation_ref=allocation)),flush=True)

if __name__=='__main__':run()
