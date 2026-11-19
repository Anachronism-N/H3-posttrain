"""Finite existing external caption selection using published per-file SHA receipts."""
import argparse
from pathlib import Path
import re
import time
import ms_fullframe_vlm_postprocess_batch_v11 as v

ROOT=v.ROOT
BASE=ROOT/'external_ms_unified_clusters80k_20261027_v1'
STATUS='extracted_rule_draft_unapproved'

def published_rows(commit_path,commit,excluded,assignment=None):
    v.need(commit.get('schema')=='ms_unified_novlm_batch_commit_v1' and commit.get('edition')=='external_ms_unified_clusters80k_20261027_v1' and commit.get('worker') in ('a100-extra-01','a100-extra-02'),'Exact authorized external epoch required')
    v.need(commit.get('prefetch_drained') is True and all(commit.get(k) is False for k in ('quality_approved','semantic_approval','training_eligible')),'Published unapproved metadata scope required')
    package=commit['package'];v.need(package['path']==str(commit_path.parent/'batch.tar') and package['sha256']==commit['batch_sha256'] and type(package['bytes']) is int and package['bytes']>0,'Canonical committed package binding required')
    filemap={x['path']:x for x in commit['files']};v.need(len(filemap)==len(commit['files']),'Published file paths must be unique')
    rows=[];seen=set()
    for receipt in commit['receipts']:
        row=receipt['source_row'];id_=row['id'];split=row['split']
        v.need(isinstance(id_,str) and re.fullmatch('[A-Za-z0-9_-]+',id_) and isinstance(split,str) and re.fullmatch('[A-Za-z0-9_-]+',split),'Stable source ID and split required')
        v.need(id_.casefold() not in seen,'Duplicate source in commit');seen.add(id_.casefold())
        if receipt['status']!=STATUS or id_.casefold() in excluded:continue
        v.need(receipt.get('vlm_called') is False and all(receipt.get(k) is False for k in ('quality_approved','semantic_approval','training_eligible')) and receipt.get('needs_visual_revision') is True,'Exact successful unapproved no-VLM receipt required')
        v.need(type(row.get('assignment_index')) is int and 0<=row['assignment_index']<40000,'Original assignment source index required')
        if assignment is not None:
            original=assignment[row['assignment_index']]
            v.need({k:x for k,x in row.items() if k!='assignment_index'}==original,'Actual published source differs from original assignment row')
        directory=commit_path.parent/'cases'/split/id_
        draft=directory/'draft.json';manifest=directory/'fullframe_manifest.json';members=[]
        for p in (draft,manifest):
            item=filemap.get(str(p));v.need(item is not None and type(item['bytes']) is int and item['bytes']>0 and re.fullmatch('[a-f0-9]{64}',item['sha256']),'Frozen published standalone JSON required')
            members.append(dict(path=str(p),sha256=item['sha256']))
        result=dict(id=id_,split=split,video_path=row['video_path'],rule_draft_ref=members[0],fullframe_manifest_ref=members[1])
        v.input_pair(result)
        for ref in members:v.need(Path(ref['path']).stat().st_size==filemap[ref['path']]['bytes'],'Published file byte length differs')
        rows.append(result)
    return rows


def run(count,out):
    v.need(type(count) is int and 1<=count<=10000,'Finite caption-only source input selection required')
    out=Path(out);v.need(out.parent==ROOT and out.name.startswith('ms_vlm_external_direct_inputs_') and not out.exists(),'Fresh shared input namespace required')
    registry=v.load(ROOT/'runtime/ms_vlm_postprocess_jobs_registry_v1.json');excluded=set()
    for job in registry['jobs']:
        if 'selected_source_ids' in job:excluded.update(i.casefold() for i in job['selected_source_ids'])
        else:
            p=Path(job['config_path'])
            if p.exists():excluded.update(i.casefold() for i in v.load(p)['selected_source_ids'])
            else:
                v.need(job.get('expected_running') is False and 'plan_ref' in job,'Missing unfrozen prior job config')
                r=job['plan_ref'];v.need(v.base.sha(r['path'])==r['sha256'],'Frozen pending plan differs');excluded.update(v.h.bench.strict(line)['id'].casefold() for line in Path(r['path']).read_bytes().splitlines())
    batches=[];rows=[];unique=set()
    for worker in ('a100-extra-01','a100-extra-02'):
        ticket_path=BASE/'assignments'/(worker+'.ticket.json');ticket=v.load(ticket_path)
        v.need(ticket['source_count']==40000 and ticket['worker_id']==worker,'Original external assignment scope differs')
        assignment_path=Path(ticket['assignment_path']);assignment_raw=assignment_path.read_bytes()
        import hashlib
        v.need(hashlib.sha256(assignment_raw).hexdigest()==ticket['assignment_sha256'],'Original assignment SHA differs')
        assignment=[v.h.bench.strict(line) for line in assignment_raw.splitlines() if line.strip()]
        v.need(len(assignment)==40000 and len({r['id'].casefold() for r in assignment})==40000,'Full exact unique assignment required')
        root=BASE/'results'/worker/'batches'
        for directory in sorted(root.iterdir(),key=lambda p:p.name):
            cp=directory/'commit.json'
            if not cp.is_file():continue
            commit=v.load(cp)
            if commit.get('schema')!='ms_unified_novlm_batch_commit_v1' or commit.get('edition')!='external_ms_unified_clusters80k_20261027_v1':continue
            v.need(commit['worker']==worker and commit['assignment_sha256']==ticket['assignment_sha256'],'Published worker/assignment binding differs')
            values=published_rows(cp,commit,excluded|unique,assignment)
            values=values[:count-len(rows)]
            if values:
                unique.update(r['id'].casefold() for r in values);rows.extend(values);batches.append(dict(commit_ref=v.ref(cp),source_rows=[dict(id=r['id'],split=r['split']) for r in values],assignment_sha256=commit['assignment_sha256'],worker=worker))
            if len(rows)==count:break
        if len(rows)==count:break
    v.need(len(rows)==count and len(unique)==count,'Existing published metadata insufficient for requested exact count')
    out.mkdir();p=out/'manifest.jsonl';raw=b''.join(v.h.bench.encoded(r) for r in rows)
    import os
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    pr=v.ref(p);proof=v.write_once(out/'input_proof.json',dict(schema='ms_vlm_external_standalone_caption_inputs_v1',observed_unix=time.time(),source_count=count,plan_ref=pr,selected_source_ids=[r['id'] for r in rows],published_batch_refs=batches,input_pair_verified_count=count,source_claims_created=0,models_called=0,archive_payload_reads=0,video_payload_reads=0,original_files_modified=0,global_registered_overlap=0,observed_input_selection_not_reservation=True,quality_approved=False,training_eligible=False));print(v.h.bench.encoded(dict(event='external_direct_caption_inputs_ready',source_count=count,plan_ref=pr,proof_ref=proof)).decode(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,required=True);p.add_argument('--out',required=True);x=p.parse_args();run(x.count,x.out)
