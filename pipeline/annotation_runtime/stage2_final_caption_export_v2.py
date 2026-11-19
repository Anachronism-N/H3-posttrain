"""Incremental final-caption export: consume committed results once, never rerun models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time
from stage2_local_publication_v1 import encoded,write_once,rename_noreplace,require

ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
SHARED=ROOT/'stage2_dynamic_local_shared_results_20261028_v1'
OUT=ROOT/'stage2_final_caption_export_20261028_v1'
LOCAL=Path('/root/stage2_final_caption_export_20261028_v1')
NODES=('N1','N2','N3','N6','N8','a100-extra-03','a100-extra-04','a100-extra-05','a100-extra-06')


def row(result,reference,member):
    require(result['quality_approved'] is False and result['training_eligible'] is False and result['semantic_approval'] is False,'Only unapproved source outputs may be exported')
    fields=('id','split','status','fused_caption','visual_audit','corrections','limitations','evidence_policy_violations','error','owner','pool_claim_batch','selected_frames')
    value={k:result[k] for k in fields if k in result}
    require(value['status'] in ('postprocessed_unapproved','postprocessed_needs_visual_revision','insufficient_visible_evidence','failed_sealed_no_retry'),'Unexpected final source status')
    value.update(original_result=dict(shared_commit_ref=reference,archive_member=member),quality_approved=False,semantic_approval=False,training_eligible=False,human_review_completed=False)
    return value


def insert(db,origin,values,ref):
    db.execute('BEGIN IMMEDIATE')
    try:
        for v in values:
            k=v['id'].casefold();raw=encoded(v).decode();existing=db.execute('SELECT origin,split,id,payload FROM results WHERE source_key=?',(k,)).fetchone()
            if existing is not None:require(existing==(origin,v['split'],v['id'],raw),'Conflicting source output in final export');continue
            db.execute('INSERT INTO results(source_key,split,id,status,origin,payload) VALUES(?,?,?,?,?,?)',(k,v['split'],v['id'],v['status'],origin,raw))
        db.execute('INSERT INTO inputs(origin,ref,source_count) VALUES(?,?,?)',(origin,json.dumps(ref,sort_keys=True),len(values)));db.execute('COMMIT')
    except BaseException:db.execute('ROLLBACK');raise


def consume(db,path):
    origin=str(path)
    if db.execute('SELECT 1 FROM inputs WHERE origin=?',(origin,)).fetchone() or db.execute('SELECT 1 FROM errors WHERE origin=?',(origin,)).fetchone():return
    raw=path.read_bytes();c=json.loads(raw);ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest());keys=set(map(tuple,c['source_keys']));require(c['schema']=='stage2_shared_metadata_commit_v1' and c['node']==path.parent.parent.name and c['batch_id']==path.parent.name and 1<=len(keys)==len(c['source_keys'])==c['source_count']<=50,'Bounded canonical source commit required');archive=Path(c['archive']['path']);require(archive==path.parent/'batch.tar' and archive.stat().st_size==c['archive']['bytes'],'Reliable committed archive absent or size differs')
    wanted={'sources/'+split+'/'+id_+'/postprocessed.json':(split,id_) for split,id_ in keys};results={};names=set()
    with tarfile.open(archive,'r:') as tar:
        for member in tar:
            require(member.isfile() and not member.issparse() and member.sparse is None and not any(k.startswith(('GNU.sparse','SCHILY.realsize')) for k in member.pax_headers),'Regular nonsparse evidence archive required')
            parts=member.name.split('/');require(not member.name.startswith('/') and all(x not in ('','.','..') for x in parts) and '\\' not in member.name and member.name not in names,'Safe unique archive member required');names.add(member.name)
            if member.name not in wanted:continue
            require(0<member.size<=16*2**20,'Bounded final result JSON required');b=tar.extractfile(member).read(16*2**20+1);require(len(b)==member.size,'Torn source output JSON');v=json.loads(b);require((v['split'],v['id'])==wanted[member.name],'Source result/commit mapping differs');results[member.name]=row(v,ref,member.name)
    require(set(results)==set(wanted) and dict(Counter(v['status'] for v in results.values()))==c['status_counts'],'Committed source status coverage differs');insert(db,origin,[results[k] for k in sorted(results)],ref)


def old_sources():
    r=json.loads((ROOT/'runtime/ms_vlm_postprocess_jobs_registry_v1.json').read_bytes());sources=[];seen=set()
    for j in r['jobs']:
        if not j['phase'].startswith('ms_fullframe_vlm_postprocess_'):continue
        cp=Path(j['config_path'])
        if not cp.exists():continue
        c=json.loads(cp.read_bytes());summary=Path(c['out'])/'summary.json'
        if not summary.exists():continue
        value=json.loads(summary.read_bytes())
        for x in value.get('all_outcomes',value.get('outcomes',[])):
            ref=x.get('commit_ref')
            if ref and ref['path'] not in seen:sources.append((x,ref));seen.add(ref['path'])
    return sources


def old_chunk(db,sources):
    pending=[]
    while sources and len(pending)<64:
        x,ref=sources.pop();origin=ref['path']
        if db.execute('SELECT 1 FROM inputs WHERE origin=?',(origin,)).fetchone() or db.execute('SELECT 1 FROM errors WHERE origin=?',(origin,)).fetchone():continue
        pending.append((x,ref))
    def load(item):
        x,ref=item
        try:
            raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original closed-stage source commit differs');commit=json.loads(raw);rr=commit['result_ref'];result_raw=Path(rr['path']).read_bytes();require(hashlib.sha256(result_raw).hexdigest()==rr['sha256'],'Original final result bytes differ');v=json.loads(result_raw);require(v['id']==x['id'] and v['split']==x['split'] and v['status']==x['status'],'Closed-stage source identity differs');out=row(v,ref,None);out['original_result']=dict(source_commit_ref=ref,result_ref=rr);return out,None
        except Exception as e:return None,repr(e)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values=list(pool.map(load,pending))
    for (_,ref),(v,error) in zip(pending,values):
        if error:db.execute('INSERT OR IGNORE INTO errors VALUES(?,?)',(ref['path'],error))
        else:insert(db,ref['path'],[v],ref)


def publish(db,force=False):
    values=db.execute('SELECT source_key,split,id,payload FROM results WHERE shard IS NULL ORDER BY rowid LIMIT 1000').fetchall()
    if not values or len(values)<1000 and not force:return False
    count=db.execute('SELECT count(*) FROM shards').fetchone()[0];name='part-%06d'%count;raw=''.join(v[3] for v in values).encode();digest=hashlib.sha256(raw).hexdigest();local=LOCAL/name;local.mkdir(exist_ok=True);p=local/'captions.jsonl';commit=local/'commit.json';target=OUT/'shards'/name;target.mkdir(parents=True,exist_ok=True)
    if p.exists():require(p.read_bytes()==raw,'Existing local export shard differs')
    else:
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    final=target/'captions.jsonl'
    if not final.exists():
        temporary=target/('captions.partial-'+str(os.getpid()));fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        rename_noreplace(temporary,final);df=os.open(target,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
    else:require(final.stat().st_size==len(raw) and hashlib.sha256(final.read_bytes()).hexdigest()==digest,'Existing export shard differs')
    v=dict(schema='stage2_final_caption_export_shard_v1',shard=name,source_count=len(values),source_keys=[[x[1],x[2]] for x in values],jsonl_ref=dict(path=str(final),sha256=digest,bytes=len(raw)),quality_approved=False,training_eligible=False,all_results_from_reliable_commits=True)
    pc=target/'commit.json'
    if pc.exists():require(json.loads(pc.read_bytes())==v,'Existing final export commit differs')
    else:write_once(pc,v)
    db.execute('BEGIN IMMEDIATE')
    try:
        db.execute('INSERT INTO shards VALUES(?,?,?)',(name,len(values),json.dumps(v,sort_keys=True)))
        for key,split,id_,payload in values:require(db.execute('UPDATE results SET shard=? WHERE source_key=? AND shard IS NULL',(name,key)).rowcount==1,'Export assignment changed')
        db.execute('COMMIT')
    except BaseException:db.execute('ROLLBACK');raise
    write_once(local/'shared_ack.json',dict(shard=name,shared_commit_ref=dict(path=str(pc),sha256=hashlib.sha256(pc.read_bytes()).hexdigest()),models_called=0));print(json.dumps(dict(event='final_caption_shard_published',shard=name,sources=len(values),models_called=0)),flush=True);return True


def summary(db):
    value=dict(schema='stage2_final_caption_export_progress_v1',observed_unix=time.time(),source_rows_read_once=db.execute('SELECT count(*) FROM results').fetchone()[0],source_rows_exported=db.execute('SELECT count(*) FROM results WHERE shard IS NOT NULL').fetchone()[0],shards=db.execute('SELECT count(*) FROM shards').fetchone()[0],status_counts=dict(db.execute('SELECT status,count(*) FROM results GROUP BY status')),blocked_inputs=db.execute('SELECT count(*) FROM errors').fetchone()[0],full_corpus_complete=False,quality_approved=False,training_eligible=False,models_called=0,original_videos_read=0,whole_original_archive_hash_repeats=0)
    p=OUT/'latest_summary.json';tmp=p.with_name('latest_summary.json.tmp');raw=encoded(value);fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p);return value


def run():
    LOCAL.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);fd=os.open(LOCAL/'export.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);db=sqlite3.connect(LOCAL/'export.db',timeout=30,isolation_level=None);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.executescript('CREATE TABLE IF NOT EXISTS inputs(origin TEXT PRIMARY KEY,ref TEXT,source_count INTEGER);CREATE TABLE IF NOT EXISTS results(source_key TEXT PRIMARY KEY,split TEXT,id TEXT,status TEXT,origin TEXT,payload TEXT,shard TEXT);CREATE TABLE IF NOT EXISTS shards(name TEXT PRIMARY KEY,source_count INTEGER,receipt TEXT);CREATE TABLE IF NOT EXISTS errors(origin TEXT PRIMARY KEY,error TEXT);');sources=None;last_flush=time.monotonic()
    while not (LOCAL/'STOP_V2').exists():
        for node in NODES:
            directory=SHARED/node
            if not directory.exists():continue
            for batch in directory.iterdir():
                p=batch/'commit.json'
                if not p.is_file():continue
                try:consume(db,p)
                except Exception as e:db.execute('INSERT OR IGNORE INTO errors VALUES(?,?)',(str(p),repr(e)));print(json.dumps(dict(event='final_export_input_blocked',input=str(p),error=repr(e),models_called=0)),flush=True)
                while publish(db):last_flush=time.monotonic()
        if sources is None:sources=old_sources()
        old_chunk(db,sources)
        while publish(db):last_flush=time.monotonic()
        if time.monotonic()-last_flush>=300 and publish(db,True):last_flush=time.monotonic()
        print(json.dumps(summary(db)),flush=True);time.sleep(30)
    while publish(db,True):pass
    summary(db);db.close();os.close(fd)

if __name__=='__main__':run()
