"""Reuse all original MS+BeatThis sources, freeze new correct revision input and central ledger."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time
import motion_revision_codec_v1 as codec
from stage2_local_publication_v1 import write_once,encoded,require

ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
OUT=ROOT/'motion_revision_english_merged_edges_20261028_v2'
LOCAL=Path('/root/stage2_motion_revision_inputs_20261028_v2')
RICH=Path('/apdcephfs_gy4/share_302533218/cedricnie/OmniDance/text')


def gather():
    registry=json.loads((ROOT/'runtime/ms_vlm_postprocess_jobs_registry_v1.json').read_bytes());rows={};refs=[]
    paths=[ROOT/'ms_vlm_external_direct_inputs_20261028_v1/manifest.jsonl',ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/first250_manifest.jsonl']
    for j in registry['jobs']:
        if j.get('plan_ref'):paths.append(Path(j['plan_ref']['path']))
        cp=Path(j['config_path'])
        if cp.exists():
            c=json.loads(cp.read_bytes())
            if c.get('plan'):paths.append(Path(c['plan']['path']))
    # Qualified prepared manifests supersede source-index rows, and never use old fused captions.
    for group in ('stage2_native_remaining_corpus_20261028_v1','stage2_external03to06_remaining_inputs_20261028_v2','stage2_legacy_remaining_inputs_20261028_v1','stage2_archive_gap452_inputs_20261028_v1','external_stage2_vlm_50000_20261028_v1','stage2_main_remaining_direct_20261028_v1'):
        paths.append(ROOT/group/'manifest.jsonl')
    for p in dict.fromkeys(paths):
        if not p.is_file():continue
        raw=p.read_bytes();refs.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()))
        try:values=[json.loads(x) for x in raw.splitlines() if x.strip()]
        except ValueError:continue
        for row in values:
            if not isinstance(row,dict) or not all(k in row for k in ('id','split','video_path','rule_draft_ref','fullframe_manifest_ref')):continue
            key=row['id'].casefold()
            if key in rows:require(rows[key]['id']==row['id'] and rows[key]['split']==row['split'] and rows[key]['video_path']==row['video_path'],'Conflicting original source identity')
            if key not in rows or 'metadata_archive_ref' in row or 'original_native_commit_ref' in row:rows[key]=row
    require(len(rows)>170000,'Expected complete17w+ original draft source set; no partial pool disguised as all data')
    return list(rows.values()),refs


def original_commit(row):
    if row.get('original_native_commit_ref'):return row['original_native_commit_ref']
    if row.get('original_external_commit_ref'):return row['original_external_commit_ref']
    path=Path(row.get('fullframe_origin_path',row['fullframe_manifest_ref']['path']))
    if str(path).startswith(str(ROOT)) and len(path.parents)>=4:return dict(path=str(path.parents[3]/'commit.json'),sha256=None)
    return None


def packed(group):
    # Original batch tar read once, without pose decoding or video reads.
    ref=group[0]['_commit'];raw=Path(ref['path']).read_bytes();require(ref['sha256'] is None or hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original native commit changed');c=json.loads(raw);archive=c.get('archive_path')
    if not archive:
        package=c.get('package')
        if not package:package=next(x for x in c['publications'] if x['path'].endswith('/batch.tar'))
        archive=package['path']
    needed={}
    for row in group:
        if c.get('archive_index'):names=c['archive_index'][row['id']]['members']
        else:names={n:'cases/'+row['split']+'/'+row['id']+'/'+n for n in ('draft.json','fullframe_manifest.json','beats.json')}
        for n in ('draft.json','fullframe_manifest.json','beats.json'):needed[names[n]]=(row['id'],n)
    values={}
    fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        data=f.read(128*2**20+1);require(len(data)<=128*2**20,'Bounded original metadata archive required')
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
        names=set()
        for member in tar:
            require(member.isfile() and not member.issparse() and member.sparse is None and not any(k.startswith(('GNU.sparse','SCHILY.realsize')) for k in member.pax_headers),'Original regular metadata archive required');parts=member.name.split('/');require(not member.name.startswith('/') and all(x not in ('','.','..') for x in parts) and member.name not in names,'Safe unique metadata members');names.add(member.name)
            lookup=member.name
            if lookup not in needed:
                lookup=next((name for name in needed if member.name.endswith('/'+name)),None)
            if lookup in needed:
                require(0<member.size<=2**20,'Bounded original JSON member');values[needed[lookup]]=tar.extractfile(member).read(2**20+1)
    require(len(values)==3*len(group),'Original beats/draft/manifest missing')
    return values


def prepare(row,values=None):
    if values:
        draft_raw=values[(row['id'],'draft.json')];manifest_raw=values[(row['id'],'fullframe_manifest.json')];beats_raw=values[(row['id'],'beats.json')];beat_ref=dict(original_commit_ref=row['_commit'],member=row['split']+'/'+row['id']+'/beats.json',sha256=hashlib.sha256(beats_raw).hexdigest())
    else:
        draft_raw=Path(row['rule_draft_ref']['path']).read_bytes();manifest_raw=Path(row['fullframe_manifest_ref']['path']).read_bytes();beatpath=Path(row.get('fullframe_origin_path',row['fullframe_manifest_ref']['path'])).parent/'beats.json';beats_raw=beatpath.read_bytes();beat_ref=dict(path=str(beatpath),sha256=hashlib.sha256(beats_raw).hexdigest())
    require(hashlib.sha256(draft_raw).hexdigest()==row['rule_draft_ref']['sha256'] and hashlib.sha256(manifest_raw).hexdigest()==row['fullframe_manifest_ref']['sha256'],'Frozen original raw JSON differs');draft=json.loads(draft_raw);manifest=json.loads(manifest_raw);beats=json.loads(beats_raw)
    require(draft['id']==manifest['id']==beats['id']==row['id'] and draft['split']==manifest['split']==row['split'],'Original source identity differs');require(draft['source']['video_sha256']==manifest['video_sha256'] and draft['source']['video_path']==manifest['video_path']==row['video_path'],'Original video binding differs')
    richpath=RICH/(row['id']+'.txt');rich=richpath.read_text() if richpath.exists() else '';style=codec.style_hint(rich);plan=codec.merged_plan(manifest,beats,beat_ref);windows=codec.select_window_frames(manifest,plan)
    return dict(schema='motion_revision_prepared_metadata_v1',revision=codec.REVISION,row={k:v for k,v in row.items() if k!='_commit' and k!='rich_text'},draft=draft,manifest=manifest,plan=plan,style_hint=style,window_frames=windows,rich_body_forwarded_to_VLM=False,rich_style_source_path=str(richpath) if richpath.exists() else None,original_draft_sha256=hashlib.sha256(draft_raw).hexdigest(),original_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),window_plan_sha256=hashlib.sha256(codec.encoded(plan)).hexdigest(),quality_approved=False,training_eligible=False)


def run():
    LOCAL.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False);rows,refs=gather();write_once(OUT/'revision_contract.json',dict(revision=codec.REVISION,source_count=len(rows),source_manifest_refs=refs,window_policy='merge_short_edges',min_edge_seconds=1.5,rich_role='dance_style_only',language='English',cross_window_continuity=True,global_amplitude_required=False,clothes_background_forbidden=True,old_caption_outputs_not_eligible=True,old_artifacts_preserved=True,created_unix=time.time()))
    db=sqlite3.connect(LOCAL/'catalog.db',timeout=30,isolation_level=None);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);CREATE TABLE sources(ordinal INTEGER PRIMARY KEY,source_key TEXT UNIQUE NOT NULL,id TEXT,split TEXT,state TEXT DEFAULT "unused",owner TEXT,batch_id TEXT,payload TEXT);CREATE TABLE batches(batch_id TEXT PRIMARY KEY,owner TEXT,requested_size INTEGER,source_keys TEXT,assigned_unix REAL);CREATE TABLE errors(source_key TEXT PRIMARY KEY,error TEXT);CREATE TABLE metadata_packs(name TEXT PRIMARY KEY,payload TEXT);')
    direct=[];groups={}
    for r in rows:
        ref=original_commit(r)
        if ref:r['_commit']=ref;groups.setdefault(ref['path'],[]).append(r)
        else:direct.append([r])
    batches=list(groups.values())+direct;started=time.time();pack_index=0;buffer=[];failed=[];count=0
    def work(group):
        try:values=packed(group) if group[0].get('_commit') else None
        except Exception as e:return [],[dict(id=r['id'],split=r['split'],error=repr(e)) for r in group]
        good=[];bad=[]
        for row in group:
            try:good.append(prepare(row,values))
            except Exception as e:bad.append(dict(id=row['id'],split=row['split'],error=repr(e)))
        return good,bad
    def flush(items):
        nonlocal pack_index,count
        name='inputs%06d'%pack_index;path=LOCAL/(name+'.tar');fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);index=[]
        with os.fdopen(fd,'wb') as f:
            with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
                for v in items:
                    raw=codec.encoded(v);member=v['row']['split']+'/'+v['row']['id']+'.json';info=tarfile.TarInfo(member);info.size=len(raw);tar.addfile(info,io.BytesIO(raw));index.append((v,member,hashlib.sha256(raw).hexdigest()))
            f.flush();os.fsync(f.fileno())
        archive=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size);db.execute('BEGIN IMMEDIATE')
        try:
            for v,member,digest in index:
                row=dict(id=v['row']['id'],split=v['row']['split'],video_path=v['row']['video_path'],revision_input_archive=archive,revision_input_member=member,revision_input_sha256=digest,revision=codec.REVISION)
                db.execute('INSERT INTO sources(source_key,id,split,payload) VALUES(?,?,?,?)',(row['id'].casefold(),row['id'],row['split'],json.dumps(row,separators=(',',':'))))
            db.execute('INSERT INTO metadata_packs VALUES(?,?)',(name,json.dumps(archive)));db.execute('COMMIT')
        except BaseException:db.execute('ROLLBACK');raise
        count+=len(items);pack_index+=1
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(batches),64):
            for good,bad in pool.map(work,batches[offset:offset+64]):
                buffer+=good;failed+=bad
                for r in bad:db.execute('INSERT INTO errors VALUES(?,?)',(r['id'].casefold(),r['error']))
                while len(buffer)>=50:flush(buffer[:50]);buffer=buffer[50:]
            print(json.dumps(dict(event='correct_revision_original_inputs_ready',ready_sources=count,blocked_sources=len(failed),total_sources=len(rows),seconds=time.time()-started,model_calls=0)),flush=True)
    if buffer:flush(buffer)
    write_once(OUT/'input_catalog_ready.json',dict(revision=codec.REVISION,total_original_sources=len(rows),qualified_sources=count,blocked_sources=failed,local_catalog=str(LOCAL/'catalog.db'),source_keys=[[r['split'],r['id']] for r in rows],model_calls=0,old_inference_outputs_reused=False,completed_unix=time.time()));db.close();print(json.dumps(dict(event='correct_revision_full_input_catalog_ready',total_sources=len(rows),qualified_sources=count,blocked_sources=len(failed),model_calls=0)),flush=True)

if __name__=='__main__':run()
