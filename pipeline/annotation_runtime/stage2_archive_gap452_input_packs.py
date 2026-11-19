"""Once-per-native-batch JSON extraction and bounded shared metadata packs, no GPU work."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
sys.dont_write_bytecode=True
sys.path.append('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
import ms_vlm_archive_inputs_v1 as bridge
from stage2_local_publication_v1 import write_once,rename_noreplace,require

ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
LOCAL=Path('/root/stage2_archive_gap452_input_packs_20261028_v1')
OUT=ROOT/'stage2_archive_gap452_inputs_20261028_v1'


def pack(entry,allowed):
    ref=entry['commit_ref'];raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Frozen native commit changed');commit=json.loads(raw)
    ids=[x for x in entry['source_ids'] if x.casefold() in allowed]
    if not ids:return [],None
    require(commit['schema']=='ms_unified_novlm_batch_commit_v3','Original native commit required')
    name=hashlib.sha256(ref['path'].encode()).hexdigest();directory=LOCAL/name;directory.mkdir(exist_ok=False);wanted={commit['archive_index'][id_]['members'][n] for id_ in ids for n in ('draft.json','fullframe_manifest.json')}
    require(1<=len(ids)<=20,'Bounded original source batch required')
    batch=bridge.read_archive(ref,commit,wanted,directory,{})
    rows=[];members=[];seen=set()
    for id_ in ids:
        item=commit['archive_index'][id_];original=item['source_row'];receipt=next(x for x in commit['receipts'] if x['source_row']['id']==id_)
        require(receipt['status']=='extracted_rule_draft_unapproved' and receipt['source_row']==original and original['split'] in ('train','val','test'),'Only original unapproved successful source may enter pack')
        require(all(receipt.get(k) is False for k in ('quality_approved','semantic_approval','training_eligible')),'Unapproved input scope changed')
        data={n:batch['extracted'][item['members'][n]] for n in ('draft.json','fullframe_manifest.json')};draft=json.loads(data['draft.json']);manifest=json.loads(data['fullframe_manifest.json']);split=original['split']
        require(draft['id']==manifest['id']==id_ and draft['split']==manifest['split']==split,'Original source JSON identity differs')
        origin=str(Path(ref['path']).parent/item['members']['fullframe_manifest.json']);require(draft['source']['fullframe_manifest_path']==origin and draft['source']['video_path']==manifest['video_path']==original['video_path'] and draft['source']['video_sha256']==manifest['video_sha256'],'Original logical binding differs')
        names={n:split+'/'+id_+'/'+n for n in data}
        for n,b in data.items():
            require(names[n] not in seen and len(b)<=2**20,'Safe bounded raw JSON member required');seen.add(names[n]);members.append((names[n],b))
        rows.append(dict(id=id_,split=split,video_path=original['video_path'],rule_draft_ref=dict(path=str(Path(ref['path']).parent/item['members']['draft.json']),sha256=hashlib.sha256(data['draft.json']).hexdigest()),fullframe_manifest_ref=dict(path=origin,sha256=hashlib.sha256(data['fullframe_manifest.json']).hexdigest()),fullframe_origin_path=origin,original_native_commit_ref=ref,original_native_verification_ref=batch['verification_ref'],metadata_archive_members=dict(rule_draft_ref=names['draft.json'],fullframe_manifest_ref=names['fullframe_manifest.json']),rich_text=''))
    archive=directory/'inputs.tar';fd=os.open(archive,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
            for member,b in members:m=tarfile.TarInfo(member);m.size=len(b);tar.addfile(m,io.BytesIO(b))
        f.flush();os.fsync(f.fileno())
    b=archive.read_bytes();require(len(b)<=64*2**20,'Bounded metadata pack exceeded');sha=hashlib.sha256(b).hexdigest();shared=OUT/'input_packs'/name;shared.mkdir(parents=True,exist_ok=False);temporary=shared/('inputs.partial-'+str(os.getpid()));fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    target=shared/'inputs.tar';rename_noreplace(temporary,target);df=os.open(shared,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df);ar=dict(path=str(target),sha256=sha,bytes=len(b))
    for row in rows:row['metadata_archive_ref']=ar
    receipt=write_once(shared/'commit.json',dict(schema='stage2_native_input_pack_v1',original_commit_ref=ref,original_verification=batch['verification'],source_count=len(rows),source_keys=[[r['split'],r['id']] for r in rows],rows=rows,archive_ref=ar,no_npz_decoded=True,no_video_payload_read=True,original_raw_json_preserved=True,quality_approved=False,training_eligible=False,created_unix=time.time()))
    write_once(directory/'shared_input_pack_ack.json',dict(shared_commit=receipt,archive_sha256=sha,source_count=len(rows),original_archive_hash_reads=1,models_called=0))
    return rows,receipt


def run():
    require(hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()=='21e908102b20ba91533953c1e6d46af36f32e1622e8b54cf808870bbff1dbf75','Frozen original archive bridge changed')
    LOCAL.mkdir(exist_ok=False);require(shutil.disk_usage(LOCAL).free>=8*2**30,'Local metadata preparation disk headroom required')
    allocation=json.loads((OUT/'allocation.json').read_bytes());ref=allocation['source_index_ref'];raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Reserved native source index changed');index=[json.loads(x) for x in raw.splitlines()];allowed={x['id'].casefold() for x in index};require(len(allowed)==len(index)==allocation['source_count'],'Original unique native source group required')
    rows=[];receipts=[];failures=[];started=time.time()
    def one(entry):
        try:return pack(entry,allowed),None
        except Exception as e:return None,dict(commit_ref=entry['commit_ref'],source_ids=entry['source_ids'],error=repr(e),model_calls=0,source_claims_created=0,automatic_retry=False)
    batches=allocation['selected_batches']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0,len(batches),16):
            for result,error in pool.map(one,batches[offset:offset+16]):
                if error:failures.append(error);continue
                values,receipt=result;rows.extend(values)
                if receipt:receipts.append(receipt)
            print(json.dumps(dict(event='native_input_packs_published',ready_sources=len(rows),failed_batches=len(failures),elapsed_seconds=time.time()-started,models_called=0)),flush=True)
    require(len({r['id'].casefold() for r in rows})==len(rows) and {r['id'].casefold() for r in rows}<=allowed,'Qualified native source list differs')
    raw=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'))+chr(10)).encode() for r in rows);p=OUT/'manifest.jsonl';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    df=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
    write_once(OUT/'input_ready.json',dict(schema='stage2_native_input_group_ready_v1',source_count=len(rows),original_reserved_count=len(index),manifest_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),input_pack_receipts=receipts,failed_batches=failures,no_models_called=True,no_source_claims_created=True,quality_approved=False,training_eligible=False,completed_unix=time.time()))
    print(json.dumps(dict(event='native_input_group_ready',ready_sources=len(rows),failed_batches=len(failures),models_called=0)),flush=True)

if __name__=='__main__':run()
