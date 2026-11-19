"""Prepare mixed external direct JSON/archive-only inputs once, no video or model work."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import time
sys.dont_write_bytecode=True
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime');sys.path.append(str(RT))
import ms_vlm_archive_inputs_v1 as bridge
from stage2_local_publication_v1 import write_once,require,rename_noreplace

ROOT=RT.parent;OUT=ROOT/'stage2_external03to06_remaining_inputs_20261028_v2';LOCAL=Path('/root/stage2_external_remaining_packs_20261028_v1')


def extract(group):
    ref=group[0]['original_external_commit_ref'];raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original external metadata commit changed');commit=json.loads(raw);by_id={r['id']:r for r in group};require(commit['schema']=='ms_unified_novlm_batch_commit_v1' and len(by_id)==len(group)<=20,'Bounded original external commit required')
    ar=group[0]['original_archive_ref'];require(all(r['original_archive_ref']==ar for r in group),'Original metadata archive binding differs');path=Path(ar['path']);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size==ar['bytes'] and before.st_size<=bridge.MAX_ARCHIVE_BYTES,'Original archive type/size exceeds bound');data=f.read(bridge.MAX_ARCHIVE_BYTES+1);after=os.fstat(f.fileno())
    require(bridge.signature(before)==bridge.signature(after)==bridge.signature(path.stat()) and len(data)==ar['bytes'] and hashlib.sha256(data).hexdigest()==ar['sha256']==commit['batch_sha256'],'Original archive stable SHA differs')
    extracted={};names=set();original_names={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:',tarinfo=bridge.RegularTarInfo) as tar:
        for member in tar:
            bridge.safe_member(member);require(member.name not in names,'Duplicate original archive member');names.add(member.name);parts=member.name.split('/')
            if len(parts)<4 or parts[-4]!='cases' or parts[-2] not in by_id or parts[-1] not in ('draft.json','fullframe_manifest.json'):continue
            row=by_id[parts[-2]];require(parts[-3]==row['split'] and 0<member.size<=2**20,'Original JSON/source split differs');k=(row['id'],parts[-1]);require(k not in extracted,'Duplicate original source JSON');b=tar.extractfile(member).read(2**20+1);require(len(b)==member.size,'Original JSON byte size differs');json.loads(b);extracted[k]=b;original_names[k]=member.name
    require(len(extracted)==2*len(group),'Selected original metadata members missing')
    name=hashlib.sha256(ref['path'].encode()).hexdigest();directory=LOCAL/name;directory.mkdir(exist_ok=False);members=[];rows=[]
    for id_,original in by_id.items():
        d=extracted[(id_,'draft.json')];m=extracted[(id_,'fullframe_manifest.json')];draft=json.loads(d);manifest=json.loads(m);split=original['split'];require(draft['id']==manifest['id']==id_ and draft['split']==manifest['split']==split and draft['source']['video_path']==manifest['video_path']==original['video_path'] and draft['source']['video_sha256']==manifest['video_sha256'],'Original source metadata identity differs')
        receipt=next(r for r in commit['receipts'] if r['source_row']['id']==id_);require(receipt['status']=='extracted_rule_draft_unapproved' and receipt['source_row']['split']==split and receipt['vlm_called'] is False,'Only original successful noVLM source allowed')
        fullframe_origin=draft['source']['fullframe_manifest_path'];row=dict(id=id_,split=split,video_path=original['video_path'],rule_draft_ref=dict(path=str(Path(ref['path']).parent/original_names[(id_,'draft.json')]),sha256=hashlib.sha256(d).hexdigest()),fullframe_manifest_ref=dict(path=fullframe_origin,sha256=hashlib.sha256(m).hexdigest()),fullframe_origin_path=fullframe_origin,original_external_commit_ref=ref,original_archive_ref=ar,metadata_archive_members=dict(rule_draft_ref=split+'/'+id_+'/draft.json',fullframe_manifest_ref=split+'/'+id_+'/fullframe_manifest.json'),rich_text='');rows.append(row);members.extend([(row['metadata_archive_members']['rule_draft_ref'],d),(row['metadata_archive_members']['fullframe_manifest_ref'],m)])
    archive=directory/'inputs.tar';fd=os.open(archive,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
        with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
            for n,b in members:info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
        f.flush();os.fsync(f.fileno())
    b=archive.read_bytes();require(len(b)<=64*2**20,'Bounded JSON-only input pack required');shared=OUT/'input_packs'/name;shared.mkdir(parents=True,exist_ok=False);p=shared/('input.partial-'+str(os.getpid()));fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    target=shared/'inputs.tar';rename_noreplace(p,target);fd=os.open(shared,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);input_ref=dict(path=str(target),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    for row in rows:row['metadata_archive_ref']=input_ref
    receipt=write_once(shared/'commit.json',dict(schema='stage2_external_original_input_pack_v1',original_commit_ref=ref,original_archive_ref=ar,original_archive_sha_verified_once=True,source_count=len(rows),source_keys=[[r['split'],r['id']] for r in rows],rows=rows,archive_ref=input_ref,no_npz_decoded=True,no_video_read=True,original_raw_JSON_preserved=True,quality_approved=False,training_eligible=False));write_once(directory/'shared_ack.json',dict(receipt=receipt,model_calls=0));return rows


def run():
    require(hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()=='21e908102b20ba91533953c1e6d46af36f32e1622e8b54cf808870bbff1dbf75','Frozen original safe tar reader differs');LOCAL.mkdir(exist_ok=False);deadline=time.monotonic()+7200
    while not (OUT/'allocation.json').exists():require(time.monotonic()<deadline,'Remaining-source metadata still unavailable');time.sleep(5)
    a=json.loads((OUT/'allocation.json').read_bytes());ref=a['source_index_ref'];raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Reserved source index changed');index=[json.loads(x) for x in raw.splitlines() if x];require(len(index)==a['source_count'],'Reserved source group differs');rows=[x for x in index if not x.get('metadata_extract_required')];groups=defaultdict(list)
    for x in index:
        if x.get('metadata_extract_required'):groups[x['original_external_commit_ref']['path']].append(x)
    failures=[];started=time.time()
    def one(group):
        try:return extract(group),None
        except Exception as e:return [],dict(source_keys=[[r['split'],r['id']] for r in group],original_commit_ref=group[0]['original_external_commit_ref'],error=repr(e),model_calls=0,automatic_retry=False)
    batches=list(groups.values())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0,len(batches),16):
            for values,failure in pool.map(one,batches[offset:offset+16]):
                rows.extend(values)
                if failure:failures.append(failure)
            print(json.dumps(dict(event='external_original_inputs_ready',ready_sources=len(rows),failed_batches=len(failures),seconds=time.time()-started,model_calls=0)),flush=True)
    require(len({r['id'].casefold() for r in rows})==len(rows),'Duplicate ready source input');p=OUT/'manifest.jsonl';raw=b''.join((json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+chr(10)).encode() for r in rows);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);write_once(OUT/'input_ready.json',dict(schema='stage2_external_remaining_input_group_ready_v1',source_count=len(rows),original_reserved_count=len(index),manifest_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),failed_batches=failures,model_calls=0,original_input_raw_bytes_preserved=True,quality_approved=False,training_eligible=False,completed_unix=time.time()));print(json.dumps(dict(event='external_remaining_inputs_ready',ready_sources=len(rows),failed_batches=len(failures),model_calls=0)),flush=True)

if __name__=='__main__':run()
