"""CPU feeder alone accesses shared inputs; model workers read only local durable packets."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import secrets
from stage2_local_publication_v1 import write_once,checked,require
import stage2_feeder_media_v1 as media


from concurrent.futures import Future
import io
import tarfile
import threading
import re
from collections import OrderedDict

_METADATA = OrderedDict()
_METADATA_LOCK = threading.Lock()


def archive_metadata(row, root):
    ref=row['metadata_archive_ref'];require(re.fullmatch('[a-f0-9]{64}',ref['sha256']) is not None,'Canonical input pack SHA required');key=(ref['path'],ref['sha256'])
    with _METADATA_LOCK:
        future=_METADATA.get(key);owner=future is None
        if owner:
            future=Future();_METADATA[key]=future
            for old in list(_METADATA):
                if len(_METADATA)<=16:break
                if old!=key and _METADATA[old].done():_METADATA.pop(old)
        else:_METADATA.move_to_end(key)
    if owner:
        try:
            cache=Path(root)/'metadata_archives';cache.mkdir(exist_ok=True);local=cache/(ref['sha256']+'.tar')
            require(type(ref['bytes']) is int and 0<ref['bytes']<=64*2**20,'Bounded original JSON-only input pack required')
            fd=os.open(cache/(ref['sha256']+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            try:
                st=os.fstat(fd);require(__import__('stat').S_ISREG(st.st_mode) and st.st_uid==os.getuid() and st.st_size==0,'Owned local pack copy lock required')
                fcntl.flock(fd,fcntl.LOCK_EX)
                if local.exists():
                    require(local.is_file() and not local.is_symlink() and local.stat().st_uid==os.getuid(),'Owned local input pack cache required');raw=local.read_bytes()
                else:
                    incoming=os.open(ref['path'],os.O_RDONLY|os.O_NOFOLLOW)
                    with os.fdopen(incoming,'rb') as f:raw=f.read(64*2**20+1)
                    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original input pack SHA/bytes differs')
                    fdout=os.open(local,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                    with os.fdopen(fdout,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                    df=os.open(cache,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
                require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Cached input pack differs')
            finally:os.close(fd)
            values={}
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
                for member in tar:
                    require(member.isfile() and not member.issparse() and member.sparse is None and not any(k.startswith(('GNU.sparse','SCHILY.realsize')) for k in member.pax_headers),'Only regular nonsparse JSON input members required')
                    parts=member.name.split('/');require(len(parts)==3 and parts[0] in ('train','val','test') and re.fullmatch('[A-Za-z0-9_-]+',parts[1]) and parts[2] in ('draft.json','fullframe_manifest.json'),'Safe original source JSON input member required')
                    require(member.name not in values and 0<member.size<=2**20 and len(values)<40,'Duplicate/oversized input metadata pack')
                    f=tar.extractfile(member);b=f.read(2**20+1);require(len(b)==member.size,'Original input metadata byte length differs');values[member.name]=b
            future.set_result(values)
        except BaseException as e:
            future.set_exception(e)
            raise
    return future.result()

def run(path):
    cfg=json.loads(Path(path).read_bytes());root=Path(cfg['out']);prepared=root/'prepared';queue=root/'ready'/cfg['owner'];queue.mkdir(parents=True,exist_ok=True);prepared.mkdir(exist_ok=True);token=Path(cfg['token_path']).read_text().strip()
    claimlog=root/'feeder_claims'/cfg['owner'];claimlog.mkdir(parents=True,exist_ok=True);stop=root/'FEEDER_STOP';pool_attempts=0
    def prepare(row):
        d=prepared/row['id'];d.mkdir(exist_ok=False)
        packed=archive_metadata(row,root) if 'metadata_archive_ref' in row else None
        values={}
        for key,name in (('rule_draft_ref','draft.json'),('fullframe_manifest_ref','manifest.json')):
            ref=row[key];raw=packed[row['metadata_archive_members'][key]] if packed is not None else Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original shared metadata SHA changed')
            local=d/name;fd=os.open(local,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            values[key]=dict(path=str(local),sha256=ref['sha256']);require(local.read_bytes()==raw,'Local raw metadata differs')
        manifest=checked(values['fullframe_manifest_ref']);draft=checked(values['rule_draft_ref']);require(manifest['id']==draft['id']==row['id'] and manifest['split']==draft['split']==row['split'],'Original source identity differs')
        frames,images=media.stage_images(row,manifest,d,cfg['ffmpeg_path']);video=d/(row['id']+Path(row['video_path']).suffix);df=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
        return dict(id=row['id'],split=row['split'],draft_ref=values['rule_draft_ref'],manifest_ref=values['fullframe_manifest_ref'],selected_frames=frames,image_refs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images],local_video_ref=dict(path=str(video),sha256=manifest['video_sha256']),single_source_video_copy=True)
    while not stop.exists():
        waiting=[p for p in queue.glob('*.ready.json') if not (root/'consumed_ready'/cfg['owner']/p.name).exists()]
        if len(waiting)>=cfg.get('prefetch_packets',8):time.sleep(.5);continue
        request_id=cfg['owner']+'-'+secrets.token_hex(16);payload=json.dumps(dict(owner=cfg['owner'],request_id=request_id,size=8)).encode();request=urllib.request.Request(cfg['claim_endpoint'],data=payload,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST');pool_attempts+=1
        try:
            with urllib.request.urlopen(request,timeout=30) as response:raw=response.read(2*2**20+1)
            require(len(raw)<=2*2**20,'Dynamic claim response exceeds bound');claim=json.loads(raw)
        except Exception as e:
            write_once(claimlog/(request_id+'_uncertain.json'),dict(request_id=request_id,owner=cfg['owner'],automatic_retry=False,error=type(e).__name__+': '+str(e),models_called=0,claim_may_have_been_committed=True));write_once(queue/'FEEDER_FAILED',dict(error=str(e),claims_not_released=True));raise
        if claim is None:break
        require(claim['owner']==cfg['owner'] and claim['automatic_retry'] is False and 1<=claim['source_count']<=8,'Original dynamic claim owner/size differs');write_once(claimlog/(claim['batch_id']+'.json'),claim)
        started=time.time()
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:sources=list(executor.map(prepare,claim['rows']))
            packet=dict(schema='stage2_prepared_packet_v1',owner=cfg['owner'],claim=claim,sources=sources,prepared_unix=time.time());p=root/'packets'/cfg['owner']/(claim['batch_id']+'.json');ref=write_once(p,packet);write_once(queue/(p.name+'.ready.json'),dict(packet_ref=ref,source_count=claim['source_count'],owner=cfg['owner'],prepared_seconds=time.time()-started))
            print(json.dumps(dict(event='prepared_local_dynamic_packet',owner=cfg['owner'],sources=claim['source_count'],seconds=time.time()-started)),flush=True)
        except Exception as e:
            write_once(claimlog/(claim['batch_id']+'_prepare_failure.json'),dict(original_claim=claim,error=type(e).__name__+': '+str(e),sources_permanently_claimed=True,automatic_retry=False,models_called=0));write_once(queue/'FEEDER_FAILED',dict(error=str(e),claims_not_released=True));raise
    write_once(queue/'FEEDER_DRAINED',dict(owner=cfg['owner'],pool_attempts=pool_attempts,ended_unix=time.time(),models_called=0,source_claims_released=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
