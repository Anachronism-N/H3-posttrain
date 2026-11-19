"""CPU-only original-video preparation for all corrected immutable windows, one copy/source."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.request
from stage2_local_publication_v1 import write_once,checked,require
import motion_revision_codec_v1 as codec


def prepare(context,cfg):
    row=context['row'];root=Path(cfg['out']);local=root/'prepared'/row['id'];local.mkdir(parents=True,exist_ok=False);video=local/(row['id']+Path(row['video_path']).suffix);source=Path(row['video_path']);fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW);digest=hashlib.sha256();total=0
    with os.fdopen(fd,'rb') as incoming:
        before=os.fstat(incoming.fileno());require(0<before.st_size<=2**30,'Bounded original video required');outfd=os.open(video,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(outfd,'wb') as f:
            for b in iter(lambda:incoming.read(8*2**20),b''):total+=len(b);require(total<=before.st_size,'Source video grew');digest.update(b);f.write(b)
            f.flush();os.fsync(f.fileno())
        after=os.fstat(incoming.fileno());require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns) and total==before.st_size and digest.hexdigest()==context['manifest']['video_sha256'],'Original video changed or SHA differs')
    indices=sorted({i for w in context['window_frames'] for i in w['frame_indices']});require(1<=len(indices)<=64,'Every window has its own bounded native frames');expression='+'.join('eq(n\,%d)'%i for i in indices);m=context['manifest'];scale=min(1.,math.sqrt(200704/(m['width']*m['height'])));width=max(1,int(m['width']*scale));height=max(1,int(m['height']*scale));env=os.environ.copy()
    if env.get('LD_LIBRARY_PATH'):env['LD_LIBRARY_PATH']=':'.join(x for x in env['LD_LIBRARY_PATH'].split(':') if x)
    subprocess.run([cfg['ffmpeg_path'],'-hide_banner','-loglevel','error','-nostdin','-threads','1','-noautorotate','-i',str(video),'-an','-sn','-filter_threads','1','-vf','select='+expression+',scale=%d:%d'%(width,height),'-vsync','0','-threads','1','-q:v','3',str(local/'frame-%03d.jpg')],cwd='/tmp',env=env,check=True,capture_output=True);images=sorted(local.glob('frame-*.jpg'));require(len(images)==len(indices),'Native sampled images missing')
    image_refs={}
    for i,p in zip(indices,images):raw=p.read_bytes();require(raw.startswith(bytes([255,216])) and raw.endswith(bytes([255,217])),'Incomplete JPEG');image_refs[str(i)]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())
    value=dict(context,local_video_ref=dict(path=str(video),sha256=digest.hexdigest()),image_refs=image_refs);ref=write_once(local/'revision_context.json',value);return ref


def run(path):
    cfg=json.loads(Path(path).read_bytes());out=Path(cfg['out']);owner=cfg['owner'];queue=out/'ready'/owner;queue.mkdir(parents=True,exist_ok=True);claimlog=out/'feeder_claims'/owner;claimlog.mkdir(parents=True,exist_ok=True);token=Path(cfg['token_path']).read_text().strip()
    while not (out/'FEEDER_STOP_BATCH16').exists():
        waiting=[p for p in queue.glob('*.ready.json') if not (out/'consumed_ready'/owner/p.name).exists()]
        if len(waiting)>=8:time.sleep(.5);continue
        rid=owner+'-'+secrets.token_hex(16);data=json.dumps(dict(owner=owner,request_id=rid,size=cfg.get('claim_size',8))).encode();request=urllib.request.Request(cfg['claim_endpoint'],data=data,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=60) as f:raw=f.read(16*2**20+1)
            require(len(raw)<=16*2**20,'Bounded new revision input response required');claim=json.loads(raw)
        except Exception as e:write_once(claimlog/(rid+'_uncertain.json'),dict(original_request_id=rid,error=repr(e),original_claim_may_exist=True,automatic_retry=False,models_called=0));write_once(queue/'FEEDER_FAILED_BATCH16',dict(error=repr(e),claims_preserved=True));raise
        if claim is None:break
        if claim.get('waiting_for_original_inputs'):time.sleep(2);continue
        require(claim['owner']==owner and claim['automatic_retry'] is False and len(claim['prepared_metadata'])==claim['source_count'],'New revision original ownership required');metadata=claim.pop('prepared_metadata');write_once(claimlog/(claim['batch_id']+'.json'),claim);begin=time.time()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:refs=list(pool.map(lambda v:prepare(v,cfg),metadata))
            packet=dict(schema='motion_revision_prepared_packet_v1',revision=codec.REVISION,owner=owner,claim=claim,context_refs=refs,source_count=len(refs));p=out/'packets'/owner/(claim['batch_id']+'.json');ref=write_once(p,packet);write_once(queue/(p.name+'.ready.json'),dict(packet_ref=ref,source_count=len(refs),owner=owner,seconds=time.time()-begin));print(json.dumps(dict(event='correct_revision_window_packet_ready',owner=owner,sources=len(refs),seconds=time.time()-begin)),flush=True)
            if cfg.get('one_packet_only'):break
        except Exception as e:write_once(claimlog/(claim['batch_id']+'_prepare_failure.json'),dict(original_claim=claim,error=repr(e),models_called=0,automatic_retry=False));write_once(queue/'FEEDER_FAILED',dict(error=repr(e),claims_preserved=True));raise
    write_once(queue/'FEEDER_DRAINED_BATCH16',dict(owner=owner,models_called=0,ended_unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
