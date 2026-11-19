"""CPU feeder alone accesses shared inputs; model workers read only local durable packets."""
import argparse
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


def run(path):
    cfg=json.loads(Path(path).read_bytes());root=Path(cfg['out']);prepared=root/'prepared';queue=root/'ready'/cfg['owner'];queue.mkdir(parents=True,exist_ok=True);prepared.mkdir(exist_ok=True);token=Path(cfg['token_path']).read_text().strip()
    claimlog=root/'feeder_claims'/cfg['owner'];claimlog.mkdir(parents=True,exist_ok=True);stop=root/'FEEDER_STOP';pool_attempts=0
    def prepare(row):
        d=prepared/row['id'];d.mkdir(exist_ok=False)
        values={}
        for key,name in (('rule_draft_ref','draft.json'),('fullframe_manifest_ref','manifest.json')):
            ref=row[key];p=Path(ref['path']);raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original shared metadata SHA changed')
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
