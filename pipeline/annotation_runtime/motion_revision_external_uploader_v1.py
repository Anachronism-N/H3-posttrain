"""One external node uploader; sealed archives only, ACK-before-cleanup, no model retries."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
from stage2_local_publication_v1 import LocalOutbox,checked,encoded,require,write_once


def upload(box,ref,cfg):
    value=checked(ref);local=box.local/'outbox'/value['batch_id'];ack=local/'upload_receipt.json'
    if ack.exists():return json.loads(ack.read_bytes())
    value['_upload_seal_ref']=ref;raw=encoded(value);archive=local/'batch.tar';u=urlsplit(cfg['relay_url']);require(u.scheme in ('http','https') and u.hostname and not u.username and not u.password and u.path in ('','/') and not u.query and not u.fragment,'Explicit relay origin required')
    connection=(http.client.HTTPSConnection if u.scheme=='https' else http.client.HTTPConnection)(u.hostname,u.port,timeout=180)
    try:
        connection.putrequest('POST','/upload');connection.putheader('Authorization','Bearer '+Path(cfg['token_path']).read_text().strip());connection.putheader('Content-Type','application/octet-stream');connection.putheader('Content-Length',str(8+len(raw)+value['archive_bytes']));connection.endheaders();connection.send(struct.pack('!Q',len(raw)));connection.send(raw);digest=hashlib.sha256();total=0
        with archive.open('rb') as f:
            for b in iter(lambda:f.read(8*2**20),b''):connection.send(b);digest.update(b);total+=len(b)
        require(total==value['archive_bytes'] and digest.hexdigest()==value['archive_sha256'],'Local sealed archive changed')
        response=connection.getresponse();body=response.read(2**20+1);require(response.status==200 and len(body)<=2**20,'Reliable relay commit not ACKed; preserve archive for upload-only recovery');receipt=json.loads(body)
        require(receipt['batch_id']==value['batch_id'] and receipt['source_count']==value['source_count'] and receipt['archive_sha256']==value['archive_sha256'] and receipt['models_called']==0 and receipt['may_cleanup_local'] is True,'Original sealed ACK differs');write_once(ack,receipt);return receipt
    finally:connection.close()


def run(path):
    cfg=json.loads(Path(path).read_bytes());box=LocalOutbox(cfg['out'],cfg['shared_root'],cfg['node']);lock=box.uploader_lock();metrics=box.local/'upload_metrics';metrics.mkdir(exist_ok=True);attempt=max((int(p.stem) for p in metrics.glob('*.json')),default=0);sealed_keys=set()
    for d in (box.local/'outbox').iterdir():
        if (d/'sealed.json').exists():sealed_keys.update(map(tuple,json.loads((d/'sealed.json').read_bytes())['source_keys']))
    try:
        while not (box.local/'UPLOAD_STOP').exists():
            complete=[]
            for p in (box.local/'sources').glob('*/*/commit.json'):
                parts=p.relative_to(box.local/'sources').parts;k=(parts[0],parts[1])
                if k not in sealed_keys:complete.append((p.stat().st_mtime,k))
            complete.sort();drained=(box.local/'EXPECTED_OWNERS.json').exists() and all((box.local/'generation_drained'/owner).exists() for owner in json.loads((box.local/'EXPECTED_OWNERS.json').read_bytes())['owners'])
            if len(complete)>=50 or complete and (time.time()-complete[0][0]>=300 or drained):
                keys=[list(k) for _,k in complete[:50]];box.seal(keys);sealed_keys.update(map(tuple,keys))
            pending=[p/'sealed.json' for p in (box.local/'outbox').iterdir() if (p/'sealed.json').exists() and not (p/'upload_receipt.json').exists()]
            for p in sorted(pending):
                attempt+=1;ref=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest());started=time.time()
                try:
                    ack=upload(box,ref,cfg);event=dict(attempt=attempt,batch_id=ack['batch_id'],source_count=ack['source_count'],start_unix=started,end_unix=time.time(),wall_seconds=time.time()-started,committed=True,shared_commit=ack['shared_commit'],models_called=0)
                except Exception as e:event=dict(attempt=attempt,batch_id=p.parent.name,start_unix=started,end_unix=time.time(),wall_seconds=time.time()-started,committed=False,error=repr(e),local_archive_preserved=True,models_called=0)
                write_once(metrics/('%08d.json'%attempt),event);print(json.dumps(dict(event='external_upload',node=cfg['node'],batch=event['batch_id'],committed=event['committed'],seconds=event['wall_seconds'])),flush=True)
                if not event['committed']:break
            if drained and not complete and not pending:
                write_once(box.local/'UPLOAD_DRAINED',dict(all_local_completed_sources_shared_acked=True,observed_unix=time.time(),models_called=0));break
            time.sleep(5)
    finally:os.close(lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
