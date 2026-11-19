"""One bounded upload loop per node, independent of GPU inference."""
import argparse
import json
import os
from pathlib import Path
import time
from stage2_local_publication_v1 import LocalOutbox,checked,require,write_once


def run(local_root,shared_root,node,seconds=21600):
    require(type(seconds) is int and 1<=seconds<=86400,'Finite uploader lifetime required')
    box=LocalOutbox(local_root,shared_root,node);lock=box.uploader_lock();metrics=box.local/'upload_metrics';metrics.mkdir(exist_ok=True);deadline=time.monotonic()+seconds
    previous=[p for p in metrics.iterdir() if p.name.endswith('.json')];require(len(previous)<=10000,'Bounded existing upload metrics required');attempt=max((int(p.stem) for p in previous),default=0)
    sealed_keys=set()
    for directory in (box.local/'outbox').iterdir():
        p=directory/'sealed.json'
        if p.is_file():sealed_keys.update(tuple(k) for k in json.loads(p.read_bytes())['source_keys'])
    try:
        while time.monotonic()<deadline and not (box.local/'UPLOAD_STOP').exists():
            complete=[]
            source_root=box.local/'sources'
            if source_root.exists():
                for commit in source_root.glob('*/*/commit.json'):
                    relative=commit.relative_to(source_root);key=(relative.parts[0],relative.parts[1])
                    if key not in sealed_keys:complete.append((commit.stat().st_mtime,key))
            complete.sort()
            if len(complete)>=50 or complete and time.time()-complete[0][0]>=300 or (box.local/'GENERATION_DRAINED').exists() and complete or (box.local/'generation_drained').exists() and complete and not [p for p in (box.local/'ready').glob('*') if not (box.local/'generation_drained'/p.name).exists()]:
                keys=[list(k) for _,k in complete[:50]];box.seal(keys);sealed_keys.update(tuple(k) for k in keys)
            names=list((box.local/'outbox').iterdir());require(len(names)<=10000,'Bounded local outbox scan required')
            pending=[]
            for p in names:
                seal=p/'sealed.json';ack=p/'upload_receipt.json'
                if seal.is_file() and not ack.exists():pending.append(seal)
            for p in sorted(pending,key=lambda x:x.parent.name):
                attempt+=1;started=time.time();began=time.monotonic();seal=dict(path=str(p),sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest());value=checked(seal)
                try:
                    receipt=box.upload(seal);event=dict(schema='stage2_upload_attempt_v1',attempt=attempt,batch_id=value['batch_id'],source_count=value['source_count'],start_unix=started,end_unix=time.time(),wall_seconds=time.monotonic()-began,committed=True,shared_commit=receipt['shared_commit'],models_called=0)
                except Exception as e:
                    event=dict(schema='stage2_upload_attempt_v1',attempt=attempt,batch_id=value['batch_id'],source_count=value['source_count'],start_unix=started,end_unix=time.time(),wall_seconds=time.monotonic()-began,committed=False,error=type(e).__name__+': '+str(e),local_archive_preserved=True,models_called=0)
                write_once(metrics/('%08d.json'%attempt),event)
                print(json.dumps(dict(event='stage2_batch_upload',node=node,batch=value['batch_id'],committed=event['committed'],wall_seconds=event['wall_seconds'])),flush=True)
                if not event['committed']:break
            time.sleep(5)
    finally:os.close(lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local-root',required=True);p.add_argument('--shared-root',required=True);p.add_argument('--node',required=True);p.add_argument('--seconds',type=int,default=21600);a=p.parse_args();run(a.local_root,a.shared_root,a.node,a.seconds)
