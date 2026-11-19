"""One-shot external live/ACK snapshot only; no model load, claim, or process signal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

NODES=tuple('a100-extra-0'+str(i) for i in range(3,7))


def alive(expected):
    p=Path('/proc',str(expected['pid']))
    try:
        s=(p/'stat').read_text().rsplit(') ',1)[1].split();a=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
        return int(s[19])==expected['start_ticks'] and s[0] not in ('Z','X','x') and a==expected['argv'] and p.stat().st_uid==expected['uid'] and (p/'cgroup').read_text()==expected['cgroup'] and os.readlink(p/'exe')==expected['exe']
    except FileNotFoundError:return False


def collect(node):
    root=Path('/root/stage2_external50000_'+node+'_v1');cfg=json.loads((root/'manager.json').read_bytes());assert cfg['node']==node and cfg['expected_hostname']==subprocess.check_output(['hostname'],text=True).strip();q=Path(cfg['queue']);ad=json.loads((q/'admission.json').read_bytes());assert ad['config_sha256']==hashlib.sha256((root/'manager.json').read_bytes()).hexdigest();live=json.loads((q/'live.json').read_bytes());manager=live['manager_identity'];worker=live['child_identity'];state=root/'worker_state'/cfg['owner'];p=state/'summary.json';ps=sorted(state.glob('progress*.json'));p=p if p.exists() else ps[-1] if ps else None;progress=json.loads(p.read_bytes()) if p else {};load=json.loads((state/'load.receipt.json').read_bytes()) if (state/'load.receipt.json').exists() else None
    acks=[]
    for directory in (Path(cfg['out'])/'outbox').iterdir():
        p=directory/'upload_receipt.json'
        if p.exists():
            v=json.loads(p.read_bytes());assert v['models_called']==0;acks.append(dict(batch_id=v['batch_id'],source_count=v['source_count'],shared_commit=v['shared_commit'],committed_unix=v['committed_unix']))
    return dict(schema='stage2_external_local_live_snapshot_v1',node=node,expected_hostname=cfg['expected_hostname'],observed_unix=time.time(),manager_identity=manager,worker_identity=worker,manager_alive=alive(manager),worker_alive=alive(worker),model_loaded=load is not None,model_load_receipt=load,local_processed=progress.get('processed',0),acknowledged_shared_sources=sum(x['source_count'] for x in acks),ack_receipts=acks,source_claims_created_by_observer=0,model_calls_by_observer=0,process_signals_by_observer=0,quality_approved=False,training_eligible=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=NODES);p.add_argument('--out',required=True);a=p.parse_args();v=collect(a.node);out=Path(a.out);assert out.is_absolute() and not out.exists();out.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(v,sort_keys=True)+chr(10)).encode();tmp=out.with_name(out.name+'.partial-'+str(os.getpid()));fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.link(tmp,out,follow_symlinks=False);df=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df);tmp.unlink();print(json.dumps({k:v[k] for k in ('node','observed_unix','manager_alive','worker_alive','model_loaded','local_processed','acknowledged_shared_sources','model_calls_by_observer')}))
