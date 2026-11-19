"""ACK-to-ledger reconciliation only; never claims sources or invokes a model."""
import argparse
import fcntl
import hashlib
import json
import re
import os
from pathlib import Path
import subprocess
import time
from stage2_dynamic_pool_v1 import DynamicPool
from stage2_local_publication_v1 import write_once

ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
LOCAL=Path('/root/stage2_ack_service_20261028_v1')
SHARED=ROOT/'stage2_dynamic_local_shared_results_20261028_v1'
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
STATUS={'postprocessed_unapproved':'published_unapproved','postprocessed_needs_visual_revision':'needs_visual_revision','insufficient_visible_evidence':'insufficient_visible_evidence','failed_sealed_no_retry':'failed_sealed_no_retry'}


def remote(node,batch,keys,commit_ref,group):
    program='''import pathlib,json,hashlib
root=pathlib.Path('/root/stage2_dynamic_'+GROUP+'_'+NODE+'_20261028_v1/out');batch=BATCH;keys=KEYS;ref=REF
ack=json.loads((root/'outbox'/batch/'upload_receipt.json').read_bytes())
assert ack['batch_id']==batch and ack['shared_commit']==ref and ack['models_called']==0
results=[]
for split,id_ in keys:
 p=root/'sources'/split/id_/'commit.json';v=json.loads(p.read_bytes());assert v['id']==id_ and v['split']==split and v['quality_approved'] is False and v['training_eligible'] is False and v['new_model_calls']<=2
 results.append({k:v[k] for k in ('id','split','owner','pool_claim_batch','status')})
print(json.dumps(results))
'''
    values=dict(NODE=node,BATCH=batch,KEYS=keys,REF=commit_ref,GROUP=group)
    program=re.sub(r'\b(?:NODE|BATCH|KEYS|REF|GROUP)\b',lambda m:repr(values[m.group()]),program)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HOSTS[node],'/opt/venv_eval/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=20)
    if r.returncode:raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout)


def run(group):
    LOCAL=Path('/root/stage2_ack_service_'+group+'_20261028_v1')
    LOCAL.mkdir(exist_ok=True);(LOCAL/'reconciled').mkdir(exist_ok=True)
    fd=os.open(LOCAL/'service.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pool=DynamicPool('/root/stage2_pool_'+group+'_20261028_v1/pool.db');allkeys={x[0] for x in pool.db.execute('SELECT source_key FROM sources')}
    while not (LOCAL/'STOP').exists():
        for node in HOSTS:
            directory=SHARED/node
            if not directory.exists():continue
            for batch in directory.iterdir():
                done=LOCAL/'reconciled'/(batch.name+'.json');commit=batch/'commit.json'
                if done.exists() or not commit.exists():continue
                try:
                    raw=commit.read_bytes();v=json.loads(raw);keys=[k for k in v['source_keys'] if k[1].casefold() in allkeys]
                    if not keys:continue
                    if len(keys)!=v['source_count']:raise ValueError('Mixed original source groups in one upload')
                    ref=dict(path=str(commit),sha256=hashlib.sha256(raw).hexdigest());outcomes=remote(node,batch.name,keys,ref,group)
                    if sorted((x['split'],x['id']) for x in outcomes)!=sorted(map(tuple,keys)):raise ValueError('ACK/source mapping differs')
                    for x in outcomes:
                        status=STATUS[x['status']];row=pool.db.execute('SELECT state,owner,batch_id,split,id FROM sources WHERE source_key=?',(x['id'].casefold(),)).fetchone()
                        if row[1:]!=(x['owner'],x['pool_claim_batch'],x['split'],x['id']):raise ValueError('Original claim identity differs')
                        if row[0]=='assigned':pool.terminal(x['owner'],x['pool_claim_batch'],x['split'],x['id'],status)
                        elif row[0]!=status:raise ValueError('Existing terminal must never be changed')
                    write_once(done,dict(shared_commit=ref,source_count=len(keys),outcomes=outcomes,ack_verified=True,source_claims_created=0,models_called=0,reconciled_unix=time.time()))
                    print(json.dumps(dict(event='ack_ledger_closed',batch=batch.name,sources=len(keys),models_called=0)),flush=True)
                except Exception as e:print(json.dumps(dict(event='ack_reconciliation_wait',batch=batch.name,error=repr(e),models_called=0)),flush=True)
        time.sleep(15)
    pool.close();os.close(fd)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',choices=('main3936','remaining25965'),default='main3936');a=p.parse_args();run(a.group)
