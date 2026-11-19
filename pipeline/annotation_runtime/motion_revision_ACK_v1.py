"""ACK-only terminal closure for the final correct revision; never rerun or re-claim."""
import json,hashlib,sqlite3,subprocess,time,re,fcntl,os
from pathlib import Path
from stage2_dynamic_pool_v1 import DynamicPool
from stage2_local_publication_v1 import write_once
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
SERVICE=Path('/root/stage2_motion_revision_final_ACK_service_20261028_v1')
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}
MAP={'postprocessed_unapproved':'published_unapproved','postprocessed_needs_visual_revision':'needs_visual_revision','insufficient_visible_evidence':'insufficient_visible_evidence','failed_sealed_no_retry':'failed_sealed_no_retry'}


def run():
 SERVICE.mkdir(exist_ok=True);(SERVICE/'closed').mkdir(exist_ok=True);fd=os.open(SERVICE/'service.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);pool=DynamicPool('/root/stage2_motion_revision_claim_service_20261028_v3/model_pool.db')
 while not (SERVICE/'STOP').exists():
  for node,host in HOSTS.items():
   directory=ROOT/'motion_revision_english_merged_edges_final_20261028_v1/production_results'/node
   for b in directory.iterdir() if directory.exists() else []:
    done=SERVICE/'closed'/(b.name+'.json');p=b/'commit.json'
    if done.exists() or not p.exists():continue
    try:
     raw=p.read_bytes();c=json.loads(raw);keys=c['source_keys'];ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest());program='''import pathlib,json
root=pathlib.Path('/root/stage2_motion_revision_final_main_'+NODE+'_20261028_v1/out');ack=json.loads((root/'outbox'/BATCH/'upload_receipt.json').read_bytes());assert ack['shared_commit']==REF and ack['models_called']==0;out=[]
for split,id_ in KEYS:
 c=json.loads((root/'sources'/split/id_/'commit.json').read_bytes());assert c['revision']=='english_merged_edges_motion_revision_v3' and c['training_eligible'] is False and c['quality_approved'] is False;out.append({k:c[k] for k in ('id','split','owner','pool_claim_batch','status')})
print(json.dumps(out))
''';values=dict(NODE=node,BATCH=b.name,REF=ref,KEYS=keys);program=re.sub(r'\b(?:NODE|BATCH|REF|KEYS)\b',lambda m:repr(values[m.group()]),program);r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,'/opt/venv_eval/bin/python','-B','-'],input=program,text=True,capture_output=True,timeout=20);assert r.returncode==0,r.stderr[-400:];rows=json.loads(r.stdout);assert sorted((x['split'],x['id']) for x in rows)==sorted(map(tuple,keys))
     for x in rows:
      old=pool.db.execute('SELECT state,owner,batch_id,split,id FROM sources WHERE source_key=?',(x['id'].casefold(),)).fetchone();assert old and old[1:]==(x['owner'],x['pool_claim_batch'],x['split'],x['id']);status=MAP[x['status']]
      if old[0]=='assigned':pool.terminal(x['owner'],x['pool_claim_batch'],x['split'],x['id'],status)
      else:assert old[0]==status
     write_once(done,dict(revision='english_merged_edges_motion_revision_v3',shared_commit=ref,source_count=len(keys),local_ACK_verified=True,model_calls=0,source_claims_created=0,closed_unix=time.time()));print(json.dumps(dict(event='final_revision_ACK_closed',node=node,sources=len(keys),model_calls=0)),flush=True)
    except Exception as e:print(json.dumps(dict(event='final_ACK_wait',batch=b.name,error=repr(e),model_calls=0)),flush=True)
  time.sleep(15)
 pool.close();os.close(fd)
if __name__=='__main__':run()
