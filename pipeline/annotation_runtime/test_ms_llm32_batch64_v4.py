"""CPU32 text stage: isolated source mapping, one budget, external exclusion and reliable publication."""
import hashlib,json,tempfile,types
from pathlib import Path
import ms_llm32_codec_batch64_v4 as c
import sys
sys.modules['ms_llm32_codec_v4']=c
import ms_llm32_worker_batch64_v4 as worker
from ms_llm32_pool_v4 import DynamicPool
from stage2_local_publication_v1 import write_once,checked,LocalOutbox
from ms_llm32_server_v4 import acknowledge,finish_ledger
class Fake:
 def __init__(self):self.calls=[]
 def chat(self,requests,sampling_params,use_tqdm):
  self.calls.append(len(requests));out=[]
  for request in requests:
   assert all(isinstance(m['content'],str) for m in request);p=json.loads(request[1]['content']);assert 'visual_audit' not in p and 'images' not in p
   value=dict(id=p['id'],split=p['split'],dance_style=p['dance_style'],windows=[dict(window_id=w['window_id'],start=w['start'],end=w['end'],caption='The arms move toward image-left, then the elbows bend.') for w in p['immutable_window_plan']],continuity_notes=[])
   out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=json.dumps(value),token_ids=[1]*60,finish_reason='stop')]))
  return out
with tempfile.TemporaryDirectory(prefix='MS32text-',dir='/tmp') as d:
 root=Path(d)/'local';root.mkdir();owner='N2-g23-msllm4';rows=[dict(id='TextSource'+str(i),split='train',video_path='/never/read/'+str(i)+'.mp4') for i in range(80)];pool=DynamicPool.create(Path(d)/'model_pool.db',rows,[]);pool.db.execute('CREATE TABLE external_allocations(source_key TEXT PRIMARY KEY,node TEXT,allocation_ordinal INTEGER,revision TEXT)')
 for i in range(72,80):pool.db.execute('UPDATE sources SET state="external_reserved" WHERE id=?',(rows[i]['id'],));pool.db.execute('INSERT INTO external_allocations VALUES(?,?,?,?)',(rows[i]['id'].casefold(),'a100-extra-01',i,c.REVISION))
 refs=[]
 for i in range(8):
  claim=pool.claim(owner,'textClaimGroup'+str(i),8);contexts=[]
  for row in claim['rows']:
   plan=dict(policy='merge_short_edges',windows=[dict(window_id=0,start=0.,end=1.)]);context=dict(revision=c.REVISION,row=row,plan=plan,window_plan_sha256=hashlib.sha256(c.encoded(plan)).hexdigest(),draft=dict(events=[dict(event_id='e0',start=0.,end=.8,text='两臂向画面左侧移动。')],observation={}),style_hint=dict(candidates=['urban']),rich_body_forwarded_to_VLM=False)
   contexts.append(write_once(root/'prepared'/row['id']/'context.json',context))
  refs.append(write_once(root/'packets'/owner/(claim['batch_id']+'.json'),dict(schema='MS_LLM32_text_packet_v4',revision=c.REVISION,owner=owner,claim=claim,context_refs=contexts,source_count=8)))
 fake=Fake();result=worker.run_packets(refs,root,owner,fake,lambda schema:schema,lambda:None);assert len(result)==64 and fake.calls==[64] and all(r['new_model_calls']==1 for r in result)
 again=Fake()
 try:worker.run_packets(refs,root,owner,again,lambda schema:schema,lambda:None);raise AssertionError('Repeated model budget')
 except FileExistsError:pass
 assert not again.calls
 try:worker.run_packets([refs[0],refs[0]],root,owner,again,lambda schema:schema,lambda:None);raise AssertionError('Repeated source accepted')
 except ValueError:pass
 ext=pool.claim('a100-extra-01-g0123-msllm4','externalClaimBatch1',8);assert all(row['id'] in {r['id'] for r in rows[72:]} for row in ext['rows']);assert pool.claim('a100-extra-02-g0123-msllm4','foreignNodeClaim1',8) is None
 box=LocalOutbox(root,Path(d)/'shared','N2');seal=checked(box.seal([[r['split'],r['id']] for r in result[:50]]));cfg=dict(database=str(pool.path),allowed_owners=[owner]);outcomes=acknowledge(cfg,seal,box.local/'outbox'/seal['batch_id']/'batch.tar');finish_ledger(cfg,outcomes);second=checked(box.seal([[r['split'],r['id']] for r in result[50:]]));finish_ledger(cfg,acknowledge(cfg,second,box.local/'outbox'/second['batch_id']/'batch.tar'));assert pool.progress()['published_unapproved']==64;ack=box.upload(dict(path=str(box.local/'outbox'/seal['batch_id']/'sealed.json'),sha256=hashlib.sha256((box.local/'outbox'/seal['batch_id']/'sealed.json').read_bytes()).hexdigest()));assert ack['source_count']==50
 pool.close()
print(json.dumps(dict(text_only64_calls=[64],single_stage_budget=True,no_video_copy=True,duplicate_and_retry_blocked=True,external_source_isolation=True,shared_commit_ACK=True,new_production_GPU_calls=0)))
