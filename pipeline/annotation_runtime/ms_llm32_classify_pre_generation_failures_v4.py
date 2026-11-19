"""Classify only proven pre-generation schema failures; never reclaim sources or reset budgets."""
import hashlib,json,os,sqlite3,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
RT=Path(__file__).parent
ROOT=RT.parent/'MS_LLM32_english_merged_edges_v4'
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N6':'28.216.19.225','N8':'28.216.18.136'}

def collect(item):
 node,host=item
 code='''import pathlib,json,hashlib
r=pathlib.Path('/root/stage2_MS_LLM32_main_'+NODE+'_v4/out');records=[];cache={}
for p in (r/'sources').glob('*/*/postprocessed.json'):
 raw=p.read_bytes();v=json.loads(raw);error=v.get('error') or ''
 if v['status']!='failed_sealed_no_retry' or 'VLLMValidationError' not in error or 'JSON schema contains features not supported by xgrammar' not in error:continue
 stage=v['stages'][0];ref=stage['batch_response_ref'];key=ref['path']
 if key not in cache:
  response_raw=pathlib.Path(key).read_bytes();assert hashlib.sha256(response_raw).hexdigest()==ref['sha256'];cache[key]=json.loads(response_raw)
 response=cache[key];assert response['responses']==[] and 'JSON schema contains features not supported by xgrammar' in response['error']
 records.append(dict(id=v['id'],split=v['split'],owner=v['owner'],pool_claim_batch=v['pool_claim_batch'],original_failure_result_ref=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()),pre_generation_batch_response_ref=ref,classification='schema_rejected_before_generation',model_generation_count=0,original_claim_budgets_preserved=True))
print(json.dumps(dict(node=NODE,records=records)))
'''.replace('NODE',repr(node))
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,'/opt/venv_eval/bin/python','-B','-'],input=code,capture_output=True,text=True,timeout=90)
 if r.returncode:raise RuntimeError(node+': '+r.stderr[-500:])
 return json.loads(r.stdout)

def run():
 with ThreadPoolExecutor(max_workers=5) as pool:values=list(pool.map(collect,HOSTS.items()))
 db=sqlite3.connect('/root/stage2_MS_LLM32_claim_service_v4/model_pool.db',timeout=30,isolation_level=None);db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE zero_generation_schema_failures(source_key TEXT PRIMARY KEY,owner TEXT NOT NULL,batch_id TEXT NOT NULL,proof TEXT NOT NULL,closure_state TEXT NOT NULL DEFAULT "pending_first_generation_completion")');db.execute('BEGIN IMMEDIATE')
 try:
  for node in values:
   for r in node['records']:
    source=db.execute('SELECT state,owner,batch_id,split,id FROM sources WHERE source_key=?',(r['id'].casefold(),)).fetchone();assert source and source[1:]==(r['owner'],r['pool_claim_batch'],r['split'],r['id']) and source[0] in ('assigned','failed_sealed_no_retry');db.execute('INSERT INTO zero_generation_schema_failures(source_key,owner,batch_id,proof) VALUES(?,?,?,?)',(r['id'].casefold(),r['owner'],r['pool_claim_batch'],json.dumps(r)))
  db.execute('COMMIT')
 except BaseException:db.execute('ROLLBACK');raise
 records=[r for v in values for r in v['records']];p=ROOT/'zero_generation_schema_failures.jsonl';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as out:
  for r in records:out.write(json.dumps(r)+'\n')
  out.flush();os.fsync(out.fileno())
 report=dict(total=len(records),by_node={v['node']:len(v['records']) for v in values},model_generation_count=0,ordinary_failed_sources_not_included=True,source_owners_claims_budgets_unchanged=True,closure_complete=False,observed_unix=time.time());(ROOT/'zero_generation_schema_failure_report.json').write_text(json.dumps(report));print(json.dumps(report));db.close()
if __name__=='__main__':run()
