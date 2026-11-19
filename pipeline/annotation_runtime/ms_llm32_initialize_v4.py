"""Initialize text-only revision once from existing metadata, preserving old budgets and three reusable diagnostics."""
import hashlib,json,os,sqlite3,time
from pathlib import Path
import ms_llm32_codec_v4 as codec
from ms_llm32_pool_v4 import DynamicPool,text
from stage2_local_publication_v1 import write_once
RT=Path(__file__).parent
SERVICE=Path('/root/stage2_MS_LLM32_claim_service_v4')
SHARED=RT.parent/'MS_LLM32_english_merged_edges_v4'


def run():
 SERVICE.mkdir(exist_ok=False);SHARED.mkdir(exist_ok=False);inputdb=sqlite3.connect('file:/root/stage2_motion_revision_inputs_20261028_v2/catalog.db?mode=ro',uri=True);rows=[json.loads(r[0]) for r in inputdb.execute('SELECT payload FROM sources ORDER BY ordinal')];blocked=inputdb.execute('SELECT count(*) FROM errors').fetchone()[0];inputdb.close();pool=DynamicPool.create(SERVICE/'model_pool.db',rows,[])
 for row in rows:row['revision']=codec.REVISION
 pool.db.execute('BEGIN IMMEDIATE')
 try:
  pool.db.execute('CREATE TABLE external_allocations(source_key TEXT PRIMARY KEY,node TEXT NOT NULL,allocation_ordinal INTEGER NOT NULL,revision TEXT NOT NULL)')
  # Preserve prior external source-node quotas, but initialize NEW text-stage budgets.
  previous=sqlite3.connect('file:/root/stage2_motion_revision_claim_service_20261028_v3/model_pool.db?mode=ro',uri=True)
  allocations=previous.execute('SELECT source_key,node,allocation_ordinal FROM external_allocations ORDER BY node,allocation_ordinal').fetchall();previous.close();assert len(allocations)==72000
  for key,node,ordinal in allocations:
   changed=pool.db.execute('UPDATE sources SET state="external_reserved" WHERE source_key=? AND state="unused"',(key,));assert changed.rowcount==1
   pool.db.execute('INSERT INTO external_allocations VALUES(?,?,?,?)',(key,node,ordinal,codec.REVISION))
  pool.db.execute('INSERT INTO meta VALUES(?,?)',('revision',codec.REVISION));pool.db.execute('COMMIT')
 except BaseException:pool.db.execute('ROLLBACK');raise
 allocation=dict(revision=codec.REVISION,source_count=72000,nodes=[dict(node='a100-extra-0'+str(i),source_count=12000,owners=['a100-extra-0'+str(i)+'-g'+pair+'-msllm4' for pair in ('0123','4567')]) for i in range(1,7)],old_MS_video_beat_extraction_reused=True,old_VLM_captions_not_read=True,new_stage_max_calls=1)
 aref=write_once(SHARED/'external_allocation_v4.json',allocation)
 # Import the same three previously validated pure-MS32B outputs, without re-inference.
 root=Path('/root/stage2_MS_LLM32_main_N2_v4');root.mkdir(exist_ok=False);out=root/'out';out.mkdir();owner='N2-g23-msllm4';batch='N2-seed-reuse-three-msllm4';seed_directory=RT.parent/'motion_revision_english_merged_edges_final_20261028_v1/fixed_three_MS_LLM_wording_v2';seedids=['wToyUHvrVoLgBvUd','XiCMwgEceOlirSYX','pImhVAYNzZmZiVZX'];selected=[];refs=[]
 for sid in seedids:
  source=pool.db.execute('SELECT split,id,payload,state FROM sources WHERE source_key=?',(sid.casefold(),)).fetchone();assert source and source[3]=='unused','Previously diagnosed seed must not consume external quota';selected.append(json.loads(source[2]))
 claim=dict(schema='stage2_dynamic_claim_receipt_v1',batch_id=batch,owner=owner,source_keys=[[r['split'],r['id']] for r in selected],rows=selected,source_count=3,assigned_unix=time.time(),automatic_retry=False,maximum_model_calls_per_source=1)
 pool.db.execute('BEGIN IMMEDIATE')
 try:
  pool.db.execute('INSERT INTO batches VALUES(?,?,?,?,?)',(batch,owner,3,text(claim['source_keys']),claim['assigned_unix']))
  for row in selected:assert pool.db.execute('UPDATE sources SET state="assigned",owner=?,batch_id=? WHERE source_key=? AND state="unused"',(owner,batch,row['id'].casefold())).rowcount==1
  pool.db.execute('COMMIT')
 except BaseException:pool.db.execute('ROLLBACK');raise
 packet=write_once(out/'packets'/owner/(batch+'.json'),dict(schema='MS_LLM32_reused_diagnostic_packet_v4',revision=codec.REVISION,owner=owner,claim=claim,source_count=3,models_called=0))
 for row in selected:
  p=seed_directory/(row['id']+'.json');raw=p.read_bytes();v=json.loads(raw);caption=v['caption'];context=json.loads((RT.parent/'motion_revision_english_merged_edges_final_20261028_v1/fixed_three_evidence_diagnostics_v1'/row['id']/'context.json').read_bytes());codec.validate(caption,context);codec.validate(codec.normalize(caption,context),context)
  d=out/'sources'/row['split']/row['id'];write_once(out/'budgets'/row['split']/row['id']/'source.claim.json',dict(revision=codec.REVISION,owner=owner,original_pool_claim=claim,maximum_model_calls=1,automatic_retry=False,reused_prior_LLM_call=True,new_model_calls=0));write_once(d/'import_evidence.json',dict(original_validated_result_path=str(p),original_validated_result_sha256=hashlib.sha256(raw).hexdigest(),original_LLM_call_ref=v['raw_LLM_result_ref'],new_model_calls=0))
  result=dict(id=row['id'],split=row['split'],revision=codec.REVISION,status='postprocessed_unapproved',owner=owner,input_refs=row,original_pool_claim_ref=packet,pool_claim_batch=batch,window_plan=context['plan'],window_plan_sha256=context['window_plan_sha256'],stages=[],error=None,pipeline='original_MS_LLM32B_text_only',image_inputs=0,video_inputs=0,VLM_patches_applied=0,reused_pure_text_diagnostic=True,dance_style=caption['dance_style'],windows=caption['windows'],fused_caption=codec.render(caption),continuity_notes=[],quality_approved=False,semantic_approval=False,training_eligible=False,human_review_completed=False)
  rr=write_once(d/'postprocessed.json',result);write_once(d/'commit.json',dict(id=row['id'],split=row['split'],revision=codec.REVISION,status='postprocessed_unapproved',result_ref=rr,new_model_calls=0,owner=owner,pool_claim_batch=batch,quality_approved=False,training_eligible=False))
 token=Path('/root/stage2_motion_revision_claim_service_20261028_v3/token').read_bytes();fd=os.open(SERVICE/'token',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(token);f.flush();os.fsync(f.fileno())
 cfg=dict(database=str(SERVICE/'model_pool.db'),token_path=str(SERVICE/'token'),local_root=str(SERVICE/'relay'),shared_root=str(SHARED/'production_results'),bind='0.0.0.0',port=19116,revision=codec.REVISION,source_count=len(rows),allowed_owners=[n+'-g'+pair+'-msllm4' for n in ('N1','N2','N3','N6','N8') for pair in ('01','23','45','67')]+[n['node']+'-g'+pair+'-msllm4' for n in allocation['nodes'] for pair in ('0123','4567')])
 write_once(SERVICE/'server.json',cfg);write_once(SHARED/'revision_contract.json',dict(revision=codec.REVISION,original_inputs=len(rows),blocked_original_inputs=blocked,language='English',rich_role='dance_style_only',window_policy='merge_short_edges',model='existing Qwen3-VL-32B-Instruct, language-only mode',dtype='bfloat16',quantization=None,image_inputs=0,video_inputs=0,VLM_actions='none',maximum_model_calls_per_source=1,batch_size=32,reused_diagnostic_sources=3,old_results_budgets_preserved=True,created_unix=time.time()));pool.close();print(json.dumps(dict(initialized=True,revision=codec.REVISION,inputs=len(rows),blocked_inputs=blocked,seed_sources_reused=3,external_reserved=72000,new_model_calls=0,server_config=str(SERVICE/'server.json'))))

if __name__=='__main__':run()
