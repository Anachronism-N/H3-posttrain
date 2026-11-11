"""Bounded model-only factual review with explicit controlled-test labels.

Reviewer predictions never become human scores or generator feedback. Known
control expectations remain private; correct labels do not validate real motion,
source independence, or general metric validity on unseen songs.
"""
import argparse
import copy
from pathlib import Path

from .baseline import _Journal,_sealed,_verify
from .io import file_lock,file_sha,json_text,object_sha,read_json,read_json_snapshot,write_json
from .memory_replay import _dispatch,_ReplayInfrastructureError,_accounting
from .providers import create_provider,parse_json_response
from .request_budget import guard_request
from .schema import keys,integer,text,identifier


PROMPT=(
 "Evaluate only the proposed text edit, not music or observed physical performance. "
 "You have not heard audio or seen a dancer. The source is a possible choreography. "
 "Use the public instruction and explicitly protected motion facts as requirements. "
 "Do not reward a sentence saying preserve the original motions unless those motions "
 "are actually stated or clearly equivalent in the final self-contained action. "
 "Distinguish retained/equivalent facts from omitted facts, contradictory facts and uncertainty. "
 "A retained positive phrase plus a later negation can be a contradiction. Equivalent paraphrases "
 "may retain facts without reusing words. Do not require choreography to match one unique dance. "
 "Judge only clear textual relations; unknown physical support/contact states remain unknown. "
 "Return a JSON object {item_id,goal,facts,new_textual_conflict,reason}. Goal is pass/partial/fail/unknown. "
 "For each supplied protected fact return {fact_id,status,evidence}; status is retained/omitted/contradicted/unknown. "
 "new_textual_conflict is yes/no/unknown. Retained and contradicted require a nonempty exact candidate quote; "
 "for omitted or unknown, use an exact candidate quote or empty text. "
 "Do not use method identities, predicted scores, word length or unknown output source as a preference cue."
)


def _schema(item):
 facts=[dict(type='object',additionalProperties=False,required=['fact_id','status','evidence'],properties=dict(
  fact_id=dict(type='string',enum=[f['fact_id']]),status=dict(type='string',enum=['retained','omitted','contradicted','unknown']),
  evidence=dict(type='string',maxLength=300)),allOf=[{
  'if':dict(properties=dict(status=dict(enum=['retained','contradicted']))),
  'then':dict(properties=dict(evidence=dict(minLength=1,pattern=r'\S')))
 }]) for f in item['protected_facts']]
 return dict(type='object',additionalProperties=False,required=['item_id','goal','facts','new_textual_conflict','reason'],properties=dict(
  item_id=dict(type='string',enum=[item['item_id']]),goal=dict(type='string',enum=['pass','partial','fail','unknown']),
  facts=dict(type='array',minItems=len(facts),maxItems=len(facts),prefixItems=facts,items=False),
  new_textual_conflict=dict(type='string',enum=['yes','no','unknown']),reason=dict(type='string',minLength=1,maxLength=800)))


def parse_review(item,response):
 value=parse_json_response(response)
 keys(value,('item_id','goal','facts','new_textual_conflict','reason'),where='model factual review')
 if value['item_id']!=item['item_id'] or value['goal'] not in ('pass','partial','fail','unknown') or value['new_textual_conflict'] not in ('yes','no','unknown'):
  raise ValueError('Review identity or verdict changed')
 text(value['reason'],'review reason')
 if len(value['reason'])>800 or not isinstance(value['facts'],list) or len(value['facts'])!=len(item['protected_facts']):
  raise ValueError('Factual review exceeds explicit inventory/budget')
 for expected,fact in zip(item['protected_facts'],value['facts']):
  keys(fact,('fact_id','status','evidence'),where='review fact')
  if fact['fact_id']!=expected['fact_id'] or fact['status'] not in ('retained','omitted','contradicted','unknown'):
   raise ValueError('Review fact identity/status invalid')
  if not isinstance(fact['evidence'],str) or len(fact['evidence'])>300:
   raise ValueError('Fact evidence invalid')
  if fact['status'] in ('retained','contradicted') and not fact['evidence'].strip():
   raise ValueError('Retained/contradicted facts require nonempty candidate evidence')
  if fact['evidence'] and fact['evidence'] not in item['candidate_action']:
   raise ValueError('Review evidence is not an exact span of the candidate')
 statuses=[f['status'] for f in value['facts']]
 value['protected_fact_preservation']=(False if any(s in ('omitted','contradicted') for s in statuses) else
   None if 'unknown' in statuses else True)
 value['text_joint_success']=(value['goal']=='pass' and value['protected_fact_preservation'] is True and value['new_textual_conflict']=='no')
 return value


def run_fact_review(inventory_path,provider_spec,output_dir,resume=False):
 path,output=Path(inventory_path).resolve(),Path(output_dir).resolve()
 inventory,sha=read_json_snapshot(path)
 keys(inventory,('schema','seed','items'),where='fact review inventory')
 if inventory['schema']!='choreobench.fact_review_inventory.v1' or not isinstance(inventory['items'],list) or not 1<=len(inventory['items'])<=24:
  raise ValueError('Model-only review permits 1-24 frozen items')
 integer(inventory['seed'],'seed')
 if inventory['seed']>=2**63:raise ValueError('Review seed outside range')
 seen=set();requests=[]
 for item in inventory['items']:
  keys(item,('item_id','source_action','candidate_action','instruction','protected_facts','private_expected','private_source'),where='review item')
  identifier(item['item_id'],'review item_id')
  for field in ('source_action','candidate_action','instruction'):text(item[field],field)
  if item['item_id'] in seen:raise ValueError('Duplicate review item')
  seen.add(item['item_id'])
  if not isinstance(item['protected_facts'],list) or not 1<=len(item['protected_facts'])<=8:
   raise ValueError('Protected fact list outside bound')
  for fact in item['protected_facts']:
   keys(fact,('fact_id','description','source_evidence'),where='protected motion fact')
   for field in fact:text(fact[field],field)
   if fact['source_evidence'] not in item['source_action']:raise ValueError('Protected fact does not cite source action')
  public={k:copy.deepcopy(item[k]) for k in ('item_id','source_action','candidate_action','instruction','protected_facts')}
  request=dict(call_id=item['item_id']+'/fact-review',seed=inventory['seed'],max_output_tokens=1200,
   response_kind='factual_edit_review',messages=[dict(role='system',content=PROMPT),dict(role='user',content=json_text(public))],response_schema=_schema(item))
  guard_request(request,dict(max_request_characters=20000));requests.append((item,request))
 provider=create_provider(provider_spec,role='planner',base_dir=path.parent)
 if provider.identity['kind'] not in ('chat_completions','replay'):raise ValueError('Factual review requires explicit text provider')
 contract=dict(schema='choreobench.fact_review_contract.v1',sources={str(path):sha},provider_identity=copy.deepcopy(provider.identity),
  maximum_calls=len(requests),code_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')},
  request_sha256={r['call_id']:object_sha(r) for _,r in requests},model_review_not_human=True,audio_seen=False,
  private_control_labels_hidden=True,formal_metric_calibration_verified=False)
 if output.exists() and not resume:raise FileExistsError(str(output))
 if resume and not (output/'contract.json').is_file():raise ValueError('Review resume requires original contract')
 output.mkdir(parents=True,exist_ok=True)
 with file_lock(output/'.run.lock'):
  def save(relative,value):
   p=output/relative
   if p.exists():
    if read_json(p)!=value:raise ValueError('Immutable factual review artifact changed')
   else:write_json(p,value)
  save('contract.json',contract)
  journal=_Journal(output,object_sha(contract),len(requests));previous=None
  if (output/'results.json').exists():
   previous=_verify(read_json(output/'results.json'))
   if previous['complete']:
    ids={r['call_id'] for _,r in requests}
    if set(journal.records)!=ids or set(journal.started)!=ids or previous['call_records']!={k:v['record_sha256'] for k,v in journal.records.items()}:
     raise ValueError('Completed review calls missing/changed; no replacement sampling')
    for relative,digest in previous['artifacts'].items():
     if file_sha(output/relative)!=digest:raise ValueError('Completed review artifact changed')
  rows,artifacts,fatal=[],{},False
  for item,request in requests:
   row=dict(item_id=item['item_id'],origin='model_judge',audio_seen=False,status='pending',human_review=False)
   try:
    if file_sha(path)!=sha:raise ValueError('Factual review inventory changed')
    if previous and previous['complete'] and request['call_id'] not in journal.records:raise ValueError('Completed review would dispatch')
    response=_dispatch(journal,provider,request,contract)
    review=parse_review(item,response['text']);row.update(status='valid',review=review)
    relative='reviews/'+item['item_id']+'.json';save(relative,review);artifacts[relative]=file_sha(output/relative)
   except (ValueError,TypeError,KeyError) as error:
    row.update(status='failed',error=str(error),retained_in_denominator=True)
    raw=journal.records.get(request['call_id']);info=raw.get('result') if raw else None
    if isinstance(error,_ReplayInfrastructureError) or not raw or not raw.get('success') or not isinstance(info,dict) or info.get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response'):
     fatal=True
   rows.append(row)
   if fatal:break
  if file_sha(path)!=sha:raise ValueError('Review source changed')
  result=_sealed(dict(schema='choreobench.fact_review_result.v1',complete=len(rows)==len(requests) and not fatal,
   contract_sha256=object_sha(contract),rows=rows,artifacts=artifacts,calls_used=len(journal.started),
   call_records={k:v['record_sha256'] for k,v in journal.records.items()},**_accounting(journal),
   human_reviews=0,audio_seen=False,formal_metric_validity_verified=False,quality_superiority_established=False))
  if previous and previous['complete'] and previous!=result:raise ValueError('Reconstructed completed review differs')
  save('results.json',result)
  return result


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--inventory',required=True);parser.add_argument('--provider',required=True)
 parser.add_argument('--out-dir',required=True);parser.add_argument('--resume',action='store_true')
 args=parser.parse_args(argv)
 r=run_fact_review(args.inventory,read_json(args.provider),args.out_dir,args.resume)
 print(json_text({k:r[k] for k in ('complete','calls_used','rows','formal_metric_validity_verified')}))


if __name__=='__main__':main()
