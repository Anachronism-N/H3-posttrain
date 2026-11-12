"""Four bounded composition adaptations sharing evidence and draft permissions.

No independent grader, test score or raw music access enters the tool loop.
"""
import copy
from pathlib import Path
from .baseline import _Journal,_empty_plan,_sealed,_verify,_recorded_result
from .coordination import _outline_request,_read_outline,_unit_request,_read_unit,_candidates
from .io import file_lock,file_sha,object_sha,read_json,write_json,json_text
from .memory_replay import _dispatch,_unchanged,_ReplayInfrastructureError,_snapshot
from .planning_units import build_planning_units
from .providers import create_provider,parse_json_response
from .request_budget import guard_request
from .schema import keys,text,integer,validate_plan
from .song_agent import load_song_source
from .timing import event_clock
from .untimed_catalog import load_catalog_song

METHODS=('workflow','self_refine','react','choreography')
TOOL_PROMPT=(
 'Replace block-only output with exactly {tool,proposal,reason}. '
 'tool is read_evidence/read_history/propose_block/check_block/revise_block/commit_block/finish. '
 'proposal is {unit_id,events} only for propose_block/revise_block; otherwise null. '
 'Read tools return public evidence or full committed history. Propose/revise replaces only the current '
 'uncommitted draft. Check gives structural feedback, not a music or semantic quality score. '
 'Commit validates and atomically appends the current draft; prior committed actions cannot be edited. '
 'Finish succeeds only after all units. Choose tools freely within the common budget; inspections are optional. '
 'All methods have the same full history, local evidence, current draft, permissions and clock. '
 'No actual extra listening tool exists in this version; do not claim one. '
)
POLICY={'react':'Act as a general ReAct-style tool planner: decide, observe feedback and continue the music-to-choreography task.',
 'choreography':'Track active constraints, already committed phrase development and unresolved draft issues. '
 'Use duration-aware primary movements with concurrent modulation, avoid uncontrolled long-chain copying, '
 'and revise only where feedback or evidence supports a change. Unknown physical state must remain unknown.'}


def _source(path,song_id,cfg,sources):
 path=Path(path).resolve()
 if path.is_dir():return load_song_source(path,song_id,sources)
 header=read_json(path)
 if header.get('schema')=='choreobench.complete_catalog_song.v1':
  from .catalog_song import load_song_view
  original,options,synthetic,digest=load_song_view(path,song_id,sources)
 else:original,options,synthetic,digest=load_catalog_song(path,song_id,sources)
 units=build_planning_units(original,config=cfg['unit_config'])
 return dict(original=original,song=copy.deepcopy(original),options=options,units=units,config=cfg,
             synthetic=synthetic,source_contract_sha256=digest)


class CompositionState:
 def __init__(self,source,outline):
  self.source=source;self.outline=outline;self.plan=_empty_plan(source['original'])
  self.plan['motifs']=copy.deepcopy(outline['motifs']);self.index=0;self.draft=None;self.observations=[]
 def validate(self,proposal):
  units=self.source['units']['units']
  if self.index==len(units):raise ValueError('No current unit')
  unit=units[self.index]
  events=_read_unit(json_text(proposal),self.source['original'],unit,self.outline,self.source['options'],
       self.source['config']['max_events_per_unit'],event_clock(unit,self.plan['sample_rate'],self.source['config']))
  plan=copy.deepcopy(self.plan);plan['events'].extend(events);validate_plan(plan,self.source['original'])
  return events,plan
 def execute(self,decision):
  keys(decision,('tool','proposal','reason'),where='composition tool');text(decision['reason'],'tool reason')
  if len(decision['reason'])>1200:raise ValueError('Tool reason budget exceeded')
  tool=decision['tool'];proposal=decision['proposal']
  if tool not in ('read_evidence','read_history','propose_block','check_block','revise_block','commit_block','finish'):
   raise ValueError('Unknown composition tool')
  if tool not in ('propose_block','revise_block') and proposal is not None:raise ValueError('Read/check/commit/finish cannot mutate draft')
  if tool=='read_evidence':return dict(tool=tool,status='observed',candidates=_candidates(self.source['options']),units=self.source['units'])
  if tool=='read_history':return dict(tool=tool,status='observed',committed_plan=copy.deepcopy(self.plan))
  if tool=='finish':return dict(tool=tool,status='finished' if self.index==len(self.source['units']['units']) else 'incomplete_finish')
  if tool in ('propose_block','revise_block'):
   if tool=='revise_block' and self.draft is None:raise ValueError('No current draft to revise')
   # Invalid draft is visible for repair but never becomes committed history.
   keys(proposal,('unit_id','events'),where='block draft')
   if self.index==len(self.source['units']['units']) or proposal['unit_id']!=self.source['units']['units'][self.index]['unit_id']:
    raise ValueError('Draft targets a different unit')
   self.draft=copy.deepcopy(proposal)
   return dict(tool=tool,status='drafted',draft_sha256=object_sha(self.draft),quality_verified=False)
  if self.draft is None:raise ValueError('No current draft')
  events,plan=self.validate(self.draft)
  observation=dict(tool=tool,status='structurally_valid',unit_id=self.draft['unit_id'],quality_verified=False)
  if tool=='commit_block':self.plan=plan;self.index+=1;self.draft=None;observation['status']='committed'
  return observation


def _request(state,cfg,method,phase,feedback=None):
 units=state.source['units']['units'];unit=units[state.index] if state.index<len(units) else None
 if unit is not None:
  req=_unit_request(state.source['song'],state.source['options'],state.source['units'],unit,state.outline,state.plan,state.source['config'])
  payload=parse_json_response(req['messages'][-1]['content']);schema=req['response_schema'];system=req['messages'][0]['content']
 else:payload={};schema=dict(type='null');system='All units are committed.'
 payload.update(committed_plan=copy.deepcopy(state.plan),current_draft=copy.deepcopy(state.draft),
    committed_units=state.index,total_units=len(units),tool_observations=state.observations[-2:],feedback=feedback,
    shared_permissions=True,actual_audio_access=False)
 if phase=='feedback':
  system+=' Give short source-grounded feedback about current draft action load, development and boundary clarity. Never invent heard audio or physical state. Return only {feedback}.'
  schema=dict(type='object',additionalProperties=False,required=['feedback'],properties=dict(feedback=dict(type='string',minLength=1,maxLength=2400)))
 elif method in ('react','choreography'):
  system+=' '+TOOL_PROMPT+POLICY[method]
  schema=dict(type='object',additionalProperties=False,required=['tool','proposal','reason'],properties=dict(
   tool=dict(enum=['read_evidence','read_history','propose_block','check_block','revise_block','commit_block','finish']),
   proposal=dict(anyOf=[dict(type='null'),schema]),reason=dict(type='string',minLength=1,maxLength=1200)))
 elif phase=='revise':system+=' Revise the same current-unit draft using supplied feedback. Keep the original output contract and permissions.'
 request=dict(call_id=state.plan['song_id']+'/four-method/%04d'%cfg['call_index'],seed=cfg['seed'],max_output_tokens=cfg['max_output_tokens'],
  response_kind='composition_'+phase,messages=[dict(role='system',content=system),dict(role='user',content=json_text(payload))],response_schema=schema)
 return guard_request(request,cfg)


def run_composition(source_path,song_id,provider_spec,out_dir,*,method='workflow',seed=42,max_output_tokens=1024,
                    outline_max_output_tokens=4096,max_request_characters=200000,max_input_tokens=2000000,max_total_output_tokens=150000,resume=False,outline_reuse_run=None):
 if method not in METHODS:raise ValueError('Unknown composition method')
 for name,value in [('seed',seed),('max_output_tokens',max_output_tokens),('outline_max_output_tokens',outline_max_output_tokens),
                    ('max_request_characters',max_request_characters),('max_input_tokens',max_input_tokens),('max_total_output_tokens',max_total_output_tokens)]:
  integer(value,name,0 if name=='seed' else 1)
 cfg=dict(seed=seed,max_output_tokens=max_output_tokens,outline_max_output_tokens=outline_max_output_tokens,
  max_request_characters=max_request_characters,max_events_per_unit=3,structured_unit_output=True,use_outline=True,use_history=True,
  event_time_format='boundary_ids',event_grid_seconds=.25,min_event_seconds=.5,event_schedule_policy='balanced_full_unit',
  unit_config=dict(bars_per_block=4,fallback_seconds=10.125,max_unit_seconds=20,meter_status='unverified',min_time_block_seconds=2))
 sources={};source=_source(source_path,song_id,cfg,sources);cfg['max_calls']=1+4*len(source['units']['units'])
 # Historical input cfg can contain a different provider; only timing/structure is used from it.
 source['config']=dict(source['config'],max_output_tokens=max_output_tokens,max_request_characters=max_request_characters,structured_unit_output=True,use_outline=True,use_history=True)
 if provider_spec.get('kind') not in ('chat_completions','replay'):raise ValueError('Composition requires explicit real/replay text provider')
 provider=create_provider(provider_spec,'planner',Path(source_path).resolve().parent)
 reused_outline=None
 if outline_reuse_run is not None:
  reuse=Path(outline_reuse_run).resolve();prior=_snapshot(reuse/'contract.json',sources)
  prior_result=_verify(_snapshot(reuse/'results.json',sources))
  if (prior['method']!=method or prior['provider_identity']!=provider.identity or not prior_result['complete']
      or prior_result['contract_sha256']!=object_sha(prior)):
   raise ValueError('Outline reuse method/model or complete result differs')
  old_journal=_Journal(reuse,object_sha(prior),cfg['max_calls']);cid=song_id+'/four-method/0000'
  if cid not in old_journal.records:raise ValueError('Original outline raw missing; no replacement')
  expected=_outline_request(source['song'],source['options'],source['units'],cfg);expected['call_id']=cid
  if old_journal.started[cid]['request']!=expected:raise ValueError('Original outline request differs')
  record=old_journal.records[cid]
  if prior_result['call_records'].get(cid)!=record['record_sha256']:raise ValueError('Original result does not bind outline raw')
  reused_outline=copy.deepcopy(_recorded_result(record,provider.identity,expected))
  for suffix in ('.started.json','.raw.json'):_snapshot(reuse/'calls'/(_Journal._name(cid)+suffix),sources)
  reused_outline['raw']=dict(original=copy.deepcopy(reused_outline['raw']),reused_original_outline=True,original_record_sha256=record['record_sha256'])
  reused_outline['costs']=dict(reused_outline['costs'],replayed=True,historical_calls=1)
 contract=dict(schema='choreobench.four_composition_contract.v1',sources=sources,method=method,config=cfg,provider_identity=provider.identity,
  max_input_tokens=max_input_tokens,max_total_output_tokens=max_total_output_tokens,source_inputs_sha256={k:object_sha(source[k]) for k in ('original','song','options','units')},
  code_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')},same_tool_permissions=True,fresh_method_outline=outline_reuse_run is None,historical_event_prefix_reused=False,
  original_outline_reused=outline_reuse_run is not None)
 output=Path(out_dir).resolve()
 if output.exists() and not resume:raise FileExistsError(str(output))
 if resume and not (output/'contract.json').exists():raise ValueError('Resume requires original contract')
 output.mkdir(parents=True,exist_ok=True)
 with file_lock(output/'.run.lock'):
  def save(name,value):
   path=output/name
   if path.exists():
    if read_json(path)!=value:raise ValueError('Immutable composition evidence changed: '+name)
   else:write_json(path,value)
  save('contract.json',contract);journal=_Journal(output,object_sha(contract),cfg['max_calls'])
  old=_verify(read_json(output/'results.json')) if (output/'results.json').exists() else None
  if old is not None:
   if not old.get('complete'):raise ValueError('Unknown infrastructure outcome requires diagnosis, not automatic replacement')
   if old['call_records']!={k:v['record_sha256'] for k,v in journal.records.items()} or old['calls_used']!=len(journal.started):raise ValueError('Completed composition raw missing; no replacement')
   for name,digest in old['artifacts'].items():
    if file_sha(output/name)!=digest:raise ValueError('Completed composition artifact changed')
  index=0;input_tokens=0;output_tokens=0;artifacts={};history=[];fatal=None;stop='budget_exhausted'
  def call(req):
   nonlocal index,input_tokens,output_tokens
   if index>=cfg['max_calls'] or output_tokens+req['max_output_tokens']>max_total_output_tokens or input_tokens>=max_input_tokens:
    raise ValueError('Shared total model budget exhausted')
   if len(json_text(req['messages']).encode('utf-8'))+1024>max_input_tokens-input_tokens:
    raise ValueError('Conservative context byte bound exceeds remaining token budget; no dispatch')
   _unchanged(sources)
   if old is not None and req['call_id'] not in journal.records:raise ValueError('Completed reconstruction would dispatch')
   try:
    if index==0 and reused_outline is not None:
     replay=type('VerifiedOutlineReplay',(),{'identity':provider.identity,'generate':lambda self,request:copy.deepcopy(reused_outline)})()
     response=journal.call(replay,req)
    else:
     response=_dispatch(journal,provider,req,contract)
   finally:
    # A returned truncation or rejected response still spent this call. Advance
    # from immutable dispatch records, not from successful parsing.
    if req['call_id'] in journal.started:index+=1
    record=journal.records.get(req['call_id'])
    usage=(record.get('result') or {}).get('costs',{}) if record else {}
    known=all(type(usage.get(k)) is int and usage[k]>=0 for k in ('input_tokens','output_tokens'))
    if known:
     input_tokens+=usage['input_tokens'];output_tokens+=usage['output_tokens']
   if not known:raise ValueError('Cannot verify total token budget with unknown usage')
   if input_tokens>max_input_tokens or output_tokens>max_total_output_tokens:raise ValueError('Actual token budget exceeded')
   if response.get('stop_reason') not in ('stop','eos'):raise ValueError('Incomplete model output retained')
   return parse_json_response(response['text'])
  state=None
  try:
   req=_outline_request(source['song'],source['options'],source['units'],cfg);req['call_id']=song_id+'/four-method/0000'
   outline=_read_outline(json_text(call(req)),source['song']);save('outline.json',outline);artifacts['outline.json']=file_sha(output/'outline.json')
   state=CompositionState(source,outline);feedback=None;phase='propose'
   while index<cfg['max_calls']:
    if state.index==len(source['units']['units']) and method in ('workflow','self_refine'):stop='finished';break
    req=_request(state,dict(cfg,call_index=index),method,phase,feedback)
    before=object_sha(state.plan);cursor=state.index;decision=None
    try:
     decision=call(req)
     if method in ('react','choreography'):observation=state.execute(decision)
     elif phase=='feedback':
      keys(decision,('feedback',));text(decision['feedback'],'feedback');feedback=decision['feedback'];observation=dict(status='feedback_recorded',quality_verified=False);phase='revise'
     else:
      state.execute(dict(tool='revise_block' if phase=='revise' else 'propose_block',proposal=decision,reason='Fixed workflow stage'))
      if method=='self_refine' and phase=='propose':observation=dict(status='drafted',quality_verified=False);phase='feedback'
      else:observation=state.execute(dict(tool='commit_block',proposal=None,reason='Fixed structural check and commit'));phase='propose';feedback=None
    except (ValueError,TypeError,KeyError) as error:
     observation=dict(status='rejected',error=str(error),quality_verified=False)
     raw=journal.records.get(req['call_id'])
     info=raw.get('result') if raw else None
     if (isinstance(error,_ReplayInfrastructureError) or not raw or not raw.get('success') or not isinstance(info,dict)
         or info.get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response')
         or any(type(info.get('costs',{}).get(k)) is not int or info['costs'][k]<0 for k in ('input_tokens','output_tokens'))):fatal=str(error)
     if method in ('workflow','self_refine'):stop='model_or_structure_failed';fatal=fatal or str(error)
    record=dict(call_id=req['call_id'],source_prefix_sha256=before,committed_units_before=cursor,decision=decision,observation=observation,committed_units_after=state.index)
    name='steps/%04d.json'%len(history);save(name,record);artifacts[name]=file_sha(output/name);history.append(record);state.observations.append(observation)
    if state.index!=cursor:
     name='prefixes/%04d.json'%cursor;save(name,state.plan);artifacts[name]=file_sha(output/name)
    if fatal:break
    if observation.get('status') in ('finished','incomplete_finish'):stop=observation['status'];break
   if state.index==len(source['units']['units']) and method in ('workflow','self_refine'):stop='finished'
  except (ValueError,TypeError,KeyError) as error:fatal=str(error);stop='failed_or_budget_exhausted'
  plan=state.plan if state is not None else _empty_plan(source['original']);full=state is not None and state.index==len(source['units']['units']) and stop=='finished' and fatal is None
  name='plan.json' if full else 'partial-plan.not-success.json';save(name,plan);artifacts[name]=file_sha(output/name)
  _unchanged(sources)
  infrastructure_failed=any(not raw.get('success') or not isinstance(raw.get('result'),dict) or
    raw['result'].get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response') for raw in journal.records.values())
  result=_sealed(dict(schema='choreobench.four_composition_result.v1',complete=not infrastructure_failed,full_song_succeeded=full,method=method,stop_reason=stop,error=fatal,
   committed_units=0 if state is None else state.index,total_units=len(source['units']['units']),calls_used=len(journal.started),input_tokens=input_tokens,output_tokens=output_tokens,
   call_records={k:v['record_sha256'] for k,v in journal.records.items()},contract_sha256=object_sha(contract),artifacts=artifacts,
   formal_E3_completed=False,quality_verified=False,extra_audio_calls=0,
   original_outline_reused=reused_outline is not None,new_model_calls=len(journal.started)-(1 if reused_outline is not None and journal.started else 0),
   reused_outline_calls=1 if reused_outline is not None and journal.started else 0,
   reused_outline_source_costs=None if reused_outline is None else copy.deepcopy(reused_outline['costs']),
   token_totals_include_reused_outline=reused_outline is not None))
  if old is not None and result!=old:raise ValueError('Completed composition reconstruction changed')
  save('results.json',result);return result
