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
from .public_style_prior import load_public_style_prior
from .request_budget import guard_request
from .schema import keys,text,integer,validate_dataset,validate_plan
from .song_agent import load_song_source
from .timing import event_clock
from .untimed_catalog import load_catalog_song

METHODS=('workflow','self_refine','react','choreography','structured','music_structured')
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
 'music_structured':'Treat section_requirement and music_evidence as binding: translate the section intent and the '
 'acoustic measurements into concrete movements. Acoustic repeat/contrast entries are similarity guesses, not detected sections. '
 'A sustained or sparse section is allowed; never add adjectives or padding instead of movements.',
 'structured':'Treat section_requirement and block_requirement as binding design targets derived from the same public evidence. '
 'Commit as soon as the block is structurally valid and carries any required motif; commit immediately when no concrete issue exists. '
 'When an issue_scan lists problems, change only what those problems require. Reasonable motif recurrence is not a defect and must not be rewritten.',
 'choreography':'Track active constraints, already committed phrase development and unresolved draft issues. '
 'Use duration-aware primary movements with concurrent modulation, avoid uncontrolled long-chain copying, '
 'and revise only where feedback or evidence supports a change. Unknown physical state must remain unknown.'}


def _load_style_prior(path,source_path,source,sources):
 # Bind to the original dataset, not a result/plan hash or effective music metadata.
 root=Path(source_path).resolve()
 if root.is_dir():contract_path=root/'contract.json'
 else:
  header=_snapshot(root,sources)
  contract_path=(Path(header['catalog_dir'])/'contract.json' if header.get('schema')=='choreobench.complete_catalog_song.v1'
                 else root.parent/'contract.json')
 contract=_snapshot(contract_path,sources)
 if contract.get('schema') not in ('choreobench.coordination_contract.v1','choreobench.untimed_catalog_contract.v1','choreobench.untimed_catalog_contract.v2'):
  raise ValueError('Style prior requires the original source dataset contract')
 dataset=contract['dataset_snapshot'];validate_dataset(dataset)
 original=next((s for s in dataset['songs'] if s['song_id']==source['original']['song_id']),None)
 if original!=source['original']:raise ValueError('Style prior source song differs from original dataset')
 path=Path(path).expanduser()
 if not path.is_absolute():path=(root if root.is_dir() else root.parent)/path
 prior=load_public_style_prior(path,contract['dataset_sha256'],dataset['songs'])
 if str(prior.path) in sources and sources[str(prior.path)]!=prior.digest:raise ValueError('Style prior source changed')
 sources[str(prior.path)]=prior.digest
 return prior


def _source(path,song_id,cfg,sources):
 path=Path(path).resolve()
 if path.is_dir():return load_song_source(path,song_id,sources)
 header=read_json(path)
 if header.get('schema')=='choreobench.complete_catalog_song.v1':
  from .catalog_song import load_song_view
  original,options,synthetic,digest=load_song_view(path,song_id,sources)
 else:original,options,synthetic,digest=load_catalog_song(path,song_id,sources)
 units=build_planning_units(original,config=cfg['unit_config'])
 if options.get('bar_windows'):
  if cfg.get('bar_structure'):raise ValueError('Bound bar-window candidates cannot use a different bar_structure partition')
  from .planning_units import build_bar_planning_units
  units=build_bar_planning_units(original,options['bar_windows'])
 if cfg.get('bar_structure'):
  from . import bar_structure
  structure=bar_structure.load(cfg['bar_structure'])
  section_id=original['music']['sections'][0]['section_id']
  units=bar_structure.blocks(structure,section_id,original['audio']['num_samples'],original['audio']['sample_rate'],
                             bars_per_block=cfg.get('bars_per_block',4),max_unit_seconds=cfg['unit_config']['max_unit_seconds'])
  units['song_id']=original['song_id']
 return dict(original=original,song=copy.deepcopy(original),options=options,units=units,config=cfg,
             synthetic=synthetic,source_contract_sha256=digest)


OUTLINE_CANDIDATE_CHARS=60

def _shared_request(request,source,cfg,unit=None):
 """Equal public evidence for all strategies; constraints remain strategy-specific."""
 request=copy.deepcopy(request)
 payload=parse_json_response(request['messages'][-1]['content'])
 if source.get('music_summary') is not None:
  summary=source['music_summary']
  from .music_structure_summary import compact
  if unit is None:
   # Outline needs block identity and timing, not every spectral diagnostic.
   payload['music_summary']=dict(
       blocks=[{k:b[k] for k in ('block_id','start_seconds','end_seconds','energy_trend') if k in b}
               for b in summary['blocks']],
       relations=summary['relations'],parameters=summary['parameters'],
       functional_labels_verified=False,beat_or_meter_assumed=False)
  else:
   payload['music_summary']=compact(summary,unit['start'],unit['end'])
 condition=source.get('global_style')
 if condition is not None:
  # The full signal already travels as style_prior; keep one bounded copy so
  # long songs do not duplicate every observation window in the prompt.
  compact_condition={k:condition[k] for k in ('label','header','status','probabilities',
                                               'model_identity','usage') if k in condition}
  if 'aggregation' in condition:
   compact_condition['aggregation']={k:condition['aggregation'][k]
                                     for k in ('method','window_count','coverage_fraction')}
  payload['global_style']=compact_condition
  request['messages'][0]['content']=condition['header']+'\nUse this shared global creative condition; output only the requested JSON, never a style header inside action text.\n'+request['messages'][0]['content']
 if unit is None and isinstance(payload.get('local_proposals'),list):
  kept=[];seen=set()
  for proposal in payload['local_proposals']:
   cid=str(proposal.get('call_id',''))
   parts=cid.split('/')
   window=parts[-2] if len(parts)>=2 else cid
   if window in seen: continue
   seen.add(window); kept.append(proposal)
  payload['local_proposals']=kept
  for proposal in payload['local_proposals']:
   text=proposal.get('text')
   if isinstance(text,str) and len(text)>OUTLINE_CANDIDATE_CHARS:
    proposal['text']=text[:OUTLINE_CANDIDATE_CHARS]+'...'
  if condition is not None and 'style_prior' in payload:
   payload.pop('style_prior')
  payload['outline_context_compression']=dict(version=3,candidate_chars=OUTLINE_CANDIDATE_CHARS,
                                              applied_to='outline_only',
                                              style_prior_compacted_to='global_style')
 request['messages'][-1]['content']=json_text(payload)
 return guard_request(request,cfg)


def _global_condition(prior,song_id,override=None):
 if prior is None and override is None:return None
 signal=prior.signal(song_id) if prior is not None else {}
 if 'header' in signal:condition=copy.deepcopy(signal)
 else:
  p=signal.get('probabilities');labels=signal.get('model_identity',{}).get('class_order',[])
  label=labels[max(range(len(p)),key=lambda i:p[i])] if p else 'unspecified'
  condition=dict(label=label,header='Dance style is '+label+'.',status='available' if p else 'no_valid_coverage',
                 source=signal,scope='global')
 if override is not None:
  import re
  if not isinstance(override,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 /_+()-]{0,79}',override):
   raise ValueError('Explicit user style must be a safe single label')
  condition.update(original_label=condition['label'],label=override,header='Dance style is '+override+'.',status='explicit_user_override')
 return condition


class CompositionState:
 def __init__(self,source,outline):
  self.source=source;self.outline=outline;self.plan=_empty_plan(source['original'])
  self.plan['motifs']=copy.deepcopy(outline['motifs']);self.index=0;self.draft=None;self.observations=[];self.revisions_used=0
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
   if tool=='revise_block' and self.revisions_used>=1:raise ValueError('Shared one-revision-per-unit limit reached; check, commit or finish')
   if tool=='propose_block' and self.draft is not None:raise ValueError('Current draft already exists; use revise_block to change it or commit_block to advance')
   # Invalid draft is visible for repair but never becomes committed history.
   keys(proposal,('unit_id','events'),where='block draft')
   if self.index==len(self.source['units']['units']) or proposal['unit_id']!=self.source['units']['units'][self.index]['unit_id']:
    raise ValueError('Draft targets a different unit')
   unchanged=proposal==self.draft
   if tool=='revise_block':self.revisions_used+=1
   self.draft=copy.deepcopy(proposal)
   return dict(tool=tool,status='unchanged_draft' if unchanged else 'drafted',draft_sha256=object_sha(self.draft),
               revisions_used=self.revisions_used,revision_limit=1,quality_verified=False)
  if self.draft is None:raise ValueError('No current draft')
  events,plan=self.validate(self.draft)
  observation=dict(tool=tool,status='structurally_valid',unit_id=self.draft['unit_id'],quality_verified=False)
  if tool=='commit_block':self.plan=plan;self.index+=1;self.draft=None;self.revisions_used=0;observation['status']='committed'
  return observation




def _outline_text(call, req, song_id, repair=True):
 """Dispatch the outline call; on truncation ask once for a shorter complete JSON."""
 if not repair: return json_text(call(req)), False
 try:
  return json_text(call(req)), False
 except ValueError as error:
  if 'Incomplete model output' not in str(error): raise
  repair=copy.deepcopy(req)
  repair['call_id']=song_id+'/four-method/0000-trunc-repair'
  repair['response_kind']=str(req.get('response_kind'))+'_repair'
  repair['messages']=list(req['messages'])+[dict(role='user',content=
    'Your previous outline was truncated. Return a COMPLETE but shorter JSON: fewer motifs, concise descriptions, same schema. No prose.')]
  return json_text(call(repair)), True


def _snap_schedule(state):
 """Deterministically adopt a permitted complete-unit schedule; returns True on success."""
 from .timing import event_clock
 if not getattr(state,'source',None) or getattr(state,'draft',None) is None: return False
 units=state.source['units']['units']
 if state.index>=len(units): return False
 unit=units[state.index]
 clock=event_clock(unit,state.plan['sample_rate'],state.source['config'])
 schedules=clock.get('permitted_schedules') or []
 events=(state.draft.get('events') or [])
 if not schedules or not events: return False
 feasible=[s for s in schedules if len(s)<=len(events)] or [min(schedules,key=len)]
 schedule=min(feasible,key=lambda s:(abs(len(s)-len(events)),len(s)))
 groups=[[] for _ in schedule]
 for i,event in enumerate(events): groups[i%len(schedule)].append(event)
 rebuilt=[]
 for (left,right),group in zip(schedule,groups):
  if not group: return False
  action=' '.join(dict.fromkeys(g['action'] for g in group))
  sources=[]
  for g in group: sources.extend(g.get('source_call_ids') or [])
  rebuilt.append(dict(start_boundary=left,end_boundary=right,track=group[0].get('track'),
                      action=action,motif_id=next((g.get('motif_id') for g in group if g.get('motif_id')),None),
                      source_call_ids=list(dict.fromkeys(sources))))
 snapped=copy.deepcopy(state.draft);snapped['events']=rebuilt
 try: state.validate(snapped)
 except Exception: return False
 state.draft=snapped
 return True

def _fallback_unit(state):
 """Commit one grounded event for the current unit after repeated model failure.

 Only fires when the planner produced no committable draft twice. The action is
 the first cited local caption for the unit and the schedule is a permitting
 complete-unit schedule, so the fallback stays inside the frozen evidence.
 """
 from .timing import event_clock
 from .coordination import LIMITS, _candidates
 if not getattr(state,'source',None): return False
 units=state.source['units']['units']
 if state.index>=len(units): return False
 unit=units[state.index]
 clock=event_clock(unit,state.plan['sample_rate'],state.source['config'])
 schedules=clock.get('permitted_schedules') or []
 if not schedules: return False
 schedule=min(schedules,key=len)
 candidates=_candidates(state.source['options'],unit)
 if not candidates: return False
 motif_ids=next(s['motif_ids'] for s in state.outline['sections'] if s['section_id']==unit['section_id'])
 events=[]
 for i,frame in enumerate(schedule):
  candidate=candidates[i%len(candidates)]
  events.append(dict(start_boundary=frame['start_boundary'],end_boundary=frame['end_boundary'],
    track='body',action=candidate['text'][:LIMITS['action_characters']],
    motif_id=motif_ids[0] if motif_ids else None,source_call_ids=[candidate['call_id']]))
 draft=dict(unit_id=unit['unit_id'],events=events)
 try:
  events_out,plan=state.validate(draft)
 except Exception:
  return False
 state.plan=plan;state.index+=1;state.draft=None;state.revisions_used=0
 state.observations.append(dict(tool='fallback_block',status='committed',unit_id=unit['unit_id'],quality_verified=False))
 return True


def _repair_outline(call, req, raw_text, error, cfg, song_id):
    """One bounded, fully journaled outline repair; never a silent retry."""
    from .schema import text as _text
    repair = copy.deepcopy(req)
    repair['call_id'] = song_id + '/four-method/outline-repair'
    repair['response_kind'] = str(req.get('response_kind')) + '_repair'
    repair['messages'] = list(req['messages']) + [
        dict(role='assistant', content=raw_text),
        dict(role='user', content=('Your previous answer failed validation: ' + str(error) +
                                   ' Return a corrected JSON object that follows the required schema exactly. '
                                   'Only output the corrected JSON.'))]
    from .request_budget import guard_request
    return guard_request(repair, cfg)

def _request(state,cfg,method,phase,feedback=None):
 bounded=cfg.get('controller_version')=='v2'
 if bounded:
  from .composition_controller_v2 import TOOL_PROMPT as BOUNDED_TOOL_PROMPT,observation_context,choreography_context
 units=state.source['units']['units'];unit=units[state.index] if state.index<len(units) else None
 if unit is not None and method=='music_structured':
  from .composition_music import unit_request as music_unit_request
  req=music_unit_request(state,unit,state.source['music_summary'],cfg)
  payload=parse_json_response(req['messages'][-1]['content']);schema=req['response_schema'];system=req['messages'][0]['content']
 elif unit is not None and method=='structured':
  from .composition_structured import unit_request as structured_unit_request
  req=structured_unit_request(state,unit,cfg)
  payload=parse_json_response(req['messages'][-1]['content']);schema=req['response_schema'];system=req['messages'][0]['content']
 elif unit is not None:
  req=_unit_request(state.source['song'],state.source['options'],state.source['units'],unit,state.outline,state.plan,state.source['config'],state.source.get('style_prior'))
  payload=parse_json_response(req['messages'][-1]['content']);schema=req['response_schema'];system=req['messages'][0]['content']
 else:payload={};schema=dict(type='null');system='All units are committed.'
 if unit is None:
  prior=state.source.get('style_prior')
  if prior is not None:payload['style_prior']=prior.signal(state.source['original']['song_id'])
 payload.update(committed_plan=copy.deepcopy(state.plan),current_draft=copy.deepcopy(state.draft),
    committed_units=state.index,total_units=len(units),tool_observations=state.observations[-2:],feedback=feedback,
    shared_permissions=True,actual_audio_access=False,remaining_model_calls=cfg['max_calls']-cfg['call_index'] if 'max_calls' in cfg else None,
    draft_revision_limit=1,draft_revisions_used=state.revisions_used)
 if bounded:
  payload['tool_observations']=observation_context(state.observations)
  payload.pop('recent_committed_events',None)  # Already present in committed_plan.
  payload.update(controller_version='v2',remaining_input_tokens=cfg['remaining_input_tokens'],
    remaining_output_tokens=cfg['remaining_output_tokens'],
    reserved_commit_calls=len(units)-state.index,no_progress_streak=state.no_progress_streak,
    controller_completion='All units committed finishes automatically; no model finish call required.')
  if method in ('choreography','structured'):payload['choreography_state']=choreography_context(state)
 if phase=='feedback':
  system+=' Give short source-grounded feedback about current draft action load, development and boundary clarity. Never invent heard audio or physical state. Return only {feedback}.'
  schema=dict(type='object',additionalProperties=False,required=['feedback'],properties=dict(feedback=dict(type='string',minLength=1,maxLength=2400)))
 elif method in ('react','choreography','structured'):
  if bounded:tools=state.available_tools()
  else:
   tools=['read_evidence','read_history','finish']
   if unit is not None:
    if state.draft is None:tools+=['propose_block']
    else:
     tools+=['check_block','commit_block']
     if state.revisions_used<1:tools+=['revise_block']
  payload['available_tools']=tools
  payload['draft_state']='absent' if state.draft is None else 'uncommitted'
  if bounded:
   payload['block_contract']=payload.pop('output_contract',{})
   payload['output_contract']=dict(fields=['tool','proposal','reason'],
      proposal_for_tools=['propose_block','revise_block','submit_block'],proposal_for_other_tools=None)
  system+=' '+(BOUNDED_TOOL_PROMPT if bounded else TOOL_PROMPT)+POLICY[method]
  if not bounded:system+=' Creating or revising a draft does not advance the song. Every method has at most ONE revision attempt per current unit, including unchanged revisions. Once that attempt is spent, check or commit the current draft, or finish with an incomplete outcome. A draft you call ready should be committed with proposal=null, not re-emitted. Use only available_tools. Completion requires actual commits, not descriptions of readiness.'
  branches=[]
  for tool in tools:
   branches.append(dict(type='object',additionalProperties=False,required=['tool','proposal','reason'],properties=dict(
    tool=dict(type='string',enum=[tool]),proposal=copy.deepcopy(schema) if tool in ('propose_block','revise_block','submit_block') else dict(type='null'),
    reason=dict(type='string',minLength=1,maxLength=1200))))
  schema=dict(type='object',additionalProperties=False,required=['tool','proposal','reason'],properties=dict(
   tool=dict(type='string',enum=tools),proposal=dict(anyOf=[dict(type='null'),schema]),
   reason=dict(type='string',minLength=1,maxLength=1200)),anyOf=branches)
 elif phase=='revise':system+=' Revise the same current-unit draft using supplied feedback. Keep the original output contract and permissions.'
 request=dict(call_id=state.plan['song_id']+'/four-method/%04d'%cfg['call_index'],seed=cfg['seed'],max_output_tokens=cfg['max_output_tokens'],
  response_kind='composition_'+phase,messages=[dict(role='system',content=system),dict(role='user',content=json_text(payload))],response_schema=schema)
 return _shared_request(request,state.source,cfg,unit)


def run_composition(source_path,song_id,provider_spec,out_dir,*,method='workflow',seed=42,max_output_tokens=1024,
                    outline_max_output_tokens=4096,max_request_characters=200000,max_input_tokens=2000000,max_total_output_tokens=150000,resume=False,outline_reuse_run=None,controller_version='v1',music_summary_path=None,
                    outline_proposal_char_limit=None,outline_candidates_per_window=None,acoustic_guidance=False,bar_structure=None,bars_per_block=4,motif_quota=True,motif_variation=False,style_prior=None,rewrite_repeats=False,repeat_groups=3,global_style_override=None,repair_blocks=True,max_calls_cap=None,snap_schedule=False,continuity_prompt=False,fallback_unit=False):
 if method not in METHODS:raise ValueError('Unknown composition method')
 if controller_version not in ('v1','v2'):raise ValueError('Unknown composition controller version')
 if method=='structured' and controller_version!='v2':
  raise ValueError('The structured method requires the bounded v2 controller')
 if method=='music_structured' and music_summary_path is None:
  raise ValueError('The music_structured method requires music_summary_path')
 if rewrite_repeats and method not in ('workflow','music_structured'):
  raise ValueError('Repeat rewriting is supported for matched workflow/music_structured ablations')
 integer(repeat_groups,'repeat_groups',0)
 bounded=controller_version=='v2'
 for name,value in [('seed',seed),('max_output_tokens',max_output_tokens),('outline_max_output_tokens',outline_max_output_tokens),
                    ('max_request_characters',max_request_characters),('max_input_tokens',max_input_tokens),('max_total_output_tokens',max_total_output_tokens)]:
  integer(value,name,0 if name=='seed' else 1)
 cfg=dict(seed=seed,max_output_tokens=max_output_tokens,outline_max_output_tokens=outline_max_output_tokens,
  max_request_characters=max_request_characters,max_events_per_unit=3,structured_unit_output=True,use_outline=True,use_history=True,draft_revision_limit=1,
  event_time_format='boundary_ids',event_grid_seconds=.25,min_event_seconds=.5,event_schedule_policy='balanced_full_unit',
  unit_config=dict(bars_per_block=4,fallback_seconds=10.125,max_unit_seconds=20,meter_status='unverified',min_time_block_seconds=2),
  outline_proposal_char_limit=outline_proposal_char_limit,
  outline_candidates_per_window=outline_candidates_per_window,acoustic_guidance=acoustic_guidance,
  bar_structure=bar_structure,bars_per_block=bars_per_block,motif_quota=motif_quota,motif_variation=motif_variation,
   rewrite_repeats=rewrite_repeats,repeat_groups=repeat_groups,global_style_override=global_style_override,repair_blocks=repair_blocks,snap_schedule=snap_schedule,continuity_prompt=continuity_prompt,fallback_unit=fallback_unit)
 if bounded:
  from .composition_controller_v2 import CompositionStateV2,NO_PROGRESS_LIMIT
  cfg.update(controller_version='v2',no_progress_limit=NO_PROGRESS_LIMIT,
    controller_policy='bounded-revision-reserved-commit-auto-finish-v2')
 sources={};source=_source(source_path,song_id,cfg,sources);cfg['max_calls']=1+4*len(source['units']['units'])
 if max_calls_cap is not None:cfg['max_calls']=min(cfg['max_calls'],max_calls_cap)
 if style_prior is not None:source['style_prior']=_load_style_prior(style_prior,source_path,source,sources)
 source['global_style']=_global_condition(source.get('style_prior'),song_id,global_style_override)
 if music_summary_path is not None:
  from .music_structure_summary import load as load_music_summary
  source['music_summary']=load_music_summary(music_summary_path)
  summary=_snapshot(Path(music_summary_path).resolve(),sources)
  audio=source['original']['audio'];bound=summary.get('audio',{})
  if any(bound.get(k)!=audio[k] for k in ('sha256','sample_rate','num_samples')):
   raise ValueError('Public acoustic summary must bind this exact song waveform and clock')
 # Historical input cfg can contain a different provider; only timing/structure is used from it.
 source['config']=dict(source['config'],max_output_tokens=max_output_tokens,max_request_characters=max_request_characters,structured_unit_output=True,use_outline=True,use_history=True)
 if provider_spec.get('kind') not in ('chat_completions','replay'):raise ValueError('Composition requires explicit real/replay text provider')
 provider=create_provider(provider_spec,'planner',Path(source_path).resolve().parent)
 reused_outline=None
 if outline_reuse_run is not None and method=='structured':
  raise ValueError('A reused legacy outline has a different schema; run the structured method in a new directory')
 if outline_reuse_run is not None:
  reuse=Path(outline_reuse_run).resolve();prior=_snapshot(reuse/'contract.json',sources)
  prior_result=_verify(_snapshot(reuse/'results.json',sources))
  if (prior['method']!=method or prior['provider_identity']!=provider.identity or not prior_result['complete']
      or prior_result['contract_sha256']!=object_sha(prior)):
   raise ValueError('Outline reuse method/model or complete result differs')
  old_journal=_Journal(reuse,object_sha(prior),cfg['max_calls']);cid=song_id+'/four-method/0000'
  if cid not in old_journal.records:raise ValueError('Original outline raw missing; no replacement')
  expected=_shared_request(_outline_request(source['song'],source['options'],source['units'],cfg,source.get('style_prior')),source,cfg);expected['call_id']=cid
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
 if source.get('style_prior') is not None:
  contract.update(source['style_prior'].binding(),style_prior_usage='optional recommendation; never a hard genre constraint')
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
  state=None;outline=None
  repaired_units={}
  try:
   outline_repair=False
   if method=='music_structured':
    from .composition_music import outline_request as music_outline_request, read_outline as read_music_outline
    from .baseline import ModelOutputError as _MOE
    _need=not cfg.get('acoustic_guidance')
    req=music_outline_request(source['song'],source['options'],source['units'],source['music_summary'],cfg,_need,source.get('style_prior'));req['call_id']=song_id+'/four-method/0000'
    req=_shared_request(req,source,cfg)
    raw,truncated=_outline_text(call,req,song_id,cfg.get('repair_blocks',True))
    outline_repair=outline_repair or truncated
    try:
     outline=read_music_outline(raw,source['song'],source['music_summary'],source['units'],_need)
    except _MOE as error:
     repair=_repair_outline(call,req,raw,error,cfg,song_id);outline_repair=True
     outline=read_music_outline(json_text(call(repair)),source['song'],source['music_summary'],source['units'],_need)
   elif method=='structured':
    from .composition_structured import outline_request as structured_outline_request, read_outline as read_structured_outline, StructuredState
    from .baseline import ModelOutputError as _MOE
    req=structured_outline_request(source['song'],source['options'],source['units'],cfg,source.get('style_prior'));req['call_id']=song_id+'/four-method/0000'
    req=_shared_request(req,source,cfg)
    raw,truncated=_outline_text(call,req,song_id,cfg.get('repair_blocks',True))
    outline_repair=outline_repair or truncated
    try:
     outline=read_structured_outline(raw,source['song'],source['options'])
    except _MOE as error:
     repair=_repair_outline(call,req,raw,error,cfg,song_id);outline_repair=True
     outline=read_structured_outline(json_text(call(repair)),source['song'],source['options'])
   else:
    from .baseline import ModelOutputError as _MOE
    req=_outline_request(source['song'],source['options'],source['units'],cfg,source.get('style_prior'));req['call_id']=song_id+'/four-method/0000'
    req=_shared_request(req,source,cfg)
    raw,truncated=_outline_text(call,req,song_id,cfg.get('repair_blocks',True))
    outline_repair=outline_repair or truncated
    try:
     outline=_read_outline(raw,source['song'])
    except _MOE as error:
     repair=_repair_outline(call,req,raw,error,cfg,song_id);outline_repair=True
     outline=_read_outline(json_text(call(repair)),source['song'])
   save('outline.json',outline);artifacts['outline.json']=file_sha(output/'outline.json')
   if outline_repair:save('outline_repair.json',dict(used=True,call_id=song_id+'/four-method/outline-repair'))
   state=StructuredState(source,outline) if method=='structured' else (CompositionStateV2 if bounded else CompositionState)(source,outline)
   feedback=None;phase='propose'
   while index<cfg['max_calls']:
    if state.index==len(source['units']['units']) and (bounded or method in ('workflow','self_refine','music_structured')):stop='finished';break
    request_cfg=dict(cfg,call_index=index)
    if bounded:
     request_cfg.update(remaining_input_tokens=max(0,max_input_tokens-input_tokens),
                        remaining_output_tokens=max(0,max_total_output_tokens-output_tokens))
     state.remaining_calls=cfg['max_calls']-index if method in ('react','choreography','structured') else None
     if method in ('react','choreography','structured') and not state.available_tools():stop='unrepairable_draft';break
     progress_before=state.progress_signature()
    req=_request(state,request_cfg,method,phase,feedback)
    before=object_sha(state.plan);cursor=state.index;decision=None
    def apply_decision(value):
     nonlocal phase,feedback
     if method in ('react','choreography','structured'):return state.execute(value)
     if phase=='feedback':
      keys(value,('feedback',));text(value['feedback'],'feedback');feedback=value['feedback'];phase='revise'
      return dict(status='feedback_recorded',quality_verified=False)
     draft_observation=state.execute(dict(tool='revise_block' if phase=='revise' else 'propose_block',proposal=value,reason='Fixed workflow stage'))
     if method=='self_refine' and phase=='propose':
      phase='feedback';return dict(status='drafted',quality_verified=False)
     if bounded and draft_observation.get('status')=='committed':
      phase='propose';feedback=None;return draft_observation
     if bounded and getattr(state,'draft',None) is not None and hasattr(state,'validate'):
      try:
       state.validate(state.draft)
      except (ValueError,TypeError,KeyError):
       if not (cfg.get('snap_schedule') and _snap_schedule(state)): raise
     observation_commit=state.execute(dict(tool='commit_block',proposal=None,reason='Fixed structural check and commit'))
     phase='propose';feedback=None
     return observation_commit
    def repair_request(error):
     source_text=(info or {}).get('text') or ''
     repair=copy.deepcopy(req)
     repair['call_id']=req['call_id']+'-block-repair'
     repair['response_kind']=str(req.get('response_kind'))+'_repair'
     was_truncated=isinstance(info,dict) and info.get('stop_reason')=='budget_truncation'
     if was_truncated: repair['max_output_tokens']=8192
     instruction=('Your previous response was truncated. Return a COMPLETE but SHORTER JSON object for the same request: '
                  'at most 2 events with at most 15 words each, and copy the permitted schedule exactly. No prose.'
                  if was_truncated else
                  'Correction required: '+str(error)+' Return only one corrected JSON object for the same request, with no prose.')
     repair['messages']=list(req['messages'])+[dict(role='assistant',content=source_text),
        dict(role='user',content=instruction)]
     return repair
    try:
     decision=call(req)
     observation=apply_decision(decision)
    except (ValueError,TypeError,KeyError) as error:
     observation=dict(status='rejected',error=str(error),quality_verified=False)
     raw=journal.records.get(req['call_id'])
     info=raw.get('result') if raw else None
     infrastructure=(isinstance(error,_ReplayInfrastructureError) or not raw or not raw.get('success') or not isinstance(info,dict)
         or info.get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response')
         or any(type(info.get('costs',{}).get(k)) is not int or info['costs'][k]<0 for k in ('input_tokens','output_tokens')))
     repaired=False
     truncated=isinstance(info,dict) and info.get('stop_reason')=='budget_truncation'
     if (cfg.get('repair_blocks',True) and not (infrastructure and not truncated) and method in ('workflow','self_refine','music_structured')
         and repaired_units.get(cursor,0)<2 and isinstance(info,dict)
         and info.get('stop_reason') in ('stop','eos','budget_truncation')):
      repaired_units[cursor]=repaired_units.get(cursor,0)+1
      if method in ('workflow','self_refine','music_structured') and state is not None:
       if hasattr(state,'draft'):state.draft=None
       if hasattr(state,'revisions_used'):state.revisions_used=0
      try:
       decision=call(repair_request(error))
       observation=apply_decision(decision)
       repaired=True;fatal=None
      except (ValueError,TypeError,KeyError) as second:
       observation=dict(status='rejected',error=str(second),quality_verified=False)
       error=second
       raw=journal.records.get(req['call_id'])
       info=raw.get('result') if raw else None
       if cfg.get('snap_schedule') and state is not None and _snap_schedule(state):
        try:
         observation=state.execute(dict(tool='commit_block',proposal=None,
                                        reason='Deterministic permitted-schedule adoption after repair failure'))
         repaired=True;fatal=None
        except (ValueError,TypeError,KeyError):pass
     if not repaired and cfg.get('fallback_unit') and method=='music_structured' and state is not None:
      if _fallback_unit(state):
       observation=state.observations[-1];repaired=True;fatal=None;stop=None
     if not repaired:
      if infrastructure:fatal=str(error)
      if method in ('workflow','self_refine','music_structured'):stop='model_or_structure_failed';fatal=fatal or str(error)
      if bounded and (input_tokens>max_input_tokens or output_tokens>max_total_output_tokens):
       fatal=fatal or str(error);stop='failed_or_budget_exhausted'
    controller_action=None
    if bounded and method in ('react','choreography','structured') and not fatal:
     controller_action=state.after_decision(progress_before)
     if controller_action and controller_action['status']=='stalled_no_content_progress':stop='stalled_no_content_progress'
    record=dict(call_id=req['call_id'],source_prefix_sha256=before,committed_units_before=cursor,decision=decision,observation=observation,committed_units_after=state.index)
    if controller_action:record['controller_actions']=[controller_action]
    name='steps/%04d.json'%len(history);save(name,record);artifacts[name]=file_sha(output/name);history.append(record);state.observations.append(observation)
    if controller_action:state.observations.append(controller_action)
    if state.index!=cursor:
     name='prefixes/%04d.json'%cursor;save(name,state.plan);artifacts[name]=file_sha(output/name)
    if fatal:break
    if controller_action and controller_action['status']=='stalled_no_content_progress':break
    if observation.get('status') in ('finished','incomplete_finish'):stop=observation['status'];break
   if state.index==len(source['units']['units']) and (bounded or method in ('workflow','self_refine','music_structured')) and not fatal:stop='finished'
  except (ValueError,TypeError,KeyError) as error:fatal=str(error);stop='failed_or_budget_exhausted'
  repeat_stats=None
  if (cfg.get('rewrite_repeats') and method in ('workflow','music_structured') and state is not None
      and state.index==len(source['units']['units']) and fatal is None):
   from .repeat_revision import duplicate_groups,rewrite_request,read_rewrites,apply_rewrites,summary as repeat_summary
   original=copy.deepcopy(state.plan);groups=duplicate_groups(state.plan)[:cfg.get('repeat_groups',3)]
   before_costs=dict(calls_used=len(journal.started),input_tokens=input_tokens,output_tokens=output_tokens,
                      full_song_succeeded=True,max_calls=cfg['max_calls'],max_input_tokens=max_input_tokens,
                      max_total_output_tokens=max_total_output_tokens)
   save('before_rewrite.json',original);artifacts['before_rewrite.json']=file_sha(output/'before_rewrite.json')
   save('before_rewrite_costs.json',before_costs);artifacts['before_rewrite_costs.json']=file_sha(output/'before_rewrite_costs.json')
   rewrite_call_start=len(journal.started)
   applied=[];failures=[]
   for group_index,group in enumerate(groups):
    if index>=cfg['max_calls']:break
    try:
     request=guard_request(rewrite_request(state.plan,group,dict(cfg,call_index=index),
       song_id+'/four-method/%04d'%index),dict(cfg,call_index=index))
     decision=call(request)
     rewrites,skipped=read_rewrites(decision,group)
     if not rewrites:
      failures.append(dict(group=group['event_ids'],error='no acceptable rewrite',skipped=skipped))
      if len(failures)>=2:break
      continue
     final,changed=apply_rewrites(state.plan,rewrites)
     name='steps/%04d.json'%len(history)
     record=dict(call_id=request['call_id'],kind='repeat_rewrite',decision=decision,
                  observation=dict(status='rewritten',event_ids=changed,quality_verified=False))
     save(name,record)
     artifacts[name]=file_sha(output/name)
     history.append(record)
     state.plan=final;applied.append(dict(event_ids=changed,group=group['event_ids'],skipped=skipped))
    except (ValueError,TypeError,KeyError) as error:
     failures.append(dict(group=group['event_ids'],error=str(error)))
     if len(failures)>=2:break
   repeat_stats=dict(groups=len(groups),applied=len(applied),failed=failures,
      calls_used=len(journal.started)-rewrite_call_start,**repeat_summary(original,state.plan))
   save('final_plan.json',state.plan);artifacts['final_plan.json']=file_sha(output/'final_plan.json')
  plan=state.plan if state is not None else _empty_plan(source['original']);full=state is not None and state.index==len(source['units']['units']) and stop=='finished' and fatal is None
  name='plan.json' if full else 'partial-plan.not-success.json';save(name,plan);artifacts[name]=file_sha(output/name)
  if full:
   from .global_style import presentation
   if source.get('global_style') is not None:
    save('global_style.json',source['global_style']);artifacts['global_style.json']=file_sha(output/'global_style.json')
   for filename,content in [('presentation.txt',presentation(plan,source.get('global_style')))]+(
       [('before_rewrite_presentation.txt',presentation(original,source.get('global_style')))] if repeat_stats is not None else []):
    path=output/filename
    if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('Immutable presentation changed')
    if not path.exists():path.write_text(content,encoding='utf-8')
    artifacts[filename]=file_sha(path)
  _unchanged(sources)
  infrastructure_failed=any(not raw.get('success') or not isinstance(raw.get('result'),dict) or
    raw['result'].get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response') for raw in journal.records.values())
  extra=dict()
  if bounded:
   extra=dict(controller_version='v2',controller_statistics=dict(
      automatic_commits=0 if state is None else state.automatic_commits,
      no_progress_steps=0 if state is None else state.no_progress_steps,
      no_progress_limit=NO_PROGRESS_LIMIT),
      token_projection_ratio=(round((input_tokens/len(journal.started)*cfg['max_calls'])/max_input_tokens,3)
                              if journal.started else None))
   if method=='music_structured':
    from .composition_music import contribution as music_contribution
    if outline is not None:
     extra['music_structure']=music_contribution(outline)
   if method=='structured' and state is not None:
    extra['structured_statistics']=dict(
      blocks_with_detected_issues=state.blocks_with_issues,
      revision_calls=state.revision_calls,
      auto_commits_without_issue=state.automatic_commits,
      projected_input_tokens_at_current_rate=(round(input_tokens/len(journal.started)*cfg['max_calls'])
                                              if journal.started else 0),
      max_input_tokens=max_input_tokens,budget_unchanged=True)
  if repeat_stats is not None:extra['repeat_rewrite']=repeat_stats
  result=_sealed(dict(schema='choreobench.four_composition_result.v1',complete=not infrastructure_failed,full_song_succeeded=full,method=method,stop_reason=stop,error=fatal,
   committed_units=0 if state is None else state.index,total_units=len(source['units']['units']),calls_used=len(journal.started),input_tokens=input_tokens,output_tokens=output_tokens,
   call_records={k:v['record_sha256'] for k,v in journal.records.items()},contract_sha256=object_sha(contract),artifacts=artifacts,
   formal_E3_completed=False,quality_verified=False,extra_audio_calls=0,
   original_outline_reused=reused_outline is not None,new_model_calls=len(journal.started)-(1 if reused_outline is not None and journal.started else 0),
   reused_outline_calls=1 if reused_outline is not None and journal.started else 0,
   reused_outline_source_costs=None if reused_outline is None else copy.deepcopy(reused_outline['costs']),
   token_totals_include_reused_outline=reused_outline is not None,**extra))
  if old is not None and result!=old:raise ValueError('Completed composition reconstruction changed')
  save('results.json',result);return result
