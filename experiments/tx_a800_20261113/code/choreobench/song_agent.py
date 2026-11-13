"""Causal whole-song tool policies sharing evidence, permissions and budgets.

A historical outline/candidate catalog is fixed for a development comparison;
no historical event prefix is borrowed. Tool checks never score choreography.
"""
import argparse
import copy
import hashlib
from pathlib import Path

from .baseline import _Journal, _empty_plan, _sealed, _verify
from .coordination import _candidates, _unit_request, _read_unit
from .evaluation import evaluate_plan
from .io import file_lock, file_sha, json_text, object_sha, read_json, write_json
from .memory_replay import _snapshot, _unchanged, _dispatch, _ReplayInfrastructureError, _accounting
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .revision_protocol import public_plan
from .schema import identifier, integer, keys, text, validate_dataset, validate_plan, validate_song
from .timing import event_clock, literal_source_copies


POLICIES = {
    'generic': 'Act as a general-purpose tool-using planner. Fulfill the complete music-to-choreography task, '
               'using available context and tools, checking important constraints and stopping when complete. '
               'Use an appropriate global organization, clear actions, continuity and musical contrast.',
    'choreography': 'Use a choreography-specific content policy: give each event one continuing primary '
                    'movement with concurrent modulation rather than a long chain of completed transitions. '
                    'Use actual duration to limit movement load, develop recognizable phrases without copying '
                    'past action sentences, and plan contrast, purposeful recurrence and a settled ending. '
                    'Inspect or check only when it can change a decision; commit the smallest coherent phrase.'
}
TOOLS = (
    'You control the causal whole-song composition through shared tools. '
    'Both policies have the same evidence, full committed plan, outline, schedules and tools. '
    'inspect_context returns the complete frozen candidate catalog and planning-unit map, without changing history. '
    'check_block validates a proposal for the current unit without committing it. '
    'commit_block validates and atomically appends a proposal for the current unit only. '
    'finish succeeds only after every unit is committed; early finish ends as an incomplete outcome. '
    'Each check/commit proposal follows the current block output contract below. '
    'Replace the earlier block-only root instruction with exactly {tool,proposal,reason}; '
    'proposal is {unit_id,events} for check/commit, null for inspect/finish. '
    'No tool supplies an independent quality rating. Only committed events enter future state. '
    'You may use fewer than the allowed calls; budget is a ceiling, not a target. '
)


def load_song_source(source_run, song_id, sources):
    root = Path(source_run).resolve(strict=True)
    contract = _snapshot(root/'contract.json', sources)
    result = _verify(_snapshot(root/'results.json', sources))
    if (contract.get('schema')!='choreobench.coordination_contract.v1' or not result['complete']
            or result['contract_sha256']!=object_sha(contract) or song_id not in contract['selected_songs']):
        raise ValueError('Song agent requires a bound completed coordination source')
    validate_dataset(contract['dataset_snapshot'])
    original = next(s for s in contract['dataset_snapshot']['songs'] if s['song_id']==song_id)
    effective = copy.deepcopy(original)
    structure = contract.get('public_structure_snapshot')
    if structure is not None:
        if structure['dataset_sha256']!=contract['dataset_sha256']:
            raise ValueError('Public structure differs from source dataset')
        row = next(s for s in structure['songs'] if s['song_id']==song_id)
        if row['audio_sha256']!=original['audio']['sha256'] or any(row[k]!=original['audio'][k] for k in ('sample_rate','num_samples')):
            raise ValueError('Public structure differs from canonical audio')
        effective['music']=copy.deepcopy(row['music'])
    validate_song(effective)
    if object_sha(effective)!=contract['planning_songs_sha256'][song_id]:
        raise ValueError('Effective song differs from source contract')
    refs={r['path']:r['sha256'] for r in result['artifacts']}
    prefix=hashlib.sha256(song_id.encode()).hexdigest()
    def artifact(relative):
        p=(root/relative).resolve(strict=True)
        if not p.is_relative_to(root) or relative not in refs:
            raise ValueError('Unbound or escaped song source artifact')
        return _snapshot(p,sources,refs[relative])
    units=artifact('units/'+prefix+'.json')
    options=artifact('options/'+prefix+'.json')
    outline=artifact('outlines/'+prefix+'.json')
    cfg=copy.deepcopy(contract['config'])
    cfg.update(use_outline=True,use_history=True)
    if not units['units'] or units['units'][0]['start']!=0 or units['units'][-1]['end']!=original['audio']['num_samples']:
        raise ValueError('Planning units do not cover the canonical song')
    if any(a['end']!=b['start'] for a,b in zip(units['units'],units['units'][1:])):
        raise ValueError('Planning units are not a causal contiguous partition')
    _unchanged(sources)
    return dict(original=original,song=effective,units=units,options=options,outline=outline,config=cfg,
                source_contract_sha256=object_sha(contract),synthetic=contract['synthetic'])


def _request(source, plan, index, observations, cfg, policy):
    units=source['units']['units']
    unit=units[index] if index<len(units) else None
    if unit is None:
        payload=dict(song_id=plan['song_id'],all_units_committed=True)
        system='All song units have been committed. You may inspect context or finish.'
        proposal_schema=dict(type='null')
    else:
        base=_unit_request(source['song'],source['options'],source['units'],unit,source['outline'],plan,source['config'])
        payload=parse_json_response(base['messages'][-1]['content'])
        system=base['messages'][0]['content']
        proposal_schema=base.get('response_schema')
        if proposal_schema is None:raise ValueError('Song tool comparison requires structured unit output')
    visible_plan=public_plan(plan) if plan['events'] else dict(copy.deepcopy(plan),provenance={})
    payload.update(committed_plan=visible_plan,committed_units=index,total_units=len(units),
                   tool_observations=observations[-2:],remaining_model_calls=cfg['max_calls']-cfg['call_index'],
                   independently_verified_quality=False)
    if cfg.get('history_mode','full')=='metadata_only':
        for event in payload['committed_plan']['events']:event.pop('action')
        for event in payload.get('recent_committed_events',[]):event.pop('action')
        history=payload.get('boundary_context')
        if history is not None:
            for part in history['body_parts'].values():
                if part['latest_event'] is not None:part['latest_event'].pop('action')
            for motif in history['motif_memory']:
                for event in motif['recent_examples']:event.pop('action')
        payload['history_representation']='committed timing/scope/motif metadata; prior action text withheld'
        system += (' This is an explicit history-text ablation: all prior action sentences are withheld, '
                   'not absent movement. Prior support, pose, direction and semantic continuity remain unknown. '
                   'The authoritative committed plan is preserved outside the model. Do not infer its hidden action '
                   'content from motif counts, timing or the shared future design. Current local proposals remain visible.')
    # Tool observations are bounded by count, not silently stripped of their contents.
    request=dict(call_id=plan['song_id']+'/song-agent/%03d'%cfg['call_index'],seed=cfg['seed'],
        max_output_tokens=cfg['max_output_tokens'],response_kind='whole_song_tool',
        messages=[dict(role='system',content=system+' '+TOOLS+' '+POLICIES[policy]),
                  dict(role='user',content=json_text(payload))],
        response_schema=dict(type='object',additionalProperties=False,required=['tool','proposal','reason'],properties=dict(
            tool=dict(type='string',enum=['inspect_context','check_block','commit_block','finish']),
            proposal=dict(anyOf=[dict(type='null'),proposal_schema]) if unit is not None else proposal_schema,
            reason=dict(type='string',minLength=1,maxLength=1200))))
    return guard_request(request,cfg)


def _execute(source, plan, index, decision):
    keys(decision,('tool','proposal','reason'),where='whole-song tool decision')
    text(decision['reason'],'tool reason')
    if len(decision['reason'])>1200:raise ValueError('Tool reason exceeds budget')
    tool=decision['tool'];units=source['units']['units']
    if tool not in ('inspect_context','check_block','commit_block','finish'):
        raise ValueError('Unknown whole-song tool')
    if tool in ('inspect_context','finish'):
        if decision['proposal'] is not None:raise ValueError('Read/finish tool cannot contain a proposal')
        if tool=='finish':
            return dict(tool=tool,status='finished' if index==len(units) else 'incomplete_finish'),plan,index,True
        return dict(tool=tool,status='observed',planning_units=source['units'],
                    all_local_proposals=_candidates(source['options']),whole_song_design=source['outline']),plan,index,False
    if index==len(units):raise ValueError('No uncommitted unit remains')
    unit=units[index]
    events=_read_unit(json_text(decision['proposal']),source['original'],unit,source['outline'],source['options'],
                      source['config']['max_events_per_unit'],event_clock(unit,plan['sample_rate'],source['config']))
    proposed=copy.deepcopy(plan);proposed['events'].extend(events)
    validate_plan(proposed,source['original'])
    observation=dict(tool=tool,status='structurally_valid',unit_id=unit['unit_id'],
                     candidate_prefix_sha256=object_sha(proposed),quality_verified=False)
    if tool=='commit_block':return observation,proposed,index+1,False
    return observation,plan,index,False


def run_song_agent(source_run,song_id,provider_spec,output_dir,*,policy='generic',seed=7,
                   max_calls=100,max_output_tokens=2048,max_request_characters=200000,history_mode='full',resume=False):
    if policy not in POLICIES:raise ValueError('Unknown song policy')
    if history_mode not in ('full','metadata_only'):raise ValueError('Unknown whole-song history mode')
    identifier(song_id,'song_id')
    for name,value,low,high in (('seed',seed,0,2**63-1),('max_calls',max_calls,1,512),
        ('max_output_tokens',max_output_tokens,1,8192),('max_request_characters',max_request_characters,1,2000000)):
        integer(value,name,low)
        if value>high:raise ValueError('Song-agent budget outside bounds')
    sources={};source=load_song_source(source_run,song_id,sources)
    if max_calls<len(source['units']['units'])+1:raise ValueError('Budget cannot cover one commit per unit and finish')
    provider=create_provider(provider_spec,role='planner',base_dir=Path(source_run).resolve())
    if provider.identity['kind'] not in ('chat_completions','replay'):raise ValueError('Song agent requires real/replay text provider')
    cfg=dict(seed=seed,max_calls=max_calls,max_output_tokens=max_output_tokens,max_request_characters=max_request_characters,
             history_mode=history_mode)
    contract=dict(schema='choreobench.song_agent_contract.v1',sources=sources,policy=policy,config=cfg,
        provider_identity=copy.deepcopy(provider.identity),source_contract_sha256=source['source_contract_sha256'],
        source_input_sha256={k:object_sha(source[k]) for k in ('original','song','units','options','outline')},
        code_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')},
        shared_tool_permissions=True,historical_outline_reused=True,historical_prefix_reused=False,
        synthetic=source['synthetic'] or provider.identity['synthetic'],new_audio_calls=0,formal_e3_completed=False)
    output=Path(output_dir).resolve()
    if output.exists() and not resume:raise FileExistsError(str(output))
    if resume and not (output/'contract.json').is_file():raise ValueError('Song agent resume needs original contract')
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/'.run.lock'):
        def save(relative,value):
            p=output/relative
            if p.exists():
                if read_json(p)!=value:raise ValueError('Immutable song-agent artifact changed: '+relative)
            else:write_json(p,value)
        save('contract.json',contract)
        journal=_Journal(output,object_sha(contract),max_calls)
        previous=None
        if (output/'results.json').is_file():
            previous=_verify(read_json(output/'results.json'))
            if not previous['complete']:raise ValueError('Infrastructure-incomplete trajectory requires explicit diagnosis, not automatic retry')
            if previous['call_records']!={k:v['record_sha256'] for k,v in journal.records.items()} or previous['calls_used']!=len(journal.started):
                raise ValueError('Completed song-agent call records missing/changed; no resampling')
            for relative,digest in previous['artifacts'].items():
                if file_sha(output/relative)!=digest:raise ValueError('Completed song-agent artifact changed')
        plan=_empty_plan(source['original']);plan['motifs']=copy.deepcopy(source['outline']['motifs'])
        index=0;observations=[];steps=[];artifacts={};stop='budget_exhausted';fatal=None
        for call_index in range(max_calls):
            _unchanged(sources)
            request=_request(source,plan,index,observations,dict(cfg,call_index=call_index),policy)
            if previous is not None and request['call_id'] not in journal.records:
                raise ValueError('Completed song-agent reconstruction would dispatch; refused')
            prefix_sha=object_sha(plan);decision=None
            try:
                response=_dispatch(journal,provider,request,contract)
                decision=parse_json_response(response['text'])
                observation,proposed,next_index,done=_execute(source,plan,index,decision)
            except (ValueError,TypeError,KeyError) as error:
                observation=dict(status='rejected',error=str(error),quality_verified=False)
                raw=journal.records.get(request['call_id']);info=raw.get('result') if raw else None
                if (isinstance(error,_ReplayInfrastructureError) or not raw or not raw.get('success')
                        or not isinstance(info,dict) or info.get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response')):
                    fatal=str(error)
                proposed,next_index,done=plan,index,False
            step=dict(call_id=request['call_id'],source_prefix_sha256=prefix_sha,committed_units_before=index,
                decision=decision,observation=observation,committed_units_after=next_index)
            relative='steps/%03d.json'%call_index;save(relative,step);artifacts[relative]=file_sha(output/relative)
            if next_index!=index:
                relative='prefixes/%03d.json'%index;save(relative,proposed);artifacts[relative]=file_sha(output/relative)
            plan,index=proposed,next_index;steps.append(step);observations.append(observation)
            if fatal:stop='infrastructure_failed';break
            if done:stop=observation['status'];break
        _unchanged(sources)
        full=index==len(source['units']['units']) and stop=='finished'
        plan['provenance']=dict(mode='whole_song_tool_agent',policy=policy,contract_sha256=object_sha(contract),
            synthetic=contract['synthetic'],history_mode=history_mode,historical_outline_reused=True,historical_prefix_reused=False,
            citation_semantics='Input inspiration, not verified movement facts')
        # Incomplete plans remain partial artifacts, never published as complete scored plans.
        relative='plan.json' if full else 'partial-plan.not-success.json'
        save(relative,plan);artifacts[relative]=file_sha(output/relative)
        diagnostics=evaluate_plan(plan,source['original']) if plan['events'] else None
        if diagnostics is not None:
            diagnostics['literal_source_copies']=literal_source_copies(plan,source['options'])
            save('diagnostics.json',diagnostics);artifacts['diagnostics.json']=file_sha(output/'diagnostics.json')
        result=_sealed(dict(schema='choreobench.song_agent_result.v1',contract_sha256=object_sha(contract),
            complete=fatal is None,full_song_succeeded=full,stop_reason=stop,committed_units=index,
            total_units=len(source['units']['units']),calls_used=len(journal.started),
            call_records={k:v['record_sha256'] for k,v in journal.records.items()},artifacts=artifacts,
            plan_sha256=object_sha(plan),source_input_sha256=contract['source_input_sha256'],
            policy=policy,history_mode=history_mode,**_accounting(journal),new_audio_calls=0,human_reviews=0,quality_verified=False,
            formal_e3_completed=False,infrastructure_error=fatal,synthetic=contract['synthetic']))
        if previous is not None and result!=previous:raise ValueError('Completed song-agent reconstructed result differs')
        save('results.json',result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','song-id','provider','out-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--policy',choices=tuple(POLICIES),default='generic')
    parser.add_argument('--max-calls',type=int,default=100)
    parser.add_argument('--history-mode',choices=('full','metadata_only'),default='full')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    result=run_song_agent(args.source_run,args.song_id,read_json(args.provider),args.out_dir,
                          policy=args.policy,max_calls=args.max_calls,history_mode=args.history_mode,resume=args.resume)
    print(json_text(result))


if __name__=='__main__':main()
