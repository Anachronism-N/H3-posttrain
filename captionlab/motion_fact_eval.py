"""Bounded open-predicate reference diagnostics with original model evidence.

References are only text agreements, not unique choreography or music truth.
Private expected labels never reach either model request.
"""
import argparse
import copy
from pathlib import Path

from choreobench.baseline import _Journal, _sealed, _verify, _recorded_result
from choreobench.io import file_lock, file_sha, json_text, object_sha, read_json, read_json_snapshot, write_json
from choreobench.providers import create_provider, parse_json_response
from choreobench.memory_replay import _dispatch, _ReplayInfrastructureError
from choreobench.request_budget import guard_request
from choreobench.schema import keys, text, integer, identifier
from .motion_facts import EXTRACTION_PROMPT, QUOTE_EXTRACTION_PROMPT, MATCH_PROMPT, bind_extraction, bind_quote_extraction, bind_matches, score_facts

MAX_FACTS=8
PAIR_BATCH=8


def _extraction_schema(evidence_mode='spans',source_side_frame='unspecified'):
    fact=dict(type='object',additionalProperties=False,required=['id','action','body_part','side','side_frame','start','end','quote'],properties=dict(
        id=dict(type='string',minLength=1,maxLength=64),action=dict(type='string',minLength=1,maxLength=120),
        body_part=dict(type='string',minLength=1,maxLength=120),side=dict(enum=['left','right','both','unspecified']),
        side_frame=dict(enum=['image','anatomical','unspecified']),start=dict(type='integer',minimum=0),end=dict(type='integer',minimum=1),
        quote=dict(type='string',minLength=1,maxLength=600)))
    relation=dict(type='object',additionalProperties=False,required=['from','to','kind','start','end','quote'],properties={
        'from':dict(type='string',minLength=1,maxLength=64),'to':dict(type='string',minLength=1,maxLength=64),
        'kind':dict(enum=['before','simultaneous']),'start':dict(type='integer',minimum=0),'end':dict(type='integer',minimum=1),
        'quote':dict(type='string',minLength=1,maxLength=1000)})
    if source_side_frame=='image':fact['properties']['side_frame']['enum']=['image','unspecified']
    if evidence_mode=='quotes':
        for node in (fact,relation):
            for name in ('start','end'):
                node['required'].remove(name);node['properties'].pop(name)
            node['required'].append('occurrence');node['properties']['occurrence']=dict(type='integer',minimum=0,maximum=1023)
    return dict(type='object',additionalProperties=False,required=['facts','relations','uncertainties'],properties=dict(
        facts=dict(type='array',maxItems=MAX_FACTS,items=fact),relations=dict(type='array',maxItems=24,items=relation),
        uncertainties=dict(type='array',maxItems=8,items=dict(type='string',minLength=1,maxLength=300))))


def _pair_schema(pairs):
    slots=[]
    for a,b in pairs:
        slots.append(dict(type='object',additionalProperties=False,required=['candidate_id','reference_id','status','reason','candidate_quote','reference_quote'],properties=dict(
            candidate_id=dict(type='string',enum=[a['id']]),reference_id=dict(type='string',enum=[b['id']]),
            status=dict(enum=['match','no_match','unknown']),reason=dict(type='string',minLength=1,maxLength=300),
            candidate_quote=dict(type='string',minLength=1,maxLength=600),reference_quote=dict(type='string',minLength=1,maxLength=600))))
    return dict(type='object',additionalProperties=False,required=['judgments'],properties=dict(judgments=dict(
        type='array',minItems=len(slots),maxItems=len(slots),prefixItems=slots,items=False)))


def run_diagnostics(inventory_path,provider_path,out_dir,*,max_calls=30,resume=False,provider_factory=None,evidence_mode='spans'):
    if evidence_mode not in ('spans','quotes'):raise ValueError('Unknown motion evidence mode')
    inventory_path,provider_path,output=[Path(p).resolve() for p in (inventory_path,provider_path,out_dir)]
    inventory,inventory_sha=read_json_snapshot(inventory_path);spec,spec_sha=read_json_snapshot(provider_path)
    keys(inventory,('schema','seed','cases'),where='motion diagnostic inventory')
    if inventory['schema']!='captionlab.motion_fact_inventory.v1' or not isinstance(inventory['cases'],list) or not 1<=len(inventory['cases'])<=3:
        raise ValueError('Freeze one to three distinct motion diagnostic cases')
    integer(inventory['seed'],'seed',0);integer(max_calls,'max_calls',1)
    if inventory['seed']>=2**63 or max_calls>128 or type(resume) is not bool:raise ValueError('Diagnostic seed/budget/resume invalid')
    if max_calls<2*len(inventory['cases']):raise ValueError('Budget cannot cover the frozen extraction inventory')
    seen=set()
    for case in inventory['cases']:
        keys(case,('case_id','candidate_text','reference_text','private_expected','private_source'),('source_side_frames',),where='diagnostic case')
        frames=case.get('source_side_frames',{})
        keys(frames,(),('candidate','reference'),where='source direction provenance')
        if any(value not in ('image','anatomical','unspecified') for value in frames.values()):raise ValueError('Invalid declared source direction frame')
        identifier(case['case_id'],'case_id')
        if case['case_id'] in seen:raise ValueError('Duplicate diagnostic case')
        seen.add(case['case_id'])
        for field in ('candidate_text','reference_text'):
            text(case[field],field)
            if len(case[field])>12000:raise ValueError('Motion source exceeds bounded text input')
    if not isinstance(spec,dict) or spec.get('kind') not in ('chat_completions','replay'):
        raise ValueError('Explicit text judge required, never mock or audio generator')
    sources={str(inventory_path):inventory_sha,str(provider_path):spec_sha}
    module=Path(__file__);code={str(p.resolve()):file_sha(p) for p in (module,module.with_name('motion_facts.py'),module.parents[1]/'choreobench/providers.py',module.parents[1]/'choreobench/io.py',module.parents[1]/'choreobench/baseline.py')}
    base=dict(schema='captionlab.motion_fact_contract.v1',sources=sources,code_sha256=code,seed=inventory['seed'],max_calls=max_calls,
        max_facts=MAX_FACTS,pair_batch_size=PAIR_BATCH,max_output_tokens=2048,max_request_characters=60000,
        evidence_mode=evidence_mode,registered_cases=len(inventory['cases']),private_expected_visible=False,actual_audio_supplied=False,metric_calibrated=False)
    if output.exists() and not resume:raise FileExistsError(str(output))
    if resume and not (output/'contract.json').is_file():raise ValueError('Resume requires exact original diagnostic contract')
    output.mkdir(parents=True,exist_ok=True)
    factory=provider_factory or create_provider
    with file_lock(output/'.run.lock'):
        provider=None
        if resume:
            contract=_verify(read_json(output/'contract.json'))
            if set(contract)!=set(base)|{'provider_identity','record_sha256'} or any(contract[k]!=v for k,v in base.items()):
                raise ValueError('Diagnostic source/config/code drift; no model retries')
        else:
            provider=factory(copy.deepcopy(spec),role='planner',base_dir=provider_path.parent)
            contract=_sealed(dict(base,provider_identity=copy.deepcopy(provider.identity)));write_json(output/'contract.json',contract)
        identity=contract['provider_identity'];journal=_Journal(output,object_sha(contract),max_calls)
        previous=_verify(read_json(output/'results.json')) if (output/'results.json').exists() else None
        if previous is not None:
            if previous['call_records']!={k:v['record_sha256'] for k,v in journal.records.items()} or previous['calls_used']!=len(journal.started):
                raise ValueError('Completed diagnostic raw missing/changed; no resampling')
            for name,digest in previous['artifacts'].items():
                if file_sha(output/name)!=digest:raise ValueError('Completed diagnostic artifact changed')
        def unchanged():
            for path,digest in dict(sources,**code).items():
                if file_sha(path)!=digest:raise ValueError('Diagnostic source changed')
            if provider is not None and provider.identity!=identity:raise ValueError('Diagnostic provider identity changed')
        artifacts={};rows=[];used=set();fatal=False
        def save(name,value):
            p=output/name
            if p.exists():
                if read_json(p)!=value:raise ValueError('Immutable diagnostic artifact differs')
            elif previous is not None:raise ValueError('Completed diagnostic artifact missing')
            else:write_json(p,value)
            artifacts[name]=file_sha(p)
        def call(cid,prompt,payload,schema):
            nonlocal provider
            request=dict(call_id=cid,seed=inventory['seed'],max_output_tokens=2048,response_kind='open_motion_diagnostic',
                messages=[dict(role='system',content=prompt),dict(role='user',content=json_text(payload))],response_schema=schema)
            guard_request(request,contract);unchanged()
            if previous is not None and cid not in journal.records:raise ValueError('Completed diagnostic would dispatch replacement')
            if cid not in journal.started and len(journal.started)>=max_calls:raise ValueError('Frozen motion diagnostic call budget exhausted')
            if provider is None and cid not in journal.records:
                provider=factory(copy.deepcopy(spec),role='planner',base_dir=provider_path.parent)
                if provider.identity!=identity:raise ValueError('Diagnostic model differs from frozen identity')
            if cid not in journal.records:
                try:
                    _dispatch(journal,provider,request,contract)
                finally:
                    if cid in journal.records:used.add(cid)
            start=journal.started[cid];raw=journal.records[cid];used.add(cid)
            if start['request']!=request or start['provider_identity']!=identity:raise ValueError('Original diagnostic request changed')
            response=_recorded_result(raw,identity,request);unchanged()
            return parse_json_response(response['text'])
        for case in inventory['cases']:
            row=dict(case_id=case['case_id'],status='not_attempted',origin='model_judge',retained_in_denominator=True)
            if fatal:rows.append(row);continue
            try:
                values={}
                for side in ('candidate','reference'):
                    source=case[side+'_text'];frame=case.get('source_side_frames',{}).get(side,'unspecified')
                    prompt=QUOTE_EXTRACTION_PROMPT if evidence_mode=='quotes' else EXTRACTION_PROMPT
                    prompt+=' A movement direction is not a limb side. Do not infer a right arm merely because a hand moves right. Preserve unknown limb identity.'
                    response=call(case['case_id']+'/'+side+'/extract',prompt,
                        dict(source_text=source,maximum_facts=MAX_FACTS,source_direction_frame=frame),_extraction_schema(evidence_mode,frame))
                    values[side]=(bind_quote_extraction if evidence_mode=='quotes' else bind_extraction)(source,response)
                    if frame=='image' and any(f['side_frame']=='anatomical' for f in values[side]['facts']):
                        raise ValueError('Image-space reference cannot create anatomical side facts')
                    if len(values[side]['facts'])>MAX_FACTS:raise ValueError('Extraction exceeds actual declared fact budget')
                    save(case['case_id']+'/'+side+'.json',values[side])
                pairs=[(a,b) for a in values['candidate']['facts'] for b in values['reference']['facts']]
                batches=[pairs[i:i+PAIR_BATCH] for i in range(0,len(pairs),PAIR_BATCH)]
                if len(used)+len(batches)>max_calls:
                    raise ValueError('Matching cannot fit remaining frozen budget; no partial match score')
                judgments=[]
                for index,pair_batch in enumerate(batches):
                    payload=dict(pairs=[dict(candidate=a,reference=b) for a,b in pair_batch])
                    answer=call(case['case_id']+'/match/%02d'%index,MATCH_PROMPT,payload,_pair_schema(pair_batch))
                    keys(answer,('judgments',),where='fact pair batch')
                    if [(x['candidate_id'],x['reference_id']) for x in answer['judgments']]!=[(a['id'],b['id']) for a,b in pair_batch]:
                        raise ValueError('Fact pair response inventory changed')
                    judgments.extend(answer['judgments'])
                matches=bind_matches(values['candidate'],values['reference'],judgments)
                score=score_facts(values['candidate'],values['reference'],matches)
                score.update(extraction_fact_limit=MAX_FACTS,
                    extraction_limit_reached={side:len(value['facts'])==MAX_FACTS for side,value in values.items()},
                    extraction_uncertainty_counts={side:len(value['uncertainties']) for side,value in values.items()},
                    full_source_recall_verified=False,
                    metric_denominator_scope='Only extracted bounded source occurrences, not all true motion facts or unique correct choreography')
                save(case['case_id']+'/matches.json',matches);save(case['case_id']+'/score.json',score)
                row.update(status='complete',score=score,private_expected=copy.deepcopy(case['private_expected']))
            except (ValueError,TypeError,KeyError) as error:
                row.update(status='failed',error=str(error))
                # Unknown API outcomes end new dispatch; they are not replaced.
                fatal=isinstance(error,_ReplayInfrastructureError) or any(not r.get('success') or not isinstance(r.get('result'),dict) or
                    r['result'].get('identity')!=identity or r['result'].get('stop_reason') in (None,'http_error','infrastructure_error','malformed_api_response') for r in journal.records.values())
                if len(journal.started)>=max_calls:fatal=True
            rows.append(row)
        unchanged()
        if set(journal.records)!=used:raise ValueError('Diagnostic journal has unexpected or unused calls')
        from choreobench.edit_agent import _accounting
        result=_sealed(dict(schema='captionlab.motion_fact_results.v1',rows=rows,registered_cases=len(rows),
            successful_cases=sum(r['status']=='complete' for r in rows),failed_cases=sum(r['status']=='failed' for r in rows),
            not_attempted_cases=sum(r['status']=='not_attempted' for r in rows),calls_used=len(journal.started),
            contract_sha256=object_sha(contract),call_records={k:v['record_sha256'] for k,v in journal.records.items()},
            artifacts=artifacts,**_accounting(journal),human_reviews=0,audio_supplied=False,
            extraction_or_matching_truth_verified=False,formal_E5_completed=False))
        if previous is not None and result!=previous:raise ValueError('Completed diagnostic reconstruction changed')
        result_path=output/'results.json'
        if result_path.exists():
            if read_json(result_path)!=result:raise ValueError('Immutable diagnostic result changed')
        else:write_json(result_path,result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','provider','out-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-calls',type=int,default=30);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evidence-mode',choices=('spans','quotes'),default='spans')
    args=parser.parse_args(argv)
    result=run_diagnostics(args.inventory,args.provider,args.out_dir,max_calls=args.max_calls,resume=args.resume,evidence_mode=args.evidence_mode)
    print(json_text({k:result[k] for k in ('registered_cases','successful_cases','failed_cases','not_attempted_cases','calls_used')}))


if __name__=='__main__':main()
