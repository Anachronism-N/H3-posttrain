"""Regression coverage for multi-group rewrites and reusable no-rewrite drafts."""
import copy
import json
from types import SimpleNamespace

from choreobench import composition_workflows as cw
from choreobench.io import read_json
from test_choreobench_composition_workflows import material


def test_zero_group_still_exports_original_and_cost_prefix(tmp_path,monkeypatch):
    def unique(answer,request,data,index):
        if 'events' in answer:
            answer['events'][0]['action']='Arms rise.' if index==2 else 'Knees bend.'
    path,sid,out,calls=material(tmp_path,monkeypatch,'workflow',unique)
    result=cw.run_composition(path,sid,{'kind':'replay'},out,method='workflow',rewrite_repeats=True)
    assert result['full_song_succeeded'] and result['repeat_rewrite']['calls_used']==0
    assert read_json(out/'before_rewrite.json')==read_json(out/'plan.json')
    assert read_json(out/'before_rewrite_costs.json')['calls_used']==result['calls_used']
    assert (out/'before_rewrite_presentation.txt').exists()


def test_two_successful_groups_have_distinct_logs_and_exact_call_count(tmp_path,monkeypatch):
    def two_events(answer,request,data,index):
        if 'events' in answer:
            first=answer['events'][0];second=copy.deepcopy(first)
            split=(first['start']+first['end'])//2
            first['end']=split;second['start']=split
            first['action']='Arm sweeps left and right.';second['action']='Knee bends slowly.'
            answer['events']=[first,second]
    path,sid,out,calls=material(tmp_path,monkeypatch,'workflow',two_events)
    original_factory=cw.create_provider
    def factory(*args,**kwargs):
        provider=original_factory(*args,**kwargs)
        def generate(request):
            if request['response_kind']!='repeat_rewrite':return provider.generate(request)
            calls.append(copy.deepcopy(request));payload=json.loads(request['messages'][-1]['content'])
            action=('Arm sweeps overhead then folds down.' if 'Arm' in payload['repeated_action'] else 'Knee bends outward and rises slowly.')
            answer=dict(rewrites=[dict(event_id=e['event_id'],action=action) for e in payload['occurrences'] if e['rewrite_this']])
            return dict(text=json.dumps(answer),identity=provider.identity,synthetic=True,raw={},stop_reason='stop',
                        costs=dict(input_tokens=100,output_tokens=50))
        return SimpleNamespace(identity=provider.identity,generate=generate)
    monkeypatch.setattr(cw,'create_provider',factory)
    result=cw.run_composition(path,sid,{'kind':'replay'},out,method='workflow',rewrite_repeats=True)
    assert result['full_song_succeeded']
    assert result['repeat_rewrite']['applied']==2 and result['repeat_rewrite']['failed']==[]
    assert result['repeat_rewrite']['calls_used']==2
    assert result['calls_used']-read_json(out/'before_rewrite_costs.json')['calls_used']==2
    records=[read_json(p) for p in (out/'steps').glob('*.json')]
    assert sum(r.get('kind')=='repeat_rewrite' for r in records)==2
