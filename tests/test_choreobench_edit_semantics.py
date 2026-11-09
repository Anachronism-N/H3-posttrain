"""Synthetic verdicts never imply actual independent choreography reviews."""
import copy
import pytest

from choreobench import edit_semantics as semantics
from choreobench.io import read_json, write_json, object_sha, file_sha
from choreobench.reviews import build_review_pack
from test_choreobench_edit_inventory import material as inventory_material


def material(tmp_path, prepare=True, literal_failure=False, multiple_goals=False, no_goals=False):
    inventory,attempts=inventory_material(tmp_path)
    inv=read_json(inventory);runs=read_json(attempts);items=[]
    if literal_failure or multiple_goals or no_goals:
        for entry in inv['tasks']:
            task_path=tmp_path/entry['task']['path'];task=read_json(task_path)
            if literal_failure:task['goals']=[dict(event_id='e',field='action',op='contains',value='unmet-extra-goal')]
            if multiple_goals:task['semantic_goals'].append('Keep shoulders relaxed')
            if no_goals:task['semantic_goals']=[]
            write_json(task_path,task,overwrite=True);entry['task']['sha256']=file_sha(task_path)
        write_json(inventory,inv,overwrite=True);runs['inventory_sha256']=file_sha(inventory);write_json(attempts,runs,overwrite=True)
    for row in runs['attempts']:
        entry=next(e for e in inv['tasks'] if e['task_id']==row['task_id'])
        dataset=read_json(tmp_path/entry['dataset']['path']);song=dataset['songs'][0]
        items.append(dict(song=song,plan=read_json(tmp_path/row['after']['path']),
            source_plan=read_json(tmp_path/entry['source_plan']['path']),edit_task=read_json(tmp_path/entry['task']['path']),
            method_id='alias-'+row['method_id'],source_binding=dict(kind='submission',sha256=object_sha(row),
                dataset_sha256=entry['dataset']['sha256'],synthetic=True)))
    bundle=build_review_pack(items)
    public=tmp_path/'public.json';private=tmp_path/'private.json';mapping=tmp_path/'mapping.json';out=tmp_path/'prepared'
    write_json(public,bundle['public']);write_json(private,bundle['private'])
    mappings=[]
    for item in bundle['private']['items']:
        mappings.append(dict(review_id=item['review_id'],task_id=item['original']['edit_task']['task_id'],
            method_id=item['method_id'].removeprefix('alias-'),seed=7))
    write_json(mapping,dict(schema='choreobench.edit_semantic_mapping.v1',inventory_sha256=file_sha(inventory),
        attempts_sha256=file_sha(attempts),pack_sha256=object_sha(bundle['public']),cells=mappings))
    args=(inventory,attempts,public,private,mapping,out)
    if prepare:semantics.prepare_edit_semantics(*args)
    return args


def verdict(req, reviewer='test-observer-not-human',status='pass'):
    return dict(review_id=req['review_id'],item_sha256=req['item_sha256'],source_plan_sha256=req['source_plan_sha256'],
        after_plan_sha256=req['after_plan_sha256'],reviewer_id=reviewer,origin='human',model_id=None,
        reviewed_ranges=copy.deepcopy(req['required_review_ranges']),
        goals=[dict(goal_index=g['goal_index'],goal_sha256=g['goal_sha256'],status=status,evidence='Software fixture only') for g in req['goals']],
        boundary=dict(status='pass',evidence='Software boundary fixture only'),comments='Not an actual human assessment')


def analyze(args,rows=None):
    inventory,attempts,_,_,_,out=args
    requirements=read_json(out/'requirements.json')
    path=out/'verdicts.json';write_json(path,dict(schema='choreobench.edit_semantic_verdicts.v1',requirements_sha256=object_sha(requirements),
        records=rows or []),overwrite=True)
    return semantics.analyze_edit_semantics(inventory,attempts,out/'requirements.json',out/'bridge.private.json',[path])


def test_prepare_uses_explicit_alias_bridge_and_keeps_missing_seed_inventory(tmp_path):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json');bridge=read_json(args[-1]/'bridge.private.json')
    assert len(req['items'])==4 and all(len(r['goals'])==1 for r in req['items'])
    assert 'alias-' not in str(req) and 'task-a' not in str(req)
    assert all(g['goal_sha256']==object_sha(g['description']) for r in req['items'] for g in r['goals'])
    report=analyze(args)
    assert report['expected_inventory_cells']==8 and report['mapped_review_items']==4
    assert len(report['review_items_without_any_verdict'])==4 and len(report['unrepresented_inventory_cells'])==4
    assert report['observer_decisions']==[] and report['consensus'] is None and not report['quality_verified']
    assert report['record_sha256']==object_sha({k:v for k,v in report.items() if k!='record_sha256'})
    with pytest.raises(FileExistsError):semantics.prepare_edit_semantics(*args)


@pytest.mark.parametrize('status,result,success',[('pass','declared_joint_pass',True),('partial','declared_partial',False),
    ('fail','declared_semantic_fail',False),('unknown','pending_or_insufficient_review',None)])
def test_explicit_goal_status_joint_semantics_not_inferred_from_patch(tmp_path,status,result,success):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0]
    report=analyze(args,[verdict(req,status=status)])
    decision=report['observer_decisions'][0]
    assert decision['status']==result and decision['joint_success'] is success
    assert len(report['review_items_without_any_verdict'])==3 and not report['quality_verified']


def test_partial_review_or_missing_goal_cannot_pass_but_known_failure_dominates(tmp_path):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0];row=verdict(req)
    row['reviewed_ranges']=[]
    d=analyze(args,[row])['observer_decisions'][0]
    assert not d['required_range_covered'] and d['joint_success'] is None
    row['goals'][0]['status']='fail'
    assert analyze(args,[row])['observer_decisions'][0]['joint_success'] is False
    row=verdict(req);row['goals']=[]
    d=analyze(args,[row])['observer_decisions'][0]
    assert d['missing_goal_indices']==[0] and d['joint_success'] is None


def test_known_literal_failure_dominates_declared_semantic_pass(tmp_path):
    args=material(tmp_path,literal_failure=True);req=read_json(args[-1]/'requirements.json')['items'][0]
    d=analyze(args,[verdict(req)])['observer_decisions'][0]
    assert d['status']=='mechanical_or_literal_fail' and d['joint_success'] is False


def test_each_goal_required_and_partial_precedes_unknown_missing(tmp_path):
    args=material(tmp_path,multiple_goals=True);req=read_json(args[-1]/'requirements.json')['items'][0];row=verdict(req)
    row['goals'].pop()
    assert analyze(args,[row])['observer_decisions'][0]['joint_success'] is None
    row['goals'][0]['status']='partial'
    d=analyze(args,[row])['observer_decisions'][0]
    assert d['joint_success'] is False and d['missing_goal_indices']==[1]
    assert analyze(args,[verdict(req)])['observer_decisions'][0]['joint_success'] is True


def test_no_goals_and_boundary_pass_is_not_semantic_success(tmp_path):
    args=material(tmp_path,no_goals=True);req=read_json(args[-1]/'requirements.json')['items'][0]
    d=analyze(args,[verdict(req)])['observer_decisions'][0]
    assert d['joint_success'] is None and d['status']=='not_evaluable'


def test_boundary_unknown_partial_or_failure_is_not_ignored(tmp_path):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0];row=verdict(req)
    for state,answer in [('unknown',None),('partial',False),('fail',False)]:
        row['boundary']['status']=state
        assert analyze(args,[row])['observer_decisions'][0]['joint_success'] is answer


def test_ranges_are_union_on_integer_half_open_clock(tmp_path):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0];row=verdict(req)
    a=req['required_review_ranges'][0]['start'];b=req['required_review_ranges'][0]['end'];mid=(a+b)//2
    row['reviewed_ranges']=[dict(start=mid,end=b),dict(start=a,end=mid+1)]
    d=analyze(args,[row])['observer_decisions'][0]
    assert d['reviewed_range_union']==[dict(start=a,end=b)] and d['joint_success'] is True
    row['reviewed_ranges']=[dict(start=a,end=mid),dict(start=mid+1,end=b)]
    assert analyze(args,[row])['observer_decisions'][0]['joint_success'] is None


def test_disagreement_preserves_independent_records_no_majority(tmp_path):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0]
    report=analyze(args,[verdict(req,'observer-a','pass'),verdict(req,'observer-b','fail')])
    assert [d['joint_success'] for d in report['observer_decisions']]==[True,False]
    assert report['consensus'] is None


@pytest.mark.parametrize('change',['placeholder','review_id','item_sha','before_sha','after_sha','origin','human_model','range_bool','range_float',
    'range_end','range_reversed','goal_sha','goal_index','goal_duplicate','status','evidence','boundary_extra','record_extra','duplicate'])
def test_changed_or_forged_verdicts_rejected(tmp_path,change):
    args=material(tmp_path);req=read_json(args[-1]/'requirements.json')['items'][0];row=verdict(req);rows=[row]
    if change=='placeholder':row['reviewer_id']='FILL_REVIEWER_ID'
    elif change=='review_id':row['review_id']='unknown'
    elif change=='item_sha':row['item_sha256']=object_sha('wrong')
    elif change=='before_sha':row['source_plan_sha256']=object_sha('wrong')
    elif change=='after_sha':row['after_plan_sha256']=object_sha('wrong')
    elif change=='origin':row['origin']='unknown'
    elif change=='human_model':row['model_id']='v1'
    elif change=='range_bool':row['reviewed_ranges'][0]['start']=False
    elif change=='range_float':row['reviewed_ranges'][0]['start']=0.0
    elif change=='range_end':row['reviewed_ranges'][0]['end']=req['num_samples']+1
    elif change=='range_reversed':row['reviewed_ranges'][0]['start']=req['num_samples']
    elif change=='goal_sha':row['goals'][0]['goal_sha256']=object_sha('wrong')
    elif change=='goal_index':row['goals'][0]['goal_index']=9
    elif change=='goal_duplicate':row['goals']*=2
    elif change=='status':row['goals'][0]['status']='4/5'
    elif change=='evidence':row['goals'][0]['evidence']=' '
    elif change=='boundary_extra':row['boundary']['rating']=4
    elif change=='record_extra':row['rating']=4
    else:rows.append(copy.deepcopy(row))
    with pytest.raises(ValueError):analyze(args,rows)


@pytest.mark.parametrize('change',['task','method','seed','duplicate','missing','pack','before_in_pack'])
def test_bad_explicit_mapping_rejected_without_publishing(tmp_path,change):
    args=material(tmp_path,prepare=False);mapping=read_json(args[4]);row=mapping['cells'][0]
    if change=='task':row['task_id']='unknown'
    elif change=='method':row['method_id']='alias-editor'
    elif change=='seed':row['seed']=11
    elif change=='duplicate':mapping['cells'].append(copy.deepcopy(row))
    elif change=='missing':mapping['cells'].pop()
    elif change=='pack':mapping['pack_sha256']=object_sha('wrong')
    else:
        b=read_json(args[3]);b['items'][0]['original']['source_plan']['events'][0]['action']='changed';write_json(args[3],b,overwrite=True)
    write_json(args[4],mapping,overwrite=True)
    with pytest.raises(ValueError):semantics.prepare_edit_semantics(*args)
    assert not args[-1].exists()


def test_original_goal_cannot_be_replaced_by_rehashing_requirements_and_bridge(tmp_path):
    args=material(tmp_path);out=args[-1];req=read_json(out/'requirements.json');req['items'][0]['goals'][0]['description']='easier target'
    req['items'][0]['goals'][0]['goal_sha256']=object_sha('easier target');write_json(out/'requirements.json',req,overwrite=True)
    bridge=read_json(out/'bridge.private.json');bridge['requirements_sha256']=object_sha(req)
    bridge['cells'][0]['requirement_sha256']=object_sha(req['items'][0]);bridge.pop('record_sha256');bridge['record_sha256']=object_sha(bridge)
    write_json(out/'bridge.private.json',bridge,overwrite=True)
    with pytest.raises(ValueError,match='original evidence'):analyze(args)


def test_duplicate_records_across_files_and_model_identity_drift_rejected(tmp_path):
    args=material(tmp_path);out=args[-1];req=read_json(out/'requirements.json');row=verdict(req['items'][0])
    analyze(args,[row]);p=out/'verdicts.json'
    with pytest.raises(ValueError,match='duplicate'):
        semantics.analyze_edit_semantics(args[0],args[1],out/'requirements.json',out/'bridge.private.json',[p,p])
    other=verdict(req['items'][1]);other.update(origin='model_judge',model_id='v1')
    with pytest.raises(ValueError,match='identity'):analyze(args,[row,other])


def test_null_unfilled_template_and_input_drift_are_rejected(tmp_path,monkeypatch):
    args=material(tmp_path);out=args[-1]
    template=read_json(out/'verdicts.template.json')
    assert all(v['status'] is None for row in template['records'] for v in row['goals'])
    with pytest.raises(ValueError,match='placeholder'):
        semantics.analyze_edit_semantics(args[0],args[1],out/'requirements.json',out/'bridge.private.json',[out/'verdicts.template.json'])
    req=read_json(out/'requirements.json');row=verdict(req['items'][0])
    analyze(args,[row])
    original=semantics._ranges
    def mutate(rows,samples):
        answer=original(rows,samples)
        p=out/'verdicts.json';p.write_bytes(p.read_bytes()+b' ')
        return answer
    monkeypatch.setattr(semantics,'_ranges',mutate)
    with pytest.raises(ValueError,match='source changed'):analyze(args,[row])


def test_required_context_covers_neighbors_from_before_and_after():
    events=[dict(event_id='a',start=0,end=10),dict(event_id='b',start=10,end=20),dict(event_id='c',start=20,end=30)]
    before=dict(events=events,num_samples=30);after=copy.deepcopy(before)
    after['events'][1].update(start=12,end=25)
    assert semantics._required_range(before,after,dict(allowed_event_ids=['b'],repair_event_ids=[]))==[dict(start=0,end=30)]


@pytest.mark.parametrize('module',[True,False])
def test_prepare_and_analyze_cli_exclusive(tmp_path,module):
    args=material(tmp_path,prepare=False);inv,attempts,public,private,mapping,out=args
    prepare=['--inventory',str(inv),'--attempts',str(attempts),'--public',str(public),'--private',str(private),'--mapping',str(mapping),'--out-dir',str(out)]
    if module:semantics.main(['prepare']+prepare)
    else:
        from choreobench.cli import main
        assert main(['prepare-edit-semantics']+prepare)==0
    report=out/'analysis.json';analyze_args=['--inventory',str(inv),'--attempts',str(attempts),'--requirements',str(out/'requirements.json'),
        '--bridge',str(out/'bridge.private.json'),'--verdicts',str(out/'verdicts.none-submitted.json'),'--out',str(report)]
    if module:
        semantics.main(['analyze']+analyze_args)
        with pytest.raises(FileExistsError):semantics.main(['analyze']+analyze_args)
    else:
        assert main(['analyze-edit-semantics']+analyze_args)==0 and main(['analyze-edit-semantics']+analyze_args)==2
    assert read_json(report)['observer_decisions']==[]
