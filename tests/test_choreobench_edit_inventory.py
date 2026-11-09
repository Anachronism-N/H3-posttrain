"""Synthetic inventories; returned patches never imply real semantic judgments."""
import copy

import pytest

from choreobench import edit_inventory as inventory
from choreobench.editing import apply_task_patch
from choreobench.io import file_sha, object_sha, read_json, write_json
from test_choreobench_review_sources import fixture


def material(tmp_path, semantic=True):
    original = fixture();before = original['plan'];song = original['song']
    dataset = dict(schema='choreobench.dataset.v1', dataset_id='fixture', synthetic=True,
        grouping=dict(status='unverified', evidence=''), songs=[song])
    write_json(tmp_path/'dataset.json', dataset);write_json(tmp_path/'before.json', before)
    def ref(name):return dict(path=name, sha256=file_sha(tmp_path/name))
    tasks=[];attempts=[]
    for tid in ('task-a','task-b'):
        task=dict(schema='choreobench.edit_task.v1',task_id=tid,song_id=song['song_id'],source_plan_sha256=object_sha(before),
            instruction='Lower movement energy only in event e.',allowed_event_ids=['e'],repair_event_ids=[],
            protected=[dict(event_id='*',field='start')],goals=[],semantic_goals=['Lower energy'] if semantic else [])
        if not semantic:task['goals']=[dict(event_id='e',field='action',op='contains',value='soft')]
        write_json(tmp_path/(tid+'.task.json'),task)
        tasks.append(dict(task_id=tid,dataset=ref('dataset.json'),source_plan=ref('before.json'),task=ref(tid+'.task.json')))
        for method in ('editor','noop'):
            patch=dict(schema='choreobench.patch.v1',base_plan_sha256=object_sha(before),instruction=task['instruction'],
                allowed_event_ids=task['allowed_event_ids'],repair_event_ids=[],protected=task['protected'],
                operations=[dict(event_id='e',field='action',value='soft steps')] if method=='editor' else [])
            after=apply_task_patch(before,patch,task)
            stem=tid+'.'+method
            write_json(tmp_path/(stem+'.patch.json'),patch);write_json(tmp_path/(stem+'.after.json'),after)
            attempts.append(dict(task_id=tid,method_id=method,seed=7,status='returned',after=ref(stem+'.after.json'),
                patch=ref(stem+'.patch.json'),error=None))
    declaration=dict(schema='choreobench.edit_inventory.v1',inventory_id='fixture',methods=['editor','noop'],seeds=[7,11],tasks=tasks)
    path=tmp_path/'inventory.json';write_json(path,declaration)
    runs=tmp_path/'attempts.json';write_json(runs,dict(schema='choreobench.edit_attempts.v1',inventory_sha256=file_sha(path),attempts=attempts))
    return path,runs


def save(path,value):write_json(path,value,overwrite=True)


def bind_second_dataset(tmp_path, paths, grouping, reverse=False):
    declaration=read_json(paths[0]);dataset=read_json(tmp_path/'dataset.json')
    dataset['dataset_id']='second-dataset';dataset['grouping']=copy.deepcopy(grouping)
    second=tmp_path/'second.dataset.json';write_json(second,dataset)
    declaration['tasks'][1]['dataset']=dict(path=second.name,sha256=file_sha(second))
    if reverse:declaration['tasks'].reverse()
    save(paths[0],declaration)
    attempts=read_json(paths[1]);attempts['inventory_sha256']=file_sha(paths[0]);save(paths[1],attempts)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('grouping',[
    dict(status='reviewed',evidence='Declared source grouping audit'),
    dict(status='unverified',evidence='Different grouping evidence declaration'),
])
def test_same_song_conflicting_dataset_grouping_rejected_in_either_task_order(tmp_path,reverse,grouping):
    paths=material(tmp_path)
    bind_second_dataset(tmp_path,paths,grouping,reverse)
    with pytest.raises(ValueError,match='inconsistent dataset grouping declarations'):
        inventory.audit_edit_inventory(*paths)


@pytest.mark.parametrize('reverse',[False,True])
def test_same_song_multiple_datasets_with_equal_grouping_accepted(tmp_path,reverse):
    paths=material(tmp_path);grouping=read_json(tmp_path/'dataset.json')['grouping']
    bind_second_dataset(tmp_path,paths,grouping,reverse)
    report=inventory.audit_edit_inventory(*paths)
    assert report['song_count']==1 and report['task_count']==2 and report['expected_cells']==8
    assert report['songs'][0]['grouping']==grouping
    assert len({c['dataset_sha256'] for c in report['cells']})==2
    assert sum(c['status']=='missing_attempt' for c in report['cells'])==4


def test_multi_task_same_song_methods_and_seeds_preserve_missing_and_noop_pending(tmp_path):
    paths=material(tmp_path);report=inventory.audit_edit_inventory(*paths)
    assert report['task_count']==2 and report['song_count']==1 and report['expected_cells']==8 and report['recorded_attempts']==4
    assert len(report['cells'])==8
    assert sum(c['status']=='missing_attempt' for c in report['cells'])==4
    returned=[c for c in report['cells'] if c['mechanical_valid']]
    assert len(returned)==4 and all(c['status']=='returned_pending_semantics' and c['joint_success'] is None for c in returned)
    assert sum(c['empty_operations'] is True for c in returned)==2 and sum(c['no_op'] is True for c in returned)==2
    assert report['songs'][0]['task_ids']==['task-a','task-b'] and report['songs'][0]['results']['editor']['expected_cells']==4
    assert report['quality_verified'] is False and report['numeric_quality_or_ci'] is None
    assert report['record_sha256']==object_sha({k:v for k,v in report.items() if k!='record_sha256'})


def test_literal_failure_precedes_pending_semantics_and_noop_is_not_automatically_a_fail(tmp_path):
    paths=material(tmp_path,semantic=False);report=inventory.audit_edit_inventory(*paths)
    returned=[c for c in report['cells'] if c['mechanical_valid']]
    assert sum(c['status']=='literal_fail' and c['joint_success'] is False for c in returned)==2
    assert sum(c['status']=='literal_pass_semantics_not_requested' and c['joint_success'] is True for c in returned)==2
    declaration=read_json(paths[0]);task_ref=declaration['tasks'][0]['task'];task=read_json(tmp_path/task_ref['path'])
    task['goals'][0]['value']='Step left';save(tmp_path/task_ref['path'],task);task_ref['sha256']=file_sha(tmp_path/task_ref['path'])
    save(paths[0],declaration);attempts=read_json(paths[1]);attempts['inventory_sha256']=file_sha(paths[0]);save(paths[1],attempts)
    report=inventory.audit_edit_inventory(*paths)
    cell=next(c for c in report['cells'] if c['task_id']=='task-a' and c['method_id']=='noop' and c['seed']==7)
    assert cell['joint_success'] is True and cell['no_op']


def test_known_literal_failure_precedes_semantic_pending(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[0])
    for entry in doc['tasks']:
        p=tmp_path/entry['task']['path'];task=read_json(p)
        task['goals']=[dict(event_id='e',field='action',op='contains',value='absent-required-text')]
        save(p,task);entry['task']['sha256']=file_sha(p)
    save(paths[0],doc);attempts=read_json(paths[1]);attempts['inventory_sha256']=file_sha(paths[0]);save(paths[1],attempts)
    report=inventory.audit_edit_inventory(*paths)
    assert all(c['status']=='literal_fail' and c['joint_success'] is False for c in report['cells'] if c['mechanical_valid'])
    assert all(c['semantic_status']=='pending_independent_review' for c in report['cells'] if c['mechanical_valid'])


def test_unauthorized_patch_remains_invalid_attempt_not_discarded(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[1]);row=doc['attempts'][0]
    p=tmp_path/row['patch']['path'];patch=read_json(p);patch['operations']=[dict(event_id='e',field='start',value=1)]
    save(p,patch);row['patch']['sha256']=file_sha(p);save(paths[1],doc)
    report=inventory.audit_edit_inventory(*paths);cell=report['cells'][0]
    assert cell['status']=='invalid_output' and cell['joint_success'] is False
    assert 'protected' in cell['errors'][0] and report['expected_cells']==8


def test_reported_failure_and_unlisted_task_stay_in_denominator(tmp_path):
    paths=material(tmp_path);attempts=read_json(paths[1]);attempts['attempts']=attempts['attempts'][:1]
    attempts['attempts'][0].update(status='failed',after=None,patch=None,error='Fixture backend failure')
    save(paths[1],attempts);report=inventory.audit_edit_inventory(*paths)
    assert report['expected_cells']==8 and len(report['cells'])==8
    assert report['cells'][0]['status']=='declared_failed' and report['cells'][0]['joint_success'] is False
    assert sum(c['status']=='missing_attempt' for c in report['cells'])==7


def test_invalid_returned_output_or_patch_is_not_dropped(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[1]);row=doc['attempts'][0]
    after_path=tmp_path/row['after']['path'];after=read_json(after_path);after['events'][0]['start']=1
    save(after_path,after);row['after']['sha256']=file_sha(after_path);save(paths[1],doc)
    report=inventory.audit_edit_inventory(*paths);cell=report['cells'][0]
    assert cell['status']=='invalid_output' and cell['joint_success'] is False and report['expected_cells']==8
    assert 'exact execution' in cell['errors'][0]


def test_empty_goal_set_remains_not_evaluable(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[0])
    for entry in doc['tasks']:
        p=tmp_path/entry['task']['path'];task=read_json(p);task['semantic_goals']=[];save(p,task);entry['task']['sha256']=file_sha(p)
    save(paths[0],doc);attempts=read_json(paths[1]);attempts['inventory_sha256']=file_sha(paths[0]);save(paths[1],attempts)
    report=inventory.audit_edit_inventory(*paths)
    assert all(c['status']=='not_evaluable' and c['joint_success'] is None for c in report['cells'] if c['mechanical_valid'])


@pytest.mark.parametrize('change',['inventory_binding','method','task','seed_bool','seed_unknown','duplicate','status','failure_plan','returned_missing','extra'])
def test_invalid_attempt_identity_or_shape_rejected(tmp_path,change):
    paths=material(tmp_path);doc=read_json(paths[1]);row=doc['attempts'][0]
    if change=='inventory_binding':doc['inventory_sha256']=object_sha('wrong')
    elif change=='method':row['method_id']='unknown'
    elif change=='task':row['task_id']='unknown'
    elif change=='seed_bool':row['seed']=True
    elif change=='seed_unknown':row['seed']=12
    elif change=='duplicate':doc['attempts'].append(copy.deepcopy(row))
    elif change=='status':row['status']='success'
    elif change=='failure_plan':row.update(status='failed',error='reported')
    elif change=='returned_missing':row['after']=None
    else:row['scores']=5
    save(paths[1],doc)
    with pytest.raises(ValueError):inventory.audit_edit_inventory(*paths)


@pytest.mark.parametrize('change',['schema','methods','seeds','seed_bool','tasks_empty','task_duplicate','unknown_field','wrong_task','wrong_dataset','wrong_source'])
def test_bad_independent_inventory_rejected(tmp_path,change):
    paths=material(tmp_path);doc=read_json(paths[0])
    if change=='schema':doc['schema']='wrong'
    elif change=='methods':doc['methods']*=2
    elif change=='seeds':doc['seeds']*=2
    elif change=='seed_bool':doc['seeds']=[True]
    elif change=='tasks_empty':doc['tasks']=[]
    elif change=='task_duplicate':doc['tasks'].append(copy.deepcopy(doc['tasks'][0]))
    elif change=='unknown_field':doc['quality']=[]
    elif change=='wrong_task':doc['tasks'][0]['task_id']='wrong'
    elif change=='wrong_dataset':
        p=tmp_path/'dataset.json';data=read_json(p);data['songs'][0]['audio']['num_samples']+=1;save(p,data)
        for t in doc['tasks']:t['dataset']['sha256']=file_sha(p)
    else:doc['tasks'][0]['source_plan']['sha256']=object_sha('wrong')
    save(paths[0],doc)
    with pytest.raises(ValueError):inventory.audit_edit_inventory(*paths)


def test_hash_drift_is_integrity_error_not_declared_generation_failure(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[1]);p=tmp_path/doc['attempts'][0]['after']['path'];p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='source hash'):inventory.audit_edit_inventory(*paths)


def test_source_mutation_during_evaluation_rejected(tmp_path,monkeypatch):
    paths=material(tmp_path);old=inventory.evaluate_edit
    def mutate(*args):
        answer=old(*args);paths[1].write_bytes(paths[1].read_bytes()+b' ');return answer
    monkeypatch.setattr(inventory,'evaluate_edit',mutate)
    with pytest.raises(ValueError,match='source changed'):inventory.audit_edit_inventory(*paths)


@pytest.mark.parametrize('module',[True,False])
def test_both_cli_report_original_denominator_exclusively(tmp_path,module):
    paths=material(tmp_path);out=tmp_path/'report.json';argv=['--inventory',str(paths[0]),'--attempts',str(paths[1]),'--out',str(out)]
    if module:
        inventory.main(argv)
        with pytest.raises(FileExistsError):inventory.main(argv)
    else:
        from choreobench.cli import main
        assert main(['audit-edit-inventory']+argv)==0 and main(['audit-edit-inventory']+argv)==2
    assert read_json(out)['expected_cells']==8
