"""Range arithmetic fixtures are not quality observations or actual reviewers."""
import copy
import pytest

from choreobench import review_ranges as ranges
from choreobench.io import file_sha, object_sha, read_json, write_json
from test_choreobench_review_issues import material as issue_material


def material(tmp_path):
    pack, issues = issue_material();row = issues['records'][0]
    original = row['issues'][0]
    row['issues'] = [dict(original,issue_id='a',type='unintended_repeat',start=10000,end=30000,event_ids=['event_0001']),
        dict(original,issue_id='b',type='unintended_repeat',start=20000,end=35000,event_ids=['event_0001']),
        dict(original,issue_id='c',type='ambiguous_action',start=25000,end=45000),
        dict(original,issue_id='d',type='reasonable_recurrence',severity='not_applicable',start=50000,end=60000,event_ids=['event_0002']),
        dict(original,issue_id='u',type='uncertain',severity='uncertain',start=70000,end=80000,event_ids=['event_0002'])]
    scope = dict(review_id=row['review_id'], item_sha256=row['item_sha256'],reviewer_id=row['reviewer_id'],
        origin='human',model_id=None,audio_seen=False,
        checked_types=[t for t in ranges.ISSUE_TYPES if t!='music_mismatch'],ranges=[dict(start=0,end=81000)])
    paths = (tmp_path/'pack.json',tmp_path/'issues.json',tmp_path/'ranges.json')
    write_json(paths[0],pack);write_json(paths[1],issues)
    write_json(paths[2],dict(schema='choreobench.review_ranges.v1',pack_sha256=object_sha(pack),issues_file_sha256=file_sha(paths[1]),records=[scope]))
    return paths


def update(path,doc):write_json(path,doc,overwrite=True)


def rebind_issue_file(paths, issues):
    update(paths[1],issues);doc=read_json(paths[2]);doc['issues_file_sha256']=file_sha(paths[1]);update(paths[2],doc)


def analyze(paths):return ranges.analyze_review_ranges(paths[0],paths[1],[paths[2]])


def test_overlap_union_positive_recurrence_and_unknown_are_separate(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[2]);doc['records'][0]['audio_seen']=True
    doc['records'][0]['checked_types']=list(ranges.ISSUE_TYPES);update(paths[2],doc)
    issues=read_json(paths[1]);issues['records'][0]['audio_seen']=True;rebind_issue_file(paths,issues)
    report=analyze(paths);row=report['rows'][0]
    assert row['reviewed_samples']==81000
    assert row['by_type']['unintended_repeat']['affected_samples']==25000
    assert row['by_type']['unintended_repeat']['reviewed_fraction']==25000/81000
    assert row['all_major_problem_union']['affected_samples']==35000
    assert row['all_major_problem_union']['at_least_one_major_problem'] is True
    assert row['by_type']['reasonable_recurrence']['classification']=='counterexample'
    assert row['by_type']['reasonable_recurrence']['affected_samples']==10000
    assert row['by_type']['uncertain']['affected_samples']==10000 and row['quality_pass'] is None
    assert row['declared_uncertainty_samples']==10000 and report['consensus'] is None


def test_partial_scope_clips_intersections_and_accounts_excluded_samples(tmp_path):
    paths=material(tmp_path);scope=read_json(paths[2]);scope['records'][0]['ranges']=[dict(start=15000,end=25000),dict(start=20000,end=40000)]
    update(paths[2],scope);row=analyze(paths)['rows'][0]
    assert row['reviewed_samples']==25000 and row['reviewed_intervals']==[dict(start=15000,end=40000)]
    assert row['by_type']['unintended_repeat']['affected_samples']==20000
    assert row['by_type']['ambiguous_action']['affected_samples']==15000
    assert row['by_type']['reasonable_recurrence']['affected_samples']==0
    assert next(i for i in row['issue_scope_accounting'] if i['issue_id']=='c')['outside_reviewed_samples']==5000
    assert row['all_major_problem_union']['reviewed_fraction'] is None  # music mismatch was not checked


def test_unchecked_types_are_na_not_zero(tmp_path):
    paths=material(tmp_path);doc=read_json(paths[2]);doc['records'][0]['checked_types']=['unintended_repeat'];update(paths[2],doc)
    row=analyze(paths)['rows'][0]
    assert row['by_type']['unintended_repeat']['affected_samples']==25000
    assert row['by_type']['ambiguous_action']['affected_samples'] is None
    assert row['major_by_type']['ambiguous_action']['reviewed_fraction'] is None
    assert row['all_major_problem_union']['at_least_one_major_problem'] is None


@pytest.mark.parametrize('missing',['scope','issues','time','both'])
def test_missing_record_or_no_reviewed_time_is_na(tmp_path,missing):
    paths=material(tmp_path);issue=read_json(paths[1]);scope=read_json(paths[2])
    if missing in ('scope','both'):scope['records']=[]
    if missing=='time':scope['records'][0]['ranges']=[]
    if missing in ('issues','both'):issue['records']=[];update(paths[1],issue);scope['issues_file_sha256']=file_sha(paths[1])
    update(paths[2],scope);report=analyze(paths)
    if missing=='both':
        assert not report['rows'] and len(report['items_without_any_scope_record'])==1 and len(report['items_without_any_issue_record'])==1
    else:
        assert report['rows'][0]['by_type']['unintended_repeat']['reviewed_fraction'] is None
        assert report['rows'][0]['quality_pass'] is None


def test_explicit_empty_issues_with_positive_checked_scope_means_zero_declarations_not_pass(tmp_path):
    paths=material(tmp_path);issues=read_json(paths[1]);issues['records'][0]['issues']=[];rebind_issue_file(paths,issues)
    row=analyze(paths)['rows'][0]
    assert row['by_type']['unintended_repeat']['affected_samples']==0
    assert row['by_type']['unintended_repeat']['reviewed_fraction']==0
    assert row['quality_pass'] is None and not row['reviewer_identity_listening_or_issue_truth_authenticated']


def test_observers_remain_separate_no_union_or_consensus(tmp_path):
    paths=material(tmp_path);issues=read_json(paths[1]);second=copy.deepcopy(issues['records'][0]);second['reviewer_id']='different-observer';second['issues']=[]
    issues['records'].append(second);rebind_issue_file(paths,issues)
    scope=read_json(paths[2]);another=copy.deepcopy(scope['records'][0]);another['reviewer_id']='different-observer';another['ranges']=[dict(start=0,end=40000)]
    scope['records'].append(another);update(paths[2],scope);report=analyze(paths)
    assert len(report['rows'])==2 and report['consensus'] is None
    empty=next(r for r in report['rows'] if r['reviewer_id']=='different-observer')
    assert empty['reviewed_samples']==40000 and empty['by_type']['unintended_repeat']['affected_samples']==0


@pytest.mark.parametrize('change',['schema','pack','issue_hash','unknown_item','item_hash','placeholder','duplicate','extra',
    'origin','model','listening','audio_bool','empty_types','duplicate_types','unknown_types','music_no_audio',
    'negative','float','bool','end','reversed','range_extra','not_list'])
def test_invalid_scope_or_cross_original_binding_rejected(tmp_path,change):
    paths=material(tmp_path);doc=read_json(paths[2]);row=doc['records'][0]
    if change=='schema':doc['schema']='wrong'
    elif change=='pack':doc['pack_sha256']=object_sha('wrong')
    elif change=='issue_hash':doc['issues_file_sha256']=object_sha('wrong')
    elif change=='unknown_item':row['review_id']='unknown'
    elif change=='item_hash':row['item_sha256']=object_sha('wrong')
    elif change=='placeholder':row['reviewer_id']='FILL_REVIEWER_ID'
    elif change=='duplicate':doc['records'].append(copy.deepcopy(row))
    elif change=='extra':row['quality_score']=5
    elif change=='origin':row['origin']='other'
    elif change=='model':row.update(origin='model_judge',model_id='v1')
    elif change=='listening':row['audio_seen']=True
    elif change=='audio_bool':row['audio_seen']=1
    elif change=='empty_types':row['checked_types']=[]
    elif change=='duplicate_types':row['checked_types']*=2
    elif change=='unknown_types':row['checked_types']=['unknown']
    elif change=='music_no_audio':row['checked_types'].append('music_mismatch')
    elif change=='negative':row['ranges'][0]['start']=-1
    elif change=='float':row['ranges'][0]['start']=0.0
    elif change=='bool':row['ranges'][0]['start']=False
    elif change=='end':row['ranges'][0]['end']=81001
    elif change=='reversed':row['ranges'][0]['start']=81000
    elif change=='range_extra':row['ranges'][0]['seconds']=1
    else:row['ranges']={}
    update(paths[2],doc)
    with pytest.raises(ValueError):analyze(paths)


def test_original_issue_whitespace_change_requires_new_scope_binding(tmp_path):
    paths=material(tmp_path);paths[1].write_bytes(paths[1].read_bytes()+b' ')
    with pytest.raises(ValueError,match='original issue-file'):analyze(paths)


def test_duplicate_scope_across_files_rejected(tmp_path):
    paths=material(tmp_path)
    with pytest.raises(ValueError,match='duplicate'):
        ranges.analyze_review_ranges(paths[0],paths[1],[paths[2],paths[2]])


def test_input_drift_after_scope_snapshot_aborts(tmp_path,monkeypatch):
    paths=material(tmp_path);original=ranges._intersect
    def drift(left,right):
        result=original(left,right);paths[2].write_bytes(paths[2].read_bytes()+b' ');return result
    monkeypatch.setattr(ranges,'_intersect',drift)
    with pytest.raises(ValueError,match='source changed'):analyze(paths)


def test_half_open_adjacent_or_distant_ranges_not_double_counted():
    assert ranges._union([(5,10),(0,5),(20,25),(3,7)])==[[0,10],[20,25]]
    assert ranges._intersect([(0,5)],[(5,10)])==[]
    assert ranges._intersect([(0,10),(20,25)],[(8,22)])==[[8,10],[20,22]]


@pytest.mark.parametrize('module',[True,False])
def test_both_cli_reports_are_exclusive_and_originals_unchanged(tmp_path,module):
    paths=material(tmp_path);snapshots=[p.read_bytes() for p in paths];out=tmp_path/'report.json'
    argv=['--public',str(paths[0]),'--issues',str(paths[1]),'--ranges',str(paths[2]),'--out',str(out)]
    if module:
        ranges.main(argv)
        with pytest.raises(FileExistsError):ranges.main(argv)
    else:
        from choreobench.cli import main
        assert main(['analyze-review-ranges']+argv)==0 and main(['analyze-review-ranges']+argv)==2
    assert [p.read_bytes() for p in paths]==snapshots
    assert read_json(out)['rows'][0]['quality_pass'] is None
