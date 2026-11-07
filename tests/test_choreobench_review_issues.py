import copy
import pytest

from choreobench import review_issues as issues
from choreobench.io import object_sha, read_json, write_json
from choreobench.reviews import build_review_pack
from test_choreobench_review_sources import fixture


def material():
    original = fixture()
    original['plan']['events'][0]['end'] = 40000
    second = copy.deepcopy(original['plan']['events'][0]);second.update(event_id='e2', start=40000, end=81000)
    original['plan']['events'].append(second)
    pack = build_review_pack([original])['public']
    doc = issues.blank_issue_template(pack)
    row = doc['records'][0];row['reviewer_id'] = 'test-not-human'
    row['issues'] = [dict(issue_id='i1', type='abrupt_transition', severity='major', start=39999, end=40001,
        event_ids=[e['event_id'] for e in pack['items'][0]['plan']['events']],
        evidence='Step left, then another step.', reason='Synthetic test, not an actual quality observation.')]
    return pack, doc


def test_blank_template_contains_no_scores_and_no_auto_findings():
    pack, _ = material();doc = issues.blank_issue_template(pack)
    assert doc['pack_sha256'] == object_sha(pack)
    assert all(not row['issues'] and not row['audio_seen'] for row in doc['records'])
    assert 'ratings' not in str(doc)
    with pytest.raises(ValueError):issues.validate_review_issues(pack, doc)


def test_integer_half_open_cross_event_observation_preserved_without_retiming():
    pack, doc = material();before = copy.deepcopy(doc)
    assert issues.validate_review_issues(pack, doc) == before and doc == before
    row = doc['records'][0];issue = row['issues'][0]
    issue.update(start=40000, end=81000, event_ids=[pack['items'][0]['plan']['events'][1]['event_id']])
    assert issues.validate_review_issues(pack, doc) == doc
    issue['event_ids'].append(pack['items'][0]['plan']['events'][0]['event_id'])
    with pytest.raises(ValueError, match='intersect'):issues.validate_review_issues(pack, doc)


@pytest.mark.parametrize('kind,severity', [('reasonable_recurrence','not_applicable'),('uncertain','uncertain'),
    ('ambiguous_action','minor'),('unintended_repeat','major'),('overdense_action','major')])
def test_positive_recurrence_and_uncertainty_remain_distinct(kind, severity):
    pack, doc = material();doc['records'][0]['issues'][0].update(type=kind, severity=severity)
    assert issues.validate_review_issues(pack, doc) == doc


def test_music_observation_requires_audio_declaration_but_does_not_prove_listening():
    pack, doc = material();row = doc['records'][0];row['issues'][0]['type'] = 'music_mismatch'
    with pytest.raises(ValueError, match='audio_seen'):issues.validate_review_issues(pack, doc)
    row['audio_seen'] = True
    assert issues.validate_review_issues(pack, doc) == doc


@pytest.mark.parametrize('change', ['schema','pack','item','extra','record_extra','unknown_item','placeholder',
    'row_duplicate','identity','audio_bool','issues_list','issue_extra','issue_duplicate','issue_id','type',
    'severity','positive_severity','uncertain_severity','negative','end','reversed','empty_interval','float',
    'bool_sample','event_unknown','event_duplicate','events_empty','event_type','nonintersect','evidence','reason'])
def test_forged_or_incompatible_declarations_are_rejected(change):
    pack, doc = material();row = doc['records'][0];entry = row['issues'][0]
    if change=='schema':doc['schema']='wrong'
    elif change=='pack':doc['pack_sha256']=object_sha('wrong')
    elif change=='item':row['item_sha256']=object_sha('wrong')
    elif change=='extra':doc['score']=1
    elif change=='record_extra':row['ratings']={}
    elif change=='unknown_item':row['review_id']='unknown'
    elif change=='placeholder':row['reviewer_id']='FILL_REVIEWER_ID'
    elif change=='row_duplicate':doc['records'].append(copy.deepcopy(row))
    elif change=='identity':row['model_id']='model'
    elif change=='audio_bool':row['audio_seen']=1
    elif change=='issues_list':row['issues']={}
    elif change=='issue_extra':entry['seconds']=2.5
    elif change=='issue_duplicate':row['issues'].append(copy.deepcopy(entry))
    elif change=='issue_id':entry['issue_id']='../private'
    elif change=='type':entry['type']='unknown'
    elif change=='severity':entry['severity']='fatal'
    elif change=='positive_severity':entry['type']='reasonable_recurrence'
    elif change=='uncertain_severity':entry['type']='uncertain'
    elif change=='negative':entry['start']=-1
    elif change=='end':entry['end']=81001
    elif change=='reversed':entry['start']=40002
    elif change=='empty_interval':entry['end']=entry['start']
    elif change=='float':entry['start']=39999.0
    elif change=='bool_sample':entry['start']=True
    elif change=='event_unknown':entry['event_ids']=['original-private-id']
    elif change=='event_duplicate':entry['event_ids']*=2
    elif change=='events_empty':entry['event_ids']=[]
    elif change=='event_type':entry['event_ids']=[1]
    elif change=='nonintersect':entry['end']=40000
    elif change=='evidence':entry['evidence']=' '
    else:entry['reason']=None
    with pytest.raises(ValueError):issues.validate_review_issues(pack, doc)


def test_same_reviewer_cannot_change_origin_across_public_items():
    first = fixture();second = fixture();second['song']['song_id']=second['plan']['song_id']='second'
    pack = build_review_pack([first,second])['public'];doc=issues.blank_issue_template(pack)
    for row in doc['records']:row['reviewer_id']='test-observer'
    doc['records'][1].update(origin='model_judge',model_id='model-v1')
    with pytest.raises(ValueError,match='identity changed'):issues.validate_review_issues(pack,doc)


def test_missing_records_and_empty_lists_never_become_good_quality(tmp_path):
    pack, doc = material();public=tmp_path/'pack.json';path=tmp_path/'issues.json'
    write_json(public,pack);doc['records']=[];write_json(path,doc)
    report=issues.check_review_issues(public,path)
    assert report['declared_observations']==0 and report['items_without_observer_record']
    assert not report['quality_verified'] and not report['reviewer_identity_or_listening_authenticated']
    row=issues.blank_issue_template(pack)['records'][0];row['reviewer_id']='test-empty'
    doc['records']=[row];write_json(path,doc,overwrite=True)
    report=issues.check_review_issues(public,path)
    assert not report['items_without_observer_record'] and not report['quality_verified']
    assert report['records'][0]['issues']==[]


def test_input_drift_after_validation_rejected(tmp_path,monkeypatch):
    pack,doc=material();public=tmp_path/'pack.json';path=tmp_path/'issues.json'
    write_json(public,pack);write_json(path,doc);validate=issues.validate_review_issues
    def drift(p,d):
        validate(p,d);path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(issues,'validate_review_issues',drift)
    with pytest.raises(ValueError,match='source changed'):issues.check_review_issues(public,path)


@pytest.mark.parametrize('module',[True,False])
def test_cli_exclusive_reports_keep_original_records(tmp_path,module):
    pack,doc=material();public=tmp_path/'pack.json';path=tmp_path/'issues.json';out=tmp_path/'report.json'
    write_json(public,pack);write_json(path,doc)
    args=['--public',str(public),'--issues',str(path),'--out',str(out)]
    if module:
        issues.main(args)
        with pytest.raises(FileExistsError):issues.main(args)
    else:
        from choreobench.cli import main
        assert main(['check-review-issues']+args)==0
        assert main(['check-review-issues']+args)==2
    report=read_json(out)
    assert report['records']==doc['records'] and report['declared_observations']==1
    assert not report['quality_verified'] and read_json(path)==doc
