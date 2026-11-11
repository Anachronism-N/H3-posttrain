"""Explicit model-review contracts, not human calibration data."""
import copy,json
from types import SimpleNamespace
import pytest
from choreobench import fact_review
from choreobench.io import write_json,read_json


@pytest.fixture
def item():
 return dict(item_id='review-1',source_action='Stomp sharply while hands strike downward.',
  candidate_action='Stomp sharply while wrists circle softly.',
  instruction='Change hand strikes to circular wrists; retain sharp stomps.',
  protected_facts=[dict(fact_id='foot',description='Sharp percussive stomps remain explicit.',source_evidence='Stomp sharply')],
  private_expected=True,private_source=dict(control_type='known-retain'))


def verdict(item,status='retained',evidence='Stomp sharply'):
 return dict(item_id=item['item_id'],goal='pass',facts=[dict(fact_id='foot',status=status,evidence=evidence)],
  new_textual_conflict='no',reason='Synthetic review contract only, not human evidence.')


def test_fact_retention_and_unknown_have_different_joint_status(item):
 r=fact_review.parse_review(item,json.dumps(verdict(item)))
 assert r['protected_fact_preservation'] is True and r['text_joint_success']
 for status in ('omitted','contradicted','unknown'):
  r=fact_review.parse_review(item,json.dumps(verdict(item,status,'Stomp sharply' if status=='contradicted' else '')))
  assert not r['text_joint_success']
  assert (r['protected_fact_preservation'] is None)==(status=='unknown')


def test_invented_quote_or_unknown_fact_cannot_bind(item):
 v=verdict(item,evidence='This did not appear in action.')
 with pytest.raises(ValueError,match='exact span'):fact_review.parse_review(item,json.dumps(v))
 v=verdict(item);v['facts'][0]['fact_id']='private'
 with pytest.raises(ValueError):fact_review.parse_review(item,json.dumps(v))


@pytest.mark.parametrize('status',['retained','contradicted'])
@pytest.mark.parametrize('evidence',['',' ',' '])
def test_retained_or_contradicted_cannot_pass_without_candidate_quote(item,status,evidence):
 with pytest.raises(ValueError,match='nonempty candidate evidence'):
  fact_review.parse_review(item,json.dumps(verdict(item,status,evidence)))


def test_unknown_does_not_hide_a_known_omission(item):
 item['protected_facts'].append(dict(fact_id='hands',description='Hand strikes',source_evidence='hands strike downward'))
 v=verdict(item,'omitted','')
 v['facts'].append(dict(fact_id='hands',status='unknown',evidence=''))
 r=fact_review.parse_review(item,json.dumps(v))
 assert r['protected_fact_preservation'] is False and r['text_joint_success'] is False


def test_structured_schema_matches_conditional_evidence_contract(item):
 import re
 schema=fact_review._schema(item)['properties']['facts']['prefixItems'][0]
 condition=schema['allOf'][0]
 assert condition['if']['properties']['status']['enum']==['retained','contradicted']
 evidence=condition['then']['properties']['evidence']
 assert evidence['minLength']==1 and not re.search(evidence['pattern'],' ')
 assert re.search(evidence['pattern'],'Stomp sharply')


def test_control_labels_never_sent_and_resume_no_new_calls(item,tmp_path,monkeypatch):
 inv=tmp_path/'inventory.json';write_json(inv,dict(schema='choreobench.fact_review_inventory.v1',seed=7,items=[item]))
 calls=[];identity=dict(kind='replay',synthetic=True,fixture='fake-review')
 def generate(request):
  calls.append(copy.deepcopy(request));p=json.loads(request['messages'][-1]['content'])
  assert set(p)=={'item_id','source_action','candidate_action','instruction','protected_facts'}
  return dict(text=json.dumps(verdict(item)),raw={},identity=identity,synthetic=True,stop_reason='stop',costs=dict(input_tokens=5,output_tokens=5,latency_seconds=0))
 monkeypatch.setattr(fact_review,'create_provider',lambda *a,**kw:SimpleNamespace(identity=identity,generate=generate))
 out=tmp_path/'runs';r=fact_review.run_fact_review(inv,dict(kind='replay'),out)
 assert r['complete'] and r['calls_used']==1 and r['human_reviews']==0 and not r['audio_seen']
 assert not r['formal_metric_validity_verified']
 assert fact_review.run_fact_review(inv,dict(kind='replay'),out,resume=True)==r and len(calls)==1
 raw=next((out/'calls').glob('*.raw.json'));raw.unlink()
 with pytest.raises(ValueError,match='missing/changed'):fact_review.run_fact_review(inv,dict(kind='replay'),out,resume=True)
 assert len(calls)==1


def test_fact_source_evidence_must_be_in_original_before_provider(item,tmp_path,monkeypatch):
 item['protected_facts'][0]['source_evidence']='Invented support foot.'
 inv=tmp_path/'inventory.json';write_json(inv,dict(schema='choreobench.fact_review_inventory.v1',seed=7,items=[item]))
 monkeypatch.setattr(fact_review,'create_provider',lambda *a,**kw:pytest.fail('provider before evidence validation'))
 with pytest.raises(ValueError,match='source action'):fact_review.run_fact_review(inv,dict(kind='replay'),tmp_path/'runs')



@pytest.mark.parametrize('bad_id',['../outside','/absolute/path','nested/item'])
def test_review_display_id_cannot_escape_output_directory(item,tmp_path,monkeypatch,bad_id):
 item['item_id']=bad_id
 inv=tmp_path/'inventory.json';write_json(inv,dict(schema='choreobench.fact_review_inventory.v1',seed=7,items=[item]))
 monkeypatch.setattr(fact_review,'create_provider',lambda *a,**kw:pytest.fail('invalid ID before provider'))
 with pytest.raises(ValueError):fact_review.run_fact_review(inv,dict(kind='replay'),tmp_path/'runs')
 assert not (tmp_path/'runs').exists()
