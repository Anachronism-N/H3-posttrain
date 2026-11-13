import copy
import pytest
from captionlab import motion_facts as m


def fact(t,id,q):
 s=t.index(q);return dict(id=id,action=q,body_part='arm',side='unspecified',side_frame='unspecified',start=s,end=s+len(q),quote=q)
def extract(t,fs,rs=()):return m.bind_extraction(t,dict(facts=fs,relations=list(rs),uncertainties=[]))
def pair(a,b,status):return dict(candidate_id=a['id'],reference_id=b['id'],status=status,reason='Synthetic semantic fixture.',candidate_quote=a['quote'],reference_quote=b['quote'])
def relation(t,a,b,k):return dict(**{'from':a,'to':b},kind=k,start=0,end=len(t),quote=t)


def test_open_predicate_equivalence_not_fixed_keyword_matching():
 c=extract('Unfurls.',[fact('Unfurls.','c','Unfurls')]);r=extract('Opens.',[fact('Opens.','r','Opens')])
 v=m.bind_matches(c,r,[pair(c['facts'][0],r['facts'][0],'match')]);s=m.score_facts(c,r,v)
 assert s['known_fact_metrics']['f1']==1 and s['musical_fit'] is None and not s['extractor_or_matcher_accuracy_verified']


def test_maximum_one_to_one_and_unknown_bounds():
 c=extract('Lift then repeat.',[fact('Lift then repeat.','a','Lift'),fact('Lift then repeat.','b','repeat')]);r=extract('Rise then repeat.',[fact('Rise then repeat.','x','Rise'),fact('Rise then repeat.','y','repeat')])
 yes={('a','x'),('a','y'),('b','x')};v=m.bind_matches(c,r,[pair(a,b,'match' if (a['id'],b['id']) in yes else 'no_match') for a in c['facts'] for b in r['facts']]);s=m.score_facts(c,r,v)
 assert s['known_fact_metrics']['matched']==2 and s['candidate_to_reference']=={'a':'y','b':'x'}
 v=m.bind_matches(c,r,[pair(a,b,'unknown') for a in c['facts'] for b in r['facts']]);s=m.score_facts(c,r,v)
 assert s['known_fact_metrics']['matched']==0 and s['potential_fact_metrics']['matched']==2
 r=extract('Rise.',[fact('Rise.','x','Rise')]);v=m.bind_matches(c,r,[pair(a,r['facts'][0],'match') for a in c['facts']]);s=m.score_facts(c,r,v)
 assert s['known_fact_metrics']['precision']==.5 and s['fact_alignment_ambiguous']


@pytest.mark.parametrize('kind',['before','simultaneous'])
def test_relations_and_ambiguous_alignment(kind):
 t='Lift then fold.';u='Rise then bend.'
 c=extract(t,[fact(t,'a','Lift'),fact(t,'b','fold')],[relation(t,'a','b',kind)])
 r=extract(u,[fact(u,'x','Rise'),fact(u,'y','bend')],[relation(u,'y' if kind=='simultaneous' else 'x','x' if kind=='simultaneous' else 'y',kind)])
 v=m.bind_matches(c,r,[pair(a,b,'match' if (a['id'],b['id']) in {('a','x'),('b','y')} else 'no_match') for a in c['facts'] for b in r['facts']]);assert m.score_facts(c,r,v)['relation_metrics']['f1']==1
 v=m.bind_matches(c,r,[pair(a,b,'match') for a in c['facts'] for b in r['facts']]);assert m.score_facts(c,r,v)['relation_metrics']['f1'] is None


def test_image_anatomical_not_assumed_equal():
 t='Left wrist rises.';a=fact(t,'a',t);a.update(side='left',side_frame='anatomical');b=dict(a,id='b',side_frame='image')
 c=extract(t,[a]);r=extract(t,[b])
 with pytest.raises(ValueError,match='side equivalence'):m.bind_matches(c,r,[pair(a,b,'match')])


@pytest.mark.parametrize('change',['quote','bounds','duplicate','frame','cycle'])
def test_spans_and_occurrences(change):
 t='Lift then fold.';a=fact(t,'a','Lift');b=fact(t,'b','fold');v=dict(facts=[a,b],relations=[],uncertainties=[])
 if change=='quote':a['quote']='Fake'
 elif change=='bounds':a['start']=True
 elif change=='duplicate':v['facts'].append(copy.deepcopy(a))
 elif change=='frame':a['side_frame']='unknown'
 else:v['relations']=[relation(t,'a','b','before'),relation(t,'b','a','before')]
 with pytest.raises(ValueError):m.bind_extraction(t,v)


def test_empty_not_perfect_and_all_pair_slots_required():
 c=extract('Unknown.',[]);r=extract('Unspecified.',[]);assert m.score_facts(c,r,m.bind_matches(c,r,[]))['known_fact_metrics']['f1'] is None
 c=extract('Lift.',[fact('Lift.','a','Lift')]);r=extract('Rise.',[fact('Rise.','b','Rise')])
 with pytest.raises(ValueError,match='Cartesian'):m.bind_matches(c,r,[])
 v=m.bind_matches(c,r,[pair(c['facts'][0],r['facts'][0],'match')]);c['facts'][0]['action']='tampered'
 with pytest.raises(ValueError):m.score_facts(c,r,v)



def test_unknown_relation_endpoint_not_definite_zero():
 t='Lift then fold.';u='Rise then bend.'
 c=extract(t,[fact(t,'a','Lift'),fact(t,'b','fold')],[relation(t,'a','b','before')])
 r=extract(u,[fact(u,'x','Rise'),fact(u,'y','bend')],[relation(u,'x','y','before')])
 rows=[pair(a,b,'match' if (a['id'],b['id'])==('a','x') else 'unknown' if (a['id'],b['id'])==('b','y') else 'no_match') for a in c['facts'] for b in r['facts']]
 result=m.score_facts(c,r,m.bind_matches(c,r,rows))
 assert result['relation_metrics']['f1'] is None and result['relation_metrics']['status']=='unknown_unresolved_fact_alignment'



def test_quote_occurrence_compiles_unicode_and_actual_repeat():
 source='抬臂。 Arm lifts. Arm lifts.'
 f=fact(source,'a','Arm lifts.');f.pop('start');f.pop('end');f['occurrence']=1
 value=m.bind_quote_extraction(source,dict(facts=[f],relations=[],uncertainties=[]))
 assert value['facts'][0]['start']==source.rindex('Arm lifts.') and value['facts'][0]['end']==len(source)
 assert 'occurrence' not in value['facts'][0]


@pytest.mark.parametrize('change',['missing','paraphrase','negative','boolean'])
def test_quote_occurrence_never_repairs_nonliteral_evidence(change):
 source='Arm lifts.';f=fact(source,'a',source);f.pop('start');f.pop('end');f['occurrence']=0
 if change=='missing':f['occurrence']=1
 elif change=='paraphrase':f['quote']='Arm rises.'
 elif change=='negative':f['occurrence']=-1
 else:f['occurrence']=False
 with pytest.raises(ValueError):m.bind_quote_extraction(source,dict(facts=[f],relations=[],uncertainties=[]))



def test_unknown_alternative_pair_keeps_relation_alignment_unresolved():
 t='Lift then fold.';u='Rise then bend.'
 c=extract(t,[fact(t,'a','Lift'),fact(t,'b','fold')],[relation(t,'a','b','before')])
 r=extract(u,[fact(u,'x','Rise'),fact(u,'y','bend')],[relation(u,'x','y','before')])
 rows=[pair(a,b,'match' if (a['id'],b['id']) in {('a','x'),('b','y')} else 'unknown') for a in c['facts'] for b in r['facts']]
 result=m.score_facts(c,r,m.bind_matches(c,r,rows))
 assert result['known_fact_metrics']['f1']==1 and result['relation_metrics']['f1'] is None
 assert result['relation_metrics']['status']=='unknown_unresolved_fact_alignment'
