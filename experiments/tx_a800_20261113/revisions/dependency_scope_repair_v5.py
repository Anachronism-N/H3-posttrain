"""Non-generative, token-anchored motion agreement; conservative coreference scope."""
import argparse,json,re,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean
from captionlab.graph_metrics import evaluate_graph

PARTS={'arm','forearm','hand','wrist','elbow','knee'}
EQUIVALENTS={'lift':'raise','rise':'raise','drop':'lower','descend':'lower','flex':'bend','fold':'bend','straighten':'extend','turn':'rotate','bring':'move','gather':'move','send':'move'}
DIRECTIONS={'up':'up','upward':'up','upwards':'up','down':'down','downward':'down','downwards':'down','left':'left','leftward':'left','right':'right','rightward':'right','forward':'forward','forwards':'forward','backward':'backward','backwards':'backward','back':'backward','outward':'outward','outwards':'outward','inward':'inward','inwards':'inward','clockwise':'clockwise','counterclockwise':'counterclockwise'}


def part_name(token):
 p=token.lemma_.lower()
 if p not in PARTS:return None
 if p=='arm' and any(x.lower_=='upper' for x in token.children):return 'upper_arm'
 return p


def side(token):
 words={t.lower_ for t in token.children if t.dep_ in ('amod','det','poss','compound')}
 if token.head.lower_ in DIRECTIONS:words.update(t.lower_ for t in token.head.children if t.dep_ in ('amod','det') and t!=token)
 words.update(x.lower_ for x in token.doc[max(token.sent.start,token.i-2):token.i] if x.lower_ in ('both','left','right'))
 return 'both' if 'both' in words else 'left' if 'left' in words else 'right' if 'right' in words else 'unspecified'


def extract(doc):
 anchors=defaultdict(list);issues=[];nodes=[]
 for token in doc:
  part=part_name(token)
  if not part:continue
  owner=token.head
  boundary=next((x.i for x in doc[token.i+1:token.sent.end] if x.lower_ in ('before','after','while','as','then','and') or x.text in (',',';','.')),token.sent.end)
  if owner.i>=boundary or owner==token:
   trailing=[x for x in doc[token.i+1:boundary] if x.pos_=='VERB' or x.lemma_.lower() in set(EQUIVALENTS)|set(EQUIVALENTS.values()) or x.lower_ in ('lower','bend','extend','rotate','move','raise','open','closed','bent','extended','still','motionless','relaxed','straight')]
   if trailing:owner=trailing[0]
   elif owner.i>=boundary:
    issues.append(dict(token=token.text,offset=token.idx,reason='Predicate attached outside explicit clause; no guess'))
    continue
  if token.dep_ in ('dobj','obj','nsubj','nsubjpass') and owner!=token:
   anchors[owner.i].append((token,part))
  elif token.dep_=='appos' and owner.lemma_.lower() in set(EQUIVALENTS)|set(EQUIVALENTS.values()):
   anchors[owner.i].append((token,part))
  elif owner!=token.head:
   anchors[owner.i].append((token,part))
  elif token.dep_=='compound' and owner.lower_ in DIRECTIONS and owner.head!=owner:
   anchors[owner.head.i].append((token,part))
  elif token.dep_=='compound' and owner.pos_ in ('NOUN','VERB'):anchors[owner.i].append((token,part))
  elif token.dep_=='conj' and token.head.i in {t.i for values in anchors.values() for t,_ in values}:
   anchors[token.head.head.i].append((token,part))
  elif token.dep_=='pobj' and token.head.lower_ in ('before','after','while','as'):
   trailing=[x for x in token.children if x.i>token.i and (x.pos_ in ('VERB','ADJ','ADV') or x.lemma_.lower() in set(EQUIVALENTS)|set(EQUIVALENTS.values())) and x.lower_ not in DIRECTIONS]
   if trailing:anchors[trailing[0].i].append((token,part))
   else:issues.append(dict(token=token.text,offset=token.idx,reason='Unresolved nominal temporal clause'))
  elif token.dep_ not in ('pobj','poss','appos'):
   issues.append(dict(token=token.text,offset=token.idx,reason='No directly attached motion predicate; not completed by a language model'))
 # An unambiguous conjoined predicate inherits the explicitly named same subject.
 for token in doc:
  if token.i in anchors or token.lower_ in DIRECTIONS or token.lemma_.lower() in ('let','make','do') or (token.pos_!='VERB' and token.lemma_.lower() not in set(EQUIVALENTS)|set(EQUIVALENTS.values())) or token.dep_ not in ('conj','dep'):continue
  subjects=[x for x in token.children if x.dep_ in ('nsubj','nsubjpass')]
  objects=[x for x in token.children if x.dep_ in ('dobj','obj','compound')]
  if any(x.lower_ not in ('it','them') and part_name(x) is None for x in objects):continue
  if not subjects and token.head.i in anchors:anchors[token.i]=list(anchors[token.head.i])
  elif any(x.lower_ in ('it','them') for x in subjects) and token.head.i in anchors and len(anchors[token.head.i])==1:anchors[token.i]=list(anchors[token.head.i])
 for sent in doc.sents:
  unique={(part,side(subject)) for vi,values in anchors.items() if sent.start<=vi<sent.end for subject,part in values}
  if len(unique)!=1:continue
  for token in sent:
   if token.i in anchors or token.lemma_.lower() not in set(EQUIVALENTS)|set(EQUIVALENTS.values()):continue
   previous=[vi for vi in anchors if sent.start<=vi<token.i]
   if not previous:continue
   vi=max(previous)
   if not any(x.lower_=='then' for x in doc[vi+1:token.i]):continue
   real_subjects=[x for x in token.children if x.dep_ in ('nsubj','nsubjpass') and x.lemma_.lower() not in set(EQUIVALENTS)|set(EQUIVALENTS.values())]
   if not real_subjects:anchors[token.i]=list(anchors[vi])
 by_verb=defaultdict(list)
 for vi,parts in sorted(anchors.items()):
  verb=doc[vi];lemma=verb.lower_ if verb.lower_ in set(EQUIVALENTS)|set(EQUIVALENTS.values()) else verb.lemma_.lower();action=EQUIVALENTS.get(lemma,lemma);local=[]
  for child in verb.children:
   if child.i in anchors:continue
   if child.dep_ in ('nsubj','nsubjpass','dobj','obj'):
    if child.lower_ in DIRECTIONS and child.i not in {p.i for p,_ in parts}:local.append(child)
    local.extend(t for t in child.children if t.dep_=='acl' and t.lower_ in DIRECTIONS)
    continue
   if child.lower_ in DIRECTIONS and child.i>min(p.i for p,_ in parts):local.append(child)
   if child.dep_ in ('advmod','prt','prep','acomp','oprd','neg','ccomp'):
    if child.dep_=='prep':local.extend(t for t in child.subtree if t.i not in anchors and not any(a.i in anchors for a in t.ancestors if a!=verb and a!=child))
    elif child.pos_!='VERB':local.append(child)
  boundary=next((x.i for x in doc[vi+1:verb.sent.end] if x.lower_ in ('before','after','while','as','then') or x.lower_=='and' and not (x.i+1<verb.sent.end and doc[x.i+1].lower_ in DIRECTIONS) or x.text in (',',';','.')),verb.sent.end)
  boundary=min(boundary,next((x for x in sorted(anchors) if vi<x<boundary),boundary))
  directional_tokens=[x for x in doc[vi+1:boundary] if x.lower_ in DIRECTIONS and not (x.i+1<len(doc) and part_name(doc[x.i+1])) and not (x.dep_=='amod' and x.head!=verb and part_name(x.head) is None) and not (x.lower_=='back' and x.i+1<boundary and doc[x.i+1].lower_ in ('left','right','leftward','rightward'))]
  directions=[DIRECTIONS[x.lower_] for x in directional_tokens]
  local=[x for x in local if vi<x.i<boundary]
  local_text=' '.join(t.lower_ for t in local)
  if any(t.lower_ in ('side','sides') for t in local) and any(t.lower_ in ('out','to') for t in local):directions.append('outward')
  state=[t.lower_ for t in local if t.lower_ in ('bent','extended','straight','still','motionless','relaxed')]
  if verb.lemma_.lower() in ('keep','hold','maintain','remain') and state:action='hold '+('still' if state[0]=='motionless' else 'extended' if state[0]=='straight' else state[0])
  negated=any(t.dep_=='neg' or t.lower_ in ('never','not') for t in verb.children)
  for subject,part in parts:
   stated_state=[x.lower_ for x in subject.children if x.lower_ in ('bent','extended','straight','still','motionless','relaxed')]
   node_action='hold '+('still' if stated_state[0]=='motionless' else 'extended' if stated_state[0]=='straight' else stated_state[0]) if verb.lemma_.lower() in ('pose','stance','posture') and stated_state else action
   for direction in dict.fromkeys(directions or ['unspecified']):
    node=dict(id='n'+str(len(nodes)),body_part=part,side=side(subject),action=node_action,direction=direction,negated=negated,quote=verb.sent.text,predicate_offset=verb.idx,part_offset=subject.idx)
    nodes.append(node);by_verb[vi].append(node['id'])
 edges=set()
 for vi in by_verb:
  verb=doc[vi];marker=None;parent=verb.head
  if parent.lower_ in ('before','after','while','as'):
   marker=parent.lower_;parent=parent.head
  else:
   markers=[x.lower_ for x in verb.children if x.dep_=='mark' and x.lower_ in ('before','after','while','as')]
   if markers:marker=markers[0]
  if marker is None:
   temporal=next((a for a in verb.ancestors if a.lower_ in ('before','after','while','as')),None)
   if temporal is not None:marker=temporal.lower_;parent=temporal.head
  if marker and parent.i not in by_verb:
   for ancestor in parent.ancestors:
    if ancestor.i in by_verb:parent=ancestor;break
  if marker and parent.i in by_verb and parent.i!=vi:
   for a in by_verb[parent.i]:
    for b in by_verb[vi]:edges.add((b,a,'before') if marker=='after' else (a,b,'before') if marker=='before' else (min(a,b),max(a,b),'concurrent'))
 for sent in doc.sents:
  verbs=sorted(v for v in by_verb if sent.start<=v<sent.end)
  if len(verbs)==2:
   left,right=verbs;markers=[x.lower_ for x in doc[left+1:right] if x.lower_ in ('before','after','while','as','then')]
   if len(markers)==1:
    relation=markers[0]
    for a in by_verb[left]:
     for b in by_verb[right]:edges.add((b,a,'before') if relation=='after' else (a,b,'before') if relation in ('before','then') else (min(a,b),max(a,b),'concurrent'))
  for token in sent:
   if token.lower_!='then':continue
   earlier=[v for v in verbs if v<token.i];later=[v for v in verbs if v>token.i]
   if not earlier or not later:continue
   # THEN orders the preceding and following explicit phases within one sentence;
   # no before edges are added merely from punctuation or sentence order.
   for left in earlier:
    for right in later:
     for a in by_verb[left]:
      for b in by_verb[right]:edges.add((a,b,'before'))
 graph=dict(nodes=nodes,edges=[dict(source=a,target=b,relation=r) for a,b,r in sorted(edges)])
 return dict(graph=graph,unsupported_mentions=issues,scope='token-anchored explicit supported parts; no anatomical side inferred from travel direction; implicit cross-sentence coreference not supplied',generator=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--items',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--controls',type=Path);p.add_argument('--bindings',type=Path);a=p.parse_args();a.out.mkdir(exist_ok=False)
 import spacy,en_core_web_sm
 nlp=en_core_web_sm.load(disable=['ner']);items=json.loads(a.items.read_text())['items'];rows=[]
 with (a.out/'graphs.jsonl').open('x') as f:
  for item,doc in zip(items,nlp.pipe((x['text'] for x in items),batch_size=32)):
   row=dict(text_id=item['text_id'],status='computed',**extract(doc));rows.append(row);f.write(json.dumps(row)+chr(10))
 index={r['text_id']:r for r in rows}
 receipt=dict(registered_unique=len(items),spacy=spacy.__version__,parser=en_core_web_sm.__version__,generator=False,code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_texts_complete=True,silent_truncation=False,scope='non-generative dependency-rule extraction; not choreography/video/music correctness; unresolved coreference excluded explicitly',canonical_equivalents=EQUIVALENTS)
 if a.controls:
  controls=[]
  for pair in json.loads(a.controls.read_text())['pairs']:
   for kind,tid in pair['ids'].items():controls.append(dict(case_id=pair['case_id'],kind=kind,metrics=evaluate_graph(pair['expected_graph'],index[tid]['graph'])))
  stats={k:{d:mean(x['metrics'][d]['f1'] for x in controls if x['kind']==k) for d in ('facts','relations')} for k in ('original','paraphrase','reversed','concurrent')}
  receipt['controls']=dict(stats=stats,results=controls,accepted=all(stats[k]['facts']>=.9 for k in ('original','paraphrase','reversed','concurrent')) and stats['original']['relations']>=.9 and stats['paraphrase']['relations']>=.9 and stats['reversed']['relations']<=.25 and stats['concurrent']['relations']<=.25)
 if a.bindings:
  results=[]
  for pair in json.loads(a.bindings.read_text())['bindings']:
   row={k:pair[k] for k in ('case_id','method_id','source_group_id','item_sha256')}
   if pair['generation_status']!='ok':row['status']='generation_not_evaluable'
   else:
    ref=index[pair['reference_text_id']];cand=index[pair['candidate_text_id']]
    if not ref['graph']['nodes']:row['status']='reference_no_measurable_explicit_facts'
    else:row.update(status='computed',metrics=evaluate_graph(ref['graph'],cand['graph']))
   results.append(row)
  with (a.out/'records.jsonl').open('x') as f:
   for row in results:f.write(json.dumps(row)+chr(10))
  receipt['methods']={}
  for m in sorted({x['method_id'] for x in results}):
   group=[x for x in results if x['method_id']==m];valid=[x for x in group if x['status']=='computed'];record=dict(registered=len(group),statuses=dict(Counter(x['status'] for x in group)))
   for dim in ('facts','relations'):
    v=[x['metrics'][dim]['f1'] for x in valid if not x['metrics'][dim]['empty_both']];record[dim+'_macroF1']=dict(n=len(v),mean=mean(v) if v else None)
   receipt['methods'][m]=record
 (a.out/'summary.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps({k:v for k,v in receipt.items() if k not in ('controls','canonical_equivalents')},indent=2))
 if 'controls' in receipt:print('CONTROL_STATS',json.dumps(receipt['controls']['stats']),'accepted',receipt['controls']['accepted'])


if __name__=='__main__':main()
