"""Extractor+fixed-NLI one-to-one reference graph agreement, not music quality."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from captionlab.nli_reference import NLI
from captionlab.graph_metrics import SUPPORTED_PARTS,prf,relation_set


def proposition(node):
 side={'both':'both','left':'the left','right':'the right','unspecified':'the'}[node['side']]
 direction='' if node['direction']=='unspecified' else ' '+node['direction']
 return side+' '+node['body_part'].replace('_',' ')+' '+('does not ' if node['negated'] else '')+node['action']+direction+'.'


def compatible(a,b):
 if a['negated']!=b['negated']:return False
 if a['side'] in ('left','right') and b['side'] in ('left','right') and a['side']!=b['side']:return False
 opposites=[{'left','right'},{'up','down'},{'forward','backward'},{'inward','outward'},{'clockwise','counterclockwise'}]
 if any({a['direction'],b['direction']}==v for v in opposites):return False
 return a['body_part']==b['body_part']


def evaluate(reference,candidate,nli,threshold=.7):
 refs=[n for n in reference['nodes'] if n['body_part'] in SUPPORTED_PARTS]
 props=[n for n in candidate['nodes'] if n['body_part'] in SUPPORTED_PARTS]
 pairs=[];cells=[]
 for i,a in enumerate(props):
  for j,b in enumerate(refs):
   if compatible(a,b):cells.append((i,j));pairs.extend([(proposition(a),proposition(b)),(proposition(b),proposition(a))])
 probs=nli.score(pairs) if pairs else [];weight=np.zeros((len(props),len(refs)),dtype=float)
 for k,(i,j) in enumerate(cells):
  a,b=probs[2*k:2*k+2]
  if a is not None and b is not None:weight[i,j]=min(a['entailment'],b['entailment'])
 # Dummy rows/columns allow arbitrary unmatched nodes; count valid matches first,
 # then weight among equally sized one-to-one alignments.
 n=max(len(props),len(refs));objective=np.zeros((n,n))
 if n:
  objective[:len(props),:len(refs)]=np.where(weight>=threshold,1+weight/(n+1),0)
  chosen=linear_sum_assignment(-objective);matches=[(i,j) for i,j in zip(*chosen) if i<len(props) and j<len(refs) and weight[i,j]>=threshold]
 else:matches=[]
 mapping={props[i]['id']:refs[j]['id'] for i,j in matches};fact=prf(len(matches),len(props),len(refs))
 refrel,refcycles=relation_set(reference,{n['id'] for n in refs});candrel,candcycles=relation_set(candidate,{n['id'] for n in props});aligned=set()
 for a,b,r in candrel:
  if a in mapping and b in mapping:
   x,y=mapping[a],mapping[b];aligned.add((min(x,y),max(x,y),r) if r=='concurrent' else (x,y,r))
 rel=prf(len(aligned&refrel),len(candrel),len(refrel))
 return dict(facts=fact,relations=rel,mapping=mapping,alignment_weights=[dict(candidate=props[i]['id'],reference=refs[j]['id'],bidirectional_entailment=weight[i,j]) for i,j in matches],threshold=threshold,reference_cycles=refcycles,candidate_cycles=candcycles,scope='supported body parts, model-extracted facts with fixed NLI aligner; no truth claim')


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.out.mkdir(exist_ok=False)
 nli=NLI(a.model,a.device);index={x['text_id']:x for x in [json.loads(l) for l in a.graphs.open() if l.strip()]};bridge=json.loads(a.bridge.read_text());results=[]
 for pair in json.loads(a.pairs.read_text())['pairs']:
  x=index.get(bridge[pair['original']]);y=index.get(bridge[pair['variant']]);row=dict(case_id=pair['case_id'],kind=pair['kind'],status='unknown')
  if x and y and x['status']=='valid' and y['status']=='valid':row.update(status='computed',metrics=evaluate(x['graph'],y['graph'],nli))
  results.append(row)
 (a.out/'controls.json').write_text(json.dumps(dict(results=results,model=str(a.model),threshold=.7,threshold_status='development_default_not_formal_frozen',extractor_dependent=True,quality_validated=False),indent=2)+'\n')
 print(json.dumps(dict(completed=True,controls=len(results))),flush=True)


if __name__=='__main__':main()
