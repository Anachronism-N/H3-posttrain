"""Frozen music-evidence summary and operational expert selection, not human truth."""
import argparse,json,itertools,collections
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scores',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 mapping={e['blind_item_sha256']:e['true_item'] for e in json.loads((a.root/'test384-review-6methods/private/mapping.json').read_text())['entries']}
 rows=[json.loads(l) for path in a.scores for l in path.open() if l.strip()]
 if len(rows)!=2304 or len({x['item_sha256'] for x in rows})!=2304 or {x['item_sha256'] for x in rows}!=set(mapping):raise ValueError('Incomplete or duplicated music inventory')
 methods=sorted({x['method_id'] for x in mapping.values()});values={m:{} for m in methods};statuses={m:collections.Counter() for m in methods}
 for row in rows:
  meta=mapping[row['item_sha256']];row['true_method']=meta['method_id'];row['true_case']=meta['case_id'];status='generation_not_evaluable' if meta['generation_status']!='ok' else row['status']
  score=row.get('rating',{}).get('music_fit') if status=='valid' else None
  if status=='valid' and score is None:status='unknown'
  row['final_status']=status;statuses[meta['method_id']][status]+=1
  if score is not None:values[meta['method_id']][meta['case_id']]=(meta['source_group_id'],score)
 common=sorted(set.intersection(*(set(values[m]) for m in methods)))
 if not common:raise ValueError('No common evaluable cases; selection unresolved')
 table=[];paired=[]
 for m in methods:
  groups=collections.defaultdict(list)
  for case in common:g,v=values[m][case];groups[g].append(v)
  table.append(dict(method=m,registered=384,status=dict(statuses[m]),valid_scores=len(values[m]),common_cases=len(common),common_source_groups=len(groups),common_group_mean=float(np.mean([np.mean(v) for v in groups.values()])),valid_mean=float(np.mean([v for _,v in values[m].values()])) if values[m] else None,full_score_fraction=sum(v==5 for _,v in values[m].values())/len(values[m]) if values[m] else None))
 for x,y in itertools.combinations(methods,2):
  cases=sorted(set(values[x])&set(values[y]));groups=collections.defaultdict(list)
  for case in cases:
   ga,va=values[x][case];gb,vb=values[y][case]
   if ga!=gb:raise ValueError('Source pairing changed')
   groups[ga].append(va-vb)
  diffs=np.array([np.mean(groups[g]) for g in sorted(groups)]);rng=np.random.default_rng(42);boots=diffs[rng.integers(0,len(diffs),(5000,len(diffs)))].mean(1)
  paired.append(dict(a=x,b=y,common_cases=len(cases),source_groups=len(groups),mean_difference=float(diffs.mean()),ci95=np.quantile(boots,[.025,.975]).tolist()))
 leader=max(table,key=lambda t:t['common_group_mean'])['method'];unresolved={leader}
 for q in paired:
  if leader in (q['a'],q['b']) and q['ci95'][0]<=0<=q['ci95'][1]:unresolved.add(q['b'] if q['a']==leader else q['a'])
 generation=json.loads((a.root/'results_summary.json').read_text())['methods_statistics'];selected=min(unresolved,key=lambda m:(-generation[m]['valid_generations'],generation[m]['mean_inference_latency_seconds'],m))
 result=dict(registered=2304,results=table,paired=paired,seed=42,resamples=5000,selected_expert=selected,music_mean_leader=leader,unresolved_candidates=sorted(unresolved),statistically_established_unique_winner=len(unresolved)==1,selection_rule='common-source music mean; unresolved pairwise95 then valid generation and input-matched recorded latency operational tie-break',latency_scope='recorded E2 runs; architecture costs not normalized',origin='model-assisted score using independent audio observations; not directly heard by text scorer; acoustic diagnostics separate',human=False,unknown_not_imputed=True,reference_similarity_not_used_for_music_selection=True)
 a.out.mkdir(exist_ok=False)
 with (a.out/'records.jsonl').open('x') as f:
  for row in rows:f.write(json.dumps(row)+chr(10))
 (a.out/'results.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
