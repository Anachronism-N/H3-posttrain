"""One-to-one non-generative NLI agreement over token-anchored explicit facts."""
import argparse,json,time,statistics,hashlib
from pathlib import Path
from collections import Counter
from captionlab.nli_reference import NLI
from captionlab.semantic_graph_metrics import evaluate
from captionlab.graph_metrics import evaluate_graph


def main():
 p=argparse.ArgumentParser();p.add_argument('--graphs',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 graph_rows=list(map(json.loads,a.graphs.open()));graphs={r['text_id']:r for r in graph_rows};bindings=json.loads(a.bindings.read_text())['bindings']
 if len(graph_rows)!=2469 or len(graphs)!=2469 or len(bindings)!=2304 or len({x['item_sha256'] for x in bindings})!=2304:raise ValueError('Registered graph/binding inventory incomplete or duplicated')
 for pair in bindings:
  if pair['generation_status']=='ok' and (pair['reference_text_id'] not in graphs or pair['candidate_text_id'] not in graphs):raise ValueError('Binding names a missing original text graph')
 extractor=json.loads(a.graphs.with_name('summary.json').read_text())
 if extractor.get('generator') is not False or extractor.get('registered_unique')!=2469:raise ValueError('Non-generative extractor receipt differs')
 a.out.mkdir(exist_ok=False);started=time.monotonic();nli=NLI(a.model,a.device);rows=[]
 with (a.out/'records.jsonl').open('x') as f:
  for i,pair in enumerate(bindings):
   row={k:pair[k] for k in ('item_sha256','case_id','method_id','source_group_id')};row['status']='generation_not_evaluable'
   if pair['generation_status']=='ok':
    ref=graphs[pair['reference_text_id']];cand=graphs[pair['candidate_text_id']]
    row.update(status='computed',exact=evaluate_graph(ref['graph'],cand['graph']),semantic=evaluate(ref['graph'],cand['graph'],nli,threshold=.7),unresolved_reference_mentions=len(ref['unsupported_mentions']),unresolved_candidate_mentions=len(cand['unsupported_mentions']))
   rows.append(row);f.write(json.dumps(row)+chr(10));f.flush()
   if (i+1)%192==0:print(json.dumps(dict(completed=i+1,total=len(bindings),cached_proposition_pairs=len(nli.cache))),flush=True)
 if len(rows)!=2304 or len({x['item_sha256'] for x in rows})!=2304:raise ValueError('Registered inventory incomplete or duplicated')
 methods={}
 for m in sorted({r['method_id'] for r in rows}):
  group=[r for r in rows if r['method_id']==m];valid=[r for r in group if r['status']=='computed'];summary=dict(registered=384,status_counts=dict(Counter(r['status'] for r in group)),unresolved_reference_mentions=sum(r['unresolved_reference_mentions'] for r in valid),unresolved_candidate_mentions=sum(r['unresolved_candidate_mentions'] for r in valid))
  for mode in ('exact','semantic'):
   for dimension in ('facts','relations'):
    values=[r[mode][dimension]['f1'] for r in valid if not r[mode][dimension]['empty_both']];summary[mode+'_'+dimension+'_F1']=dict(n=len(values),mean=statistics.mean(values) if values else None,empty_both_not_scored=sum(r[mode][dimension]['empty_both'] for r in valid))
  methods[m]=summary
 (a.out/'summary.json').write_text(json.dumps(dict(registered=2304,methods=methods,wallclock_seconds=time.monotonic()-started,platform='TX A800 cluster',device=str(a.device),model='cross-encoder/nli-deberta-v3-base',threshold=.7,code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),non_generative_extraction=True,non_generative_matching=True,extractor_code_sha=extractor['code_sha'],input_graphs_sha256=hashlib.sha256(a.graphs.read_bytes()).hexdigest(),input_bindings_sha256=hashlib.sha256(a.bindings.read_bytes()).hexdigest(),model_config_sha256=hashlib.sha256((a.model/'config.json').read_bytes()).hexdigest(),extractor_open_text_quality_validated=False,reference_scope='explicit supported body/side/predicate/direction facts and explicit before/concurrent relations; partial source agreement, not unique music-dance or video correctness',unresolved_coreference_not_invented=True,all_original_candidate_texts_preserved=True),indent=2)+chr(10))
 print(json.dumps(dict(completed=True,registered=2304,methods=methods,wallclock_seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
