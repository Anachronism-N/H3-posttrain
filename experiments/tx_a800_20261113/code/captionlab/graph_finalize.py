"""Complete extractor-dependent reference graph metrics with explicit unknowns."""
import argparse,json,collections,statistics
from pathlib import Path
from captionlab.graph_metrics import evaluate_graph
from captionlab.semantic_graph_metrics import evaluate
from captionlab.nli_reference import NLI


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--graphs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--nli-model',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args();a.out.mkdir(exist_ok=False)
 prefix=json.loads((a.run/'prefix_finalized.json').read_text())['records'];suffix=[json.loads(l) for path in a.graphs for l in path.open() if l.strip()];rows=prefix+suffix;index={r['text_id']:r for r in rows}
 expected=json.loads((a.run/'freeze.json').read_text())['registered_unique']
 if len(rows)!=len(index) or len(rows)!=expected:raise ValueError('Graph inventory incomplete or duplicated')
 bindings=json.loads((a.run/'bindings.private.json').read_text())['bindings'];nli=NLI(a.nli_model,a.device) if a.nli_model else None;result=[]
 with (a.out/'records.jsonl').open('x') as stream:
  for b in bindings:
   row={k:b[k] for k in ('item_sha256','case_id','method_id','source_group_id')};row['status']='unknown'
   if b['generation_status']!='ok':row['status']='generation_not_evaluable'
   else:
    ref=index[b['reference_text_id']];cand=index[b['candidate_text_id']]
    if ref['status']=='valid' and cand['status']=='valid':
     row.update(status='computed',exact=evaluate_graph(ref['graph'],cand['graph']))
     if nli:row['semantic']=evaluate(ref['graph'],cand['graph'],nli,threshold=.7)
    else:row['error']=dict(reference=ref['status'],candidate=cand['status'])
   result.append(row);stream.write(json.dumps(row)+chr(10));stream.flush()
 summary={}
 for m in sorted({r['method_id'] for r in result}):
  group=[r for r in result if r['method_id']==m];valid=[r for r in group if r['status']=='computed'];out=dict(registered=len(group),statuses=dict(collections.Counter(r['status'] for r in group)))
  for mode in ('exact','semantic'):
   for dim in ('facts','relations'):
    vals=[r[mode][dim]['f1'] for r in valid if mode in r and not r[mode][dim]['empty_both']];out[mode+'_'+dim+'_macroF1']=dict(n=len(vals),mean=statistics.mean(vals) if vals else None)
  summary[m]=out
 (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,registered=2304,unique_extractions=expected,semantic_NLI_threshold=.7 if nli else None,non_generative_comparison=True,model_assisted_extraction=True,scope='source reference agreement on supported body parts; not unique dance truth or music quality',calibration='explicit two-event32 controls; unrestricted source extraction reliability separately audited',interrupted_prefix_preserved=True),indent=2)+chr(10))


if __name__=='__main__':main()
