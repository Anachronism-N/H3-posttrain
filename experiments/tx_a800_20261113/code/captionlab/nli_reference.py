"""Fixed non-generative bidirectional reference entailment; not music fit or truth."""
import argparse,json,re,hashlib
from pathlib import Path
from statistics import mean
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from captionlab.reference_metrics import inputs


def sentences(text):
 return [x.strip() for x in re.split(r'(?<=[.!?;])\s+',text) if x.strip()]


class NLI:
 def __init__(self,model,device,alignment='sentence'):
  self.alignment=alignment
  self.tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
  self.model=AutoModelForSequenceClassification.from_pretrained(model,local_files_only=True).to(device).eval().requires_grad_(False)
  self.device=device;self.cache={};labels={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
  if set(labels)!={'contradiction','entailment','neutral'}:raise ValueError('NLI label mapping unknown')
  self.labels=labels
 def score(self,pairs):
  unique=list(dict.fromkeys(p for p in pairs if p not in self.cache))
  for start in range(0,len(unique),32):
   batch=unique[start:start+32];eligible=[]
   for a,b in batch:
    count=len(self.tokenizer(a,b,truncation=False)['input_ids'])
    if count>512:self.cache[a,b]=None
    else:eligible.append((a,b))
   if not eligible:continue
   tokens=self.tokenizer([p[0] for p in eligible],[p[1] for p in eligible],padding=True,truncation=False,return_tensors='pt')
   with torch.inference_mode():probs=self.model(**{k:v.to(self.device) for k,v in tokens.items()}).logits.float().softmax(-1).cpu().numpy()
   for pair,p in zip(eligible,probs):self.cache[pair]={k:float(p[v]) for k,v in self.labels.items()}
  return [self.cache[p] for p in pairs]
 def compare(self,reference,candidate):
  refs=[reference] if self.alignment=='whole_text' else sentences(reference);cands=[candidate] if self.alignment=='whole_text' else sentences(candidate)
  if self.alignment=='contextual':
   forward=self.score([(reference,c) for c in cands]);backward=self.score([(candidate,r) for r in refs])
   if any(x is None for x in forward+backward):return dict(status='context_overflow_no_truncation',pairs=len(forward)+len(backward))
   precision=float(np.mean([x['entailment'] for x in forward]));recall=float(np.mean([x['entailment'] for x in backward]));relation_ids=[i for i,t in enumerate(refs) if re.search(r'\b(?:then|before|after|while|as|followed by|simultaneous)\b',t,re.I)]
   return dict(status='computed',entailment_precision=precision,entailment_recall=recall,entailment_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,reference_candidate_contradiction=float(np.mean([x['contradiction'] for x in forward])),reference_relation_sentence_entailment=float(np.mean([backward[i]['entailment'] for i in relation_ids])) if relation_ids else None,reference_relation_sentences=len(relation_ids),reference_sentences=len(refs),candidate_sentences=len(cands),pair_comparisons=len(forward)+len(backward))
  if not refs or not cands:return dict(status='empty_text')
  pairs=[(r,c) for r in refs for c in cands]+[(c,r) for r in refs for c in cands];p=self.score(pairs);n=len(refs)*len(cands)
  if any(x is None for x in p):return dict(status='context_overflow_no_truncation',pairs=len(p))
  precision=np.array([x['entailment'] for x in p[:n]]).reshape(len(refs),len(cands)).max(axis=0).mean()
  recall=np.array([x['entailment'] for x in p[n:]]).reshape(len(refs),len(cands)).max(axis=1).mean()
  contradiction=np.array([x['contradiction'] for x in p[:n]]).reshape(len(refs),len(cands)).max(axis=0).mean()
  return dict(status='computed',entailment_precision=float(precision),entailment_recall=float(recall),entailment_f1=float(2*precision*recall/(precision+recall)) if precision+recall else 0.,reference_candidate_contradiction=float(contradiction),reference_sentences=len(refs),candidate_sentences=len(cands),pair_comparisons=len(p))


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--controls',type=Path,required=True);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--alignment',choices=['sentence','whole_text','contextual'],default='sentence');a=p.parse_args();a.out.mkdir(exist_ok=False)
 nli=NLI(a.model,a.device,a.alignment);items=json.loads((a.controls/'control_items.json').read_text())['items'];texts={x['item_sha256']:x['text'] for x in items};control=[]
 for pair in json.loads((a.controls/'control_pairs.json').read_text())['pairs']:
  whole=nli.score([(texts[pair['original']],texts[pair['variant']])])[0]
  row=dict(pair,whole_reference_to_candidate=whole,metrics=nli.compare(texts[pair['original']],texts[pair['variant']]));control.append(row)
 (a.out/'control_results.json').write_text(json.dumps(dict(controls=control,origin='assistant constructed development, not human',nli_label_map=nli.labels,not_music_truth=True),indent=2)+'\n')
 print('NLI control comparisons computed',len(control),flush=True)
 if a.root:
  rows=inputs(a.root);results=[]
  with (a.out/'records.jsonl').open('x') as f:
   for i,row in enumerate(rows):
    rec={k:row[k] for k in ('case_id','method_id','source_group_id','item_sha256','generation_status')}
    rec['metrics']=nli.compare(row['reference'],row['text']) if row['generation_status']=='ok' else dict(status='generation_not_evaluable')
    f.write(json.dumps(rec)+'\n');f.flush();results.append(rec)
    if (i+1)%96==0:print(json.dumps(dict(completed=i+1,total=len(rows),unique_pair_cache=len(nli.cache))),flush=True)
  summary={}
  for method in sorted({x['method_id'] for x in results}):
   group=[x for x in results if x['method_id']==method];valid=[x for x in group if x['metrics']['status']=='computed']
   summary[method]=dict(registered=len(group),evaluable=len(valid),statuses=dict(Counter(x['metrics']['status'] for x in group)),metrics={k:mean(x['metrics'][k] for x in valid) if valid else None for k in ('entailment_precision','entailment_recall','entailment_f1','reference_candidate_contradiction')})
  (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,scope=a.alignment+' NLI probabilities, not action F1 or music quality',non_generative=True,no_silent_truncation=True,code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model=str(a.model),label_map=nli.labels,human=False,validation_pending=True),indent=2)+'\n')
  print('NLI full reference metrics completed',flush=True)


if __name__=='__main__':main()
