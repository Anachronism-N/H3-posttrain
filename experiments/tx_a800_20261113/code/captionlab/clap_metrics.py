"""Audio-text embedding diagnostics, not learned motion-audio synchronization."""
import argparse,json,hashlib,math
from pathlib import Path
from statistics import mean
from collections import Counter
import numpy as np
import torch,soundfile as sf
from scipy.signal import resample_poly
from transformers import ClapModel,ClapProcessor
from captionlab.reference_metrics import inputs


def feature(value):
 if torch.is_tensor(value):return value
 return value.pooler_output


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.out.mkdir(exist_ok=False)
 processor=ClapProcessor.from_pretrained(a.model,local_files_only=True);model=ClapModel.from_pretrained(a.model,local_files_only=True).to(a.device).eval().requires_grad_(False)
 rows=inputs(a.root);sources=json.loads((a.root/'test384/manifest.public.json').read_text())['cases'];text_items={hashlib.sha256(r['text'].encode()).hexdigest():r['text'] for r in rows if r['generation_status']=='ok'};texts={};unknown=set()
 for key,text in text_items.items():
  if len(processor.tokenizer(text,truncation=False)['input_ids'])>512:unknown.add(key)
 eligible=[(k,v) for k,v in text_items.items() if k not in unknown]
 for start in range(0,len(eligible),32):
  batch=eligible[start:start+32];encoded=processor.tokenizer([v for k,v in batch],padding=True,truncation=False,return_tensors='pt')
  with torch.inference_mode():emb=feature(model.get_text_features(**{k:v.to(a.device) for k,v in encoded.items()})).cpu().numpy()
  for (key,text),f in zip(batch,emb):texts[key]=f
 audios={}
 for start in range(0,len(sources),8):
  group=sources[start:start+8];waves=[]
  for item in group:
   path=Path(item['audio']['path']);x,rate=sf.read(path,dtype='float32',always_2d=True)
   if x.shape[1]!=1 or len(x)!=item['audio']['num_samples']:raise ValueError('Audio geometry differs')
   div=math.gcd(rate,48000);waves.append(resample_poly(x[:,0],48000//div,rate//div).astype(np.float32))
  features=processor.feature_extractor(waves,sampling_rate=48000,padding='repeatpad',return_tensors='pt')
  with torch.inference_mode():emb=feature(model.get_audio_features(**{k:v.to(a.device) for k,v in features.items()})).cpu().numpy()
  for item,f in zip(group,emb):audios[item['case_id']]=f
  if start%64==0:print(json.dumps(dict(audio_completed=min(start+8,len(sources)),audio_total=len(sources))),flush=True)
 keys=sorted(audios);matrix=np.stack([audios[k] for k in keys]);positions={k:i for i,k in enumerate(keys)};results=[]
 for row in rows:
  result={k:row[k] for k in ('case_id','method_id','source_group_id','item_sha256')};key=hashlib.sha256(row['text'].encode()).hexdigest()
  if row['generation_status']!='ok':result['status']='generation_not_evaluable'
  elif key not in texts:result['status']='text_overflow_not_truncated'
  else:
   scores=matrix@texts[key];idx=positions[row['case_id']];actual=float(scores[idx]);others=np.delete(scores,idx);rank_min=1+int(np.sum(others>actual));ties=int(np.sum(others==actual));result.update(status='computed',audio_text_cosine=actual,other_audio_mean_cosine=float(others.mean()),paired_vs_other_mean=actual-float(others.mean()),retrieval_rank_min=rank_min,retrieval_ties=ties)
  results.append(result)
 summary={}
 for method in sorted({r['method_id'] for r in results}):
  group=[r for r in results if r['method_id']==method];valid=[r for r in group if r['status']=='computed'];summary[method]=dict(registered=len(group),evaluable=len(valid),statuses=dict(Counter(r['status'] for r in group)),mean_cosine=mean(r['audio_text_cosine'] for r in valid) if valid else None,mean_matching_minus_other=mean(r['paired_vs_other_mean'] for r in valid) if valid else None,retrieval_top1_tie_fractional=mean(1/(1+r['retrieval_ties']) if r['retrieval_rank_min']==1 else 0 for r in valid) if valid else None)
 with (a.out/'records.jsonl').open('x') as f:
  for r in results:f.write(json.dumps(r)+'\n')
 (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,scope='generic pretrained audio-text embedding agreement; no evidence of dance-specific calibration or motion synchronization',model=str(a.model),non_generative=True,source_audio_unchanged=True,native_processor_padding='repeatpad to10s, internal only, original5.0625s retained',silent_text_truncation=False,audio_pool_size=len(keys),similar_songs_unverified=True,quality_metric_validated=False),indent=2)+'\n')
 print(json.dumps(dict(completed=True,registered=len(results))),flush=True)


if __name__=='__main__':main()
