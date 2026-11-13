"""Non-generative full-song audio-text sensitivity, not motion synchronization."""
import argparse,json,math,hashlib,statistics,collections
from pathlib import Path
import numpy as np
import torch,soundfile as sf
from scipy.signal import resample_poly
from transformers import ClapModel,ClapProcessor
from captionlab.clap_metrics import feature


def main():
 p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();a.out.mkdir(exist_ok=False)
 bindings=json.loads((a.observations/'bindings.json').read_text())['bindings'];assets={x['sha256']:x for x in json.loads((a.observations/'audio_assets.json').read_text())['assets']};processor=ClapProcessor.from_pretrained(a.model,local_files_only=True);model=ClapModel.from_pretrained(a.model,local_files_only=True).to(a.device).eval().requires_grad_(False);audio_features={};song_rows=collections.defaultdict(list)
 for start in range(0,len(bindings),8):
  batch=bindings[start:start+8];waves=[]
  for row in batch:
   asset=assets[row['audio_sha256']];x,rate=sf.read(asset['path'],dtype='float32');div=math.gcd(rate,48000);waves.append(resample_poly(x,48000//div,rate//div).astype(np.float32))
  encoded=processor.feature_extractor(waves,sampling_rate=48000,padding='repeatpad',return_tensors='pt')
  with torch.inference_mode():vectors=feature(model.get_audio_features(**{k:v.to(a.device) for k,v in encoded.items()})).cpu().numpy()
  for row,v in zip(batch,vectors):audio_features[row['audio_sha256']]=v;song_rows[row['song_id']].append(row)
 records=[];cells=json.loads((a.formal/'execution_results.json').read_text())['rows'];text_inputs=[]
 for cell in cells:
  sid=cell['song_id'];path=a.formal/'runs'/sid/cell['method']/'plan.json';plan=json.loads(path.read_text()) if cell['full_song_succeeded'] else None
  source=song_rows[sid]
  for pos,row in enumerate(source):
   out=dict(song_id=sid,method=cell['method'],window=row['window']['index'],status='failed_generation')
   if plan is not None:
    w=row['window'];text=' '.join(e['action'] for e in plan['events'] if e['start']<w['observation_end'] and e['end']>w['observation_start']);out['text_sha256']=hashlib.sha256(text.encode()).hexdigest()
    if not text:out['status']='empty_text'
    elif len(processor.tokenizer(text,truncation=False)['input_ids'])>512:out['status']='context_overflow_no_truncation'
    else:
     out['status']='pending';others=[]
     for other,rows in song_rows.items():
      if other==sid:continue
      index=round(pos*(len(rows)-1)/max(1,len(source)-1));others.append(audio_features[rows[index]['audio_sha256']])
     text_inputs.append((out,text,audio_features[row['audio_sha256']],np.stack(others)))
   records.append(out)
 for start in range(0,len(text_inputs),32):
  batch=text_inputs[start:start+32];encoded=processor.tokenizer([x[1] for x in batch],padding=True,truncation=False,return_tensors='pt')
  with torch.inference_mode():vectors=feature(model.get_text_features(**{k:v.to(a.device) for k,v in encoded.items()})).cpu().numpy()
  for (row,text,actual,others),v in zip(batch,vectors):row.update(status='computed',matching_cosine=float(actual@v),other_recording_mean=float((others@v).mean()),matching_minus_other=float(actual@v-(others@v).mean()),comparison_recordings=len(others))
 with (a.out/'records.jsonl').open('x') as f:
  for row in records:f.write(json.dumps(row)+chr(10))
 summary={}
 for m in ('workflow','self_refine','react','choreography'):
  group=[r for r in records if r['method']==m];songs=collections.defaultdict(list)
  for r in group:
   if r['status']=='computed':songs[r['song_id']].append(r['matching_minus_other'])
  summary[m]=dict(registered_songs=24,registered_windows=len(group),status_counts=dict(collections.Counter(r['status'] for r in group)),valid_songs=len(songs),song_equal_mean=statistics.mean(statistics.mean(v) for v in songs.values()) if songs else None,song_means={s:statistics.mean(v) for s,v in songs.items()})
 (a.out/'summary.json').write_text(json.dumps(dict(methods=summary,scope='generic audio-text embedding sensitivity to original vs23 other recordings at matched normalized progress; not dance-grounded music quality or verified movement timing',non_generative=True,no_silent_text_truncation=True,native_audio_padding='repeatpad internal only, original PCM unchanged',human=False),indent=2)+chr(10))


if __name__=='__main__':main()
