"""Ordinal music evidence audit from actual audio, not text-only similarity."""
import argparse,json,hashlib,math,re
from pathlib import Path
import numpy as np
import torch,soundfile as sf
from scipy.signal import resample_poly

PROMPT = (
 'Listen to this ACTUAL complete 5.0625-second audio. No choreography, movement text, model names or reference labels are provided. '
 'Describe ONLY audible rhythm and dynamics. Do not propose any dance and do not guess exact counts, beat times, tempo values, instruments or song-ending structure. '
 'Give pulse_rate=slow/moderate/fast/uncertain based on audible repetition, energy=low/moderate/high/uncertain, '
 'articulation=sustained/percussive/mixed/uncertain, development=stable/changing/uncertain. '
 'Silence between sounds is not continuous sound. Unknown or ambiguous pulse is uncertain, not invented. '
 'Use short audio_evidence and limitations strings, both at most240 characters. '
 'Return only JSON with pulse_rate,energy,articulation,development,audio_evidence,limitations.'
)
SCHEMA=dict(type='object',additionalProperties=False,required=['pulse_rate','energy','articulation','development','audio_evidence','limitations'],properties=dict(
 pulse_rate=dict(type='string',enum=['slow','moderate','fast','uncertain']),energy=dict(type='string',enum=['low','moderate','high','uncertain']),
 articulation=dict(type='string',enum=['sustained','percussive','mixed','uncertain']),development=dict(type='string',enum=['stable','changing','uncertain']),
 audio_evidence=dict(type='string',minLength=1,maxLength=240),limitations=dict(type='string',minLength=1,maxLength=240)))


def parse(raw,item):
 value=json.loads(raw)
 if set(value)!=set(SCHEMA['required']):raise ValueError('Audio evidence schema differs')
 for k in value:
  if 'enum' in SCHEMA['properties'][k] and value[k] not in SCHEMA['properties'][k]['enum']:raise ValueError('Evidence enum invalid')
  if not isinstance(value[k],str) or not value[k].strip():raise ValueError('Empty evidence')
 if re.search(r'\d+(?:\.\d+)?\s*(?:s\b|seconds?\b|bpm\b)',value['audio_evidence'],re.I):raise ValueError('Exact audible values not supported')
 return value


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--items',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
 if not 0<=a.shard<a.shards:raise ValueError('Invalid shard')
 a.out.mkdir(exist_ok=False)
 from transformers import AutoProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
 from lmformatenforcer import JsonSchemaParser
 from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
 processor=AutoProcessor.from_pretrained(a.model,local_files_only=True);model=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map='balanced',local_files_only=True).eval().requires_grad_(False)
 tokenizer=processor.tokenizer;data=build_token_enforcer_tokenizer_data(tokenizer);eos=tokenizer.eos_token_id
 items=json.loads(a.items.read_text())['items'][a.shard::a.shards];assets={x['sha256']:x for x in json.loads(a.assets.read_text())['assets']}
 (a.out/'receipt.json').write_text(json.dumps(dict(prompt=PROMPT,schema=SCHEMA,code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model=str(a.model),actual_audio=True,no_reference=True,candidate_text_input=False,enable_thinking=False,constrained_labels=True,budget=512,items_sha=hashlib.sha256(a.items.read_bytes()).hexdigest(),human=False,calibrated=False))+'\n')
 with (a.out/'observations.jsonl').open('x') as f:
  for i,item in enumerate(items):
   row={k:item[k] for k in ('item_sha256','case_id','method_id','source_group_id')};row.update(status='unknown',human=False)
   if item['generation_status']!='ok':row['status']='generation_not_evaluable'
   else:
    try:
     asset=assets[item['audio_sha256']];path=a.assets.parent/asset['path']
     if hashlib.sha256(path.read_bytes()).hexdigest()!=item['audio_sha256']:raise ValueError('Audio identity changed')
     audio,rate=sf.read(path,dtype='float32',always_2d=True)
     if audio.shape[1]!=1 or len(audio)!=asset['num_samples']:raise ValueError('Audio geometry changed')
     divisor=math.gcd(rate,16000);wave=resample_poly(audio[:,0],16000//divisor,rate//divisor).astype(np.float32) if rate!=16000 else audio[:,0]
     prompt=processor.apply_chat_template([dict(role='user',content=[dict(type='audio',audio='native-waveform'),dict(type='text',text=PROMPT)])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
     batch=processor(text=[prompt],audio=[wave],sampling_rate=16000,return_tensors='pt',padding=False);n=batch['input_ids'].shape[1];device=model.get_input_embeddings().weight.device
     batch={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()};constraint=build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(SCHEMA))
     with torch.inference_mode():seq=model.generate(**batch,do_sample=False,max_new_tokens=512,eos_token_id=eos,use_cache=True,prefix_allowed_tokens_fn=constraint)
     ids=seq[0,n:].cpu().tolist();raw=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
     (a.out/(str(i)+'.raw.json')).write_text(json.dumps(dict(item_sha256=item['item_sha256'],text=raw,tokens=ids,prompt=prompt))+'\n')
     if not ids or ids[-1]!=eos:raise ValueError('Music gate response truncated')
     row.update(status='valid',observation=parse(raw,item))
    except (ValueError,TypeError,KeyError,OSError) as e:row['error']=str(e)
   f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(dict(completed=i+1,total=len(items),status=row['status'])),flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(completed=True,registered=len(items),semantic_validated=False,human=False))+'\n')


if __name__=='__main__':main()
