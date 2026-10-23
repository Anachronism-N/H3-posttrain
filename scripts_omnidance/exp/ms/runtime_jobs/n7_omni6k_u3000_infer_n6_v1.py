import argparse, collections, copy, hashlib, importlib.util, json, os, pathlib, re, sys, tempfile
import torch
from safetensors import safe_open
from peft import PeftModel
from peft.utils import get_peft_model_state_dict
P=pathlib.Path
MS=P('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS')
CORE=P('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms/omni_dance_sft.py')
CORE_SHA='4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2'
DEPLOY=P('/opt/omni_n7_u3000_deploy_v1')
BASE=P('/opt/omni30b_dance_official_v1')
OUT=MS/'n7_omni6k_u3000_eval_n6_v1'
INPUT=MS/'omni_h3_ms_style_eval3_v1/omni_infer3.jsonl'
IDS=['wJPptuhclZmKAEWs','wJPrpFsThWAcBQIw','wJQTclrlOLEYYfTB']
FP='c5e4955f0b0dd90ebf6afdc3556be2c6127cc4443a9eb150c8d8cff4ac57d315'
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with P(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(k,**v):print(json.dumps(dict(event=k,pid=os.getpid(),**v),ensure_ascii=False),flush=True)
def scope(keys):
 if not keys or not any('audio_tower.layers.' in k and '.lora_' in k for k in keys):raise ValueError('Missing audio LoRA')
 for proj in ('audio_tower.proj1','audio_tower.proj2'):
  if not any(proj in k for k in keys):raise ValueError('Missing audio projector '+proj)
 if not any('mlp.experts.' in k and '.lora_' in k for k in keys):raise ValueError('Missing MoE adapter parameters')
 if not any('model.layers.' in k and '.lora_' in k for k in keys):raise ValueError('Missing LM LoRA')
 if any('talker' in k or 'code2wav' in k or 'visual.' in k for k in keys):raise ValueError('Forbidden adapter scope')
def validate(base,deploy,receipt,marker,keys):
 if str(base)!=receipt['base_path'] or base!=BASE or not (base/'model.safetensors.index.json').is_file():raise ValueError('Fake/missing/wrong base')
 if not receipt['complete'] or not receipt['deployment_only'] or receipt['resume_checkpoint'] or receipt['training_state_read_or_copied']:raise ValueError('Wrong deployment contract')
 if marker['schema']!='omni-dance-sft-v1' or (marker['updates'],marker['epoch'],marker['position'])!=(3000,4,0) or marker['fingerprint']!=FP:raise ValueError('Wrong checkpoint marker')
 if not (deploy/'adapter/adapter_model.safetensors').is_file() or not (deploy/'adapter/adapter_config.json').is_file():raise ValueError('Missing adapter')
 for n in ('adapter/adapter_model.safetensors','adapter/adapter_config.json'):
  if marker['files'][n]!=receipt['deployment_files_sha256'][n]:raise ValueError('Adapter marker mismatch')
 scope(keys)
def main(cpu=False):
 torch.set_num_threads(2)
 assert sha(CORE)==CORE_SHA
 receipt=json.loads((DEPLOY/'deployment_receipt.json').read_text());marker=json.loads((DEPLOY/'source_checkpoint_marker.json').read_text())
 assert sha(DEPLOY/'deployment_receipt.json')==json.loads((OUT/'stage_complete.json').read_text())['deployment_receipt_sha256']
 for n,expected in receipt['deployment_files_sha256'].items():
  if sha(DEPLOY/n)!=expected:raise ValueError('Deployment file SHA differs: '+n)
 with safe_open(str(DEPLOY/'adapter/adapter_model.safetensors'),framework='pt',device='cpu') as f:keys=list(f.keys())
 validate(BASE,DEPLOY,receipt,marker,keys)
 spec=importlib.util.spec_from_file_location('omni_core_frozen',CORE);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 provenance=json.loads((DEPLOY/'provenance.json').read_text())
 assert provenance['source_sha256']==CORE_SHA and provenance['model_id']==receipt['model_id']
 family=core.identify_family(BASE);assert family=='qwen3'
 processor_cls,config_cls,model_cls=core.model_classes(family)
 processor=processor_cls.from_pretrained(DEPLOY/'processor',local_files_only=True)
 config=config_cls.from_pretrained(BASE,local_files_only=True)
 rows,_,_=core.read_metadata(INPUT,training=False)
 assert [r['id'] for r in rows]==IDS
 if cpu:
  tests=['real_deployment_positive']
  trials=[('fake_base',P('/opt/fake-base'),DEPLOY,marker,keys),('missing_adapter',BASE,P('/opt/nonexistent-u3000-adapter'),marker,keys),('missing_projector',BASE,DEPLOY,marker,[k for k in keys if 'audio_tower.proj2' not in k]),('wrong_marker',BASE,DEPLOY,dict(marker,updates=100),keys)]
  for label,b,d,m,ks in trials:
   try:validate(b,d,receipt,m,ks)
   except ValueError:tests.append(label+'_rejected')
   else:raise AssertionError(label+' accepted')
  original=processor_cls.from_pretrained(BASE,local_files_only=True)
  inp=core.prepare_inputs(processor,rows[0],training=False,max_length=8192,audio_n_window=config.audio_config.n_window)
  baseline=core.prepare_inputs(original,rows[0],training=False,max_length=8192,audio_n_window=config.audio_config.n_window)
  assert set(inp)==set(baseline) and all(torch.equal(inp[k],baseline[k]) for k in inp)
  assert 'labels' not in inp
  save(OUT/'cpu_functional_verified_v2.json',dict(complete=True,cpu_only=True,pretrained_model_loaded=False,tests=tests,adapter_tensors=len(keys),saved_processor_base_input_parity=True,input_shapes={k:list(v.shape) for k,v in inp.items()}))
  event('cpu_functional_complete',tests=tests);return
 assert not (OUT/'omni_prediction3_raw.json').exists()
 event('omni_model_loading',model_class=model_cls.__name__,speech_modules_loaded=False,updates=3000)
 model=model_cls.from_pretrained(BASE,config=config,dtype=torch.bfloat16,device_map='cuda',local_files_only=True,attn_implementation='sdpa')
 # modules_to_save must be created in the stored dtype, before PEFT copies/loads them.
 with safe_open(str(DEPLOY/'adapter/adapter_model.safetensors'),framework='pt',device='cpu') as f:
  for projector in ('proj1','proj2'):
   stored=f.get_tensor('base_model.model.audio_tower.'+projector+'.weight')
   model.audio_tower.get_submodule(projector).to(dtype=stored.dtype)
 model=PeftModel.from_pretrained(model,DEPLOY/'adapter',is_trainable=False)
 base=model.get_base_model()
 assert not hasattr(base,'talker') and not hasattr(base,'code2wav')
 base.visual.register_forward_pre_hook(lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Visual tower called')))
 model.eval().requires_grad_(False)
 state=get_peft_model_state_dict(model)
 if set(state)!=set(keys):raise ValueError('Loaded adapter key set differs from saved tensor keys')
 verified=[]
 with safe_open(str(DEPLOY/'adapter/adapter_model.safetensors'),framework='pt',device='cpu') as f:
  for k in keys:
   local=f.get_tensor(k);actual=state[k].detach().cpu()
   if local.dtype!=actual.dtype or local.shape!=actual.shape or not torch.equal(local,actual):raise ValueError('Loaded tensor differs: '+k)
   verified.append(dict(key=k,dtype=str(local.dtype),shape=list(local.shape),sha256=hashlib.sha256(local.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
 save(OUT/'actual_adapter_tensor_identity.json',dict(complete=True,all_saved_keys_exact=True,records=verified,adapter_sha256=receipt['deployment_files_sha256']['adapter/adapter_model.safetensors']))
 del state
 event('omni_model_loaded_and_adapter_verified',adapter_tensors=len(keys),speech_modules_loaded=False)
 predictions=[]
 def th(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 with torch.inference_mode():
  for row in rows:
   waveform=core.read_audio(row)
   processed=core.prepare_inputs(processor,row,training=False,waveform=waveform,max_length=8192,audio_n_window=config.audio_config.n_window)
   assert 'labels' not in processed
   audit=dict(audio_path=row['audios'][0],audio_sha256=sha(row['audios'][0]),audio_window_seconds=row.get('audio_window_seconds'),waveform_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),waveform_samples=len(waveform),processed_inputs={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=th(v)) for k,v in processed.items()},labels_passed=False)
   inputs=core.to_device(processed,model.device,torch.bfloat16)
   event('actual_omni_generate_started',id=row['id'],input_audit=audit)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16):generated=model.generate(**inputs,do_sample=False,max_new_tokens=512,eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
   assert isinstance(generated,torch.Tensor)
   tokens=generated[:,inputs['input_ids'].shape[-1]:]; values=tokens[0].detach().cpu().tolist()
   caption=processor.batch_decode(tokens,skip_special_tokens=True)[0]
   counts=collections.Counter(tuple(values[i:i+4]) for i in range(max(0,len(values)-3)))
   genre=re.search(r'Dance style:\s*([^\.\n]+)',caption,re.I)
   prediction=dict(id=row['id'],caption=caption,**audit,generation=dict(do_sample=False,max_new_tokens=512,max_length=8192,eos_token_id=processor.tokenizer.eos_token_id,eos_observed=processor.tokenizer.eos_token_id in values,generated_tokens=len(values),hit_generation_cap=len(values)>=512,generated_token_sha256=th(tokens),generated_token_ids=values,repeated_4gram_excess=sum(max(0,n-1) for n in counts.values()),most_repeated_4gram=max(counts.values(),default=0),predicted_genre=genre.group(1).strip() if genre else None,raw_caption_not_edited=True))
   predictions.append(prediction)
   save(OUT/(row['id']+'.omni_raw.json'),prediction)
   event('actual_omni_generate_complete',id=row['id'],caption=caption,generation=prediction['generation'])
 save(OUT/'omni_prediction3_raw.json',dict(model_id=receipt['model_id'],source_checkpoint=receipt['source_checkpoint'],source_checkpoint_marker_sha256=receipt['source_checkpoint_marker_sha256'],checkpoint_adapter_sha256=receipt['deployment_files_sha256']['adapter/adapter_model.safetensors'],updates=3000,full_training_completed=False,predictions=predictions,human_review=False))
 event('omni_fixed3_complete',actual_forward_cases=3)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-test',action='store_true');main(p.parse_args().cpu_test)
