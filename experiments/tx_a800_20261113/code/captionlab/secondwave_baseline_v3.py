"""Distinct native Qwen2.5/AudioFlamingo loaders; no substitution by related model families."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math
import sys
import types
import torch
from captionlab.baseline_eval import run_baseline


def terminal_ids(model,tokenizer):
    value=model.generation_config.eos_token_id
    if value is None:
        value=model.config.eos_token_id
    if value is None:
        raise ValueError('No native text EOS configured')
    ids=list(value) if isinstance(value,(tuple,list)) else [value]
    if not ids or any(not isinstance(i,int) or i<0 or i>=len(tokenizer) for i in ids):
        raise ValueError('Invalid native EOS')
    if tokenizer.eos_token_id not in ids:
        raise ValueError('Model EOS differs from native tokenizer EOS')
    return ids


class SecondWaveProvider:
    def __init__(self,spec,role,base_dir):
        if role!='local' or spec['family'] not in ('qwen25_omni','audio_flamingo3','moss_audio'):raise ValueError('Distinct native architecture required')
        self.spec=spec;self.root=Path(spec['model']);self.model=None
        files=[]
        for p in sorted(self.root.iterdir()):
            if p.is_file() and p.suffix in ('.json','.safetensors','.jinja'):
                digest=hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda:stream.read(8388608),b''):digest.update(block)
                files.append(dict(name=p.name,size=p.stat().st_size,sha256=digest.hexdigest()))
        self.identity=dict(kind='native',family=spec['family'],model=str(self.root),files=files,
                           task_prompt=spec['task_prompt'],sampling={k:spec[k] for k in ('temperature','top_p','top_k')},
                           device=spec['device'],synthetic=False,protocol='sol61-native-secondwave-v1',checkpoint=None)
    def load(self):
        if self.model is not None:return
        from transformers import AutoProcessor,Qwen2_5OmniThinkerForConditionalGeneration,AudioFlamingo3ForConditionalGeneration
        if self.spec['family']=='moss_audio':
            sys.path.insert(0,self.spec['official_source_root'])
            from src.modeling_moss_audio import MossAudioModel,SinusoidsPositionEmbedding
            from src.processing_moss_audio import MossAudioProcessor
            cls=MossAudioModel
            self.processor=MossAudioProcessor.from_pretrained(self.root,local_files_only=True,enable_time_marker=True)
        else:
            cls=Qwen2_5OmniThinkerForConditionalGeneration if self.spec['family']=='qwen25_omni' else AudioFlamingo3ForConditionalGeneration
            self.processor=AutoProcessor.from_pretrained(self.root,local_files_only=True)
        self.model=cls.from_pretrained(self.root,dtype=torch.bfloat16,device_map={'':self.spec['device']},local_files_only=True).eval().requires_grad_(False)
        if self.spec['family']=='moss_audio':
            # Restore the official constructor-only, nonpersistent frequency buffer.
            owner=self.model.audio_encoder.embed_positions
            cfg=self.model.config.audio_config
            canonical=SinusoidsPositionEmbedding(cfg.max_source_positions,cfg.d_model).inv_timescales.float()
            if not bool(torch.isfinite(canonical).all()):raise ValueError('Invalid native position frequencies')
            owner._buffers['inv_timescales']=canonical.to(owner.inv_timescales.device)
            for layer in self.model.audio_encoder.layers:
                layer._attn_implementation='eager'
                old=layer.forward
                def compatible(*args,_old=old,**kwargs):
                    out=_old(*args,**kwargs)
                    return (out,) if isinstance(out,torch.Tensor) else out
                layer.forward=compatible
            original=self.model.prepare_inputs_for_generation
            def prepare(model,input_ids,past_key_values=None,cache_position=None,**kwargs):
                cached=int(past_key_values.get_seq_length()) if past_key_values is not None else 0
                if cached and cache_position is None:cache_position=input_ids.new_tensor([cached])
                return original(input_ids,past_key_values=past_key_values,cache_position=cache_position,**kwargs)
            self.model.prepare_inputs_for_generation=types.MethodType(prepare,self.model)
        self.forward_count=0
        def finite(module,inputs,out):
            if not bool(torch.isfinite(out.logits[:,-1]).all()):raise FloatingPointError('Nonfinite native generation logits')
            self.forward_count+=1
        self.model.register_forward_hook(finite)
    def generate(self,request):
        if request['response_kind']!='local_caption':raise ValueError('Audio-only local caption request')
        self.load();p=Path(request['audio']['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=request['audio']['sha256']:raise ValueError('Original WAV changed')
        audio,rate=sf.read(p,dtype='float32',always_2d=True)
        if audio.shape[1]!=1 or len(audio)!=request['audio']['num_samples']:raise ValueError('Original mono geometry mismatch')
        divisor=math.gcd(rate,16000);waveform=resample_poly(audio[:,0],16000//divisor,rate//divisor).astype(np.float32) if rate!=16000 else audio[:,0]
        prompt=self.processor._build_default_prompt(self.spec['task_prompt'],has_audio=True) if self.spec['family']=='moss_audio' else self.processor.apply_chat_template([dict(role='user',content=[dict(type='audio',audio='native-waveform'),dict(type='text',text=self.spec['task_prompt'])])],tokenize=False,add_generation_prompt=True)
        if self.spec['family']=='moss_audio':
            batch=self.processor(text=prompt,audios=[waveform],return_tensors='pt')
            batch['audio_input_mask']=batch['input_ids'].eq(self.processor.audio_token_id)
        elif self.spec['family']=='audio_flamingo3':
            batch=self.processor(text=[prompt],audio=[waveform],sampling_rate=16000,return_tensors='pt',text_kwargs={'padding':False},audio_kwargs={'padding':'max_length'})
        else:
            batch=self.processor(text=[prompt],audio=[waveform],sampling_rate=16000,return_tensors='pt',padding=False)
        moved={k:v.to(device=self.spec['device'],dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
        n=batch['input_ids'].shape[1];budget=request['max_output_tokens']
        if n+budget>self.spec['max_length']:raise ValueError('Context truncation forbidden')
        eos=terminal_ids(self.model,self.processor.tokenizer)
        torch.manual_seed(request['seed']);start=time.monotonic();before=self.forward_count
        with torch.inference_mode():
            seq=self.model.generate(**moved,do_sample=True,temperature=self.spec['temperature'],top_p=self.spec['top_p'],top_k=self.spec['top_k'],max_new_tokens=budget,eos_token_id=eos,repetition_penalty=1.,no_repeat_ngram_size=0,use_cache=True)
        ids=seq[0,n:].cpu().tolist();text=self.processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        stop='eos' if ids and ids[-1] in eos else 'budget_truncation'
        return dict(text=text,identity=self.identity,synthetic=False,stop_reason=stop,
                    raw=dict(actual_prompt=prompt,new_token_ids=ids,eos_ids=eos,input_tokens=n,processor_resampled_rate=16000,original_samples=len(audio),parameter_storage='BF16',audio_feature_shapes={k:list(v.shape) for k,v in batch.items() if k not in ('input_ids','attention_mask')},native_audio_padding='max_length' if self.spec['family']=='audio_flamingo3' else 'native',actual_finite_forwards=self.forward_count-before),
                    costs=dict(input_tokens=n,output_tokens=len(ids),latency_seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--provider-config',required=True);p.add_argument('--method-id',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    result=run_baseline(a.manifest,a.provider_config,a.method_id,a.out_dir,provider_factory=lambda spec,role,base_dir:SecondWaveProvider(spec,role,base_dir))
    print(json.dumps({k:result[k] for k in ('case_count','complete_count','failed_count','unknown_count')}),flush=True)


if __name__=='__main__':main()
