"""Native adapters for Qwen2.5-Omni, Audio-Flamingo3 and actual MOSS-Audio.

No label/genre/reference reaches generation. This is not a trained captioner
loader and never aliases MOSS-Audio to MOSS-Music.
"""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import time
import types

from choreobench.audio import inspect_audio
from choreobench.io import file_sha,object_sha,read_json
from choreobench.providers import _allowed,_budget

def _stat(path):
    s=path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

FAMILIES={'qwen25_omni':('qwen2_5_omni','Qwen2_5OmniModel'),
          'audio_flamingo3':('audioflamingo3','AudioFlamingo3ForConditionalGeneration'),
          'moss_audio':('moss_audio','MossAudioModel')}


def _source_classes(source_dir):
    source=Path(source_dir).resolve()
    package='choreobench_moss_audio_'+object_sha(dict(path=str(source),sha256={
        name:file_sha(source/'src'/name) for name in ('configuration_moss_audio.py','modeling_moss_audio.py','processing_moss_audio.py')}))[:12]
    if package not in sys.modules:
        root=types.ModuleType(package);root.__path__=[str(source)];sys.modules[package]=root
        src=types.ModuleType(package+'.src');src.__path__=[str(source/'src')];sys.modules[package+'.src']=src
    config_name=package+'.src.configuration_moss_audio'
    if config_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(config_name,source/'src/configuration_moss_audio.py')
        module=importlib.util.module_from_spec(spec);sys.modules[config_name]=module;spec.loader.exec_module(module)
    # Official model uses one absolute local import; bind that import to its
    # exact source sibling without stealing a process-global package name.
    model_name=package+'.src.modeling_moss_audio'
    if model_name not in sys.modules:
        path=source/'src/modeling_moss_audio.py';text=path.read_text()
        old='from src.configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig'
        if text.count(old)!=1:raise ValueError('Official MOSS-Audio source import contract changed')
        module=types.ModuleType(model_name);module.__file__=str(path);module.__package__=package+'.src';sys.modules[model_name]=module
        exec(compile(text.replace(old,'from .configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig'),str(path),'exec'),module.__dict__)
    processor_name=package+'.src.processing_moss_audio'
    if processor_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(processor_name,source/'src/processing_moss_audio.py')
        module=importlib.util.module_from_spec(spec);sys.modules[processor_name]=module;spec.loader.exec_module(module)
    return sys.modules[model_name].MossAudioModel,sys.modules[processor_name].MossAudioProcessor


def _whisper_compat(native):
    def forward(*args,**kwargs):
        result=native(*args,**kwargs)
        import torch
        return (result,) if isinstance(result,torch.Tensor) else result
    return forward


def _moss_cache(native):
    def prepare(input_ids,past_key_values=None,cache_position=None,**kwargs):
        cached=int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        if cached and cache_position is None:cache_position=input_ids.new_tensor([cached])
        if cached and int(cache_position[0])!=cached:raise ValueError('MOSS-Audio cache position differs from stored sequence length')
        if cached and input_ids.shape[1] not in (1,cached+1):raise ValueError('Unexpected MOSS-Audio cached prefix')
        return native(input_ids,past_key_values=past_key_values,cache_position=cache_position,**kwargs)
    return prepare


def _restore_moss_position(model, model_path):
    import torch
    from safetensors import safe_open
    position=model.audio_encoder.embed_positions
    with torch.device('cpu'):
        reference=type(position)(model.config.audio_config.max_source_positions,model.config.audio_config.d_model).inv_timescales.float()
    if not torch.isfinite(reference).all():raise ValueError('Nonfinite native position constructor')
    key='audio_encoder.embed_positions.inv_timescales'
    index=read_json(model_path/'model.safetensors.index.json')
    if key in index['weight_map']:
        shard=(model_path/index['weight_map'][key]).resolve()
        if shard.parent!=model_path:raise ValueError('MOSS position shard escapes model root')
        with safe_open(str(shard),framework='pt',device='cpu') as stream:official=stream.get_tensor(key)
        if not torch.equal(reference.to(official.dtype),official):raise ValueError('Official MOSS position differs from native constructor')
        origin='checkpoint-equal-native-constructor'
    else:
        if 'inv_timescales' not in position._non_persistent_buffers_set:
            raise ValueError('Missing persistent MOSS position weights')
        origin='official-nonpersistent-native-constructor'
    position._buffers['inv_timescales']=reference
    return dict(origin=origin,checkpoint_contains_buffer=key in index['weight_map'],
                buffer_values_sha256=object_sha(reference.tolist()),dtype='float32',weights_modified=False)


class NativeBaselineProvider:
    def __init__(self,spec,role='local',base_dir=None):
        _allowed(spec,('kind','family','model','device','max_length','temperature','top_p','top_k','task_prompt','source_dir','source_sha256'))
        if role!='local' or spec.get('family') not in FAMILIES or spec.get('kind')!='native':
            raise ValueError('Require explicit supported native audio baseline')
        self.spec=copy.deepcopy(spec);self.family=spec['family'];self.model_path=Path(spec['model'])
        if not self.model_path.is_absolute():self.model_path=Path(base_dir or '.')/self.model_path
        self.model_path=self.model_path.resolve();config=read_json(self.model_path/'config.json')
        expected=FAMILIES[self.family]
        if config.get('model_type')!=expected[0] or expected[1] not in config.get('architectures',[]):
            raise ValueError('Audio baseline family differs from actual model architecture')
        prompt=spec.get('task_prompt')
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>16384 or '<|' in prompt or '<audio>' in prompt:
            raise ValueError('Require bounded plain task prompt')
        if self.family!='moss_audio' and ('source_dir' in spec or 'source_sha256' in spec):
            raise ValueError('External model source applies only to MOSS-Audio')
        self.max_length=spec.get('max_length',8192)
        if type(self.max_length) is not int or not 1<=self.max_length<=65536:raise ValueError('Invalid native context limit')
        self.sampling=dict(temperature=spec.get('temperature',.9),top_p=spec.get('top_p',.95),top_k=spec.get('top_k',50))
        from scripts_omnidance.exp.ms.music_planner_search import validate_sampling
        validate_sampling(dict(self.sampling,max_new_tokens=1))
        files=[p for p in self.model_path.iterdir() if p.is_file()]
        if not any(p.suffix=='.safetensors' for p in files):raise ValueError('Actual native weights required')
        self.refs={str(p):file_sha(p) for p in files};self.source=None
        if self.family=='moss_audio':
            self.source=Path(spec.get('source_dir','')).resolve();sha=spec.get('source_sha256')
            if not isinstance(sha,dict) or set(sha)!={'configuration_moss_audio.py','modeling_moss_audio.py','processing_moss_audio.py'}:
                raise ValueError('MOSS-Audio needs three exact pinned official source files')
            for name,digest in sha.items():
                p=self.source/'src'/name
                if file_sha(p)!=digest:raise ValueError('MOSS-Audio official source SHA differs')
                self.refs[str(p)]=digest
        self.identity=dict(kind='native',family=self.family,model=str(self.model_path),model_files_sha256=dict(self.refs),
            source_model_type=config['model_type'],checkpoint=None,task_prompt=prompt,task_prompt_sha256=object_sha(prompt),
            max_length=self.max_length,sampling=self.sampling,device=spec.get('device','cpu'),synthetic=False,
            protocol='captionlab-native-independent-audio-baselines-v1',provider_source_sha256=file_sha(Path(__file__)),
            native_compatibility='MOSS-Audio Whisper return/cache bridge only; no MOSS-Music alias' if self.family=='moss_audio' else 'installed Transformers native architecture',
            loaded_verified=False,quality_verified=False)
        # Runtime status is separate so provider identity never changes on load.
        self.stat_signatures={path:_stat(Path(path)) for path in self.refs}
        self.implementation_sha256=file_sha(Path(__file__))
        self.model=None;self.processor=None;self.requires_restart=False;self.verification=None

    def _unchanged(self):
        if file_sha(Path(__file__))!=self.implementation_sha256:raise ValueError('Native implementation changed')
        for path,signature in self.stat_signatures.items():
            if _stat(Path(path))!=signature:raise ValueError('Audited model or source changed')

    def load(self):
        if self.requires_restart:raise RuntimeError('Native baseline initialization failed; restart needed')
        if self.model is not None:return
        import torch
        self._unchanged()
        try:
            position_verification=None
            if self.family=='moss_audio':
                cls,processor_cls=_source_classes(self.source)
                from transformers import AutoConfig
                AutoConfig.register('moss_audio',cls.config_class,exist_ok=True)
                config=cls.config_class(**read_json(self.model_path/'config.json'))
                processor=processor_cls.from_pretrained(self.model_path,local_files_only=True,enable_time_marker=True)
                model,loading=cls.from_pretrained(self.model_path,config=config,local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
                position_verification=_restore_moss_position(model,self.model_path)
                for layer in model.audio_encoder.layers:layer.forward=_whisper_compat(layer.forward)
                model.prepare_inputs_for_generation=_moss_cache(model.prepare_inputs_for_generation)
            else:
                import transformers
                cls=getattr(transformers,'Qwen2_5OmniThinkerForConditionalGeneration' if self.family=='qwen25_omni' else 'AudioFlamingo3ForConditionalGeneration')
                processor=transformers.AutoProcessor.from_pretrained(self.model_path,local_files_only=True)
                model,loading=cls.from_pretrained(self.model_path,local_files_only=True,dtype=torch.bfloat16,output_loading_info=True)
            loading={key:sorted(value) if isinstance(value,set) else value for key,value in loading.items()}
            if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
                raise ValueError('Incomplete native baseline weight loading')
            unexpected=loading.get('unexpected_keys',[])
            allowed=(lambda k:k=='audio_encoder.embed_positions.inv_timescales') if self.family=='moss_audio' else (
                (lambda k:k.startswith(('talker.','token2wav.'))) if self.family=='qwen25_omni' else (lambda k:False))
            if any(not allowed(k) for k in unexpected):raise ValueError('Unexpected native weight keys')
            model=model.to(self.spec.get('device','cpu')).eval()
            if getattr(model,'peft_config',None):raise ValueError('Base baseline unexpectedly loaded a trained adapter')
            self._unchanged()
            self.processor,self.model=processor,model
            self.verification=dict(model_class=type(model).__name__,processor_class=type(processor).__name__,
                loading_info=loading,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                model_artifact_identity_sha256=object_sha(self.identity),trained_adapter_loaded=False,position_buffer=position_verification)
        except Exception:
            self.requires_restart=True;raise

    def generate(self,request):
        _budget(request)
        if request['response_kind']!='local_caption':raise ValueError('Native baseline only consumes audio caption requests')
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly
        from scripts_omnidance.exp.ms.music_planner_search import classify_stop
        audio=request['audio'];actual=inspect_audio(audio['path'])
        if any(actual[k]!=audio[k] for k in ('sha256','sample_rate','num_samples')) or actual['num_samples']/actual['sample_rate']!=5.0625:
            raise ValueError('Require unchanged complete original 5.0625s waveform')
        wave,rate=sf.read(audio['path'],dtype='float32',always_2d=True)
        if wave.shape[1]!=1 or not np.isfinite(wave).all():raise ValueError('Require finite mono source waveform')
        wave=wave[:,0]
        if rate!=16000:
            gcd=math.gcd(rate,16000);wave=resample_poly(wave,16000//gcd,rate//gcd).astype(np.float32)
        if len(wave)!=81000:raise ValueError('Resampled full waveform has wrong length')
        self._unchanged();self.load()
        if self.family=='moss_audio':
            batch=dict(self.processor(text=self.spec['task_prompt'],audios=[wave],return_tensors='pt'))
            batch['audio_input_mask']=batch['input_ids'].eq(self.processor.audio_token_id)
            prompt=self.processor._build_default_prompt(self.spec['task_prompt'],has_audio=True)
        else:
            content=[dict(type='audio',audio='native-waveform'),dict(type='text',text=self.spec['task_prompt'])]
            prompt=self.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
            batch=dict(self.processor(text=[prompt],audio=[wave],sampling_rate=16000,return_tensors='pt',
                text_kwargs=dict(padding=False),audio_kwargs=dict(padding='max_length',return_attention_mask=True)))
        if any('pixel_values' in k or 'grid_thw' in k for k in batch):raise ValueError('Visual inputs forbidden in audio baseline')
        device=next(self.model.parameters()).device
        batch={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in batch.items()}
        input_tokens=batch['input_ids'].shape[1]
        if input_tokens+request['max_output_tokens']>self.max_length:raise ValueError('Native context exceeded; no truncation')
        eos=self.processor.tokenizer.eos_token_id
        if type(eos) is not int:raise ValueError('Native tokenizer EOS must be explicit')
        devices=[] if device.type=='cpu' else [device.index or 0]
        started=time.monotonic()
        with torch.no_grad(),torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(request['seed'])
            for index in devices:torch.cuda.default_generators[index].manual_seed(request['seed'])
            generated=self.model.generate(**batch,max_new_tokens=request['max_output_tokens'],do_sample=True,num_beams=1,
                **self.sampling,eos_token_id=eos,pad_token_id=self.processor.tokenizer.pad_token_id or eos,
                repetition_penalty=1.,no_repeat_ngram_size=0,use_cache=True)
        if generated.ndim!=2 or generated.shape[0]!=1 or not torch.equal(generated[:,:input_tokens],batch['input_ids']):
            raise ValueError('Native generation changed its input prefix')
        tokens=generated[0,input_tokens:].tolist();stop=classify_stop(tokens,[eos],request['max_output_tokens'])
        if eos in tokens and tokens.index(eos)!=len(tokens)-1:stop='tokens_after_eos'
        text=self.processor.tokenizer.decode(tokens,skip_special_tokens=True)
        self._unchanged()
        if file_sha(audio['path'])!=audio['sha256']:raise ValueError('Audio changed during native generation')
        return dict(text=text,identity=self.identity,synthetic=False,stop_reason=stop,
            raw=dict(new_token_ids=tokens,formatted_task_prompt=prompt,native_eos_ids=[eos],
                     processor_audio_samples=len(wave),processor_audio_rate=16000,source_audio_sha256=actual['sha256'],
                     loaded_model_class=type(self.model).__name__,loaded_processor_class=type(self.processor).__name__,
                     loaded_artifact_verification=self.verification),
            costs=dict(input_tokens=input_tokens,output_tokens=len(tokens),latency_seconds=time.monotonic()-started))
