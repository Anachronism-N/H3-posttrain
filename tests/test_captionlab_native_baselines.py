"""Synthetic native adapters verify contracts, never real audio quality."""
import copy,json,sys
from types import SimpleNamespace
import pytest
import torch
from captionlab import native_baselines as native
from choreobench.audio import inspect_audio
from choreobench.io import write_json
from choreobench.providers import create_provider
from test_choreobench_baseline import dataset_file


def material(tmp_path,family='qwen25_omni'):
    model=tmp_path/'model';model.mkdir()
    kind,arch=native.FAMILIES[family]
    write_json(model/'config.json',dict(model_type=kind,architectures=[arch]))
    (model/'model.safetensors').write_bytes(b'Synthetic artifact; never loaded as real weights')
    dataset,song=dataset_file(tmp_path,seconds=5.0625)
    return dict(kind='native',family=family,model=str(model),task_prompt='Compose one possible solo phrase.'),song['audio']|dict(path=str(tmp_path/'song.wav'))


def runtime(monkeypatch,family,missing=False,unexpected=None):
    calls=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.param=torch.nn.Parameter(torch.ones(1));self.peft_config=None
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            assert kwargs['local_files_only'] and kwargs['output_loading_info']
            return cls(),dict(missing_keys=['unloaded'] if missing else [],unexpected_keys=unexpected or [],mismatched_keys=[],error_msgs=[])
        def generate(self,**kwargs):
            calls.append(kwargs)
            return torch.tensor([[10,11,12,2]])
    class Processor:
        tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0,decode=lambda tokens,**kw:'An original possible phrase.')
        def apply_chat_template(self,messages,**kwargs):
            assert 'audio' in str(messages) and 'Compose one possible solo phrase.' in str(messages)
            return json.dumps(messages)
        def __call__(self,**kwargs):
            assert kwargs['sampling_rate']==16000 and len(kwargs['audio'][0])==81000
            return dict(input_ids=torch.tensor([[10,11]]),attention_mask=torch.tensor([[1,1]]),input_features=torch.ones(1,2))
    name='Qwen2_5OmniThinkerForConditionalGeneration' if family=='qwen25_omni' else 'AudioFlamingo3ForConditionalGeneration'
    transformers=SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**kw:Processor()))
    setattr(transformers,name,Model)
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    return calls


@pytest.mark.parametrize('family',['qwen25_omni','audio_flamingo3'])
def test_real_architecture_dispatch_audio_shape_and_native_eos(tmp_path,monkeypatch,family):
    spec,audio=material(tmp_path,family)
    provider=create_provider(spec,'local');before=copy.deepcopy(provider.identity)
    calls=runtime(monkeypatch,family)
    result=provider.generate(dict(response_kind='local_caption',call_id='x',audio=audio,seed=42,max_output_tokens=16,
                                  reference='private answer not sent',song={'style':'not sent'}))
    assert provider.identity==before and result['identity']==before
    assert result['stop_reason']=='eos' and result['text']=='An original possible phrase.'
    assert result['costs']['input_tokens']==2 and result['costs']['output_tokens']==2
    assert result['raw']['processor_audio_samples']==81000
    assert result['raw']['loaded_artifact_verification']['trained_adapter_loaded'] is False
    assert calls[0]['eos_token_id']==2 and calls[0]['use_cache']
    assert 'reference' not in calls[0] and 'song' not in calls[0]


@pytest.mark.parametrize('change',['architecture','missing_weights','role','checkpoint','bad_prompt','source_for_wrong_family'])
def test_bad_model_contract_rejected_before_load(tmp_path,change):
    spec,audio=material(tmp_path)
    role='local'
    if change=='architecture':write_json(tmp_path/'model/config.json',dict(model_type='moss_music',architectures=['MossMusicModel']),overwrite=True)
    elif change=='missing_weights':(tmp_path/'model/model.safetensors').unlink()
    elif change=='role':role='planner'
    elif change=='checkpoint':spec['checkpoint']='arbitrary-adapter'
    elif change=='bad_prompt':spec['task_prompt']='<|im_start|>'
    else:spec['source_dir']='unrelated'
    with pytest.raises(ValueError):create_provider(spec,role)


def test_failed_load_marks_restart_required_and_no_provider_identity_drift(tmp_path,monkeypatch):
    spec,audio=material(tmp_path);p=native.NativeBaselineProvider(spec);calls=runtime(monkeypatch,'qwen25_omni',missing=True)
    identity=copy.deepcopy(p.identity)
    with pytest.raises(ValueError,match='Incomplete'):p.load()
    assert p.requires_restart and p.model is None and p.identity==identity and not calls
    with pytest.raises(RuntimeError,match='restart'):p.load()


@pytest.mark.parametrize('keys,accepted',[
    (['talker.model.layers.0.weight','token2wav.decoder.weight'],True),
    (['thinker.model.layers.0.foreign_weight'],False),
    (['audio_tower.unknown.weight'],False),
])
def test_qwen25_only_excludes_actual_speech_output_subtrees(tmp_path,monkeypatch,keys,accepted):
    spec,audio=material(tmp_path);provider=native.NativeBaselineProvider(spec)
    runtime(monkeypatch,'qwen25_omni',unexpected=keys)
    if accepted:
        provider.load();assert provider.verification['loading_info']['unexpected_keys']==keys
    else:
        with pytest.raises(ValueError,match='Unexpected native weight'):provider.load()


def test_audio_drift_rejected_before_model_load(tmp_path,monkeypatch):
    spec,audio=material(tmp_path);p=native.NativeBaselineProvider(spec)
    monkeypatch.setattr(p,'load',lambda:pytest.fail('Model loaded before input verification'))
    audio['sha256']='0'*64
    with pytest.raises(ValueError,match='unchanged'):p.generate(dict(response_kind='local_caption',audio=audio,seed=42,max_output_tokens=16))


def test_model_drift_rejected_before_load(tmp_path):
    spec,audio=material(tmp_path);p=native.NativeBaselineProvider(spec)
    (tmp_path/'model/config.json').write_bytes(b'changed model config')
    with pytest.raises(ValueError,match='changed'):p.load()


def test_moss_cached_bridge_keeps_original_prefill_but_passes_cache_position():
    calls=[]
    fn=native._moss_cache(lambda ids,**kw:calls.append((ids,kw)))
    ids=torch.tensor([[1,2,3]])
    fn(ids,past_key_values=None,audio_data='real audio')
    assert calls[0][1]['cache_position'] is None
    fn(ids,past_key_values=SimpleNamespace(get_seq_length=lambda:2),audio_data='original audio')
    assert calls[1][1]['cache_position'].tolist()==[2]
    with pytest.raises(ValueError,match='cached prefix'):fn(ids,past_key_values=SimpleNamespace(get_seq_length=lambda:9))
    with pytest.raises(ValueError,match='cache position'):fn(ids,past_key_values=SimpleNamespace(get_seq_length=lambda:2),cache_position=torch.tensor([1]))


@pytest.mark.parametrize('saved_tensor',['absent','matches','differs'])
def test_moss_native_nonpersistent_position_reconstructed_without_inventing_weight(tmp_path,monkeypatch,saved_tensor):
    spec,_=material(tmp_path,'moss_audio');source=tmp_path/'official'/'src';source.mkdir(parents=True)
    sha={}
    from choreobench.io import file_sha
    for name in ('configuration_moss_audio.py','modeling_moss_audio.py','processing_moss_audio.py'):
        p=source/name;p.write_text('Synthetic source identity only');sha[name]=file_sha(p)
    spec.update(source_dir=str(source.parent),source_sha256=sha)
    key='audio_encoder.embed_positions.inv_timescales'
    write_json(tmp_path/'model/model.safetensors.index.json',dict(weight_map={} if saved_tensor=='absent' else {key:'model.safetensors'}))
    class Position(torch.nn.Module):
        def __init__(self,n,d):
            super().__init__();self.register_buffer('inv_timescales',torch.arange(1,d//2+1,dtype=torch.float32),persistent=False)
    class Config:
        def __init__(self,**kwargs):pass
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoConfig=SimpleNamespace(register=lambda *a,**k:None)))
    class Model(torch.nn.Module):
        config_class=Config
        def __init__(self):
            super().__init__();self.param=torch.nn.Parameter(torch.ones(1));self.peft_config=None
            self.config=SimpleNamespace(audio_config=SimpleNamespace(max_source_positions=10,d_model=4))
            self.audio_encoder=torch.nn.Module();self.audio_encoder.embed_positions=Position(10,4)
            self.audio_encoder.embed_positions._buffers['inv_timescales']=torch.full((2,),float('nan'))
            self.audio_encoder.layers=[]
        @classmethod
        def from_pretrained(cls,*a,**k):return cls(),dict(missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[])
        def prepare_inputs_for_generation(self,*a,**k):pass
    Model.config_class=lambda **kwargs:SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoConfig=SimpleNamespace(register=lambda *a,**k:None)))
    monkeypatch.setattr(native,'_source_classes',lambda root:(Model,SimpleNamespace(from_pretrained=lambda *a,**k:SimpleNamespace())))
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def get_tensor(self,k):return torch.tensor([1.,2.]) if saved_tensor=='matches' else torch.tensor([1.,3.])
    def safe_open(*a,**k):
        assert saved_tensor!='absent','No invented checkpoint position tensor'
        return Stream()
    monkeypatch.setitem(sys.modules,'safetensors',SimpleNamespace(safe_open=safe_open))
    provider=native.NativeBaselineProvider(spec)
    if saved_tensor=='differs':
        with pytest.raises(ValueError,match='position differs'):provider.load()
        assert provider.requires_restart and provider.model is None
    else:
        provider.load();position=provider.model.audio_encoder.embed_positions
        assert torch.equal(position.inv_timescales,torch.tensor([1.,2.]))
        assert 'inv_timescales' not in position.state_dict()
        assert provider.verification['position_buffer']['checkpoint_contains_buffer']==(saved_tensor=='matches')
        assert not provider.verification['position_buffer']['weights_modified']


def test_whisper_return_bridge_handles_tensor_and_original_tuple():
    tensor=torch.ones(2,3)
    assert native._whisper_compat(lambda *a,**kw:tensor)()[0] is tensor
    value=(tensor,'extra')
    assert native._whisper_compat(lambda *a,**kw:value)() is value


@pytest.mark.parametrize('persistent',[False,True])
def test_moss_official_nonpersistent_constructor_not_missing_checkpoint(tmp_path,persistent):
 class Position(torch.nn.Module):
  def __init__(self,n,d):
   super().__init__();self.register_buffer('inv_timescales',torch.exp(-torch.arange(d//2).float()),persistent=persistent)
 pos=Position(5,8);pos.inv_timescales.fill_(float('nan'))
 model=SimpleNamespace(audio_encoder=SimpleNamespace(embed_positions=pos),config=SimpleNamespace(audio_config=SimpleNamespace(max_source_positions=5,d_model=8)))
 write_json(tmp_path/'model.safetensors.index.json',dict(weight_map={}))
 if persistent:
  with pytest.raises(ValueError,match='Missing persistent'):native._restore_moss_position(model,tmp_path)
 else:
  receipt=native._restore_moss_position(model,tmp_path)
  assert receipt['origin']=='official-nonpersistent-native-constructor' and not receipt['weights_modified']
  assert torch.equal(pos.inv_timescales,Position(5,8).inv_timescales)


def test_moss_checkpoint_position_mismatch_remains_failure(tmp_path):
 from safetensors.torch import save_file
 class Position(torch.nn.Module):
  def __init__(self,n,d):
   super().__init__();self.register_buffer('inv_timescales',torch.ones(d//2),persistent=False)
 model=SimpleNamespace(audio_encoder=SimpleNamespace(embed_positions=Position(5,8)),config=SimpleNamespace(audio_config=SimpleNamespace(max_source_positions=5,d_model=8)))
 key='audio_encoder.embed_positions.inv_timescales'
 write_json(tmp_path/'model.safetensors.index.json',dict(weight_map={key:'weights.safetensors'}))
 save_file({key:torch.zeros(4)},str(tmp_path/'weights.safetensors'))
 with pytest.raises(ValueError,match='differs'):native._restore_moss_position(model,tmp_path)
