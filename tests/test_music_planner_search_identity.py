"""Portable-byte/model-load checks use tiny CPU fixtures, never real pretrained weights."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts_omnidance.exp.ms import music_planner_adapters as adapters
from scripts_omnidance.exp.ms.music_planner_search import file_sha, object_sha
from scripts_omnidance.exp.ms.music_planner_search_identity import audit_model_artifacts, ref, training_sha, verify_loaded_portable


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fingerprint(config):
    canonical = {k: v for k, v in config.items() if k not in ('fingerprint', 'trainable_parameters')}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.fixture
def portable(tmp_path):
    base, run, source = tmp_path / 'base', tmp_path / 'run', tmp_path / 'frozen'
    base.mkdir()
    run.mkdir()
    source.mkdir()
    dump(base / 'config.json', {'model_type': 'qwen3_omni_moe'})
    (base / 'model.safetensors').write_bytes(b'CPU fixture bytes, not a real pretrained model')
    dump(base / 'tokenizer_config.json', {'eos_token_id': 2})
    (source / 'music_planner_adapters.py').write_bytes(Path(adapters.__file__).read_bytes())
    (source / 'music_planner_train.py').write_text('# frozen fixture trainer source\n', encoding='utf-8')
    checkpoint = run / 'checkpoint-0000001'
    export = checkpoint / 'adapter'
    export.mkdir(parents=True)
    signature = {'lora_a': {'shape': [2], 'dtype': 'torch.float32'}}
    tensor = torch.tensor([.5, 1.5])
    torch.save({'lora_a': tensor}, checkpoint / 'trainable.pt')
    torch.save({'lora_a': tensor}, export / 'adapter_model.bin')
    dump(export / 'adapter_config.json', {'peft_type': 'LORA'})
    dump(export / 'tokenizer_config.json', {'eos_token_id': 2})
    dump(export / 'planner_adapter.json', dict(family='omni_captioner', max_length=8192,
        prefix_policy='native-audio-only-no-target-in-prefix', time_markers=False,
        trainable_info={'trainable': [{'name': 'lora_a'}]}))
    config = dict(family='omni_captioner', optimizer='AdamW-FP32-full-states', max_length=8192,
        world_size=1, trainable_signature=signature,
        data=dict(test_used=False, length_policy='one-full-5.0625s-audio-two-text-windows-no-truncation-v1'),
        model=dict(config_sha256=file_sha(base / 'config.json'),
                   weights=[dict(name='model.safetensors', size=(base / 'model.safetensors').stat().st_size)],
                   tokenizer_config_json='ignored'),
        adapter_source_sha256=file_sha(source / 'music_planner_adapters.py'),
        trainer_sha256=file_sha(source / 'music_planner_train.py'))
    config['model'].pop('tokenizer_config_json')
    config['model']['tokenizer_config.json'] = file_sha(base / 'tokenizer_config.json')
    config['fingerprint'] = fingerprint(config)
    dump(run / 'run_config.json', config)
    marker = dict(schema='music-planner-train-v1', complete=True, fingerprint=config['fingerprint'],
                  trainable_signature=signature, world_size=1, zero3=False,
                  cursor=dict(epoch=0, position=1, update=1, source_exposures=1, best_val_nll=None),
                  trainable_sha256=file_sha(checkpoint / 'trainable.pt'),
                  files=[ref(path) for path in sorted(checkpoint.rglob('*')) if path.is_file()])
    dump(checkpoint / 'trainer_state.json', marker)
    model = torch.nn.Module()
    model.lora_a = torch.nn.Parameter(tensor.clone(), requires_grad=False)
    return base, checkpoint, source, SimpleNamespace(family='omni_captioner', model=model)


def test_portable_checkpoint_audits_real_bytes_sources_and_actual_loaded_tensors(portable):
    base, checkpoint, source, adapter = portable
    identity = audit_model_artifacts('omni_captioner', base, checkpoint, source)
    assert identity['mode'] == 'fair_portable_checkpoint'
    assert identity['checkpoint']['training_base_weight_content_sha_verified'] is False
    verified = verify_loaded_portable(adapter, identity)
    assert verified['loaded_portable_state_verified'] is True
    assert verified['artifact_identity_sha256'] == object_sha(identity)
    adapter.model.lora_a.data.add_(1)
    with pytest.raises(ValueError, match='Actual loaded adapter differs'):
        verify_loaded_portable(adapter, identity)


def test_base_only_is_explicit_and_cannot_silently_load_peft(portable):
    base, _, _, adapter = portable
    identity = audit_model_artifacts('omni_captioner', base)
    assert identity['checkpoint'] is None and identity['mode'] == 'native_base_only'
    assert verify_loaded_portable(adapter, identity)['loaded_portable_state_verified'] is False
    adapter.model.peft_config = {'not': 'base-only'}
    with pytest.raises(ValueError, match='unexpectedly loaded an adapter'):
        verify_loaded_portable(adapter, identity)


@pytest.mark.parametrize('change', ('incomplete', 'family', 'protocol', 'test', 'fingerprint', 'cursor'))
def test_incompatible_or_unfinished_training_is_rejected_before_model_load(portable, change):
    base, checkpoint, source, _ = portable
    config_path, marker_path = checkpoint.parent / 'run_config.json', checkpoint / 'trainer_state.json'
    config, marker = json.loads(config_path.read_text()), json.loads(marker_path.read_text())
    if change == 'incomplete':
        marker['complete'] = False
    elif change == 'family':
        config['family'] = 'moss'
    elif change == 'protocol':
        config['schema'] = 'short-plan-cot-training-v1'
    elif change == 'test':
        config['data']['test_used'] = True
    elif change == 'cursor':
        marker['cursor']['update'] = -1
    else:
        marker['fingerprint'] = 'f' * 64
    if change in ('family', 'protocol', 'test'):
        config['fingerprint'] = fingerprint(config)
        marker['fingerprint'] = config['fingerprint']
    dump(config_path, config)
    dump(marker_path, marker)
    with pytest.raises(ValueError):
        audit_model_artifacts('omni_captioner', base, checkpoint, source)


def test_changed_adapter_export_or_unrecorded_payload_is_rejected(portable):
    base, checkpoint, source, _ = portable
    (checkpoint / 'adapter' / 'adapter_model.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='export bytes changed'):
        audit_model_artifacts('omni_captioner', base, checkpoint, source)


def test_arbitrary_peft_folder_is_not_a_completed_portable_checkpoint(portable):
    base, checkpoint, source, _ = portable
    with pytest.raises(FileNotFoundError):
        audit_model_artifacts('omni_captioner', base, checkpoint / 'adapter', source)


def test_exact_training_source_and_actual_inference_adapter_are_both_required(portable, monkeypatch, tmp_path):
    base, checkpoint, source, _ = portable
    with pytest.raises(ValueError, match='Training source identity differs'):
        audit_model_artifacts('omni_captioner', base, checkpoint)
    altered = tmp_path / 'different_adapter.py'
    altered.write_bytes(Path(adapters.__file__).read_bytes() + b'\n# changed inference preprocessing\n')
    monkeypatch.setattr(adapters, '__file__', str(altered))
    with pytest.raises(ValueError, match='Actual inference adapter source differs'):
        audit_model_artifacts('omni_captioner', base, checkpoint, source)


def as_captioner_direct(portable, shared):
    base, checkpoint, source, adapter = portable
    old = json.loads((checkpoint.parent / 'run_config.json').read_text())
    instruction = 'Fixture direct two-window caption instruction, not a real training run.'
    (source / 'music_planner_cot.py').write_text('DIRECT_INSTRUCTION = ' + repr(instruction) + '\n', encoding='utf-8')
    (source / 'music_planner_segment_loss.py').write_text('# CPU fixture segment loss\n', encoding='utf-8')
    (source / 'music_planner_eval.py').write_text('# CPU fixture evaluator\n', encoding='utf-8')
    schema = 'music-planner-captioner-ablation-v3/direct-caption/' + ('reviewed-shared' if shared else 'weak-caption')
    config = dict(schema=schema, mode='direct-caption', require_shared_annotations=shared,
                  family='omni_captioner', model=old['model'],
                  data=dict(old['data'], annotation_schema=schema, provenance_checked=shared), max_length=8192,
                  segment_weights=[0., 0., 1.], parameter_signature=old['trainable_signature'],
                  objective='weighted-mean-of-segment-mean-causal-NLL-EOS-in-caption',
                  code_sha256={path.name: file_sha(path) for path in source.iterdir()})
    config['fingerprint'] = training_sha(config)
    dump(checkpoint.parent / 'run_config.json', config)
    dump(checkpoint / 'adapter' / 'short_plan_adapter.json', dict(schema=schema, mode='direct-caption',
        require_shared_annotations=shared, family='omni_captioner', max_length=8192,
        captioner_audio_only=True, instruction_sha256=training_sha(instruction)))
    marker = json.loads((checkpoint / 'trainer_state.json').read_text())
    marker['fingerprint'] = config['fingerprint']
    marker['files'] = [ref(path) for path in sorted(checkpoint.rglob('*'))
                       if path.is_file() and path.name != 'trainer_state.json']
    dump(checkpoint / 'trainer_state.json', marker)
    return base, checkpoint, source, adapter


@pytest.mark.parametrize('shared', (False, True))
def test_two_explicit_captioner_direct_protocols_use_native_audio_checkpoint_validation(portable, shared):
    base, checkpoint, source, adapter = as_captioner_direct(portable, shared)
    identity = audit_model_artifacts('omni_captioner', base, checkpoint, source)
    assert '/direct-caption/' in identity['checkpoint']['training_protocol']
    assert len(identity['checkpoint']['training_sources']) == 5
    assert verify_loaded_portable(adapter, identity)['loaded_portable_state_verified']


@pytest.mark.parametrize('change', ('short-plan', 'music-flamingo-direct', 'weighted-plan', 'bad-source'))
def test_other_planner_protocols_or_code_changes_are_not_implicitly_treated_as_direct(portable, change):
    base, checkpoint, source, _ = as_captioner_direct(portable, False)
    path = checkpoint.parent / 'run_config.json'
    config = json.loads(path.read_text())
    if change == 'short-plan':
        config['schema'] = 'music-planner-captioner-ablation-v3/short-plan/reviewed-shared'
        config['mode'] = 'short-plan'
    elif change == 'music-flamingo-direct':
        config['family'] = 'music_flamingo'
    elif change == 'weighted-plan':
        config['segment_weights'] = [.2, .3, 1.]
    else:
        with (source / 'music_planner_cot.py').open('a', encoding='utf-8') as stream:
            stream.write('# changed frozen source\n')
    config['fingerprint'] = training_sha({k: v for k, v in config.items() if k != 'fingerprint'})
    dump(path, config)
    marker_path = checkpoint / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    marker['fingerprint'] = config['fingerprint']
    dump(marker_path, marker)
    with pytest.raises(ValueError):
        audit_model_artifacts(config['family'], base, checkpoint, source)
