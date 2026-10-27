"""CPU protocol tests; no official model loading or GPU quality claims."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts_omnidance.exp.ms import music_planner_schema as schema
from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter


def contract():
    return dict(schema=schema.SCHEMA, caption_version='test-text-v1',
        task_instruction='Describe style and timed choreography.', audio_min_seconds=1.,
        audio_max_seconds=15., required_markers=['Style:'], require_target_windows=False)


def source(tmp_path, sid='a', split='train', seconds=3.):
    audio = tmp_path / (sid + '.wav')
    sf.write(audio, np.sin(np.arange(round(seconds * 24000)) /
                           (100 + sum(sid.encode()))).astype(np.float32), 24000)
    return dict(id=sid, source_group_id='group-' + sid, audio_path=str(audio),
        audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), split=split,
        audio_duration_seconds=seconds, audio_window_seconds=[0., seconds],
        target_text='Style: test. [0-3s] Raise the left hand.', caption_version='test-text-v1',
        semantic_approval=False, human_review_completed=False)


def json_file(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def manifest(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return path


def args(tmp_path, train, val, selected=None):
    return SimpleNamespace(train=manifest(tmp_path / 'train.jsonl', train),
        val=manifest(tmp_path / 'val.jsonl', val),
        caption_contract=json_file(tmp_path / 'contract.json', selected or contract()),
        genre_classes=None)


def test_arbitrary_continuous_caption_preserved_without_fixed_two_windows(tmp_path):
    a, b = source(tmp_path), source(tmp_path, 'b', 'val', 7.)
    values = args(tmp_path, [a], [b])
    train, val, receipt = schema.read_training_data(values)
    assert train[0]['target_text'] == a['target_text']
    assert val[0]['audio_duration_seconds'] == 7.
    assert receipt['length_policy'] == schema.LENGTH_POLICY
    assert receipt['test_used'] is False
    assert receipt['declared_isolation']['song_id'] == 'partial-or-unknown'
    assert receipt['review_claims_verified_by_program'] is False


@pytest.mark.parametrize('key,value', [('caption_version', 'wrong'), ('split', 'test'),
    ('audio_duration_seconds', 16.), ('audio_window_seconds', [0., 2.]),
    ('target_text', 'No style marker.'), ('target_text', 'Style: <|im_end|>'),
    ('human_review_completed', True), ('song_id', '')])
def test_reject_invalid_data_before_model_loading(tmp_path, key, value):
    a, b = source(tmp_path), source(tmp_path, 'b', 'val')
    a[key] = value
    with pytest.raises(ValueError):
        schema.read_training_data(args(tmp_path, [a], [b]))


@pytest.mark.parametrize('key', ['id', 'source_group_id', 'audio_sha256', 'audio_path',
                                 'song_id', 'near_duplicate_group_id'])
def test_train_val_disjoint_declared_identities(tmp_path, key):
    a, b = source(tmp_path), source(tmp_path, 'b', 'val')
    if key not in a:
        a[key] = 'shared'
    b[key] = a[key]
    with pytest.raises(ValueError, match='leakage'):
        schema.read_training_data(args(tmp_path, [a], [b]))


def test_required_windows_allow_variable_boundaries_but_reject_overlap(tmp_path):
    selected = dict(contract(), require_target_windows=True)
    a, b = source(tmp_path), source(tmp_path, 'b', 'val')
    a['target_windows'] = b['target_windows'] = [dict(start=0., end=1.2, caption='Turn.'),
                                               dict(start=1.2, end=3., caption='Step.')]
    values = args(tmp_path, [a], [b], selected)
    assert len(schema.read_training_data(values)[0]) == 1
    a['target_windows'][1]['start'] = 1.
    manifest(values.train, [a])
    with pytest.raises(ValueError, match='nonoverlapping'):
        schema.read_training_data(values)


def test_contract_unknown_fields_and_nonfinite_bounds_rejected(tmp_path):
    with pytest.raises(ValueError, match='exactly'):
        schema.load_contract(json_file(tmp_path / 'c.json', dict(contract(), unknown=True)))
    with pytest.raises(ValueError, match='Finite'):
        schema.load_contract(json_file(tmp_path / 'c.json', dict(contract(), audio_max_seconds=float('inf'))))


def test_optional_independent_jev_labels_join_without_changing_caption(tmp_path):
    a, b = source(tmp_path), source(tmp_path, 'b', 'val')
    values = args(tmp_path, [a], [b])
    values.genre_classes = json_file(tmp_path / 'classes.json', ['街舞', '古典舞'])
    label = {k: a[k] for k in ('id', 'source_group_id', 'audio_sha256', 'split')}
    label.update(genre='街舞', label_source='independent-human-review', label_version='test-v1')
    values.train_genre_labels = manifest(tmp_path / 'genres.jsonl', [label])
    values.val_genre_labels = None
    train, val, receipt = schema.read_training_data(values)
    assert train[0]['jev_target'] == [1., 0.]
    assert val[0]['jev_target'] is None
    assert train[0]['target_text'] == a['target_text']
    assert receipt['jev']['train']['known_labels'] == 1
    assert receipt['jev']['val']['known_labels'] == 0
    values.train_genre_labels = None
    with pytest.raises(ValueError, match='at least one'):
        schema.read_training_data(values)


def test_genre_file_without_formal_vocabulary_is_rejected(tmp_path):
    values = args(tmp_path, [source(tmp_path)], [source(tmp_path, 'b', 'val')])
    values.train_genre_labels = tmp_path / 'unread.jsonl'
    with pytest.raises(ValueError, match='explicit --genre-classes'):
        schema.read_training_data(values)


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [2] if text == '<|im_end|>' else [ord(c) + 10 for c in text]


class Processor:
    tokenizer = Tokenizer()
    audio_token_id = 7

    def _build_default_prompt(self, text, has_audio=True):
        return 'native-moss:' + text

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]['content'] == [{'type': 'audio', 'audio': 'native-waveform'}]
        return 'native-captioner-audio-only'

    def __call__(self, text, **kwargs):
        self.last_text = text
        return dict(input_ids=torch.tensor([[5, 7, 7, 6]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                    input_features=torch.tensor([[1., 2.]]))


def configured(family, selected=None, enable_jev=True):
    model = torch.nn.Linear(1, 1)
    model.generation_config = SimpleNamespace(eos_token_id=2)
    native = PlannerAdapter(family, model, Processor(), max_length=1000)
    return schema.ConfiguredPlannerAdapter(native, contract=selected or contract(), enable_jev=enable_jev)


@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
def test_configured_native_prefix_exact_target_and_jev_exclusive_boundary(tmp_path, family):
    adapter, row = configured(family), source(tmp_path)
    prefix = adapter.prefix(row)
    batch = adapter.collate([row])
    assert torch.equal(batch['input_ids'][:, :4], prefix['input_ids'])
    assert batch['labels'][0, :4].tolist() == [-100] * 4
    assert batch['labels'][0, 4:].tolist() == adapter.tokenizer.encode(row['target_text']) + [2]
    assert batch['jev_prefix_end'].tolist() == [4]
    if family == 'omni_captioner':
        assert adapter.processor.last_text == ['native-captioner-audio-only']
    else:
        assert adapter.processor.last_text == 'native-moss:' + contract()['task_instruction']
        assert not bool(batch['audio_input_mask'][0, 4:].any())


def test_variable_full_audio_resamples_without_padding_or_cropping(tmp_path):
    adapter, row = configured('omni_captioner'), source(tmp_path, seconds=7.)
    audio = adapter._audio(row)
    assert len(audio) == 112000
    assert audio.dtype == np.float32
    assert not adapter._audio(row, silence=True).any()
    sf.write(row['audio_path'], np.zeros(100), 24000)
    with pytest.raises(ValueError, match='Audio bytes'):
        adapter._audio(row)
    row['audio_sha256'] = hashlib.sha256(open(row['audio_path'], 'rb').read()).hexdigest()
    with pytest.raises(ValueError, match='Waveform length'):
        adapter._audio(row)


def test_inference_prefix_never_reads_caption_or_labels(tmp_path):
    class InputOnly(dict):
        def get(self, key, *default):
            assert key not in ('target_text', 'target_windows', 'genre', 'jev_target')
            return super().get(key, *default)

        def __getitem__(self, key):
            assert key not in ('target_text', 'target_windows', 'genre', 'jev_target')
            return super().__getitem__(key)
    row = source(tmp_path)
    row = InputOnly({k: v for k, v in row.items() if not k.startswith('target')})
    assert configured('omni_captioner').prefix(row)['input_ids'].shape == (1, 4)


def test_no_new_flags_preserves_native_adapter_object():
    native = PlannerAdapter('omni_captioner', torch.nn.Linear(1, 1), Processor())
    assert schema.configure_adapter(native, SimpleNamespace(caption_contract=None, genre_classes=None)) is native


def exported_checkpoint(tmp_path):
    from scripts_omnidance.exp.ms.music_planner_train import object_sha
    checkpoint = tmp_path / 'checkpoint-0000002'
    (checkpoint / 'adapter').mkdir(parents=True)
    json_file(checkpoint / 'adapter' / 'caption_contract.json', contract())
    (checkpoint / 'adapter' / 'adapter_model.safetensors').write_bytes(b'fake-adapter')
    torch.save({'fake': torch.ones(1)}, checkpoint / 'trainable.pt')
    config = dict(family='omni_captioner', data=dict(contract=contract(), test_used=False),
                  max_length=500, jev=None)
    fingerprint = object_sha(config)
    json_file(tmp_path / 'run_config.json', dict(config, fingerprint=fingerprint))
    refs = [dict(path=str(p), sha256=schema._file_hash(p)) for p in checkpoint.rglob('*') if p.is_file()]
    json_file(checkpoint / 'trainer_state.json', dict(schema='music-planner-train-v1', complete=True,
        fingerprint=fingerprint, trainable_sha256=schema._file_hash(checkpoint / 'trainable.pt'), files=refs))
    return checkpoint


def test_restore_contract_after_native_adapter_loading(tmp_path):
    native = PlannerAdapter('omni_captioner', torch.nn.Linear(1, 1), Processor())
    restored = schema.restore_adapter_protocol(native, exported_checkpoint(tmp_path))
    assert restored.model is native.model
    assert restored.contract == contract()
    assert restored.max_length == 500
    assert restored.training_protocol_identity['base_weight_content_verified_by_this_helper'] is False


@pytest.mark.parametrize('changed', ['adapter', 'config', 'portable'])
def test_restore_rejects_mixed_or_mutated_artifacts(tmp_path, changed):
    checkpoint = exported_checkpoint(tmp_path)
    if changed == 'adapter':
        (checkpoint / 'adapter' / 'adapter_model.safetensors').write_bytes(b'other')
    elif changed == 'config':
        value = json.loads((tmp_path / 'run_config.json').read_text())
        value['max_length'] = 1
        json_file(tmp_path / 'run_config.json', value)
    else:
        torch.save({'fake': torch.zeros(1)}, checkpoint / 'trainable.pt')
    native = PlannerAdapter('omni_captioner', torch.nn.Linear(1, 1), Processor())
    with pytest.raises(ValueError, match='identity'):
        schema.restore_adapter_protocol(native, checkpoint)
