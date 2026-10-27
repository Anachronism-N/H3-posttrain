"""CPU protocol tests; no official model loading or GPU quality claims."""
import hashlib
import ast
import json
import shutil
from pathlib import Path
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

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(('<|im_end|>' if i == 2 else chr(i - 10)) for i in ids
                       if i != 2 or not skip_special_tokens)


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


def conditioned_contract():
    return dict(contract(), schema=schema.CONDITIONED_SCHEMA, dance_caption_format=dict(
        schema='music-planner-dance-caption-v1', genre_header_template='Style: {genre}\n',
        time_tolerance_seconds=.001, require_full_coverage=True))


def conditioned_source(tmp_path, sid='a', split='train'):
    row = source(tmp_path, sid, split)
    row['target_text'] = 'Style: alpha\n[0.000-1.200s] Turn left.\n[1.200-3.000s] Step right.'
    row['target_windows'] = [dict(start=0., end=1.2, caption='Turn left.'),
                             dict(start=1.2, end=3., caption='Step right.')]
    return row


def conditioned_args(tmp_path):
    a, b = conditioned_source(tmp_path), conditioned_source(tmp_path, 'b', 'val')
    values = args(tmp_path, [a], [b], conditioned_contract())
    values.genre_classes = json_file(tmp_path / 'classes.json', ['alpha', 'beta'])
    label = {k: a[k] for k in ('id', 'source_group_id', 'audio_sha256', 'split')}
    label.update(genre='alpha', label_source='independent-label', label_version='v1')
    values.train_genre_labels = manifest(tmp_path / 'genres.jsonl', [label])
    values.val_genre_labels = None
    return values, a, b, label


def test_v2_enforces_finite_genre_timestamps_and_independent_label_consistency(tmp_path):
    values, a, b, label = conditioned_args(tmp_path)
    train, val, data = schema.read_training_data(values)
    assert train[0]['caption_genre'] == val[0]['caption_genre'] == 'alpha'
    assert data['jev']['task'] == 'single-categorical-hard-v2'
    label['genre'] = 'beta'
    manifest(values.train_genre_labels, [label])
    with pytest.raises(ValueError, match='disagree'):
        schema.read_training_data(values)


def test_v2_does_not_silently_convert_soft_labels_to_hard_targets(tmp_path):
    values, a, b, label = conditioned_args(tmp_path)
    label.pop('genre')
    label['genre_distribution'] = dict(alpha=.8, beta=.2)
    manifest(values.train_genre_labels, [label])
    with pytest.raises(ValueError, match='disagree'):
        schema.read_training_data(values)


@pytest.mark.parametrize('mutation', ['unknown_class', 'overlap', 'metadata', 'no_classes'])
def test_v2_rejects_wrong_caption_before_model_loading(tmp_path, mutation):
    values, a, b, label = conditioned_args(tmp_path)
    if mutation == 'unknown_class':
        a['target_text'] = a['target_text'].replace('alpha', 'unknown')
    elif mutation == 'overlap':
        a['target_text'] = a['target_text'].replace('[1.200-3.000s]', '[1.000-3.000s]')
    elif mutation == 'metadata':
        a['target_windows'][0]['caption'] = 'A different movement.'
    else:
        values.genre_classes = None
    manifest(values.train, [a])
    with pytest.raises(ValueError):
        schema.read_training_data(values)


@pytest.mark.parametrize('family', ['moss', 'omni_captioner'])
def test_v2_collate_uses_explicit_header_boundary_not_whole_text_bpe(tmp_path, family):
    class BoundaryTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.startswith('Style:') and '\n[' in text:
                left, right = text.split('\n[', 1)
                return super().encode(left) + [999999] + super().encode(right)
            return super().encode(text)
    native = PlannerAdapter(family, torch.nn.Linear(1, 1), Processor(), max_length=1000)
    native.processor.tokenizer = BoundaryTokenizer()
    native.model.generation_config = SimpleNamespace(eos_token_id=2)
    selected = schema.ConfiguredPlannerAdapter(native, contract=conditioned_contract(),
        enable_jev=True, classes=['alpha', 'beta'])
    row = conditioned_source(tmp_path)
    batch = selected.collate([row])
    header = selected.header_token_ids['alpha']
    assert batch['labels'][0, :4].tolist() == [-100] * 4
    assert batch['labels'][0, 4:4 + len(header)].tolist() == header
    assert batch['labels'][0, 4:].tolist() != native.tokenizer.encode(row['target_text']) + [2]
    assert 999999 not in batch['labels']
    assert batch['labels'][0, -1] == 2
    assert batch['jev_prefix_end'].tolist() == [4]
    assert selected.token_protocol == schema.TOKEN_PROTOCOL
    if family == 'moss':
        assert not batch['audio_input_mask'][0, 4:].any()


def test_v2_generate_routes_to_conditioned_path_and_v1_rejects_fake_control(tmp_path, monkeypatch):
    from scripts_omnidance.exp.ms import music_planner_moss_generate as generation
    native = PlannerAdapter('moss', torch.nn.Linear(1, 1), Processor(), max_length=1000)
    native.model.generation_config = SimpleNamespace(eos_token_id=2)
    selected = schema.ConfiguredPlannerAdapter(native, contract=conditioned_contract(),
        enable_jev=True, classes=['alpha', 'beta'])
    observed = []
    def generate(adapter, row, **kwargs):
        observed.append((adapter, kwargs))
        return dict(caption_valid=True, final_caption='test')
    monkeypatch.setattr(generation, 'generate_conditioned', generate)
    result = selected.generate(conditioned_source(tmp_path), selected_genre='beta', max_new_tokens=30)
    assert result['caption_valid'] is True
    assert observed == [(selected, dict(selected_genre='beta', max_new_tokens=30))]
    with pytest.raises(ValueError, match='explicit verified audio'):
        selected.generate(conditioned_source(tmp_path), silence=True)
    with pytest.raises(ValueError, match='trained v2'):
        configured('moss').generate(source(tmp_path), selected_genre='alpha')


def test_v2_rounding_tolerance_applies_to_text_and_metadata_together(tmp_path):
    values, a, b, label = conditioned_args(tmp_path)
    a['target_text'] = a['target_text'].replace('3.000s', '3.001s')
    a['target_windows'][-1]['end'] = 3.001
    manifest(values.train, [a])
    train, _, _ = schema.read_training_data(values)
    assert train[0]['target_windows'][-1]['end'] == 3.001


def protocol_native():
    model = torch.nn.Module()
    model.config = SimpleNamespace(language_config=SimpleNamespace(hidden_size=3))
    model.language_model = torch.nn.Module()
    model.language_model.norm = torch.nn.LayerNorm(3)
    model.generation_config = SimpleNamespace(eos_token_id=2)
    return PlannerAdapter('moss', model, Processor(), max_length=1000)


def exported_conditioned_checkpoint(tmp_path, profile=None, caption_classes=None, semantic_classes=None):
    from scripts_omnidance.exp.ms import music_planner_jev as jev
    from scripts_omnidance.exp.ms.music_planner_train import object_sha, trainable_state
    profile = conditioned_contract() if profile is None else profile
    caption_classes = ['alpha', 'beta'] if caption_classes is None else caption_classes
    semantic_classes = caption_classes if semantic_classes is None else semantic_classes
    adapter = schema.ConfiguredPlannerAdapter(protocol_native(), contract=profile,
        enable_jev=True, classes=caption_classes)
    installed = jev.install_jev(adapter, semantic_classes, head_dim=4)
    checkpoint = tmp_path / 'checkpoint-0000002'
    (checkpoint / 'adapter').mkdir(parents=True)
    json_file(checkpoint / 'adapter' / 'caption_contract.json', profile)
    json_file(checkpoint / 'adapter' / 'dance_caption_protocol.json', adapter.dance_caption_protocol)
    (checkpoint / 'adapter' / 'adapter_model.safetensors').write_bytes(b'fixture-native-export')
    portable = trainable_state(adapter.model)
    torch.save(portable, checkpoint / 'trainable.pt')
    config = dict(family='moss', data=dict(contract=profile, test_used=False,
        caption_classes=caption_classes, jev=dict(classes=semantic_classes)), max_length=1000, jev=installed)
    fingerprint = object_sha(config)
    json_file(tmp_path / 'run_config.json', dict(config, fingerprint=fingerprint))
    sha = schema._file_hash(checkpoint / 'trainable.pt')
    jev.export_jev(adapter, checkpoint / 'jev', identity=dict(run_fingerprint=fingerprint,
        trainable_sha256=sha, installation=installed), state_dict=portable)
    refs = [dict(path=str(p), sha256=schema._file_hash(p)) for p in checkpoint.rglob('*') if p.is_file()]
    json_file(checkpoint / 'trainer_state.json', dict(schema='music-planner-train-v1', complete=True,
        fingerprint=fingerprint, trainable_sha256=sha, files=refs,
        cursor=dict(update=2, jev_head_exercised=True)))
    return checkpoint, portable


def test_v2_restore_binds_same_head_classes_header_tokens_and_update_cursor(tmp_path):
    checkpoint, portable = exported_conditioned_checkpoint(tmp_path)
    native = protocol_native()
    native.model.load_state_dict({name: value for name, value in portable.items() if not name.startswith('jev_head.')})
    restored = schema.restore_adapter_protocol(native, checkpoint)
    assert restored.genre_classes == ['alpha', 'beta']
    assert restored.header_token_ids['alpha'] == restored.tokenizer.encode('Style: alpha\n')
    assert restored.training_protocol_identity['cursor'] == dict(update=2, jev_head_exercised=True)
    assert restored.training_protocol_identity['marker_sha256'] == schema._file_hash(checkpoint / 'trainer_state.json')
    for name, value in restored.model.jev_head.state_dict().items():
        torch.testing.assert_close(value, portable['jev_head.' + name], rtol=0, atol=0)


def test_v2_restore_rejects_semantic_token_protocol_mismatch_even_if_file_rehashed(tmp_path):
    checkpoint, _ = exported_conditioned_checkpoint(tmp_path)
    protocol_path = checkpoint / 'adapter' / 'dance_caption_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol['header_token_ids']['alpha'][0] += 1
    json_file(protocol_path, protocol)
    marker_path = checkpoint / 'trainer_state.json'
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    for ref in marker['files']:
        if ref['path'] == str(protocol_path):
            ref['sha256'] = schema._file_hash(protocol_path)
    json_file(marker_path, marker)
    with pytest.raises(ValueError, match='token/class protocol'):
        schema.restore_adapter_protocol(protocol_native(), checkpoint)


def test_v2_restore_supports_copied_renamed_checkpoint_and_verifies_its_artifacts(tmp_path):
    original, _ = exported_conditioned_checkpoint(tmp_path)
    relocated_parent = tmp_path / 'relocated'
    relocated_parent.mkdir()
    shutil.copyfile(tmp_path / 'run_config.json', relocated_parent / 'run_config.json')
    relocated = relocated_parent / 'renamed-checkpoint'
    shutil.copytree(original, relocated)
    restored = schema.restore_adapter_protocol(protocol_native(), relocated)
    assert restored.training_protocol_identity['checkpoint_path'] == str(relocated.resolve())
    protocol_path = relocated / 'adapter' / 'dance_caption_protocol.json'
    protocol_path.write_bytes(b'{}')
    with pytest.raises(ValueError, match='identity'):
        schema.restore_adapter_protocol(protocol_native(), relocated)


@pytest.mark.parametrize('changed', ['contract', 'classes'])
def test_training_rejects_contract_or_class_file_changed_during_model_load(tmp_path, changed):
    values, _, _, _ = conditioned_args(tmp_path)
    _, _, data = schema.read_training_data(values)
    if changed == 'contract':
        json_file(values.caption_contract, dict(conditioned_contract(), task_instruction='A changed instruction.'))
    else:
        json_file(values.genre_classes, ['beta', 'alpha'])
    adapter = schema.configure_adapter(protocol_native(), values)
    with pytest.raises(ValueError, match='changed between dataset validation'):
        schema.assert_adapter_data_identity(adapter, data)


PROJECT = Path(__file__).resolve().parents[1]


def official_profile():
    return schema.load_contract(PROJECT / 'examples/planner_training/caption_contract_moss_v2.example.json')


def official_classes():
    return json.loads((PROJECT / 'examples/planner_training/dance_styles_23.json').read_text(encoding='utf-8'))


def test_official_23_values_match_existing_vlm_22_plus_distinct_unknown():
    module = ast.parse((PROJECT / 'scripts_omnidance/exp/ms/vlm_label.py').read_text(encoding='utf-8'))
    genres = next(ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'GENRES' for target in node.targets))
    assert len(genres) == 22
    assert official_classes() == genres + ['unknown']
    assert official_classes()[-2:] == ['freestyle / other', 'unknown']
    profile = official_profile()
    assert profile['dance_caption_format']['genre_header_template'] == 'Dance style: {genre}\n'
    assert profile['dance_caption_format']['timestamp_decimal_places'] == 3


def formal_data(tmp_path):
    profile = official_profile()
    classes = official_classes()
    rows = [source(tmp_path, 'known', seconds=5.0625), source(tmp_path, 'unknown', seconds=5.0625)]
    val = [source(tmp_path, 'val', 'val', seconds=5.0625)]
    for row in rows + val:
        genre = 'unknown' if row['id'] == 'unknown' else 'contemporary'
        row.update(caption_version=profile['caption_version'],
            target_text='Dance style: ' + genre + '\n[0.000-1.960s] Step to the side.\n[1.960-5.062s] Raise the left arm.',
            target_windows=[dict(start=0., end=1.96, caption='Step to the side.'),
                            dict(start=1.96, end=5.0625, caption='Raise the left arm.')],
            style_metadata_not_in_target=dict(label_status='unreviewed', source_model='235B', confidence=.3))
    values = args(tmp_path, rows, val, profile)
    values.genre_classes = json_file(tmp_path / 'classes.json', classes)
    for split, sources in (('train', rows), ('val', val)):
        labels = []
        for row in sources:
            label = {key: row[key] for key in ('id', 'source_group_id', 'audio_sha256', 'split')}
            label.update(genre='unknown' if row['id'] == 'unknown' else 'contemporary',
                label_source='historical-VLM-235B-unreviewed', label_version='weak-v1')
            labels.append(label)
        setattr(values, split + '_genre_labels', manifest(tmp_path / (split + '-genres.jsonl'), labels))
    return values


def test_official_unknown_is_abstention_not_23rd_music_class(tmp_path):
    values = formal_data(tmp_path)
    train, val, data = schema.read_training_data(values)
    assert data['caption_classes'] == official_classes()
    assert data['jev']['classes'] == official_classes()[:-1]
    assert data['caption_style_counts']['train']['unknown'] == 1
    assert data['caption_style_counts']['train']['contemporary'] == 1
    assert data['jev']['train']['known_labels'] == 1
    assert data['jev']['train']['masked_visual_unknown_labels'] == 1
    assert data['jev']['train']['missing_independent_labels'] == 0
    assert len(train[0]['jev_target']) == 22
    assert train[1]['jev_target'] is None
    assert train[1]['genre_supervision_status'] == 'visual-label-abstention'
    assert data['human_review_completed'] is False


def test_unknown_header_mask_never_moves_jev_prefix_into_teacher_answer(tmp_path):
    values = formal_data(tmp_path)
    train, _, data = schema.read_training_data(values)
    adapter = schema.configure_adapter(protocol_native(), values)
    schema.assert_adapter_data_identity(adapter, data)
    unknown = adapter.collate([train[1]])
    prefix_length = 4
    header_end = prefix_length + len(adapter.header_token_ids['unknown'])
    assert unknown['labels'][0, :header_end].eq(-100).all()
    assert unknown['labels'][0, header_end:].ne(-100).all()
    assert unknown['jev_prefix_end'].tolist() == [prefix_length]
    assert int(unknown['labels'].ne(-100).long().argmax()) == header_end
    assert adapter.dance_caption_protocol['genre_supervision'] == official_profile()['genre_supervision']
    known = adapter.collate([train[0]])
    assert known['labels'][0, prefix_length].item() != -100
    assert known['jev_prefix_end'].tolist() == [prefix_length]
    assert 'confidence' not in train[1]['target_text'] and '235B' not in train[1]['target_text']


def test_only_visual_unknown_labels_cannot_claim_semantic_head_training(tmp_path):
    values = formal_data(tmp_path)
    original_rows = [json.loads(line) for line in values.train.read_text().splitlines()]
    manifest(values.train, [original_rows[1]])
    original_labels = [json.loads(line) for line in values.train_genre_labels.read_text().splitlines()]
    manifest(values.train_genre_labels, [original_labels[1]])
    with pytest.raises(ValueError, match='real dance label'):
        schema.read_training_data(values)


def test_official_restore_retains_22_class_head_and_23_header_protocol(tmp_path):
    checkpoint, portable = exported_conditioned_checkpoint(tmp_path,
        profile=official_profile(), caption_classes=official_classes(), semantic_classes=official_classes()[:-1])
    restored = schema.restore_adapter_protocol(protocol_native(), checkpoint)
    assert restored.genre_classes == official_classes()
    assert restored.model._planner_jev_capture.metadata['classes'] == official_classes()[:-1]
    assert restored.header_token_ids['unknown'] == restored.tokenizer.encode('Dance style: unknown\n')
    assert restored.genre_supervision == official_profile()['genre_supervision']
    for name, value in restored.model.jev_head.state_dict().items():
        torch.testing.assert_close(value, portable['jev_head.' + name], rtol=0, atol=0)
