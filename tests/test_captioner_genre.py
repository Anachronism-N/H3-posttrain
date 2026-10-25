"""CPU autograd and native-interface fixtures, not pretrained Captioner quality tests."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts_omnidance.exp.planner_ideas import captioner_genre as joint
from scripts_omnidance.exp.planner_ideas import genre_decision as genre


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


ACTION = '[0-2s] Steps left and lifts both arms. [2-5.062s] Lowers arms and turns right.'
CLASSES = ['hip hop', 'gesture dance']


class BoundaryTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        if text == '<|im_end|>':
            return [2]
        parts = text.encode('utf-8').split(b'\n[')
        result = []
        for index, part in enumerate(parts):
            if index:
                result.append(300)
            result.extend(value + 8 for value in part)
        return result

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        data = b''
        for token in ids:
            if token == 300:
                data += b'\n['
            elif 8 <= token < 264:
                data += bytes([token - 8])
        return data.decode('utf-8', errors='replace')


class NativeProcessor:
    tokenizer = BoundaryTokenizer()
    audio_token_id = 3

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        assert [item['type'] for item in messages[0]['content']] == ['audio']
        return '<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n'

    def __call__(self, text, **kwargs):
        return dict(input_ids=torch.tensor([[1, 3, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                    input_features=torch.ones(1, 1), mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]))

    def save_pretrained(self, path):
        (Path(path) / 'mock_processor.json').write_text('{}')


class TinyCaptioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rng = torch.Generator().manual_seed(123)
        self.base_embedding = torch.nn.Parameter(torch.randn(512, 8, generator=rng), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(8, 512, generator=rng) * .01)
        self.audio_projector = torch.nn.Parameter(torch.randn(1, 8, generator=rng) * .01)
        self.config = SimpleNamespace(use_cache=True, audio_token_id=3)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.generated_style = 'gesture dance'
        self.generation_calls = []

    def forward(self, input_ids, input_features=None, **kwargs):
        hidden = self.base_embedding[input_ids] + (input_features @ self.audio_projector).unsqueeze(1)
        return SimpleNamespace(logits=hidden @ self.lora_A, hidden_states=(hidden, hidden))

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        ids = kwargs['input_ids']
        headers = {name: BoundaryTokenizer().encode(joint.header_text(name)) for name in CLASSES}
        forced = any(ids[0, -len(tokens):].tolist() == tokens for tokens in headers.values())
        prefix = [] if forced else headers[self.generated_style]
        target = prefix + BoundaryTokenizer().encode(ACTION) + [2]
        if kwargs.get('eos_token_id') != [2]:
            target += [17, 18]
        target = target[:kwargs['max_new_tokens']]
        return torch.cat((ids, torch.tensor([target])), -1)

    def save_pretrained(self, path, state_dict=None, **kwargs):
        torch.save(state_dict, Path(path) / 'mock_native.pt')


def make_row(tmp_path, name, split, frequency):
    path = tmp_path / (name + '.wav')
    audio = np.sin(np.arange(121500) * (2 * np.pi * frequency / 24000)) * .2
    sf.write(path, audio, 24000, subtype='PCM_16')
    return dict(id=name, source_group_id='song-' + name, split=split, audio_path=str(path),
        audio_sha256=genre.sha256(path), audio_window_seconds=[0., 5.0625], target_text=ACTION,
        caption_version='fixture-actions-v1', semantic_approval=False, human_review_completed=False,
        target_windows=[dict(start=0., end=2., caption='Steps left and lifts both arms.'),
                        dict(start=2., end=5.0625, caption='Lowers arms and turns right.')])


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    rows = [make_row(tmp_path, 'a', 'train', 220), make_row(tmp_path, 'b', 'train', 330)]
    val = [make_row(tmp_path, 'v', 'val', 440)]
    write_rows(tmp_path / 'train.jsonl', rows)
    write_rows(tmp_path / 'val.jsonl', val)
    for split, values in [('train', rows), ('val', val)]:
        labels = [dict(id=row['id'], split=split, source_group_id=row['source_group_id'],
            audio_sha256=row['audio_sha256'], genre=CLASSES[i % 2],
            label_provenance='synthetic-fixture-only', label_version='fixture-v1') for i, row in enumerate(values)]
        write_rows(tmp_path / (split + '-labels.jsonl'), labels)
    (tmp_path / 'classes.json').write_text(json.dumps(CLASSES), encoding='utf-8')
    model_path = tmp_path / 'model'
    model_path.mkdir()
    (model_path / 'config.json').write_text('{}')
    (model_path / 'fixture.bin').write_bytes(b'CPU fixture, not pretrained model weights')
    loaded = []

    def native_loader(family, model_path, checkpoint=None, **kwargs):
        assert family == 'omni_captioner'
        model = TinyCaptioner()
        metadata = [dict(name=name, shape=list(p.shape), dtype=str(p.dtype))
                    for name, p in model.named_parameters() if p.requires_grad]
        if checkpoint:
            saved = torch.load(Path(checkpoint) / 'adapter/mock_native.pt', weights_only=True)
            joint.training.restore_trainable(model, saved)
            model.requires_grad_(False)
        native = joint.PlannerAdapter(family, model, NativeProcessor(), max_length=kwargs.get('max_length', 2048))
        native.trainable_info = dict(rank=kwargs.get('lora_rank', 32), alpha=kwargs.get('lora_alpha', 64),
                                    trainable=metadata)
        loaded.append(native)
        return native

    monkeypatch.setattr(joint, 'load_planner_adapter', native_loader)
    args = ['--model', str(model_path), '--train', str(tmp_path / 'train.jsonl'), '--val', str(tmp_path / 'val.jsonl'),
        '--train-labels', str(tmp_path / 'train-labels.jsonl'), '--val-labels', str(tmp_path / 'val-labels.jsonl'),
        '--classes', str(tmp_path / 'classes.json'), '--out', str(tmp_path / 'run'), '--device', 'cpu',
        '--dtype', 'float32', '--no-gradient-checkpointing', '--epochs', '2', '--max-length', '2048']
    return SimpleNamespace(root=tmp_path, rows=rows, val=val, model=model_path, args=args, loaded=loaded,
                           native_loader=native_loader)


def test_segmented_header_protocol_native_prefix_and_gradient(setup_run):
    setup = setup_run
    native = setup.native_loader('omni_captioner', setup.model)
    adapter = joint.JointCaptionerAdapter(native, CLASSES)
    row = dict(setup.rows[0], selected_style_label='gesture dance')
    prefix = native.prefix(row)
    batch = adapter.collate([row])
    header = adapter.header_ids['gesture dance']
    segmented = header + adapter.tokenizer.encode(ACTION) + [2]
    assert segmented[:-1] != adapter.tokenizer.encode(joint.header_text('gesture dance') + ACTION)
    assert batch['input_ids'][0, 4:].tolist() == segmented
    assert torch.equal(prefix['input_ids'], batch['input_ids'][:, :4])
    assert batch['labels'][:, :4].eq(-100).all()
    assert batch['mm_token_type_ids'][:, 4:].eq(0).all()
    logits = joint._forward(adapter, batch, torch.float32)
    loss, sums, counts = joint.style_action_loss(logits, batch['labels'], batch['style_action_segments'])
    assert counts.tolist() == [len(header), len(adapter.tokenizer.encode(ACTION)) + 1]
    loss.backward()
    assert native.model.lora_A.grad.abs().sum() > 0
    assert native.model.audio_projector.grad.abs().sum() > 0
    assert native.model.base_embedding.grad is None
    assert all([item['type'] for item in message[0]['content']] == ['audio']
               for message in native.processor.messages)
    assert all('pixel_values' not in key for key in batch)
    with pytest.raises(ValueError, match='validated checkpoint'):
        adapter.generate(row, selected_style='hip hop')


def test_two_updates_resume_third_and_generate_conditioned_and_joint_self_style(setup_run):
    setup = setup_run
    first = joint.main(['train'] + setup.args + ['--max-updates', '2'])
    assert first['cursor']['update'] == 2
    assert all(n > 0 for n in first['cursor']['conditioning_supervised_tokens'])
    second = joint.main(['train'] + setup.args + ['--max-updates', '3', '--resume', first['checkpoint']])
    assert second['cursor']['update'] == 3 and second['cursor']['source_exposures'] == 3
    assert second['invocation_optimizer_updates'] == 1
    assert all(a > b for a, b in zip(second['cursor']['conditioning_supervised_tokens'],
                                    first['cursor']['conditioning_supervised_tokens']))
    adapter, config, descriptor, receipt = joint.load_joint(setup.model, second['checkpoint'], generation=True)

    class AudioOnly(dict):
        def __getitem__(self, key):
            assert key not in ('target_text', 'genre', 'selected_style_label', 'target_windows')
            return super().__getitem__(key)

    row = AudioOnly(setup.rows[0])
    conditional = adapter.generate(row, selected_style='hip hop', max_new_tokens=200)
    assert conditional['caption_valid'] and conditional['generated_style'] == 'hip hop'
    assert conditional['style_header_generated_by_joint_model'] is False
    assert conditional['forced_header_token_ids'] == adapter.header_ids['hip hop']
    assert adapter.model.generation_calls[-1]['input_ids'][0, 4:].tolist() == adapter.header_ids['hip hop']
    assert conditional['raw_tail_text'].startswith('[0-2s]')
    assert conditional['input_tokens'] == 4 + len(adapter.header_ids['hip hop'])
    free = adapter.generate(row, selected_style=None, max_new_tokens=200)
    assert free['caption_valid'] and free['generated_style'] == 'gesture dance'
    assert free['route'] == 'joint-self-style' and free['style_header_generated_by_joint_model']
    assert free['forced_header_token_ids'] == []
    assert adapter.model.generation_calls[-1]['input_ids'][0].tolist() == [1, 3, 3, 4]
    assert receipt['applied_eos_patch_ref'] == free['applied_generation_eos_patch_ref']
    assert descriptor['trainable_sha256'] == genre.sha256(Path(second['checkpoint']) / 'trainable.pt')


def test_missing_labels_wrong_dictionary_and_plain_baseline_checkpoint_rejected(setup_run):
    setup = setup_run
    labels = genre.read_jsonl(setup.root / 'train-labels.jsonl')
    del labels[0]['label_provenance']
    write_rows(setup.root / 'train-labels.jsonl', labels)
    with pytest.raises(ValueError, match='provenance'):
        joint.main(['train'] + setup.args + ['--max-updates', '1'])
    assert not (setup.root / 'run').exists()


def test_checkpoint_rejects_dictionary_model_source_and_adapter_drift(setup_run):
    setup = setup_run
    result = joint.main(['train'] + setup.args + ['--max-updates', '1'])
    with pytest.raises(ValueError, match='dictionary'):
        joint.validate_joint_checkpoint(result['checkpoint'], setup.model, ['different', 'dictionary'])
    weight = setup.model / 'fixture.bin'
    original = weight.read_bytes()
    weight.write_bytes(b'x' * len(original))
    with pytest.raises(ValueError, match='weight content'):
        joint.load_joint(setup.model, result['checkpoint'])
    weight.write_bytes(original)
    export = Path(result['checkpoint']) / 'adapter/mock_native.pt'
    export.write_bytes(export.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='contents changed'):
        joint.load_joint(setup.model, result['checkpoint'])


def test_head_export_generate_binding_and_low_confidence_joint_route(setup_run):
    setup = setup_run
    trained = joint.main(['train'] + setup.args + ['--max-updates', '2'])
    sources = setup.root / 'sources.jsonl'
    write_rows(sources, [dict(setup.val[0], split='predict')])
    features = setup.root / 'features'
    shared = ['--model', str(setup.model), '--checkpoint', trained['checkpoint'], '--input', str(sources),
              '--device', 'cpu', '--dtype', 'float32']
    joint.main(['export-features'] + shared + ['--out-dir', str(features)])
    cache, identity = genre.cache_manifest([features / 'features.jsonl'])
    assert identity['joint_checkpoint']['trainable_sha256']
    assert cache['v']['extraction']['target_consumed'] is False
    values, _ = genre.load_features(cache['v'])
    head = genre.GenreHead(values.shape[-1], 2, 4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    head_path = setup.root / 'head.pt'
    genre.save_head(head_path, head, CLASSES, identity, 1., {})
    out = setup.root / 'joint-output.jsonl'
    record = joint.main(['generate'] + shared + ['--head', str(head_path), '--out', str(out)])[0]
    assert record['routing']['route'] == 'joint-self-style'
    assert record['routing']['native_unconditioned_fallback'] is False
    assert record['generation']['generated_style'] == 'gesture dance'
    assert record['style_agreement']['forced_by_head'] is False
    assert record['caption_valid']
    wrong = dict(identity, model_identity='old-base-fair-head')
    genre.save_head(head_path, head, CLASSES, wrong, 1., {})
    with pytest.raises(ValueError, match='exact trained joint checkpoint'):
        joint.main(['generate'] + shared + ['--head', str(head_path), '--out', str(setup.root / 'wrong.jsonl')])


def test_untrained_and_other_protocol_checkpoint_cannot_generate(setup_run):
    setup = setup_run
    result = joint.main(['train'] + setup.args + ['--max-updates', '1'])
    path = Path(result['checkpoint'])
    marker_path = path / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    marker['cursor']['update'] = 0
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='No actual style'):
        joint.load_joint(setup.model, path)
    config_path = path.parent / 'run_config.json'
    config = json.loads(config_path.read_text())
    config['protocol'] = 'fair-direct-action-only'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='joint Captioner protocol'):
        joint.load_joint(setup.model, path)


def make_fair_checkpoint(setup):
    native = setup.native_loader('omni_captioner', setup.model)
    groups, _ = joint.training.optimizer_groups(native.model, 1e-4, 1e-4, torch.float32)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    batch = native.collate([setup.rows[0]])
    logits = native.model(**native._move(batch)).logits
    loss, count = joint.training.shifted_nll(logits, batch['labels'])
    (loss / count).backward()
    optimizer.step()
    scheduler.step()
    folder = setup.root / 'fair'
    folder.mkdir()
    config = dict(family='omni_captioner', model=joint.training.local_model_identity(setup.model),
                  data=dict(test_used=False), adapter_source_sha256='a' * 64, trainer_sha256='b' * 64)
    fingerprint = joint.object_sha(config)
    joint.genre.write_json(folder / 'run_config.json', dict(config, fingerprint=fingerprint))
    path = folder / 'checkpoint-0000001'
    joint.training.save_checkpoint(path, native.model, optimizer, scheduler,
        dict(epoch=0, position=1, update=1, source_exposures=1, best_val_nll=None), fingerprint, adapter=native)
    marker_path = path / 'trainer_state.json'
    marker = json.loads(marker_path.read_text())
    # Simulate 8-rank ZeRO provenance with only its portable weights copied locally.
    marker.update(world_size=8, zero3=True)
    marker['files'].append(dict(path=str(path / 'distributed/unused-rank7.bin'), sha256='d' * 64))
    marker_path.write_text(json.dumps(marker))
    return path


def test_warm_start_reenables_only_declared_lora_and_projector(setup_run):
    setup = setup_run
    initialized = make_fair_checkpoint(setup)
    with pytest.raises(ValueError, match='single-device|joint Captioner protocol'):
        joint.load_joint(setup.model, initialized)
    later_args = list(setup.args)
    later_args[later_args.index('--out') + 1] = str(setup.root / 'warm')
    warmed = joint.main(['train'] + later_args + ['--init-checkpoint', str(initialized), '--max-updates', '1'])
    model = setup.loaded[-1].model
    assert model.lora_A.requires_grad and model.audio_projector.requires_grad
    assert model.base_embedding.requires_grad is False
    config = json.loads((setup.root / 'warm/run_config.json').read_text())
    assert config['initialization']['trainable_sha256'] == genre.sha256(initialized / 'trainable.pt')
    with pytest.raises(ValueError, match='choose one'):
        joint.main(['train'] + later_args + ['--init-checkpoint', str(initialized), '--resume', warmed['checkpoint']])
    with pytest.raises(ValueError, match='rank/alpha'):
        joint.validate_initialization(initialized, setup.model, 16, 64)


@pytest.mark.parametrize('split', ['cal', 'test'])
def test_cal_and_test_feature_export_cannot_reuse_joint_training_or_validation_sources(setup_run, split):
    setup = setup_run
    trained = joint.main(['train'] + setup.args + ['--max-updates', '1'])
    sources = setup.root / ('leaking-' + split + '.jsonl')
    write_rows(sources, [dict(setup.val[0], split=split)])
    out = setup.root / ('forbidden-' + split)
    with pytest.raises(ValueError, match=split + ' source overlaps joint SFT'):
        joint.main(['export-features', '--model', str(setup.model), '--checkpoint', trained['checkpoint'],
            '--input', str(sources), '--out-dir', str(out), '--device', 'cpu', '--dtype', 'float32'])
    assert not out.exists()


def test_initializer_rejects_short_plan_or_test_selected_sources(setup_run):
    setup = setup_run
    checkpoint = make_fair_checkpoint(setup)
    config_path = checkpoint.parent / 'run_config.json'
    marker_path = checkpoint / 'trainer_state.json'
    config = json.loads(config_path.read_text())
    marker = json.loads(marker_path.read_text())
    config['data']['test_used'] = True
    config.pop('fingerprint')
    fingerprint = joint.object_sha(config)
    config['fingerprint'] = marker['fingerprint'] = fingerprint
    config_path.write_text(json.dumps(config))
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='non-test'):
        joint.validate_initialization(checkpoint, setup.model, 32, 64)
    config['data']['test_used'] = False
    config['mode'] = 'short-plan'
    config['schema'] = 'music-planner-reviewed-ablation-v2/short-plan'
    config.pop('fingerprint')
    config['fingerprint'] = marker['fingerprint'] = joint.object_sha(config)
    config_path.write_text(json.dumps(config))
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='fair/direct-action'):
        joint.validate_initialization(checkpoint, setup.model, 32, 64)


@pytest.mark.parametrize('reviewed', [False, True])
def test_v3_direct_initializer_protocol_is_explicitly_supported(setup_run, reviewed):
    setup = setup_run
    path = make_fair_checkpoint(setup)
    config_path, marker_path = path.parent / 'run_config.json', path / 'trainer_state.json'
    config, marker = json.loads(config_path.read_text()), json.loads(marker_path.read_text())
    schema = 'music-planner-captioner-ablation-v3/direct-caption/' + ('reviewed-shared' if reviewed else 'weak-caption')
    config.update(schema=schema, mode='direct-caption', require_shared_annotations=reviewed)
    config.pop('fingerprint')
    config['fingerprint'] = marker['fingerprint'] = joint.object_sha(config)
    metadata = path / 'adapter/short_plan_adapter.json'
    genre.write_json(metadata, dict(schema=schema, mode='direct-caption', family='omni_captioner',
                                   require_shared_annotations=reviewed))
    marker['files'].append(dict(path=str(metadata), sha256=genre.sha256(metadata)))
    config_path.write_text(json.dumps(config))
    marker_path.write_text(json.dumps(marker))
    identity, _ = joint.validate_initialization(path, setup.model, 32, 64)
    assert identity['fingerprint'] == config['fingerprint']


def test_head_validation_cannot_relabel_a_joint_training_source(setup_run):
    setup = setup_run
    config = dict(data=dict(isolation=genre.check_isolation(dict(train=setup.rows, val=setup.val))))
    with pytest.raises(ValueError, match='val source overlaps joint SFT train'):
        joint.check_joint_holdout([dict(setup.rows[0], split='val')], config)
    # Reusing the joint validation partition is allowed; calibration needs a separate partition.
    joint.check_joint_holdout(setup.val, config)
