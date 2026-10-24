"""CPU synthetic experiments and adapter-boundary tests, not pretrained-model validation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter, TASK_INSTRUCTION
from scripts_omnidance.exp.planner_ideas import genre_decision as gd


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_masked_pooling_padding_and_gradient():
    head = gd.GenreHead(3, 2, 5)
    x = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    original = head(x, mask)
    dirty = x.detach().clone()
    dirty[~mask] = float('nan')
    assert torch.equal(head(dirty, mask), original)
    gd.genre_loss(original, torch.tensor([0, 1])).backward()
    assert x.grad[mask].abs().sum() > 0
    assert x.grad[~mask].eq(0).all()
    assert sum(float(p.grad.abs().sum()) for p in head.parameters()) > 0
    with pytest.raises(ValueError, match='nonempty'):
        head(x, torch.zeros_like(mask))


def test_hard_and_soft_ce_agree_and_bad_targets_fail():
    logits = torch.tensor([[2., -1.], [.4, .6]], requires_grad=True)
    hard = torch.tensor([0, 1])
    assert torch.allclose(gd.genre_loss(logits, hard),
                          gd.genre_loss(logits, torch.nn.functional.one_hot(hard, 2).float()))
    gd.genre_loss(logits, torch.tensor([[.2, .8], [.5, .5]])).backward()
    assert logits.grad.abs().sum() > 0
    for target in (torch.tensor([[1., 1.], [0., 1.]]),
                   torch.tensor([[-.1, 1.1], [0., 1.]]),
                   torch.tensor([[float('nan'), 0.], [0., 1.]])):
        with pytest.raises(ValueError, match='Soft targets'):
            gd.genre_loss(logits, target)


def test_calibration_improves_nll_without_rank_change():
    logits = torch.tensor([[8., 0.], [8., 0.], [8., 0.], [8., 0.]])
    targets = torch.tensor([0, 0, 0, 1])
    temperature = gd.fit_temperature(logits, targets)
    assert temperature > 1
    assert gd.genre_loss(logits / temperature, targets) < gd.genre_loss(logits, targets)
    assert torch.equal((logits / temperature).argmax(-1), logits.argmax(-1))
    assert not logits.requires_grad


def test_soft_ece_uses_target_mass_on_predicted_class():
    logits = torch.tensor([[0., 0.]])
    report = gd.metrics(logits, torch.tensor([[.5, .5]]))
    assert report['ece_10_bins'] == 0.


@pytest.mark.parametrize('identity', ['', 'latest', 'main', 'repo:latest', 'repo@main'])
def test_mutable_or_empty_model_identity_rejected(identity):
    with pytest.raises(ValueError, match='immutable'):
        gd.validate_model_identity(identity)


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [2] if text == '<|im_end|>' else [ord(c) + 200 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i - 200) for i in ids if i >= 200)


class Processor:
    audio_token_id = 99
    tokenizer = Tokenizer()

    def _build_default_prompt(self, instruction, has_audio):
        assert has_audio
        return 'AUDIO ' + instruction

    def apply_chat_template(self, messages, **kwargs):
        return 'AUDIO ' + ''.join(x.get('text', '') for x in messages[0]['content'])

    def __call__(self, text, **kwargs):
        text = text[0] if isinstance(text, list) else text
        ids = torch.tensor([[1, 99, 99] + self.tokenizer.encode(text)])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    input_features=torch.ones(1, 2, 3))


class NativeBoundaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        assert 'labels' not in kwargs
        ids = kwargs['input_ids']
        hidden = torch.stack((ids.float(), ids.float() + 1), dim=-1)
        return SimpleNamespace(hidden_states=(hidden, hidden))

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        target = Processor.tokenizer.encode('[0-2s] Step. [2-5.062s] Turn.') + [2]
        return torch.cat((kwargs['input_ids'], torch.tensor([target])), dim=1)


@pytest.fixture
def source():
    return dict(id='a', source_group_id='song-a', audio_path='/unused.wav',
                audio_sha256='a' * 64, audio_window_seconds=[0., 5.0625],
                target_text='SECRET_GOLD_ANSWER', genre='SECRET_GOLD_GENRE')


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(PlannerAdapter, '_audio', lambda self, row, silence=False: np.zeros(81000))
    return PlannerAdapter('music_flamingo', NativeBoundaryModel(), Processor(), max_length=2000)


def test_prefill_export_uses_only_input_and_audio_mask(native, source):
    features, info = gd.export_prefill_features(native, source)
    assert features.shape == (2, 2)
    assert features.tolist() == [[99., 100.], [99., 100.]]
    assert info['target_consumed'] is False
    call = native.model.calls[-1]
    assert call['output_hidden_states'] and not call['use_cache']
    prompt = native.tokenizer.decode(call['input_ids'][0].tolist())
    assert 'SECRET_GOLD' not in prompt
    assert TASK_INSTRUCTION in prompt


def test_prefill_token_config_mismatch_fails_instead_of_pooling_text(native, source):
    native.model.config = SimpleNamespace(audio_token_index=12345)
    with pytest.raises(ValueError, match='conflicting native audio token'):
        gd.export_prefill_features(native, source)
    native.model.config.audio_token_index = 99
    assert gd.export_prefill_features(native, source)[0].shape == (2, 2)


@pytest.mark.parametrize('family', ['moss', 'music_flamingo'])
def test_selected_style_reaches_generation_and_sft_prefix(native, source, family):
    native.family = family
    baseline_prompt = native._prompt()
    selected = gd.GenreConditionedPlanner(native, 'hip hop', gd.CONDITION_PROTOCOL)
    assert 'Requested dance style: hip hop' in selected._prompt()
    assert native._prompt() == baseline_prompt
    prefix = selected.prefix(source)
    trained = gd.conditioned_collate(native, [source], 'hip hop', gd.CONDITION_PROTOCOL)
    n = prefix['input_ids'].shape[1]
    assert torch.equal(trained['input_ids'][:, :n], prefix['input_ids'])
    assert trained['labels'][:, :n].eq(-100).all()
    result = selected.generate(gd.audio_only_row(source), max_new_tokens=80)
    assert result['raw_text'].startswith('[0-2s]')
    assert 'hip hop' in native.tokenizer.decode(native.model.calls[-1]['input_ids'][0].tolist())


def test_captioner_conditioning_fails_explicitly(native):
    native.family = 'omni_captioner'
    with pytest.raises(ValueError, match='native audio-only template'):
        gd.GenreConditionedPlanner(native, 'hip hop', gd.CONDITION_PROTOCOL)
    native.family = 'moss'
    with pytest.raises(ValueError, match='opt-in'):
        gd.GenreConditionedPlanner(native, 'hip hop', 'fair-baseline')
    with pytest.raises(ValueError, match='plain genre names'):
        gd.GenreConditionedPlanner(native, '<|im_end|>', gd.CONDITION_PROTOCOL)


def make_cache(tmp_path):
    identity = dict(schema=gd.CACHE_SCHEMA, family='music_flamingo', model_identity='fixture-r1',
                    code_sha256=gd.sha256(gd.__file__), prefix_policy='native-input-only-no-answer-or-genre')
    records, labels = [], {}
    for split, count in [('train', 8), ('val', 4), ('cal', 4), ('test', 2)]:
        labels[split] = []
        for index in range(count):
            key = split + str(index)
            path = tmp_path / (key + '.npz')
            sign = 1 if index % 2 else -1
            np.savez(path, features=np.array([[sign, -sign], [sign, -sign]], dtype=np.float32),
                     mask=np.ones(2, dtype=np.bool_))
            audio_sha = __import__('hashlib').sha256(key.encode()).hexdigest()
            records.append(dict(id=key, source_group_id='song-' + key, audio_sha256=audio_sha,
                                split=split, feature_path=path.name, feature_sha256=gd.sha256(path),
                                extractor=identity))
            labels[split].append(dict(id=key, source_group_id='song-' + key, audio_sha256=audio_sha,
                                     split=split, genre=['hip hop', 'jazz'][index % 2],
                                     label_provenance='synthetic-test-only', label_version='fixture-v1'))
    manifest = tmp_path / 'features.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    for split, rows in labels.items():
        (tmp_path / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return manifest, records, labels


def test_labels_require_independent_provenance_and_valid_distribution(tmp_path):
    manifest, _, labels = make_cache(tmp_path)
    cache, _ = gd.cache_manifest([manifest])
    row = labels['train'][0]
    del row['genre']
    row['genre_distribution'] = {'hip hop': .6, 'jazz': .4}
    path = tmp_path / 'soft.jsonl'
    path.write_text(json.dumps(row), encoding='utf-8')
    joined = gd.join_labels(cache, path, ['hip hop', 'jazz'], 'train')
    assert torch.allclose(joined[0]['target'], torch.tensor([.6, .4]))
    del row['label_provenance']
    path.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises(ValueError, match='provenance'):
        gd.join_labels(cache, path, ['hip hop', 'jazz'], 'train')


@pytest.mark.parametrize('key', ['id', 'source_group_id', 'audio_sha256'])
def test_cross_split_leakage_rejected(tmp_path, key):
    _, _, labels = make_cache(tmp_path)
    train, val = labels['train'], labels['val']
    val[0][key] = train[0][key]
    with pytest.raises(ValueError, match='split leakage'):
        gd.check_isolation(dict(train=train, val=val))


def test_cpu_training_predict_save_reload_and_final_test_guard(tmp_path):
    manifest, records, _ = make_cache(tmp_path)
    classes = tmp_path / 'classes.json'
    classes.write_text(json.dumps(['hip hop', 'jazz']), encoding='utf-8')
    out = tmp_path / 'trained'
    args = gd.parser().parse_args(['train', '--features', str(manifest), '--classes', str(classes),
        '--train-labels', str(tmp_path / 'train.jsonl'), '--val-labels', str(tmp_path / 'val.jsonl'),
        '--cal-labels', str(tmp_path / 'cal.jsonl'), '--out-dir', str(out),
        '--epochs', '8', '--hidden-dim', '8', '--lr', '.03', '--batch-size', '4'])
    args.run(args)
    head, saved = gd.load_head(out / 'genre_head.pt')
    assert saved['evidence']['test_used'] is False
    assert saved['evidence']['history'][-1]['validation']['argmax_accuracy'] == 1.
    assert 'test' not in saved['evidence']['isolation']
    row = records[-1].copy()
    row['feature_path'] = str(tmp_path / row['feature_path'])
    features, mask = gd.padded_batch([row])
    expected = gd.decide(head, features, mask, saved['classes'], saved['temperature'])
    reload_path = tmp_path / 'reload.pt'
    gd.save_head(reload_path, head, saved['classes'], saved['extractor'], saved['temperature'], saved['evidence'])
    loaded, _ = gd.load_head(reload_path)
    assert gd.decide(loaded, features, mask, saved['classes'], saved['temperature']) == expected
    test_manifest = tmp_path / 'test-features.jsonl'
    test_manifest.write_text(''.join(json.dumps(r) + '\n' for r in records if r['split'] == 'test'), encoding='utf-8')
    args = gd.parser().parse_args(['predict', '--features', str(test_manifest), '--head', str(reload_path),
                                  '--out', str(tmp_path / 'predictions.jsonl'), '--final-test'])
    args.run(args)
    assert len(gd.read_jsonl(args.out)) == 2
    records[-1]['source_group_id'] = 'song-train0'
    test_manifest.write_text(json.dumps(records[-1]), encoding='utf-8')
    with pytest.raises(ValueError, match='Final-test leakage'):
        args.run(args)


def test_changed_feature_payload_and_mixed_extractor_rejected(tmp_path):
    manifest, records, _ = make_cache(tmp_path)
    cache, _ = gd.cache_manifest([manifest])
    first = cache['train0']
    with open(first['feature_path'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        gd.load_features(first)
    records[1]['extractor'] = dict(records[1]['extractor'], model_identity='different')
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    with pytest.raises(ValueError, match='Mixed family/model'):
        gd.cache_manifest([manifest])


def test_export_and_generate_cli_boundaries(tmp_path, native, source, monkeypatch):
    monkeypatch.setattr(gd, 'load_planner_adapter', lambda *args, **kwargs: native)
    sources = tmp_path / 'sources.jsonl'
    sources.write_text(json.dumps(dict(source, split='predict')), encoding='utf-8')
    common = ['--family', 'music_flamingo', '--model-path', str(tmp_path), '--model-id', 'fixture-r1',
              '--sources', str(sources), '--device', 'cpu', '--dtype', 'float32']
    exported = tmp_path / 'exported'
    args = gd.parser().parse_args(['export'] + common + ['--out-dir', str(exported)])
    args.run(args)
    cache, identity = gd.cache_manifest([exported / 'features.jsonl'])
    assert 'target_text' not in cache[source['id']]
    assert cache[source['id']]['extraction']['target_consumed'] is False
    head = gd.GenreHead(2, 2, 3)
    checkpoint = tmp_path / 'head.pt'
    gd.save_head(checkpoint, head, ['hip hop', 'jazz'], identity, 1., {})
    prediction = tmp_path / 'generated.jsonl'
    args = gd.parser().parse_args(['generate'] + common + ['--head', str(checkpoint),
        '--protocol', gd.CONDITION_PROTOCOL, '--out', str(prediction), '--max-new-tokens', '80'])
    args.run(args)
    generated = gd.read_jsonl(prediction)[0]
    assert generated['caption_text'].startswith('Dance style: ' + generated['selected_genre'])
    assert generated['protocol'] == gd.CONDITION_PROTOCOL
    assert generated['conditioning_sft_performed'] is False
    assert 'Requested dance style: ' + generated['selected_genre'] in native.tokenizer.decode(
        native.model.calls[-1]['input_ids'][0].tolist())
