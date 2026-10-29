"""Encoder experiment contracts and free-caption inference, CPU fixtures only."""
from types import SimpleNamespace

import pytest
import torch

from scripts_omnidance.exp.ms import music_planner_adapters as adapters
from scripts_omnidance.exp.ms import music_planner_schema as schema
from scripts_omnidance.exp.ms import music_planner_train as trainer
from scripts_omnidance.exp.ms import music_planner_encoder_generate as generation
from test_music_planner_schema import formal_data, protocol_native
from test_music_planner_moss_generate import formal_profile, ENGLISH_ACTION_DRAFT
from test_music_planner_musicfm import setup, audio_row


def native_scope_model():
    model = torch.nn.Module()
    model.audio_encoder = torch.nn.Module()
    model.audio_encoder.layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(3, 3)) for _ in range(12)])
    model.language_model = torch.nn.Module()
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(3, 3)
        layer.mlp = torch.nn.Linear(3, 3)
        layers.append(layer)
    model.language_model.layers = torch.nn.ModuleList(layers)
    model.audio_adapter = torch.nn.Linear(3, 3)
    model.lm_head = torch.nn.Linear(3, 3)
    return model


def test_audio_last_eight_expands_only_native_audio_lora_targets():
    model = native_scope_model()
    four, experts4, indices4 = adapters.select_native_lora_targets(model, 'audio_encoder', 'language_model', 4)
    eight, experts8, indices8 = adapters.select_native_lora_targets(model, 'audio_encoder', 'language_model', 8)
    assert indices4 == [8, 9, 10, 11] and indices8 == list(range(4, 12))
    assert {x for x in four if x.startswith('language_model.')} == {x for x in eight if x.startswith('language_model.')}
    assert set(eight) - set(four) == {'audio_encoder.layers.' + str(i) + '.0' for i in range(4, 8)}
    assert experts4 == experts8 == []
    assert 'lm_head' not in eight and 'audio_adapter' not in eight


@pytest.mark.parametrize('count', [0, -1, True, 13, 4.0])
def test_audio_scope_rejects_unavailable_or_ambiguous_layer_counts(count):
    with pytest.raises(ValueError):
        adapters.select_native_lora_targets(native_scope_model(), 'audio_encoder', 'language_model', count)


def test_formal_caption_only_does_not_need_independent_genre_labels(tmp_path):
    values = formal_data(tmp_path)
    values.enable_jev = False
    values.train_genre_labels = values.val_genre_labels = None
    train, _, data = schema.read_training_data(values)
    assert data['jev']['train']['known_labels'] == 0
    adapter = schema.configure_adapter(protocol_native(), values)
    schema.assert_adapter_data_identity(adapter, data)
    assert adapter.enable_jev is False and len(adapter.genre_classes) == 23
    batch = adapter.collate([train[0]])
    assert 'jev_prefix_end' not in batch and batch['labels'].ne(-100).any()
    assert not hasattr(adapter.model, 'jev_head')


def test_explicit_jev_retains_independent_label_requirement(tmp_path):
    values = formal_data(tmp_path)
    values.enable_jev = True
    values.train_genre_labels = values.val_genre_labels = None
    with pytest.raises(ValueError, match='independently labelled'):
        schema.read_training_data(values)


def test_jev_opt_out_with_positive_weight_fails_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match='enabled JEV'):
        trainer.train(SimpleNamespace(enable_jev=False, genre_classes=None, genre_weight=.1))


def test_encoder_cli_parses_independent_vocab_and_fusion_controls():
    args = trainer.parser().parse_args(['--family', 'moss', '--model', '/models/moss',
        '--train', '/data/train.jsonl', '--val', '/data/val.jsonl', '--out', '/runs/new',
        '--genre-classes', '/data/classes.json', '--no-enable-jev', '--audio-last-n', '8',
        '--musicfm-cache', '/features/index.json', '--fusion-hidden-dim', '128'])
    assert args.enable_jev is False and args.audio_last_n == 8 and args.fusion_hidden_dim == 128
    assert str(args.musicfm_cache).replace('\\', '/') == '/features/index.json'


def test_free_generation_has_no_forced_genre_or_head_forward(formal_profile):
    adapter, source, _, _ = formal_profile
    text = 'Dance style: contemporary\n' + ENGLISH_ACTION_DRAFT
    adapter.model.continuation = adapter.tokenizer.encode(text) + [1]
    # The cursor need not claim that a classification head was trained.
    # bind_checkpoint mirrors the marker and protocol identity together.
    from test_music_planner_moss_generate import bind_checkpoint
    from pathlib import Path
    root = Path(adapter.training_protocol_identity['checkpoint_path']).parent.parent / 'free-only'
    bind_checkpoint(adapter, root, exercised=False)
    output = generation.generate_free(adapter, source, max_new_tokens=512)
    assert output['caption_valid'] and output['caption'] == text
    assert output['selection_mode'] == 'model-generated-header' and output['forced_header_tokens'] == 0
    assert output['target_consumed'] is False and output['actual_generation_forward_count'] > 0
    assert adapter.model.seen_generate[-1]['input_ids'].tolist() == [[2, 3, 4]]
    assert all('jev_prefix_end' not in row for row in adapter.model.forward_inputs)


@pytest.mark.parametrize('fault', ['bad-format', 'early-eos', 'truncated'])
def test_free_generation_preserves_failed_raw_output(formal_profile, fault):
    adapter, source, _, _ = formal_profile
    text = 'Dance style: contemporary\n' + ENGLISH_ACTION_DRAFT
    tokens = adapter.tokenizer.encode(text) + [1]
    if fault == 'bad-format':
        tokens = adapter.tokenizer.encode(text.replace('1.960', '1.96')) + [1]
    elif fault == 'early-eos':
        tokens.insert(5, 1)
    else:
        tokens = tokens[:-1]
    adapter.model.continuation = tokens
    output = generation.generate_free(adapter, source, max_new_tokens=512)
    assert output['caption'] is None and not output['caption_valid']
    assert output['raw_new_token_ids'] == tokens and output['validation_errors']


def test_free_generation_does_not_read_gold_fields(formal_profile):
    adapter, source, _, _ = formal_profile
    class AudioOnly(dict):
        def __getitem__(self, key):
            assert key in source, 'Gold field was consumed'
            return super().__getitem__(key)
    adapter.model.continuation = adapter.tokenizer.encode('Dance style: contemporary\n' + ENGLISH_ACTION_DRAFT) + [1]
    assert generation.generate_free(adapter, AudioOnly(source), max_new_tokens=512)['caption_valid']


def test_move_keeps_musicfm_time_grid_fp32_for_bf16_moss():
    model = torch.nn.Linear(1, 1).bfloat16()
    adapter = adapters.PlannerAdapter('moss', model, SimpleNamespace())
    batch = dict(musicfm_frame_times_seconds=torch.tensor([[0., .04, .08]]),
                 musicfm_features=torch.ones(1, 3, 4), audio_data=torch.ones(1, 3))
    moved = adapter._move(batch)
    assert moved['musicfm_frame_times_seconds'].dtype == torch.float32
    assert moved['musicfm_features'].dtype == torch.float32
    assert moved['audio_data'].dtype == torch.bfloat16


def test_fusion_uses_independent_optimizer_group_and_learning_rate():
    from scripts_omnidance.exp.ms.music_planner_encoder_fusion import DenseMusicFusion
    model = torch.nn.Module()
    model.lora_probe = torch.nn.Parameter(torch.ones(2))
    model.projector = torch.nn.Linear(2, 2)
    model.planner_fusion = DenseMusicFusion(6, 4, 8)
    model.frozen_base = torch.nn.Linear(2, 2).bfloat16().requires_grad_(False)
    groups, names = trainer.optimizer_groups(model, 1e-5, 5e-5, fusion_lr=3e-4)
    assert {group['name']: group['lr'] for group in groups} == dict(lora=1e-5, projector=5e-5, fusion=3e-4)
    assert names['fusion'] and all(name.startswith('planner_fusion.') for name in names['fusion'])
    actual = [id(p) for group in groups for p in group['params']]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in model.parameters() if p.requires_grad}


def test_fusion_checkpoint_restores_with_new_audio_cache_of_same_extractor(setup, monkeypatch):
    import json
    from pathlib import Path
    from scripts_omnidance.exp.ms import music_planner_musicfm as musicfm
    from scripts_omnidance.exp.ms import music_planner_encoder_fusion as fusion
    from test_music_planner_encoder_fusion import TinyNative
    from test_music_planner_schema import Processor, official_profile, official_classes
    monkeypatch.setattr(adapters, '_prepare_moss_runtime', lambda model: None)
    musicfm.prepare(**setup.args)
    store = musicfm.FeatureStore(setup.args['output_dir'] / 'index.json')
    def native():
        processor = Processor()
        processor.config = SimpleNamespace(mel_sr=16000, mel_hop_length=160, mel_dim=128)
        return adapters.PlannerAdapter('moss', TinyNative(), processor, max_length=1000)
    adapter = schema.ConfiguredPlannerAdapter(native(), contract=official_profile(),
        enable_jev=False, classes=official_classes())
    adapter.musicfm_store = store
    installed = fusion.install_fusion(adapter, feature_dim=store.feature_dim, hidden_dim=4)
    checkpoint = setup.tmp / 'trained' / 'checkpoint-0000002'
    export = checkpoint / 'adapter'
    export.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
    write(export / 'caption_contract.json', adapter.contract)
    write(export / 'dance_caption_protocol.json', adapter.dance_caption_protocol)
    (export / 'native-fixture.bin').write_bytes(b'Native PEFT loading covered by its separate adapter tests')
    portable = trainer.trainable_state(adapter.model)
    torch.save(portable, checkpoint / 'trainable.pt')
    config = dict(family='moss', max_length=1000, jev=None,
        data=dict(contract=adapter.contract, caption_classes=official_classes(), test_used=False, musicfm=store.identity),
        fusion=dict(installation=installed, cache_path=str(store.index_path), cache_identity=store.identity))
    fingerprint = trainer.object_sha(config)
    write(checkpoint.parent / 'run_config.json', dict(config, fingerprint=fingerprint))
    state_sha = schema._file_hash(checkpoint / 'trainable.pt')
    fusion.export_fusion(adapter, checkpoint / 'fusion', identity=dict(run_fingerprint=fingerprint,
        trainable_sha256=state_sha, installation=installed), state_dict=portable)
    write(checkpoint / 'trainer_state.json', dict(schema='music-planner-train-v1', complete=True,
        fingerprint=fingerprint, trainable_sha256=state_sha, cursor=dict(update=2),
        files=[trainer.file_ref(p) for p in checkpoint.rglob('*') if p.is_file()]))
    restored = schema.restore_adapter_protocol(native(), checkpoint)
    assert restored.enable_jev is False and restored.musicfm_store.identity == store.identity
    for name, value in restored.model.planner_fusion.state_dict().items():
        torch.testing.assert_close(value, portable['planner_fusion.' + name], rtol=0, atol=0)
    # New inference music has a different index, but the same extractor protocol.
    row = audio_row(setup.tmp / 'unseen.wav', seconds=2.)
    manifest = setup.tmp / 'unseen.jsonl'
    write(manifest, row)
    new_args = dict(setup.args, manifests=[manifest], output_dir=setup.tmp / 'new-cache')
    musicfm.prepare(**new_args)
    other = new_args['output_dir'] / 'index.json'
    restored = schema.restore_adapter_protocol(native(), checkpoint, musicfm_cache=other)
    assert restored.musicfm_store.identity != store.identity
    assert restored.musicfm_store.identity['extractor_identity'] == store.identity['extractor_identity']
    assert restored.musicfm_store.load(row).shape == (50, 1024)
    new_args.update(layer_ix=8, output_dir=setup.tmp / 'wrong-layer-cache')
    musicfm.prepare(**new_args)
    with pytest.raises(ValueError, match='extractor differs'):
        schema.restore_adapter_protocol(native(), checkpoint, musicfm_cache=new_args['output_dir'] / 'index.json')
    # Portable fusion bytes are among the actual checkpoint member references.
    with (checkpoint / 'fusion/fusion.pt').open('ab') as stream:
        stream.write(b'mutated')
    with pytest.raises(ValueError, match='identity differs'):
        schema.restore_adapter_protocol(native(), checkpoint)
