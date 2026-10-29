"""CPU integration with real WAVs and a tiny encoder stub, not pretrained MusicFM."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts_omnidance.exp.ms import music_planner_musicfm as musicfm


class StubMusicFM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls = []
        self.eval()

    def forward(self, waveform):
        raise AssertionError('Pretraining forward/masking must never run')

    def get_latent(self, waveform, layer_ix):
        self.calls.append((waveform.detach().cpu().clone(), layer_ix))
        frames = musicfm._feature_length(waveform.shape[1])
        time = torch.arange(frames, device=waveform.device, dtype=torch.float32)
        return (time[None, :, None] + self.weight).expand(1, frames, 1024).contiguous()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'official-checkout'
    (root / 'model').mkdir(parents=True)
    (root / 'modules').mkdir()
    (root / 'model' / 'musicfm_25hz.py').write_text('# local official source identity\n')
    (root / 'modules' / 'features.py').write_text('# local feature source identity\n')
    checkpoint = tmp_path / 'weights.pt'
    torch.save({'state_dict': {'model.weight': torch.ones(())}}, checkpoint)
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'melspec_2048_mean': -10., 'melspec_2048_std': 5.}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hidden_size': 1024, 'num_hidden_layers': 12}))
    row = audio_row(tmp_path / 'audio.wav', rate=16000, seconds=5.0625)
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps(dict(row, target_text='DO NOT USE THIS TARGET',
                                      genre='DO NOT USE THIS LABEL')) + '\n')
    model = StubMusicFM()
    monkeypatch.setattr(musicfm, 'load_local_musicfm', lambda *args, **kwargs: model)
    args = dict(manifests=[manifest], output_dir=tmp_path / 'cache', musicfm_root=root,
                checkpoint=checkpoint, stats=stats, conformer_config=config, device='cpu')
    return types.SimpleNamespace(root=root, row=row, manifest=manifest, model=model, args=args,
                                 checkpoint=checkpoint, stats=stats, config=config, tmp=tmp_path)


def audio_row(path, rate=24000, seconds=1., channels=1):
    waveform = np.sin(np.arange(round(rate * seconds)) * .03).astype(np.float32)
    if channels != 1:
        waveform = np.repeat(waveform[:, None], channels, axis=1)
    sf.write(str(path), waveform, rate, subtype='FLOAT')
    return dict(audio_path=str(path), audio_sha256=musicfm.file_sha256(path),
                audio_duration_seconds=seconds, audio_window_seconds=[0., seconds])


def read_index(setup):
    path = setup.args['output_dir'] / 'index.json'
    return path, json.loads(path.read_text())


def write_index(path, value):
    path.write_text(json.dumps(value))


def test_full_audio_resample_and_audit(setup):
    audio, metadata = musicfm.load_audio(setup.row)
    assert audio.dtype == torch.float32 and audio.shape == (121500,)
    assert metadata['source_num_samples'] == 81000
    assert metadata['source_sample_rate'] == 16000
    assert metadata['resampled_num_samples'] == 121500
    assert metadata['audio_duration_seconds'] == 5.0625


def test_no_resampling_when_already_24k(tmp_path):
    row = audio_row(tmp_path / 'source.wav', rate=24000)
    waveform, metadata = musicfm.load_audio(row)
    original, _ = sf.read(row['audio_path'], dtype='float32')
    assert torch.equal(waveform, torch.from_numpy(original))
    assert metadata['source_num_samples'] == metadata['resampled_num_samples']


@pytest.mark.parametrize('change', ['sha', 'duration', 'crop', 'stereo', 'nan', 'bool_duration'])
def test_invalid_original_audio_rejected(setup, change):
    row = dict(setup.row)
    if change == 'sha':
        row['audio_sha256'] = '0' * 64
    elif change == 'duration':
        row['audio_duration_seconds'] = 3.
        row['audio_window_seconds'] = [0., 3.]
    elif change == 'crop':
        row['audio_window_seconds'] = [1., 2.]
    elif change == 'stereo':
        row = audio_row(setup.tmp / 'stereo.wav', channels=2)
    elif change == 'bool_duration':
        row['audio_duration_seconds'] = True
        row['audio_window_seconds'] = [0., True]
    else:
        sf.write(row['audio_path'], np.full(81000, np.nan, dtype=np.float32), 16000, subtype='FLOAT')
        row['audio_sha256'] = musicfm.file_sha256(row['audio_path'])
    with pytest.raises(ValueError):
        musicfm.load_audio(row)


def test_prepare_roundtrip_keeps_entire_dense_sequence_and_no_targets(setup):
    identity = musicfm.prepare(**setup.args)
    path, index = read_index(setup)
    store = musicfm.FeatureStore(path)
    features = store.load(setup.row)
    assert features.shape == (127, 1024)
    assert features.dtype == torch.float32
    assert torch.equal(features[:, 0], torch.arange(127, dtype=torch.float32) + 1)
    waveform, layer = setup.model.calls[0]
    assert waveform.shape == (1, 121500) and layer == 7
    assert store.feature_dim == 1024
    assert identity == store.identity
    assert store.validate_rows([setup.row]) == dict(rows=1, feature_files_verified=False)
    assert store.validate_rows([setup.row], verify_bytes=True)['feature_files_verified'] is True
    assert 'DO NOT USE' not in path.read_text()
    assert 'audio_path' not in path.read_text() and 'target_text' not in path.read_text()
    assert index['extractor_identity']['layer_ix'] == 7
    assert 'index_sha256' not in index['extractor_identity']


def test_native_timing_is_not_uniformly_stretched(setup):
    musicfm.prepare(**setup.args)
    store = musicfm.FeatureStore(setup.args['output_dir'] / 'index.json')
    times = store.frame_times(setup.row)
    assert times.dtype == torch.float32 and times.shape == (127,)
    assert times[0] == 0
    assert times[1].item() == pytest.approx(.04)
    assert times[-1].item() == pytest.approx(5.04)
    assert times[-1].item() < setup.row['audio_duration_seconds']
    assert store.identity['extractor_identity']['timing'] == musicfm.TIMING


@pytest.mark.parametrize('samples,frames', [(24000, 25), (24120, 25), (24240, 26),
                                           (121500, 127), (1200, 2)])
def test_centered_stft_drop_last_and_stride2_length(samples, frames):
    assert musicfm._feature_length(samples) == frames


def test_cache_relocates_without_paths_or_identity_changes(setup):
    musicfm.prepare(**setup.args)
    old = musicfm.FeatureStore(setup.args['output_dir'] / 'index.json')
    destination = setup.tmp / 'moved-cache'
    shutil.copytree(setup.args['output_dir'], destination)
    new = musicfm.FeatureStore(destination / 'index.json')
    assert old.identity == new.identity
    assert torch.equal(old.load(setup.row), new.load(setup.row))


def test_new_inference_audio_changes_index_but_not_extractor(setup):
    identity = musicfm.prepare(**setup.args)
    row = audio_row(setup.tmp / 'new.wav', seconds=2.)
    path = setup.tmp / 'inference.jsonl'
    path.write_text(json.dumps(row) + '\n')
    args = dict(setup.args, manifests=[path], output_dir=setup.tmp / 'inference-cache')
    other = musicfm.prepare(**args)
    assert identity['index_sha256'] != other['index_sha256']
    assert identity['extractor_identity'] == other['extractor_identity']


def test_existing_cache_reuses_verified_file_without_running_encoder(setup):
    musicfm.prepare(**setup.args)
    expected_calls = len(setup.model.calls)
    identity = musicfm.prepare(**setup.args)
    assert len(setup.model.calls) == expected_calls
    assert identity['entries'] == 1


def test_duplicate_rows_deduplicate_during_first_run(setup):
    setup.manifest.write_text((json.dumps(setup.row) + '\n') * 3)
    result = musicfm.prepare(**setup.args)
    assert result['entries'] == 1
    assert len(setup.model.calls) == 1


def test_cache_can_extend_with_same_extractor(setup):
    first = musicfm.prepare(**setup.args)
    row = audio_row(setup.tmp / 'new.wav', seconds=2.)
    setup.manifest.write_text(json.dumps(row) + '\n')
    second = musicfm.prepare(**setup.args)
    assert second['entries'] == 2
    assert first['index_sha256'] != second['index_sha256']
    assert first['extractor_identity'] == second['extractor_identity']


def test_interrupted_file_can_be_recovered_without_encoder(setup):
    musicfm.prepare(**setup.args)
    path, _ = read_index(setup)
    path.unlink()
    calls = len(setup.model.calls)
    result = musicfm.prepare(**setup.args)
    assert result['entries'] == 1 and len(setup.model.calls) == calls


@pytest.mark.parametrize('field', ['checkpoint', 'stats', 'conformer_config', 'source', 'layer_ix'])
def test_same_audio_different_encoder_cannot_share_cache(setup, field):
    musicfm.prepare(**setup.args)
    args = dict(setup.args)
    if field == 'source':
        (setup.root / 'modules' / 'features.py').write_text('# source changed\n')
    elif field == 'layer_ix':
        args['layer_ix'] = 8
    else:
        path = args[field]
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='mix different'):
        musicfm.prepare(**args)


def test_interrupted_file_other_extractor_rejected(setup):
    musicfm.prepare(**setup.args)
    path, _ = read_index(setup)
    path.unlink()
    with pytest.raises(ValueError, match='another audio/extractor'):
        musicfm.prepare(**dict(setup.args, layer_ix=8))


def test_fast_validation_reads_no_feature_files(setup, monkeypatch):
    musicfm.prepare(**setup.args)
    store = musicfm.FeatureStore(setup.args['output_dir'] / 'index.json')
    monkeypatch.setattr(musicfm, '_load_feature_file', lambda *args: pytest.fail('Unexpected I/O'))
    assert store.validate_rows([setup.row, setup.row])['rows'] == 2


@pytest.mark.parametrize('mutation', ['sha', 'duration', 'crop'])
def test_row_cache_identity_mismatch_rejected(setup, mutation):
    musicfm.prepare(**setup.args)
    store = musicfm.FeatureStore(setup.args['output_dir'] / 'index.json')
    row = dict(setup.row)
    if mutation == 'sha':
        row['audio_sha256'] = 'a' * 64
    elif mutation == 'duration':
        row['audio_duration_seconds'] = 5.
        row['audio_window_seconds'] = [0., 5.]
    else:
        row['audio_window_seconds'] = [.5, 5.]
    with pytest.raises(ValueError):
        store.validate_rows([row])


def test_bytes_are_checked_when_loaded(setup):
    musicfm.prepare(**setup.args)
    path, index = read_index(setup)
    store = musicfm.FeatureStore(path)
    feature = path.parent / next(iter(index['entries'].values()))['feature_file']
    feature.write_bytes(feature.read_bytes() + b'changed')
    store.validate_rows([setup.row])
    with pytest.raises(ValueError, match='bytes differ'):
        store.load(setup.row)


@pytest.mark.parametrize('mutation', ['extractor_sha', 'mixed', 'path', 'dtype', 'shape',
                                     'time_length', 'resampled', 'source_samples'])
def test_invalid_index_rejected(setup, mutation):
    musicfm.prepare(**setup.args)
    path, index = read_index(setup)
    entry = next(iter(index['entries'].values()))
    if mutation == 'extractor_sha':
        index['extractor_identity_sha256'] = 'a' * 64
    elif mutation == 'mixed':
        entry['extractor_identity_sha256'] = 'b' * 64
    elif mutation == 'path':
        entry['feature_file'] = '../outside.safetensors'
    elif mutation == 'dtype':
        entry['feature_dtype'] = 'float16'
    elif mutation == 'shape':
        entry['feature_shape'][1] = 128
    elif mutation == 'time_length':
        entry['feature_shape'][0] += 1
    elif mutation == 'resampled':
        entry['resampled_num_samples'] += 1
    else:
        entry['source_num_samples'] -= 100
    write_index(path, index)
    with pytest.raises(ValueError):
        musicfm.FeatureStore(path)


def test_cache_payload_metadata_is_verified_even_if_index_sha_updated(setup):
    from safetensors.torch import save_file
    musicfm.prepare(**setup.args)
    path, index = read_index(setup)
    entry = next(iter(index['entries'].values()))
    feature = path.parent / entry['feature_file']
    save_file({'features': torch.zeros(entry['feature_shape'])}, str(feature), metadata={'wrong': 'identity'})
    entry['feature_sha256'] = musicfm.file_sha256(feature)
    write_index(path, index)
    with pytest.raises(ValueError, match='payload metadata'):
        musicfm.FeatureStore(path).load(setup.row)


@pytest.mark.parametrize('mode', ['train', 'gradient', 'half', 'nan', 'wrong_shape', 'pooled'])
def test_extractor_requires_frozen_fp32_temporal_model(mode):
    model = StubMusicFM()
    if mode == 'train':
        model.train()
    elif mode == 'gradient':
        model.weight.requires_grad_(True)
    elif mode == 'half':
        model.half()
    elif mode == 'nan':
        model.weight.fill_(float('nan'))
    elif mode == 'wrong_shape':
        model.get_latent = lambda *args, **kwargs: torch.zeros(1, 30, 1024)
    else:
        model.get_latent = lambda *args, **kwargs: torch.zeros(1, 1024)
    with pytest.raises(ValueError):
        musicfm.extract_features(model, torch.zeros(24000))


def test_offline_network_guard_is_scoped_and_restored():
    original = socket.socket.connect
    with musicfm._offline():
        with pytest.raises(RuntimeError, match='Network access'):
            socket.create_connection(('example.com', 80))
        with socket.socket() as connection:
            with pytest.raises(RuntimeError, match='Network access'):
                connection.connect(('127.0.0.1', 80))
    assert socket.socket.connect is original


def test_real_loader_uses_local_config_strict_tensor_weights_and_restores_patch(tmp_path, monkeypatch):
    root = tmp_path / 'arbitrary-clone-name'
    (root / 'model').mkdir(parents=True)
    (root / 'model' / 'musicfm_25hz.py').write_text('''
import types
import torch
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
class MusicFM25Hz(torch.nn.Module):
    def __init__(self, is_flash, stat_path, model_path):
        super().__init__()
        assert is_flash is False and model_path is None
        self.local_config = Wav2Vec2ConformerConfig.from_pretrained('remote-id-must-not-be-used')
        self.weight = torch.nn.Parameter(torch.zeros(()))
        spec = types.SimpleNamespace(n_fft=2048, hop_length=240, center=True, pad=0)
        self.preprocessor_melspec_2048 = types.SimpleNamespace(mel_stft=types.SimpleNamespace(spectrogram=spec))
        def block():
            def conv(stride):
                return types.SimpleNamespace(kernel_size=(3,3),padding=(1,1),stride=stride)
            return types.SimpleNamespace(conv1=conv((2,2)),conv2=conv((1,1)),conv3=conv((2,2)))
        self.conv = types.SimpleNamespace(conv=[block(),block()])
''')
    module_name = 'transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer'
    fake = types.ModuleType(module_name)
    class Config:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError('Remote loader called')
        @classmethod
        def from_dict(cls, value):
            return value
    fake.Wav2Vec2ConformerConfig = Config
    monkeypatch.setitem(sys.modules, module_name, fake)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'state_dict': {'model.weight': torch.tensor(3.)}}, checkpoint)
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'melspec_2048_mean': 0., 'melspec_2048_std': 1.}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'local': 'only'}))
    original_method = Config.from_pretrained.__func__
    model = musicfm.load_local_musicfm(root, checkpoint, stats, config)
    assert model.weight.item() == 3.
    assert model.local_config == {'local': 'only'}
    assert model.training is False and model.weight.requires_grad is False
    assert Config.from_pretrained.__func__ is original_method
    assert not any(key == 'musicfm' or key.startswith('musicfm.') for key in sys.modules)


def test_source_change_during_model_load_rejected(setup, monkeypatch):
    def changed(*args, **kwargs):
        (setup.root / 'model' / 'musicfm_25hz.py').write_text('# changed during load')
        return setup.model
    monkeypatch.setattr(musicfm, 'load_local_musicfm', changed)
    with pytest.raises(ValueError, match='changed during model construction'):
        musicfm.prepare(**setup.args)


def test_relative_manifest_audio_paths_and_target_independence(setup):
    source = dict(setup.row, audio_path='audio.wav', target_text={'arbitrary': 'ignored'},
                  style_metadata_not_in_target={'unused': True})
    setup.manifest.write_text(json.dumps(source) + '\n')
    rows = musicfm.read_audio_manifest(setup.manifest)
    assert rows == [setup.row]


def test_cli_validate(setup, capsys):
    musicfm.prepare(**setup.args)
    musicfm.main(['validate', '--index', str(setup.args['output_dir'] / 'index.json'),
                  '--manifest', str(setup.manifest), '--verify-bytes'])
    result = json.loads(capsys.readouterr().out)
    assert result['rows'] == 1 and result['feature_files_verified'] is True
