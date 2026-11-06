"""Offline, frozen MusicFM frame features for the planner encoder experiment.

The official MusicFM25Hz ``get_latent`` path is used without waveform masking,
pooling, cropping or target labels. Model/source files must already be local.
Cache identities bind their contents, not absolute paths, so caches can move.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import types
from unittest.mock import patch
import uuid

import numpy as np
import torch


SCHEMA = 'music-planner-musicfm-cache-v1'
EXTRACTOR_SCHEMA = 'music-planner-musicfm-extractor-v1'
FEATURE_SCHEMA = 'music-planner-musicfm-feature-v1'
FEATURE_DIM = 1024
SAMPLE_RATE = 24000
PREPROCESSING = {
    'schema': 'complete-mono-soundfile-float32-scipy-resample-poly-24khz-v1',
    'source': 'complete-original-mono-audio',
    'resampling': 'scipy.signal.resample_poly-default-window-padtype',
    'target_sample_rate': SAMPLE_RATE,
    'normalization': 'official-MusicFM-melspec-stats-no-waveform-normalization',
    'inference': 'eval-frozen-float32-get_latent-no-masking-no-pooling',
    'tf32': 'disabled-for-extraction',
}
TIMING = {
    'schema': 'official-musicfm-centered-stft-two-res2d-stride2-v1',
    'sample_rate': SAMPLE_RATE, 'stft_n_fft': 2048, 'stft_hop_length': 240,
    'stft_center': True, 'drop_final_mel_frame': True,
    'convolution_time_strides': [2, 2], 'convolution_kernel_size': 3,
    'convolution_padding': 1, 'first_frame_seconds': 0., 'frame_hop_seconds': .04,
    'length_rule': 'ceil(floor(resampled_num_samples/240)/4)',
    'edge_context': 'official-centered-stft-reflect-and-convolution-zero-padding',
    'all_centers_inside_source': True,
}


def _feature_length(samples):
    return (samples // 240 + 3) // 4


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file_stat(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256: ' + name)
    return value


def _duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('A finite positive complete audio duration is required')
    return float(value)


def _audio_metadata(row):
    digest = _sha(row.get('audio_sha256'), 'audio_sha256')
    duration = _duration(row.get('audio_duration_seconds'))
    window = row.get('audio_window_seconds')
    if (not isinstance(window, list) or len(window) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in window) or
            window != [0., duration]):
        raise ValueError('MusicFM requires complete source [0,duration], never a chunk crop')
    return digest, duration


def read_audio_manifest(path):
    """Read only source-audio identity fields; captions/review labels are unused."""
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f'{path}:{number}: audio row must be an object')
            row = {key: raw[key] for key in ('audio_path', 'audio_sha256',
                   'audio_duration_seconds', 'audio_window_seconds') if key in raw}
            _audio_metadata(row)
            if not isinstance(row.get('audio_path'), str) or not row['audio_path'].strip():
                raise ValueError(f'{path}:{number}: nonempty audio_path required')
            audio = Path(row['audio_path']).expanduser()
            row['audio_path'] = str((path.parent / audio).resolve() if not audio.is_absolute()
                                    else audio.resolve())
            rows.append(row)
    if not rows:
        raise ValueError('Empty audio manifest: ' + str(path))
    return rows


def load_audio(row):
    """Audit exact source bytes and return its entire mono float32 24k waveform."""
    import soundfile as sf
    from scipy.signal import resample_poly
    digest, duration = _audio_metadata(row)
    path = Path(row['audio_path']).expanduser().resolve()
    before = _file_stat(path)
    if file_sha256(path) != digest:
        raise ValueError('Audio bytes differ from declared audio_sha256')
    audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if _file_stat(path) != before:
        raise ValueError('Audio changed while checking and decoding source bytes')
    if not isinstance(rate, int) or rate <= 0 or not len(audio):
        raise ValueError('Nonempty audio with a positive sample rate required')
    if audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError('MusicFM requires finite original mono audio; no silent downmix')
    if abs(len(audio) / rate - duration) > 1. / rate + 1e-12:
        raise ValueError('Declared duration differs from complete source audio')
    source_frames = len(audio)
    waveform = audio[:, 0]
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // divisor,
                                 rate // divisor).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError('Resampling produced invalid audio')
    metadata = dict(audio_sha256=digest, audio_duration_seconds=duration,
                    source_num_samples=source_frames, source_sample_rate=rate,
                    source_channels=1, resampled_num_samples=len(waveform))
    return torch.from_numpy(np.ascontiguousarray(waveform)), metadata


def extractor_identity(musicfm_root, checkpoint, stats, conformer_config, layer_ix=7):
    if type(layer_ix) is not int or not 0 <= layer_ix <= 12:
        raise ValueError('MusicFM layer_ix must be an integer in [0,12]')
    root = Path(musicfm_root).expanduser().resolve()
    required = ('model/musicfm_25hz.py', 'modules/features.py')
    if any(not (root / name).is_file() for name in required):
        raise ValueError('--musicfm-root must contain model/ and modules/ from the official checkout')
    sources = {path.relative_to(root).as_posix(): file_sha256(path)
               for directory in ('model', 'modules')
               for path in sorted((root / directory).rglob('*.py'))}
    versions = {}
    for package in ('torch', 'torchaudio', 'transformers', 'numpy', 'scipy', 'soundfile', 'einops'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not-installed'
    identity = dict(schema=EXTRACTOR_SCHEMA, model='official-MusicFM25Hz',
                    layer_ix=layer_ix, feature_dim=FEATURE_DIM,
                    source_files=sources, source_tree_sha256=object_sha256(sources),
                    checkpoint_sha256=file_sha256(checkpoint), stats_sha256=file_sha256(stats),
                    conformer_config_sha256=file_sha256(conformer_config),
                    extractor_code_sha256=file_sha256(__file__),
                    preprocessing=dict(PREPROCESSING), timing=dict(TIMING), runtime_versions=versions)
    return identity


@contextmanager
def _offline():
    """Fail closed on ordinary network calls during local import/construction."""
    def forbidden(*args, **kwargs):
        raise RuntimeError('Network access is forbidden during offline MusicFM extraction')
    with patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                'HF_HUB_DISABLE_TELEMETRY': '1'}), \
            patch.object(socket.socket, 'connect', forbidden), \
            patch.object(socket.socket, 'connect_ex', forbidden), \
            patch.object(socket, 'getaddrinfo', forbidden), \
            patch.object(socket, 'create_connection', forbidden):
        yield


@contextmanager
def _local_musicfm_namespace(root):
    """Use this checkout even when its folder name is not literally 'musicfm'."""
    previous = {key: value for key, value in sys.modules.items()
                if key == 'musicfm' or key.startswith('musicfm.')}
    if previous:
        raise RuntimeError('MusicFM is already imported; use a fresh extraction process for a local source checkout')
    package = types.ModuleType('musicfm')
    package.__path__ = [str(root)]
    sys.modules['musicfm'] = package
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == 'musicfm' or key.startswith('musicfm.'):
                del sys.modules[key]


def load_local_musicfm(musicfm_root, checkpoint, stats, conformer_config, device='cpu'):
    """Construct the official class with local config and tensor-only weights.

    Upstream hardcodes Config.from_pretrained(remote_id) and torch.load(). Both
    are avoided: the former is scoped to the supplied local JSON and the latter
    is disabled via model_path=None, followed by weights_only=True strict load.
    """
    root = Path(musicfm_root).expanduser().resolve()
    config_path = Path(conformer_config).expanduser().resolve()
    config_value = json.loads(config_path.read_text(encoding='utf-8'))
    stats_value = json.loads(Path(stats).read_text(encoding='utf-8'))
    if not isinstance(config_value, dict) or not isinstance(stats_value, dict):
        raise ValueError('Local Conformer config and MusicFM stats must be JSON objects')
    for key in ('melspec_2048_mean', 'melspec_2048_std'):
        value = stats_value.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('MusicFM stats require finite scalar ' + key)
    if stats_value['melspec_2048_std'] <= 0:
        raise ValueError('MusicFM normalization std must be positive')
    with _offline(), _local_musicfm_namespace(root):
        from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
        def local_config(*args, **kwargs):
            return Wav2Vec2ConformerConfig.from_dict(dict(config_value))
        with patch.object(Wav2Vec2ConformerConfig, 'from_pretrained', side_effect=local_config):
            module = importlib.import_module('musicfm.model.musicfm_25hz')
            model = module.MusicFM25Hz(is_flash=False, stat_path=str(Path(stats).resolve()),
                                      model_path=None)
        checkpoint_value = torch.load(str(Path(checkpoint).resolve()), map_location='cpu',
                                      weights_only=True)
        if not isinstance(checkpoint_value, dict) or not isinstance(checkpoint_value.get('state_dict'), dict):
            raise ValueError('Require official MusicFM checkpoint with state_dict')
        state = checkpoint_value['state_dict']
        if (not state or any(not isinstance(key, str) or not key.startswith('model.') or
                             not isinstance(value, torch.Tensor) for key, value in state.items())):
            raise ValueError('Official MusicFM state_dict must contain tensor-only model.* keys')
        model.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True)
        model = model.to(device=device, dtype=torch.float32).eval()
        model.requires_grad_(False)
        _validate_model_timing(model)
    return model


def _validate_model_timing(model):
    spectrogram = model.preprocessor_melspec_2048.mel_stft.spectrogram
    if (spectrogram.n_fft != 2048 or spectrogram.hop_length != 240 or
            spectrogram.center is not True or spectrogram.pad != 0):
        raise ValueError('Local MusicFM STFT differs from the declared frame timing protocol')
    blocks = model.conv.conv
    if len(blocks) != 2:
        raise ValueError('Local MusicFM must have two residual subsampling blocks')
    for block in blocks:
        for name, stride in (('conv1', (2, 2)), ('conv2', (1, 1)), ('conv3', (2, 2))):
            conv = getattr(block, name)
            if conv.kernel_size != (3, 3) or conv.padding != (1, 1) or conv.stride != stride:
                raise ValueError('Local MusicFM convolution differs from declared frame timing protocol')


def extract_features(model, waveform, layer_ix=7, device='cpu'):
    if type(layer_ix) is not int or not 0 <= layer_ix <= 12:
        raise ValueError('MusicFM layer_ix must be an integer in [0,12]')
    if waveform.ndim != 1 or waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
        raise ValueError('Require a finite full mono float32 waveform')
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('MusicFM must be frozen and in eval mode')
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError('Frozen MusicFM extraction requires FP32 parameters')
    device = torch.device(device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with _offline(), torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            # get_latent calls get_predictions, never pretraining forward/masking.
            result = model.get_latent(waveform.to(device).unsqueeze(0), layer_ix=layer_ix)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
    if (not isinstance(result, torch.Tensor) or result.ndim != 3 or result.shape[0] != 1 or
            result.shape[1] <= 0 or result.shape[1] != _feature_length(waveform.numel()) or
            result.shape[2] != FEATURE_DIM or
            result.dtype != torch.float32 or not torch.isfinite(result).all()):
        raise ValueError('MusicFM get_latent must return finite FP32 [1,time,1024]')
    return result[0].detach().cpu().contiguous()


def _feature_metadata(entry):
    return {'schema': FEATURE_SCHEMA,
            **{key: json.dumps(entry[key], sort_keys=True, separators=(',', ':'))
               for key in ('audio_sha256', 'audio_duration_seconds', 'source_num_samples',
                           'source_sample_rate', 'source_channels', 'resampled_num_samples',
                           'feature_shape', 'feature_dtype', 'extractor_identity_sha256')}}


def _safe_member(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Cache feature path must be a relative POSIX path')
    member = Path(relative)
    if member.is_absolute() or '..' in member.parts or member.drive:
        raise ValueError('Cache feature path must stay inside cache root')
    resolved = (root / member).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError('Cache feature path escapes cache root')
    return resolved


def _load_feature_file(root, entry):
    from safetensors import safe_open
    path = _safe_member(root, entry['feature_file'])
    before = _file_stat(path)
    if file_sha256(path) != entry['feature_sha256']:
        raise ValueError('MusicFM feature file bytes differ from index')
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        if set(stream.keys()) != {'features'} or stream.metadata() != _feature_metadata(entry):
            raise ValueError('MusicFM feature payload metadata differs from index')
        features = stream.get_tensor('features')
    if _file_stat(path) != before:
        raise ValueError('MusicFM feature file changed while verifying and reading bytes')
    if (list(features.shape) != entry['feature_shape'] or features.dtype != torch.float32 or
            not torch.isfinite(features).all()):
        raise ValueError('MusicFM feature tensor differs from declared shape/dtype/finiteness')
    return features


class FeatureStore:
    """Read a portable cache; validation is metadata-only unless requested.

    ``identity`` binds the index for training fingerprints. Inference on new
    music compares only ``identity['extractor_identity']`` with training.
    """
    def __init__(self, index_path):
        self.index_path = Path(index_path).expanduser().resolve()
        raw = self.index_path.read_bytes()
        self.index = json.loads(raw)
        if (not isinstance(self.index, dict) or set(self.index) != {
                'schema', 'extractor_identity', 'extractor_identity_sha256', 'entries'} or
                self.index['schema'] != SCHEMA):
            raise ValueError('Invalid MusicFM feature index schema')
        extractor = self.index['extractor_identity']
        expected = object_sha256(extractor)
        if (not isinstance(extractor, dict) or extractor.get('schema') != EXTRACTOR_SCHEMA or
                extractor.get('feature_dim') != FEATURE_DIM or
                extractor.get('preprocessing') != PREPROCESSING or
                extractor.get('timing') != TIMING or
                self.index['extractor_identity_sha256'] != expected):
            raise ValueError('Invalid MusicFM extractor identity')
        entries = self.index['entries']
        if not isinstance(entries, dict) or not entries:
            raise ValueError('Nonempty MusicFM feature entries required')
        for digest, entry in entries.items():
            _sha(digest, 'entry audio identity')
            if not isinstance(entry, dict) or entry.get('audio_sha256') != digest:
                raise ValueError('MusicFM entry key differs from audio identity')
            _duration(entry.get('audio_duration_seconds'))
            if entry.get('extractor_identity_sha256') != expected:
                raise ValueError('Mixed MusicFM extractors in one cache')
            shape = entry.get('feature_shape')
            if (not isinstance(shape, list) or len(shape) != 2 or
                    type(shape[0]) is not int or shape[0] <= 0 or shape[1] != FEATURE_DIM or
                    entry.get('feature_dtype') != 'float32'):
                raise ValueError('MusicFM cache requires dense [time,1024] float32 features')
            for key in ('source_num_samples', 'source_sample_rate', 'resampled_num_samples'):
                if type(entry.get(key)) is not int or entry[key] <= 0:
                    raise ValueError('Invalid MusicFM source sample metadata: ' + key)
            if entry.get('source_channels') != 1:
                raise ValueError('Original mono source required')
            rate = entry['source_sample_rate']
            if abs(entry['source_num_samples'] / rate - entry['audio_duration_seconds']) > 1. / rate + 1e-12:
                raise ValueError('Cache duration differs from source sample count')
            if entry['resampled_num_samples'] != math.ceil(entry['source_num_samples'] * SAMPLE_RATE / rate):
                raise ValueError('Cache resampled sample count differs from protocol')
            if shape[0] != _feature_length(entry['resampled_num_samples']):
                raise ValueError('MusicFM dense feature length differs from native timing protocol')
            _sha(entry.get('feature_sha256'), 'feature_sha256')
            expected_path = 'features/' + digest + '.safetensors'
            if entry.get('feature_file') != expected_path:
                raise ValueError('Cache feature filename must bind its audio SHA256')
        self.root = self.index_path.parent
        self.entries = entries
        self.feature_dim = FEATURE_DIM
        self.identity = dict(schema=SCHEMA, extractor_identity=extractor,
                             extractor_identity_sha256=expected,
                             index_sha256=hashlib.sha256(raw).hexdigest(), entries=len(entries))

    def _entry(self, row):
        digest, duration = _audio_metadata(row)
        entry = self.entries.get(digest)
        if entry is None:
            raise ValueError('MusicFM feature cache misses audio: ' + digest)
        if abs(duration - entry['audio_duration_seconds']) > 1e-9:
            raise ValueError('MusicFM cache duration differs from row')
        return entry

    def validate_rows(self, rows, verify_bytes=False):
        count = 0
        for row in rows:
            self._entry(row)
            if verify_bytes:
                self.load(row)
            count += 1
        return {'rows': count, 'feature_files_verified': bool(verify_bytes)}

    def load(self, row):
        return _load_feature_file(self.root, self._entry(row))

    def frame_times(self, row):
        entry = self._entry(row)
        times = torch.arange(entry['feature_shape'][0], dtype=torch.float32) * .04
        if times[-1].item() >= entry['source_num_samples'] / entry['source_sample_rate']:
            raise ValueError('MusicFM frame centers must lie inside the source audio')
        return times


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
                             encoding='utf-8')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(manifests, output_dir, musicfm_root, checkpoint, stats, conformer_config,
            device='cpu', layer_ix=7):
    """Create/extend one extractor's cache, reusing only verified exact matches."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    identity = extractor_identity(musicfm_root, checkpoint, stats, conformer_config, layer_ix)
    identity_sha = object_sha256(identity)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / 'index.json'
    entries = {}
    if index_path.exists():
        existing = FeatureStore(index_path)
        if existing.identity['extractor_identity'] != identity:
            raise ValueError('Refuse to mix different MusicFM extractor identities in one output cache')
        entries.update(existing.entries)
    rows = [row for manifest in manifests for row in read_audio_manifest(manifest)]
    if not rows:
        raise ValueError('At least one nonempty audio manifest required')
    model = None
    for row in rows:
        waveform, audio = load_audio(row)
        digest = audio['audio_sha256']
        relative = 'features/' + digest + '.safetensors'
        destination = _safe_member(root, relative)
        if digest in entries:
            old = entries[digest]
            if any(old[key] != value for key, value in audio.items()):
                raise ValueError('Same audio SHA256 has conflicting declared source metadata')
            # Verify the actual cached file, not only its previous index declaration.
            _load_feature_file(root, old)
            continue
        if destination.exists():
            # Recover an interrupted run only if the unindexed tensor binds this
            # exact audio and extractor. Otherwise fail rather than overwrite it.
            with safe_open(str(destination), framework='pt', device='cpu') as stream:
                metadata = stream.metadata()
                if (not isinstance(metadata, dict) or metadata.get('schema') != FEATURE_SCHEMA or
                        set(stream.keys()) != {'features'}):
                    raise ValueError('Unindexed MusicFM cache file has invalid metadata')
                entry = {key: json.loads(value) for key, value in metadata.items() if key != 'schema'}
                recovered = stream.get_tensor('features')
            if (any(entry.get(key) != value for key, value in audio.items()) or
                    entry.get('extractor_identity_sha256') != identity_sha or
                    entry.get('feature_dtype') != 'float32' or
                    entry.get('feature_shape') != list(recovered.shape) or
                    recovered.ndim != 2 or recovered.shape[1] != FEATURE_DIM or
                    recovered.shape[0] != _feature_length(audio['resampled_num_samples']) or
                    recovered.dtype != torch.float32 or not torch.isfinite(recovered).all()):
                raise ValueError('Unindexed MusicFM cache file belongs to another audio/extractor')
            if metadata != _feature_metadata(entry):
                raise ValueError('Unindexed MusicFM cache metadata differs from the feature protocol')
            entry.update(feature_file=relative, feature_sha256=file_sha256(destination))
        else:
            if model is None:
                model = load_local_musicfm(musicfm_root, checkpoint, stats, conformer_config, device)
                if extractor_identity(musicfm_root, checkpoint, stats, conformer_config, layer_ix) != identity:
                    raise ValueError('MusicFM source or model files changed during model construction')
            features = extract_features(model, waveform, layer_ix, device)
            entry = dict(audio, feature_shape=list(features.shape), feature_dtype='float32',
                         extractor_identity_sha256=identity_sha, feature_file=relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                save_file({'features': features}, str(temporary), metadata=_feature_metadata(entry))
                os.replace(str(temporary), str(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
            entry['feature_sha256'] = file_sha256(destination)
        entries[digest] = entry
    _atomic_json(index_path, dict(schema=SCHEMA, extractor_identity=identity,
                                 extractor_identity_sha256=identity_sha, entries=entries))
    store = FeatureStore(index_path)
    store.validate_rows(rows, verify_bytes=False)
    return store.identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare', help='Extract full frozen MusicFM frame sequences offline')
    create.add_argument('--manifest', action='append', required=True,
                        help='Repeat for train/val or new inference audio; target fields are ignored')
    create.add_argument('--output-dir', required=True)
    create.add_argument('--musicfm-root', required=True)
    create.add_argument('--checkpoint', required=True)
    create.add_argument('--stats', required=True)
    create.add_argument('--conformer-config', required=True,
                        help='Local wav2vec2-conformer-rope-large-960h-ft config.json; never fetched')
    create.add_argument('--device', default='cuda:0')
    create.add_argument('--layer-ix', type=int, default=7)
    validate = commands.add_parser('validate', help='Audit a feature index against source audio identities')
    validate.add_argument('--index', required=True)
    validate.add_argument('--manifest', action='append', required=True)
    validate.add_argument('--verify-bytes', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.manifest, args.output_dir, args.musicfm_root, args.checkpoint,
                         args.stats, args.conformer_config, args.device, args.layer_ix)
    else:
        store = FeatureStore(args.index)
        result = store.validate_rows((row for path in args.manifest for row in read_audio_manifest(path)),
                                     verify_bytes=args.verify_bytes)
        result['identity'] = store.identity
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
