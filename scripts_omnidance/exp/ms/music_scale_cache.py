"""Fresh sharded native H3/MusicFM caches with the online 32kHz waveform path.

Teacher: /opt/venv_h3/bin/python music_scale_cache.py --stage teacher ...
MusicFM: /opt/venv_music/bin/python music_scale_cache.py --stage music ...
Merge: either environment, music_scale_cache.py merge --parts p0.pt p1.pt --caps ... --out ...
No adapter, text encoder, DiT, training, or old-cache overwrite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
AUDIO_ROOT = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio'
SOURCE_RATE, INPUT_SAMPLES, LATENT_FRAMES = 32000, 162000, 203
DEFAULT_WEIGHTS = '/dev/shm/h3/Ref2VA/audio_vae/model.safetensors'
DEFAULT_MODEL_PATH = '/opt/models/MusicFM-MSD'
DYNAMIC_METADATA = {'shard', 'count', 'effective_frames', 'reference_max_abs_error',
                    'tensor_sha256', 'created_utc', 'merged_input_sha256', 'part_shards'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_caps(path):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key in caps: {key}')
            result[key] = value
        return result

    with Path(path).open() as stream:
        caps = json.load(stream, object_pairs_hook=no_duplicate_keys)
    if not isinstance(caps, dict) or not caps:
        raise ValueError('Caps must be a nonempty actual-ID caption dictionary, not an audit/manifest')
    schemas = set()
    for sid, entry in caps.items():
        if (not isinstance(sid, str) or not sid or sid in ('.', '..', 'audit_only')
                or '/' in sid or '\\' in sid):
            raise ValueError(f'Invalid actual clip ID: {sid!r}')
        if isinstance(entry, str):
            if not entry.strip():
                raise ValueError(f'Empty caption: {sid}')
            schemas.add('caption_string')
        elif isinstance(entry, dict):
            if entry.get('audit_only') or entry.get('split') == 'audit_only' or entry.get('scope') == 'audit_only':
                raise ValueError(f'Audit-only entry forbidden in cache: {sid}')
            caption = entry.get('caption', entry.get('text', entry.get('prompt')))
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f'Expected actual caption entry: {sid}')
            schema = entry.get('text_schema', 'caption_dict')
            if not isinstance(schema, str) or not schema:
                raise ValueError(f'Invalid text_schema: {sid}')
            schemas.add(schema)
        else:
            raise ValueError(f'Expected actual caption entry, not manifest metadata: {sid}')
    if len(schemas) != 1:
        raise ValueError('Caps must use one consistent text schema')
    return sorted(caps), schemas.pop()


def check_output(path):
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    if not path.parent.is_dir():
        raise FileNotFoundError(f'Output parent must already exist: {path.parent}')
    return path


def log_progress(event, stage, count, total, start, device=None, **extra):
    peak = torch.cuda.max_memory_allocated(device) if device is not None and device.type == 'cuda' else 0
    record = dict(event=event, stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic() - start, 3), count=count, total=total,
                  cuda_max_allocated_bytes=peak, cpu_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record.update(extra)
    print(json.dumps(record, allow_nan=False), flush=True)


def base_metadata(stage, caps_path, ids, text_schema, shard, nshard):
    return dict(cache_version=1, stage=stage, caps_path=str(Path(caps_path).resolve()),
                caps_sha256=sha256(caps_path), caps_count=len(ids), text_schema=text_schema,
                source=str(AUDIO_ROOT.resolve()), waveform_path='sourcewav->32k' + ('->24k' if stage == 'music' else ''),
                waveform_loader='music_clip_bridge.load_audio(sid)', source_sample_rate=SOURCE_RATE,
                source_input_samples=INPUT_SAMPLES, input_samples=INPUT_SAMPLES,
                audio_mask='arange(203) < effective_frames[sid], at 40Hz; excludes right padding',
                audio_mask_frame_hz=40.0, effective_frames={}, shard=shard, nshard=nshard,
                frozen=True, compute_dtype='float32', created_utc=datetime.now(timezone.utc).isoformat())


def load_teacher(weights, device):
    from safetensors.torch import load_file
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    from diffsynth.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
    model = MiniMaxH3AudioVAE()
    state = load_file(weights, device='cpu')
    for name in ('q_bias', 'zero_k_bias', 'v_bias'):
        key = f'pre_block.attn.{name}'
        state[key + '.weight'] = state.pop(key)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


@torch.no_grad()
def teacher_batch(model, waveform):
    batch = len(waveform)
    audio = waveform.reshape(2 * batch, 1, INPUT_SAMPLES)
    z = model.encoder(model.preprocess(audio))
    z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
    mean, std = model._latent_stats(waveform.device)
    if mean.shape != (1, 32, 1) or std.shape != (1, 32, 1) or not torch.all(std > 0):
        raise ValueError('Unexpected native latent normalization statistics')
    latent = (model.mean_proj(z).float() - mean) / std
    if latent.shape != (2 * batch, 32, LATENT_FRAMES) or not torch.isfinite(latent).all():
        raise ValueError('Expected finite normalized native [2B,32,203] teacher latents')
    return latent.reshape(batch, 2, 32, LATENT_FRAMES)


@torch.no_grad()
def music_batch(model, waveform):
    from torchaudio.functional import resample
    batch = len(waveform)
    mono = resample(waveform.reshape(2 * batch, INPUT_SAMPLES), SOURCE_RATE, 24000)
    sequence = model(mono)
    if (sequence.ndim != 3 or sequence.shape[0] != 2 * batch or sequence.shape[1] < 1
            or sequence.shape[-1] != 1024 or not torch.isfinite(sequence).all()):
        raise ValueError('Expected finite dense MusicFM [2B,T,1024] features')
    return sequence.reshape(batch, 2, sequence.shape[1], 1024)


def validate_cache(state, expected_ids=None):
    if not isinstance(state, dict) or not isinstance(state.get('metadata'), dict):
        raise ValueError('Expected a cache dictionary with metadata')
    meta = state['metadata']
    stage = meta.get('stage')
    if stage not in ('teacher', 'music'):
        raise ValueError('Invalid cache stage')
    key = 'latents' if stage == 'teacher' else 'features'
    if set(state) != {key, 'metadata'} or not isinstance(state[key], dict) or not state[key]:
        raise ValueError('Expected exactly the original latents/features and metadata cache interface')
    values = state[key]
    if expected_ids is not None and set(values) != set(expected_ids):
        raise ValueError('Cache IDs do not match the expected actual caps/shard IDs')
    common = dict(cache_version=1, source_sample_rate=32000, source_input_samples=162000,
                  input_samples=162000, frozen=True, compute_dtype='float32', audio_mask_frame_hz=40.0)
    interface = (dict(normalized=True, sample_rate=32000, hop_length=800, frame_hz=40.0,
                      latent_channels=32, latent_frames=203, dtype='float32', waveform_path='sourcewav->32k')
                 if stage == 'teacher' else
                 dict(backend='musicfm', sample_rate=24000, frame_hz=25.0, hidden_size=1024,
                      dtype='float16', waveform_path='sourcewav->32k->24k'))
    if any(meta.get(name) != value for name, value in {**common, **interface}.items()):
        raise ValueError('Cache interface metadata disagrees with the online backbone waveform path')
    provenance = ('caps_path', 'caps_sha256', 'text_schema', 'source', 'waveform_loader', 'audio_mask')
    provenance += ('weights_path', 'weights_sha256') if stage == 'teacher' else ('model_path',)
    if any(not isinstance(meta.get(name), str) or not meta[name] for name in provenance):
        raise ValueError('Cache must include backbone/source/caps provenance metadata')
    if stage == 'music':
        model_hashes = meta.get('model_sha256')
        if (not isinstance(model_hashes, dict) or set(model_hashes) != {'msd_stats.json', 'pretrained_msd.pt'}
                or any(not isinstance(value, str) or len(value) != 64 for value in model_hashes.values())
                or type(meta.get('dense_parameters')) is not int or meta['dense_parameters'] < 1
                or meta.get('trainable_parameters') != 0):
            raise ValueError('MusicFM cache must preserve original dense count and frozen model hashes')
    frames, hashes = meta.get('effective_frames'), meta.get('tensor_sha256')
    if not isinstance(frames, dict) or set(frames) != set(values) or not isinstance(hashes, dict) or set(hashes) != set(values):
        raise ValueError('Effective frames and tensor hashes must be keyed by exactly the cached IDs')
    if meta.get('count') != len(values):
        raise ValueError('Cache count disagrees with its tensor dictionary')
    shapes = set()
    for sid, value in values.items():
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.requires_grad:
            raise ValueError(f'Expected detached CPU tensor: {sid}')
        expected_dtype = torch.float32 if stage == 'teacher' else torch.float16
        shape_ok = (value.shape == (2, 32, 203) if stage == 'teacher' else
                    value.ndim == 3 and value.shape[0] == 2 and value.shape[1] > 0 and value.shape[2] == 1024)
        if not shape_ok or value.dtype != expected_dtype or not torch.isfinite(value).all():
            raise ValueError(f'Invalid cache tensor shape/dtype/finite values: {sid}')
        if type(frames[sid]) is not int or not 1 <= frames[sid] <= LATENT_FRAMES:
            raise ValueError(f'Invalid effective frame count: {sid}')
        if hashes[sid] != tensor_sha256(value):
            raise ValueError(f'Cache tensor checksum failed: {sid}')
        shapes.add(tuple(value.shape))
    if len(shapes) != 1:
        raise ValueError('Fixed-length waveform cache must have consistent tensor shapes')
    if stage == 'teacher':
        errors = meta.get('reference_max_abs_error')
        if not isinstance(errors, dict) or not errors or not set(errors).issubset(values):
            raise ValueError('Teacher must record its first-three native encode_audio checks')
        if meta.get('shard') is not None and set(errors) != set(sorted(values)[:3]):
            raise ValueError('Teacher shard must compare exactly its first three IDs with encode_audio')
        if any(not isinstance(error, (int, float)) or not 0 <= error <= 1e-4 for error in errors.values()):
            raise ValueError('Teacher native encode_audio error exceeds 1e-4 or is non-finite')
    return key


def atomic_save(state, path, expected_ids):
    path = check_output(path)
    validate_cache(state, expected_ids)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = torch.load(temporary, map_location='cpu', weights_only=True)
        validate_cache(loaded, expected_ids)
        if loaded['metadata'] != state['metadata']:
            raise ValueError('Serialized checkpoint metadata changed')
        del loaded
        digest = sha256(temporary)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return digest


def cache(args):
    output = check_output(args.out)
    all_ids, text_schema = read_caps(args.caps)
    if args.nshard < 1 or not 0 <= args.shard < args.nshard or args.batch_size < 1:
        raise ValueError('Require positive batch_size/nshard and 0 <= shard < nshard')
    ids = all_ids[args.shard::args.nshard]
    if not ids:
        raise ValueError('Empty cache shard')
    from music_clip_bridge import load_audio
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    meta = base_metadata(args.stage, args.caps, all_ids, text_schema, args.shard, args.nshard)
    if args.stage == 'teacher':
        model = load_teacher(args.weights, device)
        meta.update(normalized=True, sample_rate=32000, hop_length=800, frame_hz=40.0,
                    latent_channels=32, latent_frames=203, dtype='float32',
                    weights_path=str(Path(args.weights).resolve()), weights_sha256=sha256(args.weights),
                    reference_max_abs_error={})
        key, forward = 'latents', teacher_batch
    else:
        from music_encoder_adapter import DenseMusicEncoder
        model = DenseMusicEncoder('musicfm', args.model_path)
        # Preserve the official dense parameter count before freezing extraction.
        meta.update(model.metadata())
        model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        meta.update(dtype='float16', trainable_parameters=0,
                    model_path=str(Path(args.model_path).resolve()),
                    model_sha256={name: sha256(Path(args.model_path) / name)
                                  for name in ('msd_stats.json', 'pretrained_msd.pt')})
        key, forward = 'features', music_batch
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise ValueError('Extraction model must be frozen and in eval mode')
    values = {}
    log_progress('start', args.stage, 0, len(ids), start, device, shard=args.shard, nshard=args.nshard)
    with torch.no_grad():
        for index in range(0, len(ids), args.batch_size):
            chunk = ids[index:index + args.batch_size]
            loaded = [load_audio(sid) for sid in chunk]
            for sid, (waveform, frames) in zip(chunk, loaded):
                if waveform.shape != (2, INPUT_SAMPLES) or waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
                    raise ValueError(f'Expected finite online source [2,162000] FP32 at 32kHz: {sid}')
                if type(frames) is not int or not 1 <= frames <= LATENT_FRAMES:
                    raise ValueError(f'Invalid online effective frames: {sid}')
                meta['effective_frames'][sid] = frames
            waveform = torch.stack([item[0] for item in loaded]).to(device=device, dtype=torch.float32)
            result = forward(model, waveform)
            for offset, sid in enumerate(chunk):
                if args.stage == 'teacher' and index + offset < 3:
                    reference = model.encode_audio(waveform[offset])
                    if reference.shape != result[offset].shape or not torch.isfinite(reference).all():
                        raise ValueError(f'Invalid native encode_audio reference: {sid}')
                    error = (reference.float() - result[offset]).abs().max().item()
                    if not 0 <= error <= 1e-4:
                        raise ValueError(f'Batched teacher differs from encode_audio: {sid}, maxerr={error}')
                    meta['reference_max_abs_error'][sid] = error
                values[sid] = result[offset].detach().to(device='cpu', dtype=torch.float32 if args.stage == 'teacher' else torch.float16).contiguous()
                if not torch.isfinite(values[sid]).all():
                    raise ValueError(f'Non-finite tensor after cache dtype conversion: {sid}')
                if len(values) % 25 == 0:
                    log_progress('progress', args.stage, len(values), len(ids), start, device)
            del waveform, result, loaded
    meta.update(count=len(values), tensor_sha256={sid: tensor_sha256(value) for sid, value in values.items()})
    checkpoint_hash = atomic_save({key: values, 'metadata': meta}, output, ids)
    log_progress('done', args.stage, len(values), len(ids), start, device,
                 output=str(output), checkpoint_sha256=checkpoint_hash, tensors_verified=True)


def merge(args):
    output = check_output(args.out)
    ids, text_schema = read_caps(args.caps)
    if len(set(str(Path(path).resolve()) for path in args.parts)) != len(args.parts):
        raise ValueError('Merge parts must be explicit distinct files')
    values, frames, errors, hashes, input_hashes, shards = {}, {}, {}, {}, {}, []
    original_fixed, original_meta, key = None, None, None
    start = time.monotonic()
    for path in args.parts:
        state = torch.load(path, map_location='cpu', weights_only=True)
        part_key = validate_cache(state)
        meta = state['metadata']
        if meta.get('caps_sha256') != sha256(args.caps) or meta.get('caps_count') != len(ids) or meta.get('text_schema') != text_schema:
            raise ValueError(f'Merge part caps hash/count/text schema mismatch: {path}')
        fixed = {name: value for name, value in meta.items() if name not in DYNAMIC_METADATA}
        if original_fixed is None:
            original_fixed, original_meta, key = fixed, meta.copy(), part_key
        elif fixed != original_fixed or part_key != key:
            raise ValueError(f'Merge parts disagree on strict backbone/interface/path metadata: {path}')
        shard, nshard = meta.get('shard'), meta.get('nshard')
        if type(shard) is not int or type(nshard) is not int or not 0 <= shard < nshard or shard in shards:
            raise ValueError(f'Invalid or duplicate part shard: {path}')
        if set(state[key]) != set(ids[shard::nshard]):
            raise ValueError(f'Merge part IDs disagree with declared shard: {path}')
        overlap = set(values).intersection(state[key])
        if overlap:
            raise ValueError(f'Duplicate cache IDs across merge parts: {sorted(overlap)[:3]}')
        values.update(state[key])
        frames.update(meta['effective_frames'])
        hashes.update(meta['tensor_sha256'])
        errors.update(meta.get('reference_max_abs_error', {}))
        input_hashes[str(Path(path).resolve())] = sha256(path)
        shards.append(shard)
        log_progress('merge_part', meta['stage'], len(values), len(ids), start, part=str(path))
    if original_meta is None or set(values) != set(ids):
        raise ValueError('Merged union must contain every actual caps ID exactly once')
    meta = {name: value for name, value in original_meta.items() if name not in DYNAMIC_METADATA}
    meta.update(shard=None, count=len(values), effective_frames=frames, tensor_sha256=hashes,
                created_utc=datetime.now(timezone.utc).isoformat(), part_shards=sorted(shards),
                merged_input_sha256=input_hashes)
    if key == 'latents':
        meta['reference_max_abs_error'] = errors
    digest = atomic_save({key: {sid: values[sid] for sid in ids}, 'metadata': meta}, output, ids)
    log_progress('done', meta['stage'], len(values), len(ids), start,
                 output=str(output), checkpoint_sha256=digest, tensors_verified=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', required=True)
    parser.add_argument('--out', required=True)
    if argv and argv[0] == 'merge':
        parser.add_argument('--parts', nargs='+', required=True)
        args = parser.parse_args(argv[1:])
        merge(args)
    else:
        parser.add_argument('--stage', choices=('teacher', 'music'), required=True)
        parser.add_argument('--shard', type=int, default=0)
        parser.add_argument('--nshard', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
        parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
        cache(parser.parse_args(argv))


if __name__ == '__main__':
    main()
