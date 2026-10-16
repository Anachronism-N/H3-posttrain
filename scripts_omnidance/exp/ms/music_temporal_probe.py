"""同网格音乐时序线性诊断；onset/log-RMS/beat 均为音频代理弱标签。
不证明人类节拍识别或音乐条件生成能力；不加载/训练 backbone、text 或 DiT。
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch

SR, N_FFT, HOP, N_MELS = 24000, 1024, 240, 64
N_ITEMS, N_TRAIN, N_VAL, FEATURE_DIM = 2000, 1800, 200, 1024
TARGET_NAMES = ('onset_flux_proxy', 'log_rms', 'beat_proxy')
WEAK_LABEL_NOTE = (
    'Audio-derived proxies only, not human-annotated onset/beat ground truth. '
    'beat_dp uses a strong 120 BPM tempo prior, 60-180 BPM search and tightness=100; '
    'the approximately 5-second window has not been human validated. '
    'Silence/constant onset or insufficient real windows yield no beat pulses. '
    'Readability of these labels is not evidence of human beat ability or generation quality.'
)


def common_times():
    return torch.arange(1, 125, dtype=torch.float64) / 25.0


def finite_guard(name, tensor):
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all().item():
        raise ValueError(f'{name}: missing/non-finite tensor')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_output(path):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if os.path.lexists(path):
        raise FileExistsError(f'Refusing existing output: {path}')
    return path


def atomic_write_new(path, payload, binary=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b' if binary else 'w', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if binary:
                torch.save(payload, stream)
            else:
                json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def valid_ids(ids):
    if len(ids) != N_ITEMS or len(set(ids)) != N_ITEMS:
        raise ValueError('Exactly 2000 unique IDs are required')
    if any(not isinstance(sid, str) or not sid or '/' in sid or '\\' in sid
           or sid in ('.', '..') for sid in ids):
        raise ValueError('Invalid clip ID')


def interpolate_sequence(values, times, source_hz, source_offset):
    """[T,D] 按真实秒插值；范围外返回0并mask，绝不把clamp尾帧当有效值。"""
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError('Expected nonempty [T,D] sequence')
    if not math.isfinite(source_hz) or source_hz <= 0 or not math.isfinite(source_offset):
        raise ValueError('Invalid source time specification')
    times = times.to(device=values.device, dtype=torch.float64)
    end = source_offset + (values.shape[0] - 1) / source_hz
    mask = (times >= source_offset - 1e-10) & (times <= end + 1e-10)
    positions = ((times - source_offset) * source_hz).clamp(0, values.shape[0] - 1)
    low = positions.floor().long()
    high = (low + 1).clamp(max=values.shape[0] - 1)
    weight = (positions - low).to(values.dtype).unsqueeze(1)
    result = values[low] * (1 - weight) + values[high] * weight
    return result.masked_fill(~mask[:, None], 0), mask


def complete_window_mask(times, duration):
    # Flux needs both the current centered window and its lag-1 predecessor.
    half = N_FFT / (2 * SR)
    return (times - HOP / SR - half >= -1e-10) & (times + half <= duration + 1e-10)


@torch.no_grad()
def audio_targets(waveform, duration, mel_transform):
    from audio_feats import beat_dp

    finite_guard('waveform', waveform)
    if waveform.shape != (2, 121500) or not 0 < duration <= 5.0625:
        raise ValueError('Expected cached stereo 24kHz/5.0625s waveform and real duration')
    mono = waveform.float().mean(0)
    mel = torch.log1p(10000 * mel_transform(mono)).transpose(0, 1)
    flux = torch.zeros(len(mel), dtype=torch.float32)
    flux[1:] = (mel[1:] - mel[:-1]).clamp_min(0).mean(1)
    frames = torch.nn.functional.pad(mono[None, None], (N_FFT // 2, N_FFT // 2),
                                     mode='reflect')[0, 0].unfold(0, N_FFT, HOP)
    log_rms = frames.square().mean(1).sqrt().clamp_min(1e-8).log()
    if len(log_rms) != len(mel):
        raise RuntimeError('Mel/RMS frame grids differ')
    times = common_times()
    mel25, mel_range = interpolate_sequence(mel, times, SR / HOP, 0.0)
    values25, target_range = interpolate_sequence(torch.stack((flux, log_rms), 1),
                                                  times, SR / HOP, 0.0)
    mask = complete_window_mask(times, duration) & mel_range & target_range
    times100 = torch.arange(len(flux), dtype=torch.float64) * HOP / SR
    real100 = complete_window_mask(times100, duration)
    indices = real100.nonzero(as_tuple=True)[0]
    tempo, beats, beat_status = None, [], 'insufficient_real_windows'
    if len(indices) >= 35:
        real_flux = flux[indices].numpy()
        if float(real_flux.std()) > 1e-8:
            tempo, beats = beat_dp(real_flux, sr=SR, hop=HOP)
            origin = times100[indices[0]].item()
            beats = [origin + beat for beat in beats]
            beat_status = 'weak_dp_proxy'
        else:
            beat_status = 'constant_onset_no_beats'
    beat25 = torch.zeros(len(times), dtype=torch.float64)
    if beats:
        distance = (times[:, None] - torch.tensor(beats, dtype=torch.float64)[None]).abs().min(1).values
        beat25 = torch.exp(-0.5 * (distance / 0.05).square())
    targets = torch.cat((values25, beat25.float()[:, None]), 1).masked_fill(~mask[:, None], 0)
    mel25 = mel25.masked_fill(~mask[:, None], 0)
    finite_guard('audio targets', targets)
    finite_guard('log mel features', mel25)
    return targets, mel25, mask, dict(tempo_proxy_bpm=tempo, beat_status=beat_status,
                                     beat_count=len(beats), complete_100hz_windows=len(indices))


@torch.no_grad()
def build_targets(args):
    import soundfile as sf
    import torchaudio
    from music_encoder_adapter import AUDIO_ROOT, load_waveform

    output = fresh_output(args.out)
    with open(args.caps) as stream:
        caps = json.load(stream)
    if not isinstance(caps, dict):
        raise ValueError('Captions must be an ID dictionary')
    ids = sorted(caps)
    valid_ids(ids)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, center=True,
        power=2.0)
    targets = torch.zeros(N_ITEMS, 124, 3, dtype=torch.float32)
    mel_features = torch.zeros(N_ITEMS, 124, N_MELS, dtype=torch.float32)
    mask = torch.zeros(N_ITEMS, 124, dtype=torch.bool)
    source_info = {}
    for index, sid in enumerate(ids):
        with sf.SoundFile(str(AUDIO_ROOT / (sid + '.wav'))) as stream:
            source_frames, source_sr = int(stream.frames), int(stream.samplerate)
        duration = min(source_frames / source_sr, 5.0625)
        values, mel, valid, beat_info = audio_targets(load_waveform(sid), duration, mel_transform)
        targets[index], mel_features[index], mask[index] = values, mel, valid
        source_info[sid] = dict(source_frames=source_frames, source_sample_rate=source_sr,
                                effective_duration_seconds=duration, **beat_info)
        if (index + 1) % 100 == 0:
            print(f'targets {index + 1}/{N_ITEMS}', flush=True)
    metadata = dict(
        schema='music_temporal_targets_v1', sample_rate=SR, waveform_seconds=5.0625,
        waveform_loader='music_encoder_adapter.load_waveform; separate channels at 24kHz then mean',
        grid_hz=25.0, grid_first_seconds=0.04, grid_last_seconds=4.96, grid_frames=124,
        target_names=list(TARGET_NAMES), n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        mel_power=2.0, mel_center=True, mel_log='log(1+10000*power)',
        onset='mean_mels(max(logmel[t]-logmel[t-1],0)); 100Hz -> 25Hz linear interpolation',
        log_rms='log(max(sqrt(mean(raw_mono_window**2)),1e-8)); 1024 samples, centered, hop240',
        beat='audio_feats.beat_dp on complete real 100Hz flux windows; max Gaussian pulse, sigma=0.05s',
        mask='current 1024-sample window AND preceding lag-1 window fully inside real source duration; no padding',
        weak_label_boundary=WEAK_LABEL_NOTE, derived_audio_cache_only=True, publication=False,
        caps_sha256=sha256(args.caps), source_info=source_info,
        samples=N_ITEMS, valid_frames=int(mask.sum()), clips_with_no_valid_frames=int((~mask.any(1)).sum()))
    finite_guard('all targets', targets)
    finite_guard('all mel features', mel_features)
    atomic_write_new(output, dict(ids=ids, times=common_times(), mask=mask, targets=targets,
                                  mel_features=mel_features, metadata=metadata), binary=True)
    print(json.dumps(dict(done='targets', out=str(output), samples=N_ITEMS,
                          valid_frames=int(mask.sum()), sha256=sha256(output)), allow_nan=False), flush=True)


def validate_feature_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('Feature metadata must be a dictionary')
    backend = metadata.get('backend')
    if backend not in ('muq', 'musicfm', 'mert'):
        raise ValueError('Expected backend muq/musicfm/mert')
    rate, offset = (75.0, 0.0083125) if backend == 'mert' else (25.0, 0.0)
    for key, expected in (('sample_rate', SR), ('hidden_size', FEATURE_DIM),
                          ('frame_hz', rate), ('frame_offset_seconds', offset)):
        value = metadata.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value - expected) > 1e-9:
            raise ValueError(f'{backend}: {key}={value!r}, expected {expected}')
    return backend, rate, offset


class AlignedInputs:
    def __init__(self, features, target_cache, source_hz, source_offset):
        self.features, self.targets = features, target_cache['targets'].float()
        self.mel = target_cache['mel_features'].float()
        self.ids, self.times = target_cache['ids'], target_cache['times'].double()
        self.rate, self.offset = source_hz, source_offset
        self.mask = target_cache['mask'].clone()
        self.original_mask = self.mask.clone()
        for index, sid in enumerate(self.ids):
            feature = features[sid]
            if not isinstance(feature, torch.Tensor) or feature.ndim != 3 or feature.shape[0] != 2 \
                    or feature.shape[1] < 1 or feature.shape[2] != FEATURE_DIM:
                raise ValueError(f'{sid}: expected Tensor[2,T>=1,1024]')
            finite_guard(sid, feature)
            end = self.offset + (feature.shape[1] - 1) / self.rate
            self.mask[index] &= (self.times >= self.offset - 1e-10) & (self.times <= end + 1e-10)

    def sequence(self, index, mel=False):
        if mel:
            return self.mel[index]
        source = self.features[self.ids[index]].detach().float().mean(0)
        result, _ = interpolate_sequence(source, self.times, self.rate, self.offset)
        return result

    def batches(self, indices, chunk_frames, mel=False):
        xs, ys, count = [], [], 0
        for index in indices:
            selected = self.mask[index]
            x, y = self.sequence(index, mel)[selected], self.targets[index][selected]
            for start in range(0, len(x), chunk_frames):
                piece_x, piece_y = x[start:start + chunk_frames], y[start:start + chunk_frames]
                if count and count + len(piece_x) > chunk_frames:
                    yield torch.cat(xs), torch.cat(ys)
                    xs, ys, count = [], [], 0
                xs.append(piece_x)
                ys.append(piece_y)
                count += len(piece_x)
        if count:
            yield torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def fit_ridge(batch_factory, dim, alpha, device):
    """Train-only population mean/std; float64 centered sufficient stats, unpenalized intercept."""
    sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros_like(sum_x)
    sum_y = torch.zeros(3, dtype=torch.float64, device=device)
    n = 0
    for x, y in batch_factory():
        x = x.to(device=device, dtype=torch.float64)
        y = y.to(device=device, dtype=torch.float64)
        sum_x += x.sum(0)
        sum_x2 += x.square().sum(0)
        sum_y += y.sum(0)
        n += len(x)
    if n < 2:
        raise ValueError('At least two valid training frames are required')
    mean, target_mean = sum_x / n, sum_y / n
    std = (sum_x2 / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    xtx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    xty = torch.zeros(dim, 3, dtype=torch.float64, device=device)
    for x, y in batch_factory():
        x = (x.to(device=device, dtype=torch.float64) - mean) / std
        y = y.to(device=device, dtype=torch.float64) - target_mean
        xtx += x.T @ x
        xty += x.T @ y
    weights = torch.linalg.solve(xtx + alpha * torch.eye(dim, dtype=torch.float64, device=device), xty)
    head = dict(mean=mean, std=std, weights=weights, intercept=target_mean, training_frames=n)
    for key in ('mean', 'std', 'weights', 'intercept'):
        finite_guard('ridge ' + key, head[key])
    return head


@torch.no_grad()
def predict(head, inputs, mel=False):
    predictions = torch.zeros_like(inputs.targets)
    device = head['weights'].device
    for index in range(len(inputs.ids)):
        selected = inputs.mask[index]
        x = inputs.sequence(index, mel)[selected].to(device=device, dtype=torch.float64)
        y = ((x - head['mean']) / head['std']) @ head['weights'] + head['intercept']
        predictions[index, selected] = y.float().cpu()
    finite_guard('predictions', predictions)
    return predictions


def pearson(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(), y - y.mean()
    xx, yy = x.square().sum(), y.square().sum()
    if len(x) < 2 or xx <= 1e-12 or yy <= 1e-12:
        return None
    return max(-1.0, min(1.0, (x @ y / (xx * yy).sqrt()).item()))


def metrics(predictions, targets, mask, indices):
    prediction, target, selected = predictions[indices], targets[indices], mask[indices]
    result = dict(samples=len(indices), samples_with_valid_frames=int(selected.any(1).sum()),
                  valid_frames=int(selected.sum()), per_target={})
    for k, name in enumerate(TARGET_NAMES):
        p, y = prediction[..., k][selected].double(), target[..., k][selected].double()
        if not len(y):
            raise ValueError('Evaluation split contains no valid frames')
        sst = (y - y.mean()).square().sum()
        correlations, target_constant, prediction_constant = [], 0, 0
        for row in range(len(indices)):
            clip_y, clip_p = target[row, selected[row], k].double(), prediction[row, selected[row], k].double()
            if len(clip_y) < 2:
                continue
            target_constant += int((clip_y - clip_y.mean()).square().sum() <= 1e-12)
            prediction_constant += int((clip_p - clip_p.mean()).square().sum() <= 1e-12)
            value = pearson(clip_p, clip_y)
            if value is not None:
                correlations.append(value)
        result['per_target'][name] = dict(
            pooled_R2=None if sst <= 1e-12 else (1 - (p - y).square().sum() / sst).item(),
            MAE=(p - y).abs().mean().item(), pooled_Pearson=pearson(p, y),
            per_clip_Pearson_macro=sum(correlations) / len(correlations) if correlations else None,
            per_clip_Pearson_n=len(correlations), target_constant_clips=target_constant,
            prediction_constant_clips=prediction_constant,
            clips_with_fewer_than_2_frames=int((selected.sum(1) < 2).sum()))
    return result


def temporal_controls(predictions, mask):
    mean_control, shift_control = torch.zeros_like(predictions), torch.zeros_like(predictions)
    shifts = []
    for index in range(len(predictions)):
        selected = mask[index]
        p = predictions[index, selected]
        shift = min(25, max(1, len(p) // 2)) if len(p) > 1 else 0
        if len(p):
            mean_control[index, selected] = p.mean(0)
            shift_control[index, selected] = p.roll(shift, dims=0)
        shifts.append(shift)
    return mean_control, shift_control, shifts


def load_inputs(args):
    if not Path(args.features).name.endswith('_features.pt') or any(char in args.features for char in '*?['):
        raise ValueError('--features must be one exact *_features.pt path, not a glob')
    cache = torch.load(args.features, map_location='cpu', weights_only=True)
    target_cache = torch.load(args.targets, map_location='cpu', weights_only=True)
    with open(args.split) as stream:
        split = json.load(stream)
    if not isinstance(cache, dict) or not isinstance(cache.get('features'), dict):
        raise ValueError('Expected features and metadata dictionaries')
    backend, rate, offset = validate_feature_metadata(cache.get('metadata'))
    if not isinstance(target_cache, dict) or not isinstance(target_cache.get('ids'), list):
        raise ValueError('Invalid target cache')
    ids = target_cache['ids']
    valid_ids(ids)
    if set(cache['features']) != set(ids):
        raise ValueError('Feature IDs and target IDs must be the exact same 2000 IDs')
    if not isinstance(split, dict) or not isinstance(split.get('train'), list) or not isinstance(split.get('val'), list):
        raise ValueError('Split must contain train and val ID lists')
    train, val = split['train'], split['val']
    if len(train) != N_TRAIN or len(val) != N_VAL or len(set(train + val)) != N_ITEMS or set(train + val) != set(ids):
        raise ValueError('Expected exact disjoint 1800/200 split, matching all cached IDs')
    for key, shape in (('targets', (N_ITEMS, 124, 3)), ('mel_features', (N_ITEMS, 124, N_MELS)),
                       ('mask', (N_ITEMS, 124)), ('times', (124,))):
        value = target_cache.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'Target {key} must have shape {shape}')
        finite_guard('target ' + key, value)
    if target_cache['mask'].dtype != torch.bool or not torch.allclose(target_cache['times'].double(), common_times(), atol=1e-10, rtol=0):
        raise ValueError('Expected bool mask and exact 0.04..4.96 second grid')
    tm = target_cache.get('metadata')
    if not isinstance(tm, dict) or tm.get('schema') != 'music_temporal_targets_v1' \
            or tm.get('sample_rate') != SR or tm.get('target_names') != list(TARGET_NAMES):
        raise ValueError('Target metadata schema/sample rate/target names mismatch')
    lookup = {sid: index for index, sid in enumerate(ids)}
    indices = dict(train=[lookup[sid] for sid in train], val=[lookup[sid] for sid in val])
    inputs = AlignedInputs(cache['features'], target_cache, rate, offset)
    return inputs, indices, cache['metadata'], target_cache['metadata'], backend


@torch.no_grad()
def run_probe(args):
    output = fresh_output(args.out)
    if not math.isfinite(args.alpha) or args.alpha <= 0 or args.chunk_frames < 1:
        raise ValueError('alpha must be finite and positive; chunk_frames must be positive')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU or CUDA devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    inputs, indices, feature_metadata, target_metadata, backend = load_inputs(args)
    print(f'probe {backend}: aligned frames train={int(inputs.mask[indices["train"]].sum())} '
          f'val={int(inputs.mask[indices["val"]].sum())}', flush=True)
    head = fit_ridge(lambda: inputs.batches(indices['train'], args.chunk_frames), FEATURE_DIM, args.alpha, device)
    predictions = predict(head, inputs)
    mean_control, shift_control, shifts = temporal_controls(predictions, inputs.mask)
    constant = head['intercept'].float().cpu().view(1, 1, 3).expand_as(inputs.targets)
    main_metrics = {name: metrics(predictions, inputs.targets, inputs.mask, index)
                    for name, index in indices.items()}
    constant_metrics = {name: metrics(constant, inputs.targets, inputs.mask, index)
                        for name, index in indices.items()}
    val = indices['val']
    controls = dict(
        clip_mean_broadcast=metrics(mean_control, inputs.targets, inputs.mask, val),
        within_clip_circular_shift=metrics(shift_control, inputs.targets, inputs.mask, val),
        same_trained_head=True,
        clip_mean_definition='Broadcast mean of valid aligned features before the same affine head; identical to mean predictions.',
        shift_definition='Circular +25 frames (1s) within each valid clip; short clips use min(25,n//2), n<2 use 0.',
        shift_frames_by_val_id={inputs.ids[index]: shifts[index] for index in val})
    del head, predictions, mean_control, shift_control
    mel_head = fit_ridge(lambda: inputs.batches(indices['train'], args.chunk_frames, mel=True), N_MELS, args.alpha, device)
    mel_predictions = predict(mel_head, inputs, mel=True)
    mel_metrics = {name: metrics(mel_predictions, inputs.targets, inputs.mask, index)
                   for name, index in indices.items()}
    report = dict(
        stage='frozen_music_temporal_readability_proxy', backend=backend, args=vars(args),
        input_sha256={key: sha256(getattr(args, key)) for key in ('features', 'targets', 'split')},
        feature_metadata=feature_metadata,
        target_metadata={key: value for key, value in target_metadata.items() if key != 'source_info'},
        samples=N_ITEMS, training_samples=N_TRAIN, validation_samples=N_VAL,
        valid_frames_by_split={name: int(inputs.mask[index].sum()) for name, index in indices.items()},
        valid_frames_per_clip={sid: int(inputs.mask[index].sum()) for index, sid in enumerate(inputs.ids)},
        target_valid_frames=int(inputs.original_mask.sum()), aligned_valid_frames=int(inputs.mask.sum()),
        frames_excluded_by_source_range=int((inputs.original_mask & ~inputs.mask).sum()),
        alignment='Stereo-mean hidden [T,1024], linear interpolation by exact metadata Hz/offset onto 25Hz elapsed seconds; no temporal global pooling or valid endpoint clamp.',
        mask='Real complete current+lag-1 audio windows intersect actual source offset..offset+(T-1)/Hz; no inferred encoder receptive field.',
        ridge=dict(alpha=args.alpha, validation_tuning=False, intercept_regularized=False,
                   normalization='Train-only per-dimension population mean/std; std min1e-6; no LayerNorm',
                   statistics_dtype='float64', fit='Centered XTX/XTY; solve(XTX+alpha*I,XTY), bias=train target mean',
                   feature_transfer='One backend CPU cache; align float32 and upload bounded frame chunks for float64 statistics'),
        trainable_parameters=dict(readout_weights=FEATURE_DIM * 3, intercept=3, total=FEATURE_DIM * 3 + 3,
                                  backbone=0, text=0, dit=0),
        backbone_updates=0, backbone_loaded=False, text_loaded=False, dit_loaded=False,
        metrics=main_metrics, train_mean_constant_baseline=constant_metrics, val_temporal_controls=controls,
        direct_acoustic_baseline=dict(input='64-dimensional logmel on the same valid 25Hz frames',
                                     independent_encoder=False, derived_from_target_audio=True,
                                     alpha=args.alpha, readout_parameters=N_MELS * 3 + 3, metrics=mel_metrics),
        metric_note='Pooled R2/MAE/Pearson use all valid frames; per-clip Pearson macro excludes either constant vector or <2 frames and reports n. Undefined R2/Pearson are null, never NaN.',
        weak_label_boundary=WEAK_LABEL_NOTE, finite=True, publication=False)
    atomic_write_new(output, report)
    print(json.dumps(dict(done='probe', backend=backend, out=str(output), metrics=main_metrics['val']), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    targets = subparsers.add_parser('targets')
    targets.add_argument('--caps', required=True)
    targets.add_argument('--out', required=True, help='Fresh derived PT path; existing paths are refused')
    probe = subparsers.add_parser('probe')
    probe.add_argument('--features', required=True, help='One exact *_features.pt cache')
    probe.add_argument('--targets', required=True)
    probe.add_argument('--split', required=True, help='Existing muq_linear_probe/split.json with 1800/200 IDs')
    probe.add_argument('--out', required=True, help='Fresh diagnostic JSON path; existing paths are refused')
    probe.add_argument('--alpha', type=float, default=10.0)
    probe.add_argument('--device', default='cuda')
    probe.add_argument('--chunk_frames', type=int, default=4096)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single-process probe only; no multi-rank torchrun')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.command == 'targets':
        build_targets(args)
    else:
        run_probe(args)


if __name__ == '__main__':
    main()
