"""Fixed 57-video counterfactual diagnostics, not musical-style truth or human quality."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, find_peaks, resample_poly

from music_generation_validate import TAGS, atomic_write_new, fixed_cases, sha256
from dwpose_full_extract import MAX_SECONDS, atomic_write, extract_clip, load_wholebody, validate_cache

DEFAULT_EXP_ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp')
WINDOW = (1., 5.)
GROUPS = {'body': (0, 18), 'arms': (2, 8), 'legs': (8, 14), 'hands': (92, 134)}
METRICS = ('precision', 'recall', 'f1', 'bas_music', 'bas_motion')
CACHE_VERSION = 1


def cases(exp_root, tags=TAGS):
    if len(set(tags)) != len(tags) or any(re.fullmatch(r'[A-Za-z0-9_-]+', tag) is None or tag in ('native_cleanpad', 'GT') for tag in tags):
        raise ValueError('Tags must be distinct bare generated configuration names')
    fixed = fixed_cases(exp_root)
    with (exp_root / 'E2/full_joint.csv').open(newline='') as stream:
        joint = {row['sample_id']: row for row in list(csv.DictReader(stream))[:3]}
    original = {c['sample_id']: c for c in fixed if c['condition'] == 'joint'}
    swapped = {c['sample_id']: c for c in fixed if c['condition'] == 'aswap'}
    if set(original) != set(swapped) or set(original) != set(joint):
        raise ValueError('Fixed joint/aswap/GT IDs must match exactly')
    result = []
    for tag in (*tags, 'native_cleanpad', 'GT'):
        for case in (fixed[:3] if tag == 'GT' else fixed):
            sid, condition = case['sample_id'], case['condition']
            correct = case['audio_path']
            alt = (swapped if condition == 'joint' else original)[sid]['audio_path']
            if correct == alt:
                raise ValueError('Counterfactual audio must differ')
            root = 'val_nativeclip_base_cleanpad' if tag == 'native_cleanpad' else f'val_musicreplace_{tag}'
            video = Path(joint[sid]['video']) if tag == 'GT' else exp_root / 'H3AR' / root / condition / case['filename']
            key = f'{tag}/{condition}/{sid}/{Path(correct).stem}'
            result.append(dict(tag=tag, condition=condition, sample_id=sid, key=key,
                               video_path=str(video), correct_audio=correct, alt_audio=alt))
    expected = len(tags) * 6 + 9
    if len(result) != expected or len({c['key'] for c in result}) != expected:
        raise ValueError(f'Expected exactly {expected} unique fixed-case configuration videos')
    return result


def cache_path(out, case):
    return out / (hashlib.sha256(case['key'].encode()).hexdigest() + '.npz')


def source_fingerprint(case):
    import decord
    path = Path(case['video_path'])
    before = path.stat()
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1, fault_tol=0)
    fps, total = float(reader.get_avg_fps()), len(reader)
    if not math.isfinite(fps) or fps <= 0 or total / fps < MAX_SECONDS:
        raise ValueError(f'{path}: incomplete fixed 5.0625s source range')
    h, w = reader[0].shape[:2]
    fingerprint = dict(version=CACHE_VERSION, key=case['key'], source=str(path.resolve()),
                       sha256=sha256(path), native_fps=fps, total_frames=total,
                       frames=int(math.floor(MAX_SECONDS * fps + 1e-6)), h=h, w=w)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'{path}: source changed during fingerprinting')
    return fingerprint


def load_verified(path, case, fingerprint):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(data.pop('source_fingerprint')))
    quality = json.loads(str(data.pop('quality_json')))
    if metadata != fingerprint:
        raise ValueError(f'{path}: existing fingerprint mismatch; refusing overwrite')
    validate_cache(data, case['key'])
    if (float(data['fps']) != fingerprint['native_fps'] or len(data['xy']) != fingerprint['frames']
            or int(data['h']) != fingerprint['h'] or int(data['w']) != fingerprint['w']):
        raise ValueError(f'{path}: cache does not contain complete native frame range')
    return data, quality


def quality_rois(case, data):
    import cv2
    import decord
    cv2.setNumThreads(1)
    reader = decord.VideoReader(case['video_path'], ctx=decord.cpu(0), num_threads=1, fault_tol=0)
    indices = np.flatnonzero((np.arange(len(data['xy'])) / float(data['fps']) >= 1)
                            & (np.arange(len(data['xy'])) / float(data['fps']) < 5))
    summaries = {}
    for name, (start, end) in {'face': (24, 92), 'left_hand': (92, 113), 'right_hand': (113, 134)}.items():
        values, confidence = [], []
        roi_valid = []
        for t in indices:
            valid = data['valid'][t, start:end]
            confidence.append(float(valid.mean()))
            points = data['xy'][t, start:end][valid]
            usable = len(points) >= math.ceil(.6 * (end - start))
            if usable:
                low, high = points.min(0), points.max(0)
                pad = .1 * (high - low)
                low = np.maximum(0, np.floor(low - pad)).astype(int)
                high = np.minimum([int(data['w']), int(data['h'])], np.ceil(high + pad)).astype(int)
                usable = bool(np.min(high - low) >= 12)
            roi_valid.append(usable)
            if not usable:
                continue
            crop = reader[int(t)].asnumpy()[low[1]:high[1], low[0]:high[0]]
            scale = 128 / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        summaries[name] = dict(roi_frames=len(values), window_frames=len(indices),
                               roi_coverage=float(np.mean(roi_valid)) if len(indices) else 0.,
                               laplacian_variance_mean=float(np.mean(values)) if values else None,
                               temporal_valid_fraction_mean=float(np.mean(confidence)) if confidence else None,
                               temporal_valid_fraction_std=float(np.std(confidence)) if confidence else None)
    return dict(parts=summaries, note='ROI sharpness/score-coverage proxies only; not quality verdicts. '
                'Pixel crops use valid joints, 10% padding, min side 12px, long side resized to 128px.')


def extract(all_cases, out, shard, nshard):
    model = None
    for case in all_cases[shard::nshard]:
        fingerprint = source_fingerprint(case)
        path = cache_path(out, case)
        if path.exists():
            load_verified(path, case, fingerprint)
            print(f"[validated] {case['key']} source_sha256={fingerprint['sha256']}", flush=True)
            continue
        lock = path.with_suffix('.lock')
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if path.exists():
                raise FileExistsError(f'Refusing concurrent existing cache: {path}')
            if model is None:
                model = load_wholebody(1)
                for session in (model.session_det, model.session_pose):
                    session.disable_fallback()
            data = extract_clip(Path(case['video_path']), model, case['key'], 1)
            validate_cache(data, case['key'])
            quality = quality_rois(case, data)
            if source_fingerprint(case) != fingerprint:
                raise ValueError('Source changed during extraction')
            data['source_fingerprint'] = np.array(json.dumps(fingerprint, sort_keys=True))
            data['quality_json'] = np.array(json.dumps(quality, allow_nan=False))
            atomic_write(path, lambda stream: np.savez_compressed(stream, **data))
            load_verified(path, case, fingerprint)
            print(f"[cached] {case['key']} source_sha256={fingerprint['sha256']}", flush=True)
        finally:
            os.close(descriptor)
            lock.unlink()


def segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def motion_series(data, start, end):
    xy, valid = data['xy'][:, start:end], data['valid'][:, start:end]
    fps, h = float(data['fps']), int(data['h'])
    times = (np.arange(len(xy) - 1) + .5) / fps
    pairs = valid[:-1] & valid[1:] & np.isfinite(xy[:-1]).all(-1) & np.isfinite(xy[1:]).all(-1)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps / h
    usable = pairs.mean(1) >= .6
    mean = np.divide(np.where(pairs, speed, 0).sum(1), pairs.sum(1),
                     out=np.full(len(speed), np.nan), where=usable & (pairs.sum(1) > 0))
    window = (times >= WINDOW[0]) & (times < WINDOW[1])
    times, mean, pairs = times[window], mean[window], pairs[window]
    for low, high in segments(np.isfinite(mean)):
        mean[low:high] = gaussian_filter1d(mean[low:high], .04 * fps, mode='nearest')
    return times, mean, pairs


def motion_events(times, speed, fps):
    events = []
    order, distance = max(1, math.ceil(.08 * fps)), max(1, math.ceil(.2 * fps))
    finite = speed[np.isfinite(speed)]
    prominence = .1 * np.std(finite) if len(finite) else 0.
    for low, high in segments(np.isfinite(speed)):
        part = speed[low:high]
        if len(part) < 2 * order + 1 or np.std(part) <= 1e-12:
            continue
        local = argrelextrema(part, np.less_equal, order=order)[0]
        prominent = find_peaks(-part, prominence=prominence, distance=distance)[0]
        events.extend((np.intersect1d(local, prominent) + low).tolist())
    selected = []
    for index in sorted(events, key=lambda i: speed[i]):
        if all(abs(times[index] - times[other]) >= .2 for other in selected):
            selected.append(index)
    return times[sorted(selected)]


def alignment(events, beats, adequate=True):
    if not adequate or len(events) < 2 or len(beats) < 2:
        return dict.fromkeys(METRICS)
    distance = np.abs(beats[:, None] - events[None, :])
    precision = float(np.mean(distance.min(0) <= .1))
    recall = float(np.mean(distance.min(1) <= .1))
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                bas_music=float(np.exp(-distance.min(1) ** 2 / (2 * .1 ** 2)).mean()),
                bas_motion=float(np.exp(-distance.min(0) ** 2 / (2 * .1 ** 2)).mean()))


def phase_summary(events, beats, adequate):
    rows = [alignment(events, WINDOW[0] + (beats - WINDOW[0] + 4 * k / 33) % 4, adequate)
            for k in range(1, 33)]
    return {name: dict(mean=float(np.mean(values)), std=float(np.std(values)),
                       p95=float(np.quantile(values, .95))) if (values := [r[name] for r in rows if r[name] is not None])
            else dict.fromkeys(('mean', 'std', 'p95')) for name in METRICS}


def logmel_flux(samples, sr=24000):
    nfft, hop = 1024, 240
    padded = np.pad(samples, (nfft // 2, nfft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    hz = 700 * (10 ** (np.linspace(mel(0), mel(sr / 2), 66) / 2595) - 1)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    filters = np.maximum(0, np.minimum((freqs[None, :] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None],
                                      (hz[2:, None] - freqs[None, :]) / (hz[2:] - hz[1:-1])[:, None]))
    logged = np.log1p(power @ filters.T)
    flux = np.maximum(0, np.diff(logged, axis=0)).mean(1)
    return (np.arange(len(flux)) + .5) * hop / sr, flux


def audio_features(path, beats):
    import soundfile as sf
    with sf.SoundFile(path) as stream:
        sr = stream.samplerate
        samples = stream.read(math.ceil(MAX_SECONDS * sr), dtype='float64', always_2d=True).mean(1)
    if len(samples) / sr < MAX_SECONDS:
        raise ValueError(f'{path}: source audio shorter than fixed range')
    divisor = math.gcd(sr, 24000)
    times, flux = logmel_flux(resample_poly(samples, 24000 // divisor, sr // divisor))
    selected = np.asarray(beats[path], dtype=float)
    if selected.ndim != 1 or not np.isfinite(selected).all() or np.any(np.diff(selected) <= 0):
        raise ValueError(f'{path}: invalid DP beat proxy')
    selected = selected[(selected >= 1) & (selected < 5)]
    return dict(times=times, flux=flux, beats=selected, sha256=sha256(Path(path)),
                silent=bool(np.max(np.abs(samples)) < 1e-8))


def correlation(times, values, audio, adequate):
    sampled = np.interp(times, audio['times'], audio['flux'], left=np.nan, right=np.nan)
    valid = np.isfinite(values) & np.isfinite(sampled)
    if not adequate or audio['silent'] or valid.sum() < 3:
        return None
    x, y = values[valid], sampled[valid]
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else None


def delta(left, right):
    return left - right if left is not None and right is not None else None


def score_group(data, group, correct, alt):
    times, speed, pairs = motion_series(data, *GROUPS[group])
    coverage = float(np.isfinite(speed).mean()) if len(speed) else 0.
    events = motion_events(times, speed, float(data['fps']))
    adequate = coverage >= .7 and len(events) >= 2
    scores = {label: alignment(events, audio['beats'], adequate and not audio['silent'])
              for label, audio in (('correct', correct), ('alt', alt))}
    phase = phase_summary(events, correct['beats'], adequate and not correct['silent'])
    acceleration = np.diff(speed) * float(data['fps'])
    acceleration_times = (times[:-1] + times[1:]) / 2
    onset = {label: dict(speed=correlation(times, speed, audio, adequate),
                        acceleration=correlation(acceleration_times, acceleration, audio, adequate))
             for label, audio in (('correct', correct), ('alt', alt))}
    return dict(events=events.tolist(), event_count=len(events), valid_pair_coverage=float(pairs.mean()) if pairs.size else 0.,
                usable_speed_coverage=coverage, inadequate_reason=('coverage_below_0.7' if coverage < .7 else
                'fewer_than_2_motion_events' if len(events) < 2 else None),
                correct_beats=len(correct['beats']), alt_beats=len(alt['beats']), **scores, phase=phase,
                margin={m: delta(scores['correct'][m], scores['alt'][m]) for m in METRICS},
                correct_minus_phase={m: delta(scores['correct'][m], phase[m]['mean']) for m in METRICS},
                onset_0lag=onset, onset_margin={m: delta(onset['correct'][m], onset['alt'][m]) for m in ('speed', 'acceleration')})


def score(all_cases, out, exp_root):
    tags = tuple(dict.fromkeys(case['tag'] for case in all_cases if case['tag'] not in ('native_cleanpad', 'GT')))
    expected = len(all_cases)
    available, missing, manifest = {}, [], []
    for case in all_cases:
        fingerprint, path = source_fingerprint(case), cache_path(out, case)
        entry = dict(**case, source=fingerprint, cache_path=str(path.absolute()))
        if not os.path.lexists(path):
            missing.append(dict(**entry, status='missing_cache'))
        else:
            available[case['key']] = load_verified(path, case, fingerprint)
        manifest.append(entry)
    if missing:
        return dict(schema_version=1, expected_videos=expected, complete=False, cache_complete=False,
                    validated_caches=len(available), missing=missing, exp_root=str(exp_root),
                    cases=[dict(**entry, status=('missing_cache' if entry['key'] not in available else
                           'not_scored_incomplete'), groups=None) for entry in manifest], configurations=None,
                    scope='Incomplete cache preflight: no scores computed; existing corrupt/hash-mismatched caches are hard errors.')
    beats_path = exp_root / 'beats.json'
    beats = json.loads(beats_path.read_text())
    audio = {path: audio_features(path, beats) for path in
             sorted({c[field] for c in all_cases for field in ('correct_audio', 'alt_audio')})}
    rows = []
    for entry in manifest:
        data, quality = available[entry['key']]
        groups = {g: score_group(data, g, audio[entry['correct_audio']], audio[entry['alt_audio']]) for g in GROUPS}
        rows.append(dict(**entry, status='scored', quality_proxies=quality, groups=groups))
    native = {(r['condition'], r['sample_id']): r for r in rows if r['tag'] == 'native_cleanpad'}
    for row in rows:
        if row['tag'] in ('GT', 'native_cleanpad'):
            continue
        base = native[row['condition'], row['sample_id']]
        row['increment_over_native_pair'] = {g: {m: delta(row['groups'][g]['margin'][m], base['groups'][g]['margin'][m])
                                                   for m in METRICS} for g in GROUPS}
    summaries = {}
    for tag in (*tags, 'native_cleanpad', 'GT'):
        selected = [r for r in rows if r['tag'] == tag]
        summary = {}
        for g in GROUPS:
            summary[g] = {}
            for field in ('margin', 'correct_minus_phase', 'increment_over_native_pair'):
                for m in METRICS:
                    values = [(r.get(field, {}).get(g, {}).get(m) if field == 'increment_over_native_pair'
                               else r['groups'][g][field][m]) for r in selected]
                    values = [v for v in values if v is not None]
                    summary[g][f'{field}_{m}'] = dict(mean=float(np.mean(values)) if values else None, valid_cases=len(values))
        by_condition = {}
        for condition in ('joint', 'aswap'):
            subset = [r for r in selected if r['condition'] == condition]
            if not subset:
                continue
            by_condition[condition] = {'video_cases': len(subset), 'groups': {}}
            for g in GROUPS:
                fields = {}
                for field in ('correct', 'alt', 'margin', 'correct_minus_phase', 'onset_margin'):
                    for m in (('speed', 'acceleration') if field == 'onset_margin' else METRICS):
                        values = [r['groups'][g][field][m] for r in subset if r['groups'][g][field][m] is not None]
                        fields[f'{field}_{m}'] = dict(mean=float(np.mean(values)) if values else None, valid_cases=len(values))
                by_condition[condition]['groups'][g] = fields
        summaries[tag] = dict(video_cases=len(selected), groups=summary, by_condition=by_condition)
    gt = [dict(sample_id=r['sample_id'], group=g, f1_margin=r['groups'][g]['margin']['f1'])
          for r in rows if r['tag'] == 'GT' for g in GROUPS
          if r['groups'][g]['margin']['f1'] is None or r['groups'][g]['margin']['f1'] <= 0]
    return dict(schema_version=1, expected_videos=expected, complete=True, cache_complete=True, validated_caches=expected,
                missing=[], exp_root=str(exp_root), cases=rows, configurations=summaries,
                valid_body_f1_cases=sum(r['groups']['body']['correct']['f1'] is not None for r in rows),
                audio_sources={p: dict(sha256=a['sha256'], beat_count=len(a['beats']), silent=a['silent']) for p, a in audio.items()},
                beats_proxy=dict(path=str(beats_path), sha256=sha256(beats_path), method='existing librosa DP proxy; NOT GT beat labels'),
                gt_calibration=dict(low_or_null_margin=gt, note='GT low/negative margins expose metric calibration limits, not GT dance failure.'),
                protocol=dict(window=list(WINDOW), phase_shifts=32, phase_fractions=[k / 33 for k in range(1, 33)],
                              event_tolerance_s=.1, bas_sigma_s=.1, smoothing_sigma_s=.04, event_order_s=.08,
                              event_prominence_std=.1, event_distance_s=.2, group_pair_min=.6, coverage_min=.7,
                              audio_sr=24000, nfft=1024, hop=240, onset='positive logmel flux, zero-lag only',
                              bas_music='mean over music beats of exp(-nearest motion distance^2/(2*sigma^2))',
                              bas_motion='traditional BAS: mean over motion events of exp(-nearest beat distance^2/(2*sigma^2))',
                              completeness='complete means all selected source/cache fingerprints validated, NOT metric/human acceptance'),
                scope='Only three fixed source cases per condition, paired deltas and configuration means; '
                      'Configuration videos are NOT IID. No significance claim, metric tuning, max-lag selection, '
                      'music-style truth, or automatic human-quality verdict.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('extract', 'score'), required=True)
    parser.add_argument('--exp_root', type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--out_json', type=Path)
    parser.add_argument('--tags', nargs='+', default=list(TAGS), help='Generated configurations, always the same three source cases plus native/GT controls')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    args = parser.parse_args(argv)
    if args.nshard < 1 or not 0 <= args.shard < args.nshard:
        parser.error('Require nshard > 0 and 0 <= shard < nshard')
    if not args.out.parent.is_dir() or not args.exp_root.is_dir():
        parser.error('Experiment root and output parent must already exist')
    if args.stage == 'score' and (args.out_json is None or args.shard != 0 or args.nshard != 1):
        parser.error('score requires --out_json and is unsharded')
    if args.out_json is not None and (not args.out_json.parent.is_dir() or os.path.lexists(args.out_json)):
        parser.error('out_json parent must exist; refusing any existing output')
    if args.stage == 'extract' and args.out_json is not None:
        parser.error('--out_json is only valid for score')
    try:
        selected = cases(args.exp_root.absolute(), args.tags)
        if args.stage == 'extract':
            args.out.mkdir(exist_ok=True)
            extract(selected, args.out, args.shard, args.nshard)
        else:
            result = score(selected, args.out, args.exp_root.absolute())
            atomic_write_new(args.out_json, result)
            print(json.dumps(dict(videos=len(selected), complete=result['complete'], validated_caches=result['validated_caches'],
                                  missing=len(result['missing']), out_json=str(args.out_json))), flush=True)
            return 0 if result['complete'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
