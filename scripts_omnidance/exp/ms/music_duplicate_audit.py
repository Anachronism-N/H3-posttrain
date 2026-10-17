"""5s配乐重复候选审计；不是songID、匹配真值、检索评测或唯一编舞判定。"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F
from music_encoder_adapter import AUDIO_ROOT, load_waveform

SR, SECONDS, NFFT, HOP = 24000, 5.0625, 2048, 960
MEL_BANK = CHROMA_BANK = WINDOW = RESOLVED_AUDIO_ROOT = None


def finite(value, name):
    if not np.isfinite(np.asarray(value)).all():
        raise ValueError(f'Non-finite {name}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    with open(path) as stream:
        return json.load(stream, object_pairs_hook=unique)


def init_worker():
    global MEL_BANK, CHROMA_BANK, WINDOW, RESOLVED_AUDIO_ROOT
    RESOLVED_AUDIO_ROOT = AUDIO_ROOT.resolve(strict=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    WINDOW = torch.hann_window(NFFT)
    MEL_BANK = torchaudio.functional.melscale_fbanks(NFFT // 2 + 1, 0, SR / 2, 64, SR)
    frequencies = torch.fft.rfftfreq(NFFT, 1 / SR)
    valid = frequencies >= 27.5  # Never apply log2 to DC/zero frequencies.
    pitch = torch.round(69 + 12 * torch.log2(frequencies[valid] / 440)).long() % 12
    CHROMA_BANK = torch.zeros(NFFT // 2 + 1, 12)
    CHROMA_BANK[torch.where(valid)[0], pitch] = 1


def waveform_features(waveform, waveform_corr=False):
    """Only actual samples enter center=False STFT and frame RMS; no cache padding."""
    if waveform.ndim != 2 or waveform.shape[0] != 2 or waveform.shape[1] < NFFT:
        raise ValueError('Need stereo waveform with at least one full FFT window')
    finite(waveform.numpy(), 'waveform')
    mono = waveform.mean(0)
    power = torch.stft(mono, NFFT, HOP, window=WINDOW, center=False,
                       return_complex=True).abs().square().T
    logmel = (power @ MEL_BANK).clamp_min(1e-12).log()
    chroma = F.normalize(power @ CHROMA_BANK, dim=1, eps=1e-12)
    envelope = mono.unfold(0, NFFT, HOP).square().mean(1).sqrt()
    signature = torch.cat([v for x in (logmel, chroma, envelope[:, None])
                           for v in (x.mean(0), x.std(0, unbiased=False))])
    features = dict(logmel=logmel.numpy(), chroma=chroma.numpy(), envelope=envelope.numpy(),
                    signature=signature.numpy(), samples=waveform.shape[1])
    if waveform_corr:
        features['waveform'] = torchaudio.functional.resample(mono, SR, 4000).numpy()
    for key, value in features.items():
        finite(value, key)
    return features


def extract(item):
    sid, waveform_corr = item
    path = RESOLVED_AUDIO_ROOT / (sid + '.wav')
    if path.is_symlink() and path.resolve().parent != RESOLVED_AUDIO_ROOT:
        raise ValueError(f'Audio path escapes original root: {sid}')
    with sf.SoundFile(str(path)) as stream:
        source_sr, source_frames, channels = stream.samplerate, len(stream), stream.channels
    read_frames = min(source_frames, math.ceil(SECONDS * source_sr))
    samples = min(121500, math.ceil(read_frames * SR / source_sr))
    waveform = load_waveform(sid)[:, :samples].contiguous()
    raw = waveform.numpy().astype('<f4', copy=False)
    h = hashlib.sha256(b'float32-le|24000|' + json.dumps(list(raw.shape)).encode() + raw.tobytes())
    result = waveform_features(waveform, waveform_corr)
    result.update(id=sid, fingerprint=h.hexdigest(), audio_path=str(path), source_sr=source_sr,
                  source_channels=channels, source_frames=source_frames,
                  source_duration_seconds=source_frames / source_sr,
                  effective_duration_seconds=read_frames / source_sr)
    return result


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    # Constant/silent sequences cannot establish near-duplicate evidence.
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 1e-20 else 0.0


def score_pair(a, b):
    best = None
    for lag in range(-12, 13):
        ia, ib = max(0, -lag), max(0, lag)
        frames = min(len(a['chroma']) - ia, len(b['chroma']) - ib)
        overlap = min(a['samples'] - ia * HOP, b['samples'] - ib * HOP) / SR
        if overlap < 3 or frames < 75:
            continue
        ca, cb = a['chroma'][ia:ia + frames], b['chroma'][ib:ib + frames]
        cosine = float(np.clip(np.einsum('ij,ij->i', ca, cb).mean(), -1, 1))
        if best is None or (cosine, -abs(lag)) > (best['chroma_cosine'], -abs(best['lag_frames'])):
            best = dict(lag_frames=lag, lag_seconds=lag * HOP / SR,
                        overlap_seconds=overlap, overlap_frames=frames, chroma_cosine=cosine)
    if best is None:
        return dict(eligible=False, reason='Less than 3s real overlap or 75 full FFT frames')
    ia, ib, n = max(0, -best['lag_frames']), max(0, best['lag_frames']), best['overlap_frames']
    best.update(eligible=True, logmel_pearson=pearson(a['logmel'][ia:ia + n], b['logmel'][ib:ib + n]),
                envelope_pearson=pearson(a['envelope'][ia:ia + n], b['envelope'][ib:ib + n]))
    if 'waveform' in a and 'waveform' in b:
        wa, wb = a['waveform'][ia * 160:], b['waveform'][ib * 160:]
        length = min(len(wa), len(wb))
        best['waveform_normalizedcorr'] = pearson(wa[:length], wb[:length])
    return best


def suspected(row, chroma=0.98, mel=0.98, envelope=0.95):
    return (row['eligible'] and row['chroma_cosine'] >= chroma and
            row['logmel_pearson'] >= mel and row['envelope_pearson'] >= envelope)


def components(ids, train, edges):
    parent = {sid: sid for sid in ids}
    def root(sid):
        while sid != parent[sid]:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    for a, b in edges:
        a, b = root(a), root(b)
        parent[b] = a
    groups = {}
    for sid in ids:
        groups.setdefault(root(sid), []).append(sid)
    rows = [dict(ids=sorted(g), train_ids=sorted(set(g) & train), val_ids=sorted(set(g) - train))
            for g in groups.values() if len(g) > 1]
    rows.sort(key=lambda r: (-len(r['ids']), r['ids']))
    cross = [r for r in rows if r['train_ids'] and r['val_ids']]
    return dict(n_groups=len(rows), n_grouped_ids=sum(len(r['ids']) for r in rows), groups=rows,
                crosssplit_n_groups=len(cross), crosssplit_n_ids=sum(len(r['ids']) for r in cross),
                crosssplit_ids=sorted(sid for r in cross for sid in r['ids']), crosssplit_groups=cross)


def atomic_json(path, report):
    if os.path.lexists(path):
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create without replacing any existing output, including a racing writer.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--waveform_corr', action='store_true', help='Optional 4kHz mono correlation at selected lag')
    args = parser.parse_args()
    out = Path(args.out).absolute()
    if os.path.lexists(out):
        raise FileExistsError(out)
    if not out.parent.is_dir():
        raise FileNotFoundError(out.parent)
    if not 1 <= args.workers <= 64:
        raise ValueError('workers must be in [1,64]')
    hashes = {name: sha256(path) for name, path in (('caps', args.caps), ('split', args.split))}
    caps, split = read_json(args.caps), read_json(args.split)
    if not isinstance(caps, dict) or len(caps) < 4 or not isinstance(split, dict):
        raise ValueError('At least four caption IDs and an explicit split dictionary are required')
    for sid in caps:
        if not sid or '/' in sid or '\\' in sid or '\0' in sid or sid in ('.', '..'):
            raise ValueError(f'Invalid original clip ID: {sid!r}')
    for key in ('train', 'val'):
        values = split.get(key)
        if (not isinstance(values, list) or len(values) < 2 or
                any(not isinstance(s, str) for s in values) or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain at least two unique string IDs')
    train, val = set(split['train']), set(split['val'])
    if train & val or train | val != set(caps):
        raise ValueError('Split must partition exactly all caption IDs; no intersection-only audit')
    ids = sorted(caps)
    init_worker()
    records = {}
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker) as pool:
        for index, record in enumerate(pool.imap_unordered(extract, [(sid, args.waveform_corr) for sid in ids]), 1):
            records[record['id']] = record
            if index % 25 == 0 or index == len(ids):
                print(f'features {index}/{len(ids)}', flush=True)
    signatures = np.stack([records[sid]['signature'] for sid in ids]).astype(np.float64)
    train_indices = [i for i, sid in enumerate(ids) if sid in train]
    training = signatures[train_indices]
    mean, std = training.mean(0), np.maximum(training.std(0, ddof=1), 1e-4)
    normalized = (signatures - mean) / std
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    finite(normalized, 'train-only standardized signature')
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -2)
    candidates = set()
    for i, sid in enumerate(ids):
        for j in np.argsort(-similarities[i], kind='stable')[:3]:
            candidates.add(tuple(sorted((i, int(j)))))
        if sid in val:
            for j in sorted(train_indices, key=lambda j: (-similarities[i, j], j))[:5]:
                candidates.add(tuple(sorted((i, j))))
    by_hash = {}
    for sid in ids:
        by_hash.setdefault(records[sid]['fingerprint'], []).append(sid)
    exact = [dict(fingerprint=h, ids=g, n_pairs=len(g) * (len(g) - 1) // 2,
                  crosssplit_n_pairs=len(set(g) & train) * len(set(g) & val))
             for h, g in by_hash.items() if len(g) > 1]
    exact_edges = [(g['ids'][0], sid) for g in exact for sid in g['ids'][1:]]
    scores = []
    for index, (i, j) in enumerate(sorted(candidates), 1):
        row = dict(ids=[ids[i], ids[j]], signature_cosine=float(similarities[i, j]),
                   crosssplit=(ids[i] in train) != (ids[j] in train),
                   exact=records[ids[i]]['fingerprint'] == records[ids[j]]['fingerprint'],
                   **score_pair(records[ids[i]], records[ids[j]]))
        row['suspected'] = suspected(row)
        scores.append(row)
        if index % 250 == 0 or index == len(candidates):
            print(f'candidate scoring {index}/{len(candidates)}', flush=True)
    hits = [r for r in scores if r['suspected']]
    sensitivity = []
    for threshold in (0.95, 0.98, 0.995):
        selected = [r for r in scores if suspected(r, threshold, threshold, threshold)]
        graph = components(ids, train, exact_edges + [r['ids'] for r in selected])
        sensitivity.append(dict(all_three_threshold=threshold, suspected_n_pairs=len(selected),
                                crosssplit_n_pairs=sum(r['crosssplit'] for r in selected),
                                crosssplit_n_groups=graph['crosssplit_n_groups'],
                                crosssplit_n_ids=graph['crosssplit_n_ids']))
    scores.sort(key=lambda r: (r['suspected'], r.get('chroma_cosine', -2),
                               r.get('logmel_pearson', -2), r.get('envelope_pearson', -2)), reverse=True)
    report = dict(stage='music_duplicate_candidate_audit', input_sha256=hashes, args=vars(args),
                  n_ids=len(ids), n_train=len(train), n_val=len(val), audio_root=str(AUDIO_ROOT),
                  scope='Only first actual 5.0625s decoded at 24k stereo. Candidates/suspected pairs are not songID or truth. '
                        'Gate can miss duplicates; absence cannot establish no shared song. Different excerpts of one song remain unresolved; '
                        'Exact hashes do not normalize gain/channel variation; mono near-scoring may miss or collapse channel changes. '
                        'Silence lacks near-match evidence. '
                        'Require human songID; do not modify split/train or assign unique choreography per audio.',
                  method=dict(signature='64 logmel + 12 normalized chroma + mono RMS; temporal mean/population std; '
                              'train-only mean/std(ddof=1,min=1e-4), then L2; all-ID top3 union per-val top5 train',
                              fft=NFFT, hop=HOP, frame_hz=25, center=False, lag_grid_seconds=0.04,
                              max_abs_lag_seconds=0.48, minimum_overlap_seconds=3,
                              lag_selection='maximum mean frame chroma cosine; ties prefer smaller absolute lag',
                              pearson='flattened mean-centered logmel and centered RMS; constant signal returns 0',
                              exact='SHA256 of dtype/rate/shape and actual resampled float32 stereo bytes, no padded samples',
                              suspected_thresholds=dict(chroma=0.98, logmel=0.98, envelope=0.95)),
                  normalization=dict(train_n=len(train), mean=mean.tolist(), std=std.tolist()),
                  audio_metadata=[{k: v for k, v in records[sid].items()
                                   if k not in ('logmel', 'chroma', 'envelope', 'signature', 'waveform')} for sid in ids],
                  exact_groups=exact, exact_n_pairs=sum(g['n_pairs'] for g in exact),
                  exact_crosssplit_n_pairs=sum(g['crosssplit_n_pairs'] for g in exact),
                  candidate_n_pairs=len(scores), candidate_crosssplit_n_pairs=sum(r['crosssplit'] for r in scores),
                  ineligible_n_pairs=sum(not r['eligible'] for r in scores),
                  suspected_n_pairs=len(hits), suspected_crosssplit_n_pairs=sum(r['crosssplit'] for r in hits),
                  suspected_pairs=hits, candidate_top50=scores[:50], sensitivity=sensitivity,
                  exact_components=components(ids, train, exact_edges),
                  exact_or_suspected_components=components(ids, train, exact_edges + [r['ids'] for r in hits]))
    if hashes != {name: sha256(path) for name, path in (('caps', args.caps), ('split', args.split))}:
        raise RuntimeError('Input JSON changed during audit')
    atomic_json(out, report)
    print(f'DONE {out} exact_pairs={report["exact_n_pairs"]} suspected_pairs={len(hits)}', flush=True)


if __name__ == '__main__':
    main()
