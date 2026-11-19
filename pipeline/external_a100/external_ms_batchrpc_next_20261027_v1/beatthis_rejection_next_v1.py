"""CPU-safe rejection evidence; no model calls and no validation relaxation."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np

MAX_ARRAY_BYTES = 2 * 2**20
MODEL_SHA = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def once(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != path: raise ValueError('redirected diagnostic path')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.diagnostic-', delete=False) as f:
            temporary = Path(f.name); f.write(raw); f.flush(); os.fsync(f.fileno())
        try: os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != raw: raise ValueError('diagnostic write-once conflict')
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def predicate_failures(beats, downbeats, duration):
    failures = []
    for name, values in (('beats', beats), ('downbeats', downbeats)):
        if values.ndim != 1: failures.append(name + '.ndim_eq_1')
        if not np.isfinite(values).all(): failures.append(name + '.all_finite')
        if np.any(values < 0): failures.append(name + '.nonnegative')
        if np.any(values > duration + .02): failures.append(name + '.within_audio_duration_plus_0.02')
        if values.ndim and np.any(np.diff(values) <= 0): failures.append(name + '.strictly_increasing')
    if beats.ndim == downbeats.ndim == 1 and any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats):
        failures.append('downbeats.belongs_to_beats_atol_1e-6_rtol_0')
    return failures

class BeatThisRejected(ValueError):
    def __init__(self, paths, failures):
        super().__init__('Invalid actual BeatThis beat/downbeat output')
        self.diagnostic_paths, self.failed_predicates = paths, failures

def validate_actual(actual, duration, row, hashes, local_case):
    original = tuple(np.asarray(values) for values in actual)
    beats, downbeats = (np.asarray(values, dtype=float) for values in original)
    # This is byte-for-byte the previous rejection expression, not a new policy.
    invalid = any(values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > duration + 0.02) or np.any(np.diff(values) <= 0) for values in (beats, downbeats)) or any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats)
    if not invalid: return beats, downbeats
    failures = predicate_failures(beats, downbeats, duration)
    if any(values.dtype.hasobject for values in original):
        raise ValueError('Unsupported object-dtype BeatThis diagnostic; never pickle or inject objects')
    if sum(values.nbytes for values in original) > MAX_ARRAY_BYTES - 65536:
        raise ValueError('Rejected BeatThis diagnostic exceeds bounded array budget; source remains sealed')
    buffer = io.BytesIO(); np.savez(buffer, beats=original[0], downbeats=original[1]); raw = buffer.getvalue()
    if len(raw) > MAX_ARRAY_BYTES: raise ValueError('Rejected BeatThis NPZ exceeds bounded budget')
    directory = Path(local_case).parent.parent.parent / 'diagnostics' / row['split'] / row['id']
    array_path, evidence_path = directory / 'beatthis_rejected.npz', directory / 'beatthis_rejected.json'
    evidence = dict(schema='ms_beatthis_rejection_diagnostic_v1', source_row=row,
        actual_audio_duration_seconds=duration, audio_sha256=hashes['audio_path'], failed_predicates=failures,
        raw_arrays=dict(path='beatthis_rejected.npz', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        beats=dict(dtype=original[0].dtype.str, shape=list(original[0].shape)),
                        downbeats=dict(dtype=original[1].dtype.str, shape=list(original[1].shape))),
        method='beat_this.inference.File2Beats', model_sha256=MODEL_SHA,
        algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device='cuda:0', float16=False, dbn=False, vlm_called=False, quality_approved=False,
        evidence_only=True, original_rejection_policy_unchanged=True)
    once(array_path, raw); once(evidence_path, encoded(evidence))
    raise BeatThisRejected([str(array_path), str(evidence_path)], failures)
