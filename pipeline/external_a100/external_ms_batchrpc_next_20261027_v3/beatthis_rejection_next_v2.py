"""Transparent per-source audit tap; only the frozen core rejects model outputs."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np

MAX_ARRAY_BYTES = 2 * 2**20
MAX_JSON_BYTES = 64 * 2**10
MODEL_SHA = '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
CORE_SHA = '85ea6781a433a7f3cfe50fa074f56e7d7c291a815dfbf7b0220d4eb9caf8dda9'
ORIGINAL_POLICY = 'core303-304: ndim=1; finite; nonnegative; <=duration+0.02; strictly increasing; downbeats belong to beats atol=1e-6 rtol=0'
METHOD = 'beat_this.inference.File2Beats'

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def once(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != path: raise ValueError('redirected diagnostic path')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.diagnostic-', delete=False) as f:
            temporary = Path(f.name); f.write(raw); f.flush(); os.fsync(f.fileno())
        try: os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw: raise ValueError('diagnostic write-once conflict')
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

def audit_actual(actual, metadata, row, index, hashes, local_case):
    # Observe original arrays, never return converted/sorted/clipped arrays to core.
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise ValueError('unsupported output container; audit gap')
    original = tuple(np.asarray(values) for values in actual)
    if any(v.dtype.hasobject or v.dtype.kind not in 'biufc' for v in original):
        raise ValueError('unsupported object/non-numeric dtype; audit gap')
    if sum(v.nbytes for v in original) > MAX_ARRAY_BYTES - MAX_JSON_BYTES:
        raise ValueError('array budget exceeded; audit gap, not full-array coverage')
    duration = float(metadata['duration_seconds'])
    beats, downbeats = (np.asarray(v, dtype=float) for v in original)
    invalid = any(values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > duration + 0.02) or np.any(np.diff(values) <= 0) for values in (beats, downbeats)) or any(not np.isclose(beat, beats, atol=1e-6, rtol=0).any() for beat in downbeats)
    if not invalid: return {}
    failures = predicate_failures(beats, downbeats, duration)
    buffer = io.BytesIO(); np.savez(buffer, beats=original[0], downbeats=original[1]); raw = buffer.getvalue()
    if len(raw) > MAX_ARRAY_BYTES: raise ValueError('NPZ budget exceeded; audit gap')
    directory = Path(local_case).parent.parent.parent / 'diagnostics' / row['split'] / row['id']
    array_path, evidence_path = directory / 'beatthis_rejected.npz', directory / 'beatthis_rejected.json'
    evidence = dict(schema='ms_beatthis_rejection_diagnostic_v2', source_row=row, assignment_index=index,
        actual_audio_duration_seconds=duration, actual_audio_metadata=metadata, audio_sha256=hashes['audio_path'], failed_predicates=failures,
        raw_arrays=dict(path='beatthis_rejected.npz', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                        beats=dict(dtype=original[0].dtype.str, shape=list(original[0].shape)),
                        downbeats=dict(dtype=original[1].dtype.str, shape=list(original[1].shape))),
        method=METHOD, model_sha256=MODEL_SHA, algorithm_sha256=CORE_SHA,
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device='cuda:0', float16=False, dbn=False, vlm_called=False, quality_approved=False,
        evidence_only=True, original_rejection_policy_unchanged=True, original_rejection_policy=ORIGINAL_POLICY,
        array_coverage='complete_bounded_numeric_candidate')
    eraw = encoded(evidence)
    if len(eraw) > MAX_JSON_BYTES: raise ValueError('JSON budget exceeded; audit gap')
    once(array_path, raw); once(evidence_path, eraw)
    return dict(diagnostic_paths=[str(array_path), str(evidence_path)], failed_predicates=failures)

class AuditTap:
    def __init__(self, model, audio_info):
        self.model, self.audio_info_original = model, audio_info
        self.metadata = None; self.result = {}; self.gap = None
    def configure(self, row, index, hashes, local_case):
        self.row, self.index, self.hashes, self.local_case = row, index, hashes, local_case
        self.metadata = None; self.result = {}; self.gap = None
    def audio_info(self, path):
        self.metadata = self.audio_info_original(path)
        return self.metadata
    def __call__(self, path):
        actual = self.model(path)
        try: self.result = audit_actual(actual, self.metadata, self.row, self.index, self.hashes, self.local_case)
        except Exception as error:
            self.gap = dict(reason=str(error)[:512], array_coverage='gap_not_full_array_coverage',
                            max_npz_bytes=MAX_ARRAY_BYTES, max_json_bytes=MAX_JSON_BYTES)
        return actual
