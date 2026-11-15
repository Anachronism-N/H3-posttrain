"""Bar-aligned planning blocks from a fixed-parameter beat estimate.

The local expert window stays exactly 5.0625 s; what changes here is only how
expert windows are grouped into planning blocks. Beats come from librosa; the
meter is ASSUMED 4/4 and the downbeat phase is chosen by low-band accent.

Everything is recorded as an assumption, never as detected meter or a phrase.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "choreobench.bar_structure.v1"
PARAMS = dict(sample_rate=16000, hop_length=320, n_fft=2048, low_band_hz=150.0,
              assumed_beats_per_bar=4, phase_search=True, bars_per_block=4, max_unit_seconds=20.0)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(wav_path, bars_per_block=4, max_unit_seconds=20.0):
    import librosa
    import soundfile as sf
    path = Path(wav_path).resolve()
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] != 1 or rate != PARAMS["sample_rate"]:
        raise ValueError("Bar analysis requires the canonical 16 kHz mono waveform")
    signal = np.ascontiguousarray(audio[:, 0])
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=rate, hop_length=PARAMS["hop_length"],
                                                 units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=rate, hop_length=PARAMS["hop_length"])
    if len(beats) < 8:
        raise ValueError("Too few beats for a bar-aligned partition")
    # Low-band onset strength to pick the downbeat phase under an assumed 4/4.
    spec = np.abs(librosa.stft(signal, n_fft=PARAMS["n_fft"], hop_length=PARAMS["hop_length"]))
    freqs = librosa.fft_frequencies(sr=rate, n_fft=PARAMS["n_fft"])
    low = spec[freqs <= PARAMS["low_band_hz"]].sum(axis=0)
    accent = np.zeros_like(low)
    accent[1:] = np.maximum(0.0, np.diff(low))
    accent = np.convolve(accent, np.ones(3) / 3.0, mode="same")
    beats_per_bar = PARAMS["assumed_beats_per_bar"]
    phase_scores = []
    for phase in range(beats_per_bar):
        idx = beat_frames[phase::beats_per_bar]
        idx = idx[idx < len(accent)]
        phase_scores.append(float(accent[idx].mean()) if len(idx) else 0.0)
    phase = int(np.argmax(phase_scores))
    downbeats = beats[phase::beats_per_bar]
    intervals = np.diff(beats)
    bar_seconds = float(np.median(intervals) * beats_per_bar) if len(intervals) else None
    contrast = (max(phase_scores) / (sorted(phase_scores)[-2] or 1e-9)) if len(phase_scores) > 1 else 1.0
    return dict(
        schema=SCHEMA, detector="librosa-beat-track-plus-lowband-phase-v1",
        parameters=dict(PARAMS, bars_per_block=bars_per_block, max_unit_seconds=max_unit_seconds),
        audio=dict(path=str(path), sha256=_sha(path), sample_rate=rate,
                   num_samples=int(len(signal)), duration_seconds=round(len(signal) / rate, 3)),
        tempo_bpm=round(float(np.atleast_1d(tempo)[0]), 3),
        beats=[round(float(b), 4) for b in beats],
        downbeats=[round(float(d), 4) for d in downbeats],
        bar_seconds=round(bar_seconds, 3) if bar_seconds else None,
        meter_assumed="4/4", meter_detected=False,
        downbeat_phase=phase, phase_accent_scores=[round(s, 6) for s in phase_scores],
        phase_contrast=round(float(contrast), 3),
        note=("Beats are a librosa estimate; the meter is ASSUMED 4/4 and the downbeat phase is chosen by "
              "low-band accent. This is not detected meter, not a musical phrase and not ground truth."))


def blocks(structure, section_id, num_samples, sample_rate, bars_per_block=4, max_unit_seconds=20.0,
           min_unit_samples=None):
    """Disjoint bar-aligned planning units covering the whole song."""
    downbeats = [int(round(d * sample_rate)) for d in structure["downbeats"]]
    downbeats = [d for d in downbeats if 0 < d < num_samples]
    capacity = int(round(max_unit_seconds * sample_rate))
    floor = int(min_unit_samples if min_unit_samples is not None else round(0.5 * sample_rate))
    units, cursor, j = [], 0, 0
    if downbeats and downbeats[0] > 0:
        units.append(dict(unit_id="bu%04d" % len(units), section_id=section_id, start=0, end=downbeats[0],
                          kind="partial_fragment", bars=0, reasons=["audio_before_first_downbeat"]))
        cursor = downbeats[0]
    while j < len(downbeats):
        start = max(cursor, downbeats[j])
        chosen = []
        k = j
        while (k < len(downbeats)
               and len(chosen) < bars_per_block
               and (downbeats[k] - start) <= capacity
               and (k + 1 >= len(downbeats) or downbeats[k + 1] - start <= capacity)):
            chosen.append(k)
            k += 1
        if not chosen:
            break
        end = downbeats[chosen[-1] + 1] if chosen[-1] + 1 < len(downbeats) else num_samples
        if end > start:
            units.append(dict(unit_id="bu%04d" % len(units), section_id=section_id, start=start, end=end,
                              kind="bar_block" if len(chosen) == bars_per_block else "capacity_block",
                              bars=len(chosen), bar_indices=chosen))
        cursor = end
        j = chosen[-1] + 1
    if cursor < num_samples:
        units.append(dict(unit_id="bu%04d" % len(units), section_id=section_id, start=cursor, end=num_samples,
                          kind="partial_fragment", bars=0, reasons=["unclosed_tail"]))
    # Merge or extend anything shorter than the declared minimum action duration;
    # a planning unit that cannot hold one event is not emitted.
    merged = []
    for unit in units:
        if unit["end"] - unit["start"] < floor and merged:
            merged[-1]["end"] = unit["end"]
            merged[-1]["reasons"] = list(merged[-1].get("reasons", [])) + ["absorbed_short_fragment"]
            continue
        if unit["end"] - unit["start"] < floor and not merged:
            unit["start"] = 0
            if unit["end"] - unit["start"] < floor:
                continue
        merged.append(unit)
    units = merged
    if units and units[0]["start"] > 0:
        # Absorb any pre-downbeat prefix so coverage is exact and gapless.
        units[0]["start"] = 0
        units[0]["reasons"] = list(units[0].get("reasons", [])) + ["absorbed_prefix"]
    for index, unit in enumerate(units):
        unit["unit_id"] = "bu%04d" % index
    if not units or units[-1]["end"] - units[-1]["start"] < floor:
        if len(units) > 1:
            units[-2]["end"] = units[-1]["end"]
            units.pop()
    if not units:
        raise ValueError("Bar partition produced no unit long enough for one event")
    return dict(schema="choreobench.planning_units.v1", song_id=None, sample_rate=sample_rate,
                num_samples=num_samples, config=dict(bars_per_block=bars_per_block,
                                                     max_unit_seconds=max_unit_seconds,
                                                     unit_source="bar_aligned"),
                provenance=dict(algorithm="bar_aligned_blocks.v1", meter_assumed="4/4",
                                meter_detected=False, semantic_phrase_detection=False,
                                boundaries_are_action_boundaries=False,
                                detector=structure["detector"]),
                units=units)


def save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    return _sha(path)


def load(path):
    return json.loads(Path(path).read_text())
