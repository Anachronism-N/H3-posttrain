"""Public music-structure summary computed from the original audio only.

Nothing here reads local captions, reference choreography or reviews. Every
number is a fixed-parameter acoustic measurement on the canonical waveform, and
any functional interpretation (repeat, contrast, section type) is explicitly
marked as an unverified acoustic guess.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "choreobench.music_summary.v1"
CODE_VERSION = "music-summary-v1"
BLOCK_SECONDS = 10.0
PARAMS = dict(sample_rate=16000, block_seconds=BLOCK_SECONDS, n_fft=2048, hop_length=320,
              onset_delta=0.07, onset_wait_frames=3, chroma_bins=12,
              recurrence_min_similarity=0.97, recurrence_min_gap_blocks=2,
              contrast_max_similarity=0.45, max_relations_per_block=2, energy_rise_db=1.5)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _blocks(num_samples, rate):
    step = int(round(BLOCK_SECONDS * rate))
    return [(start, min(start + step, num_samples)) for start in range(0, num_samples, step)]


def analyze(wav_path):
    """Return a deterministic summary; no model, no randomness."""
    import librosa
    import soundfile as sf
    path = Path(wav_path).resolve()
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("Summary requires the canonical mono waveform")
    if rate != PARAMS["sample_rate"]:
        raise ValueError("Summary expects the 16 kHz canonical rate")
    signal = np.ascontiguousarray(audio[:, 0])
    chroma = librosa.feature.chroma_stft(y=signal, sr=rate, n_fft=PARAMS["n_fft"],
                                         hop_length=PARAMS["hop_length"])
    onset_env = librosa.onset.onset_strength(y=signal, sr=rate, hop_length=PARAMS["hop_length"])
    onsets = librosa.onset.onset_detect(y=signal, sr=rate, onset_envelope=onset_env,
                                        hop_length=PARAMS["hop_length"], delta=PARAMS["onset_delta"],
                                        wait=PARAMS["onset_wait_frames"])
    onset_times = librosa.frames_to_time(onsets, sr=rate, hop_length=PARAMS["hop_length"])
    centroid = librosa.feature.spectral_centroid(y=signal, sr=rate, n_fft=PARAMS["n_fft"],
                                                 hop_length=PARAMS["hop_length"])[0]
    frames_per_second = rate / PARAMS["hop_length"]
    blocks = []
    for index, (start, end) in enumerate(_blocks(len(signal), rate)):
        left, right = int(start / rate * frames_per_second), int(math.ceil(end / rate * frames_per_second))
        chunk = signal[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        energy_db = 20.0 * math.log10(max(rms, 1e-9))
        span = max((end - start) / rate, 1e-9)
        onsets_in = int(np.sum((onset_times >= start / rate) & (onset_times < end / rate)))
        blocks.append(dict(
            block_id="b%02d" % index, start=start, end=end,
            start_seconds=round(start / rate, 3), end_seconds=round(end / rate, 3),
            energy_db=round(energy_db, 2),
            onset_count=onsets_in, onset_rate_per_second=round(onsets_in / span, 3),
            spectral_centroid_hz=round(float(np.mean(centroid[left:right])) if right > left else 0.0, 1),
            chroma=np.asarray(chroma[:, left:right].mean(axis=1)) if right > left else np.zeros(12)))
    energies = np.array([b["energy_db"] for b in blocks])
    median = float(np.median(energies)) if energies.size else 0.0
    for index, block in enumerate(blocks):
        block["relative_energy_db"] = round(block["energy_db"] - median, 2)
        if index == 0:
            block["energy_trend"] = "unknown"
        else:
            delta = block["energy_db"] - blocks[index - 1]["energy_db"]
            block["energy_trend"] = ("rise" if delta >= PARAMS["energy_rise_db"]
                                     else "fall" if delta <= -PARAMS["energy_rise_db"] else "steady")
    vectors = [b.pop("chroma") for b in blocks]
    candidates = []
    for i in range(len(vectors)):
        for j in range(i + PARAMS["recurrence_min_gap_blocks"], len(vectors)):
            left, right = vectors[i], vectors[j]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarity = float(np.dot(left, right) / denominator) if denominator else 0.0
            if similarity >= PARAMS["recurrence_min_similarity"]:
                kind = "possible_recurrence"
            elif similarity <= PARAMS["contrast_max_similarity"]:
                kind = "possible_contrast"
            else:
                continue
            candidates.append(dict(block_a=blocks[i]["block_id"], block_b=blocks[j]["block_id"],
                                   chroma_cosine_similarity=round(similarity, 4), kind=kind,
                                   interpretation="acoustic_similarity_only_not_a_section_label"))
    # Keep only the strongest few per block so the summary stays readable; a long
    # list of near-identical chroma pairs is not extra evidence.
    limit = PARAMS["max_relations_per_block"]
    kept = []
    for kind, reverse in (("possible_recurrence", True), ("possible_contrast", False)):
        per_block, seen = {}, set()
        for item in sorted((c for c in candidates if c["kind"] == kind),
                           key=lambda c: c["chroma_cosine_similarity"], reverse=reverse):
            if item["block_a"] in seen or item["block_b"] in seen:
                continue
            count = per_block.get(item["block_a"], 0)
            if count >= limit:
                continue
            per_block[item["block_a"]] = count + 1
            seen.update((item["block_a"], item["block_b"]))
            kept.append(item)
    relations = kept
    return dict(
        schema=SCHEMA, code_version=CODE_VERSION, parameters=dict(PARAMS),
        audio=dict(path=str(path), sha256=_sha(path), sample_rate=rate, num_samples=int(len(signal)),
                   duration_seconds=round(len(signal) / rate, 3)),
        blocks=blocks, relations=relations,
        functional_labels_verified=False, beat_or_meter_assumed=False,
        note=("Energy/onset/timbre/chroma are fixed-parameter acoustic measurements of this waveform. "
              "possible_recurrence/possible_contrast are chroma-similarity guesses, NOT detected chorus, "
              "verse, phrase, downbeat, meter or any functional section. No beat grid is assumed."))


def save(summary, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    return _sha(out_path)


def load(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError("Unknown music summary schema")
    return value


def direction(summary, start, end, rise_db=1.5):
    """Deterministic movement-amount direction for a time span (never model output)."""
    blocks = [b for b in summary["blocks"] if b["start"] < end and b["end"] > start]
    if not blocks:
        return "unknown"
    mean = sum(b["relative_energy_db"] for b in blocks) / len(blocks)
    if mean >= rise_db:
        return "increase"
    if mean <= -rise_db:
        return "decrease"
    return "steady"


def action_profile(direction):
    """Concrete, checkable shape of a block's movement amount."""
    return {
        "increase": dict(event_count="2 to 3", mean_duration="shorter than the previous block",
                         guidance="add a second concurrent track or split the block into two phrases"),
        "decrease": dict(event_count="1", mean_duration="at least half of the block",
                         guidance="one sustained phrase; do not add filler events"),
        "steady": dict(event_count="1 to 2", mean_duration="similar to the previous block",
                       guidance="keep the same amount of movement"),
        "unknown": dict(event_count="1 to 2", mean_duration="unconstrained",
                        guidance="no acoustic evidence; keep it simple"),
    }[direction]


def blocks_overlapping(summary, start, end):
    return [b for b in summary["blocks"] if b["start"] < end and b["end"] > start]


def compact(summary, start, end):
    """Only the blocks/relations that overlap a planning unit."""
    blocks = blocks_overlapping(summary, start, end)
    ids = {b["block_id"] for b in blocks}
    return dict(
        overlapping_blocks=[{k: b[k] for k in ("block_id", "start_seconds", "end_seconds", "relative_energy_db",
                                              "onset_rate_per_second", "spectral_centroid_hz", "energy_trend")}
                            for b in blocks],
        relations=[r for r in summary["relations"] if r["block_a"] in ids or r["block_b"] in ids],
        functional_labels_verified=False, beat_or_meter_assumed=False,
        note="Acoustic measurements and chroma-similarity guesses only; not chorus/verse/phrase/beat detection.")
