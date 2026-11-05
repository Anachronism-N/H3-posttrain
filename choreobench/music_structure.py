"""Offline public beat/novelty evidence, not verse/chorus or semantic phrase detection."""
import argparse
import copy
import json
from pathlib import Path

from .dataset import resolve_audio, verify_dataset_audio
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .schema import validate_dataset, validate_song


NOVELTY_CONFIG = dict(context_seconds=3, min_section_seconds=20, minimum_cosine_novelty=0.25,
                      novelty_quantile=0.8, bands=32, nperseg=1024, noverlap=512,
                      feature_normalization="songwise-log-band-standardization", std_floor=0.05)


def analyze_structure(dataset_path, detections_path, output):
    """Produce a versioned public sidecar without editing or rehashing the candidate bank."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import spectrogram, find_peaks
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    output = Path(output)
    evidence_path = output.with_name(output.stem + ".analysis.json")
    if output.exists() or evidence_path.exists():
        raise FileExistsError("Use new public structure output and evidence paths")
    detection_sha = file_sha(detections_path)
    raw = Path(detections_path).read_text(encoding="utf-8").splitlines()
    rows = {}
    for line in raw:
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in rows:
            raise ValueError("Invalid/duplicate beat detection record")
        rows[row["id"]] = row
    evidence, public = [], []
    for original in dataset["songs"]:
        row = rows.get(original["song_id"])
        if row is None or row.get("status") != "ok" or row.get("audio_sha256") != original["audio"]["sha256"]:
            raise ValueError("Beat predictions must bind the exact successful canonical waveform")
        path = resolve_audio(original, dataset_path)
        samples, rate = sf.read(path, dtype="float64")
        if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) != original["audio"]["num_samples"]:
            raise ValueError("Canonical analysis audio is not finite mono with the original clock")
        if rate != original["audio"]["sample_rate"]:
            raise ValueError("Canonical sample rate differs")
        def to_samples(values):
            if not isinstance(values, list) or any(type(t) not in (int, float) or not np.isfinite(t) for t in values):
                raise ValueError("Beat tool times must be finite seconds")
            return [round(t * rate) for t in values]
        beats, downbeats = to_samples(row["beats"]), to_samples(row["downbeats"])
        freq, times, power = spectrogram(samples, fs=rate, nperseg=NOVELTY_CONFIG["nperseg"],
                                         noverlap=NOVELTY_CONFIG["noverlap"], mode="psd")
        edges = np.geomspace(30, rate / 2, NOVELTY_CONFIG["bands"] + 1)
        bands = np.stack([np.log1p(power[(freq >= lo) & (freq < hi)].sum(axis=0) * 1e6)
                          for lo, hi in zip(edges, edges[1:])])
        bands = (bands - bands.mean(axis=1, keepdims=True)) / np.maximum(
            bands.std(axis=1, keepdims=True), NOVELTY_CONFIG["std_floor"])
        context = NOVELTY_CONFIG["context_seconds"]
        positions, scores = [], []
        for sample in downbeats:
            seconds = sample / rate
            if seconds < context or seconds > len(samples) / rate - context:
                continue
            left = bands[:, (times >= seconds - context) & (times < seconds)].mean(axis=1)
            right = bands[:, (times >= seconds) & (times < seconds + context)].mean(axis=1)
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            score = 1 - float(left @ right) / denominator if denominator > 1e-12 else 0.0
            positions.append(sample)
            scores.append(max(0.0, min(2.0, score)))
        threshold = max(NOVELTY_CONFIG["minimum_cosine_novelty"],
                        float(np.quantile(scores, NOVELTY_CONFIG["novelty_quantile"])) if scores else 0)
        peaks, _ = find_peaks(scores, height=threshold) if scores else ([], {})
        selected = []
        margin = round(NOVELTY_CONFIG["min_section_seconds"] * rate)
        for i in sorted(peaks, key=lambda n: (-scores[n], positions[n])):
            sample = positions[i]
            if margin <= sample <= len(samples) - margin and all(abs(sample - s) >= margin for s in selected):
                selected.append(sample)
        boundaries = [0] + sorted(selected) + [len(samples)]
        music = dict(source="beatthis+spectral-novelty-v1:" + detection_sha, beats=beats, downbeats=downbeats,
            sections=[dict(section_id="ns%03d" % i, start=a, end=b, label="unknown", repeat_group=None)
                      for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))])
        effective = copy.deepcopy(original)
        effective["music"] = music
        validate_song(effective)
        if file_sha(path) != original["audio"]["sha256"]:
            raise ValueError("Waveform changed during public analysis")
        public.append(dict(song_id=original["song_id"], audio_sha256=original["audio"]["sha256"], sample_rate=rate,
                           num_samples=len(samples), music=music, synthetic=dataset["synthetic"]))
        evidence.append(dict(song_id=original["song_id"], canonical_audio_sha256=original["audio"]["sha256"],
            detector_model_sha256=row.get("model_sha256"), downbeat_samples_used_for_novelty=positions,
            cosine_novelty=scores, selected_boundary_samples=sorted(selected), threshold=threshold,
            parameters=NOVELTY_CONFIG, threshold_calibrated=False, functional_labels_predicted=False,
            semantic_phrase_detected=False, beat_accuracy_verified=False,
            interpretation="Experimental spectral contrast peaks aligned to existing predicted downbeats; not musicologist-audited form"))
    if file_sha(detections_path) != detection_sha:
        raise ValueError("Beat prediction file changed during analysis")
    document = dict(schema="choreobench.public_structure.v1", dataset_sha256=dataset_sha, songs=public)
    write_json(output, document)
    write_json(evidence_path, dict(schema="choreobench.structure_analysis.v1", public_structure_sha256=object_sha(document),
        detections_sha256=detection_sha, source_code_sha256=file_sha(__file__), evidence=evidence))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze_structure(args.dataset, args.detections, args.out)
    print(json.dumps(dict(songs=len(result["songs"]), section_counts=[len(s["music"]["sections"]) for s in result["songs"]],
                         functional_labels="unknown", accuracy_verified=False)), flush=True)


if __name__ == "__main__":
    main()
