"""Offline public beat/novelty evidence, not verse/chorus or semantic phrase detection."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .dataset import resolve_audio, verify_dataset_audio
from .io import _pairs, _reject_constant, file_sha, finite_json, object_sha, read_json_snapshot, write_json
from .schema import integer, keys, validate_dataset, validate_song


NOVELTY_CONFIG = dict(context_seconds=3, min_section_seconds=20, minimum_cosine_novelty=0.25,
                      novelty_quantile=0.8, bands=32, nperseg=1024, noverlap=512,
                      feature_normalization="songwise-log-band-standardization", std_floor=0.05)
DETECTIONS_SCHEMA = "choreobench.beat_detections.v1"


def _sha(value, where, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(where + " must be a lowercase SHA256")


def _detector(value, synthetic):
    keys(value, ("name", "version", "model_sha256", "code_sha256", "parameters"), where="detector")
    for name in ("name", "version"):
        if not isinstance(value[name], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", value[name]):
            raise ValueError("detector." + name + " must be a nonempty portable identity label")
    _sha(value["model_sha256"], "detector.model_sha256", nullable=synthetic)
    _sha(value["code_sha256"], "detector.code_sha256")
    if not isinstance(value["parameters"], dict):
        raise ValueError("detector.parameters must be an object")
    finite_json(value)


def _jsonl_snapshot(path):
    """Hash exactly the parsed bytes; per-record hashes exclude CR/LF terminators."""
    payload = Path(path).read_bytes()
    result = []
    for number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            raise ValueError("Blank beat detection record at line " + str(number))
        row = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
        finite_json(row)
        if not isinstance(row, dict):
            raise ValueError("Beat detection record must be an object")
        result.append((row, number, hashlib.sha256(raw).hexdigest()))
    if not result:
        raise ValueError("Beat detections must contain at least one record")
    return result, hashlib.sha256(payload).hexdigest()


def _times(values, rate, count, name):
    if not isinstance(values, list):
        raise ValueError(name + " must be a list of seconds")
    previous, samples = None, []
    duration = count / rate
    for value in values:
        if (type(value) not in (int, float) or type(value) is float and not math.isfinite(value)
                or value < 0 or value >= duration):
            raise ValueError(name + " must contain finite seconds inside the canonical audio")
        if previous is not None and value <= previous:
            raise ValueError(name + " must be strictly increasing and unique")
        sample = round(value * rate)
        if not 0 <= sample < count or samples and sample <= samples[-1]:
            raise ValueError(name + " collides or escapes the canonical sample clock after rounding")
        samples.append(sample)
        previous = value
    return samples


def _validate_detection(row, song):
    if row.get("schema") != DETECTIONS_SCHEMA:
        raise ValueError("Require " + DETECTIONS_SCHEMA + "; legacy detections need explicit --wrap-legacy migration")
    keys(row, ("schema", "id", "status", "audio_sha256", "sample_rate", "num_samples", "time_unit",
               "synthetic", "detector", "beats", "downbeats"), ("source_binding",), where="beat detection")
    if row["id"] != song["song_id"] or row["status"] != "ok":
        raise ValueError("Beat detections require the exact song ID and successful status")
    audio = song["audio"]
    _sha(row["audio_sha256"], "detection.audio_sha256")
    integer(row["sample_rate"], "detection.sample_rate", 1)
    integer(row["num_samples"], "detection.num_samples", 1)
    if (row["audio_sha256"], row["sample_rate"], row["num_samples"]) != (
            audio["sha256"], audio["sample_rate"], audio["num_samples"]):
        raise ValueError("Beat detection audio hash/clock differs from the canonical input")
    if row["time_unit"] != "seconds" or type(row["synthetic"]) is not bool:
        raise ValueError("Beat detections require time_unit=seconds and explicit boolean synthetic")
    _detector(row["detector"], row["synthetic"])
    if "source_binding" in row:
        source = row["source_binding"]
        keys(source, ("artifact_sha256", "record_sha256", "run_receipt_sha256"), where="source_binding")
        _sha(source["artifact_sha256"], "source_binding.artifact_sha256")
        _sha(source["record_sha256"], "source_binding.record_sha256")
        _sha(source["run_receipt_sha256"], "source_binding.run_receipt_sha256", nullable=True)
    beats = _times(row["beats"], audio["sample_rate"], audio["num_samples"], "beats")
    downbeats = _times(row["downbeats"], audio["sample_rate"], audio["num_samples"], "downbeats")
    if not set(row["downbeats"]).issubset(row["beats"]):
        raise ValueError("Raw downbeats must be included in the same detector's raw beats")
    return beats, downbeats


def _detections(path, songs):
    records, digest = _jsonl_snapshot(path)
    by_id = {s["song_id"]: s for s in songs}
    rows = {}
    for row, number, raw_sha in records:
        song_id = row.get("id")
        if not isinstance(song_id, str) or song_id not in by_id or song_id in rows:
            raise ValueError("Duplicate/unknown beat detection ID")
        beats, downbeats = _validate_detection(row, by_id[song_id])
        rows[song_id] = dict(record=row, line_number=number, record_sha256=raw_sha,
                             beats=beats, downbeats=downbeats)
    if set(rows) != set(by_id):
        raise ValueError("Beat detections must contain exactly the dataset songs; no silent subset or failed rows")
    return rows, digest


def _write_jsonl(path, rows):
    """Exclusive durable publication; never rewrite the legacy source."""
    path = Path(path)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def wrap_legacy_detections(dataset_path, detections_path, output, *, detector, synthetic, time_unit,
                           run_receipt_path=None):
    """Wrap successful legacy rows without inference or an implicit real declaration.

    ``detector`` and ``synthetic`` are explicit operator declarations, not source
    authentication. ``time_unit`` must explicitly be seconds. Existing model,
    clock, params and synthetic declarations may not be contradicted. Every raw
    source line and the complete artifact stay bound; receipt binding is optional.
    """
    if type(synthetic) is not bool or time_unit != "seconds":
        raise ValueError("Migration requires explicit boolean synthetic and time_unit=seconds")
    _detector(detector, synthetic)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Migration requires a new output; old detections are never rewritten")
    dataset, _ = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    records, digest = _jsonl_snapshot(detections_path)
    receipt_sha = None
    if run_receipt_path is not None:
        receipt, receipt_sha = read_json_snapshot(run_receipt_path)
        if not isinstance(receipt, dict) or receipt.get("output_sha256") != digest:
            raise ValueError("Run receipt does not bind the exact legacy detection file")
        model = receipt.get("model", {})
        if isinstance(model, dict) and "checkpoint_sha256" in model and model["checkpoint_sha256"] != detector["model_sha256"]:
            raise ValueError("Run receipt model identity contradicts the migration detector")
        runtime = receipt.get("runtime", {})
        if (isinstance(runtime, dict) and "beatthis_inference_sha256" in runtime
                and runtime["beatthis_inference_sha256"] != detector["code_sha256"]):
            raise ValueError("Run receipt code identity contradicts the migration detector")
        if "params" in receipt and object_sha(receipt["params"]) != object_sha(detector["parameters"]):
            raise ValueError("Run receipt parameters contradict the migration detector")
        if "synthetic" in receipt and object_sha(receipt["synthetic"]) != object_sha(synthetic):
            raise ValueError("Run receipt synthetic declaration contradicts the migration declaration")
    songs = {s["song_id"]: s for s in dataset["songs"]}
    rows, seen = [], set()
    for old, _, raw_sha in records:
        song_id = old.get("id")
        if not isinstance(song_id, str) or song_id not in songs or song_id in seen:
            raise ValueError("Duplicate/unknown legacy detection ID")
        seen.add(song_id)
        if old.get("schema") == DETECTIONS_SCHEMA:
            raise ValueError("Input already uses the strict detections schema; do not wrap it again")
        if old.get("status") != "ok":
            raise ValueError("Migration accepts only successful rows; failed records are never filtered or replaced")
        if not synthetic and any(old.get(name) is not True for name in ("inference_called", "model_prediction_returned") if name in old):
            raise ValueError("Legacy record explicitly lacks successful model inference")
        audio = songs[song_id]["audio"]
        for name, expected in (("sample_rate", audio["sample_rate"]), ("num_samples", audio["num_samples"]),
                               ("audio_frames", audio["num_samples"])):
            if name in old and (type(old[name]) is not int or old[name] != expected):
                raise ValueError("Legacy " + name + " contradicts the canonical clock")
        for name, expected in (("synthetic", synthetic), ("time_unit", time_unit),
                               ("model_sha256", detector["model_sha256"]), ("library_source_sha256", detector["code_sha256"])):
            if name in old and object_sha(old[name]) != object_sha(expected):
                raise ValueError("Legacy " + name + " contradicts the explicit migration declaration")
        if "duration_seconds" in old:
            duration = old["duration_seconds"]
            if (type(duration) not in (int, float) or type(duration) is float and not math.isfinite(duration)
                    or abs(duration * audio["sample_rate"] - audio["num_samples"]) > .5):
                raise ValueError("Legacy duration contradicts the canonical clock")
        if "detector" in old and old["detector"] != detector["name"] and object_sha(old["detector"]) != object_sha(detector):
            raise ValueError("Legacy detector identity contradicts the migration detector")
        if "params" in old and object_sha(old["params"]) != object_sha(detector["parameters"]):
            raise ValueError("Legacy parameters contradict the migration detector")
        if (detector["name"].lower() in ("beat_this", "beat-this", "beatthis")
                and isinstance(old.get("library_versions"), dict) and "beat-this" in old["library_versions"]
                and old["library_versions"]["beat-this"] != detector["version"]):
            raise ValueError("Legacy Beat This library version contradicts the migration detector")
        row = dict(schema=DETECTIONS_SCHEMA, id=song_id, status="ok", audio_sha256=old.get("audio_sha256"),
            sample_rate=audio["sample_rate"], num_samples=audio["num_samples"], time_unit=time_unit,
            synthetic=synthetic, detector=copy.deepcopy(detector), beats=old.get("beats"), downbeats=old.get("downbeats"),
            source_binding=dict(artifact_sha256=digest, record_sha256=raw_sha, run_receipt_sha256=receipt_sha))
        _validate_detection(row, songs[song_id])
        rows.append(row)
    if seen != set(songs):
        raise ValueError("Legacy detections must contain exactly the dataset songs")
    if file_sha(detections_path) != digest:
        raise ValueError("Legacy detection file changed during migration")
    if run_receipt_path is not None and file_sha(run_receipt_path) != receipt_sha:
        raise ValueError("Run receipt changed during migration")
    _write_jsonl(output, rows)
    return rows


def analyze_structure(dataset_path, detections_path, output, *, splits=None):
    """Produce a versioned public sidecar without editing or rehashing the candidate bank."""
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    if splits is not None and (not isinstance(splits, list) or not splits
            or any(not isinstance(s, str) or s not in ("train", "dev", "test") for s in splits)
            or len(set(splits)) != len(splits)):
        raise ValueError("splits must be a unique nonempty train/dev/test list or None")
    songs = [s for s in dataset["songs"] if splits is None or s["split"] in splits]
    if not songs:
        raise ValueError("No songs match the requested analysis splits")
    output = Path(output)
    evidence_path = output.with_name(output.stem + ".analysis.json")
    if output.exists() or evidence_path.exists():
        raise FileExistsError("Use new public structure output and evidence paths")
    rows, detection_sha = _detections(detections_path, dataset["songs"])
    import numpy as np
    import soundfile as sf
    from scipy.signal import spectrogram, find_peaks
    evidence, public = [], []
    for original in songs:
        detection = rows[original["song_id"]]
        row = detection["record"]
        path = resolve_audio(original, dataset_path)
        samples, rate = sf.read(path, dtype="float64")
        if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) != original["audio"]["num_samples"]:
            raise ValueError("Canonical analysis audio is not finite mono with the original clock")
        if rate != original["audio"]["sample_rate"]:
            raise ValueError("Canonical sample rate differs")
        beats, downbeats = detection["beats"], detection["downbeats"]
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
        detector = row["detector"]
        source = (detector["name"] + "@" + detector["version"] + "+spectral-novelty-v1:"
                  + object_sha(detector) + ":" + detection_sha)
        music = dict(source=source, beats=beats, downbeats=downbeats,
            sections=[dict(section_id="ns%03d" % i, start=a, end=b, label="unknown", repeat_group=None)
                      for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))])
        effective = copy.deepcopy(original)
        effective["music"] = music
        validate_song(effective)
        if file_sha(path) != original["audio"]["sha256"]:
            raise ValueError("Waveform changed during public analysis")
        synthetic = bool(dataset["synthetic"] or row["synthetic"])
        public.append(dict(song_id=original["song_id"], audio_sha256=original["audio"]["sha256"], sample_rate=rate,
                           num_samples=len(samples), music=music, synthetic=synthetic))
        evidence.append(dict(song_id=original["song_id"], canonical_audio_sha256=original["audio"]["sha256"],
            detector=copy.deepcopy(detector), detector_model_sha256=detector["model_sha256"],
            detector_identity_status="declared_not_independently_authenticated",
            detection_line_number=detection["line_number"], detection_record_sha256=detection["record_sha256"],
            detection_source_binding=copy.deepcopy(row.get("source_binding")),
            detection_synthetic=row["synthetic"], dataset_synthetic=dataset["synthetic"], output_synthetic=synthetic,
            downbeat_samples_used_for_novelty=positions,
            cosine_novelty=scores, selected_boundary_samples=sorted(selected), threshold=threshold,
            parameters=NOVELTY_CONFIG, threshold_calibrated=False, functional_labels_predicted=False,
            semantic_phrase_detected=False, beat_accuracy_verified=False,
            interpretation="Experimental spectral contrast peaks aligned to existing predicted downbeats; not musicologist-audited form"))
    if file_sha(detections_path) != detection_sha:
        raise ValueError("Beat prediction file changed during analysis")
    document = dict(schema="choreobench.public_structure.v1", dataset_sha256=dataset_sha, songs=public)
    write_json(output, document)
    write_json(evidence_path, dict(schema="choreobench.structure_analysis.v1", public_structure_sha256=object_sha(document),
        detections_schema=DETECTIONS_SCHEMA, detections_sha256=detection_sha,
        source_code_sha256=file_sha(__file__), evidence=evidence))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wrap-legacy", action="store_true", help="Wrap successful legacy rows; no model inference")
    parser.add_argument("--detector-identity", help="Strict detector JSON for explicit legacy migration")
    parser.add_argument("--synthetic", choices=("true", "false"), help="Explicit detector output declaration; never inferred")
    parser.add_argument("--legacy-time-unit", choices=("seconds",), help="Explicit legacy timestamp unit declaration")
    parser.add_argument("--run-receipt", help="Optional original receipt; must bind the legacy file SHA")
    parser.add_argument("--split", choices=("train", "dev", "test"), action="append",
                        help="Analyze only these splits; detections must still cover the complete source dataset")
    args = parser.parse_args()
    if args.wrap_legacy:
        if args.split is not None:
            parser.error("--split applies to analysis only; legacy wrapping preserves the complete input inventory")
        if args.detector_identity is None or args.synthetic is None or args.legacy_time_unit is None:
            parser.error("--wrap-legacy requires --detector-identity, --synthetic and --legacy-time-unit")
        detector, _ = read_json_snapshot(args.detector_identity)
        rows = wrap_legacy_detections(args.dataset, args.detections, args.out, detector=detector,
            synthetic=args.synthetic == "true", time_unit=args.legacy_time_unit, run_receipt_path=args.run_receipt)
        print(json.dumps(dict(schema=DETECTIONS_SCHEMA, records=len(rows), output=str(Path(args.out)),
            detector_identity_status="operator_declared_not_independently_authenticated", inference_called=False)), flush=True)
        return
    if any(value is not None for value in (args.detector_identity, args.synthetic, args.legacy_time_unit, args.run_receipt)):
        parser.error("Migration declarations require --wrap-legacy")
    result = analyze_structure(args.dataset, args.detections, args.out, splits=args.split)
    print(json.dumps(dict(songs=len(result["songs"]), section_counts=[len(s["music"]["sections"]) for s in result["songs"]],
                         functional_labels="unknown", accuracy_verified=False)), flush=True)


if __name__ == "__main__":
    main()
