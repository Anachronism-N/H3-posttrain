"""Source-bound, duration-weighted global JEV conditions; no model inference.

v2 imports probabilities with their original audio clock. Model/class identities
must be supplied from the frozen checkpoint; this module never guesses labels.
"""
import copy
import math
import re
from pathlib import Path

from .io import file_sha, read_json_snapshot, write_json
from .schema import keys, integer, validate_dataset

SCHEMA = "choreobench.public_style_prior.v2"


def _sha(value, where):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(where + " must be lowercase SHA256")


def validate_identity(identity):
    keys(identity, ("architecture", "checkpoint_sha256", "encoder_weights_sha256", "class_order",
                    "supported_training_classes", "temperature", "calibration_scope"),
         optional=("source_revision",), where="JEV v2 model identity")
    for key in ("checkpoint_sha256", "encoder_weights_sha256"):
        _sha(identity[key], key)
    labels = identity["class_order"]
    supported = identity["supported_training_classes"]
    if (not isinstance(labels, list) or len(labels) != 22 or len(set(labels)) != 22
            or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 /_+()-]{0,79}", x) for x in labels)):
        raise ValueError("JEV v2 requires a frozen ordered list of 22 distinct safe labels")
    if (not isinstance(supported, list) or len(supported) != 15 or len(set(supported)) != 15
            or not set(supported) <= set(labels)):
        raise ValueError("JEV v2 requires exactly 15 explicitly training-supported classes")
    temperature = identity["temperature"]
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Recorded temperature must be finite and positive; probabilities are not recalibrated here")
    for key in ("architecture", "calibration_scope"):
        if not isinstance(identity[key], str) or not identity[key].strip():
            raise ValueError("Missing JEV identity " + key)


def validate(document, dataset_sha, songs):
    keys(document, ("schema", "dataset_sha256", "model_identity", "songs"),
         optional=("import_provenance",), where="JEV v2 sidecar")
    if document["schema"] != SCHEMA or document["dataset_sha256"] != dataset_sha:
        raise ValueError("JEV v2 must bind the exact dataset")
    _sha(dataset_sha, "dataset_sha256")
    validate_identity(document["model_identity"])
    originals = {s["song_id"]: s for s in songs}
    seen = set()
    for row in document["songs"]:
        keys(row, ("song_id", "audio_sha256", "sample_rate", "num_samples", "windows"),
             optional=("failure",), where="JEV v2 song")
        sid = row["song_id"]
        if sid in seen or sid not in originals:
            raise ValueError("Unknown or duplicate JEV song")
        seen.add(sid)
        audio = originals[sid]["audio"]
        if row["audio_sha256"] != audio["sha256"] or any(row[k] != audio[k] for k in ("sample_rate", "num_samples")):
            raise ValueError("JEV audio identity differs from dataset")
        integer(row["sample_rate"], "sample_rate", 1)
        integer(row["num_samples"], "num_samples", 1)
        if not isinstance(row["windows"], list):
            raise ValueError("JEV windows must be a list")
        if row.get("failure") and row["windows"]:
            raise ValueError("Failed song cannot also contribute predictions")
        intervals = set()
        for w in row["windows"]:
            keys(w, ("start", "end", "audio_sha256", "probabilities"),
                 optional=("ownership_start", "ownership_end"), where="JEV v2 observation")
            for key in ("start", "end"):
                integer(w[key], key)
            if not 0 <= w["start"] < w["end"] <= row["num_samples"]:
                raise ValueError("JEV observation outside source audio")
            if ("ownership_start" in w) != ("ownership_end" in w):
                raise ValueError("Both ownership boundaries are required")
            left, right = w.get("ownership_start", w["start"]), w.get("ownership_end", w["end"])
            integer(left, "ownership_start"); integer(right, "ownership_end")
            if not w["start"] <= left < right <= w["end"]:
                raise ValueError("Ownership must lie within the observed waveform")
            interval = (w["start"], w["end"], left, right)
            if interval in intervals:
                raise ValueError("Duplicate JEV observation/ownership")
            intervals.add(interval)
            _sha(w["audio_sha256"], "window.audio_sha256")
            p = w["probabilities"]
            if (not isinstance(p, list) or len(p) != 22
                    or any(type(x) not in (int, float) or not 0 <= x <= 1 or not math.isfinite(x) for x in p)
                    or abs(math.fsum(p) - 1) > 1e-6):
                raise ValueError("JEV requires 22 finite probabilities summing to one")
    if seen != set(originals):
        raise ValueError("JEV sidecar must include every source song, including failures")


def global_signal(row, identity, digest, dataset_sha):
    """Partition overlap ownership equally, so every covered sample has weight one."""
    windows = sorted(row["windows"], key=lambda w: (w["start"], w["end"],
                      w.get("ownership_start", w["start"]), w.get("ownership_end", w["end"])))
    spans = [(w.get("ownership_start", w["start"]), w.get("ownership_end", w["end"])) for w in windows]
    bounds = sorted({t for span in spans for t in span})
    weights = [0.] * len(windows)
    coverage = 0
    for start, end in zip(bounds, bounds[1:]):
        active = [i for i, (a, b) in enumerate(spans) if a <= start and b >= end]
        if active:
            coverage += end - start
            for i in active:
                weights[i] += (end - start) / len(active)
    probabilities = ([math.fsum(w["probabilities"][i] * weight for w, weight in zip(windows, weights)) / coverage
                      for i in range(22)] if coverage else None)
    labels = identity["class_order"]
    winner = max(range(22), key=lambda i: probabilities[i]) if probabilities else None
    reason = ("prediction_failure" if row.get("failure") else "no_valid_coverage" if winner is None
              else "unsupported_argmax" if labels[winner] not in identity["supported_training_classes"] else "available")
    label = labels[winner] if reason == "available" else "unspecified"
    return dict(schema=SCHEMA, sidecar_sha256=digest, dataset_sha256=dataset_sha,
                song_id=row["song_id"], audio_sha256=row["audio_sha256"], model_identity=copy.deepcopy(identity),
                scope=dict(kind="global", start=0, end=row["num_samples"]), probabilities=probabilities,
                argmax_label=None if winner is None else labels[winner], label=label, status=reason,
                header="Dance style is " + label + ".",
                aggregation=dict(method="ownership_duration_overlap_partition", window_count=len(windows),
                                 windows=[dict(start=w["start"], end=w["end"], ownership_start=a, ownership_end=b,
                                               audio_sha256=w["audio_sha256"], weight_samples=weight)
                                          for w, (a,b), weight in zip(windows, spans, weights)],
                                 coverage_samples=coverage, coverage_fraction=coverage / row["num_samples"],
                                 weight_sum=math.fsum(weights), window_audio_hashes_verified=False),
                usage="Shared global creative condition, not a unique true dance genre; explicit user style takes precedence.")


class GlobalStylePrior:
    def __init__(self, path, document, digest):
        self.path = Path(path).resolve(); self.document = copy.deepcopy(document); self.digest = digest
        self._songs = {row["song_id"]: row for row in document["songs"]}

    def binding(self):
        return dict(public_style_prior_sha256=self.digest, public_style_prior_snapshot=copy.deepcopy(self.document))

    def assert_unchanged(self):
        if file_sha(self.path) != self.digest:
            raise ValueError("Public style prior source bytes changed; start a new run")

    def signal(self, song_id, unit=None):
        # The same global condition is used for every block; no local label drift.
        return global_signal(self._songs[song_id], self.document["model_identity"], self.digest,
                             self.document["dataset_sha256"])


def import_predictions(dataset_path, predictions_path, identity_path, out_path):
    """Import native predictions exported as {songs:[source-clock observation rows]}.

    Input rows have the same strict song/window fields as the v2 sidecar. A missing
    song is retained as an explicit failure, never selected away. No waveform is
    resampled or inferred by this importer and no probability is renormalized.
    """
    dataset, dataset_sha = read_json_snapshot(Path(dataset_path)); validate_dataset(dataset)
    predictions, prediction_sha = read_json_snapshot(Path(predictions_path))
    identity, identity_sha = read_json_snapshot(Path(identity_path))
    keys(predictions, ("songs",), where="native JEV exported predictions")
    rows = copy.deepcopy(predictions["songs"])
    present = {r["song_id"] for r in rows}
    for song in dataset["songs"]:
        if song["song_id"] not in present:
            audio = song["audio"]
            rows.append(dict(song_id=song["song_id"], audio_sha256=audio["sha256"], sample_rate=audio["sample_rate"],
                             num_samples=audio["num_samples"], windows=[], failure="missing_native_prediction"))
    value = dict(schema=SCHEMA, dataset_sha256=dataset_sha, model_identity=identity, songs=rows,
                 import_provenance=dict(predictions_sha256=prediction_sha, model_identity_sha256=identity_sha,
                                        inference_performed=False, window_audio_hashes_verified=False))
    validate(value, dataset_sha, dataset["songs"])
    write_json(Path(out_path), value)
    return dict(path=str(Path(out_path).resolve()), sha256=file_sha(out_path), songs=len(rows),
                missing_songs=sum(bool(r.get("failure")) for r in rows))


def presentation(plan, style):
    """Presentation only: the fixed condition never becomes an action event."""
    def clock(sample):
        seconds = sample / plan["sample_rate"]
        return "%02d:%05.2f" % (int(seconds // 60), seconds % 60)
    lines = [style["header"], ""] if style else []
    lines += ["[%s\u2013%s] %s" % (clock(e["start"]), clock(e["end"]), e["action"]) for e in plan["events"]]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--model-identity', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError('Refusing to overwrite an existing style prior')
    print(json.dumps(import_predictions(args.dataset, args.predictions, args.model_identity, args.out)))


if __name__ == '__main__':
    main()
