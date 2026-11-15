"""Source-bound Beat This observations of two to four complete predicted bars.

The target duration is a soft objective. Waveforms are never time stretched or
silently cut to the old five-second contract. Residuals stay explicit.
"""
import argparse
import copy
import math
from pathlib import Path

from .dataset import verify_dataset_audio
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .music_structure import _detections, _validate_detection
from .schema import keys, validate_dataset, validate_song

SCHEMA = "choreobench.bar_windows.v2"
SONG_SCHEMA = "choreobench.bar_window_song.v2"
CONFIG = dict(target_seconds=5.0625, max_window_seconds=20.0, min_fragment_seconds=0.5,
              allowed_bars=[2, 3, 4])


def _config(value=None):
    value = {} if value is None else value
    keys(value, (), CONFIG, "bar-window config")
    cfg = dict(CONFIG, **copy.deepcopy(value))
    if cfg["allowed_bars"] != [2, 3, 4]:
        raise ValueError("The v2 contract fixes allowed_bars=[2,3,4]")
    for key in ("target_seconds", "max_window_seconds", "min_fragment_seconds"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    if not cfg["min_fragment_seconds"] < cfg["target_seconds"] <= cfg["max_window_seconds"]:
        raise ValueError("Require fragment minimum < target <= maximum duration")
    return cfg


def _groups(points, rate, cfg):
    """DP: minimize orphan bars, then squared duration error, then window count.

    A singleton is explicit residual evidence, never a normal two-to-four-bar
    window. Including it makes the exact coverage contract feasible for short
    sections without inventing a downbeat or dropping a real bar.
    """
    n = len(points) - 1
    best = {n: ((0, 0.0, 0), [])}
    for i in range(n - 1, -1, -1):
        choices = []
        for k in (2, 3, 4, 1):
            j = i + k
            if j > n or j not in best:
                continue
            duration = (points[j] - points[i]) / rate
            if duration > cfg["max_window_seconds"]:
                continue
            score, suffix = best[j]
            candidate = (score[0] + (k == 1),
                         score[1] + (duration - cfg["target_seconds"]) ** 2,
                         score[2] + 1)
            choices.append((candidate, k, [(i, j)] + suffix))
        if choices:
            score, _, groups = min(choices, key=lambda item: (item[0], item[1]))
            best[i] = (score, groups)
    if n and 0 not in best:
        raise ValueError("A predicted bar exceeds the explicit duration capacity; no fixed-time fallback")
    return best[0][1] if n else []


def make_song_windows(song, detection, config=None):
    validate_song(song)
    cfg = _config(config)
    beats, downbeats = _validate_detection(detection, song)
    detector = detection["detector"]
    if detector["name"].lower().replace("-", "_") not in ("beat_this", "beatthis"):
        raise ValueError("Bar windows require Beat This model detections, not inferred 4/4 or librosa")
    if detector["parameters"].get("dbn") is not False:
        raise ValueError("Beat This detections must declare dbn=false")
    if len(downbeats) < 2:
        raise ValueError("Insufficient model downbeats: no complete bar, no silent time fallback")
    rate = song["audio"]["sample_rate"]
    raw = []
    for section in song["music"]["sections"]:
        lo, hi = section["start"], section["end"]
        points = [d for d in downbeats if lo <= d <= hi]
        def emit(a, b, kind, bar_indices=()):
            if a < b:
                raw.append(dict(start=a, end=b, kind=kind, section_id=section["section_id"],
                                bar_intervals=[dict(start=points[i], end=points[i + 1]) for i in bar_indices],
                                reasons=[] if kind == "complete_bars" else [kind]))
        if len(points) < 2:
            emit(lo, hi, "partial_fragment")
            continue
        emit(lo, points[0], "prefix_fragment")
        for i, j in _groups(points, rate, cfg):
            emit(points[i], points[j], "complete_bars" if j - i >= 2 else "single_bar_residual", range(i, j))
        emit(points[-1], hi, "suffix_fragment")
    # Merge tiny residuals only within the same public section. Preserve their
    # reason and mixed-window status; do not claim the enlarged window is pure bars.
    i = 0
    while i < len(raw):
        item = raw[i]
        if (item["end"] - item["start"]) / rate < cfg["min_fragment_seconds"]:
            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(raw)
                          and raw[j]["section_id"] == item["section_id"]
                          and (max(raw[j]["end"], item["end"]) - min(raw[j]["start"], item["start"])) / rate
                          <= cfg["max_window_seconds"]]
            if neighbours:
                j = neighbours[0]
                other = raw[j]
                other.update(start=min(other["start"], item["start"]), end=max(other["end"], item["end"]),
                             kind="partial_fragment", bar_intervals=sorted(other["bar_intervals"] + item["bar_intervals"], key=lambda x: x["start"]),
                             reasons=other["reasons"] + item["reasons"] + ["absorbed_short_fragment"])
                raw.pop(i)
                i = max(0, i - 1)
                continue
        i += 1
    windows = []
    for index, item in enumerate(raw):
        seconds = (item["end"] - item["start"]) / rate
        if seconds > cfg["max_window_seconds"]:
            raise ValueError("Residual exceeds duration capacity; inspect detections instead of time fallback")
        windows.append(dict(index=index, observation_start=item["start"], observation_end=item["end"],
            ownership_start=item["start"], ownership_end=item["end"], actual_window_seconds=seconds,
            nominal_window_seconds=cfg["target_seconds"], sample_rounding_error=0,
            boundaries_are_action_boundaries=False, kind=item["kind"], section_id=item["section_id"],
            complete_bar_count=len(item["bar_intervals"]), bar_intervals=item["bar_intervals"], reasons=item["reasons"]))
    if (not windows or windows[0]["ownership_start"] != 0
            or windows[-1]["ownership_end"] != song["audio"]["num_samples"]
            or any(a["ownership_end"] != b["ownership_start"] for a, b in zip(windows, windows[1:]))):
        raise ValueError("Bar ownership must cover the full canonical song exactly once")
    return dict(schema=SONG_SCHEMA, song_id=song["song_id"], audio_sha256=song["audio"]["sha256"],
        sample_rate=rate, num_samples=song["audio"]["num_samples"], music_sha256=object_sha(song["music"]),
        detection=copy.deepcopy(detection), config=cfg, windows=windows,
        semantic_phrase_verified=False, predicted_bar_accuracy_verified=False,
        algorithm="beat_this_closed_bars_dp.v2", silent_fixed_window_fallback=False)


def validate_song_windows(document, song):
    if not isinstance(document, dict) or document.get("schema") != SONG_SCHEMA:
        raise ValueError("Require a versioned bar-window song")
    expected = make_song_windows(song, document["detection"], document["config"])
    if document != expected:
        raise ValueError("Bar windows differ from source-bound exact recomputation")
    return document


def validate_manifest(document, dataset, dataset_sha):
    keys(document, ("schema", "dataset_sha256", "detections_sha256", "songs"), where="bar window manifest")
    if document["schema"] != SCHEMA or document["dataset_sha256"] != dataset_sha:
        raise ValueError("Bar window manifest belongs to a different dataset")
    from .music_structure import _sha
    _sha(document["detections_sha256"], "detections_sha256")
    if not isinstance(document["songs"], list) or [r.get("song_id") for r in document["songs"]] != [s["song_id"] for s in dataset["songs"]]:
        raise ValueError("Bar window inventory must match every dataset song in order")
    for row, song in zip(document["songs"], dataset["songs"]):
        validate_song_windows(row, song)
        if row["detection"]["synthetic"] and not dataset["synthetic"]:
            raise ValueError("Synthetic beat detections cannot become real-dataset bar evidence")
    return document


def load_manifest(path, dataset, dataset_sha):
    document, sha = read_json_snapshot(path)
    return validate_manifest(document, dataset, dataset_sha), sha


def build_manifest(dataset_path, detections_path, out_path, *, target_seconds=5.0625, max_window_seconds=20.0):
    dataset, sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    rows, detection_sha = _detections(detections_path, dataset["songs"])
    cfg = _config(dict(target_seconds=target_seconds, max_window_seconds=max_window_seconds))
    document = dict(schema=SCHEMA, dataset_sha256=sha, detections_sha256=detection_sha,
                    songs=[make_song_windows(s, rows[s["song_id"]]["record"], cfg) for s in dataset["songs"]])
    if file_sha(dataset_path) != sha or file_sha(detections_path) != detection_sha:
        raise ValueError("Window inputs changed during construction")
    validate_manifest(document, dataset, sha)
    write_json(out_path, document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "detections", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target-seconds", type=float, default=5.0625)
    parser.add_argument("--max-window-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    value = build_manifest(args.dataset, args.detections, args.out,
                           target_seconds=args.target_seconds, max_window_seconds=args.max_window_seconds)
    print({"songs": len(value["songs"]), "windows": sum(len(s["windows"]) for s in value["songs"]),
           "new_model_calls": 0, "schema": SCHEMA})


if __name__ == "__main__":
    main()
