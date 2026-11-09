"""Lossless caption target preparation; no audio I/O, relabeling or training.

The deliberately explicit ``captionlab.source.v1`` input separates authoritative
window metadata from rounded timestamps in the original caption. Character spans
are Python Unicode indices; byte spans address its unchanged UTF-8 encoding.
All sample intervals are half-open. Declared hashes and context flags are not
independently verified by this CPU-only module.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import ntpath
import posixpath
import re


SOURCE_SCHEMA = "captionlab.source.v1"
TARGET_SCHEMA = "captionlab.target.v1"
_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_STAMP = re.compile(r"\[(" + _NUMBER + r")-(" + _NUMBER + r")s\]")
_SOURCE_KEYS = {"schema", "id", "split", "source_group_id", "song_id",
                "near_duplicate_group_id", "raw_caption", "genre_header_template", "audio", "windows"}
_AUDIO_KEYS = {"path", "sha256", "pcm_sha256", "sample_rate", "num_samples",
               "source_start_seconds", "pad_intervals"}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(name + " requires exactly " + ", ".join(sorted(expected)))


def _text(value, name, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(name + " must be nonempty text")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(name + " must have a valid UTF-8 encoding") from error


def _number(value, name, *, minimum=0):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(name + " must be finite numeric seconds >= " + str(minimum))
    return Fraction(str(value))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(name + " must be a lowercase SHA256")


def _object_sha(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("Input must be finite UTF-8 JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _sample(seconds, rate):
    value = Fraction(str(seconds)) * rate
    # Nonnegative nearest integer, exact half rounded upward; never banker's rounding.
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _path_key(path):
    if ntpath.splitdrive(path)[0] or path.startswith("\\"):
        return ntpath.normcase(ntpath.normpath(path))
    return posixpath.normpath(path)


def _validate_source(record):
    _keys(record, _SOURCE_KEYS, "source record")
    if record["schema"] != SOURCE_SCHEMA or record["split"] not in ("train", "val", "test"):
        raise ValueError("Unsupported source schema or split")
    for field in ("id", "source_group_id", "raw_caption"):
        _text(record[field], field)
    for field in ("song_id", "near_duplicate_group_id"):
        _text(record[field], field, nullable=True)
    if "<|" in record["raw_caption"] or "<audio>" in record["raw_caption"].lower():
        raise ValueError("Reserved conversation/audio tokens are not action facts")
    audio = record["audio"]
    _keys(audio, _AUDIO_KEYS, "audio")
    _text(audio["path"], "audio.path")
    _sha(audio["sha256"], "audio.sha256")
    _sha(audio["pcm_sha256"], "audio.pcm_sha256")
    _integer(audio["sample_rate"], "audio.sample_rate", 1)
    _integer(audio["num_samples"], "audio.num_samples", 1)
    if audio["source_start_seconds"] is not None:
        _number(audio["source_start_seconds"], "audio.source_start_seconds")
    pads = audio["pad_intervals"]
    if pads is not None:
        if not isinstance(pads, list):
            raise ValueError("pad_intervals must be declared intervals or null (unknown)")
        previous = 0
        for span in pads:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("Each padding interval must be [start_sample,end_sample]")
            start, end = span
            _integer(start, "padding start")
            _integer(end, "padding end", 1)
            if not previous <= start < end <= audio["num_samples"]:
                raise ValueError("Padding intervals must be ordered, nonoverlapping and inside audio")
            previous = end
    if not isinstance(record["windows"], list) or not record["windows"]:
        raise ValueError("Explicit nonempty authoritative windows are required")
    duration = Fraction(audio["num_samples"], audio["sample_rate"])
    previous = Fraction(0)
    for window in record["windows"]:
        _keys(window, {"start", "end", "context_dependency"}, "window")
        start, end = (_number(window[field], "window." + field) for field in ("start", "end"))
        if not previous <= start < end <= duration:
            raise ValueError("Authoritative windows must be ordered, nonoverlapping and within audio")
        if window["context_dependency"] not in ("unknown", "dependent", "independent"):
            raise ValueError("context_dependency needs an explicit known/unknown declaration")
        if _sample(window["start"], audio["sample_rate"]) >= _sample(window["end"], audio["sample_rate"]):
            raise ValueError("Window collapses under the declared sample rounding")
        previous = end
    _object_sha(record)


def validate_source_records(records):
    """Reject known cross-split links; null song/near-duplicate groups stay unknown."""
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty source record list is required")
    indices = {key: {} for key in ("id", "source_group_id", "song_id", "near_duplicate_group_id",
                                  "audio_sha256", "pcm_sha256", "audio_path")}
    ids = set()
    for record in records:
        _validate_source(record)
        if record["id"] in ids:
            raise ValueError("Duplicate complete source id: " + record["id"])
        ids.add(record["id"])
        values = {key: record[key] for key in ("id", "source_group_id", "song_id", "near_duplicate_group_id")}
        values.update(audio_sha256=record["audio"]["sha256"], pcm_sha256=record["audio"]["pcm_sha256"],
                      audio_path=_path_key(record["audio"]["path"]))
        for key, value in values.items():
            if value is None:
                continue
            previous = indices[key].setdefault(value, record["split"])
            if previous != record["split"]:
                raise ValueError("Cross-split risk: " + key + " links " + previous + " and " + record["split"])
    return {"source_records": len(records), "known_cross_split_links": 0,
            "song_groups_complete": all(row["song_id"] is not None for row in records),
            "near_duplicate_groups_complete": all(row["near_duplicate_group_id"] is not None for row in records),
            "content_similarity_audited": False, "audio_hashes_verified": False}


def _parse(record, classes, tolerance):
    if (not isinstance(classes, list) or not classes or
            any(not isinstance(c, str) or not c.strip() or c != c.strip() or "\n" in c or "\r" in c for c in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("genre_classes must be an explicit unique plain-text vocabulary")
    real_classes = [value for value in classes if value != "unknown"]
    if not real_classes:
        raise ValueError("At least one non-unknown genre class is required")
    raw, template = record["raw_caption"], record["genre_header_template"]
    header_end, label = 0, None
    if template is not None:
        _text(template, "genre_header_template")
        if template.count("{genre}") != 1 or template.count("\n") != 1 or not template.endswith("\n"):
            raise ValueError("Explicit genre_header_template must be one complete physical line")
        prefix, suffix = template.split("{genre}")
        header_end = raw.find("\n") + 1
        header = raw[:header_end]
        if not header_end or not header.startswith(prefix) or not header.endswith(suffix):
            raise ValueError("Raw caption does not match its declared genre line")
        label = header[len(prefix):len(header) - len(suffix)]
        if label not in real_classes and label != "unknown":
            raise ValueError("Explicit genre label is outside the declared vocabulary")
        if prefix.strip() or suffix.strip():
            repeated = re.compile(r"(?m)^" + re.escape(prefix) + r"[^\r\n]+" + re.escape(suffix))
            if repeated.search(raw[header_end:]):
                raise ValueError("Only one explicit genre line is supported; no extra line is silently removed")
    body = raw[header_end:]
    stamps = list(_STAMP.finditer(body))
    if len(stamps) != len(record["windows"]) or not stamps or stamps[0].start() != 0:
        raise ValueError("Exactly one explicit timestamp per declared window is required")
    parsed, previous_display_end = [], Fraction(0)
    for index, (stamp, window) in enumerate(zip(stamps, record["windows"])):
        if stamp.start() and body[stamp.start() - 1] != "\n":
            raise ValueError("Only physical line-start timestamp markers can be removed")
        start, end = Fraction(stamp[1]), Fraction(stamp[2])
        if start >= end or start < previous_display_end:
            raise ValueError("Malformed, reversed or overlapping displayed timestamp")
        if any(abs(value - Fraction(str(window[key]))) > tolerance for value, key in ((start, "start"), (end, "end"))):
            raise ValueError("Displayed timestamp differs from authoritative window metadata")
        following = stamps[index + 1].start() if index + 1 < len(stamps) else len(body)
        action = body[stamp.end():following]
        if not action.strip() or "[" in action or "]" in action:
            raise ValueError("Malformed timestamp/bracket or empty action; no repair is performed")
        parsed.append({"index": index, "start": window["start"], "end": window["end"],
                       "context_dependency": window["context_dependency"],
                       "timestamp_char_span": [header_end + stamp.start(), header_end + stamp.end()],
                       "action_char_span": [header_end + stamp.end(), header_end + following],
                       "action_text": action, "displayed_timestamp": stamp[0]})
        previous_display_end = end
    return real_classes, label, header_end, parsed


def _spans(raw, removals):
    parts, cursor, target_chars, target_bytes = [], 0, 0, 0
    intervals = []
    for start, end, reason in removals:
        if cursor < start:
            intervals.append((cursor, start, "retained"))
        intervals.append((start, end, reason))
        cursor = end
    if cursor < len(raw):
        intervals.append((cursor, len(raw), "retained"))
    for start, end, kind in intervals:
        text = raw[start:end]
        retained = kind == "retained"
        char_end = target_chars + (len(text) if retained else 0)
        byte_end = target_bytes + (len(text.encode("utf-8")) if retained else 0)
        parts.append({"kind": kind, "source_char_span": [start, end],
                      "source_byte_span": [len(raw[:start].encode("utf-8")), len(raw[:end].encode("utf-8"))],
                      "target_char_span": [target_chars, char_end], "target_byte_span": [target_bytes, byte_end],
                      "text": text})
        target_chars, target_bytes = char_end, byte_end
    return parts


def convert_record(record, *, keep_timestamps=False, genre_classes, timestamp_tolerance_seconds=0.001):
    """Remove only the declared genre line/optional line-start timestamp bytes.

    Unknown/missing genre is masked, never assigned to a real class. Window times
    come from explicit metadata; displayed rounding tolerance is recorded and
    never used to snap, stretch, infer or repair source intervals.
    """
    _validate_source(record)
    if type(keep_timestamps) is not bool:
        raise ValueError("keep_timestamps must be an explicit boolean")
    tolerance = _number(timestamp_tolerance_seconds, "timestamp_tolerance_seconds")
    classes, label, header_end, windows = _parse(record, genre_classes, tolerance)
    removals = [(0, header_end, "removed_genre_header")] if header_end else []
    if not keep_timestamps:
        removals += [(*window["timestamp_char_span"], "removed_timestamp") for window in windows]
    spans = _spans(record["raw_caption"], removals)
    target = "".join(part["text"] for part in spans if part["kind"] == "retained")
    return {"schema": TARGET_SCHEMA, "id": record["id"], "split": record["split"],
            "source": deepcopy(record), "source_sha256": _object_sha(record),
            "raw_caption_sha256": hashlib.sha256(record["raw_caption"].encode("utf-8")).hexdigest(),
            "target_text": target, "keep_timestamps": keep_timestamps,
            "timestamp_tolerance_seconds": timestamp_tolerance_seconds, "genre_classes": classes,
            "jev": {"label": label, "class_index": classes.index(label) if label in classes else None,
                    "known_mask": label in classes, "label_source": "explicit_original_genre_line" if header_end else "missing",
                    "source_char_span": [0, header_end], "independently_verified": False},
            "windows": windows, "source_spans": spans, "context_dependency_inferred": False,
            "window_boundaries_are_action_boundaries": False,
            "semantic_approval_added": False}


def restore_raw(converted):
    """Validate the deterministic conversion and reconstruct its original text."""
    if not isinstance(converted, dict) or converted.get("schema") != TARGET_SCHEMA:
        raise ValueError("A captionlab target record is required")
    try:
        expected = convert_record(converted["source"], keep_timestamps=converted["keep_timestamps"],
                                  genre_classes=converted["genre_classes"],
                                  timestamp_tolerance_seconds=converted["timestamp_tolerance_seconds"])
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete target conversion evidence") from error
    if _object_sha(expected) != _object_sha(converted):
        raise ValueError("Converted target or reversible provenance was modified")
    restored = "".join(part["text"] for part in converted["source_spans"])
    if restored != converted["source"]["raw_caption"]:
        raise ValueError("Source span reconstruction differs from raw caption")
    return restored


def build_crop_manifest(converted_records, *, min_seconds=1.0, max_seconds=3.0):
    """Plan full existing-window crops, without reading/writing/cropping audio.

    Parent file/PCM hashes remain parent identities. A child PCM hash is unknown
    until a separate materializer actually produces and verifies its bytes.
    """
    low, high = _number(min_seconds, "min_seconds"), _number(max_seconds, "max_seconds")
    if low <= 0 or high < low:
        raise ValueError("Positive ordered crop duration bounds required")
    if not isinstance(converted_records, list) or not converted_records:
        raise ValueError("Nonempty converted_records required")
    for converted in converted_records:
        restore_raw(converted)
    classes = converted_records[0]["genre_classes"]
    if any(converted["genre_classes"] != classes for converted in converted_records):
        raise ValueError("Crop records must use the same ordered JEV genre vocabulary")
    sources = [converted["source"] for converted in converted_records]
    isolation = validate_source_records(sources)
    crops, excluded = [], []
    for converted in converted_records:
        source, audio = converted["source"], converted["source"]["audio"]
        for window in converted["windows"]:
            length = Fraction(str(window["end"])) - Fraction(str(window["start"]))
            if not low <= length <= high:
                excluded.append({"source_id": source["id"], "window_index": window["index"],
                                 "reason": "existing_window_outside_duration_range"})
                continue
            start, end = (_sample(window[key], audio["sample_rate"]) for key in ("start", "end"))
            if not 0 <= start < end <= audio["num_samples"]:
                raise ValueError("Rounded crop exceeds the actual parent sample clock")
            pads = audio["pad_intervals"]
            intersections = None if pads is None else [[max(start, a), min(end, b)] for a, b in pads
                                                       if max(start, a) < min(end, b)]
            origin = audio["source_start_seconds"]
            absolute = None if origin is None else [float(Fraction(str(origin)) + Fraction(value, audio["sample_rate"]))
                                                    for value in (start, end)]
            row = {"id": source["id"] + "::window-" + format(window["index"], "06d"),
                   "source_id": source["id"], "source_sha256": converted["source_sha256"],
                   "split": source["split"], "source_group_id": source["source_group_id"],
                   "song_id": source["song_id"], "near_duplicate_group_id": source["near_duplicate_group_id"],
                   "audio": deepcopy(audio), "window_index": window["index"],
                   "declared_window_seconds": [window["start"], window["end"]],
                   "start_sample": start, "end_sample": end, "num_samples": end - start,
                   "actual_duration_seconds": (end - start) / audio["sample_rate"],
                   "absolute_source_seconds": absolute, "sample_rounding": "nearest-ties-up-half-open-v1",
                   "parent_pad_intersections": intersections,
                   "crop_local_pad_intersections": None if intersections is None else [[a - start, b - start] for a, b in intersections],
                   "context_dependency": window["context_dependency"], "context_dependency_inferred": False,
                   "window_boundaries_are_action_boundaries": False,
                   "target_text": window["action_text"], "action_char_span": window["action_char_span"],
                   "action_byte_span": [len(source["raw_caption"][:value].encode("utf-8")) for value in window["action_char_span"]],
                   "source_timestamp": window["displayed_timestamp"], "jev": deepcopy(converted["jev"]),
                   "jev_observation_samples": [0, audio["num_samples"]],
                   "output_audio_path": None, "output_pcm_sha256": None, "audio_cropped": False}
            crops.append(row)
    document = {"schema": "captionlab.crop_manifest.v1", "min_seconds": min_seconds, "max_seconds": max_seconds,
                "sources": deepcopy(sources), "source_record_count": len(sources),
                "genre_classes": deepcopy(classes),
                "windows": crops, "excluded_windows": excluded, "isolation": isolation,
                "audio_cropped": False, "ready_for_training": False,
                "exposure_note": "Derived windows share parent split; track parent exposure, seconds and window counts separately."}
    _object_sha(document)
    return document
