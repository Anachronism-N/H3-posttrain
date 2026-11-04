"""Deterministic planning contexts, not musical phrases or action boundaries.

No inference runs here. ``reviewed`` is an author's declaration about meter,
not a calibrated confidence or proof that the imported downbeats are correct.
Expert observation/ownership windows remain exactly those from ``make_windows``.
"""
import bisect
import math

from .audio import WINDOW_SECONDS, make_windows
from .io import finite_json, object_sha
from .schema import ValidationError, integer, keys, validate_song


DEFAULT_CONFIG = {"bars_per_block": 4, "max_unit_seconds": 20,
                  "fallback_seconds": 10.125, "meter_status": "unverified"}


def _json(value, where):
    try:
        finite_json(value)
    except ValueError as error:
        raise ValidationError(where + " must be finite JSON") from error


def _config(config, rate):
    if config is None:
        config = {}
    keys(config, (), DEFAULT_CONFIG, "planning config")
    value = dict(DEFAULT_CONFIG, **config)
    integer(value["bars_per_block"], "bars_per_block", 1)
    for name in ("max_unit_seconds", "fallback_seconds"):
        number = value[name]
        if (type(number) not in (int, float) or number <= 0
                or type(number) is float and not math.isfinite(number)):
            raise ValidationError(name + " must be a finite positive number, not a boolean")
        try:
            product = number * rate
        except OverflowError as error:
            raise ValidationError(name + " overflows the sample clock") from error
        if type(product) is float and not math.isfinite(product):
            raise ValidationError(name + " overflows the sample clock")
    if value["meter_status"] not in ("unverified", "reviewed"):
        raise ValidationError("meter_status must be unverified/reviewed")
    # A capacity is a hard upper bound; fallback length uses the native window's
    # nearest-sample convention. These are different, explicit operations.
    capacity = math.floor(value["max_unit_seconds"] * rate)
    fallback_product = value["fallback_seconds"] * rate
    fallback = fallback_product if type(fallback_product) is int else math.floor(fallback_product + .5)
    if capacity < 1 or fallback < 1:
        raise ValidationError("Planning durations must represent at least one sample")
    return value, capacity, fallback


def _windows(song, windows):
    audio = song["audio"]
    count, rate = audio["num_samples"], audio["sample_rate"]
    try:
        if windows is None:
            return make_windows(count, rate)
        _json(windows, "windows")
        if not isinstance(windows, list) or not windows:
            raise ValidationError("windows must be a nonempty make_windows result")
        for index, window in enumerate(windows):
            keys(window, ("index", "observation_start", "observation_end", "ownership_start",
                          "ownership_end", "nominal_window_seconds", "sample_rounding_error",
                          "boundaries_are_action_boundaries"), where="window")
            for name in ("index", "observation_start", "observation_end", "ownership_start", "ownership_end"):
                integer(window[name], "window." + name)
            if window["index"] != index:
                raise ValidationError("Window indices must be consecutive from zero")
        stride = None
        if len(windows) > 1:
            gap = windows[1]["observation_start"] - windows[0]["observation_start"]
            length = math.floor(WINDOW_SECONDS * rate + .5)
            stride = None if gap == length else gap / rate
        expected = make_windows(count, rate, stride_seconds=stride)
        if object_sha(windows) != object_sha(expected):
            raise ValidationError("windows must exactly match the unchanged make_windows contract")
        return expected
    except (ValueError, OverflowError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError(str(error)) from error


def _overlap(start, end, other_start, other_end):
    left, right = max(start, other_start), min(end, other_end)
    return {"start": left, "end": right} if left < right else None


def build_planning_units(song, windows=None, config=None):
    """Build a source-bound, exact full-song partition and window intersections.

    Config accepts a subset of ``DEFAULT_CONFIG``. Unverified meter always uses
    time blocks, even for synthetic songs. Reviewed meter groups *closed*
    downbeat intervals; no BPM extrapolation, beat repair or meter detection is
    performed. Supplying nondefault-stride windows binds that exact schedule;
    pass the same windows when validating the result.
    """
    _json(song, "song")
    validate_song(song)
    rate, count = song["audio"]["sample_rate"], song["audio"]["num_samples"]
    cfg, capacity, fallback = _config(config, rate)
    expert_windows = _windows(song, windows)
    downbeats = song["music"]["downbeats"]
    units = []

    def emit(section, start, end, kind, reasons, bars=()):
        if not start < end:
            return
        references = []
        for window in expert_windows:
            observation = _overlap(start, end, window["observation_start"], window["observation_end"])
            if observation is not None:
                references.append({"window_index": window["index"], "observation_overlap": observation,
                    "ownership_overlap": _overlap(start, end, window["ownership_start"], window["ownership_end"])})
        units.append({"unit_id": "pu%04d" % len(units), "section_id": section["section_id"],
            "kind": kind, "start": start, "end": end,
            "bars": len(bars) if kind == "metrical_block" else None,
            "complete_bar_count": len(bars),
            "bar_intervals": [{"start": left, "end": right} for left, right in bars],
            "reasons": list(reasons),
            "boundary_source": "section_bounds_and_sample_clock" if kind == "time_block"
                else "tool_downbeats" if kind == "metrical_block" else "tool_downbeats_and_section_bounds",
            "semantic_phrase_verified": False, "window_refs": references})

    def time_blocks(section, start, end, reasons):
        size = min(fallback, capacity)
        why = list(reasons) + (["fallback_reduced_to_capacity"] if fallback > capacity else [])
        while start < end:
            stop = min(start + size, end)
            emit(section, start, stop, "time_block", why)
            start = stop

    def fragment(section, start, end, reasons, bars=()):
        if end - start > capacity:
            time_blocks(section, start, end, list(reasons) + ["fragment_exceeds_capacity"])
        else:
            emit(section, start, end, "partial_fragment", reasons, bars)

    for section in song["music"]["sections"]:
        start, end = section["start"], section["end"]
        fallback_reasons = []
        if cfg["meter_status"] == "unverified":
            fallback_reasons.append("meter_unverified")
        if not song["music"]["beats"]:
            fallback_reasons.append("beats_unavailable")
        if len(downbeats) < 2:
            fallback_reasons.append("insufficient_downbeats_for_closed_bar")
        if fallback_reasons:
            time_blocks(section, start, end, fallback_reasons)
            continue

        i = bisect.bisect_left(downbeats, start)
        cursor = start
        first = min(downbeats[i], end) if i < len(downbeats) else end
        if cursor < first:
            reason = "before_first_downbeat" if start < downbeats[0] else (
                "after_last_downbeat" if start > downbeats[-1] else "section_starts_inside_predicted_bar")
            fragment(section, cursor, first, [reason])
            cursor = first
        while cursor < end and i + 1 < len(downbeats):
            if downbeats[i + 1] > end:
                fragment(section, cursor, end, ["section_ends_inside_predicted_bar"])
                cursor = end
                break
            if downbeats[i + 1] - cursor > capacity:
                time_blocks(section, cursor, downbeats[i + 1], ["predicted_bar_exceeds_capacity"])
                cursor = downbeats[i + 1]
                i += 1
                continue
            j = i
            bars = []
            while (j + 1 < len(downbeats) and len(bars) < cfg["bars_per_block"]
                   and downbeats[j + 1] <= end and downbeats[j + 1] - cursor <= capacity):
                bars.append((downbeats[j], downbeats[j + 1]))
                j += 1
            capacity_limited = (len(bars) < cfg["bars_per_block"] and j + 1 < len(downbeats)
                                and downbeats[j + 1] <= end and downbeats[j + 1] - cursor > capacity)
            if len(bars) == cfg["bars_per_block"]:
                emit(section, cursor, downbeats[j], "metrical_block", ["requested_bar_count"], bars)
            elif capacity_limited:
                emit(section, cursor, downbeats[j], "metrical_block", ["bar_count_reduced_to_capacity"], bars)
            else:
                fragment(section, cursor, downbeats[j], ["fewer_than_requested_complete_bars"], bars)
            cursor, i = downbeats[j], j
        if cursor < end:
            fragment(section, cursor, end, ["unclosed_tail_after_last_downbeat"])

    result = {"schema": "choreobench.planning_units.v1", "song_id": song["song_id"],
        "sample_rate": rate, "num_samples": count, "audio_sha256": song["audio"]["sha256"],
        "song_sha256": object_sha(song), "music_sha256": object_sha(song["music"]),
        "windows_sha256": object_sha(expert_windows), "window_count": len(expert_windows),
        "config": cfg, "provenance": {"algorithm": "section_downbeat_partition.v1",
            "music_source": song["music"]["source"], "meter_status_basis": "author_declaration_not_confidence",
            "semantic_phrase_detection": False, "boundaries_are_action_boundaries": False,
            "decision_scope": "observation_window", "cross_unit_window_refs": "shared_single_decision",
            "capacity_samples": capacity, "fallback_samples": fallback,
            "effective_time_block_samples": min(capacity, fallback)}, "units": units}
    _json(result, "planning units")
    return result


def validate_planning_units(document, song, windows=None):
    """Reject any field, type, source, partition or overlap differing from recomputation."""
    _json(document, "planning units")
    if not isinstance(document, dict) or document.get("schema") != "choreobench.planning_units.v1":
        raise ValidationError("Unsupported planning units schema")
    if "config" not in document:
        raise ValidationError("Planning units config is required")
    expected = build_planning_units(song, windows, document["config"])
    if object_sha(document) != object_sha(expected):
        raise ValidationError("Planning units differ from exact source-bound recomputation")
    return document
