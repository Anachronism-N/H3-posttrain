"""Literal final-plan scheduling diagnostics, never musical or physical scores."""
import math
from statistics import median

from .io import finite_json, object_sha
from .schema import TRACKS, ValidationError, identifier, integer, validate_plan
from .timing import event_clock


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _timing_config(config):
    _require(isinstance(config, dict), "schedule diagnostics config must be an object")
    finite_json(config)
    result = dict(event_time_format=config.get("event_time_format", "global_samples"),
                  event_schedule_policy=config.get("event_schedule_policy", "free"),
                  event_grid_seconds=config.get("event_grid_seconds", .25),
                  min_event_seconds=config.get("min_event_seconds", 0),
                  max_events_per_unit=config.get("max_events_per_unit", 3))
    _require(result["event_time_format"] in ("global_samples", "boundary_ids"), "Unknown event_time_format")
    _require(result["event_schedule_policy"] in ("free", "balanced_full_unit"), "Unknown event_schedule_policy")
    if result["event_schedule_policy"] == "balanced_full_unit":
        _require(result["event_time_format"] == "boundary_ids", "balanced_full_unit requires boundary_ids")
    _require(type(result["max_events_per_unit"]) is int and 1 <= result["max_events_per_unit"] <= 8,
             "max_events_per_unit must be an integer in [1,8]")
    for name in ("event_grid_seconds", "min_event_seconds"):
        value = result[name]
        _require(type(value) in (int, float) and value >= 0
                 and (type(value) is int or math.isfinite(value)), name + " must be finite and nonnegative")
    _require(result["event_grid_seconds"] > 0, "event_grid_seconds must be positive")
    return result


def _units(document, plan):
    _require(isinstance(document, dict), "planning units document must be an object")
    finite_json(document)
    _require(document.get("schema") == "choreobench.planning_units.v1", "Unsupported planning units schema")
    for name in ("sample_rate", "num_samples"):
        integer(document.get(name), "planning units " + name, 1)
    _require((document.get("song_id"), document["sample_rate"], document["num_samples"]) ==
             (plan["song_id"], plan["sample_rate"], plan["num_samples"]), "Plan and planning unit song/clock differ")
    units = document.get("units")
    _require(isinstance(units, list) and bool(units), "Planning units must be a nonempty list")
    cursor, seen = 0, set()
    for unit in units:
        _require(isinstance(unit, dict), "Planning unit must be an object")
        identifier(unit.get("unit_id"), "planning unit_id")
        _require(unit["unit_id"] not in seen, "Duplicate planning unit_id")
        seen.add(unit["unit_id"])
        integer(unit.get("start"), "planning unit start")
        integer(unit.get("end"), "planning unit end", 1)
        _require(unit["start"] == cursor and cursor < unit["end"] <= plan["num_samples"],
                 "Planning units must be an ordered, disjoint full-song partition")
        cursor = unit["end"]
    _require(cursor == plan["num_samples"], "Planning units must cover the real final sample boundary")
    return units


def _durations(events, rate):
    if not events:
        return dict(minimum_samples=None, median_samples=None, minimum_seconds=None, median_seconds=None)
    durations = [event["end"] - event["start"] for event in events]
    shortest, middle = min(durations), median(durations)
    return dict(minimum_samples=shortest, median_samples=middle,
                minimum_seconds=shortest / rate, median_seconds=middle / rate)


def schedule_diagnostics(plan, units_document, config):
    """Count actual events and describe their declared scheduling constraints.

    Every event must be fully contained in exactly one unchanged planning unit.
    Unit counts, not events, form saturation denominators. Configured-maximum
    saturation is separate from saturation at the maximum offered menu size:
    short balanced units may offer fewer slots. Free-mode feasible counts are
    not enumerated: a configured 1..N range is not evidence that each N fits.

    The caller retains source/structure audit responsibility. Hashes bind these
    input documents; this function checks their clock/partition consistency,
    not the authenticity of their declared music or model evidence.
    """
    validate_plan(plan)
    cfg = _timing_config(config)
    units = _units(units_document, plan)
    assigned = {unit["unit_id"]: [] for unit in units}
    for event in plan["events"]:
        matches = [unit for unit in units if unit["start"] <= event["start"] < event["end"] <= unit["end"]]
        _require(len(matches) == 1, "Event " + event["event_id"] + " is not uniquely contained in one planning unit")
        assigned[matches[0]["unit_id"]].append(event)
    maximum, balanced = cfg["max_events_per_unit"], cfg["event_schedule_policy"] == "balanced_full_unit"
    rows = []
    for unit in units:
        events = assigned[unit["unit_id"]]
        clock = event_clock(unit, plan["sample_rate"], cfg)
        allowed = sorted({len(schedule) for schedule in clock["permitted_schedules"]}) if balanced else None
        if balanced:
            samples = clock["boundary_samples"]
            schedules = [[(samples[pair["start_boundary"]], samples[pair["end_boundary"]]) for pair in schedule]
                         for schedule in clock["permitted_schedules"]]
            matches_schedule = [(event["start"], event["end"]) for event in events] in schedules
        else:
            matches_schedule = None
        count = len(events)
        rows.append(dict(unit_id=unit["unit_id"], start=unit["start"], end=unit["end"], event_count=count,
            event_ids=[event["event_id"] for event in events],
            track_counts={track: sum(event["track"] == track for event in events) for track in sorted(TRACKS)},
            configured_event_count_range=dict(minimum=1, maximum=maximum),
            count_within_configured_range=1 <= count <= maximum,
            allowed_event_counts=allowed,
            allowed_counts_basis="public_permitted_schedules" if balanced else "not_enumerated_configured_range_only",
            count_in_allowed_set=count in allowed if balanced else None,
            maximum_allowed_event_count=max(allowed) if balanced else None,
            at_configured_maximum=count == maximum,
            at_allowed_maximum=count == max(allowed) if balanced else None,
            matches_permitted_schedule=matches_schedule,
            minimum_duration_samples=clock["minimum_duration_samples"],
            all_events_meet_minimum_duration=all(event["end"] - event["start"] >= clock["minimum_duration_samples"]
                                                 for event in events) if events else None,
            event_durations=_durations(events, plan["sample_rate"])))
    unit_count = len(rows)
    at_configured = sum(row["at_configured_maximum"] for row in rows)
    at_allowed = sum(row["at_allowed_maximum"] for row in rows) if balanced else None
    result = dict(schema="choreobench.schedule_diagnostics.v1", song_id=plan["song_id"],
        plan_sha256=object_sha(plan), units_sha256=object_sha(units_document), config_sha256=object_sha(config),
        sample_rate=plan["sample_rate"], num_samples=plan["num_samples"],
        event_schedule_policy=cfg["event_schedule_policy"], timing_config=cfg, max_events_per_unit=maximum,
        unit_count=unit_count, event_count=len(plan["events"]),
        track_counts={track: sum(event["track"] == track for event in plan["events"]) for track in sorted(TRACKS)},
        track_count_basis="declared_track_labels_not_inferred_body_motion",
        event_durations=_durations(plan["events"], plan["sample_rate"]),
        configured_event_capacity=unit_count * maximum,
        total_event_capacity_fraction=len(plan["events"]) / (unit_count * maximum),
        units_at_configured_maximum=at_configured, configured_maximum_saturation_fraction=at_configured / unit_count,
        units_at_allowed_maximum=at_allowed,
        allowed_maximum_saturation_fraction=at_allowed / unit_count if balanced else None,
        saturation_denominator=dict(unit="planning_unit", count=unit_count),
        units_with_no_events=sum(row["event_count"] == 0 for row in rows),
        units_outside_configured_count_range=sum(not row["count_within_configured_range"] for row in rows),
        units=rows, musical_or_physical_quality_verified=False,
        note=("Literal counts and durations only; no musicality, adaptive complexity or physical feasibility score. "
              "Configured capacity and saturation do not prove feasible free-mode counts. "
              "Capacity ratios are not clipped; values above one expose budget violations. "
              "Balanced menu matching is an engineering constraint, not quality or beat synchronization. "
              "Planning units and events are not independent songs for statistical inference."))
    finite_json(result)
    return result
