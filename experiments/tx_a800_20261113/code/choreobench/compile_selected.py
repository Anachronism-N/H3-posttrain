"""Compile source-bound whole responses; never silently crop action semantics."""
import copy

from .io import object_sha
from .schema import validate_plan
from .selection import _seal, _validated_selection


def _gaps(events, duration):
    result, cursor = [], 0
    for event in events:
        if event["start"] > cursor:
            result.append(dict(start=cursor, end=event["start"]))
        cursor = max(cursor, event["end"])
    if cursor < duration:
        result.append(dict(start=cursor, end=duration))
    return result


def compile_selection(bank, song, selection, timing_policy="strict"):
    """Return {plan, report}; an entirely omitted schedule has plan=None.

    strict: retain only entire nominal source fragments inside ownership.
    proposed_crop: retain intersections as *new*, semantically unreviewed timing
    proposals. Neither policy infers measured motion bounds or body state.
    Current audio bytes are checked by the runner; this module binds the exact
    source song/audio declarations and immutable bank/request/raw evidence.
    """
    if timing_policy not in ("strict", "proposed_crop"):
        raise ValueError("timing_policy must be strict or proposed_crop")
    options = _validated_selection(bank, song, selection)
    choices = {c["window_id"]: c["candidate_index"] for c in selection["choices"]}
    events, mappings, omitted_sources, review = [], [], [], []
    for row in options["windows"]:
        window = row["window"]
        candidate = next(c for c in row["candidates"] if c["candidate_index"] == choices[row["window_id"]])
        for fragment in candidate["fragments"]:
            start, end = fragment["nominal_interval"]
            cut_start, cut_end = max(start, window["ownership_start"]), min(end, window["ownership_end"])
            intersects = cut_start < cut_end
            partial = intersects and (cut_start != start or cut_end != end)
            retain = intersects and (not partial or timing_policy == "proposed_crop")
            event_id = row["window_id"] + "-p%d" % fragment["part_index"]
            transform = ("crop_proposed" if partial else "identity") if retain else (
                "omitted_partial_source" if partial else "omitted_outside_ownership")
            mapping = dict(window_id=row["window_id"], candidate_index=candidate["candidate_index"],
                call_id=candidate["call_id"], part_index=fragment["part_index"],
                raw_record_sha256=candidate["raw_record_sha256"], request_sha256=candidate["request_sha256"],
                action_sha256=object_sha(fragment["text"]), source_interval=[start, end],
                source_declared_interval_seconds=copy.deepcopy(fragment["declared_interval_seconds"]),
                observation_interval=[window["observation_start"], window["observation_end"]],
                ownership_interval=[window["ownership_start"], window["ownership_end"]],
                ownership_intersection=[cut_start, cut_end] if intersects else None,
                scheduled_interval=[cut_start, cut_end] if retain else None,
                event_id=event_id if retain else None, retained=retain, partial_source=partial,
                time_transform=transform, semantic_timing_verified=False)
            mappings.append(mapping)
            if partial:
                review.append(dict(window_id=row["window_id"], call_id=candidate["call_id"],
                    part_index=fragment["part_index"], event_id=event_id if retain else None,
                    reason="Source caption is only partly inside ownership; duration/count semantics not verified",
                    status="pending", retained=retain))
            if not retain:
                omitted_sources.append(dict(call_id=candidate["call_id"], part_index=fragment["part_index"],
                    source_interval=[start, end], reason=transform))
                continue
            events.append(dict(event_id=event_id, start=cut_start, end=cut_end, track="body",
                action=fragment["text"], attributes={}, entry={}, exit={}, motif_id=None,
                evidence=[candidate["call_id"]]))
    events.sort(key=lambda e: (e["start"], e["track"], e["event_id"]))
    omitted = _gaps(events, song["audio"]["num_samples"])
    covered = sum(e["end"] - e["start"] for e in events)
    plan = None
    if events:
        plan = dict(schema="choreobench.plan.v1", song_id=song["song_id"],
            sample_rate=song["audio"]["sample_rate"], num_samples=song["audio"]["num_samples"], motifs=[],
            events=events, provenance=dict(selection_sha256=selection["record_sha256"],
                source_binding=copy.deepcopy(options["binding"]), timing_policy=timing_policy,
                source_mappings_sha256=object_sha(mappings), synthetic=selection["synthetic"],
                selection_provenance=copy.deepcopy(selection.get("selection_provenance")),
                timing_semantics="Proposed ownership schedule; nominal caption times are not measured action boundaries",
                music_or_physical_quality_verified=False))
        validate_plan(plan, song)
    report = _seal(dict(schema="choreobench.compilation_report.v1", song_id=song["song_id"],
        binding=copy.deepcopy(options["binding"]), selection_sha256=selection["record_sha256"],
        plan_sha256=object_sha(plan) if plan is not None else None, timing_policy=timing_policy,
        synthetic=selection["synthetic"], selection_provenance=copy.deepcopy(selection.get("selection_provenance")),
        source_mappings=mappings, omitted_source_intervals=omitted_sources,
        omitted_intervals=omitted, partial_source_count=sum(m["partial_source"] for m in mappings),
        needs_review=review, retained_event_count=len(events), covered_samples=covered,
        num_samples=song["audio"]["num_samples"], coverage_fraction=covered / song["audio"]["num_samples"],
        status="compiled" if plan is not None else "abstained_no_retained_events",
        nominal_time_normalization="Legacy fixed-window parser tolerance mapped to nominal sample bounds; raw declared seconds retained",
        empty_intervals_filled_with_rest=False, source_action_text_unchanged=True,
        music_or_physical_quality_verified=False, new_local_calls=0))
    bank.assert_unchanged(full=True)
    return dict(plan=plan, report=report)
