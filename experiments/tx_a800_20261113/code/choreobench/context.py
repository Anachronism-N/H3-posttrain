"""Bounded, literal history for section planning; no inferred physical state."""
import copy

from .io import finite_json, object_sha
from .schema import TRACKS, ValidationError, integer, validate_plan


def _event_summary(event):
    return {key: copy.deepcopy(event[key]) for key in
            ("event_id", "start", "end", "track", "action", "motif_id", "exit")}


def _validate_prefix(plan):
    """Validate the complete envelope, allowing only the initial empty schedule.

    A detached one-sample event is solely a schema-validation probe. It is never
    added to the source prefix, its hash, or the resulting context.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("events"), list):
        raise ValidationError("A complete plan prefix is required")
    if plan["events"]:
        validate_plan(plan)
        return
    probe = copy.deepcopy(plan)
    probe["events"] = [{
        "event_id": "context-validation-only", "start": 0, "end": 1,
        "track": "body", "action": "Schema validation placeholder.",
        "attributes": {}, "entry": {}, "exit": {}, "motif_id": None, "evidence": [],
    }]
    validate_plan(probe)


def build_boundary_context(plan, boundary, *, motif_examples=2):
    """Summarize a validated plan prefix without assuming poses persist in gaps.

    The first section may have an empty prefix. Later prefixes must be valid
    plans whose events all end before or exactly at the next section boundary.
    A motif ID only denotes the planner's declaration, not semantic equivalence.
    """
    try:
        finite_json(plan)
    except ValueError as error:
        raise ValidationError("Plan prefix must be finite JSON") from error
    _validate_prefix(plan)
    integer(boundary, "boundary")
    integer(motif_examples, "motif_examples", 1)
    if motif_examples > 8:
        raise ValidationError("motif_examples must be at most 8")
    if boundary >= plan["num_samples"]:
        raise ValidationError("Boundary must precede the end of the song")
    events = plan["events"]
    if not events and boundary != 0:
        raise ValidationError("An empty prefix is allowed only for the first section")
    if any(event["end"] > boundary for event in events):
        raise ValidationError("Plan prefix crosses the next section boundary")
    ordered = sorted(events, key=lambda event: (event["end"], event["start"], event["event_id"]))
    tracks = {}
    for leaf in sorted(TRACKS["body"]):
        preceding = [event for event in ordered if leaf in TRACKS[event["track"]]]
        latest = preceding[-1] if preceding else None
        tracks[leaf] = {
            "latest_event": _event_summary(latest) if latest else None,
            "gap_samples": boundary - latest["end"] if latest else None,
            "touches_boundary": latest["end"] == boundary if latest else False,
            "state_at_boundary_verified": False,
        }
    memory = []
    for motif in plan["motifs"]:
        uses = [event for event in ordered if event["motif_id"] == motif["motif_id"]]
        memory.append({
            "motif_id": motif["motif_id"], "description": motif["description"],
            "prior_event_count": len(uses),
            "last_end": uses[-1]["end"] if uses else None,
            "recent_examples": [_event_summary(event) for event in uses[-motif_examples:]],
            "semantic_recurrence_verified": False,
        })
    return {
        "schema": "choreobench.boundary_context.v1", "boundary": boundary,
        "source_prefix_sha256": object_sha(plan), "song_id": plan["song_id"],
        "sample_rate": plan["sample_rate"], "num_samples": plan["num_samples"],
        "body_parts": tracks, "motif_memory": memory,
        "evidence_kind": "prior planned text and declared exit attributes; not observed body state",
        "instruction": "Use the history to plan continuity and motif variation. Unknown exit fields remain unknown. "
                       "A gap does not imply the previous pose persists. Motif counts are not quality scores.",
    }
