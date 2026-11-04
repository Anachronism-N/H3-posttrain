"""Resolve explicit edit selectors; never interpret arbitrary natural language.

Selectors use one of ``section_id``, ``section_label`` plus a one-based
``occurrence``, or integer ``start``/``end`` samples. An optional ``track``
matches an exact stored track; ``body_scope`` instead includes its descendant
tracks using the same body hierarchy as the plan validator.

``strict=False`` is preview-only: partial temporal or body-scope matches are
shown but cannot authorize a task. Task construction always resolves strictly.
No waveform is read, no music structure is inferred, and no input is mutated.
"""

import copy

from .editing import TASK_SCHEMA, validate_edit_task
from .io import finite_json, object_sha
from .schema import TRACKS, ValidationError, integer, text, validate_plan, validate_song


class TargetResolutionError(ValidationError):
    """An explicit selection cannot safely authorize whole-event edits.

    ``preview`` is a detached, public-only diagnostic for a CLI to display.
    It never contains private goals or the full source plan/provenance.
    """

    def __init__(self, message, *, code, preview):
        super().__init__(message)
        self.code = code
        self.preview = copy.deepcopy(preview)


def _selector(selector, song):
    if not isinstance(selector, dict):
        raise ValidationError("selector must be an explicit JSON object, not a natural-language query")
    try:
        finite_json(selector)
    except ValueError as error:
        raise ValidationError("selector must be finite JSON") from error
    modes = []
    if "section_id" in selector:
        modes.append(("section_id", {"section_id"}))
    if "section_label" in selector or "occurrence" in selector:
        modes.append(("section_label", {"section_label", "occurrence"}))
    if "start" in selector or "end" in selector:
        modes.append(("interval", {"start", "end"}))
    if len(modes) != 1:
        raise ValidationError("selector requires exactly one of section_id, section_label+occurrence, or start+end")
    mode, required = modes[0]
    if not required.issubset(selector) or set(selector) - required - {"track", "body_scope"}:
        raise ValidationError("selector has missing or unknown fields for " + mode)
    if "track" in selector and "body_scope" in selector:
        raise ValidationError("choose exact track OR hierarchical body_scope, not both")
    scope_mode = "all"
    scope_name = "body"
    if "track" in selector or "body_scope" in selector:
        scope_mode = "exact_track" if "track" in selector else "body_scope"
        scope_name = selector["track"] if "track" in selector else selector["body_scope"]
        if not isinstance(scope_name, str) or scope_name not in TRACKS:
            raise ValidationError("track/body_scope must name a supported body track")

    section = None
    occurrence = None
    if mode == "section_id":
        text(selector["section_id"], "selector.section_id")
        matches = [item for item in song["music"]["sections"] if item["section_id"] == selector["section_id"]]
        if len(matches) != 1:
            raise ValidationError("section_id must identify exactly one existing public music section")
        section = matches[0]
    elif mode == "section_label":
        text(selector["section_label"], "selector.section_label")
        integer(selector["occurrence"], "selector.occurrence (one-based)", 1)
        occurrence = selector["occurrence"]
        matches = [item for item in song["music"]["sections"] if item["label"] == selector["section_label"]]
        if not matches:
            raise ValidationError("Unknown section_label; matching is exact and does not infer music structure")
        if occurrence > len(matches):
            raise ValidationError("section occurrence is out of range: %d matches, requested %d" % (len(matches), occurrence))
        section = matches[occurrence - 1]

    if section is not None:
        start, end = section["start"], section["end"]
    else:
        start, end = selector["start"], selector["end"]
        integer(start, "selector.start")
        integer(end, "selector.end", 1)
        if not start < end <= song["audio"]["num_samples"]:
            raise ValidationError("selector interval must satisfy 0 <= start < end <= num_samples")
    return {
        "kind": mode,
        "start": start,
        "end": end,
        "section_id": section["section_id"] if section else None,
        "section_label": section["label"] if section else None,
        "occurrence": occurrence,
        "scope_mode": scope_mode,
        "scope": scope_name,
        "body_parts": sorted(TRACKS[scope_name]),
    }


def _temporal_relation(event, start, end):
    before, after = event["start"] < start, event["end"] > end
    if before and after:
        return "spans_interval"
    if before:
        return "crosses_start"
    if after:
        return "crosses_end"
    return "contained"


def resolve_targets(plan, song, selector, *, strict=True):
    """Return an auditable preview for an explicit selector.

    ``matched_event_ids`` includes intersecting cross-boundary/scope events;
    ``allowed_event_ids`` contains only events wholly within both requested
    scopes. Any cross-match makes the whole selection unsafe to build, even
    when other events are contained. Strict resolution raises an actionable
    :class:`TargetResolutionError` with this same preview attached.
    """
    if type(strict) is not bool:
        raise ValidationError("strict must be boolean; false is preview-only")
    validate_song(song)
    validate_plan(plan, song)
    selected = _selector(selector, song)
    start, end = selected["start"], selected["end"]
    requested_parts = set(selected["body_parts"])
    matches = []
    for event in plan["events"]:
        if event["start"] >= end or event["end"] <= start:
            continue
        event_parts = TRACKS[event["track"]]
        if selected["scope_mode"] == "exact_track":
            if event["track"] != selected["scope"]:
                continue
            body_relation = "contained"
        else:
            if not requested_parts.intersection(event_parts):
                continue
            body_relation = "contained" if event_parts.issubset(requested_parts) else "crosses_scope"
        temporal_relation = _temporal_relation(event, start, end)
        matches.append({
            "event_id": event["event_id"],
            "start": event["start"],
            "end": event["end"],
            "track": event["track"],
            "action": event["action"],
            "overlap": {"start": max(start, event["start"]), "end": min(end, event["end"])},
            "temporal_relation": temporal_relation,
            "body_relation": body_relation,
            "targetable": temporal_relation == body_relation == "contained",
        })
    boundary_ids = [row["event_id"] for row in matches if row["temporal_relation"] != "contained"]
    body_ids = [row["event_id"] for row in matches if row["body_relation"] != "contained"]
    allowed = [row["event_id"] for row in matches if row["targetable"]]
    preview = {
        "schema": "choreobench.target_resolution.v1",
        "song_id": song["song_id"],
        "sample_rate": plan["sample_rate"],
        "num_samples": plan["num_samples"],
        "source_plan_sha256": object_sha(plan),
        "song_sha256": object_sha(song),
        "audio_sha256": song["audio"]["sha256"],
        "selector": copy.deepcopy(selector),
        "selection": selected,
        "matched_event_ids": [row["event_id"] for row in matches],
        "allowed_event_ids": allowed,
        "cross_boundary_event_ids": boundary_ids,
        "cross_body_scope_event_ids": body_ids,
        "matched_events": matches,
        "can_build_task": bool(allowed) and not boundary_ids and not body_ids,
        "semantics": {
            "time": "integer samples, half-open intervals on the source song clock",
            "label_matching": "exact public section labels; occurrence is one-based",
            "audio_identity": "declared canonical SHA256; targeting does not read waveform bytes",
            "instruction_understood": False,
            "private_goals_included": False,
        },
    }
    if not matches:
        raise TargetResolutionError("No plan events match the explicit time/body selector; no edit task was created",
                                    code="no_matches", preview=preview)
    if strict and (boundary_ids or body_ids):
        reasons = []
        if boundary_ids:
            reasons.append("events cross the requested time boundary: " + ", ".join(boundary_ids))
        if body_ids:
            reasons.append("events also cover body parts outside the requested body_scope: " + ", ".join(body_ids))
        raise TargetResolutionError(
            "; ".join(reasons) + ". Whole-event edits would expand the scope. Choose a broader explicit selector "
            "or a plan with separately authored events; strict=False only previews the conflict",
            code="cross_boundary_and_scope" if boundary_ids and body_ids else "cross_boundary" if boundary_ids else "cross_body_scope",
            preview=preview,
        )
    return preview


def build_edit_task(plan, song, selector, *, task_id, instruction, repair_event_ids=None,
                    protected=None, goals=None, semantic_goals=None):
    """Build a v1 private edit task from explicit author-supplied constraints.

    Repair IDs are never inferred from neighbors. Protections and private goals
    are copied as supplied and checked by the existing task validator; this
    function does not infer their meaning from ``instruction`` or solve their
    feasibility. Use ``public_edit_task`` to obtain the model-facing view.
    """
    preview = resolve_targets(plan, song, selector, strict=True)
    task = {
        "schema": TASK_SCHEMA,
        "task_id": task_id,
        "song_id": song["song_id"],
        "source_plan_sha256": preview["source_plan_sha256"],
        "instruction": instruction,
        "allowed_event_ids": copy.deepcopy(preview["allowed_event_ids"]),
        "repair_event_ids": [] if repair_event_ids is None else copy.deepcopy(repair_event_ids),
        "protected": [] if protected is None else copy.deepcopy(protected),
        "goals": [] if goals is None else copy.deepcopy(goals),
        "semantic_goals": [] if semantic_goals is None else copy.deepcopy(semantic_goals),
    }
    validate_edit_task(task, plan)
    if set(task["allowed_event_ids"]).intersection(task["repair_event_ids"]):
        raise ValidationError("repair_event_ids must be disjoint from selected target IDs; no automatic scope expansion")
    return task
