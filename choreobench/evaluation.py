"""Literal diagnostics and song-clustered uncertainty, never aesthetic proxy scores."""

import copy
import math
import random
from collections import Counter
from statistics import mean

from .editing import (
    _MISSING, event_value, literal_equal, protection_results, validate_edit_task,
)
from .io import object_sha
from .schema import ValidationError, validate_plan, validate_song


def _coverage(events, duration):
    merged = []
    for start, end in sorted((event["start"], event["end"]) for event in events):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    cursor, gaps = 0, []
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if cursor < duration:
        gaps.append([cursor, duration])
    covered = sum(end - start for start, end in merged)
    return {"covered_samples": covered, "total_samples": duration,
            "fraction": covered / duration, "gaps": gaps,
            "interpretation": "literal interval coverage; no motion quality or rest inference"}


def evaluate_plan(plan, song):
    """Return schema/clock/coverage diagnostics. Invalid inputs never receive scores."""
    validate_song(song)
    report = {
        "schema": "choreobench.plan_evaluation.v1", "song_id": song["song_id"],
        "plan_sha256": object_sha(plan), "song_sha256": object_sha(song),
        "valid": False, "errors": [], "coverage": None,
        "music_alignment": {"status": "not_measured"},
        "physical_quality": {"status": "not_measured"},
        "semantic_quality": {"status": "pending_independent_review"},
    }
    try:
        validate_plan(plan, song)
    except (ValidationError, TypeError, KeyError) as error:
        report["errors"] = [str(error)]
        return report
    report["valid"] = True
    report["coverage"] = _coverage(plan["events"], plan["num_samples"])
    tracks = sorted({event["track"] for event in plan["events"]})
    report["track_coverage"] = {
        track: _coverage([event for event in plan["events"] if event["track"] == track], plan["num_samples"])
        for track in tracks
    }
    motif_counts = Counter(event["motif_id"] for event in plan["events"] if event["motif_id"] is not None)
    action_counts = Counter(" ".join(event["action"].casefold().split()) for event in plan["events"])
    report["literal_diagnostics"] = {
        "event_count": len(plan["events"]), "motif_event_counts": dict(sorted(motif_counts.items())),
        "normalized_action_counts": dict(sorted(action_counts.items())),
        "note": "Text repetition and declared motif IDs do not establish semantic repetition or motif quality.",
    }
    return report


def _literal_goal(event, goal):
    actual = event_value(event, goal["field"])
    expected = goal["value"]
    if actual is _MISSING:
        return False
    if goal["op"] == "eq":
        return literal_equal(actual, expected)
    if goal["op"] == "contains":
        # Deliberately literal, case-sensitive substring; never a semantic classifier.
        return isinstance(actual, str) and expected in actual
    numeric = not isinstance(actual, bool) and isinstance(actual, (int, float))
    if not numeric:
        return False
    return actual >= expected if goal["op"] == "gte" else actual <= expected


def evaluate_edit(before, after, task):
    """Evaluate a frozen private task. Semantic goals remain pending external review."""
    validate_edit_task(task, before)
    report = {
        "schema": "choreobench.edit_evaluation.v1", "task_id": task["task_id"],
        "song_id": task["song_id"], "task_sha256": object_sha(task),
        "before_sha256": object_sha(before), "after_sha256": object_sha(after),
        "after_valid": False, "errors": [], "literal_goals": [], "protections": [],
        "literal_goal_satisfaction_rate": None, "protection_preservation_rate": None,
        "literal_goals_status": "not_requested" if not task["goals"] else "not_evaluated",
        "literal_joint_success": False,
        "semantic_goals": [{"goal": goal, "status": "pending_independent_review"}
                           for goal in task["semantic_goals"]],
        "semantic_status": "pending_independent_review" if task["semantic_goals"] else "not_requested",
        "overall_status": "literal_fail", "boundary_quality": {"status": "not_measured"},
        "music_alignment": {"status": "not_measured"},
        "physical_quality": {"status": "not_measured"},
    }
    try:
        validate_plan(after)
    except (ValidationError, KeyError, TypeError) as error:
        report["errors"] = [str(error)]
        return report
    report["after_valid"] = True
    immutable = ("schema", "song_id", "sample_rate", "num_samples", "motifs")
    changed_immutable = [field for field in immutable if not literal_equal(before[field], after[field])]
    before_events = {event["event_id"]: event for event in before["events"]}
    after_events = {event["event_id"]: event for event in after["events"]}
    old_ids, new_ids = set(before_events), set(after_events)
    changed = sorted(event_id for event_id in old_ids | new_ids
                     if not literal_equal(before_events.get(event_id, _MISSING), after_events.get(event_id, _MISSING)))
    allowed, repair = set(task["allowed_event_ids"]), set(task["repair_event_ids"])
    out_of_scope = sorted(set(changed) - allowed - repair)
    protection_rows = protection_results(before, after, task["protected"])
    goal_rows = []
    for goal in task["goals"]:
        actual = event_value(after_events.get(goal["event_id"], {}), goal["field"])
        goal_rows.append({
            "event_id": goal["event_id"], "field": goal["field"], "op": goal["op"],
            "expected": copy.deepcopy(goal["value"]), "observed": None if actual is _MISSING else copy.deepcopy(actual),
            "observed_missing": actual is _MISSING,
            "satisfied": _literal_goal(after_events.get(goal["event_id"], {}), goal),
            "interpretation": "literal JSON check, not semantic truth",
        })
    goal_rate = sum(row["satisfied"] for row in goal_rows) / len(goal_rows) if goal_rows else None
    protection_rate = sum(row["preserved"] for row in protection_rows) / len(protection_rows) if protection_rows else None
    # Evidence is immutable in patch v1, even on allowed events.
    changed_evidence = sorted(event_id for event_id in old_ids & new_ids
                              if not literal_equal(before_events[event_id]["evidence"], after_events[event_id]["evidence"]))
    scope_valid = not out_of_scope and old_ids == new_ids and not changed_immutable and not changed_evidence
    literal_pass = scope_valid and all(row["satisfied"] for row in goal_rows) and all(row["preserved"] for row in protection_rows)
    # Preserving constraints is meaningful even without a target, but all([])
    # cannot turn a task with no goals into successful editing. A no-op with
    # explicit goals already satisfied remains a valid literal success.
    literal_success = literal_pass if goal_rows or not literal_pass else None
    if not literal_pass:
        overall_status = "literal_fail"
    elif task["semantic_goals"]:
        overall_status = "pending_independent_review"
    elif goal_rows:
        overall_status = "literal_pass"
    else:
        overall_status = "not_evaluable"
    report.update({
        "literal_goals": goal_rows, "protections": protection_rows,
        "literal_goal_satisfaction_rate": goal_rate, "protection_preservation_rate": protection_rate,
        "literal_goals_status": "evaluated" if goal_rows else "not_requested",
        "changed_event_ids": changed, "out_of_scope_event_ids": out_of_scope,
        "repair_changed_event_ids": sorted((set(changed) & repair) - allowed),
        "changed_immutable_fields": changed_immutable, "changed_evidence_event_ids": changed_evidence,
        "event_identity_preserved": old_ids == new_ids,
        "scope_valid": scope_valid, "no_op": not changed and not changed_immutable,
        "literal_joint_success": literal_success,
        "overall_status": overall_status,
        "note": ("Literal success does not establish action semantics, musicality, executability, or boundary quality. "
                 "Without literal goals, joint literal success is NA unless a constraint fails. "
                 "A no-op alone does not establish successful editing or correct abstention."),
    })
    return report


def _percentile(values, quantile):
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def song_bootstrap(rows, metric="value", n_resamples=2000, confidence=0.95, seed=0):
    """Equal-song mean and percentile CI. First average all rows within each song.

    For paired comparisons pass within-pair differences, retaining the song_id.
    This is descriptive uncertainty, not a significance or release gate.
    """
    if not isinstance(rows, list) or not isinstance(metric, str) or not metric:
        raise ValidationError("rows must be a list and metric must name a numeric field")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValidationError("n_resamples must be a positive integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValidationError("confidence must be between zero and one")
    if type(seed) is not int:
        raise ValidationError("seed must be an integer")
    grouped = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("song_id"), str) or not row["song_id"].strip():
            raise ValidationError("every bootstrap row requires song_id")
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError("bootstrap values must be finite numbers; missing values cannot be dropped silently")
        grouped.setdefault(row["song_id"], []).append(value)
    song_values = [(song_id, mean(values)) for song_id, values in sorted(grouped.items())]
    report = {
        "schema": "choreobench.song_bootstrap.v1", "metric": metric,
        "n_rows": len(rows), "n_songs": len(song_values), "n_resamples": n_resamples,
        "confidence": confidence, "seed": seed, "estimate": None, "interval": None,
        "status": "insufficient_songs", "resampling_unit": "song",
        "aggregation": "mean within song, then equally weighted mean across songs",
        "song_means": [{"song_id": song_id, "value": value} for song_id, value in song_values],
        "note": "Percentile bootstrap; not an automatic significance test or scientific quality gate.",
    }
    if not song_values:
        return report
    values = [value for _, value in song_values]
    report["estimate"] = mean(values)
    if len(values) < 2:
        return report
    rng = random.Random(seed)
    samples = sorted(mean(values[rng.randrange(len(values))] for _ in values)
                     for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    report["interval"] = [_percentile(samples, alpha), _percentile(samples, 1 - alpha)]
    report["status"] = "computed"
    return report
