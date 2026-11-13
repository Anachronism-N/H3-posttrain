"""Strict frozen-draft revision messages and action-only internal edit scopes.

Diagnoses and verdicts are model proposals for an internal revision experiment.
They are neither independent benchmark annotations nor measured quality scores.
No model calls, private evaluation goals, retries or in-place changes occur here.
"""

import copy
import random

from .editing import TASK_SCHEMA, validate_edit_task
from .io import object_sha
from .providers import parse_json_response
from .schema import ValidationError, identifier, keys, text, validate_plan


DIAGNOSIS_SCHEMA = "choreobench.revision_diagnosis.v1"
VERDICT_SCHEMA = "choreobench.revision_verdict.v1"
RANDOM_REVIEW_REASON = (
    "Consider whether this event benefits from an action-only revision; "
    "keep it unchanged if no revision is justified."
)
_PROTECTED_FIELDS = ("start", "end", "track", "motif_id", "attributes", "entry", "exit")


def _reason(value, where):
    text(value, where)
    if len(value) > 2000:
        raise ValidationError(where + " must contain at most 2000 characters")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValidationError(where + " must be valid UTF-8 text") from error


def _diagnosis(value, plan):
    validate_plan(plan)
    keys(value, ("schema", "source_plan_sha256", "issues"), where="revision diagnosis")
    if value["schema"] != DIAGNOSIS_SCHEMA:
        raise ValidationError("Unsupported revision diagnosis schema")
    if value["source_plan_sha256"] != object_sha(plan):
        raise ValidationError("Revision diagnosis does not bind the frozen source plan")
    issues = value["issues"]
    if not isinstance(issues, list) or len(issues) > len(plan["events"]):
        raise ValidationError("Diagnosis issues must be a list no longer than the source event list")
    events, seen = {event["event_id"] for event in plan["events"]}, set()
    for issue in issues:
        keys(issue, ("event_id", "priority", "reason"), where="revision issue")
        event_id = issue["event_id"]
        identifier(event_id, "revision issue event_id")
        if event_id not in events or event_id in seen:
            raise ValidationError("Diagnosis event IDs must be unique and present in the frozen source plan")
        seen.add(event_id)
        if type(issue["priority"]) is not int or not 1 <= issue["priority"] <= 3:
            raise ValidationError("Issue priority must be an integer in [1,3]")
        _reason(issue["reason"], "revision issue reason")
    return value


def parse_diagnosis(text, plan):
    """Parse exact JSON and bind every diagnosed event to the original draft."""
    return copy.deepcopy(_diagnosis(parse_json_response(text), plan))


def select_events(plan, diagnosis, policy, max_repairs, seed):
    """Freeze a matched-size selection from the original plan, without mutation.

    Targeted selection ranks priority descending, then original event order.
    Random selection uses the same diagnosed-count budget but samples ALL source
    events. Its selection rows carry neutral reasons; the runner supplies the
    same complete frozen diagnosis to both arms when asking for revisions.
    Both policies execute their selected set in original event order to avoid
    making revision order another intervention. Empty diagnoses yield zero
    selections for both policies, rather than inventing repair targets.
    """
    _diagnosis(diagnosis, plan)
    if not isinstance(policy, str) or policy not in ("targeted", "random"):
        raise ValidationError("Revision selection policy must be targeted or random")
    if type(max_repairs) is not int or not 1 <= max_repairs <= 8:
        raise ValidationError("max_repairs must be an integer in [1,8]")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValidationError("seed must be an integer in [0,2**63)")
    n = min(max_repairs, len(diagnosis["issues"]))
    order = {event["event_id"]: index for index, event in enumerate(plan["events"])}
    if policy == "targeted":
        chosen = sorted(diagnosis["issues"], key=lambda issue: (-issue["priority"], order[issue["event_id"]]))[:n]
        return copy.deepcopy(sorted(chosen, key=lambda issue: order[issue["event_id"]]))
    positions = sorted(random.Random(seed).sample(range(len(plan["events"])), n))
    return [{"event_id": plan["events"][position]["event_id"], "priority": None,
             "reason": RANDOM_REVIEW_REASON} for position in positions]


def action_task(plan, event_id, instruction):
    """Bind a task permitting only one existing event's action text to change.

    Empty goal lists deliberately provide no claimed semantic success. Every
    other event field is protected globally; other event actions are outside the
    allowlist. Existing editing code also excludes evidence/top-level mutation.
    """
    validate_plan(plan)
    identifier(event_id, "revision event_id")
    if event_id not in {event["event_id"] for event in plan["events"]}:
        raise ValidationError("Revision task event is absent from the source plan")
    text(instruction, "revision instruction")
    source_sha = object_sha(plan)
    task = {
        "schema": TASK_SCHEMA,
        "task_id": "revision-action-" + object_sha({"source_plan_sha256": source_sha,
                                                    "event_id": event_id, "instruction": instruction}),
        "song_id": plan["song_id"], "source_plan_sha256": source_sha,
        "instruction": instruction, "allowed_event_ids": [event_id], "repair_event_ids": [],
        "protected": [{"event_id": "*", "field": field} for field in _PROTECTED_FIELDS],
        "goals": [], "semantic_goals": [],
    }
    validate_edit_task(task, plan)
    return task


def public_plan(plan):
    """Detach a model-input view, excluding provenance and source evidence."""
    validate_plan(plan)
    result = copy.deepcopy(plan)
    result["provenance"] = {}
    for event in result["events"]:
        event["evidence"] = []
    return result


def parse_verdict(text, before, after, event_id):
    """Validate an internal candidate-choice verdict, never a quality verdict."""
    validate_plan(before)
    validate_plan(after)
    identifier(event_id, "verdict event_id")
    if any(event_id not in {event["event_id"] for event in plan["events"]} for plan in (before, after)):
        raise ValidationError("Verdict event must exist in both bound plans")
    if any(before[field] != after[field] for field in ("song_id", "sample_rate", "num_samples")):
        raise ValidationError("Verdict plans must identify the same song and clock")
    value = parse_json_response(text)
    keys(value, ("schema", "before_plan_sha256", "after_plan_sha256", "event_id", "decision", "reason"),
         where="revision verdict")
    if value["schema"] != VERDICT_SCHEMA:
        raise ValidationError("Unsupported revision verdict schema")
    if (value["before_plan_sha256"] != object_sha(before) or
            value["after_plan_sha256"] != object_sha(after) or value["event_id"] != event_id):
        raise ValidationError("Revision verdict does not bind its exact plans and event")
    if not isinstance(value["decision"], str) or value["decision"] not in (
            "prefer_revision", "keep_original", "uncertain"):
        raise ValidationError("Invalid internal revision decision")
    _reason(value["reason"], "revision verdict reason")
    return copy.deepcopy(value)
