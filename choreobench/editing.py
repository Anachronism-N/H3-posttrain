"""Bounded, literal plan editing. No model calls and no in-place mutation."""

import copy
import math
import re

from .io import object_sha
from .schema import TRACKS, ValidationError, validate_plan


PATCH_SCHEMA = "choreobench.patch.v1"
TASK_SCHEMA = "choreobench.edit_task.v1"
_PATCH_KEYS = {
    "schema", "base_plan_sha256", "instruction", "allowed_event_ids",
    "repair_event_ids", "protected", "operations",
}
_TASK_KEYS = {
    "schema", "task_id", "song_id", "source_plan_sha256", "instruction",
    "allowed_event_ids", "repair_event_ids", "protected", "goals", "semantic_goals",
}
_SCALAR_FIELDS = {"action", "start", "end", "track", "motif_id"}
_OBJECT_FIELDS = {"attributes", "entry", "exit"}
_MISSING = object()


def _fail(message):
    raise ValidationError(message)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail("{} must contain exactly {}".format(label, sorted(expected)))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        _fail("{} must be a nonempty string".format(label))


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail("{} must be a lowercase SHA256".format(label))


def _ids(value, label):
    if not isinstance(value, list):
        _fail("{} must be a list".format(label))
    for item in value:
        _text(item, label + " item")
    if len(set(value)) != len(value):
        _fail("{} contains duplicate IDs".format(label))


def _field(field, whole_objects=True):
    """Return (root, scalar key or None). Object keys may themselves contain dots."""
    if not isinstance(field, str):
        _fail("field must be a string")
    if field in _SCALAR_FIELDS or (whole_objects and field in _OBJECT_FIELDS):
        return field, None
    if "." in field:
        root, key = field.split(".", 1)
        if root in _OBJECT_FIELDS and key:
            return root, key
    _fail("unsupported event field: {!r}".format(field))


def event_value(event, field):
    """Read a supported field; absent scalar keys are represented internally only."""
    root, key = _field(field)
    if root not in event:
        return _MISSING
    if key is None:
        return event[root]
    container = event[root]
    if not isinstance(container, dict):
        return _MISSING
    return container.get(key, _MISSING)


def literal_equal(left, right):
    """JSON equality without Python's True == 1 coercion."""
    if left is _MISSING or right is _MISSING:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(literal_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(literal_equal(a, b) for a, b in zip(left, right))
    return left == right


def _json_value(value):
    # Hashing also checks representability, but explicitly reject nonfinite numbers here.
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    _fail("value must be finite JSON")


def _protection_rows(rows, events=None):
    if not isinstance(rows, list):
        _fail("protected must be a list")
    seen = set()
    for row in rows:
        _keys(row, {"event_id", "field"}, "protection")
        _text(row["event_id"], "protected event_id")
        _field(row["field"])
        identity = (row["event_id"], row["field"])
        if identity in seen:
            _fail("duplicate protection")
        seen.add(identity)
        if events is not None and row["event_id"] != "*" and row["event_id"] not in events:
            _fail("protection references an unknown event")


def _scope(contract, events=None):
    _text(contract["instruction"], "instruction")
    _ids(contract["allowed_event_ids"], "allowed_event_ids")
    _ids(contract["repair_event_ids"], "repair_event_ids")
    _protection_rows(contract["protected"], events)
    if events is not None:
        referenced = set(contract["allowed_event_ids"]) | set(contract["repair_event_ids"])
        if not referenced.issubset(events):
            _fail("edit scope references an unknown event")


def validate_edit_task(task, plan=None):
    """Validate a private task, optionally binding it to its exact source plan."""
    _keys(task, _TASK_KEYS, "edit task")
    if task["schema"] != TASK_SCHEMA:
        _fail("unsupported edit task schema")
    _text(task["task_id"], "task_id")
    _text(task["song_id"], "song_id")
    _sha(task["source_plan_sha256"], "source_plan_sha256")
    events = None
    if plan is not None:
        validate_plan(plan)
        if task["song_id"] != plan["song_id"] or task["source_plan_sha256"] != object_sha(plan):
            _fail("edit task does not bind to this source plan")
        events = {event["event_id"]: event for event in plan["events"]}
    _scope(task, events)
    if not isinstance(task["goals"], list):
        _fail("goals must be a list")
    for goal in task["goals"]:
        _keys(goal, {"event_id", "field", "op", "value"}, "goal")
        _text(goal["event_id"], "goal event_id")
        _field(goal["field"])
        if not isinstance(goal["op"], str) or goal["op"] not in {"eq", "contains", "gte", "lte"}:
            _fail("unsupported literal goal operator")
        _json_value(goal["value"])
        if goal["op"] == "contains" and not isinstance(goal["value"], str):
            _fail("contains goals require a string value")
        if goal["op"] in {"gte", "lte"} and (
            isinstance(goal["value"], bool) or not isinstance(goal["value"], (int, float))
        ):
            _fail("numeric goals require a finite numeric value")
        if events is not None and goal["event_id"] not in events:
            _fail("goal references an unknown event")
    if not isinstance(task["semantic_goals"], list):
        _fail("semantic_goals must be a list")
    for goal in task["semantic_goals"]:
        _text(goal, "semantic goal")
    return task


def public_edit_task(task):
    """Return a detached allowlist view. Private answers never reach the planner."""
    validate_edit_task(task)
    return copy.deepcopy({
        "schema": "choreobench.public_edit_task.v1",
        "task_id": task["task_id"],
        "song_id": task["song_id"],
        "source_plan_sha256": task["source_plan_sha256"],
        "instruction": task["instruction"],
        "allowed_event_ids": task["allowed_event_ids"],
        "repair_event_ids": task["repair_event_ids"],
        "protected": task["protected"],
    })


def protection_results(before, after, protected):
    """Expand wildcard protections and compare literal JSON values, including absence."""
    before_events = {event["event_id"]: event for event in before["events"]}
    after_events = {event["event_id"]: event for event in after.get("events", [])}
    rows = []
    seen = set()
    for protection in protected:
        event_ids = sorted(before_events) if protection["event_id"] == "*" else [protection["event_id"]]
        for event_id in event_ids:
            identity = (event_id, protection["field"])
            if identity in seen:
                continue
            seen.add(identity)
            original = event_value(before_events[event_id], protection["field"])
            current = event_value(after_events.get(event_id, {}), protection["field"])
            rows.append({
                "event_id": event_id, "field": protection["field"],
                "preserved": event_id in after_events and literal_equal(original, current),
            })
    return rows


def apply_patch(plan, patch):
    """Apply an exact-version patch transactionally; reject every invalid operation."""
    validate_plan(plan)
    _keys(patch, _PATCH_KEYS, "patch")
    if patch["schema"] != PATCH_SCHEMA:
        _fail("unsupported patch schema")
    _sha(patch["base_plan_sha256"], "base_plan_sha256")
    if patch["base_plan_sha256"] != object_sha(plan):
        _fail("patch base_plan_sha256 does not match the source plan")
    events = {event["event_id"]: event for event in plan["events"]}
    _scope(patch, events)
    if not isinstance(patch["operations"], list):
        _fail("operations must be a list")
    permitted = set(patch["allowed_event_ids"]) | set(patch["repair_event_ids"])
    result = copy.deepcopy(plan)
    result_events = {event["event_id"]: event for event in result["events"]}
    seen = set()
    for operation in patch["operations"]:
        _keys(operation, {"event_id", "field", "value"}, "operation")
        event_id = operation["event_id"]
        _text(event_id, "operation event_id")
        if event_id not in permitted or event_id not in events:
            _fail("operation targets an event outside the allowed/repair scope")
        root, key = _field(operation["field"], whole_objects=False)
        identity = (event_id, operation["field"])
        if identity in seen:
            _fail("duplicate operation for the same event field")
        seen.add(identity)
        _json_value(operation["value"])
        value = operation["value"]
        if key is not None and value is not None and type(value) not in (str, bool, int, float):
            _fail("attribute/state values must be JSON scalars")
        if key is None:
            if root == "action":
                _text(value, "operation action")
            elif root in {"start", "end"} and (type(value) is not int or value < 0):
                _fail("operation times must be nonnegative integer samples")
            elif root == "track" and (not isinstance(value, str) or value not in TRACKS):
                _fail("operation track is unsupported")
            elif root == "motif_id" and value is not None and not isinstance(value, str):
                _fail("operation motif_id must be a string or null")
        target = result_events[event_id]
        if key is None:
            if root not in target:
                _fail("operations can only replace existing event fields")
            target[root] = copy.deepcopy(operation["value"])
        else:
            if root not in target or key not in target[root]:
                _fail("operations cannot add an attribute/state key")
            target[root][key] = copy.deepcopy(operation["value"])
    if any(not row["preserved"] for row in protection_results(plan, result, patch["protected"])):
        _fail("patch changes a protected field")
    # Retiming may alter list order. Execution canonicalizes the output, never the source.
    result["events"].sort(key=lambda event: (event["start"], event["track"], event["event_id"]))
    result["provenance"]["choreobench_edit"] = {
        "base_plan_sha256": object_sha(plan), "patch_sha256": object_sha(patch),
    }
    validate_plan(result)
    return result


def apply_task_patch(plan, patch, task):
    """Apply a model proposal only if it preserves the benchmark's frozen contract."""
    validate_edit_task(task, plan)
    _keys(patch, _PATCH_KEYS, "patch")
    if patch["instruction"] != task["instruction"]:
        _fail("patch rewrites the frozen edit instruction")
    _scope(patch, {event["event_id"]: event for event in plan["events"]})
    for name in ("allowed_event_ids", "repair_event_ids"):
        if set(patch[name]) != set(task[name]):
            _fail("patch changes the frozen " + name)
    normalize = lambda rows: sorted((row["event_id"], row["field"]) for row in rows)
    if normalize(patch["protected"]) != normalize(task["protected"]):
        _fail("patch changes the frozen protected fields")
    return apply_patch(plan, patch)
