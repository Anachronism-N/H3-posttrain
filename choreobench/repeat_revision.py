"""Rewrite duplicated action sentences in a committed plan (action-only).

The whole-song planner often reuses one sentence for consecutive blocks. This
pass finds groups of events whose action text is identical after normalisation,
keeps the first occurrence, and asks the model to rewrite the remaining ones as
variations of the same theme. Only `action` may change; `apply_patches`
re-validates the plan and rejects any other difference.

Not a quality judge: it removes verbatim copying, nothing more.
"""
import copy
import json
import re

from .schema import keys, text, validate_plan
from .whole_song_revision import apply_patches

INSTRUCTION = (
    "Rewrite repeated movement sentences so that a returning theme reads as variation, not copying. "
    "Keep the same body parts, the same base movement vocabulary and the stated start/end; change at "
    "least one of direction, level (high/mid/low), concurrency, or which limb leads, so the return is "
    "still recognisable as the same theme. Paraphrasing a few words is not enough: change the spatial "
    "path, the level sequence, or the timing of the movement. Do not add props, partners or scenery. "
    "Return only {\"rewrites\":[{\"event_id\":\"...\",\"action\":\"...\"}]} covering exactly the "
    "requested event ids."
)

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise(action):
    return " ".join(_PUNCT.sub(" ", action.lower()).split())


def duplicate_groups(plan, min_count=2, similarity=0.80):
    """Cluster events whose sentences are identical or near-identical (Jaccard >= threshold)."""
    groups = []
    for event in plan.get("events", []):
        placed = False
        for group in groups:
            if overlap(event["action"], group["text"]) >= similarity:
                group["event_ids"].append(event["event_id"])
                placed = True
                break
        if not placed:
            groups.append(dict(text=event["action"], event_ids=[event["event_id"]]))
    return [g for g in groups if len(g["event_ids"]) >= min_count]


def _view(plan, group, limit=180):
    by_id = {e["event_id"]: e for e in plan["events"]}
    rows = []
    for index, event_id in enumerate(group["event_ids"]):
        event = by_id[event_id]
        position = [e["event_id"] for e in plan["events"]].index(event_id)
        previous = plan["events"][position - 1]["action"][:limit] if position else None
        following = plan["events"][position + 1]["action"][:limit] if position + 1 < len(plan["events"]) else None
        rows.append(dict(event_id=event_id, occurrence=index + 1, start=event["start"], end=event["end"],
                         motif_id=event.get("motif_id"), previous_action=previous, next_action=following,
                         rewrite_this=(index > 0)))
    return rows


def rewrite_request(plan, group, cfg, call_id):
    payload = dict(song_id=plan["song_id"], repeated_action=group["text"][:400],
                   occurrences=_view(plan, group), note="occurrence 1 is kept; rewrite the rest")
    return dict(call_id=call_id, seed=cfg["seed"], max_output_tokens=cfg["max_output_tokens"],
                response_kind="repeat_rewrite",
                messages=[dict(role="system", content=INSTRUCTION),
                          dict(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True))],
                response_schema=dict(type="object", additionalProperties=False, required=["rewrites"],
                                     properties=dict(rewrites=dict(type="array", items=dict(
                                         type="object", additionalProperties=False,
                                         required=["event_id", "action"],
                                         properties=dict(event_id=dict(type="string", minLength=1),
                                                         action=dict(type="string", minLength=1, maxLength=600)))))))


def _words(text):
    return set(normalise(text).split())


def overlap(a, b):
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def read_rewrites(response, group, min_overlap=0.20, max_similarity=0.80):
    """Accept valid rewrites individually; report skipped rows instead of failing the group."""
    keys(response, ("rewrites",), where="repeat rewrite")
    targets = set(group["event_ids"][1:])
    rows = response["rewrites"]
    if not isinstance(rows, list):
        raise ValueError("Repeat rewrite must be a list")
    seen, accepted, skipped = [], [], []
    for row in rows:
        keys(row, ("event_id", "action"), where="repeat rewrite row")
        event_id = row["event_id"]
        if event_id not in targets:
            raise ValueError("Repeat rewrite targets an unexpected event: " + str(event_id))
        if event_id in seen:
            raise ValueError("Repeat rewrite repeats an event: " + str(event_id))
        seen.append(event_id)
        action = row["action"]
        text(action, "rewritten action")
        if len(action) > 600:
            skipped.append(dict(event_id=event_id, reason="length"))
        elif normalise(action) == normalise(group["text"]):
            skipped.append(dict(event_id=event_id, reason="unchanged"))
        elif overlap(action, group["text"]) < min_overlap:
            skipped.append(dict(event_id=event_id, reason="vocabulary_drift"))
        elif overlap(action, group["text"]) > max_similarity:
            skipped.append(dict(event_id=event_id, reason="near_duplicate"))
        else:
            accepted.append(dict(event_id=event_id, action=action))
    return accepted, skipped


def apply_rewrites(plan, rewrites):
    final, changed = apply_patches(plan, rewrites)
    validate_plan(final)
    return final, changed


def summary(before, after):
    def dup(plan):
        events = plan["events"]
        return 1 - len({normalise(e["action"]) for e in events}) / len(events) if events else 0.0
    return dict(duplicate_rate_before=round(dup(before), 3), duplicate_rate_after=round(dup(after), 3))
