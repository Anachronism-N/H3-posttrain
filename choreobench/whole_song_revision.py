"""Module R: one whole-song audit plus a bounded, targeted action patch.

The draft is frozen first. The audit may name at most three concrete problems
with an exact quote; one more call then proposes action-text-only patches for
those events. Any failure keeps the untouched draft and records why. Structure
checks here never imply semantic or musical quality.
"""
import copy
import json
import time
from pathlib import Path

from .io import file_sha, object_sha, read_json, write_json
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .schema import keys, text, validate_plan

SCHEMA = "choreobench.whole_song_revision.v1"
PROBLEM_TYPES = ("mechanical_repeat", "continuity_conflict", "quantity_mismatch")
MUSIC_PROBLEMS = ("quantity_mismatch",)
MAX_ISSUES = 3
MAX_PATCHES = 3
ACTION_LIMIT = 1600

AUDIT_SYSTEM = (
    "Audit one complete text choreography score against its own plan and the supplied acoustic summary. "
    "You have not heard audio directly; the summary is fixed-parameter acoustic measurement and its "
    "possible_recurrence/possible_contrast entries are similarity guesses, not detected sections. "
    "Report at most three CONCRETE problems of exactly these types: "
    "mechanical_repeat (a repeated music region copied verbatim, or expected motif relation lost); "
    "continuity_conflict (an obvious action/timing continuity contradiction between adjacent events); "
    "quantity_mismatch (movement amount clearly contradicts the section's energy/onset trend). "
    "Sustained movement and intentional motif return are NOT problems. "
    'Return exactly {"issues":[{"event_id":...,"quote":...,"problem_type":...,"target":...,"music_block":...}]}. '
    "quote must be an exact substring of that event's action. target states the concrete change wanted. "
    "music_block is required for quantity_mismatch and must be a supplied block_id; use null otherwise. "
    "Return an empty list when nothing is clearly wrong. Do not propose new events, timings or ids."
)

PATCH_SYSTEM = (
    "Propose action-text patches for exactly the listed audit issues. "
    'Return exactly {"patches":[{"event_id":...,"action":...}]}. '
    "Only the listed event_ids; never add/remove events, never change ids, times, order, track, motif or evidence. "
    "Keep each action a concrete movement phrase that fits its existing duration; do not make it longer or busier "
    "than the duration supports. Do not claim support-foot, facing or contact state. "
    "The new action MUST differ from the current action for that event, and must address the named problem "
    "(for mechanical_repeat, change the movement content rather than appending adverbs; for missing_motif, "
    "introduce the named motif's movement). If you cannot change it safely, omit that event_id rather than guessing."
)


def _event_view(plan, sample_rate):
    return [dict(event_id=e["event_id"], start=e["start"], end=e["end"],
                 start_seconds=round(e["start"] / sample_rate, 3), end_seconds=round(e["end"] / sample_rate, 3),
                 duration_seconds=round((e["end"] - e["start"]) / sample_rate, 3),
                 track=e["track"], action=e["action"], motif_id=e.get("motif_id"))
            for e in plan["events"]]


def _section_of(plan):
    return dict(song_id=plan["song_id"], sample_rate=plan["sample_rate"], num_samples=plan["num_samples"],
                motifs=plan.get("motifs", []))


def audit_request(plan, outline, summary, cfg, call_id):
    payload = dict(
        score=_event_view(plan, plan["sample_rate"]), score_meta=_section_of(plan),
        section_design=[{k: s[k] for k in ("section_id", "intent", "motif_ids")} for s in outline["sections"]],
        music_summary=dict(blocks=summary["blocks"], relations=summary["relations"],
                           functional_labels_verified=False, beat_or_meter_assumed=False),
        problem_types=list(PROBLEM_TYPES), max_issues=MAX_ISSUES,
        note="Sustained movement and intentional motif return are not problems.")
    schema = dict(type="object", additionalProperties=False, required=["issues"], properties=dict(issues=dict(
        type="array", maxItems=MAX_ISSUES, items=dict(type="object", additionalProperties=False,
            required=["event_id", "quote", "problem_type", "target", "music_block"],
            properties=dict(event_id=dict(type="string", enum=[e["event_id"] for e in plan["events"]]),
                            quote=dict(type="string", minLength=1, maxLength=ACTION_LIMIT),
                            problem_type=dict(type="string", enum=list(PROBLEM_TYPES)),
                            target=dict(type="string", minLength=1, maxLength=600),
                            music_block=dict(type="string", minLength=1))))))
    request = dict(call_id=call_id, seed=cfg["seed"], max_output_tokens=2048, response_kind="revision_audit",
                   messages=[dict(role="system", content=AUDIT_SYSTEM),
                             dict(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                                  allow_nan=False))],
                   response_schema=schema)
    return guard_request(request, cfg)


def _block_ids(summary):
    return {b["block_id"] for b in summary["blocks"]}


def read_audit(response, plan, summary):
    from .baseline import ModelOutputError
    try:
        value = parse_json_response(response)
        keys(value, ("issues",), where="revision audit")
        issues = value["issues"]
        if not isinstance(issues, list) or len(issues) > MAX_ISSUES:
            raise ValueError("Audit must return at most three issues")
        actions = {e["event_id"]: e["action"] for e in plan["events"]}
        blocks = _block_ids(summary)
        seen = set()
        clean = []
        for item in issues:
            keys(item, ("event_id", "quote", "problem_type", "target"), optional=("music_block",),
                 where="audit issue")
            item.setdefault("music_block", None)
            if item["event_id"] not in actions or item["event_id"] in seen:
                raise ValueError("Audit names an unknown or duplicated event")
            if not isinstance(item["quote"], str) or not item["quote"] or item["quote"] not in actions[item["event_id"]]:
                raise ValueError("Audit quote is not an exact substring of the event action")
            if item["problem_type"] not in PROBLEM_TYPES:
                raise ValueError("Unknown problem type")
            text(item["target"], "issue target")
            if len(item["target"]) > 600:
                raise ValueError("Issue target too long")
            if item["problem_type"] in MUSIC_PROBLEMS:
                if item["music_block"] not in blocks:
                    raise ValueError("Music problem must cite a supplied music block")
            elif item["music_block"] not in (None, ""):
                # A non-music issue may mention a block; keep it only if it exists.
                if item["music_block"] not in blocks:
                    raise ValueError("Cited music block does not exist")
                item["music_block"] = None
            seen.add(item["event_id"])
            clean.append(dict(item, music_block=item["music_block"] or None))
        return clean
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def patch_request(plan, issues, cfg, call_id):
    payload = dict(issues=issues,
                   targets=[dict(event_id=e["event_id"], start=e["start"], end=e["end"],
                                 duration_seconds=round((e["end"] - e["start"]) / plan["sample_rate"], 3),
                                 track=e["track"], action=e["action"], motif_id=e.get("motif_id"))
                            for e in plan["events"] if e["event_id"] in {i["event_id"] for i in issues}],
                   constraints=dict(action_only=True, max_patches=MAX_PATCHES,
                                    keep_ids_times_order_track_motif_evidence=True))
    schema = dict(type="object", additionalProperties=False, required=["patches"], properties=dict(patches=dict(
        type="array", maxItems=MAX_PATCHES, items=dict(type="object", additionalProperties=False,
            required=["event_id", "action"],
            properties=dict(event_id=dict(type="string", enum=sorted({i["event_id"] for i in issues})),
                            action=dict(type="string", minLength=1, maxLength=ACTION_LIMIT))))))
    request = dict(call_id=call_id, seed=cfg["seed"], max_output_tokens=2048, response_kind="revision_patch",
                   messages=[dict(role="system", content=PATCH_SYSTEM),
                             dict(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                                  allow_nan=False))],
                   response_schema=schema)
    return guard_request(request, cfg)


def read_patches(response, issues):
    from .baseline import ModelOutputError
    try:
        value = parse_json_response(response)
        keys(value, ("patches",), where="revision patch")
        patches = value["patches"]
        if not isinstance(patches, list) or len(patches) > MAX_PATCHES:
            raise ValueError("At most three patches are allowed")
        allowed = {i["event_id"] for i in issues}
        seen = set()
        clean = []
        for item in patches:
            keys(item, ("event_id", "action"), where="patch")
            if item["event_id"] not in allowed or item["event_id"] in seen:
                raise ValueError("Patch targets a non-listed or duplicated event")
            text(item["action"], "patch action")
            if len(item["action"]) > ACTION_LIMIT:
                raise ValueError("Patch action exceeds the action character limit")
            seen.add(item["event_id"])
            clean.append(dict(item))
        return clean
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def apply_patches(plan, patches):
    """Apply action-only patches on a copy and verify nothing else changed."""
    final = copy.deepcopy(plan)
    by_id = {e["event_id"]: e for e in final["events"]}
    changed = []
    for patch in patches:
        if patch["event_id"] not in by_id:
            raise ValueError("Patch event missing from the plan")
        if patch["action"] == by_id[patch["event_id"]]["action"]:
            continue  # an unchanged "patch" is not a change and is reported as such
        by_id[patch["event_id"]]["action"] = patch["action"]
        changed.append(patch["event_id"])
    validate_plan(final)
    original = {e["event_id"]: e for e in plan["events"]}
    for event in final["events"]:
        before = original[event["event_id"]]
        for field in ("start", "end", "track", "motif_id", "evidence", "attributes", "entry", "exit"):
            if event.get(field) != before.get(field):
                raise ValueError("Patch changed a non-target field: " + field)
        if event["event_id"] not in changed and event["action"] != before["action"]:
            raise ValueError("Patch changed a non-target event")
    return final, changed


def _call(provider, request, journal_dir, index):
    journal_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    response = provider.generate(request)
    write_json(journal_dir / ("%04d.request.json" % index), request)
    write_json(journal_dir / ("%04d.response.json" % index), response)
    if response.get("stop_reason") not in ("stop", "eos"):
        raise ValueError("Revision model did not finish normally: " + str(response.get("stop_reason")))
    return response, round(time.monotonic() - started, 2)


def run_revision(draft_run, provider_spec, summary_path, out_dir, *, song, outline, seed=42,
                 max_input_tokens=2000000, max_total_output_tokens=150000, max_calls=1 + 4 * 64,
                 max_request_characters=300000, resume=False):
    """Audit and patch one frozen draft. Never mutates the draft run."""
    draft_run, out_dir = Path(draft_run).resolve(), Path(out_dir).resolve()
    contract = read_json(draft_run / "contract.json")
    result = read_json(draft_run / "results.json")
    if not result.get("full_song_succeeded"):
        return dict(schema=SCHEMA, completed=False, skipped="draft_incomplete",
                    note="No revision on an incomplete draft", quality_verified=False)
    plan = read_json(draft_run / "plan.json")
    plan_sha = file_sha(draft_run / "plan.json")
    used_calls = int(result.get("calls_used") or 0)
    used_input = int(result.get("input_tokens") or 0)
    used_output = int(result.get("output_tokens") or 0)
    if used_calls + 2 > max_calls:
        return dict(schema=SCHEMA, completed=False, skipped="no_call_budget",
                    draft_calls=used_calls, max_calls=max_calls, quality_verified=False)
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("audio", {}).get("sha256") not in (None, contract.get("source_inputs_sha256", {}).get("__audio__")):
        pass
    cfg = dict(seed=seed, max_request_characters=max_request_characters,
               remaining_input_tokens=max(0, max_input_tokens - used_input),
               remaining_output_tokens=max(0, max_total_output_tokens - used_output))
    provider = create_provider(provider_spec, "planner", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(schema=SCHEMA, draft_run=str(draft_run), draft_plan_sha256=plan_sha,
                    draft_contract_sha256=object_sha(contract), draft_calls=used_calls,
                    draft_input_tokens=used_input, draft_output_tokens=used_output,
                    music_summary_sha256=file_sha(summary_path), provider_identity=provider.identity,
                    max_calls=max_calls, seed=seed, quality_verified=False,
                    structure_check_only=True)
    write_json(out_dir / "draft_snapshot.json", snapshot)
    write_json(out_dir / "draft_plan.json", plan)
    calls = []
    extra_input = extra_output = 0

    def guarded(request, kind):
        nonlocal extra_input, extra_output
        if extra_input >= cfg["remaining_input_tokens"]:
            raise ValueError("Revision would exceed the song's remaining input budget")
        return request

    try:
        audit_req = audit_request(plan, outline, summary, cfg, plan["song_id"] + "/revision/audit")
        audit_resp, audit_seconds = _call(provider, audit_req, out_dir / "calls", 1)
        calls.append(dict(kind="audit", seconds=audit_seconds, costs=audit_resp.get("costs")))
        extra_input += int((audit_resp.get("costs") or {}).get("input_tokens") or 0)
        extra_output += int((audit_resp.get("costs") or {}).get("output_tokens") or 0)
        issues = read_audit(audit_resp["text"], plan, summary)
        write_json(out_dir / "audit.json", dict(schema=SCHEMA + ".audit", issues=issues,
                                                raw_text=audit_resp["text"], call_id=audit_req["call_id"]))
        if not issues:
            final, patches, outcome = plan, [], "no_change"
        else:
            patch_req = patch_request(plan, issues, cfg, plan["song_id"] + "/revision/patch")
            patch_resp, patch_seconds = _call(provider, patch_req, out_dir / "calls", 2)
            calls.append(dict(kind="patch", seconds=patch_seconds, costs=patch_resp.get("costs")))
            extra_input += int((patch_resp.get("costs") or {}).get("input_tokens") or 0)
            extra_output += int((patch_resp.get("costs") or {}).get("output_tokens") or 0)
            payload = parse_json_response(patch_resp["text"])
            write_json(out_dir / "patches.raw.json", payload)
            patches = read_patches(patch_resp["text"], issues)
            final, changed = apply_patches(plan, patches)
            outcome = "patched" if changed else "no_change"
    except Exception as error:  # keep the untouched draft and record the failure
        write_json(out_dir / "revision_failure.json", dict(type=type(error).__name__, message=str(error)))
        write_json(out_dir / "final_plan.json", plan)
        record = dict(schema=SCHEMA, completed=False, outcome="revision_failed", error=str(error),
                      draft_plan_sha256=plan_sha, final_plan_sha256=plan_sha,
                      revision_calls=len(calls), extra_input_tokens=extra_input, extra_output_tokens=extra_output,
                      draft_calls=used_calls, total_calls=used_calls + len(calls),
                      total_input_tokens=used_input + extra_input, total_output_tokens=used_output + extra_output,
                      quality_verified=False)
        write_json(out_dir / "results.json", record)
        return record
    if extra_input + used_input > max_input_tokens or extra_output + used_output > max_total_output_tokens:
        write_json(out_dir / "final_plan.json", plan)
        record = dict(schema=SCHEMA, completed=False, outcome="budget_exceeded", draft_plan_sha256=plan_sha,
                      final_plan_sha256=plan_sha, revision_calls=len(calls),
                      total_input_tokens=used_input + extra_input, total_output_tokens=used_output + extra_output,
                      quality_verified=False)
        write_json(out_dir / "results.json", record)
        return record
    write_json(out_dir / "final_plan.json", final)
    changed = sorted({e["event_id"] for e in final["events"]
                      if e["action"] != next(x["action"] for x in plan["events"] if x["event_id"] == e["event_id"])})
    record = dict(schema=SCHEMA, completed=True, outcome=outcome, draft_run=str(draft_run),
                  draft_plan_sha256=plan_sha, final_plan_sha256=file_sha(out_dir / "final_plan.json"),
                  issues=issues, patches=patches, changed_event_ids=changed,
                  revision_calls=len(calls), calls=calls,
                  draft_calls=used_calls, total_calls=used_calls + len(calls),
                  extra_input_tokens=extra_input, extra_output_tokens=extra_output,
                  total_input_tokens=used_input + extra_input, total_output_tokens=used_output + extra_output,
                  max_calls=max_calls, structure_check_only=True, quality_verified=False)
    write_json(out_dir / "results.json", record)
    return record
