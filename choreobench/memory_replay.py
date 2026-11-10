"""Bounded fixed-prefix memory ablation, not whole-song regeneration.

The same frozen prefix, candidates, outline, public clock, prompt and decoding
budget are reused. Only historical event.action fields are projected out.
No audio, critic, repair, or training calls are dispatched by this module.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .baseline import _Journal, _journal_accounting, _recorded_result, _sealed, _verify
from .context import build_boundary_context
from .coordination import _candidates, _read_unit
from .io import atomic_write, file_lock, file_sha, json_text, object_sha, read_json, read_json_snapshot, write_json
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .schema import keys, validate_dataset, validate_plan
from .timing import event_clock


MODES = ("full", "metadata_only")


def project_history(request, mode):
    """Remove action from all three declared history paths, not from the outline."""
    if mode not in MODES:
        raise ValueError("Unknown fixed-prefix history mode")
    result = copy.deepcopy(request)
    if mode == "full":
        return result
    payload = parse_json_response(result["messages"][-1]["content"])
    history = payload.get("boundary_context")
    if not isinstance(history, dict) or payload.get("history_available") is not True:
        raise ValueError("Projection requires a real supplied history context")
    for event in payload["recent_committed_events"]:
        event.pop("action")
    for part in history["body_parts"].values():
        if part["latest_event"] is not None:
            part["latest_event"].pop("action")
    for motif in history["motif_memory"]:
        for event in motif["recent_examples"]:
            event.pop("action")
    result["messages"][-1]["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return result


def _snapshot(path, sources, expected=None):
    path = Path(path).resolve(strict=True)
    value, sha = read_json_snapshot(path)
    if expected is not None and expected != sha:
        raise ValueError("Frozen replay source bytes differ: " + str(path))
    if str(path) in sources and sources[str(path)] != sha:
        raise ValueError("Replay source changed")
    sources[str(path)] = sha
    return value


def _unchanged(sources):
    for path, sha in sources.items():
        if file_sha(path) != sha:
            raise ValueError("Frozen replay source changed: " + path)


def _case(entry, inventory_path, sources):
    keys(entry, ("run_dir", "contract_sha256", "results_sha256", "song_id", "unit_id"), where="replay case")
    root = Path(entry["run_dir"])
    if not root.is_absolute():
        root = inventory_path.parent / root
    root = root.resolve(strict=True)
    contract = _snapshot(root / "contract.json", sources, entry["contract_sha256"])
    result = _verify(_snapshot(root / "results.json", sources, entry["results_sha256"]))
    if (contract["schema"] != "choreobench.coordination_contract.v1" or not result["complete"]
            or result["contract_sha256"] != object_sha(contract) or not contract["config"]["use_history"]):
        raise ValueError("Replay requires a complete bound full-history source run")
    validate_dataset(contract["dataset_snapshot"])
    sid, uid = entry["song_id"], entry["unit_id"]
    original = next(s for s in contract["dataset_snapshot"]["songs"] if s["song_id"] == sid)
    prefix_name = hashlib.sha256(sid.encode()).hexdigest()
    refs = {r["path"]: r["sha256"] for r in result["artifacts"] + result["plans"]}
    def artifact(relative):
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or relative not in refs:
            raise ValueError("Replay artifact escapes or lacks an original binding")
        return _snapshot(path, sources, refs[relative])
    units = artifact("units/" + prefix_name + ".json")
    options = artifact("options/" + prefix_name + ".json")
    outline = artifact("outlines/" + prefix_name + ".json")
    index = next(i for i, u in enumerate(units["units"]) if u["unit_id"] == uid)
    if index == 0:
        raise ValueError("A memory probe needs a nonempty original prefix")
    unit = units["units"][index]
    before = artifact("blocks/" + prefix_name + "/" + units["units"][index - 1]["unit_id"] + ".json")
    after = artifact("blocks/" + prefix_name + "/" + uid + ".json")
    validate_plan(before, original)
    validate_plan(after, original)
    if before["motifs"] != outline["motifs"] or any(e["end"] > unit["start"] for e in before["events"]):
        raise ValueError("Original prefix/outline crosses the probe boundary")
    call_id = sid + "/coordinate/" + uid
    name = hashlib.sha256(call_id.encode()).hexdigest()
    started = _verify(_snapshot(root / "calls" / (name + ".started.json"), sources))
    raw = _verify(_snapshot(root / "calls" / (name + ".raw.json"), sources))
    if (started["call_id"] != call_id or raw["call_id"] != call_id
            or started["contract_sha256"] != object_sha(contract)
            or raw["contract_sha256"] != object_sha(contract)
            or started["provider_identity"] != contract["planner_identity"]
            or started["request_sha256"] != object_sha(started["request"])
            or raw["started_sha256"] != started["record_sha256"]
            or result["call_records"][call_id] != raw["record_sha256"]):
        raise ValueError("Original replay request/response identity mismatch")
    request = started["request"]
    response = _recorded_result(raw, contract["planner_identity"], request)
    payload = parse_json_response(request["messages"][-1]["content"])
    expected_history = build_boundary_context(before, unit["start"])
    section = next(s for s in outline["sections"] if s["section_id"] == unit["section_id"])
    expected_history["motif_memory"] = [m for m in expected_history["motif_memory"] if m["motif_id"] in section["motif_ids"]]
    clock = event_clock(unit, before["sample_rate"], contract["config"])
    expected_recent = [{k: copy.deepcopy(e[k]) for k in
                        ("event_id", "start", "end", "track", "action", "motif_id")} for e in before["events"][-8:]]
    if (payload["boundary_context"] != expected_history or payload["recent_committed_events"] != expected_recent
            or payload["planning_unit"] != {k: unit[k] for k in ("unit_id", "section_id", "start", "end", "kind")}
            or payload["event_clock"] != clock or payload["local_proposals"] != _candidates(options, unit)
            or payload["current_section_design"] != section
            or payload["whole_song_design"] != {"sections": outline["sections"]}
            or payload["current_motif_definitions"] != [m for m in outline["motifs"] if m["motif_id"] in section["motif_ids"]]):
        raise ValueError("Original request does not match the frozen prefix/candidates/design")
    old_events = _read_unit(response["text"], original, unit, outline, options,
                            contract["config"]["max_events_per_unit"], clock)
    if after["events"] != before["events"] + old_events:
        raise ValueError("Original response does not compile to the saved block")
    return dict(song_id=sid, unit_id=uid, original=original, unit=unit, outline=outline, options=options,
                before=before, historical_events=old_events, request=request, config=contract["config"], clock=clock,
                original_contract_sha256=object_sha(contract), synthetic=contract["synthetic"])


def run_memory_replay(inventory_path, provider_spec, output_dir, resume=False):
    inventory_path, output = Path(inventory_path).resolve(), Path(output_dir).resolve()
    sources = {}
    inventory = _snapshot(inventory_path, sources)
    keys(inventory, ("schema", "cases"), where="memory replay inventory")
    if inventory["schema"] != "choreobench.memory_replay_inventory.v1" or not isinstance(inventory["cases"], list) or not 1 <= len(inventory["cases"]) <= 3:
        raise ValueError("Fixed-prefix development probe permits one to three cases only")
    cases = [_case(entry, inventory_path, sources) for entry in inventory["cases"]]
    if len({c["song_id"] for c in cases}) != len(cases):
        raise ValueError("Use one preselected probe per song")
    requests = []
    for case in cases:
        for mode in MODES:
            request = project_history(case["request"], mode)
            request["call_id"] = case["song_id"] + "/memory-replay/" + case["unit_id"] + "/" + mode
            # Keep source request seed, prompt, schema and token cap identical.
            guard_request(request, case["config"])
            requests.append((case, mode, request))
    _unchanged(sources)
    provider = create_provider(provider_spec, role="planner", base_dir=inventory_path.parent)
    if provider.identity["kind"] not in ("chat_completions", "replay"):
        raise ValueError("Memory replay requires an explicit text provider")
    contract = dict(schema="choreobench.memory_replay_contract.v1", sources=sources,
        code_sha256={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        provider_identity=provider.identity, max_calls=len(requests),
        request_sha256={r["call_id"]: object_sha(r) for _, _, r in requests},
        modes=list(MODES), fixed_original_prefix=True, same_request_seed_and_token_cap=True,
        full_prompt_unchanged=True, metadata_only_removes_historical_action_fields=True,
        new_audio_calls=0, new_outline_calls=0, model_weights_updated=False,
        whole_song_quality_verified=False, synthetic=any(c["synthetic"] for c in cases) or provider.identity["synthetic"])
    if output.exists() and not resume:
        raise FileExistsError(str(output))
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume requires the original replay contract")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        if (output / "contract.json").exists():
            if read_json(output / "contract.json") != contract:
                raise ValueError("Replay input/config/provider/code changed")
        else:
            write_json(output / "contract.json", contract)
        journal = _Journal(output, object_sha(contract), len(requests))
        rows = []
        for case, mode, request in requests:
            directory = output / case["song_id"] / mode
            row = dict(song_id=case["song_id"], unit_id=case["unit_id"], mode=mode, status="pending",
                       historical_prefix_sha256=object_sha(case["before"]), new_plan_sha256=None)
            try:
                _unchanged(sources)
                if provider.identity != contract["provider_identity"]:
                    raise ValueError("Replay provider identity changed before dispatch")
                response = journal.call(provider, request)
                events = _read_unit(response["text"], case["original"], case["unit"], case["outline"], case["options"],
                                    case["config"]["max_events_per_unit"], case["clock"])
                plan = copy.deepcopy(case["before"])
                plan["events"].extend(events)
                validate_plan(plan, case["original"])
                history_actions = {e["action"] for e in case["before"]["events"]}
                row.update(status="valid", event_count=len(events), exact_prior_action_copies=sum(e["action"] in history_actions for e in events),
                           action_characters=sum(len(e["action"]) for e in events), new_plan_sha256=object_sha(plan),
                           actual_intervals=[dict(start=e["start"], end=e["end"]) for e in events],
                           semantic_quality_verified=False)
                for name, value in (("prefix.json", case["before"]), ("historical_events.json", case["historical_events"]),
                                    ("events.json", events), ("plan.json", plan)):
                    path = directory / name
                    if path.exists():
                        if read_json(path) != value:
                            raise ValueError("Published replay artifact changed")
                    else:
                        write_json(path, value)
            except (ValueError, TypeError, KeyError) as error:
                row.update(status="failed", error=str(error), retained_in_denominator=True)
                # Unknown/infrastructure results must not trigger subsequent calls.
                raw = journal.records.get(request["call_id"])
                if raw is None or not raw.get("success") or raw.get("result", {}).get("stop_reason") in (
                        None, "http_error", "infrastructure_error", "malformed_api_response"):
                    rows.append(row)
                    break
            rows.append(row)
        _unchanged(sources)
        accounting = _journal_accounting(journal)
        accounting["costs"].sort(key=lambda row: row["call_id"])
        accounting["unknown_outcome_call_ids"].sort()
        accounting["unknown_usage_call_ids"].sort()
        result = _sealed(dict(schema="choreobench.memory_replay_result.v1", contract_sha256=object_sha(contract), rows=rows,
            complete=len(rows) == len(requests), expected_calls=len(requests), calls_used=len(journal.started),
            call_records={k: r["record_sha256"] for k, r in journal.records.items()}, **accounting,
            new_audio_calls=0, new_outline_calls=0, whole_song_generation=False, model_weights_updated=False,
            synthetic=contract["synthetic"], quality_verified=False,
            note="Fixed original-prefix local probe only. Exact copies and shorter text do not prove dance quality."))
        if (output / "results.json").exists():
                previous = _verify(read_json(output / "results.json"))
                if previous["complete"]:
                    if previous != result:
                        raise ValueError("Completed replay result changed")
                else:
                    atomic_write(output / "results.json", result)
        else:
            write_json(output / "results.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_memory_replay(args.inventory, read_json(args.provider), args.out_dir, args.resume)
    print(json_text({k: result[k] for k in ("complete", "expected_calls", "calls_used", "rows", "quality_verified")}))


if __name__ == "__main__":
    main()
