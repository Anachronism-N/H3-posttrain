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
from .coordination import _read_unit, _unit_request
from .io import atomic_write, file_lock, file_sha, json_text, object_sha, read_json, read_json_snapshot, write_json
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .schema import keys, validate_dataset, validate_plan, validate_song
from .timing import event_clock


MODES = ("full", "metadata_only")


class _ReplayInfrastructureError(ValueError):
    """An identity/infrastructure violation stops further dispatches."""


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
    effective = copy.deepcopy(original)
    structure = contract.get("public_structure_snapshot")
    if structure is not None:
        matches = [row for row in structure["songs"] if row["song_id"] == sid]
        if (structure.get("schema") != "choreobench.public_structure.v1"
                or structure.get("dataset_sha256") != contract["dataset_sha256"] or len(matches) != 1):
            raise ValueError("Frozen public music structure has an invalid source identity")
        row = matches[0]
        if any(row[key] != original["audio"][key] for key in ("sample_rate", "num_samples")) or row["audio_sha256"] != original["audio"]["sha256"]:
            raise ValueError("Frozen public music structure differs from the original audio")
        effective["music"] = copy.deepcopy(row["music"])
    validate_song(effective)
    if contract["planning_songs_sha256"][sid] != object_sha(effective):
        raise ValueError("Frozen effective song does not match its original contract")
    clock = event_clock(unit, before["sample_rate"], contract["config"])
    expected_request = _unit_request(effective, options, units, unit, outline, before, contract["config"])
    expected_payload = parse_json_response(expected_request["messages"][-1]["content"])
    # Preserve the original system prompt, but reject unknown/private payload
    # fields and inconsistent public song, clock, position or availability data.
    if (object_sha(payload) != object_sha(expected_payload)
            or object_sha({key: value for key, value in request.items() if key != "messages"}) !=
               object_sha({key: value for key, value in expected_request.items() if key != "messages"})
            or len(request["messages"]) != 2
            or any(set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str)
                   for message, role in zip(request["messages"], ("system", "user")))):
        raise ValueError("Original request does not match the frozen prefix/candidates/design")
    old_events = _read_unit(response["text"], original, unit, outline, options,
                            contract["config"]["max_events_per_unit"], clock)
    if after["events"] != before["events"] + old_events:
        raise ValueError("Original response does not compile to the saved block")
    return dict(song_id=sid, unit_id=uid, original=original, unit=unit, outline=outline, options=options,
                before=before, historical_events=old_events, request=request, config=contract["config"], clock=clock,
                original_contract_sha256=object_sha(contract), synthetic=contract["synthetic"])


def _prepare_requests(inventory_path, sources):
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
    return cases, requests


def _decode_probe(case, response):
    if case.get("phase_representation"):
        from .motion_phases import decode_phases
        response = decode_phases(case, response)
    return _read_unit(response, case["original"], case["unit"], case["outline"], case["options"],
                      case["config"]["max_events_per_unit"], case["clock"])


def _valid_row(case, mode, events):
    if case.get("content_limits") is not None:
        from .duration_content import validate_content
        validate_content(case, events)
    plan = copy.deepcopy(case["before"])
    plan["events"].extend(events)
    validate_plan(plan, case["original"])
    history_actions = {event["action"] for event in case["before"]["events"]}
    row = dict(song_id=case["song_id"], unit_id=case["unit_id"], mode=mode, status="valid",
        historical_prefix_sha256=object_sha(case["before"]), new_plan_sha256=object_sha(plan), event_count=len(events),
        exact_prior_action_copies=sum(event["action"] in history_actions for event in events),
        action_characters=sum(len(event["action"]) for event in events),
        actual_intervals=[dict(start=event["start"], end=event["end"]) for event in events], semantic_quality_verified=False)
    if case.get("content_limits") is not None:
        from .duration_content import content_statistics
        row["content_lengths"] = content_statistics(events, case["content_limits"])
    return row, {"prefix.json": case["before"], "historical_events.json": case["historical_events"],
                 "events.json": events, "plan.json": plan}


def _accounting(journal):
    value = _journal_accounting(journal)
    for name in ("unknown_outcome_call_ids", "unknown_usage_call_ids"):
        value[name].sort()
    value["costs"].sort(key=lambda row: row["call_id"])
    return value


def _canonical_artifact(path, expected):
    if not path.is_file():
        raise ValueError("Missing completed replay artifact: " + str(path))
    value, sha = read_json_snapshot(path)
    if value != expected or sha != hashlib.sha256(json_text(expected).encode("utf-8")).hexdigest():
        raise ValueError("Completed replay artifact bytes changed: " + str(path))


def _check_completed(output, contract, journal, requests):
    previous = _verify(read_json(output / "results.json"))
    _canonical_artifact(output / "results.json", previous)
    if previous.get("complete") is not True:
        raise ValueError("Read-only replay audit requires a completed result")
    expected_ids = {request["call_id"] for _, _, request in requests}
    if (previous.get("schema") != "choreobench.memory_replay_result.v1"
            or previous.get("contract_sha256") != object_sha(contract)
            or previous.get("expected_calls") != len(requests) or previous.get("calls_used") != len(journal.started)
            or set(journal.started) != expected_ids or set(journal.records) != expected_ids
            or previous.get("call_records") != {key: value["record_sha256"] for key, value in journal.records.items()}
            or not isinstance(previous.get("rows"), list) or len(previous["rows"]) != len(requests)
            or contract.get("request_sha256") != {request["call_id"]: object_sha(request) for _, _, request in requests}
            or contract.get("max_calls") != len(requests)
            or contract.get("modes") != list(dict.fromkeys(mode for _, mode, _ in requests))):
        raise ValueError("Completed replay journal/request inventory changed")
    for (case, mode, request), stored_row in zip(requests, previous["rows"]):
        call_id = request["call_id"]
        start, raw = journal.started[call_id], journal.records[call_id]
        if start["request"] != request or start["provider_identity"] != contract["provider_identity"]:
            raise ValueError("Completed replay request/provider changed")
        name = journal._name(call_id)
        _canonical_artifact(output / "calls" / (name + ".started.json"), start)
        _canonical_artifact(output / "calls" / (name + ".raw.json"), raw)
        try:
            response = _recorded_result(raw, contract["provider_identity"], request)
            events = _decode_probe(case, response["text"])
            expected_row, artifacts = _valid_row(case, mode, events)
        except (ValueError, TypeError, KeyError):
            # An actual invalid/truncated response can be a completed, failed
            # attempt. Its original explanation is retained, not regenerated.
            expected_row = dict(song_id=case["song_id"], unit_id=case["unit_id"], mode=mode, status="failed",
                historical_prefix_sha256=object_sha(case["before"]), new_plan_sha256=None,
                error=stored_row.get("error"), retained_in_denominator=True)
            if not isinstance(expected_row["error"], str) or not expected_row["error"]:
                raise ValueError("Completed failed replay lacks its retained explanation")
            artifacts = {}
        if object_sha(stored_row) != object_sha(expected_row):
            raise ValueError("Completed replay row/statistics differ from recorded response")
        directory = output / case["song_id"] / mode
        for name, expected in artifacts.items():
            _canonical_artifact(directory / name, expected)
        if not artifacts and any((directory / name).exists() for name in ("prefix.json", "historical_events.json", "events.json", "plan.json")):
            raise ValueError("Failed replay unexpectedly contains published valid artifacts")
    if any(previous.get(key) != value for key, value in _accounting(journal).items()):
        raise ValueError("Completed replay accounting differs from its journal")
    if (previous.get("synthetic") != contract["synthetic"] or previous.get("new_audio_calls") != 0
            or previous.get("new_outline_calls") != 0 or previous.get("whole_song_generation") is not False
            or previous.get("model_weights_updated") is not False or previous.get("quality_verified") is not False):
        raise ValueError("Completed replay scope/status differs from its contract")
    _unchanged(contract["sources"])
    _canonical_artifact(output / "results.json", previous)
    return previous


def audit_completed_replay(output_dir):
    """Audit saved v1 source/journal/artifacts without a provider or any writes.

    Saved code hashes remain historical declarations; current code need not
    match them. Derived files use the original v1 canonical JSON publication.
    """
    output = Path(output_dir).resolve()
    contract, contract_sha = read_json_snapshot(output / "contract.json")
    if contract.get("schema") != "choreobench.memory_replay_contract.v1" or not (output / "calls").is_dir():
        raise ValueError("Missing/invalid completed replay contract or journal")
    sources, inventories = {}, []
    for path, sha in contract["sources"].items():
        value = _snapshot(path, sources, sha)
        if isinstance(value, dict) and value.get("schema") == "choreobench.memory_replay_inventory.v1":
            inventories.append(Path(path))
    if len(inventories) != 1:
        raise ValueError("Replay contract must bind exactly one source inventory")
    _, requests = _probe_requests(inventories[0], sources, contract.get("duration_source_run"), contract.get("adaptive_boundary", False), contract.get("motion_phase_probe", False))
    if sources != contract["sources"]:
        raise ValueError("Rebuilt replay sources differ from the saved contract")
    journal = _Journal(output, object_sha(contract), contract["max_calls"])
    result = _check_completed(output, contract, journal, requests)
    if file_sha(output / "contract.json") != contract_sha:
        raise ValueError("Replay contract changed during read-only audit")
    return dict(schema="choreobench.memory_replay_audit.v1", audit_complete=True, new_model_calls=0,
        contract_file_sha256=contract_sha, result_file_sha256=hashlib.sha256(json_text(result).encode("utf-8")).hexdigest(),
        historical_code_sha256=contract["code_sha256"], cases=len(requests) // len(MODES),
        calls_used=result["calls_used"], valid_rows=sum(row["status"] == "valid" for row in result["rows"]),
        failed_rows=sum(row["status"] == "failed" for row in result["rows"]), quality_verified=False,
        scope="Saved source/request/response/artifact/statistic consistency; not independent model or quality authentication")


def _dispatch(journal, provider, request, contract):
    expected = contract["provider_identity"]
    if provider.identity != expected:
        raise _ReplayInfrastructureError("Replay provider identity changed before dispatch")
    if read_json(journal.root / "contract.json") != contract:
        raise _ReplayInfrastructureError("On-disk replay contract changed")
    start = journal.started.get(request["call_id"])
    if start and start["provider_identity"] != expected:
        raise _ReplayInfrastructureError("Recorded replay provider identity differs from the frozen contract")
    try:
        return journal.call(provider, request)
    finally:
        raw = journal.records.get(request["call_id"])
        response = raw.get("result") if raw else None
        if provider.identity != expected or (isinstance(response, dict) and response.get("identity") != expected):
            raise _ReplayInfrastructureError("Replay provider identity changed during dispatch")
        if read_json(journal.root / "contract.json") != contract:
            raise _ReplayInfrastructureError("On-disk replay contract changed during dispatch")


def _probe_requests(inventory_path, sources, duration_source_run=None, adaptive_boundary=False, motion_phase_probe=False):
    if motion_phase_probe:
        from .motion_phases import prepare_phase_requests
        return prepare_phase_requests(inventory_path, sources, duration_source_run)
    if duration_source_run is None:
        return _prepare_requests(inventory_path, sources)
    from .duration_content import prepare_duration_requests
    return prepare_duration_requests(inventory_path, sources, duration_source_run, adaptive_boundary)


def run_memory_replay(inventory_path, provider_spec, output_dir, resume=False, *, duration_source_run=None, adaptive_boundary=False, motion_phase_probe=False):
    inventory_path, output = Path(inventory_path).resolve(), Path(output_dir).resolve()
    sources = {}
    cases, requests = _probe_requests(inventory_path, sources, duration_source_run, adaptive_boundary, motion_phase_probe)
    _unchanged(sources)
    provider = create_provider(provider_spec, role="planner", base_dir=inventory_path.parent)
    if provider.identity["kind"] not in ("chat_completions", "replay"):
        raise ValueError("Memory replay requires an explicit text provider")
    contract = dict(schema="choreobench.memory_replay_contract.v1", sources=sources,
        code_sha256={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        provider_identity=copy.deepcopy(provider.identity), max_calls=len(requests),
        request_sha256={r["call_id"]: object_sha(r) for _, _, r in requests},
        modes=list(dict.fromkeys(mode for _, mode, _ in requests)), fixed_original_prefix=True, same_request_seed_and_token_cap=True,
        full_prompt_unchanged=duration_source_run is None, metadata_only_removes_historical_action_fields=True,
        new_audio_calls=0, new_outline_calls=0, model_weights_updated=False,
        whole_song_quality_verified=False, synthetic=any(c["synthetic"] for c in cases) or provider.identity["synthetic"])
    if motion_phase_probe:
        contract["motion_phase_probe"] = True
        contract["structured_actions_are_proposals"] = True
    if duration_source_run is not None:
        contract["duration_source_run"] = str(Path(duration_source_run).resolve())
        contract["fixed_schedule_content_probe"] = True
        contract["common_content_limits"] = cases[0]["content_limits"]
        if adaptive_boundary:
            contract["adaptive_boundary"] = True
            contract["adaptive_slot_minimum_seconds"] = 2
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
        completed = (output / "results.json").is_file() and _verify(read_json(output / "results.json")).get("complete") is True
        if completed and not (output / "calls").is_dir():
            raise ValueError("Completed replay journal directory is missing")
        journal = _Journal(output, object_sha(contract), len(requests))
        if completed:
            return _check_completed(output, contract, journal, requests)
        rows, infrastructure_stopped = [], False
        for case, mode, request in requests:
            directory = output / case["song_id"] / mode
            row = dict(song_id=case["song_id"], unit_id=case["unit_id"], mode=mode, status="pending",
                       historical_prefix_sha256=object_sha(case["before"]), new_plan_sha256=None)
            try:
                _unchanged(sources)
                response = _dispatch(journal, provider, request, contract)
                events = _decode_probe(case, response["text"])
                row, artifacts = _valid_row(case, mode, events)
                for name, value in artifacts.items():
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
                info = raw.get("result") if raw else None
                if (isinstance(error, _ReplayInfrastructureError) or raw is None or not raw.get("success")
                        or not isinstance(info, dict) or info.get("stop_reason") in (
                            None, "http_error", "infrastructure_error", "malformed_api_response")):
                    infrastructure_stopped = True
                    rows.append(row)
                    break
            rows.append(row)
        _unchanged(sources)
        if read_json(output / "contract.json") != contract:
            raise ValueError("On-disk replay contract changed before result publication")
        result = _sealed(dict(schema="choreobench.memory_replay_result.v1", contract_sha256=object_sha(contract), rows=rows,
            complete=len(rows) == len(requests) and not infrastructure_stopped, expected_calls=len(requests), calls_used=len(journal.started),
            call_records={k: r["record_sha256"] for k, r in journal.records.items()}, **_accounting(journal),
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
    parser.add_argument("--inventory")
    parser.add_argument("--provider")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.audit_only:
        if args.inventory is not None or args.provider is not None or args.resume:
            parser.error("--audit-only reads the saved contract; do not provide inventory, provider or resume")
        print(json_text(audit_completed_replay(args.out_dir)))
        return
    if args.inventory is None or args.provider is None:
        parser.error("Execution requires --inventory and --provider")
    result = run_memory_replay(args.inventory, read_json(args.provider), args.out_dir, args.resume)
    print(json_text({k: result[k] for k in ("complete", "expected_calls", "calls_used", "rows", "quality_verified")}))


if __name__ == "__main__":
    main()
