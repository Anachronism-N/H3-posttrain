"""Prepare and audit fixed-outline history comparisons; never run a provider."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from . import experiments as helpers
from .audio import make_windows
from .baseline import ModelOutputError, _empty_plan, _journal_accounting, _recorded_result, _verify
from .candidates import load_candidates, match_candidates
from .dataset import resolve_audio, verify_dataset_audio
from .harness import _config
from .io import file_sha, object_sha, read_json, read_json_snapshot, write_json
from .planning_units import build_planning_units
from .public_structure import load_public_structure
from .schema import validate_dataset, validate_plan
from .selection import candidate_options
from .timing import event_clock


SCHEMA = "choreobench.history_experiment.v1"
VARIANTS = (("main", True), ("no_history", False))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _context(dataset_path, candidates_path, config):
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    songs = [song for song in dataset["songs"] if song["split"] in config["splits"]]
    _require(bool(songs), "No songs in selected history comparison splits")
    bank = load_candidates(candidates_path)
    bank_info = match_candidates(bank, dataset_sha, songs, {s["song_id"]: s["audio"] for s in songs},
                                 bank.data["source"]["contract"]["config"])
    path = config["public_structure_path"]
    if path is not None:
        path = Path(path).expanduser()
        path = path.resolve() if path.is_absolute() else (Path(dataset_path).parent / path).resolve()
    effective, structure, structure_sha = load_public_structure(path, dataset_sha, songs)
    source_cfg = bank.data["source"]["contract"]["config"]
    units = {song["song_id"]: build_planning_units(song, make_windows(song["audio"]["num_samples"],
        song["audio"]["sample_rate"], source_cfg["window_seconds"], source_cfg["stride_seconds"]), config["unit_config"])
        for song in effective}
    return dict(dataset=dataset, dataset_sha=dataset_sha, songs=songs, bank=bank, bank_info=bank_info,
                effective=effective, structure=structure, structure_sha=structure_sha, structure_path=path, units=units)


def _load_outlines(path, context):
    from .outlines import load_shared_outlines
    return load_shared_outlines(path, context["dataset_sha"], context["bank"].bytes_sha,
                                context["effective"], context["units"], context["structure_sha"])


def prepare_history_experiment(dataset_path, candidates_path, base_config_path, outlines_path, output_dir):
    """Write a new two-arm preparation, with shared historical outline costs once."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    dataset_path, candidates_path, base_config_path, outlines_path = [Path(p).resolve() for p in
        (dataset_path, candidates_path, base_config_path, outlines_path)]
    base, base_sha = read_json_snapshot(base_config_path)
    config = _config(dict(base, use_outline=True))
    _require(config["mode"] == "coordinated", "Fixed-outline preparation requires coordinated mode")
    existing = config.get("shared_outline_path")
    if existing is not None:
        existing = Path(existing).expanduser()
        existing = existing.resolve() if existing.is_absolute() else (dataset_path.parent / existing).resolve()
        _require(existing == outlines_path, "base config and outlines argument select different outline sources")
    config["shared_outline_path"] = str(outlines_path)
    context = _context(dataset_path, candidates_path, config)
    if context["structure_path"] is not None:
        config["public_structure_path"] = str(context["structure_path"])
    outlines, outline_sha = _load_outlines(outlines_path, context)
    sources = [helpers._source(dataset_path, context["dataset_sha"], "dataset"),
        helpers._source(candidates_path, context["bank"].bytes_sha, "candidates"),
        helpers._source(base_config_path, base_sha, "base_config"),
        helpers._source(outlines_path, outline_sha, "shared_outlines")]
    if context["structure_path"] is not None:
        sources.append(helpers._source(context["structure_path"], context["structure_sha"], "public_structure"))
    sources.extend(helpers._source(resolve_audio(s, dataset_path), s["audio"]["sha256"], "audio:" + s["song_id"])
                   for s in context["dataset"]["songs"])
    workload = helpers._workload(context["songs"], context["effective"], context["bank"], config)
    blocks = sum(row["expected_block_calls"] for row in workload)
    failures = [dict(song_id=row["song_id"], reason=row["preparation_failure"])
                for row in workload if row["preparation_failure"] is not None]
    common_sha = object_sha(helpers._shared_config(config))
    configs, variants = {}, []
    for name, history in VARIANTS:
        value = _config(dict(config, use_outline=True, use_history=history))
        _require(object_sha(helpers._shared_config(value)) == common_sha, "Non-history parameter changed")
        configs[name] = value
        variants.append(dict(variant_id=name, use_outline=True, use_history=history,
            config_path=str(output / "configs" / (name + ".json")), run_dir=str(output / "runs" / name),
            expected_new_text_calls=blocks, expected_block_calls=blocks, expected_outline_calls=0,
            max_calls=value["max_calls"], budget_sufficient=blocks <= value["max_calls"],
            output_token_cap=blocks * value["max_output_tokens"], new_local_model_calls=0))
    helpers._assert_sources(sources)
    context["bank"].assert_unchanged(full=True)
    output.mkdir(parents=True, exist_ok=False)
    for row in variants:
        write_json(row["config_path"], configs[row["variant_id"]])
        row["config_sha256"] = file_sha(row["config_path"])
    sufficient, preparable = all(row["budget_sufficient"] for row in variants), not failures
    commands = dict(schema="choreobench.history_commands.v1", cwd=str(Path(__file__).resolve().parents[1]),
        preflight_argv=[sys.executable, "-m", "choreobench.history_experiment", "--verify", str(output)],
        runs=[dict(variant_id=row["variant_id"], ready_to_run=sufficient and preparable,
            argv=[sys.executable, "-m", "choreobench", "run-harness", "--dataset", str(dataset_path),
                  "--candidates", str(candidates_path), "--config", row["config_path"], "--out-dir", row["run_dir"]])
            for row in variants],
        execution_guidance="Not executed. Run preflight before each arm; use subprocess.run(argv,cwd=cwd,shell=False,check=True). "
            "Audit actual runs separately. Input/configuration hashes are not independent model or quality verification.")
    write_json(output / "commands.json", commands)
    code_sha = helpers._code_hashes()
    manifest = helpers._seal(dict(schema=SCHEMA, prepared=True, executed=False, results_verified=False,
        model_or_service_verified=False, training_performed=False, ready_to_run=sufficient and preparable,
        readiness_scope="Local source/outline bindings and declared call budget only; no service/context/quality acceptance",
        all_budgets_sufficient=sufficient, all_preparations_valid=preparable, known_preparation_failures=failures,
        sources=sources, selected_songs=[s["song_id"] for s in context["songs"]], song_workload=workload,
        code_sha256=code_sha, python=helpers._python_identity(), shared_config_sha256=common_sha,
        local_candidates_identity_sha256=object_sha(context["bank_info"]),
        public_structure_snapshot_sha256=object_sha(context["structure"]),
        shared_outline_sha256=outline_sha, outline_sha256_by_song={s["song_id"]: s["outline_sha256"] for s in outlines["songs"]},
        shared_outline_source_costs=copy.deepcopy(outlines["source_costs"]),
        shared_outline_cost_scope="Historical outline collection once shared across both arms; no new outline calls",
        original_collection=dict(recorded_calls=context["bank_info"]["original_collection_calls"],
            expected_calls=context["bank_info"]["expected_local_calls"],
            costs_sha256=object_sha(context["bank_info"]["original_collection_costs"]),
            scope="One shared local collection; never counted twice or claimed free"),
        input_synthetic=bool(context["dataset"]["synthetic"] or context["bank"].data["synthetic"] or outlines["synthetic"]
            or context["structure"] and any(s["synthetic"] for s in context["structure"]["songs"])),
        actual_input_tokens=None, actual_output_tokens=None, actual_cost=None,
        block_seeds_identical_across_variants=True, variants=variants,
        commands=dict(path=str(output / "commands.json"), sha256=file_sha(output / "commands.json")),
        limitations=["Only use_history changes; the SAME externally bound outline is supplied to both arms.",
            "No-history retains the shared whole-song design, music context and overlapping source proposals.",
            "Prompt length and actual cost differ; identical caps do not equal equal billed cost.",
            "Independent quality reviews remain pending; events/blocks are not independent songs."]))
    helpers._assert_sources(sources)
    _require(code_sha == helpers._code_hashes(), "History implementation changed during preparation")
    write_json(output / "manifest.json", manifest)
    return manifest


def _verified_preparation(output_dir, environment=True):
    output = Path(output_dir).resolve()
    manifest = read_json(output / "manifest.json")
    _require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "Unsupported history preparation schema")
    _verify(manifest)
    helpers._assert_sources(manifest["sources"])
    if environment:
        _require(manifest["code_sha256"] == helpers._code_hashes(), "History implementation changed; prepare a new experiment")
        _require(manifest["python"] == helpers._python_identity(), "History Python executable/version changed")
    _require([row["variant_id"] for row in manifest["variants"]] == [v[0] for v in VARIANTS], "History variant inventory differs")
    for row, (name, history) in zip(manifest["variants"], VARIANTS):
        path = output / "configs" / (name + ".json")
        _require(row["config_path"] == str(path) and row["run_dir"] == str(output / "runs" / name)
                 and file_sha(path) == row["config_sha256"], "History config/path changed")
        config = _config(read_json(path))
        _require(config["mode"] == "coordinated" and config["use_outline"] is True and config["use_history"] is history
            and object_sha(helpers._shared_config(config)) == manifest["shared_config_sha256"], "History flags/shared config changed")
    _require(manifest["commands"]["path"] == str(output / "commands.json") and
             file_sha(output / "commands.json") == manifest["commands"]["sha256"], "History commands changed")
    return manifest


def verify_history_experiment(output_dir):
    """Strict preflight: unchanged sources, configs, commands, code and Python."""
    return _verified_preparation(output_dir)


def _artifact(root, ref):
    _require(isinstance(ref, dict) and isinstance(ref.get("path"), str), "Invalid run artifact reference")
    path = (root / ref["path"]).resolve(strict=True)
    _require(not Path(ref["path"]).is_absolute() and root in path.parents, "Run artifact escapes run directory")
    value, digest = read_json_snapshot(path)
    _require(digest == ref["sha256"], "Published run artifact bytes changed")
    return value


def _journal(root, contract):
    """Read the immutable journal without _Journal's directory-creation side effect."""
    starts, records = {}, {}
    contract_sha = object_sha(contract)
    for path in sorted((root / "calls").glob("*.started.json")):
        value = _verify(read_json(path))
        call_id = value["call_id"]
        name = hashlib.sha256(call_id.encode()).hexdigest()
        _require(call_id not in starts and path.name == name + ".started.json"
            and value.get("contract_sha256") == contract_sha
            and value.get("request_sha256") == object_sha(value.get("request"))
            and value.get("provider_identity") == contract["planner_identity"], "Invalid history journal start binding")
        starts[call_id] = value
    for path in sorted((root / "calls").glob("*.raw.json")):
        value = _verify(read_json(path))
        call_id = value["call_id"]
        name = hashlib.sha256(call_id.encode()).hexdigest()
        _require(call_id in starts and call_id not in records and path.name == name + ".raw.json"
            and value.get("contract_sha256") == contract_sha
            and value.get("started_sha256") == starts[call_id]["record_sha256"], "Invalid history raw journal binding")
        records[call_id] = value
    _require(len(starts) <= contract["config"]["max_calls"], "Actual calls exceed prepared budget")
    return SimpleNamespace(started=starts, records=records)


def _audit_arm(row, manifest, context, outlines, current_code_matches):
    from .coordination import _read_unit, _unit_request
    root = Path(row["run_dir"]).resolve()
    selected = manifest["selected_songs"]
    absent = {sid: dict(status="missing_run", complete_plan=False) for sid in selected}
    if not (root / "contract.json").is_file():
        return dict(variant_id=row["variant_id"], status="missing_run", executed=False,
                    complete=False, songs=absent, calls_used=0, contract=None, payloads={})
    contract = read_json(root / "contract.json")
    cfg = _config(read_json(row["config_path"]))
    _require(isinstance(contract, dict) and isinstance(contract.get("software"), dict)
             and contract["software"].get("python") == manifest["python"]["version"],
             "Actual producer Python declaration differs from preparation; this is not process authentication")
    _require(contract.get("schema") == "choreobench.coordination_contract.v1"
        and contract.get("config") == cfg and contract.get("dataset_sha256") == context["dataset_sha"]
        and contract.get("dataset_snapshot") == context["dataset"] and contract.get("selected_songs") == selected
        and contract.get("code_sha256") == manifest["code_sha256"]
        and object_sha(contract.get("local_candidates_identity")) == manifest["local_candidates_identity_sha256"]
        and contract.get("public_structure_sha256") == context["structure_sha"]
        and object_sha(contract.get("public_structure_snapshot")) == manifest["public_structure_snapshot_sha256"]
        and contract.get("shared_outline_sha256") == manifest["shared_outline_sha256"]
        and contract.get("shared_outline_snapshot") == outlines,
        "Actual history contract differs from prepared sources/config/code/outline")
    _require(type(contract.get("synthetic")) is bool and (not manifest["input_synthetic"] or contract["synthetic"]),
             "Actual run lost synthetic provenance")
    journal = _journal(root, contract)
    accounting = _journal_accounting(journal)
    if not (root / "results.json").is_file():
        return dict(variant_id=row["variant_id"], status="missing_results", executed=bool(journal.started), complete=False,
            songs={sid: dict(status="missing_results", complete_plan=False) for sid in selected}, calls_used=len(journal.started),
            contract=contract, payloads={}, **accounting)
    result = _verify(read_json(root / "results.json"))
    _require(all(type(result.get(key)) is int and result[key] >= 0 for key in
                 ("selected_count", "success_count", "calls_used", "expected_calls")), "Actual counts must be nonnegative integers")
    _require(result.get("schema") == "choreobench.coordination_result.v1"
        and result.get("contract_sha256") == object_sha(contract)
        and result.get("shared_outline_sha256") == manifest["shared_outline_sha256"]
        and result.get("shared_outline_source_costs") == outlines["source_costs"]
        and result.get("selected_count") == len(selected)
        and result.get("calls_used") == len(journal.started)
        and result.get("call_records") == {key: value["record_sha256"] for key, value in journal.records.items()}
        and result.get("synthetic") is contract["synthetic"], "Actual history results/journal binding differs")
    for key, value in accounting.items():
        recorded = result.get(key)
        if isinstance(value, list) and isinstance(recorded, list):
            sort_key = (lambda item: item["call_id"]) if key == "costs" else None
            recorded, value = sorted(recorded, key=sort_key), sorted(value, key=sort_key)
        _require(recorded == value, "Actual journal accounting differs: " + key)
    _require(result.get("new_local_model_calls") == 0, "History comparison unexpectedly sampled new local candidates")
    artifacts, plans = {}, {}
    for ref in result["artifacts"]:
        _require(ref["path"] not in artifacts, "Duplicate published artifact")
        artifacts[ref["path"]] = _artifact(root, ref)
    for ref in result["plans"]:
        sid = ref["song_id"]
        _require(sid in selected and sid not in plans, "Unknown/duplicate published song")
        plans[sid] = _artifact(root, ref)
    failures = {failure["song_id"]: failure for failure in result["failed_songs"]}
    _require(len(failures) == len(result["failed_songs"]) and set(failures) <= set(selected)
        and not set(plans) & set(failures), "Invalid failure denominator")
    _require(all(failure.get("retained_in_evaluation_denominator") is True for failure in failures.values()),
             "Failed song missing denominator declaration")
    _require(result.get("success_count") == len(plans) and type(result.get("complete")) is bool,
             "Invalid success/completion counts")
    if result["complete"]:
        _require(set(plans) | set(failures) == set(selected), "Completed run omitted selected songs")
    _require(result.get("all_songs_succeeded") is (result["complete"] and len(plans) == len(selected) and not failures),
             "Whole-song success misreported")
    _require(contract.get("expected_calls") == row["expected_new_text_calls"]
        and result.get("expected_calls") == row["expected_new_text_calls"], "Expected call workload changed")
    shared = {item["song_id"]: item for item in outlines["songs"]}
    original = {song["song_id"]: song for song in context["songs"]}
    effective = {song["song_id"]: song for song in context["effective"]}
    seen_calls, payloads, songs = set(), {}, {}
    for sid in selected:
        prefix = hashlib.sha256(sid.encode()).hexdigest()
        units = context["units"][sid]
        outline = shared[sid]["outline"]
        published_units = artifacts.get("units/" + prefix + ".json")
        if published_units is not None:
            _require(published_units == units, "Actual planning units changed")
        published_outline = artifacts.get("outlines/" + prefix + ".json")
        if published_outline is not None:
            _require(object_sha(published_outline) == shared[sid]["outline_sha256"], "Actual imported outline changed")
        options = candidate_options(context["bank"], original[sid], units) if sid in plans or any(
            call_id.startswith(sid + "/coordinate/") for call_id in journal.started) else None
        partial = _empty_plan(original[sid])
        partial["motifs"] = copy.deepcopy(outline["motifs"])
        usable = True
        payloads[sid] = []
        for unit in units["units"]:
            call_id = sid + "/coordinate/" + unit["unit_id"]
            if call_id not in journal.started:
                usable = False
                continue
            _require(usable, "A call follows an absent/failed block; hidden continuation or retry")
            seen_calls.add(call_id)
            request = journal.started[call_id]["request"]
            _require(request.get("response_kind") == "coordination_unit", "Shared history run generated an outline or unexpected call")
            payload = json.loads(request["messages"][-1]["content"])
            has_history = "boundary_context" in payload and "recent_committed_events" in payload
            no_history = "boundary_context" not in payload and "recent_committed_events" not in payload
            _require(has_history if cfg["use_history"] else no_history, "Actual history fields violate the ablation")
            _require(payload.get("history_available") is cfg["use_history"] and payload.get("outline_available") is True,
                     "Actual history/outline availability flags differ")
            _require(payload.get("whole_song_design") == dict(sections=outline["sections"]), "Actual request did not consume shared outline")
            clock = event_clock(unit, original[sid]["audio"]["sample_rate"], cfg)
            _require(payload.get("event_clock") == clock, "Actual event menu differs")
            payloads[sid].append(dict(call_id=call_id, seed=request["seed"], payload=payload, clock_sha256=object_sha(clock)))
            if current_code_matches:
                _require(request == _unit_request(effective[sid], options, units, unit, outline, partial, cfg),
                         "Actual request differs from reconstructed bound public inputs/history")
            record = journal.records.get(call_id)
            if record is None:
                usable = False
                continue
            try:
                response = _recorded_result(record, contract["planner_identity"], request)
                events = _read_unit(response["text"], original[sid], unit, outline, options, cfg["max_events_per_unit"], clock)
                proposed = copy.deepcopy(partial)
                proposed["events"].extend(events)
                validate_plan(proposed, original[sid])
            except (ValueError, TypeError, KeyError):
                usable = False
                continue
            partial = proposed
            _require(artifacts.get("blocks/" + prefix + "/" + unit["unit_id"] + ".json") == partial,
                     "Published block does not match the recorded response")
        if sid in plans:
            validate_plan(plans[sid], original[sid])
            _require(usable and published_outline is not None and published_units is not None,
                     "Published final plan lacks complete actual block/outline evidence")
            _require({k: v for k, v in plans[sid].items() if k != "provenance"} ==
                     {k: v for k, v in partial.items() if k != "provenance"},
                     "Published final plan differs from actual responses")
            provenance = plans[sid]["provenance"]
            _require(provenance.get("outline_sha256") == shared[sid]["outline_sha256"]
                and provenance.get("shared_outline_sha256") == manifest["shared_outline_sha256"]
                and provenance.get("outline_generated_in_this_run") is False
                and provenance.get("outline_generated_by_model") is True
                and provenance.get("candidate_bank_sha256") == context["bank"].bytes_sha
                and provenance.get("contract_sha256") == object_sha(contract)
                and provenance.get("ablation") == dict(use_outline=True, use_history=cfg["use_history"])
                and provenance.get("synthetic") is contract["synthetic"], "Final plan shared outline provenance differs")
            metadata = artifacts.get("outline_metadata/" + prefix + ".json", {})
            _require(metadata.get("source") == "shared_import" and metadata.get("generated_in_this_run") is False
                and metadata.get("generated_by_model") is True
                and metadata.get("outline_sha256") == shared[sid]["outline_sha256"]
                and metadata.get("shared_outline_sha256") == manifest["shared_outline_sha256"],
                "Published outline source metadata differs")
        songs[sid] = dict(status="success" if sid in plans else "failed" if sid in failures else "incomplete",
            complete_plan=sid in plans, outline_sha256=object_sha(published_outline) if published_outline is not None else None,
            successful_blocks=sum(1 for unit in units["units"] if "blocks/" + prefix + "/" + unit["unit_id"] + ".json" in artifacts),
            actual_block_calls=len(payloads[sid]), failure=failures.get(sid), plan=plans.get(sid))
    _require(seen_calls == set(journal.started), "Unexpected non-block calls or song outside selected denominator")
    return dict(variant_id=row["variant_id"], status="complete" if result["complete"] else "incomplete", executed=bool(journal.started),
        complete=result["complete"], songs=songs, calls_used=len(journal.started), contract=contract, payloads=payloads,
        synthetic=result["synthetic"], **accounting)


def audit_history_experiment(output_dir):
    """Read actual two-arm evidence; absent/failed pairs remain in the denominator."""
    manifest = _verified_preparation(output_dir, environment=False)
    sources = {row["role"]: row["path"] for row in manifest["sources"]}
    cfg = _config(read_json(manifest["variants"][0]["config_path"]))
    context = _context(Path(sources["dataset"]), Path(sources["candidates"]), cfg)
    outlines, digest = _load_outlines(sources["shared_outlines"], context)
    _require(digest == manifest["shared_outline_sha256"], "Shared outline bytes changed")
    current_code = manifest["code_sha256"] == helpers._code_hashes()
    arms = {row["variant_id"]: _audit_arm(row, manifest, context, outlines, current_code) for row in manifest["variants"]}
    identity_equal = (arms["main"]["contract"] is not None and arms["no_history"]["contract"] is not None
                      and arms["main"]["contract"]["planner_identity"] == arms["no_history"]["contract"]["planner_identity"])
    pairs = []
    for sid in manifest["selected_songs"]:
        a, b = [arms[name]["songs"][sid] for name, _ in VARIANTS]
        left, right = [arms[name]["payloads"].get(sid, []) for name, _ in VARIANTS]
        def common_payload(item):
            return {k: v for k, v in item["payload"].items() if k not in
                    ("boundary_context", "recent_committed_events", "history_available")}
        payloads_equal = (bool(left) and len(left) == len(right) and all(x["call_id"] == y["call_id"]
            and x["seed"] == y["seed"] and common_payload(x) == common_payload(y) for x, y in zip(left, right)))
        outline_equal = (a.get("outline_sha256") == b.get("outline_sha256") == manifest["outline_sha256_by_song"][sid])
        complete = a["complete_plan"] and b["complete_plan"]
        pair = dict(song_id=sid, main_status=a["status"], no_history_status=b["status"],
            complete_pair=complete, actual_shared_outline_equal=outline_equal,
            actual_common_inputs_menus_seeds_equal=payloads_equal,
            content_control_verified=bool(complete and outline_equal and payloads_equal and identity_equal and current_code),
            event_count_equal=None, time_boundaries_equal=None, track_sequence_equal=None,
            time_track_sequence_equal=None, action_sequence_equal=None)
        if complete:
            ae, be = a["plan"]["events"], b["plan"]["events"]
            pair.update(event_count_equal=len(ae) == len(be),
                time_boundaries_equal=[(e["start"], e["end"]) for e in ae] == [(e["start"], e["end"]) for e in be],
                track_sequence_equal=[e["track"] for e in ae] == [e["track"] for e in be],
                time_track_sequence_equal=[(e["start"], e["end"], e["track"]) for e in ae] == [(e["start"], e["end"], e["track"]) for e in be],
                action_sequence_equal=[e["action"] for e in ae] == [e["action"] for e in be])
        pairs.append(pair)
    for arm in arms.values():
        del arm["contract"], arm["payloads"]
        for song in arm["songs"].values():
            song.pop("plan", None)
        costs = arm.get("costs")
        arm["reported_usage_totals"] = {key: sum(item[key] for item in costs)
            if costs is not None and all(type(item.get(key)) is int and item[key] >= 0 for item in costs) else None
            for key in ("input_tokens", "output_tokens")}
    synthetic = bool(manifest["input_synthetic"] or any(arm.get("synthetic") is True for arm in arms.values()))
    run_completion = all(arm.get("complete") is True and arm.get("unknown_outcomes_count", 0) == 0 for arm in arms.values())
    return helpers._seal(dict(schema="choreobench.history_audit.v1", preparation_record_sha256=manifest["record_sha256"],
        selected_song_count=len(pairs), selected_song_ids=manifest["selected_songs"], pairs=pairs, arms=arms,
        complete_pair_count=sum(pair["complete_pair"] for pair in pairs),
        content_controlled_pair_count=sum(pair["content_control_verified"] for pair in pairs),
        missing_or_failed_pair_count=sum(not pair["complete_pair"] for pair in pairs),
        paired_control_verified=run_completion and all(pair["content_control_verified"] for pair in pairs),
        run_completion_verified=run_completion,
        incomplete_run_variants=[name for name, arm in arms.items() if arm.get("complete") is not True],
        current_implementation_matches_preparation=current_code,
        current_python_matches_preparation=manifest["python"] == helpers._python_identity(),
        planner_reported_identity_equal=identity_equal, model_identity_independently_verified=False,
        input_synthetic=manifest["input_synthetic"], synthetic=synthetic,
        scientific_result_status="synthetic" if synthetic else "declared_real_not_independently_audited",
        original_collection=manifest["original_collection"], shared_outline_source_costs=manifest["shared_outline_source_costs"],
        new_outline_calls=0, quality_status="pending_independent_review", musical_or_physical_quality_verified=False,
        note="Read-only recorded evidence audit, not fresh inference or authenticated service/tensor verification. "
             "Missing and failed songs remain in the selected-song denominator. Equality of text, tracks or timings is descriptive only. "
             "Current code mismatch prevents full request-reconstruction acceptance; run the frozen audit implementation."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", metavar="EXPERIMENT_DIR")
    mode.add_argument("--audit", metavar="EXPERIMENT_DIR")
    for name in ("dataset", "candidates", "config", "outlines", "out-dir", "out"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if any(getattr(args, name) is None for name in ("dataset", "candidates", "config", "outlines", "out_dir")):
            parser.error("--prepare requires --dataset --candidates --config --outlines --out-dir")
        report = prepare_history_experiment(args.dataset, args.candidates, args.config, args.outlines, args.out_dir)
        passed = report["ready_to_run"]
    elif args.verify:
        report, passed = verify_history_experiment(args.verify), True
        passed = report["ready_to_run"]
    else:
        report = audit_history_experiment(args.audit)
        passed = report["paired_control_verified"]
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
