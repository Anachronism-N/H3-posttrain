"""Frozen-model, single-pass selection harness over a fixed local candidate bank.

Planning units only organize selection context. A source observation crossing
units still receives exactly one whole-response choice, never a retimed input.
No training, hidden-goal optimization or automatic repair loop is performed.
"""
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

from .audio import make_windows
from .baseline import ModelOutputError, _Journal, _journal_accounting, _request, _sealed, _verify
from .candidates import load_candidates, match_candidates
from .dataset import public_song, resolve_audio, verify_dataset_audio
from .io import atomic_write, file_lock, file_sha, finite_json, object_sha, read_json, read_json_snapshot, write_json
from .planning_units import build_planning_units
from .providers import create_provider, parse_json_response
from .public_structure import load_public_structure
from .public_style_prior import load_public_style_prior
from .schema import validate_dataset, validate_plan
from .selection import NoFeasibleSelectionError, candidate_options, make_selection, select_with_scores
from .compile_selected import compile_selection
from .request_budget import guard_request


_IMPORT_SIGNATURE = (Path(__file__).stat().st_ino, Path(__file__).stat().st_mtime_ns, Path(__file__).stat().st_ctime_ns)

def _config(value):
    allowed = {"mode", "selector_provider", "scores_path", "seed", "max_calls", "max_output_tokens",
               "splits", "unit_config", "timing_policy", "continue_on_model_error", "max_request_characters",
               "public_structure_path", "planner_provider", "outline_max_output_tokens", "structured_unit_output",
               "max_events_per_unit", "event_time_format", "event_grid_seconds", "min_event_seconds",
               "event_schedule_policy", "use_outline", "use_history", "shared_outline_path"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown inference harness config fields")
    finite_json(value)
    cfg = dict(mode="index0", selector_provider=None, scores_path=None, seed=7, max_calls=1000,
               max_output_tokens=4096, splits=["dev"], unit_config={}, timing_policy="strict",
               continue_on_model_error=True, max_request_characters=120000, public_structure_path=None,
               planner_provider=None, outline_max_output_tokens=4096, structured_unit_output=False, max_events_per_unit=3,
               event_time_format="global_samples", event_grid_seconds=.25, min_event_seconds=0,
               event_schedule_policy="free", use_outline=True, use_history=True, shared_outline_path=None)
    cfg.update(copy.deepcopy(value))
    if cfg["mode"] not in ("index0", "llm", "scores", "coordinated"):
        raise ValueError("Harness mode must be index0/llm/scores/coordinated")
    for name in ("use_outline", "use_history"):
        if type(cfg[name]) is not bool:
            raise ValueError(name + " must be a boolean")
        if cfg["mode"] != "coordinated" and name in value:
            raise ValueError("Outline/history ablations are only consumed by coordinated composition")
    if cfg["mode"] != "coordinated" and "shared_outline_path" in value:
        raise ValueError("Shared outlines are only consumed by coordinated composition")
    if cfg["shared_outline_path"] is not None:
        if not isinstance(cfg["shared_outline_path"], str) or not cfg["shared_outline_path"].strip():
            raise ValueError("shared_outline_path must be a nonempty path or null")
        if not cfg["use_outline"]:
            raise ValueError("shared_outline_path requires use_outline=true")
    if cfg["event_time_format"] not in ("global_samples", "boundary_ids"):
        raise ValueError("event_time_format must be global_samples/boundary_ids")
    if cfg["event_schedule_policy"] not in ("free", "balanced_full_unit"):
        raise ValueError("event_schedule_policy must be free/balanced_full_unit")
    if cfg["event_schedule_policy"] != "free" and cfg["event_time_format"] != "boundary_ids":
        raise ValueError("balanced_full_unit requires boundary_ids")
    for name in ("event_grid_seconds", "min_event_seconds"):
        if type(cfg[name]) not in (int, float) or cfg[name] < 0 or (name == "event_grid_seconds" and cfg[name] == 0):
            raise ValueError(name + " must be a finite number in its positive/nonnegative range")
    if cfg["mode"] != "coordinated" and set(value) & {"event_time_format", "event_grid_seconds", "min_event_seconds", "event_schedule_policy"}:
        raise ValueError("Event clock settings are only consumed by coordinated composition")
    if cfg["public_structure_path"] is not None and (not isinstance(cfg["public_structure_path"], str)
            or not cfg["public_structure_path"].strip()):
        raise ValueError("public_structure_path must be a nonempty path or null")
    if type(cfg["outline_max_output_tokens"]) is not int or cfg["outline_max_output_tokens"] < 1:
        raise ValueError("outline_max_output_tokens must be positive")
    if type(cfg["max_events_per_unit"]) is not int or not 1 <= cfg["max_events_per_unit"] <= 8:
        raise ValueError("max_events_per_unit must be an integer in [1,8]")
    if type(cfg["structured_unit_output"]) is not bool or (cfg["structured_unit_output"] and cfg["mode"] != "coordinated"):
        raise ValueError("structured_unit_output is a boolean option for coordinated composition only")
    if cfg["mode"] == "coordinated":
        if not isinstance(cfg["planner_provider"], dict):
            raise ValueError("coordinated mode requires planner_provider")
    elif cfg["planner_provider"] is not None:
        raise ValueError("Only coordinated mode consumes planner_provider")
    if cfg["timing_policy"] not in ("strict", "proposed_crop"):
        raise ValueError("timing_policy must be strict/proposed_crop")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2 ** 63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if type(cfg["max_calls"]) is not int or cfg["max_calls"] < 0:
        raise ValueError("max_calls must be a nonnegative integer")
    if type(cfg["max_output_tokens"]) is not int or cfg["max_output_tokens"] < 1:
        raise ValueError("max_output_tokens must be positive")
    if type(cfg["max_request_characters"]) is not int or cfg["max_request_characters"] < 1:
        raise ValueError("max_request_characters must be positive")
    if type(cfg["continue_on_model_error"]) is not bool or not isinstance(cfg["unit_config"], dict):
        raise ValueError("Invalid continue_on_model_error/unit_config")
    if (not isinstance(cfg["splits"], list) or not cfg["splits"]
            or any(not isinstance(s, str) or s not in ("train", "dev", "test") for s in cfg["splits"])
            or len(set(cfg["splits"])) != len(cfg["splits"])):
        raise ValueError("splits must be a unique nonempty train/dev/test subset")
    if cfg["mode"] == "llm":
        if not isinstance(cfg["selector_provider"], dict) or cfg["scores_path"] is not None:
            raise ValueError("llm mode requires selector_provider and no scores_path")
    elif cfg["selector_provider"] is not None:
        raise ValueError("Only llm mode consumes selector_provider")
    if cfg["mode"] == "scores":
        if not isinstance(cfg["scores_path"], str) or not cfg["scores_path"].strip():
            raise ValueError("scores mode requires scores_path")
    elif cfg["scores_path"] is not None:
        raise ValueError("Only scores mode consumes scores_path")
    if cfg["mode"] != "coordinated":
        for name in ("event_time_format", "event_grid_seconds", "min_event_seconds", "event_schedule_policy", "use_outline", "use_history", "shared_outline_path"):
            del cfg[name]
    return cfg


def _publish(path, value):
    if path.exists():
        if object_sha(read_json(path)) != object_sha(value):
            raise ValueError("Immutable harness artifact differs: " + str(path))
    else:
        write_json(path, value)


def _owned_windows(options, units):
    """One authoritative unit per window; all intersections remain context."""
    result = {u["unit_id"]: [] for u in units["units"]}
    for option in options["windows"]:
        window = option["window"]
        midpoint = (window["ownership_start"] + window["ownership_end"] - 1) // 2
        matches = [u for u in units["units"] if u["start"] <= midpoint < u["end"]]
        if len(matches) != 1:
            raise ValueError("Each observation window must have exactly one authoritative planning unit")
        result[matches[0]["unit_id"]].append(option["window_id"])
    return result


def _unit_request(song, options, units, unit, owned, choices, cfg, style_prior=None):
    by_id = {row["window_id"]: row for row in options["windows"]}
    context = []
    for choice in choices[-2:]:
        row = by_id[choice["window_id"]]
        candidate = next(c for c in row["candidates"] if c["candidate_index"] == choice["candidate_index"])
        retained = []
        for fragment in candidate["fragments"]:
            start, end = fragment["nominal_interval"]
            left = max(start, row["window"]["ownership_start"])
            right = min(end, row["window"]["ownership_end"])
            partial = left != start or right != end
            if left < right and (not partial or cfg["timing_policy"] == "proposed_crop"):
                retained.append(dict(text=fragment["text"], source_interval=[start, end],
                    scheduled_interval=[left, right], partial_source=partial,
                    semantic_timing_verified=False))
        context.append(dict(window_id=choice["window_id"], candidate_index=choice["candidate_index"],
                            scheduled_fragments=retained, timing_policy=cfg["timing_policy"],
                            omitted_fragment_count=len(candidate["fragments"]) - len(retained)))
    relevant = [row for row in options["windows"] if row["window_id"] in owned or
                row["window"]["observation_start"] < unit["end"] and row["window"]["observation_end"] > unit["start"]]
    # Send each source text once. Hashes and duplicate parsed text stay in the
    # auditable options artifact; the model needs IDs and scheduling metadata.
    relevant = [dict(window_id=row["window_id"], window=row["window"], unit_refs=row["unit_refs"],
                    candidates=[dict(candidate_index=c["candidate_index"], call_id=c["call_id"], raw_text=c["raw_text"],
                        fragments=[{key: f[key] for key in ("part_index", "nominal_interval", "ownership_intersection",
                                                          "fully_inside_ownership", "partial_source")} for f in c["fragments"]])
                                for c in row["candidates"]]) for row in relevant]
    public = public_song(song)
    payload = dict(song_id=song["song_id"], brief=public["brief"], sample_rate=public["sample_rate"],
        num_samples=public["num_samples"], planning_unit=unit, authoritative_window_ids=owned,
        music_context={"source": public["music"]["source"],
            "sections": [s for s in public["music"]["sections"] if s["start"] < unit["end"] and s["end"] > unit["start"]],
            "beats": [t for t in public["music"]["beats"] if unit["start"] <= t < unit["end"]],
            "downbeats": [t for t in public["music"]["downbeats"] if unit["start"] <= t < unit["end"]]},
        windows=relevant, prior_selected_text=context,
        response_template={"choices": [{"window_id": window_id, "candidate_index": 0} for window_id in owned]})
    if style_prior is not None:
        payload["style_prior"] = style_prior.signal(song["song_id"], unit)
    messages = [dict(role="system", content=(
        "Select existing WHOLE audio-expert responses to organize a choreography plan. You have not heard audio directly. "
        "Return only {\"choices\":[{\"window_id\":...,\"candidate_index\":...}]}. Choose exactly one available candidate "
        "for each authoritative_window_id in that order. Other overlapping windows are context only. "
        "Never rewrite actions, split a response's two captions, change times, or select outside the authoritative IDs. "
        "Use the public brief, structure and prior selected text; prefer coherent development and avoid unintended repetition. "
        "These preferences are proposed text organization, not proof of musicality, physical state or synchronization. "
        "Prior selected text contains only retained scheduled fragments; omitted fragments are not committed actions. "
        "Partial fragments have unverified duration semantics. No prior text establishes measured body state. "
        "Observation/ownership/block boundaries are not measured movement boundaries.")),
        dict(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))]
    if sum(len(message["content"]) for message in messages) > cfg["max_request_characters"]:
        raise ModelOutputError("Selector input exceeds max_request_characters; no request sent or source text truncated. "
                               "This is a character guard, not a model-specific tokenizer limit.")
    return guard_request(_request(song["song_id"] + "/select/" + unit["unit_id"], cfg,
                    response_kind="window_selection", messages=messages), cfg)


def _read_choices(text, expected, options):
    try:
        value = parse_json_response(text)
        if not isinstance(value, dict) or set(value) != {"choices"} or not isinstance(value["choices"], list):
            raise ValueError("Selector must return only a choices list")
        choices = value["choices"]
        if len(choices) != len(expected):
            raise ValueError("Selector omitted or added an authoritative window")
        by_id = {row["window_id"]: row for row in options["windows"]}
        for row, window_id in zip(choices, expected):
            if (not isinstance(row, dict) or set(row) != {"window_id", "candidate_index"}
                    or row["window_id"] != window_id or type(row["candidate_index"]) is not int
                    or row["candidate_index"] not in {c["candidate_index"] for c in by_id[window_id]["candidates"]}):
                raise ValueError("Selector changed scope/order or selected an unavailable whole response")
        return choices
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def run_harness(dataset_path, candidates_path, output_dir, config, resume=False, style_prior=None):
    current = Path(__file__).stat()
    if (current.st_ino, current.st_mtime_ns, current.st_ctime_ns) != _IMPORT_SIGNATURE:
        raise ValueError("Harness source changed after process import; start a fresh frozen process before model calls")
    cfg = _config(config)
    if cfg["mode"] == "coordinated":
        from .coordination import run_coordinated
        return run_coordinated(dataset_path, candidates_path, output_dir, cfg, resume=resume, style_prior=style_prior)
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    songs = [s for s in dataset["songs"] if s["split"] in cfg["splits"]]
    if not songs:
        raise ValueError("No songs in selected splits")
    bank = load_candidates(candidates_path)
    source_cfg = copy.deepcopy(bank.data["source"]["contract"]["config"])
    inputs = {s["song_id"]: s["audio"] for s in songs}
    bank_identity = match_candidates(bank, dataset_sha, songs, inputs, source_cfg)
    structure_path = cfg["public_structure_path"]
    if structure_path is not None:
        structure_path = Path(structure_path).expanduser()
        if not structure_path.is_absolute():
            structure_path = dataset_path.parent / structure_path
    planning_songs, structure_snapshot, structure_sha = load_public_structure(structure_path, dataset_sha, songs)
    if style_prior is not None:
        prior_path = Path(style_prior).expanduser()
        if not prior_path.is_absolute():
            prior_path = dataset_path.parent / prior_path
        style_prior = load_public_style_prior(prior_path, dataset_sha, dataset["songs"])
    def assert_style_prior_unchanged():
        if style_prior is not None:
            style_prior.assert_unchanged()
    assert_style_prior_unchanged()
    planning = {s["song_id"]: s for s in planning_songs}
    prepared, expected_calls = {}, 0
    for song in songs:
        windows = make_windows(song["audio"]["num_samples"], song["audio"]["sample_rate"],
                               source_cfg["window_seconds"], source_cfg["stride_seconds"])
        units = build_planning_units(planning[song["song_id"]], windows=windows, config=cfg["unit_config"])
        try:
            options = candidate_options(bank, song, units)
            owners = _owned_windows(options, units)
            expected_calls += sum(bool(v) for v in owners.values()) if cfg["mode"] == "llm" else 0
            prepared[song["song_id"]] = dict(units=units, options=options, owners=owners, error=None)
        except ModelOutputError as error:
            prepared[song["song_id"]] = dict(units=units, error=str(error))
    if expected_calls > cfg["max_calls"]:
        raise ValueError("max_calls cannot cover all authoritative planning units: " + str(expected_calls))
    provider = create_provider(cfg["selector_provider"], role="planner", base_dir=dataset_path.parent) if cfg["mode"] == "llm" else None
    if provider and provider.identity["kind"] == "mock":
        raise ValueError("Use index0 for deterministic smoke; llm selection requires chat_completions or replay")
    scores, scores_sha = None, None
    if cfg["mode"] == "scores":
        path = Path(cfg["scores_path"]).expanduser()
        if not path.is_absolute():
            path = dataset_path.parent / path
        scores, scores_sha = read_json_snapshot(path)
        if (not isinstance(scores, dict) or set(scores) != {"schema", "dataset_sha256", "bank_sha256", "songs"}
                or scores["schema"] != "choreobench.selection_scores.v1"
                or scores["dataset_sha256"] != dataset_sha or scores["bank_sha256"] != bank.bytes_sha
                or not isinstance(scores["songs"], dict) or set(scores["songs"]) != {s["song_id"] for s in songs}):
            raise ValueError("Selection scores must bind the exact dataset, bank and selected song inventory")
    synthetic = bool(dataset["synthetic"] or bank.data["synthetic"] or provider and provider.identity["synthetic"]
                     or structure_snapshot and any(s["synthetic"] for s in structure_snapshot["songs"]))
    if scores:
        for item in scores["songs"].values():
            if not isinstance(item, dict) or set(item) != {"options_sha256", "unary", "transitions", "score_provenance"}:
                raise ValueError("Invalid per-song selection score fields")
            provenance = item["score_provenance"]
            if not isinstance(provenance, dict) or type(provenance.get("synthetic")) is not bool:
                raise ValueError("Scores require explicit synthetic provenance")
            synthetic = synthetic or provenance["synthetic"]
    contract = dict(schema="choreobench.harness_contract.v1", dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha, dataset_snapshot=dataset, config=cfg,
        public_structure_sha256=structure_sha, public_structure_snapshot=structure_snapshot,
        planning_songs_sha256={s["song_id"]: object_sha(s) for s in planning_songs},
        selected_songs=[s["song_id"] for s in songs], local_candidates_identity=bank_identity,
        selector_identity=provider.identity if provider else None, scores_sha256=scores_sha, scores_snapshot=scores,
        units_sha256={sid: object_sha(p["units"]) for sid, p in prepared.items()}, expected_calls=expected_calls,
        code_sha256={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        software=dict(python=sys.version, platform=platform.platform()), synthetic=synthetic,
        model_weights_updated=False, requires_new_training_data=False, automatic_repair_loop=False)
    if style_prior is not None:
        contract.update(style_prior.binding(), sources={str(style_prior.path): style_prior.digest})
    if output.exists() and not resume:
        raise ValueError("Output exists; use unchanged resume or a new directory")
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume requires an existing harness contract")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        _publish(output / "contract.json", contract)
        contract_sha = object_sha(contract)
        journal = _Journal(output, contract_sha, cfg["max_calls"])
        previous = _verify(read_json(output / "results.json")) if (output / "results.json").exists() else None
        if previous and previous.get("complete"):
            if previous["contract_sha256"] != contract_sha or previous["call_records"] != {
                    key: r["record_sha256"] for key, r in journal.records.items()}:
                raise ValueError("Completed harness journal identity differs")
            bank.assert_unchanged(full=True)
            for ref in previous["plans"] + previous["artifacts"]:
                if file_sha(output / ref["path"]) != ref["sha256"]:
                    raise ValueError("Published harness artifact changed")
            assert_style_prior_unchanged()
            return previous
        plans, artifacts, failures = [], [], []
        def publish(relative, value):
            _publish(output / relative, value)
            ref = dict(path=relative, sha256=file_sha(output / relative))
            artifacts.append(ref)
            return ref
        def result(complete, error=None):
            return _sealed(dict(schema="choreobench.harness_result.v1", complete=complete,
                contract_sha256=contract_sha, synthetic=synthetic, mode=cfg["mode"], plans=plans, artifacts=artifacts,
                selected_count=len(songs), success_count=len(plans), failed_songs=failures,
                all_songs_succeeded=complete and not failures, calls_used=len(journal.started),
                expected_calls=expected_calls, call_records={key: r["record_sha256"] for key, r in journal.records.items()},
                **_journal_accounting(journal), new_local_model_calls=0,
                original_collection_calls=bank.data["original_collection_calls"], original_collection_costs=bank.costs,
                model_weights_updated=False, music_or_physical_quality_verified=False,
                semantic_status="pending_independent_review", automatic_repair_loop=False, error=error))
        try:
            for song in songs:
                sid, prefix = song["song_id"], hashlib.sha256(song["song_id"].encode()).hexdigest()
                data = prepared[sid]
                publish("units/" + prefix + ".json", data["units"])
                try:
                    if data["error"]:
                        raise ModelOutputError(data["error"])
                    options = data["options"]
                    publish("options/" + prefix + ".json", options)
                    choices, score_report = [], None
                    if cfg["mode"] == "index0":
                        choices = [dict(window_id=w["window_id"], candidate_index=0) for w in options["windows"]]
                    elif cfg["mode"] == "scores":
                        item = scores["songs"][sid]
                        if item["options_sha256"] != object_sha(options):
                            raise ValueError("Scores belong to another exact candidate options artifact")
                        try:
                            score_report = select_with_scores(options, item["unary"], item["transitions"],
                                                             score_provenance=item["score_provenance"])
                        except NoFeasibleSelectionError as error:
                            raise ModelOutputError(str(error)) from error
                        choices = score_report["choices"]
                    else:
                        for unit in data["units"]["units"]:
                            owned = data["owners"][unit["unit_id"]]
                            if not owned:
                                continue
                            request = _unit_request(planning[sid], options, data["units"], unit, owned, choices, cfg, style_prior)
                            assert_style_prior_unchanged()
                            response = journal.call(provider, request)
                            choices.extend(_read_choices(response["text"], owned, options))
                    selection = make_selection(bank, song, choices, selection_provenance=dict(
                        source=cfg["mode"], synthetic=synthetic, contract_sha256=contract_sha,
                        selector_identity_sha256=object_sha(provider.identity) if provider else None,
                        score_provenance=score_report["score_provenance"] if score_report else None))
                    publish("selections/" + prefix + ".json", selection)
                    if score_report:
                        publish("scores/" + prefix + ".json", score_report)
                    compilation = compile_selection(bank, song, selection, timing_policy=cfg["timing_policy"])
                    publish("compilation/" + prefix + ".json", compilation["report"])
                    if compilation["plan"] is None:
                        raise ModelOutputError("Strict source compilation retained no complete caption; no action invented")
                    validate_plan(compilation["plan"], song)
                    relative = "plans/" + prefix + ".json"
                    _publish(output / relative, compilation["plan"])
                    plans.append(dict(song_id=sid, path=relative, sha256=file_sha(output / relative)))
                except ModelOutputError as error:
                    failures.append(dict(song_id=sid, error_type=type(error).__name__, message=str(error),
                                         retained_in_evaluation_denominator=True))
                    if not cfg["continue_on_model_error"]:
                        raise
            bank.assert_unchanged(full=True)
            assert_style_prior_unchanged()
            final = result(True)
            atomic_write(output / "results.json", final)
            return final
        except Exception as error:
            atomic_write(output / "results.json", result(False, dict(type=type(error).__name__, message=str(error))))
            raise
