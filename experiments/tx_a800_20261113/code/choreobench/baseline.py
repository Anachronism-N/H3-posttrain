"""Finite concat/global/hierarchical baselines with immutable call evidence.

Global and hierarchical providers see text proposals and public music structure,
not raw audio. The existing native model remains an audio-only local proposer.
"""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

from .audio import inspect_audio, make_windows, write_window
from .io import atomic_write, file_lock, file_sha, finite_json, object_sha, read_json, read_json_snapshot, write_json
from .providers import create_provider, parse_json_response
from .schema import validate_dataset, validate_plan


class ModelOutputError(ValueError):
    """A completed provider returned an unusable model answer, retained as raw."""


def _model_json(text):
    try:
        return parse_json_response(text)
    except (ValueError, TypeError) as error:
        raise ModelOutputError(str(error)) from error


def _model_plan(plan, song):
    try:
        return validate_plan(plan, song)
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def _config(config):
    allowed = {"mode", "local_provider", "planner_provider", "seed", "max_calls", "max_output_tokens",
               "planner_max_output_tokens", "window_seconds", "stride_seconds", "candidates_per_window", "splits", "continue_on_model_error", "local_candidates_path"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown baseline config fields")
    finite_json(config)
    cfg = dict(mode="concat", local_provider={"kind": "mock"}, planner_provider={"kind": "mock"},
               seed=42, max_calls=1000, max_output_tokens=2048, window_seconds=5.0625,
               stride_seconds=5.0625, candidates_per_window=1, splits=["train", "dev", "test"],
               continue_on_model_error=False, local_candidates_path=None)
    cfg.update(copy.deepcopy(config))
    cfg.setdefault("planner_max_output_tokens", cfg["max_output_tokens"])
    if cfg["mode"] not in ("concat", "global", "hierarchical"):
        raise ValueError("Unknown baseline mode")
    if type(cfg["continue_on_model_error"]) is not bool:
        raise ValueError("continue_on_model_error must be boolean")
    if cfg["local_candidates_path"] is not None and (
            not isinstance(cfg["local_candidates_path"], str) or not cfg["local_candidates_path"].strip()):
        raise ValueError("local_candidates_path must be a nonempty path or null")
    for key in ("max_calls", "max_output_tokens", "planner_max_output_tokens", "candidates_per_window"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Positive integer " + key + " required")
    if cfg["candidates_per_window"] > 32:
        raise ValueError("At most32 candidates per window")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2 ** 63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if (not isinstance(cfg["splits"], list) or not cfg["splits"] or len(set(cfg["splits"])) != len(cfg["splits"])
            or any(split not in ("train", "dev", "test") for split in cfg["splits"])):
        raise ValueError("splits must be a nonempty unique subset of train/dev/test")
    # Validate clock options even before a dataset is loaded.
    make_windows(81000, 16000, cfg["window_seconds"], cfg["stride_seconds"])
    return cfg


def _sealed(value):
    value = copy.deepcopy(value)
    value["record_sha256"] = object_sha(value)
    return value


def _verify(value):
    if not isinstance(value, dict) or value.get("record_sha256") != object_sha(
            {k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("Journal/result content hash differs")
    return value


class _Journal:
    def __init__(self, root, contract_sha, max_calls):
        self.root, self.contract_sha, self.max_calls = Path(root), contract_sha, max_calls
        self.directory = self.root / "calls"
        self.directory.mkdir(exist_ok=True)
        self.records = {}
        self.started = {}
        for path in sorted(self.directory.glob("*.started.json")):
            item = _verify(read_json(path))
            call_id = item["call_id"]
            if (item.get("contract_sha256") != contract_sha or call_id in self.started
                    or path.name != self._name(call_id) + ".started.json"
                    or item.get("request_sha256") != object_sha(item.get("request"))):
                raise ValueError("Invalid call start identity")
            self.started[call_id] = item
        for path in sorted(self.directory.glob("*.raw.json")):
            item = _verify(read_json(path))
            call_id = item["call_id"]
            start = self.started.get(call_id)
            if (start is None or path.name != self._name(call_id) + ".raw.json"
                    or item.get("started_sha256") != start["record_sha256"]
                    or item.get("contract_sha256") != contract_sha):
                raise ValueError("Unbound call response")
            self.records[call_id] = item
        if len(self.started) > max_calls:
            raise ValueError("Historical calls exceed immutable max_calls")

    @staticmethod
    def _name(call_id):
        return hashlib.sha256(call_id.encode()).hexdigest()

    def call(self, provider, request):
        call_id = request["call_id"]
        if call_id in self.started:
            if self.started[call_id]["request"] != request:
                raise ValueError("Cached request identity changed")
            if call_id not in self.records:
                raise ValueError("Pending/unknown provider outcome; automatic retry is forbidden: " + call_id)
        else:
            if len(self.started) >= self.max_calls:
                raise ValueError("Entire-run max_calls budget exhausted")
            start = _sealed(dict(schema="choreobench.call_started.v1", call_id=call_id,
                contract_sha256=self.contract_sha, request=request, request_sha256=object_sha(request),
                provider_identity=provider.identity, started_unix=time.time()))
            write_json(self.directory / (self._name(call_id) + ".started.json"), start)
            self.started[call_id] = start
            try:
                result = provider.generate(copy.deepcopy(request))
                record = dict(success=True, result=result, error=None)
            except Exception as error:
                record = dict(success=False, result=None, error=dict(type=type(error).__name__, message=str(error)))
            record = _sealed(dict(record, schema="choreobench.call_raw.v1", call_id=call_id,
                contract_sha256=self.contract_sha, started_sha256=start["record_sha256"], completed_unix=time.time()))
            write_json(self.directory / (self._name(call_id) + ".raw.json"), record)
            self.records[call_id] = record
        return _recorded_result(self.records[call_id], provider.identity, request)


def _recorded_result(record, identity, request):
    """Shared interpretation for live journal and immutable expert banks."""
    if record.get("success") is not True:
        raise ValueError("Recorded provider failure; not retried: " + str(record.get("error")))
    result = record.get("result")
    if (not isinstance(result, dict) or not isinstance(result.get("text"), str)
            or result.get("identity") != identity or type(result.get("synthetic")) is not bool):
        raise ValueError("Invalid provider result identity")
    if result.get("synthetic") != identity.get("synthetic"):
        raise ValueError("Provider result synthetic declaration differs from its identity")
    if not isinstance(result.get("costs"), dict):
        raise ValueError("Provider result lacks cost/unknown-cost metadata")
    if result.get("stop_reason") in (None, "http_error", "malformed_api_response", "infrastructure_error"):
        raise ValueError("API/infrastructure response unusable; raw retained: " + str(result.get("stop_reason")))
    if result.get("stop_reason") not in ("stop", "eos"):
        raise ModelOutputError("Provider did not finish normally; raw response retained: " + str(result.get("stop_reason")))
    if not result["text"].strip():
        raise ModelOutputError("Empty provider output retained; no automatic replacement")
    reported_tokens = result.get("costs", {}).get("output_tokens")
    if reported_tokens is not None and (type(reported_tokens) is not int or reported_tokens < 0
                                        or reported_tokens > request["max_output_tokens"]):
        raise ModelOutputError("Provider-reported output exceeds or violates the requested token budget; raw retained")
    return result


def _journal_accounting(journal):
    costs, unknown_outcomes, unknown_usage = [], [], []
    for call_id in journal.started:
        record = journal.records.get(call_id)
        result = record.get("result") if record else None
        reported = result.get("costs") if isinstance(result, dict) else None
        item = copy.deepcopy(reported) if isinstance(reported, dict) else {}
        item.setdefault("input_tokens", None)
        item.setdefault("output_tokens", None)
        costs.append(dict(item, call_id=call_id))
        if any(type(item[key]) is not int or item[key] < 0 for key in ("input_tokens", "output_tokens")):
            unknown_usage.append(call_id)
        if (record is None or record.get("success") is not True or not isinstance(result, dict)
                or result.get("stop_reason") in (None, "http_error", "malformed_api_response", "infrastructure_error")):
            unknown_outcomes.append(call_id)
    return dict(costs=costs, unknown_outcomes_count=len(unknown_outcomes), unknown_outcome_call_ids=unknown_outcomes,
                unknown_usage_count=len(unknown_usage), unknown_usage_call_ids=unknown_usage)


def _request(call_id, cfg, *, output_budget=None, **values):
    seed = int.from_bytes(hashlib.sha256((str(cfg["seed"]) + "/" + call_id).encode()).digest()[:8], "big") % (2 ** 63)
    return dict(call_id=call_id, seed=seed,
                max_output_tokens=cfg["max_output_tokens"] if output_budget is None else output_budget, **values)


def _empty_plan(song):
    return dict(schema="choreobench.plan.v1", song_id=song["song_id"], sample_rate=song["audio"]["sample_rate"],
                num_samples=song["audio"]["num_samples"], motifs=[], events=[], provenance={})


def _local_actions(text):
    from scripts_omnidance.exp.ms.music_planner_eval import parse_windows
    parsed = parse_windows(text)
    if not parsed["matches_fixed_windows"] or not parsed["nonempty_window_text"]:
        raise ModelOutputError("Local proposal violates existing fixed-two-window caption contract; raw retained")
    return parsed


def _concat(song, proposals):
    plan = _empty_plan(song)
    rate = song["audio"]["sample_rate"]
    observations = []
    for proposal in proposals:
        if proposal["candidate_index"] != 0:
            continue  # Concat has no hidden best-of-N selector.
        window, parsed = proposal["window"], proposal["parsed"]
        length = window["observation_end"] - window["observation_start"]
        split = min(length, int(math.floor(2. * rate + .5)))
        local_bounds = ((0, split), (split, length))
        for part, (bounds, caption) in enumerate(zip(local_bounds, parsed["windows"])):
            start = max(window["ownership_start"], window["observation_start"] + bounds[0])
            end = min(window["ownership_end"], window["observation_start"] + bounds[1])
            if end > start:
                plan["events"].append(dict(event_id="w%04d-p%d" % (window["index"], part), start=start,
                    end=end, track="body", action=caption["text"], attributes={}, entry={}, exit={}, motif_id=None,
                    evidence=[proposal["call_id"]]))
        observations.append(dict(window=window,
            raw_caption_intervals=[dict(start=w["start"], end=w["end"]) for w in parsed["windows"]],
            normalized_local_sample_intervals=[list(bounds) for bounds in local_bounds],
            normalization="Existing two-window format tolerance mapped to its nominal sample clock, then clipped to ownership"))
    plan["events"].sort(key=lambda e: (e["start"], e["track"], e["event_id"]))
    plan["provenance"] = dict(candidate_selection="index0_no_quality_selection", observations=observations,
                              timing_semantics="planned ownership schedule; not observed action onset/offset")
    validate_plan(plan, song)
    return plan


def _public_proposals(proposals, section=None):
    return [dict(call_id=p["call_id"], candidate_index=p["candidate_index"], observation=p["window"],
                 caption=p["text"])
            for p in proposals if section is None or (p["window"]["observation_start"] < section["end"]
                                                       and p["window"]["observation_end"] > section["start"])]


def _messages(payload, response_kind):
    common = ("You plan choreography from public musical structure and local audio-model text proposals. "
        "You have not heard the audio directly. All times are integer samples on the supplied global clock, "
        "with half-open intervals. Observation/beat/section boundaries are not measured action boundaries. "
        "Return only one JSON object, no Markdown. Do not invent audio observations or private references. "
        "Unknown attributes and entry/exit states must remain null or empty objects. "
        "Do not invent support feet, contact or facing states merely to satisfy validation; "
        "only carry state values explicitly supported by the supplied local text. ")
    if response_kind == "outline":
        instruction = ("Return {motifs:[{motif_id,description}],sections:[{section_id,instruction}]}, "
                       "one instruction for every supplied section in order. No other keys.")
    else:
        instruction = ("Return the COMPLETE choreobench.plan.v1 object ITSELF as the JSON root, "
            "not a wrapper such as {proposed_plan:...}. The root has exactly these keys: "
            "schema,song_id,sample_rate,num_samples,motifs,events,provenance. "
            "Use schema=choreobench.plan.v1; provenance,attributes,entry,exit are objects; motifs,events,evidence are arrays. "
            "The supplied proposed_plan is a draft. You may consolidate redundant proposals into coherent "
            "movement phrases rather than copying every draft event; keep actions concise and do not discard the public music evidence. "
            "Every event requires event_id,start,end,track,action,attributes,entry,exit,motif_id,evidence. "
            "Allowed tracks: body,upper,lower,head,torso,left_arm,right_arm,left_leg,right_leg. "
            "No overlapping incompatible scopes: body conflicts with all, upper with torso/head/arms, lower with legs. "
            "Events must be ordered by (start,track,event_id). Entry/exit/attributes values are scalar or null. "
            "Use only motifs declared in the plan. Keep the supplied song_id/sample_rate/num_samples. "
            "For section_plan keep every event within the supplied section, on the GLOBAL clock. "
            "This is a proposed schedule, not proof of execution or beat synchronization.")
    return [dict(role="system", content=common + instruction),
            dict(role="user", content=json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))]


def _validate_outline(outline, song, draft):
    if not isinstance(outline, dict) or set(outline) != {"motifs", "sections"} or not isinstance(outline["sections"], list):
        raise ModelOutputError("Malformed hierarchy outline")
    if len(outline["sections"]) != len(song["music"]["sections"]):
        raise ModelOutputError("Outline must include every section exactly once")
    for actual, expected in zip(outline["sections"], song["music"]["sections"]):
        if (not isinstance(actual, dict) or set(actual) != {"section_id", "instruction"}
                or actual["section_id"] != expected["section_id"]
                or not isinstance(actual["instruction"], str) or not actual["instruction"].strip()):
            raise ModelOutputError("Outline section identity/order/instruction differs")
    # Reuse actual valid local events solely to validate motif declarations;
    # the public plan schema deliberately does not accept an empty schedule.
    probe = copy.deepcopy(draft)
    probe["motifs"] = outline["motifs"]
    _model_plan(probe, song)
    return outline


def _section_draft(draft, section, index, motifs):
    result = copy.deepcopy(draft)
    result["motifs"] = copy.deepcopy(motifs)
    result["provenance"] = {}
    result["events"] = []
    for event in draft["events"]:
        if event["start"] < section["end"] and event["end"] > section["start"]:
            event = copy.deepcopy(event)
            event.update(start=max(section["start"], event["start"]), end=min(section["end"], event["end"]),
                         event_id="s%03d-%s" % (index, event["event_id"]))
            result["events"].append(event)
    return result


def _outer_plan(song, public, proposals, draft, provider, journal, cfg):
    if cfg["mode"] == "global":
        payload = dict(song=public, proposals=_public_proposals(proposals), proposed_plan=dict(draft, provenance={}))
        request = _request(song["song_id"] + "/global", cfg, output_budget=cfg["planner_max_output_tokens"],
                           response_kind="plan", **payload,
                           messages=_messages(payload, "plan"))
        plan = _model_json(journal.call(provider, request)["text"])
        _model_plan(plan, song)
        return plan
    payload = dict(song=public, proposals=_public_proposals(proposals))
    request = _request(song["song_id"] + "/outline", cfg, output_budget=cfg["planner_max_output_tokens"],
                       response_kind="outline", **payload,
                       messages=_messages(payload, "outline"))
    outline = _validate_outline(_model_json(journal.call(provider, request)["text"]), song, draft)
    plan = _empty_plan(song)
    plan["motifs"] = copy.deepcopy(outline["motifs"])
    for index, section in enumerate(song["music"]["sections"]):
        from .context import build_boundary_context
        payload = dict(song=public, section=section, outline=outline,
            section_instruction=outline["sections"][index]["instruction"],
            previous_section_tail=plan["events"][-2:], proposals=_public_proposals(proposals, section),
            boundary_context=build_boundary_context(plan, section["start"]),
            proposed_plan=_section_draft(draft, section, index, outline["motifs"]))
        request = _request(song["song_id"] + "/section/%04d" % index, cfg,
            output_budget=cfg["planner_max_output_tokens"], response_kind="section_plan", **payload,
            messages=_messages(payload, "section_plan"))
        part = _model_json(journal.call(provider, request)["text"])
        _model_plan(part, song)
        if any(e["start"] < section["start"] or e["end"] > section["end"] for e in part["events"]):
            raise ModelOutputError("Section plan escapes its permitted global interval")
        for motif in part["motifs"]:
            known = next((m for m in plan["motifs"] if m["motif_id"] == motif["motif_id"]), None)
            if known is not None and known != motif:
                raise ModelOutputError("Conflicting motif definitions across sections")
            if known is None:
                plan["motifs"].append(motif)
        plan["events"].extend(part["events"])
    # Sections are already ordered. The validator rejects duplicated IDs or
    # incompatible overlap; we do not repair/reorder malformed model events.
    _model_plan(plan, song)
    plan["provenance"]["outline"] = outline
    return plan


def _plan_for_song(song, source_identity, windows, output, cfg, local, planner, journal, public, candidate_bank=None):
    sid = song["song_id"]
    proposals = []
    safe = hashlib.sha256(sid.encode()).hexdigest()
    source_audio = inspect_audio(source_identity["path"])
    if {k: v for k, v in source_audio.items() if k != "pcm"} != source_identity:
        raise ValueError("Song bytes changed after preflight")
    if candidate_bank is not None:
        candidate_bank.assert_unchanged(full=True)
    for window in windows:
        audio = write_window(source_audio, window, output / "audio" / (safe + "-%04d.wav" % window["index"]))
        for candidate in range(cfg["candidates_per_window"]):
            call_id = sid + "/local/%04d/%02d" % (window["index"], candidate)
            request = _request(call_id, cfg, response_kind="local_caption", audio=audio)
            response = candidate_bank.call(request) if candidate_bank is not None else journal.call(local, request)
            proposals.append(dict(call_id=call_id, candidate_index=candidate, window=window,
                                  text=response["text"], parsed=_local_actions(response["text"])))
    draft = _concat(song, proposals)
    return draft if cfg["mode"] == "concat" else _outer_plan(song, public, proposals, draft, planner, journal, cfg)


def run_baseline(dataset_path, output_dir, config, resume=False):
    """Run once or resume unchanged cached calls; failures are retained, never resampled."""
    from .dataset import public_song, resolve_audio
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    cfg = _config(config)
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    songs = [song for song in dataset["songs"] if song["split"] in cfg["splits"]]
    if not songs:
        raise ValueError("No songs in selected splits")
    inputs, windows = {}, {}
    for song in songs:
        sid = song["song_id"]
        audio = inspect_audio(resolve_audio(song, dataset_path))
        if any(audio[key] != song["audio"][key] for key in ("sha256", "sample_rate", "num_samples")):
            raise ValueError("Actual audio differs from song identity")
        inputs[sid] = {k: v for k, v in audio.items() if k != "pcm"}
        windows[sid] = make_windows(audio["num_samples"], audio["sample_rate"], cfg["window_seconds"], cfg["stride_seconds"])
    bank = None
    if cfg["local_candidates_path"] is not None:
        from .candidates import load_candidates, match_candidates
        bank_path = Path(cfg["local_candidates_path"])
        if not bank_path.is_absolute():
            bank_path = dataset_path.parent / bank_path
        bank = load_candidates(bank_path)
        match_candidates(bank, dataset_sha, songs, inputs, cfg)
    known_failed = {state["song_id"] for state in bank.states if state["status"] != "complete"} if bank else set()
    expected_local_calls = sum(len(windows[s["song_id"]]) * cfg["candidates_per_window"] for s in songs)
    expected_outer_calls = sum(0 if s["song_id"] in known_failed or cfg["mode"] == "concat" else
        1 if cfg["mode"] == "global" else 1 + len(s["music"]["sections"]) for s in songs)
    expected_calls = (0 if bank else expected_local_calls) + expected_outer_calls
    if expected_calls > cfg["max_calls"]:
        raise ValueError("max_calls is insufficient for the complete run: requires " + str(expected_calls))
    if file_sha(dataset_path) != dataset_sha:
        raise ValueError("Dataset bytes changed after its immutable JSON snapshot")
    local = bank if bank else create_provider(cfg["local_provider"], role="local", base_dir=dataset_path.parent)
    planner = None if cfg["mode"] == "concat" else create_provider(cfg["planner_provider"], role="planner", base_dir=dataset_path.parent)
    sources = {name: file_sha(Path(__file__).with_name(name)) for name in
               ("audio.py", "providers.py", "baseline.py", "schema.py", "io.py", "dataset.py", "candidates.py", "context.py")}
    parser_source = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms" / "music_planner_eval.py"
    sources["music_planner_eval.py"] = file_sha(parser_source)
    packages = {}
    if bank is None and cfg["local_provider"].get("kind") == "native":
        for name in ("torch", "transformers", "peft", "accelerate", "deepspeed", "numpy", "scipy", "soundfile", "torchaudio"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
    software = dict(python=sys.version, executable=sys.executable, implementation=platform.python_implementation(),
                    platform=platform.platform(), native_packages=packages)
    contract = dict(schema="choreobench.run_contract.v1", dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha, dataset_snapshot=dataset, dataset_id=dataset["dataset_id"], grouping=dataset["grouping"],
        selected_songs=[s["song_id"] for s in songs], config=cfg,
        audio_identities=inputs, expected_calls=expected_calls, expected_local_calls=expected_local_calls,
        expected_new_local_calls=0 if bank else expected_local_calls, expected_outer_calls=expected_outer_calls,
        local_candidates_identity=bank.summary() if bank else None, software=software,
        provider_identities=dict(local=local.identity, planner=planner.identity if planner else None), code_sha256=sources,
        global_provider_input="text proposals and public music metadata; not raw audio",
        native_provider_input="derived full-window audio only; no global context or edits",
        synthetic=dataset["synthetic"] or (bank.data["synthetic"] if bank else local.identity["synthetic"])
                  or bool(planner and planner.identity["synthetic"]))
    if output.exists() and not resume:
        raise ValueError("Output already exists; use unchanged --resume or a new directory")
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume needs an existing complete run contract")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        if (output / "contract.json").exists():
            if read_json(output / "contract.json") != contract:
                raise ValueError("Resume dataset/config/model/source identity changed")
        else:
            write_json(output / "contract.json", contract)
        contract_sha = object_sha(contract)
        journal = _Journal(output, contract_sha, cfg["max_calls"])
        previous = _verify(read_json(output / "results.json")) if (output / "results.json").exists() else None
        if previous and previous.get("complete"):
            if previous.get("contract_sha256") != contract_sha or previous.get("call_records") != {
                    sid: record["record_sha256"] for sid, record in journal.records.items()}:
                raise ValueError("Completed result/call identity changed")
            if bank:
                bank.assert_unchanged(full=True)
                if any(key not in bank.entries or bank.entries[key]["raw"]["record_sha256"] != value
                       for key, value in previous.get("reused_call_records", {}).items()):
                    raise ValueError("Completed result/candidate bank identity changed")
            for ref in previous["plans"]:
                path = output / ref["path"]
                if file_sha(path) != ref["sha256"]:
                    raise ValueError("Published plan bytes changed")
                validate_plan(read_json(path), next(s for s in songs if s["song_id"] == ref["song_id"]))
            return previous
        plan_refs, failed_songs = [], []
        try:
            for song in songs:
                sid = song["song_id"]
                safe = hashlib.sha256(sid.encode()).hexdigest()
                try:
                    plan = _plan_for_song(song, inputs[sid], windows[sid], output, cfg, local, planner, journal, public_song(song), bank)
                except ModelOutputError as error:
                    failed_songs.append(dict(song_id=sid, error_type=type(error).__name__, message=str(error),
                                             retained_in_evaluation_denominator=True))
                    if cfg["continue_on_model_error"]:
                        continue
                    raise
                plan["provenance"]["baseline"] = dict(mode=cfg["mode"], contract_sha256=contract_sha,
                    synthetic=contract["synthetic"], observation_windows=windows[sid],
                    timing_semantics="proposed schedule; window boundaries are not measured action boundaries",
                    global_context_consumed_by_native_local_model=False)
                validate_plan(plan, song)
                # Portable IDs are validated by schema; hashing also keeps paths
                # safe for all platforms and prevents source IDs escaping output.
                relative = "plans/" + safe + ".json"
                destination = output / relative
                if destination.exists():
                    if read_json(destination) != plan:
                        raise ValueError("Existing published plan differs")
                else:
                    write_json(destination, plan)
                plan_refs.append(dict(song_id=sid, path=relative, sha256=file_sha(destination)))
            if bank:
                bank.assert_unchanged(full=True)
            result = dict(schema="choreobench.run_result.v1", complete=True, mode=cfg["mode"],
                synthetic=contract["synthetic"], contract_sha256=contract_sha, plans=plan_refs,
                selected_count=len(songs), success_count=len(plan_refs), failed_songs=failed_songs,
                all_songs_succeeded=not failed_songs,
                calls_used=len(journal.started), max_calls=cfg["max_calls"], max_output_tokens_per_call=cfg["max_output_tokens"],
                planner_max_output_tokens_per_call=cfg["planner_max_output_tokens"],
                new_calls_used=len(journal.started), local_candidates_reused=bank is not None,
                local_calls_reused=len(bank.reused) if bank else 0,
                original_collection_calls=bank.data["original_collection_calls"] if bank else 0,
                original_collection_costs=bank.costs if bank else [],
                reused_call_records=dict(bank.reused) if bank else {},
                call_records={sid: record["record_sha256"] for sid, record in journal.records.items()},
                **_journal_accounting(journal),
                music_or_physical_quality_verified=False, native_large_model_server_verified=False,
                remote_model_weight_identity_verified=False)
            result = _sealed(result)
            atomic_write(output / "results.json", result)
            return result
        except Exception as error:
            atomic_write(output / "results.json", _sealed(dict(schema="choreobench.run_result.v1", complete=False,
                synthetic=contract["synthetic"], contract_sha256=contract_sha, plans=plan_refs,
                selected_count=len(songs), success_count=len(plan_refs), failed_songs=failed_songs,
                all_songs_succeeded=False,
                calls_used=len(journal.started), error=dict(type=type(error).__name__, message=str(error)),
                new_calls_used=len(journal.started), local_candidates_reused=bank is not None,
                local_calls_reused=len(bank.reused) if bank else 0,
                original_collection_calls=bank.data["original_collection_calls"] if bank else 0,
                original_collection_costs=bank.costs if bank else [],
                reused_call_records=dict(bank.reused) if bank else {},
                **_journal_accounting(journal),
                call_records={sid: record["record_sha256"] for sid, record in journal.records.items()})))
            raise
