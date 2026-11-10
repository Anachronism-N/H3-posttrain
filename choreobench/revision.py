"""Budgeted diagnosis, audio reinspection and action-only plan revision.

The text judge is part of the inference policy, never an independent evaluator.
Diagnosis is collected once and shared across selection-policy experiments.
"""
import copy
import re
from pathlib import Path

from .audio import inspect_audio, write_window
from .baseline import ModelOutputError, _Journal, _journal_accounting, _request, _sealed, _verify
from .dataset import public_song, resolve_audio
from .editing import apply_task_patch, public_edit_task
from .io import atomic_write, file_lock, file_sha, json_text, object_sha, read_json, read_json_snapshot, write_json
from .music_recheck import create_music_observer, make_observation_window
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .revision_protocol import action_task, parse_diagnosis, parse_verdict, public_plan, select_events
from .schema import keys, validate_dataset, validate_plan


class _ModelFailure(ValueError):
    """A retained model attempt failed; never resample to hide this outcome."""


class _RunFailure(ValueError):
    """Provider/transport identity or outcome is unusable; stop further calls."""


def _config(config):
    defaults = dict(seed=7, selection_policy="targeted", max_repairs=2, listen=True,
                    max_calls=6, max_output_tokens=4096, max_request_characters=160000,
                    source_plan_synthetic=True, observer_provider=None)
    if not isinstance(config, dict) or set(config) - (set(defaults) | {"planner_provider"}):
        raise ValueError("Unknown revision configuration fields")
    cfg = dict(defaults, **copy.deepcopy(config))
    if not isinstance(cfg.get("planner_provider"), dict):
        raise ValueError("A text planner_provider is required")
    for key in ("listen", "source_plan_synthetic"):
        if type(cfg[key]) is not bool:
            raise ValueError(key + " must be boolean")
    for key, low, high in (("seed", 0, 2**63 - 1), ("max_repairs", 1, 8),
                           ("max_calls", 1, 24), ("max_output_tokens", 1, 65536),
                           ("max_request_characters", 1, 2000000)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError("Invalid revision " + key)
    if cfg["selection_policy"] not in ("targeted", "random"):
        raise ValueError("selection_policy must be targeted or random")
    if cfg["listen"] and not isinstance(cfg["observer_provider"], dict):
        raise ValueError("listen=true requires an explicit audio observer_provider")
    if not cfg["listen"] and cfg["observer_provider"] is not None:
        raise ValueError("Text-only control must not configure an audio observer")
    return cfg


def _tainted(value):
    if isinstance(value, dict):
        return value.get("synthetic") is True or any(_tainted(v) for v in value.values())
    return isinstance(value, list) and any(_tainted(v) for v in value)


def _inputs(dataset_path, plan_path):
    dataset, ds_sha = read_json_snapshot(dataset_path)
    before, plan_sha = read_json_snapshot(plan_path)
    validate_dataset(dataset)
    song = next((s for s in dataset["songs"] if s["song_id"] == before.get("song_id")), None)
    if song is None:
        raise ValueError("Source plan song is absent from the dataset")
    validate_plan(before, song)
    audio = inspect_audio(resolve_audio(song, dataset_path))
    if any(audio[k] != song["audio"][k] for k in ("sha256", "sample_rate", "num_samples")):
        raise ValueError("Source audio differs from dataset identity")
    sources = {str(Path(dataset_path).resolve()): ds_sha, str(Path(plan_path).resolve()): plan_sha,
               audio["path"]: audio["sha256"]}
    return dataset, song, before, audio, sources


def _unchanged(sources):
    for path, expected in sources.items():
        if file_sha(path) != expected:
            raise ValueError("Revision input/artifact changed: " + path)


def _provider(cfg, dataset_path):
    provider = create_provider(cfg["planner_provider"], role="planner", base_dir=Path(dataset_path).resolve().parent)
    if provider.identity["kind"] not in ("chat_completions", "replay"):
        raise ValueError("Revision requires an explicit text chat_completions or replay provider")
    return provider


def _contract(stage, sources, cfg, provider, synthetic, **extra):
    code = {name: file_sha(Path(__file__).with_name(name)) for name in
            ("revision.py", "revision_protocol.py", "music_recheck.py", "providers.py", "resident.py",
             "audio.py", "baseline.py", "request_budget.py", "editing.py", "dataset.py", "schema.py", "io.py")}
    return copy.deepcopy(dict(schema="choreobench.revision_contract.v1", stage=stage, sources=sources,
                config=cfg, provider_identity=provider.identity, code_sha256=code,
                synthetic=synthetic, model_weights_updated=False, **extra))


def _open_output(output, contract, resume):
    if output.exists() and not resume:
        raise ValueError("Revision output exists; use unchanged --resume or a new directory")
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume requires the original revision contract")
    output.mkdir(parents=True, exist_ok=True)


def _bind_output(output, contract):
    path = output / "contract.json"
    if path.exists():
        if read_json(path) != contract:
            raise ValueError("Revision input/config/provider/code identity changed")
    else:
        write_json(path, contract)


def _check_completed(output, contract, journal):
    path = output / "results.json"
    if not path.exists():
        return
    previous = _verify(read_json(path))
    if previous.get("contract_sha256") != object_sha(contract):
        raise ValueError("Previous revision result belongs to another contract")
    _unchanged({str(output / name): sha for name, sha in previous.get("artifacts", {}).items()})
    if previous.get("complete") and previous.get("call_records") != {
            k: r["record_sha256"] for k, r in journal.records.items()}:
        raise ValueError("Completed revision call inventory changed")


def _publish(output, name, value, artifacts):
    path = output / name
    if path.exists():
        if object_sha(read_json(path)) != object_sha(value):
            raise ValueError("Existing revision artifact changed: " + name)
    else:
        write_json(path, value)
    artifacts[name] = file_sha(path)


def _call(journal, provider, request, cfg, sources):
    try:
        guard_request(request, cfg)
    except ModelOutputError as error:
        raise _RunFailure(str(error)) from error
    _unchanged(sources)
    frozen = read_json(journal.root / "contract.json")
    if object_sha(frozen) != journal.contract_sha:
        raise ValueError("On-disk revision contract changed")
    expected_identity = copy.deepcopy(provider.identity)
    if expected_identity not in (frozen["provider_identity"], frozen.get("observer_identity")):
        raise _RunFailure("Provider identity differs from the frozen contract")
    start = journal.started.get(request["call_id"])
    if start and (start["request"] != request or start["provider_identity"] != provider.identity):
        raise ValueError("Cached revision request/provider changed")
    if start and request["call_id"] not in journal.records:
        raise _RunFailure("Pending/unknown revision call; no automatic retry")
    try:
        result = journal.call(provider, request)
    except ModelOutputError as error:
        raise _ModelFailure(str(error)) from error
    except (ValueError, TypeError, KeyError) as error:
        recorded = journal.records.get(request["call_id"])
        if recorded and recorded.get("success") is False:
            # Error dict key order differs between live construction and JSON
            # reload; decision artifacts must remain identical on resume.
            raise _RunFailure("Recorded provider failure; not retried: " + json_text(recorded["error"]).strip()) from error
        raise _RunFailure(str(error)) from error
    finally:
        _unchanged(sources)
        if provider.identity != expected_identity:
            raise _RunFailure("Provider identity changed during dispatch")
    return result


def _parse(parser, *args):
    try:
        return parser(*args)
    except (ValueError, TypeError, KeyError) as error:
        raise _ModelFailure(str(error)) from error


def _text_request(call_id, cfg, kind, instruction, payload):
    return _request(call_id, cfg, response_kind=kind, messages=[
        dict(role="system", content=instruction), dict(role="user", content=json_text(payload))])


def _accounting(journal):
    accounting = _journal_accounting(journal)
    accounting["costs"].sort(key=lambda row: row["call_id"])
    accounting["unknown_outcome_call_ids"].sort()
    accounting["unknown_usage_call_ids"].sort()
    return accounting


def _result(output, contract, journal, artifacts, complete, **extra):
    _unchanged(contract["sources"])
    if read_json(output / "contract.json") != contract:
        raise ValueError("On-disk revision contract changed")
    _unchanged({str(output / name): sha for name, sha in artifacts.items()})
    value = _sealed(dict(schema="choreobench.revision_result.v1", stage=contract["stage"], complete=complete,
        contract_sha256=object_sha(contract), synthetic=contract["synthetic"],
        artifacts=copy.deepcopy(artifacts), calls_used=len(journal.started),
        call_records={k: r["record_sha256"] for k, r in journal.records.items()},
        **_accounting(journal), semantic_success=None, quality_verified=False,
        model_weights_updated=False, **extra))
    path = output / "results.json"
    if path.exists() and _verify(read_json(path)).get("complete"):
        if read_json(path) != value:
            raise ValueError("Completed revision result changed")
    else:
        atomic_write(path, value)
    return value


def _bounded_diagnosis(text, plan):
    value = parse_diagnosis(text, plan)
    if len(value['issues']) > 8 or any(len(i['reason']) > 400 for i in value['issues']):
        raise ValueError('Diagnosis exceeds the explicit 8-issue / 400-character reason bound')
    return value


def _no_new_timing_literals(before, after, event_id):
    # A text-only action revision must not manufacture a precise onset while
    # preserving its event schedule. Existing source claims remain unverified.
    pattern = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:milliseconds?|ms|seconds?|secs?|s|samples?)\b'
        r'|\b(?:downbeat|beat|onset|timestamp|sample)\s+(?:at\s+)?\d+(?:\.\d+)?\b', re.IGNORECASE)
    action = lambda plan: next(e['action'] for e in plan['events'] if e['event_id'] == event_id)
    old = {v.lower().replace(' ', '') for v in pattern.findall(action(before))}
    new = {v.lower().replace(' ', '') for v in pattern.findall(action(after))}
    if not new <= old:
        raise ValueError('Action-only revision added a new precise timing literal; original schedule cannot be retimed in prose')


def diagnose_plan(dataset_path, plan_path, output_dir, config, resume=False):
    """Collect one initial-plan diagnosis for reuse by all experimental arms."""
    cfg = _config(config)
    dataset, song, before, audio, sources = _inputs(dataset_path, plan_path)
    provider = _provider(cfg, dataset_path)
    synthetic = dataset["synthetic"] or cfg["source_plan_synthetic"] or _tainted(before) or provider.identity["synthetic"]
    contract = _contract("diagnosis", sources, cfg, provider, synthetic, source_plan_sha256=object_sha(before))
    output = Path(output_dir).resolve()
    _open_output(output, contract, resume)
    with file_lock(output / ".run.lock"):
        _bind_output(output, contract)
        journal, artifacts = _Journal(output, object_sha(contract), 1), {}
        _check_completed(output, contract, journal)
        request = _text_request(song["song_id"] + "/revision/diagnosis", cfg, "revision_diagnosis",
            "Inspect a proposed choreography score using ONLY the supplied text and public music metadata. "
            "You have not heard audio. Identify concrete suspected problems worth checking, not verified faults. "
            "Reasonable motif recurrence is not automatically an error. Do not invent observed physical state. "
            "Return exactly {schema,source_plan_sha256,issues}. schema is choreobench.revision_diagnosis.v1; "
            "copy source_plan_sha256. issues is an array of {event_id,priority,reason}; each existing event at most "
            "once, priority integer 1..3 (3 most urgent), reason at most 400 characters. Return at most 8 issues, "
            "prioritizing a few concrete checks rather than listing every event. Return [] if no issue is supported. "
            "All event/beat numeric values are integer SAMPLES, not milliseconds; seconds = samples / sample_rate. "
            "Equal adjacent end/start is a valid half-open transition, not by itself a hard-cut error. "
            "A phrase boundary need not coincide with a beat. Do not treat any boundary mismatch alone as proof of a problem.",
            dict(source_plan_sha256=object_sha(before), source_plan=public_plan(before), music=public_song(song)))
        request["response_schema"] = dict(type="object", additionalProperties=False,
            required=["schema", "source_plan_sha256", "issues"], properties=dict(
                schema=dict(type="string", enum=["choreobench.revision_diagnosis.v1"]),
                source_plan_sha256=dict(type="string", enum=[object_sha(before)]),
                issues=dict(type="array", maxItems=min(8, len(before["events"])), items=dict(type="object",
                    additionalProperties=False, required=["event_id", "priority", "reason"], properties=dict(
                        event_id=dict(type="string", enum=[e["event_id"] for e in before["events"]]),
                        priority=dict(type="integer", minimum=1, maximum=3),
                        reason=dict(type="string", minLength=1, maxLength=400))))))
        try:
            response = _call(journal, provider, request, cfg, sources)
            diagnosis = _parse(_bounded_diagnosis, response["text"], before)
            record = _sealed(dict(schema="choreobench.revision_diagnosis_record.v1", diagnosis=diagnosis,
                dataset_sha256=file_sha(dataset_path), source_file_sha256=file_sha(plan_path),
                source_contract_sha256=object_sha(contract), synthetic=synthetic,
                call_records={k: r["record_sha256"] for k, r in journal.records.items()},
                collection_accounting=_accounting(journal)))
            _publish(output, "diagnosis.json", record, artifacts)
            return _result(output, contract, journal, artifacts, True, issue_count=len(diagnosis["issues"]))
        except (_ModelFailure, _RunFailure) as error:
            _result(output, contract, journal, artifacts, False, error=str(error))
            raise


def _diagnosis_record(path, dataset_path, plan_path, before):
    record, sha = read_json_snapshot(path)
    _verify(record)
    keys(record, ("schema", "diagnosis", "dataset_sha256", "source_file_sha256", "source_contract_sha256",
                  "synthetic", "call_records", "collection_accounting", "record_sha256"), where="diagnosis record")
    if (record["schema"] != "choreobench.revision_diagnosis_record.v1" or type(record["synthetic"]) is not bool
            or record["dataset_sha256"] != file_sha(dataset_path) or record["source_file_sha256"] != file_sha(plan_path)):
        raise ValueError("Diagnosis belongs to different original input bytes")
    # Reused records must satisfy the same current bounds as newly collected
    # diagnoses. Historical oversized records remain unchanged and are refused.
    _bounded_diagnosis(json_text(record["diagnosis"]), before)
    return record, sha


def _comparison(before, after, event_id, seed):
    revision_label = "a" if int(object_sha([seed, event_id])[:8], 16) % 2 else "b"
    plans = {revision_label: after, "b" if revision_label == "a" else "a": before}
    return dict(revision_label=revision_label, event_id=event_id,
                a_plan_sha256=object_sha(plans["a"]), b_plan_sha256=object_sha(plans["b"])), {
                    label: public_plan(plan) for label, plan in plans.items()}


def _choice_verdict(text, comparison, before, after):
    choice = parse_json_response(text)
    keys(choice, ("schema", "a_plan_sha256", "b_plan_sha256", "event_id", "choice", "reason"), where="revision choice")
    if (choice["schema"] != "choreobench.revision_choice.v1"
            or any(choice[k] != comparison[k] for k in ("a_plan_sha256", "b_plan_sha256", "event_id"))
            or choice["choice"] not in ("a", "b", "uncertain")):
        raise ValueError("Comparison choice identity or selection is invalid")
    decision = ("uncertain" if choice["choice"] == "uncertain" else
                "prefer_revision" if choice["choice"] == comparison["revision_label"] else "keep_original")
    return parse_verdict(json_text(dict(schema="choreobench.revision_verdict.v1",
        before_plan_sha256=object_sha(before), after_plan_sha256=object_sha(after),
        event_id=comparison["event_id"], decision=decision, reason=choice["reason"])),
        before, after, comparison["event_id"])


def run_revision(dataset_path, plan_path, diagnosis_path, output_dir, config, resume=False):
    """One attempt per selected event; failed/uncertain steps keep current plan."""
    cfg = _config(config)
    dataset, song, before, audio, sources = _inputs(dataset_path, plan_path)
    record, diagnosis_sha = _diagnosis_record(diagnosis_path, dataset_path, plan_path, before)
    sources[str(Path(diagnosis_path).resolve())] = diagnosis_sha
    diagnosis = record["diagnosis"]
    selected = select_events(before, diagnosis, cfg["selection_policy"], cfg["max_repairs"], cfg["seed"])
    upper_calls = len(selected) * (3 if cfg["listen"] else 2)
    if cfg["max_calls"] < upper_calls:
        raise ValueError("max_calls cannot cover all selected events: requires " + str(upper_calls))
    # Check every requested audio interval before opening a model or dispatching.
    initial_events = {e["event_id"]: e for e in before["events"]}
    windows = {s["event_id"]: make_observation_window(audio, initial_events[s["event_id"]])
               for s in selected} if cfg["listen"] else {}
    provider = _provider(cfg, dataset_path)
    observer = create_music_observer(cfg["observer_provider"], Path(dataset_path).resolve().parent) if cfg["listen"] and selected else None
    synthetic = (dataset["synthetic"] or cfg["source_plan_synthetic"] or _tainted(before)
                 or record["synthetic"] or provider.identity["synthetic"] or bool(observer and observer.identity["synthetic"]))
    contract = _contract("revision", sources, cfg, provider, synthetic,
        dataset_sha256=file_sha(dataset_path), source_plan_sha256=object_sha(before),
        selected=selected, maximum_new_calls=upper_calls,
        observer_identity=observer.identity if observer else None,
        diagnosis_collection_accounting=record["collection_accounting"], diagnosis_new_calls=0,
        source_execution_independently_verified=False, internal_judge_is_independent_evaluation=False)
    output = Path(output_dir).resolve()
    _open_output(output, contract, resume)
    with file_lock(output / ".run.lock"):
        _bind_output(output, contract)
        journal, artifacts, steps = _Journal(output, object_sha(contract), cfg["max_calls"]), {}, []
        _check_completed(output, contract, journal)
        current, fatal_error = copy.deepcopy(before), None
        _publish(output, "before.json", before, artifacts)
        _publish(output, "original.submission.json", dict(schema="choreobench.submission.v1", method_id="original",
            dataset_sha256=contract["dataset_sha256"], synthetic=synthetic,
            predictions=[dict(song_id=song["song_id"], path="before.json", sha256=artifacts["before.json"])]), artifacts)
        _publish(output, "selection.json", dict(policy=cfg["selection_policy"], selected=selected), artifacts)
        for index, selection in enumerate(selected):
            event_id = selection["event_id"]
            prefix = "steps/%02d" % index
            call_prefix = song["song_id"] + "/revision/" + object_sha(event_id)[:20]
            step = dict(event_id=event_id, status="pending", proposed=False, judged=False, accepted=False,
                        before_plan_sha256=object_sha(current), semantic_success=None)
            evidence = None
            _unchanged(sources)
            _publish(output, prefix + "/before.json", current, artifacts)
            try:
                if observer:
                    derived = write_window(audio, windows[event_id], output / (prefix + "/observation.wav"))
                    artifacts[prefix + "/observation.wav"] = file_sha(derived["path"])
                    response = _call(journal, observer, _request(call_prefix + "/listen", cfg,
                        response_kind="local_caption", audio=derived), cfg, sources)
                    event = initial_events[event_id]
                    evidence = dict(schema="choreobench.music_observation.v1", audio=derived, text=response["text"],
                        observed_start=derived["parent_start"], observed_end=derived["parent_end"],
                        target_overlap_start=max(event["start"], derived["parent_start"]),
                        target_overlap_end=min(event["end"], derived["parent_end"]),
                        covers_entire_event=derived["parent_start"] <= event["start"] and derived["parent_end"] >= event["end"],
                        synthetic=response["synthetic"], musical_facts_verified=False)
                    _publish(output, prefix + "/observation.json", evidence, artifacts)
                instruction = ("Review this event in its musical and choreographic context. Change only its action text "
                    "if the supplied evidence supports a useful improvement. Preserve the schedule, body scope, "
                    "motif, attributes, declared entry/exit states and all other events. Reasonable recurrence is allowed. "
                    "If evidence is insufficient or no improvement is justified, abstain with empty operations.")
                task = action_task(current, event_id, instruction)
                template = dict(schema="choreobench.patch.v1", base_plan_sha256=object_sha(current),
                    instruction=task["instruction"], allowed_event_ids=task["allowed_event_ids"],
                    repair_event_ids=task["repair_event_ids"], protected=task["protected"], operations=[])
                # Same inputs and prompts in both selection arms; no policy label or priority-derived instruction.
                observation_view = None if evidence is None else {k: v for k, v in evidence.items() if k not in ("audio", "synthetic")}
                payload = dict(source_plan=public_plan(current), target_event_id=event_id,
                    initial_plan_diagnosis=diagnosis, diagnosis_is_unverified=True, music=public_song(song),
                    music_observation=observation_view, task=public_edit_task(task), patch_template=template)
                response = _call(journal, provider, _text_request(call_prefix + "/edit", cfg, "patch",
                    "You revise one choreography event. Return exactly a complete JSON patch matching patch_template; "
                    "copy its first six fields and replace only operations. Each operation is {event_id,field,value}; "
                    "only the target event's action may change. Never treat supplied diagnosis or music observations as "
                    "verified truth. No invented audio or physical observations. Text must remain consistent with frozen "
                    "scope, timing, attributes and states. Numeric event/beat values are SAMPLES, with seconds = samples/sample_rate. "
                    "Do not add new numeric seconds/millisecond onset claims to action text; this action-only task cannot retime the schedule. "
                    "Equal adjacent end/start is legal and is not alone a transition fault. Empty operations is permitted.", payload), cfg, sources)
                patch = _parse(parse_json_response, response["text"])
                _publish(output, prefix + "/patch.json", patch, artifacts)
                candidate = _parse(apply_task_patch, current, patch, task)
                _parse(_no_new_timing_literals, current, candidate, event_id)
                validate_plan(candidate, song)
                step["proposed"] = True
                _publish(output, prefix + "/candidate.json", candidate, artifacts)
                _publish(output, prefix + "/task.json", task, artifacts)
                if current["events"] == candidate["events"]:
                    step["status"] = "no_op"
                else:
                    comparison, pair = _comparison(current, candidate, event_id, cfg["seed"])
                    _publish(output, prefix + "/comparison.json", comparison, artifacts)
                    response = _call(journal, provider, _text_request(call_prefix + "/judge", cfg, "revision_verdict",
                        "Compare choreography alternatives a and b using only supplied evidence. Neither is designated "
                        "as the preferred answer. This is an internal "
                        "candidate-selection heuristic, not an independent quality evaluation. Choose a or b only if "
                        "one alternative is better supported and does not conflict with timing, scope, attributes, states "
                        "or neighboring actions. Event and beat numbers are SAMPLES, not milliseconds; equal adjacent boundaries are legal. "
                        "Extra musical timing adjectives or repeated diagnosis claims do not alone establish an improvement. "
                        "Return uncertain for a tie or insufficient evidence. Return exactly "
                        "{schema,a_plan_sha256,b_plan_sha256,event_id,choice,reason}; schema is "
                        "choreobench.revision_choice.v1; copy hashes and event_id; choice is a, b or uncertain; "
                        "reason is nonempty text.",
                        dict(a_plan_sha256=comparison["a_plan_sha256"], b_plan_sha256=comparison["b_plan_sha256"],
                            event_id=event_id, pair=pair,
                            music=public_song(song), music_observation=observation_view)), cfg, sources)
                    verdict = _parse(_choice_verdict, response["text"], comparison, current, candidate)
                    _publish(output, prefix + "/verdict.json", verdict, artifacts)
                    step["judged"] = True
                    step["status"] = verdict["decision"]
                    if verdict["decision"] == "prefer_revision":
                        current = candidate
                        step["accepted"] = True
            except _ModelFailure as error:
                step.update(status="failed", error=str(error))
            except _RunFailure as error:
                fatal_error = str(error)
                step.update(status="failed", error=fatal_error)
            step["after_plan_sha256"] = object_sha(current)
            steps.append(step)
            _publish(output, prefix + "/decision.json", step, artifacts)
            if fatal_error is not None:
                for remaining in selected[index + 1:]:
                    steps.append(dict(event_id=remaining["event_id"], status="not_attempted_after_error",
                        proposed=False, judged=False, accepted=False, semantic_success=None,
                        before_plan_sha256=object_sha(current), after_plan_sha256=object_sha(current)))
                break
        _publish(output, "plan.json", current, artifacts)
        return _result(output, contract, journal, artifacts, fatal_error is None, steps=steps, error=fatal_error,
            plans=[dict(song_id=song["song_id"], path="plan.json", sha256=artifacts["plan.json"])],
            selected_count=len(selected), proposed_count=sum(s["proposed"] for s in steps),
            judged_count=sum(s["judged"] for s in steps), accepted_count=sum(s["accepted"] for s in steps),
            failed_count=sum(s["status"] == "failed" for s in steps),
            not_attempted_count=sum(s["status"] == "not_attempted_after_error" for s in steps),
            all_selected_attempts_usable=all(s["status"] not in ("failed", "not_attempted_after_error") for s in steps),
            final_plan_sha256=object_sha(current), diagnosis_new_calls=0,
            diagnosis_collection_accounting=record["collection_accounting"],
            note="Completion means processing the selected inventory, not successful choreography repair. "
                 "Failed, uncertain and no-op attempts retain the current plan and remain in the denominator.")
