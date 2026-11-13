"""Global design plus sequential block composition, not source-verbatim selection.

Generated actions and their timings are new proposals. Input citations bind what
was shown to the planner, not proof of musical fit, execution or semantic entailment.
"""
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

from .audio import make_windows
from .baseline import ModelOutputError, _Journal, _journal_accounting, _request, _sealed, _verify, _empty_plan
from .candidates import load_candidates, match_candidates
from .context import build_boundary_context
from .dataset import public_song, verify_dataset_audio
from .harness import _publish
from .io import atomic_write, file_lock, file_sha, object_sha, read_json, read_json_snapshot
from .planning_units import build_planning_units
from .providers import create_provider, parse_json_response
from .public_structure import load_public_structure
from .schema import identifier, keys, text, validate_dataset, validate_plan
from .selection import candidate_options
from .evaluation import evaluate_plan
from .timing import TimingInfeasibleError, event_clock, event_interval, literal_source_copies
from .request_budget import guard_request
from .schedule_diagnostics import schedule_diagnostics


_IMPORT_SIGNATURE = (Path(__file__).stat().st_ino, Path(__file__).stat().st_mtime_ns, Path(__file__).stat().st_ctime_ns)

LIMITS = dict(motifs=8, events_per_unit=8, action_characters=1600, design_characters=2400)
COMMON = (
    "Compose a POSSIBLE choreography, not an observed or uniquely recoverable performance. "
    "You have not heard audio directly. Local captions and public music structure are fallible predictions, "
    "not ground truth. Never claim measured beat synchronization, support-foot state, or verified physical feasibility. "
    "Stored plan times are GLOBAL integer samples with half-open intervals. Follow the declared event clock for output. "
    "A planning block is not a measured action boundary. "
    "Use concise connected movement phrases, not a list of isolated poses or paraphrases of every 2-second caption. "
    "Output one complete JSON object at the root, no wrapper, Markdown or commentary. "
)


def _candidates(options, unit=None):
    result = []
    for row in options["windows"]:
        window = row["window"]
        if unit and not (window["observation_start"] < unit["end"] and window["observation_end"] > unit["start"]):
            continue
        for c in row["candidates"]:
            result.append(dict(call_id=c["call_id"], observation=[window["observation_start"], window["observation_end"]],
                               text=c["raw_text"]))
    return result


def _bounded_text(value, name, maximum):
    text(value, name)
    if len(value) > maximum:
        raise ValueError(name + " exceeds the response length contract")


def _read_outline(response, song):
    try:
        value = parse_json_response(response)
        keys(value, ("motifs", "sections"), where="coordination outline")
        if not isinstance(value["motifs"], list) or len(value["motifs"]) > LIMITS["motifs"]:
            raise ValueError("Outline requires at most eight motifs")
        motifs = set()
        for motif in value["motifs"]:
            keys(motif, ("motif_id", "description"), where="outline motif")
            identifier(motif["motif_id"], "motif_id")
            _bounded_text(motif["description"], "motif.description", LIMITS["design_characters"])
            if motif["motif_id"] in motifs:
                raise ValueError("Duplicate outline motif")
            motifs.add(motif["motif_id"])
        expected = song["music"]["sections"]
        if not isinstance(value["sections"], list) or len(value["sections"]) != len(expected):
            raise ValueError("Outline must assign every public section in order")
        for section, source in zip(value["sections"], expected):
            keys(section, ("section_id", "intent", "motif_ids"), where="outline section")
            if section["section_id"] != source["section_id"]:
                raise ValueError("Outline cannot rename, omit or reorder public sections")
            _bounded_text(section["intent"], "section.intent", LIMITS["design_characters"])
            ids = section["motif_ids"]
            if (not isinstance(ids, list) or any(not isinstance(i, str) or i not in motifs for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Section references unknown/duplicate motifs")
        return value
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def _messages(payload, instruction):
    return [dict(role="system", content=COMMON + instruction),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))]


def _outline_request(song, options, units, cfg):
    payload = dict(song=public_song(song), planning_units=[{k: u[k] for k in
        ("unit_id", "section_id", "start", "end", "kind")} for u in units["units"]],
        local_proposals=_candidates(options), local_proposals_are_unverified=True)
    instruction = (
        "Design whole-song contrast, motif introduction, recurrence/variation and an ending. "
        "If functional sections are unknown, do not invent chorus/verse detections; your organization is a design goal. "
        "Return exactly {motifs:[{motif_id,description}],sections:[{section_id,intent,motif_ids}]}. "
        "At most eight motifs; each description must describe an identifiable movement phrase. "
        "Assign each supplied section exactly once, in order. Intent describes choreography development "
        "across that section and its blocks. Motif IDs are design declarations, not verified repeated audio. "
        "Do not output events, captions, a complete score, or extra keys."
    )
    return _guard_request(song["song_id"] + "/coordinate/outline", cfg,
        _messages(payload, instruction), "coordination_outline", cfg["outline_max_output_tokens"])


def _neutral_outline(song):
    """A schema placeholder, not an inferred or generated whole-song design."""
    return dict(motifs=[], sections=[dict(section_id=s["section_id"], intent="No generated section design provided.",
                                         motif_ids=[]) for s in song["music"]["sections"]])


def _unit_request(song, options, units, unit, outline, plan, cfg):
    clock = event_clock(unit, plan["sample_rate"], cfg)
    time_fields = ["start_boundary", "end_boundary"] if clock["format"] == "boundary_ids" else ["start", "end"]
    section_design = next(s for s in outline["sections"] if s["section_id"] == unit["section_id"])
    siblings = [u for u in units["units"] if u["section_id"] == unit["section_id"]]
    use_history, use_outline = cfg.get("use_history", True), cfg.get("use_outline", True)
    history = build_boundary_context(plan, unit["start"]) if use_history else None
    if history is not None:
        history["motif_memory"] = [m for m in history["motif_memory"] if m["motif_id"] in section_design["motif_ids"]]
    payload = dict(song_id=song["song_id"], sample_rate=plan["sample_rate"], num_samples=plan["num_samples"],
        brief=song["brief"], planning_unit={k: unit[k] for k in ("unit_id", "section_id", "start", "end", "kind")},
        position_in_section=dict(index=siblings.index(unit) + 1, total=len(siblings)),
        whole_song_design=dict(sections=outline["sections"]), current_section_design=section_design,
        current_motif_definitions=[m for m in outline["motifs"] if m["motif_id"] in section_design["motif_ids"]],
        unit_clock=dict(duration_samples=unit["end"] - unit["start"],
                        duration_seconds=(unit["end"] - unit["start"]) / plan["sample_rate"]),
        music_context=dict(source=song["music"]["source"],
            sections=[s for s in song["music"]["sections"] if s["start"] < unit["end"] and s["end"] > unit["start"]],
            beats=[t for t in song["music"]["beats"] if unit["start"] <= t < unit["end"]],
            downbeats=[t for t in song["music"]["downbeats"] if unit["start"] <= t < unit["end"]]),
        local_proposals=_candidates(options, unit), boundary_context=history,
        recent_committed_events=[{k: copy.deepcopy(e[k]) for k in
            ("event_id", "start", "end", "track", "action", "motif_id")} for e in plan["events"][-8:]],
        event_clock=clock,
        output_contract=dict(unit_id=unit["unit_id"], fields=time_fields + ["track", "action", "motif_id", "source_call_ids"],
            max_events=cfg["max_events_per_unit"], allowed_motif_ids=section_design["motif_ids"]))
    payload.update(outline_available=use_outline, history_available=use_history)
    if not use_outline:
        for name in ("whole_song_design", "current_section_design", "current_motif_definitions"):
            del payload[name]
    if not use_history:
        del payload["boundary_context"], payload["recent_committed_events"]
    instruction = (
        "Compose this block from the inputs explicitly supplied in this request; do not merely copy the fixed-window captions. "
        "You may synthesize a coherent phrase or vary its repetition, body scope and proposed timing WITHIN this unit. "
        "Return exactly {unit_id,events:[event,...]}; each event has exactly output_contract.fields, no extra keys. "
        "Prefer one to three connected movement phrases; eight is a SAFETY CEILING, not a target. "
        "Do not enumerate the motif list. Choose only motif_ids in output_contract.allowed_motif_ids or null; "
        "global future-section themes are NOT permission to introduce those motifs now. "
        "A phrase may span several seconds: use unit_clock and event_clock to choose time, "
        "and leave enough proposed time for the movements instead of packing many gestures into milliseconds. "
        "Use body for multi-part narratives; upper must not describe knee/foot actions and lower must not describe arms. "
        "Each event must cite one or more call_ids from the supplied local_proposals as input inspiration. "
        "Citations do not prove the new action; do not describe your synthesis as original source text. "
        "Times must remain within planning_unit.start/end; choose phrase boundaries, not automatic two-second cuts. "
        "Tracks: body,upper,lower,head,torso,left_arm,right_arm,left_leg,right_leg. "
        "body conflicts with every other track; upper with torso/head/arms, lower with legs. "
        "No overlapping incompatible scopes; order by start,track and preserve that order. "
        "Keep connections feasible but do not infer known support or facing from unknown prior states. "
        "Do not output entry/exit/attributes or event IDs; the controller leaves states unknown and assigns stable IDs. "
        "Under free scheduling a gap is allowed and reported, never filled with invented rest. "
        "When a public complete-unit schedule is supplied, use exactly one of its permitted interval arrays. "
        "Before returning check complete JSON and unit bounds."
    )
    if clock["format"] == "boundary_ids":
        instruction += ("Output start_boundary/end_boundary as integer indices into event_clock.boundary_samples. "
                        "Do NOT output start/end sample values or seconds. The controller compiles indices to exact samples. "
                        "Schedule time first, then describe each phrase. For sequential phrases, the next start_boundary "
                        "must be at least the previous end_boundary; equal boundaries are allowed for half-open intervals. "
                        "Do not assign the first body event the whole block and then layer more events inside it: "
                        "body owns every body part over its interval. A whole-block body phrase must be the sole event. "
                        "Boundary IDs restart from zero in EACH block; never carry an index from previous blocks. "
                        "History event times are GLOBAL samples, not indices into this block's boundary_samples. ")
    else:
        instruction += "Output start/end as GLOBAL integer sample indices. "
    instruction += ("Every event must last at least event_clock.minimum_duration_samples after compilation. "
                    "The grid is a scheduling device, not proof of musical or action boundaries. ")
    if clock.get("schedule_policy") == "balanced_full_unit":
        instruction += ("This request uses a PUBLIC complete-unit scheduling constraint: choose exactly ONE array from "
                        "event_clock.permitted_schedules and create one connected movement phrase for each of its intervals. "
                        "Copy the chosen start_boundary/end_boundary pairs exactly, in order. Prefer the number of phrases "
                        "that fits the public section intent and source proposals; three is not mandatory. "
                        "Compose movement across the entire chosen schedule, not invented rest to fill holes. "
                        "The offered durations are engineering design constraints, not verified musical or physical timing. "
                        "Do not return events outside or missing from the chosen schedule. ")
    if not use_outline:
        instruction += ("No generated whole-song outline or motifs are supplied in this ablation. "
                        "Compose from the public brief, current music context and local proposals; motif_id must be null. ")
    if not use_history:
        instruction += ("Previous committed action text, motif counts and body-state history are deliberately withheld. "
                        "This does not mean no previous movement happened; do not invent its content or state. ")
    request = _guard_request(song["song_id"] + "/coordinate/" + unit["unit_id"], cfg,
                             _messages(payload, instruction), "coordination_unit", cfg["max_output_tokens"])
    if cfg["structured_unit_output"]:
        request["response_schema"] = unit_response_schema(unit, section_design["motif_ids"],
                                                        [c["call_id"] for c in payload["local_proposals"]],
                                                        max_events=cfg["max_events_per_unit"], clock=clock,
                                                        max_request_characters=cfg["max_request_characters"])
    return guard_request(request, cfg)


def unit_response_schema(unit, motif_ids, call_ids, max_events=3, clock=None, max_request_characters=None):
    """Public decoder constraints only; original validation still checks relations/order."""
    fields = dict(
        start=dict(type="integer", minimum=unit["start"], maximum=unit["end"] - 1),
        end=dict(type="integer", minimum=unit["start"] + 1, maximum=unit["end"]),
        track=dict(type="string", enum=["body", "upper", "lower", "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]),
        action=dict(type="string", minLength=1, maxLength=LIMITS["action_characters"]),
        motif_id=dict(enum=list(motif_ids) + [None]),
        source_call_ids=dict(type="array", minItems=1, maxItems=len(call_ids),
                             items=dict(type="string", enum=list(call_ids))))
    if clock is not None:
        minimum = clock["minimum_duration_samples"]
        if clock["format"] == "boundary_ids":
            samples = clock["boundary_samples"]
            fields = dict(start_boundary=dict(type="integer", minimum=0,
                maximum=max(i for i, s in enumerate(samples) if samples[-1] - s >= minimum)),
                end_boundary=dict(type="integer", minimum=next(i for i, s in enumerate(samples) if s - samples[0] >= minimum),
                                  maximum=len(samples) - 1),
                **{name: value for name, value in fields.items() if name not in ("start", "end")})
        else:
            fields["start"]["maximum"] = unit["end"] - minimum
            fields["end"]["minimum"] = unit["start"] + minimum
    item = dict(type="object", additionalProperties=False, required=list(fields), properties=fields)
    if (clock is not None and clock["format"] == "boundary_ids"
            and clock.get("schedule_policy") != "balanced_full_unit"):
        if max_request_characters is not None:
            # Lower bound before copying potentially thousands of complete
            # branches. Zero indices have the shortest possible numeric width.
            # Root/message overhead can only increase the final request size.
            minimal = copy.deepcopy(item)
            minimal["properties"]["start_boundary"] = dict(type="integer", enum=[0])
            minimal["properties"]["end_boundary"]["minimum"] = 0
            lower_bound = len(json.dumps(minimal, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            if (fields["start_boundary"]["maximum"] + 1) * lower_bound > max_request_characters:
                raise ModelOutputError("Decoder schema alone exceeds max_request_characters; no branches expanded or request sent")
        branches = []
        for start_index in range(fields["start_boundary"]["maximum"] + 1):
            end_index = next(i for i, sample in enumerate(samples) if sample - samples[start_index] >= minimum)
            branch = copy.deepcopy(item)
            branch["properties"]["start_boundary"] = dict(type="integer", enum=[start_index])
            branch["properties"]["end_boundary"]["minimum"] = end_index
            branches.append(branch)
        # Complete alternatives avoid backends that ignore sibling properties
        # when anyOf is present. Bounds and field order live in every branch.
        item = dict(anyOf=branches)
    events_schema = dict(type="array", minItems=1, maxItems=max_events, items=item)
    if clock is not None and clock.get("schedule_policy") == "balanced_full_unit":
        alternatives = []
        for schedule in clock["permitted_schedules"]:
            if len(schedule) > max_events:
                continue
            slots = []
            for interval in schedule:
                slot = copy.deepcopy(item)
                for name, boundary in interval.items():
                    slot["properties"][name] = dict(type="integer", enum=[boundary])
                slots.append(slot)
            alternatives.append(dict(type="array", minItems=len(slots), maxItems=len(slots),
                                     prefixItems=slots, items=False))
        events_schema = dict(anyOf=alternatives)
    return dict(type="object", additionalProperties=False, required=["unit_id", "events"],
        properties=dict(unit_id=dict(type="string", enum=[unit["unit_id"]]), events=events_schema))


def _guard_request(call_id, cfg, messages, kind, budget):
    if sum(len(m["content"]) for m in messages) > cfg["max_request_characters"]:
        raise ModelOutputError("Coordination input exceeds max_request_characters; no text truncated or call sent")
    return guard_request(_request(call_id, cfg, response_kind=kind, messages=messages, output_budget=budget), cfg)


def _read_unit(response, original, unit, outline, options, max_events=3, clock=None):
    try:
        value = parse_json_response(response)
        keys(value, ("unit_id", "events"), where="coordinated unit")
        if value["unit_id"] != unit["unit_id"]:
            raise ValueError("Planner changed unit identity")
        if not isinstance(value["events"], list) or not 1 <= len(value["events"]) <= max_events:
            raise ValueError("Unit event count exceeds the explicit phrase budget")
        allowed = {c["call_id"] for c in _candidates(options, unit)}
        motif_ids = next(s["motif_ids"] for s in outline["sections"] if s["section_id"] == unit["section_id"])
        events = []
        clock = clock or event_clock(unit, original["audio"]["sample_rate"], {})
        time_fields = ("start_boundary", "end_boundary") if clock["format"] == "boundary_ids" else ("start", "end")
        for i, event in enumerate(value["events"]):
            keys(event, time_fields + ("track", "action", "motif_id", "source_call_ids"), where="coordinated event")
            start, end = event_interval(event, clock)
            if not unit["start"] <= start < end <= unit["end"]:
                raise ValueError("Proposed event escapes its immutable planning unit")
            _bounded_text(event["action"], "event.action", LIMITS["action_characters"])
            if event["motif_id"] is not None and (not isinstance(event["motif_id"], str) or event["motif_id"] not in motif_ids):
                raise ValueError("Event motif is outside the global section design")
            refs = event["source_call_ids"]
            if (not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in allowed for r in refs)
                    or len(refs) != len(set(refs))):
                raise ValueError("Event cites missing, duplicate or out-of-unit local input")
            events.append(dict(event_id=unit["unit_id"] + "-e%02d" % i,
                start=start, end=end, track=event["track"], action=event["action"],
                attributes={}, entry={}, exit={}, motif_id=event["motif_id"], evidence=copy.deepcopy(refs)))
        if clock.get("schedule_policy") == "balanced_full_unit":
            proposed_schedule = [{name: event[name] for name in time_fields} for event in value["events"]]
            if proposed_schedule not in clock["permitted_schedules"]:
                raise ValueError("Events do not match a publicly permitted full-unit schedule; no retiming or gap filling")
        probe = _empty_plan(original)
        probe.update(motifs=copy.deepcopy(outline["motifs"]), events=events)
        validate_plan(probe, original)
        return events
    except (ValueError, TypeError, KeyError) as error:
        raise ModelOutputError(str(error)) from error


def run_coordinated(dataset_path, candidates_path, output_dir, cfg, resume=False):
    current = Path(__file__).stat()
    if (current.st_ino, current.st_mtime_ns, current.st_ctime_ns) != _IMPORT_SIGNATURE:
        raise ValueError("Harness source changed after process import; start a fresh frozen process before model calls")
    dataset_path, output = Path(dataset_path).resolve(), Path(output_dir).resolve()
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    songs = [s for s in dataset["songs"] if s["split"] in cfg["splits"]]
    if not songs:
        raise ValueError("No songs in selected splits")
    bank = load_candidates(candidates_path)
    source_cfg = bank.data["source"]["contract"]["config"]
    bank_identity = match_candidates(bank, dataset_sha, songs, {s["song_id"]: s["audio"] for s in songs}, source_cfg)
    path = cfg["public_structure_path"]
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = dataset_path.parent / path
    effective, structure, structure_sha = load_public_structure(path, dataset_sha, songs)
    shared_path = cfg.get("shared_outline_path")
    if shared_path is not None:
        shared_path = Path(shared_path).expanduser()
        shared_path = shared_path.resolve() if shared_path.is_absolute() else (dataset_path.parent / shared_path).resolve()
    prepared, expected_calls = {}, 0
    for original, song in zip(songs, effective):
        windows = make_windows(song["audio"]["num_samples"], song["audio"]["sample_rate"],
                               source_cfg["window_seconds"], source_cfg["stride_seconds"])
        units = build_planning_units(song, windows, cfg["unit_config"])
        try:
            for unit in units["units"]:
                event_clock(unit, song["audio"]["sample_rate"], cfg)
            options = candidate_options(bank, original, units)
            prepared[song["song_id"]] = dict(song=song, units=units, options=options, error=None)
            expected_calls += int(cfg["use_outline"] and shared_path is None) + len(units["units"])
        except (ModelOutputError, TimingInfeasibleError) as error:
            prepared[song["song_id"]] = dict(song=song, units=units, error=str(error))
    shared, shared_sha, shared_rows = None, None, {}
    if shared_path is not None:
        from .outlines import load_shared_outlines
        shared, shared_sha = load_shared_outlines(shared_path, dataset_sha, bank.bytes_sha, effective,
            {sid: data["units"] for sid, data in prepared.items()}, structure_sha)
        shared_rows = {row["song_id"]: row for row in shared["songs"]}
    def assert_shared_unchanged():
        if shared_path is not None and file_sha(shared_path) != shared_sha:
            raise ValueError("Shared outline source bytes changed; start a new run")
    assert_shared_unchanged()
    if expected_calls > cfg["max_calls"]:
        raise ValueError("max_calls cannot cover enabled outline plus all coordination units: " + str(expected_calls))
    provider = create_provider(cfg["planner_provider"], role="planner", base_dir=dataset_path.parent)
    if provider.identity["kind"] == "mock":
        raise ValueError("Coordinated composition requires a real/replay text provider, never mock")
    synthetic = bool(dataset["synthetic"] or bank.data["synthetic"] or provider.identity["synthetic"]
                     or shared is not None and shared["synthetic"]
                     or structure and any(s["synthetic"] for s in structure["songs"]))
    contract = dict(schema="choreobench.coordination_contract.v1", dataset_sha256=dataset_sha,
        dataset_snapshot=dataset, config=cfg, selected_songs=[s["song_id"] for s in songs],
        local_candidates_identity=bank_identity, planner_identity=provider.identity,
        public_structure_sha256=structure_sha, public_structure_snapshot=structure,
        shared_outline_sha256=shared_sha, shared_outline_snapshot=shared,
        planning_songs_sha256={s["song_id"]: object_sha(s) for s in effective}, expected_calls=expected_calls,
        code_sha256={p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        software=dict(python=sys.version, platform=platform.platform()), synthetic=synthetic,
        original_local_sampling_unchanged=True, automatic_repair_loop=False, model_weights_updated=False,
        ablation=dict(use_outline=cfg["use_outline"], use_history=cfg["use_history"]),
        limits=LIMITS, generated_actions_are_new_proposals=True)
    if output.exists() and not resume:
        raise ValueError("Output exists; use unchanged resume or a new directory")
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume requires an existing coordination contract")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        _publish(output / "contract.json", contract)
        journal = _Journal(output, object_sha(contract), cfg["max_calls"])
        if (output / "results.json").is_file():
            previous = _verify(read_json(output / "results.json"))
            if previous.get("complete"):
                if previous["call_records"] != {k: r["record_sha256"] for k, r in journal.records.items()}:
                    raise ValueError("Completed coordination journal changed")
                for ref in previous["plans"] + previous["artifacts"]:
                    if file_sha(output / ref["path"]) != ref["sha256"]:
                        raise ValueError("Published coordination artifact changed")
                bank.assert_unchanged(full=True)
                assert_shared_unchanged()
                return previous
        plans, artifacts, failures = [], [], []
        def publish(relative, value):
            _publish(output / relative, value)
            artifacts.append(dict(path=relative, sha256=file_sha(output / relative)))
        def result(complete, error=None):
            return _sealed(dict(schema="choreobench.coordination_result.v1", complete=complete,
                contract_sha256=object_sha(contract), mode="coordinated", synthetic=synthetic,
                selected_count=len(songs), success_count=len(plans), failed_songs=failures,
                all_songs_succeeded=complete and not failures, plans=plans, artifacts=artifacts,
                expected_calls=expected_calls, calls_used=len(journal.started),
                call_records={k: r["record_sha256"] for k, r in journal.records.items()}, **_journal_accounting(journal),
                new_local_model_calls=0, original_collection_calls=bank.data["original_collection_calls"],
                original_collection_costs=bank.costs, model_weights_updated=False, automatic_repair_loop=False,
                shared_outline_sha256=shared_sha,
                shared_outline_source_costs=copy.deepcopy(shared["source_costs"]) if shared is not None else None,
                generated_actions_are_new_proposals=True, music_or_physical_quality_verified=False,
                ablation=dict(use_outline=cfg["use_outline"], use_history=cfg["use_history"]),
                semantic_status="pending_independent_review", error=error))
        try:
            for original in songs:
                sid = original["song_id"]
                prefix = hashlib.sha256(sid.encode()).hexdigest()
                data = prepared[sid]
                publish("units/" + prefix + ".json", data["units"])
                try:
                    if data["error"]:
                        raise ModelOutputError(data["error"])
                    options, units, song = data["options"], data["units"], data["song"]
                    publish("options/" + prefix + ".json", options)
                    assert_shared_unchanged()
                    if shared is not None:
                        outline = copy.deepcopy(shared_rows[sid]["outline"])
                    elif cfg["use_outline"]:
                        response = journal.call(provider, _outline_request(song, options, units, cfg))
                        outline = _read_outline(response["text"], song)
                    else:
                        outline = _neutral_outline(song)
                    publish("outlines/" + prefix + ".json", outline)
                    publish("outline_metadata/" + prefix + ".json", dict(generated_by_model=cfg["use_outline"],
                        generated_in_this_run=cfg["use_outline"] and shared is None,
                        source="shared_import" if shared is not None else "model_response" if cfg["use_outline"] else "deterministic_empty_placeholder",
                        outline_sha256=object_sha(outline), shared_outline_sha256=shared_sha,
                        use_outline=cfg["use_outline"], use_history=cfg["use_history"]))
                    plan = _empty_plan(original)
                    plan["motifs"] = copy.deepcopy(outline["motifs"])
                    for unit in units["units"]:
                        request = _unit_request(song, options, units, unit, outline, plan, cfg)
                        assert_shared_unchanged()
                        response = journal.call(provider, request)
                        events = _read_unit(response["text"], original, unit, outline, options, cfg["max_events_per_unit"],
                                            clock=event_clock(unit, song["audio"]["sample_rate"], cfg))
                        proposed = copy.deepcopy(plan)
                        proposed["events"].extend(events)
                        validate_plan(proposed, original)
                        publish("blocks/" + prefix + "/" + unit["unit_id"] + ".json", proposed)
                        # Only validated and committed blocks enter future body/motif context.
                        plan = proposed
                    plan["provenance"] = dict(mode="coordinated", contract_sha256=object_sha(contract),
                        outline_sha256=object_sha(outline), candidate_bank_sha256=bank.bytes_sha,
                        outline_generated_by_model=cfg["use_outline"],
                        outline_generated_in_this_run=cfg["use_outline"] and shared is None,
                        shared_outline_sha256=shared_sha,
                        public_structure_sha256=structure_sha, generated_actions_are_new_proposals=True,
                        ablation=dict(use_outline=cfg["use_outline"], use_history=cfg["use_history"]),
                        citation_semantics="Input inspiration, not verbatim source support or verified action facts",
                        boundaries="New proposed block-local schedule; no observed motion boundaries", synthetic=synthetic)
                    validate_plan(plan, original)
                    report = evaluate_plan(plan, original)
                    report.update(generated_actions_are_new_proposals=True, no_inferred_entry_exit=True,
                                  literal_source_copies=literal_source_copies(plan, options),
                                  event_timing_policy={k: cfg[k] for k in ("event_time_format", "event_grid_seconds", "min_event_seconds", "event_schedule_policy")},
                                  schedule_diagnostics=schedule_diagnostics(plan, units, cfg),
                                  ablation=dict(use_outline=cfg["use_outline"], use_history=cfg["use_history"]),
                                  full_song_design_sha256=object_sha(outline) if cfg["use_outline"] else None,
                                  semantic_goal_verification="pending")
                    publish("diagnostics/" + prefix + ".json", report)
                    relative = "plans/" + prefix + ".json"
                    _publish(output / relative, plan)
                    plans.append(dict(song_id=sid, path=relative, sha256=file_sha(output / relative)))
                except ModelOutputError as error:
                    failures.append(dict(song_id=sid, error_type=type(error).__name__, message=str(error),
                                         retained_in_evaluation_denominator=True))
                    if not cfg["continue_on_model_error"]:
                        raise
            bank.assert_unchanged(full=True)
            assert_shared_unchanged()
            final = result(True)
            atomic_write(output / "results.json", final)
            return final
        except Exception as error:
            atomic_write(output / "results.json", result(False, dict(type=type(error).__name__, message=str(error))))
            raise
