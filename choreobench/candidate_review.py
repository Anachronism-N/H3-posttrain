"""Read-only revision candidate audit and independently filled review material.

Historical acceptance, current guard checks and human preference stay separate.
No provider is constructed; all nonempty candidates, including rejected ones,
remain eligible for review. Source files and historical results are never edited.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

from .baseline import _sealed, _verify
from .dataset import resolve_audio
from .editing import apply_task_patch
from .evaluation import evaluate_edit
from .io import file_sha, json_text, object_sha, read_json_snapshot, write_json
from .review_page import render_review_page
from .reviews import _validate_bindings, blank_review_template, build_review_pack, validate_review_pack
from .revision import _no_new_timing_literals
from .revision_protocol import parse_diagnosis, parse_verdict, select_events
from .schema import integer, keys, text, validate_dataset, validate_plan


ARMS = ("targeted", "random", "text_only")
INVENTORY_SCHEMA = "choreobench.candidate_review_inventory.v1"
FOCUS_SCHEMA = "choreobench.candidate_focus.v1"
PREFERENCE_SCHEMA = "choreobench.candidate_preferences.v1"


def _within(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError("Artifact reference must remain inside its run")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("Artifact escapes its bound run")
    return result


def _snapshot(path, sources, expected=None):
    path = Path(path).resolve()
    value, sha = read_json_snapshot(path)
    if expected is not None and sha != expected:
        raise ValueError("Source bytes differ: " + str(path))
    if str(path) in sources and sources[str(path)] != sha:
        raise ValueError("Shared source changed")
    sources[str(path)] = sha
    return value


def _unchanged(sources):
    for path, sha in sources.items():
        if file_sha(path) != sha:
            raise ValueError("Candidate review source changed: " + path)


def _required_range(plan, event_id, context_seconds):
    event = next(e for e in plan["events"] if e["event_id"] == event_id)
    margin = int(context_seconds * plan["sample_rate"])
    start, end = max(0, event["start"] - margin), min(plan["num_samples"], event["end"] + margin)
    preceding = [e["end"] for e in plan["events"] if e["end"] <= event["start"]]
    following = [e["start"] for e in plan["events"] if e["start"] >= event["end"]]
    if preceding:
        last = max(preceding)
        start = min(start, min(e["start"] for e in plan["events"] if e["end"] == last))
    if following:
        first = min(following)
        end = max(end, max(e["end"] for e in plan["events"] if e["start"] == first))
    return dict(start=start, end=end)


def _read_run(root, sources, song, original, diagnosis):
    contract = _snapshot(root / "contract.json", sources)
    result = _verify(_snapshot(root / "results.json", sources))
    if (contract.get("stage") != "revision" or result.get("stage") != "revision"
            or result.get("contract_sha256") != object_sha(contract)):
        raise ValueError("Revision result/contract mismatch")
    for path, sha in contract["sources"].items():
        resolved = str(Path(path).resolve())
        if resolved in sources and sources[resolved] != sha:
            raise ValueError("Revision contract source mismatch")
        if file_sha(resolved) != sha:
            raise ValueError("Revision contract source bytes differ")
        sources[resolved] = sha
    artifacts = {}
    for relative, sha in result["artifacts"].items():
        path = _within(root, relative)
        if file_sha(path) != sha:
            raise ValueError("Historical revision artifact changed: " + relative)
        sources[str(path)] = sha
        artifacts[relative] = path
    started_files = sorted((root / "calls").glob("*.started.json"))
    for path in started_files:
        started = _verify(_snapshot(path, sources))
        if (started["contract_sha256"] != result["contract_sha256"]
                or started["request_sha256"] != object_sha(started["request"])
                or path.name != hashlib.sha256(started["call_id"].encode()).hexdigest() + ".started.json"):
            raise ValueError("Historical started-call identity mismatch")
    raw_files = sorted((root / "calls").glob("*.raw.json"))
    if len(started_files) != result["calls_used"] or len(raw_files) != len(result["call_records"]):
        raise ValueError("Historical call inventory differs from result")
    for call_id, sha in result["call_records"].items():
        name = hashlib.sha256(call_id.encode()).hexdigest()
        start = _verify(_snapshot(root / "calls" / (name + ".started.json"), sources))
        raw = _verify(_snapshot(root / "calls" / (name + ".raw.json"), sources))
        if (raw["record_sha256"] != sha or raw["started_sha256"] != start["record_sha256"]
                or raw["contract_sha256"] != result["contract_sha256"]
                or start["contract_sha256"] != result["contract_sha256"]
                or raw["call_id"] != call_id or start["call_id"] != call_id):
            raise ValueError("Historical call hash/identity mismatch")
    before = _snapshot(artifacts["before.json"], sources)
    if before != original or contract["source_plan_sha256"] != object_sha(original):
        raise ValueError("Arm does not start from the frozen original")
    cfg = contract["config"]
    selected = select_events(original, diagnosis, cfg["selection_policy"], cfg["max_repairs"], cfg["seed"])
    if selected != contract["selected"] or len(selected) != len(result["steps"]):
        raise ValueError("Historical selected-event inventory mismatch")
    if result["selected_count"] != len(selected):
        raise ValueError("Historical selected count mismatch")
    current, rows = copy.deepcopy(original), []
    for index, (selection, step) in enumerate(zip(selected, result["steps"])):
        event_id = selection["event_id"]
        if step["event_id"] != event_id or step["before_plan_sha256"] != object_sha(current):
            raise ValueError("Revision step order/base mismatch")
        prefix = "steps/%02d/" % index
        row = dict(event_id=event_id, status=step["status"], internal_accepted=step["accepted"],
                   candidate_available=False, nonempty_candidate=False, latest_timing_guard=None,
                   hard_scope_valid=None, before=current, candidate=None, task=None,
                   required_range=None)
        if prefix + "decision.json" in artifacts:
            if _snapshot(artifacts[prefix + "decision.json"], sources) != step:
                raise ValueError("Step decision differs from historical result")
        elif step["status"] != "not_attempted_after_error":
            raise ValueError("Executed step lacks a bound decision")
        if prefix + "candidate.json" in artifacts:
            candidate = _snapshot(artifacts[prefix + "candidate.json"], sources)
            task = _snapshot(artifacts[prefix + "task.json"], sources)
            patch = _snapshot(artifacts[prefix + "patch.json"], sources)
            if _snapshot(artifacts[prefix + "before.json"], sources) != current:
                raise ValueError("Step before does not match committed state")
            if apply_task_patch(current, patch, task) != candidate:
                raise ValueError("Candidate does not match the bound patch")
            validate_plan(candidate, song)
            if task["allowed_event_ids"] != [event_id] or task["repair_event_ids"]:
                raise ValueError("Candidate is not a single-event action-only revision")
            report = evaluate_edit(current, candidate, task)
            old = {e["event_id"]: e for e in current["events"]}
            new = {e["event_id"]: e for e in candidate["events"]}
            action_only = set(old) == set(new) and all(
                {k: v for k, v in e.items() if k != "action"} ==
                {k: v for k, v in new[eid].items() if k != "action"}
                and (eid == event_id or e["action"] == new[eid]["action"])
                for eid, e in old.items())
            if not action_only or not report["after_valid"] or report["overall_status"] == "literal_fail":
                raise ValueError("Candidate violates frozen action-only scope")
            nonempty = old[event_id]["action"] != new[event_id]["action"]
            try:
                _no_new_timing_literals(current, candidate, event_id)
                guard = dict(passed=True, reason=None)
            except ValueError as error:
                guard = dict(passed=False, reason=str(error))
            row.update(candidate_available=True, nonempty_candidate=nonempty, latest_timing_guard=guard,
                       hard_scope_valid=True, candidate=candidate, task=task)
            if step["judged"]:
                verdict = _snapshot(artifacts[prefix + "verdict.json"], sources)
                parse_verdict(json_text(verdict), current, candidate, event_id)
                if verdict["decision"] != step["status"]:
                    raise ValueError("Internal verdict and historical decision mismatch")
            if step["accepted"]:
                if not step["judged"] or step["status"] != "prefer_revision" or not nonempty:
                    raise ValueError("Historical acceptance lacks a changed, judged candidate")
                current = candidate
        elif step["accepted"] or step["proposed"] or step["judged"]:
            raise ValueError("Historical step claims a missing candidate")
        if step["after_plan_sha256"] != object_sha(current):
            raise ValueError("Historical after-state mismatch")
        rows.append(row)
    final = _snapshot(artifacts["plan.json"], sources)
    if final != current or result["final_plan_sha256"] != object_sha(current):
        raise ValueError("Final plan differs from historical commits")
    if result["accepted_count"] != sum(r["internal_accepted"] for r in rows):
        raise ValueError("Historical accepted count mismatch")
    return rows, result, contract


def prepare_candidate_review(inventory_path, output_dir, seed=7, context_seconds=5):
    """Audit the complete manifest-defined arm/step inventory, then blind pairs."""
    integer(seed, "seed")
    integer(context_seconds, "context_seconds", 1)
    if seed >= 2 ** 63 or context_seconds > 60:
        raise ValueError("Seed or review context exceeds the declared bound")
    output, inventory_path = Path(output_dir).resolve(), Path(inventory_path).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    sources = {}
    inventory = _snapshot(inventory_path, sources)
    keys(inventory, ("schema", "experiments"), where="candidate review inventory")
    if inventory["schema"] != INVENTORY_SCHEMA or not isinstance(inventory["experiments"], list) or not inventory["experiments"]:
        raise ValueError("Invalid candidate review experiment inventory")
    pairs, audit_rows, targets, songs, manifests = [], [], {}, {}, set()
    historical_calls = 0
    for entry in inventory["experiments"]:
        keys(entry, ("experiment_dir", "manifest_sha256"), where="experiment reference")
        root = Path(entry["experiment_dir"])
        if not root.is_absolute():
            root = inventory_path.parent / root
        root = root.resolve(strict=True)
        if root in manifests:
            raise ValueError("Duplicate revision experiment")
        manifests.add(root)
        manifest = _verify(_snapshot(root / "manifest.json", sources, entry["manifest_sha256"]))
        if manifest["schema"] != "choreobench.revision_experiment.v1" or manifest["execution_order"] != ["diagnosis", *ARMS]:
            raise ValueError("Expected a complete three-arm revision experiment")
        roles = {s["role"]: s for s in manifest["sources"]}
        if len(roles) != len(manifest["sources"]) or set(roles) != {"dataset", "plan", "base_config", "audio"}:
            raise ValueError("Invalid experiment source inventory")
        _snapshot(roles["base_config"]["path"], sources, roles["base_config"]["sha256"])
        dataset = _snapshot(roles["dataset"]["path"], sources, roles["dataset"]["sha256"])
        validate_dataset(dataset)
        original = _snapshot(roles["plan"]["path"], sources, roles["plan"]["sha256"])
        song = next(s for s in dataset["songs"] if s["song_id"] == manifest["song_id"])
        validate_plan(original, song)
        if manifest["source_plan_sha256"] != object_sha(original):
            raise ValueError("Manifest frozen plan identity mismatch")
        if song["song_id"] in songs:
            raise ValueError("Use one frozen experiment per song, not duplicate source versions")
        songs[song["song_id"]] = song
        audio = resolve_audio(song, roles["dataset"]["path"])
        if file_sha(audio) != song["audio"]["sha256"] or roles["audio"]["sha256"] != song["audio"]["sha256"]:
            raise ValueError("Canonical audio identity mismatch")
        sources[str(Path(audio).resolve())] = song["audio"]["sha256"]
        diagnosis_path = Path(manifest["shared_diagnosis_path"]).resolve()
        record = _verify(_snapshot(diagnosis_path, sources))
        if (record["dataset_sha256"] != roles["dataset"]["sha256"]
                or record["source_file_sha256"] != roles["plan"]["sha256"]):
            raise ValueError("Shared diagnosis source mismatch")
        diagnosis = parse_diagnosis(json_text(record["diagnosis"]), original)
        for arm in ARMS:
            run = Path(manifest["run_directories"][arm]).resolve(strict=True)
            rows, result, contract = _read_run(run, sources, song, original, diagnosis)
            if str(diagnosis_path) not in {str(Path(p).resolve()) for p in contract["sources"]}:
                raise ValueError("Arm does not bind the shared diagnosis")
            expected_policy, expected_listen = ("random" if arm == "random" else "targeted"), arm != "text_only"
            if contract["config"]["selection_policy"] != expected_policy or contract["config"]["listen"] != expected_listen:
                raise ValueError("Arm policy/listening intervention mismatch")
            historical_calls += result["calls_used"]
            for index, row in enumerate(rows):
                row.update(song_id=song["song_id"], arm=arm, step_index=index, run_dir=str(run),
                           source_code_sha256=contract["code_sha256"]["revision.py"],
                           result_file_sha256=sources[str(run / "results.json")])
                row["before_plan_sha256"] = object_sha(row["before"])
                row["candidate_plan_sha256"] = object_sha(row["candidate"]) if row["candidate"] is not None else None
                row["required_range"] = _required_range(original, row["event_id"], context_seconds)
                identity = (song["song_id"], row["event_id"])
                targets.setdefault(identity, dict(song=song, before=original, event_id=row["event_id"],
                                                  required_range=row["required_range"], source_arms=[]))
                targets[identity]["source_arms"].append(arm)
                if row["nonempty_candidate"]:
                    row["pair_id"] = "pair_%04d" % (len(pairs) + 1)
                    row["action_pair_sha256"] = object_sha([song["song_id"], row["before"]["events"], row["candidate"]["events"]])
                    pairs.append(row)
                audit_rows.append(row)
    if not pairs:
        raise ValueError("No nonempty candidates; retain the failed/no-op source inventory separately")
    rng, items = random.Random(seed), []
    for pair in pairs:
        pair["candidate_label"] = rng.choice(("a", "b"))
        for label in ("a", "b"):
            plan = pair["candidate"] if label == pair["candidate_label"] else pair["before"]
            items.append(dict(song=songs[pair["song_id"]], plan=plan,
                              method_id=pair["pair_id"] + "_" + label))
    package = build_review_pack(items, seed=seed)
    public, private = package["public"], package["private"]
    _validate_bindings(public, private)
    by_method = {row["method_id"]: row for row in private["items"]}
    focus = dict(schema=FOCUS_SCHEMA, pack_sha256=object_sha(public), pairs=[])
    for pair in pairs:
        a, b = (by_method[pair["pair_id"] + "_" + label] for label in ("a", "b"))
        focus["pairs"].append(dict(pair_id=pair["pair_id"], song_id=next(
            i["song"]["song_id"] for i in public["items"] if i["review_id"] == a["review_id"]),
            a_review_id=a["review_id"], b_review_id=b["review_id"],
            target_event_id=a["event_id_map"][pair["event_id"]], required_range=pair["required_range"]))
        pair.update(a_review_id=a["review_id"], b_review_id=b["review_id"])
    validate_focus(public, focus)
    preference = blank_preferences(public, focus)
    # Target judgments use the frozen original, without critic text or policy labels.
    target_items = [dict(song=t["song"], plan=t["before"], method_id="target_%04d" % (i + 1))
                    for i, t in enumerate(targets.values())]
    target_package = build_review_pack(target_items, seed=seed)
    target_public, target_private = target_package["public"], target_package["private"]
    target_methods = {r["method_id"]: r for r in target_private["items"]}
    target_focus = dict(schema="choreobench.target_focus.v1", pack_sha256=object_sha(target_public), targets=[])
    target_bindings = []
    for index, target in enumerate(targets.values()):
        bound = target_methods["target_%04d" % (index + 1)]
        target_focus["targets"].append(dict(review_id=bound["review_id"],
            target_event_id=bound["event_id_map"][target["event_id"]], required_range=target["required_range"]))
        target_bindings.append(dict(review_id=bound["review_id"], song_id=target["song"]["song_id"],
                                    event_id=target["event_id"], source_arms=target["source_arms"]))
    _unchanged(sources)
    output.mkdir(parents=True, exist_ok=False)
    for folder, pack, bindings in ((output, public, private), (output / "targets", target_public, target_private)):
        write_json(folder / "public/pack.json", pack)
        write_json(folder / "private/bindings.json", bindings)
        write_json(folder / "public/reviews.template.json", blank_review_template(pack))
        copied = set()
        for row in bindings["items"]:
            dest = folder / "public" / row["public_audio_path"]
            if dest in copied:
                continue
            song = row["original"]["song"]
            source = next(p for p, sha in sources.items() if sha == song["audio"]["sha256"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            if file_sha(dest) != song["audio"]["sha256"]:
                raise ValueError("Copied audio identity differs")
            copied.add(dest)
    write_json(output / "private/focus.json", focus)
    write_json(output / "private/preferences.template.json", preference)
    write_json(output / "targets/private/focus.json", target_focus)
    write_json(output / "targets/private/judgments.template.json", blank_target_judgments(target_public, target_focus))
    (output / "public/index.html").write_text(render_candidate_page(public, focus), encoding="utf-8")
    (output / "targets/public/index.html").write_text(render_target_page(target_public, target_focus), encoding="utf-8")
    private_rows = [{k: v for k, v in row.items() if k not in ("before", "candidate", "task")} for row in audit_rows]
    receipt = _sealed(dict(schema="choreobench.candidate_review_receipt.v1", sources=sources,
        code_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                     ("candidate_review.py", "revision.py", "revision_protocol.py", "reviews.py", "review_page.py", "editing.py", "io.py")},
        seed=seed, context_seconds=context_seconds,
        pairs_sha256=object_sha(focus), public_pack_sha256=object_sha(public),
        target_pack_sha256=object_sha(target_public), target_focus_sha256=object_sha(target_focus),
        targets=target_bindings, rows=private_rows,
        song_count=len(songs), arm_count=len(songs) * len(ARMS), selected_step_count=len(audit_rows),
        candidate_pair_count=len(pairs), unique_action_pair_count=len({p["action_pair_sha256"] for p in pairs}),
        target_count=len(targets), no_op_count=sum(r["status"] == "no_op" for r in audit_rows),
        historical_internal_accepted=sum(r["internal_accepted"] for r in audit_rows),
        latest_guard_rejected_nonempty=sum(not p["latest_timing_guard"]["passed"] for p in pairs),
        historical_revision_calls=historical_calls, new_model_calls=0, gpu_count=0,
        actual_human_reviews=0, quality_verified=False, historical_results_unchanged=True,
        limitations=["Hashes bind local evidence, not independent authentication of model weights or reviewers.",
                     "Current timing guard is a literal check, not a semantic timing validator.",
                     "Pairs include repeated originals and duplicate candidates; do not count them as independent songs.",
                     "All historical candidates remain in review, regardless of internal acceptance or current guard.",
                     "Reviewer roster is not supplied; blank forms and deployment are not independent evaluation."]))
    _unchanged(sources)
    write_json(output / "private/audit.json", receipt)
    return receipt


def validate_focus(pack, focus):
    validate_review_pack(pack)
    keys(focus, ("schema", "pack_sha256", "pairs"), where="candidate focus")
    if focus["schema"] != FOCUS_SCHEMA or focus["pack_sha256"] != object_sha(pack):
        raise ValueError("Candidate focus belongs to another public pack")
    by_id, seen, pair_ids = {i["review_id"]: i for i in pack["items"]}, set(), set()
    if not isinstance(focus["pairs"], list) or not focus["pairs"]:
        raise ValueError("Candidate pairs must be nonempty")
    for pair in focus["pairs"]:
        keys(pair, ("pair_id", "song_id", "a_review_id", "b_review_id", "target_event_id", "required_range"), where="candidate pair")
        text(pair["pair_id"], "pair_id")
        if pair["pair_id"] in pair_ids:
            raise ValueError("Duplicate pair ID")
        pair_ids.add(pair["pair_id"])
        for label in ("a", "b"):
            rid = pair[label + "_review_id"]
            if rid not in by_id or rid in seen:
                raise ValueError("Pair review ID missing or reused")
            seen.add(rid)
        a, b = (by_id[pair[label + "_review_id"]] for label in ("a", "b"))
        if a["song"] != b["song"] or a["song"]["song_id"] != pair["song_id"]:
            raise ValueError("Pair alternatives differ in music")
        if a["source_plan"] is not None or b["source_plan"] is not None:
            raise ValueError("Pair must not expose which alternative is the source")
        old, new = a["plan"], b["plan"]
        if {k: v for k, v in old.items() if k != "events"} != {k: v for k, v in new.items() if k != "events"}:
            raise ValueError("Pair differs outside event actions")
        old_events, new_events = {e["event_id"]: e for e in old["events"]}, {e["event_id"]: e for e in new["events"]}
        target = pair["target_event_id"]
        if set(old_events) != set(new_events) or target not in old_events:
            raise ValueError("Pair event identity mismatch")
        changes = [eid for eid in old_events if old_events[eid] != new_events[eid]]
        if changes != [target] or {k: v for k, v in old_events[target].items() if k != "action"} != {
                k: v for k, v in new_events[target].items() if k != "action"}:
            raise ValueError("Pair must differ only in one nonempty action")
        _range(pair["required_range"], old["num_samples"])
        if pair["required_range"]["start"] > old_events[target]["start"] or pair["required_range"]["end"] < old_events[target]["end"]:
            raise ValueError("Required context excludes target")
    if seen != set(by_id):
        raise ValueError("Pair focus must cover the entire public inventory")
    return focus


def _range(value, count):
    keys(value, ("start", "end"), where="review range")
    integer(value["start"], "range.start")
    integer(value["end"], "range.end", 1)
    if not value["start"] < value["end"] <= count:
        raise ValueError("Review range outside song")


def blank_preferences(pack, focus):
    validate_focus(pack, focus)
    by_id = {i["review_id"]: i for i in pack["items"]}
    return dict(schema=PREFERENCE_SCHEMA, pack_sha256=object_sha(pack), focus_sha256=object_sha(focus), reviews=[
        dict(pair_id=p["pair_id"], a_item_sha256=by_id[p["a_review_id"]]["item_sha256"],
             b_item_sha256=by_id[p["b_review_id"]]["item_sha256"], reviewer_id="FILL_REVIEWER_ID",
             origin="human", model_id=None, audio_seen=False, choice=None, reason="", reviewed_ranges=[])
        for p in focus["pairs"]])


def analyze_candidate_preferences(review_dir, reviews_paths):
    root, sources = Path(review_dir).resolve(), {}
    pack = _snapshot(root / "public/pack.json", sources)
    focus = validate_focus(pack, _snapshot(root / "private/focus.json", sources))
    bindings = _snapshot(root / "private/bindings.json", sources)
    _validate_bindings(pack, bindings)
    receipt = _verify(_snapshot(root / "private/audit.json", sources))
    if receipt["public_pack_sha256"] != object_sha(pack) or receipt["pairs_sha256"] != object_sha(focus):
        raise ValueError("Audit receipt public identity mismatch")
    historical = {r["pair_id"]: r for r in receipt["rows"] if "pair_id" in r}
    private_items = {r["review_id"]: r for r in bindings["items"]}
    if set(historical) != {p["pair_id"] for p in focus["pairs"]}:
        raise ValueError("Audit receipt omits or adds candidate pairs")
    for pair in focus["pairs"]:
        source = historical[pair["pair_id"]]
        if source["candidate_label"] not in ("a", "b"):
            raise ValueError("Invalid private candidate label")
        for label in ("a", "b"):
            expected = source["candidate_plan_sha256"] if label == source["candidate_label"] else source["before_plan_sha256"]
            if private_items[pair[label + "_review_id"]]["plan_sha256"] != expected:
                raise ValueError("Private preference direction does not match original evidence")
    by_pair = {p["pair_id"]: p for p in focus["pairs"]}
    template_rows = {r["pair_id"]: r for r in blank_preferences(pack, focus)["reviews"]}
    imported, seen = [], set()
    for path in reviews_paths:
        document = _snapshot(path, sources)
        keys(document, ("schema", "pack_sha256", "focus_sha256", "reviews"), where="candidate preferences")
        if (document["schema"] != PREFERENCE_SCHEMA or document["pack_sha256"] != object_sha(pack)
                or document["focus_sha256"] != object_sha(focus) or not isinstance(document["reviews"], list)):
            raise ValueError("Preference document identity mismatch")
        for row in document["reviews"]:
            keys(row, tuple(next(iter(template_rows.values()))), where="candidate preference")
            pid = row["pair_id"]
            if pid not in by_pair or any(row[k] != template_rows[pid][k] for k in ("a_item_sha256", "b_item_sha256")):
                raise ValueError("Preference alternatives differ from frozen pair")
            text(row["reviewer_id"], "reviewer_id")
            if row["reviewer_id"] == "FILL_REVIEWER_ID" or (pid, row["reviewer_id"]) in seen:
                raise ValueError("Missing reviewer or duplicate reviewer/pair")
            seen.add((pid, row["reviewer_id"]))
            if row["origin"] not in ("human", "model_judge") or (row["origin"] == "human" and row["model_id"] is not None):
                raise ValueError("Invalid reviewer origin/model identity")
            if row["origin"] == "model_judge":
                text(row["model_id"], "model_id")
            if type(row["audio_seen"]) is not bool or row["choice"] not in (None, "a", "b", "tie", "unknown"):
                raise ValueError("Invalid preference choice or audio declaration")
            if not isinstance(row["reason"], str) or not isinstance(row["reviewed_ranges"], list):
                raise ValueError("Invalid preference reason/ranges")
            if row["choice"] is not None:
                text(row["reason"], "preference reason")
            pair = by_pair[pid]
            count = next(i["plan"]["num_samples"] for i in pack["items"] if i["review_id"] == pair["a_review_id"])
            for interval in row["reviewed_ranges"]:
                _range(interval, count)
            cursor = pair["required_range"]["start"]
            for interval in sorted(row["reviewed_ranges"], key=lambda r: r["start"]):
                if interval["start"] <= cursor:
                    cursor = max(cursor, interval["end"])
            sufficient = cursor >= pair["required_range"]["end"]
            eligible = row["origin"] == "human" and row["audio_seen"] and sufficient and row["choice"] is not None
            source = historical[pid]
            preferred = ("candidate" if row["choice"] == source["candidate_label"] else "before") if row["choice"] in ("a", "b") else row["choice"]
            imported.append(dict(copy.deepcopy(row), range_sufficient=sufficient, eligible_human_judgment=eligible,
                preferred_version=preferred, historical_internal_accepted=source["internal_accepted"],
                possible_internal_misaccept=eligible and source["internal_accepted"] and preferred == "before",
                possible_internal_missed_improvement=eligible and not source["internal_accepted"] and preferred == "candidate"))
    _unchanged(sources)
    return _sealed(dict(schema="choreobench.candidate_preference_analysis.v1", sources=sources,
        pair_count=len(by_pair), independent_song_count=receipt["song_count"], rows=imported,
        eligible_human_judgments=sum(r["eligible_human_judgment"] for r in imported),
        pair_status={pid: "reviewed" if any(r["pair_id"] == pid and r["eligible_human_judgment"] for r in imported) else "pending"
                     for pid in by_pair}, expected_reviewer_roster_supplied=False, full_design_complete=None,
        quality_superiority_established=False, new_model_calls=0,
        note="Per-reviewer evidence only; no consensus, statistical winner, or completion without a registered roster."))


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", chr(92) + "u003c")


def render_candidate_page(pack, focus):
    validate_focus(pack, focus)
    page = render_review_page(pack)
    payload = dict(focus=focus, preferences=blank_preferences(pack, focus))
    return page.replace("</html>", '<script id="candidate-data" type="application/json">' + _encoded(payload) +
                        '</script><script>' + _PAIR_SCRIPT + '</script></html>')


def validate_target_focus(pack, focus):
    validate_review_pack(pack)
    keys(focus, ("schema", "pack_sha256", "targets"), where="target focus")
    if focus["schema"] != "choreobench.target_focus.v1" or focus["pack_sha256"] != object_sha(pack):
        raise ValueError("Target focus pack binding differs")
    by_id = {i["review_id"]: i for i in pack["items"]}
    seen = set()
    for row in focus["targets"]:
        keys(row, ("review_id", "target_event_id", "required_range"), where="target focus row")
        if row["review_id"] not in by_id or row["review_id"] in seen:
            raise ValueError("Target focus missing or duplicated")
        seen.add(row["review_id"])
        plan = by_id[row["review_id"]]["plan"]
        event = next((e for e in plan["events"] if e["event_id"] == row["target_event_id"]), None)
        _range(row["required_range"], plan["num_samples"])
        if event is None or row["required_range"]["start"] > event["start"] or row["required_range"]["end"] < event["end"]:
            raise ValueError("Target range excludes event")
    if seen != set(by_id):
        raise ValueError("Target focus must cover all selected locations")
    return focus


def blank_target_judgments(pack, focus):
    validate_target_focus(pack, focus)
    by_id = {i["review_id"]: i for i in pack["items"]}
    return dict(schema="choreobench.target_judgments.v1", pack_sha256=object_sha(pack),
        focus_sha256=object_sha(focus), reviews=[dict(review_id=t["review_id"],
            item_sha256=by_id[t["review_id"]]["item_sha256"], reviewer_id="FILL_REVIEWER_ID",
            origin="human", model_id=None, audio_seen=False, needs_revision=None, reason="", reviewed_ranges=[])
            for t in focus["targets"]])


def analyze_target_judgments(review_dir, reviews_paths):
    root, sources = Path(review_dir).resolve(), {}
    pack = _snapshot(root / "targets/public/pack.json", sources)
    focus = validate_target_focus(pack, _snapshot(root / "targets/private/focus.json", sources))
    _validate_bindings(pack, _snapshot(root / "targets/private/bindings.json", sources))
    receipt = _verify(_snapshot(root / "private/audit.json", sources))
    if receipt["target_pack_sha256"] != object_sha(pack) or receipt["target_focus_sha256"] != object_sha(focus):
        raise ValueError("Target audit identity mismatch")
    expected = {r["review_id"]: r for r in blank_target_judgments(pack, focus)["reviews"]}
    targets = {r["review_id"]: r for r in focus["targets"]}
    private_targets = {r["review_id"]: r for r in receipt["targets"]}
    rows, seen = [], set()
    for path in reviews_paths:
        document = _snapshot(path, sources)
        keys(document, ("schema", "pack_sha256", "focus_sha256", "reviews"), where="target judgments")
        if (document["schema"] != "choreobench.target_judgments.v1" or document["pack_sha256"] != object_sha(pack)
                or document["focus_sha256"] != object_sha(focus) or not isinstance(document["reviews"], list)):
            raise ValueError("Target judgment document identity mismatch")
        for row in document["reviews"]:
            keys(row, tuple(next(iter(expected.values()))), where="target judgment")
            rid = row["review_id"]
            if rid not in expected or row["item_sha256"] != expected[rid]["item_sha256"]:
                raise ValueError("Target judgment item identity mismatch")
            text(row["reviewer_id"], "reviewer_id")
            if row["reviewer_id"] == "FILL_REVIEWER_ID" or (rid, row["reviewer_id"]) in seen:
                raise ValueError("Missing reviewer or duplicate reviewer/target")
            seen.add((rid, row["reviewer_id"]))
            if row["origin"] not in ("human", "model_judge") or (row["origin"] == "human" and row["model_id"] is not None):
                raise ValueError("Invalid target reviewer origin/model identity")
            if row["origin"] == "model_judge":
                text(row["model_id"], "model_id")
            if type(row["audio_seen"]) is not bool or row["needs_revision"] not in (None, "yes", "no", "unknown"):
                raise ValueError("Invalid target judgment or audio declaration")
            if not isinstance(row["reason"], str) or not isinstance(row["reviewed_ranges"], list):
                raise ValueError("Invalid target judgment reason/ranges")
            if row["needs_revision"] is not None:
                text(row["reason"], "target judgment reason")
            count = next(i["plan"]["num_samples"] for i in pack["items"] if i["review_id"] == rid)
            for interval in row["reviewed_ranges"]:
                _range(interval, count)
            cursor = targets[rid]["required_range"]["start"]
            for interval in sorted(row["reviewed_ranges"], key=lambda r: r["start"]):
                if interval["start"] <= cursor:
                    cursor = max(cursor, interval["end"])
            sufficient = cursor >= targets[rid]["required_range"]["end"]
            rows.append(dict(copy.deepcopy(row), source_arms=private_targets[rid]["source_arms"], range_sufficient=sufficient,
                eligible_human_judgment=row["origin"] == "human" and row["audio_seen"] and sufficient and row["needs_revision"] is not None))
    _unchanged(sources)
    return _sealed(dict(schema="choreobench.target_judgment_analysis.v1", sources=sources, target_count=len(targets), rows=rows,
        eligible_human_judgments=sum(r["eligible_human_judgment"] for r in rows),
        target_status={rid: "reviewed" if any(r["review_id"] == rid and r["eligible_human_judgment"] for r in rows) else "pending" for rid in targets},
        expected_reviewer_roster_supplied=False, full_design_complete=None, diagnosis_recall=None, new_model_calls=0,
        note="Judgments of selected locations only, not diagnosis recall; no inferred consensus or reviewer roster."))


def render_target_page(pack, focus):
    validate_target_focus(pack, focus)
    page = render_review_page(pack)
    payload = dict(focus=focus, judgments=blank_target_judgments(pack, focus))
    return page.replace("</html>", '<script id="target-data" type="application/json">' + _encoded(payload) +
                        '</script><script>' + _TARGET_SCRIPT + '</script></html>')


_PAIR_SCRIPT = r'''
const candidateData=JSON.parse(document.getElementById('candidate-data').textContent);
document.title='局部候选独立盲评';document.querySelector('h1').textContent=document.title;
document.querySelector('#rubric p').textContent='先听取共同上下文，对同一比较的A/B独立评分，再填写局部偏好、具体理由及实际已审范围。没有内部接受/拒绝标签；不默认任何一侧更好。整体组织维度在此局部包禁用，须到整曲包单独评价。';
const cards=new Map([...document.querySelectorAll('.card')].map(c=>[c.querySelector('h3').textContent,c]));
function markFocus(card,target){
 card.querySelectorAll('tr[data-start]').forEach(row=>{if(row.cells[0].textContent===target){row.style.outline='2px solid #3366cc';row.dataset.target='true'}});
 [...card.querySelectorAll('label')].find(l=>l.textContent.startsWith(labels.long_range_structure)).querySelector('select').disabled=true;
}
candidateData.focus.pairs.forEach(pair=>{
 const a=cards.get(pair.a_review_id),b=cards.get(pair.b_review_id),parent=a.parentElement;
 [a,b].forEach((card,index)=>{const h=card.querySelector('h3');h.dataset.reviewId=h.textContent;h.textContent=pair.pair_id+' · '+(index?'B':'A');markFocus(card,pair.target_event_id)});
 parent.append(a,b);
 const form=E('section');form.className='card pair-form';form.dataset.pair=pair.pair_id;form.style.gridColumn='1 / -1';
 const song=pack.items.find(i=>i.review_id===pair.a_review_id).song;
 const seconds=v=>(v/song.audio.sample_rate).toFixed(4);
 form.append(E('h3',pair.pair_id+' · 独立局部比较'),E('p','目标 '+pair.target_event_id+'；共同上下文 '+seconds(pair.required_range.start)+'–'+seconds(pair.required_range.end)+' 秒。蓝框为目标，完整时间线可查。'));
 const seek=E('button','播放共同上下文');seek.onclick=()=>{const audio=form.closest('[id^="case-"]').querySelector('audio');audio.currentTime=pair.required_range.start/song.audio.sample_rate;audio.play().catch(()=>{});
 const stop=()=>{if(audio.currentTime>=pair.required_range.end/song.audio.sample_rate){audio.pause();audio.removeEventListener('timeupdate',stop)}};audio.addEventListener('timeupdate',stop)};form.append(seek);
 const review=candidateData.preferences.reviews.find(r=>r.pair_id===pair.pair_id),choice=E('select');choice.className='pair-choice';
 [['','未评价'],['a','A更好'],['b','B更好'],['tie','相当'],['unknown','无法判断']].forEach(([v,t])=>choice.append(new Option(t,v)));
 choice.onchange=()=>review.choice=choice.value||null;
 const reason=E('textarea');reason.className='pair-reason';reason.placeholder='引用动作/音乐依据，区分新增错误、合理重复和无法判断';reason.oninput=()=>review.reason=reason.value;
 const start=E('input'),end=E('input');start.className='reviewed-start';end.className='reviewed-end';start.placeholder='实际已审起点（采样点）';end.placeholder='实际已审终点（采样点）';
 const add=E('button','记录已审区间'),ranges=E('pre');add.onclick=()=>{
 if(!/^(0|[1-9][0-9]*)$/.test(start.value)||!/^(0|[1-9][0-9]*)$/.test(end.value)||!Number.isSafeInteger(Number(start.value))||!Number.isSafeInteger(Number(end.value))||Number(start.value)>=Number(end.value)||Number(end.value)>song.audio.num_samples){alert('填写音频内非空整数采样区间');return}
 review.reviewed_ranges.push({start:Number(start.value),end:Number(end.value)});ranges.textContent=JSON.stringify(review.reviewed_ranges);start.value='';end.value=''};
 const clear=E('button','清空区间草稿');clear.onclick=()=>{review.reviewed_ranges=[];ranges.textContent=''};
 form.append(choice,reason,E('p','实际已审区间留空，不自动填为已听；可记录多段半开区间。'),start,end,add,clear,ranges);parent.append(form);
});
const exportPreferences=E('button','另存独立候选偏好JSON');exportPreferences.id='export-preferences';document.querySelector('nav').after(exportPreferences);
exportPreferences.onclick=()=>{
 const id=document.getElementById('reviewer').value.trim();if(!id||id==='FILL_REVIEWER_ID'){alert('请填真实预登记匿名评审ID');return}
 for(const pair of candidateData.focus.pairs){const row=candidateData.preferences.reviews.find(r=>r.pair_id===pair.pair_id);row.reviewer_id=id;row.audio_seen=[pair.a_review_id,pair.b_review_id].every(rid=>template.reviews.find(r=>r.review_id===rid).audio_seen);
 if(row.choice!==null&&(!row.reason.trim()||!row.audio_seen||!row.reviewed_ranges.length)){alert('已填偏好需要双方听音声明、理由和实际已审范围；否则留空');return}}
 const blob=new Blob([JSON.stringify(candidateData.preferences,null,2)],{type:'application/json'}),link=E('a');link.href=URL.createObjectURL(blob);link.download='candidate-preferences.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
'''

_TARGET_SCRIPT = r'''
const targetData=JSON.parse(document.getElementById('target-data').textContent),targetFocus=targetData.focus;
document.title='原稿选点必要性独立审查';document.querySelector('h1').textContent=document.title;
document.querySelector('#rubric p').textContent='先听取目标及相邻上下文，独立判断原稿是否值得修改。未显示模型诊断/优先级/策略；不能预设被选中的事件一定有问题。填写必要性、具体依据和实际审过的半开采样区间；未知留空。此页没有候选，整体组织另评。';
const targetCards=new Map([...document.querySelectorAll('.card')].map(c=>[c.querySelector('h3').textContent,c]));
targetFocus.targets.forEach(f=>{const card=targetCards.get(f.review_id),item=pack.items.find(i=>i.review_id===f.review_id),audio=card.closest('[id^="case-"]').querySelector('audio');
 card.prepend(E('p','审查目标 '+f.target_event_id+'；共同上下文 samples=['+f.required_range.start+','+f.required_range.end+')，请独立填写修改必要性与已审范围。'));
 const seek=E('button','播放审查上下文');seek.onclick=()=>{audio.currentTime=f.required_range.start/item.plan.sample_rate;audio.play().catch(()=>{})};card.prepend(seek);
 card.querySelectorAll('tr[data-start]').forEach(row=>{if(row.cells[0].textContent===f.target_event_id){row.style.outline='2px solid #3366cc';row.dataset.target='true'}});
 [...card.querySelectorAll('label')].find(l=>l.textContent.startsWith(labels.long_range_structure)).querySelector('select').disabled=true;
 const row=targetData.judgments.reviews.find(r=>r.review_id===f.review_id),choice=E('select');choice.className='target-choice';[['','未评价'],['yes','值得修改'],['no','无需修改'],['unknown','无法判断']].forEach(([v,t])=>choice.append(new Option(t,v)));choice.onchange=()=>row.needs_revision=choice.value||null;
 const reason=E('textarea');reason.className='target-reason';reason.placeholder='具体动作/音乐依据，不引用未显示的模型诊断';reason.oninput=()=>row.reason=reason.value;
 const start=E('input'),end=E('input'),ranges=E('pre');start.className='target-reviewed-start';end.className='target-reviewed-end';start.placeholder='实际已审起点（采样点）';end.placeholder='实际已审终点（采样点）';const add=E('button','记录已审区间');add.onclick=()=>{
 if(!/^(0|[1-9][0-9]*)$/.test(start.value)||!/^(0|[1-9][0-9]*)$/.test(end.value)||!Number.isSafeInteger(Number(start.value))||!Number.isSafeInteger(Number(end.value))||Number(start.value)>=Number(end.value)||Number(end.value)>item.plan.num_samples){alert('填写音频内非空整数采样区间');return}
 row.reviewed_ranges.push({start:Number(start.value),end:Number(end.value)});ranges.textContent=JSON.stringify(row.reviewed_ranges);start.value='';end.value=''};
 const clear=E('button','清空区间草稿');clear.onclick=()=>{row.reviewed_ranges=[];ranges.textContent=''};card.append(E('h4','独立修改必要性（另存）'),choice,reason,start,end,add,clear,ranges);
});
const exportTargets=E('button','另存独立选点判断JSON');exportTargets.id='export-targets';document.querySelector('nav').after(exportTargets);exportTargets.onclick=()=>{
 const id=document.getElementById('reviewer').value.trim();if(!id||id==='FILL_REVIEWER_ID'){alert('请填真实预登记匿名评审ID');return}
 for(const row of targetData.judgments.reviews){row.reviewer_id=id;row.audio_seen=template.reviews.find(r=>r.review_id===row.review_id).audio_seen;if(row.needs_revision!==null&&(!row.reason.trim()||!row.audio_seen||!row.reviewed_ranges.length)){alert('判断需要听音声明、理由和实际已审区间；否则留空');return}}
 const blob=new Blob([JSON.stringify(targetData.judgments,null,2)],{type:'application/json'}),link=E('a');link.href=URL.createObjectURL(blob);link.download='target-judgments.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--inventory", required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.add_argument("--context-seconds", type=int, default=5)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--review-dir", required=True)
    analyze.add_argument("--reviews", nargs="*", default=[])
    analyze.add_argument("--out", required=True)
    targets = sub.add_parser("analyze-targets")
    targets.add_argument("--review-dir", required=True)
    targets.add_argument("--reviews", nargs="*", default=[])
    targets.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_candidate_review(args.inventory, args.out_dir, args.seed, args.context_seconds)
        print(json_text({k: result[k] for k in ("song_count", "selected_step_count", "candidate_pair_count", "unique_action_pair_count",
                                               "latest_guard_rejected_nonempty", "new_model_calls", "actual_human_reviews")}))
    elif args.command == "analyze-targets":
        result = analyze_target_judgments(args.review_dir, args.reviews)
        write_json(args.out, result)
        print(json_text({k: result[k] for k in ("target_count", "eligible_human_judgments", "target_status", "full_design_complete")}))
    else:
        result = analyze_candidate_preferences(args.review_dir, args.reviews)
        write_json(args.out, result)
        print(json_text({k: result[k] for k in ("pair_count", "eligible_human_judgments", "pair_status", "full_design_complete")}))


if __name__ == "__main__":
    main()
