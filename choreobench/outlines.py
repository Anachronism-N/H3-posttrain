"""Audited, fixed public outlines. No provider is instantiated or called.

Source hashes preserve traceability, not model authentication or quality. Only
whole successful source runs are exported; selecting individual good songs or
re-exporting imported outlines is deliberately unsupported.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .baseline import _Journal, _journal_accounting, _recorded_result, _verify
from .candidates import load_candidates, match_candidates
from .dataset import resolve_audio, verify_dataset_audio
from .io import file_sha, finite_json, object_sha, read_json_snapshot, write_json
from .public_structure import load_public_structure
from .schema import integer, keys, validate_dataset, validate_plan, validate_song
from .selection import candidate_options


SCHEMA = "choreobench.shared_outlines.v1"
ACCOUNTING = "historical_outline_generation_once_shared_not_new_calls"


def _sha(value, name, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(name + " must be a lowercase SHA256")


def _same(left, right):
    return object_sha(left) == object_sha(right)


def _seal(value):
    value["record_sha256"] = object_sha(value)
    return value


def _unit_context(units, song):
    finite_json(units)
    if not isinstance(units, dict) or units.get("schema") != "choreobench.planning_units.v1":
        raise ValueError("A complete planning_units document is required")
    integer(units.get("sample_rate"), "planning units sample_rate", 1)
    integer(units.get("num_samples"), "planning units num_samples", 1)
    if (units.get("song_id"), units.get("sample_rate"), units.get("num_samples"), units.get("audio_sha256"),
            units.get("song_sha256")) != (song["song_id"], song["audio"]["sample_rate"], song["audio"]["num_samples"],
                                         song["audio"]["sha256"], object_sha(song)):
        raise ValueError("Planning units do not bind the effective song/clock")
    if not isinstance(units.get("units"), list) or not units["units"]:
        raise ValueError("Planning units must be a complete nonempty partition")
    sections = {s["section_id"]: s for s in song["music"]["sections"]}
    cursor, ids = 0, set()
    for unit in units["units"]:
        if not isinstance(unit, dict) or not isinstance(unit.get("unit_id"), str) or unit["unit_id"] in ids:
            raise ValueError("Duplicate/invalid planning unit ID")
        ids.add(unit["unit_id"])
        integer(unit.get("start"), "unit.start")
        integer(unit.get("end"), "unit.end", 1)
        section = sections.get(unit.get("section_id"))
        if (section is None or unit["start"] != cursor or not unit["start"] < unit["end"] <= section["end"]
                or unit["start"] < section["start"]):
            raise ValueError("Planning units must partition the effective sections without crossing boundaries")
        cursor = unit["end"]
    if cursor != song["audio"]["num_samples"]:
        raise ValueError("Planning units omit the song tail")


def _cost(call_id, costs):
    if not isinstance(costs, dict):
        raise ValueError("Source outline lacks generation cost metadata")
    result = dict(call_id=call_id, raw_costs_sha256=object_sha(costs))
    for name in ("input_tokens", "output_tokens"):
        value = costs.get(name)
        if value is not None:
            integer(value, "source cost." + name)
        result[name] = value
    latency = costs.get("latency_seconds")
    if latency is not None and (type(latency) not in (int, float) or latency < 0
                                or type(latency) is float and not math.isfinite(latency)):
        raise ValueError("Source latency must be finite nonnegative or null")
    result["latency_seconds"] = latency
    return result


def _validate_document(document, dataset_sha, bank_sha, effective_songs, units_by_song, structure_sha):
    from .coordination import _read_outline  # Keep coordination -> outlines imports acyclic.
    finite_json(document)
    _verify(document)
    keys(document, ("schema", "dataset_sha256", "candidate_bank_sha256", "public_structure_sha256", "synthetic",
                    "songs", "source", "source_costs", "record_sha256"), where="shared outlines")
    if document["schema"] != SCHEMA:
        raise ValueError("Unsupported shared outline schema")
    for name, expected in (("dataset_sha256", dataset_sha), ("candidate_bank_sha256", bank_sha),
                           ("public_structure_sha256", structure_sha)):
        _sha(document[name], name, nullable=name == "public_structure_sha256")
        if document[name] != expected:
            raise ValueError("Shared outlines " + name + " differs from the current input")
    if type(document["synthetic"]) is not bool:
        raise ValueError("Shared outlines require explicit synthetic provenance")
    if not isinstance(effective_songs, list) or not effective_songs:
        raise ValueError("Nonempty effective song inventory required")
    for song in effective_songs:
        validate_song(song)
    ids = [s["song_id"] for s in effective_songs]
    if len(set(ids)) != len(ids) or not isinstance(units_by_song, dict) or set(units_by_song) != set(ids):
        raise ValueError("Effective songs and planning units must have exactly the same unique inventory")
    source = document["source"]
    keys(source, ("kind", "contract_file_sha256", "contract_sha256", "results_file_sha256", "results_record_sha256",
                  "planner_identity_sha256", "code_sha256", "synthetic", "identity_status"), where="outline source")
    if source["kind"] != "completed_coordinated_run" or source["identity_status"] != "source_records_not_independent_model_authentication":
        raise ValueError("Only audited original coordinated generation is a shared outline source")
    if type(source["synthetic"]) is not bool or source["synthetic"] != document["synthetic"]:
        raise ValueError("Shared outline source synthetic taint differs")
    for name in ("contract_file_sha256", "contract_sha256", "results_file_sha256", "results_record_sha256", "planner_identity_sha256"):
        _sha(source[name], "source." + name)
    if not isinstance(source["code_sha256"], dict) or not source["code_sha256"]:
        raise ValueError("Original frozen code identity must be retained")
    for name, digest in source["code_sha256"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", name):
            raise ValueError("Invalid original source filename")
        _sha(digest, "source code digest")
    rows = document["songs"]
    if not isinstance(rows, list) or [r.get("song_id") for r in rows if isinstance(r, dict)] != ids or len(rows) != len(ids):
        raise ValueError("Shared outlines must contain every selected song exactly once, in source order")
    calls = []
    for row, song in zip(rows, effective_songs):
        keys(row, ("song_id", "planning_song_sha256", "planning_units_sha256", "outline", "outline_sha256", "source_call"),
             where="shared outline song")
        units = units_by_song[song["song_id"]]
        _unit_context(units, song)
        if row["planning_song_sha256"] != object_sha(song) or row["planning_units_sha256"] != object_sha(units):
            raise ValueError("Shared outline effective song or units changed")
        outline = _read_outline(json.dumps(row["outline"], ensure_ascii=False, allow_nan=False), song)
        if row["outline_sha256"] != object_sha(outline):
            raise ValueError("Shared outline content hash differs")
        call = row["source_call"]
        keys(call, ("call_id", "request_sha256", "started_record_sha256", "raw_record_sha256", "provider_identity_sha256"),
             where="outline source call")
        if call["call_id"] != song["song_id"] + "/coordinate/outline":
            raise ValueError("Shared outline call belongs to another song or stage")
        for name in ("request_sha256", "started_record_sha256", "raw_record_sha256", "provider_identity_sha256"):
            _sha(call[name], "source_call." + name)
        if call["provider_identity_sha256"] != source["planner_identity_sha256"]:
            raise ValueError("Source call planner identity differs")
        calls.append(call["call_id"])
    costs = document["source_costs"]
    keys(costs, ("recorded_calls", "costs", "unknown_usage_call_ids", "accounting"), where="outline source costs")
    integer(costs["recorded_calls"], "source recorded_calls", 1)
    if costs["recorded_calls"] != len(calls) or costs["accounting"] != ACCOUNTING:
        raise ValueError("Shared outline costs must reference each original generation exactly once")
    if not isinstance(costs["costs"], list) or len(costs["costs"]) != len(calls):
        raise ValueError("Source outline cost inventory differs")
    unknown = []
    for item, call_id in zip(costs["costs"], calls):
        keys(item, ("call_id", "input_tokens", "output_tokens", "latency_seconds", "raw_costs_sha256"), where="source cost")
        if item["call_id"] != call_id:
            raise ValueError("Source cost call order/identity differs")
        for name in ("input_tokens", "output_tokens"):
            if item[name] is not None:
                integer(item[name], "source cost." + name)
        latency = item["latency_seconds"]
        if latency is not None and (type(latency) not in (int, float) or latency < 0
                                    or type(latency) is float and not math.isfinite(latency)):
            raise ValueError("Invalid source latency")
        _sha(item["raw_costs_sha256"], "raw source costs")
        if item["input_tokens"] is None or item["output_tokens"] is None:
            unknown.append(call_id)
    if not _same(costs["unknown_usage_call_ids"], unknown):
        raise ValueError("Unknown outline usage may not be silently zeroed")
    return document


def load_shared_outlines(path, dataset_sha, bank_sha, effective_songs, units_by_song, structure_sha):
    """Validate an exact snapshot; caller binds its bytes SHA in the new run.

    The original source code hashes are preserved, not compared to the current
    code. The runner must continue checking the sidecar bytes before dispatch
    and resume. This loader neither resolves source paths nor calls a provider.
    """
    document, digest = read_json_snapshot(path)
    _validate_document(document, dataset_sha, bank_sha, effective_songs, units_by_song, structure_sha)
    return document, digest


def export_shared_outlines(run_dir, out_path):
    """Audit one complete successful original run, then publish once, exclusively."""
    from .coordination import _outline_request, _read_outline
    root, out_path = Path(run_dir).resolve(), Path(out_path)
    if out_path.exists():
        raise FileExistsError(str(out_path))
    snapshots = []
    def snapshot(path):
        value, digest = read_json_snapshot(path)
        snapshots.append((Path(path), digest))
        return value, digest
    def safe(relative):
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or relative.startswith("/") or any(p in ("", ".", "..") for p in relative.split("/"))):
            raise ValueError("Source artifact must be a canonical relative path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source artifact escapes its run directory")
        return path
    contract, contract_file_sha = snapshot(root / "contract.json")
    result, result_file_sha = snapshot(root / "results.json")
    _verify(result)
    if contract.get("schema") != "choreobench.coordination_contract.v1" or result.get("schema") != "choreobench.coordination_result.v1":
        raise ValueError("Shared outlines require an original coordinated run")
    cfg = contract["config"]
    if (not isinstance(cfg, dict) or cfg.get("mode") != "coordinated" or cfg.get("use_outline") is not True
            or cfg.get("shared_outline_path") is not None or contract.get("shared_outline_snapshot") is not None):
        raise ValueError("Cannot export disabled, placeholder or previously imported shared outlines")
    if (result.get("complete") is not True or result.get("all_songs_succeeded") is not True
            or result.get("failed_songs") != [] or result.get("error") is not None
            or result.get("contract_sha256") != object_sha(contract)):
        raise ValueError("Export requires the entire selected source run to have completed successfully")
    dataset = validate_dataset(contract["dataset_snapshot"])
    selected = contract["selected_songs"]
    originals = [s for s in dataset["songs"] if s["split"] in cfg["splits"]]
    if not originals or selected != [s["song_id"] for s in originals]:
        raise ValueError("Source selected-song inventory differs from its dataset/config")
    for name in ("selected_count", "success_count"):
        integer(result.get(name), name, 1)
        if result[name] != len(selected):
            raise ValueError("Source result drops a selected song")
    bank_info = contract["local_candidates_identity"]
    bank = load_candidates(bank_info["path"])
    if bank.bytes_sha != bank_info["file_sha256"]:
        raise ValueError("Original candidate bank bytes changed")
    dataset_path = Path(bank.data["source"]["contract"]["dataset_path"])
    actual_dataset, actual_dataset_sha = snapshot(dataset_path)
    if actual_dataset_sha != contract["dataset_sha256"] or not _same(actual_dataset, dataset):
        raise ValueError("Original dataset bytes/snapshot changed")
    verify_dataset_audio(dataset, dataset_path)
    snapshots.extend((resolve_audio(song, dataset_path), song["audio"]["sha256"]) for song in dataset["songs"])
    matched = match_candidates(bank, actual_dataset_sha, originals, {s["song_id"]: s["audio"] for s in originals},
                               bank.data["source"]["contract"]["config"])
    if not _same(matched, bank_info):
        raise ValueError("Source candidate identity differs from the actual bank")
    structure_path = cfg["public_structure_path"]
    if structure_path is not None:
        structure_path = Path(structure_path).expanduser()
        if not structure_path.is_absolute():
            structure_path = dataset_path.parent / structure_path
    effective, structure, structure_sha = load_public_structure(structure_path, actual_dataset_sha, originals)
    if structure_sha != contract["public_structure_sha256"] or not _same(structure, contract["public_structure_snapshot"]):
        raise ValueError("Original public structure changed")
    if structure_path is not None:
        snapshots.append((structure_path, structure_sha))
    if not _same(contract["planning_songs_sha256"], {s["song_id"]: object_sha(s) for s in effective}):
        raise ValueError("Source effective song snapshots differ")
    identity = contract["planner_identity"]
    if not isinstance(identity, dict) or type(identity.get("synthetic")) is not bool or type(contract.get("synthetic")) is not bool:
        raise ValueError("Source planner/run must declare synthetic provenance")
    tainted = bool(dataset["synthetic"] or bank.data["synthetic"] or identity["synthetic"]
                   or structure and any(s["synthetic"] for s in structure["songs"]))
    if (tainted and not contract["synthetic"]) or type(result.get("synthetic")) is not bool or result["synthetic"] != contract["synthetic"]:
        raise ValueError("Source synthetic provenance was erased")
    references = {}
    for ref in result["artifacts"] + result["plans"]:
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or ref["path"] in references:
            raise ValueError("Duplicate/invalid published source artifact")
        _sha(ref.get("sha256"), "artifact.sha256")
        path = safe(ref["path"])
        if file_sha(path) != ref["sha256"]:
            raise ValueError("Published source artifact changed")
        references[ref["path"]] = ref
        snapshots.append((path, ref["sha256"]))
    def artifact(relative):
        if relative not in references:
            raise ValueError("Missing published source artifact: " + relative)
        value, digest = snapshot(safe(relative))
        if digest != references[relative]["sha256"]:
            raise ValueError("Source artifact changed while reading")
        return value
    if [p.get("song_id") for p in result["plans"]] != selected:
        raise ValueError("Source plans omit or reorder selected songs")
    if not safe("calls").is_dir():
        raise ValueError("Source run has no complete call journal")
    journal = _Journal(root, object_sha(contract), cfg["max_calls"])
    if set(journal.started) != set(journal.records) or result["call_records"] != {k: r["record_sha256"] for k, r in journal.records.items()}:
        raise ValueError("Source journal has missing, changed or unknown calls")
    for call_id, started in journal.started.items():
        if not _same(started["provider_identity"], identity):
            raise ValueError("Source journal planner differs from its contract")
        for suffix, expected in (("started", started), ("raw", journal.records[call_id])):
            actual, _ = snapshot(safe("calls/" + _Journal._name(call_id) + "." + suffix + ".json"))
            if not _same(actual, expected):
                raise ValueError("Source journal changed during export")
        _recorded_result(journal.records[call_id], identity, started["request"])
    if type(result.get("calls_used")) is not int or result["calls_used"] != len(journal.started):
        raise ValueError("Source call count differs")
    accounting = _journal_accounting(journal)
    def accounting_value(name, value):
        if name == "costs" and isinstance(value, list):
            return sorted(value, key=lambda item: item["call_id"])
        if name.endswith("_call_ids") and isinstance(value, list):
            return sorted(value)
        return value
    if any(not _same(accounting_value(name, result.get(name)), accounting_value(name, value))
           for name, value in accounting.items()):
        raise ValueError("Source costs/unknown-outcome accounting differs from its journal")
    rows, costs, units_by_song, expected_calls = [], [], {}, set()
    for original, song, plan_ref in zip(originals, effective, result["plans"]):
        sid = song["song_id"]
        prefix = hashlib.sha256(sid.encode()).hexdigest()
        units = artifact("units/" + prefix + ".json")
        _unit_context(units, song)
        units_by_song[sid] = units
        options = artifact("options/" + prefix + ".json")
        if not _same(options, candidate_options(bank, original, units)):
            raise ValueError("Published candidate options differ from the original source bank")
        metadata = artifact("outline_metadata/" + prefix + ".json")
        if (metadata.get("generated_by_model") is not True or metadata.get("source") != "model_response"
                or metadata.get("generated_in_this_run", True) is not True):
            raise ValueError("Cannot recursively export placeholder or imported outlines")
        outline = artifact("outlines/" + prefix + ".json")
        call_id = sid + "/coordinate/outline"
        expected_calls.add(call_id)
        expected_calls.update(sid + "/coordinate/" + u["unit_id"] for u in units["units"])
        started, raw = journal.started.get(call_id), journal.records.get(call_id)
        if started is None or raw is None or not _same(started["request"], _outline_request(song, options, units, cfg)):
            raise ValueError("Source outline request is missing or differs from the public input protocol")
        response = _recorded_result(raw, identity, started["request"])
        parsed = _read_outline(response["text"], song)
        if not _same(outline, parsed):
            raise ValueError("Published outline differs from its original model response")
        plan = artifact(plan_ref["path"])
        validate_plan(plan, original)
        if plan["provenance"].get("outline_sha256") != object_sha(outline) or plan["provenance"].get("outline_generated_by_model") is not True:
            raise ValueError("Source plan does not bind an original generated outline")
        rows.append(dict(song_id=sid, planning_song_sha256=object_sha(song), planning_units_sha256=object_sha(units),
            outline=copy.deepcopy(outline), outline_sha256=object_sha(outline), source_call=dict(call_id=call_id,
                request_sha256=started["request_sha256"], started_record_sha256=started["record_sha256"],
                raw_record_sha256=raw["record_sha256"], provider_identity_sha256=object_sha(identity))))
        costs.append(_cost(call_id, response["costs"]))
    if set(journal.records) != expected_calls or len(expected_calls) != len(rows) + sum(len(u["units"]) for u in units_by_song.values()):
        raise ValueError("Source journal includes extra or missing generation attempts")
    document = _seal(dict(schema=SCHEMA, dataset_sha256=actual_dataset_sha, candidate_bank_sha256=bank.bytes_sha,
        public_structure_sha256=structure_sha, synthetic=contract["synthetic"], songs=rows,
        source=dict(kind="completed_coordinated_run", contract_file_sha256=contract_file_sha,
            contract_sha256=object_sha(contract), results_file_sha256=result_file_sha,
            results_record_sha256=result["record_sha256"], planner_identity_sha256=object_sha(identity),
            code_sha256=copy.deepcopy(contract["code_sha256"]), synthetic=contract["synthetic"],
            identity_status="source_records_not_independent_model_authentication"),
        source_costs=dict(recorded_calls=len(costs), costs=costs,
            unknown_usage_call_ids=[c["call_id"] for c in costs if c["input_tokens"] is None or c["output_tokens"] is None],
            accounting=ACCOUNTING)))
    _validate_document(document, actual_dataset_sha, bank.bytes_sha, effective, units_by_song, structure_sha)
    bank.assert_unchanged(full=True)
    for path, digest in snapshots:
        if file_sha(path) != digest:
            raise ValueError("Source changed during shared outline export")
    write_json(out_path, document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    document = export_shared_outlines(args.run_dir, args.out)
    print(json.dumps(dict(schema=SCHEMA, songs=len(document["songs"]), synthetic=document["synthetic"],
                          new_model_calls=0, source_outline_calls=document["source_costs"]["recorded_calls"])))


if __name__ == "__main__":
    main()
