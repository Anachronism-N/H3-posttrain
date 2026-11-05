"""Synthetic source-record audits only; never real model or music-quality claims."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from choreobench import baseline, candidates, coordination, harness, outlines
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.outlines import ACCOUNTING, export_shared_outlines, load_shared_outlines
from choreobench.public_structure import load_public_structure
from test_choreobench_coordination import config, fake_planner, sidecar
from test_choreobench_harness import fixture
from test_choreobench_baseline import dataset_file


def seal(value):
    value = copy.deepcopy(value)
    value.pop("record_sha256", None)
    value["record_sha256"] = object_sha(value)
    return value


def source_run(tmp_path, monkeypatch, *, structure=False, unknown=False):
    dataset, original, bank = fixture(tmp_path)
    requests = []
    fake_planner(monkeypatch, requests)
    if unknown:
        factory = coordination.create_provider
        def provider(*args, **kwargs):
            value = factory(*args, **kwargs)
            generate = value.generate
            def unknown_usage(request):
                response = generate(request)
                response["costs"]["input_tokens"] = None
                return response
            value.generate = unknown_usage
            return value
        monkeypatch.setattr(coordination, "create_provider", provider)
    changes, structure_path = {}, None
    if structure:
        structure_path, _ = sidecar(tmp_path, dataset, original)
        changes["public_structure_path"] = str(structure_path)
    output = tmp_path / "source-run"
    result = harness.run_harness(dataset, bank, output, config(**changes))
    units = {original["song_id"]: next(read_json(output / r["path"]) for r in result["artifacts"]
                                      if r["path"].startswith("units/"))}
    effective, _, structure_sha = load_public_structure(structure_path, file_sha(dataset), [original])
    return dict(run=output, dataset=dataset, original=original, bank=bank, requests=requests,
                result=result, effective=effective, units=units, structure_sha=structure_sha)


def load(path, source):
    return load_shared_outlines(path, file_sha(source["dataset"]), file_sha(source["bank"]),
                                source["effective"], source["units"], source["structure_sha"])


def test_export_matches_runner_home_relative_structure_resolution(tmp_path, monkeypatch):
    dataset, original, bank = fixture(tmp_path)
    structure, _ = sidecar(tmp_path, dataset, original)
    expand = Path.expanduser
    def expand_fixture(path):
        return structure if path.parts == ("~", "structure.json") else expand(path)
    monkeypatch.setattr(Path, "expanduser", expand_fixture)
    requests = []
    fake_planner(monkeypatch, requests)
    run = tmp_path / "tilde-run"
    result = harness.run_harness(dataset, bank, run, config(public_structure_path="~/structure.json"))
    assert result["all_songs_succeeded"]
    exported = export_shared_outlines(run, tmp_path / "shared.json")
    assert exported["public_structure_sha256"] == file_sha(structure)
    assert len(requests) == result["calls_used"]  # Export must not generate anything.


def published(source, prefix):
    return next(source["run"] / r["path"] for r in source["result"]["artifacts"] if r["path"].startswith(prefix))


def rebind_source(source, *, contract_change=None, start_change=None, raw_change=None, artifact_change=None):
    """Keep fixture hashes internally consistent so semantic binding gets tested."""
    root = source["run"]
    contract = read_json(root / "contract.json")
    result = read_json(root / "results.json")
    if contract_change:
        contract_change(contract)
    contract_sha = object_sha(contract)
    write_json(root / "contract.json", contract, overwrite=True)
    records = {}
    for path in (root / "calls").glob("*.started.json"):
        start = read_json(path)
        if start_change:
            start_change(start)
        start["contract_sha256"] = contract_sha
        start["request_sha256"] = object_sha(start["request"])
        start = seal(start)
        write_json(path, start, overwrite=True)
        raw_path = path.with_name(path.name.replace(".started.json", ".raw.json"))
        raw = read_json(raw_path)
        if raw_change:
            raw_change(raw)
        raw.update(contract_sha256=contract_sha, started_sha256=start["record_sha256"])
        raw = seal(raw)
        write_json(raw_path, raw, overwrite=True)
        records[raw["call_id"]] = raw["record_sha256"]
    for ref in result["plans"]:
        path = root / ref["path"]
        plan = read_json(path)
        plan["provenance"]["contract_sha256"] = contract_sha
        write_json(path, plan, overwrite=True)
        ref["sha256"] = file_sha(path)
    if artifact_change:
        for ref in result["artifacts"]:
            path = root / ref["path"]
            value = read_json(path)
            artifact_change(ref["path"], value)
            write_json(path, value, overwrite=True)
            ref["sha256"] = file_sha(path)
    result.update(contract_sha256=contract_sha, call_records=records)
    write_json(root / "results.json", seal(result), overwrite=True)


def test_roundtrip_public_snapshot_and_costs_without_any_provider(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch)
    before = len(source["requests"])
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: pytest.fail("Export/load must not instantiate a provider"))
    path = tmp_path / "shared.json"
    document = export_shared_outlines(source["run"], path)
    loaded, digest = load(path, source)
    assert loaded == document and digest == file_sha(path)
    assert len(source["requests"]) == before
    assert document["synthetic"] is True
    assert len(document["songs"]) == 1
    row = document["songs"][0]
    assert row["outline_sha256"] == object_sha(read_json(published(source, "outlines/")))
    assert row["source_call"]["call_id"] == "song1/coordinate/outline"
    assert document["source_costs"]["recorded_calls"] == 1  # Unit calls are not charged again.
    assert document["source_costs"]["accounting"] == ACCOUNTING
    assert document["source_costs"]["costs"][0]["input_tokens"] == 100
    assert document["source_costs"]["costs"][0]["output_tokens"] == 50
    raw = path.read_text(encoding="utf-8")
    for private in ("private-song-group", "private-artist-group", "song.wav", str(tmp_path), '"semantic_goals"'):
        assert private not in raw
    with pytest.raises(FileExistsError):
        export_shared_outlines(source["run"], path)


def test_public_structure_and_unknown_cost_taint_survive(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch, structure=True, unknown=True)
    path = tmp_path / "shared.json"
    document = export_shared_outlines(source["run"], path)
    assert load(path, source)[0] == document
    assert document["public_structure_sha256"] == source["structure_sha"]
    assert document["source_costs"]["unknown_usage_call_ids"] == ["song1/coordinate/outline"]
    assert document["source_costs"]["costs"][0]["input_tokens"] is None


def test_different_historical_source_code_is_recorded_not_rejected(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch)
    frozen = {"historical_coordinator.py": "9" * 64}
    rebind_source(source, contract_change=lambda c: c.update(code_sha256=frozen))
    path = tmp_path / "shared.json"
    document = export_shared_outlines(source["run"], path)
    assert document["source"]["code_sha256"] == frozen
    assert load(path, source)[0] == document


@pytest.mark.parametrize("kind", ["dataset", "audio", "bank", "structure", "outline", "journal"])
def test_changed_actual_source_is_rejected(tmp_path, monkeypatch, kind):
    source = source_run(tmp_path, monkeypatch, structure=kind == "structure")
    if kind == "dataset": path = source["dataset"]
    elif kind == "audio": path = tmp_path / "song.wav"
    elif kind == "bank": path = source["bank"]
    elif kind == "structure": path = tmp_path / "structure.json"
    elif kind == "outline": path = published(source, "outlines/")
    else: path = next((source["run"] / "calls").glob("*.raw.json"))
    if kind == "journal":
        value = read_json(path)
        value["result"]["text"] += " changed content"
        write_json(path, value, overwrite=True)
    elif kind == "audio":
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        export_shared_outlines(source["run"], tmp_path / "shared.json")
    assert not (tmp_path / "shared.json").exists()


@pytest.mark.parametrize("kind", ["imported", "placeholder", "missing_call", "incomplete", "dropped_song", "taint"])
def test_export_cannot_pick_good_prefixes_or_recursively_import_source(tmp_path, monkeypatch, kind):
    source = source_run(tmp_path, monkeypatch)
    if kind == "imported":
        rebind_source(source, contract_change=lambda c: c["config"].update(shared_outline_path="some-import.json"))
    elif kind == "placeholder":
        def change(path, value):
            if path.startswith("outline_metadata/"):
                value.update(source="deterministic_empty_placeholder", generated_by_model=False)
        rebind_source(source, artifact_change=change)
    elif kind == "missing_call":
        name = hashlib.sha256(b"song1/coordinate/outline").hexdigest()
        (source["run"] / "calls" / (name + ".raw.json")).unlink()
    else:
        path = source["run"] / "results.json"
        result = read_json(path)
        if kind == "incomplete": result["complete"] = False
        elif kind == "dropped_song": result["plans"] = []
        else: result["synthetic"] = False
        write_json(path, seal(result), overwrite=True)
    with pytest.raises(ValueError):
        export_shared_outlines(source["run"], tmp_path / "shared.json")


def test_consistently_rehashed_private_outline_request_is_still_rejected(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch)
    def change(start):
        if start["request"]["response_kind"] == "coordination_outline":
            payload = json.loads(start["request"]["messages"][-1]["content"])
            payload["semantic_goals"] = ["PRIVATE_GOAL"]
            start["request"]["messages"][-1]["content"] = json.dumps(payload)
    rebind_source(source, start_change=change)
    with pytest.raises(ValueError, match="public input protocol"):
        export_shared_outlines(source["run"], tmp_path / "shared.json")


@pytest.mark.parametrize("change", ["private", "outline_private", "cost_private", "missing_song", "duplicate_song",
    "outline_hash", "call_identity", "provider", "cost_count", "cost_type", "source_taint", "source_code"])
def test_loader_strict_schema_and_internal_source_invariants(tmp_path, monkeypatch, change):
    source = source_run(tmp_path, monkeypatch)
    path = tmp_path / "shared.json"
    value = export_shared_outlines(source["run"], path)
    if change == "private": value["semantic_goals"] = ["hidden"]
    elif change == "outline_private": value["songs"][0]["outline"]["goals"] = ["hidden"]
    elif change == "cost_private": value["source_costs"]["costs"][0]["goals"] = ["hidden"]
    elif change == "missing_song": value["songs"] = []
    elif change == "duplicate_song": value["songs"] *= 2
    elif change == "outline_hash": value["songs"][0]["outline_sha256"] = "a" * 64
    elif change == "call_identity": value["songs"][0]["source_call"]["call_id"] = "other/coordinate/outline"
    elif change == "provider": value["songs"][0]["source_call"]["provider_identity_sha256"] = "a" * 64
    elif change == "cost_count": value["source_costs"]["recorded_calls"] = True
    elif change == "cost_type": value["source_costs"]["costs"][0]["input_tokens"] = False
    elif change == "source_taint": value["source"]["synthetic"] = False
    else: value["source"]["code_sha256"] = {"../secret.py": "a" * 64}
    write_json(path, seal(value), overwrite=True)
    with pytest.raises(ValueError):
        load(path, source)


@pytest.mark.parametrize("binding", ["dataset", "bank", "structure", "song", "units", "inventory"])
def test_loader_rejects_another_current_context(tmp_path, monkeypatch, binding):
    source = source_run(tmp_path, monkeypatch)
    path = tmp_path / "shared.json"
    export_shared_outlines(source["run"], path)
    dataset_sha, bank_sha, structure_sha = file_sha(source["dataset"]), file_sha(source["bank"]), source["structure_sha"]
    songs, units = copy.deepcopy(source["effective"]), copy.deepcopy(source["units"])
    if binding == "dataset": dataset_sha = "a" * 64
    elif binding == "bank": bank_sha = "a" * 64
    elif binding == "structure": structure_sha = "a" * 64
    elif binding == "song": songs[0]["brief"] += " changed"
    elif binding == "units": units[songs[0]["song_id"]]["provenance"]["algorithm"] += " changed"
    else: songs = []
    with pytest.raises(ValueError):
        load_shared_outlines(path, dataset_sha, bank_sha, songs, units, structure_sha)


def test_export_cli_is_cpu_only_and_keeps_cost_reference(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch)
    output = tmp_path / "shared.json"
    command = subprocess.run([sys.executable, "-m", "choreobench.outlines", "--run-dir", str(source["run"]),
                              "--out", str(output)], capture_output=True, text=True, check=True)
    report = json.loads(command.stdout)
    assert report["new_model_calls"] == 0 and report["source_outline_calls"] == 1
    assert load(output, source)[0]["synthetic"] is True


def test_source_change_during_export_never_publishes_a_partial_artifact(tmp_path, monkeypatch):
    source = source_run(tmp_path, monkeypatch)
    actual_sha = outlines.file_sha
    def altered(path):
        if Path(path) == source["run"] / "contract.json":
            return "a" * 64
        return actual_sha(path)
    monkeypatch.setattr(outlines, "file_sha", altered)
    with pytest.raises(ValueError, match="Source changed during"):
        export_shared_outlines(source["run"], tmp_path / "shared.json")
    assert not (tmp_path / "shared.json").exists()


def test_full_multi_song_inventory_cannot_be_selected_after_source_completion(tmp_path, monkeypatch):
    dataset_path, song = dataset_file(tmp_path)
    dataset = read_json(dataset_path)
    second = copy.deepcopy(song)
    second["song_id"] = "song2"
    dataset["songs"].append(second)  # Synthetic duplicate media only, not independent benchmark samples.
    write_json(dataset_path, dataset, overwrite=True)
    baseline.run_baseline(dataset_path, tmp_path / "collection", {})
    bank_path = tmp_path / "bank.json"
    candidates.export_candidates(tmp_path / "collection", bank_path)
    fake_planner(monkeypatch, [])
    run = tmp_path / "source-run"
    result = harness.run_harness(dataset_path, bank_path, run, config())
    document = export_shared_outlines(run, tmp_path / "shared.json")
    assert [r["song_id"] for r in document["songs"]] == ["song1", "song2"]
    assert document["source_costs"]["recorded_calls"] == 2
    units = {read_json(run / ref["path"])["song_id"]: read_json(run / ref["path"])
             for ref in result["artifacts"] if ref["path"].startswith("units/")}
    with pytest.raises(ValueError, match="every selected song"):
        load_shared_outlines(tmp_path / "shared.json", file_sha(dataset_path), file_sha(bank_path),
                             [song], {"song1": units["song1"]}, None)
