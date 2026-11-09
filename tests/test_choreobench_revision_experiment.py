"""Preparation freezes comparisons, never loading providers or running a model."""
import copy
from pathlib import Path
import sys
import wave

import pytest

from choreobench import revision_experiment as experiment
from choreobench.io import file_sha, object_sha, read_json, write_json


def materials(tmp_path, **options):
    audio = tmp_path / "music fixture.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(100)
        handle.writeframes(b"\0\0" * 600)
    song = {"song_id": "revision-song", "split": "dev", "song_group_id": None,
            "artist_group_id": None, "near_duplicate_group_id": None, "brief": "A synthetic preparation fixture.",
            "audio": {"path": audio.name, "sha256": file_sha(audio), "sample_rate": 100, "num_samples": 600},
            "music": {"source": "synthetic", "beats": [], "downbeats": [], "sections": [
                {"section_id": "s", "start": 0, "end": 600, "label": "unknown", "repeat_group": None}]}}
    dataset = {"schema": "choreobench.dataset.v1", "dataset_id": "revision-fixture", "synthetic": True,
               "grouping": {"status": "unverified", "evidence": ""}, "songs": [song]}
    plan = {"schema": "choreobench.plan.v1", "song_id": song["song_id"], "sample_rate": 100,
            "num_samples": 600, "motifs": [], "provenance": {"fixture": True}, "events": [
                {"event_id": "e", "start": 0, "end": 600, "track": "body", "action": "Sway.",
                 "attributes": {}, "entry": {}, "exit": {}, "motif_id": None, "evidence": []}]}
    config = {"planner_provider": {"kind": "chat_completions", "base_url": "http://not-contacted.invalid"},
              "observer_provider": {"kind": "native_service", "base_url": "http://also-not-contacted.invalid"},
              "listen": True, "max_repairs": 2, "max_calls": 6, "seed": 7}
    config.update(options)
    paths = (tmp_path / "dataset.json", tmp_path / "original plan.json", tmp_path / "config.json")
    for path, value in zip(paths, (dataset, plan, config)):
        write_json(path, value)
    return paths


def save(path, value):
    write_json(path, value, overwrite=True)


def argument(argv, name):
    return argv[argv.index(name) + 1]


def test_preparation_freezes_exact_sources_four_configs_and_execution_graph(tmp_path):
    paths = materials(tmp_path, selection_policy="random", max_calls=8)
    hashes = {str(path.resolve()): file_sha(path) for path in (*paths, tmp_path / "music fixture.wav")}
    output = tmp_path / "new experiment"
    report = experiment.prepare_revision_experiment(*paths, output)
    assert report == read_json(output / "manifest.json")
    assert report["record_sha256"] == object_sha({k: v for k, v in report.items() if k != "record_sha256"})
    assert {ref["path"]: ref["sha256"] for ref in report["sources"]} == hashes
    assert {ref["role"] for ref in report["sources"]} == {"dataset", "plan", "base_config", "audio"}
    assert all(file_sha(path) == digest for path, digest in hashes.items())
    assert report["prepared"] is True and report["executed"] is False
    assert report["quality_verified"] is False and report["deployment_verified"] is False
    assert report["model_or_service_verified"] is False and report["training_performed"] is False
    assert report["model_weights_updated"] is False
    assert report["actual_cost"] is None and report["actual_calls"] is None
    configs = {name: read_json(reference["path"]) for name, reference in report["configs"].items()}
    assert all(file_sha(reference["path"]) == reference["sha256"] for reference in report["configs"].values())
    assert configs["diagnosis"]["listen"] is False and configs["diagnosis"]["observer_provider"] is None
    assert configs["diagnosis"]["max_calls"] == 1
    assert configs["targeted"]["listen"] is True and configs["targeted"]["selection_policy"] == "targeted"
    targeted = copy.deepcopy(configs["targeted"]); randomized = copy.deepcopy(configs["random"])
    assert targeted.pop("selection_policy") == "targeted" and randomized.pop("selection_policy") == "random"
    assert targeted == randomized and configs["targeted"]["max_calls"] == 8
    assert configs["text_only"]["selection_policy"] == "targeted"
    assert configs["text_only"]["listen"] is False and configs["text_only"]["observer_provider"] is None
    assert configs["text_only"]["max_calls"] == 4
    for key in ("seed", "max_repairs", "max_output_tokens", "max_request_characters", "planner_provider"):
        assert len({object_sha(config[key]) for config in configs.values()}) == 1
    commands = read_json(output / "commands.json")
    assert file_sha(report["commands"]["path"]) == report["commands"]["sha256"]
    assert commands["execution_order"] == ["diagnosis", "targeted", "random", "text_only"]
    assert [row["stage"] for row in commands["steps"]] == commands["execution_order"]
    for index, step in enumerate(commands["steps"]):
        argv = step["argv"]
        assert argv[:3] == [sys.executable, "-m", "choreobench"]
        assert argv[3] == ("diagnose-plan" if index == 0 else "revise-plan")
        assert argument(argv, "--dataset") == str(paths[0].resolve())
        assert argument(argv, "--plan") == str(paths[1].resolve())
        if index:
            assert argument(argv, "--diagnosis") == str(output / "diagnosis" / "diagnosis.json")
        assert not Path(argument(argv, "--out-dir")).exists()
    assert not (output / "diagnosis").exists() and not (output / "runs").exists()
    assert sorted(str(path.relative_to(output)).replace("\\", "/") for path in output.rglob("*") if path.is_file()) == [
        "commands.json", "configs/diagnosis.json", "configs/random.json", "configs/targeted.json",
        "configs/text_only.json", "manifest.json"]
    assert report["original_baseline"]["path"] == str(paths[1].resolve())
    assert report["original_baseline"]["plan_sha256"] == object_sha(read_json(paths[1]))
    assert report["original_baseline"]["new_model_calls"] == 0
    assert report["call_upper_bounds"] == {"diagnosis_once": 1, "targeted": 6, "random": 6, "text_only": 4}
    assert any("not an equal-cost control" in note for note in report["limitations"])


def test_prepare_never_constructs_provider_observer_or_runs_revision(tmp_path, monkeypatch):
    from choreobench import revision, providers, music_recheck
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must never contact or instantiate a provider or execute a stage")
    for module, name in ((revision, "create_provider"), (revision, "create_music_observer"),
                         (providers, "create_provider"), (music_recheck, "create_provider"),
                         (revision, "diagnose_plan"), (revision, "run_revision")):
        monkeypatch.setattr(module, name, forbidden)
    paths = materials(tmp_path)
    report = experiment.prepare_revision_experiment(*paths, tmp_path / "prepared")
    assert report["executed"] is False


@pytest.mark.parametrize("options", [
    {"listen": False, "observer_provider": None}, {"observer_provider": None},
    {"observer_provider": {}}, {"observer_provider": {"kind": "mock"}},
    {"planner_provider": {}}, {"planner_provider": {"kind": "native"}},
    {"max_calls": 5}, {"max_calls": True}, {"max_repairs": 0}, {"max_repairs": 9},
    {"max_repairs": True}, {"seed": -1}, {"new_private_goal": "not accepted"},
])
def test_invalid_config_or_insufficient_complete_budget_refused_before_output(tmp_path, options):
    paths = materials(tmp_path, **options); output = tmp_path / "refused"
    with pytest.raises(ValueError):
        experiment.prepare_revision_experiment(*paths, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["song", "clock", "audio_hash", "audio_geometry", "missing_audio"])
def test_wrong_input_identity_refused_before_output(tmp_path, change):
    paths = materials(tmp_path); dataset, plan = read_json(paths[0]), read_json(paths[1])
    if change == "song": plan["song_id"] = "absent"
    elif change == "clock": plan["sample_rate"] = 101
    elif change == "audio_hash": dataset["songs"][0]["audio"]["sha256"] = "0" * 64
    elif change == "audio_geometry": dataset["songs"][0]["audio"]["sample_rate"] = 101; plan["sample_rate"] = 101
    else: (tmp_path / "music fixture.wav").unlink()
    save(paths[0], dataset); save(paths[1], plan)
    output = tmp_path / "refused"
    with pytest.raises((ValueError, FileNotFoundError)):
        experiment.prepare_revision_experiment(*paths, output)
    assert not output.exists()


@pytest.mark.parametrize("directory", [True, False])
def test_existing_output_is_exclusive_and_untouched(tmp_path, directory):
    paths = materials(tmp_path); output = tmp_path / "existing"
    if directory: output.mkdir(); marker = output / "marker"
    else: marker = output
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        experiment.prepare_revision_experiment(*paths, output)
    assert marker.read_bytes() == b"original"


def test_source_change_during_preflight_refused_before_output(tmp_path, monkeypatch):
    paths = materials(tmp_path); original = experiment._inputs
    def changed(*args):
        values = original(*args)
        paths[1].write_bytes(paths[1].read_bytes() + b" ")
        return values
    monkeypatch.setattr(experiment, "_inputs", changed)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="source bytes changed"):
        experiment.prepare_revision_experiment(*paths, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["source", "product"])
def test_no_commit_manifest_when_sources_or_generated_products_change(tmp_path, monkeypatch, change):
    paths = materials(tmp_path); output = tmp_path / "partial"; original = experiment.write_json
    def modified(path, value):
        original(path, value)
        if Path(path).name == "commands.json":
            altered = paths[2] if change == "source" else output / "configs" / "random.json"
            altered.write_bytes(altered.read_bytes() + b" ")
    monkeypatch.setattr(experiment, "write_json", modified)
    with pytest.raises(ValueError, match="changed"):
        experiment.prepare_revision_experiment(*paths, output)
    assert output.is_dir() and not (output / "manifest.json").exists()


def test_maximum_repair_budget_produces_24_and_16_call_caps(tmp_path):
    paths = materials(tmp_path, max_repairs=8, max_calls=24)
    report = experiment.prepare_revision_experiment(*paths, tmp_path / "prepared")
    assert read_json(report["configs"]["targeted"]["path"])["max_calls"] == 24
    assert read_json(report["configs"]["text_only"]["path"])["max_calls"] == 16
