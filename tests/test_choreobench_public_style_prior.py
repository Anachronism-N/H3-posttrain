"""CPU-only toy probabilities and synthetic providers; no model/quality claims."""
import copy
import json

import pytest

from choreobench import composition_workflows as composition, coordination, harness
from choreobench.audio import make_windows
from choreobench.baseline import ModelOutputError, _empty_plan
from choreobench.candidates import load_candidates
from choreobench.cli import main
from choreobench.io import file_sha, json_text, read_json, write_json
from choreobench.planning_units import build_planning_units
from choreobench.public_style_prior import CLASS_ORDER, MODEL_IDENTITY, SCHEMA, load_public_style_prior
from choreobench.selection import candidate_options
from test_choreobench_baseline import dataset_file
from test_choreobench_composition_workflows import material, state
from test_choreobench_coordination import config, fake_planner
from test_choreobench_harness import fake_selector, fixture


def sidecar(tmp_path, dataset_path):
    songs = read_json(dataset_path)["songs"]
    rows = []
    for song in songs:
        audio = song["audio"]
        windows = []
        for window in make_windows(audio["num_samples"], audio["sample_rate"]):
            p = [0.] * 7
            p[window["index"] % 7] = .7
            p[(window["index"] + 1) % 7] = .3
            windows.append(dict(start=window["observation_start"], end=window["observation_end"],
                                audio_sha256="a" * 64, probabilities=p))
        rows.append(dict(song_id=song["song_id"], audio_sha256=audio["sha256"],
                         sample_rate=audio["sample_rate"], num_samples=audio["num_samples"], windows=windows))
    value = dict(schema=SCHEMA, dataset_sha256=file_sha(dataset_path),
                 model_identity=copy.deepcopy(MODEL_IDENTITY), songs=rows)
    path = tmp_path / "style-prior.json"
    write_json(path, value)
    return path, value


def load(path, dataset_path):
    return load_public_style_prior(path, file_sha(dataset_path), read_json(dataset_path)["songs"])


def payload(request):
    return json.loads(request["messages"][-1]["content"])


def test_validation_readonly_deterministic_aggregation_and_canonical_ties(tmp_path):
    dataset, song = dataset_file(tmp_path)
    path, value = sidecar(tmp_path, dataset)
    before = {p: file_sha(p) for p in (dataset, path, tmp_path / "song.wav")}
    prior = load(path, dataset)
    global_signal = prior.signal(song["song_id"])
    expected = [sum(w["probabilities"][i] for w in value["songs"][0]["windows"]) / 3 for i in range(7)]
    assert global_signal["probabilities"] == pytest.approx(expected)
    assert global_signal["aggregation"]["method"] == "equal_window_mean"
    assert global_signal["aggregation"]["coverage_samples"] == song["audio"]["num_samples"]
    assert not global_signal["aggregation"]["window_audio_hashes_verified"]
    assert global_signal["model_identity"] == MODEL_IDENTITY
    assert global_signal["model_identity"]["calibrated"]
    assert "not human" in global_signal["model_identity"]["calibration_scope"]
    assert "video VLM labels" in global_signal["model_identity"]["pseudolabel_caveat"]
    assert global_signal["sidecar_sha256"] == file_sha(path)
    unit = dict(start=0, end=96000)
    unit_signal = prior.signal(song["song_id"], unit)
    assert unit_signal["aggregation"]["window_count"] == 2
    assert unit_signal["aggregation"]["weight_sum"] == unit_signal["aggregation"]["coverage_samples"] == 96000
    assert unit_signal["probabilities"] == pytest.approx([
        (81000 * value["songs"][0]["windows"][0]["probabilities"][i]
         + 15000 * value["songs"][0]["windows"][1]["probabilities"][i]) / 96000 for i in range(7)])
    # Serialization/input ordering must not change the chosen mean or ranking.
    value["songs"][0]["windows"].reverse()
    write_json(path, value, overwrite=True)
    reordered = load(path, dataset).signal(song["song_id"])
    assert reordered["probabilities"] == global_signal["probabilities"]
    assert reordered["top3"] == global_signal["top3"]
    for w in value["songs"][0]["windows"]:
        w["probabilities"] = [1 / 7] * 7
    write_json(path, value, overwrite=True)
    tied = load(path, dataset).signal(song["song_id"])
    assert [r["label"] for r in tied["top3"]] == list(CLASS_ORDER[:3])
    assert {p: file_sha(p) for p in (dataset, tmp_path / "song.wav")} == {p: before[p] for p in (dataset, tmp_path / "song.wav")}
    with pytest.raises(ValueError, match="source bytes changed"):
        prior.assert_unchanged()


@pytest.mark.parametrize("problem", [
    "dataset", "source_hash", "rate", "samples", "bool_clock", "unknown_song", "duplicate_song", "missing_song",
    "private_song", "private_root", "reference_window", "identity_hash", "identity_architecture", "class_order",
    "calibrated", "calibrated_integer", "caveat", "window_hash", "window_negative", "window_outside", "window_short",
    "window_bool", "window_duplicate", "windows_null",
])
def test_unbound_or_private_sidecar_rejected_before_provider(tmp_path, monkeypatch, problem):
    dataset, song, bank = fixture(tmp_path)
    path, value = sidecar(tmp_path, dataset)
    row, identity = value["songs"][0], value["model_identity"]
    window = row["windows"][0]
    if problem == "dataset": value["dataset_sha256"] = "0" * 64
    elif problem == "source_hash": row["audio_sha256"] = "0" * 64
    elif problem == "rate": row["sample_rate"] += 1
    elif problem == "samples": row["num_samples"] += 1
    elif problem == "bool_clock": row["sample_rate"] = True
    elif problem == "unknown_song": row["song_id"] = "other-song"
    elif problem == "duplicate_song": value["songs"].append(copy.deepcopy(row))
    elif problem == "missing_song": value["songs"] = []
    elif problem == "private_song": row["private_labels"] = ["hip-hop"]
    elif problem == "private_root": value["reference_captions"] = ["secret"]
    elif problem == "reference_window": window["reference_caption"] = "secret"
    elif problem == "identity_hash": identity["archive_sha256"] = "0" * 64
    elif problem == "identity_architecture": identity["architecture"] = "six zero-shot"
    elif problem == "class_order": identity["class_order"].reverse()
    elif problem == "calibrated": identity["calibrated"] = False
    elif problem == "calibrated_integer": identity["calibrated"] = 0
    elif problem == "caveat": identity["pseudolabel_caveat"] = "audited ground truth"
    elif problem == "window_hash": window["audio_sha256"] = "unknown"
    elif problem == "window_negative": window["start"] = -1
    elif problem == "window_outside": window.update(start=row["num_samples"], end=row["num_samples"] + 81000)
    elif problem == "window_short": window["end"] -= 1
    elif problem == "window_bool": window["start"] = False
    elif problem == "window_duplicate": row["windows"].append(copy.deepcopy(window))
    else: row["windows"] = None
    write_json(path, value, overwrite=True)
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: pytest.fail("No provider before validation"))
    with pytest.raises(ValueError):
        harness.run_harness(dataset, bank, tmp_path / "out", config(), style_prior=path)
    assert not (tmp_path / "out").exists()
    assert read_json(dataset)["songs"][0] == song


@pytest.mark.parametrize("probabilities", [
    [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [True, 0, 0, 0, 0, 0, 0],
    ["1", 0, 0, 0, 0, 0, 0], [-.1, 1.1, 0, 0, 0, 0, 0], [.1] * 7, [0] * 7,
    [float("nan"), 0, 0, 0, 0, 0, 0], [float("inf"), 0, 0, 0, 0, 0, 0],
    [10 ** 400, 0, 0, 0, 0, 0, 0], {"hip-hop": 1}, None,
])
def test_invalid_probabilities_never_become_a_distribution(tmp_path, probabilities):
    dataset, _ = dataset_file(tmp_path)
    path, value = sidecar(tmp_path, dataset)
    value["songs"][0]["windows"][0]["probabilities"] = probabilities
    # Raw JSON also exercises the strict nonfinite-reader boundary.
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path, dataset)


@pytest.mark.parametrize("delta,accepted", [(0.0000005, True), (0.000002, False)])
def test_sum_tolerance_without_silent_renormalization(tmp_path, delta, accepted):
    dataset, song = dataset_file(tmp_path)
    path, value = sidecar(tmp_path, dataset)
    value["songs"][0]["windows"] = [value["songs"][0]["windows"][0]]
    value["songs"][0]["windows"][0]["probabilities"] = [1 + delta, 0, 0, 0, 0, 0, 0]
    write_json(path, value, overwrite=True)
    if accepted:
        assert load(path, dataset).signal(song["song_id"])["probabilities"][0] == 1 + delta
    else:
        with pytest.raises(ValueError): load(path, dataset)


def test_missing_windows_and_nonoverlap_are_explicit_no_global_fallback(tmp_path):
    dataset, song = dataset_file(tmp_path)
    path, value = sidecar(tmp_path, dataset)
    row = value["songs"][0]
    row["windows"] = row["windows"][:1]
    write_json(path, value, overwrite=True)
    prior = load(path, dataset)
    missing = prior.signal(song["song_id"], dict(start=96000, end=192000))
    assert missing["status"] == "no_overlapping_windows"
    assert missing["probabilities"] is None and missing["top3"] == []
    assert missing["aggregation"]["coverage_samples"] == missing["aggregation"]["weight_sum"] == 0
    assert prior.signal(song["song_id"])["probabilities"] is not None
    row["windows"] = []
    write_json(path, value, overwrite=True)
    empty = load(path, dataset).signal(song["song_id"])
    assert empty["status"] == "missing_windows" and empty["probabilities"] is None and empty["top3"] == []


def test_exact_dataset_songset_includes_unselected_songs(tmp_path):
    dataset, song = dataset_file(tmp_path)
    value = read_json(dataset)
    other = copy.deepcopy(song)
    other.update(song_id="other-song", split="test")
    value["songs"].append(other)
    write_json(dataset, value, overwrite=True)
    path, side = sidecar(tmp_path, dataset)
    assert load(path, dataset).signal("other-song")["song_id"] == "other-song"
    side["songs"].pop()
    write_json(path, side, overwrite=True)
    with pytest.raises(ValueError, match="song set"): load(path, dataset)


def test_none_is_backward_compatible_and_added_signal_counts_in_request_budget(tmp_path):
    dataset, song, bank = fixture(tmp_path)
    path, _ = sidecar(tmp_path, dataset)
    prior = load(path, dataset)
    units = build_planning_units(song)
    options = candidate_options(load_candidates(bank), song, units)
    cfg = harness._config(config(structured_unit_output=True))
    outline = coordination._neutral_outline(song)
    plan = _empty_plan(song)
    for request_fn, args in [
        (coordination._outline_request, (song, options, units)),
        (coordination._unit_request, (song, options, units, units["units"][0], outline, plan)),
    ]:
        original = request_fn(*args, cfg)
        assert request_fn(*args, cfg, style_prior=None) == original
        assert "style_prior" not in payload(original)
        enabled = request_fn(*args, cfg, style_prior=prior)
        unchanged = payload(enabled)
        unchanged.pop("style_prior")
        assert unchanged == payload(original)
        assert enabled.get("response_schema") == original.get("response_schema")
        assert enabled["messages"][0] == original["messages"][0]
        limit = len(json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        bounded = dict(cfg, max_request_characters=limit)
        assert request_fn(*args, bounded) == original
        with pytest.raises(ModelOutputError, match="max_request_characters"):
            request_fn(*args, bounded, style_prior=prior)
    assert load_public_style_prior(None, "unused", []) is None


@pytest.mark.parametrize("mode", ["coordinated", "llm", "index0"])
def test_harness_binding_journaling_resume_and_bank_readonly(tmp_path, monkeypatch, mode):
    dataset, song, bank = fixture(tmp_path)
    path, side = sidecar(tmp_path, dataset)
    immutable = [dataset, bank, tmp_path / "song.wav"] + [p for p in (tmp_path / "collection").rglob("*") if p.is_file()]
    before = {p: file_sha(p) for p in immutable}
    calls = []
    if mode == "coordinated": fake_planner(monkeypatch, calls); cfg = config()
    elif mode == "llm": fake_selector(monkeypatch, calls); cfg = dict(mode=mode, selector_provider={"kind": "fixture"})
    else: cfg = dict(mode=mode, max_calls=0)
    out = tmp_path / "out"
    result = harness.run_harness(dataset, bank, out, cfg, style_prior=path.name)
    assert result["all_songs_succeeded"]
    contract = read_json(out / "contract.json")
    assert contract["public_style_prior_snapshot"] == side
    assert contract["public_style_prior_sha256"] == contract["sources"][str(path)] == file_sha(path)
    assert contract["dataset_snapshot"]["songs"][0] == song
    prior = load(path, dataset)
    for request in calls:
        data = payload(request)
        assert data["style_prior"] == prior.signal(song["song_id"], data.get("planning_unit"))
        assert "private-song-group" not in request["messages"][-1]["content"]
        assert "reference_captions" not in request["messages"][-1]["content"]
        assert data.get("brief", data.get("song", {}).get("brief")) == song["brief"]
    count = len(calls)
    assert harness.run_harness(dataset, bank, out, cfg, resume=True, style_prior=path) == result
    assert len(calls) == count
    assert {p: file_sha(p) for p in immutable} == before
    # Even semantically identical raw sidecar bytes must invalidate resume.
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable"):
        harness.run_harness(dataset, bank, out, cfg, resume=True, style_prior=path)
    with pytest.raises(ValueError, match="Immutable"):
        harness.run_harness(dataset, bank, out, cfg, resume=True)
    assert len(calls) == count


@pytest.mark.parametrize("controller", ["v1", "v2"])
@pytest.mark.parametrize("method", composition.METHODS)
def test_all_composition_methods_get_identical_public_signal_and_resume(tmp_path, monkeypatch, method, controller):
    root, sid, out, calls = material(tmp_path, monkeypatch, method)
    dataset = tmp_path / "dataset.json"
    path, side = sidecar(tmp_path, dataset)
    prior = load(path, dataset)
    result = composition.run_composition(root, sid, {"kind": "replay"}, out,
                                         method=method, controller_version=controller, style_prior=path)
    assert result["full_song_succeeded"]
    contract = read_json(out / "contract.json")
    assert contract["public_style_prior_snapshot"] == side
    assert contract["public_style_prior_sha256"] == contract["sources"][str(path)] == file_sha(path)
    for req in calls:
        data = payload(req)
        assert data["style_prior"] == prior.signal(sid, data.get("planning_unit"))
        assert "style_prior" not in data.get("output_contract", {})
    count = len(calls)
    assert composition.run_composition(root, sid, {"kind": "replay"}, out, method=method,
        controller_version=controller, style_prior=path, resume=True) == result
    assert len(calls) == count
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Immutable composition"):
        composition.run_composition(root, sid, {"kind": "replay"}, out, method=method,
            controller_version=controller, style_prior=path, resume=True)
    assert len(calls) == count


def test_composition_none_and_all_method_request_signals_match(tmp_path, monkeypatch):
    s = state(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset.json"
    path, _ = sidecar(tmp_path, dataset)
    prior = load(path, dataset)
    cfg = dict(seed=42, max_output_tokens=1024, max_request_characters=200000, call_index=1)
    original = composition._request(s, cfg, "workflow", "propose")
    assert "style_prior" not in payload(original)
    s.source["style_prior"] = None
    assert composition._request(s, cfg, "workflow", "propose") == original
    s.source["style_prior"] = prior
    signals = [payload(composition._request(s, cfg, method, "propose"))["style_prior"] for method in composition.METHODS]
    assert all(signal == signals[0] for signal in signals)
    limit = len(json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ModelOutputError, match="max_request_characters"):
        composition._request(s, dict(cfg, max_request_characters=limit), "workflow", "propose")


def test_style_fields_count_toward_total_input_budget_before_dispatch(tmp_path, monkeypatch):
    root, sid, out, calls = material(tmp_path, monkeypatch, "workflow")
    source = composition._source(root, sid, {}, {})
    cfg = dict(seed=42, max_output_tokens=1024, outline_max_output_tokens=4096, max_request_characters=200000)
    req = coordination._outline_request(source["song"], source["options"], source["units"], cfg)
    budget = len(json_text(req["messages"]).encode("utf-8")) + 1024
    path, _ = sidecar(tmp_path, tmp_path / "dataset.json")
    result = composition.run_composition(root, sid, {"kind": "replay"}, out, style_prior=path, max_input_tokens=budget)
    assert not result["full_song_succeeded"] and result["calls_used"] == 0 and calls == []
    assert "context byte bound" in result["error"]


@pytest.mark.parametrize("command", ["run-harness", "run-composition"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cli_sidecar_is_opt_in_and_forwarded(tmp_path, monkeypatch, command, enabled):
    spec = tmp_path / "spec.json"
    write_json(spec, {})
    calls = []
    def capture(*args, **kwargs): calls.append(kwargs); return {"fixture": True}
    if command == "run-harness":
        monkeypatch.setattr(harness, "run_harness", capture)
        flags = ["--dataset", "dataset.json", "--candidates", "bank.json", "--config", str(spec)]
    else:
        monkeypatch.setattr(composition, "run_composition", capture)
        flags = ["--source", "source", "--song-id", "song1", "--provider", str(spec)]
    path = tmp_path / "not-loaded.json"
    assert main([command, "--out-dir", str(tmp_path / "out")] + flags
                + (["--style-prior", str(path)] if enabled else [])) == 0
    assert len(calls) == 1 and calls[0]["style_prior"] == (path if enabled else None)


@pytest.mark.parametrize("kind", ["catalog", "song_view"])
def test_style_binding_uses_original_dataset_for_catalog_sources_readonly(tmp_path, monkeypatch, kind):
    from choreobench.catalog_song import export_song
    from test_choreobench_catalog_song import source as catalog_source
    song, root, _, calls = catalog_source(tmp_path, monkeypatch)
    source_path = root / "results.json"
    if kind == "song_view":
        source_path = tmp_path / "song-view.json"
        export_song(root, song["song_id"], source_path)
    before = {p: file_sha(p) for p in root.rglob("*") if p.is_file()}
    sources = {}
    source = composition._source(source_path, song["song_id"], dict(unit_config={}), sources)
    path, _ = sidecar(tmp_path, tmp_path / "dataset.json")
    relative = "../" + path.name if kind == "catalog" else path.name
    prior = composition._load_style_prior(relative, source_path, source, sources)
    assert prior.signal(song["song_id"])["dataset_sha256"] == file_sha(tmp_path / "dataset.json")
    assert sources[str(path)] == prior.digest
    assert {p: file_sha(p) for p in before} == before and len(calls) == 6
    source["original"] = copy.deepcopy(source["original"])
    source["original"]["audio"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source song differs"):
        composition._load_style_prior(path, source_path, source, sources)


def test_outline_reuse_requires_identical_style_prior_request(tmp_path, monkeypatch):
    root, sid, out, calls = material(tmp_path, monkeypatch, "workflow")
    path, _ = sidecar(tmp_path, tmp_path / "dataset.json")
    composition.run_composition(root, sid, {"kind": "replay"}, out, style_prior=path)
    count = len(calls)
    reused = composition.run_composition(root, sid, {"kind": "replay"}, tmp_path / "reused",
                                          style_prior=path, outline_reuse_run=out)
    assert reused["full_song_succeeded"] and reused["reused_outline_calls"] == 1
    assert len(calls) == count + 2
    count = len(calls)
    with pytest.raises(ValueError, match="outline request differs"):
        composition.run_composition(root, sid, {"kind": "replay"}, tmp_path / "disabled-reuse", outline_reuse_run=out)
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="outline request differs"):
        composition.run_composition(root, sid, {"kind": "replay"}, tmp_path / "changed-reuse",
                                     style_prior=path, outline_reuse_run=out)
    assert len(calls) == count


def test_composition_rejects_wrong_dataset_sidecar_before_provider(tmp_path, monkeypatch):
    root, sid, out, calls = material(tmp_path, monkeypatch, "workflow")
    path, value = sidecar(tmp_path, tmp_path / "dataset.json")
    value["dataset_sha256"] = "0" * 64
    write_json(path, value, overwrite=True)
    monkeypatch.setattr(composition, "create_provider", lambda *a, **k: pytest.fail("No provider before validation"))
    with pytest.raises(ValueError, match="exact source dataset"):
        composition.run_composition(root, sid, {"kind": "replay"}, out, style_prior=path)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize("runner", ["harness", "composition"])
def test_sidecar_drift_during_run_stops_subsequent_calls(tmp_path, monkeypatch, runner):
    if runner == "harness":
        dataset, _, bank = fixture(tmp_path)
        path, _ = sidecar(tmp_path, dataset)
        calls = []
        def mutate(answer, req, data):
            path.write_text(path.read_text() + " ", encoding="utf-8")
        fake_planner(monkeypatch, calls, mutate)
        with pytest.raises(ValueError, match="style prior source bytes changed"):
            harness.run_harness(dataset, bank, tmp_path / "out", config(), style_prior=path)
    else:
        def mutate(answer, req, data, index):
            if index == 1: path.write_text(path.read_text() + " ", encoding="utf-8")
        root, sid, out, calls = material(tmp_path, monkeypatch, "workflow", mutate)
        path, _ = sidecar(tmp_path, tmp_path / "dataset.json")
        with pytest.raises(ValueError, match="source changed"):
            composition.run_composition(root, sid, {"kind": "replay"}, out, style_prior=path)
    assert len(calls) == 1


def test_shared_outline_without_style_binding_is_rejected(tmp_path, monkeypatch):
    dataset, _, bank = fixture(tmp_path)
    path, _ = sidecar(tmp_path, dataset)
    monkeypatch.setattr(coordination, "create_provider", lambda *a, **k: pytest.fail("No provider before validation"))
    with pytest.raises(ValueError, match="Shared outline v1 cannot bind"):
        harness.run_harness(dataset, bank, tmp_path / "out", config(shared_outline_path="unused"), style_prior=path)
