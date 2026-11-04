import copy
import itertools
import json

import pytest

from choreobench import baseline, candidates
from choreobench.compile_selected import compile_selection
from choreobench.io import object_sha, read_json, write_json
from choreobench.schema import validate_plan
from choreobench.selection import NoFeasibleSelectionError, candidate_options, make_selection, select_with_scores, validate_selection
from test_choreobench_baseline import dataset_file


def bank_fixture(tmp_path, *, seconds=12, count=2, stride=5.0625):
    source, song = dataset_file(tmp_path, seconds=seconds)
    origin = tmp_path / "origin"
    baseline.run_baseline(source, origin, dict(candidates_per_window=count, stride_seconds=stride))
    bank_path = tmp_path / "bank.json"
    candidates.export_candidates(origin, bank_path)
    return song, candidates.load_candidates(bank_path)


def choose(options, index=0):
    return [dict(window_id=row["window_id"], candidate_index=index) for row in options["windows"]]


def reseal(document):
    document = copy.deepcopy(document)
    document.pop("record_sha256", None)
    document["record_sha256"] = object_sha(document)
    return document


def test_public_options_keep_exact_whole_pairs_and_no_private_fields(tmp_path):
    song, bank = bank_fixture(tmp_path)
    units = [dict(unit_id="block-a", start=0, end=100000, kind="time_block", private_goal="SECRET"),
             dict(unit_id="block-b", start=100000, end=192000, kind="time_block")]
    options = candidate_options(bank, song, units)
    serialized = json.dumps(options)
    for hidden in ("private-song-group", "private-artist-group", "private-duplicate-group", "song.wav", "SECRET"):
        assert hidden not in serialized
    assert "split" not in options and options["synthetic"]
    assert options["windows"][1]["unit_refs"] == ["block-a", "block-b"]
    middle = options["windows"][1]["candidates"][0]["fragments"]
    assert middle[0]["partial_source"] and middle[0]["ownership_intersection"] == [81000, 111000]
    assert not middle[0]["fully_inside_ownership"]
    assert middle[1]["ownership_intersection"] is None
    for window in options["windows"]:
        assert len(window["candidates"]) == 2
        for candidate in window["candidates"]:
            entry = bank.entries[candidate["call_id"]]
            assert candidate["raw_text"] == entry["raw"]["result"]["text"]
            assert len(candidate["fragments"]) == 2
            assert candidate["raw_record_sha256"] == entry["raw"]["record_sha256"]
            assert candidate["request_sha256"] == entry["started"]["request_sha256"]
    # Unit context cannot change the authority to select one pair per window.
    choices = choose(options, 1)
    selection = make_selection(bank, song, list(reversed(choices)))
    assert selection["choices"] == choices
    assert validate_selection(selection, bank, song) is selection


@pytest.mark.parametrize("change", ["missing", "duplicate", "other_window", "other_candidate", "bool", "part", "dict"])
def test_selection_requires_every_window_once_and_one_whole_pair(tmp_path, change):
    song, bank = bank_fixture(tmp_path)
    choices = choose(candidate_options(bank, song))
    if change == "missing":
        choices.pop()
    elif change == "duplicate":
        choices.append(copy.deepcopy(choices[0]))
    elif change == "other_window":
        choices[0]["window_id"] = "another-song-w0000"
    elif change == "other_candidate":
        choices[0]["candidate_index"] = 2
    elif change == "bool":
        choices[0]["candidate_index"] = True
    elif change == "part":
        choices[0]["part_index"] = 0
    else:
        choices = {"choices": choices}
    with pytest.raises(ValueError):
        make_selection(bank, song, choices)


def test_tail_partial_source_strict_leaves_real_gap_and_crop_is_explicit(tmp_path):
    song, bank = bank_fixture(tmp_path)
    options = candidate_options(bank, song)
    selection = make_selection(bank, song, choose(options, 1))
    strict = compile_selection(bank, song, selection)
    plan, report = strict["plan"], strict["report"]
    validate_plan(plan, song)
    assert [(e["start"], e["end"]) for e in plan["events"]] == [
        (0, 32000), (32000, 81000), (111000, 143000), (143000, 192000)]
    assert report["omitted_intervals"] == [dict(start=81000, end=111000)]
    assert report["covered_samples"] == 162000
    assert report["coverage_fraction"] == 162000 / 192000
    assert report["partial_source_count"] == 1
    assert not report["empty_intervals_filled_with_rest"]
    middle = next(m for m in report["source_mappings"] if m["window_id"] == "w0001" and m["part_index"] == 0)
    assert middle["source_interval"] == [81000, 113000]
    assert middle["ownership_intersection"] == [81000, 111000]
    assert middle["scheduled_interval"] is None
    assert middle["time_transform"] == "omitted_partial_source"
    assert report["needs_review"][0]["retained"] is False
    for event in plan["events"]:
        assert event["entry"] == event["exit"] == {}
        pair = next(c for row in options["windows"] for c in row["candidates"] if c["call_id"] == event["evidence"][0])
        assert pair["candidate_index"] == 1
        assert event["action"] in [f["text"] for f in pair["fragments"]]
    cropped = compile_selection(bank, song, selection, timing_policy="proposed_crop")
    validate_plan(cropped["plan"], song)
    assert cropped["report"]["coverage_fraction"] == 1.
    assert cropped["report"]["omitted_intervals"] == []
    proposal = next(m for m in cropped["report"]["source_mappings"] if m["time_transform"] == "crop_proposed")
    assert proposal["source_interval"] == [81000, 113000]
    assert proposal["scheduled_interval"] == [81000, 111000]
    assert proposal["partial_source"] and not proposal["semantic_timing_verified"]
    assert cropped["report"]["needs_review"][0]["status"] == "pending"
    assert not cropped["report"]["music_or_physical_quality_verified"]
    assert cropped["report"]["synthetic"] and cropped["report"]["new_local_calls"] == 0


def test_exact_full_window_has_identity_timing_and_no_inferred_state(tmp_path):
    song, bank = bank_fixture(tmp_path, seconds=5.0625)
    options = candidate_options(bank, song)
    result = compile_selection(bank, song, make_selection(bank, song, choose(options)))
    assert result["report"]["coverage_fraction"] == 1
    assert result["report"]["needs_review"] == []
    assert all(m["time_transform"] == "identity" for m in result["report"]["source_mappings"])
    assert all(not e["entry"] and not e["exit"] for e in result["plan"]["events"])


@pytest.mark.parametrize("field", ["brief", "path", "audio_sha", "music"])
def test_changed_current_song_is_not_the_source_snapshot(tmp_path, field):
    song, bank = bank_fixture(tmp_path)
    selection = make_selection(bank, song, choose(candidate_options(bank, song)))
    changed = copy.deepcopy(song)
    if field == "brief":
        changed["brief"] += " changed"
    elif field == "path":
        changed["audio"]["path"] = "other.wav"
    elif field == "audio_sha":
        changed["audio"]["sha256"] = "a" * 64
    else:
        changed["music"]["source"] = "other tool identity"
    with pytest.raises(ValueError, match="snapshot differs"):
        compile_selection(bank, changed, selection)


@pytest.mark.parametrize("change", ["choice", "source", "binding", "extra"])
def test_resealed_selection_cannot_change_bound_sources(tmp_path, change):
    song, bank = bank_fixture(tmp_path)
    selection = make_selection(bank, song, choose(candidate_options(bank, song)))
    if change == "choice":
        selection["choices"][0]["candidate_index"] = 1
    elif change == "source":
        selection["sources"][0]["call_id"] = "other/local/0000/00"
    elif change == "binding":
        selection["binding"]["bank_file_sha256"] = "a" * 64
    else:
        selection["hidden_override"] = True
    with pytest.raises(ValueError, match="binding differs"):
        compile_selection(bank, song, reseal(selection))


def test_bank_byte_changes_and_in_memory_raw_tampering_are_rejected(tmp_path):
    song, bank = bank_fixture(tmp_path)
    options = candidate_options(bank, song)
    selection = make_selection(bank, song, choose(options))
    first = next(iter(bank.entries.values()))
    first["raw"]["result"]["text"] += " tampered"
    with pytest.raises(ValueError, match="hash differs"):
        compile_selection(bank, song, selection)
    # A fresh instance still detects a byte-only change despite identical JSON.
    bank = candidates.load_candidates(bank.path)
    bank.path.write_bytes(bank.path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        candidate_options(bank, song)


def test_identical_bank_bytes_are_portable_but_another_collection_is_not(tmp_path):
    song, bank = bank_fixture(tmp_path)
    selection = make_selection(bank, song, choose(candidate_options(bank, song)))
    copied = tmp_path / "copied-bank.json"
    copied.write_bytes(bank.path.read_bytes())
    assert compile_selection(candidates.load_candidates(copied), song, selection)["plan"]
    other = tmp_path / "other"
    other.mkdir()
    same_song, other_bank = bank_fixture(other)
    # New source journals/timestamps give a distinct collection even with the
    # same deterministic synthetic responses and unchanged source song.
    assert object_sha(same_song) == object_sha(song)
    with pytest.raises(ValueError, match="binding differs"):
        compile_selection(other_bank, song, selection)


def test_current_parser_change_rejected_and_no_local_provider_is_created(tmp_path, monkeypatch):
    song, bank = bank_fixture(tmp_path)
    monkeypatch.setattr(baseline, "create_provider", lambda *a, **k: pytest.fail("No provider permitted"))
    options = candidate_options(bank, song)
    assert compile_selection(bank, song, make_selection(bank, song, choose(options)))["plan"]
    original = candidates.file_sha
    monkeypatch.setattr(candidates, "file_sha", lambda p: "a" * 64 if str(p).endswith("music_planner_eval.py") else original(p))
    with pytest.raises(ValueError, match="different actual local-caption parser"):
        candidate_options(bank, song)


def test_declared_time_tolerance_remains_distinct_from_nominal_sample_clock(tmp_path, monkeypatch):
    source, song = dataset_file(tmp_path, seconds=5.0625)
    create = baseline.create_provider
    def tolerant(spec, role, base_dir):
        provider = create(spec, role, base_dir)
        original = provider.generate
        def generate(request):
            result = original(request)
            result["text"] = "[0.01-1.99s] Turn twice. [2.01-5.06s] Step right."
            return result
        provider.generate = generate
        return provider
    monkeypatch.setattr(baseline, "create_provider", tolerant)
    baseline.run_baseline(source, tmp_path / "origin", {})
    path = tmp_path / "tolerant-bank.json"
    candidates.export_candidates(tmp_path / "origin", path)
    bank = candidates.load_candidates(path)
    options = candidate_options(bank, song)
    fragment = options["windows"][0]["candidates"][0]["fragments"][0]
    assert fragment["declared_interval_seconds"] == [.01, 1.99]
    assert fragment["nominal_interval"] == [0, 2 * song["audio"]["sample_rate"]]
    report = compile_selection(bank, song, make_selection(bank, song, choose(options)))["report"]
    assert report["source_mappings"][0]["source_declared_interval_seconds"] == [.01, 1.99]
    assert "raw declared seconds retained" in report["nominal_time_normalization"]


def test_noninteger_window_length_uses_actual_source_sample_clock(tmp_path):
    # At this sample rate the 5.0625s observation is not an integer sample.
    source, song = dataset_file(tmp_path, seconds=12, rate=44101)
    baseline.run_baseline(source, tmp_path / "origin", {})
    path = tmp_path / "clock-bank.json"
    candidates.export_candidates(tmp_path / "origin", path)
    bank = candidates.load_candidates(path)
    options = candidate_options(bank, song)
    for row in options["windows"]:
        first, second = row["candidates"][0]["fragments"]
        assert first["nominal_interval"][1] == second["nominal_interval"][0]
        assert second["nominal_interval"][1] == row["window"]["observation_end"]
    result = compile_selection(bank, song, make_selection(bank, song, choose(options)))
    validate_plan(result["plan"], song)
    assert result["report"]["covered_samples"] + sum(g["end"] - g["start"] for g in result["report"]["omitted_intervals"]) == song["audio"]["num_samples"]


def test_known_failed_song_cannot_select_its_valid_prefix(tmp_path, monkeypatch):
    source, song = dataset_file(tmp_path)
    create = baseline.create_provider
    def partly_bad(spec, role, base_dir):
        provider = create(spec, role, base_dir)
        original = provider.generate
        def generate(request):
            result = original(request)
            if request["call_id"].endswith("/01"):
                result["text"] = "untimed original bad answer"
            return result
        provider.generate = generate
        return provider
    monkeypatch.setattr(baseline, "create_provider", partly_bad)
    baseline.run_baseline(source, tmp_path / "origin", dict(candidates_per_window=2, continue_on_model_error=True))
    path = tmp_path / "failed.json"
    candidates.export_candidates(tmp_path / "origin", path)
    bank = candidates.load_candidates(path)
    assert len(bank.entries) == 2  # A valid response, then the retained failure.
    with pytest.raises(baseline.ModelOutputError, match="prefix cannot be selected"):
        candidate_options(bank, song)


def test_dp_matches_brute_force_and_has_deterministic_path_ties(tmp_path):
    song, bank = bank_fixture(tmp_path)
    options = candidate_options(bank, song)
    unary = [[2, 0], [0, 1], [3, 0]]
    edges = [[[0, 3], [2, 0]], [[0, 0], [None, 2]]]
    provenance = dict(source="synthetic score fixture; no music evaluation", synthetic=True)
    result = select_with_scores(options, unary, edges, score_provenance=provenance)
    scores = []
    for path in itertools.product(range(2), repeat=3):
        if any(edges[i][path[i]][path[i + 1]] is None for i in range(2)):
            continue
        score = sum(unary[i][k] for i, k in enumerate(path)) + sum(edges[i][path[i]][path[i + 1]] for i in range(2))
        scores.append((score, path))
    expected = min(scores, key=lambda row: (-row[0], row[1]))
    assert result["total_score"] == expected[0]
    assert tuple(c["candidate_index"] for c in result["choices"]) == expected[1]
    assert result["score_provenance"] == provenance and result["synthetic"]
    selection = make_selection(bank, song, result["choices"])
    validate_plan(compile_selection(bank, song, selection)["plan"], song)
    ties = select_with_scores(options, [[0, 0]] * 3, [[[0, 0], [0, 0]]] * 2, score_provenance=provenance)
    assert [c["candidate_index"] for c in ties["choices"]] == [0, 0, 0]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1"])
def test_dp_rejects_non_numeric_nonfinite_or_bool_scores(tmp_path, bad):
    song, bank = bank_fixture(tmp_path, seconds=5.0625)
    options = candidate_options(bank, song)
    with pytest.raises(ValueError, match="finite numbers"):
        select_with_scores(options, [[bad, 0]], [], score_provenance=dict(source="fixture", synthetic=True))


def test_dp_unknown_scores_are_not_silently_zero_or_compatible(tmp_path):
    song, bank = bank_fixture(tmp_path)
    options = candidate_options(bank, song)
    provenance = dict(source="fixture", synthetic=True)
    with pytest.raises(NoFeasibleSelectionError, match="No complete path"):
        select_with_scores(options, [[None, None], [1, 1], [1, 1]], [[[0, 0], [0, 0]]] * 2,
                           score_provenance=provenance)
    with pytest.raises(NoFeasibleSelectionError, match="No complete path"):
        select_with_scores(options, [[1, 1]] * 3, [[[None, None], [None, None]], [[0, 0], [0, 0]]],
                           score_provenance=provenance)
    with pytest.raises(ValueError, match="synthetic"):
        select_with_scores(options, [[1, 1]] * 3, [[[0, 0], [0, 0]]] * 2,
                           score_provenance=dict(source="fixture"))


def test_dp_requires_correct_dimensions_and_untampered_options(tmp_path):
    song, bank = bank_fixture(tmp_path, seconds=5.0625)
    options = candidate_options(bank, song)
    with pytest.raises(ValueError, match="dimensions"):
        select_with_scores(options, [[1]], [], score_provenance=dict(source="fixture", synthetic=True))
    options["windows"][0]["candidates"][0]["raw_text"] = "edited candidate"
    with pytest.raises(ValueError, match="hash differs"):
        select_with_scores(options, [[1, 1]], [], score_provenance=dict(source="fixture", synthetic=True))


def test_dp_tie_is_over_the_full_path_not_the_last_candidate(tmp_path):
    song, bank = bank_fixture(tmp_path, seconds=10.125)
    options = candidate_options(bank, song)
    result = select_with_scores(options, [[0, 0], [0, 0]], [[[0, 1], [1, 0]]],
                                score_provenance=dict(source="fixture", synthetic=True))
    assert [c["candidate_index"] for c in result["choices"]] == [0, 1]


@pytest.mark.parametrize("provenance", [
    [], {"source": "", "synthetic": False}, {"source": "test", "synthetic": 0},
    {"source": "test", "synthetic": "false"}, {"source": "test"},
    {"source": "test", "synthetic": False, "extra": float("nan")}])
def test_selection_provenance_requires_explicit_boolean_and_finite_json(tmp_path, provenance):
    song, bank = bank_fixture(tmp_path, seconds=5.0625)
    options = candidate_options(bank, song)
    with pytest.raises(ValueError):
        make_selection(bank, song, choose(options), selection_provenance=provenance)


def test_synthetic_selector_taints_declared_real_bank_and_compilation(tmp_path, monkeypatch):
    # This deliberately declared-real recording fixture tests taint propagation;
    # its values do not constitute a verified real-model benchmark.
    source, song = dataset_file(tmp_path, seconds=5.0625)
    dataset = read_json(source)
    dataset["synthetic"] = False
    write_json(source, dataset, overwrite=True)
    create = baseline.create_provider
    def declared_recording(spec, role, base_dir):
        provider = create(spec, role, base_dir)
        provider.identity = dict(kind="declared_recording_fixture", synthetic=False, audited=False)
        original = provider.generate
        def generate(request):
            result = original(request)
            result["synthetic"] = False
            return result
        provider.generate = generate
        return provider
    monkeypatch.setattr(baseline, "create_provider", declared_recording)
    baseline.run_baseline(source, tmp_path / "origin", {})
    path = tmp_path / "declared-bank.json"
    candidates.export_candidates(tmp_path / "origin", path)
    bank = candidates.load_candidates(path)
    assert not bank.data["synthetic"]
    options = candidate_options(bank, song)
    provenance = dict(source="mock selector fixture", synthetic=True, contract_sha256="a" * 64,
                      selector_identity_sha256=None, extra={"version": 1})
    selection = make_selection(bank, song, choose(options), selection_provenance=provenance)
    assert selection["synthetic"] and selection["selection_provenance"] == provenance
    assert validate_selection(selection, bank, song) is selection
    compiled = compile_selection(bank, song, selection)
    assert compiled["report"]["synthetic"] and compiled["plan"]["provenance"]["synthetic"]
    assert compiled["report"]["selection_provenance"] == provenance
    tampered = copy.deepcopy(selection)
    tampered["synthetic"] = False
    with pytest.raises(ValueError, match="binding differs"):
        compile_selection(bank, song, reseal(tampered))
