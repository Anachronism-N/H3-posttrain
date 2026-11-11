"""Synthetic baseline -> blinded assets -> external review -> private summary."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from captionlab.baseline_eval import run_baseline
from captionlab.choreo_quality import DIMENSIONS, review_template
from captionlab.review_bundle import BundleError, collect_review_bundle, main, unblind_reviews
from choreobench.baseline import _sealed
from choreobench.io import file_sha, object_sha, read_json, write_json


@pytest.mark.parametrize("phase", ["quality", "reference"])
def test_bundle_to_judge_to_unblind_and_metric_validation(source, tmp_path, phase):
    """Exercise the actual module contracts, including reference-only NA rules."""
    from captionlab.quality_judge import run_judge, TEXT_DIMENSIONS
    from captionlab.metric_validation import validate_metric
    result_paths, reference_path, bundle = source
    identity = {"kind": "chat_completions", "synthetic": True, "fixture": "No real judge"}
    roster = {"h1": {"origin": "human", "judge_identity": None},
              "h2": {"origin": "human", "judge_identity": None},
              "j": {"origin": "model_judge", "judge_identity": object_sha(identity)}}
    collect_review_bundle(result_paths, bundle, references_path=reference_path,
                          reviewers=roster, seed=42, phase=phase)
    stage = bundle / ("public" if phase == "quality" else "private/reference")
    items = read_json(stage / "items.json")["items"]
    config = tmp_path / "judge.json"
    write_json(config, {"kind": "chat_completions", "model": "synthetic-judge", "base_url": "http://unused.invalid/v1"})

    def generate(request):
        data = json.loads(request["messages"][1]["content"])
        answer = {}
        for name in TEXT_DIMENSIONS:
            rated = name == "reference_fidelity" if phase == "reference" else name != "reference_fidelity"
            answer[name] = {"status": "rated" if rated else "not_applicable", "score": 3 if rated else None,
                            "reason": "Synthetic integration fixture.", "text_quotes": [data["text"]] if rated else [],
                            "reference_quotes": [data["reference_text"]] if rated and phase == "reference" else []}
        return {"text": json.dumps(answer), "identity": identity, "synthetic": True,
                "raw": {"synthetic": True}, "stop_reason": "stop", "costs": {}}

    judged = tmp_path / "judge-run"
    outcome = run_judge(stage / "items.json", config, "j", judged,
        provider_factory=lambda *a, **kw: SimpleNamespace(identity=identity, generate=generate))
    assert outcome["valid_reviews"] == len(items)
    humans = []
    for reviewer in ("h1", "h2"):
        for item in items:
            row = review_template(item, reviewer)
            row["review_status"] = "completed"
            row["evidence"].update(text_seen=True, text_quotes=[item["text"]],
                                   evidence_scope="Synthetic test only, no human listened.")
            for name in DIMENSIONS:
                row["ratings"][name] = {"status": "rated" if phase == "quality" else "not_applicable",
                                        "score": 3 if phase == "quality" else None, "reason": "Test fixture."}
            if phase == "quality":
                row["evidence"].update(audio_seen=True, audio_sha256=item["audio_sha256"],
                                       audio_intervals_seconds=[[0, item["duration_seconds"]]])
            else:
                row["reference_fidelity"] = {"status": "rated", "score": 3, "reason": "Test fixture."}
                row["evidence"].update(reference_seen=True, reference_quotes=[item["reference_text"]])
            humans.append(row)
    human_path = tmp_path / "humans.json"
    write_json(human_path, {"reviews": humans})
    restored = unblind_reviews(bundle, [human_path, judged / "reviews.json"], n_resamples=10)
    report = validate_metric(restored["items"], restored["reviews"], humans=["h1", "h2"], judge="j", n_resamples=10)
    dimension = "reference_fidelity" if phase == "reference" else "action_clarity"
    assert report["dimensions"][dimension]["judge_human_pair_coverage"] == 1
    assert report["dimensions"][dimension]["judge_vs_human_mean_spearman"]["estimate"] is None
    assert report["dimensions"]["music_fit"]["judge_human_pair_coverage"] == 0


def _records(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _reseal(value):
    return _sealed({key: val for key, val in value.items() if key != "record_sha256"})


def _rewrite_run(result_path, mutate, contract_mutate=None):
    result = read_json(result_path)
    contract = read_json(result_path.parent / "contract.json")
    mutate(result)
    if contract_mutate:
        contract_mutate(contract)
        contract = _reseal(contract)
        write_json(result_path.parent / "contract.json", contract, overwrite=True)
    result["contract_sha256"] = object_sha(contract)
    result["rows"] = [_reseal(row) for row in result["rows"]]
    result["case_count"] = result["calls_used"] = len(result["rows"])
    for status in ("complete", "failed", "unknown"):
        result[status + "_count"] = sum(row["status"] == status for row in result["rows"])
    raw = _records(result["rows"])
    result["records_sha256"] = hashlib.sha256(raw).hexdigest()
    (result_path.parent / "records.jsonl").write_bytes(raw)
    write_json(result_path, _reseal(result), overwrite=True)


@pytest.fixture
def source(tmp_path):
    cases = []
    for index in range(3):
        path = tmp_path / ("input" + str(index) + ".wav")
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 16000, 81000, "NONE", "not compressed"))
            handle.writeframes(bytes([index]) * 162000)
        cases.append({"case_id": "original_case_" + str(index), "source_group_id": "original_group_" + str(index),
                      "split": "test", "audio": {"path": str(path), "sha256": file_sha(path), "sample_rate": 16000, "num_samples": 81000},
                      "duration_seconds": 5.0625})
    manifest = {"schema": "captionlab.baseline_manifest.v1", "synthetic": True,
                "dataset": {"repo_id": "synthetic-not-HF-evidence", "source_files": {"test": "frozen-file-sha"}},
                "selection": {"split": "test", "limit": 3, "seed": 1}, "cases": cases}
    manifest_path, provider_path = tmp_path / "manifest.json", tmp_path / "provider.json"
    write_json(manifest_path, manifest)
    write_json(provider_path, {"kind": "native", "family": "omni_captioner", "model": "/never-loaded", "task_prompt": "Audio-only task."})
    results = []
    for index, method in enumerate(("PRIVATE_METHOD_ALPHA", "PRIVATE_METHOD_BETA")):
        identity = {"kind": "native", "synthetic": True, "fixture": "No model, no GPU", "nested": {"version": 1}}
        def generate(request, identity=identity, index=index):
            return {"text": "Raise one arm, then lower it." if index == 0 else "Lean gently, then extend an arm.",
                    "identity": identity, "synthetic": True, "raw": {"fixture": True}, "stop_reason": "stop",
                    "costs": {"input_tokens": 10, "output_tokens": 8, "latency_seconds": 0}}
        provider = SimpleNamespace(identity=identity, generate=generate)
        out = tmp_path / ("run" + str(index))
        run_baseline(manifest_path, provider_path, method, out, provider_factory=lambda *args, provider=provider, **kwargs: provider)
        results.append(out / "results.json")
    refs = []
    for case in cases:
        raw = "Dance style: PRIVATE_STYLE_GOLD\nSECRET_REFERENCE: one wrist moves left."
        refs.append({"case_id": case["case_id"], "source_group_id": case["source_group_id"], "split": "test",
                     "raw_caption": raw, "action_reference": raw.partition("\n")[2],
                     "raw_caption_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    reference_path = tmp_path / "references.private.json"
    write_json(reference_path, {"schema": "captionlab.sol61_references.v1", "selection_sha256": object_sha(manifest["selection"]),
                                "source_files": manifest["dataset"]["source_files"], "records": refs})
    return results, reference_path, tmp_path / "bundle"


def collect(source, **kwargs):
    return collect_review_bundle(source[0], source[2], seed=7, **kwargs)


def _filled(item, *, reference=False):
    review = review_template(item, "reviewer_1")
    review["review_status"] = "completed"
    evaluable = item["generation_status"] == "ok" and not item["mechanical"]["empty_body"]
    review["reference_fidelity"] = {"status": "not_applicable", "score": None, "reason": "Separate training reference not used for quality."}
    for dimension in DIMENSIONS:
        review["ratings"][dimension] = {"status": "rated" if evaluable and not reference else "not_applicable",
                                        "score": 4 if evaluable and not reference else None,
                                        "reason": "Human judgment of the bound evidence."}
    if evaluable:
        review["evidence"].update(text_seen=True, text_quotes=[item["text"]], evidence_scope="I inspected this synthetic test text and full audio.")
        if not reference:
            review["evidence"].update(audio_seen=True, audio_sha256=item["audio_sha256"], audio_intervals_seconds=[[0, 5.0625]])
        else:
            review["reference_fidelity"] = {"status": "rated", "score": 2, "reason": "Subset agreement only, not overall quality."}
            review["evidence"].update(reference_seen=True, reference_quotes=[item["reference_text"]])
    return review


def _items(bundle, phase="quality"):
    return read_json(bundle / ("public" if phase == "quality" else "private/reference") / "items.json")["items"]


def test_full_synthetic_roundtrip_anonymizes_and_keeps_gt_private(source, tmp_path):
    manifest = collect(source, references_path=source[1])
    assert manifest["synthetic"] and manifest["item_count"] == 6 and manifest["case_count"] == 3
    assert not manifest["quality_verified"]
    public = source[2] / "public"
    assert (public / "index.html").is_file()
    items = _items(source[2])
    for bound in items:
        assert bound["reference_text"] is None
        assert bound["method_id"].startswith("candidate_")
        assert bound["case_id"].startswith("case_")
        assert bound["source_group_id"].startswith("source_")
    for path in public.rglob("*"):
        if path.is_file() and path.suffix != ".wav":
            value = path.read_text(encoding="utf-8")
            for secret in ("SECRET_REFERENCE", "PRIVATE_STYLE_GOLD", "PRIVATE_METHOD_ALPHA", "PRIVATE_METHOD_BETA", "original_case_", "original_group_"):
                assert secret not in value
    mapping = read_json(source[2] / "private/mapping.json")
    assert mapping["private_references"]
    assets = read_json(public / "audio_assets.json")["assets"]
    for asset in assets:
        assert file_sha(public / asset["path"]) == asset["sha256"]
    path = tmp_path / "returned.json"
    write_json(path, {"reviews": [_filled(bound) for bound in items]})
    report = unblind_reviews(source[2], [path], method_a="PRIVATE_METHOD_ALPHA", method_b="PRIVATE_METHOD_BETA", n_resamples=20)
    assert report["synthetic"]
    assert {bound["method_id"] for bound in report["items"]} == {"PRIVATE_METHOD_ALPHA", "PRIVATE_METHOD_BETA"}
    assert all(bound["reference_text"] is None for bound in report["items"])
    assert len(report["review_bindings"]) == 6
    assert report["summary"]["paired_comparisons"]["human"]["action_clarity"]["estimate_group_equal"] == 0
    assert len(report["summary"]["coverage"]["missing_review_slots"]) == 6  # reviewer_2 remains missing.


def test_statuses_failure_unknown_and_truncation_are_retained(source):
    def mutate(result):
        result["rows"][0].update(status="unknown", stop_reason=None, text="")
        result["rows"][1].update(status="failed", stop_reason="length", text="Partial uncompleted action")
        result["rows"][2].update(status="failed", stop_reason="content_filter", text="")
    _rewrite_run(source[0][1], mutate)
    collect(source)
    rows = [row for row in _items(source[2]) if row["generation_status"] != "ok"]
    assert {row["generation_status"] for row in rows} == {"unknown", "truncated", "failed"}
    assert len(_items(source[2])) == 6
    report = unblind_reviews(source[2], [], n_resamples=10)
    assert report["summary"]["coverage"]["registered_items"] == 6
    assert report["summary"]["coverage"]["reviewable_items"] == 3


def test_mapping_is_per_case_randomized_not_a_fixed_public_method_label(source):
    collect(source)
    entries = read_json(source[2] / "private/mapping.json")["entries"]
    public = {bound["item_sha256"]: bound for bound in _items(source[2])}
    positions = {}
    for entry in entries:
        positions.setdefault(entry["true_item"]["method_id"], set()).add(public[entry["blind_item_sha256"]]["method_id"])
    assert all(len(labels) > 1 for labels in positions.values())
    html = (source[2] / "public/index.html").read_text(encoding="utf-8")
    assert "NOT consistent method IDs" in html


def test_nonzero_method_difference_survives_case_specific_permutations_and_unblind_is_read_only(source, tmp_path):
    collect(source)
    mapping = read_json(source[2] / "private/mapping.json")
    real_methods = {entry["blind_item_sha256"]: entry["true_item"]["method_id"] for entry in mapping["entries"]}
    reviews = []
    for bound in _items(source[2]):
        review = _filled(bound)
        for rating in review["ratings"].values():
            rating["score"] = 1 if real_methods[bound["item_sha256"]] == "PRIVATE_METHOD_ALPHA" else 5
        reviews.append(review)
    path = tmp_path / "completed.json"
    write_json(path, reviews)
    before = {str(file): file_sha(file) for file in source[2].rglob("*") if file.is_file()}
    report = unblind_reviews(source[2], [path], method_a="PRIVATE_METHOD_ALPHA", method_b="PRIVATE_METHOD_BETA", n_resamples=20)
    pair = report["summary"]["paired_comparisons"]["human"]["action_clarity"]
    assert pair["estimate_group_equal"] == 4
    assert pair["paired_cases"] == pair["n_source_groups_observed"] == 3
    assert pair["interval_group_equal"] == [4, 4]
    assert before == {str(file): file_sha(file) for file in source[2].rglob("*") if file.is_file()}


def test_html_escapes_generated_instructions_and_script(source):
    danger = '<script>alert("no")</script><img src=x onerror=alert(1)>'
    _rewrite_run(source[0][0], lambda result: result["rows"][0].update(text=danger))
    collect(source)
    page = (source[2] / "public/index.html").read_text(encoding="utf-8")
    assert "<script>" not in page and "<img" not in page
    assert "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page


@pytest.mark.parametrize("field,value", [("source_group_id", "other-group"), ("split", "val")])
def test_cross_method_case_binding_mismatch_fails_before_publication(source, field, value):
    _rewrite_run(source[0][1], lambda result: result["rows"][0].update({field: value}),
                 lambda contract: contract["cases"][0].update({field: value}))
    with pytest.raises(BundleError, match="Methods differ"):
        collect(source)
    assert not source[2].exists()


def test_cross_method_audio_identity_mismatch_rejected(source):
    other = read_json(source[0][1])["rows"][1]["audio"]
    _rewrite_run(source[0][1], lambda result: result["rows"][0].update(audio=other, audio_sha256=other["sha256"]),
                 lambda contract: contract["cases"][0].update(audio=other))
    with pytest.raises(BundleError, match="Methods differ"):
        collect(source)


def test_omitted_registered_case_is_not_fabricated_or_silently_intersected(source):
    _rewrite_run(source[0][1], lambda result: result["rows"].pop())
    with pytest.raises(BundleError, match="omit"):
        collect(source)
    assert not source[2].exists()


def test_duplicate_method_rejected(source):
    with pytest.raises(BundleError, match="Duplicate method"):
        collect_review_bundle([source[0][0], source[0][0]], source[2])


def test_changed_audio_bytes_rejected_without_creating_output(source):
    audio = Path(read_json(source[0][0])["rows"][0]["audio"]["path"])
    raw = bytearray(audio.read_bytes()); raw[-1] ^= 1; audio.write_bytes(raw)
    with pytest.raises(BundleError, match="Actual audio SHA"):
        collect(source)
    assert not source[2].exists()


@pytest.mark.parametrize("tamper", ["result_seal", "row_seal", "records"])
def test_source_content_bindings_are_checked(source, tamper):
    path = source[0][0]
    if tamper == "records":
        (path.parent / "records.jsonl").write_text("{}\n", encoding="utf-8")
    else:
        result = read_json(path)
        if tamper == "result_seal":
            result["case_count"] += 1
        else:
            result["rows"][0]["text"] = "Changed without resealing the row"
            result = _reseal(result)
        write_json(path, result, overwrite=True)
    with pytest.raises(ValueError):
        collect(source)


@pytest.mark.parametrize("field", ["selection_sha256", "source_files", "action_reference"])
def test_reference_bindings_are_checked_even_when_only_kept_private(source, field):
    refs = read_json(source[1])
    if field == "action_reference":
        refs["records"][0][field] = "Mismatched text"
    else:
        refs[field] = "wrong"
    write_json(source[1], refs, overwrite=True)
    with pytest.raises(BundleError, match="Reference"):
        collect(source, references_path=source[1])
    assert not source[2].exists()


def test_reference_stage_stays_private_and_cannot_reassess_quality(source, tmp_path):
    manifest = collect(source, phase="reference", references_path=source[1])
    assert manifest["stage"] == "private/reference"
    assert not (source[2] / "public").exists()
    items = _items(source[2], "reference")
    assert all("SECRET_REFERENCE" in bound["reference_text"] and "Dance style:" not in bound["reference_text"] for bound in items)
    path = tmp_path / "reference-returned.json"
    reviews = [_filled(bound, reference=True) for bound in items]
    write_json(path, reviews)
    result = unblind_reviews(source[2], [path], n_resamples=10)
    method = result["summary"]["origins"]["human"]["methods"]["PRIVATE_METHOD_ALPHA"]
    assert method["reference_fidelity"]["estimate_group_equal"] == 2
    assert method["action_clarity"]["estimate_group_equal"] is None
    reviews[0]["ratings"]["action_clarity"] = {"status": "rated", "score": 5, "reason": "Illegitimate post-reference quality score"}
    write_json(path, reviews, overwrite=True)
    with pytest.raises(BundleError, match="cannot re-score"):
        unblind_reviews(source[2], [path], n_resamples=10)


@pytest.mark.parametrize("asset", ["items.json", "index.html", "audio_assets.json", "rubric.json"])
def test_unblind_rejects_changed_published_assets(source, asset):
    collect(source)
    path = source[2] / "public" / asset
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BundleError, match="asset changed"):
        unblind_reviews(source[2], [])


def test_extra_public_file_and_private_mapping_drift_are_rejected(source):
    collect(source)
    extra = source[2] / "public" / "unregistered.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError, match="inventory"):
        unblind_reviews(source[2], [])
    extra.unlink()
    path = source[2] / "private/mapping.json"
    mapping = read_json(path); mapping["seed"] += 1
    write_json(path, _reseal(mapping), overwrite=True)
    with pytest.raises(BundleError, match="mapping"):
        unblind_reviews(source[2], [])


def test_returned_review_unknown_item_duplicate_identity_or_bad_hash_is_rejected(source, tmp_path):
    collect(source)
    bound = _items(source[2])[0]
    path = tmp_path / "returned.json"
    rows = [_filled(bound), _filled(bound)]
    write_json(path, rows)
    with pytest.raises(BundleError, match="Duplicate"):
        unblind_reviews(source[2], [path], n_resamples=10)
    rows = [_filled(bound)]; rows[0]["reviewer_id"] = "not-registered"
    write_json(path, rows, overwrite=True)
    with pytest.raises(BundleError, match="roster"):
        unblind_reviews(source[2], [path], n_resamples=10)
    rows[0]["reviewer_id"] = "reviewer_1"; rows[0]["item_sha256"] = "f" * 64
    write_json(path, rows, overwrite=True)
    with pytest.raises(BundleError, match="unknown blind"):
        unblind_reviews(source[2], [path], n_resamples=10)


def test_collect_is_exclusive_and_reference_mode_requires_explicit_reference(source):
    with pytest.raises(BundleError, match="requires explicit"):
        collect(source, phase="reference")
    collect(source)
    before = {str(path): file_sha(path) for path in source[2].rglob("*") if path.is_file()}
    with pytest.raises(BundleError, match="new directory"):
        collect(source)
    assert before == {str(path): file_sha(path) for path in source[2].rglob("*") if path.is_file()}


def test_cli_collect_unblind_is_offline_and_private_output_not_public(source, tmp_path):
    args = ["collect", "--out-dir", str(source[2]), "--seed", "7"]
    for path in source[0]:
        args += ["--result", str(path)]
    main(args)
    out = tmp_path / "unblinded.private.json"
    result = main(["unblind", "--bundle", str(source[2]), "--out", str(out), "--resamples", "10"])
    assert result["summary"]["coverage"]["registered_items"] == 6
    assert read_json(out)["synthetic"]
    with pytest.raises(SystemExit):
        main(["unblind", "--bundle", str(source[2]), "--out", str(source[2] / "public" / "private-leak.json")])
    assert not (source[2] / "public" / "private-leak.json").exists()
