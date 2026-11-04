import copy
import json

from choreobench.cli import main
from choreobench.fixtures import create_demo
from choreobench.io import file_sha, read_json, write_json


def test_target_authoring_cli_keeps_private_goals_out_of_preview(tmp_path, capsys):
    root = tmp_path / "demo"
    create_demo(root)
    selector = root / "selector.json"
    write_json(selector, {"section_label": "chorus", "occurrence": 1, "body_scope": "upper"})
    constraints = root / "constraints.json"
    write_json(constraints, {"protected": [{"event_id": "*", "field": "start"}],
                             "semantic_goals": ["PRIVATE_REVIEW_CRITERION"]})
    common = ["--plan", str(root / "source_plan.json"), "--dataset", str(root / "dataset.json"),
              "--selector", str(selector)]
    preview = root / "preview.json"
    assert main(["preview-target"] + common + ["--out", str(preview)]) == 0
    assert read_json(preview)["allowed_event_ids"] == ["arm1"]
    output = root / "target"
    assert main(["target-edit"] + common + ["--task-id", "pilot-edit", "--instruction", "Raise the arm higher.",
                                             "--constraints", str(constraints), "--out-dir", str(output)]) == 0
    assert read_json(output / "task.private.json")["semantic_goals"] == ["PRIVATE_REVIEW_CRITERION"]
    assert "PRIVATE_REVIEW_CRITERION" not in (output / "task.public.json").read_text(encoding="utf-8")
    assert "PRIVATE_REVIEW_CRITERION" not in (output / "resolution.json").read_text(encoding="utf-8")
    # Crossing an existing event boundary is inspectable, but cannot authorize a task.
    write_json(selector, {"start": 1, "end": 96000, "track": "right_arm"}, overwrite=True)
    assert main(["preview-target"] + common + ["--out", str(root / "conflict.json")]) == 0
    assert not read_json(root / "conflict.json")["can_build_task"]
    assert main(["target-edit"] + common + ["--task-id", "crossing", "--instruction", "Change action.",
                                             "--out-dir", str(root / "forbidden")]) == 2
    assert not (root / "forbidden").exists()


def test_external_method_can_enter_same_blind_review_without_baseline_run(tmp_path, capsys):
    root = tmp_path / "demo"
    create_demo(root)
    submission = {"schema": "choreobench.submission.v1", "method_id": "independent-author-method",
                  "dataset_sha256": file_sha(root / "dataset.json"), "synthetic": True,
                  "predictions": [{"song_id": "demo-song", "path": "source_plan.json",
                                   "sha256": file_sha(root / "source_plan.json")}]}
    write_json(root / "external.json", submission)
    command = ["review-pack", "--dataset", str(root / "dataset.json"), "--split", "dev",
               "--submission", str(root / "external.json")]
    output = root / "blind"
    assert main(command + ["--out-dir", str(output), "--copy-audio"]) == 0
    pack = read_json(output / "public" / "pack.json")
    assert len(pack["items"]) == 1
    assert "independent-author-method" not in json.dumps(pack)
    assert "independent-author-method" in (output / "private" / "bindings.json").read_text(encoding="utf-8")
    wrong = copy.deepcopy(submission)
    wrong["predictions"] = []
    write_json(root / "external.json", wrong, overwrite=True)
    assert main(command + ["--out-dir", str(root / "missing")]) == 2
    assert not (root / "missing").exists()
    submission["predictions"][0]["sha256"] = "a" * 64
    write_json(root / "external.json", submission, overwrite=True)
    assert main(command + ["--out-dir", str(root / "changed")]) == 2


def test_paired_cli_reports_groups_and_never_silently_omits_missing_method(tmp_path, capsys):
    rows = [dict(song_id=song, group_id=group, method_id=method, value=value, reviewer_id="r1")
            for song, group in (("s1", "g1"), ("s2", "g1"), ("s3", "g2"))
            for method, value in (("a", 2), ("b", 3))]
    source = tmp_path / "ratings.json"
    write_json(source, rows)
    common = ["compare", "--rows", str(source), "--method-a", "a", "--method-b", "b", "--resamples", "100"]
    assert main(common + ["--out", str(tmp_path / "comparison.json")]) == 0
    report = read_json(tmp_path / "comparison.json")
    assert report["estimate"] == 1 and report["n_independent_groups"] == 2
    assert report["n_songs"] == 3
    write_json(source, rows[:-1], overwrite=True)
    assert main(common) == 2
    assert main(common + ["--allow-incomplete", "--out", str(tmp_path / "partial.json")]) == 0
    report = read_json(tmp_path / "partial.json")
    assert report["coverage"]["omitted_song_count"] == 1
    assert report["interval"] is None and report["n_independent_groups"] == 1


def test_candidate_cli_and_estimate_reuse_local_samples_without_local_provider(tmp_path, capsys, monkeypatch):
    from choreobench import baseline
    from choreobench.cli import resources
    root = tmp_path / "demo"
    create_demo(root)
    dataset = root / "dataset.json"
    assert main(["run", "--dataset", str(dataset), "--config", str(root / "concat.json"),
                 "--out-dir", str(root / "collection")]) == 0
    bank_path = root / "candidates.json"
    assert main(["export-candidates", "--run-dir", str(root / "collection"), "--out", str(bank_path)]) == 0
    config = read_json(root / "global.json")
    config.update(local_candidates_path=str(bank_path), max_calls=1)
    write_json(root / "reuse.json", config)
    estimate = resources(dataset, root / "reuse.json")
    assert estimate["expected_generation_calls"] == 1
    assert estimate["songs"][0]["new_local_calls"] == 0
    assert estimate["songs"][0]["reused_local_calls"] == 3
    original_create = baseline.create_provider
    def no_local(spec, role="local", base_dir=None):
        assert role != "local", "Reusing samples must not instantiate a local GPU model"
        return original_create(spec, role=role, base_dir=base_dir)
    monkeypatch.setattr(baseline, "create_provider", no_local)
    command = ["run", "--dataset", str(dataset), "--config", str(root / "reuse.json"),
               "--out-dir", str(root / "reused-global")]
    assert main(command) == 0
    result = read_json(root / "reused-global" / "results.json")
    assert result["calls_used"] == 1 and result["local_calls_reused"] == 3
    assert result["original_collection_calls"] == 3 and result["synthetic"] is True
    assert main(command + ["--resume"]) == 0
