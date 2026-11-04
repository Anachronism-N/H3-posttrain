import json
from pathlib import Path

from choreobench.cli import main
from choreobench.io import file_sha, read_json, write_json


def test_complete_cli_smoke_review_submission_and_undo(tmp_path, capsys):
    root = tmp_path / "demo"
    assert main(["demo", "--out-dir", str(root), "--run"]) == 0
    assert read_json(root / "NOTICE.json")["benchmark_results"] is False
    dataset = str(root / "dataset.json")
    assert main(["validate-data", "--dataset", dataset]) == 0
    assert main(["validate-data", "--dataset", dataset, "--release-check"]) == 2
    assert main(["resources", "--dataset", dataset, "--config", str(root / "hierarchical.json")]) == 0
    for mode in ("concat", "global", "hierarchical"):
        assert main(["run", "--dataset", dataset, "--config", str(root / (mode + ".json")),
                     "--out-dir", str(root / ("run-" + mode)), "--resume"]) == 0
    restored = root / "restored.json"
    assert main(["undo-edit", "--plan", str(root / "source_plan.json"), "--after", str(root / "edited_plan.json"),
                 "--task", str(root / "edit_task.private.json"), "--patch", str(root / "example_patch.json"),
                 "--out", str(restored)]) == 0
    assert read_json(restored) == read_json(root / "source_plan.json")
    submission = root / "submission.json"
    assert main(["export-submission", "--run-dir", str(root / "run-concat"),
                 "--method-id", "concat", "--out", str(submission)]) == 0
    report = root / "report.json"
    assert main(["evaluate-submission", "--dataset", dataset, "--submission", str(submission),
                 "--split", "dev", "--out", str(report)]) == 0
    assert read_json(report)["selected_songs"] == 1
    assert read_json(report)["valid_plans"] == 1
    assert read_json(report)["music_quality"] is None
    assert read_json(report)["synthetic"] is True
    reviews = root / "review"
    assert main(["review-pack", "--dataset", dataset, "--split", "dev",
                 "--run", "concat=" + str(root / "run-concat"),
                 "--run", "hierarchical=" + str(root / "run-hierarchical"),
                 "--out-dir", str(reviews), "--copy-audio"]) == 0
    pack = read_json(reviews / "public" / "pack.json")
    assert len(pack["items"]) == 2
    for item in pack["items"]:
        audio = item["song"]["audio"]
        assert file_sha(reviews / "public" / audio["path"]) == audio["sha256"]
    assert "contract_sha256" not in json.dumps(pack)
    template = read_json(reviews / "public" / "reviews.template.json")
    # Blank template must never be mistaken for a quality result.
    for review in template["reviews"]:
        review["reviewer_id"] = "reviewer-1"
    write_json(reviews / "public" / "blank-filled-id.json", template)
    assert main(["review-import", "--public", str(reviews / "public" / "pack.json"),
                 "--private", str(reviews / "private" / "bindings.json"),
                 "--reviews", str(reviews / "public" / "blank-filled-id.json"),
                 "--out", str(reviews / "imported.json")]) == 0


def test_cli_missing_submission_predictions_remain_denominator(tmp_path, capsys):
    root = tmp_path / "demo"
    assert main(["demo", "--out-dir", str(root)]) == 0
    submission = {"schema": "choreobench.submission.v1", "method_id": "empty",
                  "dataset_sha256": file_sha(root / "dataset.json"), "synthetic": True, "predictions": []}
    write_json(root / "submission.json", submission)
    assert main(["evaluate-submission", "--dataset", str(root / "dataset.json"),
                 "--submission", str(root / "submission.json"), "--split", "dev",
                 "--out", str(root / "report.json")]) == 0
    report = read_json(root / "report.json")
    assert report["valid_plan_rate"] == 0
    assert report["missing_or_invalid_plans"] == 1
