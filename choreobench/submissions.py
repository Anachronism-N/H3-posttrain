"""Method-independent benchmark submission format and complete-denominator checks."""
import os
from pathlib import Path

from .dataset import release_report, verify_dataset_audio
from .evaluation import evaluate_plan
from .io import file_sha, object_sha, read_json, read_json_snapshot, write_json
from .schema import keys, text, validate_dataset, validate_plan


def export_submission(run_dir, output, method_id):
    from .baseline import _verify
    text(method_id, "method_id")
    root, output = Path(run_dir).resolve(), Path(output).resolve()
    run = _verify(read_json(root / "results.json"))
    contract = read_json(root / "contract.json")
    if run.get("contract_sha256") != object_sha(contract):
        raise ValueError("Run result/contract mismatch")
    rows = []
    for ref in run["plans"]:
        path = (root / ref["path"]).resolve()
        if file_sha(path) != ref["sha256"]:
            raise ValueError("Published plan identity changed")
        try:
            portable = Path(os.path.relpath(str(path), str(output.parent))).as_posix()
        except ValueError:
            portable = str(path)
        rows.append(dict(ref, path=portable))
    submission = {"schema": "choreobench.submission.v1", "method_id": method_id,
                  "dataset_sha256": contract["dataset_sha256"], "synthetic": contract["synthetic"],
                  "predictions": rows}
    write_json(output, submission)
    return submission


def evaluate_submission(dataset_path, submission_path, split="test", track="full_song"):
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    submission = read_json(submission_path)
    keys(submission, ("schema", "method_id", "dataset_sha256", "synthetic", "predictions"), where="submission")
    if submission["schema"] != "choreobench.submission.v1" or type(submission["synthetic"]) is not bool:
        raise ValueError("Invalid submission schema/synthetic declaration")
    text(submission["method_id"], "method_id")
    if submission["dataset_sha256"] != dataset_sha:
        raise ValueError("Submission belongs to a different dataset manifest")
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    selected = [song for song in dataset["songs"] if song["split"] == split]
    if not selected:
        raise ValueError("No songs in requested evaluation split")
    if not isinstance(submission["predictions"], list):
        raise ValueError("predictions must be a list")
    by_id, all_ids = {}, {song["song_id"] for song in dataset["songs"]}
    for ref in submission["predictions"]:
        keys(ref, ("song_id", "path", "sha256"), where="prediction reference")
        if ref["song_id"] not in all_ids or ref["song_id"] in by_id:
            raise ValueError("Unknown/duplicate prediction song_id")
        text(ref["path"], "prediction.path")
        by_id[ref["song_id"]] = ref
    rows = []
    for song in selected:
        sid = song["song_id"]
        if sid not in by_id:
            rows.append({"song_id": sid, "valid": False, "status": "missing", "diagnostics": None})
            continue
        try:
            ref = by_id[sid]
            path = Path(ref["path"]).expanduser()
            if not path.is_absolute():
                path = Path(submission_path).resolve().parent / path
            plan, plan_sha = read_json_snapshot(path)
            if plan_sha != ref["sha256"]:
                raise ValueError("Prediction bytes differ from submitted hash")
            validate_plan(plan, song)
            diagnostics = evaluate_plan(plan, song)
            rows.append({"song_id": sid, "valid": True, "status": "valid", "diagnostics": diagnostics})
        except (ValueError, OSError, TypeError, KeyError) as error:
            rows.append({"song_id": sid, "valid": False, "status": "invalid", "diagnostics": None,
                         "error": {"type": type(error).__name__, "message": str(error)}})
    valid = sum(row["valid"] for row in rows)
    eligibility = release_report(dataset, track)
    if submission["synthetic"]:
        eligibility["eligible_by_declared_contract"] = False
        eligibility["blockers"].append("Synthetic model/replay predictions are excluded from scientific benchmark results")
    return {"schema": "choreobench.submission_report.v1", "dataset_sha256": dataset_sha,
            "submission_sha256": object_sha(submission), "split": split, "track": track, "method_id": submission["method_id"],
            "selected_songs": len(rows), "valid_plans": valid, "valid_plan_rate": valid / len(rows),
            "missing_or_invalid_plans": len(rows) - valid, "songs": rows,
            "synthetic": dataset["synthetic"] or submission["synthetic"],
            "release_contract": eligibility, "music_quality": None, "human_quality": None,
            "quality_status": "pending_independent_review",
            "limitations": ["Validity is structural only, never musicality, physical feasibility or semantic edit success.",
                            "Missing and invalid outputs remain in the denominator.",
                            "External method identities and grouping declarations require organizer audit."]}
