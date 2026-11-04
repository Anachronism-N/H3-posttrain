"""Method-independent benchmark submission format and complete-denominator checks."""
import os
from pathlib import Path

from .dataset import release_report, resolve_audio, verify_dataset_audio
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


def _references(dataset, dataset_sha, submission):
    keys(submission, ("schema", "method_id", "dataset_sha256", "synthetic", "predictions"), where="submission")
    if submission["schema"] != "choreobench.submission.v1" or type(submission["synthetic"]) is not bool:
        raise ValueError("Invalid submission schema/synthetic declaration")
    text(submission["method_id"], "method_id")
    if submission["dataset_sha256"] != dataset_sha:
        raise ValueError("Submission belongs to a different dataset manifest")
    if not isinstance(submission["predictions"], list):
        raise ValueError("predictions must be a list")
    by_id, all_ids = {}, {song["song_id"] for song in dataset["songs"]}
    for ref in submission["predictions"]:
        keys(ref, ("song_id", "path", "sha256"), where="prediction reference")
        if ref["song_id"] not in all_ids or ref["song_id"] in by_id:
            raise ValueError("Unknown/duplicate prediction song_id")
        text(ref["path"], "prediction.path")
        by_id[ref["song_id"]] = ref
    return by_id


def review_submission_items(dataset_path, submission_path, *, split="test", song_ids=None):
    """Load independent method plans for the same blinded review protocol.

    Missing or invalid plans abort package creation; submission evaluation is the
    separate place to report failures in the full requested denominator.
    """
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    submission, submission_sha = read_json_snapshot(submission_path)
    refs = _references(dataset, dataset_sha, submission)
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    selected = [song for song in dataset["songs"] if song["split"] == split
                and (song_ids is None or song["song_id"] in song_ids)]
    if not selected or (song_ids is not None and set(song_ids) != {s["song_id"] for s in selected}):
        raise ValueError("Missing or out-of-split review songs")
    items = []
    for song in selected:
        if song["song_id"] not in refs:
            raise ValueError("Missing review plan; report failure in submission evaluation")
        ref = refs[song["song_id"]]
        path = Path(ref["path"]).expanduser()
        if not path.is_absolute():
            path = Path(submission_path).resolve().parent / path
        plan, sha = read_json_snapshot(path)
        if sha != ref["sha256"]:
            raise ValueError("Review prediction bytes differ from submitted hash")
        validate_plan(plan, song)
        song = dict(song, audio=dict(song["audio"], path=str(resolve_audio(song, dataset_path))))
        items.append({"song": song, "plan": plan, "method_id": submission["method_id"],
                      "source_binding": {"kind": "submission", "sha256": submission_sha,
                                         "dataset_sha256": dataset_sha,
                                         "synthetic": dataset["synthetic"] or submission["synthetic"]}})
    return items


def evaluate_submission(dataset_path, submission_path, split="test", track="full_song"):
    dataset, dataset_sha = read_json_snapshot(dataset_path)
    validate_dataset(dataset)
    verify_dataset_audio(dataset, dataset_path)
    submission = read_json(submission_path)
    by_id = _references(dataset, dataset_sha, submission)
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    selected = [song for song in dataset["songs"] if song["split"] == split]
    if not selected:
        raise ValueError("No songs in requested evaluation split")
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
