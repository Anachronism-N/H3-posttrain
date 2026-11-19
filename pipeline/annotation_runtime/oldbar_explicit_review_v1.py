"""Frozen, independent old-bar claim audit; never a production/training terminal.

prepare-only decodes original RGB and validates native same-frame hand ROIs on
CPU. Model execution must be scheduled separately. AI errors are human-review
candidates, and even sampled pass cannot approve training or an original batch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REVIEW_LIB = Path(__file__).resolve().parent / "source_curated_review_v2"
sys.path.insert(0, str(REVIEW_LIB))
import ms_batch_review as audit
import ms_visual_batch as batch
import ms_caption_rewrite as core
import ms_bar_annotation as bar

SCHEMA = "oldbar_explicit_independent_review_v1"
PILOT_SHA256 = "98611de703011815aee9a586f500462565bbfb12d2a237af30486c231f4aaf17"
PILOT_IDS = {"hwrlCvOCYUKpljlV", "KMcaVVwQmswUfTuP", "qKgmlBaDdOeqvUZZ"}
FALSE_FLAGS = ("training_eligible", "semantic_approval", "human_review_completed",
               "formal_training", "formal_approval", "human_approved", "training_entry_updated")
NO_APPROVAL = {k: False for k in FALSE_FLAGS}
SIX_FILES = {"request.json", "binding.json", "response.raw.txt", "generation.json", "response.json", "final.json"}
need = audit.need


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checked_sha(value):
    need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "explicit full SHA256 required")
    return value


class References:
    """Hash every referenced file once per process; changed stats invalidate cache."""
    def __init__(self):
        self.files = {}

    def check(self, ref):
        need(isinstance(ref, dict) and isinstance(ref.get("path"), str), "file path/SHA reference required")
        path = Path(ref["path"])
        need(path.is_absolute(), "reference path must be absolute")
        expected = checked_sha(ref.get("sha256"))
        path = path.resolve(strict=True)
        key = str(path)
        before = batch.snapshot(path)
        if key not in self.files or self.files[key]["stat"] != before:
            self.files[key] = batch.file_identity(path)
        identity = self.files[key]
        need(identity["sha256"] == expected, "reference SHA mismatch: " + key)
        return identity

    def visit(self, value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.check(value)
            for child in value.values():
                self.visit(child)
        elif isinstance(value, list):
            for child in value:
                self.visit(child)

    def document(self, ref):
        self.check(ref)
        return audit.read_json(ref["path"])


def line_at(ref):
    need(type(ref.get("line")) is int and ref["line"] > 0, "one-based original line required")
    with Path(ref["path"]).open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            if index == ref["line"]:
                return raw, audit.strict_json(raw)
    raise ValueError("referenced line outside file")


def validate_claim(row):
    need(isinstance(row, dict) and row.get("schema") == "old_bar_joint_explicit_alternative_import_v1", "explicit old-bar import schema required")
    need(row.get("id") == row.get("source_id") and re.fullmatch(r"[A-Za-z0-9_-]+", row.get("id", "")), "safe unchanged original source ID required")
    audit.normalized_record(row)
    need(all(row.get(k) is False for k in FALSE_FLAGS), "candidate cannot claim training/human/semantic approval")
    need(row.get("independent_review_required") is True and row.get("batch_review_status") == "pending_independent_review", "candidate requires fresh independent review")
    need(row.get("population_status") == "same_curated_source_alternative_not_new_population", "import alternative is not new population")
    need(row.get("dedup_key") == {"source_id": row["id"], "split": row["split"]}, "dedup must bind original id/split")
    need(isinstance(row.get("edition_id"), str) and row["edition_id"], "explicit import edition required")
    need(isinstance(row.get("batch_id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", row["batch_id"]), "safe import batch ID required")
    need(row.get("original_curated_batch_id") != row["batch_id"], "import population cannot impersonate original curated batch")
    need(row.get("original_timed_motion_unchanged") is True and row.get("original_windows_unchanged") is True, "original MS/windows must remain unchanged")
    label = row.get("label")
    need(isinstance(label, dict) and set(label) == {"dance_style", "timed_motion_script"}, "only style/MS semantic claim fields allowed")
    need(all(isinstance(v, str) and v.strip() for v in label.values()), "nonempty style and verbatim timed MS required")
    need(row.get("dance_style_label") == label["dance_style"], "style claim mismatch")
    expected = "Dance style: " + label["dance_style"] + "\n" + label["timed_motion_script"]
    need(row.get("target_text") == expected and row.get("target_text_sha256") == text_sha(expected), "target text changed or SHA mismatch")
    need(row.get("original_timed_motion_sha256") == text_sha(label["timed_motion_script"]), "verbatim MS SHA mismatch")
    need(audit.label_payload(row) == label and audit.label_reviewable(row), "candidate label payload must be unchanged/reviewable")
    provenance = row.get("source_refs", {}).get("import_provenance", {})
    need(provenance.get("historical_video_sha256", "missing") is None and provenance.get("historical_video_binding") == "path_only_unverified", "current MS SHA cannot upgrade historical style binding")
    metadata = row.get("style_result", {}).get("metadata", {})
    need(metadata.get("historical_video_sha256", "missing") is None and metadata.get("historical_video_binding") == "path_only_unverified" and metadata.get("historical_video_binding_verified") is False, "historical style must remain path-only/unverified")
    need(metadata.get("source_refs", {}).get("historical_label_record") == provenance.get("historical_style_label_record"), "historical style reference changed")
    need(metadata.get("historical_video_path") == row["video_path"], "historical path-only style source path changed")
    for value in (row.get("style_result", {}), metadata):
        need(all(value.get(k) is False for k in ("semantic_approval", "human_review_completed", "formal_approval", "formal_training", "training_entry_updated")), "historical style cannot claim fresh approval")
    need(provenance.get("current_curated_batch", {}).get("batch_id") == row["original_curated_batch_id"], "original batch mapping changed")
    need(provenance.get("current_media_content_SHA_matches_old_MS") is True and provenance.get("current_actual_pts_matches_old_MS") is True, "explicit old/current MS binding attestations required")
    need(provenance.get("old_to_current_plan_equality_claimed") is False, "old/current plans are not equal")


def load_manifest(path, *, manifest_sha256=None, population_kind="three-source-pilot"):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = manifest_sha256 or (PILOT_SHA256 if population_kind == "three-source-pilot" else None)
    need(expected is not None and digest == checked_sha(expected), "frozen complete manifest SHA mismatch")
    rows = [audit.strict_json(line) for line in raw.splitlines() if line.strip()]
    need(rows, "nonempty frozen manifest required")
    keys = [(r.get("id"), r.get("split")) for r in rows]
    need(len(set(keys)) == len(keys), "duplicate (id,split) in frozen import manifest")
    for row in rows:
        validate_claim(row)
    need(len({r["edition_id"] for r in rows}) == 1 and len({r["batch_id"] for r in rows}) == 1 and len({r["split"] for r in rows}) == 1, "one frozen edition/import population/split required")
    if population_kind == "three-source-pilot":
        need(len(rows) == 3 and {r["id"] for r in rows} == PILOT_IDS and rows[0]["batch_id"] == "oldbar3-import-v1", "pilot population must be exactly the frozen three, not 500")
    else:
        need(population_kind == "complete500" and len(rows) == 500 and len({r["original_curated_batch_id"] for r in rows}) == 1, "complete500 requires 500 explicit claims from one original curated batch")
        refs = References()
        descriptor = rows[0]["source_refs"]["import_provenance"]["current_curated_batch"]
        refs.check(descriptor)
        members = [audit.strict_json(line) for line in Path(descriptor["path"]).read_bytes().splitlines() if line.strip()]
        need(len(members) == 500 and {(r["id"], r["split"]) for r in members} == set(keys), "complete500 cohort must equal authoritative original batch members")
        need(all(r["batch_id"] == rows[0]["original_curated_batch_id"] for r in members), "original full-cohort batch identity mismatch")
    return rows, {"path": str(path), "sha256": digest, "population_kind": population_kind,
                  "source_count": len(rows), "edition_id": rows[0]["edition_id"], "import_batch_id": rows[0]["batch_id"]}


def validate_source(row, refs, *, ffprobe):
    validate_claim(row)
    p = row["source_refs"]["import_provenance"]
    refs.visit(row["source_refs"])
    for name in ("visual_result_ref", "raw_rgb_manifest", "hand_roi"):
        refs.check(row[name])
    result = refs.document(p["old_result"])
    need(row["visual_result_ref"] == p["old_result"] and result["id"] == row["id"], "original visual result/source mismatch")
    need(row["source_binding"] == result["binding"] == refs.document(p["old_binding"]) == refs.document(p["old_prepared"])["binding"], "original source_binding mismatch")
    origin = Path(row["origin_root"]).resolve(strict=True)
    source = Path(row["source_hint"]).resolve(strict=True)
    need(source == origin / "sources" / row["id"], "real origin_root/source_hint mismatch")
    for name in ("old_source_identity", "old_actual_pts", "old_binding", "old_prepared", "old_prepared_receipt", "old_roi_gate", "old_raw_rgb_manifest"):
        need(Path(p[name]["path"]).resolve(strict=True).is_relative_to(source), "old reference outside actual source bundle: " + name)
    need(Path(p["old_result"]["path"]).resolve(strict=True).parent == source / "results", "visual result outside original result namespace")
    need(Path(p["old_completed_record"]["path"]).resolve(strict=True) == origin / "completed" / (row["id"] + ".json"), "completed source origin mismatch")
    completed = refs.document(p["old_completed_record"])
    need(completed.get("id") == row["id"] and completed.get("status") == "candidate", "old completion is not source-bound candidate")
    need(result["final_timed_caption"] == row["label"]["timed_motion_script"], "label differs from original MS verbatim")
    need(result["original_windows"] == p["old_original_windows"] and result["final"]["json"]["windows"] == p["old_normalized_final_windows"], "original timed windows changed")
    need(row["raw_rgb_manifest"] == result["raw_rgb_manifest"] == p["old_raw_rgb_manifest"] and row["hand_roi"] == result["hand_roi"] == p["old_roi_ref"], "original RGB/ROI references changed")
    raw, plan = line_at(p["old_plan"])
    binding = row["source_binding"]
    need(hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest() == p["old_plan"]["raw_line_sha256"] == binding["raw_line_sha256"], "original plan raw line SHA mismatch")
    need(batch.digest(plan) == p["old_plan"]["full_source_row_sha256"] == binding["full_source_row_sha256"], "original full source row SHA mismatch")
    need(batch.digest(plan["manifest_record"]) == binding["manifest_record_sha256"] and plan["manifest_record"] == result["manifest_record"], "original manifest record SHA mismatch")
    need(plan["id"] == row["id"] and plan["manifest_record"]["split"] == row["split"] and p["old_plan"]["sha256"] == binding["plan_sha256"], "old plan source/split/SHA mismatch")
    current_raw, current = line_at(p["current_curated_batch"])
    need(hashlib.sha256(current_raw).hexdigest() == p["current_curated_batch"]["raw_source_row_sha256"], "current original batch raw source row SHA mismatch")
    need(all(current[k] == row[k] for k in ("id", "split", "video_path", "audio_path")) and current["batch_id"] == row["original_curated_batch_id"], "current original membership/media mismatch")
    summary = refs.document(p["current_curated_summary"])
    descriptors = [d for d in summary["batches"] if d["batch_id"] == row["original_curated_batch_id"]]
    need(len(descriptors) == 1 and all(descriptors[0][k] == p["current_curated_batch"][k] for k in ("path", "sha256", "count", "split")), "authoritative original population descriptor mismatch")
    _, historical = line_at(p["historical_style_label_record"])
    need(historical["sid"] == row["id"] and batch.digest(historical) == p["historical_style_label_record"]["row_sha256"], "historical style source row SHA mismatch")
    need(historical["label"] == row["style_result"]["metadata"]["historical_label"] and historical["label"]["genre"] == row["label"]["dance_style"], "historical style claim changed")
    identity = refs.document(p["old_source_identity"])
    media = binding["source_binding"]
    for name, suffix in (("video_path", "video"), ("audio_path", "audio")):
        ref = p["current_" + suffix]
        need(row[name] == ref["path"] == identity["files"][name]["path"] == result["manifest_record"][name] and ref["sha256"] == media[suffix + "_sha256"] == identity["files"][name]["sha256"], "old/current media path/SHA mismatch: " + name)
        refs.check(ref)
    for ref in identity["files"].values():
        if ref is not None:
            refs.check(ref)
    metadata = row["style_result"]["metadata"]
    need(metadata.get("video_sha256") == metadata.get("current_video_sha256") == media["video_sha256"], "current style metadata media SHA mismatch")
    probe = bar.probe_actual(row["video_path"], ffprobe)
    previous = refs.document(p["old_actual_pts"])
    need(len(probe["times"]) == media["decoded_video_frames"] and probe["actual_video_pts_sha256"] == media["actual_video_pts_sha256"], "actual original video frame/PTS SHA mismatch")
    for name in ("times", "source_timestamps"):
        need(len(previous[name]) == len(probe[name]) and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-8) for a, b in zip(previous[name], probe[name])), "actual original PTS values changed")
    roi, roi_ref = bar.load_hand_roi({"id": row["id"], "binding": {"video_sha256": media["video_sha256"]}}, probe, row["hand_roi"]["path"])
    need(roi is not None and roi_ref["available"] is True and all(roi_ref[k] == row["hand_roi"][k] for k in ("path", "sha256", "source_manifest_sha256", "extract_receipt_sha256")), "native original ROI identity mismatch")
    return probe


def review_row(row, manifest_ref):
    # All producer scores, explanations, pose data, captions and historic metadata
    # are excluded from the model request. The immutable input digest binds them.
    return {"source_id": row["id"], "id": row["id"], "split": row["split"],
            "video_path": row["video_path"], "audio_path": row["audio_path"],
            "label": copy.deepcopy(row["label"]), "label_status": "completed",
            "hand_roi": copy.deepcopy(row["hand_roi"]),
            "annotator_model_sha256": row["annotator_model_sha256"],
            "explicit_review_binding": {"schema": SCHEMA, "manifest": manifest_ref,
                "import_candidate_sha256": batch.digest(row), "import_batch_id": row["batch_id"],
                "original_curated_batch_id": row["original_curated_batch_id"],
                "label_sha256": batch.digest(row["label"]), "target_text_sha256": row["target_text_sha256"],
                "source_binding_sha256": batch.digest(row["source_binding"]),
                "provenance_sha256": batch.digest(row["source_refs"]),
                "historical_video_sha256": None, "historical_video_binding": "path_only_unverified"}}


def verify_evidence(evidence, row, root, probe):
    need(evidence["binding"]["input_row_sha256"] == batch.digest(row) and evidence["binding"]["label_sha256"] == batch.digest(row["label"]), "native evidence input/label binding mismatch")
    need(evidence["probe"] == probe, "decoded evidence PTS differs from original video")
    indices = {f["index"] for f in evidence["frames"]}
    need(2 <= len(indices) == len(evidence["frames"]) <= 16, "native body frame count invalid")
    for frame in evidence["frames"]:
        path = (Path(root) / frame["file"]).resolve(strict=True)
        need(path.is_relative_to(Path(root).resolve()) and batch.sha256(path) == frame["sha256"], "native RGB path/SHA mismatch")
        need(frame["time"] == probe["times"][frame["index"]] and frame["source_timestamp"] == probe["source_timestamps"][frame["index"]], "decoded source frame actual PTS mismatch")
    for crop in evidence["hand_rois"]:
        need(crop["frame_index"] in indices and batch.sha256(Path(crop["path"])) == crop["sha256"], "ROI missing native same-frame body context/SHA")
    roi_binding = evidence["binding"]["hand_roi_ref"]
    need((roi_binding.get("path") == row["hand_roi"]["path"] or ("hands" not in audit.risk_flags(row) and roi_binding.get("reason") == "not_requested")) and len(evidence["hand_rois"]) <= 6, "wrong native ROI evidence binding/budget")


class ManifestReviewer(audit.LocalReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(__file__).resolve().parent
        paths = [Path(v["path"]) for v in self.code["files"].values()] + [Path(__file__).resolve()]
        self.code = batch.identity_tree(root, paths)

    def identity(self):
        identity = super().identity()
        if not getattr(self, "_checkpoint_content_rehashed", False):
            batch.verify_current(identity, model=True, rehash=True)
            need(self._annotator_identity is not None, "full annotator checkpoint reference required")
            self._checkpoint_content_rehashed = True
        return identity

    def runtime(self):
        if getattr(self, "_runtime_error", None) is not None:
            raise RuntimeError("shared reviewer load previously failed") from self._runtime_error
        try:
            return super().runtime()
        except Exception as error:
            self._runtime_error = error
            raise

    def __call__(self, row, root, *, prepare_only=False):
        self.hand_roi_root = row["hand_roi"]["path"]
        return super().__call__(row, root, prepare_only=prepare_only)


def verify_review_receipt(root, row, evidence, reviewer):
    transaction = Path(root) / "review"
    receipt = audit.read_json(transaction / "receipt.json")
    need(receipt["status"] == "candidate" and set(receipt["files_sha256"]) == SIX_FILES, "successful independent review requires all six receipt artifacts")
    for name in SIX_FILES:
        need(batch.sha256(transaction / name) == receipt["files_sha256"][name], "independent review receipt artifact SHA mismatch: " + name)
    binding = audit.read_json(transaction / "binding.json")
    need(binding == receipt["binding"], "independent receipt binding file mismatch")
    for key, expected in (("source_id", row["id"]), ("split", row["split"]),
                          ("label_file_ref", None), ("video_sha256", evidence["source_identity"]["files"]["video_path"]["sha256"]),
                          ("source_refs", evidence["binding"]["source_refs"]),
                          ("frame_indices", evidence["binding"]["frame_indices"]),
                          ("hand_roi_ref", evidence["binding"]["hand_roi_ref"]),
                          ("input_row_sha256", batch.digest(row)), ("label_sha256", batch.digest(row["label"])),
                          ("evidence_sha256", batch.sha256(Path(root) / "evidence.json")),
                          ("model_sha256", reviewer.identity()["sha256"]), ("code_sha256", reviewer.code["sha256"]),
                          ("reviewer_checkpoint_sha256", audit.checkpoint_identity(reviewer.identity()))):
        need(binding.get(key) == expected, "independent review binding mismatch: " + key)
    need(reviewer._annotator_identity is not None and reviewer._annotator_identity["sha256"] == row["annotator_model_sha256"], "source annotator must equal full original checkpoint inventory")
    need(reviewer._annotator_identity is not None and audit.checkpoint_identity(reviewer._annotator_identity) != audit.checkpoint_identity(reviewer.identity()), "annotator/reviewer checkpoint content must differ")
    need(binding.get("annotator_model_reference") == reviewer._annotator_reference_ref, "annotator full checkpoint reference mismatch")
    need(binding.get("checkpoint_content_independence_verified") is True and binding.get("annotator_model_sha256") == row["annotator_model_sha256"], "independent checkpoint attestation required")
    request = audit.read_json(transaction / "request.json")
    generation = core.validate_generation(transaction, request)
    need(generation.get("model_identity_sha256") == reviewer.identity()["sha256"], "generation checkpoint SHA differs from independent reviewer")
    need(request == audit.review_request(row, evidence, root, reviewer.model), "receipt request differs from original semantic-only/native evidence request")
    checked = audit.validate_review(audit.strict_json((transaction / "response.raw.txt").read_bytes()), evidence)
    final = audit.read_json(transaction / "final.json")
    need(final.get("status") == "candidate" and final.get("json") == checked == audit.read_json(transaction / "response.json"), "raw/validated/final independent review mismatch")
    return checked, batch.file_identity(transaction / "receipt.json")


def aggregate(rows, selected, results, manifest_ref, sampling, *, prepare_only):
    reviewed = [r for r in results.values() if r["status"] in ("pass", "confirmed_error", "uncertain")]
    errors = sum(r["status"] == "confirmed_error" for r in reviewed)
    complete = not prepare_only and len(reviewed) == len(selected)
    return {"schema": SCHEMA, "manifest": manifest_ref, "population_source_count": len(rows),
            "sample_count": sampling["policy"]["sample_count"], "risk_count": sampling["policy"]["risk_count"],
            "selected_source_ids": selected, "unsampled_source_ids": sorted({r["id"] for r in rows} - set(selected)),
            "sampled_reviewed_denominator": len(reviewed), "confirmed_error_count": errors,
            "sampled_candidate_error_rate": errors / len(reviewed) if reviewed else None,
            "random_reviewed_denominator": sum(s in results and results[s]["status"] in ("pass", "confirmed_error", "uncertain") for s in sampling["random_source_ids"]),
            "statuses": dict(Counter(r["status"] for r in results.values())),
            "pending_source_ids": [s for s in selected if results.get(s, {}).get("status") not in ("pass", "confirmed_error", "uncertain")],
            "independent_sample_review_completed": complete,
            "state": "prepared_pending_independent_review" if prepare_only else "sample_review_completed_pending_human" if complete else "pending",
            "GPU_model_load_performed": bool(not prepare_only and reviewed),
            "semantic_approval": False, "training_eligible": False, "eligible_for_training": False,
            "human_review_completed": False, "human_approved": False, "formal_approval": False,
            "formal_training": False, "training_entry_updated": False,
            "old_scores_reused": False, "original_curated_batches_completed": False,
            "original_curated_batch_mapping": {r["id"]: r["original_curated_batch_id"] for r in rows},
            "full_batch_semantic_pass_claimed": False, "statistical_population_error_estimate": False,
            "confirmed_errors_are_AI_candidates_not_ground_truth": True}


def execute(rows, manifest_ref, out, *, prepare_only=True, reviewer=None, ffprobe="/opt/ms_tools/ffprobe", ffmpeg="/usr/local/bin/ffmpeg", seed=0):
    out = Path(out).resolve()
    protected = [Path(manifest_ref["path"]).parent] + [Path(r["origin_root"]) for r in rows] + [Path(r["hand_roi"]["path"]).parent for r in rows]
    need(all(out != p.resolve() and not p.resolve().is_relative_to(out) and not out.is_relative_to(p.resolve()) for p in protected), "output overlaps frozen input/old source/ROI namespace")
    need(prepare_only or reviewer is not None, "run requires independently configured LocalReviewer")
    count, risk = (3, 0) if manifest_ref["population_kind"] == "three-source-pilot" else (25, 5)
    sampling = audit.sampling_plan(rows, manifest_ref["import_batch_id"], seed=seed, sample_count=count, risk_count=risk)
    binding = {"schema": SCHEMA, "manifest": manifest_ref, "sampling": sampling,
               "candidate_rows_sha256": [batch.digest(r) for r in rows], **NO_APPROVAL}
    out.mkdir(parents=True, exist_ok=True)
    core.save_json(out / "frozen.population.json", binding)
    refs, results = References(), {}
    by_id = {r["id"]: r for r in rows}
    target, rounds = count, []
    while True:
        selected = list(dict.fromkeys(sampling["random_order"][:target] + sampling["risk_source_ids"]))
        for sid in selected:
            if sid in results:
                continue
            candidate = by_id[sid]
            row = review_row(candidate, manifest_ref)
            source_root = out / "sources" / (candidate["split"] + "-" + sid) / batch.digest(row)[:24]
            base = {"source_id": sid, "split": candidate["split"], "import_batch_id": candidate["batch_id"],
                    "original_curated_batch_id": candidate["original_curated_batch_id"], "manifest_sha256": manifest_ref["sha256"],
                    "input_row_sha256": batch.digest(row), "import_candidate_sha256": batch.digest(candidate),
                    "label_sha256": batch.digest(candidate["label"]), "target_text_sha256": candidate["target_text_sha256"],
                    "historical_video_sha256": None, "historical_video_binding": "path_only_unverified",
                    "AI_candidate_not_ground_truth": True, "old_scores_reused": False, **NO_APPROVAL}
            try:
                probe = validate_source(candidate, refs, ffprobe=ffprobe)
                evidence = audit.prepare_evidence(row, source_root, ffprobe=ffprobe, ffmpeg=ffmpeg, hand_roi_root=candidate["hand_roi"]["path"])
                verify_evidence(evidence, row, source_root, probe)
                evidence_ref = batch.file_identity(source_root / "evidence.json")
                base.update({"evidence_ref": evidence_ref, "evidence_frames": len(evidence["frames"]), "native_hand_rois": len(evidence["hand_rois"]), "real_hand_roi_ref": candidate["hand_roi"]})
                if prepare_only:
                    result = {**base, "status": "prepared", "GPU_model_load_performed": False, "actual_VLM_calls": 0}
                    core.save_json(source_root / "explicit.prepare.receipt.json", result)
                else:
                    # Before dispatch, incomplete/tampered prior receipts are
                    # pending, never erased or treated as a fresh model request.
                    receipt_path = source_root / "review" / "receipt.json"
                    if receipt_path.exists():
                        need(audit.read_json(receipt_path).get("status") == "candidate", "prior failed/incomplete receipt remains pending; no redispatch")
                        checked, receipt = verify_review_receipt(source_root, row, evidence, reviewer)
                    else:
                        need(not (source_root / "review").exists() or not any((source_root / "review").iterdir()), "partial prior transaction remains pending; no redispatch")
                        raw_result = reviewer(row, source_root, prepare_only=False)
                        need(raw_result["status"] in ("pass", "confirmed_error", "uncertain"), "independent model transaction incomplete: " + str(raw_result.get("error", raw_result["status"])))
                        checked, receipt = verify_review_receipt(source_root, row, evidence, reviewer)
                    result = {**base, **checked, "review_receipt": receipt, **NO_APPROVAL}
                    core.save_json(source_root / "explicit.review.receipt.json", result)
                results[sid] = result
            except Exception as error:
                results[sid] = {**base, "status": "pending", "error": type(error).__name__ + ": " + str(error), "actual_VLM_calls": None if not prepare_only else 0}
        reviewed = [r for r in results.values() if r["status"] in ("pass", "confirmed_error", "uncertain")]
        errors = sum(r["status"] == "confirmed_error" for r in reviewed)
        rounds.append({"random_target": min(target, len(rows)), "sampled_reviewed_denominator": len(reviewed), "AI_error_candidates": errors})
        next_target = target if prepare_only else audit.next_sample_target(target, len(rows), errors, len(reviewed))
        if next_target == target:
            break
        target = next_target
    summary = aggregate(rows, selected, results, manifest_ref, sampling, prepare_only=prepare_only)
    summary["GPU_model_load_performed"] = bool(not prepare_only and getattr(reviewer, "_loaded", None) is not None)
    summary["model_execution_scheduled_externally"] = not prepare_only
    summary["rounds"] = rounds
    summary["sampling"] = sampling
    summary["processed_source_count"] = len(results)
    summary["audited_source_count"] = summary["sampled_reviewed_denominator"]
    summary["total_curated_source_count"] = 222939
    summary["original_curated_batch_completed"] = False
    summary["native_source_frames"] = sum(r.get("evidence_frames", 0) for r in results.values())
    summary["native_hand_rois"] = sum(r.get("native_hand_rois", 0) for r in results.values())
    batch.atomic_write(out / "independent.ledger.jsonl", b"".join((json.dumps(results[s], ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for s in selected))
    batch.atomic_write(out / "independent.summary.json", batch.json_bytes(summary))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256")
    p.add_argument("--out", required=True)
    p.add_argument("--population-kind", choices=("three-source-pilot", "complete500"), default="three-source-pilot")
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--run", action="store_true")
    p.add_argument("--model")
    p.add_argument("--model-reference")
    p.add_argument("--annotator-model-reference")
    p.add_argument("--ffprobe", default="/opt/ms_tools/ffprobe")
    p.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-pixels", type=int, default=49152)
    p.add_argument("--gpu-memory-utilization", type=float, default=.75)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    rows, manifest = load_manifest(args.manifest, manifest_sha256=args.manifest_sha256, population_kind=args.population_kind)
    reviewer = None
    if args.run:
        need(args.model and args.annotator_model_reference, "run requires --model and --annotator-model-reference; schedule GPU separately")
        reviewer = ManifestReviewer(args.model, args.out, model_reference=args.model_reference,
            annotator_model_reference=args.annotator_model_reference, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg,
            max_model_len=args.max_model_len, max_pixels=args.max_pixels, gpu_memory_utilization=args.gpu_memory_utilization)
        reviewer.identity()  # Verify checkpoint content independence before any model loading.
    summary = execute(rows, manifest, args.out, prepare_only=args.prepare_only, reviewer=reviewer,
                      ffprobe=args.ffprobe, ffmpeg=args.ffmpeg, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if (args.prepare_only and not summary["statuses"].get("pending")) or summary["independent_sample_review_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
