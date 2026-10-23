#!/usr/bin/env python3
"""CPU-only, provenance-preserving sidecar; never edits the fusion writer outputs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import unittest
from datetime import datetime, timezone

VERSION = "fusion-music-motion-dimensions-v1"
FUSION_VERSION = "rich-ms-fusion6k-30b-weak-v2"
SOURCE_PLAN_VERSION = "rich-ms-fusion6k-30b-weak-v1"
KIN_FIELDS = ("speed", "arm_speed", "leg_speed", "torso_speed", "travel", "disp", "turn", "net_turn")
VLM_FIELDS = {
    "energy_vlm": "energy", "tempo_feel_visual": "tempo_feel",
    "body_focus_vlm": "body_focus", "quality_vlm": "quality",
    "level_vlm": "level", "travel_vlm": "travel",
}
FORBIDDEN_MOTION = re.compile(r"\b(?:audio|music|soundtrack|bpm)\b|\bbeats?\s+per\s+minute\b", re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "sha256": sha(data), "bytes": len(data)}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def require_version(value, expected, label):
    if value != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {value!r}")


def require_ids(index, ids, label):
    missing = sorted(set(ids) - set(index))
    if missing:
        raise ValueError(f"{label}: missing {len(missing)} train IDs: {missing[:10]}")


def stream_rows(path, ids, id_key, reduce_row=None):
    # Hash the aggregate and each physical JSONL row during the same sequential read.
    wanted = set(ids)
    index = {}
    h = hashlib.sha256()
    size = count = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            h.update(line)
            size += len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            sid = row.get(id_key)
            if sid not in wanted:
                continue
            if sid in index:
                raise ValueError(f"{path}: duplicate train ID {sid}")
            provenance = {"path": str(path), "line_number": line_number,
                          "row_sha256": sha(line), "row_hash_basis": "exact_physical_line_bytes_including_newline"}
            index[sid] = (reduce_row(row) if reduce_row else row, provenance)
            if reduce_row and len(index) % 500 == 0:
                print(json.dumps({"stage": "source_stream", "rows": len(index), "gpu_count": 0}), flush=True)
    aggregate = {"path": str(path), "sha256": h.hexdigest(), "bytes": size,
                 "jsonl_rows": count, "train_rows": len(index)}
    for _, provenance in index.values():
        provenance["aggregate_sha256"] = aggregate["sha256"]
    return index, aggregate


def reduce_source(row):
    return {"id": row.get("id"), "split": row.get("split"), "source_group": row.get("source_group"),
            "kin": (row.get("raw_MS") or {}).get("kin") or {},
            "raw_MS_sid": (row.get("raw_MS") or {}).get("sid"),
            "ms_source": (row.get("source_provenance") or {}).get("ms"),
            "raw_MS_text_sha256": row.get("raw_MS_text_sha256"),
            "actual_text_input_sha256": row.get("actual_text_input_sha256"),
            "window_mapping": row.get("window_mapping")}


def validate_audio(row, sid):
    if row.get("err") is not None or row.get("error") is not None:
        raise ValueError(f"audio feature error for {sid}: {row.get('err', row.get('error'))}")
    for key in ("bpm", "dur", "rms", "onset_rate", "centroid"):
        value = row.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"nonfinite/non-numeric audio feature {sid}/{key}")
    if row.get("bpm") is not None and row["bpm"] <= 0:
        raise ValueError(f"non-positive BPM for {sid}")


def make_context(sid, ref, source, audio, vlm, row_provenance, fusion_link, aggregates):
    validate_audio(audio, sid)
    label = vlm.get("label") or {}
    music = {
        "tempo_bpm_est": audio.get("bpm"), "duration_s": audio.get("dur"),
        "rms": audio.get("rms"), "onset_rate": audio.get("onset_rate"),
        "centroid_hz": audio.get("centroid"), "audio_window": [0, 5.0625],
        "confidence": "uncalibrated", "historical_audio_content_binding": "unverified",
        "feature_duration_basis": "legacy_whole_wav_resampled_22050hz_with_rounding",
        "audio_window_basis": "normalized_reference_protocol_not_media_revalidated",
        "units": {"tempo_bpm_est": "beats/minute", "duration_s": "seconds",
                  "rms": "audio_amplitude_rms_not_body_energy", "onset_rate": "onsets/second",
                  "centroid_hz": "Hz", "audio_window": "seconds"},
    }
    motion = {key: label.get(field) for key, field in VLM_FIELDS.items()}
    motion.update({"kin": {key: source["kin"].get(key) for key in KIN_FIELDS},
                   "kin_units": {key: "m/s" if "speed" in key else "degrees" if "turn" in key else "m" for key in KIN_FIELDS},
                   "window_mapping": "unknown", "vlm_evidence_type": "model_assessment_not_action_truth",
                   "teacher_supervision_only": True})
    return {
        "schema_version": VERSION, "id": sid, "source_group": ref["source_group"], "split": "train",
        "music_context": music, "motion_dims": motion,
        "fusion_target_link": fusion_link,
        "semantic_approval": False, "human_review_completed": False, "window_mapping": "unknown",
        "conditioning_policy": {"student_gt_visual_motion_dims_allowed": False,
                                "motion_dims_role": "teacher_supervision_only",
                                "music_context_role": "audio_derived_future_controlled_experiment_only_not_current_training"},
        "provenance": {
            "split": aggregates["split"], "references": aggregates["references"],
            "audio_features": row_provenance["audio"], "vlm": row_provenance["vlm"],
            "source_records": row_provenance["source"],
            "reference_audio_expected": {"path": ref.get("audio_path"), "expected_sha256": ref.get("audio_sha256"),
                                         "binding_status": "expected_reference_only_not_historical_feature_byte_verification",
                                         "media_rehashed": False},
            "raw_MS_original_source": source.get("ms_source"),
            "raw_MS_text_sha256": source.get("raw_MS_text_sha256"),
            "raw_MS_hash_verification": "recorded_source_hash_not_original_file_rehashed",
            "source_actual_text_input_sha256": source.get("actual_text_input_sha256"),
        },
    }


def motion_caption_risks(target):
    caption = (target or {}).get("motion_caption")
    if not isinstance(caption, str):
        return []
    return [match.group(0) for match in FORBIDDEN_MOTION.finditer(caption)]


def fusion_record_link(record, sid, group, path, data_sha, source_sha):
    require_version(record.get("caption_version"), FUSION_VERSION, f"fusion version {sid}")
    if record.get("id") != sid or record.get("split") != "train" or record.get("source_group") != group:
        raise ValueError(f"fusion identity mismatch: {sid}")
    if record.get("source_records_sha256") != source_sha:
        raise ValueError(f"fusion source_records hash mismatch: {sid}")
    caption = (record.get("target") or {}).get("motion_caption")
    return {
        "id": sid, "split": "train", "source_group": group, "caption_version": record.get("caption_version"),
        "path": str(path), "sha256": data_sha, "completion_status": record.get("completion_status"),
        "format_valid": record.get("format_valid"), "validation_errors": record.get("validation_errors"),
        "target": record.get("target"), "raw_generation": record.get("raw_generation"),
        "motion_caption_sha256": sha(caption.encode()) if isinstance(caption, str) else None,
        "motion_caption_forbidden_matches": motion_caption_risks(record.get("target")),
        "original_semantic_approval": record.get("semantic_approval"),
        "original_human_review_completed": record.get("human_review_completed"),
        "original_review_status": record.get("review_status"),
        "semantic_approval": False, "human_review_completed": False, "window_mapping": "unknown",
        "source_records_sha256": record.get("source_records_sha256"),
        "raw_source_text_input_sha256": record.get("raw_source_text_input_sha256"),
        "actual_text_input_sha256": record.get("actual_text_input_sha256"),
        "body_preservation": "target and raw_generation retained without text edits",
    }


def build(args):
    ms = args.ms.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    split, split_meta = read_json(ms / "music_direct6k_v2/split.json")
    ids = split["train"]
    if len(ids) != 6000 or len(set(ids)) != 6000:
        raise ValueError(f"expected 6000 unique train IDs, got {len(ids)}/{len(set(ids))}")
    references, refs_meta = read_json(ms / "music_direct6k_v2/references.json")
    refs = references["records"]
    require_ids(refs, ids, "references")
    source_plan, source_plan_meta = read_json(ms / "dance_caption_fusion6k_source_v2/source_plan.json")
    require_version(source_plan.get("version"), SOURCE_PLAN_VERSION, "source_plan version")
    run, run_meta = read_json(ms / "dance_caption_fusion6k_rewrite_v2/run_started.json")
    require_version(run.get("version"), FUSION_VERSION, "run version")
    sources, source_meta = stream_rows(ms / "dance_caption_fusion6k_source_v2/records.jsonl", ids, "id", reduce_source)
    require_ids(sources, ids, "source_records")
    if run.get("source_records_sha256") != source_meta["sha256"]:
        raise ValueError("run/source_records SHA mismatch")
    for key, actual in (("split", split_meta), ("references", refs_meta)):
        expected = source_plan.get("provenance", {}).get(key, {}).get("sha256")
        if expected != actual["sha256"]:
            raise ValueError(f"source_plan/{key} SHA mismatch")
    audio, audio_meta = stream_rows(ms / "audio_feats.jsonl", ids, "sid")
    require_ids(audio, ids, "audio_features")
    vlm = {}
    vlm_metas = []
    for shard in range(4):
        rows, meta = stream_rows(ms / f"vlm/labels_s{shard}.jsonl", ids, "sid")
        overlap = set(rows) & set(vlm)
        if overlap:
            raise ValueError(f"duplicate VLM IDs across shards: {sorted(overlap)[:10]}")
        vlm.update(rows)
        vlm_metas.append(meta)
    require_ids(vlm, ids, "vlm")
    for sid in ids:
        source = sources[sid][0]
        if source["split"] != "train" or source["source_group"] != refs[sid]["source_group"] or source["raw_MS_sid"] != sid:
            raise ValueError(f"source/reference identity mismatch: {sid}")
        validate_audio(audio[sid][0], sid)
    print(json.dumps({"stage": "aggregates_joined", "context_rows": len(ids), "gpu_count": 0}), flush=True)
    aggregates = {"split": split_meta, "references": refs_meta, "source_plan": source_plan_meta,
                  "source_records": source_meta, "audio_features": audio_meta, "vlm_shards": vlm_metas, "fusion_run_started": run_meta}
    snapshot_started = now()
    fusion_dir = ms / "dance_caption_fusion6k_rewrite_v2"
    # Directory membership is frozen once; only completed files in this snapshot are opened.
    names = {entry.name for entry in os.scandir(fusion_dir) if entry.name.endswith(".json")}
    links = {}
    unreadable = []
    for sid in ids:
        name = sid + ".json"
        if name not in names:
            continue
        path = fusion_dir / name
        try:
            data = path.read_bytes()
            record = json.loads(data)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            unreadable.append({"id": sid, "path": str(path), "reason": type(exc).__name__})
            continue
        links[sid] = fusion_record_link(record, sid, refs[sid]["source_group"], path, sha(data), source_meta["sha256"])
    snapshot_ended = now()
    out.mkdir(parents=True, exist_ok=False)
    schema = {
        "schema_version": VERSION, "gpu_count": 0, "current_training_modified": False,
        "audio_field_mapping": {"tempo_bpm_est": "bpm", "duration_s": "dur", "rms": "rms", "onset_rate": "onset_rate", "centroid_hz": "centroid"},
        "vlm_field_mapping": VLM_FIELDS, "MS_kin_fields": list(KIN_FIELDS),
        "semantic_approval": False, "human_review_completed": False, "window_mapping": "unknown",
        "rules": ["RMS is audio amplitude, never body energy.",
                  "Visual tempo_feel is a VLM opinion, not BPM; fast music can accompany slow motion.",
                  "No beat phase extrapolation or one-action-per-beat assumption.",
                  "GT visual motion_dims are teacher-only; never supply them to a music-only student.",
                  "Deployment-consistent audio-derived BPM may be tested as input in a separate controlled future experiment only.",
                  "Existing N7 training and N1 captions are unchanged.",
                  "No props, sub_style, key_moves or entire VLM scene labels are injected.",
                  "Missing dimensions are null, not false or new guessed labels.",
                  "Original caption text and failed/candidate status are never rewritten.",
                  "Historical feature/audio byte binding is unverified; expected audio hash is from references only."],
        "forbidden_check_scope": "target.motion_caption only, never context JSON",
        "forbidden_motion_regex": FORBIDDEN_MOTION.pattern,
        "row_hash_basis": "exact_physical_JSONL_line_bytes_including_newline",
    }
    schema_path = out / "schema.json"
    schema_path.write_bytes(json_bytes(schema))
    link_file = out / "fusion_links_snapshot.jsonl"
    link_rows = {}
    with link_file.open("xb") as stream:
        for sid in ids:
            if sid not in links:
                continue
            line = json_bytes(links[sid])
            stream.write(line)
            link_rows[sid] = {"path": str(link_file), "line_number": len(link_rows) + 1, "row_sha256": sha(line)}
    link_sha = file_sha(link_file)
    context_file = out / "context_rows.jsonl"
    missing_values = {"music_context." + k: 0 for k in ("tempo_bpm_est", "duration_s", "rms", "onset_rate", "centroid_hz")}
    missing_values.update({"motion_dims." + k: 0 for k in VLM_FIELDS})
    missing_values.update({"motion_dims.kin." + k: 0 for k in KIN_FIELDS})
    examples = []
    statuses = {}
    forbidden_count = 0
    candidate_target_count = 0
    with context_file.open("xb") as stream:
        for number, sid in enumerate(ids, 1):
            link = links.get(sid)
            if link is None:
                summary = {"status": "pending_fusion", "snapshot_started_UTC": snapshot_started,
                           "snapshot_ended_UTC": snapshot_ended,
                           "reason": "not_observed_as_complete_JSON_in_snapshot"}
            else:
                summary = {"status": link["completion_status"], "path": link["path"], "sha256": link["sha256"],
                           "motion_caption_sha256": link["motion_caption_sha256"],
                           "processed_link_record": {**link_rows[sid], "aggregate_sha256": link_sha},
                           "snapshot_started_UTC": snapshot_started, "snapshot_ended_UTC": snapshot_ended}
                forbidden_count += bool(link["motion_caption_forbidden_matches"])
                candidate_target_count += (link["completion_status"] == "candidate" and isinstance((link["target"] or {}).get("motion_caption"), str))
            statuses[summary["status"] or "null"] = statuses.get(summary["status"] or "null", 0) + 1
            context = make_context(sid, refs[sid], sources[sid][0], audio[sid][0], vlm[sid][0],
                                   {"source": sources[sid][1], "audio": audio[sid][1], "vlm": vlm[sid][1]}, summary, aggregates)
            stream.write(json_bytes(context))
            for field in missing_values:
                value = context
                for part in field.split("."):
                    value = value[part]
                missing_values[field] += value is None
            if number <= 3:
                examples.append({"id": sid, "source_group": context["source_group"],
                                 "music_context": context["music_context"], "motion_dims": context["motion_dims"],
                                 "fusion_target_link": summary,
                                 "example_music_context_text_not_injected_into_caption":
                                     f"Estimated audio tempo: {audio[sid][0].get('bpm'):.4f} BPM (uncalibrated; does not determine motion speed)."})
            if number % 500 == 0:
                print(json.dumps({"stage": "context_written", "rows": number, "linked_records": len(links), "gpu_count": 0}), flush=True)
    examples_path = out / "examples3.json"
    examples_path.write_bytes(json_bytes(examples))
    manifest = {
        "schema_version": VERSION, "created_UTC": now(), "host": socket.gethostname(), "gpu_count": 0,
        "script": {"path": str(Path(__file__).resolve()), "sha256": file_sha(Path(__file__))},
        "source_aggregates": aggregates,
        "counts": {"train_ids": len(ids), "context_rows": len(ids), "unique_context_ids": len(set(ids)),
                   "source_group_aligned": len(ids), "split_aligned": len(ids),
                   "fusion_records_linked": len(links), "candidate_targets_linked": candidate_target_count,
                   "pending_fusion": len(ids) - len(links), "snapshot_unreadable_files": len(unreadable),
                   "linked_caption_forbidden_risk_count": forbidden_count},
        "fusion_status_counts": statuses, "missing_dimension_counts": missing_values,
        "fusion_snapshot": {"started_UTC": snapshot_started, "ended_UTC": snapshot_ended,
                            "semantics": "single directory snapshot plus immutable byte-hashed files read; running writer continues independently",
                            "unreadable_files": unreadable},
        "outputs": [{"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}
                    for path in (schema_path, context_file, link_file, examples_path)],
        "semantic_approval": False, "human_review_completed": False, "window_mapping": "unknown",
        "media_files_read": 0, "source_per_ID_JSON_files_read": 0, "GPU_or_LLM_used": False,
        "current_training_or_fusion_writer_changed": False,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_bytes(json_bytes(manifest))
    print(json.dumps({"stage": "complete", "manifest": str(manifest_path), "manifest_sha256": file_sha(manifest_path),
                      "script_sha256": manifest["script"]["sha256"], **manifest["counts"]}), flush=True)


class SidecarSchemaTests(unittest.TestCase):
    def test_missing_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            require_ids({"A": {}}, ["A", "B"], "fixture")

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_bytes(b'{"sid":"A"}\n{"sid":"A"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                stream_rows(path, ["A"], "sid")

    def test_audio_error_rejected(self):
        with self.assertRaisesRegex(ValueError, "audio feature error"):
            validate_audio({"bpm": 90, "err": "decode_failure"}, "A")

    def test_wrong_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            require_version("wrong-version", FUSION_VERSION, "fixture")

    def test_tempo_visual_and_missing_null(self):
        row = make_context("A", {"source_group": "A"}, {"kin": {}}, {"bpm": 151.99908088235293, "rms": 0.99},
                           {"label": {"energy": "low", "tempo_feel": "slow"}},
                           {"audio": {}, "vlm": {}, "source": {}}, {"status": "pending_fusion"},
                           {"split": {}, "references": {}})
        self.assertAlmostEqual(row["music_context"]["tempo_bpm_est"], 151.99908088235293)
        self.assertEqual(row["motion_dims"]["tempo_feel_visual"], "slow")
        self.assertEqual(row["motion_dims"]["energy_vlm"], "low")
        self.assertIsNone(row["motion_dims"]["body_focus_vlm"])
        self.assertIsNone(row["motion_dims"]["kin"]["turn"])
        self.assertFalse(row["conditioning_policy"]["student_gt_visual_motion_dims_allowed"])

    def test_forbidden_only_motion_caption(self):
        context = {"music_context": {"tempo_bpm_est": 90, "audio_window": [0, 5.0625]},
                   "motion_caption": "The dancer bends both elbows with rhythmic control."}
        self.assertEqual(motion_caption_risks(context), [])
        self.assertEqual(motion_caption_risks({"motion_caption": "Dance to music at 90 BPM."}), ["music", "BPM"])

    def test_failed_and_caption_bytes_preserved(self):
        caption = "The dancer moves to music.  Keep original spaces."
        original = {"id": "A", "split": "train", "source_group": "A", "caption_version": FUSION_VERSION,
                    "source_records_sha256": "fixture-sha", "completion_status": "failed",
                    "format_valid": False, "validation_errors": ["original-error"],
                    "target": {"motion_caption": caption}, "raw_generation": "exact original raw"}
        before = copy.deepcopy(original)
        link = fusion_record_link(original, "A", "A", Path("A.json"), "file-sha", "fixture-sha")
        self.assertEqual(original, before)
        self.assertEqual(link["completion_status"], "failed")
        self.assertEqual(link["target"]["motion_caption"], caption)
        self.assertEqual(link["motion_caption_sha256"], sha(caption.encode()))
        self.assertEqual(link["motion_caption_forbidden_matches"], ["music"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SidecarSchemaTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.ms is None or args.out is None:
        parser.error("--ms and --out are required unless --self-test")
    build(args)


if __name__ == "__main__":
    main()
