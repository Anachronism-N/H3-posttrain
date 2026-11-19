"""Independent sampled source-video review. AI errors are human-review candidates.

prepare_batches is CPU-only: it never loads a processor/model or labels unsampled
sources as semantically passed. run uses native local vLLM/xgrammar transactions.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
from types import SimpleNamespace

import ms_visual_batch as batch
import ms_caption_rewrite as core
import ms_bar_annotation as bar
from ms_visual_observe import strict_json

SCHEMA = "ms_independent_batch_review_v1"
ERROR_TYPES = ("incorrectstyle", "missingmajor", "hallucination", "hands", "left_right", "counts", "boundary")
SYSTEM = """You are an independent visual auditor of Dance style and timedMotionScript.
Use ONLY the supplied original-video RGB frames and same-frame native hand ROIs.
The label is a claim to test, NOT evidence. Do not use the annotator's reasoning,
confidence, scores, original caption, music genre, filenames or prompt as truth.
Check major body motion, hallucination, anatomical left/right, counts, hands,
and unsupported action boundaries. A time window is NOT an action boundary.
A confirmed error needs at least TWO DISTINCT supplied original source frames
and explicit positive visual evidence: a visible contradiction, or major visible
motion omitted from the label. NOT seeing an action between sparse samples is
NOT evidence that it never happened. Absence-only, blurred hands, one-frame
facts, uncertain anatomical side and unobservable counts must be uncertain.
ROIs from one frame never count as extra temporal support. Unknown style is not
a confirmed style error merely because sparse samples do not establish it.
Use status pass only for no supported error in these samples; this is not a
full-video or human semantic approval. confirmed_error is only an AI candidate
for human checking. For missingmajor claimed_text may identify the affected
label window rather than invent an omitted claim. Return ONLY schema JSON.
"""
POLICY = {"sample_count": 25, "risk_count": 5, "error_threshold": .10,
          "expansion_counts": [50, 100], "engineering_defaults_not_user_requirements": True,
          "statistical_population_error_estimate": False}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return strict_json(Path(path).read_bytes())


def source_id(row):
    sid = row.get("source_id", row.get("id"))
    need(isinstance(sid, str) and sid.strip(), "source_id/id must be a nonempty string")
    return sid


def normalized_record(row):
    value = copy.deepcopy(row)
    original = value.get("manifest_record", value)
    value["source_id"] = source_id(value)
    value["split"] = original.get("split", value.get("split"))
    need(value["split"] in ("train", "val", "validation", "test"), "explicit source split required")
    for key in ("video_path", "audio_path", "original_caption"):
        if key not in value and key in original:
            value[key] = original[key]
    need(isinstance(value.get("video_path"), str) and Path(value["video_path"]).is_absolute(),
         "explicit absolute original video_path required")
    return value


def label_payload(row):
    """Only semantic label text, never producer scores/reasoning, enters review."""
    if "label" in row:
        label = row["label"]
    elif row.get("label_path"):
        label = read_json(row["label_path"])
    elif "dance_style_label" in row or "target_text" in row:
        label = {"dance_style": row.get("dance_style_label", ""), "timed_motion_script": row.get("target_text", "")}
    else:
        return None
    if isinstance(label, str):
        return {"dance_style": row.get("dance_style", ""), "timed_motion_script": label}
    need(isinstance(label, dict), "label must be a text or object")
    style = label.get("dance_style", label.get("Dance style", label.get("style_text", "")))
    motion = label.get("timed_motion_script", label.get("timedMotionScript", label.get("motion_script", "")))
    if isinstance(style, dict):
        style = next((style[k] for k in ("dance_style", "style", "style_name", "style_text", "label", "primary_style")
                      if isinstance(style.get(k), str)), "")
    if not motion and "windows" in label:
        motion = [{k: w[k] for k in ("window_id", "start", "end", "caption", "cleaned_motion_text") if k in w}
                  for w in label["windows"]]
    if not motion:
        motion = label.get("timed_motion_caption", label.get("motion_caption", ""))
    return {"dance_style": style, "timed_motion_script": motion}


def label_complete(row):
    payload = label_payload(row)
    style = row.get("dance_style_label")
    statuses = (row.get("label_status"), row.get("motion_status"), row.get("visual_status"),
                row.get("style_status"), style.get("status") if isinstance(style, dict) else None)
    pending = ("failed", "partial", "partially_completed", "pending", "prepared", "needs_rgb",
               "needs_style", "labeling", "unknown", "abstain")
    return bool(payload and str(payload["dance_style"]).strip().lower() not in ("", "unknown")
                and payload["dance_style"] and payload["timed_motion_script"]
                and row.get("joint_complete") is not False and row.get("all_windows_nonempty") is not False
                and all(str(status).strip().lower() not in pending for status in statuses))


def label_reviewable(row):
    payload = label_payload(row)
    style = row.get("dance_style_label")
    style_status = style.get("status") if isinstance(style, dict) else row.get("style_status")
    unfinished = ("pending", "prepared", "needs_rgb", "needs_style", "labeling")
    return bool(payload and payload["dance_style"] and payload["timed_motion_script"]
                and row.get("label_status") not in unfinished and row.get("motion_status") not in unfinished
                and style_status not in unfinished)


def risk_flags(row):
    payload = label_payload(row)
    text = json.dumps(payload, ensure_ascii=False) if payload else ""
    flags = []
    for name, pattern in (("hands", r"\b(hand|finger|palm|fist)s?\b|手|指"),
                          ("left_right", r"\b(left|right)\b|左|右"),
                          ("counts", r"\b(twice|thrice|[2-9] times|two|three|four)\b|次"),
                          ("boundary", r"\b(start|stop|begin|finish)s?\b|开始|停止")):
        if re.search(pattern, text, re.I):
            flags.append(name)
    if not payload or not payload["dance_style"] or not payload["timed_motion_script"]:
        flags.append("incomplete_label")
    return flags


def sampling_plan(records, batch_id, *, seed=0, sample_count=25, risk_count=5):
    need(type(sample_count) is int and sample_count > 0 and type(risk_count) is int and risk_count >= 0,
         "positive sample_count and nonnegative risk_count required")
    ids = [source_id(r) for r in records]
    need(len(ids) == len(set(ids)), "duplicate source IDs in batch")
    order = sorted(ids)
    rng = random.Random(int(batch.digest({"seed": seed, "batch_id": batch_id})[:16], 16))
    rng.shuffle(order)
    by_id = {source_id(r): r for r in records}
    risk = sorted((sid for sid in ids if risk_flags(by_id[sid])),
                  key=lambda sid: (-len(risk_flags(by_id[sid])), batch.digest({"seed": seed, "id": sid})))
    selected = order[:sample_count]
    extras = [sid for sid in risk if sid not in selected][:risk_count]
    return {"seed": seed, "random_order": order, "random_source_ids": selected,
            "risk_source_ids": extras, "selected_source_ids": selected + extras,
            "risk_flags": {sid: risk_flags(by_id[sid]) for sid in extras},
            "policy": {**POLICY, "sample_count": sample_count, "risk_count": risk_count}}


def prepare_batches(records, out=None, *, batch_size=500, seed=0, sample_count=25, risk_count=5):
    need(type(batch_size) is int and batch_size > 0, "positive batch_size required")
    groups, seen = defaultdict(list), set()
    for raw in records:
        row = normalized_record(raw)
        sid = source_id(row)
        need(sid not in seen, "source ID cannot recur across splits/batches")
        seen.add(sid)
        groups[row["split"]].append(row)
    plans = []
    for split in sorted(groups):
        rows = sorted(groups[split], key=source_id)
        random.Random(int(batch.digest({"seed": seed, "split": split})[:16], 16)).shuffle(rows)
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset:offset + batch_size]
            bid = f"{split}-{offset // batch_size:05d}"
            plans.append({"schema": SCHEMA, "batch_id": bid, "split": split, "records": chunk,
                          "source_ids": [source_id(r) for r in chunk], "source_count": len(chunk),
                          "sampling": sampling_plan(chunk, bid, seed=seed, sample_count=sample_count, risk_count=risk_count),
                          "semantic_review_completed": False, "human_review_completed": False})
    if out is not None:
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        core.save_once(path / "batch-list.jsonl", b"".join(batch.json_bytes(p) + b"\n" for p in plans))
        core.save_json(path / "prepare.receipt.json", {"schema": SCHEMA, "source_count": len(seen),
            "batch_count": len(plans), "batch_list_sha256": batch.sha256(path / "batch-list.jsonl"),
            "GPU_model_load_performed": False, "semantic_review_completed": False})
    return plans


def review_schema(indices):
    need(2 <= len(indices) <= 16 and indices == sorted(set(indices)) and all(type(i) is int for i in indices),
         "two to sixteen distinct ordered actual frame indices required")
    evidence = {"type": "array", "items": {"type": "integer", "enum": indices}, "maxItems": 16}
    issue = {"type": "object", "additionalProperties": False, "properties": {
        "claimed_text": {"type": "string"}, "actual_visual_evidence": {"type": "string", "minLength": 1},
        "frame_indices": evidence, "error_type": {"type": "string", "enum": list(ERROR_TYPES)},
        "severity": {"type": "string", "enum": ["major", "minor"]},
        "evidence_kind": {"type": "string", "enum": ["direct_contradiction", "positive_missing", "absence_only", "ambiguous"]}},
        "required": ["claimed_text", "actual_visual_evidence", "frame_indices", "error_type", "severity", "evidence_kind"]}
    props = {"status": {"type": "string", "enum": ["pass", "confirmed_error", "uncertain"]},
             "errors": {"type": "array", "items": issue, "maxItems": 20},
             "uncertainties": {"type": "array", "items": issue, "maxItems": 20},
             "coverage_note": {"type": "string", "minLength": 1}}
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


def validate_review(value, evidence):
    indices = [f["index"] for f in evidence["frames"]]
    review_schema(indices)
    need(isinstance(value, dict) and set(value) == {"status", "errors", "uncertainties", "coverage_note"}, "strict review object fields required")
    need(value["status"] in ("pass", "confirmed_error", "uncertain"), "invalid review status")
    need(isinstance(value["coverage_note"], str) and value["coverage_note"].strip(), "coverage note required")
    for key in ("errors", "uncertainties"):
        need(isinstance(value[key], list) and len(value[key]) <= 20, "bounded issue list required")
        for issue in value[key]:
            need(isinstance(issue, dict) and set(issue) == {"claimed_text", "actual_visual_evidence", "frame_indices", "error_type", "severity", "evidence_kind"}, "strict issue fields required")
            need(isinstance(issue["claimed_text"], str) and isinstance(issue["actual_visual_evidence"], str)
                 and issue["actual_visual_evidence"].strip(), "claimed text and actual visual evidence required")
            need(issue["error_type"] in ERROR_TYPES and issue["severity"] in ("major", "minor")
                 and issue["evidence_kind"] in ("direct_contradiction", "positive_missing", "absence_only", "ambiguous"), "invalid issue enums")
            need(isinstance(issue["frame_indices"], list) and len(issue["frame_indices"]) <= 16
                 and all(type(i) is int for i in issue["frame_indices"]), "actual integer source frame evidence required")
    checked = copy.deepcopy(value)
    supported, weak = [], []
    for issue in checked["errors"] + checked["uncertainties"]:
        fids = issue["frame_indices"]
        need(len(fids) == len(set(fids)) and fids == sorted(fids), "duplicate/unordered source frame evidence")
        need(set(fids) <= set(indices), "foreign source frame evidence")
    for issue in checked["errors"]:
        if len(issue["frame_indices"]) >= 2 and issue["evidence_kind"] in ("direct_contradiction", "positive_missing"):
            supported.append(issue)
        else:
            weak.append(issue)
    checked["errors"] = supported
    checked["uncertainties"].extend(weak)
    expected = "confirmed_error" if supported else "uncertain" if checked["uncertainties"] else "pass"
    need(value["status"] != "pass" or expected == "pass", "pass cannot conceal errors/uncertainties")
    need(value["status"] != "confirmed_error" or bool(value["errors"]), "confirmed_error requires an error")
    need(value["status"] != "uncertain" or bool(value["uncertainties"] or weak), "uncertain requires an uncertainty")
    checked["status"] = expected
    checked.update({"semantic_approval": False, "human_review_completed": False,
                    "confirmed_errors_are_AI_candidates_not_ground_truth": True,
                    "full_continuous_video_observed": False})
    return checked


def next_sample_target(current, total, errors, reviewed, *, threshold=.10, expansion_counts=(50, 100)):
    need(0 <= threshold <= 1 and reviewed >= 0 and 0 <= errors <= reviewed, "invalid expansion denominator/threshold")
    need(all(type(n) is int and n > 0 for n in expansion_counts)
         and list(expansion_counts) == sorted(set(expansion_counts)), "finite ascending expansion counts required")
    if not reviewed or errors / reviewed < threshold:
        return current
    for target in expansion_counts:
        if min(target, total) > current:
            return min(target, total)
    return current


def code_identity():
    root = Path(__file__).resolve().parent
    return batch.identity_tree(root, [root / n for n in ("ms_batch_review.py", "ms_bar_annotation.py",
        "ms_caption_rewrite.py", "ms_visual_batch.py", "ms_visual_observe.py", "ms_optimize.py",
        "ms_landmarks.py", "ms_hand_roi.py")])


def source_references(row):
    refs = []
    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                identity = batch.file_identity(Path(value["path"]))
                need(identity["sha256"] == value["sha256"], "source reference SHA mismatch")
                refs.append(identity)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(row.get("sourceRefs", row.get("source_refs", {})))
    for key in ("raw_rgb_manifest", "hand_roi", "visual_result_ref", "style_result_ref", "source_identity"):
        visit(row.get(key))
    return refs


def prepare_evidence(row, root, *, ffprobe="ffprobe", ffmpeg="ffmpeg", hand_roi_root=None):
    """Decode sampled ORIGINAL frames once; hand proposals never supply semantics."""
    root = Path(root)
    video = batch.file_identity(Path(row["video_path"]))
    payload = label_payload(row)
    need(label_reviewable(row), "two label components not ready for source audit")
    source = {"files": {"video_path": video}}
    if row.get("audio_path"):
        source["files"]["audio_path"] = batch.file_identity(Path(row["audio_path"]))
    label_ref = batch.file_identity(Path(row["label_path"])) if row.get("label_path") else None
    refs = source_references(row)
    binding = {"source_id": source_id(row), "split": row["split"], "video_sha256": video["sha256"],
               "input_row_sha256": batch.digest(row), "source_refs": refs,
               "label_sha256": batch.digest(payload), "label_file_ref": label_ref}
    probe = bar.probe_actual(row["video_path"], ffprobe)
    selected = [round(k * (len(probe["times"]) - 1) / (min(16, len(probe["times"])) - 1))
                for k in range(min(16, len(probe["times"])))]
    need(len(selected) >= 2 and selected == sorted(set(selected)), "insufficient actual frames")
    crops, roi_ref = [], {"available": False, "reason": "not_requested"}
    if hand_roi_root and "hands" in risk_flags(row):
        roi_row = {"id": source_id(row), "binding": {"video_sha256": video["sha256"]}}
        roi_manifest, roi_ref = bar.load_hand_roi(roi_row, probe, hand_roi_root)
        if roi_manifest:
            import ms_hand_roi as roi
            hand = roi.select_roi_evidence(roi_manifest, 0., probe["media_end"], max_per_hand=3)
            crops = [c for p in ("left_hand", "right_hand") for c in hand[p]["crops"]]
            required = {c["frame_index"] for c in crops}
            selected = sorted(required | {selected[0], selected[-1]})
            candidates = [round(k * (len(probe["times"]) - 1) / 15) for k in range(16)]
            for i in candidates:
                if len(selected) >= min(16, len(probe["times"])):
                    break
                selected = sorted(set(selected) | {i})
    binding["frame_indices"] = selected
    binding["hand_roi_ref"] = roi_ref
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "evidence.json"
    if manifest_path.exists():
        evidence = read_json(manifest_path)
        need(evidence["binding"] == binding, "source/label/frame binding changed on resume")
        need(read_json(root / "evidence.receipt.json")["sha256"] == batch.sha256(manifest_path), "evidence manifest changed")
        for f in evidence["frames"]:
            need(batch.sha256(root / f["file"]) == f["sha256"], "source RGB evidence changed")
        for crop in evidence["hand_rois"]:
            need(batch.sha256(Path(crop["path"])) == crop["sha256"], "native hand crop changed")
        batch.verify_source(source)
        return evidence
    frame_dir = Path(tempfile.mkdtemp(prefix="original-rgb-", dir=root))
    expression = "+".join(f"eq(n\\,{i})" for i in selected)
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", row["video_path"], "-map", "0:v:0",
                    "-vf", f"select={expression}", "-fps_mode", "vfr", "-q:v", "2", str(frame_dir / "%06d.jpg")],
                   check=True, capture_output=True)
    images = sorted(frame_dir.glob("*.jpg"))
    need(len(images) == len(selected), "original source frame extraction count mismatch")
    frames = [{"index": i, "time": probe["times"][i], "source_timestamp": probe["source_timestamps"][i],
               "file": p.relative_to(root).as_posix(), "sha256": batch.sha256(p)} for i, p in zip(selected, images)]
    batch.verify_source(source)
    if label_ref:
        need(batch.snapshot(Path(label_ref["path"])) == label_ref["stat"], "label changed during extraction")
    evidence = {"schema": SCHEMA, "binding": binding, "source_identity": source, "frames": frames,
                "hand_rois": crops, "probe": probe, "frame_indices_are_original_global": True,
                "source_RGB_saved_native_size": True, "sparse_samples_are_not_full_video": True,
                "image_budget": {"source_frames": len(frames), "native_hand_rois": len(crops), "max_source_frames": 16, "max_ROIs": 6}}
    core.save_json(manifest_path, evidence)
    core.save_json(root / "evidence.receipt.json", {"sha256": batch.sha256(manifest_path)})
    return evidence


def review_request(row, evidence, root, model, *, max_tokens=1800):
    payload = label_payload(row)
    content = [{"type": "text", "text": json.dumps({"label_to_test": payload,
        "allowed_source_frame_indices": [f["index"] for f in evidence["frames"]],
        "sparse_observation_only": True, "absence_between_samples_not_an_error": True}, ensure_ascii=False)}]
    for frame in evidence["frames"]:
        content.extend([{"type": "text", "text": f"Source frame {frame['index']}; relative {frame['time']:.9f}s; original PTS {frame['source_timestamp']:.9f}s."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode((Path(root) / frame["file"]).read_bytes()).decode("ascii")}}])
        for crop in evidence["hand_rois"]:
            if crop["frame_index"] != frame["index"]:
                continue
            need(batch.sha256(Path(crop["path"])) == crop["sha256"], "ROI changed before dispatch")
            content.extend([{"type": "text", "text": json.dumps({k: crop[k] for k in ("frame_index", "part", "crop_xyxy", "crop_pixel_size", "source_frame_size")}) + " Same original frame; anatomical side unverified; native unresized ROI."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(crop["path"]).read_bytes()).decode("ascii")}}])
    need(sum(c["type"] == "image_url" for c in content) <= 22, "review image budget exceeded")
    return {"model": str(Path(model).resolve()), "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "sampling": {"temperature": 0., "seed": 0, "max_tokens": max_tokens},
            "json_schema": review_schema([f["index"] for f in evidence["frames"]]), "prompt_sha256": batch.digest(SYSTEM)}


def checkpoint_identity(identity):
    weights = sorted(v["sha256"] for name, v in identity["files"].items()
                     if Path(name).suffix in (".safetensors", ".bin", ".pt", ".gguf"))
    need(weights, "model reference requires actual weight SHA inventory")
    return batch.digest({"weight_file_content_sha256": weights})


def read_model_reference(path, *, permit_attested=False):
    value = read_json(path)
    identity = value.get("model_identity", value.get("model", value))
    need(isinstance(identity, dict) and isinstance(identity.get("files"), dict) and identity.get("root"), "full model SHA inventory reference required")
    content = {name: {"path": ref["path"], "sha256": ref["sha256"]} for name, ref in identity["files"].items()}
    need(batch.digest({"root": identity["root"], "files": content}) == identity["sha256"], "model reference inventory SHA mismatch")
    attested = permit_attested and value.get("source_sha_verified_once") is True
    if not attested:
        batch.verify_current(identity, model=True)
    return identity, {"path": str(Path(path).resolve()), "sha256": batch.sha256(Path(path)),
                      "source_SHA_verified_attestation_used": attested,
                      "weights_rehashed_on_reviewer_node": False}


class LocalReviewer:
    """One native vLLM/processor instance across every source and batch."""
    def __init__(self, model, out, *, model_reference=None, max_model_len=32768, max_pixels=49152,
                 gpu_memory_utilization=.75, annotator_model_sha256=None, annotator_model_reference=None, hand_roi_root=None,
                 ffprobe="ffprobe", ffmpeg="ffmpeg"):
        self.model, self.out = str(Path(model).resolve()), Path(out)
        self.model_reference = model_reference
        self.max_model_len, self.max_pixels = max_model_len, max_pixels
        self.gpu_memory_utilization = gpu_memory_utilization
        self.annotator_model_sha256 = annotator_model_sha256
        self.annotator_model_reference = annotator_model_reference
        self._annotator_identity = None
        self._annotator_reference_ref = None
        self.hand_roi_root, self.ffprobe, self.ffmpeg = hand_roi_root, ffprobe, ffmpeg
        self._loaded, self._identity = None, None
        self.code = code_identity()

    def identity(self):
        if self._identity is None:
            if self.model_reference:
                self._identity, _ = read_model_reference(self.model_reference)
                need(str(Path(self._identity["root"]).resolve()) == self.model, "reviewer reference root mismatch")
                batch.verify_current(self._identity, model=True)
            else:
                self._identity = batch.model_identity(self.model)
            if self.annotator_model_reference:
                self._annotator_identity, self._annotator_reference_ref = read_model_reference(self.annotator_model_reference, permit_attested=True)
                need(not self.annotator_model_sha256 or self.annotator_model_sha256 == self._annotator_identity["sha256"],
                     "annotator model SHA differs from its reference")
                self.annotator_model_sha256 = self._annotator_identity["sha256"]
                need(checkpoint_identity(self._annotator_identity) != checkpoint_identity(self._identity),
                     "reviewer checkpoint weights equal annotator despite different deployment roots")
            need(not self.annotator_model_sha256 or self.annotator_model_sha256 != self._identity["sha256"],
                 "reviewer must use a different annotator model identity")
        return self._identity

    def runtime(self):
        if self._loaded is not None:
            return self._loaded
        identity = self.identity()
        batch.verify_current(identity, model=True)
        batch.verify_current(self.code)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
        from vllm import LLM, SamplingParams
        llm = LLM(model=self.model, tensor_parallel_size=1, dtype="bfloat16", max_model_len=self.max_model_len,
                  max_num_seqs=1, gpu_memory_utilization=self.gpu_memory_utilization,
                  limit_mm_per_prompt={"image": 22, "video": 0}, mm_processor_kwargs={"max_pixels": self.max_pixels},
                  structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
                  enable_prefix_caching=True, trust_remote_code=False, seed=0)
        llm._ms_bar_model_sha256 = identity["sha256"]
        self.out.mkdir(parents=True, exist_ok=True)
        core.save_json(self.out / "reviewer.model.identity.json", identity)
        core.save_json(self.out / "reviewer.code.identity.json", self.code)
        loads = self.out / "loads"
        loads.mkdir(exist_ok=True)
        core.save_json(loads / f"{os.getpid()}.json", {"schema": SCHEMA, "model_load_completed": True,
            "model_identity_sha256": identity["sha256"], "code_sha256": self.code["sha256"],
            "annotator_model_sha256": self.annotator_model_sha256, "different_model_identity_verified": bool(self.annotator_model_sha256),
            "checkpoint_content_independence_verified": self._annotator_identity is not None,
            "annotator_model_reference": self._annotator_reference_ref,
            "reviewer_checkpoint_sha256": checkpoint_identity(identity),
            "annotator_checkpoint_sha256": checkpoint_identity(self._annotator_identity) if self._annotator_identity else None,
            "model_family_independence_claimed": False, "shared_model_and_processor_for_all_batches": True,
            "dtype": "bfloat16", "tensor_parallel_size": 1, "image_limit": 22,
            "max_model_len": self.max_model_len, "max_pixels": self.max_pixels,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True}})
        self._loaded = (llm, SamplingParams)
        return self._loaded

    def __call__(self, row, root, *, prepare_only=False):
        root = Path(root)
        evidence = prepare_evidence(row, root, ffprobe=self.ffprobe, ffmpeg=self.ffmpeg, hand_roi_root=self.hand_roi_root)
        request = review_request(row, evidence, root, self.model)
        identity = self.identity() if not prepare_only else {"sha256": None}
        teacher = row.get("annotator_model_sha256", self.annotator_model_sha256)
        need(prepare_only or bool(teacher), "run requires annotator_model_sha256 to attest checkpoint independence")
        need(prepare_only or teacher != identity["sha256"], "independent reviewer model identity equals annotator")
        need(prepare_only or self._annotator_identity is None or teacher == self.annotator_model_sha256,
             "source annotator differs from verified annotator checkpoint reference")
        binding = {**evidence["binding"], "model_sha256": identity["sha256"], "code_sha256": self.code["sha256"],
                   "annotator_model_sha256": teacher, "model_family_independence_claimed": False,
                   "checkpoint_content_independence_verified": self._annotator_identity is not None,
                   "annotator_model_reference": self._annotator_reference_ref,
                   "reviewer_checkpoint_sha256": checkpoint_identity(identity) if not prepare_only else None,
                   "runtime_parameters": {"dtype": "bfloat16", "tp": 1, "max_model_len": self.max_model_len,
                                          "max_pixels": self.max_pixels, "image_limit": 22},
                   "evidence_sha256": batch.sha256(root / "evidence.json")}
        # Prepare without model identity is a separate immutable transaction from run.
        transaction_root = root / ("prepare" if prepare_only else "review")
        item = {"root": transaction_root, "request": request,
                "validator": lambda v: validate_review(v, evidence), "binding": binding}
        result = bar.transactions([item], self.runtime, SimpleNamespace(stage="prepare" if prepare_only else "run", batch_size=1))[str(transaction_root)]
        batch.verify_source(evidence["source_identity"])
        need(batch.digest(label_payload(row)) == binding["label_sha256"], "label changed during reviewer generation")
        for ref in [binding.get("label_file_ref")] + binding.get("source_refs", []):
            if ref:
                need(batch.snapshot(Path(ref["path"])) == ref["stat"], "source/label reference changed during reviewer generation")
        if result["status"] == "candidate":
            value = {**result["json"], "transaction_status": "candidate", "source_id": source_id(row),
                     "review_receipt": {"path": str(transaction_root / "receipt.json"), "sha256": batch.sha256(transaction_root / "receipt.json")},
                     "label_sha256": binding["label_sha256"], "evidence": evidence, "evidence_root": str(root)}
        else:
            value = {"status": result["status"], "source_id": source_id(row), "error": result.get("error"),
                     "label_sha256": binding["label_sha256"], "evidence": evidence, "evidence_root": str(root)}
        return value


def human_resolved(result, decisions):
    sid = result["source_id"]
    decision = decisions.get(sid, {})
    if not isinstance(decision, dict) or decision.get("decision") != "false_positive":
        return False
    return (bool(result.get("label_sha256")) and bool(result.get("review_receipt", {}).get("sha256"))
            and decision.get("human_review_completed") is True and bool(decision.get("note"))
            and decision.get("label_sha256") == result.get("label_sha256")
            and decision.get("review_receipt_sha256") == result.get("review_receipt", {}).get("sha256"))


def batch_acceptance(plan, results, *, decisions=None, expansion_complete=True):
    decisions = decisions or {}
    expected = set(plan["sampling"]["selected_source_ids"])
    reviewed = {sid for sid, v in results.items() if v["status"] in ("pass", "confirmed_error", "uncertain")}
    complete = expected <= reviewed and expansion_complete
    unresolved_errors = [sid for sid, v in results.items() if v["status"] == "confirmed_error" and not human_resolved(v, decisions)]
    unresolved_major = [sid for sid, v in results.items() if v["status"] in ("uncertain", "confirmed_error")
                        and any(i["severity"] == "major" for i in v.get("uncertainties", [])) and not human_resolved(v, decisions)]
    labels_done = all(label_complete(r) for r in plan["records"])
    return {"all_source_labels_completed": labels_done, "review_complete": complete,
            "unresolved_confirmed_error_source_ids": unresolved_errors,
            "unresolved_major_uncertain_source_ids": unresolved_major,
            "eligible_for_training": labels_done and complete and not unresolved_errors and not unresolved_major,
            "human_approval_automatically_granted": False, "unsampled_sources_individually_reviewed": False}


def review_queue(plan, results, *, representative_limit=3):
    by_id = {source_id(r): r for r in plan["records"]}
    representatives, optional, counts = [], [], Counter()
    for sid in sorted(results):
        result = results[sid]
        if result["status"] not in ("confirmed_error", "uncertain"):
            continue
        evidence, root = result.get("evidence", {}), Path(result.get("evidence_root", "."))
        issues = result.get("errors", []) if result["status"] == "confirmed_error" else result.get("uncertainties", [])
        frame_ids = {i for issue in issues for i in issue["frame_indices"]}
        item = {"source_id": sid, "batch_id": plan["batch_id"], "video_path": by_id[sid]["video_path"],
                "audio_path": by_id[sid].get("audio_path"), "original_caption": by_id[sid].get("original_caption", by_id[sid].get("caption", "")),
                "new_label": label_payload(by_id[sid]), "status": result["status"], "errors": issues,
                "evidence_frames": [{**f, "path": str(root / f["file"])} for f in evidence.get("frames", []) if f["index"] in frame_ids],
                "native_hand_rois": [c for c in evidence.get("hand_rois", []) if c["frame_index"] in frame_ids],
                "label_sha256": result.get("label_sha256"), "review_receipt": result.get("review_receipt"),
                "human_decision": "pending", "human_review_completed": False, "AI_error_candidate_not_ground_truth": True}
        if result["status"] == "uncertain":
            optional.append(item)
        else:
            types = set(i["error_type"] for i in issues)
            if any(counts[t] < representative_limit for t in types):
                representatives.append(item)
                counts.update(types)
    return {"schema": SCHEMA, "batch_id": plan["batch_id"], "items": representatives,
            "optional_uncertain_items": optional, "all_error_source_ids": [sid for sid, v in results.items() if v["status"] == "confirmed_error"],
            "error_count": sum(v["status"] == "confirmed_error" for v in results.values()),
            "needs_human_count": sum(v["status"] in ("confirmed_error", "uncertain") for v in results.values()),
            "human_decision": "pending", "human_approval_automatically_granted": False,
            "representative_limit_per_error_type": representative_limit}


def review_batch(plan, out, reviewer, *, prepare_only=False, decisions=None, error_threshold=.10, expansion_counts=(50, 100)):
    records = {source_id(r): r for r in plan["records"]}
    need(set(plan["source_ids"]) == set(records), "batch records/source IDs differ")
    need(len({r["split"] for r in records.values()}) == 1, "review batch cannot cross splits")
    root = Path(out) / plan["batch_id"]
    need(root.resolve().is_relative_to(Path(out).resolve()), "unsafe batch_id")
    root.mkdir(parents=True, exist_ok=True)
    core.save_json(root / "batch.membership.json", {"batch_id": plan["batch_id"], "split": plan["split"],
        "source_ids": plan["source_ids"], "random_order": plan["sampling"]["random_order"],
        "seed": plan["sampling"]["seed"], "policy": plan["sampling"]["policy"]})
    batch.atomic_write(root / "batch.plan.json", batch.json_bytes(plan))
    sample = plan["sampling"]
    order = sample["random_order"]
    need(set(order) == set(records) and len(order) == len(records), "random review order must bind complete source batch")
    target = len(sample["random_source_ids"])
    results, rounds = {}, []
    expansion_complete = False
    while True:
        selected = list(dict.fromkeys(order[:target] + sample["risk_source_ids"]))
        for sid in selected:
            if sid in results:
                continue
            if not label_reviewable(records[sid]):
                results[sid] = {"status": "label_incomplete", "source_id": sid}
                continue
            revision = batch.digest({"label": label_payload(records[sid]), "label_path": records[sid].get("label_path"),
                                     "input_row_sha256": batch.digest(records[sid]),
                                     "annotator_model_sha256": records[sid].get("annotator_model_sha256")})[:16]
            source_root = root / "sources" / hashlib.sha256(sid.encode()).hexdigest()[:24] / revision
            result = reviewer(records[sid], source_root, prepare_only=prepare_only)
            need(result.get("source_id") == sid, "review result source ID mismatch")
            results[sid] = result
            if result["status"] == "confirmed_error":
                core.save_json(root / "isolation" / (source_root.parent.name + "-" + source_root.name + ".json"), {
                    "source_id": sid, "AI_candidate_pending_human": True, "excluded_until_resolved": True,
                    "label_sha256": result.get("label_sha256"), "review_receipt": result.get("review_receipt"), "errors": result["errors"]})
        reviewed = [v for v in results.values() if v["status"] in ("pass", "confirmed_error", "uncertain")]
        errors = sum(v["status"] == "confirmed_error" for v in reviewed)
        next_target = target if prepare_only else next_sample_target(target, len(order), errors, len(reviewed),
                                                                     threshold=error_threshold, expansion_counts=expansion_counts)
        rounds.append({"random_target": target, "sampled_reviewed_denominator": len(reviewed), "AI_error_candidates": errors,
                       "sampled_candidate_error_rate": errors / len(reviewed) if reviewed else None})
        if next_target == target:
            expansion_complete = not prepare_only
            break
        target = next_target
    effective = copy.deepcopy(plan)
    effective["sampling"]["selected_source_ids"] = list(results)
    acceptance = batch_acceptance(effective, results, decisions=decisions, expansion_complete=expansion_complete)
    reviewed_ids = sorted(sid for sid, v in results.items() if v["status"] in ("pass", "confirmed_error", "uncertain"))
    denominator = len(reviewed_ids)
    random_reviewed = [sid for sid in order[:target] if sid in reviewed_ids]
    summary = {"schema": SCHEMA, "batch_id": plan["batch_id"], "source_count": len(records),
        "reviewed_source_ids": reviewed_ids, "selected_source_ids": list(results),
        "unsampled_source_ids": sorted(set(records) - set(results)), "unsampled_source_status": "not_individually_reviewed",
        "sampled_reviewed_denominator": denominator, "sampled_fraction": denominator / len(records),
        "confirmed_error_count": errors, "sampled_candidate_error_rate": errors / denominator if denominator else None,
        "random_reviewed_denominator": len(random_reviewed), "risk_extra_source_ids": sample["risk_source_ids"],
        "random_candidate_error_rate": sum(results[sid]["status"] == "confirmed_error" for sid in random_reviewed) / len(random_reviewed) if random_reviewed else None,
        "rounds": rounds, "policy": {**sample["policy"], "error_threshold": error_threshold, "expansion_counts": list(expansion_counts)},
        "acceptance": acceptance, "statuses": dict(Counter(v["status"] for v in results.values())),
        "full_batch_semantic_pass_claimed": False, "human_decision": "pending"}
    # Summaries are snapshots of immutable individual receipts; only these aggregate files refresh.
    batch.atomic_write(root / "summary.json", batch.json_bytes(summary))
    batch.atomic_write(root / "review_web.json", batch.json_bytes(review_queue(plan, results)))
    return summary


def review_batch_ready(records, out, reviewer, *, batch_id, expected_source_ids=None,
                       seed=0, sample_count=25, risk_count=5, prepare_only=False, decisions=None,
                       error_threshold=.10, expansion_counts=(50, 100)):
    """Review ready sampled sources now; pending labels never block production.

    records should contain the complete fixed batch membership, with label fields
    attached as they finish. If only completed rows are available, supply the full
    expected_source_ids so a partial batch can NEVER be accepted for training.
    Human decisions are external inputs; confirmed_error remains an AI candidate.
    """
    rows = [normalized_record(r) for r in records]
    need(rows, "batch membership/source records required")
    by_id = {source_id(r): r for r in rows}
    need(len(by_id) == len(rows), "duplicate ready source IDs")
    expected = sorted(expected_source_ids if expected_source_ids is not None else by_id)
    need(len(expected) == len(set(expected)) and set(by_id) <= set(expected), "ready records must belong to fixed batch membership")
    splits = {r["split"] for r in rows}
    need(len(splits) == 1, "ready batch cannot cross splits")
    split = rows[0]["split"]
    # Missing rows have no media identity or semantic label; selected missing IDs
    # are explicitly pending and never passed to the VLM.
    for sid in expected:
        if sid not in by_id:
            by_id[sid] = {"source_id": sid, "split": split, "label_status": "pending"}
    full_rows = [by_id[sid] for sid in expected]
    plan = {"schema": SCHEMA, "batch_id": str(batch_id), "split": split, "source_ids": expected,
            "source_count": len(expected), "records": full_rows,
            "sampling": sampling_plan(full_rows, str(batch_id), seed=seed, sample_count=sample_count, risk_count=risk_count),
            "semantic_review_completed": False, "human_review_completed": False}
    summary = review_batch(plan, out, reviewer, prepare_only=prepare_only, decisions=decisions,
                           error_threshold=error_threshold, expansion_counts=expansion_counts)
    summary["pending_source_ids"] = [source_id(r) for r in full_rows if not label_complete(r)]
    summary["state"] = "pending" if summary["pending_source_ids"] or not summary["acceptance"]["review_complete"] else "accepted" if summary["acceptance"]["eligible_for_training"] else "human_review_required"
    summary["confirmed_errors_are_AI_candidates"] = True
    summary["producer_processing_allowed"] = True
    batch.atomic_write(Path(out) / str(batch_id) / "summary.json", batch.json_bytes(summary))
    return summary


def review_batches(plans, out, reviewer, **kwargs):
    return [review_batch(plan, out, reviewer, **kwargs) for plan in plans]


def read_batch_list(path, *, batch_size=500, seed=0, sample_count=25, risk_count=5, results_root=None):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    try:
        decoded = strict_json(text)
    except ValueError:
        decoded = [strict_json(line) for line in text.splitlines() if line.strip()]
    if isinstance(decoded, dict) and "batches" in decoded and results_root:
        rows = []
        root = Path(results_root).resolve()
        seen = set()
        # Automatic snapshots are per-batch increments, NOT cumulative manifests.
        # Inspect only the fixed batch-level publications named in the manifest.
        for descriptor in decoded["batches"]:
            bid = str(descriptor["batch_id"])
            need(bid not in seen, "duplicate batch ID in source manifest")
            seen.add(bid)
            ready = root / "batches" / bid / "batch_ready.json"
            need(ready.resolve().is_relative_to(root / "batches"), "unsafe manifest batch ID")
            if not ready.is_file():
                continue
            obj = read_json(ready)
            need(str(obj["batch_id"]) == bid, "ready batch ID differs from source manifest")
            need("split" not in descriptor or obj["split"] == descriptor["split"], "ready batch split differs from source manifest")
            need("count" not in descriptor or len(obj["source_ids"]) == descriptor["count"], "ready batch count differs from source manifest")
            rows.append(obj)
        need(rows, "no fixed batch-ready publication yet; use completed pilot source rows directly")
    elif isinstance(decoded, dict) and "published" in decoded:
        rows = []
        for item in decoded["published"]:
            ready = Path(item["path"])
            if not ready.is_absolute():
                ready = (Path(results_root) if results_root else path.parent) / ready
            if item.get("sha256"):
                need(batch.sha256(ready) == item["sha256"], "published batch-ready SHA mismatch")
            rows.append(read_json(ready))
    else:
        rows = [decoded] if isinstance(decoded, dict) else decoded
    need(isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows), "nonempty object batch/source list required")
    if all("records" in row and "sampling" not in row for row in rows):
        plans = []
        for obj in rows:
            members = [normalized_record(r) for r in obj["records"]]
            ids = [source_id(r) for r in members]
            need(members and set(obj["source_ids"]) == set(ids) and len(ids) == len(set(ids)), "complete matching batch-ready membership required")
            need(len({r["split"] for r in members}) == 1 and members[0]["split"] == obj["split"], "batch-ready split mismatch")
            bid = str(obj["batch_id"])
            plans.append({"schema": SCHEMA, "batch_id": bid, "split": obj["split"], "records": members,
                "source_ids": list(obj["source_ids"]), "source_count": len(members),
                "sampling": sampling_plan(members, bid, seed=seed, sample_count=sample_count, risk_count=risk_count),
                "semantic_review_completed": False, "human_review_completed": False})
        return plans
    if all("batch_id" in row and "records" not in row for row in rows):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row["batch_id"])].append(normalized_record(row))
        plans = []
        for bid in sorted(groups):
            members = sorted(groups[bid], key=source_id)
            need(len({r["split"] for r in members}) == 1, "existing production batch cannot cross splits")
            plans.append({"schema": SCHEMA, "batch_id": bid, "split": members[0]["split"], "records": members,
                "source_ids": [source_id(r) for r in members], "source_count": len(members),
                "sampling": sampling_plan(members, bid, seed=seed, sample_count=sample_count, risk_count=risk_count),
                "semantic_review_completed": False, "human_review_completed": False})
        return plans
    if all("records" in row and "sampling" in row for row in rows):
        return rows
    need(not any("records" in row or "sampling" in row for row in rows), "mixed source and batch rows")
    return prepare_batches(rows, batch_size=batch_size, seed=seed, sample_count=sample_count, risk_count=risk_count)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--results-root")
    p.add_argument("--model")
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--run", action="store_true")
    p.add_argument("--sample-count", type=int, default=25)
    p.add_argument("--risk-count", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-reference")
    p.add_argument("--annotator-model-sha256")
    p.add_argument("--annotator-model-reference")
    p.add_argument("--hand-roi-root")
    p.add_argument("--human-decisions")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-pixels", type=int, default=49152)
    p.add_argument("--gpu-memory-utilization", type=float, default=.75)
    p.add_argument("--error-threshold", type=float, default=.10)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    plans = read_batch_list(args.batch_list, batch_size=args.batch_size, seed=args.seed,
                           sample_count=args.sample_count, risk_count=args.risk_count, results_root=args.results_root)
    out = Path(args.out)
    if args.prepare_only:
        out.mkdir(parents=True, exist_ok=True)
        core.save_once(out / "batch-list.jsonl", b"".join(batch.json_bytes(p) + b"\n" for p in plans))
        core.save_json(out / "prepare.receipt.json", {"schema": SCHEMA, "batch_count": len(plans),
            "source_count": sum(p["source_count"] for p in plans), "input_sha256": batch.sha256(Path(args.batch_list)),
            "code_sha256": code_identity()["sha256"], "GPU_model_load_performed": False,
            "semantic_review_completed": False, "human_review_completed": False})
        print(json.dumps({"prepared_batches": len(plans), "model_loaded": False}))
        return 0
    need(args.model, "--run requires local --model")
    reviewer = LocalReviewer(args.model, out, model_reference=args.model_reference,
        max_model_len=args.max_model_len, max_pixels=args.max_pixels,
        gpu_memory_utilization=args.gpu_memory_utilization, annotator_model_sha256=args.annotator_model_sha256,
        annotator_model_reference=args.annotator_model_reference,
        hand_roi_root=args.hand_roi_root, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg)
    decisions = read_json(args.human_decisions) if args.human_decisions else {}
    summaries = review_batches(plans, out, reviewer, decisions=decisions, error_threshold=args.error_threshold)
    batch.atomic_write(out / "run.summary.json", batch.json_bytes({"schema": SCHEMA,
        "batch_count": len(summaries), "review_complete_batches": sum(s["acceptance"]["review_complete"] for s in summaries),
        "eligible_for_training_batches": sum(s["acceptance"]["eligible_for_training"] for s in summaries),
        "all_batches_completed": all(s["acceptance"]["review_complete"] for s in summaries),
        "batch_summaries": [str(out / s["batch_id"] / "summary.json") for s in summaries],
        "human_approval_automatically_granted": False}))
    print(json.dumps({"batches": len(summaries), "eligible_batches": sum(s["acceptance"]["eligible_for_training"] for s in summaries)}))
    return 0 if all(s["acceptance"]["review_complete"] for s in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
