"""Independent open-vocabulary video observations, separate from rule labels.

prepare samples actual video frames and exports a VLM request without MotionScript
text. import-response validates an externally produced response. observe optionally
calls a user-specified VLM endpoint. export requires a content-bound review record.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import urllib.parse

from ms_optimize import atomic_write, json_bytes, overlaps, sha256


SCHEMA = "motionscript_open_visual_v1"
SYSTEM = """Observe only the dancer's visible body movements in the ordered video frames.
MotionScript labels and an action-class vocabulary are deliberately not supplied.
Describe the actual sequence and transitions in natural English; action names are
open vocabulary. Do not guess a named dance step from genre or music. Do not
describe identity, clothing, scene, camera style, emotion, or unobserved contacts.
Keep anatomical sides distinct from image-left/right. Mark ambiguity instead of
guessing. Repeated physical movements should be described once with their pattern,
not by repeating the same sentence. Do not pad the caption to a minimum length.
Return JSON only, with exactly these keys:
motion_caption: a concise chronological paragraph, at most 140 English words;
events: list of {id, description, frame_indices}, each supported by at least two
supplied frames. Use real supplied frame indices, not invented timestamps. Frame
evidence bounds are not precise event boundaries. Action descriptions are free
text, not an enum. uncertainties: list of strings. abstain: boolean. If no movement
is supported, return abstain=true, motion_caption="", events=[].
These outputs are unreviewed candidates, not ground truth.
"""


def sentence_key(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


def parse_response(raw: bytes) -> dict:
    value = strict_json(raw.decode("utf-8-sig"))
    if isinstance(value, dict) and "choices" in value:
        try:
            value = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("VLM service response has no assistant content") from error
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```json") and value.endswith("```"):
            value = value[7:-3].strip()
        value = strict_json(value)
    if not isinstance(value, dict):
        raise ValueError("VLM response must contain a JSON object")
    return value


def validate_candidate(value: dict, manifest: dict) -> dict:
    if set(value) != {"motion_caption", "events", "uncertainties", "abstain"}:
        raise ValueError("candidate keys do not match the open-visual schema")
    caption, events, uncertainties, abstain = (value[k] for k in
        ("motion_caption", "events", "uncertainties", "abstain"))
    if not isinstance(abstain, bool) or not isinstance(caption, str) or not isinstance(events, list):
        raise ValueError("invalid caption, events or abstain type")
    if not isinstance(uncertainties, list) or not all(isinstance(x, str) for x in uncertainties):
        raise ValueError("uncertainties must be a string list")
    if abstain:
        if caption.strip() or events:
            raise ValueError("abstention must have empty caption and events")
    elif not caption.strip() or not events:
        raise ValueError("non-abstention needs a caption and supported event facts")
    if len(caption.split()) > 140:
        raise ValueError("caption exceeds 140 words; do not pad detailed rule clauses")
    clauses = [sentence_key(x) for x in re.split(r"[.!?]+", caption) if sentence_key(x)]
    if len(clauses) != len(set(clauses)):
        raise ValueError("caption repeats an identical clause; describe the repetition pattern once")
    frames = {row["index"]: row["time"] for row in manifest["frames"]}
    checked, used = [], set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"id", "description", "frame_indices"}:
            raise ValueError("invalid visual event schema")
        identifier, text, indices = event["id"], event["description"], event["frame_indices"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used:
            raise ValueError("visual event IDs must be unique nonempty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("visual event description must be nonempty open text")
        if (not isinstance(indices, list) or len(indices) < 2
                or any(type(i) is not int or i not in frames for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("event evidence must name at least two distinct supplied frames")
        used.add(identifier)
        checked.append({**event, "evidence_time_range": [min(frames[i] for i in indices), max(frames[i] for i in indices)],
                        "time_range_is_exact_event_boundary": False})
    return {"schema": SCHEMA, "source": "independent_video_open_vocabulary",
            "motion_caption": caption.strip(), "events": checked,
            "uncertainties": uncertainties, "abstain": abstain,
            "format_valid": True, "semantic_approval": False, "human_review_completed": False,
            "rule_labels_supplied_to_observer": False,
            "vocabulary_policy": "free_text_actions_with_frame_evidence; not a rule-class enum"}


def verify_bundle(bundle: Path) -> dict:
    raw_manifest = (bundle / "manifest.json").read_bytes()
    receipt = strict_json((bundle / "manifest.receipt.json").read_text(encoding="utf-8"))
    if receipt.get("manifest_sha256") != hashlib.sha256(raw_manifest).hexdigest():
        raise ValueError("visual evidence manifest was altered")
    manifest = strict_json(raw_manifest)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("not an open-visual bundle")
    if sha256(bundle / "request.json") != manifest["request_sha256"]:
        raise ValueError("VLM request changed after preparation")
    video = Path(manifest["video_path"])
    if not video.is_file() or sha256(video) != manifest["video_sha256"]:
        raise ValueError("source video is missing or changed; source review cannot be attested")
    for row in manifest["frames"]:
        path = (bundle / row["file"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != row["sha256"]:
            raise ValueError("frame evidence is missing, altered or outside its bundle")
    return manifest


def prepare(args):
    video, out = Path(args.video).resolve(), Path(args.out).resolve()
    if not video.is_file():
        raise ValueError("video input does not exist")
    if overlaps(out, video.parent):
        raise ValueError("use an output directory separate from the video input directory")
    if (not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or args.max_frames < 2
            or not math.isfinite(args.start) or args.start < 0
            or args.end is not None and (not math.isfinite(args.end) or args.end <= args.start)):
        raise ValueError("invalid sampling rate, frame budget or requested time window")
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare requires a new empty output directory")
    video_sha = sha256(video)
    probe = subprocess.run([args.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        check=True, capture_output=True)
    frames = json.loads(probe.stdout)["frames"]
    times = [float(row["best_effort_timestamp_time"]) for row in frames]
    if (not times or any(not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("video needs finite, strictly increasing decoded frame timestamps")
    times = [t - times[0] for t in times]
    available = [i for i, t in enumerate(times) if t >= args.start and (args.end is None or t <= args.end)]
    if len(available) < 2:
        raise ValueError("requested window contains fewer than two frames")
    selected = [available[0]]
    for i in available[1:]:
        if times[i] - times[selected[-1]] >= 1 / args.sample_fps - 1e-9:
            selected.append(i)
    if selected[-1] != available[-1]:
        selected.append(available[-1])
    if len(selected) > args.max_frames:
        selected = [selected[round(k * (len(selected) - 1) / (args.max_frames - 1))]
                    for k in range(args.max_frames)]
    out.mkdir(parents=True, exist_ok=True)
    frame_dir = out / "frames"
    frame_dir.mkdir(exist_ok=True)
    expression = "+".join(f"eq(n\\,{i})" for i in selected)
    subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf",
        f"select={expression},scale=640:640:force_original_aspect_ratio=decrease",
        "-fps_mode", "vfr", "-q:v", "3", str(frame_dir / "%06d.jpg")], check=True, capture_output=True)
    images = sorted(frame_dir.glob("*.jpg"))
    if len(images) != len(selected) or sha256(video) != video_sha:
        raise ValueError("frame extraction count mismatch or source video changed during extraction")
    evidence, content = [], [{"type": "text", "text": "Observe this ordered frame sequence. Only describe supported body movements."}]
    for index, image in zip(selected, images):
        evidence.append({"index": index, "time": times[index], "file": image.relative_to(out).as_posix(), "sha256": sha256(image)})
        content.extend([{"type": "text", "text": f"Source frame {index}, time {times[index]:.6f} seconds."},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}}])
    request = {"model": args.model, "messages": [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": content}], "temperature": 0, "max_tokens": 1200}
    atomic_write(out / "request.json", json_bytes(request))
    manifest = {"schema": SCHEMA, "video_path": str(video), "video_sha256": video_sha,
                "video_stream_selector": "0:v:0 (same stream as ffprobe v:0)",
                "request_sha256": sha256(out / "request.json"), "frames": evidence,
                "code_sha256": {"observer": sha256(Path(__file__)),
                                "io_helper": sha256(Path(__file__).with_name("ms_optimize.py"))},
                "source_frame_time_range": [times[available[0]], times[available[-1]]],
                "sampling_policy": {"requested_fps": args.sample_fps, "max_frames": args.max_frames,
                                    "full_continuous_video_observed_by_model": False},
                "rule_labels_supplied_to_observer": False, "semantic_approval": False}
    manifest_bytes = json_bytes(manifest)
    atomic_write(out / "manifest.receipt.json", json_bytes({
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
    atomic_write(out / "manifest.json", manifest_bytes)
    return manifest


def import_response(bundle: Path, raw: bytes):
    manifest = verify_bundle(bundle)
    raw_path = bundle / "response.raw.json"
    if (bundle / "candidate.json").exists():
        raise ValueError("response already exists; use a separate bundle for a new observation")
    if raw_path.exists() and raw_path.read_bytes() != raw:
        raise ValueError("interrupted bundle has a different raw response; use a new bundle")
    candidate = validate_candidate(parse_response(raw), manifest)
    candidate.update(video_sha256=manifest["video_sha256"], request_sha256=manifest["request_sha256"],
                     raw_response_sha256=hashlib.sha256(raw).hexdigest())
    atomic_write(raw_path, raw)
    atomic_write(bundle / "candidate.json", json_bytes(candidate))
    return candidate


def export(bundle: Path, review: Path, out: Path):
    manifest = verify_bundle(bundle)
    path = bundle / "candidate.json"
    raw = path.read_bytes()
    candidate = strict_json(raw)
    # Reparse the original response rather than treating mutable approval flags
    # in candidate.json as authority. Only the explicit review binds approval.
    original = (bundle / "response.raw.json").read_bytes()
    checked = validate_candidate(parse_response(original), manifest)
    checked.update(video_sha256=manifest["video_sha256"], request_sha256=manifest["request_sha256"],
                   raw_response_sha256=hashlib.sha256(original).hexdigest())
    if candidate != checked:
        raise ValueError("candidate differs from its original response or source evidence")
    reviewed_bytes = review.read_bytes()
    decision = strict_json(reviewed_bytes)
    if (decision.get("approved") is not True or decision.get("video_observed") is not True
            or decision.get("reviewer_type") not in ("human", "assistant_delegated")
            or not isinstance(decision.get("reviewer"), str) or not decision["reviewer"].strip()
            or decision.get("candidate_sha256") != hashlib.sha256(raw).hexdigest()
            or decision.get("video_sha256") != manifest["video_sha256"]
            or decision.get("request_sha256") != manifest["request_sha256"]):
        raise ValueError("export requires an explicit review bound to candidate, video and request hashes")
    if candidate["abstain"]:
        raise ValueError("abstention cannot be exported as a training caption")
    if out.exists() or out.resolve() == review.resolve() or out.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("export needs a new destination outside the evidence bundle")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "source": "independent_video_open_vocabulary_reviewed",
              "motion_caption": candidate["motion_caption"], "events": candidate["events"],
              "video_sha256": manifest["video_sha256"], "request_sha256": manifest["request_sha256"],
              "candidate_sha256": hashlib.sha256(raw).hexdigest(),
              "review_sha256": hashlib.sha256(reviewed_bytes).hexdigest(), "reviewer": decision["reviewer"],
              "reviewer_type": decision["reviewer_type"], "review_is_external_attestation": True,
              "semantic_approval": True, "human_review_completed": decision["reviewer_type"] == "human",
              "training_entry_updated": False, "new_action_names_are_visual_candidates_not_rule_inferences": True}
    atomic_write(out, json_bytes(result))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sample-fps", type=float, default=8.)
    p.add_argument("--max-frames", type=int, default=48)
    p.add_argument("--start", type=float, default=0.)
    p.add_argument("--end", type=float)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    for name in ("import-response", "observe", "export"):
        p = commands.add_parser(name)
        p.add_argument("--bundle", required=True)
        if name == "import-response":
            p.add_argument("--response", required=True)
        elif name == "observe":
            p.add_argument("--endpoint", required=True, help="Explicit VLM chat/completions URL; frames are sent here")
            p.add_argument("--api-key-env", default="MS_VLM_API_KEY")
        else:
            p.add_argument("--review", required=True)
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        value = prepare(args)
    elif args.command == "export":
        value = export(Path(args.bundle).resolve(), Path(args.review).resolve(), Path(args.out).resolve())
    else:
        bundle = Path(args.bundle).resolve()
        if args.command == "import-response":
            raw = Path(args.response).read_bytes()
        else:
            verify_bundle(bundle)
            if (bundle / "candidate.json").exists() or (bundle / "response.raw.json").exists():
                raise ValueError("response already exists; no additional API request sent")
            if urllib.parse.urlsplit(args.endpoint).scheme not in ("http", "https"):
                raise ValueError("endpoint must be an HTTP or HTTPS chat/completions URL")
            headers = {"Content-Type": "application/json"}
            key = os.environ.get(args.api_key_env)
            if key:
                headers["Authorization"] = "Bearer " + key
            request = urllib.request.Request(args.endpoint, data=(bundle / "request.json").read_bytes(), headers=headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read()
        value = import_response(bundle, raw)
    print(json.dumps({"command": args.command, "schema": value["schema"],
                      "semantic_approval": value.get("semantic_approval", False)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
