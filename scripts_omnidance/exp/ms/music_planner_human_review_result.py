"""Validate actual offline five-axis human choices for the frozen three-case pack.

CPU/stdlib only. No rendering, services, pairwise inference, or automatic verdict.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

PUBLIC_MANIFEST = Path("/apdcephfs_gy4/share_302533218/cedricnie/planner_h3_review_offline_v2/manifest.json")
PUBLIC_MANIFEST_SHA = "ebfde5de14ee50b936d1dfced6d47e666f269164c359d38c0d28639abe3669da"
PRIVATE_KEY = Path("/apdcephfs_gy4/share_302533218/cedricnie/planner_h3_review_offline_v2.private-key.json")
PRIVATE_KEY_SHA = "b404c15576ece3694bcfbe301c94cabaeb6dbacf513e8437d1043bfe0d2bb1a6"
BUNDLE_ID = "b0cfae50578158ca66d1cecccd49aba6"
CASES = ("case1", "case2", "case3")
AXES = ("clarity", "continuity", "choreography", "amplitude", "music_response")
FAMILIES = ("music_flamingo", "moss", "omni_captioner")
CHOICES = {"A", "B", "C", "similar", "all_unusable", "unknown", ""}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON value")


def read_snapshot(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "canonical nonsymlink input required")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "regular JSON input required")
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        require(identity(before) == identity(os.fstat(stream.fileno())), "input changed before read")
        data = stream.read()
        require(identity(before) == identity(os.fstat(stream.fileno())), "input changed during read")
    require(identity(before) == identity(path.stat()), "input changed during read")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_keys, parse_constant=reject_constant)
    return value, hashlib.sha256(data).hexdigest()


def fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value)
            and set(value) <= set(required) | set(optional), "invalid JSON object fields")


def case_index(value):
    require(type(value) is list and len(value) == 3, "exact three cases required")
    result = {}
    for case in value:
        require(type(case) is dict and type(case.get("case")) is str
                and case["case"] in CASES and case["case"] not in result,
                "exact distinct anonymous cases required")
        result[case["case"]] = case
    require(set(result) == set(CASES), "exact three cases required")
    return result


def video_index(value):
    require(type(value) is list and len(value) == 3, "exact three candidate labels required")
    result = {}
    for video in value:
        require(type(video) is dict and type(video.get("label")) is str
                and video["label"] in ("A", "B", "C") and video["label"] not in result,
                "distinct ABC labels required")
        result[video["label"]] = video
    return result


def reference_sha(value):
    require(type(value) is dict and type(value.get("sha256")) is str
            and re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is not None,
            "valid SHA reference required")
    return value["sha256"]


def frozen_mapping(public, private):
    require(type(public) is dict and type(private) is dict, "manifest objects required")
    require(public.get("schema") == "anonymous-offline-video-review-v1"
            and private.get("schema") == "private-offline-video-review-key-v1", "manifest schema mismatch")
    require(public.get("bundle_id") == BUNDLE_ID and private.get("bundle_id") == BUNDLE_ID,
            "manifest bundle mismatch")
    require(public.get("axes") == list(AXES), "manifest axes mismatch")
    public_cases, private_cases = case_index(public.get("cases")), case_index(private.get("cases"))
    mappings = {}
    for alias in CASES:
        public_case, private_case = public_cases[alias], private_cases[alias]
        public_videos = video_index(public_case.get("videos"))
        private_videos = video_index(private_case.get("videos"))
        mapping = {}
        for label in ("A", "B", "C"):
            shown, hidden = public_videos[label], private_videos[label]
            require(shown.get("file") == alias + "/" + label + ".mp4", "anonymous video file mismatch")
            require(reference_sha(shown) == reference_sha(hidden.get("copy")), "candidate copy SHA mismatch")
            require(type(hidden.get("family")) is str and hidden["family"] in FAMILIES,
                    "candidate family mismatch")
            mapping[label] = hidden["family"]
        require(set(mapping.values()) == set(FAMILIES), "exact three distinct families required")
        audio = public_case.get("original_audio")
        require(type(audio) is dict and audio.get("file") == alias + "/music.wav", "anonymous audio file mismatch")
        require(reference_sha(audio) == reference_sha(private_case.get("original_audio_copy"))
                == reference_sha(private_case.get("audio_ref")), "audio copy SHA mismatch")
        mappings[alias] = mapping
    return mappings


def choices_report(scores, mapping, hashes):
    fields(scores, ("schema", "bundle_id", "downloaded_at", "cases"), ("reviewer",))
    require(scores["schema"] == "anonymous-offline-human-scores-v1", "score schema mismatch")
    require(scores["bundle_id"] == BUNDLE_ID, "score bundle mismatch")
    require(type(scores["downloaded_at"]) is str and bool(scores["downloaded_at"].strip()),
            "downloaded_at string required")
    require(scores.get("reviewer") is None or type(scores["reviewer"]) is str, "invalid reviewer type")
    indexed = case_index(scores["cases"])
    cases = []
    counts = {axis: {"prefer_named_family": {family: 0 for family in FAMILIES},
                     "similar": 0, "all_unusable": 0, "unknown": 0, "unanswered": 0}
              for axis in AXES}
    answered = 0
    for alias in CASES:
        case = indexed[alias]
        fields(case, ("case", "axes", "reason"))
        fields(case["axes"], AXES)
        require(case["reason"] is None or type(case["reason"]) is str, "invalid reason type")
        axes = {}
        for axis in AXES:
            value = case["axes"][axis]
            require(value is None or (type(value) is str and value in CHOICES), "invalid axis choice")
            value = None if value == "" else value
            answered += value is not None
            family = mapping[alias][value] if value in ("A", "B", "C") else None
            choice = "prefer_named_family" if family is not None else value
            axes[axis] = {"choice": choice, "prefer_named_family": family}
            if family is not None:
                counts[axis]["prefer_named_family"][family] += 1
            else:
                counts[axis]["unanswered" if value is None else value] += 1
        cases.append({"case": alias, "axes": axes, "reason": case["reason"]})
    require(answered > 0, "unfilled score template is not a human review")
    return {"schema": "offline-human-review-result-v1", "bundle_id": BUNDLE_ID,
            "source_sha256": hashes, "human_review_complete": answered == len(CASES) * len(AXES),
            "answered_axis_choices": answered, "expected_axis_choices": len(CASES) * len(AXES),
            "choice_semantics": "independent_three_candidate_preferences_not_pairwise_or_accuracy",
            "human_quality_accepted": False, "requires_manual_synthesis": True,
            "cases": cases, "axis_choice_counts": counts}


def ingest(scores_path, out_path):
    scores_path, out_path = Path(scores_path).absolute(), Path(out_path).absolute()
    require(out_path == out_path.resolve() and out_path.parent.is_dir(), "fresh canonical output parent required")
    require(not os.path.lexists(out_path), "output exists; no overwrite")
    require(out_path != scores_path and not out_path.is_relative_to(PUBLIC_MANIFEST.parent),
            "output must not replace input or alter frozen public pack")
    require(PRIVATE_KEY is not None and PRIVATE_KEY_SHA is not None, "frozen private key not configured")
    paths = {"scores": scores_path, "public_manifest": PUBLIC_MANIFEST, "private_key": PRIVATE_KEY}
    snapshots = {name: read_snapshot(path) for name, path in paths.items()}
    require(snapshots["public_manifest"][1] == PUBLIC_MANIFEST_SHA, "frozen public manifest SHA mismatch")
    require(snapshots["private_key"][1] == PRIVATE_KEY_SHA, "frozen private key SHA mismatch")
    mapping = frozen_mapping(snapshots["public_manifest"][0], snapshots["private_key"][0])
    hashes = {name: snapshot[1] for name, snapshot in snapshots.items()}
    report = choices_report(snapshots["scores"][0], mapping, hashes)
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    for name, path in paths.items():
        require(read_snapshot(path)[1] == hashes[name], "input snapshot SHA changed before output")
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = ingest(args.scores, args.out)
    except (OSError, ValueError, RuntimeError):
        print("Human review result not completed; input gate/output failed. No human quality verdict inferred.",
              file=sys.stderr)
        return 1
    print(json.dumps({"report_written": True, "human_review_complete": report["human_review_complete"],
                      "human_quality_accepted": False, "requires_manual_synthesis": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
