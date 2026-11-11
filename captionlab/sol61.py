"""Prepare source-bound sol61 baselines without loading models or exposing targets.

Published split/group checks concern declared IDs and video hashes, not verified
song identity or the membership of an independently running training job.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from .io import file_sha, loads, object_sha, read_json, text_sha, write_json
from choreobench.io import file_lock, read_json_snapshot, write_json as atomic_json

REPO = "avagelion/dance-data"
SUBDIR = "sol61_captions"
REVISION = "1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a"
SPLITS = ("train", "val", "test")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
DOWNLOAD_SCHEMA = "captionlab.sol61_download.v1"
_SOURCE_FILES = tuple(SUBDIR + "/" + name for name in
    ("README.md", "summary.json", "train_captions.jsonl", "val_captions.jsonl", "test_captions.jsonl"))
_MEDIA_FILES = tuple("ms_original_planner_dataset_v1/" + split + ".jsonl" for split in ("val", "test"))


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError("Require an immutable HF commit, not main")
    return value


def _safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Metadata path escapes its download root")
    return path


def _receipt(path, revision, *, verify_files=True):
    value, digest = read_json_snapshot(path)
    legacy = isinstance(value, dict) and set(value) == {"repo_id", "revision", "files"}
    if (not isinstance(value, dict) or (not legacy and (
            set(value) != {"schema", "repo_id", "revision", "files"} or value.get("schema") != DOWNLOAD_SCHEMA))
            or value.get("repo_id") != REPO or value.get("revision") != revision
            or not isinstance(value.get("files"), dict)):
        raise ValueError("Download receipt does not bind the expected repo/revision")
    for name, expected in value["files"].items():
        if name not in _SOURCE_FILES + _MEDIA_FILES or not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError("Invalid download receipt file identity")
        target = _safe_path(path.parent, name)
        if verify_files and (not target.is_file() or file_sha(target) != expected):
            raise ValueError("Downloaded metadata changed or is missing")
    return value, digest, legacy


def _provenance(source_dir, revision, source_files):
    root = Path(source_dir).resolve()
    candidates = [p for p in (root / "download_receipt.json", root.parent / "download_receipt.json") if p.is_file()]
    if len(candidates) > 1:
        raise ValueError("Ambiguous download receipt locations")
    result = dict(repo_id=REPO, revision=revision, revision_status="declared_revision_only",
        receipt_path=None, receipt_sha256=None, legacy_receipt=None,
        independent_remote_revision_verification=False,
        verification_scope="operator-declared commit; local source hashes only")
    if not candidates:
        return result
    path = candidates[0]
    receipt, digest, legacy = _receipt(path, revision)
    for identity in source_files.values():
        if receipt["files"].get(identity["path"]) != identity["sha256"]:
            raise ValueError("Audited captions are not bound by download receipt")
    if file_sha(path) != digest:
        raise ValueError("Download receipt changed during audit")
    result.update(revision_status="download_receipt_verified", receipt_path=str(path),
        receipt_sha256=digest, legacy_receipt=legacy,
        verification_scope="local files match repo/revision declared in receipt; not independent remote authenticity")
    return result


def _rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at line {number}")
                yield value


def _validate_row(row, split):
    for field in ("id", "source_group_id", "motion_caption", "dance_style", "caption_version"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("Missing sol61 field: " + field)
    if not _ID.fullmatch(row["id"]) or row.get("split") != split:
        raise ValueError("Invalid case ID or file/split mismatch")
    if type(row.get("gate")) is not bool or not isinstance(row.get("gestures"), list):
        raise ValueError("Require explicit gate and source gesture provenance")
    header = "Dance style: " + row["dance_style"]
    first, separator, body = row["motion_caption"].partition("\n")
    if first.rstrip("\r") != header or not separator or not body.strip():
        raise ValueError("Caption must preserve its declared style header and nonempty body")


def audit_sources(source_dir, *, revision=REVISION):
    """Scan all three files; retain only compact identity metadata in memory."""
    _revision(revision)
    root = Path(source_dir)
    ids, groups, paths, reports = {}, defaultdict(set), {}, {}
    metadata = {}
    for split in SPLITS:
        path = root / (split + "_captions.jsonl")
        seen, styles, gates, versions, group_set = set(), Counter(), Counter(), Counter(), set()
        rows = []
        before = file_sha(path)
        for row in _rows(path):
            _validate_row(row, split)
            ident, group = row["id"], row["source_group_id"]
            if ident in seen:
                raise ValueError("Duplicate ID within a split")
            seen.add(ident)
            ids.setdefault(ident, set()).add(split)
            groups[group].add(split)
            group_set.add(group)
            styles[row["dance_style"]] += 1
            gates[str(row["gate"]).lower()] += 1
            versions[row["caption_version"]] += 1
            rows.append({"id": ident, "source_group_id": group, "gate": row["gate"]})
        if file_sha(path) != before:
            raise ValueError("Caption source changed during audit")
        paths[split] = {"path": SUBDIR + "/" + path.name, "sha256": before,
                        "bytes": path.stat().st_size}
        metadata[split] = rows
        reports[split] = {"records": len(rows), "source_groups": len(group_set),
                          "gate": dict(gates), "styles": dict(styles), "versions": dict(versions)}
    overlap_ids = {ident for ident, splits in ids.items() if len(splits) > 1}
    overlap_groups = {group for group, splits in groups.items() if len(splits) > 1}
    report = {
        "schema": "captionlab.sol61_audit.v1", "splits": reports, "source_files": paths,
        "cross_split_id_count": len(overlap_ids), "cross_split_group_count": len(overlap_groups),
        "group_semantics": "declared source_group_id; published original records use video_sha256",
        "song_or_near_duplicate_isolation_verified": False,
        "actual_training_membership_verified": False,
        "reference_scope": "text-only rewrite; image-space wrists/elbows/knees; no action timestamps",
        "automatic_gate_is_human_quality": False,
        "provenance": _provenance(root, revision, paths),
    }
    return report, metadata, overlap_ids, overlap_groups


def freeze_cases(source_dir, *, split, limit=12, seed=42, gate_policy="all", revision=REVISION):
    if (split not in ("val", "test") or type(limit) is not int or limit < 1 or type(seed) is not int
            or not 0 <= seed < 2 ** 63):
        raise ValueError("Select a positive number of validation or test cases")
    if gate_policy not in ("all", "pass"):
        raise ValueError("Require explicit gate policy and immutable HF commit")
    _revision(revision)
    report, metadata, bad_ids, bad_groups = audit_sources(source_dir, revision=revision)
    eligible = [r for r in metadata[split] if r["id"] not in bad_ids
                and r["source_group_id"] not in bad_groups
                and (gate_policy == "all" or r["gate"])]
    ordered = sorted(eligible, key=lambda row: text_sha(f"{seed}:{split}:{row['id']}"))
    chosen, groups = [], set()
    for row in ordered:
        if row["source_group_id"] not in groups:
            chosen.append(row)
            groups.add(row["source_group_id"])
        if len(chosen) == limit:
            break
    if len(chosen) != limit:
        raise ValueError("Insufficient distinct eligible declared source groups; no split fallback")
    selected = {r["id"] for r in chosen}
    refs = {}
    path = Path(source_dir) / (split + "_captions.jsonl")
    for row in _rows(path):
        if row["id"] in selected:
            refs[row["id"]] = {
                "case_id": row["id"], "source_group_id": row["source_group_id"], "split": split,
                "raw_caption": row["motion_caption"],
                "action_reference": row["motion_caption"].partition("\n")[2],
                "raw_caption_sha256": text_sha(row["motion_caption"]),
                "source_row_sha256": object_sha(row), "dance_style": row["dance_style"],
                "gate": row["gate"], "gestures": copy.deepcopy(row["gestures"]),
                "caption_version": row["caption_version"], "video_path": row.get("video_path"),
                "reference_is_unique_choreography_truth": False,
                "reference_timestamps_available": False,
            }
    if file_sha(path) != report["source_files"][split]["sha256"]:
        raise ValueError("Caption source changed after selection")
    dataset = {"repo_id": REPO, "revision": revision, "subdir": SUBDIR,
               "source_files": report["source_files"], "audit": report,
               "provenance": copy.deepcopy(report["provenance"]), "media_bound": False}
    manifest = {"schema": "captionlab.baseline_manifest.v1", "dataset": dataset,
                "selection": {"split": split, "seed": seed, "gate_policy": gate_policy,
                              "limit": limit, "eligible_records": len(eligible),
                              "policy": "sha256 order; one per declared video group; cross-split groups excluded"},
                "cases": [{"case_id": row["id"], "source_group_id": row["source_group_id"], "split": split,
                           "audio": {"path": None, "sha256": None, "sample_rate": None, "num_samples": None},
                           "duration_seconds": 5.0625} for row in chosen]}
    private = {"schema": "captionlab.sol61_references.v1", "selection_sha256": object_sha(manifest["selection"]),
               "source_files": report["source_files"], "records": [refs[r["id"]] for r in chosen]}
    return manifest, private, report


def bind_audio(manifest, media_manifest, audio_root):
    """Join the original media identity by ID, then inspect actual local WAV bytes."""
    from choreobench.audio import inspect_audio
    if (not isinstance(manifest, dict) or manifest.get("schema") != "captionlab.baseline_manifest.v1"
            or set(manifest) - {"schema", "dataset", "selection", "cases", "synthetic"}
            or not isinstance(manifest.get("dataset"), dict) or not isinstance(manifest.get("selection"), dict)
            or not isinstance(manifest.get("cases"), list) or not manifest["cases"]
            or type(manifest.get("synthetic", False)) is not bool):
        raise ValueError("Expected baseline manifest")
    for case in manifest["cases"]:
        if (not isinstance(case, dict) or set(case) != {"case_id", "source_group_id", "split", "audio", "duration_seconds"}
                or not isinstance(case["case_id"], str) or not _ID.fullmatch(case["case_id"])
                or not isinstance(case["source_group_id"], str) or not case["source_group_id"].strip()
                or case["split"] not in ("val", "test") or case["split"] != manifest["selection"].get("split")
                or type(case["duration_seconds"]) not in (int, float) or case["duration_seconds"] != 5.0625
                or not isinstance(case["audio"], dict) or set(case["audio"]) != {"path", "sha256", "sample_rate", "num_samples"}):
            raise ValueError("Invalid public case; no private references or inferred clocks allowed")
    result = copy.deepcopy(manifest)
    cases = {r["case_id"]: r for r in result["cases"]}
    if len(cases) != len(result["cases"]):
        raise ValueError("Duplicate case")
    path = Path(media_manifest)
    before = file_sha(path)
    found = {}
    for row in _rows(path):
        if row.get("id") in cases:
            if row["id"] in found:
                raise ValueError("Duplicate media identity")
            found[row["id"]] = row
    if set(found) != set(cases) or file_sha(path) != before:
        raise ValueError("Missing media identities or changed media manifest")
    root = Path(audio_root).resolve()
    seen_pcm = set()
    for ident, case in cases.items():
        if not _ID.fullmatch(ident):
            raise ValueError("Unsafe media ID")
        row = found[ident]
        if row.get("split") != case["split"] or row.get("source_group_id") != case["source_group_id"]:
            raise ValueError("Caption and media source/split disagree")
        if row.get("audio_window_seconds") != [0.0, 5.0625] or row.get("audio_duration_seconds") != 5.0625:
            raise ValueError("Media does not declare the complete original 5.0625-second view")
        audio = inspect_audio(_safe_path(root, ident + ".wav"))
        if not _SHA.fullmatch(str(row.get("audio_sha256", ""))) or audio["sha256"] != row["audio_sha256"]:
            raise ValueError("Local audio differs from the original media SHA256")
        if audio["num_samples"] != int(5.0625 * audio["sample_rate"] + .5):
            raise ValueError("Unexpected audio duration; no implicit crop, resample or padding")
        for key in ("path", "sha256", "sample_rate", "num_samples"):
            declared = case["audio"][key]
            if declared is not None and (type(declared) is not type(audio[key]) or declared != audio[key]):
                raise ValueError("Previously bound audio identity or clock disagrees with local WAV")
        pcm = text_sha(str(audio["sample_rate"]) + ":" + hashlib.sha256(audio["pcm"]).hexdigest())
        if pcm in seen_pcm:
            raise ValueError("Selected cases contain identical PCM; revise selection explicitly")
        seen_pcm.add(pcm)
        case["audio"] = {k: audio[k] for k in ("path", "sha256", "sample_rate", "num_samples")}
    if file_sha(path) != before:
        raise ValueError("Media manifest changed during audio binding")
    result["dataset"].update(media_bound=True, media_manifest_sha256=before,
                             media_manifest_path=str(path.resolve()), selected_pcm_duplicates=0)
    return result


def fetch_sources(output_dir, revision=REVISION, include_media=False):
    """Resume pinned metadata downloads; never trust an unreceipted local file.

    A complete staged file is receipted before publication. An interrupted stream
    restarts only that file; unreceipted existing files require a matching fresh
    pinned download before adoption. No existing metadata is overwritten.
    """
    import requests
    _revision(revision)
    if type(include_media) is not bool:
        raise ValueError("include_media must be boolean")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = list(_SOURCE_FILES)
    if include_media:
        files += list(_MEDIA_FILES)
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    receipt_path = root / "download_receipt.json"
    with file_lock(root / ".download.lock"):
        if receipt_path.exists():
            receipt, _, _ = _receipt(receipt_path, revision, verify_files=False)
            # Recover a crash between the atomic receipt and final publication.
            for name, digest in receipt["files"].items():
                target = _safe_path(root, name)
                staged = _safe_path(root, name + ".partial")
                if target.is_file():
                    if file_sha(target) != digest:
                        raise ValueError("Downloaded metadata changed")
                elif staged.is_file() and file_sha(staged) == digest:
                    os.link(str(staged), str(target))
                    staged.unlink()
                else:
                    raise ValueError("Receipted metadata and its complete staged file are missing or changed")
        else:
            receipt = {"schema": DOWNLOAD_SCHEMA, "repo_id": REPO, "revision": revision, "files": {}}
        for name in files:
            if name in receipt["files"]:
                continue
            target = _safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = _safe_path(root, name + ".partial")
            # A prior unsealed stream has no trusted identity and is restarted.
            if temporary.exists():
                temporary.unlink()
            try:
                # requests strips Authorization on a cross-host CDN redirect.
                with requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}",
                                  headers=headers, stream=True, timeout=(20, 120)) as response:
                    if response.status_code != 200:
                        raise ValueError("HF metadata HTTP status " + str(response.status_code))
                    with temporary.open("xb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                        handle.flush(); os.fsync(handle.fileno())
            except requests.RequestException as error:
                # Never serialize headers, request objects, tokens or CDN URLs.
                raise RuntimeError("HF metadata request failed: " + type(error).__name__) from None
            digest = file_sha(temporary)
            if target.exists() and (not target.is_file() or file_sha(target) != digest):
                raise FileExistsError("Unreceipted existing metadata differs from fresh pinned download")
            receipt["files"][name] = digest
            atomic_json(receipt_path, receipt, overwrite=True)
            if not target.exists():
                os.link(str(temporary), str(target))
            temporary.unlink()
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--out-dir", required=True, type=Path)
    fetch.add_argument("--revision", default=REVISION)
    fetch.add_argument("--include-media", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--source-dir", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--revision", default=REVISION)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-dir", type=Path, required=True)
    freeze.add_argument("--split", choices=("val", "test"), required=True)
    freeze.add_argument("--limit", type=int, default=12)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--gate-policy", choices=("all", "pass"), default="all")
    freeze.add_argument("--revision", default=REVISION)
    freeze.add_argument("--out-dir", type=Path, required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--manifest", type=Path, required=True)
    bind.add_argument("--media-manifest", type=Path, required=True)
    bind.add_argument("--audio-root", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = fetch_sources(args.out_dir, args.revision, args.include_media)
        print(json.dumps({"revision": result["revision"], "metadata_files": len(result["files"])}))
    elif args.command == "audit":
        report, _, _, _ = audit_sources(args.source_dir, revision=args.revision)
        write_json(args.out, report)
        print(json.dumps({"splits": {k: v["records"] for k, v in report["splits"].items()},
                          "cross_split_groups": report["cross_split_group_count"]}))
    elif args.command == "freeze":
        manifest, refs, report = freeze_cases(args.source_dir, split=args.split, limit=args.limit,
            seed=args.seed, gate_policy=args.gate_policy, revision=args.revision)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.out_dir / "manifest.pending.json", manifest)
        write_json(args.out_dir / "references.private.json", refs)
        write_json(args.out_dir / "audit.json", report)
        print(json.dumps({"split": args.split, "cases": len(manifest["cases"]), "media_bound": False}))
    else:
        result = bind_audio(read_json(args.manifest), args.media_manifest, args.audio_root)
        write_json(args.out, result)
        print(json.dumps({"cases": len(result["cases"]), "media_bound": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
