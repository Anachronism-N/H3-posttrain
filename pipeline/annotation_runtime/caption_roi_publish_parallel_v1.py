"""Independent parallel ROI publication, preserving native durability and gates.

Only disjoint asset writes use four threads. Failed publications retain their
exclusive owned directory and never create the native completion marker.
No producer integration or global Path replacement is performed here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
import curated_producer_io_v2 as base

PUBLICATION_CODE = {"path": str(Path(__file__).resolve()),
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def publish_roi(roi, local, final, prepared, extraction_seconds):
    started = time.monotonic()
    local, final = Path(local), Path(final)
    if local.is_symlink():
        raise ValueError("Local ROI root is a symlink")
    local = local.resolve(strict=True)
    if final.is_symlink() or final.exists():
        raise FileExistsError("Refusing any existing ROI directory: " + str(final))
    final = final.parent.resolve() / final.name
    roi.verify_identity(prepared["source"])
    roi.verify_model(prepared["model_ref"])
    manifest = roi.validate_completed(local, prepared)
    source_bytes = (local / "source_manifest.json").read_bytes()
    if json.loads(source_bytes) != prepared:
        raise ValueError("Local source manifest differs from original source binding")
    assets = [manifest["raw"]] + manifest["crops"]
    allowed = {"source_manifest.json", "prepare_receipt.json", "raw_wholebody134.npz",
               "roi_manifest.json", "extract_receipt.json"}
    expected = {}
    for asset in assets:
        path = Path(asset["path"])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("ROI asset is not canonical")
        relative = path.relative_to(local)
        if asset is manifest["raw"]:
            if relative != Path("raw_wholebody134.npz"):
                raise ValueError("Unexpected raw ROI path")
        elif relative.parent != Path("crops") or relative.suffix != ".jpg":
            raise ValueError("Crop escaped source directory")
        key = relative.as_posix()
        if key in expected:
            raise ValueError("Duplicate ROI asset path")
        allowed.add(key)
        # Native local full validation has already read and verified these bytes.
        expected[key] = asset["sha256"]
    actual = set()
    for path in local.rglob("*"):
        if path.is_symlink() or (path.is_dir() and path != local / "crops"):
            raise ValueError("Unexpected local directory/symlink")
        if path.is_file():
            actual.add(path.relative_to(local).as_posix())
    if actual != allowed:
        raise ValueError("Unexpected/missing local ROI files")
    mapped = base.remap_paths(manifest, local, final)
    for key in ("source", "video_sha256", "binding_sha256", "model_ref", "probe", "selected_frames"):
        if mapped[key] != manifest[key]:
            raise ValueError("Publication changed source evidence: " + key)
    manifest_bytes = roi.json_bytes(mapped)
    receipt = json.loads((local / "extract_receipt.json").read_bytes())
    receipt["roi_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    receipt_bytes = roi.json_bytes(receipt)
    expected.update({"source_manifest.json": hashlib.sha256(source_bytes).hexdigest(),
                     "prepare_receipt.json": roi.sha256(local / "prepare_receipt.json"),
                     "roi_manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
                     "extract_receipt.json": hashlib.sha256(receipt_bytes).hexdigest()})
    if expected["source_manifest.json"] != manifest["source_manifest"]["sha256"]:
        raise ValueError("Local source manifest changed after native validation")
    preflight_seconds = time.monotonic() - started
    shared_started = time.monotonic()
    final.parent.mkdir(parents=True, exist_ok=True)
    final.mkdir(exist_ok=False)  # Only claim: never rename, reuse, or delete final.
    owner = uuid.uuid4().hex
    ownership = {"schema": "curated_producer_io_v4_ownership", "owner_token": owner,
                 "owner_pid": os.getpid(), "source_id": prepared["id"],
                 "source_ordinal": prepared["input_row"]["manifest_record"]["source_ordinal"],
                 "source_binding_sha256": prepared["binding_sha256"],
                 "video_sha256": prepared["video_sha256"],
                 "roi_model_sha256": prepared["model_ref"]["sha256"],
                 "source_manifest_sha256": expected["source_manifest.json"],
                 "roi_manifest_sha256": expected["roi_manifest.json"],
                 "shared_roi_directory": str(final),
                 "expected_assets": [{"relative_path": key, "sha256": expected[key]} for key in sorted(allowed)],
                 "whole_directory_atomic": False, "exclusive_directory": True,
                 "commit_marker": "extract_receipt.json",
                 "consumer_gate": "producer.terminal.json status=ready after native full validation",
                 "partial_failure_policy": "retain_final_directory_refuse_incomplete_resume",
                 "semantic_approval": False, "formal_training": False}
    publication_code = PUBLICATION_CODE
    ownership["publication_code"] = publication_code
    ownership["asset_write_max_workers"] = 4
    written_bytes, in_write_hash_bytes, direct_writes = 0, 0, 0

    def write_owned(relative, raw=None, expected_sha=None):
        hasher = hashlib.sha256()
        byte_count = 0
        with (final / relative).open("xb") as output:
            if raw is not None:
                chunks = iter((raw,))
                source = None
            else:
                fd = os.open(local / relative, os.O_RDONLY | os.O_NOFOLLOW)
                source = os.fdopen(fd, "rb")
                chunks = iter(lambda: source.read(1024 * 1024), b"")
            try:
                for chunk in chunks:
                    if output.write(chunk) != len(chunk):
                        raise OSError("Incomplete shared ROI write: " + relative)
                    hasher.update(chunk)
                    byte_count += len(chunk)
            finally:
                if source is not None:
                    source.close()
            output.flush()
            os.fsync(output.fileno())
        if expected_sha is not None and hasher.hexdigest() != expected_sha:
            raise ValueError("Shared streamed content SHA mismatch: " + relative)
        return byte_count, byte_count, 1

    def aggregate(counters):
        nonlocal written_bytes, in_write_hash_bytes, direct_writes
        written, hashed, opens = counters
        written_bytes += written
        in_write_hash_bytes += hashed
        direct_writes += opens

    aggregate(write_owned("io_publish_ownership.json", roi.json_bytes(ownership)))
    base.fsync_directory(final)
    base.fsync_directory(final.parent)
    (final / "crops").mkdir(exist_ok=False)
    # Workers own disjoint xb files; only the coordinator mutates counters.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_owned, relative, expected_sha=expected[relative])
                   for relative in sorted(allowed - {"roi_manifest.json", "extract_receipt.json"})]
        for future in as_completed(futures):
            aggregate(future.result())
    base.fsync_directory(final / "crops")
    base.fsync_directory(final)
    base.fsync_directory(final.parent)
    copied_seconds = time.monotonic() - shared_started
    roi.verify_identity(prepared["source"])
    roi.verify_model(prepared["model_ref"])
    aggregate(write_owned("roi_manifest.json", manifest_bytes, expected["roi_manifest.json"]))
    base.fsync_directory(final)
    evidence = {"schema": "curated_producer_io_v4", "owner_token": owner,
                "source_id": prepared["id"], "source_ordinal": ownership["source_ordinal"],
                "source_binding_sha256": prepared["binding_sha256"],
                "roi_model_sha256": prepared["model_ref"]["sha256"],
                "source_manifest_sha256_unchanged": expected["source_manifest.json"],
                "raw_sha256_unchanged": expected["raw_wholebody134.npz"],
                "output_path_mapping": {"local_roi_directory": str(local), "shared_roi_directory": str(final)},
                "source_media_staged": False, "source_manifest_rewritten": False,
                "all_shared_assets_sha256": expected,
                "local_extract_seconds": extraction_seconds,
                "local_preflight_seconds": preflight_seconds,
                "shared_copy_verify_seconds": copied_seconds,
                "shared_publish_seconds": time.monotonic() - shared_started,
                "shared_publish_timer_end": "before io receipt write, last extract receipt commit and native full validation",
                "publisher_pre_receipt_counters": {"shared_direct_xb_opens": direct_writes,
                    "shared_file_fsync": direct_writes, "shared_asset_link": 0, "shared_asset_unlink": 0,
                    "shared_rehash_read_bytes": 0, "written_bytes": written_bytes,
                    "in_write_hash_bytes": in_write_hash_bytes},
                "counter_scope": "actual publisher operations before this receipt; excludes native fullvalidation reads and last receipt write_new",
                "whole_directory_atomic": False, "per_file_shared_fsync": True,
                "publication_protocol": "exclusive_mkdir_owned_xb_parallel4_stream_sha_last_native_write_new_extract_receipt_v1",
                "publication_code": publication_code, "asset_write_max_workers": 4,
                "local_complete_roi_validated": True, "complete_roi_validated": False,
                "final_validation_gate": "native validate_completed after last extract receipt and directory fsync",
                "semantic_approval": False, "human_review_completed": False, "formal_training": False}
    write_owned("io_publish_receipt.json", roi.json_bytes(evidence))
    base.fsync_directory(final)
    # Original atomic no-replace file commit; partial assets never imply completion.
    roi.write_new(final / "extract_receipt.json", receipt_bytes)
    base.fsync_directory(final)
    base.fsync_directory(final.parent)
    return roi.validate_completed(final, prepared)

