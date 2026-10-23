"""Build a separate, provenance-bound MotionScript optimization dataset.

Requires NumPy and SciPy. No GPU, model download, training or LLM request occurs.
Optional original MotionScript execution additionally requires its local source,
import shim and PyTorch. See motionscript_v2/README.md before dataset admission.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile

import numpy as np
import scipy

from motionscript_v2 import SCHEMA
from motionscript_v2.coordinates import preprocess
from motionscript_v2.events import extract_events, make_llm_payload, render_caption


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    # Temporary file and destination are on the same filesystem.
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_npz(path: Path, arrays: dict) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write(path, buffer.getvalue())


def overlaps(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def check_output_path(out: Path, nlf: Path, template: Path, ids: Path | None) -> None:
    # resolve() also catches symlink / junction aliases on supported platforms.
    source_root = nlf if nlf.is_dir() else nlf.parent
    if overlaps(out, source_root):
        raise ValueError("--out must be a separate directory outside the NLF input tree")
    for source in (template, ids):
        if source is not None and (out.resolve() == source.resolve() or out.resolve() in source.resolve().parents):
            raise ValueError("--out must not contain the template or ID input")


def input_files(nlf: Path, ids: Path | None, limit: int, *, ids_content: bytes | None = None) -> list[Path]:
    if not nlf.exists():
        raise ValueError(f"NLF input does not exist: {nlf}")
    files = sorted(nlf.glob("*.npz")) if nlf.is_dir() else [nlf]
    files = [p for p in files if p.suffix == ".npz" and ".tmp" not in p.name]
    if ids is not None:
        text = (ids.read_bytes() if ids_content is None else ids_content).decode("utf-8-sig")
        requested = [line.strip() for line in text.splitlines() if line.strip()]
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate IDs in --ids")
        missing = sorted(set(requested) - {p.stem for p in files})
        if missing:
            raise ValueError(f"requested IDs have no NLF NPZ: {missing[:10]}")
        files = [p for p in files if p.stem in set(requested)]
    if limit:
        files = files[:limit]
    if not files:
        raise ValueError("no matching NLF NPZ files")
    seen = set()
    for path in files:
        sid = path.stem
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid) or sid.casefold() in {"protocol", "manifest"}:
            raise ValueError(f"SID must be a portable bare identifier, not a reserved artifact name: {sid}")
        if sid.casefold() in seen:
            raise ValueError(f"case-insensitive SID collision: {sid}")
        seen.add(sid.casefold())
    return files


def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), *sorted((here / "motionscript_v2").glob("*.py"))]
    return {str(p.relative_to(here)).replace("\\", "/"): sha256(p) for p in paths}


def upstream_hashes(src: Path | None, shim: Path | None) -> dict[str, str]:
    result = {}
    for label, root in (("source", src), ("shim", shim)):
        if root is None:
            continue
        if not root.is_dir():
            raise ValueError(f"MotionScript {label} is not a directory: {root}")
        paths = sorted(root.rglob("*.py"))
        if not paths:
            raise ValueError(f"MotionScript {label} contains no Python files: {root}")
        for path in paths:
            result[f"{label}/{path.relative_to(root).as_posix()}"] = sha256(path)
    return result


def original_captions(arrays: dict, src: Path, shim: Path | None) -> list[dict]:
    """Run unchanged upstream rules on each contiguous, fully usable interval.

    A gap is never deleted and stitched into an artificial continuous motion.
    The regression suite checks this call contract with a stub, not a real
    upstream installation or real PyTorch tensors.
    """
    for root in (src, shim):
        if root is not None:
            sys.path.insert(0, str(root.resolve()))
    import torch
    from text2pose.posescript import captioning

    if not any(Path(captioning.__file__).resolve().is_relative_to(root.resolve()) for root in (src, shim) if root):
        raise ValueError("imported MotionScript is outside the specified source/shim")
    usable = arrays["valid"] & arrays["joint_valid"].all(axis=1)
    root_segments = arrays.get("segment_id", np.zeros(len(usable), dtype=np.int32))
    usable &= root_segments >= 0
    same = root_segments[1:] == root_segments[:-1]
    if "joint_segment_id" in arrays:
        joint_segments = arrays["joint_segment_id"]
        usable &= (joint_segments >= 0).all(axis=1)
        same &= (joint_segments[1:] == joint_segments[:-1]).all(axis=1)
    starts = np.flatnonzero(usable & ~np.r_[False, usable[:-1] & same])
    stops = np.flatnonzero(usable & ~np.r_[usable[1:] & same, False]) + 1
    segments = []
    for start, stop in zip(starts, stops):
        if stop - start < 4:
            continue
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # Upstream writes auxiliary files; keep these in an isolated temporary tree.
        with tempfile.TemporaryDirectory(prefix="motionscript_v2_") as temp:
            with contextlib.redirect_stdout(io.StringIO()):
                result = captioning.main(
                    torch.as_tensor(arrays["coords"][start:stop], dtype=torch.float32),
                    save_dir=temp, babel_info=False, simplified_captions=False,
                    apply_transrel_ripple_effect=False, apply_stat_ripple_effect=False,
                    random_skip=False, motion_tracking=True, verbose=False,
                )
        _, _, nonaggregated, aggregated = result
        segments.append({"start": float(arrays["time"][start]),
                         "end": float(arrays["time"][stop - 1]),
                         "source": "original_motionscript_rules_on_v2_coordinates",
                         "ms": list(aggregated), "ms_nonagg": list(nonaggregated)})
    return segments


def run(args) -> dict:
    nlf, template, out = Path(args.nlf).resolve(), Path(args.template).resolve(), Path(args.out).resolve()
    ids = Path(args.ids).resolve() if args.ids else None
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.upstream_shim and not args.upstream_src:
        raise ValueError("--upstream-shim requires --upstream-src")
    check_output_path(out, nlf, template, ids)
    ids_content = ids.read_bytes() if ids else None
    files = input_files(nlf, ids, args.limit, ids_content=ids_content)
    src = Path(args.upstream_src).resolve() if args.upstream_src else None
    shim = Path(args.upstream_shim).resolve() if args.upstream_shim else None
    if src is not None:
        if not np.isclose(args.fps, 20.0):
            raise ValueError("original upstream MotionScript integration currently requires --fps 20")
        if overlaps(out, src) or (shim is not None and overlaps(out, shim)):
            raise ValueError("--out must be separate from upstream source and shim")
    config = {name: getattr(args, name) for name in (
        "fps", "max_gap_seconds", "smooth_seconds", "max_joint_uncertainty",
        "min_duration_seconds", "min_angle_change_degrees", "min_distance_change_m", "emit_llm_payload")}
    # Hash and parse exactly the same byte snapshot even if a producer replaces
    # its path concurrently. Never hash one revision and load another revision.
    template_bytes = template.read_bytes()
    protocol = {"schema": SCHEMA, "config": config, "code_sha256": source_hashes(),
                "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
                "upstream_sha256": upstream_hashes(src, shim),
                "selection_sha256": hashlib.sha256(json_bytes([p.stem for p in files])).hexdigest(),
                "selection_count": len(files),
                "numpy_version": np.__version__, "scipy_version": scipy.__version__}
    if src is not None:
        import torch
        protocol["torch_version"] = torch.__version__
    fingerprint = hashlib.sha256(json_bytes(protocol)).hexdigest()
    with np.load(io.BytesIO(template_bytes), allow_pickle=False) as data:
        tmpl = {key: data[key] for key in data.files}
    out.mkdir(parents=True, exist_ok=True)
    # Reject artifact aliases before either reads or atomic replacements.
    names = ["protocol.json", "manifest.json", "manifest.receipt.json"]
    for path in files:
        names.extend([f"{path.stem}.json", f"{path.stem}.coords.npz", f"{path.stem}.receipt.json"])
    if any((out / name).is_symlink() for name in names):
        raise ValueError("output artifacts must not be symbolic links")
    # A marker binds the directory to this CLI and refuses accidental legacy reuse.
    marker = out / "protocol.json"
    if marker.exists():
        if not args.resume:
            raise ValueError("output dataset already exists; use --resume for identical verified inputs")
        if marker.read_bytes() != json_bytes({"fingerprint": fingerprint, "protocol": protocol}):
            raise ValueError("resume protocol differs; use a new output directory")
    elif any(out.iterdir()):
        raise ValueError("--out must be empty or contain an identical v2 protocol")
    else:
        atomic_write(marker, json_bytes({"fingerprint": fingerprint, "protocol": protocol}))
    manifest_path = out / "manifest.json"
    manifest_receipt = out / "manifest.receipt.json"
    if manifest_path.exists():
        if not manifest_receipt.is_file():
            raise ValueError("completed manifest has no receipt; inspect interrupted output before reuse")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = json.loads(manifest_receipt.read_text(encoding="utf-8"))
        if expected.get("manifest_sha256") != sha256(manifest_path) or saved_manifest.get("fingerprint") != fingerprint:
            raise ValueError("altered manifest or incompatible dataset fingerprint")
    records = []
    for path in files:
        sid = path.stem
        report_path, coords_path = out / f"{sid}.json", out / f"{sid}.coords.npz"
        source_bytes = path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if report_path.exists():
            if not args.resume:
                raise ValueError(f"output exists: {report_path}")
            record = json.loads(report_path.read_text(encoding="utf-8"))
            if record.get("fingerprint") != fingerprint or record.get("input_sha256") != source_sha:
                raise ValueError(f"cached input or protocol mismatch: {sid}")
            if not coords_path.is_file() or sha256(coords_path) != record.get("coords_sha256"):
                raise ValueError(f"missing or altered coordinates cache: {sid}")
            # Array hash is not enough: bind report bytes to its independent receipt.
            receipt = out / f"{sid}.receipt.json"
            if not receipt.is_file() or json.loads(receipt.read_text())["report_sha256"] != sha256(report_path):
                raise ValueError(f"missing or altered report receipt: {sid}")
        else:
            # Uncommitted coordinates from a crash may be replaced; report is commit marker.
            with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
                nlf_data = {key: data[key] for key in data.files}
            arrays, quality = preprocess(nlf_data, tmpl, fps=args.fps,
                max_gap_seconds=args.max_gap_seconds, smooth_seconds=args.smooth_seconds,
                max_joint_uncertainty=args.max_joint_uncertainty)
            events = extract_events(arrays, min_duration_seconds=args.min_duration_seconds,
                min_angle_change_degrees=args.min_angle_change_degrees,
                min_distance_change_m=args.min_distance_change_m)
            upstream = original_captions(arrays, src, shim) if src is not None else None
            save_npz(coords_path, arrays)
            record = {"schema": SCHEMA, "sid": sid, "source": "motionscript_v2_rules",
                      "status": "candidate" if events else "abstain_no_supported_dynamic_event",
                      "fingerprint": fingerprint, "input_sha256": source_sha,
                      "coords_file": coords_path.name, "coords_sha256": sha256(coords_path),
                      "quality": quality, "events": events, "caption": render_caption(events),
                      "upstream": upstream, "semantic_approval": False, "human_review_completed": False,
                      "hand_fingers_observed": False}
            if args.emit_llm_payload:
                record["llm_payload"] = make_llm_payload(events)
            report_bytes = json_bytes(record)
            atomic_write(out / f"{sid}.receipt.json", json_bytes({
                "report_sha256": hashlib.sha256(report_bytes).hexdigest()}))
            atomic_write(report_path, report_bytes)  # Final per-clip commit marker.
        records.append({"sid": sid, "input_sha256": source_sha,
                        "report_file": report_path.name, "report_sha256": sha256(report_path),
                        "coords_sha256": record["coords_sha256"], "status": record["status"],
                        "events": len(record["events"])})
        print(json.dumps({"sid": sid, "status": record["status"], "events": len(record["events"])}, ensure_ascii=False), flush=True)
    manifest = {"schema": SCHEMA, "fingerprint": fingerprint, "protocol": protocol,
                "selection": {"input_count": len(files), "limit": args.limit,
                              "ids_sha256": hashlib.sha256(ids_content).hexdigest() if ids_content is not None else None},
                "records": records, "count": len(records), "complete": True,
                "semantic_approval": False, "human_review_completed": False}
    manifest_bytes = json_bytes(manifest)
    atomic_write(manifest_receipt, json_bytes({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
    atomic_write(manifest_path, manifest_bytes)  # Final dataset commit marker.
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nlf", required=True, help="Existing NLF NPZ file or directory (read only)")
    ap.add_argument("--template", required=True, help="SMPL-H neutral joint NPZ: J_template and parents")
    ap.add_argument("--out", required=True, help="Separate empty output directory")
    ap.add_argument("--ids", help="Optional exact SID allowlist; missing IDs fail")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-gap-seconds", type=float, default=0.2)
    ap.add_argument("--smooth-seconds", type=float, default=0.0625)
    ap.add_argument("--max-joint-uncertainty", type=float, default=None)
    ap.add_argument("--min-duration-seconds", type=float, default=0.15)
    ap.add_argument("--min-angle-change-degrees", type=float, default=20.0)
    ap.add_argument("--min-distance-change-m", type=float, default=0.12)
    ap.add_argument("--emit-llm-payload", action="store_true", help="Export instructions/events only; no API call")
    ap.add_argument("--resume", action="store_true", help="Reuse only matching input, code, config and output hashes")
    ap.add_argument("--upstream-src", help="Optional local original MotionScript src directory")
    ap.add_argument("--upstream-shim", help="Optional local text2pose import shim directory")
    return ap


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
