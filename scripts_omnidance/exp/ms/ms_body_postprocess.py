"""Measure and export every registered source, retaining failed cases."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ms_body_review import load_facts
from ms_landmarks import load_bundle, require_new_output
from ms_optimize import atomic_write, json_bytes, sha256, source_hashes


CONFIG = {"fps": 20., "max_gap_seconds": .2, "smooth_seconds": .0625, "max_point_uncertainty": None,
          "min_duration_seconds": .15, "min_angle_change_degrees": 20., "min_distance_change_m": .12}


def run(args):
    registry_path, out = Path(args.registry).resolve(), Path(args.out).resolve()
    require_new_output(out, [registry_path])
    registry = json.loads(registry_path.read_text())
    records = registry["records"]
    if registry["expected"] != len(records) or len({record["id"] for record in records}) != len(records):
        raise ValueError("Registered source count/IDs must be exact")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 bounded CPU postprocessing workers")
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    script_dir = Path(__file__).resolve().parent
    front, review_script = script_dir / "ms_body_front.py", script_dir / "ms_body_review.py"
    code_sha = source_hashes()
    protocol = json.loads(Path(args.protocol).read_text()) if args.protocol else None
    if registry["expected"] == 200 and protocol is None:
        raise ValueError("Frozen measurement protocol required for the acceptance panel")
    if protocol is not None and (protocol["config"] != CONFIG or protocol["measurement_code_sha256"] != code_sha
            or protocol["front_code_sha256"] != sha256(front) or protocol["review_code_sha256"] != sha256(review_script)
            or protocol["model_sha256"] != registry["model_sha256"] or protocol["mapping_sha256"] != registry["mapping_sha256"]):
        raise ValueError("Source/model/code/parameters differ from the frozen acceptance protocol")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def one(record):
        sid = record["id"]
        if record["status"] != "extracted":
            return {"id": sid, "status": "source_extraction_failed", "source_record": record}
        try:
            raw, source, manifest_sha = load_bundle(Path(record["bundle"]))
            if source["id"] != sid or record["raw_sha256"] != source["raw_sha256"]:
                raise ValueError("Raw registry and clip cache mismatch")
            facts_dir = Path(args.reuse_facts) / sid if args.reuse_facts else out / "facts" / sid
            reused_facts = args.reuse_facts and (facts_dir / "facts.json").is_file()
            if not reused_facts:
                facts_dir = out / "facts" / sid
                command = [args.python, "-B", str(front), "describe", "--bundle", record["bundle"],
                           "--out", str(facts_dir), "--ffprobe", args.ffprobe]
                with (out / "logs" / (sid + ".facts.log")).open("wb") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            facts, _, facts_sha = load_facts(facts_dir)
            if (facts["source_manifest_sha256"] != manifest_sha or facts["video_sha256"] != source["source_sha256"]["video"]
                    or facts["parameter_config"] != CONFIG or facts["measurement_code_sha256"] != code_sha
                    or facts["implementation_sha256"] != sha256(front)):
                raise ValueError("Facts are not an identical source/code/parameter revision")
            evidence_dir = Path(args.reuse_evidence) / sid if args.reuse_evidence else out / "evidence" / sid
            reused_evidence = args.reuse_evidence and (evidence_dir / "evidence.json").is_file()
            if not reused_evidence:
                evidence_dir = out / "evidence" / sid
                command = [args.python, "-B", str(review_script), "--bundle", record["bundle"],
                           "--facts", str(facts_dir), "--out", str(evidence_dir),
                           "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe]
                with (out / "logs" / (sid + ".evidence.log")).open("wb") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            payload = (evidence_dir / "evidence.json").read_bytes()
            receipt = json.loads((evidence_dir / "evidence.receipt.json").read_text())
            if receipt["evidence_sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError("Evidence receipt mismatch")
            evidence = json.loads(payload)
            if (evidence["id"] != sid or evidence["facts_sha256"] != facts_sha
                    or evidence["source_video_sha256"] != facts["video_sha256"]
                    or evidence["implementation_sha256"] != sha256(review_script)):
                raise ValueError("Evidence is not bound to this identical fact/source/code revision")
            for frame in evidence["frames"]:
                if sha256(Path(frame["image"])) != frame["sha256"]:
                    raise ValueError("Evidence RGB frame changed")
            if sha256(Path(evidence["review_video"])) != evidence["review_video_sha256"]:
                raise ValueError("Review video changed")
            return {"id": sid, "status": "candidate_evidence_complete", "bundle": record["bundle"],
                    "facts_dir": str(facts_dir), "facts_sha256": facts_sha, "evidence_dir": str(evidence_dir),
                    "evidence_sha256": hashlib.sha256(payload).hexdigest(), "frames": evidence["frame_count"],
                    "events": len(facts["events"]), "selected_occurrences": facts["summary"]["selected_event_count"],
                    "reused_facts": bool(reused_facts), "reused_evidence": bool(reused_evidence),
                    "semantic_approval": False, "human_review_completed": False}
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            return {"id": sid, "status": "postprocessing_failed", "error_type": type(error).__name__, "error": str(error)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, record) for record in records]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_write(out / "progress.json", json_bytes({"expected": len(records), "records": results, "complete": False}))
            print(json.dumps(result), flush=True)
    ordered = {record["id"]: record for record in results}
    if set(ordered) != {record["id"] for record in records}:
        raise ValueError("Postprocessing dropped registered source IDs")
    terminal = {"schema": "native-body-postprocessing-v1", "expected": len(records),
                "records": [ordered[record["id"]] for record in records], "complete": True,
                "input_registry_sha256": sha256(registry_path), "parameter_config": CONFIG,
                "all_candidates_have_evidence": all(result["status"] == "candidate_evidence_complete" for result in results),
                "failures_in_denominator": True, "quality_gate_passed": False,
                "semantic_approval": False, "human_review_completed": False,
                "implementation_sha256": sha256(Path(__file__)),
                "frozen_protocol_sha256": sha256(Path(args.protocol)) if args.protocol else None}
    atomic_write(out / "finished.json", json_bytes(terminal))
    return terminal


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("registry", "out", "python", "ffmpeg", "ffprobe"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-facts")
    ap.add_argument("--reuse-evidence")
    ap.add_argument("--protocol", help="Frozen code, model and parameter receipt; required for acceptance")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
