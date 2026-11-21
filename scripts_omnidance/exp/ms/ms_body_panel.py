"""Run a frozen development or acceptance panel on bounded CPU workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from ms_landmarks import load_bundle, require_new_output
from ms_nlf_rgb32 import read_rows
from ms_optimize import atomic_write, json_bytes, sha256


def verify_case(record, row, model_sha, mapping_sha, num_aug, batch_size):
    if record.get("status") != "extracted":
        return dict(record)
    bundle = Path(record["bundle"])
    raw, source, _ = load_bundle(bundle)
    if (source.get("id") != row["id"] or source.get("source_group") != row["source_group"]
            or source["source_sha256"]["video"] != row["video_sha256"]
            or source["source_sha256"]["model"] != model_sha
            or source["source_sha256"]["mapping"] != mapping_sha
            or source.get("num_aug") != num_aug or source.get("batch_size") != batch_size
            or source.get("point_count") != 20 or source.get("source") != "nlf_official_native_rgb_body"):
        raise ValueError("Case does not match the frozen source/model/native-body protocol")
    if sha256(Path(row["video_path"])) != row["video_sha256"]:
        raise ValueError("Selected source RGB changed")
    if record.get("raw_sha256") != source["raw_sha256"] or raw["points_m"].shape[0] != source["frame_count"]:
        raise ValueError("Case registry differs from its committed raw cache")
    return dict(record)


def run(args):
    out, manifest, model, mapping = [Path(path).resolve() for path in
                                   (args.out, args.manifest, args.model, args.mapping)]
    require_new_output(out, [manifest, model, mapping])
    rows = read_rows(manifest, 0)
    role = "development" if args.expected == 20 else "acceptance"
    if (len(rows) != args.expected or any(row.get("panel_role") != role for row in rows)
            or len({row["source_group"] for row in rows}) != len(rows)):
        raise ValueError("Frozen panel role/count/source groups do not match")
    if not 1 <= args.workers <= 8 or args.bs < 1 or args.num_aug < 1:
        raise ValueError("Use 1..8 CPU workers and positive inference settings")
    manifest_sha, model_sha, mapping_sha = map(sha256, (manifest, model, mapping))
    desired = {row["id"]: row for row in rows}
    records = {}
    if args.reuse:
        registry = json.loads(Path(args.reuse).read_text())
        for record in registry["records"]:
            sid = record["id"]
            if sid in desired and record.get("status") == "extracted":
                records[sid] = verify_case(record, desired[sid], model_sha, mapping_sha, args.num_aug, args.bs)
    pending = [row for row in rows if row["id"] not in records]
    out.mkdir(parents=True)
    (out / "manifests").mkdir()
    (out / "logs").mkdir()
    atomic_write(out / "registered.json", json_bytes({
        "schema": "native-body-panel-registration-v1", "expected": args.expected, "role": role,
        "input_manifest_sha256": manifest_sha, "model_sha256": model_sha, "mapping_sha256": mapping_sha,
        "registered_ids": [row["id"] for row in rows], "source_groups": [row["source_group"] for row in rows],
        "reused_ids": list(records), "workers": args.workers, "device": "cpu",
        "torch_threads_per_worker": 4, "num_aug": args.num_aug, "batch_size": args.bs,
        "inherited_candidate_inventory_flags_are_not_selection_authority": True,
        "selection_authority": "frozen input manifest and panel_role", "semantic_approval": False}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    script = Path(__file__).with_name("ms_body_front.py")
    processes = []
    for worker in range(args.workers):
        assigned = pending[worker::args.workers]
        if not assigned:
            continue
        shard = out / "manifests" / f"worker{worker}.jsonl"
        atomic_write(shard, b"".join(json_bytes({**row, "registered_panel_manifest_sha256": manifest_sha})
                                   .replace(b"\n", b" ") + b"\n" for row in assigned))
        raw_out = out / "raw_shards" / f"worker{worker}"
        log = (out / "logs" / f"worker{worker}.log").open("wb")
        command = [args.python, "-B", str(script), "extract", "--manifest", str(shard),
                   "--model", str(model), "--mapping", str(mapping), "--out", str(raw_out),
                   "--device", "cpu", "--bs", str(args.bs), "--num-aug", str(args.num_aug),
                   "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes.append((worker, process, log, assigned, raw_out))
    for worker, process, log, assigned, raw_out in processes:
        code = process.wait()
        log.close()
        path = raw_out / "finished.json"
        if code == 0 and path.is_file():
            result = json.loads(path.read_text())
            if result.get("expected") != len(assigned) or not result.get("complete"):
                raise ValueError("Worker registry did not cover its assigned sources")
            registered = {row["id"]: row for row in assigned}
            if {record["id"] for record in result["records"]} != set(registered):
                raise ValueError("Worker silently dropped or replaced registered sources")
            for record in result["records"]:
                sid = record["id"]
                records[sid] = verify_case(record, registered[sid], model_sha, mapping_sha, args.num_aug, args.bs)
        else:
            for row in assigned:
                records[row["id"]] = {"id": row["id"], "status": "failed", "worker": worker,
                                       "worker_returncode": code, "error": "worker failed or did not commit a complete registry"}
        atomic_write(out / "progress.json", json_bytes({"expected": len(rows), "records": list(records.values()), "complete": False}))
        print(json.dumps({"worker": worker, "returncode": code, "registered_results": len(records)}), flush=True)
    if set(records) != set(desired) or sha256(manifest) != manifest_sha:
        raise ValueError("Frozen source registry changed or is incomplete")
    ordered = [records[row["id"]] for row in rows]
    result = {"schema": "native-body-cpu-panel-v1", "expected": len(rows), "role": role,
              "records": ordered, "input_manifest_sha256": manifest_sha,
              "model_sha256": model_sha, "mapping_sha256": mapping_sha,
              "all_registered": True, "all_inferences_succeeded": all(record["status"] == "extracted" for record in ordered),
              "failures_in_denominator": True, "device": "cpu", "workers": args.workers,
              "semantic_approval": False, "quality_gate_passed": False, "human_review_completed": False,
              "implementation_sha256": sha256(Path(__file__))}
    atomic_write(out / "finished.json", json_bytes(result))
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "model", "mapping", "out", "python", "ffmpeg", "ffprobe"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--expected", type=int, choices=(20, 200), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num-aug", type=int, default=3)
    ap.add_argument("--reuse", help="Verified earlier raw registry; never silently reuse an unbound cache")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
