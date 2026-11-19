"""Finite MotionScript caller using native controller memory accounting; v1 stays frozen."""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
ROOT = RT.parent
SELF = Path(__file__).resolve()
BASE_SHA = "6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6"
HOST = "28.216.19.225"
PYTHON = "/opt/venv_eval/bin/python"
DEPS = "/opt/ms_unified_deps_copied_v1"
CORE_DIR = Path("/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain/scripts_omnidance/exp/ms")
CORE = CORE_DIR / "ms_unified_novlm_v1.py"
SCAIL = Path("/apdcephfs_gy4/share_302533218/cedricnie/pose-preprocess/third_party/scail_pose")
SCHEMA = "ms_unified_native_config_v1"
SUMMARY_SCHEMA = "ms_unified_native_summary_v1"
GIB = 2**30
_MODEL_SNAPSHOTS = {}

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

base = module("ms_unified_native_base", RT / "resource_manager.py")
h = module("ms_unified_native_ownership", RT / "bench_manager_driver_v2.py")

def need(condition, message):
    if not condition:
        raise PermissionError(message)

def load(path):
    return h.bench.strict(Path(path).read_bytes())

def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": base.sha(path)}

def checked(reference):
    path = Path(reference["path"])
    need(path.is_absolute() and str(path.resolve(strict=True)) == str(path), "Stable absolute reference required")
    need(base.sha(path) == reference["sha256"], "Frozen reference changed: " + str(path))
    return load(path)

def dependencies():
    return {str(p) for p in (SELF, RT / "resource_manager.py", RT / "bench_manager_driver_v2.py",
        RT / "curated_resource_manager.py", RT / "visual_batch_fixed3_benchmark_v1.py", CORE,
        CORE_DIR / "dwpose_full_extract.py", CORE_DIR / "ms_fullframe_pose2d_draft_v2.py",
        CORE_DIR / "ms_fullframe_pose2d_draft_v1.py", CORE_DIR / "ms_pose2d_draft.py",
        SCAIL / "DWPoseProcess/dwpose/__init__.py", SCAIL / "DWPoseProcess/dwpose/wholebody.py",
        SCAIL / "DWPoseProcess/dwpose/onnxdet.py", SCAIL / "DWPoseProcess/dwpose/onnxpose.py")}

def ticket_for(cfg):
    ticket = checked(cfg["ticket"])
    need(ticket.get("schema") == "ms_unified_novlm_ticket_v1" and ticket.get("authorization") == "native_parent",
         "Explicit new native ticket required")
    need(ticket.get("gpu_index") == cfg["gpus"][0] and type(ticket.get("gpu_index")) is int, "Ticket GPU assignment differs")
    need(ticket.get("output_root") == cfg["out"] and ticket.get("edition") == cfg["phase"], "Ticket output/edition differs")
    need(isinstance(ticket.get("ticket_id"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", ticket["ticket_id"]), "Unsafe finite ticket identifier")
    sources = ticket.get("sources")
    need(isinstance(sources, list) and len(sources) == 3, "Only finite three-source first trial is authorized")
    plan_path = Path(cfg["plan"]["path"])
    need(base.sha(plan_path) == cfg["plan"]["sha256"], "Selected plan changed")
    rows = [h.bench.strict(line) for line in plan_path.read_bytes().splitlines() if line.strip()]
    need(rows == sources, "Ticket source rows must exactly equal SHA-bound selected plan")
    ids = [row.get("id") for row in rows]
    need(all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids) and
         len({x.casefold() for x in ids}) == 3 and ids == cfg["selected_source_ids"], "Selected source identity/order differs")
    for row in rows:
        need(isinstance(row.get("split"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", row["split"]), "Unsafe source split")
        need(not any(k in row for k in ("roi_manifest_path", "bar_plan_path", "npz_path")), "Historical ROI/cache input forbidden")
        for key in ("video_path", "audio_path"):
            value = row.get(key)
            need(isinstance(value, str) and Path(value).is_absolute() and Path(value).stem == row["id"], "Source path/identity differs")
    need(type(ticket.get("max_cache_bytes")) is int and ticket["max_cache_bytes"] >= 8 * GIB, "Explicit >=8 GiB bounded cache required")
    need(ticket.get("local_cache_root") == "/tmp", "Only independent local /tmp cache is authorized")
    need(isinstance(ticket.get("beat_checkpoint"), str) and Path(ticket["beat_checkpoint"]).is_absolute(), "Explicit local BeatThis checkpoint required")
    need(ticket.get("scail_root") == str(SCAIL), "Exact frozen DWPose source/weight root required")
    for key in ("ffprobe_path", "ffmpeg_path"):
        if key in ticket:
            need(isinstance(ticket[key], str) and Path(ticket[key]).is_absolute() and str(Path(ticket[key]).resolve()) == ticket[key] and
                 Path(ticket[key]).is_file() and os.access(ticket[key], os.X_OK), "Stable actual media executable required: " + key)
    return ticket

def selected(cfg):
    ticket = ticket_for(cfg)
    return [row["id"] for row in ticket["sources"]], 3

def validate_common(cfg):
    need(cfg.get("schema") == SCHEMA and cfg.get("node") == "N6", "Default deny: only explicit new-edition N6 config authorized")
    need(cfg.get("gpus") in ([2], [3]) and all(type(g) is int for g in cfg["gpus"]), "Only N6 GPU2 or GPU3 single-card trial authorized")
    need(cfg.get("release_authorized") is True, "Explicit holder release authorization required")
    need(type(cfg.get("targetcount")) is int and cfg["targetcount"] == cfg.get("input_count") == 3, "Finite selected denominator must be three, not full corpus")
    need(isinstance(cfg.get("phase"), str) and re.fullmatch(r"ms_unified_native_v1_[A-Za-z0-9_-]+", cfg["phase"]), "Independent new phase required")
    need(base.sha(base.__file__) == BASE_SHA and Path(base.__file__).resolve() == RT / "resource_manager.py", "Original native lifecycle SHA differs")
    need(cfg.get("producer") == cfg.get("resource_manager_ref") == ref(SELF) and cfg.get("core_ref") == ref(CORE), "Frozen separate caller/core identity required")
    config_path = cfg.get("config_path")
    need(isinstance(config_path, str) and Path(config_path).is_absolute() and str(Path(config_path).resolve()) == config_path, "Stable absolute launch config path required")
    need(cfg.get("argv") == [PYTHON, "-B", str(SELF), "worker", "--config", config_path], "Exact native worker interpreter/argv required")
    need(cfg.get("cwd") == str(RT), "Exact native runtime cwd required")
    expected_env = {"CUDA_VISIBLE_DEVICES": str(cfg["gpus"][0]), "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": DEPS}
    need(cfg.get("env") == expected_env, "Exact worker environment and sole offline dependency path required")
    for key, parent in (("out", ROOT), ("queue", ROOT / "queues")):
        value = cfg.get(key)
        need(isinstance(value, str) and Path(value).is_absolute() and str(Path(value).resolve()) == value and
             Path(value).parent == parent and Path(value).name.startswith("ms_unified_native_v1_"), "Independent stable new " + key + " required")
    need(not cfg.get("reuse_existing_results") and not cfg.get("previous_generation_gate"), "Old output/generation reuse forbidden")
    owners = cfg.get("holders")
    need(isinstance(owners, list) and len(owners) == 1 and [o.get("gpu") for o in owners] == cfg["gpus"], "Exactly one ordered physical holder identity required")
    need(type(owners[0].get("pid")) is int and owners[0]["pid"] > 0 and isinstance(owners[0].get("uuid"), str) and bool(owners[0]["uuid"]), "Explicit holder PID/UUID required")
    ticket_for(cfg)
    refs = [cfg[k] for k in ("ticket", "plan", "core_ref", "modelref", "resource_previous_safe")] + cfg["holder_restored_refs"]
    required = dependencies() | {r["path"] for r in refs}
    need(set(cfg.get("source_sha256", {})) == required, "Only exact necessary source/plan/ticket/model/proof bindings accepted")
    for r in refs:
        need(cfg["source_sha256"].get(r["path"]) == r["sha256"], "Config reference/source SHA conflict")
    need(cfg["source_sha256"].get(str(RT / "resource_manager.py")) == BASE_SHA, "Native base source binding differs")

def local_budget(cfg):
    ticket = ticket_for(cfg)
    memory = base.headroom()
    required = base.required_budget(cfg)
    need(required == 160 * GIB and memory["headroom"] >= required, "Original 160 GiB native RAM/cgroup gate not met")
    filesystem = max((line.split() for line in Path("/proc/mounts").read_text().splitlines() if
                      Path("/tmp").is_relative_to(Path(line.split()[1]))), key=lambda row: len(row[1]))[2]
    need(filesystem not in {"ceph", "nfs", "nfs4", "cifs", "fuse.ceph", "lustre", "tmpfs", "ramfs"}, "Real independent local disk required")
    free = shutil.disk_usage("/tmp").free
    need(free >= ticket["max_cache_bytes"] + 8 * GIB, "Bounded cache plus 8 GiB local disk reserve not met")
    return {"memory": memory, "required_bytes": required, "local_free_bytes": free, "local_reserve_bytes": 8 * GIB}

def previous_safe(cfg):
    reference = cfg["resource_previous_safe"]
    safe_path = Path(reference["path"])
    safe = checked(reference)
    need(safe_path.name == "resources_safe.json" and safe_path.parent.parent == ROOT / "queues" and
         all(safe.get(k) is True for k in ("all_holders_safe", "locks_released", "leases_restored", "managed_parent_wait", "unknown_jobs_not_signaled")),
         "Real predecessor resources-safe proof with every native flag required")
    refs = cfg["holder_restored_refs"]
    need(len(refs) == 1, "Exact single holder-restored proof required")
    owner = cfg["holders"][0]
    need(Path(refs[0]["path"]) == safe_path.parent / ("holder_restored_gpu%d.json" % owner["gpu"]), "Holder proof must share actual safe native queue")
    restored = checked(refs[0])
    need(restored.get("uuid") == owner["uuid"] and all(restored["holder_identity"][k] == owner[k] for k in base.IDENTITY_KEYS), "Current holder differs from exact restored identity")

def validate(cfg):
    validate_common(cfg)
    need(HOST in h.subprocess.check_output(["hostname", "-I"], text=True).split(), "Wrong actual physical node")
    need(not Path(cfg["queue"]).exists() and not Path(cfg["out"]).exists(), "Fresh independent queue/output required")
    previous_safe(cfg)
    running = base.apps()
    for gpu, owner in zip(cfg["gpus"], cfg["holders"]):
        base.check_holder(owner, base.identity(owner["pid"]), gpu, base.uuid_for(gpu), running)
        need(not h.lock_snapshot(gpu)["held"], "Existing native GPU FLOCK is held")
        base.lease_snapshot(Path("/tmp/od_reserve_%d" % gpu))
    local_budget(cfg)

def check_model(cfg):
    identity = checked(cfg["modelref"])
    need(identity.get("schema") == "ms_unified_native_model_identity_v1" and
         cfg["modelref"].get("model_identity_sha256") == cfg["modelref"]["sha256"], "New small-model identity reference required")
    files = identity.get("files", {})
    expected = {"detector": str(SCAIL / "pretrained_weights/DWPose/yolox_l.onnx"),
                "pose": str(SCAIL / "pretrained_weights/DWPose/dw-ll_ucoco_384.onnx"),
                "beat": ticket_for(cfg)["beat_checkpoint"]}
    need(set(files) == set(expected), "Only detector, pose and BeatThis checkpoint assets required")
    for name, path in expected.items():
        asset = files[name]
        need(asset["path"] == path and Path(path).is_file(), "Actual small-model checkpoint identity differs: " + name)
        st = Path(path).stat()
        snapshot = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        key = (path, asset["sha256"])
        if key not in _MODEL_SNAPSHOTS:
            need(base.sha(path) == asset["sha256"], "Actual small-model checkpoint identity differs: " + name)
            after = Path(path).stat()
            need(snapshot == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Checkpoint changed during startup validation")
            _MODEL_SNAPSHOTS[key] = snapshot
        need(snapshot == _MODEL_SNAPSHOTS[key], "Actual small-model checkpoint identity changed: " + name)

def gate(cfg, before_load=False):
    code = h.parent_gate.__code__
    original_python = "/opt/venv_vllm/bin/python"
    need(code.co_consts.count(original_python) == 1, "Native parent gate must contain exactly one frozen interpreter literal")
    code = code.replace(co_consts=tuple(PYTHON if value == original_python else value for value in code.co_consts))
    fn = types.FunctionType(code, dict(h.parent_gate.__globals__, SELF=SELF, base=base,
        NODE="N6", GPUS=cfg["gpus"], CVD=cfg["env"]["CUDA_VISIBLE_DEVICES"]), "parent_gate", h.parent_gate.__defaults__)
    return fn(cfg, before_load)

def write_summary(path, value):
    h.bench.exclusive(Path(path), h.bench.encoded(value))

def summaries(cfg, out):
    ticket = ticket_for(cfg)
    path = out / "summary.json"
    value = load(path)
    need(value.get("schema") == SUMMARY_SCHEMA and value.get("selected_source_ids") == cfg["selected_source_ids"] and
         value.get("processed") == value.get("selected") == cfg["targetcount"] and value.get("prefetch_drained") is True and
         value.get("human_review_completed") is False and value.get("production_complete_claimed") is False, "Finite exact source completion summary required")
    commit_path = out / ticket["edition"] / "batches" / ticket["ticket_id"] / "commit.json"
    need(value.get("commit_ref") == ref(commit_path), "Current sealed batch commit reference differs")
    commit = load(commit_path)
    receipts = commit.get("receipts", [])
    need(commit.get("schema") == "ms_unified_novlm_batch_commit_v1" and commit.get("ticket_id") == ticket["ticket_id"] and
         commit.get("edition") == ticket["edition"] and commit.get("unique_sources_attempted") == 3 and
         len(receipts) == 3 and [r.get("source_row") for r in receipts] == ticket["sources"], "Sealed outcomes must exactly cover selected source rows once")
    need(all(commit.get(k) is False for k in ("quality_approved", "beat_alignment_claimed", "production_complete_claimed")), "Unreviewed batch must not claim quality or full production")
    archive = commit_path.parent / "batch.tar"
    need(base.sha(archive) == commit["batch_sha256"], "Sealed batch archive changed")
    counts = Counter()
    for receipt in receipts:
        status = receipt.get("status")
        need(status in ("extracted_rule_draft_unapproved", "failed_sealed_no_retry"), "Unsealed/reused source outcome forbidden")
        case = commit_path.parent / "cases" / receipt["source_row"]["split"] / receipt["source_row"]["id"]
        need(load(case / "receipt.json") == receipt, "Per-source receipt differs from sealed outcome")
        counts[status] += 1
    need(commit.get("prefetch_drained") is True and all(type(r.get("input_copy_completed")) is bool for r in receipts), "Real drained prefetch/copy evidence required")
    copied = sum(r["input_copy_completed"] for r in receipts)
    partial = sum(r["input_copy_completed"] and r["status"] == "failed_sealed_no_retry" for r in receipts)
    errors = sum(r["status"] == "failed_sealed_no_retry" for r in receipts)
    need(value.get("counts") == dict(counts) and value.get("source_outcomes") == receipts and
         value.get("copied") == copied and value.get("partial") == partial and value.get("error_sealed") == errors,
         "Summary counters/outcomes differ from sealed receipts")
    return {str(path): {"summary": {"stage": "run", "total_source_count": 3, "selected_source_count": 3,
        "processed_source_count": 3, "plan_sha256": cfg["plan"]["sha256"]},
        "selection": {"selected_ids": cfg["selected_source_ids"]}, "counts": dict(counts),
        "human_review_completed": False, "production_complete_claimed": False}}

def worker(path):
    cfg = load(path)
    need(cfg.get("config_path") == str(Path(path).resolve()), "Worker config path differs")
    validate_common(cfg)
    need(not os.environ.get("PYTHONHOME") and os.environ.get("PYTHONPATH") == DEPS and os.environ.get("DWPOSE_ALLOW_CPU") != "1", "Only the exact offline dependency PYTHONPATH is authorized; Python home/CPU fallback forbidden")
    cfg.update(_config_path=str(Path(path).resolve()), _config_sha=base.sha(path))
    base.selected_sources = selected
    base.check_sources(cfg, path, cfg["_config_sha"], BASE_SHA)
    gate(cfg, False)
    ticket = ticket_for(cfg)
    sys.path.insert(0, str(CORE_DIR))
    core = module("ms_unified_novlm_native_worker", CORE)
    for name in ("dwpose_full_extract", "ms_fullframe_pose2d_draft_v2", "ms_fullframe_pose2d_draft_v1", "ms_pose2d_draft"):
        need(Path(sys.modules[name].__file__).resolve() == CORE_DIR / (name + ".py"), "Imported core dependency shadowed: " + name)

    @contextmanager
    def admitted(actual_ticket):
        need(actual_ticket == ticket, "Resource gate ticket differs")
        ownership = gate(cfg, False)
        budget = local_budget(cfg)
        uuids = ownership["gpu_uuids"]
        running = [app for app in base.apps() if app["uuid"] in uuids]
        evidence = {"granted": ownership.get("admitted") is True and ownership.get("caller_owned") is True,
            "owner_pid": ownership["child_identity"]["pid"], "gpu_index": cfg["gpus"][0],
            "gpu_uuid": uuids[0], "gpu_observed_free": not running,
            "native_memory_gate_preserved": budget["memory"]["headroom"] >= base.required_budget(cfg),
            "observed_unix": time.time(), "native_parent": ownership, "resource_budget": budget}
        try:
            yield evidence
        finally:
            gate(cfg, False)

    def load_admitted(actual_ticket):
        need(actual_ticket == ticket, "Model-load ticket differs")
        local_budget(cfg)
        check_model(cfg)
        gate(cfg, True)
        return core.load_models(actual_ticket)

    commit = core.run_ticket(ticket, limit=3, resource_gate=admitted, model_loader=load_admitted,
                             stage_gate=lambda before=False: gate(cfg, before),
                             resource_reader=lambda local_root: {
                                 "cgroup_headroom": local_budget(cfg)["memory"]["headroom"],
                                 "local_free": shutil.disk_usage(local_root).free})
    gate(cfg, False)
    need(commit.get("prefetch_drained") is True, "Core incomplete prefetch cannot be summarized as complete")
    receipts = commit.get("receipts", [])
    need(len(receipts) == 3 and [r.get("source_row") for r in receipts] == ticket["sources"], "Finite worker source outcomes differ")
    commit_path = Path(cfg["out"]) / ticket["edition"] / "batches" / ticket["ticket_id"] / "commit.json"
    need(load(commit_path) == commit, "Core return differs from sealed commit")
    counts = dict(Counter(r.get("status") for r in receipts))
    need(all(type(r.get("input_copy_completed")) is bool for r in receipts), "Real copied/partial/error evidence required")
    copied = sum(r["input_copy_completed"] for r in receipts)
    failed = sum(r["status"] == "failed_sealed_no_retry" for r in receipts)
    value = {"schema": SUMMARY_SCHEMA, "selected_source_ids": cfg["selected_source_ids"], "selected": 3,
        "processed": len(receipts), "counts": counts, "source_outcomes": receipts, "copied": copied,
        "partial": sum(r["input_copy_completed"] and r["status"] == "failed_sealed_no_retry" for r in receipts),
        "error_sealed": failed, "prefetch_drained": commit["prefetch_drained"], "commit_ref": ref(commit_path),
        "human_review_completed": False, "production_complete_claimed": False}
    write_summary(Path(cfg["out"]) / "summary.json", value)
    summaries(cfg, Path(cfg["out"]))

def main(argv=None):
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in[:1] == ["worker"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["worker"])
        parser.add_argument("--config", required=True)
        args = parser.parse_args(args_in)
        worker(args.config)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--manager-sha", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(args_in)
    need(args.manager_sha == BASE_SHA and base.sha(base.__file__) == BASE_SHA and base.sha(args.config) == args.config_sha, "Native base/config SHA mismatch")
    cfg = load(args.config)
    need(cfg.get("config_path") == str(Path(args.config).resolve()), "CLI/config stable path mismatch")
    base.selected_sources = selected
    base.validate_pair = validate
    base.check_model = check_model
    base.stage_summaries = summaries
    if not args.check_only:
        expected = [PYTHON, "-B", str(SELF), "--config", cfg["config_path"], "--config-sha", args.config_sha, "--manager-sha", BASE_SHA]
        need(argv is None and base.identity(os.getpid())["argv"] == expected, "Exact parent native launch argv required before any resource release")
    need(not os.environ.get("PYTHONHOME") and os.environ.get("PYTHONPATH") == DEPS and os.environ.get("DWPOSE_ALLOW_CPU") != "1", "Only the exact offline dependency PYTHONPATH is authorized; Python home/CPU fallback forbidden")
    validate(cfg)
    base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    check_model(cfg)
    if args.check_only:
        print(json.dumps({"schema": "ms_unified_native_check_v1", "checked": True, "node": cfg["node"],
            "gpus": cfg["gpus"], "selected": 3, "queue_created": False, "models_loaded": False,
            "signals_sent": False, "holder_released": False, "resource_budget": local_budget(cfg)}))
        return
    need(argv is None, "Native lifecycle must use actual process argv")
    base.main()

if __name__ == "__main__":
    main()
