#!/usr/bin/env python3
"""Nebula entry for the 9 fixed H3 jobs: verify the prompt package, run 8 GPUs in parallel, upload outputs."""
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

JOB = Path(__file__).resolve().parent
CONFIG = json.loads((JOB / "deploy.json").read_text())
H3 = Path(CONFIG["oss_mount"]) / CONFIG["h3_prefix"]
RUN_ID = os.environ.get("H3_RUN_ID") or time.strftime("h3jobs_%Y%m%d_%H%M%S")
OSS_RUN = H3 / "runs" / RUN_ID
OSS_OUT = OSS_RUN / "outputs"
SCRATCH = Path(CONFIG["scratch"])
LOCAL = SCRATCH / RUN_ID
LOGS = LOCAL / "logs"
CODE = SCRATCH / "code_v3" / "nebula_job"
VENV = SCRATCH / "venv_h3"
WEIGHTS = SCRATCH / CONFIG["weights_root"]
SAFE_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES", "HOSTNAME", "NEBULA_JOB_NAME",
                 "NEBULA_PROJECT", "SIGMA_APP_SITE")

report = {"run_id": RUN_ID, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
          "run_uri": "%s/%s/runs/%s" % (CONFIG["oss_bucket_uri"], CONFIG["h3_prefix"], RUN_ID),
          "steps": {}, "jobs": {}}
WORKERS = []


def log(message):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), message)
    print(line, flush=True)
    if LOGS.exists():
        with open(LOGS / "launcher.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_to_oss(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def sync_logs():
    try:
        for path in sorted(LOGS.glob("*")):
            if path.is_file():
                copy_to_oss(path, OSS_RUN / "logs" / path.name)
    except Exception as error:
        log("log sync failed: %r" % error)


def save_report():
    try:
        (LOCAL / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        copy_to_oss(LOCAL / "report.json", OSS_RUN / "report.json")
    except Exception as error:
        log("report sync failed: %r" % error)


def step(name, fn):
    log("=== step %s ===" % name)
    started = time.time()
    entry = report["steps"][name] = {"status": "running"}
    try:
        entry["result"] = fn()
        entry["status"] = "ok"
    except Exception as error:
        entry["status"] = "fail"
        entry["error"] = repr(error)[:2000]
        entry["traceback"] = traceback.format_exc()[-4000:]
        log("step %s FAILED: %r" % (name, error))
    entry["seconds"] = round(time.time() - started, 1)
    save_report()
    return entry["status"] == "ok"


def run(cmd, log_name, env=None, timeout=None):
    log("$ %s" % " ".join(str(c) for c in cmd))
    with open(LOGS / log_name, "w", encoding="utf-8") as handle:
        process = subprocess.run([str(c) for c in cmd], stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError("%s exited %d (see logs/%s)" % (cmd[0], process.returncode, log_name))


def capture(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as error:
        return "error: %r" % error


def meminfo_gib():
    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) / 1024 ** 2
    return {"mem_total_gib": round(info.get("MemTotal", 0), 1), "mem_available_gib": round(info.get("MemAvailable", 0), 1)}


def scratch():
    LOGS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(SCRATCH).free
    if free < 220e9:
        raise RuntimeError("%s has only %.1f GB free" % (SCRATCH, free / 1e9))
    return {"scratch": str(SCRATCH), "free_gb": round(free / 1e9, 1)}


def env_report():
    return {"hostname": socket.gethostname(), "python": sys.version.split()[0],
            "nvidia_smi_L": capture(["nvidia-smi", "-L"]),
            "env": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}}


def safe_members(tar, root):
    root = root.resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root) + os.sep) or member.isdev():
            raise RuntimeError("unsafe archive member %r" % member.name)
    return tar.getmembers()


def code_package():
    import tarfile
    tarball = H3 / CONFIG["code_tarball"]
    digest = sha256_file(tarball)
    if digest != CONFIG["code_tarball_sha256"]:
        raise RuntimeError("code package sha256 mismatch")
    target = SCRATCH / "code_v3"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tarfile.open(tarball) as tar:
        tar.extractall(target, members=safe_members(tar, target))
    version = json.loads((CODE / "VERSION.json").read_text())
    if version["diffsynth_studio"]["commit"] != CONFIG["diffsynth_commit"]:
        raise RuntimeError("unexpected DiffSynth commit")
    return {"sha256": digest, "package_version": version["preparation"]["version"],
            "diffsynth_commit": CONFIG["diffsynth_commit"],
            "h3_video_only": str(CODE / "h3_posttrain/scripts_omnidance/exp/ms/h3_video_only.py")}


def bootstrap():
    env = dict(os.environ, H3_ENV_CACHE=str(H3 / "env" / "cache"))
    run(["bash", CODE / "h3env" / "bootstrap.sh", VENV, CODE], "bootstrap.log", env=env, timeout=5400)
    freeze = capture([str(VENV / "bin" / "python"), "-m", "pip", "freeze"])
    (LOGS / "pip_freeze.txt").write_text(freeze + "\n")
    return {"venv": str(VENV), "source": (VENV / ".h3_source").read_text().strip()
            if (VENV / ".h3_source").exists() else "unknown"}


def sha256_copy(src, dst, size, sha):
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    partial = Path(str(dst) + ".partial")
    with open(src, "rb") as reader, open(partial, "wb") as writer:
        for chunk in iter(lambda: reader.read(16 << 20), b""):
            digest.update(chunk)
            writer.write(chunk)
    os.replace(partial, dst)
    return dst.stat().st_size == size and digest.hexdigest() == sha


def weights():
    manifest = H3 / CONFIG["weights_manifest"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != CONFIG["weights_manifest_sha256"]:
        raise RuntimeError("weights manifest checksum mismatch")
    entries = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            rel, size, sha = line.split("\t")
            parts = PurePosixPath(rel).parts
            if PurePosixPath(rel).is_absolute() or ".." in parts or parts[0] != "Ref2VA":
                raise RuntimeError("unsafe manifest path %r" % rel)
            entries.append((rel, int(size), sha))
    if len(entries) != CONFIG["expected_weight_files"] or sum(e[1] for e in entries) != CONFIG["expected_weight_bytes"]:
        raise RuntimeError("manifest file count or size mismatch")
    started, bad = time.time(), []
    with ThreadPoolExecutor(max_workers=CONFIG["copy_threads"]) as pool:
        futures = {pool.submit(sha256_copy, manifest.parent / rel, WEIGHTS / rel, size, sha): rel
                   for rel, size, sha in entries}
        for future in as_completed(futures):
            try:
                if not future.result():
                    bad.append(futures[future])
            except Exception as error:
                bad.append("%s: %r" % (futures[future], error))
    if bad:
        raise RuntimeError("weight verification failed: %r" % bad[:5])
    return {"files": len(entries), "bytes": CONFIG["expected_weight_bytes"],
            "seconds": round(time.time() - started, 1)}


def prepare_package():
    source = H3 / CONFIG["package_prefix"]
    target = LOCAL / "package"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    manifest = json.loads((target / "manifest.json").read_text())
    spec = CONFIG["generation"]
    by_id = {item["job_id"]: item for item in manifest["jobs"]}
    jobs, problems = [], []
    for job_dir in sorted((target / "jobs").iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        hashes = json.loads((job_dir / "sha256.json").read_text())
        for name, expected in hashes.items():
            actual = sha256_file(job_dir / name)
            if actual != expected:
                problems.append("%s/%s sha256 %s != %s" % (job_id, name, actual, expected))
        item = by_id.get(job_id) or {}
        prompt_sha = hashes["prompt.txt"]
        if item.get("prompt_sha256") and item["prompt_sha256"] != prompt_sha:
            problems.append("%s manifest prompt sha mismatch" % job_id)
        request = json.loads((job_dir / "request.json").read_text())
        jobs.append({"job_id": job_id, "case_id": request["case_id"], "variant": request["caption_variant"],
                     "dance_style": request.get("dance_style"), "prompt": str(job_dir / "prompt.txt"),
                     "reference": str(job_dir / "reference.jpg"), "request": request,
                     "prompt_sha256": prompt_sha,
                     "reference_sha256": hashes["reference.jpg"]})
    if len(jobs) != CONFIG["expected_jobs"]:
        problems.append("expected %d jobs, found %d" % (CONFIG["expected_jobs"], len(jobs)))
    generation = {"seed": spec["seed"], "width": spec["width"], "height": spec["height"],
                  "num_frames": spec["num_frames"], "steps": spec["steps"], "cfg": spec["cfg"],
                  "flow_shift": spec["flow_shift"], "fps": spec["fps"]}
    for job in jobs:
        g = job["request"]["generation"]
        if (g["seed"], g["width"], g["height"], g["num_frames"], g["steps"], float(g["cfg"])) != (
                spec["seed"], spec["width"], spec["height"], spec["num_frames"], spec["steps"], float(spec["cfg"])):
            problems.append("%s request.json params deviate: %r" % (job["job_id"], g))
    if problems:
        raise RuntimeError("prompt package verification failed: %r" % problems[:5])
    payload = {"run_id": RUN_ID, "spec": generation, "jobs": jobs, "manifest": manifest}
    (LOCAL / "jobs.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    copy_to_oss(LOCAL / "jobs.json", OSS_RUN / "jobs.json")
    return {"jobs": len(jobs), "package_prefix": CONFIG["package_prefix"],
            "variants": sorted({job["variant"] for job in jobs}),
            "cases": sorted({job["case_id"] for job in jobs})}


def shard():
    payload = json.loads((LOCAL / "jobs.json").read_text())
    count = CONFIG["workers"]
    shards = [[] for _ in range(count)]
    for index, job in enumerate(payload["jobs"]):
        shards[index % count].append(job)
    (LOCAL / "shards").mkdir(parents=True, exist_ok=True)
    for index, jobs in enumerate(shards):
        (LOCAL / "shards" / ("worker_%d.json" % index)).write_text(
            json.dumps({"spec": payload["spec"], "jobs": jobs}, indent=1, ensure_ascii=False))
    return {"workers": count, "shares": [len(jobs) for jobs in shards]}


def runner_env(gpu_index):
    env = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu_index), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               DIFFSYNTH_SKIP_DOWNLOAD="true", OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONPATH=os.pathsep.join([str(CODE / "DiffSynth-Studio"),
                                           str(CODE / "h3_posttrain" / "scripts_omnidance" / "exp" / "ms")]))
    return env


def start_worker(index):
    env = runner_env(index)
    cmd = [str(VENV / "bin" / "python"), str(JOB / "h3_job_worker.py"),
           "--jobs_json", str(LOCAL / "shards" / ("worker_%d.json" % index)),
           "--out_root", str(LOCAL / "out" / ("w%d" % index)), "--oss_outputs", str(OSS_OUT),
           "--weights_root", str(WEIGHTS / "Ref2VA"), "--code_root", str(CODE),
           "--status_file", str(LOCAL / ("status_w%d.json" % index)), "--gpu_index", str(index),
           "--vram_margin_gb", str(CONFIG["vram_margin_gb"])]
    log("$ (gpu %d) %s" % (index, " ".join(cmd[:6])))
    handle = open(LOGS / ("worker_%d.log" % index), "a", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                               bufsize=1, start_new_session=True)

    def pump():
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            sys.stdout.write("[w%d] %s" % (index, line))
            sys.stdout.flush()
        handle.close()
    threading.Thread(target=pump, daemon=True, name="worker-log-%d" % index).start()
    return {"index": index, "process": process}


def launch():
    started = []
    for index in range(CONFIG["workers"]):
        memory = meminfo_gib()
        if memory["mem_available_gib"] < CONFIG["min_mem_available_gb"]:
            raise RuntimeError("MemAvailable %.1f GiB < %d before worker %d" % (
                memory["mem_available_gib"], CONFIG["min_mem_available_gb"], index))
        if index:
            time.sleep(CONFIG["stagger_s"])
        WORKERS.append(start_worker(index))
        started.append({"gpu": index, "pid": WORKERS[-1]["process"].pid,
                        "mem_available_gib": memory["mem_available_gib"]})
        log("worker %d started (mem available %.1f GiB)" % (index, memory["mem_available_gib"]))
    return started


def collect_jobs():
    payload = json.loads((LOCAL / "jobs.json").read_text())
    results = {}
    for job in payload["jobs"]:
        path = OSS_OUT / ("%s.json" % job["job_id"])
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = None
        results[job["job_id"]] = data
    return results


def supervise():
    started = time.time()
    while True:
        alive = [w for w in WORKERS if w["process"].poll() is None]
        results = collect_jobs()
        done = {k: (v or {}).get("status") for k, v in results.items()}
        report["jobs"] = done
        report["elapsed_s"] = round(time.time() - started, 1)
        save_report()
        log("progress: %d/%d reported (ok=%d fail=%d), workers alive=%d" % (
            sum(1 for v in done.values() if v), len(done), sum(1 for v in done.values() if v == "ok"),
            sum(1 for v in done.values() if v == "fail"), len(alive)))
        if not alive:
            break
        time.sleep(30)
    for index in range(CONFIG["workers"]):
        log_path = LOGS / ("worker_%d.log" % index)
        if log_path.exists():
            copy_to_oss(log_path, OSS_RUN / "logs" / log_path.name)
    sync_logs()
    results = collect_jobs()
    failed = [job_id for job_id, data in results.items() if not data or data.get("status") != "ok"]
    report["critical_failed"] = failed
    report["jobs"] = {job_id: (data or {}).get("status") for job_id, data in results.items()}
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_report()
    log("SUMMARY ok=%d fail=%r outputs=%s" % (len(results) - len(failed), failed, OSS_OUT))
    return 0 if not failed else 1


def stop(signum, frame):
    log("launcher received signal %d" % signum)
    for worker in WORKERS:
        try:
            os.killpg(worker["process"].pid, signal.SIGTERM)
        except OSError:
            pass
    report["stopped"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    save_report()
    os._exit(0)


def main():
    import signal
    if int(os.environ.get("RANK", "0") or 0) != 0:
        log("rank %s: nothing to do" % os.environ.get("RANK"))
        return 0
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for name, fn in (("scratch", scratch), ("env_report", env_report), ("code_package", code_package),
                     ("bootstrap", bootstrap), ("weights", weights), ("package", prepare_package),
                     ("shard", shard), ("launch", launch)):
        if not step(name, fn):
            report["critical_failed"] = [name]
            report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
            save_report()
            for worker in WORKERS:
                try:
                    os.killpg(worker["process"].pid, signal.SIGTERM)
                except OSError:
                    pass
            return 1
    return supervise()


if __name__ == "__main__":
    sys.exit(main())
