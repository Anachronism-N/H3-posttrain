"""Once-only finite batch dispatch after a frozen predecessor's native-safe replay."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import ms_fullframe_vlm_postprocess_batch_v6 as v

RT, ROOT = v.RT, v.ROOT
SELF = Path(__file__).resolve()
TMP = Path("/tmp")
REGISTRY = RT / "ms_vlm_postprocess_jobs_registry_v1.json"
V4_SHA = "2fe2ea66caebb7a8f195c173eb34209f85b7aa33dc589c74ec0df95a35dca23b"
CLONE = "argv config_path cwd env ffmpeg_path gpus holder_restored_refs holders input_count modelref new_stage_authorized node out phase plan producer queue release_authorized resource_manager_ref resource_safe_refs schema selected_source_ids targetcount".split()
JOB_KEYS = set("schema job_path previous_config previous_live_ref producer_ref dispatcher_ref phase node gpus host plan source_count batch_sources allocation_ref dispatcher_task_handle".split())
SAFE_FLAGS = ("all_holders_safe", "locks_released", "leases_restored", "managed_parent_wait", "unknown_jobs_not_signaled")


def stable(path, parent=None, exists=True):
    p = Path(path)
    v.need(p.is_absolute() and str(p.resolve(strict=exists)) == str(p) and
           (parent is None or p.parent == parent), "Stable scoped absolute path required")
    return p


def check_ref(reference, parent=None):
    v.need(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "Exact path/SHA reference required")
    p = stable(reference["path"], parent)
    v.need(v.base.sha(p) == reference["sha256"], "Frozen reference changed: " + str(p))
    return p


def check_dispatcher(job):
    v.need(check_ref(job["dispatcher_ref"], RT) == SELF, "Frozen dispatcher self path differs")


def plan_rows(reference):
    p = check_ref(reference)
    v.need(p.is_relative_to(ROOT), "Plan outside experiment")
    rows = [v.h.bench.strict(line) for line in p.read_bytes().splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    v.need(all(isinstance(i, str) and re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids) and
           len({i.casefold() for i in ids}) == len(ids), "Casefold-unique source IDs required")
    return rows


def registry_state(value, job, job_ref, rows):
    v.need(value.get("schema") == "ms_vlm_postprocess_jobs_registry_v1" and isinstance(value.get("jobs"), list), "Wrong registry")
    pair = [e for e in value["jobs"] if e.get("node") == job["node"] and e.get("gpus") == job["gpus"] and e.get("expected_running") is True]
    v.need(len(pair) == 1 and pair[0].get("config_path") == job["previous_config"]["path"], "Registry predecessor CAS conflict")
    pending = [e for e in value["jobs"] if e.get("phase") == job["phase"]]
    v.need(len(pending) == 1, "Exactly one self pending registration required")
    expected = dict(node=job["node"], gpus=job["gpus"], config_path=str(RT / (job["phase"] + "_config.json")),
                    plan_ref=job["plan"], sources=job["source_count"], batch_sources=job["batch_sources"],
                    expected_running=False, task_handle=job["dispatcher_task_handle"], dispatcher_job_ref=job_ref)
    v.need(all(pending[0].get(k) == x for k, x in expected.items()) and pending[0].get("expected_running") is False,
           "Self pending registration differs")
    ids = {r["id"].casefold() for r in rows}
    for entry in value["jobs"]:
        if entry is pending[0]:
            continue
        references = [entry[k] for k in ("plan_ref", "plan") if k in entry]
        other_ids = list(entry.get("selected_source_ids", []))
        if entry.get("config_path"):
            cp = stable(entry["config_path"], RT, exists=False)
            if cp.exists():
                cfg = v.load(cp)
                v.need(cfg.get("config_path") == str(cp), "Registry config path differs")
                other_ids.extend(cfg["selected_source_ids"])
                references.append(cfg["plan"])
            else:
                v.need(entry.get("expected_running") is False and isinstance(entry.get("phase"), str) and
                       re.fullmatch(v.PREFIX + r"[A-Za-z0-9_-]+", entry["phase"]) and
                       cp == RT / (entry["phase"] + "_config.json") and entry.get("node") in v.PAIRS and
                       tuple(entry.get("gpus", [])) in v.PAIRS[entry["node"]] and
                       type(entry.get("sources")) is int and 1 <= entry["sources"] <= 10000 and
                       entry["batch_sources"] in (4, 8, 16, 32) and "plan_ref" in entry, "Missing config is not a legal frozen pending job")
                v.need(len(plan_rows(entry["plan_ref"])) == entry["sources"], "Pending frozen plan count differs")
        for reference in references:
            other_ids.extend(r["id"] for r in plan_rows(reference))
        v.need(not ids.intersection(i.casefold() for i in other_ids), "Source intersects another registered plan/config")
    return pair[0], pending[0]


def registry_check(job, job_ref, rows, promote=False):
    stable(REGISTRY, RT)
    fd = os.open(REGISTRY.with_suffix(".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        st = os.fstat(lock.fileno())
        v.need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, "Unsafe registry lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = v.load(REGISTRY)
        old, pending = registry_state(value, job, job_ref, rows)
        if not promote:
            return
        old.update(expected_running=False, worker_status="completed_native_safe_fullreplay",
                   native_safe_verified=True, full_replay_verified=True)
        pending.update(expected_running=True, task_handle=job["dispatcher_task_handle"], worker_status="dispatch_validated_before_native_main")
        with tempfile.NamedTemporaryFile(dir=RT, prefix=".vlm-registry-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(v.h.bench.encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, REGISTRY)
        fd = os.open(RT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def check_job(path, digest):
    jp = stable(path, RT)
    job_ref = dict(path=str(jp), sha256=digest)
    job = v.load(check_ref(job_ref, RT))
    v.need(set(job) == JOB_KEYS and job["schema"] == "ms_vlm_batch_safe_dispatch_job_v1" and
           job["job_path"] == str(jp), "Exact self-bound dispatch job required")
    check_dispatcher(job)
    v.need(isinstance(job["phase"], str) and re.fullmatch(v.PREFIX + r"[A-Za-z0-9_-]+", job["phase"]), "Independent v4 phase required")
    v.need(job["node"] in v.PAIRS and tuple(job["gpus"]) in v.PAIRS[job["node"]] and
           all(type(g) is int for g in job["gpus"]) and job["host"] == v.HOSTS[job["node"]], "Unknown physical slot")
    v.need(job["host"] in v.subprocess.check_output(["hostname", "-I"], text=True).split(), "Wrong physical dispatcher node")
    v.need(type(job["source_count"]) is int and type(job["batch_sources"]) is int and
           1 <= job["source_count"] <= 10000 and job["batch_sources"] in (4, 8, 16, 32), "Finite one-to-10000 source budget with four/eight inference batch required")
    v.need(isinstance(job["dispatcher_task_handle"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", job["dispatcher_task_handle"]), "Frozen dispatcher task handle required")
    check_ref(job["producer_ref"], RT)
    v.need(job["producer_ref"] == dict(path=str(v.SELF), sha256=V4_SHA), "Frozen batch v4 producer required")
    previous = v.load(check_ref(job["previous_config"], RT))
    v.need(previous["config_path"] == job["previous_config"]["path"] and previous["node"] == job["node"] and
           previous["gpus"] == job["gpus"] and previous["phase"] != job["phase"], "Predecessor scope differs")
    for k, parent in (("queue", ROOT / "queues"), ("out", ROOT)):
        stable(previous[k], parent)
    check_ref(previous["producer"], RT)
    v.need(Path(previous["producer"]["path"]).name in ("ms_fullframe_vlm_postprocess_v3.py", "ms_fullframe_vlm_postprocess_batch_v4.py", "ms_fullframe_vlm_postprocess_batch_v5.py", v.SELF.name), "Only frozen native v3/v4 predecessor permitted")
    v.need(previous["resource_manager_ref"] == previous["producer"], "Predecessor producer differs")
    for p, sha in previous["source_sha256"].items():
        v.need(v.base.sha(stable(p)) == sha, "Predecessor source changed")
    live = v.load(check_ref(job["previous_live_ref"]))
    v.need(job["previous_live_ref"]["path"] == str(Path(previous["queue"]) / "live.json") and
           live["phase"] == previous["phase"] and live["targetcount"] == previous["targetcount"] and
           live["argv"] == previous["argv"] and isinstance(live.get("child_identity"), dict), "Exact frozen predecessor live record required")
    expected_argv = [v.PYTHON, "-B", previous["producer"]["path"], "--config", previous["config_path"],
                     "--config-sha", job["previous_config"]["sha256"], "--manager-sha", v.BASE_SHA]
    v.need(live["manager_identity"]["argv"] == expected_argv, "Predecessor manager argv differs")
    rows = plan_rows(job["plan"])
    v.need(len(rows) == job["source_count"], "New plan source count differs")
    v.rows_for(dict(plan=job["plan"], targetcount=len(rows), input_count=len(rows), selected_source_ids=[r["id"] for r in rows]))
    for dependency in v.dependencies() - {str(v.SELF)}:
        v.need(previous["source_sha256"].get(dependency) == v.base.sha(dependency), "Unfrozen predecessor dependency")
    v.need(previous["source_sha256"].get(previous["modelref"]["path"]) == previous["modelref"]["sha256"], "Unfrozen inherited model reference")
    allocation = v.load(check_ref(job["allocation_ref"]))
    v.need(Path(job["allocation_ref"]["path"]).is_relative_to(ROOT) and
           allocation.get("schema") == "ms_vlm_batch_exclusive_allocation_v1" and
           all(allocation.get(k) == job[k] for k in ("phase", "node", "gpus", "source_count", "batch_sources")) and
           allocation.get("selected_source_ids") == [r["id"] for r in rows] and
           (allocation.get("rows") == rows or allocation.get("plan_refs") == [job["plan"]]), "Exclusive allocation differs")
    registry_check(job, job_ref, rows)
    check_dispatcher(job)
    return job, job_ref, previous, live, rows


def manager_alive(job, live):
    check_ref(job["previous_live_ref"])
    expected = live["manager_identity"]
    current = v.base.identity(expected["pid"])
    v.need(all(current[k] == expected[k] for k in v.base.IDENTITY_KEYS) and current.get("state") not in ("Z", "X", "x"),
           "Predecessor manager identity changed or no longer live")


def wait_safe(job, previous, live):
    q = Path(previous["queue"])
    deadline = time.monotonic() + 7200
    while True:
        manager_alive(job, live)
        v.need(not (q / "failure.json").exists(), "Predecessor failure; no automatic retry")
        if (q / "worker_exit.json").exists():
            v.need(v.load(q / "worker_exit.json").get("exit_code") == 0, "Predecessor worker failed")
        if (q / "resources_safe.json").exists():
            break
        v.need(time.monotonic() < deadline, "Native safety wait exceeded bound")
        time.sleep(15)
    closure_refs = [v.ref(p) for p in (q / "worker_exit.json", Path(previous["out"]) / "summary.json", q / "resources_safe.json")]
    safe = v.checked(closure_refs[2])
    v.need(safe.get("phase") == previous["phase"] and all(safe.get(k) is True for k in SAFE_FLAGS), "Incomplete native safety")
    worker = live["child_identity"]
    try:
        current = v.base.identity(worker["pid"])
    except (FileNotFoundError, ProcessLookupError):
        current = None
    v.need(current is None or current["start_ticks"] != worker["start_ticks"], "Predecessor worker still alive")
    exit_value = v.checked(closure_refs[0])
    v.need(exit_value.get("exit_code") == 0 and exit_value.get("source_unchanged") is True and
           all(exit_value.get(k) == previous[k] for k in ("phase", "producer", "plan", "targetcount", "input_count")), "Predecessor exit scope/source differs")
    producer = v if previous["producer"]["path"] == str(v.SELF) else v.module("vlm_dispatch_predecessor", previous["producer"]["path"])
    replay = producer.summaries(previous, Path(previous["out"]))
    v.need(bool(replay) and replay == exit_value.get("summaries"), "Complete predecessor replay differs")
    check_closure(job, previous, live, closure_refs)
    return [v.ref(q / ("holder_restored_gpu%d.json" % gpu)) for gpu in job["gpus"]], closure_refs[2], closure_refs


def check_closure(job, previous, live, closure_refs):
    for reference in closure_refs:
        check_ref(reference)
    v.need(not (Path(previous["queue"]) / "failure.json").exists(), "Predecessor failure after replay")
    manager_alive(job, live)


def build_config(job, previous, rows, restored, safe):
    cfg = {k: copy.deepcopy(previous[k]) for k in CLONE}
    holders = []
    for gpu, reference in zip(job["gpus"], restored):
        proof = v.checked(reference)
        holders.append(dict(proof["holder_identity"], gpu=gpu, uuid=proof["uuid"]))
    cp = RT / (job["phase"] + "_config.json")
    cfg.update(schema=v.SCHEMA, phase=job["phase"], config_path=str(cp), node=job["node"], gpus=job["gpus"],
               producer=job["producer_ref"], resource_manager_ref=job["producer_ref"], plan=job["plan"],
               targetcount=len(rows), input_count=len(rows), batch_sources=job["batch_sources"],
               selected_source_ids=[r["id"] for r in rows], out=str(ROOT / job["phase"]), queue=str(ROOT / "queues" / job["phase"]),
               holders=holders,
               resource_safe_refs=[safe, safe], holder_restored_refs=restored,
               argv=[v.PYTHON, "-B", str(v.SELF), "worker", "--config", str(cp)], cwd="/tmp")
    cfg["env"] = v.expected_env(cfg)
    references = v.source_refs(cfg, rows)
    cfg["source_sha256"] = {p: v.base.sha(p) for p in sorted(v.dependencies() | {r["path"] for r in references})}
    return cp, cfg


def dispatch(path, digest):
    job, job_ref, previous, live, rows = check_job(path, digest)
    lockpath = TMP / ("ms-vlm-dispatch-%s-%s.lock" % (job["node"], "-".join(map(str, job["gpus"]))))
    fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        st = os.fstat(lock.fileno())
        v.need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, "Unsafe dispatcher lock")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for p in (RT / (job["phase"] + "_config.json"), ROOT / job["phase"], ROOT / "queues" / job["phase"]):
            v.need(not p.exists(), "Existing dispatch output/config; never restart")
        v.write_once(RT / (job["phase"] + "_dispatch_started.json"), dict(schema="ms_vlm_batch_dispatch_started_v1", job_ref=job_ref,
            phase=job["phase"], dispatcher_task_handle=job["dispatcher_task_handle"], dispatcher_identity=v.base.identity(os.getpid()), automatic_restart=False))
        restored, safe, closure_refs = wait_safe(job, previous, live)
        cp, cfg = build_config(job, previous, rows, restored, safe)
        v.base.headroom = v.actual_controller_headroom
        v.base.selected_sources = v.selected
        v.validate_common(cfg)
        v.validate(cfg)
        v.write_once(cp, cfg)
        sha = v.base.sha(cp)
        v.base.check_sources(cfg, str(cp), sha, v.BASE_SHA)
        v.base.check_model(cfg)
        check_job(path, digest)
        check_closure(job, previous, live, closure_refs)
        registry_check(job, job_ref, rows, promote=True)
        check_dispatcher(job)
        check_closure(job, previous, live, closure_refs)
        os.execv(v.PYTHON, [v.PYTHON, "-B", str(v.SELF), "--config", str(cp), "--config-sha", sha, "--manager-sha", v.BASE_SHA])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--job-sha", required=True)
    args = parser.parse_args()
    dispatch(args.job, args.job_sha)


if __name__ == "__main__":
    main()
