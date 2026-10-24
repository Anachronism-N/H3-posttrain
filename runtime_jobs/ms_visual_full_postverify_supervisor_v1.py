"""Single managed CPU supervisor: eight real terminals, then one original export.

No model starts, retries, media/frame scans while waiting, or resource signals.
A durable launch claim prevents a restarted parent from blindly exporting again.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path("/apdcephfs_gy4/share_302533218/cedricnie")
MS=ROOT/"diff/logs_omnidance/exp/MS"
CONTROL=MS/"visual12k_local32b_full_postverify_control_v1"
PLAN=MS/"visual12k_local32b_full_v1/launch_lead_v1.json"
INPUT=MS/"chunk_caption12k_input_v1/manifest.jsonl"
PRODUCER=ROOT/"H3-posttrain/scripts_omnidance/exp/ms/ms_visual_batch.py"
EXPORTER=PRODUCER.with_name("ms_visual_export.py")
OUTPUT=MS/"visual12k_local32b_full_export_v1"
EXPECTED={str(PLAN):"43a38cb9c5d730ea5c34a30bcc97c18394946dd1cee0a7a0f99996d53d8b2e18",
          str(INPUT):"4b2efc8233fed8dd7cdb46acf8da318db76cb575faa1177e9e2bc113d453d283",
          str(PRODUCER):"b362dc5275b19c70a6f4f4d149b974e75571ad07e7e4f1f0a2251222fd839204",
          str(EXPORTER):"4a562a5ff7d5bfcc4b898b60f2e4883e628b3f2f8482e43431286de7f558a558"}
ARGV=["/opt/venv_eval/bin/python","-B","-u",str(EXPORTER),"--plan",str(PLAN),"--out",str(OUTPUT),"--ffprobe","/opt/ms_tools/ffprobe"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8388608),b""):h.update(chunk)
    return h.hexdigest()


def current_identity(pid):
    p=Path("/proc")/str(pid)
    value=(p/"stat").read_text().rsplit(") ",1)[1].split()
    return {"pid":pid,"uid":p.stat().st_uid,"start_ticks":int(value[19]),"state":value[0],
            "argv":[a.decode(errors="replace") for a in (p/"cmdline").read_bytes().split(b"\0") if a]}


def require(condition,message):
    if not condition:raise ValueError(message)


def data(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()


def exclusive(path,value):
    with Path(path).open("xb") as stream:
        stream.write(data(value));stream.flush();os.fsync(stream.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def state(value):
    p=CONTROL/"state.json";temp=CONTROL/"state.json.tmp"
    with temp.open("wb") as stream:
        stream.write(data(value));stream.flush();os.fsync(stream.fileno())
    temp.replace(p)


def hashes_unchanged():
    for path,digest in EXPECTED.items():require(sha(path)==digest,"frozen source/input/plan changed: "+path)


def duplicate_processes():
    matches=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=current_identity(int(proc.name))["argv"]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(EXPORTER) in argv and str(OUTPUT) in argv or str(Path(__file__).resolve()) in argv:
            matches.append({"pid":int(proc.name),"argv":argv})
    return matches


def controls(job):
    return Path(job["out"])/"shards"/f"shard-{job['shard']:05d}-of-00008"


def terminal_scan(jobs):
    pending=[];proofs=[]
    for job in jobs:
        c=controls(job);q=Path(job["resource_queue"])
        paths=[q/"worker_exit.json",c/"summary.json",c/"index.jsonl",c/"run.receipt.json"]
        missing=[str(p) for p in paths if not p.is_file()]
        if missing:
            pending.append({"shard":job["shard"],"missing_artifacts":missing});continue
        exit_value=json.loads(paths[0].read_text());summary=json.loads(paths[1].read_text())
        require(type(exit_value.get("exit_code")) is int and exit_value["exit_code"] in (0,1),"abnormal original worker terminal")
        require(exit_value.get("source_unchanged") is True and exit_value.get("input_unchanged") is True,"original worker reports source/input drift")
        require(exit_value["summary_path"]==str(c/"summary.json") and data(exit_value["summary"])==data(summary),"resource/producer terminal summary mismatch")
        require(type(summary["selected_ids"]) is int and type(summary["processed_ids"]) is int and summary["selected_ids"]==summary["processed_ids"]==1500,"original shard is incomplete")
        proofs.append({"shard":job["shard"],"worker_exit_sha256":sha(paths[0]),"summary_sha256":sha(paths[1])})
    return pending,proofs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-sha",required=True)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    require(sha(__file__)==args.self_sha,"supervisor entry changed")
    hashes_unchanged()
    plan=json.loads(PLAN.read_text())
    require(plan["shards"]==8 and plan["target_ids"]==12000 and sorted(j["shard"] for j in plan["jobs"])==list(range(8)),"wrong immutable launch scope")
    require(os.access("/opt/ms_tools/ffprobe",os.X_OK),"local ffprobe unavailable")
    jobs=sorted(plan["jobs"],key=lambda j:j["shard"])
    prior=duplicate_processes()
    pending,proofs=terminal_scan(jobs)
    result={"status":"preflight_pending_terminals" if pending else "preflight_terminal_ready","same_scope_processes":prior,
        "terminal_shard_count":len(proofs),"pending_shard_count":len(pending),"entry_sha256":args.self_sha,
        "expected_identity":EXPECTED,"argv":ARGV,"existing_export":OUTPUT.exists(),"CPU_only":True,"source_or_frame_scan_performed":False}
    if args.preflight:
        print(json.dumps(result),flush=True);return 0
    require(not prior,"same-scope live exporter/supervisor exists; do not repeat")
    lock=(CONTROL/"scope.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (CONTROL/"registry.json").exists(),"scope already registered; inspect prior receipt, never restart blindly")
    require(not OUTPUT.exists(),"original export already exists; do not overwrite or blindly run another export")
    exclusive(CONTROL/"registry.json",{**result,"status":"waiting_original_terminals","owner_identity":current_identity(os.getpid()),
        "poll_seconds":60,"at_most_one_export_popen":True,"GPU_tasks_started":False})
    child=None
    try:
        while True:
            pending,proofs=terminal_scan(jobs)
            state({"status":"pending_original_workers" if pending else "checking_original_control_locks","utc":time.strftime("%FT%TZ",time.gmtime()),
                "terminal_shards":len(proofs),"pending_shards":[r["shard"] for r in pending],"exporter_started":False,"formal_training":False})
            if pending:
                time.sleep(60);continue
            with ExitStack() as guards:
                busy=[]
                for job in jobs:
                    p=controls(job)/".lock";require(p.is_file(),"original producer control lock absent")
                    stream=guards.enter_context(p.open("rb"))
                    try:fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
                    except BlockingIOError:busy.append(job["shard"])
                if busy:
                    state({"status":"pending_active_original_lock","shards":busy,"exporter_started":False,"formal_training":False})
                    time.sleep(60);continue
                hashes_unchanged();require(sha(__file__)==args.self_sha,"supervisor changed before export")
                require(not duplicate_processes(),"new same-scope exporter appeared; do not repeat")
                require(not OUTPUT.exists(),"export output appeared while waiting; preserve it, do not repeat")
                exclusive(CONTROL/"export.launch.claim.json",{"entry_sha256":args.self_sha,"argv":ARGV,"terminal_proofs":proofs,
                    "frozen_sha256":EXPECTED,"owner_identity":current_identity(os.getpid()),"claimed_once":True,"claim_is_not_export_completion":True})
                environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
                with (CONTROL/"exporter.log").open("xb") as log:
                    child=subprocess.Popen(ARGV,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    exclusive(CONTROL/"export.live.json",{"child_identity":current_identity(child.pid),"owner_identity":current_identity(os.getpid()),
                        "argv":ARGV,"frozen_sha256":EXPECTED,"CPU_only":True,"export_completed":False})
                    state({"status":"exporter_active","child_pid":child.pid,"exporter_started":True,"formal_training":False})
                    code=child.wait()
                hashes_unchanged();require(sha(__file__)==args.self_sha,"supervisor changed during export")
                receipt={"exit_code":code,"child_pid":child.pid,"frozen_sha256":EXPECTED,"source_and_inputs_unchanged":True,
                    "log_path":str(CONTROL/"exporter.log"),"CPU_only":True,"GPU_tasks_started":False,"formal_training":False}
                if code==0:
                    manifest=OUTPUT/"manifest.json";captions=OUTPUT/"captions.jsonl";value=json.loads(manifest.read_text())
                    require(value["schema"]=="motionscript_visual_terminal_export_v1" and value["status"]=="terminal_verified_unreviewed","export result is not a verified terminal ledger")
                    require(value["verified_ids"]==12000 and value["all_expected_ids_accounted_for"] is True,"original export union incomplete")
                    require(value["producer_sha256"]==EXPECTED[str(PRODUCER)] and value["exporter_sha256"]==EXPECTED[str(EXPORTER)]
                        and value["launch_plan_sha256"]==EXPECTED[str(PLAN)] and value["input_manifest_sha256"]==EXPECTED[str(INPUT)],"export source/input binding changed")
                    require(sha(captions)==value["captions_sha256"],"final caption stream SHA differs")
                    require(all(value.get(k) is False for k in ("semantic_approval","human_review_completed","formal_training","ground_truth")),"export claims an unsupported approval")
                    receipt.update(status="original_export_terminal_verified_unreviewed",manifest_path=str(manifest),manifest_sha256=sha(manifest),captions_sha256=value["captions_sha256"],
                        sid_status_counts=value["sid_status_counts"],window_status_counts=value["window_status_counts"],
                        usable_full_caption_count=value["usable_full_caption_count"],caption_delivery_complete=value["caption_delivery_complete"],
                        ledger_complete_not_semantic_approval=True)
                else:receipt.update(status="original_export_rejected_or_failed",actual_export_complete=False,no_automatic_relaunch=True)
                exclusive(CONTROL/"export.worker_exit.json",receipt);state(receipt)
                print(json.dumps(receipt),flush=True);return 0 if code==0 else 1
    except BaseException as error:
        if child is not None and child.poll() is None:
            # Do not kill a running source verification. Preserve the real child
            # identity for explicit operator inspection; no second Popen is made.
            value={"status":"supervisor_interrupted_export_may_be_active","child_pid":child.pid,"error":repr(error),"do_not_relaunch":True}
        else:value={"status":"supervisor_blocked_or_failed","error":repr(error),"do_not_relaunch_blindly":True,"export_child_pid":child.pid if child else None}
        if not (CONTROL/"failure.json").exists():exclusive(CONTROL/"failure.json",value)
        print(json.dumps(value),flush=True);return 1

if __name__=="__main__":raise SystemExit(main())
