"""Freeze explicit full-pool contract; read-only fresh holder capture; SHA-preserving build."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time

RT = Path(__file__).resolve().parent
EXP = RT.parent
spec = importlib.util.spec_from_file_location("full12k_resource", RT / "resource_manager.py")
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
spec = importlib.util.spec_from_file_location("full12k_capture", RT / "capture_build.py")
capture_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_base)
spec = importlib.util.spec_from_file_location("full12k_driver_build", RT / "full12k_driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
PAIRS = {"N3": ((2, 3), (4, 5)), "N6": ((4, 5), (6, 7)), "N5": ((6,),)}
HOSTS = {"N3": "28.216.19.213", "N5": "28.216.19.137", "N6": "28.216.19.225"}


def option(argv, key, value):
    i = argv.index(key)
    argv[i + 1] = str(value)


def freeze():
    visual = json.loads((RT / "stable_roi3_v3_launch.json").read_text())
    roi = json.loads((RT / "hand_roi_gpu_launch_v3.json").read_text())
    files = dict(visual["source_sha256"])
    files.update(roi["source_sha256"])
    for name, expected in files.items():
        if manager.sha(name) != expected:
            raise RuntimeError("Validated stable source changed: " + name)
    for name in ("full12k_driver.py", "full12k_roi_resource_manager.py", "full12k_launch.py",
                 "resource_manager.py", "capture_build.py"):
        files[str(RT / name)] = manager.sha(RT / name)
    files[roi["modelref"]["path"]] = roi["modelref"]["sha256"]
    files[visual["modelref"]["path"]] = visual["modelref"]["sha256"]
    roi_ref = json.loads(Path(roi["modelref"]["path"]).read_text())
    for value in roi_ref["files"].values():
        files[value["path"]] = value["sha256"]
    reuse = {}
    old = EXP / "hand_roi_gpu_fixed3_v3"
    proof = old / "runtime_profile" / "cuda_execution_receipt.json"
    if json.loads(proof.read_text()).get("real_CUDA_nodes_executed") is not True:
        raise RuntimeError("Fixed3 ROI has no actual CUDA compute receipt")
    for source_path in sorted(old.glob("*/source_manifest.json")):
        directory = source_path.parent
        source = json.loads(source_path.read_text())
        manifest = json.loads((directory / "roi_manifest.json").read_text())
        per = {str(directory / name): manager.sha(directory / name) for name in
               ("source_manifest.json", "prepare_receipt.json", "roi_manifest.json", "extract_receipt.json")}
        for asset in [manifest["raw"]] + manifest["crops"]:
            if manager.sha(asset["path"]) != asset["sha256"]:
                raise RuntimeError("Fixed3 ROI raw/crop changed")
            per[asset["path"]] = asset["sha256"]
        per[str(proof)] = manager.sha(proof)
        reuse[source["id"]] = {"files": per, "roi_manifest": str(directory / "roi_manifest.json"),
                               "cuda_receipt": str(proof)}
        files.update(per)
    if len(reuse) != 3:
        raise RuntimeError("Reuse must cover exactly the three verified native sources")
    contract_path = RT / "full12k_contract_v1.json"
    contract = {"schema": "full12k_frozen_pipeline_v1", "plan": visual["plan"]["path"],
        "plan_sha256": visual["plan"]["sha256"], "input_count": 12000, "files": files,
        "roi_source_dir": str(Path(roi["producer"]["path"]).parent),
        "visual_source_dir": str(Path(visual["producer"]["path"]).parent),
        "roi_modelref": roi["modelref"]["path"], "roi_params": source["params"],
        "roi_out": str(EXP / "hand_roi_full12000_v1"), "visual_out": str(EXP / "full12000_stable_roi_v1"),
        "reuse": reuse, "human_approved": False, "semantic_approval": False}
    driver.immutable(contract_path, contract)
    bindings = {**files, str(contract_path): manager.sha(contract_path)}
    producer = {"path": str(RT / "full12k_driver.py"), "sha256": manager.sha(RT / "full12k_driver.py")}
    for shard, (node, gpus) in enumerate((("N3", [2, 3]), ("N3", [4, 5]), ("N6", [4, 5]), ("N6", [6, 7]))):
        cfg = copy.deepcopy(visual)
        cfg.pop("capture", None)
        cfg.pop("previous_generation_gate", None)
        cfg.update(node=node, gpus=gpus, holders=[], producer=producer, source_sha256=bindings,
            targetcount=3000, input_count=12000, source_selection={"shard": shard, "nshard": 4, "limit": 0},
            phase="full12k_visual_s%d_v1" % shard, reuse_existing_results=False,
            out=str(Path(contract["visual_out"]) / ("shard" + str(shard))),
            queue=str(EXP / "queues" / ("full12k_visual_s%d_v1" % shard)))
        cfg["env"]["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        argv = [visual["argv"][0], "-B", "-u", producer["path"], "visual", "--contract", str(contract_path)]
        argv += visual["argv"][visual["argv"].index("--plan"):]
        for key, value in (("--out", cfg["out"]), ("--limit", 0), ("--shard", shard), ("--nshard", 4),
                           ("--hand-roi-root", contract["roi_out"])):
            option(argv, key, value)
        cfg["argv"] = argv
        driver.immutable(RT / ("full12k_visual_s%d_v1_template.json" % shard), cfg)
    cfg = copy.deepcopy(roi)
    cfg.pop("capture", None)
    cfg.update(holders=[], producer=producer, plan=visual["plan"], source_sha256=bindings,
        targetcount=12000, input_count=12000, source_selection={"shard": 0, "nshard": 1, "limit": 0},
        phase="full12k_roi_v1", out=contract["roi_out"], queue=str(EXP / "queues/full12k_roi_v1"),
        reuse_existing_results=False)
    cfg["argv"] = ["/opt/venv_eval/bin/python", "-B", "-u", producer["path"], "roi", "--contract", str(contract_path),
        "--plan", contract["plan"], "--out", cfg["out"], "--stage", "run", "--limit", "0", "--shard", "0",
        "--nshard", "1", "--ffmpeg", "/usr/local/bin/ffmpeg", "--ffprobe", "/opt/ms_tools/ffprobe"]
    cfg["env"]["PYTHONDONTWRITEBYTECODE"] = "1"
    driver.immutable(RT / "full12k_roi_v1_template.json", cfg)
    return {"contract": str(contract_path), "contract_sha256": manager.sha(contract_path),
            "manager_sha256": manager.sha(manager.__file__), "templates": 5, "signals_sent": False}


def capture(node, gpus, out):
    if tuple(gpus) not in PAIRS.get(node, ()):
        raise RuntimeError("Not a currently authorized full-pool GPU assignment")
    local_ips = set(socket.gethostbyname_ex(socket.gethostname())[2])
    local_ips.update(subprocess.check_output(["hostname", "-I"], text=True).split())
    if HOSTS[node] not in local_ips:
        raise RuntimeError("Capture command was sent to the wrong physical node")
    pids = []
    for gpu in gpus:
        uuid = manager.uuid_for(gpu)
        running = [p for p in manager.apps() if p["uuid"] == uuid]
        if len(running) != 1:
            raise RuntimeError("Expected one exact holder, not unknown compute")
        ident = manager.identity(running[0]["pid"])
        if ident["argv"][-1:] != [str(manager.HOLDER)]:
            raise RuntimeError("Occupied by non-holder; capture does not signal anything")
        pids.append(running[0]["pid"])
    if len(gpus) == 2:
        result = capture_base.capture(node, gpus, pids)
    else:
        gpu, pid = gpus[0], pids[0]
        expected = dict(manager.identity(pid), gpu=gpu, uuid=manager.uuid_for(gpu))
        fd = os.pidfd_open(pid)
        try:
            if manager.select.select([fd], [], [], 0)[0]:
                raise RuntimeError("Captured holder exited")
            manager.check_holder(expected, manager.identity(pid), gpu, expected["uuid"], manager.apps())
        finally:
            os.close(fd)
        reserve = Path("/tmp/od_reserve_%d" % gpu)
        manager.lease_snapshot(reserve)
        result = {"schema": "ms_bar_resource_capture_v1", "node": node, "gpus": gpus,
            "holders": [expected], "hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "captured_unix": time.time(), "headroom": manager.headroom(), "signals_sent": False,
            "files": [{"gpu": gpu, "lock": capture_base.snapshot(Path("/tmp/ms12k_gpu_%d.lock" % gpu)),
                       "reserve": capture_base.snapshot(reserve)}]}
    driver.immutable(out, result)
    return {"capture": out, "sha256": manager.sha(out), "signals_sent": False,
            "holders": [{"gpu": x["gpu"], "pid": x["pid"]} for x in result["holders"]]}


def build(template_path, capture_path, out):
    cfg, fresh = json.loads(Path(template_path).read_text()), json.loads(Path(capture_path).read_text())
    if (fresh["node"] != cfg["node"] or fresh["gpus"] != cfg["gpus"] or fresh.get("signals_sent") is not False
            or not 0 <= time.time() - fresh["captured_unix"] <= 1200):
        raise RuntimeError("Mismatch/stale read-only capture")
    # Never turn changed code/input into an apparently fresh validated binding.
    for name, expected in cfg["source_sha256"].items():
        if manager.sha(name) != expected:
            raise RuntimeError("Frozen source/input changed, refusing rebind: " + name)
    cfg["holders"] = fresh["holders"]
    cfg["capture"] = {"path": str(Path(capture_path).resolve()), "sha256": manager.sha(capture_path),
        "hostname": fresh["hostname"], "boot_id": fresh["boot_id"], "captured_unix": fresh["captured_unix"]}
    if len(cfg["gpus"]) == 1:
        spec = importlib.util.spec_from_file_location("full12k_roi_checks", RT / "full12k_roi_resource_manager.py")
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        wrapper.validate_roi(cfg)
    else:
        manager.validate_pair(cfg)
    manager.check_sources(cfg, template_path, manager.sha(template_path), manager.sha(manager.__file__))
    driver.immutable(out, cfg)
    return {"launch": out, "config_sha256": manager.sha(out), "manager_sha256": manager.sha(manager.__file__),
            "signals_sent": False}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("capture")
    p.add_argument("--node", required=True, choices=list(PAIRS))
    p.add_argument("--gpus", required=True, nargs="+", type=int)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--template", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    result = freeze() if a.command == "freeze" else capture(a.node, a.gpus, a.out) if a.command == "capture" else build(a.template, a.capture, a.out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
