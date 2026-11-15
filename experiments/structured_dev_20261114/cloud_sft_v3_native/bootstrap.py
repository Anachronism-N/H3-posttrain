#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))

from experiments.structured_dev_20261114.cloud_qwen32_native.bootstrap import download_runtime, extract_runtime, install_runtime
from experiments.structured_dev_20261114.cloud_qwen32_native.entry import sha_file, write_new
from experiments.structured_dev_20261114.cloud_sft_v3_native.entry import SftTransport


def main():
    if os.environ.get("RANK") not in (None, "", "0"):
        return
    config = json.loads((ROOT / "job_config.json").read_text())
    capabilities = json.loads((ROOT / "signed_io.json").read_text())
    report = dict(schema="choreobench.sft_isolated_bootstrap.v1", run_id=config["run_id"],
        started_unix=time.time(), model_calls_attempted=0, status="bootstrap_started")
    store, output = None, None
    try:
        store = SftTransport(config, capabilities)
        output = ROOT / (config["run_id"] + "-bootstrap-records")
        output.mkdir()
        write_new(store, output / "bootstrap.started.json", report)
        profile_path = ROOT / "runtime_profile.json"
        if sha_file(profile_path) != config["runtime_profile_sha256"]:
            raise ValueError("SFT runtime profile changed")
        profile = json.loads(profile_path.read_text())
        parent = next((p for p in map(Path, ("/checkpoint/binary", "/checkpoint", "/tmp"))
            if p.is_dir() and shutil.disk_usage(p).free >= config["source_asset_bytes"] + (40 << 30)), None)
        if parent is None:
            raise OSError("Insufficient private scratch; no existing data will be deleted")
        scratch = parent / (config["run_id"] + "-runtime")
        scratch.mkdir()
        archive = scratch / profile["archive_name"]
        if archive.parent != scratch:
            raise ValueError("Runtime archive name escapes its private directory")
        download_runtime(capabilities, archive, profile)
        destination = scratch / "runtime"
        extract_runtime(archive, destination)
        python, env = install_runtime(destination, profile)
        report.update(status="isolated_environment_ready", python=str(python),
                      archive_sha256=profile["archive_sha256"], lock_sha256=profile["lock_sha256"], finished_unix=time.time())
        write_new(store, output / "bootstrap.ready.json", report)
    except Exception as error:
        report.update(status="bootstrap_failed", error_type=type(error).__name__, finished_unix=time.time())
        if output is not None:
            write_new(store, output / "bootstrap.failed.json", report)
        print(json.dumps(report), flush=True)
        raise SystemExit(2)
    raise SystemExit(subprocess.run([str(python), str(ROOT / "entry.py")], env=env, check=False).returncode)


if __name__ == "__main__":
    main()
