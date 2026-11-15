#!/usr/bin/env python3
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

if __package__:
    from .entry import NoRedirect, OssTransport, sha_file, write_new
else:
    from entry import NoRedirect, OssTransport, sha_file, write_new

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "vllm019-py310-cu128-public-runtime.tar"
ARCHIVE_BYTES = 5322977280
ARCHIVE_SHA256 = "67ade6b5d360f7d8bdf0b2a49ecfd3b08f638a0fa5e67f3953b45a7ad8f617f0"
LOCK_SHA256 = "de4361d227491b6453cf90114d58251bd9b41ac9bc5aec78761a780970352de6"
ARCHIVE_KEY = "/niezichao.nzc/m2d_train_v1/runtimes/vllm019-py310-cu128/sha256-" + ARCHIVE_SHA256 + "/" + ARCHIVE_NAME


def download_runtime(capabilities, target, profile=None):
    profile = profile or dict(archive_name=ARCHIVE_NAME, archive_bytes=ARCHIVE_BYTES,
                             archive_sha256=ARCHIVE_SHA256, archive_key=ARCHIVE_KEY)
    url = capabilities["RUNTIME_GET"][profile["archive_name"]]
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com"
            or urllib.parse.unquote(parsed.path) != profile["archive_key"] or parsed.fragment):
        raise ValueError("Runtime capability violates its pinned HTTPS object scope")
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(url, timeout=600) as response, target.open("xb") as stream:
        remaining = profile["archive_bytes"]
        while remaining:
            block = response.read(min(8 << 20, remaining))
            if not block:
                raise ValueError("Runtime archive is shorter than its pinned size")
            stream.write(block)
            remaining -= len(block)
        if response.read(1):
            raise ValueError("Runtime archive exceeds its pinned size")
    if sha_file(target) != profile["archive_sha256"]:
        raise ValueError("Runtime archive SHA256 differs from the pinned public bundle")


def extract_runtime(archive, destination):
    destination.mkdir()
    root = destination.resolve()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        seen, total = set(), 0
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or str(name) != member.name.rstrip("/")
                    or str(name) in seen):
                raise ValueError("Unsafe or duplicate runtime archive member")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Runtime archive contains a special file")
            seen.add(str(name))
            total += member.size
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                base = root / name.parent if member.issym() else root
                target = (base / member.linkname).resolve()
                if link.is_absolute() or not target.is_relative_to(root):
                    raise ValueError("Runtime archive link escapes its private directory")
        if total > 16 << 30 or len(members) > 30000:
            raise ValueError("Runtime archive expansion exceeds its declared capacity")
        tar.extractall(destination, members=members, filter="data")


def install_runtime(destination, profile=None):
    profile = profile or dict(lock_sha256=LOCK_SHA256, wheel_count=183, torch_build="2.10.0+cu128",
        packages={"transformers": "4.57.6", "lm-format-enforcer": "0.11.3"},
        imports=["lmformatenforcer.integrations.transformers"])
    lock = destination / "requirements.lock"
    if sha_file(lock) != profile["lock_sha256"]:
        raise ValueError("Runtime dependency lock changed")
    manifest = json.loads((destination / "wheel_manifest.json").read_text())
    if manifest.get("dependency_closure_verified") is not True or len(manifest["files"]) != profile["wheel_count"]:
        raise ValueError("Runtime dependency closure is incomplete")
    for item in manifest["files"]:
        path = destination / item["path"]
        if (not path.resolve().is_relative_to(destination.resolve()) or path.stat().st_size != item["bytes"]
                or sha_file(path) != item["sha256"]):
            raise ValueError("Runtime wheel differs from its pinned inventory")
    python = destination / "python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10"
    uv = destination / "tools/uv"
    venv = destination / "venv"
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="never", UV_CACHE_DIR=str(destination / "uv-cache"),
               UV_LINK_MODE="copy", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    probe = (
        "import torch, transformers, importlib, importlib.metadata as m, json, platform, sys; "
        "assert sys.prefix != sys.base_prefix and platform.python_version() == '3.10.22'; "
        "profile = json.loads(" + repr(json.dumps(profile)) + "); "
        "assert torch.__version__ == profile['torch_build']; "
        "assert all(m.version(k) == v for k, v in profile['packages'].items()); "
        "[importlib.import_module(name) for name in profile['imports']]"
    )
    commands = [
        [str(uv), "venv", "--offline", "--python", str(python), str(venv)],
        [str(uv), "pip", "install", "--offline", "--python", str(venv / "bin/python"), "--no-index",
         "--find-links", str(destination / "wheels"), "--require-hashes", "-r", str(lock)],
        [str(venv / "bin/python"), "-I", "-c", probe],
    ]
    for command in commands:
        subprocess.run(command, env=env, check=True, timeout=600)
    return venv / "bin/python", env


def main():
    if os.environ.get("RANK") not in (None, "", "0"):
        return
    config = json.loads((ROOT / "job_config.json").read_text())
    capabilities = json.loads((ROOT / "signed_io.json").read_text())
    report = dict(schema="choreobench.isolated_bootstrap.v1", run_id=config["run_id"],
        started_unix=time.time(), model_calls_attempted=0, status="bootstrap_started",
        archive_sha256=ARCHIVE_SHA256, lock_sha256=LOCK_SHA256)
    store = None
    output = None
    try:
        store = OssTransport(config, capabilities)
        output = ROOT / (config["run_id"] + "-bootstrap-records")
        output.mkdir()
        write_new(store, output / "bootstrap.started.json", report)
        parent = next((p for p in map(Path, ("/checkpoint/binary", "/checkpoint", "/tmp"))
            if p.is_dir() and shutil.disk_usage(p).free >= config["expected_bytes"] + (40 << 30)), None)
        if parent is None:
            raise OSError("Insufficient private scratch; existing data will not be deleted")
        scratch = parent / (config["run_id"] + "-runtime")
        scratch.mkdir()
        archive = scratch / ARCHIVE_NAME
        download_runtime(capabilities, archive)
        destination = scratch / "runtime"
        extract_runtime(archive, destination)
        python, env = install_runtime(destination)
        report.update(status="isolated_environment_ready", python=str(python), finished_unix=time.time())
        write_new(store, output / "bootstrap.ready.json", report)
    except Exception as error:
        report.update(status="bootstrap_failed", error_type=type(error).__name__, finished_unix=time.time())
        if store is not None and output is not None:
            write_new(store, output / "bootstrap.failed.json", report)
        print(json.dumps(report), flush=True)
        raise SystemExit(2)
    raise SystemExit(subprocess.run([str(python), str(ROOT / "entry.py")], env=env, check=False).returncode)


if __name__ == "__main__":
    main()
