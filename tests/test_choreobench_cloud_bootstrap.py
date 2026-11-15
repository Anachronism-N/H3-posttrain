import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def bootstrap(monkeypatch):
    root = Path(__file__).parents[1] / "experiments/structured_dev_20261114/cloud_qwen32_native"
    spec = importlib.util.spec_from_file_location("entry", root / "entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setitem(sys.modules, "entry", entry)
    spec = importlib.util.spec_from_file_location("cloud_bootstrap_test", root / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(path, entries):
    with tarfile.open(path, "w") as tar:
        for name, kind, body in entries:
            member = tarfile.TarInfo(name)
            if kind == "file":
                member.size = len(body)
                tar.addfile(member, io.BytesIO(body))
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                tar.addfile(member)
            else:
                member.type = {"symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                member.linkname = body
                tar.addfile(member)


@pytest.mark.parametrize("entries", [
    [("../outside", "file", b"bad")],
    [("/absolute", "file", b"bad")],
    [("link", "symlink", "../outside")],
    [("link", "symlink", "/outside")],
    [("link", "hardlink", "../outside")],
    [("fifo", "fifo", "")],
    [("same", "file", b"a"), ("same", "file", b"b")],
    [("./same", "file", b"a")],
    [("dir//same", "file", b"a")],
])
def test_rejects_unsafe_archive_before_extracting(bootstrap, tmp_path, entries):
    source = tmp_path / "archive.tar"
    archive(source, entries)
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError):
        bootstrap.extract_runtime(source, destination)
    assert not list(destination.iterdir())
    assert not (tmp_path / "outside").exists()


def test_allows_only_contained_interpreter_link(bootstrap, tmp_path):
    source = tmp_path / "archive.tar"
    archive(source, [("python", "directory", b""), ("python/python3.10", "file", b"python"),
                    ("python/python", "symlink", "python3.10")])
    destination = tmp_path / "runtime"
    bootstrap.extract_runtime(source, destination)
    assert (destination / "python/python").read_bytes() == b"python"
    with pytest.raises(FileExistsError):
        bootstrap.extract_runtime(source, destination)
    assert (destination / "python/python3.10").read_bytes() == b"python"


@pytest.mark.parametrize("mutation", ["http", "host", "path", "fragment"])
def test_runtime_url_is_exactly_scoped_before_network(bootstrap, tmp_path, mutation):
    url = "https://futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com" + bootstrap.ARCHIVE_KEY + "?Signature=fixture"
    if mutation == "http":
        url = url.replace("https:", "http:")
    elif mutation == "host":
        url = url.replace("futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com", "wrong.invalid")
    elif mutation == "path":
        url = url.replace(bootstrap.ARCHIVE_NAME, "another.tar")
    else:
        url += "#fragment"
    caps = {"RUNTIME_GET": {bootstrap.ARCHIVE_NAME: url}}
    with pytest.raises(ValueError, match="scope") as error:
        bootstrap.download_runtime(caps, tmp_path / "runtime.tar")
    assert "Signature=" not in str(error.value)
    assert not (tmp_path / "runtime.tar").exists()


@pytest.mark.parametrize("body", [b"abc", b"abcde", b"xxxx"])
def test_short_oversize_and_hash_mismatch_are_rejected(bootstrap, tmp_path, monkeypatch, body):
    monkeypatch.setattr(bootstrap, "ARCHIVE_BYTES", 4)
    monkeypatch.setattr(bootstrap, "ARCHIVE_SHA256", hashlib.sha256(b"abcd").hexdigest())
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(body)))
    caps = {"RUNTIME_GET": {bootstrap.ARCHIVE_NAME:
        "https://futurelab-dpro-hz.oss-cn-hangzhou.aliyuncs.com" + bootstrap.ARCHIVE_KEY + "?Signature=fixture"}}
    target = tmp_path / "runtime.tar"
    with pytest.raises(ValueError):
        bootstrap.download_runtime(caps, target)
    assert target.stat().st_size <= 4


def test_changed_lock_prevents_installation(bootstrap, tmp_path, monkeypatch):
    (tmp_path / "requirements.lock").write_text("changed")
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="lock changed"):
        bootstrap.install_runtime(tmp_path)
    assert not calls


def test_installer_uses_fresh_isolated_offline_hashed_environment(bootstrap, tmp_path, monkeypatch):
    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture hashes")
    monkeypatch.setattr(bootstrap, "LOCK_SHA256", hashlib.sha256(lock.read_bytes()).hexdigest())
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    files = []
    for index in range(183):
        path = wheels / (str(index) + ".whl")
        path.write_bytes(bytes([index]))
        files.append(dict(path="wheels/" + path.name, bytes=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path / "wheel_manifest.json").write_text(json.dumps(dict(dependency_closure_verified=True, files=files)))
    calls = []
    monkeypatch.setenv("PYTHONPATH", "unwanted-shared-path")
    monkeypatch.setenv("PYTHONHOME", "unwanted-shared-home")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    python, env = bootstrap.install_runtime(tmp_path)
    assert python == tmp_path / "venv/bin/python"
    assert len(calls) == 3
    assert "--offline" in calls[0][0]
    assert "--no-index" in calls[1][0] and "--require-hashes" in calls[1][0] and "--offline" in calls[1][0]
    assert "--system-site-packages" not in calls[0][0]
    assert "-I" in calls[2][0]
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["UV_PYTHON_DOWNLOADS"] == "never"
    assert all(kwargs["check"] is True for _, kwargs in calls)
    (wheels / "0.whl").write_bytes(b"different")
    calls.clear()
    with pytest.raises(ValueError, match="wheel differs"):
        bootstrap.install_runtime(tmp_path)
    assert calls == []
