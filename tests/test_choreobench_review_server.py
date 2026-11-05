"""Public-only HTTP regression tests; all audio and plans are synthetic fixtures."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request
import wave

import pytest

from choreobench.io import file_sha, write_json
from choreobench.review_server import create_review_server, _range
from choreobench.reviews import build_review_pack, blank_review_template


@pytest.fixture
def public(tmp_path):
    root = tmp_path / "public"
    (root / "assets").mkdir(parents=True)
    audio = root / "assets" / "fixture.wav"
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16)
        f.writeframes(b"\x01\x00" * 81)
    song = dict(song_id="fixture", split="dev", song_group_id=None, artist_group_id=None,
        near_duplicate_group_id=None, brief="Synthetic public-server fixture, not musical evidence.",
        audio=dict(path=str(audio), sha256=file_sha(audio), sample_rate=16, num_samples=81),
        music=dict(source="unknown", beats=[], downbeats=[], sections=[dict(section_id="s0",
            start=0, end=81, label="unknown", repeat_group=None)]))
    plan = dict(schema="choreobench.plan.v1", song_id="fixture", sample_rate=16, num_samples=81,
        motifs=[], events=[dict(event_id="e0", start=0, end=81, track="body", action="Steps left.",
            attributes={}, entry={}, exit={}, motif_id=None, evidence=[])], provenance={})
    pack = build_review_pack([dict(song=song, plan=plan, method_id="private-method")])["public"]
    destination = root / pack["items"][0]["song"]["audio"]["path"]
    destination.write_bytes(audio.read_bytes())
    write_json(root / "pack.json", pack)
    write_json(root / "reviews.template.json", blank_review_template(pack))
    (root / "index.html").write_text("<html>public fixture</html>")
    (root / "secret.json").write_text('"not a public route"')
    return root, pack, destination


@contextmanager
def running(public):
    root, _, _ = public
    server = create_review_server(root, proxy_prefix="/proxy/8780")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def fetch(url, method="GET", headers=None):
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        response = urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.headers, response.read()


def test_public_prefix_head_and_audio_ranges(public):
    root, pack, audio = public
    route = "/" + pack["items"][0]["song"]["audio"]["path"]
    with running(public) as url:
        assert fetch(url + "/")[0] == fetch(url + "/proxy/8780/")[0] == 200
        status, headers, body = fetch(url + route, headers={"Range": "bytes=0-43"})
        assert status == 206 and body == audio.read_bytes()[:44]
        assert headers["Content-Range"] == "bytes 0-43/" + str(audio.stat().st_size)
        assert fetch(url + route, headers={"Range": "bytes=-8"})[2] == audio.read_bytes()[-8:]
        assert fetch(url + route, method="HEAD")[2] == b""
        assert fetch(url + route, headers={"Range": "bytes=99999-"})[0] == 416
        assert fetch(url + route, headers={"Range": "bytes=0-1,3-4"})[0] == 416
        health = json.loads(fetch(url + "/health")[2])
        assert health["read_only"] and health["review_items"] == 1 and not health["private_routes"]


@pytest.mark.parametrize("path", ["/secret.json", "/private/bindings.json", "/../private/bindings.json",
    "/%2e%2e/private/bindings.json", "/assets/../pack.json", "/assets", "/?path=/etc/passwd",
    "/assets/fixture.wav", "/proxy/8780/../private/bindings.json"])
def test_no_directory_traversal_or_unlisted_file_routes(public, path):
    with running(public) as url:
        if path.startswith("/?"):
            assert fetch(url + path)[2] == (public[0] / "index.html").read_bytes()
        else:
            assert fetch(url + path)[0] == 404
        assert fetch(url + "/", method="POST")[0] == 405


def test_mutated_audio_is_not_served_and_page_is_frozen(public):
    root, pack, audio = public
    with running(public) as url:
        audio.write_bytes(audio.read_bytes() + b"changed")
        assert fetch(url + "/" + pack["items"][0]["song"]["audio"]["path"])[0] == 409
        (root / "index.html").write_text("private accidental replacement")
        assert b"private accidental" not in fetch(url + "/")[2]


@pytest.mark.parametrize("method,headers", [("GET", {}), ("HEAD", {}), ("GET", {"Range": "bytes=0-43"}),
                                           ("HEAD", {"Range": "bytes=-8"})])
def test_same_metadata_different_audio_bytes_are_rejected(public, monkeypatch, method, headers):
    from types import SimpleNamespace
    import choreobench.review_server as module
    root, pack, audio = public
    route = "/" + pack["items"][0]["song"]["audio"]["path"]
    with audio.open("rb") as stream:
        original_stat = os.fstat(stream.fileno())
    original_fstat = os.fstat
    def stable_metadata(fd):
        result = original_fstat(fd)
        # Simulate the Windows equal-size/retained-times case on every CI OS.
        # Other files' metadata are unaffected.
        if (result.st_dev, result.st_ino) == (original_stat.st_dev, original_stat.st_ino):
            return SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino, st_size=result.st_size,
                st_mode=result.st_mode, st_mtime_ns=original_stat.st_mtime_ns,
                st_ctime_ns=original_stat.st_ctime_ns)
        return result
    monkeypatch.setattr(module.os, "fstat", stable_metadata)
    with running(public) as url:
        initial = audio.read_bytes()
        with audio.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(bytes([initial[-1] ^ 1]))
        with audio.open("rb") as stream:
            changed_stat = module.os.fstat(stream.fileno())
        assert module._signature(changed_stat) == module._signature(original_stat)
        status, _, body = fetch(url + route, method=method, headers=headers)
        assert status == 409
        assert body == (b"" if method == "HEAD" else b"Request not available")


@pytest.mark.parametrize("headers", [{}, {"Range": "bytes=-8"}])
def test_audio_changed_after_verification_cannot_change_served_bytes(public, monkeypatch, headers):
    from http.server import BaseHTTPRequestHandler
    root, pack, audio = public
    route = "/" + pack["items"][0]["song"]["audio"]["path"]
    original = audio.read_bytes()
    end_headers = BaseHTTPRequestHandler.end_headers
    mutations = []
    def mutate_after_headers(handler):
        end_headers(handler)
        if handler.path == route and handler.command == "GET":
            with audio.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(bytes([original[-1] ^ 1]))
            mutations.append(True)
    monkeypatch.setattr(BaseHTTPRequestHandler, "end_headers", mutate_after_headers)
    with running(public) as url:
        status, _, body = fetch(url + route, headers=headers)
        assert status == (206 if headers else 200)
        assert body == (original[-8:] if headers else original)
        assert mutations == [True] and audio.read_bytes() != original


def test_audio_handles_are_closed_on_success_failure_and_server_stop(public, monkeypatch):
    root, pack, audio = public
    route = "/" + pack["items"][0]["song"]["audio"]["path"]
    original_open = Path.open
    handles = []
    def observed_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == audio:
            handles.append(stream)
        return stream
    monkeypatch.setattr(Path, "open", observed_open)
    with running(public) as url:
        assert fetch(url + route)[0] == 200
        assert fetch(url + route, method="HEAD", headers={"Range": "bytes=-8"})[0] == 206
        assert fetch(url + route, headers={"Range": "bytes=99999-"})[0] == 416
        audio.write_bytes(audio.read_bytes() + b"changed")
        assert fetch(url + route)[0] == 409
        assert handles and all(stream.closed for stream in handles)
    assert all(stream.closed for stream in handles)
    # Also exercises Windows deletion semantics: no retained audio handle.
    audio.unlink()


def test_path_and_handle_timestamp_semantics_do_not_cause_false_audio_rejection(public, monkeypatch):
    from types import SimpleNamespace
    original = Path.stat
    audio = public[2]
    def different_path_ctime(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == audio:
            return SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino, st_size=result.st_size,
                st_mode=result.st_mode, st_mtime_ns=result.st_mtime_ns, st_ctime_ns=result.st_ctime_ns + 1000)
        return result
    monkeypatch.setattr(Path, "stat", different_path_ctime)
    with running(public) as url:
        route = "/" + public[1]["items"][0]["song"]["audio"]["path"]
        assert fetch(url + route)[0] == 200
        assert fetch(url + route)[2] == audio.read_bytes()


def test_symlink_cannot_publish_private_content(public):
    root, _, audio = public
    outside = root.parent / "private-audio.wav"
    outside.write_bytes(audio.read_bytes())
    audio.unlink()
    try:
        audio.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Platform does not permit test symlinks")
    with pytest.raises(ValueError, match="symlinks"):
        create_review_server(root)


@pytest.mark.parametrize("header", ["bytes=-0", "bytes=-", "invalid", "bytes=4-2", "bytes=200-"])
def test_invalid_ranges(header):
    with pytest.raises(ValueError):
        _range(header, 100)
