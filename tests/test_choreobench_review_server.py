"""Public-only HTTP regression tests; all audio and plans are synthetic fixtures."""
from contextlib import contextmanager
import json
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
