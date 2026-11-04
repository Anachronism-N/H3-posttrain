import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import wave

import pytest

from choreobench import providers, resident
from choreobench.audio import inspect_audio
from choreobench.io import object_sha


class FakeNativeProvider:
    instances = []

    def __init__(self, spec, role, base_dir=None):
        assert role == "local"
        self.identity = dict(kind="native", family="fixture", synthetic=False,
                             artifact_identity={"fixture_only": True})
        self.adapter = None
        self.verification = None
        self.loads, self.calls, self.active, self.peak = 0, [], 0, 0
        self.fail = False
        self.__class__.instances.append(self)

    def generate(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.adapter is None:
                self.loads += 1
                self.adapter = object()
                self.verification = {"fixture": "verified_not_real_model"}
            self.calls.append(copy.deepcopy(request))
            time.sleep(.015)
            if self.fail:
                raise RuntimeError("No retry after possibly completed computation")
            return dict(text="[0-5.0625s] Raise both arms.", raw={"fixture_call": len(self.calls)},
                        identity=copy.deepcopy(self.identity), synthetic=False, stop_reason="eos",
                        costs=dict(input_tokens=12, output_tokens=5, latency_seconds=.015))
        finally:
            self.active -= 1


def audio_file(path, rate=16, count=81):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * count)
    audio = inspect_audio(path)
    audio.pop("pcm")
    return audio


@contextmanager
def running_server(monkeypatch, tmp_path, *, auth=""):
    monkeypatch.setattr(providers, "NativeProvider", FakeNativeProvider)
    root = tmp_path / "audio"
    root.mkdir(exist_ok=True)
    audio = audio_file(root / "window.wav")
    server = resident.create_resident_server({"kind": "native", "model": "fixture"}, root, api_key_env=auth)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    url = "http://127.0.0.1:" + str(server.server_address[1])
    try:
        yield server, url, audio
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def config(server, url, **changes):
    spec = dict(kind="native_service", base_url=url, expected_identity_sha256=server.identity_sha256,
                timeout_seconds=3., api_key_env="")
    spec.update(changes)
    return spec


def request(audio, **changes):
    item = dict(call_id="song:0:0", response_kind="local_caption", audio=copy.deepcopy(audio),
                seed=42, max_output_tokens=32)
    item.update(changes)
    return item


def http_json(url, body=None, headers=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
    http = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(http, timeout=3.) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def envelope(server, item):
    return dict(schema=resident.REQUEST_SCHEMA, expected_identity_sha256=server.identity_sha256, request=item)


def test_health_is_lazy_and_identity_pinned_and_full_raw_preserved(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        status, health = http_json(url + "/health")
        assert status == 200 and health["loaded"] is False
        assert health["identity_sha256"] == object_sha(health["identity"])
        assert server.provider.loads == 0 and server.provider.calls == []
        assert set(health["identity"]["implementation_sha256"]) == {"resident.py", "providers.py"}
        client = providers.create_provider(config(server, url))
        assert client.identity["remote_identity_independently_verified"] is False
        assert client.identity["base_url"] == url
        first = client.generate(request(audio))
        second = client.generate(request(audio, call_id="song:1:0"))
        assert server.provider.loads == 1 and len(server.provider.calls) == 2
        assert first["identity"] == client.identity == second["identity"]
        assert first["raw"]["remote_response"]["result"]["identity"] == server.provider.identity
        assert first["raw"]["remote_response"]["result"]["raw"] == {"fixture_call": 1}
        assert first["stop_reason"] == "eos" and first["costs"]["output_tokens"] == 5
        assert first["raw"]["startup_health_attestation"]["loaded"] is False
        after_load = resident.NativeServiceProvider(config(server, url))
        assert after_load.health["loaded"] is True
        assert after_load.identity == client.identity  # Lazy loading must not break run resume.
        assert len(server.provider.calls) == 2


def test_result_matches_recorded_result_contract(monkeypatch, tmp_path):
    from choreobench.baseline import _recorded_result
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        req = request(audio)
        result = client.generate(req)
        assert _recorded_result(dict(success=True, result=result), client.identity, req) == result


def test_same_provider_serializes_concurrent_generations(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda i: client.generate(request(audio, call_id=str(i))), range(8)))
        assert all(r["stop_reason"] == "eos" for r in results)
        assert len(server.provider.calls) == 8 and server.provider.loads == 1 and server.provider.peak == 1


def test_auth_required_health_and_generation_no_secret_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOREO_TEST_SERVICE_KEY", "test-private-token")
    with running_server(monkeypatch, tmp_path, auth="CHOREO_TEST_SERVICE_KEY") as (server, url, audio):
        assert http_json(url + "/health")[0] == 401
        assert http_json(url + "/v1/local-caption", envelope(server, request(audio)))[0] == 401
        assert http_json(url + "/health", headers={"Authorization": "Bearer wrong"})[0] == 401
        with pytest.raises(ValueError, match="health unavailable"):
            resident.NativeServiceProvider(config(server, url))
        client = resident.NativeServiceProvider(config(server, url, api_key_env="CHOREO_TEST_SERVICE_KEY"))
        result = client.generate(request(audio))
        assert result["stop_reason"] == "eos"
        assert "test-private-token" not in json.dumps(result)


@pytest.mark.parametrize("change", [
    {"response_kind": "plan"}, {"messages": [{"role": "user", "content": "secret goals"}]},
    {"song": {"brief": "not audio"}}, {"seed": True}, {"seed": -1}, {"seed": 2 ** 63},
    {"max_output_tokens": True}, {"max_output_tokens": 0}, {"max_output_tokens": 65537},
    {"call_id": ""}, {"call_id": "x" * 513},
])
def test_audio_only_whitelist_and_budgets_rejected_without_generation(monkeypatch, tmp_path, change):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        item = request(audio, **change)
        client = resident.NativeServiceProvider(config(server, url))
        with pytest.raises(ValueError):
            client.generate(item)
        assert http_json(url + "/v1/local-caption", envelope(server, item))[0] == 400
        assert server.provider.calls == []


@pytest.mark.parametrize("change", [
    {"sample_rate": True}, {"sample_rate": 0}, {"sample_rate": 192001},
    {"num_samples": 80}, {"num_samples": True}, {"sha256": "0" * 64},
    {"sha256": "A" * 64}, {"transcript": "inject"}, {"path": "window.wav"},
    {"parent_start": 0},
    {"parent_start": True, "parent_end": 82, "parent_audio_sha256": "0" * 64},
    {"parent_start": 1, "parent_end": 81, "parent_audio_sha256": "0" * 64},
    {"parent_start": 0, "parent_end": 81, "parent_audio_sha256": "bad"},
])
def test_audio_metadata_rejected_on_server(monkeypatch, tmp_path, change):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        audio.update(change)
        assert http_json(url + "/v1/local-caption", envelope(server, request(audio)))[0] == 400
        assert server.provider.calls == []


def test_actual_clock_sha_outside_and_symlink_checked(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        outside = audio_file(tmp_path / "outside.wav")
        assert http_json(url + "/v1/local-caption", envelope(server, request(outside)))[0] == 400
        wrong_clock = audio_file(Path(audio["path"]), rate=32, count=162)
        wrong_clock.update(sample_rate=16, num_samples=81)
        assert http_json(url + "/v1/local-caption", envelope(server, request(wrong_clock)))[0] == 400
        # Exact bytes changed since a caller's collection are not reinterpreted.
        assert http_json(url + "/v1/local-caption", envelope(server, request(audio)))[0] == 400
        link = Path(audio["path"]).parent / "link.wav"
        try:
            link.symlink_to(tmp_path / "outside.wav")
        except OSError:
            pass  # Windows may not grant symlink privilege.
        else:
            escaped = copy.deepcopy(outside)
            escaped["path"] = str(link)
            assert http_json(url + "/v1/local-caption", envelope(server, request(escaped)))[0] == 400
        assert server.provider.calls == []


def test_valid_parent_binding_and_no_context_is_forwarded(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        audio.update(parent_start=81, parent_end=162, parent_audio_sha256="0" * 64,
                     conversion="lossless PCM16 sample slice; no padding, resampling or mixing")
        client = resident.NativeServiceProvider(config(server, url))
        req = request(audio)
        original = copy.deepcopy(req)
        assert client.generate(req)["stop_reason"] == "eos"
        assert req == original and server.provider.calls == [req]
        assert "no_parent_waveform" in server.identity["parent_metadata_verification"]


def test_identity_mismatch_rejects_health_and_every_inference(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        with pytest.raises(ValueError, match="explicit pin"):
            resident.NativeServiceProvider(config(server, url, expected_identity_sha256="0" * 64))
        body = envelope(server, request(audio))
        body["expected_identity_sha256"] = "0" * 64
        assert http_json(url + "/v1/local-caption", body)[0] == 409
        assert server.provider.calls == []


def test_failed_generation_is_recorded_without_retry(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        server.provider.fail = True
        client = resident.NativeServiceProvider(config(server, url))
        result = client.generate(request(audio))
        assert result["stop_reason"] == "http_error" and result["raw"]["status"] == 500
        assert result["raw"]["outcome"] == "unknown_no_automatic_retry"
        assert result["identity"] == client.identity
        assert len(server.provider.calls) == 1 and server.provider.loads == 1


def test_failed_loaded_verification_requires_restart(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        attempts = []
        def failed_initialization(req):
            attempts.append(req)
            server.provider.adapter = object()
            server.provider.verification = None
            raise ValueError("Loaded checkpoint does not match audited checkpoint")
        monkeypatch.setattr(server.provider, "generate", failed_initialization)
        client = resident.NativeServiceProvider(config(server, url))
        assert client.generate(request(audio))["raw"]["status"] == 500
        assert http_json(url + "/health")[1]["requires_restart"] is True
        assert client.generate(request(audio, call_id="new-call"))["raw"]["status"] == 503
        assert len(attempts) == 1


def test_audio_byte_limit_before_generation(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        with Path(audio["path"]).open("ab") as stream:
            stream.write(b"x" * resident.LIMITS["max_audio_bytes"])
        assert http_json(url + "/v1/local-caption", envelope(server, request(audio)))[0] == 400
        assert server.provider.calls == []


def test_duplicate_keys_nonfinite_and_payload_limit(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        for raw in (b'{"schema":"a","schema":"b"}', b'{"seed":NaN}', b'{"seed":1e999}',
                    b"x" * (resident.LIMITS["max_request_bytes"] + 1)):
            assert http_json(url + "/v1/local-caption", raw=raw)[0] == 400
        assert http_json(url + "/v1/train", envelope(server, request(audio)))[0] == 404
        assert http_json(url + "/health?token=bad")[0] == 404
        assert server.provider.calls == []


@pytest.mark.parametrize("change", [
    {"expected_identity_sha256": None}, {"api_key_env": "bad-name"}, {"extra": True},
    {"timeout_seconds": True}, {"timeout_seconds": float("inf")}, {"timeout_seconds": 0},
    {"base_url": "http://user:password@localhost"}, {"base_url": "http://localhost/v1"},
    {"base_url": "http://localhost?x=y"}, {"base_url": "file:///tmp/service"},
])
def test_client_configuration_failures(monkeypatch, tmp_path, change):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        with pytest.raises(ValueError):
            resident.NativeServiceProvider(config(server, url, **change))
        assert server.provider.calls == []


def test_roles_missing_secret_and_no_fallback(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        with pytest.raises(ValueError, match="only a local"):
            providers.create_provider(config(server, url), role="planner")
        monkeypatch.delenv("CHOREO_MISSING_KEY", raising=False)
        with pytest.raises(ValueError, match="empty"):
            resident.NativeServiceProvider(config(server, url, api_key_env="CHOREO_MISSING_KEY"))
        with pytest.raises(ValueError, match="explicit native"):
            resident.create_resident_server({"kind": "mock"}, tmp_path)


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.mark.parametrize("corruption", ["identity", "request", "native_identity", "nonfinite", "malformed"])
def test_client_rejects_unbound_or_malformed_response_no_retry(monkeypatch, tmp_path, corruption):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        req = request(audio)
        status, payload = http_json(url + "/v1/local-caption", envelope(server, req))
        if corruption == "identity":
            payload["identity_sha256"] = "0" * 64
        elif corruption == "request":
            payload["request_sha256"] = "0" * 64
        elif corruption == "native_identity":
            payload["result"]["identity"]["family"] = "changed"
        raw = json.dumps(payload).encode()
        if corruption == "nonfinite":
            raw = b'{"value":1e999}'
        if corruption == "malformed":
            raw = b"not JSON"
        calls = []
        monkeypatch.setattr(client, "_open", lambda http: (calls.append(http), Response(raw))[1])
        result = client.generate(req)
        assert result["stop_reason"] == "malformed_api_response"
        assert result["identity"] == client.identity and len(calls) == 1
        assert len(server.provider.calls) == 1


def test_unknown_connection_outcome_no_retry(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        calls = []
        def fail(http):
            calls.append(http)
            raise urllib.error.URLError("timeout after send")
        monkeypatch.setattr(client, "_open", fail)
        result = client.generate(request(audio))
        assert result["stop_reason"] == "infrastructure_error"
        assert result["raw"]["outcome"] == "unknown_no_automatic_retry"
        assert len(calls) == 1


def test_incomplete_http_read_is_infrastructure_failure(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        calls = []
        class BrokenResponse(Response):
            def read(self, *args):
                raise IncompleteRead(b"part", 10)
        monkeypatch.setattr(client, "_open", lambda http: (calls.append(http), BrokenResponse())[1])
        result = client.generate(request(audio))
        assert result["stop_reason"] == "infrastructure_error" and len(calls) == 1


def test_redirect_does_not_replay_request(monkeypatch, tmp_path):
    with running_server(monkeypatch, tmp_path) as (server, url, audio):
        client = resident.NativeServiceProvider(config(server, url))
        # A real redirecting endpoint must not turn a POST into a GET elsewhere.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        seen = []
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                seen.append(self.path)
                self.send_response(307)
                self.send_header("Location", url + "/v1/local-caption")
                self.send_header("Content-Length", "0")
                self.end_headers()
        redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=lambda: redirect.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        try:
            # Keep the already attested identity; intercept the transport only.
            client.url = "http://127.0.0.1:" + str(redirect.server_address[1])
            result = client.generate(request(audio))
            assert result["stop_reason"] == "http_error" and result["raw"]["status"] == 307
            assert len(seen) == 1 and server.provider.calls == []
        finally:
            redirect.shutdown()
            redirect.server_close()
            thread.join(timeout=3)
