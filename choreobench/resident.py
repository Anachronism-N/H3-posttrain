"""Resident, serialized native audio inference over an explicit shared filesystem.

Health identities are remote attestations, not independent verification of a
server's weights or behavior. There are no uploads, training routes or retries.
"""
import copy
import hashlib
import hmac
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

from .audio import WINDOW_SECONDS
from .io import file_sha, finite_json, object_sha


HEALTH_SCHEMA = "choreobench.native_service.health.v1"
REQUEST_SCHEMA = "choreobench.native_service.request.v1"
RESPONSE_SCHEMA = "choreobench.native_service.response.v1"
LIMITS = dict(max_request_bytes=65536, max_response_bytes=16 * 1024 * 1024,
              max_audio_bytes=4 * 1024 * 1024, max_output_tokens=65536,
              max_sample_rate=192000, window_seconds=WINDOW_SECONDS)
_SHA = re.compile(r"[a-f0-9]{64}")


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _credential(env_name):
    if not isinstance(env_name, str) or (env_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name)):
        raise ValueError("api_key_env must name an environment variable or be empty")
    secret = os.environ.get(env_name, "") if env_name else ""
    if env_name and not secret:
        raise ValueError("Configured API key environment variable is empty")
    if len(secret) > 4096 or any(ord(c) < 32 or ord(c) > 126 for c in secret):
        raise ValueError("Invalid API key")
    return secret


def _implementation():
    root = Path(__file__).resolve().parent
    return {name: file_sha(root / name) for name in ("resident.py", "providers.py")}


def _request(request):
    """Validate and copy an audio-only request; never forward unknown context."""
    from .providers import _budget
    if not isinstance(request, dict) or set(request) != {
            "call_id", "response_kind", "audio", "seed", "max_output_tokens"}:
        raise ValueError("Only call_id, response_kind, audio, seed and max_output_tokens are accepted")
    finite_json(request)
    _budget(request)
    if request["max_output_tokens"] > LIMITS["max_output_tokens"]:
        raise ValueError("Output token budget exceeds service limit")
    if (request["response_kind"] != "local_caption" or not isinstance(request["call_id"], str)
            or not request["call_id"].strip() or len(request["call_id"]) > 512):
        raise ValueError("A bounded call_id and local_caption response_kind are required")
    audio = request["audio"]
    required = {"path", "sha256", "sample_rate", "num_samples"}
    parent = {"parent_audio_sha256", "parent_start", "parent_end"}
    if (not isinstance(audio, dict) or not required <= set(audio)
            or set(audio) - required - parent - {"conversion"}):
        raise ValueError("Unknown or missing audio fields")
    if (not isinstance(audio["path"], str) or not audio["path"] or len(audio["path"]) > 8192
            or not _sha(audio["sha256"])):
        raise ValueError("Explicit audio path and SHA256 required")
    rate, count = audio["sample_rate"], audio["num_samples"]
    if (type(rate) is not int or not 1 <= rate <= LIMITS["max_sample_rate"]
            or type(count) is not int or count < 2
            or count != int(math.floor(WINDOW_SECONDS * rate + .5))):
        raise ValueError("Audio clock must represent one complete 5.0625s window")
    if parent & set(audio):
        if (not parent <= set(audio) or not _sha(audio["parent_audio_sha256"])
                or type(audio["parent_start"]) is not int or type(audio["parent_end"]) is not int
                or not 0 <= audio["parent_start"] < audio["parent_end"]
                or audio["parent_end"] - audio["parent_start"] != count
                or audio["parent_end"] >= 2 ** 63):
            raise ValueError("Invalid parent audio binding or sample interval")
    if "conversion" in audio and (not isinstance(audio["conversion"], str) or len(audio["conversion"]) > 2048):
        raise ValueError("Invalid audio conversion record")
    return copy.deepcopy(request)


def _audio_under_root(request, root):
    audio = request["audio"]
    path = Path(audio["path"])
    if not path.is_absolute():
        raise ValueError("Shared audio path must be absolute")
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Audio path escapes the shared audio root") from error
    if not path.is_file() or path.stat().st_size > LIMITS["max_audio_bytes"]:
        raise ValueError("Audio must be a bounded regular WAV file")
    # Limit the read itself, not only an earlier file-size observation.
    with path.open("rb") as stream:
        raw = stream.read(LIMITS["max_audio_bytes"] + 1)
    if len(raw) > LIMITS["max_audio_bytes"]:
        raise ValueError("Audio exceeds service size limit")
    try:
        with wave.open(io.BytesIO(raw), "rb") as stream:
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
                raise ValueError("Audio must be canonical uncompressed mono PCM16")
            rate, count = stream.getframerate(), stream.getnframes()
            pcm = stream.readframes(count)
    except (wave.Error, EOFError) as error:
        raise ValueError("Invalid canonical WAV") from error
    if len(pcm) != 2 * count:
        raise ValueError("Truncated WAV")
    inspected = dict(sha256=hashlib.sha256(raw).hexdigest(), sample_rate=rate, num_samples=count)
    if any(inspected[key] != audio[key] for key in ("sha256", "sample_rate", "num_samples")):
        raise ValueError("Actual audio SHA256 or clock differs from request")
    # Native prefix performs its existing waveform SHA/length checks as well.
    audio["path"] = str(path)
    return request


class _ResidentServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = True


def create_resident_server(provider_spec, audio_root, host="127.0.0.1", port=0, api_key_env=""):
    """Construct one lazy NativeProvider; caller owns serve_forever/server_close.

    Relative model paths resolve against the caller's current working directory,
    exactly as for NativeProvider. The root must already exist. No model weights
    are loaded until the first valid serialized inference request.
    """
    from .providers import NativeProvider
    if not isinstance(provider_spec, dict) or provider_spec.get("kind") != "native":
        raise ValueError("Resident service requires an explicit native provider configuration")
    if (not isinstance(host, str) or not host.strip() or type(port) is not int
            or not 0 <= port <= 65535):
        raise ValueError("Invalid service host/port")
    root = Path(audio_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Shared audio root must be a directory")
    secret = _credential(api_key_env)
    provider = NativeProvider(provider_spec, "local")
    identity = dict(kind="native_service_resident", version=1,
        native_provider_identity=copy.deepcopy(provider.identity), audio_root=str(root),
        transport="shared_filesystem_only_no_upload", api_key_env=api_key_env,
        parent_metadata_verification="format_and_clock_consistency_only_no_parent_waveform",
        implementation_sha256=_implementation(), limits=copy.deepcopy(LIMITS),
        synthetic=provider.identity["synthetic"])
    identity_sha = object_sha(identity)
    generation_lock = threading.Lock()
    state = dict(requires_restart=False)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # Avoid logging paths or credential-bearing headers.

        def _send(self, status, value):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(raw) > LIMITS["max_response_bytes"]:
                status = 500
                raw = b'{"error":"Service response exceeds size limit"}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)

        def _authorized(self):
            if secret and not hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"),
                                                  ("Bearer " + secret).encode("ascii")):
                self._send(401, dict(error="Authentication required"))
                return False
            return True

        def do_GET(self):
            if not self._authorized():
                return
            if self.path != "/health":
                self._send(404, dict(error="Unknown endpoint"))
                return
            self._send(200, dict(schema=HEALTH_SCHEMA, identity=identity,
                identity_sha256=identity_sha, loaded=provider.adapter is not None,
                requires_restart=state["requires_restart"],
                limits=LIMITS, identity_verification="server_attestation_not_independent_verification"))

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/v1/local-caption":
                self._send(404, dict(error="Unknown endpoint"))
                return
            try:
                from .providers import parse_json_response
                lengths = self.headers.get_all("Content-Length", [])
                if (self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1
                        or not re.fullmatch(r"[0-9]+", lengths[0])):
                    raise ValueError("One Content-Length and no Transfer-Encoding required")
                length = int(lengths[0])
                if not 0 < length <= LIMITS["max_request_bytes"]:
                    raise ValueError("Request body exceeds size limit or is empty")
                self.connection.settimeout(15.)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated request body")
                body = parse_json_response(raw.decode("utf-8"))
                if (not isinstance(body, dict) or set(body) != {"schema", "expected_identity_sha256", "request"}
                        or body["schema"] != REQUEST_SCHEMA):
                    raise ValueError("Invalid resident request envelope")
                if body["expected_identity_sha256"] != identity_sha:
                    self._send(409, dict(error="Resident identity differs from pinned identity"))
                    return
                request = _request(body["request"])
                # Validation and model execution share one lock. No parallel RNG,
                # lazy adapter initialization or mutation of a resident adapter.
                with generation_lock:
                    if state["requires_restart"]:
                        self._send(503, dict(error="Failed native initialization requires service restart",
                                             outcome="no_generation_attempted"))
                        return
                    request = _audio_under_root(request, root)
                    try:
                        result = provider.generate(request)
                    except Exception as error:
                        # NativeProvider sets adapter before verifying a loaded
                        # checkpoint. A failed verification must never be skipped
                        # on the next call merely because adapter is non-None.
                        if provider.adapter is not None and getattr(provider, "verification", None) is None:
                            state["requires_restart"] = True
                        # An inference may already have occurred. Never retry it.
                        self._send(500, dict(error="Native generation failed", error_type=type(error).__name__,
                            outcome="unknown_no_automatic_retry", identity_sha256=identity_sha))
                        return
                    finite_json(result)
                    if (not isinstance(result, dict) or result.get("identity") != identity["native_provider_identity"]
                            or type(result.get("synthetic")) is not bool
                            or result["synthetic"] != identity["synthetic"]):
                        raise ValueError("Native response identity changed")
                    _audio_under_root(request, root)
                self._send(200, dict(schema=RESPONSE_SCHEMA, identity_sha256=identity_sha,
                                     request_sha256=object_sha(body["request"]), result=result))
            except (ValueError, TypeError, KeyError, UnicodeError, OSError) as error:
                self._send(400, dict(error="Invalid audio-only request", error_type=type(error).__name__))
            except Exception as error:
                self._send(500, dict(error="Resident service failure", error_type=type(error).__name__,
                                    outcome="unknown_no_automatic_retry"))

    server = _ResidentServer((host, port), Handler)
    server.provider = provider
    server.identity = identity
    server.identity_sha256 = identity_sha
    server.generation_lock = generation_lock
    return server


def serve_resident(provider_spec, audio_root, host="127.0.0.1", port=0, api_key_env=""):
    """Run in the foreground until interrupted; never spawn a helper process."""
    server = create_resident_server(provider_spec, audio_root, host, port, api_key_env)
    try:
        print(json.dumps(dict(listening=list(server.server_address), health=HEALTH_SCHEMA,
                              identity_sha256=server.identity_sha256), allow_nan=False), flush=True)
        server.serve_forever()
    finally:
        server.server_close()


class NativeServiceProvider:
    """Pinned resident client. Its health evidence is a server assertion only."""
    def __init__(self, spec, role="local"):
        from .providers import _allowed, parse_json_response
        _allowed(spec, ("kind", "base_url", "timeout_seconds", "api_key_env", "expected_identity_sha256"))
        if role != "local":
            raise ValueError("Resident native service is only a local audio provider")
        url = spec.get("base_url")
        if not isinstance(url, str):
            raise ValueError("Explicit resident base_url required")
        url = url.rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path):
            raise ValueError("Resident base_url must be a credential-free http(s) origin without path")
        timeout = spec.get("timeout_seconds", 600.)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError("Resident timeout must be finite and in (0,86400]")
        expected = spec.get("expected_identity_sha256")
        if not _sha(expected):
            raise ValueError("Explicit expected_identity_sha256 pin required")
        env_name = spec.get("api_key_env", "")
        self._secret = _credential(env_name)
        self.url, self.timeout = url, timeout
        headers = self._headers()
        try:
            with self._open(urllib.request.Request(url + "/health", headers=headers, method="GET")) as response:
                if response.status != 200:
                    raise ValueError("Resident health did not return HTTP 200")
                raw = response.read(LIMITS["max_response_bytes"] + 1)
            if len(raw) > LIMITS["max_response_bytes"]:
                raise ValueError("Resident health response exceeds size limit")
            health = parse_json_response(raw.decode("utf-8"))
        except (urllib.error.URLError, OSError, UnicodeError, HTTPException) as error:
            # Do not include potentially credential-bearing remote error bodies.
            raise ValueError("Resident health unavailable: " + type(error).__name__) from None
        if (not isinstance(health, dict) or health.get("schema") != HEALTH_SCHEMA
                or not isinstance(health.get("identity"), dict)
                or health.get("identity_sha256") != expected
                or object_sha(health["identity"]) != expected):
            raise ValueError("Resident health identity does not match the explicit pin")
        remote = health["identity"]
        if (remote.get("kind") != "native_service_resident" or remote.get("synthetic") is not False
                or not isinstance(remote.get("native_provider_identity"), dict)
                or remote["native_provider_identity"].get("kind") != "native"
                or remote["native_provider_identity"].get("synthetic") is not False):
            raise ValueError("Health must attest an actual native provider identity")
        self.remote_identity = copy.deepcopy(remote)
        self.health = copy.deepcopy(health)
        self.identity = dict(kind="native_service", base_url=url, timeout_seconds=timeout,
            api_key_env=env_name, expected_identity_sha256=expected, synthetic=False,
            remote_identity=copy.deepcopy(remote),
            health_attestation=dict(schema=health["schema"], identity_sha256=expected,
                identity_verification="server_attestation_not_independent_verification"),
            remote_identity_independently_verified=False,
            identity_evidence="pinned_server_attestation_not_independent_verification",
            implementation_sha256=_implementation())

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["Authorization"] = "Bearer " + self._secret
        return headers

    def _open(self, http):
        # Disable redirects: neither credentials nor audio paths go to a second
        # endpoint, and a redirected POST is never replayed as another call.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(NoRedirect())
        return opener.open(http, timeout=self.timeout)

    def generate(self, request):
        from .providers import parse_json_response
        request = _request(request)
        body = dict(schema=REQUEST_SCHEMA, expected_identity_sha256=self.identity["expected_identity_sha256"],
                    request=request)
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > LIMITS["max_request_bytes"]:
            raise ValueError("Resident request exceeds size limit")
        started = time.monotonic()
        http = urllib.request.Request(self.url + "/v1/local-caption", data=encoded,
                                      headers=self._headers(), method="POST")
        raw_text, status = None, None
        try:
            with self._open(http) as response:
                status = response.status
                raw = response.read(LIMITS["max_response_bytes"] + 1)
            if len(raw) > LIMITS["max_response_bytes"]:
                raise ValueError("Resident response exceeds size limit")
            raw_text = self._redact(raw.decode("utf-8"))
            payload = parse_json_response(raw_text)
            if (status != 200 or not isinstance(payload, dict) or payload.get("schema") != RESPONSE_SCHEMA
                    or payload.get("identity_sha256") != self.identity["expected_identity_sha256"]
                    or payload.get("request_sha256") != object_sha(request)):
                raise ValueError("Resident response binding differs")
            result = payload.get("result")
            if (not isinstance(result, dict) or result.get("identity") != self.remote_identity["native_provider_identity"]
                    or result.get("synthetic") is not False or not isinstance(result.get("text"), str)
                    or not isinstance(result.get("costs"), dict) or not isinstance(result.get("stop_reason"), str)):
                raise ValueError("Malformed native response")
            result = copy.deepcopy(result)
            result["identity"] = copy.deepcopy(self.identity)
            result["raw"] = dict(remote_response=payload, response_body=self._redact(raw_text),
                                 status=status, request=body,
                                 startup_health_attestation=copy.deepcopy(self.health),
                                 remote_identity_independently_verified=False)
            result["costs"]["transport_latency_seconds"] = time.monotonic() - started
            return result
        except urllib.error.HTTPError as error:
            status = error.code
            try:
                raw_text = error.read(LIMITS["max_response_bytes"] + 1).decode("utf-8", errors="replace")
                stop, detail = "http_error", dict(error_type=type(error).__name__)
            except (OSError, HTTPException) as read_error:
                stop, detail = "infrastructure_error", dict(error_type=type(read_error).__name__)
        except (urllib.error.URLError, OSError, HTTPException) as error:
            stop, detail = "infrastructure_error", dict(error_type=type(error).__name__, error=self._redact(str(error)))
        except (ValueError, TypeError, KeyError, UnicodeError) as error:
            stop, detail = "malformed_api_response", dict(error_type=type(error).__name__)
        return dict(text="", identity=copy.deepcopy(self.identity), synthetic=False, stop_reason=stop,
            raw=dict(status=status, body=self._redact(raw_text), request=body,
                     outcome="unknown_no_automatic_retry", **detail),
            costs=dict(input_tokens=None, output_tokens=None, latency_seconds=time.monotonic() - started))

    def _redact(self, text):
        return text.replace(self._secret, "[REDACTED]") if self._secret and isinstance(text, str) else text
