"""Explicit synthetic, replay, native audio-only, and chat-completions providers.

No provider falls back to mock. Native weights and optional portable checkpoints
use the existing strict artifact audit. Chat calls send text only; their remote
weight identity cannot be independently verified by this client.
"""
import copy
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .io import file_sha, finite_json, object_sha, read_json, read_json_snapshot


def parse_json_response(text):
    """Reject fences, duplicate keys and nonfinite model JSON, without repair."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate model JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite model JSON: " + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    finite_json(value)
    return value


def _path(value, base_dir):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base_dir or ".") / path).resolve()


def _allowed(spec, names):
    if not isinstance(spec, dict) or set(spec) - set(names):
        raise ValueError("Unknown provider configuration fields")
    finite_json(spec)


def _budget(request):
    if (type(request.get("max_output_tokens")) is not int or request["max_output_tokens"] <= 0
            or type(request.get("seed")) is not int or not 0 <= request["seed"] < 2 ** 63):
        raise ValueError("Positive integer output budget and integer seed in [0,2**63) required")


class MockProvider:
    """Deterministic synthetic fixture, never an audio-understanding model."""
    def __init__(self, spec, role):
        _allowed(spec, ("kind",))
        self.role = role
        self.identity = dict(kind="mock", version="choreobench-mock-v1", role=role,
                             synthetic=True, understands_audio=False)

    def generate(self, request):
        _budget(request)
        kind = request["response_kind"]
        if kind == "local_caption":
            actions = ("Steps left while raising both arms.", "Turns right with a small step.",
                       "Bends the knees and opens both arms.", "Steps backward and lowers both arms.")
            index = request["seed"] % len(actions)
            text = "[0-2s] " + actions[index] + " [2-5.0625s] " + actions[(index + 1) % len(actions)]
        elif kind == "outline":
            text = json.dumps(dict(motifs=[], sections=[dict(section_id=s["section_id"],
                instruction="Synthetic fixture: retain the supplied local proposal schedule.")
                for s in request["song"]["music"]["sections"]]), ensure_ascii=False)
        elif kind in ("plan", "section_plan"):
            text = json.dumps(request["proposed_plan"], ensure_ascii=False)
        else:
            raise ValueError("Mock does not implement this response_kind")
        estimate = max(1, (len(text) + 3) // 4)
        return dict(text=text, raw=dict(synthetic=True, fixture="deterministic-copy-not-model", text=text),
                    identity=self.identity, synthetic=True,
                    stop_reason="stop" if estimate <= request["max_output_tokens"] else "budget_truncation",
                    costs=dict(output_tokens=None, estimated_output_tokens=estimate,
                               input_tokens=None, latency_seconds=0., token_counts_are_estimates=True))


class ReplayProvider:
    def __init__(self, spec, role, base_dir=None):
        _allowed(spec, ("kind", "path"))
        self.path = _path(spec["path"], base_dir)
        data, snapshot_sha = read_json_snapshot(self.path)
        if (not isinstance(data, dict) or data.get("schema") != "choreobench.replay.v1"
                or type(data.get("synthetic")) is not bool or not isinstance(data.get("responses"), dict)
                or not isinstance(data.get("source"), dict)):
            raise ValueError("Replay requires schema, explicit synthetic, source and responses-by-call_id")
        if not data["synthetic"]:
            source = data["source"]
            if (not isinstance(source.get("recording_id"), str) or not source["recording_id"].strip()
                    or not isinstance(source.get("recording_sha256"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", source["recording_sha256"])
                    or not isinstance(source.get("provider_identity"), dict) or not source["provider_identity"]):
                raise ValueError("Non-synthetic replay requires a traceable recording_id/SHA and provider_identity")
            for call_id, item in data["responses"].items():
                if (not isinstance(call_id, str) or not isinstance(item, dict)
                        or not isinstance(item.get("request_sha256"), str)
                        or not re.fullmatch(r"[a-f0-9]{64}", item["request_sha256"])
                        or "raw" not in item):
                    raise ValueError("Non-synthetic replay requires exact request_sha256 and original raw for every response")
        self.data = data
        self.identity = dict(kind="replay", role=role, path=str(self.path), sha256=snapshot_sha,
                             synthetic=data["synthetic"], source=data["source"],
                             source_model_identity_independently_verified=False)

    def generate(self, request):
        _budget(request)
        if file_sha(self.path) != self.identity["sha256"]:
            raise ValueError("Replay bytes changed during the run")
        item = copy.deepcopy(self.data["responses"].get(request["call_id"]))
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("No replay response for call_id " + request["call_id"])
        if ((not self.data["synthetic"] and item.get("request_sha256") != object_sha(request))
                or (self.data["synthetic"] and item.get("request_sha256") not in (None, object_sha(request)))):
            raise ValueError("Replay response belongs to another request")
        return dict(text=item["text"], raw=item, identity=self.identity,
                    synthetic=self.data["synthetic"], stop_reason=item.get("stop_reason", "stop"),
                    costs=dict(replayed=True, input_tokens=None, output_tokens=None, latency_seconds=0.))


class NativeProvider:
    def __init__(self, spec, role, base_dir=None):
        _allowed(spec, ("kind", "family", "model", "checkpoint", "training_source_dir", "device",
                        "max_length", "temperature", "top_p", "top_k", "task_prompt"))
        if role != "local":
            raise ValueError("Native audio-only model is a local provider, never a text editor/planner")
        self.spec = copy.deepcopy(spec)
        self.family = spec.get("family", "omni_captioner")
        if self.family not in ("omni_captioner", "moss", "music_flamingo"):
            raise ValueError("Unknown native model family")
        self.task_prompt = spec.get("task_prompt")
        if "task_prompt" in spec and (self.family != "omni_captioner" or spec.get("checkpoint")
                or not isinstance(self.task_prompt, str) or not self.task_prompt.strip()
                or len(self.task_prompt) > 16384 or "<|" in self.task_prompt or "<audio>" in self.task_prompt):
            raise ValueError("task_prompt requires a base-only Omni model and bounded plain task text")
        self.model = _path(spec["model"], base_dir)
        self.checkpoint = _path(spec["checkpoint"], base_dir) if spec.get("checkpoint") else None
        self.training_source = _path(spec["training_source_dir"], base_dir) if spec.get("training_source_dir") else None
        from scripts_omnidance.exp.ms.music_planner_search_identity import audit_model_artifacts
        artifact = audit_model_artifacts(self.family, self.model, self.checkpoint, self.training_source)
        self.max_length = spec.get("max_length", 8192)
        if type(self.max_length) is not int or self.max_length <= 0:
            raise ValueError("Positive native max_length required")
        if self.checkpoint and self.max_length > artifact["checkpoint"]["adapter_metadata"]["max_length"]:
            raise ValueError("Native context exceeds the checkpoint training contract")
        self.sampling = dict(temperature=spec.get("temperature", .9), top_p=spec.get("top_p", .95),
                             top_k=spec.get("top_k", 50))
        from scripts_omnidance.exp.ms.music_planner_search import validate_sampling
        validate_sampling(dict(self.sampling, max_new_tokens=1))
        self.identity = dict(kind="native", family=self.family, artifact_identity=artifact,
                             max_length=self.max_length, device=spec.get("device", "cpu"),
                             sampling=self.sampling, synthetic=False,
                             large_model_server_acceptance="not_verified_by_this_implementation")
        source_root = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
        source_names = ("music_planner_adapters.py", "music_planner_search.py", "music_planner_search_identity.py",
                        "music_planner_eval.py", "music_planner_moss_generation_patch.py",
                        "music_planner_moss_position_buffer_patch.py", "music_planner_generation_eos_patch.py")
        self.identity["native_source_sha256"] = {name: file_sha(source_root / name) for name in source_names}
        if self.task_prompt is not None:
            self.identity.update(task_prompt=self.task_prompt, task_prompt_sha256=object_sha(self.task_prompt),
                protocol="choreobench-prompted-omni-base-v1",
                benchmark_provider_sha256=file_sha(Path(__file__)))
        self.adapter = None
        self.verification = None

    def _task_template(self):
        return self.adapter.processor.apply_chat_template([dict(role="user", content=[
            dict(type="audio", audio="native-waveform"),
            dict(type="text", text=self.task_prompt)])], tokenize=False, add_generation_prompt=True)

    def load(self):
        from scripts_omnidance.exp.ms.music_planner_search_identity import verify_loaded_portable
        if self.adapter is None:
            from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
            self.adapter = load_planner_adapter(self.family, self.model, checkpoint=self.checkpoint,
                device=self.spec.get("device", "cpu"), max_length=self.max_length)
            self.adapter._sampling_official_model_path = self.model
            self.verification = verify_loaded_portable(self.adapter, self.identity["artifact_identity"])
            if self.task_prompt is not None:
                self.adapter._prompt = self._task_template

    def generate(self, request):
        _budget(request)
        from scripts_omnidance.exp.ms.music_planner_search import sample_native
        if request["response_kind"] != "local_caption":
            raise ValueError("Native provider cannot consume global plans or edit instructions")
        self.load()
        # Deliberately whitelist audio fields. No song brief, previous plan,
        # transcript, beat table, edit instruction or desired answer reaches it.
        audio = request["audio"]
        row = dict(id=request["call_id"], audio_path=audio["path"], audio_sha256=audio["sha256"],
                   audio_window_seconds=[0., 5.0625])
        started = time.monotonic()
        generated = sample_native(self.adapter, row, seed=request["seed"],
                                  max_new_tokens=request["max_output_tokens"], **self.sampling)
        raw = dict(generation=generated, loaded_artifact_verification=self.verification)
        if self.task_prompt is not None:
            raw["formatted_task_prompt"] = self._task_template()
        return dict(text=generated["raw_text"], raw=raw, identity=self.identity,
                    synthetic=False, stop_reason=generated["stop_reason"],
                    costs=dict(input_tokens=generated.get("input_tokens"),
                               output_tokens=len(generated["new_token_ids"]),
                               latency_seconds=time.monotonic() - started))


class ChatCompletionsProvider:
    def __init__(self, spec, role="planner"):
        _allowed(spec, ("kind", "base_url", "model", "api_key_env", "timeout_seconds", "temperature",
                        "json_mode", "token_parameter"))
        if role != "planner":
            raise ValueError("Chat-completions provider is text-only, not a local audio model")
        url = spec.get("base_url", "https://api.openai.com/v1").rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API base_url must be a credential-free http(s) endpoint")
        if not isinstance(spec.get("model"), str) or not spec["model"].strip():
            raise ValueError("Explicit API model required")
        self.spec = copy.deepcopy(spec)
        self.url = url + "/chat/completions"
        timeout = spec.get("timeout_seconds", 60.)
        temperature = spec.get("temperature", 0.)
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("Invalid API timeout/temperature")
        token_parameter = spec.get("token_parameter", "max_tokens")
        if token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Unknown API token budget parameter")
        if type(spec.get("json_mode", False)) is not bool:
            raise ValueError("json_mode must be boolean")
        env_name = spec.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(env_name, str) or (env_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name)):
            raise ValueError("api_key_env must name an environment variable (or be empty for no auth)")
        self.identity = dict(kind="chat_completions", base_url=url, model=spec["model"],
                             api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                             timeout_seconds=timeout, temperature=temperature,
                             json_mode=spec.get("json_mode", False), token_parameter=token_parameter,
                             remote_weight_identity_verified=False, synthetic=False)

    def generate(self, request):
        _budget(request)
        messages = request.get("messages")
        if (not isinstance(messages, list) or not messages or any(not isinstance(m, dict)
                or set(m) != {"role", "content"} or m["role"] not in ("system", "user", "assistant")
                or not isinstance(m["content"], str) for m in messages)):
            raise ValueError("Text-only role/content messages required")
        body = dict(model=self.identity["model"], messages=messages,
                    temperature=self.identity["temperature"], seed=request["seed"])
        body[self.identity["token_parameter"]] = request["max_output_tokens"]
        schema = request.get("response_schema")
        if schema is not None:
            if (not isinstance(schema, dict) or schema.get("type") != "object"
                    or schema.get("additionalProperties") is not False):
                raise ValueError("response_schema must be an explicit closed JSON object schema")
            finite_json(schema)
            body["response_format"] = dict(type="json_schema", json_schema=dict(
                name="choreobench_response", strict=True, schema=copy.deepcopy(schema)))
        elif self.identity["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        secret = os.environ.get(self.identity["api_key_env"], "") if self.identity["api_key_env"] else ""
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["Authorization"] = "Bearer " + secret
        http = urllib.request.Request(self.url, data=json.dumps(body, allow_nan=False).encode(),
                                      headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http, timeout=self.identity["timeout_seconds"]) as response:
                raw_text = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as error:
            raw_text = error.read().decode("utf-8", errors="replace")
            if secret:
                raw_text = raw_text.replace(secret, "[REDACTED]")
            # Returned as evidence, not retried. The caller records then fails.
            return dict(text="", raw=dict(status=error.code, body=raw_text, request=body),
                        identity=self.identity, synthetic=False, stop_reason="http_error",
                        costs=dict(latency_seconds=time.monotonic() - started, input_tokens=None, output_tokens=None))
        except (urllib.error.URLError, OSError) as error:
            message = str(error)
            if secret:
                message = message.replace(secret, "[REDACTED]")
            return dict(text="", raw=dict(error_type=type(error).__name__, error=message, request=body),
                        identity=self.identity, synthetic=False, stop_reason="infrastructure_error",
                        costs=dict(latency_seconds=time.monotonic() - started, input_tokens=None, output_tokens=None))
        if secret:
            raw_text = raw_text.replace(secret, "[REDACTED]")
        try:
            payload = parse_json_response(raw_text)
            choice = payload["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("No plain assistant text")
            finish = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError):
            payload, text, finish = None, "", "malformed_api_response"
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        usage_status = "reported_object" if isinstance(usage, dict) and usage else "unknown_or_invalid"
        if not isinstance(usage, dict):
            usage = {}  # Preserve the full raw envelope even with broken usage metadata.
        return dict(text=text, raw=dict(status=status, body=raw_text, request=body),
                    identity=self.identity, synthetic=False,
                    stop_reason="stop" if finish == "stop" else "budget_truncation" if finish == "length" else str(finish),
                    costs=dict(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                               reported_total_tokens=usage.get("total_tokens"),
                               usage_metadata_status=usage_status,
                               latency_seconds=time.monotonic() - started, price_usd=None))


def create_provider(spec, role="local", base_dir=None):
    if not isinstance(spec, dict):
        raise ValueError("Provider configuration must be an object")
    kind = spec.get("kind")
    if kind == "mock":
        return MockProvider(spec, role)
    if kind == "replay":
        return ReplayProvider(spec, role, base_dir)
    if kind == "native":
        return NativeProvider(spec, role, base_dir)
    if kind == "native_service":
        from .resident import NativeServiceProvider
        return NativeServiceProvider(spec, role)
    if kind == "chat_completions":
        return ChatCompletionsProvider(spec, role)
    raise ValueError("Unknown provider kind; no fallback exists")


def call_chat(spec, messages, *, max_output_tokens=2048, seed=42):
    """Single non-retrying text API call; root edit CLI journals this result."""
    return create_provider(spec, role="planner").generate(dict(messages=messages,
        max_output_tokens=max_output_tokens, seed=seed, response_kind="patch"))
