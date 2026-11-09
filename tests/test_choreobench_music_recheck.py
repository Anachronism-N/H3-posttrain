"""CPU protocol fixtures only; no pretrained model or audio-quality acceptance."""
from copy import deepcopy
from contextlib import contextmanager
import struct
import threading
from types import SimpleNamespace
import wave

import pytest

from choreobench import music_recheck as recheck, providers, resident
from choreobench.audio import inspect_audio
from choreobench.io import object_sha, write_json


def native_identity():
    def ref(name):
        return dict(path="/CPU-fixture/" + name, sha256="a" * 64, size_bytes=1)
    files = [dict(ref(name), relative_path=name) for name in ("config.json", "model.safetensors")]
    return dict(kind="native", family="omni_captioner", synthetic=False,
                protocol="choreobench-prompted-omni-base-v1",
                task_prompt=recheck.MUSIC_OBSERVATION_PROMPT,
                task_prompt_sha256=object_sha(recheck.MUSIC_OBSERVATION_PROMPT),
                artifact_identity=dict(schema="music_planner_local_artifact_identity_v1",
                    family="omni_captioner", mode="native_base_only",
                    protocol="native-direct-action-fixed-two-windows-v1", checkpoint=None,
                    base_files=files, inference_adapter_source=ref("music_planner_adapters.py")))


def service_identity(native=None):
    remote = dict(kind="native_service_resident", version=1,
                  synthetic=False, transport="shared_filesystem_only_no_upload",
                  native_provider_identity=native or native_identity())
    pin = object_sha(remote)
    return dict(kind="native_service", synthetic=False, remote_identity=remote,
                expected_identity_sha256=pin,
                health_attestation=dict(schema="choreobench.native_service.health.v1", identity_sha256=pin))


@pytest.mark.parametrize("event,expected", [
    (dict(start=0, end=1), (0, 81)),
    (dict(start=99, end=101), (60, 141)),  # Exact half-sample centering tie rounds up.
    (dict(start=198, end=200), (119, 200)),
])
def test_midpoint_complete_window_and_edge_shift(event, expected):
    result = recheck.make_observation_window(dict(sample_rate=16, num_samples=200), event)
    assert (result["observation_start"], result["observation_end"]) == expected
    assert result["nominal_window_seconds"] == 5.0625 and result["sample_rounding_error"] == 0
    assert result["boundaries_are_action_boundaries"] is False


def test_nonintegral_length_has_exact_half_up_error_and_long_event_is_partial():
    result = recheck.make_observation_window(dict(sample_rate=8, num_samples=100), dict(start=0, end=100))
    assert result["observation_end"] - result["observation_start"] == 41
    assert result["sample_rounding_error"] == .5
    assert result["contains_entire_event"] is False
    full = recheck.make_observation_window(dict(sample_rate=16, num_samples=81), dict(start=0, end=81))
    assert full["observation_start"] == 0 and full["observation_end"] == 81
    assert full["contains_entire_event"] is True


@pytest.mark.parametrize("audio,event", [
    (dict(sample_rate=16, num_samples=80), dict(start=0, end=1)),
    (dict(sample_rate=True, num_samples=100), dict(start=0, end=1)),
    (dict(sample_rate=16., num_samples=100), dict(start=0, end=1)),
    (dict(sample_rate=16, num_samples=100.), dict(start=0, end=1)),
    (dict(sample_rate=0, num_samples=100), dict(start=0, end=1)),
    (dict(sample_rate=16, num_samples=100), dict(start=False, end=1)),
    (dict(sample_rate=16, num_samples=100), dict(start=0, end=1.)),
    (dict(sample_rate=16, num_samples=100), dict(start=-1, end=1)),
    (dict(sample_rate=16, num_samples=100), dict(start=5, end=5)),
    (dict(sample_rate=16, num_samples=100), dict(start=0, end=101)),
    (dict(sample_rate=16, num_samples=2 ** 63), dict(start=0, end=1)),
    ({}, dict(start=0, end=1)),
    (dict(sample_rate=16, num_samples=100), {}),
])
def test_invalid_clock_or_event_never_pads_or_guesses(audio, event):
    with pytest.raises(ValueError):
        recheck.make_observation_window(audio, event)


def audio_file(path, rate=16, count=200):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"".join(struct.pack("<h", i) for i in range(count)))
    return inspect_audio(path)


def test_real_pcm_slice_matches_parent_and_reuses_identical_resume_bytes(tmp_path):
    parent = audio_file(tmp_path / "parent.wav")
    event = dict(start=99, end=101, action="Private plan text must not become an observer prompt")
    destination = tmp_path / "slice.wav"
    result = recheck.prepare_observation_audio(parent["path"], event, destination)
    actual = inspect_audio(destination)
    assert actual["pcm"] == parent["pcm"][2 * 60:2 * 141]
    assert actual["num_samples"] == 81 and result["audio"]["parent_start"] == 60
    assert result["musical_facts_verified"] is False and result["movement_ground_truth"] is False
    assert recheck.prepare_observation_audio(parent["path"], event, destination) == result
    assert "action" not in result["audio"] and "action" not in result["window"]


def test_native_prompt_is_injected_into_copy_without_loading_or_mutating(monkeypatch, tmp_path):
    supplied = dict(kind="native", model="CPU-fixture", checkpoint=None)
    before, seen = deepcopy(supplied), []
    expected = SimpleNamespace(identity=native_identity())
    def create(spec, role, base_dir):
        seen.append((spec, role, base_dir))
        return expected
    monkeypatch.setattr(recheck, "create_provider", create)
    assert recheck.create_music_observer(supplied, tmp_path) is expected
    assert supplied == before and seen[0][1:] == ("local", tmp_path)
    assert seen[0][0]["task_prompt"] == recheck.MUSIC_OBSERVATION_PROMPT
    assert seen[0][0]["family"] == "omni_captioner"


@pytest.mark.parametrize("changes", [
    dict(kind="mock"), dict(kind="chat_completions"), dict(kind="unknown"),
    dict(family="moss"), dict(family="music_flamingo"),
    dict(checkpoint="/trained"), dict(checkpoint=""), dict(checkpoint=False),
    dict(training_source_dir="/training"), dict(task_prompt="Make dance steps"), dict(task_prompt=None),
])
def test_disallowed_spec_rejected_before_provider_creation(monkeypatch, changes):
    def unexpected(*args, **kwargs):
        raise AssertionError("Rejected spec must not create, contact or load a provider")
    monkeypatch.setattr(recheck, "create_provider", unexpected)
    spec = dict(kind="native", model="CPU-fixture")
    spec.update(changes)
    with pytest.raises(ValueError):
        recheck.create_music_observer(spec)


@pytest.mark.parametrize("change", ["prompt", "prompt_sha", "protocol", "mode", "checkpoint",
                                    "family", "artifact_family", "files", "adapter", "unknown"])
def test_unknown_or_incompatible_native_identity_rejected(monkeypatch, change):
    identity = native_identity()
    if change == "prompt": identity["task_prompt"] = "Describe movement"
    elif change == "prompt_sha": identity["task_prompt_sha256"] = "0" * 64
    elif change == "protocol": identity["protocol"] = "unknown"
    elif change == "mode": identity["artifact_identity"]["mode"] = "trained"
    elif change == "checkpoint": identity["artifact_identity"]["checkpoint"] = {}
    elif change == "family": identity["family"] = "moss"
    elif change == "artifact_family": identity["artifact_identity"]["family"] = "moss"
    elif change == "files": identity["artifact_identity"]["base_files"] = []
    elif change == "adapter": del identity["artifact_identity"]["inference_adapter_source"]
    else: identity = {"kind": "native", "synthetic": False}
    monkeypatch.setattr(recheck, "create_provider", lambda *args, **kwargs: SimpleNamespace(identity=identity))
    with pytest.raises(ValueError):
        recheck.create_music_observer(dict(kind="native", model="CPU-fixture"))


def request():
    return dict(call_id="music-observation:one", response_kind="local_caption", seed=42, max_output_tokens=128,
                audio=dict(path="/CPU-fixture/window.wav", sha256="0" * 64, sample_rate=16, num_samples=81))


def replay_file(path, *, synthetic, origin=None):
    source = {} if synthetic else dict(recording_id="CPU-only-fixture", recording_sha256="b" * 64,
                                        provider_identity=origin)
    item = dict(text="An even pulse grows stronger near the end; the exact tempo is uncertain.",
                raw={"fixture": True}, request_sha256=object_sha(request()), stop_reason="eos")
    write_json(path, dict(schema="choreobench.replay.v1", synthetic=synthetic, source=source,
                         responses={request()["call_id"]: item}))


@pytest.mark.parametrize("synthetic,origin", [(True, None), (False, "native"), (False, "service")])
def test_replays_keep_explicit_synthetic_and_require_matching_real_origin(tmp_path, synthetic, origin):
    identity = native_identity() if origin == "native" else service_identity() if origin == "service" else None
    replay_file(tmp_path / "replay.json", synthetic=synthetic, origin=identity)
    observer = recheck.create_music_observer(dict(kind="replay", path="replay.json"), tmp_path)
    result = observer.generate(request())
    assert result["synthetic"] is synthetic and result["costs"]["replayed"] is True
    assert "pulse" in result["text"]
    changed = request()
    changed["seed"] += 1
    with pytest.raises(ValueError, match="another request"):
        observer.generate(changed)


@pytest.mark.parametrize("origin", [
    dict(kind="chat_completions", synthetic=False),
    dict(kind="mock", synthetic=True), dict(kind="replay", synthetic=True),
    dict(kind="native", family="omni_captioner", synthetic=False),
])
def test_real_replay_cannot_launder_wrong_or_unknown_source_identity(tmp_path, origin):
    replay_file(tmp_path / "replay.json", synthetic=False, origin=origin)
    with pytest.raises(ValueError):
        recheck.create_music_observer(dict(kind="replay", path="replay.json"), tmp_path)


class FakeObservedNative:
    """Declared protocol double; never presented as a real model execution."""
    def __init__(self, spec, role, base_dir=None):
        self.identity = native_identity()
        self.identity["task_prompt"] = spec.get("task_prompt", "different task")
        self.identity["task_prompt_sha256"] = object_sha(self.identity["task_prompt"])
        self.adapter = self.verification = None
        self.calls = []

    def generate(self, request):
        self.calls.append(deepcopy(request))
        self.adapter, self.verification = object(), {"CPU-fixture": True}
        return dict(text="The pulse is steady; the excerpt becomes louder. Details remain uncertain.",
                    raw={"CPU-fixture": True}, identity=deepcopy(self.identity), synthetic=False,
                    stop_reason="eos", costs=dict(input_tokens=10, output_tokens=12))


@contextmanager
def resident_observer(monkeypatch, tmp_path, prompt):
    monkeypatch.setattr(providers, "NativeProvider", FakeObservedNative)
    parent = audio_file(tmp_path / "parent.wav")
    audio = recheck.prepare_observation_audio(parent["path"], dict(start=10, end=20), tmp_path / "window.wav")["audio"]
    server = resident.create_resident_server(dict(kind="native", model="CPU-fixture", task_prompt=prompt), tmp_path)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    spec = dict(kind="native_service", base_url="http://127.0.0.1:" + str(server.server_address[1]),
                expected_identity_sha256=server.identity_sha256, timeout_seconds=3.)
    try:
        yield server, spec, audio
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_actual_resident_health_and_audio_only_call_share_existing_protocol(monkeypatch, tmp_path):
    with resident_observer(monkeypatch, tmp_path, recheck.MUSIC_OBSERVATION_PROMPT) as (server, spec, audio):
        observer = recheck.create_music_observer(spec)
        assert server.provider.calls == [] and observer.health["loaded"] is False
        item = request()
        item["audio"] = audio
        result = observer.generate(item)
        assert result["identity"] == observer.identity and result["stop_reason"] == "eos"
        assert server.provider.calls == [item]
        bad = dict(item, plan="private plan must not go to the observer")
        with pytest.raises(ValueError):
            observer.generate(bad)
        assert len(server.provider.calls) == 1


def test_real_resident_wrong_prompt_rejected_before_any_inference(monkeypatch, tmp_path):
    with resident_observer(monkeypatch, tmp_path, "Generate choreography.") as (server, spec, audio):
        with pytest.raises(ValueError, match="exact prompted"):
            recheck.create_music_observer(spec)
        assert server.provider.calls == []


@pytest.mark.parametrize("change", ["pin", "attestation", "version", "unknown_native", "startup_health"])
def test_malformed_service_evidence_rejected(monkeypatch, change):
    identity = service_identity()
    health = dict(schema="choreobench.native_service.health.v1", identity=deepcopy(identity["remote_identity"]),
                  identity_sha256=identity["expected_identity_sha256"])
    if change == "pin": identity["expected_identity_sha256"] = "0" * 64
    elif change == "attestation": identity["health_attestation"] = {}
    elif change == "version":
        identity["remote_identity"]["version"] = True
        identity["expected_identity_sha256"] = object_sha(identity["remote_identity"])
        identity["health_attestation"]["identity_sha256"] = identity["expected_identity_sha256"]
    elif change == "unknown_native":
        identity["remote_identity"]["native_provider_identity"] = {"kind": "native"}
        identity["expected_identity_sha256"] = object_sha(identity["remote_identity"])
        identity["health_attestation"]["identity_sha256"] = identity["expected_identity_sha256"]
    else: health["identity"] = {}
    monkeypatch.setattr(recheck, "create_provider", lambda *args, **kwargs: SimpleNamespace(identity=identity, health=health))
    with pytest.raises(ValueError):
        recheck.create_music_observer(dict(kind="native_service"))
