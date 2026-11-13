"""Fixed-window music observations, never verified facts or movement targets.

The observer receives the canonical audio slice alone. Event/plan text, repair
instructions and private goals are not part of this audio-provider protocol.
Resident health and replay source identities are declarations, not independent
authentication of remote weights or of the truth of an observation.
"""
from copy import deepcopy
from fractions import Fraction

from .audio import inspect_audio, write_window
from .io import finite_json, object_sha
from .providers import create_provider


MUSIC_OBSERVATION_PROMPT = (
    "Describe only the audible music in this audio excerpt. Summarize its rhythm "
    "and pulse, relative energy, and any changes within the excerpt. State "
    "uncertainty or unavailable detail explicitly. Do not propose choreography, "
    "describe a dancer or body movements, name a dance style, or invent precise "
    "beat times. Return concise plain English music observations."
)
_NATIVE_PROTOCOL = "choreobench-prompted-omni-base-v1"
_ARTIFACT_SCHEMA = "music_planner_local_artifact_identity_v1"
_BASE_PROTOCOL = "native-direct-action-fixed-two-windows-v1"
_HEALTH_SCHEMA = "choreobench.native_service.health.v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value < 2 ** 63:
        raise ValueError(name + " must be an integer in [" + str(minimum) + ",2**63)")


def make_observation_window(audio, event):
    """Center a complete 81/16-second window on the declared event midpoint.

    Length and half-sample center ties round upward using exact integer
    arithmetic. At song edges the whole interval shifts; it never pads or clips
    to a shorter duration. A long event need not be entirely observed.
    """
    if not isinstance(audio, dict) or not isinstance(event, dict):
        raise ValueError("Audio clock and event must be objects")
    rate, count = audio.get("sample_rate"), audio.get("num_samples")
    start, end = event.get("start"), event.get("end")
    _integer(rate, "audio.sample_rate", 1)
    _integer(count, "audio.num_samples", 1)
    _integer(start, "event.start")
    _integer(end, "event.end", 1)
    if not 0 <= start < end <= count:
        raise ValueError("Event interval must be positive and within the original song")
    length = (81 * rate + 8) // 16
    if length < 2 or count < length:
        raise ValueError("Song cannot supply a complete 5.0625s observation; no padding fallback")
    centered = (start + end - length + 1) // 2
    left = max(0, min(centered, count - length))
    return dict(observation_start=left, observation_end=left + length,
                nominal_window_seconds=5.0625,
                sample_rounding_error=float(Fraction(length) - Fraction(81 * rate, 16)),
                contains_entire_event=left <= start and end <= left + length,
                boundaries_are_action_boundaries=False)


def prepare_observation_audio(path, event, destination):
    """Inspect the parent WAV and publish its unchanged PCM slice for an observer."""
    audio = inspect_audio(path)
    window = make_observation_window(audio, event)
    return dict(window=window, audio=write_window(audio, window, destination),
                observation_kind="unverified_music_observation", musical_facts_verified=False,
                movement_ground_truth=False)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _file_reference(value, *, relative=False):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"].strip()
            or not _sha(value.get("sha256")) or type(value.get("size_bytes")) is not int
            or value["size_bytes"] < 0):
        raise ValueError("Unknown or incomplete native artifact file identity")
    if relative and (not isinstance(value.get("relative_path"), str) or not value["relative_path"].strip()):
        raise ValueError("Base file identity needs its explicit relative path")


def _validate_native(identity):
    if (not isinstance(identity, dict) or identity.get("kind") != "native"
            or identity.get("synthetic") is not False or identity.get("family") != "omni_captioner"
            or identity.get("protocol") != _NATIVE_PROTOCOL
            or identity.get("task_prompt") != MUSIC_OBSERVATION_PROMPT
            or identity.get("task_prompt_sha256") != object_sha(MUSIC_OBSERVATION_PROMPT)):
        raise ValueError("Music observer requires the exact prompted base-only native Omni identity")
    artifact = identity.get("artifact_identity")
    if (not isinstance(artifact, dict) or artifact.get("schema") != _ARTIFACT_SCHEMA
            or artifact.get("family") != "omni_captioner" or artifact.get("mode") != "native_base_only"
            or artifact.get("protocol") != _BASE_PROTOCOL
            or "checkpoint" not in artifact or artifact["checkpoint"] is not None):
        raise ValueError("Music observer rejects unknown or trained-checkpoint artifact identities")
    files = artifact.get("base_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Music observer requires the audited base file inventory")
    for item in files:
        _file_reference(item, relative=True)
    names = [item["relative_path"] for item in files]
    if len(set(names)) != len(names) or "config.json" not in names or not any(
            name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError("Base inventory needs distinct config and actual weight members")
    _file_reference(artifact.get("inference_adapter_source"))


def _validate_service(identity):
    if not isinstance(identity, dict) or identity.get("kind") != "native_service" or identity.get("synthetic") is not False:
        raise ValueError("Unknown resident observer identity")
    remote, pin = identity.get("remote_identity"), identity.get("expected_identity_sha256")
    if (not isinstance(remote, dict) or remote.get("kind") != "native_service_resident"
            or type(remote.get("version")) is not int or remote["version"] != 1
            or remote.get("synthetic") is not False
            or remote.get("transport") != "shared_filesystem_only_no_upload"
            or not _sha(pin) or object_sha(remote) != pin):
        raise ValueError("Resident music observer requires its pinned original resident protocol")
    health = identity.get("health_attestation")
    if (not isinstance(health, dict) or health.get("schema") != _HEALTH_SCHEMA
            or health.get("identity_sha256") != pin):
        raise ValueError("Resident music observer is missing the matching health attestation")
    _validate_native(remote.get("native_provider_identity"))


def _validate_origin(identity):
    if not isinstance(identity, dict):
        raise ValueError("Unknown music-observer provider identity")
    if identity.get("kind") == "native":
        _validate_native(identity)
    elif identity.get("kind") == "native_service":
        _validate_service(identity)
    else:
        raise ValueError("Replay origin must be native Omni or its original resident service, not another replay")


def create_music_observer(spec, base_dir=None):
    """Create a music-only local audio provider without loading GPU weights.

    The native spec is copied before injecting the fixed prompt. Real replays
    retain their existing request/content bindings and must declare a compatible
    origin. Explicit synthetic replay fixtures are allowed; mock/text-only
    providers and unknown identities never receive a fallback.
    """
    if not isinstance(spec, dict):
        raise ValueError("Music observer spec must be an object")
    finite_json(spec)
    config = deepcopy(spec)
    kind = config.get("kind")
    if kind == "native":
        if config.get("family", "omni_captioner") != "omni_captioner":
            raise ValueError("Music observation requires native omni_captioner")
        if config.get("checkpoint") is not None or config.get("training_source_dir") is not None:
            raise ValueError("Music observation requires a base-only model without a training checkpoint")
        if "task_prompt" in config and config["task_prompt"] != MUSIC_OBSERVATION_PROMPT:
            raise ValueError("Explicit music observation prompt differs from the frozen protocol")
        config["family"], config["task_prompt"] = "omni_captioner", MUSIC_OBSERVATION_PROMPT
    elif kind not in ("native_service", "replay"):
        raise ValueError("Music observer supports only native Omni, native_service or explicit replay")
    provider = create_provider(config, role="local", base_dir=base_dir)
    identity = getattr(provider, "identity", None)
    if not isinstance(identity, dict) or identity.get("kind") != kind:
        raise ValueError("Music observer factory returned an unknown or different provider identity")
    finite_json(identity)
    if kind == "replay":
        if type(identity.get("synthetic")) is not bool:
            raise ValueError("Replay must explicitly declare its synthetic status")
        if not identity["synthetic"]:
            source = identity.get("source")
            if not isinstance(source, dict):
                raise ValueError("Real replay requires its original source provider identity")
            _validate_origin(source.get("provider_identity"))
    else:
        _validate_origin(identity)
        if kind == "native_service":
            health = getattr(provider, "health", None)
            if (not isinstance(health, dict) or health.get("schema") != _HEALTH_SCHEMA
                    or health.get("identity_sha256") != identity["expected_identity_sha256"]
                    or health.get("identity") != identity["remote_identity"]):
                raise ValueError("Resident startup health differs from its validated observer identity")
    return provider
