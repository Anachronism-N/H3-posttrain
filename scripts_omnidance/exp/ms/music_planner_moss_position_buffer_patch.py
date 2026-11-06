"""Eval-only restoration of the shipped MOSS non-persistent sinusoid buffer."""
import hashlib
import inspect
import json
from pathlib import Path


BUFFER_KEY = "audio_encoder.embed_positions.inv_timescales"


def _file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def position_buffer_patch_ref():
    return _file_ref(__file__)


def _tensor_ref(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())


def _buffer_stats(tensor):
    import torch
    ref = _tensor_ref(tensor)
    finite = torch.isfinite(tensor)
    values = tensor[finite].float()
    return dict(ref, device=str(tensor.device), finite_count=int(finite.sum().item()),
                nan_count=int(torch.isnan(tensor).sum().item()),
                inf_count=int(torch.isinf(tensor).sum().item()),
                positive_count=int((tensor > 0).sum().item()),
                finite_min=float(values.min().item()) if values.numel() else None,
                finite_max=float(values.max().item()) if values.numel() else None)


def _native_model(adapter):
    if getattr(adapter, "family", None) != "moss":
        raise ValueError("Position buffer restoration is restricted to MOSS")
    model = adapter.model
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if getattr(model.config, "model_type", None) != "moss_music":
        raise ValueError("A native MOSS model is required")
    return model


def restore_moss_position_buffer(adapter, official_model_path):
    """Restore only inv_timescales; retain parameters, hooks and training identity.

    The official shard stores BF16-rounded frequencies. The shipped native class
    constructs FP32 frequencies (including under HF's training ZeRO-3 context).
    Verify that rounding the native FP32 result reproduces every official value;
    install that exact native FP32 result, not a BF16-to-FP32 approximation.
    """
    import torch
    from safetensors import safe_open

    model = _native_model(adapter)
    owner = model.audio_encoder.embed_positions
    buffer = owner._buffers.get("inv_timescales")
    matches = [(key, value) for key, value in model.named_buffers()
               if key.rsplit(".", 1)[-1] == "inv_timescales"]
    if (len(matches) != 1 or matches[0][0] != BUFFER_KEY or matches[0][1] is not buffer or
            "inv_timescales" not in owner._non_persistent_buffers_set or
            BUFFER_KEY in model.state_dict()):
        raise ValueError("Expected the unique native non-persistent inv_timescales buffer")
    if (not isinstance(buffer, torch.Tensor) or buffer.shape != (640,) or
            buffer.dtype not in (torch.float32, torch.bfloat16, torch.float16) or
            buffer.is_meta or buffer.requires_grad):
        raise ValueError("Expected a materialized floating-point non-gradient buffer of shape [640]")

    root = Path(official_model_path).resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_name = index.get("weight_map", {}).get(BUFFER_KEY)
    if not isinstance(shard_name, str):
        raise ValueError("Official checkpoint index must identify inv_timescales")
    shard_path = (root / shard_name).resolve()
    if shard_path.parent != root:
        raise ValueError("Official tensor shard must stay inside the model directory")
    with safe_open(str(shard_path), framework="pt", device="cpu") as stream:
        if list(stream.get_slice(BUFFER_KEY).get_shape()) != [640]:
            raise ValueError("Official frequency tensor must have shape [640] before reading its bytes")
        source = stream.get_tensor(BUFFER_KEY)
    if (source.shape != (640,) or source.dtype not in (torch.float32, torch.bfloat16) or
            not torch.isfinite(source).all().item() or not (source > 0).all().item()):
        raise ValueError("Official frequency tensor must be finite, positive FP32/BF16 [640]")

    official_source_ref = _file_ref(root / "modeling_moss_music.py")
    native_source_ref = _file_ref(inspect.getfile(type(owner)))
    if (type(owner).__name__ != "SinusoidsPositionEmbedding" or
            native_source_ref["sha256"] != official_source_ref["sha256"]):
        raise ValueError("Position class must match the shipped official native source")
    official_config = json.loads((root / "config.json").read_text())
    audio_config = official_config["audio_config"]
    if audio_config["d_model"] != 1280 or model.config.audio_config.d_model != 1280:
        raise ValueError("Expected official/native MOSS d_model=1280")
    with torch.no_grad(), torch.device("cpu"):
        reference_owner = type(owner)(audio_config["max_source_positions"], audio_config["d_model"])
        reference = reference_owner.inv_timescales
    if (reference.shape != (640,) or reference.dtype != torch.float32 or
            "inv_timescales" not in reference_owner._non_persistent_buffers_set or
            not torch.isfinite(reference).all().item() or not (reference > 0).all().item() or
            not torch.equal(reference.to(source.dtype), source)):
        raise ValueError("Native FP32 frequencies must reproduce every official checkpoint value")

    before = _buffer_stats(buffer)
    restored = reference.to(device=buffer.device, dtype=torch.float32)
    if not torch.equal(restored.cpu(), reference):
        raise ValueError("Device transfer changed canonical FP32 frequencies")
    owner._buffers["inv_timescales"] = restored
    if ("inv_timescales" not in owner._non_persistent_buffers_set or
            BUFFER_KEY in model.state_dict()):
        raise RuntimeError("Restoration changed buffer persistence")
    return dict(schema="moss_position_buffer_receipt_v1", buffer_key=BUFFER_KEY,
                applied_position_buffer_patch_ref=position_buffer_patch_ref(),
                restoration_source="official_native_constructor_fp32",
                source_tensor_ref=dict(_tensor_ref(source), key=BUFFER_KEY,
                                       shard_path=str(shard_path),
                                       shard_size_bytes=shard_path.stat().st_size,
                                       index_ref=_file_ref(index_path)),
                official_model_source_ref=official_source_ref,
                native_model_source_ref=native_source_ref,
                official_config_ref=_file_ref(root / "config.json"),
                canonical_tensor_ref=_tensor_ref(reference), before_stats=before,
                after_stats=_buffer_stats(restored), persistent=False,
                parameters_changed=False, training_identity_preserved=True)


def validate_moss_generation_logits(logits):
    """Reject non-finite true last-token logits before generation can argmax them."""
    import torch
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or
            not logits.shape[0] or not logits.shape[1] or not logits.shape[2]):
        raise ValueError("Generation requires real [batch, sequence, vocabulary] logits")
    if not torch.isfinite(logits[:, -1, :]).all().item():
        raise FloatingPointError("MOSS generation rejected non-finite last-token logits before sampling")
    return logits


def install_moss_generation_logits_validator(adapter):
    """Observe native forward outputs; return a removable generation-only hook."""
    model = _native_model(adapter)

    def validate_output(module, inputs, output):
        logits = output.get("logits") if isinstance(output, dict) else getattr(output, "logits", None)
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
        validate_moss_generation_logits(logits)
        # Returning None keeps the original native output object and values.

    return model.register_forward_hook(validate_output)
