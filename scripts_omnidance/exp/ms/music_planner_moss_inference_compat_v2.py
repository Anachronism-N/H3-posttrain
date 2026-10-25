"""Eval-only MOSS native-cache compatibility; frozen training identity is unchanged."""
import argparse
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def _file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def generation_patch_ref():
    return _file_ref(__file__)


POSITION_PATCH_PATH = Path("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1/runtime/eval_generation_patches_v2/music_planner_moss_position_buffer_patch.py")
POSITION_PATCH_SHA = "0077f742bb98f64763a38eb082ec228e99b1168ad8090813abb45af90dfcba33"
_position_helper = None


def _position_patch():
    global _position_helper
    if _file_ref(POSITION_PATCH_PATH)["sha256"] != POSITION_PATCH_SHA:
        raise ValueError("Frozen native position-buffer source changed")
    if _position_helper is None:
        _position_helper = _load_module("_moss_position_buffer_frozen", POSITION_PATCH_PATH)
    return _position_helper


def _prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                   attention_mask=None, inputs_embeds=None,
                                   cache_position=None, **kwargs):
    import torch
    cached = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
    if cached:
        # HF5 no longer supplies cache_position. Restore the shipped MOSS decode
        # branch instead of re-encoding the audio against an ever-growing prefix.
        if cache_position is None:
            cache_position = input_ids.new_tensor([cached])
        elif int(cache_position[0]) != cached:
            raise ValueError("MOSS cache_position differs from the native KV cache length")
        next_length = kwargs.get("next_sequence_length")
        if next_length not in (None, 1) or input_ids.shape[1] not in (1, cached + 1):
            raise ValueError("MOSS generation patch supports native one-token cached decode only")
    elif kwargs.get("audio_data") is not None:
        mask = kwargs.get("audio_input_mask")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
                mask.shape != input_ids.shape or
                not torch.equal(mask, input_ids.eq(self._planner_generation_audio_token_id))):
            raise ValueError("MOSS original prefix audio_input_mask must exactly match audio token positions")
    return self._planner_generation_native_prepare(
        input_ids, past_key_values=past_key_values, attention_mask=attention_mask,
        inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)


def _generate_with_patch_ref(self, *args, **kwargs):
    position = _position_patch()
    guard = position.install_moss_generation_logits_validator(self)
    native = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
    count = {"forward": 0}
    def observed(module, inputs, output):
        count["forward"] += 1
    observed_handle = native.register_forward_hook(observed)
    try:
        result = self._planner_generation_native_generate(*args, **kwargs)
    finally:
        observed_handle.remove()
        guard.remove()
    if not count["forward"]:
        raise ValueError("No real MOSS generation forward observed")
    return dict(result, applied_generation_patch_ref=self.applied_generation_patch_ref,
                applied_position_buffer_patch_ref=self.position_buffer_patch_receipt["applied_position_buffer_patch_ref"],
                native_logits_all_finite=True, actual_forward_count=count["forward"],
                finite_native_logits_forward_count=count["forward"])


def apply_moss_generation_patch(adapter):
    """Patch an already-loaded adapter, including PEFT, without touching its state."""
    if getattr(adapter, "family", None) != "moss":
        raise ValueError("This eval-only generation patch is restricted to MOSS")
    model = adapter.model
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if (getattr(model.config, "model_type", None) != "moss_music" or
            not hasattr(model, "_planner_native_forward")):
        raise ValueError("A native MOSS instance prepared by the trained planner adapter is required")
    position = _position_patch()
    adapter.position_buffer_patch_receipt = position.restore_moss_position_buffer(
        adapter, Path("/opt/music_planner_models/moss"))
    ref = generation_patch_ref()
    if "_planner_generation_native_prepare" in vars(model):
        if getattr(adapter, "applied_generation_patch_ref", None) != ref:
            raise ValueError("MOSS already has a different generation patch identity")
        return ref
    model._planner_generation_native_prepare = model.prepare_inputs_for_generation
    model._planner_generation_audio_token_id = adapter.processor.audio_token_id
    model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)
    adapter.applied_generation_patch_ref = ref
    adapter._planner_generation_native_generate = adapter.generate
    @wraps(adapter.generate)
    def generate(self, *args, **kwargs):
        return _generate_with_patch_ref(self, *args, **kwargs)
    adapter.generate = types.MethodType(generate, adapter)
    return ref


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source", required=True, type=Path)
    parser.add_argument("--entry", choices=["eval", "fixed3"], default="eval")
    parser.add_argument("--expected-fixed3-sha256")
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-eval-sha256", required=True)
    parser.add_argument("--expected-patch-sha256", required=True)
    parser.add_argument("eval_args", nargs=argparse.REMAINDER,
                        help="After --, pass the unchanged frozen music_planner_eval.py arguments")
    args = parser.parse_args(argv)
    if args.entry == "fixed3" and not args.expected_fixed3_sha256:
        parser.error("--expected-fixed3-sha256 is required for --entry fixed3")
    refs = {}
    pinned_sources = [
        ("adapter", args.frozen_source / "music_planner_adapters.py", args.expected_adapter_sha256),
        ("eval", args.frozen_source / "music_planner_eval.py", args.expected_eval_sha256),
        ("patch", Path(__file__), args.expected_patch_sha256),
    ]
    if args.entry == "fixed3":
        pinned_sources.append(("fixed3", args.frozen_source / "music_planner_fixed3_eval.py",
                               args.expected_fixed3_sha256))
    for key, path, expected in pinned_sources:
        ref = _file_ref(path)
        if ref["sha256"] != expected:
            parser.error(f"{key} source SHA256 differs from the explicitly pinned eval contract")
        refs[key] = ref
    eval_args = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    if args.entry == "fixed3" and eval_args[:1] != ["generate"]:
        parser.error("The fixed3 bridge is restricted to the frozen generate subcommand")
    boundary = argparse.ArgumentParser(add_help=False)
    boundary.add_argument("--family", required=True, choices=["moss"])
    boundary.add_argument("--out", required=True, type=Path)
    target, _ = boundary.parse_known_args(eval_args)
    if target.out.exists():
        parser.error("Output must be fresh; frozen training/evaluation outputs cannot be overwritten")
    if "music_planner_adapters" in sys.modules:
        parser.error("Use an isolated CLI process; an adapter module is already imported")
    adapters = _load_module("music_planner_adapters", refs["adapter"]["path"])
    entry_ref = refs[args.entry]
    evaluator = _load_module("_moss_generation_frozen_" + args.entry, entry_ref["path"])
    refs["position_buffer_patch"] = _file_ref(POSITION_PATCH_PATH)
    original_loader = adapters.load_planner_adapter
    applied = []
    def patched_loader(*positional, **options):
        manifest = json.loads((target.out / "manifest.json").read_text())
        trained_source_sha = manifest["checkpoint_identity"]["training_config"]["adapter_source_sha256"]
        if trained_source_sha != refs["adapter"]["sha256"]:
            raise ValueError("Eval bridge adapter source differs from the original trained adapter identity")
        adapter = original_loader(*positional, **options)
        ref = apply_moss_generation_patch(adapter)
        applied.append(ref)
        with (target.out / "generation_patch.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(schema="moss_generation_patch_receipt_v1",
                           applied_generation_patch_ref=ref, frozen_source_refs=refs,
                           position_buffer_receipt=adapter.position_buffer_patch_receipt,
                           scope="eval-only-instance-prepare-native-cache", cache_disabled=False,
                           original_training_identity_preserved=True), stream, indent=2)
        return adapter
    adapters.load_planner_adapter = patched_loader
    original_argv = sys.argv
    try:
        sys.argv = [entry_ref["path"], *eval_args]
        evaluator.main()
    finally:
        sys.argv = original_argv
        adapters.load_planner_adapter = original_loader
        manifest_path = target.out / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["requested_generation_patch_ref"] = generation_patch_ref()
            manifest["applied_generation_patch_ref"] = applied[0] if applied else None
            manifest["eval_bridge_frozen_source_refs"] = refs
            receipt_path = target.out / "generation_patch.json"
            manifest["generation_patch_receipt_ref"] = _file_ref(receipt_path) if receipt_path.is_file() else None
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
