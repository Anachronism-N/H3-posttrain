"""Eval-only native EOS argument bridge; never alter frozen code, weights or text."""
import argparse
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

FAMILIES = ("moss", "music_flamingo", "omni_captioner")


def _file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def generation_eos_patch_ref():
    return _file_ref(__file__)


def _valid_eos(value):
    ids = list(value) if isinstance(value, (list, tuple)) else [value]
    return bool(ids) and all(type(token) is int and token >= 0 for token in ids)


def apply_generation_eos_patch(adapter):
    """Supply the trained assistant EOS only when native generation lacks valid EOS."""
    if getattr(adapter, "family", None) not in FAMILIES:
        raise ValueError("A native music planner adapter is required")
    model = adapter.model.get_base_model() if hasattr(adapter.model, "get_base_model") else adapter.model
    ref = generation_eos_patch_ref()
    if "_planner_generation_eos_native_generate" in vars(model):
        if (vars(model).get("_planner_generation_eos_patch_ref") != ref or
                vars(adapter).get("applied_generation_eos_patch_ref") != ref):
            raise ValueError("Native generation already has a different EOS patch identity")
        return ref
    native_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if _valid_eos(native_eos):
        return None
    eos = adapter.eos_token_ids()
    terminator = adapter.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if (not _valid_eos(eos) or len(terminator) != 1 or
            list(eos) != terminator or not _valid_eos(terminator)):
        raise ValueError("Fallback EOS must be exactly the native trained assistant im_end ID")
    original_generate = model.generate
    original_adapter_generate = adapter.generate
    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        effective = kwargs.get("eos_token_id", getattr(
            kwargs.get("generation_config") or self.generation_config, "eos_token_id", None))
        if not _valid_eos(effective):
            kwargs["eos_token_id"] = list(eos)
        return self._planner_generation_eos_native_generate(*args, **kwargs)
    @wraps(original_adapter_generate)
    def adapter_generate(self, *args, **kwargs):
        result = original_adapter_generate(*args, **kwargs)
        return dict(result, applied_generation_eos_patch_ref=ref)
    model._planner_generation_eos_native_generate = original_generate
    model._planner_generation_eos_patch_ref = ref
    model.generate = types.MethodType(generate, model)
    adapter.applied_generation_eos_patch_ref = ref
    adapter.generation_eos_patch_details = dict(
        original_native_eos_token_id=native_eos, explicit_eos_token_ids=list(eos),
        training_assistant_terminator_ids=terminator,
        scope="eval-only-instance-native-generate-eos-argument",
        config_mutated=False, weights_mutated=False, prefix_mutated=False,
        decoding_penalties_mutated=False, raw_output_cleaning=False)
    adapter.generate = types.MethodType(adapter_generate, adapter)
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
    parser.add_argument("--entry", choices=("eval", "fixed3"), default="eval")
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-eval-sha256", required=True)
    parser.add_argument("--expected-patch-sha256", required=True)
    parser.add_argument("--expected-fixed3-sha256")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry == "fixed3" and not args.expected_fixed3_sha256:
        parser.error("--expected-fixed3-sha256 is required for --entry fixed3")
    pinned = [
        ("adapter", args.frozen_source / "music_planner_adapters.py", args.expected_adapter_sha256),
        ("eval", args.frozen_source / "music_planner_eval.py", args.expected_eval_sha256),
        ("patch", Path(__file__), args.expected_patch_sha256),
    ]
    if args.entry == "fixed3":
        pinned.append(("fixed3", args.frozen_source / "music_planner_fixed3_eval.py",
                       args.expected_fixed3_sha256))
    refs = {}
    for key, path, expected in pinned:
        ref = _file_ref(path)
        if ref["sha256"] != expected:
            parser.error(f"{key} source SHA256 differs from the explicitly pinned eval contract")
        refs[key] = ref
    eval_args = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    if args.entry == "fixed3" and eval_args[:1] != ["generate"]:
        parser.error("The fixed3 bridge permits only the unchanged generate subcommand")
    boundary = argparse.ArgumentParser(add_help=False)
    boundary.add_argument("--family", required=True, choices=FAMILIES)
    boundary.add_argument("--out", required=True, type=Path)
    target, _ = boundary.parse_known_args(eval_args)
    if target.out.exists():
        parser.error("Output must be fresh; frozen outputs cannot be overwritten")
    module_names = ("music_planner_adapters", "_eos_generation_frozen_eval", "_eos_generation_frozen_fixed3")
    if any(name in sys.modules for name in module_names):
        parser.error("Use an isolated CLI process; a bridge/adapter module is already imported")
    original_argv, original_path = sys.argv, sys.path[:]
    original_loader, adapters = None, None
    applied, details = [], []
    try:
        adapters = _load_module(module_names[0], refs["adapter"]["path"])
        evaluator = _load_module(module_names[1], refs["eval"]["path"])
        entry = evaluator
        if args.entry == "fixed3":
            entry = _load_module(module_names[2], refs["fixed3"]["path"])
            if Path(entry.FROZEN).resolve() != args.frozen_source.resolve():
                raise ValueError("Fixed3 native adapter source is not the pinned frozen source")
            entry.frozen_eval = lambda: evaluator
        original_loader = adapters.load_planner_adapter
        def patched_loader(*positional, **options):
            manifest = json.loads((target.out / "manifest.json").read_text())
            trained_sha = manifest["checkpoint_identity"]["training_config"]["adapter_source_sha256"]
            if trained_sha != refs["adapter"]["sha256"]:
                raise ValueError("Bridge source differs from the original trained adapter identity")
            adapter = original_loader(*positional, **options)
            if adapter.family != target.family:
                raise ValueError("Loaded adapter family differs from the frozen evaluation")
            ref = apply_generation_eos_patch(adapter)
            applied.append(ref)
            detail = getattr(adapter, "generation_eos_patch_details", dict(
                scope="no-patch-native-generation-EOS-already-valid",
                original_native_eos_token_id=getattr(adapter.model.generation_config, "eos_token_id", None)))
            details.append(detail)
            receipt = dict(schema="music_planner_generation_eos_patch_receipt_v1",
                requested_generation_eos_patch_ref=refs["patch"],
                applied_generation_eos_patch_ref=ref,
                applied_generation_eos_patch_source_refs=refs,
                family=adapter.family, applied=ref is not None, details=detail,
                original_training_identity_preserved=True)
            with (target.out / "generation_eos_patch.json").open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            return adapter
        adapters.load_planner_adapter = patched_loader
        sys.argv = [refs[args.entry]["path"], *eval_args]
        entry.main()
    finally:
        sys.argv, sys.path[:] = original_argv, original_path
        if original_loader is not None:
            adapters.load_planner_adapter = original_loader
        for name in module_names:
            sys.modules.pop(name, None)
        manifest_path = target.out / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["requested_generation_eos_patch_ref"] = refs["patch"]
            manifest["applied_generation_eos_patch_ref"] = applied[0] if applied else None
            manifest["applied_generation_eos_patch_source_refs"] = refs
            manifest["generation_eos_patch_details"] = details[0] if details else None
            receipt_path = target.out / "generation_eos_patch.json"
            manifest["generation_eos_patch_receipt_ref"] = _file_ref(receipt_path) if receipt_path.is_file() else None
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
