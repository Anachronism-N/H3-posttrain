"""Native consumer with one CPU future and source-scoped input reuse."""
import hashlib
import importlib.util
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
DEPS = {"curated_consumer_prepare_ahead_v1.py": "02d792a4bf9aa7977dcbbc783fedcd95f4a06a0b85c223b6c34a5e898d4a1d49",
        "curated_fast_source_v1.py": "adefcf7919a752f3ef09d9b41a4a99c316500b18852efe877eae6020c093511f",
        "curated_transaction_cache_v1.py": "f90f017ab293ddd9c25ceb41d85178a3e69990adf69e3332d5da37b4211613a5"}


def verified_module(name):
    path = RT / name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEPS[name]:
        raise ValueError("Fast consumer dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("fast_consumer_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


ahead = verified_module("curated_consumer_prepare_ahead_v1.py")
fast = verified_module("curated_fast_source_v1.py")
transactions = verified_module("curated_transaction_cache_v1.py")
pipeline = ahead.pipeline


class InputCache:
    def __init__(self, bar, rgb_root):
        self.context = fast.fast_source(bar, Path(rgb_root).parent)
        self.assets = None
        self._stats = {}

    @property
    def stats(self):
        return self.assets.stats if self.assets is not None else self._stats

    def __enter__(self):
        self.assets = self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self.context.__exit__(exc_type, exc, traceback)


def run_consumer(args, helper_factory=ahead.HelperProcess):
    cache_module = ahead.cached.image_cache
    original_cache, original_save = cache_module.SourceImageCache, pipeline.save
    metadata = {"fast_source_execution_ref": pipeline.ref(RT / "curated_fast_source_v1.py"),
                "fast_consumer_execution_ref": pipeline.ref(__file__),
                "transaction_cache_execution_ref": pipeline.ref(RT / "curated_transaction_cache_v1.py"),
                "transaction_cache_scope": "one native LOCK; existing receipts fully verified",
                "model_persistence_timing_separated": True,
                "input_validation_scope": "first read and source completion",
                "immutable_inputs_reused_in_memory": True, "indexed_PTS_frame_selection": True,
                "model_request_and_sampling_unchanged": True}
    runs = Path(args.out).resolve() / "runs"

    def save(path, value):
        path = Path(path)
        if (path.name == "protocol.json" and path.parent.parent.resolve() == runs
                and value.get("mode") == "consumer"):
            if path.exists():
                raise ValueError("Optimized consumer requires a new run protocol")
            value = {**value, **metadata}
        return original_save(path, value)

    cache_module.SourceImageCache, pipeline.save = InputCache, save
    def emit(event):
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)

    try:
        bar, _, _ = pipeline.modules(args.repo, args.ms_source)
        # Source inputs are checked once by fast_source at source completion.
        # Do not scan the entire source again for every window receipt.
        with transactions.transaction_cache(bar, emit):
            return ahead.run_consumer(args, helper_factory)
    finally:
        cache_module.SourceImageCache, pipeline.save = original_cache, original_save


def main(argv=None):
    args = pipeline.parser().parse_args(argv)
    if args.mode != "consumer":
        raise ValueError("Fast entry only supports native consumer mode")
    return run_consumer(args)


if __name__ == "__main__":
    main()
