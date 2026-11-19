"""New native consumer execution with source-scoped request-image IO caching.

Semantic configuration and generation/request identities remain native. Only a
new run protocol attests this execution wrapper; existing artifacts stay intact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
PIPELINE_PATH = RT / "source_curated_consumer_v1" / "curated_pipeline.py"
PIPELINE_SHA = "d5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc"
BAR_SHA = "df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e"
CACHE_PATH = RT / "curated_request_image_cache_v1.py"
CACHE_SHA = "8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3"


def load_verified(path, expected_sha, name):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Execution source SHA mismatch: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Execute precisely the verified bytes, not an independently cached pyc.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


pipeline = load_verified(PIPELINE_PATH, PIPELINE_SHA, "_curated_cache_native_pipeline_v1")
image_cache = load_verified(CACHE_PATH, CACHE_SHA, "_curated_cache_native_images_v1")


def run_consumer(args):
    for path, expected in ((PIPELINE_PATH, PIPELINE_SHA), (CACHE_PATH, CACHE_SHA)):
        if pipeline.sha(path) != expected:
            raise ValueError("Execution source SHA mismatch: " + str(path))
    bar, _, _ = pipeline.modules(args.repo, args.ms_source)
    if pipeline.sha(bar.__file__) != BAR_SHA:
        raise ValueError("Original bar source SHA mismatch")
    original_process, original_save = bar.process_source, pipeline.save
    if getattr(original_process, "_curated_consumer_image_cache_v1", False):
        raise ValueError("consumer image cache reentry is forbidden")
    metadata = {
        "consumer_execution_wrapper_ref": pipeline.ref(__file__),
        "request_image_cache_ref": pipeline.ref(CACHE_PATH),
        "original_pipeline_ref": pipeline.ref(PIPELINE_PATH),
        "model_input_request_bytes_unchanged": True,
        "cache_scope": "single-source",
        "params_identity_scope": "semantic_native_configuration",
    }
    runs_root = Path(args.out).resolve() / "runs"

    def cached_process(row, raw_line, root, native_args, binding, runtime, ref_cache):
        with image_cache.SourceImageCache(bar, Path(root) / "rgb") as cache:
            try:
                return original_process(row, raw_line, root, native_args, binding, runtime, ref_cache)
            finally:
                print(json.dumps({"event": "single_source_image_cache", "source_id": row["id"],
                                  "cache_scope": "single-source", **cache.stats}, sort_keys=True), flush=True)

    cached_process._curated_consumer_image_cache_v1 = True

    def save_new_protocol(path, value):
        path = Path(path)
        if (path.name == "protocol.json" and path.parent.parent.resolve() == runs_root
                and path.parent.name.startswith(f"v{args.shard}-") and value.get("mode") == "consumer"):
            if path.exists():
                raise ValueError("Refusing cache metadata on an existing run protocol")
            value = {**value, **metadata}
        return original_save(path, value)

    bar.process_source, pipeline.save = cached_process, save_new_protocol
    try:
        return pipeline.run_consumer(args)
    finally:
        bar.process_source, pipeline.save = original_process, original_save


def main(argv=None):
    args = pipeline.parser().parse_args(argv)
    if args.mode != "consumer":
        raise ValueError("This execution wrapper only runs the native consumer mode")
    return run_consumer(args)


if __name__ == "__main__":
    try:
        print(json.dumps(main(), ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({"status": "worker_stopped", "error": f"{type(error).__name__}: {error}",
                          "remaining_sources_not_marked_failed": isinstance(error, pipeline.SharedFailure),
                          **pipeline.FLAGS}), file=sys.stderr, flush=True)
        sys.exit(2)
