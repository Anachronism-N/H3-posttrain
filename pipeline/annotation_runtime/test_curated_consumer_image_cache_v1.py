"""CPU deployment tests: three real request sets plus native resume/IO trace.

Only temporary /tmp artifacts are written. Runtime/model calls are mocked;
these tests do not claim generated-video semantics or GPU throughput acceptance.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import curated_consumer_image_cache_v1 as wrapper
import test_curated_request_image_cache_v1 as fixtures

bar, pipeline = fixtures.bar, wrapper.pipeline


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.rgb_content = bar.rgb_content
        self.process_source = bar.process_source
        self.save = pipeline.save
        self.style = pipeline.observe_dance_style

    def tearDown(self):
        self.assertIs(bar.rgb_content, self.rgb_content)
        self.assertIs(bar.process_source, self.process_source)
        self.assertIs(pipeline.save, self.save)
        self.assertIs(pipeline.observe_dance_style, self.style)

    def args(self, root, extra=()):
        return pipeline.parser().parse_args(["consumer", "--manifest", str(root / "manifest.summary.json"),
            "--out", str(root), "--ms-source", str(wrapper.PIPELINE_PATH.parent),
            "--model", "/tmp/cpu-no-model", *extra])

    def configuration(self, args):
        args.hand_roi_root = str(Path(args.out).resolve() / "roi")
        config = bar.params(args, bar.code_identity())
        config["style_module_sha256"] = pipeline.sha(Path(bar.__file__).with_name("ms_dance_style.py"))
        return config

    def assert_protocol(self, protocol, configuration):
        self.assertEqual(protocol["configuration"], configuration)
        self.assertEqual(protocol["pipeline_code"], pipeline.ref(wrapper.PIPELINE_PATH))
        for field, path in (("consumer_execution_wrapper_ref", wrapper.__file__),
                            ("request_image_cache_ref", wrapper.CACHE_PATH),
                            ("original_pipeline_ref", wrapper.PIPELINE_PATH)):
            self.assertEqual(protocol[field], pipeline.ref(path))
        self.assertIs(protocol["model_input_request_bytes_unchanged"], True)
        self.assertEqual(protocol["cache_scope"], "single-source")
        self.assertEqual(protocol["params_identity_scope"], "semantic_native_configuration")

    def test_three_real_requests_through_wrapper(self):
        roots = fixtures.source_roots()
        protected = [wrapper.PIPELINE_PATH, wrapper.CACHE_PATH, Path(bar.__file__),
                     Path(fixtures.__file__)]
        sources, evidence = {}, {}
        for sid, root in zip(fixtures.IDS, roots):
            prepared = bar.load(root / "prepared.json")
            wid = prepared["windows"][0]["window"]["window_id"]
            recorded = bar.load(root / "windows" / wid / "attempt-0" / "request.json")
            request_args = SimpleNamespace(model=recorded["model"],
                visual_max_tokens=recorded["sampling"]["max_tokens"], retry_failed=False)
            with fixtures.jpeg_reads(root / "rgb") as (reads, _):
                requests, locations = fixtures.request_set(root, prepared, request_args)
            for request, location in zip(requests, locations):
                self.assertEqual(fixtures.canonical(request), fixtures.canonical(bar.load(location)))
            protected.extend([root / "prepared.json", *locations])
            sources[sid] = (root, prepared, recorded)
            evidence[sid] = {"requests": requests, "locations": locations, "reads": sum(reads.values())}
        before_files = {path: pipeline.sha(path) for path in protected}
        native_code = bar.code_identity()
        before_core_identity = bar.core.code_identity()
        output = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="consumer-cache-real-", dir="/tmp") as tmp:
            root = Path(tmp)
            args = self.args(root)
            args.model = sources[fixtures.IDS[0]][2]["model"]
            args.visual_max_tokens = sources[fixtures.IDS[0]][2]["sampling"]["max_tokens"]
            config = self.configuration(args)
            native_args = copy.deepcopy(vars(args))
            old = root / "runs" / "v0-old" / "protocol.json"
            self.save(old, {"configuration": config, "fixture": "old immutable protocol"})
            old_bytes = old.read_bytes()
            new = root / "runs" / "v0-new" / "protocol.json"
            contexts = []
            cache_class = wrapper.image_cache.SourceImageCache

            def context(module, rgb_root):
                instance = cache_class(module, rgb_root)
                contexts.append(instance)
                return instance

            def process(row, raw_line, source_root, actual_args, binding, runtime, ref_cache):
                sid = row["id"]
                self.assertIs(actual_args, args)
                self.assertEqual(vars(actual_args), native_args)
                self.assertEqual(binding, {"params_sha256": bar.batch.digest(config)})
                self.assertEqual(raw_line, b"native raw line")
                self.assertIsNot(bar.rgb_content, self.rgb_content)
                self.assertEqual(bar.code_identity(), native_code)
                self.assertEqual(bar.core.code_identity(), before_core_identity)
                self.assertEqual(self.configuration(actual_args), config)
                prepared = sources[sid][1]
                with fixtures.jpeg_reads(source_root / "rgb") as (reads, _):
                    requests, locations = fixtures.request_set(source_root, prepared, actual_args)
                self.assertEqual(locations, evidence[sid]["locations"])
                self.assertEqual(len(requests), len(evidence[sid]["requests"]))
                for original, cached in zip(evidence[sid]["requests"], requests):
                    self.assertEqual(fixtures.canonical(original), fixtures.canonical(cached))
                    self.assertEqual(bar.core.batch.json_bytes(original), bar.core.batch.json_bytes(cached))
                self.assertEqual(sum(reads.values()), contexts[-1].stats["read_count"])
                evidence[sid]["cached_reads"] = sum(reads.values())
                evidence[sid]["request_set_sha256"] = hashlib.sha256(fixtures.canonical(requests)).hexdigest()
                return {"id": sid, "status": "cpu_requests_only"}

            def driver(actual_args):
                self.assertIs(actual_args, args)
                pipeline.save(new, {"mode": "consumer", "configuration": config,
                                   "pipeline_code": pipeline.ref(wrapper.PIPELINE_PATH)})
                for sid, (source_root, _, _) in sources.items():
                    self.assertIs(bar.rgb_content, self.rgb_content)
                    bar.process_source({"id": sid}, b"native raw line", source_root, args,
                                       {"params_sha256": bar.batch.digest(config)}, None, {})
                    # The native style function is neither patched nor inside the cache.
                    self.assertIs(pipeline.observe_dance_style, self.style)
                    self.assertIs(bar.rgb_content, self.rgb_content)
                return {"run_root": str(new.parent), "cpu_trace_only": True}

            with patch.object(bar, "process_source", process), patch.object(pipeline, "run_consumer", driver), \
                    patch.object(wrapper.image_cache, "SourceImageCache", context), redirect_stdout(output):
                wrapper.run_consumer(args)
            self.assertEqual(old.read_bytes(), old_bytes)
            self.assert_protocol(pipeline.load(new), config)
            self.assertEqual(len(contexts), 3)
            for instance in contexts:
                self.assertFalse(instance._active)
                self.assertFalse(instance._bindings)
                self.assertFalse(instance._images["body"])
                self.assertFalse(instance._images["crop"])
            self.assertEqual(self.configuration(args), config)
        self.assertEqual({path: pipeline.sha(path) for path in protected}, before_files)
        traces = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([trace["source_id"] for trace in traces], list(fixtures.IDS))
        total_before = total_after = 0
        for sid, trace in zip(fixtures.IDS, traces):
            item = evidence[sid]
            self.assertEqual(trace["read_count"], item["cached_reads"])
            self.assertGreater(trace["cache_hits"], 0)
            self.assertEqual(trace["stat_checks"], trace["read_count"] * 4 + trace["cache_hits"] * 2)
            self.assertLess(item["cached_reads"], item["reads"])
            total_before += item["reads"]
            total_after += item["cached_reads"]
            print(json.dumps({"source": sid, "request_count": len(item["requests"]),
                "request_set_sha256": item["request_set_sha256"],
                "recorded_original_and_native_bytes_equal": True,
                "original_read_count": item["reads"],
                "reduced_read_count": item["reads"] - item["cached_reads"], **trace}, sort_keys=True))
        self.assertEqual(sum(len(item["requests"]) for item in evidence.values()), 11)
        self.assertEqual((total_before, total_after), (179, 122))
        print(json.dumps({"three_case_original_reads": total_before, "three_case_cached_reads": total_after,
            "native_semantic_params_sha256": bar.batch.digest(config),
            "new_protocol_wrapper_and_cache_bound": True, "old_protocol_unchanged": True,
            "bar_and_core_code_identity_unchanged": True}, sort_keys=True))

    def test_actual_native_consumer_resume_and_shard_model_arguments(self):
        with tempfile.TemporaryDirectory(prefix="consumer-cache-resume-", dir="/tmp") as tmp:
            root = Path(tmp)
            args = self.args(root, ["--shard", "2", "--nshard", "4", "--batch-size", "3",
                "--max-num-seqs", "4", "--max-model-len", "17000", "--max-pixels", "150000",
                "--visual-max-tokens", "900", "--text-max-tokens", "2500",
                "--gpu-memory-utilization", "0.77", "--boundary-review", "off",
                "--rewrite-reference-repair", "off", "--retry-failed"])
            config = self.configuration(args)
            summary = {"total": 8, "counts": {"train": 8}, "batches": []}
            self.save(args.manifest, summary)
            old_protocol = root / "runs" / "v2-old" / "protocol.json"
            self.save(old_protocol, {"mode": "consumer", "configuration": config})
            protected = [old_protocol]
            hints = []
            for ordinal in (0, 2, 3, 6):
                sid = "source-" + str(ordinal)
                hint = {"id": sid, "split": "train", "source_ordinal": ordinal, "status": "ready"}
                directory = root / "sources" / "train" / sid
                producer = directory / "producer.terminal.json"
                self.save(producer, hint)
                if ordinal % 4 == 2:
                    artifact = directory / "old-joint.json"
                    self.save(artifact, {"id": sid, "fixture": "completed original candidate"})
                    terminal = directory / "consumer.terminal.json"
                    self.save(terminal, {"status": "joint_candidate", "params_sha256": bar.batch.digest(config),
                                         "joint_candidate_ref": pipeline.ref(artifact)})
                    protected.extend([artifact, terminal])
                protected.append(producer)
                hints.append(hint)
            queue = root / "ready.jsonl"
            queue.write_bytes(b"".join(pipeline.raw_json(hint) for hint in hints))
            protected.append(queue)
            original_bytes = {path: path.read_bytes() for path in protected}
            runtime = unittest.mock.Mock(side_effect=AssertionError("model execution forbidden"))
            with patch.object(bar, "runtime_loader", return_value=runtime) as loader, \
                    patch.object(bar, "process_source", side_effect=AssertionError("terminal must skip")) as process:
                process._curated_consumer_image_cache_v1 = False
                result = wrapper.run_consumer(args)
                loader.assert_called_once()
                loader_args, loader_out, loader_code = loader.call_args.args
                self.assertIs(loader_args, args)
                self.assertEqual(loader_out, root / "visual" / "shard-0002")
                self.assertEqual(loader_code, bar.code_identity())
                process.assert_not_called()
                runtime.assert_not_called()
            self.assertEqual(result["counts"], {"resumed_terminal": 2})
            self.assertEqual(result["queue_sources_consumed"], 2)
            self.assertEqual({path: path.read_bytes() for path in protected}, original_bytes)
            protocol = pipeline.load(Path(result["run_root"]) / "protocol.json")
            self.assert_protocol(protocol, config)
            self.assertEqual((protocol["shard"], protocol["nshard"]), (2, 4))
            self.assertFalse((root / "completed.jsonl").exists())
            for ordinal in (0, 3):
                self.assertFalse((root / "sources" / "train" / f"source-{ordinal}" / "consumer.terminal.json").exists())
            # The original resume fingerprint and artifact integrity checks remain active.
            target = root / "sources" / "train" / "source-2" / "consumer.terminal.json"
            original = pipeline.load(target)
            for field, bad, error in (("params_sha256", "0" * 64, "different code/model/parameters"),
                                      ("joint_candidate_ref", {**original["joint_candidate_ref"], "sha256": "0" * 64},
                                       "artifact changed")):
                changed = {**original, field: bad}
                target.write_bytes(pipeline.raw_json(changed))
                with patch.object(bar, "runtime_loader", return_value=runtime):
                    with self.assertRaisesRegex(ValueError, error):
                        wrapper.run_consumer(args)
                target.write_bytes(pipeline.raw_json(original))
            print(json.dumps({"actual_native_resume": True, "resumed_terminal": 2, "shard": 2, "nshard": 4,
                "model_calls": 0, "params_sha256": bar.batch.digest(config),
                "old_files_unchanged": True, "original_resume_integrity_rejections": True}, sort_keys=True))

    def test_process_and_runtime_exceptions_restore_without_reentry_pollution(self):
        with tempfile.TemporaryDirectory(prefix="consumer-cache-exception-", dir="/tmp") as tmp:
            root = Path(tmp)
            args = self.args(root)
            for source_exception in (ValueError, RuntimeError):
                contexts = []
                actual_cache = wrapper.image_cache.SourceImageCache

                def context(module, rgb_root):
                    instance = actual_cache(module, rgb_root)
                    contexts.append(instance)
                    return instance

                def process(*unused):
                    self.assertIsNot(bar.rgb_content, self.rgb_content)
                    raise source_exception("source sentinel")

                def driver(actual_args):
                    with self.assertRaisesRegex(ValueError, "reentry"):
                        wrapper.run_consumer(actual_args)
                    bar.process_source({"id": "source"}, b"", root, actual_args, {}, None, {})

                with patch.object(bar, "process_source", process), patch.object(pipeline, "run_consumer", driver), \
                        patch.object(wrapper.image_cache, "SourceImageCache", context), redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(source_exception, "source sentinel"):
                        wrapper.run_consumer(args)
                    self.assertIs(bar.process_source, process)
                self.assertEqual(len(contexts), 1)
                self.assertFalse(contexts[0]._active)
                self.assertIs(bar.rgb_content, self.rgb_content)
                self.assertIs(pipeline.save, self.save)
            with patch.object(pipeline, "run_consumer", side_effect=RuntimeError("runtime sentinel")):
                with self.assertRaisesRegex(RuntimeError, "runtime sentinel"):
                    wrapper.run_consumer(args)
            self.assertIs(bar.process_source, self.process_source)
            with patch.object(pipeline, "run_consumer", return_value={"fresh": True}):
                self.assertEqual(wrapper.run_consumer(args), {"fresh": True})

    def test_existing_protocol_cannot_receive_wrapper_metadata(self):
        with tempfile.TemporaryDirectory(prefix="consumer-cache-oldprotocol-", dir="/tmp") as tmp:
            root = Path(tmp)
            args = self.args(root)
            old = root / "runs" / "v0-old" / "protocol.json"
            value = {"mode": "consumer", "configuration": {"native": True}}
            self.save(old, value)
            original = old.read_bytes()
            with patch.object(pipeline, "run_consumer", side_effect=lambda unused: pipeline.save(old, value)):
                with self.assertRaisesRegex(ValueError, "existing run protocol"):
                    wrapper.run_consumer(args)
            self.assertEqual(old.read_bytes(), original)

    def test_foreign_pipeline_cache_and_bar_hash_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="consumer-cache-hash-", dir="/tmp") as tmp:
            root = Path(tmp)
            foreign = root / "foreign.py"
            foreign.write_text("raise AssertionError('foreign code must not execute')\n")
            for expected in (wrapper.PIPELINE_SHA, wrapper.CACHE_SHA):
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    wrapper.load_verified(foreign, expected, "foreign_execution")
            for attribute in ("PIPELINE_PATH", "CACHE_PATH"):
                with patch.object(wrapper, attribute, foreign), patch.object(pipeline, "modules") as modules:
                    with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                        wrapper.run_consumer(self.args(root))
                    modules.assert_not_called()
            foreign_bar = SimpleNamespace(__file__=str(foreign))
            with patch.object(pipeline, "modules", return_value=(foreign_bar, None, None)):
                with self.assertRaisesRegex(ValueError, "Original bar source SHA mismatch"):
                    wrapper.run_consumer(self.args(root))
            self.assertFalse((root / "runs").exists())

    def test_cli_uses_exact_native_consumer_parser(self):
        argv = ["consumer", "--manifest", "/tmp/manifest", "--out", "/tmp/out",
            "--repo", "/tmp/repo", "--ms-source", str(wrapper.PIPELINE_PATH.parent),
            "--shard", "3", "--nshard", "4", "--limit", "3", "--ids", *fixtures.IDS,
            "--stage", "prepare", "--ffprobe", "/tmp/ffprobe", "--ffmpeg", "/tmp/ffmpeg",
            "--model", "/tmp/model", "--model-reference", "/tmp/model-reference", "--tp", "2",
            "--batch-size", "2", "--max-num-seqs", "3", "--sample-fps", "4",
            "--max-model-len", "16384", "--max-pixels", "200704",
            "--visual-max-tokens", "1800", "--text-max-tokens", "4096",
            "--gpu-memory-utilization", "0.88", "--boundary-review", "on",
            "--rewrite-reference-repair", "on", "--retry-failed", "--watch", "--poll-seconds", "7",
            "--style-reuse", "/tmp/style1", "/tmp/style2"]
        expected = vars(pipeline.parser().parse_args(argv))
        with patch.object(wrapper, "run_consumer", side_effect=lambda args: vars(args)):
            self.assertEqual(wrapper.main(argv), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
