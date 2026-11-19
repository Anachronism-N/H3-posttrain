"""CPU tests for input reuse, exact sampling, and consumer integration."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import curated_fast_source_v1 as m
import curated_consumer_fast_v1 as consumer


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "rgb").mkdir()
        self.image = self.root / "rgb/frame.jpg"
        self.image.write_bytes(b"original JPEG bytes")
        self.frame = {"file": "frame.jpg", "sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
                      "index": 0, "time": 0.0, "source_timestamp": 0.0}
        self.json = self.root / "prepared.json"
        self.json.write_text(json.dumps({"value": [1, 2]}))
        self.dynamic = self.root / "result.json"
        self.dynamic.write_text(json.dumps({"value": 1}))
        def load(p):
            return json.loads(Path(p).read_bytes())
        def sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        def need(condition, message):
            if not condition:
                raise ValueError(message)
        self.bar = SimpleNamespace(load=load, strict_json=json.loads, need=need,
                batch=SimpleNamespace(sha256=sha, select_frames=lambda *a: []),
                rgb_content=lambda *a: [], verify_prepared=lambda *a: {}, prepare_source=lambda *a: {},
                select_window_frames=lambda *a: [])
        self.original = (self.bar.load, self.bar.batch.sha256, self.bar.rgb_content,
                         self.bar.verify_prepared, self.bar.select_window_frames)

    def test_hash_and_load_share_single_read_and_parse(self):
        with m.fast_source(self.bar, self.root) as cache:
            expected = self.bar.batch.sha256(self.json)
            first = self.bar.load(self.json)
            first["value"].append(3)
            with patch.object(Path, "open", side_effect=AssertionError("hot disk open")), \
                 patch.object(Path, "resolve", side_effect=AssertionError("hot resolve")):
                for _ in range(30):
                    self.assertEqual(self.bar.batch.sha256(self.json), expected)
                    self.assertEqual(self.bar.load(self.json), {"value": [1, 2]})
            self.assertEqual(cache.stats["input_reads"], 1)
            self.assertEqual(cache.stats["json_parses"], 1)
        self.assertEqual(self.original, (self.bar.load, self.bar.batch.sha256, self.bar.rgb_content,
                                        self.bar.verify_prepared, self.bar.select_window_frames))

    def test_model_outputs_are_not_cached(self):
        with m.fast_source(self.bar, self.root):
            self.assertEqual(self.bar.load(self.dynamic), {"value": 1})
            self.dynamic.write_text(json.dumps({"value": 2}))
            self.assertEqual(self.bar.load(self.dynamic), {"value": 2})

    def test_repeated_JPEG_reuses_read_hash_and_base64(self):
        with m.fast_source(self.bar, self.root) as cache:
            first = self.bar.rgb_content(self.root / "rgb", [self.frame], {})
            with patch.object(Path, "open", side_effect=AssertionError("hot disk open")), \
                 patch.object(Path, "resolve", side_effect=AssertionError("hot resolve")):
                self.assertEqual(first, self.bar.rgb_content(self.root / "rgb", [self.frame], {}))
            self.assertEqual(cache.stats["input_reads"], 1)

    def test_image_budget_preserved(self):
        with m.fast_source(self.bar, self.root):
            with self.assertRaisesRegex(ValueError, "image budget"):
                self.bar.rgb_content(self.root / "rgb", [self.frame] * 17, {})

    def test_changed_input_rejected_before_return(self):
        with self.assertRaisesRegex(ValueError, "before result commit"):
            with m.fast_source(self.bar, self.root):
                self.bar.load(self.json)
                self.json.write_text(json.dumps({"value": [9]}))
        self.assertIs(self.bar.load, self.original[0])

    def test_bad_JPEG_SHA_and_escape_rejected(self):
        with m.fast_source(self.bar, self.root):
            wrong = dict(self.frame, sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "expected SHA"):
                self.bar.rgb_content(self.root / "rgb", [wrong], {})
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.bar.rgb_content(self.root / "rgb", [dict(self.frame, file="../prepared.json")], {})

    def test_symlink_escape_rejected(self):
        (self.root / "rgb/escape.jpg").symlink_to(self.json)
        with m.fast_source(self.bar, self.root):
            with self.assertRaisesRegex(ValueError, "symlink escapes"):
                self.bar.rgb_content(self.root / "rgb", [dict(self.frame, file="escape.jpg")], {})

    def test_exception_and_nested_context_restore_original_functions(self):
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with m.fast_source(self.bar, self.root):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with m.fast_source(self.bar, self.root):
                        pass
                raise RuntimeError("failure")
        self.assertIs(self.bar.rgb_content, self.original[2])
        self.assertFalse(self.bar._fast_source_active)

    def test_strict_parser_is_not_replaced(self):
        self.bar.strict_json = lambda raw: (_ for _ in ()).throw(ValueError("strict parser rejection"))
        with m.fast_source(self.bar, self.root):
            with self.assertRaisesRegex(ValueError, "strict parser rejection"):
                self.bar.load(self.json)


class FrameTests(unittest.TestCase):
    def test_random_VFR_and_short_windows_match_native(self):
        pipeline = consumer.pipeline
        args = pipeline.parser().parse_args(["consumer", "--manifest", "/fixture", "--out", "/fixture",
                                            "--model", "/fixture", "--ms-source",
                                            str(Path(m.__file__).parent / "source_curated_consumer_v1")] )
        bar, _, _ = pipeline.modules(args.repo, args.ms_source)
        random.seed(13)
        times = [0.0]
        for _ in range(2000):
            times.append(times[-1] + random.uniform(0.005, 0.05))
        index = m.FrameSelection()
        for _ in range(100):
            start = random.choice(times)
            end = start + random.uniform(0.001, 6.0)
            rate = random.choice([1.0, 2.0, 4.0, 8.0])
            self.assertEqual(index.window(times, start, end, rate), bar.select_window_frames(times, start, end, rate))
            if len([t for t in times if start <= t < end]) >= 2:
                self.assertEqual(index.select(times, start, end, rate, 16), bar.batch.select_frames(times, start, end, rate, 16))

    def test_sample_results_do_not_share_mutable_lists(self):
        index = m.FrameSelection()
        times = [0.0, 0.1, 0.2, 0.3]
        selected = index.select(times, 0.0, 0.4)
        selected.clear()
        self.assertTrue(index.select(times, 0.0, 0.4))


class ConsumerTests(unittest.TestCase):
    def args(self):
        return consumer.pipeline.parser().parse_args(["consumer", "--manifest", "/fixture",
               "--out", "/fixture/output", "--model", "/fixture", "--stage", "run",
               "--ms-source", str(Path(m.__file__).parent / "source_curated_consumer_v1")])

    def test_feature_is_enabled_in_consumer_and_restored(self):
        original = consumer.ahead.cached.image_cache.SourceImageCache
        save = consumer.pipeline.save
        args = self.args()
        bar, _, _ = consumer.pipeline.modules(args.repo, args.ms_source)
        original_transactions = bar.transactions
        factory = object()
        def run(current, helper):
            self.assertIs(current, args)
            self.assertIs(helper, factory)
            self.assertIs(consumer.ahead.cached.image_cache.SourceImageCache, consumer.InputCache)
            self.assertIsNot(consumer.pipeline.save, save)
            self.assertIsNot(bar.transactions, original_transactions)
            self.assertIsNone(bar.transactions.__self__.source_boundary)
            from test_curated_transaction_cache_v1 import FakeModel, request
            model = FakeModel()
            with tempfile.TemporaryDirectory() as root, patch.object(bar.core, "structured_sampling", return_value=None):
                item = {"root": Path(root) / "attempt", "request": request(),
                        "validator": lambda value: value, "binding": {"source": "fixture"}}
                result = bar.transactions([item], lambda: (model, None), args)
                self.assertEqual(next(iter(result.values()))["status"], "candidate")
                self.assertEqual(len(model.calls), 1)
            return "enabled"
        with patch.object(consumer.ahead, "run_consumer", side_effect=run):
            self.assertEqual(consumer.run_consumer(args, factory), "enabled")
        self.assertIs(consumer.ahead.cached.image_cache.SourceImageCache, original)
        self.assertIs(consumer.pipeline.save, save)
        self.assertIs(bar.transactions, original_transactions)

    def test_failed_consumer_restores_wrappers(self):
        original = consumer.ahead.cached.image_cache.SourceImageCache
        save = consumer.pipeline.save
        with patch.object(consumer.ahead, "run_consumer", side_effect=RuntimeError("consumer failure")):
            with self.assertRaisesRegex(RuntimeError, "consumer failure"):
                consumer.run_consumer(self.args())
        self.assertIs(consumer.ahead.cached.image_cache.SourceImageCache, original)
        self.assertIs(consumer.pipeline.save, save)


if __name__ == "__main__":
    unittest.main(verbosity=2)
