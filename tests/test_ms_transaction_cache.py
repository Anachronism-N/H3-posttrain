"""CPU-only native-bytecode parity, fail-closed, and actual model-wall timing tests."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_bar_annotation as bar
import ms_transaction_cache as m

core = bar.core


def request():
    return {"messages": [{"role": "user", "content": "fixture only"}],
            "sampling": {"max_tokens": 64, "temperature": 0.0, "seed": 0},
            "json_schema": {"type": "object"}}


class FakeModel:
    def __init__(self, text='{"value":1}', delay=0.0, failure=False):
        self.text, self.delay, self.failure = text, delay, failure
        self.calls = []
        self._ms_bar_model_sha256 = "fixture model identity"

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=2)

    def chat(self, messages, sampling, use_tqdm):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(sampling), use_tqdm))
        time.sleep(self.delay)
        if self.failure:
            raise RuntimeError("fixture chat failure")
        completion = SimpleNamespace(text=self.text, token_ids=[3, 4, 2],
                                     finish_reason="stop", stop_reason=None)
        return [SimpleNamespace(prompt_token_ids=[5, 6], outputs=[completion]) for _ in messages]


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = []
        self.state = m.NativeTransactions(bar, self.events.append)

    def item(self, root=None, validator=lambda value: value):
        return {"root": root or self.root / "attempt", "request": request(),
                "validator": validator, "binding": {"source": "fixture"}}

    def direct(self, item, optimized, model=None):
        function = self.state.transaction if optimized else core.transaction
        return function(item["root"], item["request"], model, None, item["validator"], item["binding"])

    def files(self, root):
        return {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".lock"}

    def make_candidate(self, root=None):
        item = self.item(root)
        self.direct(item, False, FakeModel())
        return item

    def test_original_bytecode_is_reused(self):
        for function in (core.transaction, core.validate_generation, core.save_json, bar.transactions):
            self.assertIs(m.clone(function).__code__, function.__code__)

    def test_fake_chat_original_new_bytes_and_flags_equal(self):
        first, second = self.item(self.root / "old"), self.item(self.root / "new")
        old_model, new_model = FakeModel(), FakeModel()
        self.assertEqual(self.direct(first, False, old_model), self.direct(second, True, new_model))
        self.assertEqual(self.files(first["root"]), self.files(second["root"]))
        self.assertEqual(old_model.calls, new_model.calls)
        final = core.strict_json((second["root"] / "final.json").read_bytes())
        for flag in core.FLAGS:
            self.assertIs(final[flag], False)
        event = self.events[-1]
        self.assertEqual(event["artifact_reads"], 0)
        self.assertGreaterEqual(event["sha_reuses"], 8)

    def test_existing_receipt_full_gate_wrong_SHA_no_dispatch(self):
        item = self.make_candidate()
        receipt = core.strict_json((item["root"] / "receipt.json").read_bytes())
        receipt["files_sha256"]["response.raw.txt"] = "0" * 64
        (item["root"] / "receipt.json").write_bytes(bar.batch.json_bytes(receipt))
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "resume artifact SHA"):
            self.direct(item, True, model)
        self.assertEqual(model.calls, [])
        self.assertTrue(self.events[-1]["receipt_resume_full_validation"])
        self.assertEqual(self.events[-1]["cache_hits"], 0)

    def test_existing_receipt_rechecks_validator_and_final(self):
        item = self.make_candidate()
        item["validator"] = lambda value: {"changed": True}
        with self.assertRaisesRegex(ValueError, "resume final differs"):
            self.direct(item, True, FakeModel())

    def test_missing_receipt_replay_generation_final_receipt_bytes(self):
        source = self.make_candidate()
        original = self.files(source["root"])
        replay = self.item(self.root / "replay")
        replay["root"].mkdir()
        for name in ("request.json", "binding.json", "response.raw.txt", "generation.json"):
            (replay["root"] / name).write_bytes(original[name])
        model = FakeModel()
        self.direct(replay, True, model)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.files(replay["root"]), original)
        self.assertEqual(self.events[-1]["artifact_reads"], 2)

    def test_raw_orphan_fails_and_never_retries(self):
        item = self.item()
        item["root"].mkdir()
        (item["root"] / "response.raw.txt").write_text('{"value":1}')
        model = FakeModel()
        for _ in range(2):
            self.assertEqual(self.direct(item, True, model)["status"], "failed")
        self.assertEqual(model.calls, [])

    def test_generation_wrong_hash_or_abnormal_EOS_fails_without_retry(self):
        for field, value in (("raw_response_sha256", "0" * 64), ("finish_reason", "length"),
                             ("max_tokens", 9), ("completion_tokens", 64), ("model_call_performed", False)):
            with self.subTest(field=field):
                item = self.make_candidate(self.root / field)
                (item["root"] / "receipt.json").unlink()
                (item["root"] / "final.json").unlink()
                generation = core.strict_json((item["root"] / "generation.json").read_bytes())
                generation[field] = value
                (item["root"] / "generation.json").write_bytes(bar.batch.json_bytes(generation))
                model = FakeModel()
                self.assertEqual(self.direct(item, True, model)["status"], "failed")
                self.assertEqual(model.calls, [])

    def test_changed_artifact_rejected_before_commit(self):
        item = self.item()
        def validator(value):
            (item["root"] / "response.raw.txt").write_text('{"value":900000}')
            return value
        item["validator"] = validator
        with self.assertRaisesRegex(ValueError, "changed before result commit"):
            self.direct(item, True, FakeModel())
        self.assertFalse((item["root"] / "receipt.json").exists())

    def test_unknown_other_path_and_wrong_expected_SHA_not_reused(self):
        cache = m.TransactionCache(self.root, core.strict_json)
        cache.active = True
        (self.root / "request.json").write_bytes(b"{}")
        other = self.root / "other" / "request.json"
        other.parent.mkdir()
        other.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "expected SHA"):
            cache.capture(self.root / "request.json", "0" * 64)
        with self.assertRaisesRegex(ValueError, "outside active LOCK"):
            cache.capture(other)
        with self.assertRaisesRegex(ValueError, "outside active LOCK"):
            cache.capture(self.root / "unknown.json")
        cache.clear()
        with self.assertRaisesRegex(ValueError, "outside active LOCK"):
            cache.capture(self.root / "request.json")

    def test_strict_json_duplicate_and_mutable_returns(self):
        cache = m.TransactionCache(self.root, core.strict_json)
        cache.active = True
        path = self.root / "generation.json"
        path.write_bytes(b'{"x":1,"x":2}')
        raw, _ = cache.capture(path)
        with self.assertRaises(ValueError):
            cache.parse(raw)
        cache.clear()
        cache.active = True
        path.write_bytes(b'{"x":[1]}')
        raw, _ = cache.capture(path)
        cache.parse(raw)["x"].append(2)
        self.assertEqual(cache.parse(raw), {"x": [1]})

    def test_text_newline_semantics(self):
        cache = m.TransactionCache(self.root, core.strict_json)
        cache.active = True
        path = self.root / "response.raw.txt"
        path.write_bytes(b"a\r\nb\rc\n")
        self.assertEqual(m.CachedPath(path, cache).read_text(encoding="utf-8"), path.read_text(encoding="utf-8"))

    def test_source_identity_callback_prevents_final_and_receipt(self):
        def changed():
            raise ValueError("source artifact identity changed")
        self.state.source_boundary = changed
        item = self.item()
        with self.assertRaisesRegex(ValueError, "source artifact identity"):
            self.direct(item, True, FakeModel())
        self.assertFalse((item["root"] / "receipt.json").exists())

    def test_batched_fake_model_timing_calls_and_output_equal(self):
        old_items = [self.item(self.root / f"old{i}") for i in range(3)]
        new_items = [self.item(self.root / f"new{i}") for i in range(3)]
        old_model, new_model = FakeModel(delay=.02), FakeModel(delay=.02)
        args = SimpleNamespace(stage="run", batch_size=2)
        sampling = lambda cls, schema, budget: {"schema": schema, "max_tokens": budget}
        def slow_save(path, raw):
            time.sleep(.002)
            return original_save(path, raw)
        original_save = core.save_once
        with patch.object(core, "structured_sampling", sampling):
            old = bar.transactions(old_items, lambda: (old_model, None), args)
            with patch.object(core, "save_once", slow_save), m.transaction_cache(bar, self.events.append):
                new = bar.transactions(new_items, lambda: (new_model, None), args)
        self.assertEqual(list(old.values()), list(new.values()))
        self.assertEqual(old_model.calls, new_model.calls)
        self.assertEqual([len(c[0]) for c in new_model.calls], [2, 1])
        for first, second in zip(old_items, new_items):
            self.assertEqual(self.files(first["root"]), self.files(second["root"]))
        timing = self.events[-1]
        self.assertEqual(timing["event"], "native_transactions_timing")
        self.assertGreater(timing["chatwall"], .035)
        self.assertGreater(timing["outputpersist"], .02)
        self.assertGreater(timing["transactionwall"], timing["chatwall"] + timing["outputpersist"])
        for phase in m.PHASES:
            self.assertGreater(timing[phase], 0)
        self.assertFalse(any("timing" in core.strict_json((i["root"] / "generation.json").read_bytes()) for i in new_items))

    def test_install_exception_restore_and_nested_reject(self):
        original = bar.transactions
        with self.assertRaisesRegex(RuntimeError, "runtime fixture failure"):
            with m.transaction_cache(bar, self.events.append):
                with self.assertRaisesRegex(ValueError, "reentry"):
                    with m.transaction_cache(bar, self.events.append):
                        pass
                def fail():
                    raise RuntimeError("runtime fixture failure")
                bar.transactions([self.item()], fail, SimpleNamespace(stage="run", batch_size=1))
        self.assertIs(bar.transactions, original)
        self.assertFalse(bar._transaction_cache_v1_active)

    def test_dispatch_exception_does_not_duplicate_dispatch_on_resume(self):
        model = FakeModel(failure=True)
        args = SimpleNamespace(stage="run", batch_size=2)
        with patch.object(core, "structured_sampling", lambda *a: None), m.transaction_cache(bar, self.events.append):
            for _ in range(2):
                result = bar.transactions([self.item()], lambda: (model, None), args)
                self.assertEqual(next(iter(result.values()))["status"], "failed")
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
