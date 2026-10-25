"""CPU-only native transactions with synthetic evidence; no vLLM/model loads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_bar_annotation as bar
import ms_caption_acceleration as acceleration


def cpu_sampling(sampling_cls, schema, max_tokens):
    return sampling_cls(temperature=0.0, seed=0, max_tokens=max_tokens,
                        structured_outputs=copy.deepcopy(schema))


class FakeLLM:
    _ms_bar_model_sha256 = "e" * 64

    def __init__(self, mode="valid"):
        self.mode, self.calls, self.sampling = mode, [], []

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append(copy.deepcopy(messages))
        self.sampling.append(copy.deepcopy(sampling))
        if self.mode == "exception":
            raise RuntimeError("CPU dispatch failure")
        outputs = []
        for request in messages:
            header = json.loads(request[1]["content"][0]["text"])
            indices = header["allowed_source_frame_indices"][:2]
            value = bar.empty_observation("unknown")
            value["part_support"][bar.PARTS.index("left_hand")] = {
                "part": "left_hand", "frame_indices": indices, "reason": "visible"}
            value.update(events=[{"id": "e1", "description": "The dancer opens the left hand.",
                                  "parts": ["left_hand"], "frame_indices": indices}], abstain=False)
            if self.mode == "unsupported":
                value["events"][0]["frame_indices"] = [9998, 9999]
            completion = SimpleNamespace(
                text="not JSON" if self.mode == "invalid" else json.dumps(value),
                token_ids=[7, 8], finish_reason="length" if self.mode == "truncated" else "stop",
                stop_reason=None)
            outputs.append(SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3]))
        return outputs

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)


class AccelerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(stage="run", batch_size=2, max_num_seqs=8,
                                    model=str(self.root / "cpu-checkpoint"), visual_max_tokens=1800,
                                    max_pixels=448 * 448, tp=2, custom={"unchanged": True})
        self.sampling_patch = patch.object(bar.core, "structured_sampling", side_effect=cpu_sampling)
        self.sampling_patch.start()
        self.addCleanup(self.sampling_patch.stop)

    def source_items(self, output):
        sources = {}
        for sid in ("source_c", "source_a", "source_b"):
            sources[sid] = []
            for wi in range(3):
                window = {"window_id": f"w{wi}", "start": float(wi), "end": float(wi + 1)}
                frames = []
                for fi in range(wi * 8, wi * 8 + 8):
                    file = Path("cpu-evidence") / sid / f"frame-{fi}.jpg"
                    image = self.root / file
                    image.parent.mkdir(parents=True, exist_ok=True)
                    # Synthetic bytes exercise native JPEG base64 preservation, not pixel quality.
                    image.write_bytes(f"CPU synthetic JPEG {sid} {fi}".encode())
                    frames.append({"index": fi, "time": fi / 8., "source_timestamp": fi / 8.,
                                   "file": str(file), "sha256": bar.batch.sha256(image)})
                manifest = {"window": window, "frames": frames}
                request = bar.visual_request(self.args, self.root, manifest, {})
                sources[sid].append({"root": self.root / output / sid / window["window_id"] / "attempt-0",
                    "request": request, "validator": lambda value, m=manifest: bar.filter_supported_observation(value, m, {}),
                    "binding": {"source_binding": {"id": sid, "video_sha256": "a" * 64},
                                "window": window, "frames_sha256": bar.batch.digest(frames)}})
        return sources

    def direct_inputs(self):
        windows = [{"window_id": "w2", "start": 0., "end": 1.},
                   {"window_id": "w0", "start": 1., "end": 2.},
                   {"window_id": "w1", "start": 2., "end": 3.}]
        texts = [("w2", "w2:e2", "The dancer opens the left hand, spreading the fingers."),
                 ("w1", "w1:e1", "The dancer closes the right hand into a fist."),
                 ("w2", "w2:e1", "The dancer turns both hands, holding a V-sign with the fingers.")]
        facts = [{"window_id": wid, "fact_id": fid, "text": text, "raw_text": text,
                  "parts": ["left_hand", "right_hand"], "frame_indices": [0, 1]} for wid, fid, text in texts]
        boundaries = [{"boundary_id": "b-w2-w0:head", "relation": "unknown",
                       "left_window_id": "w2", "right_window_id": "w0"}]
        return windows, facts, boundaries

    def test_direct_preserves_every_text_id_hand_detail_and_order(self):
        windows, facts, boundaries = self.direct_inputs()
        before = copy.deepcopy((windows, facts, boundaries))
        with patch.object(bar, "check_supported_rewrite", wraps=bar.check_supported_rewrite) as validator:
            result = acceleration.direct_caption(windows, facts, boundaries)
        validator.assert_called_once_with(result, windows, facts, boundaries)
        self.assertEqual([w["window_id"] for w in result["windows"]], ["w2", "w0", "w1"])
        self.assertEqual(result["windows"][0]["caption"], facts[0]["text"] + " " + facts[2]["text"])
        self.assertEqual(result["windows"][0]["used_fact_ids"], ["w2:e2", "w2:e1"])
        self.assertEqual(result["windows"][2]["caption"], facts[1]["text"])
        self.assertTrue(all(w["used_boundary_ids"] == [] for w in result["windows"]))
        self.assertEqual((windows, facts, boundaries), before)

    def test_direct_empty_is_not_stillness_or_filler(self):
        windows, facts, boundaries = self.direct_inputs()
        mixed = acceleration.direct_caption(windows, facts, boundaries)
        self.assertEqual(mixed["windows"][1], {"window_id": "w0", "caption": "",
                         "used_fact_ids": [], "used_boundary_ids": []})
        empty = acceleration.direct_final(windows, [], boundaries)
        self.assertEqual(empty["status"], "abstain")
        self.assertTrue(all(w["caption"] == "" for w in empty["json"]["windows"]))

    def test_direct_final_is_new_unreviewed_derivation(self):
        final = acceleration.direct_final(*self.direct_inputs())
        self.assertEqual(final["status"], "candidate")
        self.assertEqual(final["derivation"], acceleration.DIRECT_RECIPE)
        for flag in (*bar.FLAGS, "training_eligible", "semantic_validation_completed",
                     "model_call_performed", "model_rewrite_performed"):
            self.assertIs(final[flag], False)

    def test_direct_duplicate_and_out_of_window_fact_rejected(self):
        for kind in ("windows", "facts", "wrong_window", "empty_text", "missing_raw"):
            with self.subTest(kind=kind):
                windows, facts, boundaries = self.direct_inputs()
                if kind == "windows":
                    windows.append(copy.deepcopy(windows[0]))
                elif kind == "facts":
                    facts.append(copy.deepcopy(facts[0]))
                elif kind == "wrong_window":
                    facts[0]["window_id"] = "absent"
                elif kind == "empty_text":
                    facts[0]["text"] = " "
                else:
                    del facts[0]["raw_text"]
                with self.assertRaises(ValueError):
                    acceleration.direct_caption(windows, facts, boundaries)

    def test_direct_full_validator_rejects_unsupported_parts_and_excluded_claims(self):
        for mode in ("parts", "excluded", "pronoun", "count"):
            with self.subTest(mode=mode):
                windows, facts, boundaries = self.direct_inputs()
                if mode == "parts":
                    for fact in facts:
                        fact["parts"] = ["torso"]
                elif mode == "excluded":
                    facts[0]["text"] = "The dancer's hair moves in the wind."
                elif mode == "pronoun":
                    facts[0]["text"] = "She opens the left hand."
                else:
                    facts[0]["text"] = "The dancer opens the left hand three times."
                final = acceleration.direct_final(windows, facts, boundaries)
                self.assertEqual(final["status"], "failed")
                self.assertIs(final["format_valid"], False)

    def test_batch2_batch8_real_calls_exact_native_messages_and_bytes(self):
        before = copy.deepcopy(vars(self.args))
        cases = []
        for size, output in ((2, "batch2"), (8, "batch8")):
            sources = self.source_items(output)
            llm = FakeLLM()
            flat = [item for items in sources.values() for item in items]
            with patch.object(bar, "transactions", wraps=bar.transactions) as transactions:
                grouped = acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args, size)
            transactions.assert_called_once()
            dispatched_args = transactions.call_args.args[2]
            self.assertIsNot(dispatched_args, self.args)
            self.assertEqual(vars(dispatched_args), {**before, "batch_size": size})
            self.assertEqual([messages for call in llm.calls for messages in call], [i["request"]["messages"] for i in flat])
            self.assertEqual([len(c) for c in llm.calls], [2, 2, 2, 2, 1] if size == 2 else [8, 1])
            self.assertEqual(list(grouped["by_source"]), list(sources))
            for sid, items in sources.items():
                self.assertEqual(list(grouped["by_source"][sid]), [str(i["root"]) for i in items])
                for item in items:
                    result = grouped["by_source"][sid][str(item["root"])]
                    self.assertIs(result, grouped["by_root"][str(item["root"])])
                    self.assertEqual(result["status"], "candidate")
                    receipt = bar.load(item["root"] / "receipt.json")
                    self.assertEqual(receipt["binding"], item["binding"])
                    self.assertEqual(receipt["binding"]["source_binding"]["id"], sid)
                    for name, digest in receipt["files_sha256"].items():
                        self.assertEqual(bar.batch.sha256(item["root"] / name), digest)
            for item, sampling in zip(flat, [s for call in llm.sampling for s in call]):
                self.assertEqual(vars(sampling), {"temperature": 0., "seed": 0, "max_tokens": 1800,
                                                 "structured_outputs": item["request"]["json_schema"]})
            prior_calls = len(llm.calls)
            self.assertEqual(acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args, size), grouped)
            self.assertEqual(len(llm.calls), prior_calls)
            cases.append(flat)
        for first, second in zip(*cases):
            self.assertEqual({p.name: p.read_bytes() for p in first["root"].iterdir() if p.is_file()},
                             {p.name: p.read_bytes() for p in second["root"].iterdir() if p.is_file()})
        self.assertEqual(vars(self.args), before)
        # The eight-item group genuinely crosses all three independent sources.
        self.assertEqual({i["binding"]["source_binding"]["id"] for i in cases[1][:8]},
                         {"source_a", "source_b", "source_c"})

    def test_wrapper_matches_unmodified_native_transactions_bytes(self):
        native = self.source_items("native")
        portable = self.source_items("portable")
        first_llm, second_llm = FakeLLM(), FakeLLM()
        copied = copy.copy(self.args)
        copied.batch_size = 8
        bar.transactions([i for items in native.values() for i in items], lambda: (first_llm, SimpleNamespace), copied)
        acceleration.cross_source_transactions(portable, lambda: (second_llm, SimpleNamespace), self.args)
        self.assertEqual(first_llm.calls, second_llm.calls)
        for first, second in zip([i for items in native.values() for i in items],
                                 [i for items in portable.values() for i in items]):
            for name in ("request.json", "binding.json", "response.raw.txt", "generation.json",
                         "response.json", "final.json", "receipt.json"):
                self.assertEqual((first["root"] / name).read_bytes(), (second["root"] / name).read_bytes())

    def test_invalid_source_root_window_stage_schema_fail_before_dispatch_or_write(self):
        for mode in ("source", "root", "alias_root", "window", "stage", "capacity", "batch", "boundary",
                     "rewrite", "dependent", "schema", "short", "unordered", "validator"):
            with self.subTest(mode=mode):
                sources = self.source_items("invalid-" + mode)
                first, second = sources["source_c"][:2]
                args = copy.copy(self.args)
                batch_size = 8
                if mode == "source":
                    second["binding"]["source_binding"]["id"] = "source_a"
                elif mode == "root":
                    second["root"] = first["root"]
                elif mode == "alias_root":
                    second["root"] = first["root"] / ".." / "attempt-0"
                elif mode == "window":
                    second["binding"]["window"] = first["binding"]["window"]
                elif mode == "stage":
                    args.stage = "export"
                elif mode == "capacity":
                    args.max_num_seqs = 2
                elif mode == "batch":
                    batch_size = True
                elif mode == "boundary":
                    second["request"]["messages"][0]["content"] = bar.core.BOUNDARY_SYSTEM
                elif mode == "rewrite":
                    second["request"]["json_schema"] = bar.core.JSON_SCHEMAS["rewrite"]
                elif mode == "dependent":
                    second["binding"]["boundary"] = {}
                elif mode == "schema":
                    second["request"]["json_schema"]["properties"]["events"]["maxItems"] = 33
                elif mode in ("short", "unordered"):
                    header = json.loads(second["request"]["messages"][1]["content"][0]["text"])
                    header["allowed_source_frame_indices"] = [8] if mode == "short" else [9, 8]
                    second["request"]["messages"][1]["content"][0]["text"] = json.dumps(header)
                else:
                    second["validator"] = None
                runtime = unittest.mock.Mock(side_effect=AssertionError("runtime must not load"))
                with self.assertRaises(ValueError), patch.object(bar, "transactions") as transaction:
                    acceleration.cross_source_transactions(sources, runtime, args, batch_size)
                runtime.assert_not_called()
                transaction.assert_not_called()
                self.assertFalse((self.root / ("invalid-" + mode)).exists())

    def test_prepare_never_loads_runtime(self):
        sources = self.source_items("prepare")
        args = copy.copy(self.args)
        args.stage = "prepare"
        runtime = unittest.mock.Mock(side_effect=AssertionError("CPU prepare cannot load"))
        result = acceleration.cross_source_transactions(sources, runtime, args)
        runtime.assert_not_called()
        self.assertTrue(all(r["status"] == "prepared" for r in result["by_root"].values()))
        self.assertFalse(any(p.name == "response.raw.txt" for p in (self.root / "prepare").rglob("*")))

    def test_dispatch_parse_and_truncation_failures_never_resubmit(self):
        for mode in ("exception", "invalid", "truncated"):
            with self.subTest(mode=mode):
                sources = self.source_items("failed-" + mode)
                llm = FakeLLM(mode)
                result = acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args)
                self.assertTrue(all(r["status"] == "failed" for r in result["by_root"].values()))
                self.assertEqual(len(llm.calls), 2)
                resumed = acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args)
                self.assertEqual(resumed, result)
                self.assertEqual(len(llm.calls), 2)

    def test_orphan_raw_is_failed_without_any_new_call(self):
        sources = self.source_items("orphan")
        first = sources["source_c"][0]
        sources = {"source_c": [first]}
        first["root"].mkdir(parents=True)
        (first["root"] / "response.raw.txt").write_bytes(b"orphan raw")
        runtime = unittest.mock.Mock(side_effect=AssertionError("orphan cannot be resubmitted"))
        result = acceleration.cross_source_transactions(sources, runtime, self.args)
        runtime.assert_not_called()
        self.assertEqual(result["by_root"][str(first["root"])]["status"], "failed")
        self.assertTrue((first["root"] / "receipt.json").is_file())

    def test_model_load_exception_is_not_retried_per_window(self):
        sources = self.source_items("load-failure")
        runtime = unittest.mock.Mock(side_effect=RuntimeError("CPU load failure"))
        with self.assertRaisesRegex(RuntimeError, "shared model load failed; not retrying per window"):
            acceleration.cross_source_transactions(sources, runtime, self.args)
        runtime.assert_called_once()

    def test_native_support_filter_still_rejects_unsupported_frames(self):
        sources = self.source_items("support")
        llm = FakeLLM("unsupported")
        grouped = acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args)
        for result in grouped["by_root"].values():
            self.assertEqual(result["status"], "candidate")
            self.assertTrue(result["json"]["abstain"])
            self.assertEqual(result["json"]["events"], [])
            self.assertTrue(result["json"]["support_filter_audit"]["rejected_events"])

    def test_native_receipt_corruption_cannot_be_bypassed(self):
        sources = self.source_items("tamper")
        llm = FakeLLM()
        acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args)
        (sources["source_c"][0]["root"] / "final.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "resume artifact SHA mismatch"):
            acceleration.cross_source_transactions(sources, lambda: (llm, SimpleNamespace), self.args)
        self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
