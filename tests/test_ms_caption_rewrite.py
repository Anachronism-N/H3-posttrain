"""CPU-only contracts and mock LLM completions; not a GPU validation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_caption_rewrite as rewrite


def row(strategy="fixed2s", statuses=("candidate", "candidate")):
    windows = []
    for i, status in enumerate(statuses):
        wid = f"w{i:04d}"
        candidate = status == "candidate"
        body = "Both arms rise. Both arms lower."
        windows.append({"window_id": wid, "window_start": i * 2., "window_end": (i + 1) * 2.,
            "status": "candidate" if candidate else "abstain",
            "abstain_reason": None if candidate else "partial_support" if status == "partial" else "no_output",
            "support_status": "partial" if status == "partial" else "complete_available_samples",
            "selected_sample_count": 40,
            "cleaned_motion_text": f"[{i*2.:.6f}, {(i+1)*2.:.6f}) seconds: {body}" if candidate else "",
            "caption": body, "original_captions": [{"start": i * 2., "end": (i+1)*2.-.05,
                "ms": ["Both arms rise. ", "Both arms lower."], "ms_nonagg": ["Both arms rise. ", "Both arms lower."]}],
            "partial_supported_motion_text": "Original partial bytes\n" if status == "partial" else "",
            "facts": [{"id":f"{wid}.s0000.ms{j:04d}","window_id":wid,"text":s,
                       "source":"original_motionscript_aggregated"}
                      for j,s in enumerate(["Both arms rise. ", "Both arms lower."])] if candidate else []})
    return {"schema": rewrite.SCHEMA, "id": "case1", "strategy": strategy,
            "manifest_record": {"id": "case1", "source_group": "same_source", "split": "test",
                "video_path": "/inputs/video.mp4", "audio_path": "/inputs/audio.wav", "nlf_path": "/inputs/nlf.npz"},
            "source_refs": {"report": {"path": "/inputs/report.json", "sha256": "a" * 64}},
            "binding": {"input_sha256": "b" * 64}, "windows": windows}


def valid_rewrite(source, facts):
    return {"windows": [{"window_id": w["window_id"],
                         "caption": rewrite.original_caption(w) if any(f["window_id"] == w["window_id"] for f in facts) else "",
                         "used_fact_ids": [f["fact_id"] for f in facts if f["window_id"] == w["window_id"]],
                         "used_boundary_ids": []} for w in source["windows"]], "issues": []}


class FakeLLM:
    def __init__(self, value, finish="stop", tokens=None):
        self.text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        self.finish = finish
        self.tokens = [7, 8] if tokens is None else tokens
        self.calls = []

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append((copy.deepcopy(messages), sampling))
        completion = SimpleNamespace(text=self.text, finish_reason=self.finish,
                                     stop_reason=None, token_ids=self.tokens)
        return [SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3])]

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)


class CaptionRewriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        sdk = patch.dict(sys.modules, {"vllm.sampling_params": SimpleNamespace(
            StructuredOutputsParams=lambda **kwargs: SimpleNamespace(**kwargs))})
        sdk.start()
        self.addCleanup(sdk.stop)

    def test_structured_sampling_preserves_free_caption_and_shared_schema(self):
        sampling = rewrite.structured_sampling(lambda **kw: SimpleNamespace(**kw), rewrite.JSON_SCHEMAS["rewrite"], 4096)
        self.assertEqual(sampling.temperature, 0.)
        self.assertEqual(sampling.seed, 0)
        self.assertEqual(sampling.max_tokens, 4096)
        self.assertEqual(sampling.structured_outputs.json, rewrite.JSON_SCHEMAS["rewrite"])
        caption = sampling.structured_outputs.json["properties"]["windows"]["items"]["properties"]["caption"]
        self.assertEqual(caption, {"type": "string"})
        source = row()
        supplied = rewrite.rewrite_input(source, rewrite.facts_for(source))
        self.assertEqual(rewrite.rewrite_request("/local", supplied)["json_schema"],
                         rewrite.rewrite_request("/local", supplied, [{"relation": "unknown"}])["json_schema"])

    def test_structured_generation_binding_rejects_schema_change(self):
        source = row()
        facts = rewrite.facts_for(source)
        request = rewrite.rewrite_request("/local", rewrite.rewrite_input(source, facts))
        rewrite.transaction(self.path, request, FakeLLM(valid_rewrite(source, facts)), object(),
                            lambda v: rewrite.check_rewrite(v, source, facts), {})
        generation_path = self.path / "generation.json"
        generation = json.loads(generation_path.read_text())
        self.assertEqual(generation["structured_output_schema_sha256"], rewrite.digest(request["json_schema"]))
        generation["structured_output_schema_sha256"] = "wrong"
        generation_path.write_text(json.dumps(generation))
        with self.assertRaises(ValueError):
            rewrite.validate_generation(self.path, request)

    def test_neutral_subject_and_no_synthetic_seconds(self):
        source = row()
        facts = rewrite.facts_for(source)
        for caption in ("She raises his arms.", "The dancer bends a second later."):
            output = valid_rewrite(source, facts)
            output["windows"][0]["caption"] = caption
            with self.assertRaises(ValueError):
                rewrite.check_rewrite(output, source, facts)
        rewrite.check_rewrite(valid_rewrite(source, facts), source, facts)

    def test_exact_window_schema_bounds_order_ids_and_local_refs(self):
        source = row(statuses=("candidate", "no_output", "partial"))
        supplied = rewrite.rewrite_input(source, rewrite.facts_for(source))
        schema = rewrite.rewrite_schema(supplied)
        windows = schema["properties"]["windows"]
        self.assertEqual((windows["minItems"], windows["maxItems"]), (3, 3))
        self.assertIs(windows["items"], False)
        self.assertEqual([p["properties"]["window_id"]["const"] for p in windows["prefixItems"]],
                         ["w0000", "w0001", "w0002"])
        first = windows["prefixItems"][0]["properties"]
        self.assertEqual(first["used_fact_ids"]["items"]["enum"],
                         [f["fact_id"] for f in supplied["facts"]])
        for item in windows["prefixItems"][1:]:
            self.assertEqual(item["properties"]["caption"]["const"], "")
            self.assertEqual(item["properties"]["used_fact_ids"]["maxItems"], 0)
            self.assertEqual(item["properties"]["used_boundary_ids"]["maxItems"], 0)

    def test_boundary_schema_uses_six_ordered_parts_and_actual_frame_refs(self):
        schema = rewrite.boundary_schema([26, 27, 28], [32, 33])
        relations = schema["properties"]["relations"]
        self.assertEqual(relations["minItems"], len(rewrite.PARTS))
        self.assertEqual([item["properties"]["part"]["const"] for item in relations["prefixItems"]],
                         list(rewrite.PARTS))
        for item in relations["prefixItems"]:
            self.assertEqual(item["properties"]["left_frame_indices"]["items"]["enum"], [26, 27, 28])
            self.assertEqual(item["properties"]["right_frame_indices"]["maxItems"], 2)

    def test_fact_normalization_keeps_raw_and_does_not_expose_raw_to_model(self):
        source = row("whole", ("candidate",))
        source["windows"][0]["facts"][0]["text"] = "She raises his right arm a second later."
        old = copy.deepcopy(source)
        facts = rewrite.facts_for(source)
        self.assertEqual(source, old)
        self.assertEqual(facts[0]["raw_text"], "She raises his right arm a second later.")
        self.assertEqual(facts[0]["text"], "the dancer raises the dancer's right arm afterward.")
        supplied = rewrite.rewrite_input(source, facts)
        self.assertTrue(all("raw_text" not in fact for fact in supplied["facts"]))
        self.assertEqual(supplied["facts"][0]["fact_id"], old["windows"][0]["facts"][0]["id"])

    def test_exact_four_strategies(self):
        self.assertEqual(rewrite.STRATEGIES, ("whole", "fixed2s", "bpm", "mergedbar"))
        for strategy in rewrite.STRATEGIES:
            source = row(strategy, ("candidate",) if strategy == "whole" else ("candidate", "candidate"))
            rewrite.validate_row(source)
        with self.assertRaises(ValueError):
            rewrite.validate_row(row("random"))
        with self.assertRaises(ValueError):
            rewrite.validate_row(row("whole"))

    def test_preserves_original_windows_and_bytes(self):
        source = row(statuses=("candidate", "partial", "no_output"))
        old = copy.deepcopy(source)
        facts = rewrite.facts_for(source)
        baseline = rewrite.r0(source)
        self.assertEqual(source, old)
        self.assertEqual(baseline["json"]["windows"][0]["caption"], source["windows"][0]["caption"])
        self.assertEqual("".join(f["text"] for f in facts), source["windows"][0]["caption"])
        self.assertEqual(baseline["partial_sources"][0]["original"], source["windows"][1])
        self.assertNotIn("partial", baseline["timed_caption"].lower())
        self.assertEqual(len(baseline["json"]["windows"]), 3)
        self.assertEqual([w["status"] for w in source["windows"]], ["candidate", "abstain", "abstain"])

    def test_no_facts_no_stillness_or_partial_upgrade(self):
        source = row(statuses=("no_output", "partial"))
        facts = rewrite.facts_for(source)
        self.assertEqual(facts, [])
        output = valid_rewrite(source, facts)
        rewrite.check_rewrite(output, source, facts)
        output["windows"][0]["caption"] = "The dancer remains still."
        with self.assertRaises(ValueError):
            rewrite.check_rewrite(output, source, facts)
        self.assertEqual(rewrite.timed_text(source, valid_rewrite(source, facts)["windows"]), "")

    def test_identical_repeated_actions_survive_each_window(self):
        source = row()
        facts = rewrite.facts_for(source)
        self.assertEqual(len(facts), 4)
        self.assertEqual(facts[0]["text"], facts[2]["text"])
        self.assertNotEqual(facts[0]["fact_id"], facts[2]["fact_id"])
        result = valid_rewrite(source, facts)
        rewrite.check_rewrite(result, source, facts)
        result["windows"][1]["used_fact_ids"] = []
        result["windows"][1]["caption"] = ""
        with self.assertRaisesRegex(ValueError, "omitted facts"):
            rewrite.check_rewrite(result, source, facts)

    def test_fact_refs_conflicts_and_window_order(self):
        source = row()
        facts = rewrite.facts_for(source)
        output = valid_rewrite(source, facts)
        output["windows"][0]["used_fact_ids"] = [facts[-1]["fact_id"]]
        with self.assertRaisesRegex(ValueError, "another window"):
            rewrite.check_rewrite(output, source, facts)
        output = valid_rewrite(source, facts)
        missing = output["windows"][0]["used_fact_ids"].pop()
        output["issues"] = [f"Conflict involving {missing}; not resolved."]
        rewrite.check_rewrite(output, source, facts)
        output["windows"].reverse()
        with self.assertRaises(ValueError):
            rewrite.check_rewrite(output, source, facts)

    def test_r1_r2_identical_S_prompt_sampling_only_extra_relations(self):
        source = row()
        supplied = rewrite.rewrite_input(source, rewrite.facts_for(source))
        before = copy.deepcopy(supplied)
        relations = [{"boundary_id": "b:torso", "relation": "repeat"}]
        r1 = rewrite.rewrite_request("/models/local", supplied)
        r2 = rewrite.rewrite_request("/models/local", supplied, relations)
        self.assertEqual(r1["messages"][0], r2["messages"][0])
        self.assertEqual(r1["sampling"], r2["sampling"])
        r2_input = json.loads(r2["messages"][1]["content"])
        self.assertEqual(r2_input.pop("boundary_relations"), relations)
        self.assertEqual(r2_input, json.loads(r1["messages"][1]["content"]))
        self.assertEqual(before, supplied)

    def test_boundary_unknown_and_nonadjacent_refs_rejected(self):
        source = row(statuses=("candidate", "candidate", "candidate"))
        facts = rewrite.facts_for(source)
        relation = {"boundary_id": "b:torso", "relation": "repeat",
                    "left_window_id": "w0000", "right_window_id": "w0001"}
        output = valid_rewrite(source, facts)
        output["windows"][1]["used_boundary_ids"] = ["b:torso"]
        rewrite.check_rewrite(output, source, facts, [relation])
        with self.assertRaises(ValueError):
            rewrite.check_rewrite(output, source, facts)
        with self.assertRaises(ValueError):
            rewrite.check_rewrite(output, source, facts, [{**relation, "relation": "unknown"}])
        output["windows"][2]["used_boundary_ids"] = ["b:torso"]
        with self.assertRaises(ValueError):
            rewrite.check_rewrite(output, source, facts, [relation])

    def test_boundary_real_pts_budget_and_two_each_side(self):
        probe = {"times": [i / 30 for i in range(120)]}
        left, right = rewrite.boundary_indices(probe, 2.)
        self.assertLessEqual(len(left) + len(right), 16)
        self.assertGreaterEqual(len(left), 2)
        self.assertGreaterEqual(len(right), 2)
        self.assertTrue(all(1.6 <= probe["times"][i] < 2 for i in left))
        self.assertTrue(all(2 <= probe["times"][i] < 2.4 for i in right))
        value = {"relations": [{"part": part, "relation": "continue", "left_frame_indices": left[:2],
                                "right_frame_indices": right[:2]} for part in rewrite.PARTS], "issues": []}
        checked = rewrite.check_boundary(value, left, right, {"id": "b", "left_window_id": "w0000", "right_window_id": "w0001"})
        self.assertTrue(all(r["ai_candidate"] for r in checked["relations"]))
        value["relations"][0]["left_frame_indices"] = left[:1]
        with self.assertRaises(ValueError):
            rewrite.check_boundary(value, left, right, {"id": "b"})

    def test_supplement_only_genuine_empty_and_enters_both_S(self):
        source = row(statuses=("no_output", "partial", "candidate"))
        event = {"abstain": False, "events": [{"id": "e1", "description": "Arms move outward.", "frame_indices": [1, 2]}]}
        supplements = {w["window_id"]: event for w in source["windows"]}
        facts = rewrite.facts_for(source, supplements)
        visual = [f for f in facts if f["source"] == "visual_supplement"]
        self.assertEqual(len(visual), 1)
        self.assertEqual(visual[0]["window_id"], "w0000")
        self.assertTrue(visual[0]["ai_weak_candidate"])
        self.assertEqual(rewrite.r0(source)["json"]["windows"][0]["caption"], "")
        args = SimpleNamespace(model="/models/local")
        with patch.object(rewrite.batch, "prepare_window") as prepare:
            self.assertIsNone(rewrite.prepare_supplement(self.path, {}, {}, {"times": [0, 1]}, source["windows"][1], args, {}))
            result = rewrite.prepare_supplement(self.path, {}, {}, {"times": [.1]}, source["windows"][0], args, {})
            self.assertEqual(result["status"], "insufficient_real_frames")
            prepare.assert_not_called()

    def test_raw_utf8_finish_tokens_and_resume_no_call(self):
        source = row()
        facts = rewrite.facts_for(source)
        value = valid_rewrite(source, facts)
        raw = "  " + json.dumps(value, ensure_ascii=False) + "\n"
        llm = FakeLLM(raw)
        request = rewrite.rewrite_request("/models/local", rewrite.rewrite_input(source, facts))
        validator = lambda value: rewrite.check_rewrite(value, source, facts)
        final = rewrite.transaction(self.path, request, llm, object(), validator, {"code": "sha"})
        self.assertEqual(final["status"], "candidate")
        self.assertEqual((self.path / "response.raw.txt").read_bytes(), raw.encode("utf-8"))
        resumed = rewrite.transaction(self.path, request, llm, object(), validator, {"code": "sha"})
        self.assertEqual(resumed, final)
        self.assertEqual(len(llm.calls), 1)
        generation = json.loads((self.path / "generation.json").read_text())
        self.assertFalse(generation["last_output_token_is_eos"])
        self.assertEqual(generation["finish_reason"], "stop")
        with self.assertRaises(ValueError):
            rewrite.transaction(self.path, request, llm, object(), validator, {"code": "changed"})

    def test_truncation_failure_preserves_raw_no_retry(self):
        source = row()
        facts = rewrite.facts_for(source)
        request = rewrite.rewrite_request("/models/local", rewrite.rewrite_input(source, facts), max_tokens=2048)
        for name, finish, tokens in (("length", "length", [7]), ("budget", "stop", [7] * 2048)):
            root = self.path / name
            llm = FakeLLM(valid_rewrite(source, facts), finish=finish, tokens=tokens)
            validator = lambda value: rewrite.check_rewrite(value, source, facts)
            final = rewrite.transaction(root, request, llm, object(), validator, {})
            self.assertEqual(final["status"], "failed")
            self.assertTrue((root / "response.raw.txt").is_file())
            rewrite.transaction(root, request, llm, object(), validator, {})
            self.assertEqual(len(llm.calls), 1)

    def test_malformed_json_no_gpu_repair_loop(self):
        llm = FakeLLM("not json\n")
        request = rewrite.rewrite_request("/models/local", {"windows": [], "facts": []})
        final = rewrite.transaction(self.path, request, llm, object(), lambda x: x, {})
        self.assertEqual(final["status"], "failed")
        self.assertEqual((self.path / "response.raw.txt").read_bytes(), b"not json\n")
        rewrite.transaction(self.path, request, llm, object(), lambda x: x, {})
        self.assertEqual(len(llm.calls), 1)

    def test_orphan_raw_never_resent(self):
        request = rewrite.rewrite_request("/models/local", {"windows": [], "facts": []})
        rewrite.save_once(self.path / "response.raw.txt", b"{}")
        llm = FakeLLM({})
        final = rewrite.transaction(self.path, request, llm, object(), lambda x: x, {})
        self.assertEqual(final["status"], "failed")
        self.assertEqual(len(llm.calls), 0)
        self.assertIn("generation.json", final["error"])

    def test_resume_detects_raw_and_request_tampering(self):
        request = rewrite.rewrite_request("/models/local", {"windows": [], "facts": []})
        llm = FakeLLM({"windows": [], "issues": []})
        rewrite.transaction(self.path, request, llm, object(), lambda x: x, {})
        changed = copy.deepcopy(request)
        changed["sampling"]["seed"] = 5
        with self.assertRaises(ValueError):
            rewrite.transaction(self.path, changed, llm, object(), lambda x: x, {})
        (self.path / "response.raw.txt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            rewrite.transaction(self.path, request, llm, object(), lambda x: x, {})
        self.assertEqual(len(llm.calls), 1)

    def test_whole_r2_reuses_one_model_call_and_no_boundaries(self):
        source = row("whole", ("candidate",))
        facts = rewrite.facts_for(source)
        llm = FakeLLM(valid_rewrite(source, facts))
        args = SimpleNamespace(stage="run", supplement_empty=False, model="/models/local", rewrite_tokens=4096)
        runtime = lambda: (llm, lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.object(rewrite.batch, "verify_source"), patch.object(rewrite, "prepare_boundary") as boundary:
            final = rewrite.process_row(source, self.path, {"sha256": "source"}, [], {}, args, {"code_sha256": "code"}, runtime)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(final["R2"]["reused_from"], "R1")
        self.assertFalse(final["R2"]["duplicate_model_call_performed"])
        self.assertEqual(final["R1"]["json"], final["R2"]["json"])
        self.assertEqual(final["original_windows"], source["windows"])
        boundary.assert_not_called()

    def test_plan_limit_means_three_ids_all_strategies(self):
        rows = []
        for index in range(4):
            for strategy in rewrite.STRATEGIES:
                source = row(strategy, ("candidate",) if strategy == "whole" else ("candidate", "candidate"))
                source["id"] = source["manifest_record"]["id"] = f"case{index}"
                rows.append(source)
        plan = self.path / "plan.jsonl"
        plan.write_text("\n".join(json.dumps(r) for r in rows))
        selected, _ = rewrite.read_plan(plan, limit=3)
        self.assertEqual(len(selected), 12)
        selected, _ = rewrite.read_plan(plan, strategy="bpm", limit=3)
        self.assertEqual(len(selected), 3)

    def test_prepare_never_loads_or_imports_vllm(self):
        source = row("whole", ("candidate",))
        args = SimpleNamespace(stage="prepare", supplement_empty=False, model="/models/local", rewrite_tokens=4096)
        def runtime():
            self.fail("prepare must not load a model")
        with patch.object(rewrite.batch, "verify_source"):
            final = rewrite.process_row(source, self.path, {"sha256": "source"}, [], {}, args, {}, runtime)
        self.assertEqual(final["R1"]["status"], "prepared")
        self.assertNotIn("vllm", sys.modules)
        self.assertTrue((self.path / "R1" / "prepare" / "request.json").is_file())

    def test_real_ms_fields_abstain_reasons_declared_facts(self):
        source = row(statuses=("candidate", "partial", "no_output"))
        source["windows"][0]["facts"] = [{"id": "w0000:orig:1", "window_id": "w0000",
                                          "text": "Both arms rise. Both arms lower.", "source": "original_ms"}]
        source["windows"][1]["abstain_reason"] = "partial_support"
        source["windows"][2]["abstain_reason"] = "no_output"
        original = copy.deepcopy(source)
        rewrite.validate_row(source)
        self.assertEqual(rewrite.window_bounds(source["windows"][0]), (0., 2.))
        facts = rewrite.facts_for(source)
        self.assertEqual(facts[0]["text"], source["windows"][0]["facts"][0]["text"])
        self.assertEqual(facts[0]["fact_id"], "w0000:orig:1")
        self.assertEqual(rewrite.source_status(source["windows"][1]), "partial")
        self.assertEqual(rewrite.source_status(source["windows"][2]), "no_output")
        baseline = rewrite.r0(source)
        self.assertEqual(len(baseline["partial_sources"]), 1)
        self.assertEqual(baseline["json"]["windows"][1]["caption"], "")
        self.assertEqual(source, original)
        output = valid_rewrite(source, facts)
        rewrite.check_rewrite(output, source, facts)

    def test_whole_resume_does_not_load_model(self):
        source = row("whole", ("candidate",))
        facts = rewrite.facts_for(source)
        llm = FakeLLM(valid_rewrite(source, facts))
        args = SimpleNamespace(stage="run", supplement_empty=False, model="/models/local", rewrite_tokens=4096)
        def unavailable():
            self.fail("resume must not load model")
        with patch.object(rewrite.batch, "verify_source"):
            rewrite.process_row(source, self.path, {"sha256": "source"}, [], {}, args, {},
                                lambda: (llm, lambda **kw: SimpleNamespace(**kw)))
            rewrite.process_row(source, self.path, {"sha256": "source"}, [], {}, args, {}, unavailable)
        self.assertEqual(len(llm.calls), 1)

    def test_summary_unapproved_separate_counts(self):
        results = [{"strategy": "whole", "source_group": "src", "R1": {"status": "candidate"},
                    "R2": {"status": "failed"}, "visual_supplement": {"w": {"status": "abstain"}}, "partial_count": 2}]
        summary = rewrite.summary(results, "run")
        self.assertEqual(summary["counts"]["strategy:whole"]["partial"], 2)
        self.assertEqual(summary["counts"]["source:src"]["R2"], {"failed": 1})
        self.assertEqual(summary["counts"]["all"]["empty_supplement"], {"abstain": 1})
        for key in rewrite.FLAGS:
            self.assertIs(summary[key], False)


if __name__ == "__main__":
    unittest.main()
