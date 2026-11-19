"""CPU schema/serialization tests; fake completions are not semantic approval."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if os.environ.get("MS_VISUAL_CPU_SITEPATH"):
    sys.path.append(os.environ["MS_VISUAL_CPU_SITEPATH"])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_dance_style as style


def rich(body):
    return "[1. Technical Execution.] ballet clothes [2. Choreographic Content & Movement Vocabulary.] " + body + " [3. Expressiveness.] hip-hop mood"


def row():
    return {"id": "fixture", "binding": {"video_sha256": "a" * 64}, "manifest_record": {"nlf_path": None}}


def args(stage="run"):
    return SimpleNamespace(model="/input/model", style_max_tokens=1024, stage=stage, batch_size=8)


def value():
    return {"dance_style": "hip-hop", "frame_indices": [8, 12], "reason": "Visible rhythmic body isolation.", "abstain": False}


class FakeLLM:
    def __init__(self, result=None, finish="stop"):
        self.calls = []
        self.result = result if result is not None else value()
        self.finish = finish

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append(copy.deepcopy(messages))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(self.result), token_ids=[3, 4],
            finish_reason=self.finish, stop_reason=None)], prompt_token_ids=[1, 2]) for _ in messages]

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)


class DanceStyleContracts(unittest.TestCase):
    def test_old_label_priority_and_simple_target(self):
        label = {"genre": "k-pop", "genre_conf": "high", "sub_style": "K-pop girl-group choreography", "mood": "playful"}
        p = style.prepare_style(row(), label, rich("Expressive choreography."), {"label": {"sha256": "b" * 64}}, "a" * 64)
        self.assertEqual((p["status"], p["dance_style"], p["source"]), ("resolved", "k-pop", "historical_RGB_label"))
        timed = "[0.000, 1.000] The dancer raises the arm.\n[1.000, 2.000] The dancer repeats the movement.  \n"
        out = style.export_joint(p, timed)
        self.assertEqual(out["text"], "Dance style: k-pop\n" + timed)
        self.assertNotIn("playful", out["text"])
        self.assertEqual(out["metadata"]["style"]["metadata"]["historical_label"], label)
        self.assertFalse(out["formal_training"])
        self.assertTrue(p["metadata"]["missing_NLF_does_not_block_style"])

    def test_historical_binding_is_required_and_wrong_video_is_not_reused(self):
        for sha in (None, "f" * 64):
            p = style.prepare_style(row(), {"genre": "ballet", "genre_conf": "high"}, historical_video_sha256=sha)
            self.assertEqual(p["status"], "needs_rgb")
            self.assertEqual(p["dance_style"], "unknown")
        with self.assertRaisesRegex(ValueError, "video SHA"):
            style.prepare_style({"id": "fixture"})

    def test_historical_path_only_reuse_keeps_current_and_historical_hash_distinct(self):
        path = "/fixture/normalized_704x1280/main/videos/fixture.mp4"
        r = {**row(), "video_path": path}
        refs = {"historical_label_record": {"sid": "fixture", "path": "/fixture/labels_s0.jsonl",
            "sha256": "b" * 64, "line": 1, "row_sha256": "c" * 64}}
        p = style.prepare_style(r, {"genre": "ballet", "genre_conf": "high"}, source_refs=refs,
                                historical_video_path=path)
        self.assertEqual((p["status"], p["dance_style"]), ("resolved", "ballet"))
        self.assertEqual(p["metadata"]["historical_video_binding"], "path_only_unverified")
        self.assertEqual(p["metadata"]["current_video_sha256"], "a" * 64)
        self.assertIsNone(p["metadata"]["historical_video_sha256"])
        self.assertFalse(p["metadata"]["historical_video_binding_verified"])
        self.assertEqual(p["metadata"]["source_refs"], refs)
        mismatch = style.prepare_style(r, {"genre": "ballet"}, source_refs=refs,
            historical_video_sha256="f" * 64, historical_video_path=path)
        self.assertEqual(mismatch["status"], "needs_rgb")
        self.assertEqual(mismatch["metadata"]["historical_video_binding"], "sha256_mismatch")

    def test_path_only_reuse_needs_label_file_row_and_correct_normalized_source_id(self):
        path = "/fixture/normalized_704x1280/main/videos/fixture.mp4"
        refs = {"historical_label_record": {"sid": "fixture", "path": "/fixture/labels_s0.jsonl",
            "sha256": "b" * 64, "line": 1, "row_sha256": "c" * 64}}
        for change in ({"sid": "other"}, {"row_sha256": None}, {"sha256": None}, {"line": False}, {"path": "relative.jsonl"}):
            bad = {"historical_label_record": {**refs["historical_label_record"], **change}}
            p = style.prepare_style({**row(), "video_path": path}, {"genre": "ballet"},
                source_refs=bad, historical_video_path=path)
            self.assertEqual(p["status"], "needs_rgb")
        for wrong in ("/fixture/other/fixture.mp4", "/fixture/normalized_704x1280/main/videos/other.mp4"):
            p = style.prepare_style({**row(), "video_path": wrong}, {"genre": "ballet"},
                source_refs=refs, historical_video_path=wrong)
            self.assertEqual(p["status"], "needs_rgb")
        p = style.prepare_style({**row(), "video_path": path}, {"genre": "ballet"},
            source_refs=refs, historical_video_path="/elsewhere/normalized_704x1280/main/videos/fixture.mp4")
        self.assertEqual(p["status"], "needs_rgb")

    def test_actual_historical_freestyle_is_not_unknown_fallback(self):
        p = style.prepare_style(row(), {"genre": "freestyle / other", "genre_conf": "high", "sub_style": "freestyle"},
                                historical_video_sha256="a" * 64)
        self.assertEqual((p["status"], p["dance_style"]), ("resolved", "freestyle / other"))
        missing = style.prepare_style(row(), historical_video_sha256="a" * 64)
        self.assertEqual((missing["status"], missing["dance_style"]), ("needs_rgb", "unknown"))

    def test_rich_only_chapter2_explicit_and_contemporary_not_blind(self):
        self.assertEqual(style.resolve_style(None, rich("A ballet dance."))["dance_style"], "ballet")
        self.assertEqual(style.resolve_style(None, rich("Dance style: contemporary."))["dance_style"], "contemporary")
        self.assertEqual(style.resolve_style(None, rich("Contemporary dance with flowing arms."))["dance_style"], "contemporary")
        for text in ("Contemporary clothes with contemporary energy.", "A pop or urban dance style.",
                     "The dancer locks the arms, popping a shoulder.", "Not ballet dance; just movement."):
            self.assertEqual(style.resolve_style(None, rich(text))["status"], "needs_rgb", text)
        self.assertEqual(style.resolve_style(None, "A ballet dance outside any chapter.")["status"], "needs_rgb")

    def test_missing_low_conflict_and_other_do_not_fill_freestyle(self):
        for label, text in ((None, ""), ({"genre": "freestyle / other", "genre_conf": "low"}, ""),
                            ({"genre": "ballet", "genre_conf": "low"}, ""),
                            (None, rich("A ballet dance and a hip-hop dance."))):
            r = style.resolve_style(label, text)
            self.assertEqual((r["status"], r["dance_style"]), ("needs_rgb", "unknown"))
        self.assertEqual(style.resolve_style({"sub_style": "Chinese classical sleeve dance"})["dance_style"], "chinese classical")
        self.assertEqual(style.resolve_style(None, rich("Chinese folk dance."))["dance_style"], "chinese folk")

    def test_valid_old_genre_keeps_priority_and_conflict_is_review_metadata(self):
        for label, text in (({"genre": "ballet", "genre_conf": "high"}, rich("Hip-hop dance.")),
                            ({"genre": "ballet", "sub_style": "hip-hop choreography"}, ""),
                            ({"genre": "ballet"}, rich("Ballet dance and hip-hop dance."))):
            p = style.prepare_style(row(), label, text, historical_video_sha256="a" * 64)
            self.assertEqual((p["status"], p["dance_style"]), ("resolved", "ballet"))
            self.assertTrue(p["metadata"]["conflict"])
            self.assertIsNone(style.style_item(p, None, None, None, None))
        low = style.resolve_style({"genre": "ballet", "genre_conf": "low"}, rich("Hip-hop dance."))
        self.assertEqual((low["status"], low["dance_style"], low["source"]), ("resolved", "hip-hop", "rich_chapter2"))
        self.assertTrue(low["metadata"]["conflict"])

    def fixture(self, root, indices=(8, 12)):
        frames = []
        for i in indices:
            path = root / (str(i) + ".jpg")
            path.write_bytes(("CPU fake JPEG, not visual evidence " + str(i)).encode())
            frames.append({"index": i, "time": i / 16., "source_timestamp": i / 16.,
                           "file": path.name, "sha256": style.bar.batch.sha256(path)})
        return {"video_sha256": "a" * 64, "frames": frames, "motion_caption": "PRIVATE_MOTION_PRIOR", "label": "PRIVATE_LABEL_PRIOR"}

    def test_independent_request_actual_frame_and_per_request_label_enums(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m = self.fixture(root)
            p = style.prepare_style(row(), rich_text=rich("Unclear style."))
            item = style.style_item(p, root, m, root / "style", args(), ["hip-hop", "unknown"])
            request = item["request"]
            user = json.dumps(request["messages"][1])
            self.assertNotIn("PRIVATE_MOTION_PRIOR", user)
            self.assertNotIn("PRIVATE_LABEL_PRIOR", user)
            self.assertNotIn("Unclear style", user)
            self.assertNotEqual(request["messages"][0]["content"], style.bar.SYSTEM)
            schema = request["json_schema"]
            known = schema["oneOf"][0]["properties"]
            self.assertEqual(known["dance_style"]["enum"], ["hip-hop"])
            self.assertEqual(known["frame_indices"]["items"]["enum"], [8, 12])
            self.assertNotIn("uniqueItems", json.dumps(schema))
            self.assertEqual(sum(c["type"] == "image_url" for c in request["messages"][1]["content"]), 2)
            other = style.style_schema([32, 40], ["ballet", "unknown"])
            self.assertEqual(other["oneOf"][0]["properties"]["frame_indices"]["items"]["enum"], [32, 40])
            self.assertEqual(known["frame_indices"]["items"]["enum"], [8, 12])

    def test_rgb_binding_hash_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m = self.fixture(root)
            p = style.prepare_style(row())
            bad = copy.deepcopy(m)
            bad["video_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "video SHA"):
                style.style_item(p, root, bad, root / "style", args())
            bad = copy.deepcopy(m)
            bad["frames"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "path/SHA"):
                style.style_item(p, root, bad, root / "style", args())
            bad["frames"][0]["file"] = "../missing.jpg"
            with self.assertRaises((ValueError, FileNotFoundError)):
                style.style_item(p, root, bad, root / "style", args())

    def test_large_union_subsamples_existing_RGB_without_motion_redecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m = self.fixture(root, tuple(range(64)))
            item = style.style_item(style.prepare_style(row()), root, m, root / "style", args())
            indices = [f["index"] for f in item["binding"]["RGB_evidence"]["frames"]]
            self.assertEqual(len(indices), 16)
            self.assertEqual((indices[0], indices[-1]), (0, 63))
            self.assertEqual(indices, sorted(set(indices)))

    def test_validator_rejects_duplicate_foreign_single_and_extra_keys(self):
        for refs in ([8, 8], [8], [8, 999], [True, 12]):
            bad = value()
            bad["frame_indices"] = refs
            with self.assertRaises(ValueError):
                style.check_style(bad, [8, 12])
        for label in ("a prose style with mood", None, "ballet"):
            bad = value()
            bad["dance_style"] = label
            with self.assertRaises(ValueError):
                style.check_style(bad, [8, 12], ["hip-hop", "unknown"])
        bad = {**value(), "confidence": "high"}
        with self.assertRaisesRegex(ValueError, "keys"):
            style.check_style(bad, [8, 12])

    def test_unknown_is_explicit_abstain_with_no_claimed_frames(self):
        unknown = {"dance_style": "unknown", "frame_indices": [], "reason": "Movement style is not distinguishable.", "abstain": True}
        self.assertEqual(style.check_style(unknown, [8, 12]), unknown)
        for changed in ({"frame_indices": [8, 12]}, {"abstain": False}):
            with self.assertRaises(ValueError):
                style.check_style({**unknown, **changed}, [8, 12])
        self.assertEqual(style.build_target("unknown", "[0,1] Motion."), "Dance style: unknown\n[0,1] Motion.")

    def test_shared_transaction_generation_receipts_resume_no_extra_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = style.prepare_style(row())
            item = style.style_item(p, root, self.fixture(root), root / "style", args())
            llm = FakeLLM()
            loads = []
            def runtime():
                loads.append(1)
                return llm, object
            with patch.object(style.bar.core, "structured_sampling", return_value=object()):
                r = style.observe_style(p, item, runtime, args())
                resumed = style.observe_style(p, item, runtime, args())
            self.assertEqual(r, resumed)
            self.assertEqual((r["status"], r["dance_style"]), ("candidate", "hip-hop"))
            self.assertEqual((len(llm.calls), len(loads)), (1, 1))
            self.assertEqual(r["metadata"]["generation.json"]["value"]["finish_reason"], "stop")
            self.assertFalse(r["semantic_approval"])
            out = style.export_joint(r, "[0,1] Original motion.")
            self.assertNotIn("Visible rhythmic", out["text"])
            self.assertIn("receipt.json", out["metadata"]["style"]["metadata"])

    def test_preparation_never_loads_model_and_prepared_failed_not_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = style.prepare_style(row())
            item = style.style_item(p, root, self.fixture(root), root / "style", args("prepare"))
            def no_model():
                self.fail("CPU preparation must not load model")
            r = style.observe_style(p, item, no_model, args("prepare"))
            self.assertEqual(r["status"], "prepared")
            with self.assertRaisesRegex(ValueError, "cannot export"):
                style.export_joint(r, "Original motion")
            with patch.object(style.bar.core, "structured_sampling", return_value=object()):
                failed = style.observe_style(p, item, lambda: (FakeLLM(finish="length"), object), args())
            self.assertEqual(failed["status"], "failed")
            with self.assertRaises(ValueError):
                style.export_joint(failed, "Original motion")

    def test_resolved_does_not_dispatch_and_input_not_mutated(self):
        original = {"genre": "ballet", "sub_style": "ballet", "genre_conf": "high"}
        p = style.prepare_style(row(), original, historical_video_sha256="a" * 64)
        original["genre"] = "hip-hop"
        def no_model():
            self.fail("resolved historical label does not dispatch")
        self.assertIsNone(style.style_item(p, None, None, None, None))
        self.assertEqual(style.observe_style(p, None, no_model, args())["dance_style"], "ballet")

    def test_cross_source_style_item_is_not_attached_to_other_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = style.prepare_style(row())
            item = style.style_item(p, root, self.fixture(root), root / "style", args())
            other = copy.deepcopy(p)
            other["id"] = "other"
            with self.assertRaisesRegex(ValueError, "binding differs"):
                style.observe_style(other, item, lambda: self.fail("must not dispatch"), args())

    def test_invalid_schema_and_target_inputs(self):
        for indices in ([1], [2, 1], [1, 1], [False, 2], list(range(17))):
            with self.assertRaises(ValueError):
                style.style_schema(indices)
        for label in ("", "ballet\nMood: graceful", {"dance_style": "ballet"}):
            with self.assertRaises(ValueError):
                style.build_target(label, "[0,1] Motion.")
        with self.assertRaises(ValueError):
            style.build_target("ballet", "")

    def test_real_xgrammar_bounded_whitespace_and_dynamic_label_enums(self):
        try:
            import xgrammar
        except ImportError:
            self.skipTest("installed xgrammar required for real CPU grammar compilation")
        compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo([bytes([i]) for i in range(256)]), max_threads=1)
        schema = style.style_schema([8, 12], ["hip-hop", "unknown"])
        bounded = compiler.compile_json_schema(schema, any_whitespace=False)
        unbounded = compiler.compile_json_schema(schema, any_whitespace=True)
        good = json.dumps(value())
        self.assertTrue(xgrammar.GrammarMatcher(bounded).accept_string(good))
        unknown = json.dumps({"dance_style": "unknown", "frame_indices": [], "reason": "Unclear.", "abstain": True})
        self.assertTrue(xgrammar.GrammarMatcher(bounded).accept_string(unknown))
        for change in ({"dance_style": "ballet"}, {"frame_indices": [8, 999]}, {"abstain": True}):
            self.assertFalse(xgrammar.GrammarMatcher(bounded).accept_string(json.dumps({**value(), **change})))
        runaway = good.replace('[8, 12]', '[8,' + ' ' * 4096 + '12]')
        self.assertTrue(xgrammar.GrammarMatcher(unbounded).accept_string(runaway))
        self.assertFalse(xgrammar.GrammarMatcher(bounded).accept_string(runaway))


if __name__ == "__main__":
    unittest.main()
