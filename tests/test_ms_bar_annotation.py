"""CPU contracts and fake completions only; not a GPU/semantic approval."""
import builtins
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_bar_annotation as bar

FFMPEG = os.environ.get("MS_BAR_TEST_FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("MS_BAR_TEST_FFPROBE") or shutil.which("ffprobe")


def row(sid="source1"):
    binding = {"id": sid, "input_sha256": "a" * 64, "video_sha256": "b" * 64,
               "audio_sha256": "c" * 64, "actual_video_pts_sha256": "d" * 64,
               "media_start": 0., "media_end": 1., "video_fps": 8., "decoded_video_frames": 8}
    windows = [{"start": 0., "end": .5, "raw_window_indices": [0]},
               {"start": .5, "end": 1., "raw_window_indices": [1]}]
    return {"schema": bar.SCHEMA, "id": sid, "manifest_record": {"id": sid, "video_path": "/input/video.mp4",
            "audio_path": "/input/audio.wav", "nlf_path": "/input/nlf.npz", "source_group": "test", "split": "train"},
            "binding": binding, "window_manifest": {"schema": "motionscript_music_windows_v1", "id": sid,
            "binding": copy.deepcopy(binding), "windows": copy.deepcopy(windows), "params": {"policy": "merge_short"},
            "raw_windows": [{"start": 0., "end": .125}, {"start": .125, "end": 1.}],
            "fallback_metadata": {"native_short_window": True}}, "windows": windows,
            "source_refs": {"manifest": {"path": "/input/manifest.jsonl", "sha256": "a" * 64}}, "plan_status": "ready"}


def manifest(indices=(0, 1, 2, 3)):
    return {"window": {"start": 0., "end": .5, "raw_window_indices": [0], "window_id": "w00000"},
            "frames": [{"index": i, "time": i / 8., "source_timestamp": i / 8.} for i in indices]}


def observation(indices=(0, 1), part="left_hand"):
    value = bar.empty_observation("unknown or cropped")
    value["part_support"][bar.PARTS.index(part)] = {"part": part, "frame_indices": list(indices), "reason": "visible reliable RGB"}
    value["events"] = [{"id": "e1", "description": "The dancer opens the left hand." if part == "left_hand" else "The dancer turns the head.",
                        "parts": [part], "frame_indices": list(indices)}]
    value["abstain"] = False
    return value


class FakeLLM:
    def __init__(self, finish="stop", mode="valid"):
        self.calls = []
        self.finish = finish
        self.mode = mode

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append(copy.deepcopy(messages))
        outputs = []
        for msgs in messages:
            system = msgs[0]["content"]
            if system == bar.SYSTEM:
                content = msgs[1]["content"]
                header = json.loads(content[0]["text"])
                value = observation(header["allowed_source_frame_indices"][:2])
            elif system == bar.core.BOUNDARY_SYSTEM:
                content = msgs[1]["content"]
                header = json.loads(content[0]["text"])
                value = {"relations": [{"part": p, "relation": "unknown", "left_frame_indices": [], "right_frame_indices": []}
                                       for p in bar.core.PARTS], "issues": []}
            else:
                supplied = json.loads(msgs[1]["content"])
                value = {"windows": [{"window_id": w["window_id"],
                    "caption": " ".join(f["text"] for f in supplied["facts"] if f["window_id"] == w["window_id"]),
                    "used_fact_ids": [f["fact_id"] for f in supplied["facts"] if f["window_id"] == w["window_id"]],
                    "used_boundary_ids": []} for w in supplied["windows"]], "issues": []}
            text = json.dumps(value, ensure_ascii=False) if self.mode != "invalid" else "not-json"
            c = SimpleNamespace(text=text, token_ids=[7, 8], finish_reason=self.finish, stop_reason=None)
            outputs.append(SimpleNamespace(outputs=[c], prompt_token_ids=[1, 2, 3]))
        return outputs

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)


def args(stage="run", **kwargs):
    value = vars(bar.parser().parse_args(["--plan", "/input/plan", "--model", "/input/model", "--out", "/output", "--stage", stage]))
    value.update(kwargs)
    return SimpleNamespace(**value)


class AnnotationContracts(unittest.TestCase):
    def test_original_window_manifest_exact_preserved(self):
        r = row()
        bar.validate_row(r)
        normalized = bar.normalized_windows(r)
        self.assertEqual(normalized[0]["start"], r["window_manifest"]["windows"][0]["start"])
        self.assertEqual(r["window_manifest"]["fallback_metadata"], {"native_short_window": True})
        bad = copy.deepcopy(r)
        bad["windows"][0]["end"] = .4
        with self.assertRaisesRegex(ValueError, "original window_manifest"):
            bar.validate_row(bad)

    def test_short_native_selection_half_open_global(self):
        times = [0., .1, .2, .3, .4, .5]
        self.assertEqual(bar.batch.select_frames(times, .2, .4, 8., 16), [2, 3])
        self.assertNotIn(4, bar.batch.select_frames(times, .2, .4, 8., 16))
        self.assertEqual(bar.batch.select_frames(times, .2, .4, 4., 16), [2, 3])

    def roi_fixture(self, directory, indices=(0, 1, 2, 3)):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frames = [{"frame_index": i, "time": i / 8., "source_timestamp": i / 8.,
                   "source_pts": i, "rawframe_sha256": str(i) * 64} for i in indices]
        crops = []
        for part in ("left_hand", "right_hand"):
            for frame in frames:
                path = directory / (str(frame["frame_index"]) + "_" + part + ".jpg")
                path.write_bytes(b"CPU fake JPEG, not pixel truth")
                crops.append({**frame, "part": part, "path": str(path), "sha256": bar.batch.sha256(path),
                    "status": "supported", "quality_is_probability": False,
                    "crop_xyxy": [1, 2, 33, 34], "crop_pixel_size": [32, 32], "source_frame_size": [64, 64],
                    "ROI_to_source_affine": [[1., 0., 1.], [0., 1., 2.]],
                    "source_to_ROI_affine": [[1., 0., -1.], [0., 1., -2.]],
                    "resized": False, "rotated": False, "mirrored": False, "crop_adds_native_detail": False,
                    "supportpointindices": [92, 93, 94, 95, 96, 97], "hand_support_xyxy": [5., 6., 20., 21.]})
        return {"schema": "ms_native_hand_roi_v1", "id": "source1", "video_sha256": "b" * 64,
                "selected_frames": frames, "crops": crops, "probe": {"width": 64, "height": 64,
                    "decoded_video_frames": 8, "frame_times": [i / 8. for i in range(8)],
                    "actual_video_pts": [i / 8. for i in range(8)]}}

    def test_roi_same_frame_two_hands_one_time_and_total_image_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rois = self.roi_fixture(directory, indices=tuple(range(8)))
            window = {"start": 0., "end": 1., "window_id": "w0"}
            selected, evidence = bar.select_window_roi(rois, window, list(range(8)), [i / 8. for i in range(8)])
            required = set(evidence["distinct_source_frame_indices"])
            self.assertEqual(required, {0, 3, 7})
            self.assertEqual(evidence["left_hand"]["frame_indices"], evidence["right_hand"]["frame_indices"])
            self.assertTrue(required <= set(selected))
            self.assertLessEqual(len(selected), 8)
            self.assertEqual(evidence["image_budget"]["ROI"], 6)
            frames = []
            for i in selected:
                image = directory / ("body" + str(i) + ".jpg")
                image.write_bytes(b"CPU fake body JPEG")
                frames.append({"index": i, "time": i / 8., "source_timestamp": i / 8., "file": image.name})
            content = bar.rgb_content(directory, frames, {}, evidence)
            self.assertEqual(sum(c["type"] == "image_url" for c in content), len(selected) + 6)
            texts = [json.loads(c["text"]) for c in content if c["type"] == "text" and c["text"].startswith('{')]
            self.assertEqual(sum(t.get("not_another_time_observation", False) for t in texts), 6)
            m = {"frames": frames, "window": window, "hand_roi_evidence": evidence}
            request = bar.visual_request(args(), directory, m, {})
            self.assertEqual(request["json_schema"]["properties"]["events"]["items"]["properties"]["frame_indices"]["items"]["enum"], selected)
            with self.assertRaisesRegex(ValueError, "distinct actual"):
                bar.check_indices([0, 0], set(selected), 2)

    def test_roi_clips_each_bar_and_single_hand_frame_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            rois = self.roi_fixture(temp, indices=(0, 1, 2, 3))
            window = {"start": .25, "end": .5, "window_id": "w0"}
            selected, evidence = bar.select_window_roi(rois, window, [2, 3], [i / 8. for i in range(8)])
            self.assertEqual(selected, [2, 3])
            self.assertEqual(evidence["distinct_source_frame_indices"], [2, 3])
            self.assertNotIn(4, selected)
            rois["crops"] = [c for c in rois["crops"] if c["part"] == "left_hand" or c["frame_index"] == 2]
            selected, evidence = bar.select_window_roi(rois, window, [2, 3], [i / 8. for i in range(8)])
            self.assertEqual(evidence["right_hand"]["status"], "unknown")
            self.assertEqual(evidence["right_hand"]["frame_indices"], [])
            self.assertEqual(evidence["left_hand"]["frame_indices"], [2, 3])

    def test_missing_fullpool_roi_does_not_block_base_and_no_lazy_GPU_import(self):
        with tempfile.TemporaryDirectory() as temp:
            result, meta = bar.load_hand_roi(row(), {}, temp)
            self.assertIsNone(result)
            self.assertTrue(meta["base_RGB_not_blocked"])
            selected, evidence = bar.select_window_roi(None, manifest()["window"], [0, 1, 2, 3], [])
            self.assertEqual(selected, [0, 1, 2, 3])
            self.assertIsNone(evidence)

    def test_roi_binding_actual_PTS_and_native_affine_are_mandatory(self):
        import ms_hand_roi as roi
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "source1"
            value = self.roi_fixture(directory)
            path = directory / "roi_manifest.json"
            path.write_bytes(bar.batch.json_bytes(value))
            path.with_name("source_manifest.json").write_text('{}')
            path.with_name("extract_receipt.json").write_text('{}')
            probe = {"times": [i / 8. for i in range(8)], "source_timestamps": [i / 8. for i in range(8)]}
            with patch.object(roi, "validate_completed", return_value=value):
                loaded, meta = bar.load_hand_roi(row(), probe, temp)
                self.assertEqual(loaded, value)
                self.assertTrue(meta["actual_PTS_values_compared_not_cross_schema_hashes"])
            for change in ("sha", "PTS", "index", "affine", "resize"):
                bad = copy.deepcopy(value)
                if change == "sha":
                    bad["video_sha256"] = "f" * 64
                elif change == "PTS":
                    bad["probe"]["actual_video_pts"][2] += .001
                elif change == "index":
                    bad["selected_frames"][0]["frame_index"] = 7
                elif change == "affine":
                    bad["crops"][0]["ROI_to_source_affine"][0][0] = 2.
                else:
                    bad["crops"][0]["resized"] = True
                with self.subTest(change=change), patch.object(roi, "validate_completed", return_value=bad):
                    with self.assertRaises(ValueError):
                        bar.load_hand_roi(row(), probe, str(path))

    def test_short_window_retains_native_frames_without_expanding(self):
        times = [i / 30. for i in range(15)]
        indices, policy = bar.select_window_frames(times, .1, .2, 8.)
        self.assertEqual(indices, [3, 4, 5])
        self.assertTrue(policy["native_short_window_fallback"])
        self.assertEqual(policy["native_available_frame_count"], 3)
        indices, policy = bar.select_window_frames(times, .1, .2, 4.)
        self.assertEqual(indices, [3, 4, 5])

    def test_shared_manifest_identity_hashed_once_and_records_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            rows = [row("s0"), row("s1")]
            path.write_text("".join(json.dumps(r["manifest_record"]) + "\n" for r in rows))
            expected = bar.batch.sha256(path)
            for r in rows:
                r["source_refs"] = {"manifest": {"path": str(path), "sha256": expected}}
            cache = {}
            with patch.object(bar.batch, "file_identity", wraps=bar.batch.file_identity) as identity:
                bar.source_references(rows[0], cache)
                bar.source_references(rows[1], cache)
                self.assertEqual(identity.call_count, 1)
            rows[1]["manifest_record"]["source_group"] = "not-in-source"
            with self.assertRaisesRegex(ValueError, "does not match"):
                bar.source_references(rows[1], cache)

    def test_raw2d_loader_is_source_bound_and_only_false_forbids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mask.json"
            r = row()
            mask = {"schema": "bar_raw2d_part_mask_v1", "id": r["id"],
                    "video_sha256": r["binding"]["video_sha256"],
                    "actual_video_pts_sha256": r["binding"]["actual_video_pts_sha256"],
                    "parts": {"left_hand": {"0": True, "1": False}, "left_foot": {"0": False, "1": False}}}
            path.write_bytes(bar.batch.json_bytes(mask))
            r["raw2d_mask"] = {"path": str(path), "sha256": bar.batch.sha256(path)}
            forbidden, meta = bar.load_hard_mask(r, {"times": [i / 8. for i in range(8)]})
            self.assertEqual(forbidden["left_hand"], {1})
            self.assertTrue(meta["available"])
            self.assertNotIn("right_hand", forbidden)
            mask["id"] = "other"
            path.write_bytes(bar.batch.json_bytes(mask))
            r["raw2d_mask"]["sha256"] = bar.batch.sha256(path)
            with self.assertRaisesRegex(ValueError, "binding differs"):
                bar.load_hard_mask(r, {"times": [i / 8. for i in range(8)]})

    def test_foot_unknown_does_not_block_hand(self):
        value = observation()
        checked = bar.validate_observation(value, manifest(), {"left_foot": {0, 1, 2, 3}, "right_foot": {0, 1, 2, 3}})
        self.assertFalse(checked["abstain"])
        self.assertEqual(checked["part_support"][bar.PARTS.index("left_foot")]["frame_indices"], [])

    def test_plural_and_finger_claims_require_their_own_visible_parts(self):
        for text in ("the dancer opens both hands", "the dancer taps the feet twice", "the dancer opens the left fingers"):
            value = observation()
            value["events"][0].update(description=text, parts=["head"])
            for part in value["part_support"]:
                part["frame_indices"] = [0, 1] if part["part"] == "head" else []
            with self.assertRaises(ValueError):
                bar.validate_observation(value, manifest())

    def test_bad_foot_or_appearance_event_does_not_discard_reliable_hand_fact(self):
        value = observation()
        good = copy.deepcopy(value["events"][0])
        foot = dict(good, id="e2", description="the dancer shifts between the feet", parts=["head"])
        hair = dict(good, id="e3", description="the dancer's hair blows in the breeze")
        value["events"] = [good, foot, hair]
        for part in value["part_support"]:
            if part["part"] in ("left_foot", "right_foot"):
                part["frame_indices"] = []
        raw = copy.deepcopy(value)
        result = bar.filter_supported_observation(value, manifest())
        self.assertEqual(value, raw)
        self.assertEqual([{k: v for k, v in e.items() if k != "source_clause"} for e in result["events"]], [good])
        self.assertFalse(result["abstain"])
        self.assertEqual(result["support_filter_audit"]["admitted_event_count"], 1)
        self.assertEqual(len(result["support_filter_audit"]["rejected_events"]), 2)
        self.assertFalse(result["support_filter_audit"]["raw_response_changed"])

    def test_evidence_sentence_is_metadata_but_sustained_motion_remains_a_fact(self):
        value = observation()
        value["events"][0]["description"] = "The dancer opens the left hand. This motion is visible from frame 2 to frame 4."
        old = copy.deepcopy(value)
        result = bar.filter_supported_observation(value, manifest())
        self.assertEqual(value, old)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["description"], "The dancer opens the left hand.")
        notes = result["support_filter_audit"]["evidence_only_notes"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["text"], "This motion is visible from frame 2 to frame 4.")
        self.assertTrue(notes[0]["stored_as_evidence_not_motion_fact"])
        self.assertTrue(bar.evidence_only_note("This is visible from frame 6 to frame 22."))
        self.assertFalse(bar.evidence_only_note("This motion is sustained with minor adjustments across frames 0 to 6."))

    def test_real_failure_evidence_notes_and_minor_adjustment_facts(self):
        texts = (
            "The dancer opens the left hand. This motion is visible from frame 2 to frame 4.",
            "The dancer opens the left hand. This is visible from frame 6 to frame 22.",
            "The dancer opens the left hand. This motion is sustained with minor adjustments across frames 0 to 6.")
        for text in texts:
            with self.subTest(text=text):
                raw = observation()
                raw["events"][0]["description"] = text
                old = copy.deepcopy(raw)
                checked = bar.filter_supported_observation(raw, manifest())
                self.assertEqual(raw, old)
                sustained = "sustained" in text
                self.assertEqual(len(checked["events"]), 2 if sustained else 1)
                self.assertEqual(len(checked["support_filter_audit"]["evidence_only_notes"]), 0 if sustained else 1)
                facts = bar.facts_from_observations([manifest()["window"]], [{"status": "candidate", "json": checked}])
                for fact in facts:
                    source = fact["source_clause"]
                    self.assertEqual(source["original_description"], text)
                    self.assertEqual(source["raw_source_clause"], fact["raw_text"])
                    self.assertIn(source["raw_source_clause"], text)
                    self.assertEqual(fact["fact_id"], "w00000:e1_c" + str(source["raw_clause_index"]))
                if sustained:
                    self.assertIn("minor adjustments", facts[-1]["raw_text"])

    def test_evidence_only_notes_cannot_hide_invalid_raw_refs(self):
        for refs in ([0, 0], [0, 99]):
            raw = observation()
            raw["events"][0].update(description="This is visible from frame 0 to frame 1.", frame_indices=refs)
            result = bar.filter_supported_observation(raw, manifest())
            self.assertEqual(result["support_filter_audit"]["evidence_only_notes"], [])
            self.assertEqual(len(result["support_filter_audit"]["rejected_events"]), 1)

    def rewrite_case(self):
        window = manifest()["window"]
        raw = observation()
        raw["events"][0]["description"] = "The dancer opens the left hand. The left hand remains open with minor adjustments."
        checked = bar.filter_supported_observation(raw, manifest())
        facts = bar.facts_from_observations([window], [{"status": "candidate", "json": checked}])
        supplied = {"windows": [{"window_id": window["window_id"]}], "facts": facts}
        request = bar.bar_rewrite_request(args(), supplied, [])
        return window, facts, request

    def test_rewrite_local_schema_ordered_facts_and_explicit_merged_coverage(self):
        window, facts, request = self.rewrite_case()
        payload = json.loads(request["messages"][1]["content"])
        ordered = payload["ordered_fact_groups"][0]["facts_in_source_order"]
        self.assertEqual([f["fact_id"] for f in ordered], [f["fact_id"] for f in facts])
        self.assertEqual(ordered[1]["source_clause"], facts[1]["source_clause"])
        refs = request["json_schema"]["properties"]["windows"]["prefixItems"][0]["properties"]["used_fact_ids"]
        self.assertEqual(refs["items"]["enum"], [f["fact_id"] for f in facts])
        self.assertNotIn("uniqueItems", refs)
        self.assertEqual(refs["maxItems"], len(facts))
        self.assertEqual(request["json_schema"]["properties"]["windows"]["prefixItems"][0]
                         ["properties"]["used_boundary_ids"]["maxItems"], 0)
        value = {"windows": [{"window_id": window["window_id"],
            "caption": "The dancer opens the left hand and keeps it open with minor adjustments.",
            "used_fact_ids": [f["fact_id"] for f in facts], "used_boundary_ids": []}], "issues": []}
        bar.check_supported_rewrite(value, [window], facts, [])
        value["windows"][0]["used_fact_ids"].pop()
        with self.assertRaisesRegex(ValueError, "omitted facts"):
            bar.check_supported_rewrite(value, [window], facts, [])

    def test_runtime_fixes_xgrammar_whitespace_once_and_receipts_policy(self):
        class CapturingLLM:
            calls = []

            def __init__(self, **kwargs):
                self.calls.append(kwargs)

        model = {"root": "/input/model", "sha256": "a" * 64}
        config = {"backend": "xgrammar", "disable_any_whitespace": True}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=CapturingLLM, SamplingParams=object)}), \
                 patch.object(bar, "model_reference", return_value=model), \
                 patch.object(bar, "code_identity", return_value={"code": "frozen"}), \
                 patch.object(bar.batch, "verify_current"), patch.dict(os.environ):
                runtime = bar.runtime_loader(args(visual_max_tokens=1800), directory, {"code": "frozen"})
                first = runtime()
                self.assertIs(runtime(), first)
            self.assertEqual(len(CapturingLLM.calls), 1)
            self.assertEqual(CapturingLLM.calls[0]["structured_outputs_config"], config)
            self.assertEqual(CapturingLLM.calls[0]["limit_mm_per_prompt"], {"image": 16, "video": 0})
            self.assertEqual(bar.load(directory / "load.receipt.json")["structured_outputs_config"], config)
            self.assertEqual(args(visual_max_tokens=1800).visual_max_tokens, 1800)

    def test_real_xgrammar_observation_rejects_runaway_whitespace(self):
        try:
            import xgrammar
        except ImportError:
            self.skipTest("installed xgrammar is required for real grammar compilation")
        compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo([bytes([i]) for i in range(256)]),
                                            max_threads=1)
        schema = bar.observation_schema([0, 1])
        fixed = compiler.compile_json_schema(schema, any_whitespace=False)
        unbounded = compiler.compile_json_schema(schema, any_whitespace=True)
        valid = json.dumps(observation())
        self.assertTrue(xgrammar.GrammarMatcher(fixed).accept_string(valid))
        # JSON whitespace, not a quoted string: reproduce a stalled frame-ref separator.
        marker = "\"frame_indices\": [0, 1]"
        for whitespace in (" " * 4096, " \t\n" * 1200):
            runaway = valid.replace(marker, "\"frame_indices\": [0," + whitespace + "1]", 1)
            self.assertEqual(json.loads(runaway), json.loads(valid))
            self.assertTrue(xgrammar.GrammarMatcher(unbounded).accept_string(runaway))
            self.assertFalse(xgrammar.GrammarMatcher(fixed).accept_string(runaway))

    def test_duplicate_rewrite_refs_still_rejected_without_unique_items(self):
        window, facts, request = self.rewrite_case()
        boundary = {"boundary_id": "b-w00000-w00001:left_hand", "relation": "continue",
                    "left_window_id": window["window_id"], "right_window_id": "w00001"}
        valid = {"windows": [{"window_id": window["window_id"], "caption": facts[0]["text"],
            "used_fact_ids": [f["fact_id"] for f in facts],
            "used_boundary_ids": [boundary["boundary_id"]]}], "issues": []}
        bar.check_supported_rewrite(valid, [window], facts, [boundary])
        for key in ("used_fact_ids", "used_boundary_ids"):
            value = copy.deepcopy(valid)
            value["windows"][0][key] = [valid["windows"][0][key][0]] * 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "duplicate evidence refs"):
                bar.check_supported_rewrite(value, [window], facts, [boundary])

    def test_real_xgrammar_rewrite_keeps_exact_windows_and_local_ref_enums(self):
        try:
            import xgrammar
        except ImportError:
            self.skipTest("installed xgrammar is required for real grammar compilation")
        window, facts, _ = self.rewrite_case()
        windows = [window, {"window_id": "w00001"}, {"window_id": "w00002"}]
        facts = facts + [dict(facts[0], fact_id="w00001:e1", window_id="w00001")]
        boundaries = [{"boundary_id": "b-w00000-w00001:left_hand", "relation": "continue",
                       "left_window_id": "w00000", "right_window_id": "w00001"},
                      {"boundary_id": "b-w00001-w00002:left_hand", "relation": "unknown",
                       "left_window_id": "w00001", "right_window_id": "w00002"}]
        request = bar.bar_rewrite_request(args(), {"windows": windows, "facts": facts}, boundaries)
        schema = request["json_schema"]
        window_schema = schema["properties"]["windows"]
        self.assertEqual(window_schema["minItems"], len(windows))
        self.assertEqual(window_schema["maxItems"], len(windows))
        self.assertFalse(window_schema["items"])
        self.assertEqual([p["properties"]["window_id"]["const"]
                          for p in window_schema["prefixItems"]], [w["window_id"] for w in windows])
        compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo([bytes([i]) for i in range(256)]),
                                            max_threads=1)
        compiled = compiler.compile_json_schema(schema, any_whitespace=False)
        valid = {"windows": [{"window_id": w["window_id"],
            "caption": " ".join(f["text"] for f in facts if f["window_id"] == w["window_id"]),
            "used_fact_ids": [f["fact_id"] for f in facts if f["window_id"] == w["window_id"]],
            "used_boundary_ids": [boundaries[0]["boundary_id"]] if i < 2 else []}
            for i, w in enumerate(windows)], "issues": []}
        self.assertTrue(xgrammar.GrammarMatcher(compiled).accept_string(json.dumps(valid)))
        bar.check_supported_rewrite(valid, windows, facts, boundaries)
        for change in ("missing_window", "extra_window", "wrong_const", "cross_window_fact",
                       "unknown_fact", "overlong_refs", "unknown_boundary", "nonadjacent_boundary",
                       "empty_window_fact", "empty_window_caption"):
            value = copy.deepcopy(valid)
            if change == "missing_window":
                value["windows"].pop()
            elif change == "extra_window":
                value["windows"].append(copy.deepcopy(value["windows"][-1]))
            elif change == "wrong_const":
                value["windows"][0]["window_id"] = "w00001"
            elif change == "cross_window_fact":
                value["windows"][0]["used_fact_ids"] = ["w00001:e1"]
            elif change == "unknown_fact":
                value["windows"][0]["used_fact_ids"] = ["unknown:e1"]
            elif change == "overlong_refs":
                value["windows"][0]["used_fact_ids"] *= 2
            elif change == "unknown_boundary":
                value["windows"][1]["used_boundary_ids"] = [boundaries[1]["boundary_id"]]
            elif change == "nonadjacent_boundary":
                value["windows"][2]["used_boundary_ids"] = [boundaries[0]["boundary_id"]]
            elif change == "empty_window_fact":
                value["windows"][2]["used_fact_ids"] = [facts[0]["fact_id"]]
            else:
                value["windows"][2]["caption"] = "Invented motion."
            with self.subTest(change=change):
                self.assertFalse(xgrammar.GrammarMatcher(compiled).accept_string(json.dumps(value)))

    def test_issues_reference_must_match_complete_id_not_prefix(self):
        window, facts, request = self.rewrite_case()
        value = {"windows": [{"window_id": window["window_id"], "caption": facts[0]["text"],
            "used_fact_ids": [facts[0]["fact_id"]], "used_boundary_ids": []}],
            "issues": [facts[1]["fact_id"] + "0 omitted"]}
        with self.assertRaisesRegex(ValueError, "exact issues"):
            bar.check_supported_rewrite(value, [window], facts, [])
        value["issues"] = [facts[1]["fact_id"] + ": unsupported precise state omitted"]
        bar.check_supported_rewrite(value, [window], facts, [])
        value["issues"] = ["Omitted " + facts[1]["fact_id"] + " because its state detail is uncertain."]
        bar.check_supported_rewrite(value, [window], facts, [])

    def test_one_real_dispatch_reference_repair_preserves_context_and_raw(self):
        window, facts, request = self.rewrite_case()
        class MissingOnceLLM(FakeLLM):
            def chat(self, messages, sampling, use_tqdm=False):
                outputs = super().chat(messages, sampling, use_tqdm)
                if len(self.calls) == 1:
                    for output in outputs:
                        value = json.loads(output.outputs[0].text)
                        value["windows"][0]["used_fact_ids"].pop()
                        output.outputs[0].text = json.dumps(value)
                return outputs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt-0"
            item = {"root": root, "request": request, "binding": {"facts_sha256": bar.batch.digest(facts)},
                "validator": lambda v: bar.check_supported_rewrite(v, [window], facts, [])}
            llm = MissingOnceLLM()
            with patch.object(bar.core, "structured_sampling", return_value=object()):
                final = bar.rewrite_with_reference_repair(item, lambda: (llm, object), args(), facts)
                self.assertEqual(final["status"], "candidate")
                self.assertEqual(len(llm.calls), 2)
                self.assertEqual(final["original_attempt"]["status"], "failed")
                raw = (root / "response.raw.txt").read_bytes()
                original_request = (root / "request.json").read_bytes()
                repair = bar.load(root / "reference-repair/request.json")
                self.assertEqual(repair["messages"][:2], request["messages"])
                self.assertEqual(repair["messages"][2]["content"].encode(), raw)
                self.assertEqual(repair["reference_repair"]["missing_fact_ids_by_window"],
                                 {window["window_id"]: [facts[1]["fact_id"]]})
                self.assertFalse(repair["reference_repair"]["automatic_ID_fill_performed"])
                bar.rewrite_with_reference_repair(item, lambda: self.fail("resume must not dispatch"), args(), facts)
                self.assertEqual((root / "response.raw.txt").read_bytes(), raw)
                self.assertEqual((root / "request.json").read_bytes(), original_request)

    def test_repeated_missing_refs_no_infinite_retry_and_truncation_no_repair(self):
        window, facts, request = self.rewrite_case()
        class MissingAlwaysLLM(FakeLLM):
            def chat(self, messages, sampling, use_tqdm=False):
                outputs = super().chat(messages, sampling, use_tqdm)
                for output in outputs:
                    value = json.loads(output.outputs[0].text)
                    value["windows"][0]["used_fact_ids"].pop()
                    output.outputs[0].text = json.dumps(value)
                return outputs
        for finish, expected_calls in (("stop", 2), ("length", 1)):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "attempt-0"
                item = {"root": root, "request": request, "binding": {},
                    "validator": lambda v: bar.check_supported_rewrite(v, [window], facts, [])}
                llm = MissingAlwaysLLM(finish=finish)
                with patch.object(bar.core, "structured_sampling", return_value=object()):
                    final = bar.rewrite_with_reference_repair(item, lambda: (llm, object), args(), facts)
                    self.assertEqual(final["status"], "failed")
                    self.assertEqual(len(llm.calls), expected_calls)
                    bar.rewrite_with_reference_repair(item, lambda: self.fail("terminal requests must not repeat"), args(), facts)
                self.assertFalse((root / "reference-repair/reference-repair").exists())

    def test_missing_reference_does_not_repair_malformed_later_window(self):
        windows = bar.normalized_windows(row())
        checked = bar.filter_supported_observation(observation(), manifest())
        observations = [{"status": "candidate", "json": checked} for _ in windows]
        facts = bar.facts_from_observations(windows, observations)
        duplicate = dict(facts[0], fact_id=facts[0]["fact_id"] + "_extra")
        facts.insert(1, duplicate)
        request = bar.bar_rewrite_request(args(), {"windows": windows, "facts": facts}, [])
        class MalformedLaterLLM(FakeLLM):
            def chat(self, messages, sampling, use_tqdm=False):
                outputs = super().chat(messages, sampling, use_tqdm)
                for output in outputs:
                    value = json.loads(output.outputs[0].text)
                    value["windows"][0]["used_fact_ids"].pop()
                    value["windows"][1]["used_fact_ids"] *= 2
                    output.outputs[0].text = json.dumps(value)
                return outputs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt-0"
            item = {"root": root, "request": request, "binding": {},
                    "validator": lambda v: bar.check_supported_rewrite(v, windows, facts, [])}
            llm = MalformedLaterLLM()
            with patch.object(bar.core, "structured_sampling", return_value=object()):
                final = bar.rewrite_with_reference_repair(item, lambda: (llm, object), args(), facts)
            self.assertEqual(final["status"], "failed")
            self.assertEqual(len(llm.calls), 1)
            self.assertFalse((root / "reference-repair").exists())

    def test_lexical_validator_does_not_establish_motion_semantic_truth(self):
        window, facts, request = self.rewrite_case()
        # A different motion can pass structural/body-part/count checks. The
        # contract must not be presented as source-motion semantic verification.
        value = {"windows": [{"window_id": window["window_id"],
            "caption": "The dancer lowers the left hand.",
            "used_fact_ids": [f["fact_id"] for f in facts], "used_boundary_ids": []}], "issues": []}
        bar.check_supported_rewrite(value, [window], facts, [])
        self.assertIn("not human semantic verification", bar.BAR_REWRITE_RULES)

    def test_one_bad_sentence_does_not_erase_supported_motion_sentence(self):
        value = observation()
        value["events"][0]["description"] = "The dancer opens the left hand. The hair moves in the breeze."
        raw = copy.deepcopy(value)
        result = bar.filter_supported_observation(value, manifest())
        self.assertEqual(value, raw)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["description"], "The dancer opens the left hand.")
        self.assertTrue(result["events"][0]["source_clause"]["source_text_is_verbatim_substring"])
        self.assertEqual(len(result["support_filter_audit"]["rejected_events"]), 1)
        self.assertFalse(result["abstain"])

    def test_rewrite_cannot_turn_supported_hand_fact_into_foot_or_count(self):
        windows = bar.normalized_windows(row())
        facts = [{"fact_id": w["window_id"]+":e1", "window_id": w["window_id"],
                  "text": "the dancer opens the left hand", "raw_text": "the dancer opens the left hand",
                  "parts": ["left_hand"]} for w in windows]
        good = {"windows": [{"window_id": w["window_id"], "caption": facts[i]["text"],
                 "used_fact_ids": [facts[i]["fact_id"]], "used_boundary_ids": []}
                for i,w in enumerate(windows)], "issues": []}
        bar.check_supported_rewrite(good, windows, facts, [])
        for caption in ("the dancer taps the right foot exactly ten times", "the dancer opens the left hand twice"):
            bad = copy.deepcopy(good)
            bad["windows"][0]["caption"] = caption
            with self.assertRaises(ValueError):
                bar.check_supported_rewrite(bad, windows, facts, [])

    def test_two_distinct_allowed_frames_and_own_part(self):
        for refs in ([0], [0, 0], [0, 4], [1, 0], [True, 1]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                bar.validate_observation(observation(refs), manifest())
        value = observation()
        value["events"][0]["parts"] = ["right_hand"]
        with self.assertRaisesRegex(ValueError, "own two"):
            bar.validate_observation(value, manifest())
        value = observation()
        value["events"][0]["description"] = "The dancer opens the right hand."
        with self.assertRaisesRegex(ValueError, "correct part"):
            bar.validate_observation(value, manifest())

    def test_reject_outside_original_window_manifest(self):
        m = manifest()
        m["frames"][0]["time"] = .5
        with self.assertRaisesRegex(ValueError, "half-open"):
            bar.validate_observation(observation(), m)

    def test_exact_part_list_and_missing_event_support(self):
        for change in ("duplicate", "missing", "unsupported", "event_abstain", "empty_nonabstain"):
            value = observation()
            if change == "duplicate":
                value["part_support"][0] = copy.deepcopy(value["part_support"][1])
            elif change == "missing":
                value["part_support"].pop()
            elif change == "unsupported":
                value["part_support"][bar.PARTS.index("left_hand")]["frame_indices"] = []
            elif change == "event_abstain":
                value["abstain"] = True
            else:
                value["events"] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                bar.validate_observation(value, manifest())
        self.assertTrue(bar.validate_observation(bar.empty_observation("unknown"), manifest())["abstain"])

    def test_raw_mask_on_off_same_source_hard_reject(self):
        value = observation()
        bar.validate_observation(value, manifest())
        with self.assertRaisesRegex(ValueError, "raw 2D mask"):
            bar.validate_observation(value, manifest(), {"left_hand": {1}})
        bar.validate_observation(value, manifest(), {"right_hand": {1}})

    def test_actual_schema_enums_bounds_and_unique_validator(self):
        schema = bar.observation_schema([20, 21])
        refs = schema["properties"]["events"]["items"]["properties"]["frame_indices"]
        self.assertEqual(refs["items"]["enum"], [20, 21])
        self.assertEqual(refs["minItems"], 2)
        self.assertEqual(refs["maxItems"], 2)
        supports = schema["properties"]["part_support"]["prefixItems"]
        self.assertEqual([x["properties"]["part"]["const"] for x in supports], list(bar.PARTS))
        value = observation()
        value["events"].append(copy.deepcopy(value["events"][0]))
        with self.assertRaisesRegex(ValueError, "event ID"):
            bar.validate_observation(value, manifest())

    def test_full_plan_sha_stable_under_limits_and_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.jsonl"
            path.write_text("".join(json.dumps(row("s" + str(i))) + "\n" for i in range(5)), encoding="utf-8")
            all_rows, full_sha, total = bar.read_plan(path)
            a, sa, _ = bar.read_plan(path, shard=0, nshard=2, limit=1)
            b, sb, _ = bar.read_plan(path, shard=1, nshard=2)
            self.assertEqual(sa, full_sha)
            self.assertEqual(sb, full_sha)
            self.assertEqual(total, 5)
            self.assertEqual(a[0][0]["id"], "s0")
            self.assertEqual([r["id"] for r, _ in b], ["s1", "s3"])
            path.write_text(json.dumps(row("S0")) + "\n" + json.dumps(row("s0")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bar.read_plan(path)

    def test_source_hash_pts_and_manifest_mismatch_rejected(self):
        r = row()
        source = {"files": {"video_path": {"sha256": r["binding"]["video_sha256"]},
                            "audio_path": {"sha256": r["binding"]["audio_sha256"]},
                            "nlf_path": {"sha256": r["binding"]["input_sha256"]}}}
        probe = {"actual_video_pts_sha256": r["binding"]["actual_video_pts_sha256"], "times": [i / 8 for i in range(8)],
                 "media_start": 0., "media_end": 1., "video_fps": 8.}
        with patch.object(bar.core, "verify_references"):
            bar.check_source_binding(r, source, probe, [])
            for key in ("video_sha256", "actual_video_pts_sha256", "input_sha256"):
                bad = copy.deepcopy(r)
                bad["binding"][key] = "f" * 64
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bar.check_source_binding(bad, source, probe, [])

    def test_one_fulltext_exact_windows_local_fact_ids_raw_untouched(self):
        windows = bar.normalized_windows(row())
        raw_observations = [observation() for _ in windows]
        raw_observations[0]["events"][0]["description"] = "She opens the left hand a second later."
        observations = [{"status": "candidate", "json": bar.filter_supported_observation(o, manifest())}
                        for o in raw_observations]
        original = copy.deepcopy(observations)
        facts = bar.facts_from_observations(windows, observations)
        self.assertEqual(observations, original)
        self.assertIn("She", facts[0]["raw_text"])
        self.assertNotIn("She", facts[0]["text"])
        supplied = {"windows": [{"window_id": w["window_id"]} for w in windows], "facts": facts}
        schema = bar.core.rewrite_schema(supplied)
        prefixes = schema["properties"]["windows"]["prefixItems"]
        self.assertEqual(prefixes[0]["properties"]["used_fact_ids"]["items"]["enum"], ["w00000:e1"])
        value = {"windows": [{"window_id": w["window_id"], "caption": facts[i]["text"],
                    "used_fact_ids": [facts[i]["fact_id"]], "used_boundary_ids": []} for i, w in enumerate(windows)], "issues": []}
        bar.core.check_rewrite(value, {"windows": windows}, facts)
        value["windows"][0]["used_fact_ids"] = ["w00001:e1"]
        with self.assertRaisesRegex(ValueError, "another window"):
            bar.core.check_rewrite(value, {"windows": windows}, facts)

    def test_boundary_cannot_reintroduce_unknown_legs(self):
        windows = [{"window_id": "w0"}, {"window_id": "w1"}]
        spec = {"left_window_id": "w0", "right_window_id": "w1"}
        relations = {"relations": [
            {"part": "left_leg", "relation": "continue", "left_frame_indices": [0, 1], "right_frame_indices": [2, 3]},
            {"part": "left_arm", "relation": "switch", "left_frame_indices": [0, 1], "right_frame_indices": [2, 3]}], "issues": []}
        observations = [{"json": {"part_support": [
            {"part": "left_leg", "frame_indices": []},
            {"part": "left_arm", "frame_indices": indices}]}} for indices in ([0, 1], [2, 3])]
        old = copy.deepcopy(relations)
        result = bar.gate_boundary_support(relations, spec, windows, observations)
        self.assertEqual(relations, old)
        self.assertEqual(result["relations"][0]["relation"], "unknown")
        self.assertEqual(result["relations"][1]["relation"], "switch")
        self.assertEqual(result["relations"][0]["raw_relation"], "continue")

    def test_video_tail_cannot_be_dropped_even_with_matching_window_manifest(self):
        source = row()
        source["windows"][-1]["end"] -= .1
        source["window_manifest"]["windows"] = copy.deepcopy(source["windows"])
        with self.assertRaisesRegex(ValueError, "full video end"):
            bar.validate_row(source)

    def test_boundary_short_windows_do_not_expand_support(self):
        windows = [{"start": 0., "end": .05, "window_id": "short"}, {"start": .05, "end": .5, "window_id": "next"}]
        probe = {"times": [i / 20. for i in range(10)]}
        spec = bar.boundary_specs(windows, probe, True)[0]
        self.assertFalse(spec["usable"])
        self.assertEqual(spec["left"], [0])

    def transaction_items(self, root, count=2):
        items = []
        for i in range(count):
            m = manifest()
            request = {"messages": [{"role": "system", "content": bar.SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": json.dumps({"allowed_source_frame_indices": [0, 1]})}]}],
                "sampling": {"temperature": 0., "seed": 0, "max_tokens": 1800}, "json_schema": bar.observation_schema([0, 1])}
            items.append({"root": root / str(i), "request": request, "validator": lambda v: bar.validate_observation(v, m),
                          "binding": {"source_sha": "a" * 64, "window": i}})
        return items

    def test_batch_two_single_dispatch_raw_receipts_resume_no_resubmit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items = self.transaction_items(root)
            llm = FakeLLM()
            with patch.object(bar.core, "structured_sampling", return_value=object()):
                result = bar.transactions(items, lambda: (llm, object), args())
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(len(llm.calls[0]), 2)
                self.assertTrue(all(r["status"] == "candidate" for r in result.values()))
                result2 = bar.transactions(items, lambda: self.fail("must not reload/reissue"), args())
                self.assertEqual(result2, result)
            receipt = bar.load(items[0]["root"] / "receipt.json")
            self.assertIn("response.raw.txt", receipt["files_sha256"])
            self.assertIn("generation.json", receipt["files_sha256"])
            self.assertEqual(bar.load(items[0]["root"] / "generation.json")["finish_reason"], "stop")
            (items[0]["root"] / "response.raw.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                bar.transactions(items, lambda: (llm, object), args())

    def test_truncation_failed_preserves_raw_and_limited_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "window"
            first = bar.attempt_root(root, False)
            item = self.transaction_items(Path(temp), 1)[0]
            item["root"] = first
            llm = FakeLLM(finish="length")
            with patch.object(bar.core, "structured_sampling", return_value=object()):
                result = bar.transactions([item], lambda: (llm, object), args())
            self.assertEqual(result[str(first)]["status"], "failed")
            oldraw = (first / "response.raw.txt").read_bytes()
            self.assertEqual(bar.attempt_root(root, False), first)
            second = bar.attempt_root(root, True)
            self.assertEqual(second.name, "attempt-1")
            item["root"] = second
            with patch.object(bar.core, "structured_sampling", return_value=object()):
                bar.transactions([item], lambda: (FakeLLM(mode="invalid"), object), args(retry_failed=True))
            self.assertEqual(bar.attempt_root(root, True), second)
            self.assertEqual(bar.attempt_root(root, False), second)
            self.assertEqual((first / "response.raw.txt").read_bytes(), oldraw)

    def test_orphan_raw_failed_not_resubmitted_and_binding_changed_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            item = self.transaction_items(Path(temp), 1)[0]
            item["root"].mkdir()
            (item["root"] / "response.raw.txt").write_text("{}", encoding="utf-8")
            result = bar.transactions([item], lambda: self.fail("orphan must not resubmit"), args())
            self.assertEqual(result[str(item["root"])]["status"], "failed")
            item["binding"]["source_sha"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "immutable"):
                bar.transactions([item], lambda: self.fail("not called"), args())

    def test_source_failure_retained_and_counted_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            plan = temp / "plan.jsonl"
            plan.write_text(json.dumps(row()) + "\n")
            out = temp / "out"
            with patch.object(bar.batch, "model_identity", side_effect=AssertionError("no GPU/hash")), redirect_stdout(io.StringIO()):
                code = bar.main(["--plan", str(plan), "--model", str(temp / "model"), "--out", str(out), "--stage", "prepare"])
            self.assertEqual(code, 1)
            failures = list((out / "sources/source1/failures").glob("*.json"))
            self.assertEqual(len(failures), 1)
            failed = bar.load(failures[0])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["original_windows"], row()["windows"])
            self.assertFalse(failed["formal_training"])
            summary = bar.load(next((out / "snapshots").glob("*/summary.json")))["summary"]
            self.assertEqual(summary["failed_source_count"], 1)

    def test_shared_model_load_failure_is_not_retried_per_window(self):
        with tempfile.TemporaryDirectory() as temp:
            items = self.transaction_items(Path(temp))
            def failed_runtime():
                raise ValueError("unavailable model")
            with self.assertRaisesRegex(RuntimeError, "not retrying per window"):
                bar.transactions(items, failed_runtime, args())

    def test_bundle_lock_rejects_concurrent_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "locked"
            with bar.batch.bundle_lock(root):
                with self.assertRaisesRegex(ValueError, "active"):
                    with bar.batch.bundle_lock(root):
                        self.fail("lock allowed")

    def test_model_reference_current_stat_and_not_prepare_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors.index.json").write_text("{}")
            (model / "weights.bin").write_bytes(b"weights")
            identity = bar.batch.model_identity(model)
            ref = Path(temp) / "ref.json"
            ref.write_bytes(bar.batch.json_bytes(identity))
            a = args(model=str(model), model_reference=str(ref))
            observed = bar.model_reference(a)
            self.assertEqual(observed["sha256"], identity["sha256"])
            self.assertTrue(observed["local_stats_refreshed_from_verified_staged_identity"])
            (model / "weights.bin").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "changed|differs"):
                bar.model_reference(a)

    @unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/probe unavailable")
    def test_real_cpu_prepare_one_decode_no_vllm_and_pipeline_resume_export(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            video, audio, nlf, manifest_path = [temp / p for p in ("video.mp4", "audio.wav", "nlf.npz", "manifest.jsonl")]
            subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=8:d=1",
                            "-c:v", "mpeg4", str(video)], check=True, capture_output=True)
            audio.write_bytes(b"provenance audio")
            nlf.write_bytes(b"provenance nlf")
            r = row()
            r["manifest_record"].update(video_path=str(video), audio_path=str(audio), nlf_path=str(nlf))
            manifest_path.write_text(json.dumps(r["manifest_record"]) + "\n")
            probe = bar.probe_actual(video, FFPROBE)
            r["binding"].update(video_sha256=bar.batch.sha256(video), audio_sha256=bar.batch.sha256(audio),
                input_sha256=bar.batch.sha256(nlf), actual_video_pts_sha256=probe["actual_video_pts_sha256"],
                media_end=probe["media_end"], video_fps=probe["video_fps"], decoded_video_frames=len(probe["times"]))
            r["source_refs"] = {"manifest": {"path": str(manifest_path), "sha256": bar.batch.sha256(manifest_path)}}
            r["window_manifest"]["binding"] = copy.deepcopy(r["binding"])
            plan = temp / "plan.jsonl"
            plan.write_text(json.dumps(r) + "\n", encoding="utf-8")
            out = temp / "out"
            common = ["--plan", str(plan), "--model", str(temp / "unused-model"), "--out", str(out), "--boundary-review", "off", "--ffmpeg", FFMPEG, "--ffprobe", FFPROBE]
            original_import = builtins.__import__
            def no_vllm(name, *a, **kw):
                self.assertFalse(name.startswith("vllm"), "prepare imported vLLM")
                return original_import(name, *a, **kw)
            actual_run = subprocess.run
            calls = []
            def tracked_run(command, *a, **kw):
                calls.append(command)
                return actual_run(command, *a, **kw)
            with patch.object(builtins, "__import__", side_effect=no_vllm), patch.object(bar.batch, "model_identity", side_effect=AssertionError("cold model hash")), \
                 patch.object(bar.subprocess, "run", side_effect=tracked_run), redirect_stdout(io.StringIO()):
                self.assertEqual(bar.main(common + ["--stage", "prepare"]), 0)
            self.assertEqual(sum("-filter_script:v" in c for c in calls), 1)
            prepared = bar.load(out / "sources/source1/prepared.json")
            self.assertEqual(prepared["original_windows"], r["windows"])
            self.assertTrue(all(len(w["frames"]) >= 2 for w in prepared["windows"]))
            llm = FakeLLM()
            with patch.object(bar, "runtime_loader", return_value=lambda: (llm, object)), \
                 patch.object(bar.core, "structured_sampling", return_value=object()), redirect_stdout(io.StringIO()):
                self.assertEqual(bar.main(common + ["--stage", "run"]), 0)
                self.assertEqual(len(llm.calls), 2)
                self.assertEqual(len(llm.calls[0]), 2)
                self.assertEqual(llm.calls[1][0][0]["content"], bar.core.REWRITE_SYSTEM + bar.BAR_REWRITE_RULES)
                self.assertEqual(bar.main(common + ["--stage", "run"]), 0)
                self.assertEqual(len(llm.calls), 2)
                self.assertEqual(bar.main(common + ["--stage", "export"]), 0)
            exports = list((out / "snapshots").glob("*/weak_labels.jsonl"))
            self.assertEqual(len(exports), 1)
            exported = bar.strict_json(exports[0].read_text().strip())
            self.assertEqual(exported["source"], bar.OUTPUT_SCHEMA)
            self.assertFalse(exported["formal_training"])
            self.assertEqual(len(exported["facts"]), 2)
            self.assertIn("[0.000-0.500s]", exported["final_timed_caption"])
            self.assertTrue(exported["coverage"]["full_window_caption"])
            changed = copy.deepcopy(r)
            changed["manifest_record"]["source_group"] = "changed"
            plan.write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "immutable"), redirect_stdout(io.StringIO()):
                bar.main(common + ["--stage", "prepare"])


if __name__ == "__main__":
    unittest.main()
