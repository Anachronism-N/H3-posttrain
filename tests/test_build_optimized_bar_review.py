import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/review_web"))
import build_optimized_bar_review as builder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"))
import ms_bar_annotation as bar


def fixtures():
    cases, materials = [], []
    for sid in builder.IDS:
        windows = [{"start": 0., "end": .5, "window_id": "w0", "raw_window_indices": [0]},
                   {"start": .5, "end": 1., "window_id": "w1", "raw_window_indices": [1]}]
        original = {"label": "小节 merged短边缘", "source": "MotionScript",
                    "windows": [{"start": w["start"], "end": w["end"], "caption": "original MS", "status": "candidate"} for w in windows]}
        cases.append({"id": sid, "video": "/source/" + sid + ".mp4", "audio": "/source/" + sid + ".wav",
                      "views": [original, {"experiment_tag": builder.review.EXPERIMENT, "run_version": "v3", "strategy": "mergedbar", "stage": "R2", "windows": [{"caption": "old unchanged"}]}]})
        observation = {"status": "candidate", "actual_model_terminal": True, "json": {"events": [{"id": "e1", "description": "the dancer opens the left hand", "parts": ["left_hand"], "frame_indices": [0, 1]}],
                       "part_support": [{"part": "left_hand", "frame_indices": [0, 1], "reason": "visible"}, {"part": "left_foot", "frame_indices": [], "reason": "cropped"}], "uncertainties": []}}
        observation["raw_observation"] = copy.deepcopy(observation["json"])
        outputs = [{"window_id": w["window_id"], "caption": "the dancer opens the left hand", "used_fact_ids": [w["window_id"] + ":e1"], "used_boundary_ids": []} for w in windows]
        record = {"id": sid, "status": "candidate", "manifest_record": {"video_path": cases[-1]["video"], "audio_path": cases[-1]["audio"]},
                  "visual_baseline": {"windows": outputs}, "visual_baseline_timed_caption": "raw full text", "final_timed_caption": "one rewrite full text", "boundary_observations": []}
        materials.append({"root": Path("/run/sources") / sid, "record": record, "observations": [copy.deepcopy(observation), copy.deepcopy(observation)],
                          "prepared": {"windows": [{"window": w, "frames": [{"index": i * 2, "time": w["start"], "source_timestamp": w["start"], "file": "frame.jpg", "sha256": "a" * 64}]} for i, w in enumerate(windows)]},
                          "final": {"status": "candidate", "actual_model_terminal": True, "json": {"windows": copy.deepcopy(outputs)}, "review_evidence": {"receipt": {"path": "/run/receipt.json", "sha256": "b" * 64}}},
                          "refs": [{"path": "/run/result.json", "sha256": "c" * 64}]})
    return {"schema": builder.review.SCHEMA, "skeleton_cases": [{"id": "old skeleton", "raw_sha256": "d" * 64}], "caption_cases": cases}, materials


class OptimizedReviewContracts(unittest.TestCase):
    def test_deepcopy_preserves_old_views_skeleton_and_three_cases(self):
        bundle, materials = fixtures()
        before = copy.deepcopy(bundle)
        result, cases = builder.build(bundle, materials)
        self.assertEqual(bundle, before)
        self.assertEqual(result["skeleton_cases"], before["skeleton_cases"])
        self.assertEqual(len(result["caption_cases"]), 3)
        self.assertEqual(sum(len(c["views"]) for c in cases), 9)
        for old, new in zip(before["caption_cases"], result["caption_cases"]):
            self.assertEqual(new["views"][:2], old["views"])
            self.assertEqual([v["stage"] for v in new["views"][2:]], ["R0", "R1", "R2"])
            self.assertTrue(all(v["run_version"] == "stable_roi3_v2" for v in new["views"][2:]))
        result["caption_cases"][0]["views"][0]["windows"][0]["caption"] = "mutated"
        self.assertEqual(bundle, before)

    def test_unknown_foot_retains_hand_and_original_window_seek(self):
        bundle, materials = fixtures()
        _, cases = builder.build(bundle, materials)
        view = cases[0]["views"][1]
        self.assertIn("非MS补空", view["source"])
        self.assertIn("不是改写", view["label"])
        for w in view["windows"]:
            self.assertTrue(w["caption"])
            self.assertEqual(w["events"]["part_support"][1]["frame_indices"], [])
            self.assertEqual(w["events"]["raw_events"][0]["parts"], ["left_hand"])
            self.assertIn("/rgb/", w["events"]["frames"][0]["path"])
        self.assertEqual([(w["start"], w["end"]) for w in view["windows"]], [(0., .5), (.5, 1.)])

    def test_failed_window_not_hidden_or_converted_to_stillness(self):
        bundle, materials = fixtures()
        m = materials[-1]
        m["record"]["status"] = "failed"
        m["observations"][1] = {"status": "failed", "error": "missing torso support", "review_evidence": {"raw_response_path": "/run/real.raw.txt"}}
        m["record"]["visual_baseline"]["windows"][1]["caption"] = ""
        m["record"]["visual_baseline"]["windows"][1]["used_fact_ids"] = []
        m["final"]["json"]["windows"][1]["caption"] = ""
        m["final"]["json"]["windows"][1]["used_fact_ids"] = []
        _, cases = builder.build(bundle, materials)
        for view in cases[-1]["views"][1:]:
            self.assertEqual(len(view["windows"]), 2)
            self.assertEqual(view["source_status"], "failed")
            failed = view["windows"][1]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["caption"], "")
            self.assertEqual(failed["events"]["observation_error"], "missing torso support")
            self.assertEqual(failed["events"]["observation_transaction"]["raw_response_path"], "/run/real.raw.txt")

    def test_reject_foreign_case_or_changed_original_window(self):
        bundle, materials = fixtures()
        with self.assertRaisesRegex(ValueError, "fixed three"):
            builder.build(bundle, materials[:2])
        bundle["caption_cases"][0]["views"][0]["windows"][0]["end"] = .4
        with self.assertRaisesRegex(ValueError, "comparison differs"):
            builder.build(bundle, materials)

    def test_generation_must_match_real_runtime_model(self):
        value = {"status": "candidate", "review_evidence": {"generation": {"model_identity_sha256": "different"}}}
        with patch.object(builder.review, "checked_transaction", return_value=(value, [], {})):
            with self.assertRaisesRegex(ValueError, "model identity differs"):
                builder.checked_attempt("/run", {}, lambda v: v, "expected")

    def test_missing_receipt_does_not_borrow_old_output(self):
        with patch.object(builder.review, "checked_transaction", return_value=({"status": "pending"}, [], None)):
            with self.assertRaisesRegex(ValueError, "Missing terminal receipt"):
                builder.checked_attempt("/run", {}, lambda v: v, "model")

    def test_failed_raw_schema_error_is_reproduced_without_upgrade(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "response.raw.txt").write_text("{}")
            value = {"status": "failed", "error": "ValueError: missing torso", "review_evidence": {}}
            with patch.object(builder.review, "checked_transaction", return_value=(value, [], {})), patch.object(builder.review, "import_validator") as imported:
                imported.return_value.parse_response.return_value = {}
                def reject(v):
                    raise ValueError("missing torso")
                result, _ = builder.checked_attempt(root, {}, reject, "model")
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["review_evidence"]["failure_reproduced_from_raw"])

    def test_saved_failure_that_validates_now_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "response.raw.txt").write_text("{}")
            value = {"status": "failed", "error": "ValueError: missing torso", "review_evidence": {}}
            with patch.object(builder.review, "checked_transaction", return_value=(value, [], {})), patch.object(builder.review, "import_validator"):
                with self.assertRaisesRegex(ValueError, "failure not reproduced"):
                    builder.checked_attempt(root, {}, lambda v: v, "model")

    def test_default_dryrun_does_not_publish_or_write_annotation(self):
        bundle, materials = fixtures()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.json"
            path.write_text(json.dumps(bundle))
            before = path.read_bytes()
            with patch.object(builder, "validate_run", return_value=(materials, {"failed_source_count": 1})), patch.object(builder.review, "publish") as publish, redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(builder.main(["--run", "/run", "--run-version", builder.VERSION, "--validator-source", "/frozen", "--annotation", str(path)]), 0)
                publish.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(json.loads(stdout.getvalue())["dry_run"])

    def test_full_12000_plan_only_reads_selected_fixed_three(self):
        rows = [({"id": sid}, sid.encode()) for sid in builder.IDS]
        fake_bar = SimpleNamespace(read_plan=Mock(return_value=(rows, "plan_sha", 12000)))
        protocol = {"plan_path": "/plan/full12000.jsonl", "source_count": 12000, "plan_sha256": "plan_sha"}
        latest = {"selection": {"selected_ids": list(builder.IDS), "shard": 0, "nshard": 1},
                  "summary": {"selected_source_count": 3, "total_source_count": 12000, "plan_sha256": "plan_sha"}}
        self.assertEqual(builder.selected_plan(fake_bar, protocol, latest), (rows, "plan_sha"))
        fake_bar.read_plan.assert_called_once_with(protocol["plan_path"], 0, 1, limit=3)
        for field, value in (("selected_ids", list(builder.IDS) + ["foreign"]), ("selected_ids", list(reversed(builder.IDS)))):
            bad = copy.deepcopy(latest)
            bad["selection"][field] = value
            with self.assertRaisesRegex(ValueError, "fixed three"):
                builder.selected_plan(fake_bar, protocol, bad)
        bad = copy.deepcopy(latest)
        bad["summary"]["selected_source_count"] = 12000
        with self.assertRaisesRegex(ValueError, "fixed three"):
            builder.selected_plan(fake_bar, protocol, bad)

    def test_selected_plan_rejects_foreign_shard_rows_or_count(self):
        rows = [({"id": sid}, b"raw") for sid in builder.IDS]
        protocol = {"plan_path": "/plan", "source_count": 12000, "plan_sha256": "sha"}
        latest = {"selection": {"selected_ids": list(builder.IDS), "shard": 1, "nshard": 2},
                  "summary": {"selected_source_count": 3, "total_source_count": 12000, "plan_sha256": "sha"}}
        for selected, sha, total in ((rows[::-1], "sha", 12000), (rows, "changed", 12000), (rows, "sha", 11999)):
            fake_bar = SimpleNamespace(read_plan=Mock(return_value=(selected, sha, total)))
            with self.assertRaises(ValueError):
                builder.selected_plan(fake_bar, protocol, latest)

    def test_exact_executed_request_and_binding_are_required(self):
        value = {"status": "candidate", "review_evidence": {}}
        with patch.object(builder.review, "checked_transaction", return_value=(value, [], {"ROI": "changed"})):
            with self.assertRaisesRegex(ValueError, "Request differs"):
                builder.checked_attempt("/run", {"id": "s"}, lambda v: v, "model", {"ROI": "actual"})
        with patch.object(builder.review, "checked_transaction", return_value=(value, [], {"ROI": "actual"})),                 patch.object(builder.review, "load_json", return_value=({"binding": {"id": "s", "extra": "repair"}}, {})):
            with self.assertRaisesRegex(ValueError, "exact binding"):
                builder.checked_attempt("/run", {"id": "s"}, lambda v: v, "model", {"ROI": "actual"})

    def test_raw_and_admitted_clauses_are_distinct_with_roi_and_repair_audit(self):
        bundle, materials = fixtures()
        m = materials[0]
        m["observations"][0]["raw_observation"]["events"][0]["description"] += " Hair moves in wind."
        m["observations"][0]["json"]["support_filter_audit"] = {"raw_response_changed": False, "rejected_events": [{"reason": "appearance"}]}
        m["prepared"]["windows"][0]["hand_roi_evidence"] = {"distinct_source_frame_indices": [0, 1], "same_frame_two_hands_is_one_temporal_observation": True}
        m["final"]["reference_repair"] = {"maximum_additional_model_requests": 1, "automatic_ID_fill_performed": False}
        m["final"]["original_attempt"] = {"status": "failed", "review_evidence": {"raw_response_path": "/run/original.raw"}}
        _, cases = builder.build(bundle, materials)
        detail = cases[0]["views"][2]["windows"][0]["events"]
        self.assertIn("Hair", detail["raw_events"][0]["description"])
        self.assertNotIn("Hair", detail["admitted_events"][0]["description"])
        self.assertFalse(detail["support_filter_audit"]["raw_response_changed"])
        self.assertEqual(detail["hand_roi_evidence"]["distinct_source_frame_indices"], [0, 1])
        self.assertFalse(detail["reference_repair"]["automatic_ID_fill_performed"])
        self.assertEqual(detail["original_rewrite_transaction"]["raw_response_path"], "/run/original.raw")
        self.assertIn("至多一次", cases[0]["views"][2]["label"])

    def test_actual_dispatch_error_preserves_receipt_error_and_sha_strength(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "response.raw.txt").write_text("")
            (root / "dispatch_error.json").write_text(json.dumps({"error": "VLLMValidationError: Unimplemented keys: [uniqueItems]"}))
            value = {"status": "failed", "error": "FileNotFoundError: generation missing", "review_evidence": {}}
            (root / "receipt.json").write_text(json.dumps({"files_sha256": {}}))
            with patch.object(builder.review, "checked_transaction", side_effect=lambda *a: (copy.deepcopy(value), [], {})),                     patch.object(builder.review, "import_validator") as core:
                core.return_value.validate_generation.side_effect = FileNotFoundError("generation missing")
                result, refs = builder.checked_attempt(root, {}, lambda v: v, "model")
                self.assertEqual(result["error"], value["error"])
                evidence = result["review_evidence"]
                self.assertIn("uniqueItems", evidence["actualdispatch_error"])
                self.assertFalse(evidence["dispatch_error_terminal_receipt_sha_bound"])
                self.assertEqual(refs, [evidence["dispatch_error_ref"]])
                (root / "receipt.json").write_text(json.dumps({"files_sha256": {"dispatch_error.json": evidence["dispatch_error_ref"]["sha256"]}}))
                result, _ = builder.checked_attempt(root, {}, lambda v: v, "model")
                self.assertTrue(result["review_evidence"]["dispatch_error_terminal_receipt_sha_bound"])
                (root / "dispatch_error.json").write_text(json.dumps({"error": "changed"}))
                with self.assertRaisesRegex(ValueError, "Dispatch error SHA"):
                    builder.checked_attempt(root, {}, lambda v: v, "model")
        bundle, materials = fixtures()
        materials[0]["final"] = {"status": "failed", "error": "FileNotFoundError: generation missing",
                                  "review_evidence": {"actualdispatch_error": "VLLMValidationError: uniqueItems",
                                                      "dispatch_error_ref": {"path": "/attempt/dispatch_error.json", "sha256": "a" * 64},
                                                      "dispatch_error_terminal_receipt_sha_bound": False}}
        _, cases = builder.build(bundle, materials)
        detail = cases[0]["views"][2]["windows"][0]["events"]
        self.assertEqual(detail["rewrite_error"], "FileNotFoundError: generation missing")
        self.assertIn("uniqueItems", detail["actualdispatch_error"])
        self.assertFalse(detail["dispatch_error_terminal_receipt_sha_bound"])
        self.assertEqual(detail["dispatch_error_ref"]["path"], "/attempt/dispatch_error.json")
        self.assertIsNone(cases[0]["views"][1]["windows"][0]["events"]["actualdispatch_error"])

    def test_existing_v4_views_scores_and_foreign_cases_are_unchanged(self):
        bundle, materials = fixtures()
        bundle["scores"] = [{"view": "v4", "score": 2}]
        for case in bundle["caption_cases"]:
            case["views"].append({"experiment_tag": builder.review.EXPERIMENT, "run_version": "v4", "stage": "R2", "strategy": "mergedbar", "caption": "old v4"})
        foreign = copy.deepcopy(bundle["caption_cases"][0])
        foreign["id"] = "foreign"
        bundle["caption_cases"].append(foreign)
        before = copy.deepcopy(bundle)
        updated, _ = builder.build(bundle, materials)
        self.assertEqual(updated["scores"], before["scores"])
        self.assertEqual(updated["caption_cases"][-1], before["caption_cases"][-1])
        for old, new in zip(before["caption_cases"][:3], updated["caption_cases"][:3]):
            self.assertEqual(new["views"][:3], old["views"])
        rerun, _ = builder.build(updated, materials)
        self.assertEqual(rerun, updated)

    def test_cli_rejects_legacy_v4_version_before_any_annotation_read(self):
        with patch.object(builder.review, "load_json") as load, patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                builder.main(["--run", "/run", "--run-version", "v4", "--validator-source", "/frozen", "--annotation", "/data"])
            load.assert_not_called()

    def test_prepared_roi_selection_is_reconstructed_not_trusted_from_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"source line"
            (root / "plan.line.raw.json").write_bytes(raw)
            window = {"window_id": "w0", "start": 0., "end": 1.}
            binding, row = {"source": "bound"}, {"windows": [window]}
            roi_meta = {"available": True, "path": "/roi/manifest", "sha256": "roi_sha"}
            evidence = {"distinct_source_frame_indices": [0, 1], "image_budget": {"body": 2, "ROI": 4}}
            mask = {"available": False}
            policy = {"requested_fps": 8.0}
            prepared = {"binding": binding, "source_identity": {"files": {}}, "source_refs": {}, "raw2d_mask": mask, "forbidden": {},
                        "probe": {"times": [0., .5]}, "union_manifest_sha256": "sha", "hand_roi": roi_meta, "boundaries": [],
                        "windows": [{"window": window, "frames": [{"index": 0}, {"index": 1}], "hand_roi_evidence": evidence,
                                     "sampling_policy": {**policy, "body_budget_reduced_for_same_frame_ROI": True, "maximum_body_images": 8,
                                                         "actual_body_frame_count": 2, "maximum_total_body_plus_ROI_images": 16}}]}
            values = {"binding.json": binding, "plan.json": row, "prepared.json": prepared,
                      "prepared.receipt.json": {"binding": binding, "prepared_sha256": "sha"}, "manifest.json": {"binding": binding, "frames": []}}
            fake_bar = SimpleNamespace(load=lambda p: values[p.name], batch=SimpleNamespace(sha256=lambda p: "sha", SOURCE_FIELDS=()),
                                       core=SimpleNamespace(window_bounds=lambda w: (w["start"], w["end"])),
                                       load_hard_mask=lambda r, p: ({}, mask), load_hand_roi=Mock(return_value=({}, copy.deepcopy(roi_meta))),
                                       normalized_windows=lambda r: r["windows"], boundary_specs=lambda w, p, e: [],
                                       select_window_frames=lambda t, s, e, f: ([0, 1], policy),
                                       select_window_roi=Mock(return_value=([0, 1], copy.deepcopy(evidence))))
            params = {"hand_roi_root": "/executed/roi", "sample_fps": 8., "boundary_review": "on"}
            self.assertEqual(builder.checked_prepared(fake_bar, root, row, raw, binding, params), prepared)
            fake_bar.load_hand_roi.assert_called_once_with(row, prepared["probe"], params["hand_roi_root"])
            prepared["windows"][0]["hand_roi_evidence"]["distinct_source_frame_indices"] = [0, 2]
            with self.assertRaisesRegex(ValueError, "ROI same-frame"):
                builder.checked_prepared(fake_bar, root, row, raw, binding, params)
            prepared["windows"][0]["hand_roi_evidence"] = copy.deepcopy(fake_bar.select_window_roi.return_value[1])
            prepared["hand_roi"]["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "ROI source receipt"):
                builder.checked_prepared(fake_bar, root, row, raw, binding, params)

    def test_frozen_repair_algorithm_is_replayed_receipt_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "rewrite" / "attempt-0"
            original.mkdir(parents=True)
            windows = [{"window_id": "w0", "start": 0., "end": 1.}]
            observations = [{"status": "candidate"}]
            facts = [{"fact_id": "w0:e1", "id": "w0:e1", "window_id": "w0", "text": "The dancer turns the head.",
                      "raw_text": "The dancer turns the head.", "parts": ["head"], "frame_indices": [0, 1],
                      "source_clause": {"raw_source_clause": "The dancer turns the head."}},
                     {"fact_id": "w0:e2", "id": "w0:e2", "window_id": "w0", "text": "The dancer lowers the head.",
                      "raw_text": "The dancer lowers the head.", "parts": ["head"], "frame_indices": [1, 2],
                      "source_clause": {"raw_source_clause": "The dancer lowers the head."}}]
            args = SimpleNamespace(model="/model", text_max_tokens=4096, rewrite_reference_repair="on")
            supplied = {"windows": [{"window_id": "w0", "source_status": "candidate", "source_status_is_audit_only": True, "admitted_fact_count": 2}],
                        "facts": [{k: copy.deepcopy(v) for k, v in f.items() if k != "raw_text"} for f in facts]}
            request = bar.bar_rewrite_request(args, supplied, [])
            first_json = {"windows": [{"window_id": "w0", "caption": "The dancer turns the head.", "used_fact_ids": ["w0:e1"], "used_boundary_ids": []}], "issues": []}
            (original / "request.json").write_text(json.dumps(request))
            (original / "response.raw.txt").write_text(json.dumps(first_json))
            (original / "generation.json").write_text("{}")
            first = {"status": "failed", "error": "ValueError: omitted facts require exact issues naming their IDs"}
            final_json = copy.deepcopy(first_json)
            final_json["windows"][0].update(caption="The dancer turns and lowers the head.", used_fact_ids=["w0:e1", "w0:e2"])
            final = {"status": "candidate", "json": final_json}
            captured = []
            def check(path, binding, validator, model_sha, actual_request):
                captured.append((path, binding, actual_request))
                if path == original:
                    self.assertEqual(actual_request, request)
                    return copy.deepcopy(first), [{"path": "first"}]
                self.assertEqual(path, original / "reference-repair")
                self.assertEqual(actual_request["reference_repair"]["missing_fact_ids_by_window"], {"w0": ["w0:e2"]})
                self.assertEqual(binding["reference_repair_sha256"], bar.batch.digest(actual_request["reference_repair"]))
                self.assertEqual(validator(copy.deepcopy(final_json)), final_json)
                return copy.deepcopy(final), [{"path": "repair"}]
            saved = {**final, "attempt_path": str(original / "reference-repair"),
                     "original_attempt": {**first, "attempt_path": str(original)}}
            with patch.object(builder, "checked_attempt", side_effect=check),                     patch.object(bar.core, "validate_generation", return_value={}),                     patch.object(bar, "transactions", side_effect=AssertionError("runtime transactions must not run")),                     patch.object(bar.core, "save_json", side_effect=AssertionError("no bundle writes")):
                with self.assertRaisesRegex(ValueError, "one-repair binding"):
                    builder.checked_rewrite(bar, root, args, {}, windows, observations, facts, [], saved, "model")
                saved["reference_repair"] = captured[-1][2]["reference_repair"]
                captured.clear()
                result, refs = builder.checked_rewrite(bar, root, args, {}, windows, observations, facts, [], saved, "model")
                self.assertEqual(result["status"], "candidate")
                self.assertEqual(len(captured), 2)
                self.assertEqual(refs, [{"path": "first"}, {"path": "repair"}])
                bad = copy.deepcopy(saved)
                bad["reference_repair"]["automatic_ID_fill_performed"] = True
                with self.assertRaisesRegex(ValueError, "one-repair binding"):
                    builder.checked_rewrite(bar, root, args, {}, windows, observations, facts, [], bad, "model")


if __name__ == "__main__":
    unittest.main()
