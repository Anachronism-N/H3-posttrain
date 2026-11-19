"""Synthetic review contracts; no private caption text or model invocation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/review_web/build_caption_chunk4_review.py"
spec = importlib.util.spec_from_file_location("chunk4_review", PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fixture():
    bundle = {"schema": review.SCHEMA, "skeleton_cases": [{"id": "skeleton", "extra": 32}],
              "caption_cases": [{"id": "sample", "video": "/EXP/sample.mp4", "audio": "/EXP/sample.wav",
                                  "views": [{"label": "old", "source": "MotionScript", "windows": []}],
                                  "source_refs": [{"path": "legacy", "sha256": "legacy"}],
                                  "unknown_field": {"keep": True}}],
              "note": "legacy note", "reviews": [{"keep": True}], "manifest": {"models": 15}}
    case = {"id": "sample", "video": "/EXP/sample.mp4", "audio": "/EXP/sample.wav",
            "views": [{"experiment_tag": review.EXPERIMENT, "strategy": "whole", "stage": "R2",
                       "label": "new pending", "source": "MotionScript",
                       "windows": [{"start": 0, "end": 2, "status": "pending", "caption": ""}]}],
            "source_refs": [{"path": "experiment", "sha256": "synthetic"}]}
    return bundle, case


class MergeTests(unittest.TestCase):
    def test_preserves_old_data_and_does_not_mutate_input(self):
        bundle, case = fixture()
        before = copy.deepcopy(bundle)
        result = review.append_cases(bundle, [case])
        self.assertEqual(bundle, before)
        for field in ("skeleton_cases", "reviews", "manifest", "note"):
            self.assertEqual(result[field], bundle[field])
        self.assertEqual(result["caption_cases"][0]["views"][0], bundle["caption_cases"][0]["views"][0])
        self.assertEqual(result["caption_cases"][0]["source_refs"], bundle["caption_cases"][0]["source_refs"])
        self.assertEqual(result["caption_cases"][0]["unknown_field"], {"keep": True})

    def test_idempotent_and_only_replaces_own_views(self):
        bundle, case = fixture()
        bundle["caption_cases"][0]["views"].append({"experiment_tag": "other", "label": "untouched"})
        once = review.append_cases(bundle, [case])
        self.assertEqual(once, review.append_cases(once, [case]))
        case["views"][0]["windows"][0]["status"] = "candidate"
        case["views"][0]["windows"][0]["caption"] = "SYNTHETIC TEST ONLY"
        again = review.append_cases(once, [case])
        self.assertEqual(len(again["caption_cases"][0]["views"]), 3)
        self.assertEqual(again["caption_cases"][0]["views"][1]["experiment_tag"], "other")

    def test_refuses_foreign_tag_duplicate_views_or_media_collision(self):
        bundle, case = fixture()
        case["views"][0]["experiment_tag"] = "other"
        with self.assertRaises(ValueError):
            review.append_cases(bundle, [case])
        bundle, case = fixture()
        case["views"].append(copy.deepcopy(case["views"][0]))
        with self.assertRaises(ValueError):
            review.append_cases(bundle, [case])
        bundle, case = fixture()
        case["video"] = "/EXP/different.mp4"
        with self.assertRaises(ValueError):
            review.append_cases(bundle, [case])

    def test_publish_refuses_concurrent_change_and_repeat_is_noop(self):
        bundle, case = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps(bundle))
            sha = review.digest(path.read_bytes())
            merged = review.append_cases(bundle, [case])
            self.assertTrue(review.publish(path, sha, merged))
            next_sha = review.digest(path.read_bytes())
            self.assertFalse(review.publish(path, next_sha, merged))
            with self.assertRaises(ValueError):
                review.publish(path, sha, bundle)
            self.assertEqual(json.loads(path.read_bytes()), merged)

    def test_receipt_digest_is_checked_before_reading_caption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"id": "synthetic"}))
            receipt = path.with_suffix(".receipt.json")
            receipt.write_text(json.dumps({"report_sha256": review.digest(path.read_bytes())}))
            record, ref = review.checked_json(path)
            self.assertEqual(record["id"], "synthetic")
            self.assertIn("receipt", ref)
            path.write_text(json.dumps({"id": "changed"}))
            with self.assertRaises(ValueError):
                review.checked_json(path)

    def test_private_body_cannot_escape_exp(self):
        with self.assertRaises(ValueError):
            review.inside_exp("/repo/caption.json", "/EXP")
        self.assertEqual(review.inside_exp("/EXP/caption.json", "/EXP"), Path("/EXP/caption.json"))

    def test_versions_preserve_v1_failed_and_repeat_v2_without_duplicates(self):
        bundle, case = fixture()
        v1 = review.append_cases(bundle, [case])
        v2case = copy.deepcopy(case)
        for view in v2case["views"]:
            view["run_version"] = "v2"
            view["label"] = "v2 pending"
        v2 = review.append_cases(v1, [v2case], run_version="v2")
        self.assertEqual(len(v2["caption_cases"][0]["views"]), 3)
        self.assertEqual(v2["caption_cases"][0]["views"][1], case["views"][0])
        self.assertEqual(v2, review.append_cases(v2, [v2case], run_version="v2"))
        with self.assertRaises(ValueError):
            review.append_cases(v1, [v2case])

    def test_publish_guard_requires_verified_actual_terminal_not_format_status(self):
        _, case = fixture()
        view = case["views"][0]
        view["stage"] = "R1"
        for status in ("pending", "candidate", "failed"):
            view["windows"][0]["status"] = status
            view["actual_model_terminal"] = False
            self.assertFalse(review.publishable([case]))
        view["actual_model_terminal"] = True
        for status in ("candidate", "failed"):
            view["windows"][0]["status"] = status
            self.assertTrue(review.publishable([case]))
        view["windows"][0]["status"] = "pending"
        self.assertFalse(review.publishable([case]))

    def test_interruption_marks_only_pending_and_never_satisfies_publish_guard(self):
        _, case = fixture()
        view = case["views"][0]
        view["strategy"] = "fixed2s"
        view["stage"] = "R1"
        view["actual_model_terminal"] = False
        view["windows"].append({"start": 2, "end": 4, "status": "failed", "caption": ""})
        with tempfile.TemporaryDirectory() as folder:
            control = Path(folder) / "failure.json"
            control.write_text(json.dumps({"error": "InterruptedError('manager received signal 15')",
                                           "worker_started": True, "partial_outputs_preserved": True}))
            cases = review.mark_interruptions([case], {"fixed2s": control})
            self.assertEqual([w["status"] for w in view["windows"]], ["interrupted", "failed"])
            self.assertEqual(view["windows"][0]["caption"], "")
            self.assertFalse(review.publishable(cases))
            self.assertIn("interruption_control", view["windows"][0]["events"])
            control.write_text(json.dumps({"error": "unknown", "worker_started": False}))
            with self.assertRaises(ValueError):
                review.mark_interruptions(cases, {"fixed2s": control})


class ReaderTests(unittest.TestCase):
    def test_original_window_status_partial_raw_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = {"window_start": 0, "window_end": 2, "status": "abstain",
                        "abstain_reason": "partial_support", "cleaned_motion_text": "",
                        "partial_supported_motion_text": "SYNTHETIC PARTIAL TEST ONLY",
                        "original_captions": [{"ms": ["SYNTHETIC RAW TEST ONLY"]}]}
            report = root / "source.json"
            report.write_text(json.dumps({"id": "sample", "A_whole_clip": original}))
            receipt = report.with_suffix(".receipt.json")
            receipt.write_text(json.dumps({"report_sha256": review.digest(report.read_bytes())}))
            selected = {**copy.deepcopy(original), "window_id": "w0", "facts": []}
            row = {"id": "sample", "strategy": "whole", "binding": {}, "windows": [selected],
                   "source_refs": {"report": {"path": str(report), "sha256": review.digest(report.read_bytes())},
                                   "receipt": {"path": str(receipt), "sha256": review.digest(receipt.read_bytes())}}}
            review.checked_original_row(row)
            for key, changed in (("status", "candidate"), ("window_end", 3),
                                 ("partial_supported_motion_text", ""), ("original_captions", [])):
                changed_row = copy.deepcopy(row)
                changed_row["windows"][0][key] = changed
                with self.assertRaises(ValueError):
                    review.checked_original_row(changed_row)

    def test_latest_missing_receipt_is_pending_even_with_final_or_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "final.json").write_text(json.dumps({"status": "candidate", "json": {}}))
            (root / "response.raw.txt").write_text("SYNTHETIC ORPHAN RESPONSE")
            result, refs, request = review.checked_transaction(root, {}, lambda v: v)
            self.assertEqual(result["status"], "pending")
            self.assertNotIn("json", result)
            self.assertEqual(refs, [])
            self.assertIsNone(request)

    def test_terminal_receipt_source_and_raw_sha_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = root / "request.json"
            request.write_text(json.dumps({"synthetic": True}))
            final = root / "final.json"
            final.write_text(json.dumps({"status": "failed", "error": "synthetic failure"}))
            receipt = {"status": "failed", "binding": {"source_sha256": "source", "plan_sha256": "plan"},
                       "files_sha256": {"request.json": review.digest(request.read_bytes()),
                                        "final.json": review.digest(final.read_bytes())}}
            (root / "receipt.json").write_text(json.dumps(receipt))
            result, refs, _ = review.checked_transaction(root, {"source_sha256": "source", "plan_sha256": "plan"}, lambda v: v)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(refs), 3)
            with self.assertRaises(ValueError):
                review.checked_transaction(root, {"plan_sha256": "stale"}, lambda v: v)
            final.write_text(json.dumps({"status": "candidate"}))
            with self.assertRaises(ValueError):
                review.checked_transaction(root, {}, lambda v: v)

    def test_transaction_paths_cannot_escape_receipt_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            outside = Path(folder) / "outside.json"
            outside.write_text("{}")
            (root / "receipt.json").write_text(json.dumps({"binding": {}, "files_sha256": {
                "../outside.json": review.digest(outside.read_bytes())}}))
            with self.assertRaises(ValueError):
                review.checked_transaction(root, {}, lambda v: v)


class IntegrationTests(unittest.TestCase):
    def test_real_transaction_receipts_to_fulltext_views_and_pending(self):
        try:
            core = review.import_validator()
        except ModuleNotFoundError as error:
            self.skipTest(f"CPU dependency absent: {error}")
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        import sys
        sdk = ModuleType("vllm.sampling_params")
        sdk.StructuredOutputsParams = lambda **kwargs: SimpleNamespace(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"id": "sample", "source_group": "SYNTHETIC TEST ONLY", "split": "test"}
            for key in ("video_path", "audio_path", "nlf_path"):
                path = root / key
                path.write_bytes(b"SYNTHETIC IDENTITY BYTES, NOT REAL MEDIA")
                manifest[key] = str(path)
            source = core.batch.source_identity(manifest)
            binding = {"video_sha256": source["files"]["video_path"]["sha256"],
                       "audio_sha256": source["files"]["audio_path"]["sha256"]}
            rows = []
            for strategy in review.STRATEGIES:
                originals = []
                for index in range(1 if strategy == "whole" else 2):
                    originals.append({"window_start": index * 2., "window_end": (index + 1) * 2.,
                                      "status": "candidate", "abstain_reason": None,
                                      "cleaned_motion_text": f"[{index*2.:.6f}, {(index+1)*2.:.6f}) seconds: SYNTHETIC ONLY.",
                                      "partial_supported_motion_text": "", "original_captions": [{"ms": ["SYNTHETIC ONLY."]}]})
                report = root / (strategy + ".source.json")
                key = "A_whole_clip" if strategy == "whole" else "B_fixed_windows" if strategy == "fixed2s" else "MS_windows"
                report.write_text(json.dumps({"id": "sample", "binding": binding,
                                              key: originals[0] if strategy == "whole" else originals}))
                receipt = report.with_suffix(".receipt.json")
                receipt.write_text(json.dumps({"report_sha256": review.digest(report.read_bytes())}))
                windows = []
                for index, original in enumerate(originals):
                    wid = f"w{index}"
                    windows.append({**copy.deepcopy(original), "window_id": wid,
                                    "facts": [{"id": wid + ".f0", "window_id": wid,
                                               "text": "SYNTHETIC ONLY.", "source": "original_ms"}]})
                rows.append({"schema": core.SCHEMA, "id": "sample", "strategy": strategy,
                             "manifest_record": manifest, "binding": binding, "windows": windows,
                             "source_refs": {"report": {"path": str(report), "sha256": review.digest(report.read_bytes())},
                                             "receipt": {"path": str(receipt), "sha256": review.digest(receipt.read_bytes())}}})
            plan = root / "plan.jsonl"
            plan.write_text("\n".join(json.dumps(r) for r in rows))
            identity = {"model_sha256": "synthetic", "code_sha256": "synthetic",
                        "plan_sha256": review.digest(plan.read_bytes()),
                        "rewrite_prompt_sha256": review.digest(core.REWRITE_SYSTEM.encode()),
                        "boundary_prompt_sha256": review.digest(core.BOUNDARY_SYSTEM.encode()),
                        "rewrite_tokens": 4096, "supplement_empty": False, "max_pixels": 448 * 448,
                        "max_num_seqs": 1, "gpu_memory_utilization": .88}
            if hasattr(core, "JSON_SCHEMAS"):
                identity["generation_protocol"] = "json_schema_v2"
                identity["json_schemas_sha256"] = core.digest(core.JSON_SCHEMAS)
            code_root = root / "synthetic_code"
            code_root.mkdir()
            code_file = code_root / "test_only.py"
            code_file.write_text("SYNTHETIC TEST CODE IDENTITY ONLY")
            producer = code_root / "ms_caption_rewrite.py"
            producer.write_bytes(Path(core.__file__).read_bytes())
            identity["code_identity"] = core.batch.identity_tree(code_root, [code_file, producer])
            identity["code_sha256"] = identity["code_identity"]["sha256"]
            identity["model_identity"] = {"sha256": identity["model_sha256"]}
            output = root / "output"
            core.save_json(output / "run.identity.json", {**identity, "strategy_filter": None})
            core.save_json(output / "load.receipt.json", {"model_identity": identity["model_identity"],
                           "code_identity": identity["code_identity"], "transport": "local_vllm_LLM.chat",
                           "tensor_parallel_size": 2, "third_party_endpoint_used": False, "model_download_allowed": False})
            args = SimpleNamespace(stage="run", supplement_empty=False, model="/synthetic/model", rewrite_tokens=4096)
            for row in rows:
                facts = core.facts_for(row)
                value = {"windows": [{"window_id": w["window_id"], "caption": "SYNTHETIC ONLY.",
                                      "used_fact_ids": [f["fact_id"] for f in facts if f["window_id"] == w["window_id"]],
                                      "used_boundary_ids": []} for w in row["windows"]], "issues": []}
                class FakeLLM:
                    def chat(self, *args, **kwargs):
                        return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                            text=json.dumps(value), token_ids=[2, 3], finish_reason="stop", stop_reason=None)])]
                    def get_tokenizer(self):
                        return SimpleNamespace(eos_token_id=9)
                with patch.object(core, "prepare_boundary", return_value={"status": "unknown", "json": {"relations": []}}), \
                     patch.dict(sys.modules, {"vllm.sampling_params": sdk}):
                    core.process_row(row, output / "rows/sample" / row["strategy"], source, [], {}, args, identity,
                                     lambda: (FakeLLM(), lambda **k: SimpleNamespace(**k)))
            cases = review.build_cases(plan, output)
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(cases[0]["views"]), 12)
            for view in cases[0]["views"]:
                if view["stage"] == "R0":
                    self.assertEqual(view["windows"][0]["caption"], rows[review.STRATEGIES.index(view["strategy"])]["windows"][0]["cleaned_motion_text"])
                else:
                    self.assertEqual(view["windows"][0]["status"], "candidate")
                    timed = output / "rows/sample" / view["strategy"] / ("R1" if view["strategy"] == "whole" else view["stage"]) / "timed.txt"
                    self.assertEqual(view["windows"][0]["caption"], timed.read_text())
            runtime, _ = review.load_json(output / "load.receipt.json")
            changed_runtime = copy.deepcopy(runtime)
            changed_runtime["tensor_parallel_size"] = 1
            (output / "load.receipt.json").write_text(json.dumps(changed_runtime))
            with self.assertRaisesRegex(ValueError, "TP2"):
                review.build_cases(plan, output)
            (output / "load.receipt.json").write_text(json.dumps(runtime))
            code_file.write_text("TAMPERED SYNTHETIC CODE")
            with self.assertRaisesRegex(ValueError, "SHA"):
                review.build_cases(plan, output)
            code_file.write_text("SYNTHETIC TEST CODE IDENTITY ONLY")
            (output / "rows/sample/fixed2s/R2/receipt.json").unlink()
            latest = review.build_cases(plan, output)
            r2 = next(v for v in latest[0]["views"] if v["strategy"] == "fixed2s" and v["stage"] == "R2")
            self.assertTrue(all(w["status"] == "pending" and not w["caption"] for w in r2["windows"]))


if __name__ == "__main__":
    unittest.main()
