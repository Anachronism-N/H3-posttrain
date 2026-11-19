"""CPU contracts and generated test video, never model semantic approval."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_omnidance/exp/ms"))
import ms_batch_review as review

FFMPEG = os.environ.get("MS_REVIEW_TEST_FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("MS_REVIEW_TEST_FFPROBE") or shutil.which("ffprobe")


def row(i, split="train", motion="The dancer steps sideways."):
    return {"id": f"s{i:05d}", "split": split, "video_path": "/fixture/source.mp4",
            "audio_path": "/fixture/source.wav", "original_caption": "Original caption, not review evidence.",
            "label": {"dance_style": "freestyle dance", "timed_motion_script": motion,
                      "reasoning": "annotator reasoning must not be supplied", "confidence": .99}}


def evidence(indices=(0, 4, 8)):
    return {"frames": [{"index": i, "time": i / 8, "source_timestamp": .125 + i / 8,
                        "file": f"{i}.jpg", "sha256": "a" * 64} for i in indices], "hand_rois": []}


def issue(indices=(0, 4), kind="direct_contradiction", error_type="hallucination", severity="major"):
    return {"claimed_text": "The dancer raises both arms.", "actual_visual_evidence": "The feet move across both cited frames but the label only describes arm movement.",
            "frame_indices": list(indices), "error_type": error_type, "severity": severity, "evidence_kind": kind}


def value(status="pass", errors=None, uncertainties=None):
    return {"status": status, "errors": errors or [], "uncertainties": uncertainties or [],
            "coverage_note": "Only these actual sparse source frames were observed."}


class ReviewContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ms-batch-review-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_500_partition_reproducible_without_split_leakage(self):
        rows = [row(i) for i in range(1003)] + [row(i + 2000, "test") for i in range(4)]
        first = review.prepare_batches(rows, seed=13)
        again = review.prepare_batches(list(reversed(rows)), seed=13)
        self.assertEqual(first, again)
        self.assertEqual([p["source_count"] for p in first], [4, 500, 500, 3])
        ids = [sid for p in first for sid in p["source_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 1007)
        self.assertTrue(all(len({r["split"] for r in p["records"]}) == 1 for p in first))
        self.assertNotEqual(first, review.prepare_batches(rows, seed=14))
        self.assertTrue(all(not p["semantic_review_completed"] for p in first))

    def test_duplicate_source_across_splits_rejected(self):
        with self.assertRaisesRegex(ValueError, "recur"):
            review.prepare_batches([row(1), row(1, "test")])

    def test_random_25_plus_reproducible_label_risk_extras(self):
        rows = [row(i, motion="Left hand opens twice, then stops." if i < 20 else "Steps.") for i in range(500)]
        plan = review.sampling_plan(rows, "train-00000", seed=4, sample_count=25, risk_count=5)
        self.assertEqual(len(plan["random_source_ids"]), 25)
        self.assertEqual(len(plan["risk_source_ids"]), 5)
        self.assertFalse(set(plan["random_source_ids"]) & set(plan["risk_source_ids"]))
        self.assertEqual(plan, review.sampling_plan(list(reversed(rows)), "train-00000", seed=4))
        self.assertTrue(all("hands" in plan["risk_flags"][s] for s in plan["risk_source_ids"]))
        self.assertTrue(plan["policy"]["engineering_defaults_not_user_requirements"])

    def test_schema_uses_actual_source_indices_without_uniqueitems(self):
        schema = review.review_schema([32, 36, 40])
        allowed = schema["properties"]["errors"]["items"]["properties"]["frame_indices"]["items"]["enum"]
        self.assertEqual(allowed, [32, 36, 40])
        self.assertNotIn("uniqueItems", json.dumps(schema))
        for bad in ([0], [0, 0], [4, 0], [False, 4], list(range(17))):
            with self.assertRaises(ValueError):
                review.review_schema(bad)

    def test_error_requires_two_distinct_actual_source_frames(self):
        result = review.validate_review(value("confirmed_error", [issue()]), evidence())
        self.assertEqual(result["status"], "confirmed_error")
        self.assertFalse(result["semantic_approval"])
        self.assertTrue(result["confirmed_errors_are_AI_candidates_not_ground_truth"])
        single = review.validate_review(value("confirmed_error", [issue((0,))]), evidence())
        self.assertEqual(single["status"], "uncertain")
        self.assertEqual(single["errors"], [])
        for pair in ((0, 0), (0, 999), (4, 0)):
            with self.assertRaises(Exception):
                review.validate_review(value("confirmed_error", [issue(pair)]), evidence())

    def test_unobserved_between_samples_not_confirmed_missing(self):
        for kind in ("absence_only", "ambiguous"):
            result = review.validate_review(value("confirmed_error", [issue(kind=kind)]), evidence())
            self.assertEqual(result["status"], "uncertain")
            self.assertFalse(result["errors"])
        positive = review.validate_review(value("confirmed_error", [issue(kind="positive_missing", error_type="missingmajor")]), evidence())
        self.assertEqual(positive["status"], "confirmed_error")

    def test_pass_cannot_hide_errors(self):
        with self.assertRaisesRegex(ValueError, "conceal"):
            review.validate_review(value("pass", [issue()]), evidence())
        with self.assertRaises(Exception):
            review.validate_review(value("human_approved"), evidence())
        with self.assertRaises(ValueError):
            review.validate_review(value("confirmed_error"), evidence())

    def test_threshold_10pct_finite_50_100(self):
        self.assertEqual(review.next_sample_target(25, 500, 2, 25), 25)
        self.assertEqual(review.next_sample_target(25, 500, 3, 25), 50)
        self.assertEqual(review.next_sample_target(50, 500, 5, 50), 100)
        self.assertEqual(review.next_sample_target(100, 500, 99, 100), 100)
        self.assertEqual(review.next_sample_target(25, 40, 3, 25), 40)
        self.assertEqual(review.next_sample_target(40, 40, 39, 40), 40)
        self.assertEqual(review.next_sample_target(25, 500, 0, 0), 25)
        with self.assertRaises(ValueError):
            review.next_sample_target(25, 500, 1, 25, expansion_counts=(100, 50))

    def test_unsampled_not_passed_sample_denominator_only(self):
        plan = review.prepare_batches([row(i) for i in range(500)], risk_count=0)[0]
        calls = []
        def fake(r, root, prepare_only=False):
            sid = review.source_id(r)
            calls.append(sid)
            return {"status": "pass", "source_id": sid, "synthetic_CPU_fixture_not_model_review": True}
        summary = review.review_batch(plan, self.root, fake)
        self.assertEqual(len(calls), 25)
        self.assertEqual(summary["sampled_reviewed_denominator"], 25)
        self.assertEqual(summary["sampled_fraction"], .05)
        self.assertEqual(len(summary["unsampled_source_ids"]), 475)
        self.assertEqual(summary["unsampled_source_status"], "not_individually_reviewed")
        self.assertFalse(summary["full_batch_semantic_pass_claimed"])
        self.assertFalse(summary["acceptance"]["unsampled_sources_individually_reviewed"])
        self.assertTrue(summary["acceptance"]["eligible_for_training"])
        self.assertFalse(summary["acceptance"]["human_approval_automatically_granted"])

    def test_error_expands_finitely_isolates_no_autoapprove(self):
        plan = review.prepare_batches([row(i) for i in range(500)], risk_count=0)[0]
        calls = []
        def fake(r, root, prepare_only=False):
            sid = review.source_id(r)
            calls.append(sid)
            return {"status": "confirmed_error", "source_id": sid, "errors": [issue()],
                    "label_sha256": "b" * 64, "review_receipt": {"sha256": "c" * 64},
                    "synthetic_CPU_fixture_not_model_review": True}
        summary = review.review_batch(plan, self.root, fake)
        self.assertEqual(len(calls), 100)
        self.assertEqual(len(set(calls)), 100)
        self.assertEqual([r["random_target"] for r in summary["rounds"]], [25, 50, 100])
        self.assertEqual(summary["sampled_candidate_error_rate"], 1.)
        self.assertFalse(summary["acceptance"]["eligible_for_training"])
        self.assertEqual(len(list((self.root / plan["batch_id"] / "isolation").glob("*.json"))), 100)
        queue = review.read_json(self.root / plan["batch_id"] / "review_web.json")
        self.assertEqual(len(queue["items"]), 3)
        self.assertEqual(queue["error_count"], 100)
        self.assertTrue(all(i["human_decision"] == "pending" for i in queue["items"]))

    def test_major_uncertain_blocks_minor_not_every_source_manual(self):
        plan = review.prepare_batches([row(i) for i in range(25)], risk_count=0)[0]
        results = {sid: {"source_id": sid, "status": "pass"} for sid in plan["source_ids"]}
        sid = plan["source_ids"][0]
        results[sid] = {"source_id": sid, "status": "uncertain", "uncertainties": [issue(severity="major")]}
        self.assertFalse(review.batch_acceptance(plan, results)["eligible_for_training"])
        results[sid]["uncertainties"][0]["severity"] = "minor"
        self.assertTrue(review.batch_acceptance(plan, results)["eligible_for_training"])
        results[sid]["status"] = "failed"
        self.assertFalse(review.batch_acceptance(plan, results)["review_complete"])

    def test_human_falsepositive_binds_label_receipt(self):
        result = {"source_id": "s", "label_sha256": "l", "review_receipt": {"sha256": "r"}}
        decision = {"decision": "false_positive", "human_review_completed": True, "note": "checked original evidence",
                    "label_sha256": "l", "review_receipt_sha256": "r"}
        self.assertTrue(review.human_resolved(result, {"s": decision}))
        for key in ("label_sha256", "review_receipt_sha256", "human_review_completed", "note"):
            changed = dict(decision)
            changed[key] = None
            self.assertFalse(review.human_resolved(result, {"s": changed}))
        self.assertFalse(review.human_resolved(result, {"s": {"decision": "approved"}}))

    def test_filter_no_annotator_reasoning_scores_or_originalcaption(self):
        r = row(1)
        self.assertEqual(set(review.label_payload(r)), {"dance_style", "timed_motion_script"})
        for i in (0, 4, 8):
            (self.root / f"{i}.jpg").write_bytes(b"CPU hash fixture, not JPEG")
        request = review.review_request(r, evidence(), self.root, "/fixture/not-loaded-model")
        text = json.dumps(request["messages"][1]["content"])
        self.assertNotIn("annotator reasoning must not be supplied", text)
        self.assertNotIn("confidence", text)
        self.assertNotIn("Original caption, not review evidence", text)
        self.assertEqual(sum(i["type"] == "image_url" for i in request["messages"][1]["content"]), 3)

    def test_prepare_cli_no_gpu_model_import(self):
        source = self.root / "sources.jsonl"
        source.write_text(chr(10).join(json.dumps(row(i)) for i in range(30)))
        out = self.root / "prepared"
        import builtins
        old = builtins.__import__
        def no_gpu(name, *args, **kwargs):
            if name.split(".")[0] in ("vllm", "torch", "transformers"):
                raise AssertionError("prepare attempted GPU/model dependency import")
            return old(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=no_gpu):
            self.assertEqual(review.main(["--batch-list", str(source), "--out", str(out), "--prepare-only"]), 0)
        receipt = review.read_json(out / "prepare.receipt.json")
        self.assertFalse(receipt["GPU_model_load_performed"])
        self.assertFalse(receipt["semantic_review_completed"])

    @unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg/ffprobe unavailable")
    def test_real_generated_video_indices_pts_native_size_and_hash_resume(self):
        video = self.root / "generated-test.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=192x128:rate=8:duration=3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
        r = row(1)
        r["video_path"] = str(video)
        r.pop("audio_path")
        bundle = self.root / "evidence"
        manifest = review.prepare_evidence(r, bundle, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        frames = manifest["frames"]
        self.assertEqual(len(frames), 16)
        self.assertEqual((frames[0]["index"], frames[-1]["index"]), (0, 23))
        self.assertEqual([f["time"] for f in frames], [f["index"] / 8 for f in frames])
        self.assertEqual([f["source_timestamp"] for f in frames], [f["index"] / 8 for f in frames])
        from PIL import Image
        with Image.open(bundle / frames[0]["file"]) as image:
            self.assertEqual(image.size, (192, 128))
        self.assertTrue(manifest["source_RGB_saved_native_size"])
        with patch.object(review.subprocess, "run", side_effect=AssertionError("resume must not decode")):
            with patch.object(review.bar, "probe_actual", return_value=manifest["probe"]):
                self.assertEqual(review.prepare_evidence(r, bundle, ffmpeg=FFMPEG, ffprobe=FFPROBE), manifest)
        p = bundle / frames[0]["file"]
        p.write_bytes(b"changed source evidence fixture")
        with patch.object(review.bar, "probe_actual", return_value=manifest["probe"]):
            with self.assertRaisesRegex(ValueError, "RGB evidence changed"):
                review.prepare_evidence(r, bundle, ffmpeg=FFMPEG, ffprobe=FFPROBE)


class IncrementalAndTransactionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ms-review-incremental-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_production_style_object_is_text_only_and_pending_components_block(self):
        r = row(1)
        r.pop("label")
        r["dance_style_label"] = {"dance_style": "hiphop", "status": "candidate", "metadata": {"reason": "NEVER_USE_PRODUCER_REASON"}}
        r["target_text"] = "Dance style: hiphop" + chr(10) + "[0,2) seconds: The dancer steps."
        self.assertEqual(review.label_payload(r)["dance_style"], "hiphop")
        self.assertNotIn("NEVER_USE_PRODUCER_REASON", json.dumps(review.label_payload(r)))
        self.assertTrue(review.label_complete(r))
        r["dance_style_label"]["status"] = "failed"
        self.assertFalse(review.label_complete(r))
        r["dance_style_label"]["status"] = "unknown"
        self.assertFalse(review.label_complete(r))
        self.assertTrue(review.label_reviewable(r))
        r["dance_style_label"]["status"] = "candidate"
        r["motion_status"] = "prepared"
        self.assertFalse(review.label_complete(r))

    def test_explicit_completion_flags_and_component_failures_block_training(self):
        for key in ("joint_complete", "all_windows_nonempty"):
            for complete in (False, True):
                with self.subTest(key=key, complete=complete):
                    r = row(1)
                    r[key] = complete
                    self.assertEqual(review.label_complete(r), complete)
                    self.assertTrue(review.label_reviewable(r))
        for key, statuses in (("label_status", ("partial", "failed", "abstain")),
                              ("style_status", ("unknown", "abstain", "needs_rgb")),
                              ("motion_status", ("failed", "abstain", "partial")),
                              ("visual_status", ("failed", "abstain", "partial"))):
            for status in statuses:
                with self.subTest(key=key, status=status):
                    r = row(1)
                    r[key] = status
                    self.assertFalse(review.label_complete(r))
        r = row(1)
        r["dance_style_label"] = {"dance_style": "hiphop", "status": "resolved"}
        r["style_status"] = "unknown"
        self.assertFalse(review.label_complete(r))

    def test_legacy_full_25_remain_complete_but_partial_never_accepts_all_pass(self):
        rows = [row(i) for i in range(25)]
        plan = review.prepare_batches(rows, risk_count=0)[0]
        results = {sid: {"source_id": sid, "status": "pass"} for sid in plan["source_ids"]}
        self.assertTrue(all(review.label_complete(r) for r in plan["records"]))
        self.assertTrue(review.batch_acceptance(plan, results)["eligible_for_training"])
        for key in ("joint_complete", "all_windows_nonempty"):
            with self.subTest(key=key):
                plan["records"][0][key] = False
                self.assertTrue(review.label_reviewable(plan["records"][0]))
                accepted = review.batch_acceptance(plan, results)
                self.assertTrue(accepted["review_complete"])
                self.assertFalse(accepted["all_source_labels_completed"])
                self.assertFalse(accepted["eligible_for_training"])
                del plan["records"][0][key]

    def test_unknown_style_claim_is_reviewable_but_never_training_eligible(self):
        for style in ("unknown", "UNKNOWN", " Unknown "):
            with self.subTest(style=style):
                r = row(1)
                r["label"]["dance_style"] = style
                r["style_status"] = "resolved"
                self.assertFalse(review.label_complete(r))
                self.assertTrue(review.label_reviewable(r))
                plan = review.prepare_batches([r], risk_count=0)[0]
                results = {r["id"]: {"source_id": r["id"], "status": "pass"}}
                accepted = review.batch_acceptance(plan, results)
                self.assertTrue(accepted["review_complete"])
                self.assertFalse(accepted["eligible_for_training"])

    def test_representative_false_positive_does_not_resolve_other_bad_sources(self):
        plan = review.prepare_batches([row(i) for i in range(25)], risk_count=0)[0]
        results = {sid: {"source_id": sid, "status": "confirmed_error", "errors": [issue()],
                         "label_sha256": "label", "review_receipt": {"sha256": "receipt"}}
                   for sid in plan["source_ids"]}
        queue = review.review_queue(plan, results)
        self.assertEqual(len(queue["items"]), 3)
        decisions = {item["source_id"]: {"decision": "false_positive", "human_review_completed": True,
                                        "note": "checked original evidence", "label_sha256": "label",
                                        "review_receipt_sha256": "receipt"} for item in queue["items"]}
        accepted = review.batch_acceptance(plan, results, decisions=decisions)
        self.assertEqual(len(accepted["unresolved_confirmed_error_source_ids"]), 22)
        self.assertFalse(accepted["eligible_for_training"])

    def test_ready_batch_reviews_completed_samples_then_resumes_fixed_membership(self):
        rows = [row(i) for i in range(30)]
        expected = [r["id"] for r in rows]
        random = review.sampling_plan(rows, "production-batch-1", sample_count=25, risk_count=0)
        ready = set(random["random_source_ids"][:10])
        for r in rows:
            if r["id"] not in ready:
                r.pop("label")
                r["label_status"] = "pending"
        calls = []
        def fake(r, root, prepare_only=False):
            calls.append(r["id"])
            return {"source_id": r["id"], "status": "pass", "synthetic_CPU_fixture_not_model_review": True}
        first = review.review_batch_ready(rows, self.root, fake, batch_id="production-batch-1", expected_source_ids=expected, risk_count=0)
        self.assertEqual(set(calls), ready)
        self.assertEqual(first["state"], "pending")
        self.assertTrue(first["producer_processing_allowed"])
        self.assertFalse(first["acceptance"]["eligible_for_training"])
        self.assertEqual(len(first["pending_source_ids"]), 20)
        complete = [row(i) for i in range(30)]
        second = review.review_batch_ready(complete, self.root, fake, batch_id="production-batch-1", expected_source_ids=expected, risk_count=0)
        self.assertTrue(second["acceptance"]["eligible_for_training"])
        self.assertEqual(second["state"], "accepted")
        self.assertEqual(first["selected_source_ids"], second["selected_source_ids"])

    def test_partial_completed_list_cannot_accept_full_expected_batch(self):
        rows = [row(i) for i in range(20)]
        def fake(r, root, prepare_only=False):
            return {"source_id": r["id"], "status": "pass", "synthetic_CPU_fixture_not_model_review": True}
        summary = review.review_batch_ready(rows, self.root, fake, batch_id="partial", expected_source_ids=[row(i)["id"] for i in range(50)], risk_count=0)
        self.assertFalse(summary["acceptance"]["all_source_labels_completed"])
        self.assertFalse(summary["acceptance"]["eligible_for_training"])
        self.assertEqual(len(summary["pending_source_ids"]), 30)
        self.assertEqual(summary["state"], "pending")

    def test_json_array_existing_production_batch_ids_not_reshuffled(self):
        rows = [{**row(i), "batch_id": "batch-a" if i < 3 else "batch-b"} for i in range(7)]
        path = self.root / "completed.json"
        path.write_text(json.dumps(rows))
        plans = review.read_batch_list(path)
        self.assertEqual([p["batch_id"] for p in plans], ["batch-a", "batch-b"])
        self.assertEqual([p["source_count"] for p in plans], [3, 4])
        self.assertEqual(plans[0]["source_ids"], [row(i)["id"] for i in range(3)])

    def test_common_native_transactions_record_real_finish_resume_without_model_redispatch(self):
        from types import SimpleNamespace
        class FakeLLM:
            def __init__(self, finish):
                self.calls = 0
                self.finish = finish
                self._ms_bar_model_sha256 = "fixture-only-no-model-loaded"
            def chat(self, messages, sampling, use_tqdm=False):
                self.calls += 1
                completion = SimpleNamespace(text=json.dumps(value()), token_ids=[1, 2, 3], finish_reason=self.finish, stop_reason=None)
                return [SimpleNamespace(outputs=[completion], prompt_token_ids=[6, 7])]
            def get_tokenizer(self):
                return SimpleNamespace(eos_token_id=99)
        for finish, expected in (("stop", "candidate"), ("length", "failed")):
            llm = FakeLLM(finish)
            root = self.root / finish
            request = {"model": "/fixture/not-loaded-model", "messages": [], "sampling": {"max_tokens": 100}, "json_schema": review.review_schema([0, 4, 8])}
            item = {"root": root, "request": request, "binding": {"CPU_fixture_no_GPU": True}, "validator": lambda v: review.validate_review(v, evidence())}
            args = SimpleNamespace(stage="run", batch_size=1)
            with patch.object(review.core, "structured_sampling", return_value=object()):
                result = review.bar.transactions([item], lambda: (llm, object), args)[str(root)]
                self.assertEqual(result["status"], expected)
                gen = review.read_json(root / "generation.json")
                self.assertEqual(gen["finish_reason"], finish)
                self.assertEqual(gen["completion_tokens"], 3)
                self.assertFalse(gen["last_output_token_is_eos"])
                again = review.bar.transactions([item], lambda: (_ for _ in ()).throw(AssertionError("terminal resume loaded model")), args)[str(root)]
                self.assertEqual(again, result)
            self.assertEqual(llm.calls, 1)
            self.assertTrue((root / "response.raw.txt").exists())
            self.assertTrue((root / "receipt.json").exists())
            self.assertFalse(result["human_review_completed"])

    def test_checkpoint_content_identity_ignores_deployment_root_and_weight_filenames(self):
        first = {"root": "/fixture/annotator", "files": {"model.safetensors": {"sha256": "a" * 64}, "config.json": {"sha256": "c" * 64}}}
        second = {"root": "/fixture/staged-reviewer", "files": {"renamed.safetensors": {"sha256": "a" * 64}, "config.json": {"sha256": "c" * 64}}}
        self.assertEqual(review.checkpoint_identity(first), review.checkpoint_identity(second))
        second["files"]["renamed.safetensors"]["sha256"] = "b" * 64
        self.assertNotEqual(review.checkpoint_identity(first), review.checkpoint_identity(second))
        with self.assertRaisesRegex(ValueError, "actual weight"):
            review.checkpoint_identity({"files": {"config.json": {"sha256": "a" * 64}}})

    def test_production_record_label_precedes_combined_target_text(self):
        r = row(1)
        r["target_text"] = "Dance style: hiphop" + chr(10) + "[0,2) seconds: Steps."
        self.assertEqual(review.label_payload(r)["dance_style"], "freestyle dance")
        self.assertTrue(review.label_complete(r))

    def test_single_batch_ready_and_published_snapshot_reader(self):
        members = [{**row(i), "batch_id": "train-b000000"} for i in range(3)]
        ready = {"batch_id": "train-b000000", "split": "train", "source_ids": [r["id"] for r in members], "records": members}
        ready_path = self.root / "batch_ready.json"
        ready_path.write_text(json.dumps(ready))
        direct = review.read_batch_list(ready_path)
        self.assertEqual(direct[0]["source_ids"], ready["source_ids"])
        self.assertEqual(direct[0]["batch_id"], "train-b000000")
        snapshots = self.root / "production" / "batch-snapshots"
        snapshots.mkdir(parents=True)
        snap = snapshots / "fixture.json"
        snap.write_text(json.dumps({"published": [{"path": str(ready_path), "sha256": review.batch.sha256(ready_path)}]}))
        self.assertEqual(review.read_batch_list(snap), direct)
        index = self.root / "manifest.summary.json"
        index.write_text(json.dumps({"batches": [{"batch_id": "train-b000000"}, {"batch_id": "train-b000001"}]}))
        fixed = snapshots.parent / "batches" / "train-b000000" / "batch_ready.json"
        fixed.parent.mkdir(parents=True)
        fixed.write_text(json.dumps(ready))
        self.assertEqual(review.read_batch_list(index, results_root=snapshots.parent), direct)

    def test_results_root_includes_all_fixed_ready_batches_despite_incremental_snapshots(self):
        production = self.root / "production"
        descriptors = []
        expected = []
        for batch_number in range(2):
            bid = f"train-b{batch_number:06d}"
            members = [row(10 * batch_number + i) for i in range(3)]
            obj = {"batch_id": bid, "split": "train", "source_ids": [r["id"] for r in members], "records": members}
            ready = production / "batches" / bid / "batch_ready.json"
            ready.parent.mkdir(parents=True)
            ready.write_text(json.dumps(obj))
            snapshots = production / "batch-snapshots"
            snapshots.mkdir(exist_ok=True)
            (snapshots / f"increment-{batch_number}.json").write_text(json.dumps({"published": [{"path": str(ready), "sha256": review.batch.sha256(ready)}]}))
            descriptors.append({"batch_id": bid, "split": "train", "count": 3})
            expected.append(bid)
        descriptors.append({"batch_id": "train-b000002", "split": "train", "count": 3})
        manifest = self.root / "summary.json"
        manifest.write_text(json.dumps({"batches": descriptors}))
        plans = review.read_batch_list(manifest, results_root=production)
        self.assertEqual([p["batch_id"] for p in plans], expected)
        self.assertEqual(sum(p["source_count"] for p in plans), 6)
        self.assertEqual(len(review.read_batch_list(snapshots / "increment-1.json")), 1)

    def test_attested_model_identity_wrapper_is_hash_checked_without_foreign_node_stat(self):
        ident = {"root": "/foreign-node/fixture", "files": {"model.safetensors": {"path": "/foreign-node/fixture/model.safetensors", "sha256": "a" * 64}}}
        ident["sha256"] = review.batch.digest({"root": ident["root"], "files": ident["files"]})
        ref = self.root / "reference.json"
        ref.write_text(json.dumps({"model_identity": ident, "source_sha_verified_once": True, "synthetic_CPU_inventory_fixture": True}))
        with patch.object(review.batch, "verify_current", side_effect=AssertionError("foreign source stat must not be called for attested reference")):
            found, receipt = review.read_model_reference(ref, permit_attested=True)
        self.assertEqual(found, ident)
        self.assertTrue(receipt["source_SHA_verified_attestation_used"])
        ident["files"]["model.safetensors"]["sha256"] = "b" * 64
        ref.write_text(json.dumps({"model_identity": ident, "source_sha_verified_once": True}))
        with self.assertRaisesRegex(ValueError, "inventory SHA mismatch"):
            review.read_model_reference(ref, permit_attested=True)

    def test_real_partial_claim_can_be_audited_but_not_training_eligible(self):
        rows = [row(1)]
        rows[0]["label_status"] = "partial"
        self.assertFalse(review.label_complete(rows[0]))
        self.assertTrue(review.label_reviewable(rows[0]))
        calls = []
        def fake(r, root, prepare_only=False):
            calls.append(r["id"])
            return {"source_id": r["id"], "status": "pass", "synthetic_CPU_fixture_not_model_review": True}
        result = review.review_batch_ready(rows, self.root, fake, batch_id="partial-audit", risk_count=0)
        self.assertEqual(calls, [rows[0]["id"]])
        self.assertTrue(result["acceptance"]["review_complete"])
        self.assertFalse(result["acceptance"]["eligible_for_training"])

    def test_source_ref_sha_mismatch_rejected(self):
        path = self.root / "input.json"
        path.write_bytes(b"CPU source provenance fixture")
        r = row(1)
        r["sourceRefs"] = {"manifest": {"path": str(path), "sha256": review.batch.sha256(path)}}
        refs = review.source_references(r)
        self.assertEqual(refs[0]["path"], str(path))
        path.write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            review.source_references(r)


if __name__ == "__main__":
    unittest.main()
