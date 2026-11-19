"""Real CPU/source evidence checks plus mock-only receipt control-flow tests.

All tests perform ZERO actual VLM calls. Mock pass is not semantic acceptance.
Only temporary outputs are written; frozen source artifacts are never edited.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import shutil
import os
import signal
import time
import importlib.util
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import oldbar_explicit_review_v1 as m

MANIFEST = Path(m.__file__).with_name("oldbar3_explicit_alternative_candidates_v1.jsonl")
FFPROBE = "/opt/ms_tools/ffprobe"
FFMPEG = "/usr/local/bin/ffmpeg"


class MockLLM:
    """Receipt plumbing only: never instantiate a real model/processor."""
    calls = 0

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=1)

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls += 1
        value = {"status": "pass", "errors": [], "uncertainties": [],
                 "coverage_note": "TEST ONLY: mock response, zero actual VLM calls; not a semantic verdict."}
        completion = SimpleNamespace(text=json.dumps(value), token_ids=[2, 1], finish_reason="stop", stop_reason=None)
        return [SimpleNamespace(outputs=[completion], prompt_token_ids=[3])]


class MockReviewer:
    def __init__(self, model_root, code):
        self.model = str(model_root)
        self.code = code
        self._identity = m.batch.identity_tree(model_root, [model_root / "weights.bin"])
        teacher_root = model_root.parent / "MOCK-ANNOTATOR"
        teacher_root.mkdir(exist_ok=True)
        (teacher_root / "teacher.bin").write_bytes(b"TEST FIXTURE: independently different mock teacher")
        self._annotator_identity = m.batch.identity_tree(teacher_root, [teacher_root / "teacher.bin"])
        # Mock control flow uses the frozen producer inventory identifier only;
        # this fixture is NOT an actual checkpoint/content-independence attestation.
        self._annotator_identity["sha256"] = "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c"
        self._annotator_reference_ref = {"TEST_ONLY": True, "sha256": self._annotator_identity["sha256"]}
        self.llm = MockLLM()
        self.calls = 0
        self._loaded = None

    def identity(self):
        return self._identity

    def __call__(self, row, root, *, prepare_only=False):
        self.calls += 1
        evidence = m.audit.read_json(Path(root) / "evidence.json")
        request = m.audit.review_request(row, evidence, root, self.model)
        binding = {**evidence["binding"], "evidence_sha256": m.batch.sha256(Path(root) / "evidence.json"),
                   "model_sha256": self.identity()["sha256"], "code_sha256": self.code["sha256"],
                   "reviewer_checkpoint_sha256": m.audit.checkpoint_identity(self.identity()),
                   "annotator_model_sha256": row["annotator_model_sha256"],
                   "checkpoint_content_independence_verified": True,
                   "annotator_model_reference": self._annotator_reference_ref,
                   "annotator_checkpoint_sha256": m.audit.checkpoint_identity(self._annotator_identity),
                   "TEST_ONLY_mock_not_actual_checkpoint_or_VLM": True}
        transaction = Path(root) / "review"
        validator = lambda v: m.audit.validate_review(v, evidence)
        prepared = m.core.transaction(transaction, request, None, None, validator, binding)
        if prepared["status"] == "prepared":
            output = self.llm.chat([request["messages"]], None)[0]
            self.llm._ms_bar_model_sha256 = self.identity()["sha256"]
            m.bar.persist_completion(transaction, request, output, output.outputs[0], self.llm)
        result = m.core.transaction(transaction, request, None, None, validator, binding)
        return {**result.get("json", {}), "status": result.get("json", {}).get("status", result["status"]), "source_id": row["id"]}


class ExplicitReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="test-oldbar-independent-")
        cls.root = Path(cls.tmp.name)
        cls.rows, cls.ref = m.load_manifest(MANIFEST)
        cls.before = m.batch.sha256(MANIFEST)
        cls.out = cls.root / "real-prepare"
        cls.summary = m.execute(cls.rows, cls.ref, cls.out, ffprobe=FFPROBE, ffmpeg=FFMPEG)
        cls.ledger = [json.loads(line) for line in (cls.out / "independent.ledger.jsonl").read_text().splitlines()]
        cls.refs = m.References()
        cls.probes = {r["id"]: m.validate_source(r, cls.refs, ffprobe=FFPROBE) for r in cls.rows}
        cls.first = cls.rows[0]
        cls.row = m.review_row(cls.first, cls.ref)
        cls.source = cls.out / "sources" / (cls.first["split"] + "-" + cls.first["id"]) / m.batch.digest(cls.row)[:24]
        cls.evidence = m.audit.read_json(cls.source / "evidence.json")
        cls.model = cls.root / "MOCK-NOT-A-MODEL"
        cls.model.mkdir()
        (cls.model / "weights.bin").write_bytes(b"TEST FIXTURE: not real model weights")
        cls.code = m.batch.identity_tree(Path(m.__file__).parent, [Path(m.__file__)])

    @classmethod
    def tearDownClass(cls):
        if m.batch.sha256(MANIFEST) != cls.before:
            raise AssertionError("frozen source manifest changed during tests")
        cls.tmp.cleanup()

    def fixture_manifest(self, rows, name="negative.jsonl"):
        path = self.root / name
        raw = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows)
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def changed(self):
        return copy.deepcopy(self.first)

    def test_real_three_sources_native_evidence_zero_VLM(self):
        self.assertEqual(self.summary["statuses"], {"prepared": 3})
        self.assertEqual(self.summary["population_source_count"], 3)
        self.assertEqual((self.summary["sample_count"], self.summary["risk_count"]), (3, 0))
        self.assertEqual((self.summary["native_source_frames"], self.summary["native_hand_rois"]), (48, 18))
        self.assertEqual(self.summary["sampled_reviewed_denominator"], 0)
        self.assertIsNone(self.summary["sampled_candidate_error_rate"])
        self.assertFalse(self.summary["GPU_model_load_performed"])
        self.assertFalse(self.summary["independent_sample_review_completed"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)
        for row in self.ledger:
            self.assertEqual((row["evidence_frames"], row["native_hand_rois"], row["actual_VLM_calls"]), (16, 6, 0))
            self.assertTrue(all(row[k] is False for k in m.FALSE_FLAGS))

    def test_native_actual_PTS_same_frame_and_dimensions(self):
        from PIL import Image
        for candidate in self.rows:
            row = m.review_row(candidate, self.ref)
            root = self.out / "sources" / (row["split"] + "-" + row["id"]) / m.batch.digest(row)[:24]
            evidence = m.audit.read_json(root / "evidence.json")
            m.verify_evidence(evidence, row, root, self.probes[row["id"]])
            for frame in evidence["frames"]:
                with Image.open(root / frame["file"]) as image:
                    self.assertEqual(image.size, (704, 1280))
            for crop in evidence["hand_rois"]:
                with Image.open(crop["path"]) as image:
                    self.assertEqual(list(image.size), crop["crop_pixel_size"])
                self.assertIn(crop["frame_index"], evidence["binding"]["frame_indices"])
                self.assertFalse(crop["resized"])

    def test_real_ROI_paths_are_per_source_not_production_mapping(self):
        roots = {r["id"]: Path(r["hand_roi"]["path"]).parent.parent.name for r in self.rows}
        self.assertEqual(roots["KMcaVVwQmswUfTuP"], "hand_roi_full12000_v1")
        self.assertEqual(roots["hwrlCvOCYUKpljlV"], "hand_roi_gpu_fixed3_v3")
        self.assertEqual(roots["qKgmlBaDdOeqvUZZ"], "hand_roi_gpu_fixed3_v3")

    def test_manifest_wrong_SHA_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            m.load_manifest(MANIFEST, manifest_sha256="0" * 64)

    def test_manifest_duplicate_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1] = copy.deepcopy(rows[0])
        path, sha = self.fixture_manifest(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m.load_manifest(path, manifest_sha256=sha)

    def test_partial_3_cannot_claim_complete500(self):
        with self.assertRaisesRegex(ValueError, "complete500"):
            m.load_manifest(MANIFEST, manifest_sha256=m.PILOT_SHA256, population_kind="complete500")

    def test_original_batch_impersonation_rejected(self):
        row = self.changed()
        row["batch_id"] = row["original_curated_batch_id"]
        with self.assertRaisesRegex(ValueError, "impersonate"):
            m.validate_claim(row)

    def test_original_batch_mapping_tamper_rejected(self):
        row = self.changed()
        row["original_curated_batch_id"] = "train-b000999"
        with self.assertRaisesRegex(ValueError, "batch mapping"):
            m.validate_claim(row)

    def test_approval_flags_strict_boolean_false(self):
        for key in m.FALSE_FLAGS:
            for wrong in (True, 0, None):
                row = self.changed()
                row[key] = wrong
                with self.subTest(key=key, wrong=wrong), self.assertRaises(ValueError):
                    m.validate_claim(row)

    def test_independent_review_must_be_required(self):
        row = self.changed()
        row["independent_review_required"] = False
        with self.assertRaisesRegex(ValueError, "fresh independent"):
            m.validate_claim(row)

    def test_historical_style_cannot_upgrade_SHA(self):
        row = self.changed()
        row["style_result"]["metadata"]["historical_video_sha256"] = row["source_binding"]["source_binding"]["video_sha256"]
        with self.assertRaisesRegex(ValueError, "historical style"):
            m.validate_claim(row)
        row = self.changed()
        row["source_refs"]["import_provenance"]["historical_video_binding"] = "current_SHA_verified"
        with self.assertRaisesRegex(ValueError, "historical style"):
            m.validate_claim(row)

    def test_historical_style_reference_path_tamper_rejected(self):
        row = self.changed()
        row["style_result"]["metadata"]["source_refs"]["historical_label_record"] = copy.deepcopy(self.rows[1]["source_refs"]["import_provenance"]["historical_style_label_record"])
        with self.assertRaisesRegex(ValueError, "historical style reference"):
            m.validate_claim(row)
        row = self.changed()
        row["style_result"]["metadata"]["historical_video_path"] = self.rows[1]["video_path"]
        with self.assertRaisesRegex(ValueError, "style source path"):
            m.validate_claim(row)

    def test_wrong_ROI_other_source_rejected(self):
        row = self.changed()
        row["hand_roi"] = copy.deepcopy(self.rows[1]["hand_roi"])
        with self.assertRaisesRegex(ValueError, "ROI references"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_wrong_source_hint_rejected(self):
        row = self.changed()
        row["source_hint"] = self.rows[1]["source_hint"]
        with self.assertRaisesRegex(ValueError, "origin_root/source_hint"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_wrong_origin_root_rejected(self):
        row = self.changed()
        row["origin_root"] = self.rows[1]["origin_root"]
        with self.assertRaisesRegex(ValueError, "origin_root/source_hint"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_wrong_current_media_SHA_rejected(self):
        row = self.changed()
        row["source_refs"]["import_provenance"]["current_video"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_wrong_original_source_binding_rejected(self):
        row = self.changed()
        row["source_binding"]["source_binding"]["video_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_binding"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_changed_label_or_target_rejected(self):
        row = self.changed()
        row["label"]["timed_motion_script"] += " Modified."
        with self.assertRaisesRegex(ValueError, "target text"):
            m.validate_claim(row)
        row["target_text"] = "Dance style: " + row["label"]["dance_style"] + "\n" + row["label"]["timed_motion_script"]
        row["target_text_sha256"] = m.text_sha(row["target_text"])
        row["original_timed_motion_sha256"] = m.text_sha(row["label"]["timed_motion_script"])
        with self.assertRaisesRegex(ValueError, "original MS verbatim"):
            m.validate_source(row, self.refs, ffprobe=FFPROBE)

    def test_semantic_request_excludes_producer_reasoning_and_scores(self):
        row = m.review_row(self.first, self.ref)
        request = m.audit.review_request(row, self.evidence, self.source, self.model)
        claim = json.loads(request["messages"][1]["content"][0]["text"])
        self.assertEqual(claim["label_to_test"], self.first["label"])
        serialized = request["messages"][0]["content"] + " ".join(item["text"] for item in request["messages"][1]["content"] if item["type"] == "text")
        for forbidden in ("historical_label", "genre_conf", "producer_score", "window_observations", "source_hint", "NLF"):
            self.assertNotIn(forbidden, serialized)

    def test_prepare_resume_reuses_native_evidence_without_decoding(self):
        with patch.object(m.audit.subprocess, "run", side_effect=AssertionError("unexpected ffmpeg decode")):
            # probe_actual lives in ms_landmarks, not audit.subprocess.run;
            # avoid intercepting its legitimate CPU ffprobe subprocess.
            with patch.object(m.audit.bar, "probe_actual", return_value=self.probes[self.first["id"]]):
                evidence = m.audit.prepare_evidence(self.row, self.source, ffprobe=FFPROBE, ffmpeg=FFMPEG, hand_roi_root=self.first["hand_roi"]["path"])
        self.assertEqual(evidence, self.evidence)

    def test_mock_six_receipt_resume_no_repeated_call_not_semantic_verdict(self):
        reviewer = MockReviewer(self.model, self.code)
        result = reviewer(self.row, self.source)
        self.assertEqual(result["status"], "pass")  # Mock ONLY, no semantic acceptance.
        checked, ref = m.verify_review_receipt(self.source, self.row, self.evidence, reviewer)
        self.assertIn("TEST ONLY", checked["coverage_note"])
        self.assertEqual(reviewer.llm.calls, 1)
        reviewer(self.row, self.source)
        m.verify_review_receipt(self.source, self.row, self.evidence, reviewer)
        self.assertEqual(reviewer.llm.calls, 1)
        self.assertTrue(all(checked.get(k, False) is False for k in m.FALSE_FLAGS))
        self.assertEqual(set(m.audit.read_json(ref["path"])["files_sha256"]), m.SIX_FILES)
        transaction = self.source / "review"
        for name in sorted(m.SIX_FILES):
            path = transaction / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\nTAMPER")
                with self.subTest(artifact=name), self.assertRaises(ValueError):
                    m.verify_review_receipt(self.source, self.row, self.evidence, reviewer)
            finally:
                path.write_bytes(original)
        receipt_path = transaction / "receipt.json"
        original = receipt_path.read_bytes()
        receipt = json.loads(original)
        del receipt["files_sha256"]["generation.json"]
        try:
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "six receipt"):
                m.verify_review_receipt(self.source, self.row, self.evidence, reviewer)
        finally:
            receipt_path.write_bytes(original)

    def test_execute_partial_and_failed_transactions_never_redispatch(self):
        for filename in ("request.json", "response.raw.txt", "generation.json", "receipt.json"):
            out = self.root / ("partial-" + filename)
            target = out / "sources" / (self.first["split"] + "-" + self.first["id"]) / m.batch.digest(self.row)[:24]
            shutil.copytree(self.source, target, ignore=shutil.ignore_patterns("review", "explicit.review.receipt.json"))
            transaction = target / "review"
            transaction.mkdir()
            (transaction / filename).write_text(json.dumps({"status": "failed"}) if filename == "receipt.json" else "partial")
            reviewer = MockReviewer(self.model, self.code)
            with patch.object(m, "validate_source", return_value=self.probes[self.first["id"]]):
                summary = m.execute([self.first], self.ref, out, prepare_only=False, reviewer=reviewer, ffprobe=FFPROBE, ffmpeg=FFMPEG)
            self.assertEqual(summary["statuses"], {"pending": 1})
            self.assertEqual(reviewer.calls, 0)
            self.assertEqual(reviewer.llm.calls, 0)
            self.assertEqual(summary["sampled_reviewed_denominator"], 0)

    def test_execute_valid_candidate_resume_direct_result_no_reviewer_call(self):
        out = self.root / "valid-candidate-resume"
        target = out / "sources" / (self.first["split"] + "-" + self.first["id"]) / m.batch.digest(self.row)[:24]
        shutil.copytree(self.source, target, ignore=shutil.ignore_patterns("review", "explicit.review.receipt.json"))
        reviewer = MockReviewer(self.model, self.code)
        reviewer(self.row, target)
        self.assertEqual(reviewer.calls, 1)
        with patch.object(m, "validate_source", return_value=self.probes[self.first["id"]]):
            summary = m.execute([self.first], self.ref, out, prepare_only=False, reviewer=reviewer, ffprobe=FFPROBE, ffmpeg=FFMPEG)
        self.assertEqual(summary["statuses"], {"pass": 1})  # Mock ONLY.
        self.assertEqual(reviewer.calls, 1)
        self.assertEqual(reviewer.llm.calls, 1)
        self.assertFalse(summary["GPU_model_load_performed"])
        self.assertFalse(summary["training_eligible"])

    def test_mock_pass_never_approves_training_or_original_population(self):
        sampling = m.audit.sampling_plan(self.rows, self.ref["import_batch_id"], sample_count=3, risk_count=0)
        results = {r["id"]: {"status": "pass", "MOCK_ONLY_zero_actual_VLM_calls": True} for r in self.rows}
        summary = m.aggregate(self.rows, list(results), results, self.ref, sampling, prepare_only=False)
        self.assertTrue(summary["independent_sample_review_completed"])
        for key in (*m.FALSE_FLAGS, "eligible_for_training", "original_curated_batches_completed", "full_batch_semantic_pass_claimed"):
            self.assertFalse(summary[key])
        self.assertEqual(summary["population_source_count"], 3)

    def test_failure_denominator_preserves_pending_and_500_policy(self):
        sampling = m.audit.sampling_plan(self.rows, self.ref["import_batch_id"], sample_count=3, risk_count=0)
        ids = sampling["selected_source_ids"]
        results = {ids[0]: {"status": "confirmed_error"}, ids[1]: {"status": "pending"}, ids[2]: {"status": "pass"}}
        summary = m.aggregate(self.rows, ids, results, self.ref, sampling, prepare_only=False)
        self.assertEqual(summary["sampled_reviewed_denominator"], 2)
        self.assertEqual(summary["confirmed_error_count"], 1)
        self.assertEqual(summary["sampled_candidate_error_rate"], .5)
        self.assertFalse(summary["independent_sample_review_completed"])
        synthetic = [{"source_id": "fixture" + str(i), "label": {"dance_style": "mock", "timed_motion_script": "hand"}} for i in range(500)]
        plan = m.audit.sampling_plan(synthetic, "MOCK500", sample_count=25, risk_count=5)
        self.assertEqual(len(plan["random_source_ids"]), 25)
        self.assertEqual(len(plan["risk_source_ids"]), 5)
        self.assertEqual(len(plan["selected_source_ids"]), 30)

    def test_validation_failure_is_per_source_pending_not_fake_pass(self):
        out = self.root / "injected-validation-failure"
        def validation(candidate, refs, *, ffprobe):
            if candidate["id"] == self.first["id"]:
                raise ValueError("TEST injected source failure")
            return self.probes[candidate["id"]]
        with patch.object(m, "validate_source", side_effect=validation):
            summary = m.execute(self.rows, self.ref, out, ffprobe=FFPROBE, ffmpeg=FFMPEG)
        self.assertEqual(summary["statuses"], {"prepared": 2, "pending": 1})
        self.assertEqual(summary["sampled_reviewed_denominator"], 0)
        self.assertEqual(summary["pending_source_ids"], summary["selected_source_ids"])
        self.assertFalse(summary["training_eligible"])


class ResourceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rt = Path(m.__file__).parent
        def load(name):
            spec = importlib.util.spec_from_file_location(name, rt / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        cls.manager = load("oldbar_explicit_review_resource_manager_v1")
        cls.builder = load("prepare_oldbar_explicit_review_v1")

    def capture(self, *, extra=False, ignored=False, headroom=None):
        base = self.manager.base
        owner = dict(pid=123456, uid=os.getuid(), argv=["python", str(base.HOLDER)], start_ticks=456,
                     cgroup="TEST-MOCK", CVD=["CUDA_VISIBLE_DEVICES=3"], exe="/TEST-MOCK/python",
                     sig_ign=(1 << (signal.SIGTERM - 1)) if ignored else 0, sig_blk=0,
                     gpu=3, uuid="GPU-TEST-MOCK")
        active = [dict(pid=owner["pid"], uuid=owner["uuid"], used_MiB="1")]
        if extra:
            active += [dict(pid=9876, uuid=owner["uuid"], used_MiB="1")]
        fd = os.open('/dev/null', os.O_RDONLY)
        try:
            with patch.object(self.builder.subprocess, "check_output", return_value="28.216.19.225"), \
                 patch.object(base, "apps", return_value=active), \
                 patch.object(base, "identity", return_value=owner), \
                 patch.object(base, "uuid_for", return_value=owner["uuid"]), \
                 patch.object(base, "lease_snapshot", return_value=None), \
                 patch.object(base, "headroom", return_value={"headroom": headroom if headroom is not None else 160 * 2**30}), \
                 patch.object(self.builder.os, "pidfd_open", return_value=fd), \
                 patch.object(self.builder.select, "select", return_value=([], [], [])), \
                 patch.object(Path, "exists", return_value=False), patch.object(Path, "is_symlink", return_value=False):
                return self.builder.fresh_capture(self.manager)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def test_builder_capture_only_normal_exactholder_no_signals(self):
        with patch.object(self.manager.base, "release_holder", side_effect=AssertionError("signal forbidden")):
            snapshot = self.capture()
        self.assertEqual(snapshot["gpus"], [3])
        self.assertFalse(snapshot["signals_sent"])
        self.assertFalse(snapshot["gpu_started"])
        self.assertFalse(snapshot["leases_modified"])

    def test_builder_ignored_holder_or_unknown_active_not_free(self):
        with self.assertRaisesRegex(RuntimeError, "SIGTERM ignored"):
            self.capture(ignored=True)
        with self.assertRaisesRegex(RuntimeError, "one exact project holder"):
            self.capture(extra=True)

    def test_builder_original_160GiB_headroom_unchanged(self):
        self.assertEqual(self.manager.base.required_budget({}), 160 * 2**30)
        with self.assertRaisesRegex(RuntimeError, "160GiB"):
            self.capture(headroom=160 * 2**30 - 1)

    def test_process_SHA_cache_first_full_and_stat_change_invalidates(self):
        with tempfile.TemporaryDirectory(prefix="oldbar-manager-cache-") as tmp:
            path = Path(tmp) / "mock.bin"
            path.write_bytes(b"first")
            with patch.object(self.manager, "_original_sha", wraps=self.manager._original_sha) as full:
                first = self.manager.cached_sha(path)
                self.assertEqual(self.manager.cached_sha(path), first)
                self.assertEqual(full.call_count, 1)
                path.write_bytes(b"second-longer")
                self.assertNotEqual(self.manager.cached_sha(path), first)
                self.assertEqual(full.call_count, 2)

    def test_manager_full_model_content_firsthash_and_tamper_rejection(self):
        manager = self.manager
        with tempfile.TemporaryDirectory(prefix="mock-oldbar-model-") as tmp:
            root = Path(tmp)
            model = root / "MOCK-not-real-model"
            model.mkdir()
            for name in ("config.json", "part1.safetensors", "part2.safetensors"):
                (model / name).write_bytes(("MOCK content " + name).encode())
            identity = m.batch.identity_tree(model, list(model.iterdir()))
            stage = root / "stage.json"
            stage.write_text(json.dumps({"model_identity": identity}))
            teacher = {"sha256": "26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c",
                       "files": {"teacher.bin": {"sha256": "a" * 64}}}
            cfg = {"modelref": {"path": str(stage), "sha256": manager.cached_sha(stage), "model_identity_sha256": identity["sha256"]}}
            with patch.object(manager, "MODEL", model), patch.object(manager, "STAGE", stage), \
                 patch.object(manager, "STAGE_SHA", cfg["modelref"]["sha256"]), \
                 patch.object(manager.support, "check_model"), \
                 patch.object(manager.worker.audit, "read_model_reference", return_value=(teacher, {})), \
                 patch.object(manager, "_original_sha", wraps=manager._original_sha) as full:
                manager.check_model(cfg)
                self.assertEqual(full.call_count, 3)
                manager.check_model(cfg)
                self.assertEqual(full.call_count, 3)
                (model / "part2.safetensors").write_bytes(b"changed model fixture")
                with self.assertRaisesRegex(RuntimeError, "full checkpoint content SHA"):
                    manager.check_model(cfg)
                self.assertEqual(full.call_count, 4)

    def test_manager_scope_stale_capture_and_lease_are_rejected(self):
        manager = self.manager
        with tempfile.TemporaryDirectory(prefix="oldbar-manager-scope-") as tmp:
            capture = Path(tmp) / "capture.json"
            snap = dict(node='N6', gpus=[3], holders=[dict(gpu=3)], captured_unix_ns=time.time_ns(),
                        signals_sent=False, gpu_started=False, leases_modified=False, lease=None)
            capture.write_text(json.dumps(snap))
            options = {'--manifest': manager.MANIFEST, '--out': manager.OUT, '--model': manager.MODEL,
                       '--model-reference': manager.STAGE, '--annotator-model-reference': manager.ANNOTATOR,
                       '--ffprobe': '/opt/ms_tools/ffprobe', '--ffmpeg': '/usr/local/bin/ffmpeg'}
            argv = ['/opt/venv_vllm/bin/python', '-B', '-u', str(Path(m.__file__)), '--run']
            for k, v in options.items():
                argv.extend([k, str(v)])
            cfg = dict(node='N6', gpus=[3], release_authorized=True, phase=manager.PHASE,
                       input_count=222939, targetcount=3, queue=str(manager.QUEUE), out=str(manager.OUT),
                       env={'CUDA_VISIBLE_DEVICES': '3', 'MS_VISUAL_CPU_SITEPATH': '/opt/venv_eval/lib/python3.11/site-packages', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2'}, cwd='/tmp', source_selection=dict(shard=0, nshard=1, limit=3),
                       holders=snap['holders'], semantic_approval=False, training_entry_updated=False,
                       resource_manager_ref={'path': manager.__file__, 'sha256': manager.cached_sha(manager.__file__)},
                       argv=argv, producer={'path': str(Path(m.__file__))},
                       holder_capture={'path': str(capture), 'sha256': manager.cached_sha(capture)},
                       plan={'path': str(manager.MANIFEST), 'sha256': m.PILOT_SHA256},
                       ready_source_refs=[], source_sha256={})
            with patch.object(manager.subprocess, "check_output", return_value='28.216.19.225'), \
                 patch.object(manager.base, "lease_snapshot", return_value=None), \
                 patch.object(manager, "source_references", return_value=([], [])):
                manager.validate(cfg)
                for key, wrong in [('gpus', [2]), ('input_count', 3), ('targetcount', 500), ('release_authorized', False), ('semantic_approval', True), ('argv', ['/opt/venv_eval/bin/python', *argv[1:]]), ('env', {'CUDA_VISIBLE_DEVICES': '3', 'PYTHONPATH': '/opt/venv_eval/lib/python3.11/site-packages'})]:
                    changed = copy.deepcopy(cfg)
                    changed[key] = wrong
                    with self.subTest(key=key), self.assertRaises(RuntimeError):
                        manager.validate(changed)
                changed = copy.deepcopy(cfg)
                changed['argv'][0] = '/opt/venv_eval/bin/python'
                with self.assertRaisesRegex(RuntimeError, 'Native vLLM worker'):
                    manager.validate(changed)
                changed = copy.deepcopy(cfg)
                changed['env']['PYTHONPATH'] = '/opt/venv_eval/lib/python3.11/site-packages'
                with self.assertRaisesRegex(RuntimeError, 'Native vLLM env'):
                    manager.validate(changed)
                snap['captured_unix_ns'] -= 1201 * 10**9
                capture.write_text(json.dumps(snap))
                cfg['holder_capture']['sha256'] = manager.cached_sha(capture)
                with self.assertRaisesRegex(RuntimeError, "fresh exact"):
                    manager.validate(cfg)

    def test_wrapper_does_not_replace_original_lease_or_release_logic(self):
        import ast
        source = ast.parse(Path(self.manager.__file__).read_text())
        assigned = [ast.unparse(n.targets[0]) for n in source.body if isinstance(n, ast.Assign)]
        for forbidden in ("base.release_holder", "base.acquire_lease", "base.required_budget", "base.restore_step", "base.main"):
            self.assertNotIn(forbidden, assigned)
        self.assertEqual(self.manager.base.SAFETY + self.manager.base.LOAD, 160 * 2**30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
