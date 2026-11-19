"""Actual multi-conversation CPU tests; models/media preparation are synthetic."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ms_fullframe_vlm_postprocess_batch_v4 as v


class FakeLLM:
    def __init__(self, mutate=None, batch_error=None, wrong_count=False, reorder=False, blank=False):
        self.calls = []
        self.mutate, self.batch_error = mutate, batch_error
        self.wrong_count, self.reorder, self.blank = wrong_count, reorder, blank

    def chat(self, conversations, sampling_params, use_tqdm):
        assert isinstance(conversations, list) and all(isinstance(x, list) for x in conversations)
        assert isinstance(sampling_params, list) and len(sampling_params) == len(conversations)
        outputs, stage = [], None
        for position, (request, grammar) in enumerate(zip(conversations, sampling_params)):
            content = request[1]["content"]
            payload = json.loads(content[0]["text"] if isinstance(content, list) else content)
            assert grammar == payload["required_json_schema"]
            stage = "visual_audit" if isinstance(content, list) else "caption_fusion"
            row = dict(schema="ms_fullframe_" + stage + "_v2", id=payload["id"], split=payload["split"])
            if stage == "visual_audit":
                row.update(visual_motion_summary="A visible arm pose differs.",
                    supported_actions=[] if self.blank else [dict(description="One side raises an arm in the image.", pair_id=0)],
                    unsupported_claims=[], missing_evidence=["Unsampled intervals remain unknown."])
            else:
                row.update(fused_caption="" if self.blank else "One side raises an arm in the image.",
                           corrections=[], limitations=["Only sampled endpoint poses are supported."])
            candidate = SimpleNamespace(text=json.dumps(row), token_ids=list(range(30)), finish_reason="stop")
            if self.mutate:
                self.mutate(stage, position, row, candidate)
            outputs.append(SimpleNamespace(outputs=[candidate]))
        self.calls.append(dict(stage=stage, sources=len(conversations), requests=copy.deepcopy(conversations)))
        if self.batch_error == stage:
            raise RuntimeError("Synthetic invocation failure; do not retry")
        if self.reorder:
            outputs.reverse()
        return outputs[:-1] if self.wrong_count else outputs


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ms-batch-v4-cpu-", dir="/tmp")
        self.root = Path(self.tmp.name)
        self.gates = []

    def tearDown(self):
        self.tmp.cleanup()

    def setup_sources(self, count, batch=8):
        rows, manifests = [], {}
        for i in range(count):
            id_ = "Source%03d" % i
            directory = self.root / "inputs" / id_
            manifest_path = directory / "manifest.json"
            video = self.root / (id_ + ".mp4")
            manifest = dict(schema="ms_fullframe_pose2d_manifest_v1", id=id_, split="train",
                video_path=str(video), video_sha256="a" * 64, frame_count=12, frame_indices=list(range(12)),
                times=[i * .04 for i in range(12)], source_timestamps=[10 + i * .04 for i in range(12)],
                pts_origin=10., width=1280, height=704, duration_seconds=.48,
                **dict(list(v.FLAGS.items())[:4]))
            manifest_ref = v.write_once(manifest_path, manifest)
            binding = dict(schema="ms_fullframe_pose2d_binding_v1", observations_sha256="b" * 64)
            payload = dict(schema=binding["schema"], fullframe_manifest=manifest,
                           observations_sha256=binding["observations_sha256"])
            binding["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            draft = dict(schema="ms_fullframe_pose2d_rule_draft_v2", id=id_, split="train", ms_text="Observed pose",
                source=dict(fullframe_manifest_path=str(manifest_path), video_path=str(video),
                            video_sha256=manifest["video_sha256"], fullframe_binding=binding),
                **dict(list(v.FLAGS.items())[:4]))
            draft_ref = v.write_once(directory / "draft.json", draft)
            row = dict(id=id_, split="train", video_path=str(video), rule_draft_ref=draft_ref,
                       fullframe_manifest_ref=manifest_ref)
            rows.append(row)
            manifests[id_] = manifest
        cfg = dict(out=str(self.root / "out"), targetcount=count, input_count=count,
                   selected_source_ids=[r["id"] for r in rows], batch_sources=batch,
                   plan=dict(path=str(self.root / "plan.jsonl"), sha256="c" * 64),
                   modelref=dict(model_identity_sha256="d" * 64, sha256="e" * 64), ffmpeg_path="synthetic-not-executed")
        v.write_once(Path(cfg["out"]) / "load.receipt.json", dict(model_identity_sha256="d" * 64,
            source_reference_sha256="e" * 64, model_init_count=1, model_load_completed=True,
            model_root=v.MODEL, dtype="bfloat16", tensor_parallel_size=2))
        return cfg, rows, manifests

    def images(self, row, manifest, local, ffmpeg):
        image = local / "frame-001.jpg"
        image.write_bytes(b"\xff\xd8syntheticJPEG\xff\xd9")
        return v.select_frames(manifest), [image]

    def run_case(self, count=8, batch=8, model=None, fail_prepare=None, before=None):
        cfg, rows, manifests = self.setup_sources(count, batch)
        llm = model or FakeLLM()
        if before:
            before(cfg, rows)
        def stage(row, manifest, local, ffmpeg):
            if fail_prepare == row["id"]:
                raise ValueError("Synthetic media preparation failure")
            return self.images(row, manifest, local, ffmpeg)
        with patch.object(v, "rows_for", return_value=rows), patch.object(v, "gate", side_effect=lambda *a, **k:self.gates.append(True)), patch.object(v, "stage_images", side_effect=stage):
            summary = v.run_sources(cfg, llm, lambda grammar:grammar)
            replay = v.summaries(cfg, Path(cfg["out"]))
        self.assertEqual(replay[str(Path(cfg["out"]) / "summary.json")]["processed"], count)
        self.assertTrue(self.gates)
        self.assertEqual(summary["new_model_calls"], sum(len(v.checked(x["result_ref"])["stages"]) for x in summary["all_outcomes"]))
        for x in summary["all_outcomes"]:
            self.assertLessEqual(x["new_model_calls"], 2)
            self.assertFalse(x["quality_approved"])
            self.assertFalse(x["training_eligible"])
        return cfg, rows, llm, summary

    def test_eight_sources_are_two_actual_chat_invocations(self):
        c, r, llm, s = self.run_case()
        self.assertEqual([x["sources"] for x in llm.calls], [8, 8])
        self.assertEqual(s["batch_chat_invocations"], 2)
        self.assertEqual(s["new_model_calls"], 16)
        self.assertEqual(s["structure_grounded_candidate"], 8)
        for i, item in enumerate(s["all_outcomes"]):
            result = v.checked(item["result_ref"])
            self.assertEqual([x["batch_position"] for x in result["stages"]], [i, i])
            self.assertEqual(len(result["selected_frames"]), 12)

    def test_four_sources_batch_and_short_tail(self):
        c, r, llm, s = self.run_case(count=6, batch=4)
        self.assertEqual([x["sources"] for x in llm.calls], [4, 4, 2, 2])
        self.assertEqual(s["new_model_calls"], 12)
        self.assertEqual(s["batch_chat_invocations"], 4)

    def test_eight_batch_short_tail(self):
        c, r, llm, s = self.run_case(count=9)
        self.assertEqual([x["sources"] for x in llm.calls], [8, 8, 1, 1])

    def test_one_truncated_visual_does_not_discard_other_sources(self):
        def mutate(stage, i, row, candidate):
            if stage == "visual_audit" and i == 2:
                candidate.finish_reason = "length"
                candidate.token_ids = list(range(1800))
        c, r, llm, s = self.run_case(model=FakeLLM(mutate=mutate))
        self.assertEqual([x["sources"] for x in llm.calls], [8, 7])
        self.assertEqual(s["all_outcomes"][2]["status"], "failed_sealed_no_retry")
        self.assertEqual(s["all_outcomes"][2]["new_model_calls"], 1)
        self.assertEqual(s["structure_grounded_candidate"], 7)

    def test_visual_call_exception_consumes_one_stage_without_retry(self):
        c, r, llm, s = self.run_case(model=FakeLLM(batch_error="visual_audit"))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(s["new_model_calls"], 8)
        self.assertTrue(all(x["status"] == "failed_sealed_no_retry" for x in s["all_outcomes"]))

    def test_text_call_exception_consumes_two_stages_without_retry(self):
        c, r, llm, s = self.run_case(model=FakeLLM(batch_error="caption_fusion"))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(s["new_model_calls"], 16)
        self.assertTrue(all(x["status"] == "failed_sealed_no_retry" for x in s["all_outcomes"]))

    def test_missing_batch_outputs_are_sealed_not_retried(self):
        c, r, llm, s = self.run_case(model=FakeLLM(wrong_count=True))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(s["new_model_calls"], 8)
        self.assertEqual(s["structure_grounded_candidate"], 0)

    def test_reordered_batch_outputs_cannot_cross_source_identity(self):
        c, r, llm, s = self.run_case(model=FakeLLM(reorder=True))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(s["structure_grounded_candidate"], 0)
        self.assertTrue(all(x["status"] == "failed_sealed_no_retry" for x in s["all_outcomes"]))

    def test_wrong_source_id_in_one_output_is_rejected(self):
        def mutate(stage, i, row, candidate):
            if stage == "visual_audit" and i == 1:
                row["id"] = "Wrong"
                candidate.text = json.dumps(row)
        c, r, llm, s = self.run_case(model=FakeLLM(mutate=mutate))
        self.assertEqual([x["sources"] for x in llm.calls], [8, 7])
        self.assertEqual(s["all_outcomes"][1]["status"], "failed_sealed_no_retry")

    def test_one_prepare_failure_is_excluded_from_both_model_batches(self):
        c, r, llm, s = self.run_case(fail_prepare="Source002")
        self.assertEqual([x["sources"] for x in llm.calls], [7, 7])
        self.assertEqual(s["all_outcomes"][2]["new_model_calls"], 0)

    def test_existing_source_claim_is_not_consumed_again(self):
        def before(cfg, rows):
            v.write_once(Path(cfg["out"]) / "budgets/train/Source000/source.claim.json", {"already_consumed": True})
        c, r, llm, s = self.run_case(before=before)
        self.assertEqual([x["sources"] for x in llm.calls], [7, 7])
        self.assertEqual(s["all_outcomes"][0]["status"], "existing_claim_not_reassigned")
        self.assertEqual(s["all_outcomes"][0]["new_model_calls"], 0)

    def test_existing_stage_claim_is_not_retried(self):
        def before(cfg, rows):
            v.write_once(Path(cfg["out"]) / "budgets/train/Source000/visual_audit.claim.json", {"already_consumed": True})
        c, r, llm, s = self.run_case(before=before)
        self.assertEqual([x["sources"] for x in llm.calls], [7, 7])
        self.assertEqual(s["all_outcomes"][0]["new_model_calls"], 0)
        self.assertEqual(s["all_outcomes"][0]["status"], "failed_sealed_no_retry")

    def test_empty_visual_evidence_has_empty_caption(self):
        c, r, llm, s = self.run_case(model=FakeLLM(blank=True))
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(x["status"] == "insufficient_visible_evidence" for x in s["all_outcomes"]))

    def test_one_truncated_text_preserves_seven_normal_sources(self):
        def mutate(stage, i, row, candidate):
            if stage == "caption_fusion" and i == 3:
                candidate.finish_reason = "length"
        c, r, llm, s = self.run_case(model=FakeLLM(mutate=mutate))
        self.assertEqual(s["structure_grounded_candidate"], 7)
        self.assertEqual(s["all_outcomes"][3]["new_model_calls"], 2)

    def test_batch_response_receipt_tampering_is_rejected(self):
        c, rows, llm, s = self.run_case(count=4, batch=4)
        path = Path(s["batch_invocations"][0]["response_ref"]["path"])
        path.write_bytes(b"{}")
        with patch.object(v, "rows_for", return_value=rows):
            with self.assertRaises(ValueError):
                v.summaries(c, Path(c["out"]))

    def test_batch_size_outside_four_eight_is_rejected(self):
        for size in (True, 1, 2, 16):
            with self.assertRaises(ValueError):
                v.run_sources({"batch_sources": size}, FakeLLM(), lambda x:x)

    def test_original_native_resource_functions_and_recipe_unchanged(self):
        old = ast.parse((v.RT / "ms_fullframe_vlm_postprocess_v3.py").read_text())
        new = ast.parse(Path(v.__file__).read_text())
        functions = lambda tree:{x.name:ast.dump(x, include_attributes=False) for x in tree.body if isinstance(x, ast.FunctionDef)}
        a, b = functions(old), functions(new)
        for name in ("gate", "previous_safe", "validate", "actual_controller_headroom", "runtime", "actual_model", "worker", "main", "stage_images", "request_for", "response_schema", "audit_check", "evidence_policy_violations"):
            self.assertEqual(a[name], b[name], name)
        self.assertEqual(v.BASE_SHA, "6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6")

    def test_model_runtime_is_not_loaded_in_CPU_suite(self):
        with patch.object(v, "runtime", side_effect=AssertionError("No actual model in CPU test")):
            c, r, llm, s = self.run_case(count=4, batch=4)
            self.assertEqual(s["new_model_calls"], 8)


    def test_failed_error_only_raw_receipt_tampering_is_rejected(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4, model=FakeLLM(batch_error="visual_audit"))
        result = v.checked(summary["all_outcomes"][0]["result_ref"])
        self.assertNotIn("raw_text", result["stages"][0])
        path = Path(cfg["out"]) / "sources/train/Source000/visual_audit.raw.json"
        path.write_bytes(b"{}")
        with patch.object(v, "rows_for", return_value=rows), self.assertRaises(ValueError):
            v.summaries(cfg, Path(cfg["out"]))

    def test_failed_truncated_raw_receipt_tampering_is_rejected(self):
        def mutate(stage, i, row, candidate):
            if stage == "visual_audit" and i == 0:
                candidate.finish_reason = "length"
        cfg, rows, llm, summary = self.run_case(count=4, batch=4, model=FakeLLM(mutate=mutate))
        path = Path(cfg["out"]) / "sources/train/Source000/visual_audit.raw.json"
        raw = v.load(path)
        raw["finish_reason"] = "stop"
        path.write_bytes(v.h.bench.encoded(raw))
        with patch.object(v, "rows_for", return_value=rows), self.assertRaises(ValueError):
            v.summaries(cfg, Path(cfg["out"]))

    def test_invalid_json_is_sealed_without_discarding_batch(self):
        def mutate(stage, i, row, candidate):
            if stage == "visual_audit" and i == 2:
                candidate.text = "not JSON"
        cfg, rows, llm, summary = self.run_case(model=FakeLLM(mutate=mutate))
        self.assertEqual([x["sources"] for x in llm.calls], [8, 7])
        self.assertEqual(summary["all_outcomes"][2]["status"], "failed_sealed_no_retry")

    def test_multiple_candidates_seal_only_affected_source(self):
        class MultipleLLM(FakeLLM):
            def chat(self, *args, **kwargs):
                outputs = super().chat(*args, **kwargs)
                if self.calls[-1]["stage"] == "visual_audit":
                    outputs[0].outputs.append(copy.deepcopy(outputs[0].outputs[0]))
                return outputs
        cfg, rows, llm, summary = self.run_case(model=MultipleLLM())
        self.assertEqual([x["sources"] for x in llm.calls], [8, 7])
        self.assertEqual(summary["all_outcomes"][0]["new_model_calls"], 1)

    def test_zero_candidates_seal_only_affected_source(self):
        class EmptyLLM(FakeLLM):
            def chat(self, *args, **kwargs):
                outputs = super().chat(*args, **kwargs)
                if self.calls[-1]["stage"] == "visual_audit":
                    outputs[0].outputs = []
                return outputs
        cfg, rows, llm, summary = self.run_case(model=EmptyLLM())
        self.assertEqual(summary["structure_grounded_candidate"], 7)

    def test_second_run_never_calls_model_or_overwrites_sealed_result(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4)
        original = Path(summary["all_outcomes"][0]["result_ref"]["path"]).read_bytes()
        model = FakeLLM()
        with patch.object(v, "rows_for", return_value=rows), patch.object(v, "gate"), self.assertRaises(FileExistsError):
            v.run_sources(cfg, model, lambda x:x)
        self.assertEqual(model.calls, [])
        self.assertEqual(Path(summary["all_outcomes"][0]["result_ref"]["path"]).read_bytes(), original)

    def test_source_budget_tampering_rejected(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4)
        path = Path(cfg["out"]) / "budgets/train/Source000/source.claim.json"
        budget = v.load(path)
        budget["automatic_retry"] = True
        path.write_bytes(v.h.bench.encoded(budget))
        with patch.object(v, "rows_for", return_value=rows), self.assertRaises(ValueError):
            v.summaries(cfg, Path(cfg["out"]))

    def test_stage_budget_tampering_rejected_even_for_failed_call(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4, model=FakeLLM(batch_error="caption_fusion"))
        path = Path(cfg["out"]) / "budgets/train/Source000/caption_fusion.claim.json"
        budget = v.load(path)
        budget["maximum_calls"] = 2
        path.write_bytes(v.h.bench.encoded(budget))
        with patch.object(v, "rows_for", return_value=rows), self.assertRaises(ValueError):
            v.summaries(cfg, Path(cfg["out"]))

    def test_failed_caption_normalized_evidence_tampering_rejected(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4, model=FakeLLM(batch_error="caption_fusion"))
        path = Path(cfg["out"]) / "sources/train/Source000/visual_audit.normalized.json"
        path.write_bytes(b"{}")
        with patch.object(v, "rows_for", return_value=rows), self.assertRaises(ValueError):
            v.summaries(cfg, Path(cfg["out"]))

    def test_partial_errored_response_candidates_rebound_tampering_rejected(self):
        cfg, rows, llm, summary = self.run_case(count=4, batch=4, model=FakeLLM(wrong_count=True))
        invocation = summary["batch_invocations"][0]
        path = Path(invocation["response_ref"]["path"])
        response = v.load(path)
        self.assertIsNotNone(response["error"])
        self.assertEqual(response["output_count"], 3)
        response["responses"][0]["candidate_count"] = 99
        path.write_bytes(v.h.bench.encoded(response))
        invocation["response_ref"] = v.ref(path)
        for item in summary["all_outcomes"]:
            result = v.checked(item["result_ref"])
            for record in result["stages"]:
                record["batch_response_ref"] = invocation["response_ref"]
                raw_path = Path(cfg["out"]) / "sources" / item["split"] / item["id"] / (record["stage"] + ".raw.json")
                raw_path.write_bytes(v.h.bench.encoded({k:x for k,x in record.items() if k != "parsed"}))
            result_path = Path(item["result_ref"]["path"])
            result_path.write_bytes(v.h.bench.encoded(result))
            item["result_ref"] = v.ref(result_path)
            commit_path = Path(item["commit_ref"]["path"])
            commit_path.write_bytes(v.h.bench.encoded({k:x for k,x in item.items() if k != "commit_ref"}))
            item["commit_ref"] = v.ref(commit_path)
        (Path(cfg["out"]) / "summary.json").write_bytes(v.h.bench.encoded(summary))
        with patch.object(v, "rows_for", return_value=rows), self.assertRaisesRegex(ValueError, "outputs/candidates/error"):
            v.summaries(cfg, Path(cfg["out"]))


class StrictArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ms-batch-v4-archive-cpu-", dir="/tmp")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, change=None, commit_change=None):
        source = dict(id="SyntheticArchive", split="train", video_path=str(self.root / "SyntheticArchive.mp4"))
        prefix = "cases/train/" + source["id"]
        members = {name: prefix + "/" + name for name in
                   ("pose.npz", "fullframe_manifest.json", "beats.json", "draft.json", "receipt.json")}
        batch = self.root / "batch"
        origin = str(batch / members["fullframe_manifest.json"])
        npz = str(batch / members["pose.npz"])
        locator = dict(archive_path=str(batch / "batch.tar"), source_case_prefix=prefix,
                       fullframe_manifest_member=members["fullframe_manifest.json"], npz_member=members["pose.npz"])
        flags = dict(list(v.FLAGS.items())[:4])
        storage = dict(locator=locator, artifact_storage="archive_member", artifact_members=members)
        manifest = dict(schema="ms_fullframe_pose2d_manifest_v1", **source, video_sha256="a" * 64,
            frame_count=3, frame_indices=[0, 1, 2], times=[0., .04, .08], source_timestamps=[10., 10.04, 10.08],
            pts_origin=10., width=1280, height=704, duration_seconds=.12,
            fullframe_manifest_path=origin, npz_path=npz, **storage, **flags)
        draft = dict(schema="ms_fullframe_pose2d_rule_draft_v2", id=source["id"], split=source["split"],
            ms_text="Visible endpoint poses", locator=locator, artifact_storage="archive_member",
            source=dict(fullframe_manifest_path=origin, npz_path=npz, video_path=source["video_path"],
                        video_sha256="a" * 64, **storage), **flags)
        receipt = dict(source_row=source, status="extracted_rule_draft_unapproved", **storage, **flags)
        item = dict(source_row=source, members=members, source_case_prefix=prefix, receipt_member=members["receipt.json"])
        if change:
            change(draft, manifest, receipt, item)
        binding = dict(schema="ms_fullframe_pose2d_binding_v1", observations_sha256="b" * 64)
        payload = dict(schema=binding["schema"], fullframe_manifest=manifest, observations_sha256=binding["observations_sha256"])
        binding["sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        draft["source"]["fullframe_binding"] = binding
        commit = dict(schema="ms_unified_novlm_batch_commit_v3", archive_path=str(batch / "batch.tar"),
            archive_sha256="c" * 64, batch_sha256="c" * 64, archive_bytes=10240,
            vlm_called=False, prefetch_drained=True, archive_index={source["id"]:item}, receipts=[receipt])
        if commit_change:
            commit_change(commit)
        commit_ref = v.write_once(batch / "commit.json", commit)
        verification = dict(schema="ms_archive_vlm_batch_verified_v1", commit_ref=commit_ref,
            archive_path=commit["archive_path"], archive_sha256=commit["archive_sha256"], archive_bytes=commit["archive_bytes"],
            sha_verified_once=True, whole_archive_hash_reads=1, archive_stat_stable=True)
        verification_ref = v.write_once(self.root / "verification.json", verification)
        draft_ref = v.write_once(self.root / "input/draft.json", draft)
        manifest_ref = v.write_once(self.root / "input/manifest.json", manifest)
        physical = {"draft.json":draft_ref, "fullframe_manifest.json":manifest_ref}
        metadata = {name:dict(member=members[name], sha256=r["sha256"], bytes=Path(r["path"]).stat().st_size)
                    for name, r in physical.items()}
        proof = dict(schema="ms_archive_vlm_input_origin_v1", commit_ref=commit_ref, source_row=source,
            source_case_prefix=prefix, members=members, receipt=receipt, archive_path=commit["archive_path"],
            archive_sha256=commit["archive_sha256"], archive_bytes=commit["archive_bytes"], sha_verified_once=True,
            archive_verification_ref=verification_ref, logical_fullframe_origin=origin,
            original_binding=binding, original_npz_path=manifest["npz_path"], physical_refs=physical, member_metadata=metadata,
            no_npz_decoded=True, no_media_input_read=True, no_gpu_calls=True, original_raw_bytes_preserved=True)
        row = dict(**source, rule_draft_ref=draft_ref, fullframe_manifest_ref=manifest_ref,
                   fullframe_origin_path=origin, origin_proof_ref=v.write_once(self.root / "proof.json", proof))
        return row

    def test_strict_archive_positive_without_media_or_npz_reads(self):
        row = self.fixture()
        self.assertEqual(v.input_pair(row)[0]["id"], row["id"])
        self.assertFalse(Path(row["fullframe_origin_path"]).exists())
        self.assertFalse(Path(row["video_path"]).exists())

    def test_missing_draft_top_locator_rejected(self):
        row = self.fixture(lambda d, m, r, i:d.pop("locator"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_source_locator_rejected(self):
        row = self.fixture(lambda d, m, r, i:d["source"].pop("locator"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_manifest_locator_rejected(self):
        row = self.fixture(lambda d, m, r, i:m.pop("locator"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_receipt_locator_rejected(self):
        row = self.fixture(lambda d, m, r, i:r.pop("locator"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_fifth_member_rejected(self):
        row = self.fixture(lambda d, m, r, i:i["members"].pop("beats.json"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_receipt_member_rejected(self):
        row = self.fixture(lambda d, m, r, i:i.update(receipt_member="wrong"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_source_storage_rejected(self):
        row = self.fixture(lambda d, m, r, i:d["source"].pop("artifact_storage"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_manifest_storage_rejected(self):
        row = self.fixture(lambda d, m, r, i:m.pop("artifact_storage"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_virtual_npz_even_rebound_rejected(self):
        def change(draft, manifest, receipt, item):
            draft["source"]["npz_path"] = manifest["npz_path"] = "/tmp/wrong.npz"
        row = self.fixture(change)
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_origin_proof_unpaired_rejected(self):
        row = self.fixture()
        del row["origin_proof_ref"]
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_changed_original_commit_rejected(self):
        row = self.fixture()
        proof = v.checked(row["origin_proof_ref"])
        v.input_pair(row)
        Path(proof["commit_ref"]["path"]).write_bytes(b"{}")
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_top_draft_storage_rejected(self):
        row = self.fixture(lambda d, m, r, i:d.pop("artifact_storage"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_receipt_storage_rejected(self):
        row = self.fixture(lambda d, m, r, i:r.pop("artifact_storage"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_canonical_archive_path_rejected(self):
        row = self.fixture(commit_change=lambda c:c.update(archive_path="/tmp/noncanonical.tar"))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_wrong_batch_sha_rejected(self):
        row = self.fixture(commit_change=lambda c:c.update(batch_sha256="f" * 64))
        with self.assertRaises(ValueError): v.input_pair(row)

    def test_missing_prefetch_drained_rejected(self):
        row = self.fixture(commit_change=lambda c:c.pop("prefetch_drained"))
        with self.assertRaises(ValueError): v.input_pair(row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
