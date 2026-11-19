"""CPU-only native admission, fullframe evidence, sealed budget and two-stage tests."""
import copy
import importlib.util
import json
import os
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fullframe_test_target", RT / "ms_fullframe_vlm_postprocess_v2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
IDS = ["uqVfYjBeBtKElELf", "mfpqUxOTSfYTJpLV", "hFJWFhMQCFJOLwzX"]


class Llm:
    def __init__(self, mode="ok"):
        self.calls = []
        self.mode = mode

    def chat(self, request, sampling_params, use_tqdm):
        self.calls.append(request)
        content = request[1]["content"]
        payload = json.loads(content[0]["text"] if isinstance(content, list) else content)
        schema = payload["required_json_schema"]
        value = {k: v["const"] for k, v in schema["properties"].items() if "const" in v}
        if "supported_actions" in schema["properties"]:
            frames = payload["sampled_native_frames"]
            a, b = frames[:2]
            value.update(visual_motion_summary="Visible arm positions differ between snapshots.",
                supported_actions=[dict(description="The arm appears higher in the second observed pose.", pair_id=0)],
                unsupported_claims=["Music synchronization is not visually established."],
                missing_evidence=["Motion between snapshots and after the final PTS is unknown."])
            if self.mode == "invalidpair": value["supported_actions"][0]["pair_id"] = len(frames) - 1
            if self.mode == "negativepair": value["supported_actions"][0]["pair_id"] = -1
            if self.mode == "boolpair": value["supported_actions"][0]["pair_id"] = True
            if self.mode == "empty": value["supported_actions"] = []
            if self.mode == "bearingvisual": value["supported_actions"][0]["description"] = "The dancer is bearing more weight on one leg."
            if self.mode == "imageplane": value["supported_actions"][0]["description"] = "One arm raises toward image left while the other side bends."
            if self.mode == "badtime": value["supported_actions"][0]["end"] = 999
            if self.mode == "hole": value["supported_actions"][0].update(end=frames[2]["time"], frame_indices=[frames[0]["frame_index"], frames[2]["frame_index"]])
        else:
            assert "visual_audit" in payload and isinstance(content, str)
            value.update(fused_caption="A dancer changes the visible arm position.", corrections=["Removed unsupported synchronization."],
                         limitations=["Sparse snapshots cannot establish continuous motion."])
            if self.mode == "empty": value["fused_caption"] = ""
            if self.mode == "bearingcaption": value["fused_caption"] = "The dancer shifts weight and the hip moves forward."
            if self.mode == "imageplane": value["fused_caption"] = "One arm raises toward image left while the other side bends."
        if self.mode == "missing": value.pop("id")
        if self.mode == "wrongid": value["id"] = "other"
        if self.mode == "wrongsplit": value["split"] = "other"
        if self.mode == "shape": value["supported_actions"] = "wrong"
        text = "not JSON" if self.mode == "badjson" else json.dumps(value)
        if self.mode == "duplicate": text = text[:-1] + ',"id":"duplicate"}'
        tokens = list(range(1800 if self.mode == "tokens" else 20))
        result = SimpleNamespace(text=text, token_ids=tokens, finish_reason="length" if self.mode == "truncate" else "stop")
        return [SimpleNamespace(outputs=[result])]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fullframe-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rootpatch = patch.object(m, "ROOT", self.root)
        self.rootpatch.start()
        self.addCleanup(self.rootpatch.stop)
        self.rows = []
        for i, sid in enumerate(IDS, 4):
            video = self.root / (sid + ".mp4")
            video.write_bytes(b"test-video")
            draft = dict(schema="ms_fullframe_pose2d_rule_draft_v1", id=sid, split="pilot%02d" % i,
                         ms_text="untrusted arm hint", **{k: v for k, v in m.FLAGS.items() if k != "human_review_completed"})
            manifest = dict(schema="ms_fullframe_pose2d_manifest_v1", id=sid, split=draft["split"],
                video_path=str(video), video_sha256=m.base.sha(video), frame_count=24, width=640, height=480,
                frame_indices=list(range(24)), times=[i / 10 for i in range(24)],
                source_timestamps=[i / 10 + 5 for i in range(24)], pts_origin=5., duration_seconds=2.4,
                **{k: v for k, v in m.FLAGS.items() if k != "human_review_completed"})
            dp, fp = self.root / (sid + ".draft.json"), self.root / (sid + ".manifest.json")
            payload = dict(schema="ms_fullframe_pose2d_binding_v1", fullframe_manifest=manifest, observations_sha256="a" * 64)
            binding = dict(schema=payload["schema"], observations_sha256=payload["observations_sha256"],
                sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            draft["source"] = dict(fullframe_manifest_path=str(fp), video_path=str(video), video_sha256=manifest["video_sha256"], fullframe_binding=binding)
            m.write_once(dp, draft)
            m.write_once(fp, manifest)
            self.rows.append(dict(id=sid, split=draft["split"], video_path=str(video), rule_draft_ref=m.ref(dp),
                                  fullframe_manifest_ref=m.ref(fp), rich_text="untrusted diverse hint"))
        plan = self.root / "plan.jsonl"
        plan.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        model = self.root / "modelref.json"
        m.write_once(model, {"test": True})
        safe_refs, restored_refs, owners = [], [], []
        for gpu in (2, 3):
            owner = dict(pid=100 + gpu, uid=os.getuid(), argv=["holder"], start_ticks=1, cgroup="test",
                         CVD=["CUDA_VISIBLE_DEVICES=" + str(gpu)], exe="python", gpu=gpu, uuid="uuid" + str(gpu))
            queue = self.root / "queues" / ("previous-gpu%d" % gpu)
            safe_refs.append(m.write_once(queue / "resources_safe.json", dict(all_holders_safe=True, locks_released=True,
                leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True)))
            restored_refs.append(m.write_once(queue / ("holder_restored_gpu%d.json" % gpu),
                dict(gpu=gpu, uuid=owner["uuid"], holder_identity={k: owner[k] for k in m.base.IDENTITY_KEYS})))
            owners.append(owner)
        phase = m.PREFIX + "test"
        cfgpath = self.root / "config.json"
        self.cfg = dict(schema=m.SCHEMA, node="N2", gpus=[2, 3], phase=phase, config_path=str(cfgpath),
            queue=str(self.root / "queues" / phase), out=str(self.root / phase), targetcount=3, input_count=3,
            selected_source_ids=IDS, release_authorized=True, new_stage_authorized=True, env=m.expected_env({"gpus": [2, 3]}), ffmpeg_path=str(Path(shutil.which("ffmpeg")).resolve()),
            cwd="/tmp", argv=[m.PYTHON, "-B", str(m.SELF), "worker", "--config", str(cfgpath)],
            producer=m.ref(m.SELF), resource_manager_ref=m.ref(m.SELF), plan=m.ref(plan),
            modelref=dict(**m.ref(model), model_identity_sha256="test-model"), holders=owners,
            resource_safe_refs=safe_refs, holder_restored_refs=restored_refs)
        refs = m.source_refs(self.cfg, self.rows)
        self.cfg["source_sha256"] = {p: m.base.sha(p) for p in m.dependencies() | {r["path"] for r in refs}}
        m.write_once(cfgpath, self.cfg)

    def fake_images(self, row, manifest, local, ffmpeg_path):
        frames = m.select_frames(manifest)
        images = []
        for i in range(len(frames)):
            p = local / ("frame-%03d.jpg" % i)
            p.write_bytes(b"\xff\xd8test\xff\xd9")
            images.append(p)
        return frames, images

    def load_receipt(self, cfg):
        return m.write_once(Path(cfg["out"]) / "load.receipt.json", dict(model_identity_sha256=cfg["modelref"]["model_identity_sha256"],
            source_reference_sha256=cfg["modelref"]["sha256"], model_init_count=1, model_load_completed=True,
            model_root=m.MODEL, dtype="bfloat16", tensor_parallel_size=2))

    def run_cases(self, mode="ok", cfg=None):
        cfg = cfg or self.cfg
        Path(cfg["out"]).mkdir()
        self.load_receipt(cfg)
        llm = Llm(mode)
        with patch.object(m, "gate", return_value={"admitted": True}), patch.object(m, "stage_images", side_effect=self.fake_images):
            summary = m.run_sources(cfg, llm, lambda x: x)
        return summary, llm

    def admission(self, budget=160 * 2**30, lock=False):
        return [patch.object(m.subprocess, "check_output", return_value=m.HOSTS["N2"]),
                patch.object(m.base, "apps", return_value=[]), patch.object(m.base, "identity", return_value={}),
                patch.object(m.base, "uuid_for", return_value="uuid"), patch.object(m.base, "check_holder"),
                patch.object(m.h, "lock_snapshot", return_value={"held": lock}), patch.object(m.base, "lease_snapshot", return_value=None),
                patch.object(m.base, "headroom", return_value={"headroom": budget})]

    def validate_with(self, budget=160 * 2**30, lock=False):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.admission(budget, lock): stack.enter_context(p)
            return m.validate(self.cfg)

    def test_full_chain_six_calls_three_terminal(self):
        m.validate_common(self.cfg)
        self.validate_with()
        summary, llm = self.run_cases()
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(summary["new_model_calls"], 6)
        self.assertEqual(summary["processed"], 3)
        self.assertEqual(summary["model_init_count"], 1)
        self.assertEqual(len(list(Path(self.cfg["out"]).glob("budgets/*/*/*.claim.json"))), 9)
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))
        for outcome in summary["all_outcomes"]:
            result = m.checked(outcome["result_ref"])
            self.assertTrue(result["model_revision_performed"])
            self.assertFalse(result["quality_approved"])
            self.assertEqual(len(result["selected_frames"]), 12)

    def test_all_invalid_json_and_evidence_sealed(self):
        for mode in ("badjson", "duplicate", "missing", "wrongid", "wrongsplit", "shape", "badtime", "hole", "invalidpair", "negativepair", "boolpair", "truncate", "tokens"):
            with self.subTest(mode=mode):
                cfg = copy.deepcopy(self.cfg)
                cfg["out"] += "-" + mode
                summary, llm = self.run_cases(mode, cfg)
                self.assertEqual(len(llm.calls), 3)
                self.assertEqual(summary["skippedfailed"], 3)
                self.assertTrue(m.summaries(cfg, Path(cfg["out"])))

    def test_existing_claim_no_retry(self):
        out = Path(self.cfg["out"])
        m.write_once(out / "budgets" / self.rows[0]["split"] / self.rows[0]["id"] / "source.claim.json", {"old": True})
        llm = Llm()
        self.load_receipt(self.cfg)
        with patch.object(m, "gate", return_value={}), patch.object(m, "stage_images", side_effect=self.fake_images):
            summary = m.run_sources(self.cfg, llm, lambda x: x)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(summary["all_outcomes"][0]["status"], "existing_claim_not_reassigned")
        self.assertTrue(m.summaries(self.cfg, out))

    def test_reference_sha_and_exact_source_set(self):
        row = self.rows[0]
        p = Path(row["rule_draft_ref"]["path"])
        original = p.read_bytes()
        p.write_bytes(original + b" ")
        with self.assertRaises(ValueError): m.validate_common(self.cfg)
        p.write_bytes(original)
        self.cfg["source_sha256"]["/tmp/unnecessary"] = "wrong"
        with self.assertRaises(ValueError): m.validate_common(self.cfg)

    def test_input_identity_native_shape_time(self):
        row = self.rows[0]
        p = Path(row["fullframe_manifest_ref"]["path"])
        original = m.load(p)
        for key, value in (("split", "other"), ("schema", "old_roi"), ("frame_indices", [0, 2]),
                           ("times", [0.] * 24), ("source_timestamps", [float("inf")] * 24),
                           ("duration_seconds", -1), ("semantic_approval", True)):
            bad = copy.deepcopy(original)
            bad[key] = value
            p.write_text(json.dumps(bad))
            row["fullframe_manifest_ref"] = m.ref(p)
            with self.subTest(key=key), self.assertRaises(ValueError): m.input_pair(row)
        p.write_text(json.dumps(original))

    def test_admission_exact_160gib_and_minus_one(self):
        self.validate_with()
        with self.assertRaises(ValueError): self.validate_with(160 * 2**30 - 1)
        self.assertEqual(m.base.required_budget(self.cfg), 160 * 2**30)
        with self.assertRaises(ValueError): self.validate_with(lock=True)

    def test_holder_proof_flags_identity_leases(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["holders"][1]["start_ticks"] += 1
        with self.assertRaises(ValueError): m.previous_safe(cfg)
        p = Path(cfg["resource_safe_refs"][0]["path"])
        original_safe = p.read_bytes()
        value = m.load(p)
        value["unknown_jobs_not_signaled"] = False
        p.write_text(json.dumps(value))
        cfg["resource_safe_refs"][0] = m.ref(p)
        with self.assertRaises(ValueError): m.previous_safe(cfg)
        p.write_bytes(original_safe)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.admission(): stack.enter_context(p)
            stack.enter_context(patch.object(m.base, "lease_snapshot", side_effect=RuntimeError("active lease")))
            with self.assertRaises(RuntimeError): m.validate(self.cfg)

    def test_exclusive_write_and_unsealed_not_complete(self):
        p = self.root / "exclusive.json"
        m.write_once(p, {"first": True})
        with self.assertRaises(FileExistsError): m.write_once(p, {"second": True})
        self.assertEqual(m.load(p), {"first": True})
        Path(self.cfg["out"]).mkdir()
        with self.assertRaises(FileNotFoundError): m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_summary_tampering_rejected(self):
        summary, _ = self.run_cases()
        p = Path(self.cfg["out"]) / "summary.json"
        for key, bad in (("processed", 2), ("quality_approved", True), ("new_model_calls", 7), ("model_init_count", 2)):
            value = {**summary, key: bad}
            p.write_text(json.dumps(value))
            with self.subTest(key=key), self.assertRaises(ValueError): m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_decode_single_copy_exact_indices_env(self):
        row = self.rows[0]
        _, manifest = m.input_pair(row)
        local = self.root / "local"
        local.mkdir()
        frames = m.select_frames(manifest)
        calls = []
        def decoder(argv, **kwargs):
            calls.append((argv, kwargs))
            for i in range(len(frames)): (local / ("frame-%03d.jpg" % (i + 1))).write_bytes(b"\xff\xd8test\xff\xd9")
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": ":/trusted::/also:"}), patch.object(m.subprocess, "run", side_effect=decoder):
            got, images = m.stage_images(row, manifest, local, self.cfg["ffmpeg_path"])
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], ":/trusted::/also:")
        self.assertEqual(len(calls), 1)
        self.assertEqual(got[0]["frame_index"], 0)
        self.assertEqual(got[-1]["frame_index"], 23)
        argv, kwargs = calls[0]
        self.assertIn("-noautorotate", argv)
        self.assertIn("eq(n\\,0)", argv[argv.index("-vf") + 1])
        self.assertEqual(kwargs["cwd"], local)
        self.assertEqual(kwargs["env"]["LD_LIBRARY_PATH"], "/trusted:/also")
        self.assertEqual(len(images), 12)

    def test_copy_bad_sha_and_jpeg_count(self):
        row = self.rows[0]
        _, manifest = m.input_pair(row)
        local = self.root / "copybad"
        local.mkdir()
        manifest["video_sha256"] = "0" * 64
        with self.assertRaises(ValueError), patch.object(m.subprocess, "run") as decoder:
            m.stage_images(row, manifest, local, self.cfg["ffmpeg_path"])
        decoder.assert_not_called()
        _, manifest = m.input_pair(row)
        local = self.root / "jpegmissing"
        local.mkdir()
        with patch.object(m.subprocess, "run"), self.assertRaises(ValueError): m.stage_images(row, manifest, local, self.cfg["ffmpeg_path"])

    def test_worker_single_initialization_six_real_mock_chats(self):
        Path(self.cfg["out"]).mkdir()
        llm = Llm()
        loaded = dict(model_identity_sha256=self.cfg["modelref"]["model_identity_sha256"],
            source_reference_sha256=self.cfg["modelref"]["sha256"], model_init_count=1, model_load_completed=True,
            model_root=m.MODEL, dtype="bfloat16", tensor_parallel_size=2)
        with patch.object(m, "gate", return_value={"admitted": True}) as gate, patch.object(m, "stage_images", side_effect=self.fake_images), \
             patch.object(m.base, "check_model"), patch.object(m, "actual_model", return_value=loaded), \
             patch.object(m, "runtime", return_value=(llm, lambda x: x)) as factory, patch.object(m.base, "selected_sources", m.selected):
            m.worker(self.cfg["config_path"])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(sum(c.kwargs.get("before_load", False) for c in gate.call_args_list), 1)
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))

    def test_existing_stage_claim_no_retry_or_extra_call(self):
        self.load_receipt(self.cfg)
        row = self.rows[0]
        m.write_once(Path(self.cfg["out"]) / "budgets" / row["split"] / row["id"] / "visual_audit.claim.json", {"old": True})
        llm = Llm()
        with patch.object(m, "gate", return_value={}), patch.object(m, "stage_images", side_effect=self.fake_images):
            result = m.run_sources(self.cfg, llm, lambda x: x)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(result["all_outcomes"][0]["status"], "failed_sealed_no_retry")
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))

    def test_real_holder_checker_wrong_uuid_cvd_pid(self):
        owner = dict(self.cfg["holders"][0], argv=["/opt/venv_eval/bin/python", "-u", str(m.base.HOLDER)])
        current = dict(owner, sig_ign=0, sig_blk=0)
        apps = [dict(uuid=owner["uuid"], pid=owner["pid"])]
        m.base.check_holder(owner, current, 2, owner["uuid"], apps)
        for key, value in (("pid", 999), ("CVD", ["CUDA_VISIBLE_DEVICES=3"]), ("uid", os.getuid() + 1)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.base.check_holder(owner, {**current, key: value}, 2, owner["uuid"], apps)
        with self.assertRaises(RuntimeError): m.base.check_holder(owner, current, 2, "wronguuid", apps)
        with self.assertRaises(RuntimeError): m.base.check_holder(owner, current, 2, owner["uuid"], apps + [dict(uuid=owner["uuid"], pid=999)])

    def test_actual_frozen_three_v2_input_pairs_cpu_only(self):
        plan = RT / "ms_fullframe_vlm_postprocess_v1_diag3_plan.jsonl"
        self.assertEqual(m.base.sha(plan), "72edacbe2fa70e858111c8f8bd91ea6bc7b79e337221fc640df5695a03d25f87")
        rows = [json.loads(line) for line in plan.read_text().splitlines() if line.strip()]
        self.assertEqual([r["id"] for r in rows], IDS)
        for row in rows:
            draft, manifest = m.input_pair(row)
            self.assertEqual(draft["schema"], "ms_fullframe_pose2d_rule_draft_v2")
            self.assertEqual(row["split"], "train")
            self.assertEqual(manifest["frame_count"], 81)

    def test_wrong_original_binding_and_env_rejected(self):
        row = self.rows[0]
        draft = m.checked(row["rule_draft_ref"])
        draft["source"]["fullframe_binding"]["sha256"] = "0" * 64
        p = Path(row["rule_draft_ref"]["path"])
        original = p.read_bytes()
        p.write_text(json.dumps(draft))
        row["rule_draft_ref"] = m.ref(p)
        with self.assertRaises(ValueError): m.input_pair(row)
        p.write_bytes(original)
        row["rule_draft_ref"] = m.ref(p)
        cfg = copy.deepcopy(self.cfg)
        cfg["env"]["PYTHONPATH"] = "/bad/cuda"
        with self.assertRaises(ValueError): m.validate_common(cfg)

    def test_real_base_sha_and_parent_gate_preserved(self):
        self.assertEqual(m.base.sha(m.base.__file__), m.BASE_SHA)
        constants = m.h.parent_gate.__code__.co_consts
        self.assertIn(m.PYTHON, constants)
        self.assertIn("/proc/locks", constants)
        self.assertIn("GPU occupied before admitted model load", constants)

    def test_check_only_no_release_load_claim(self):
        path = self.cfg["config_path"]
        digest = m.base.sha(path)
        argv = [m.PYTHON, "-B", str(m.SELF), "--config", path, "--config-sha", digest,
                "--manager-sha", m.BASE_SHA, "--check-only"]
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(m.sys, "argv", argv[2:]))
            stack.enter_context(patch.object(m.base, "identity", return_value={"argv": argv}))
            stack.enter_context(patch.object(m, "validate"))
            stack.enter_context(patch.object(m.base, "check_model"))
            stack.enter_context(patch.object(m.base, "selected_sources", m.selected))
            base_main = stack.enter_context(patch.object(m.base, "main"))
            load = stack.enter_context(patch.object(m, "runtime"))
            release = stack.enter_context(patch.object(m.base, "release_holder"))
            m.main()
            base_main.assert_not_called()
            load.assert_not_called()
            release.assert_not_called()
        self.assertFalse(Path(self.cfg["out"]).exists())
        self.assertFalse(Path(self.cfg["queue"]).exists())

    def test_pair_lookup_every_exact_native_endpoint(self):
        row = self.rows[0]
        frames = m.select_frames(m.input_pair(row)[1])
        pairs = m.sample_pairs(frames)
        self.assertEqual(len(pairs), 11)
        raw = dict(schema="ms_fullframe_visual_audit_v2", id=row["id"], split=row["split"],
                   visual_motion_summary="Visible endpoint pose changes.", supported_actions=[],
                   unsupported_claims=[], missing_evidence=["Unsampled motion is unknown."])
        for pair_id, (a, b) in enumerate(zip(frames, frames[1:])):
            raw["supported_actions"] = [dict(description="One side raises.", pair_id=pair_id)]
            original = copy.deepcopy(raw)
            audit = m.audit_check(raw, row, frames)
            action = audit["supported_actions"][0]
            self.assertEqual(action["frame_indices"], [a["frame_index"], b["frame_index"]])
            self.assertEqual([action["start"], action["end"]], [a["time"], b["time"]])
            self.assertEqual(action["source_pts"], [a["source_pts"], b["source_pts"]])
            self.assertEqual(raw, original)
        for pair_id in (-1, 11, True, 1.0, "0"):
            raw["supported_actions"][0]["pair_id"] = pair_id
            with self.subTest(pair_id=pair_id), self.assertRaises(ValueError): m.audit_check(raw, row, frames)

    def test_pair_schema_actual_two_frames_and_no_free_indices(self):
        row = self.rows[0]
        frames = m.select_frames(m.input_pair(row)[1])[:2]
        grammar = m.response_schema(row, "visual_audit", frames)
        action_schema = grammar["properties"]["supported_actions"]["items"]
        self.assertEqual(set(action_schema["properties"]), {"description", "pair_id"})
        self.assertEqual(action_schema["properties"]["pair_id"]["maximum"], 0)
        for action in (dict(description="Visible pose change.", start=0, end=.2, frame_indices=[0, 2]),
                       dict(description="Visible pose change.", pair_id=1)):
            with self.assertRaises(ValueError): m.schema_check(action, action_schema)
        m.schema_check(dict(description="Visible pose change.", pair_id=0), action_schema)
        bad_frames = copy.deepcopy(frames)
        bad_frames[1]["source_pts"] += .1
        with self.assertRaises(ValueError): m.sample_pairs(bad_frames)

    def test_compact_schema_exact_bounds(self):
        visual = m.response_schema(self.rows[0], "visual_audit")["properties"]
        caption = m.response_schema(self.rows[0], "caption_fusion")["properties"]
        self.assertEqual(visual["supported_actions"]["maxItems"], 8)
        self.assertEqual(visual["supported_actions"]["items"]["properties"]["description"]["maxLength"], 220)
        self.assertEqual(visual["visual_motion_summary"]["maxLength"], 500)
        self.assertEqual(caption["fused_caption"]["maxLength"], 1600)
        for prop in (visual["unsupported_claims"], visual["missing_evidence"], caption["corrections"], caption["limitations"]):
            self.assertEqual(prop["maxItems"], 8)
            self.assertEqual(prop["items"]["maxLength"], 180)
        for prop in (visual["visual_motion_summary"], visual["supported_actions"]["items"]["properties"]["description"], caption["fused_caption"]):
            m.schema_check("x" * prop["maxLength"], prop)
            with self.assertRaises(ValueError): m.schema_check("x" * prop["maxLength"] + " ", prop)
        with self.assertRaises(ValueError): m.schema_check(["x"] * 9, visual["unsupported_claims"])

    def test_risk_candidates_preserved_not_grounded_success(self):
        for mode in ("bearingvisual", "bearingcaption"):
            with self.subTest(mode=mode):
                cfg = copy.deepcopy(self.cfg)
                cfg["out"] += "-" + mode
                summary, llm = self.run_cases(mode, cfg)
                self.assertEqual(len(llm.calls), 6)
                self.assertEqual(summary["structure_grounded_candidate"], 0)
                self.assertEqual(summary["skippedfailed"], 3)
                for outcome in summary["all_outcomes"]:
                    result = m.checked(outcome["result_ref"])
                    self.assertEqual(result["status"], "postprocessed_needs_visual_revision")
                    self.assertTrue(result["model_revision_performed"])
                    self.assertTrue(result["evidence_policy_violations"])
                    self.assertFalse(result["structure_grounded_candidate"])
                    self.assertEqual(result["visual_audit_raw"], json.loads(result["stages"][0]["raw_text"]))
                    self.assertEqual(result["fused_caption"], json.loads(result["stages"][1]["raw_text"])["fused_caption"])
                    for key, value in m.FLAGS.items(): self.assertIs(result[key], value)
                self.assertTrue(m.summaries(cfg, Path(cfg["out"])))

    def test_risk_patterns_and_image_plane_not_false_positive(self):
        audit = dict(visual_motion_summary="One arm raises toward image left.", supported_actions=[dict(description="The other side bends.")])
        caption = dict(fused_caption="The dancer lowers one arm at image right.")
        self.assertEqual(m.evidence_policy_violations(audit, caption), [])
        for text in ("weight bearing", "weight shift", "shifts weight", "shifting their weight", "bearing more weight",
                     "hip moving forward", "hip moves back", "palm orientation", "palms facing up", "3D motion",
                     "承重", "掌心朝向", "踩拍", "音乐节拍同步"):
            caption["fused_caption"] = text
            with self.subTest(text=text): self.assertTrue(m.evidence_policy_violations(audit, caption))
        audit["unsupported_claims"] = ["Weight bearing cannot be established."]
        caption["fused_caption"] = "The visible arm raises."
        caption["limitations"] = ["Palm orientation and 3D motion are unverified."]
        self.assertEqual(m.evidence_policy_violations(audit, caption), [])
        summary, _ = self.run_cases("imageplane")
        self.assertEqual(summary["structure_grounded_candidate"], 3)
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))

    def test_empty_evidence_two_stages_no_quality_success(self):
        summary, llm = self.run_cases("empty")
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(summary["structure_grounded_candidate"], 0)
        self.assertEqual(summary["skippedfailed"], 3)
        for outcome in summary["all_outcomes"]:
            result = m.checked(outcome["result_ref"])
            self.assertEqual(result["status"], "insufficient_visible_evidence")
            self.assertEqual(result["fused_caption"], "")
            self.assertTrue(result["insufficient_visible_evidence"])
            self.assertFalse(result["quality_approved"])
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))

    def test_raw_normalized_ref_and_replay_mapping(self):
        summary, _ = self.run_cases()
        outcome = summary["all_outcomes"][0]
        result = m.checked(outcome["result_ref"])
        self.assertNotIn("frame_indices", result["visual_audit_raw"]["supported_actions"][0])
        self.assertIn("frame_indices", result["visual_audit"]["supported_actions"][0])
        normalized_path = Path(result["normalized_audit_ref"]["path"])
        self.assertEqual(m.load(normalized_path), result["visual_audit"])
        self.assertTrue(m.summaries(self.cfg, Path(self.cfg["out"])))
        # Rebind all durable hashes to prove replay checks mapping, not merely file digests.
        result["visual_audit"]["supported_actions"][0]["frame_indices"] = [0, 23]
        normalized_path.write_text(json.dumps(result["visual_audit"]))
        result["normalized_audit_ref"] = m.ref(normalized_path)
        result_path = Path(outcome["result_ref"]["path"])
        result_path.write_text(json.dumps(result))
        outcome["result_ref"] = m.ref(result_path)
        commit_path = Path(outcome["commit_ref"]["path"])
        commit_path.write_text(json.dumps({k: v for k, v in outcome.items() if k != "commit_ref"}))
        outcome["commit_ref"] = m.ref(commit_path)
        (Path(self.cfg["out"]) / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "raw/normalized"): m.summaries(self.cfg, Path(self.cfg["out"]))

    def test_independent_v2_prefix_old_config_rejected(self):
        self.assertEqual(m.SCHEMA, "ms_fullframe_vlm_postprocess_native_v2")
        self.assertEqual(m.PREFIX, "ms_fullframe_vlm_postprocess_v2_")
        cfg = copy.deepcopy(self.cfg)
        cfg["schema"] = "ms_fullframe_vlm_postprocess_native_v1"
        with self.assertRaises(ValueError): m.validate_common(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["phase"] = "ms_fullframe_vlm_postprocess_v1_test"
        with self.assertRaises(ValueError): m.validate_common(cfg)

    def test_shared_tp2_native_safe_proof_strict_per_gpu(self):
        cfg = copy.deepcopy(self.cfg)
        shared = self.root / "queues" / "previous-tp2"
        safe = dict(all_holders_safe=True, locks_released=True, leases_restored=True,
                    managed_parent_wait=True, unknown_jobs_not_signaled=True)
        safe_ref = m.write_once(shared / "resources_safe.json", safe)
        cfg["resource_safe_refs"] = [safe_ref, dict(safe_ref)]
        cfg["holder_restored_refs"] = [m.write_once(shared / ("holder_restored_gpu%d.json" % owner["gpu"]),
            dict(gpu=owner["gpu"], uuid=owner["uuid"], holder_identity={k: owner[k] for k in m.base.IDENTITY_KEYS}))
            for owner in cfg["holders"]]
        refs = m.source_refs(cfg, self.rows)
        cfg["source_sha256"] = {p: m.base.sha(p) for p in m.dependencies() | {r["path"] for r in refs}}
        m.validate_common(cfg)
        m.previous_safe(cfg)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.admission(): stack.enter_context(p)
            m.validate(cfg)
        for key in safe:
            bad = {**safe, key: False}
            (shared / "resources_safe.json").write_text(json.dumps(bad))
            broken = copy.deepcopy(cfg)
            broken["resource_safe_refs"] = [m.ref(shared / "resources_safe.json")] * 2
            with self.subTest(flag=key), self.assertRaises(ValueError): m.previous_safe(broken)
        (shared / "resources_safe.json").write_bytes(m.h.bench.encoded(safe))
        m.previous_safe(cfg)
        for field, value in (("gpu", 2), ("uuid", cfg["holders"][0]["uuid"]),
                             ("holder_identity", {**cfg["holders"][1], "start_ticks": 999})):
            rr = cfg["holder_restored_refs"][1]
            path = Path(rr["path"])
            original = path.read_bytes()
            restored = m.load(path)
            restored[field] = value
            path.write_text(json.dumps(restored))
            broken = copy.deepcopy(cfg)
            broken["holder_restored_refs"][1] = m.ref(path)
            with self.subTest(restored_field=field), self.assertRaises(ValueError): m.previous_safe(broken)
            path.write_bytes(original)
        broken = copy.deepcopy(cfg)
        broken["holder_restored_refs"] = [cfg["holder_restored_refs"][0]] * 2
        with self.assertRaises(ValueError): m.previous_safe(broken)
        broken = copy.deepcopy(cfg)
        broken["holders"][1]["start_ticks"] += 1
        with self.assertRaises(ValueError): m.previous_safe(broken)
        broken = copy.deepcopy(cfg)
        broken["resource_safe_refs"] = broken["resource_safe_refs"][:1]
        with self.assertRaises(ValueError): m.validate_common(broken)

    def test_actual_frozen_fresh_v2_three_input_pairs(self):
        plan = RT / "ms_fullframe_vlm_postprocess_v2_diag3_plan.jsonl"
        self.assertEqual(m.base.sha(plan), "d86b2414c97dee7c2c30317e6e2f7af755ff9915a8776e29197df04a88e96e59")
        rows = [m.h.bench.strict(line) for line in plan.read_bytes().splitlines() if line.strip()]
        self.assertEqual([r["id"] for r in rows], ["foiUwxyOjHXDJDsK", "mckCcxBOybGGuzQl", "hwnvfqovhiiBQsdY"])
        self.assertFalse(set(r["id"] for r in rows) & set(IDS))
        for row in rows:
            draft, manifest = m.input_pair(row)
            self.assertEqual(draft["schema"], "ms_fullframe_pose2d_rule_draft_v2")
            self.assertEqual(row["split"], "train")
            self.assertEqual(manifest["frame_count"], 81)
            pairs = m.sample_pairs(m.select_frames(manifest))
            self.assertEqual(len(pairs), 11)
            self.assertEqual(pairs[0]["frame_indices"], [0, 7])
            self.assertEqual(pairs[-1]["frame_indices"], [73, 80])


if __name__ == "__main__":
    unittest.main()
