import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_driver", RT / "full12k_driver.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
spec = importlib.util.spec_from_file_location("full_roi_manager", RT / "full12k_roi_resource_manager.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FullPoolTests(unittest.TestCase):
    def test_immutable_refuses_changed_resume(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            d.immutable(path, {"a": 1})
            d.immutable(path, {"a": 1})
            with self.assertRaises(ValueError):
                d.immutable(path, {"a": 2})
            self.assertEqual(d.load(path), {"a": 1})

    def test_contract_cannot_rebind_changed_code(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            asset = root / "source.py"
            asset.write_text("before")
            contract = root / "contract.json"
            d.immutable(contract, {"files": {str(asset): d.sha(asset)}})
            d.verify_contract(contract)
            asset.write_text("after")
            with self.assertRaises(ValueError):
                d.verify_contract(contract)

    def test_failed_roi_needs_attempt_and_reason(self):
        row = {"id": "case", "binding": {"video_sha256": "video"}}
        value = d.bound_terminal(row, "plan", "failed", reason="native decoder failed", roi_extraction_attempted=True)
        d.check_terminal(value, row, "plan")
        for key in ("reason", "roi_extraction_attempted"):
            bad = dict(value)
            bad.pop(key)
            with self.assertRaises(ValueError):
                d.check_terminal(bad, row, "plan")

    def test_roi_terminal_rejects_different_source_or_plan(self):
        row = {"id": "case", "binding": {"video_sha256": "video"}}
        value = d.bound_terminal(row, "plan", "failed", reason="missing source", roi_extraction_attempted=True)
        for key in ("id", "video_sha256", "plan_sha256", "status"):
            bad = dict(value)
            bad[key] = "wrong"
            with self.assertRaises(ValueError):
                d.check_terminal(bad, row, "plan")

    def test_ready_roi_requires_complete_sha_and_cuda_proof(self):
        row = {"id": "case", "binding": {"video_sha256": "video"}}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("roi_manifest.json", "source_manifest.json", "extract_receipt.json"):
                d.immutable(root / name, {})
            proof = root / "cuda_execution_receipt.json"
            d.immutable(proof, {"real_CUDA_nodes_executed": True})
            files = {str(p): d.sha(p) for p in root.iterdir()}
            value = d.bound_terminal(row, "plan", "ready", roi_manifest=str(root / "roi_manifest.json"),
                cuda_execution_receipt=str(proof), files=files)
            d.check_terminal(value, row, "plan")
            bad = copy.deepcopy(value)
            bad["files"].pop(str(root / "extract_receipt.json"))
            with self.assertRaises(ValueError):
                d.check_terminal(bad, row, "plan")
            (root / "roi_manifest.json").write_text("changed")
            with self.assertRaises(ValueError):
                d.check_terminal(value, row, "plan")

    def test_bounded_profile_requires_actual_cuda_for_both_models(self):
        class Session:
            def __init__(self, path):
                self.path = path
            def end_profiling(self):
                return str(self.path)
            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = [root / "detector.json", root / "pose.json"]
            for path in paths:
                path.write_text(json.dumps([{"args": {"provider": "CUDAExecutionProvider"}},
                    {"args": {"provider": "CPUExecutionProvider"}}]))
            model = types.SimpleNamespace(session_det=Session(paths[0]), session_pose=Session(paths[1]))
            receipt = d.finish_profile(model, root)
            self.assertTrue(d.load(receipt)["profiling_ended_before_remainder_of_full_pool"])
            paths[1].write_text(json.dumps([{"args": {"provider": "CPUExecutionProvider"}}]))
            with self.assertRaises(RuntimeError):
                d.finish_profile(model, root)

    def test_full_pool_roi_manager_rejects_fixed3_and_other_gpu(self):
        cfg = {"node": "N5", "gpus": [6], "holders": [{"gpu": 6}],
            "release_authorized": True, "phase": "full12k_roi_v1", "env": {"CUDA_VISIBLE_DEVICES": "6"},
            "targetcount": 12000, "input_count": 12000,
            "source_selection": {"shard": 0, "nshard": 1, "limit": 0},
            "producer": {"path": "/producer.py"}, "argv": ["/producer.py", "roi"], "cwd": "/tmp"}
        m.validate_roi(cfg)
        for key, value in (("targetcount", 3), ("input_count", 3), ("gpus", [7]),
                           ("phase", "hand_roi_fixed3_v1"), ("release_authorized", False)):
            bad = copy.deepcopy(cfg)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                m.validate_roi(bad)

    def test_real_12000_stride_is_disjoint_complete(self):
        plan = RT.parent / "plan_full12000.jsonl"
        cfg = {"plan": {"path": str(plan)}, "argv": ["--plan", str(plan), "--limit", "0", "--nshard", "4", "--shard", "0"]}
        shards = []
        for shard in range(4):
            cfg["argv"][-1] = str(shard)
            ids, total = m.base.selected_sources(cfg)
            self.assertEqual(total, 12000)
            self.assertEqual(len(ids), 3000)
            shards.append(set(ids))
        self.assertEqual(len(set.union(*shards)), 12000)
        for i in range(4):
            for j in range(i):
                self.assertFalse(shards[i] & shards[j])

    def test_real_fixed3_eligibility_is_two_full_one_partial(self):
        results = []
        for source in sorted((RT.parent / "stable_roi3_v3/sources").iterdir()):
            paths = list((source / "results").glob("run-*.json"))
            if paths:
                results.append(d.load(paths[-1]))
        self.assertEqual(len(results), 3)
        self.assertEqual(sum(d.eligible(result) for result in results), 2)
        for result in results:
            bad = copy.deepcopy(result)
            bad["final"]["status"] = "prepared"
            self.assertFalse(d.eligible(bad))
            bad = copy.deepcopy(result)
            if bad["final"].get("json", {}).get("windows"):
                bad["final"]["json"]["windows"][0]["caption"] = ""
                self.assertFalse(d.eligible(bad))

    def test_shared_roi_model_or_profile_failure_stops_without_source_terminals(self):
        plan = RT.parent / "plan_full12000.jsonl"
        plans = [json.loads(line) for line in plan.read_text().splitlines() if line.strip()]
        rows = [{"id": row["id"]} for row in plans]
        for failure in ("model", "profile"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "roi"
                ref = Path(folder) / "model.json"
                ref.write_text("{}")
                calls = []
                def load_model(*args):
                    calls.append("load")
                    if failure == "model":
                        raise RuntimeError("shared model failure")
                    return object()
                def directory(root, source):
                    path = root / plans[0]["id"]
                    path.mkdir()
                    return path
                roi = types.SimpleNamespace(read_manifest=lambda *a, **kw: (rows, {"sha256": d.sha(plan)}),
                    validate_params=lambda p: None, verify_model=lambda p: None,
                    prepare_source=lambda *a: {"video_sha256": plans[0]["binding"]["video_sha256"]},
                    ensure_prepared=directory, load_cuda_model=load_model,
                    extract_source=lambda *a: {"crops": []})
                contract = {"roi_source_dir": str(RT), "plan": str(plan), "roi_out": str(out),
                    "roi_params": {}, "roi_modelref": str(ref), "reuse": {}}
                args = ["--plan", str(plan), "--out", str(out), "--ffmpeg", "/usr/local/bin/ffmpeg",
                        "--ffprobe", "/opt/ms_tools/ffprobe"]
                with patch.object(d.importlib, "import_module", return_value=roi), \
                     patch.object(d, "finish_profile", side_effect=RuntimeError("shared profile failure")):
                    with self.assertRaises(RuntimeError, msg=failure):
                        d.roi_main(types.SimpleNamespace(), args, contract)
                self.assertEqual(calls, ["load"])
                self.assertFalse(list((out / "terminal").glob("*.json")))

    def test_visual_roi_timeout_preserves_all_pending_without_failed_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            roi_root = folder / "roi"
            roi_root.mkdir()
            out = folder / "visual"
            contract = {"visual_source_dir": str(RT / "source_stable_roi_v3"),
                "plan": str(RT.parent / "plan_full12000.jsonl"),
                "roi_out": str(roi_root), "visual_out": str(out)}
            contract_path = folder / "contract.json"
            contract_path.write_text(json.dumps(contract))
            options = types.SimpleNamespace(contract=str(contract_path), roi_wait_seconds=1e-9,
                                            roi_poll_seconds=.01)
            args = ["--plan", contract["plan"], "--out", str(out / "shard0"), "--stage", "run",
                    "--limit", "0", "--shard", "0", "--nshard", "4", "--tp", "2",
                    "--hand-roi-root", str(roi_root), "--model", "/opt/models/Qwen3-VL-32B-Instruct"]
            with self.assertRaises(RuntimeError, msg="shared ROI timeout"):
                d.visual_main(options, args, contract)
            blocked = d.load(out / "shard0" / "blocked.json")
            self.assertEqual(len(blocked["pending_ids"]), 3000)
            self.assertTrue(blocked["pending_sources_not_marked_failed"])
            self.assertFalse(list((out / "shard0" / "completed").glob("*.json")))
            self.assertEqual((out / "shard0" / "completed_labels.jsonl").read_bytes(), b"")

    def test_candidate_without_all_original_windows_is_never_trainable(self):
        result = {"status": "candidate", "original_windows": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
            "window_observations": [{"window": {"window_id": "w00000"}, "status": "candidate"},
                {"window": {"window_id": "w00001"}, "status": "candidate"}],
            "final": {"status": "candidate", "json": {"windows": [{"window_id": "w00000", "caption": "dances"}]}},
            "final_timed_caption": "[0-1] dances"}
        self.assertFalse(d.eligible(result))
        result["final"]["json"]["windows"].append({"window_id": "w00001", "caption": "turns"})
        self.assertTrue(d.eligible(result))
        result["final"]["json"]["windows"].reverse()
        self.assertFalse(d.eligible(result))


if __name__ == "__main__":
    unittest.main()
