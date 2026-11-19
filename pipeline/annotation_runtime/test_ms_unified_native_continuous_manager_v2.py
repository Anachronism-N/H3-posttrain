"""Reuse v1 safety regressions; CPU mocks only, no native jobs or claims."""
import ast
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

fixture = module("native_v1_test_fixture", RT / "test_ms_unified_native_continuous_manager_v1.py")
m = module("native_v2_under_test", RT / "ms_unified_native_continuous_manager_v2.py")
REAL_DEPENDENCIES = m.dependencies


class RegressionMixin:
    def setUp(self):
        rebound = patch.object(fixture, "m", m)
        rebound.start()
        self.addCleanup(rebound.stop)
        super().setUp()
        if hasattr(self, "cfg"):
            ticket = m.load(self.cfg["ticket"]["path"])
            for key in ("ffprobe_path", "ffmpeg_path"):
                path = self.root / key
                path.write_bytes(b"CPU fixture, never executed")
                path.chmod(0o700)
                ticket[key] = str(path)
            self.cfg["ticket"] = self.put("ticket.json", ticket)
            self.bind()

    def test_worker_model_load_uses_true_gate_at_load_boundary(self):
        self.put("config.json", self.cfg)
        events = []
        ownership = {"admitted": True, "caller_owned": True, "gpu_uuids": [self.owner["uuid"]],
                     "child_identity": {"pid": os.getpid()}}
        media = types.SimpleNamespace(__file__=str(m.MEDIA), probe_video=Mock(return_value={"native": "PTS"}),
                                      decode_bgr=Mock(return_value=iter(())))
        def fake_gate(cfg, before_load=False):
            events.append(("gate", before_load))
            return ownership
        def fake_loader(ticket):
            events.append(("model_load", None))
            return object()
        def fake_run(ticket, limit, resource_gate, model_loader, stage_gate, resource_reader, probe, decoder):
            self.assertEqual(limit, 3)
            self.assertEqual(resource_reader("/tmp")["cgroup_headroom"], 160*m.GIB)
            path = Path("/tmp/caller-media/input.mp4")
            self.assertEqual(probe(path), {"native": "PTS"})
            self.assertEqual(list(decoder(path, 2, 1)), [])
            media.probe_video.assert_called_once_with(path, ticket["ffprobe_path"])
            media.decode_bgr.assert_called_once_with(path, 2, 1, ticket["ffmpeg_path"])
            with resource_gate(ticket) as evidence:
                self.assertTrue(evidence["granted"])
                self.assertTrue(evidence["gpu_observed_free"])
                model_loader(ticket)
                stage_gate(False)
                summary, _, _, _ = self.sealed_summary()
                result = m.load(summary)
                summary.unlink()
                return result
        fake_core = types.SimpleNamespace(load_models=fake_loader, run_ticket=fake_run)
        imports = {name: types.SimpleNamespace(__file__=str(m.CORE_DIR / (name + ".py"))) for name in
                   ("ms_unified_novlm_v2", "dwpose_full_extract", "ms_fullframe_pose2d_draft_v2", "ms_fullframe_pose2d_draft_v1", "ms_pose2d_draft")}
        def load_module(name, path):
            self.assertIn(path, (m.CORE, m.MEDIA))
            return media if path == m.MEDIA else fake_core
        saved_path = list(sys.path)
        try:
            with patch.object(m, "gate", side_effect=fake_gate), patch.object(m, "check_model"),                  patch.object(m, "module", side_effect=load_module) as loads, patch.dict(sys.modules, imports),                  patch.object(m.base, "apps", return_value=[]),                  patch.object(m, "local_budget", return_value={"memory": {"headroom": 160*m.GIB}}):
                m.worker(self.cfg["config_path"])
        finally:
            sys.path[:] = saved_path
        self.assertEqual([call.args[1] for call in loads.call_args_list], [m.CORE, m.MEDIA])
        index = events.index(("model_load", None))
        self.assertEqual(events[index-1], ("gate", True))
        self.assertEqual(sum(value for kind, value in events if kind == "gate"), 1)
        self.assertEqual(events[-1], ("gate", False))
        self.assertEqual(m.load(Path(self.cfg["out"]) / "summary.json")["counts"]["generated"], 3)


class NativeTests(RegressionMixin, fixture.NativeTests):
    def test_all_nonmedia_safety_code_is_ast_identical_to_v1(self):
        def safety_tree(path):
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.FunctionDef) and node.name in ("dependencies", "worker") or
                isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MEDIA" for target in node.targets))]
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "ticket_for":
                    node.body = [statement for statement in node.body if not (isinstance(statement, ast.Expr) and
                        isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and
                        statement.value.func.id == "need" and len(statement.value.args) == 2 and
                        isinstance(statement.value.args[1], ast.Constant) and
                        statement.value.args[1].value == "Explicit actual media executables required")]
            return ast.dump(tree, include_attributes=False)
        self.assertEqual(safety_tree(m.SELF), safety_tree(RT / "ms_unified_native_continuous_manager_v1.py"))

    def test_real_dependencies_are_v1_plus_self_and_media_only(self):
        old = module("original_dependency_identity", RT / "ms_unified_native_continuous_manager_v1.py")
        expected = old.dependencies() - {str(old.SELF)} | {str(m.SELF), str(m.MEDIA)}
        self.assertEqual(REAL_DEPENDENCIES(), expected)
        self.assertEqual(m.CORE, old.CORE)
        self.assertEqual(m.CORE.name, "ms_unified_novlm_v3.py")
        self.assertEqual(m.SELF.name, "ms_unified_native_continuous_manager_v2.py")

    def test_media_source_binding_required_and_frozen(self):
        deps = {str(m.SELF), str(Path(m.base.__file__)), str(m.CORE), str(m.MEDIA)}
        with patch.object(m, "dependencies", return_value=deps):
            self.bind()
            m.validate_common(self.cfg)
            path = str(m.MEDIA)
            changed = copy.deepcopy(self.cfg)
            del changed["source_sha256"][path]
            with self.assertRaisesRegex(PermissionError, "exact necessary"):
                m.validate_common(changed)
            self.cfg["source_sha256"][path] = "0"*64
            self.put("config.json", self.cfg)
            with self.assertRaisesRegex(RuntimeError, "source/input changed.*ms_media_local_v1"):
                m.base.check_sources(self.cfg, self.cfg["config_path"], m.base.sha(self.cfg["config_path"]), m.BASE_SHA)

    def test_shadowed_media_is_denied_before_core_run(self):
        self.put("config.json", self.cfg)
        core = types.SimpleNamespace(run_ticket=Mock(side_effect=AssertionError("run called")))
        shadow = types.SimpleNamespace(__file__="/tmp/shadow/ms_media_local_v1.py")
        saved_path = list(sys.path)
        try:
            with patch.object(m, "gate"), patch.object(m, "module", side_effect=[core, shadow]),                  self.assertRaisesRegex(PermissionError, "media dependency shadowed"):
                m.worker(self.cfg["config_path"])
        finally:
            sys.path[:] = saved_path
        core.run_ticket.assert_not_called()

    def test_worker_requires_explicit_media_executables_without_path_fallback(self):
        ticket = m.load(self.cfg["ticket"]["path"])
        for key in ("ffprobe_path", "ffmpeg_path"):
            candidate = dict(ticket)
            del candidate[key]
            self.cfg["ticket"] = self.put("ticket.json", candidate)
            self.bind()
            self.put("config.json", self.cfg)
            with self.subTest(key=key), patch.object(m, "gate"), patch.object(m, "module") as loader,                  self.assertRaisesRegex(PermissionError, "actual media executables"):
                m.worker(self.cfg["config_path"])
            loader.assert_not_called()

    def test_missing_media_tools_are_denied_at_ticket_boundary(self):
        ticket = m.load(self.cfg["ticket"]["path"])
        for missing in (("ffprobe_path",), ("ffmpeg_path",), ("ffprobe_path", "ffmpeg_path")):
            candidate = {key: value for key, value in ticket.items() if key not in missing}
            cfg = dict(self.cfg, ticket=self.put("candidate.json", candidate))
            with self.subTest(missing=missing), patch.object(m, "module") as loader, patch.object(m, "gate") as gate:
                with self.assertRaisesRegex(PermissionError, "actual media executables"):
                    m.ticket_for(cfg)
                with self.assertRaisesRegex(PermissionError, "actual media executables"):
                    m.validate_common(cfg)
                loader.assert_not_called()
                gate.assert_not_called()
            self.assertFalse(Path(cfg["queue"]).exists())
            self.assertFalse(Path(cfg["out"]).exists())

    def test_invalid_media_executable_identity_retains_v1_validation(self):
        ticket = m.load(self.cfg["ticket"]["path"])
        for key in ("ffprobe_path", "ffmpeg_path"):
            for value in ("ffprobe", "/tmp/nonexistent-native-media", str(self.root / ".." / self.root.name / key)):
                cfg = dict(self.cfg, ticket=self.put("candidate.json", dict(ticket, **{key: value})))
                with self.subTest(key=key, value=value), self.assertRaisesRegex(PermissionError, "actual media executable"):
                    m.ticket_for(cfg)


class ContinuousBatchTests(RegressionMixin, fixture.ContinuousBatchTests):
    pass


class CallerGateTests(fixture.CallerGateTests):
    def setUp(self):
        rebound = patch.object(fixture, "m", m)
        rebound.start()
        self.addCleanup(rebound.stop)
        super().setUp()


class CoreIntegrationTests(RegressionMixin, fixture.CoreIntegrationTests):
    def run_integration(self, stop_after_first=False):
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))
        subject, core = self.integration_fixture()
        stop = self.root / "STOP"
        if stop_after_first:
            ticket = dict(m.load(self.cfg["ticket"]["path"]), stop_file=str(stop))
            self.cfg["ticket"] = self.put("ticket.json", ticket)
            self.bind()
        self.put("config.json", self.cfg)
        events = []
        ownership = dict(admitted=True, caller_owned=True, gpu_uuids=[self.owner["uuid"]], child_identity={"pid": os.getpid()})
        def fake_gate(cfg, before_load=False):
            events.append(before_load)
            return ownership
        real_run, real_progress, real_sha = core.run_ticket, core.publish_progress, m.base.sha
        def run_actual(ticket, **kwargs):
            self.assertEqual(kwargs["limit"], 43)
            self.assertTrue(callable(kwargs["probe"]))
            self.assertTrue(callable(kwargs["decoder"]))
            kwargs.update(probe=subject.probe, decoder=subject.decoder, audio_info=subject.audio_info)
            return real_run(ticket, **kwargs)
        def progress(path, value):
            real_progress(path, value)
            if stop_after_first and value["processed"] == 20:
                stop.write_text("STOP")
        def no_old_input_hash(path):
            self.assertNotIn(str(subject.root), str(path))
            return real_sha(path)
        media = types.SimpleNamespace(__file__=str(m.MEDIA))
        def load_module(name, path):
            return media if path == m.MEDIA else core
        with patch.object(m, "gate", side_effect=fake_gate), patch.object(m, "check_model"),              patch.object(m, "module", side_effect=load_module), patch.object(core, "load_models", side_effect=subject.loader),              patch.object(core, "run_ticket", side_effect=run_actual) as run, patch.object(core, "publish_progress", side_effect=progress),              patch.object(m.base, "apps", return_value=[]), patch.object(m.base, "sha", side_effect=no_old_input_hash),              patch.object(m, "local_budget", return_value={"memory": {"headroom": 160*m.GIB}}),              patch.object(m.shutil, "disk_usage", return_value=types.SimpleNamespace(free=32*m.GIB)),              contextlib.redirect_stdout(io.StringIO()):
            m.worker(self.cfg["config_path"])
        result = m.load(Path(self.cfg["out"]) / "summary.json")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(subject.loader.call_count, 1)
        self.assertEqual(events.count(True), 1)
        self.assertGreaterEqual(events.count(False), result["processed"])
        self.assertEqual(result["model_init_count"], 1)
        self.assertEqual(result["total_frames"], result["processed"]*97)
        self.assertEqual(subject.pose.call_count, result["processed"]*97)
        self.assertTrue(result["allprefetch_drained"])
        self.assertFalse(result["vlm_called"])
        self.assertFalse(result["production_complete_claimed"])
        return result, stop


if __name__ == "__main__":
    unittest.main()
