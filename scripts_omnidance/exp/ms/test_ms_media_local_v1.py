"""CPU subprocess mocks only; no native media process or model load."""
import ast
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("media_under_test", Path(__file__).with_name("ms_media_local_v1.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

# Load only the original functions, without importing pose/model dependencies.
source = ast.parse(Path(__file__).with_name("ms_unified_novlm_v2.py").read_text())
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and
             node.name in ("probe_video", "decode_bgr", "video_metadata")]
original = dict(subprocess=subprocess, tempfile=tempfile, np=np, json=json, Fraction=Fraction)
exec(compile(ast.Module(body=functions, type_ignores=[]), "frozen_media_functions", "exec"), original)


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("/tmp/local-media/source.mp4")
        self.payload = {"streams": [{"width": 2, "height": 1, "duration": "0.1", "nb_frames": "2", "time_base": "1/1000"}],
                        "frames": [{"best_effort_timestamp": "0", "best_effort_timestamp_time": "0.000"},
                                   {"best_effort_timestamp": "40", "best_effort_timestamp_time": "0.040"}]}

    def assert_environment(self, before, actual):
        self.assertEqual(os.environ.copy(), before)
        expected = dict(before)
        if "LD_LIBRARY_PATH" in expected:
            items = [item for item in expected["LD_LIBRARY_PATH"].split(":") if item]
            if items:
                expected["LD_LIBRARY_PATH"] = ":".join(items)
            else:
                del expected["LD_LIBRARY_PATH"]
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, os.environ)

    def decoder(self, raw=b"\x01\x02\x03\x04\x05\x06", code=0):
        process = Mock(stdout=io.BytesIO(raw))
        process.wait.return_value = code
        return process

    def test_probe_argv_and_pts_are_exactly_original(self):
        with patch.object(m.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(self.payload))) as run:
            expected = original["probe_video"](self.path, "/opt/bin/ffprobe")
            original_call = run.call_args
            actual = m.probe_video(self.path, "/opt/bin/ffprobe")
            current_call = run.call_args
        self.assertEqual(current_call.args, original_call.args)
        self.assertEqual({k: v for k, v in current_call.kwargs.items() if k not in ("cwd", "env")}, original_call.kwargs)
        self.assertEqual(actual, expected)
        self.assertEqual(current_call.kwargs["cwd"], self.path.parent)
        self.assertNotIn("-threads", current_call.args[0])
        pts, duration, width, height = original["video_metadata"](actual)
        np.testing.assert_array_equal(pts, [0, 0.04])
        self.assertEqual((duration, width, height), (0.1, 2, 1))

    def test_probe_environment_missing_clean_empty_and_consecutive_items(self):
        values = (None, "", ":", ":::", "/opt/a", "/opt/a:", ":/opt/a::/opt/b::", " /opt/a :/opt/b")
        for value in values:
            env = {"KEEP": "unchanged", "LD_PRELOAD": "untouched"}
            if value is not None:
                env["LD_LIBRARY_PATH"] = value
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True),                  patch.object(m.subprocess, "run", return_value=types.SimpleNamespace(stdout=json.dumps(self.payload))) as run:
                before = os.environ.copy()
                m.probe_video(self.path, "/opt/bin/ffprobe")
                self.assert_environment(before, run.call_args.kwargs["env"])

    def test_decode_environment_missing_clean_empty_and_consecutive_items(self):
        for value in (None, "", "::", "/opt/a", "/opt/a:", ":/opt/a::/opt/b::"):
            env = {"KEEP": "unchanged"}
            if value is not None:
                env["LD_LIBRARY_PATH"] = value
            process = self.decoder()
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True),                  patch.object(m.subprocess, "Popen", return_value=process) as popen:
                before = os.environ.copy()
                list(m.decode_bgr(self.path, 2, 1, "/opt/bin/ffmpeg"))
                self.assert_environment(before, popen.call_args.kwargs["env"])
                self.assertEqual(popen.call_args.kwargs["cwd"], self.path.parent)

    def test_decoder_argv_bgr_shape_and_dtype_are_exactly_original(self):
        raw = bytes(range(12))
        with patch.object(m.subprocess, "Popen", side_effect=[self.decoder(raw), self.decoder(raw)]) as popen:
            expected = list(original["decode_bgr"](self.path, 2, 1, "/opt/bin/ffmpeg"))
            actual = list(m.decode_bgr(self.path, 2, 1, "/opt/bin/ffmpeg"))
        old, new = popen.call_args_list
        self.assertEqual(new.args, old.args)
        self.assertEqual(new.kwargs["stdout"], old.kwargs["stdout"])
        self.assertEqual(new.kwargs["cwd"], self.path.parent)
        self.assertEqual(len(actual), 2)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, (1, 2, 3))
            self.assertEqual(a.dtype, np.uint8)
        np.testing.assert_array_equal(actual[0][0, 0], [0, 1, 2])

    def test_truncated_bgr_keeps_original_error_and_closes_pipe(self):
        process = self.decoder(b"12345")
        with patch.object(m.subprocess, "Popen", return_value=process), self.assertRaisesRegex(ValueError, "Truncated native raw BGR frame"):
            list(m.decode_bgr(self.path, 2, 1))
        self.assertTrue(process.stdout.closed)
        process.wait.assert_called_once_with()

    def test_nonzero_decode_keeps_error_and_cleanup_without_signals(self):
        process = self.decoder(b"", 7)
        def fail(command, **kwargs):
            kwargs["stderr"].write(b"decode failed\xff")
            return process
        with patch.object(m.subprocess, "Popen", side_effect=fail), self.assertRaisesRegex(RuntimeError, "decode failed"):
            list(m.decode_bgr(self.path, 2, 1))
        self.assertTrue(process.stdout.closed)
        self.assertEqual(process.wait.call_count, 2)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_probe_nonzero_and_invalid_json_are_not_suppressed(self):
        failure = subprocess.CalledProcessError(2, ["ffprobe"])
        with patch.object(m.subprocess, "run", side_effect=failure), self.assertRaises(subprocess.CalledProcessError):
            m.probe_video(self.path)
        with patch.object(m.subprocess, "run", return_value=types.SimpleNamespace(stdout="bad JSON")), self.assertRaises(json.JSONDecodeError):
            m.probe_video(self.path)

    def test_relative_paths_fail_before_any_media_subprocess(self):
        for path in ("source.mp4", "local/source.mp4", "../source.mp4"):
            with self.subTest(path=path), patch.object(m.subprocess, "run") as run, patch.object(m.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "Absolute local media path"):
                    m.probe_video(path)
                with self.assertRaisesRegex(ValueError, "Absolute local media path"):
                    list(m.decode_bgr(path, 2, 1))
                run.assert_not_called()
                popen.assert_not_called()

    def test_empty_decode_and_early_generator_close_keep_cleanup(self):
        empty = self.decoder(b"")
        with patch.object(m.subprocess, "Popen", return_value=empty):
            self.assertEqual(list(m.decode_bgr(self.path, 2, 1)), [])
        self.assertTrue(empty.stdout.closed)
        process = self.decoder(bytes(range(12)))
        with patch.object(m.subprocess, "Popen", return_value=process):
            generator = m.decode_bgr(self.path, 2, 1)
            next(generator)
            generator.close()
        self.assertTrue(process.stdout.closed)
        process.wait.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
