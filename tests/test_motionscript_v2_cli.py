"""End-to-end cache/provenance tests using synthetic, explicitly labelled NLF data."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_optimize as cli


def write_fixture(root: Path):
    nlf = root / "nlf"
    nlf.mkdir()
    joints = np.zeros((52, 3), dtype=np.float32)
    joints[:22] = [
        [0, 0.9, 0], [.1, .85, 0], [-.1, .85, 0], [0, 1.05, 0],
        [.1, .5, 0], [-.1, .5, 0], [0, 1.2, 0], [.1, .12, 0],
        [-.1, .12, 0], [0, 1.35, 0], [.1, .05, .15], [-.1, .05, .15],
        [0, 1.5, 0], [.08, 1.4, 0], [-.08, 1.4, 0], [0, 1.7, 0],
        [.2, 1.4, 0], [-.2, 1.4, 0], [.45, 1.4, 0], [-.45, 1.4, 0],
        [.7, 1.4, 0], [-.7, 1.4, 0],
    ]
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        9, 9, 12, 13, 14, 16, 17, 18, 19] + [20] * 30)
    joints[22:] = joints[20]
    template = root / "template.npz"
    np.savez(template, J_template=joints, parents=parents)
    frames = 81
    pose = np.zeros((frames, 72), dtype=np.float32)
    pose[:, 0] = np.pi  # Camera-axis conversion gives an upright Y-up root.
    j3d = np.zeros((frames, 24, 3), dtype=np.float32)
    j3d[:, 0, 0] = np.linspace(0, .6, frames)
    j3d[:, 0, 1] = -.95
    j3d[:, 0, 2] = 3.
    source = nlf / "synthetic.npz"
    np.savez(source, pose=pose, j3d=j3d, valid=np.ones(frames, bool),
             unc=np.full((frames, 24), .01, dtype=np.float32), fps=16.)
    return nlf, source, template


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nlf, self.source, self.template = write_fixture(self.root)
        self.args = cli.parser().parse_args([
            "--nlf", str(self.nlf), "--template", str(self.template),
            "--out", str(self.root / "v2"), "--emit-llm-payload",
        ])

    def run_cli(self, args=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.run(args or self.args)

    def test_complete_dataset_and_identical_resume(self):
        before = cli.sha256(self.source)
        manifest = self.run_cli()
        self.assertEqual(manifest["count"], 1)
        self.assertFalse(manifest["semantic_approval"])
        out = Path(self.args.out)
        record = json.loads((out / "synthetic.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source"], "motionscript_v2_rules")
        self.assertFalse(record["hand_fingers_observed"])
        self.assertIsNone(record["upstream"])
        self.assertIn("llm_payload", record)
        self.assertEqual(record["event_summary"]["source_events"], record["events"])
        self.assertEqual(record["compact_caption"], record["event_summary"]["compact_caption"])
        self.assertEqual(record["llm_payload"]["task"], "fact_preserving_compact_motion_rewrite")
        self.assertEqual(record["llm_payload"]["source_events"], record["events"])
        self.assertFalse(record["event_summary"]["vocabulary_coverage"]["open_vocabulary"])
        with np.load(out / "synthetic.coords.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["coords"].shape, (101, 54, 3))
            np.testing.assert_allclose(np.diff(arrays["time"]), .05, atol=1e-12)
            self.assertEqual(arrays["global_joints"].dtype, np.float32)
        self.args.resume = True
        self.assertEqual(manifest, self.run_cli())
        self.assertEqual(before, cli.sha256(self.source))

    def test_resume_rejects_changed_input(self):
        self.run_cli()
        self.args.resume = True
        with np.load(self.source) as data:
            changed = {key: data[key] for key in data.files}
        changed["j3d"][:, 0, 0] += 1.
        np.savez(self.source, **changed)
        with self.assertRaisesRegex(ValueError, "input or protocol mismatch"):
            self.run_cli()

    def test_resume_rejects_coordinate_or_report_tampering(self):
        self.run_cli()
        self.args.resume = True
        coords = Path(self.args.out) / "synthetic.coords.npz"
        original = coords.read_bytes()
        coords.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "altered coordinates"):
            self.run_cli()
        coords.write_bytes(original)
        report = Path(self.args.out) / "synthetic.json"
        record = json.loads(report.read_text())
        record["caption"] = ["invented hand contact"]
        report.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "report receipt"):
            self.run_cli()

    def test_resume_rejects_changed_config(self):
        self.run_cli()
        self.args.resume = True
        self.args.max_gap_seconds = .5
        with self.assertRaisesRegex(ValueError, "resume protocol differs"):
            self.run_cli()

    def test_resume_rejects_altered_manifest(self):
        self.run_cli()
        self.args.resume = True
        path = Path(self.args.out) / "manifest.json"
        record = json.loads(path.read_text())
        record["semantic_approval"] = True
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "altered manifest"):
            self.run_cli()

    def test_reserved_identifier_cannot_replace_dataset_protocol(self):
        self.source.rename(self.nlf / "protocol.npz")
        with self.assertRaisesRegex(ValueError, "reserved artifact name"):
            self.run_cli()
        self.assertFalse(Path(self.args.out).exists())

    def test_interruption_before_report_commit_can_resume(self):
        original_write = cli.atomic_write

        def interrupt(path, data):
            if path.name == "synthetic.json":
                raise OSError("simulated interruption before report commit")
            original_write(path, data)

        with mock.patch.object(cli, "atomic_write", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                self.run_cli()
        out = Path(self.args.out)
        self.assertFalse((out / "synthetic.json").exists())
        self.assertTrue((out / "synthetic.receipt.json").exists())
        self.args.resume = True
        self.assertTrue(self.run_cli()["complete"])

    def test_interruption_before_manifest_commit_can_resume(self):
        original_write = cli.atomic_write

        def interrupt(path, data):
            if path.name == "manifest.json":
                raise OSError("simulated interruption before manifest commit")
            original_write(path, data)

        with mock.patch.object(cli, "atomic_write", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                self.run_cli()
        self.assertFalse((Path(self.args.out) / "manifest.json").exists())
        self.args.resume = True
        self.assertTrue(self.run_cli()["complete"])

    def test_input_hash_and_arrays_use_same_byte_snapshot(self):
        old_bytes = self.source.read_bytes()
        with np.load(io.BytesIO(old_bytes)) as data:
            changed = {key: data[key] for key in data.files}
        changed["j3d"][:, 0, 0] += 2.
        buffer = io.BytesIO()
        np.savez(buffer, **changed)
        original_read = Path.read_bytes
        replaced = []

        def replace_after_read(path):
            data = original_read(path)
            if path.resolve() == self.source.resolve() and not replaced:
                path.write_bytes(buffer.getvalue())
                replaced.append(True)
            return data

        with mock.patch.object(Path, "read_bytes", replace_after_read):
            manifest = self.run_cli()
        self.assertEqual(manifest["records"][0]["input_sha256"], hashlib.sha256(old_bytes).hexdigest())
        with np.load(Path(self.args.out) / "synthetic.coords.npz") as arrays:
            self.assertAlmostEqual(float(arrays["global_joints"][:, 0, 0].mean()), .3, places=5)
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, "input or protocol mismatch"):
            self.run_cli()

    def test_upstream_contract_does_not_stitch_hidden_joint_gap(self):
        # Test the integration boundary with an explicit stub; this is not a
        # claim that upstream MotionScript/PyTorch ran in the local environment.
        src = self.root / "upstream"
        src.mkdir()
        calls = []
        captioning = types.ModuleType("text2pose.posescript.captioning")
        captioning.__file__ = str(src / "captioning.py")

        def caption(tensor, **kwargs):
            calls.append((tensor.copy(), kwargs))
            return None, None, ["unaggregated"], ["aggregated"]

        captioning.main = caption
        posescript = types.ModuleType("text2pose.posescript")
        posescript.captioning = captioning
        package = types.ModuleType("text2pose")
        package.posescript = posescript
        torch = types.ModuleType("torch")
        torch.float32 = np.float32
        torch.manual_seed = lambda _: None
        torch.as_tensor = lambda array, dtype: np.asarray(array, dtype=dtype)
        arrays = {"valid": np.ones(12, bool), "joint_valid": np.ones((12, 22), bool),
                  "segment_id": np.zeros(12, dtype=np.int32),
                  "joint_segment_id": np.zeros((12, 22), dtype=np.int32),
                  "time": np.arange(12) * .05, "coords": np.zeros((12, 54, 3), np.float32)}
        arrays["joint_segment_id"][6:, 18] = 1
        original_path = sys.path.copy()
        try:
            with mock.patch.dict(sys.modules, {"torch": torch, "text2pose": package,
                    "text2pose.posescript": posescript, "text2pose.posescript.captioning": captioning}):
                result = cli.original_captions(arrays, src, None)
        finally:
            sys.path[:] = original_path
        self.assertEqual([len(call[0]) for call in calls], [6, 6])
        self.assertTrue(all(call[1]["motion_tracking"] for call in calls))
        np.testing.assert_allclose([row["start"] for row in result], [0., .3], atol=1e-12)
        self.assertEqual(result[0]["source"], "original_motionscript_rules_on_v2_coordinates")

    def test_refuses_output_inside_input_and_nonempty_legacy_directory(self):
        self.args.out = str(self.nlf / "v2")
        with self.assertRaisesRegex(ValueError, "outside the NLF"):
            self.run_cli()
        legacy = self.root / "ms_orig"
        legacy.mkdir()
        marker = legacy / "synthetic.json"
        marker.write_text('{"ms":["original"]}', encoding="utf-8")
        self.args.out = str(legacy)
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.run_cli()
        self.assertEqual(marker.read_text(), '{"ms":["original"]}')

    def test_missing_requested_id_does_not_create_output(self):
        ids = self.root / "ids.txt"
        ids.write_text("missing\n", encoding="utf-8")
        self.args.ids = str(ids)
        with self.assertRaisesRegex(ValueError, "no NLF NPZ"):
            self.run_cli()
        self.assertFalse(Path(self.args.out).exists())


if __name__ == "__main__":
    unittest.main()
