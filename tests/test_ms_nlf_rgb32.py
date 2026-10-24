"""Synthetic API contracts only, not RGB32 accuracy claims."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_nlf_rgb32 as cli
from ms_landmarks import load_bundle


def result(counts):
    return dict(boxes=[np.tile([1., 2., 10., 20., .9], (c, 1)) for c in counts],
        poses3d=[np.full((c, 32, 3), 1000., np.float32) for c in counts],
        poses2d=[np.full((c, 32, 2), 4., np.float32) for c in counts],
        uncertainties=[np.full((c, 32), .1, np.float32) for c in counts])


class SelectionTests(unittest.TestCase):
    def test_largest_xywh_area_not_confidence(self):
        r = result([2])
        r["boxes"][0] = np.asarray([[0, 0, 10, 10, .99], [2, 3, 20, 20, .1]], float)
        r["poses3d"][0][1] = 2000
        out = cli.select_largest_detections(r, 1)
        self.assertEqual(out["selected_index"].tolist(), [1])
        np.testing.assert_equal(out["points_m"], 2.)
        self.assertEqual(out["all_points_m"].shape, (2, 32, 3))
        np.testing.assert_equal(out["all_points_m"][0], 1.)
        self.assertEqual(out["box"][0, 4], .1)

    def test_empty_keeps_nan_mask_and_ragged_shapes(self):
        out = cli.select_largest_detections(result([0, 0]), 2)
        self.assertFalse(out["valid"].any())
        self.assertTrue(np.isnan(out["points_m"]).all())
        self.assertTrue(np.isnan(out["box"]).all())
        self.assertEqual(out["selected_index"].tolist(), [-1, -1])
        self.assertEqual(out["all_boxes"].shape, (0, 5))
        self.assertEqual(out["all_points_m"].shape, (0, 32, 3))

    def test_mixed_empty_and_tie_chooses_first(self):
        out = cli.select_largest_detections(result([2, 0, 1]), 3)
        self.assertEqual(out["selected_index"].tolist(), [0, -1, 0])
        self.assertEqual(out["detection_count"].tolist(), [2, 0, 1])
        self.assertEqual(out["valid"].sum(), 64)

    def test_numeric_invalid_never_gets_filled(self):
        r = result([1])
        r["poses3d"][0][0, 3] = np.nan
        r["poses2d"][0][0, 5] = np.inf
        r["uncertainties"][0][0, 7] = -1
        out = cli.select_largest_detections(r, 1)
        self.assertFalse(out["valid"][0, [3, 5, 7]].any())
        self.assertEqual(out["valid"].sum(), 29)

    def test_bad_batch_and_point_counts_rejected(self):
        for key in ("boxes", "poses3d", "poses2d", "uncertainties"):
            with self.subTest(key=key):
                r = result([1])
                r[key] = []
                with self.assertRaises(ValueError):
                    cli.select_largest_detections(r, 1)
        r = result([1])
        r["poses3d"][0] = np.zeros((1, 31, 3))
        with self.assertRaises(ValueError):
            cli.select_largest_detections(r, 1)

    def test_bad_boxes_rejected(self):
        for box in ([0, 0, 0, 3, .9], [0, 0, -1, 3, .9], [0, 0, 3, 4, np.nan]):
            with self.subTest(box=box):
                r = result([1])
                r["boxes"][0] = np.asarray([box], float)
                with self.assertRaises(ValueError):
                    cli.select_largest_detections(r, 1)
        r = result([1])
        r["boxes"][0] = np.zeros((1, 4))
        with self.assertRaises(ValueError):
            cli.select_largest_detections(r, 1)

    def test_non_numeric_field_rejected(self):
        r = result([1])
        r["poses2d"][0] = np.full((1, 32, 2), 'bad')
        with self.assertRaises(ValueError):
            cli.select_largest_detections(r, 1)


class ManifestTests(unittest.TestCase):
    def write(self, tmp, rows):
        p = Path(tmp) / 'manifest.jsonl'
        p.write_text(''.join(json.dumps(row) + chr(10) for row in rows))
        return p

    def row(self, sid='one'):
        return dict(id=sid, video_path='/data/' + sid + '.mp4', source_group='/source/' + sid, split='test',
            nlf_path='/deliberately/absent/legacy.npz')

    def test_provenance_only_nlf_path_not_accessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self.row()
            self.assertEqual(cli.read_rows(self.write(tmp, [row]), 0), [row])

    def test_duplicates_even_outside_limit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                cli.read_rows(self.write(tmp, [self.row(), self.row()]), 1)

    def test_unsafe_id_relative_path_and_missing_source_rejected(self):
        for field, value in [('id', '../bad'), ('video_path', 'relative.mp4'), ('source_group', ''), ('split', 'unknown')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                row = self.row()
                row[field] = value
                with self.assertRaises(ValueError):
                    cli.read_rows(self.write(tmp, [row]), 0)

    def test_limit_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.write(tmp, [self.row('one'), self.row('two')])
            self.assertEqual(len(cli.read_rows(p, 1)), 1)
            with self.assertRaises(ValueError):
                cli.read_rows(self.write(tmp, []), 0)


class ExtractionTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'inputs'
        source.mkdir()
        for name in ('video.mp4', 'model.torchscript'):
            (source / name).write_bytes(b'synthetic-not-real-media')
        spec = dict(schema='motionscript_landmark_spec_v1', units='m', canonical_frame='nlf_canonical_smplx',
            points=[dict(name='point_' + str(i), canonical=[float(i) / 100, 0., 0.]) for i in range(32)])
        (source / 'spec.json').write_text(json.dumps(spec))
        rows = [dict(id='one', video_path=str(source / 'video.mp4'), source_group='synthetic', split='test', nlf_path='/absent/not-read.npz')]
        (source / 'input.jsonl').write_text(json.dumps(rows[0]) + chr(10))
        args = cli.parser().parse_args(['--manifest', str(source / 'input.jsonl'), '--model', str(source / 'model.torchscript'),
            '--spec', str(source / 'spec.json'), '--out', str(root / 'output'), '--bs', '2', '--device', 'cpu'])
        info = dict(time=np.asarray([0., .0625, .125]), fps=16., width=8, height=10, pts_offset=7.,
            media_end=.1875, media_end_source='synthetic decoded last frame duration')
        return args, info

    def predictor(self, *unused):
        class Fake:
            method_schemas = {'detect_poses_batched': 'synthetic', 'get_weights_for_canonical_points': 'synthetic'}
            def __init__(self):
                self.called = 0
            def __call__(self, frames):
                if frames.dtype != np.uint8 or frames.ndim != 4:
                    raise AssertionError('RGB batch contract')
                self.called += 1
                return result([2, 0] if self.called == 1 else [1])
        return Fake()

    def test_full_pipeline_receipt_pts_empty_and_no_oldboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, info = self.fixture(Path(tmp))
            batches = [(0, np.zeros((2, 10, 8, 3), np.uint8)), (2, np.zeros((1, 10, 8, 3), np.uint8))]
            with mock.patch.object(cli, 'probe_video', return_value=info), mock.patch.object(cli, 'decode_batches', return_value=iter(batches)):
                terminal = cli.run(args, self.predictor)
            self.assertTrue(terminal['complete'])
            self.assertEqual(terminal['actual'], 1)
            arrays, m, _ = load_bundle(Path(args.out) / 'raw/one')
            self.assertEqual(m['source'], 'nlf_fresh_detector_rgb_landmarks')
            self.assertTrue(m['no_external_boxes_supplied'])
            self.assertFalse(m['smpl_fitting_used'])
            self.assertFalse(m['template_fk_used'])
            self.assertFalse(m['person_tracking_added'])
            self.assertEqual(m['source_pts_offset_seconds'], 7.)
            self.assertEqual(m['media_bounds']['end'], .1875)
            self.assertNotIn('boxes', m['source_paths'])
            self.assertEqual(arrays['detection_offsets'].tolist(), [0, 2, 2, 3])
            self.assertEqual(arrays['selected_index'].tolist(), [0, -1, 0])
            self.assertTrue(np.isnan(arrays['points_m'][1]).all())
            np.testing.assert_array_equal(arrays['timestamps'], info['time'])
            self.assertEqual(arrays['names'].shape, (32,))
            self.assertFalse(m['visibility_verified'])
            self.assertFalse(m['semantic_approval'])

    def test_original_guard_rejects_nested_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, info = self.fixture(root)
            args.out = str(root / 'inputs' / 'nested_output')
            with self.assertRaises(ValueError), mock.patch.object(cli, 'FreshDetector32') as factory:
                cli.run(args)
            factory.assert_not_called()

    def test_changed_input_rejected_before_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, info = self.fixture(Path(tmp))
            def baddecode(*unused):
                Path(args.manifest).write_text('changed')
                yield 0, np.zeros((3, 10, 8, 3), np.uint8)
            def allthree(*unused):
                fake = self.predictor()
                fake.__class__.__call__ = lambda self, frames: result([1, 0, 1])
                return fake
            with mock.patch.object(cli, 'probe_video', return_value=info), mock.patch.object(cli, 'decode_batches', side_effect=baddecode):
                with self.assertRaises(ValueError):
                    cli.run(args, allthree)
            self.assertFalse((Path(args.out) / 'finished.json').exists())

    def test_batch_order_or_frame_count_rejected(self):
        for batches in ([(1, np.zeros((2, 10, 8, 3), np.uint8))], [(0, np.zeros((2, 10, 8, 3), np.uint8))]):
            with self.subTest(batches=len(batches)), tempfile.TemporaryDirectory() as tmp:
                args, info = self.fixture(Path(tmp))
                with mock.patch.object(cli, 'probe_video', return_value=info), mock.patch.object(cli, 'decode_batches', return_value=iter(batches)):
                    with self.assertRaises(ValueError):
                        cli.run(args, self.predictor)

    def test_exact32_spec_required_before_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, info = self.fixture(Path(tmp))
            spec = json.loads(Path(args.spec).read_text())
            spec['points'].pop()
            Path(args.spec).write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                cli.run(args, self.predictor)


if __name__ == '__main__':
    unittest.main()
