"""CPU regression using actual MP4 decode/PTS and explicitly stubbed model outputs.

These test artifacts are not annotation results and assert no actual CUDA claim.
Run: /opt/venv_od/bin/python -m unittest discover -s runtime/smalltests -p test_curated_producer_io_v2.py -v
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import cv2
import numpy as np

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import curated_producer_io_v2 as io
sys.path.insert(0, str(RT / 'source_curated_producer_v1/repo/scripts_omnidance/exp/ms'))
import ms_hand_roi as roi


class RecordedShapeStub:
    """Known points exercise frozen code; this is not an inference/model identity."""
    def __init__(self):
        self.calls = 0

    def __call__(self, bgr):
        i = self.calls
        self.calls += 1
        if i == 5:
            return np.zeros((0, 134, 2)), np.zeros((0, 134)), np.zeros((0, 4))
        xy = np.full((1, 134, 2), 50., dtype=np.float32)
        score = np.full((1, 134), .9, dtype=np.float32)
        grid = np.array([(x, y) for y in (30., 45., 60.) for x in (30., 35., 40., 45., 50., 55., 60.)], np.float32)
        xy[0, 92:113] = grid
        xy[0, 113:134] = grid + [100., 30.]
        xy[0, 7], xy[0, 4] = xy[0, 92], xy[0, 113]
        if i == 7:
            xy[0, 92:134, 0] = -10.
        if i == 9:
            score[0, 95] = .3
        return xy, score, np.array([[10., 10., 240., 180.]], np.float32)


class LocalPublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='curated-io-v2-tests-')
        cls.root = Path(cls.temporary.name)
        cls.video = cls.root / 'actual_decode_fixture.mp4'
        subprocess.run(['/usr/local/bin/ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi',
                        '-i', 'testsrc2=size=256x192:rate=16:duration=5.0625',
                        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(cls.video)],
                       check=True, capture_output=True)
        cls.row = {'id': 'cpu_fixture', 'video_path': str(cls.video), 'source_group': 'cpu_only_fixture',
                   'source_ordinal': 500, 'split_ordinal': 500, 'split': 'train', 'batch_id': 'train-b000001'}
        cls.manifest = cls.root / 'cpu_manifest.jsonl'
        cls.manifest.write_bytes(roi.json_bytes(cls.row))
        identity_file = cls.root / 'explicitly_not_a_model.txt'
        identity_file.write_text('CPU regression shape stub, not an ONNX model')
        files = {'test_fixture_only': roi.file_identity(identity_file)}
        cls.model_ref = {'files': files, 'sha256': roi.digest(files), 'actual_inference_model': False}
        cls.prepared = roi.prepare_source({'id': cls.row['id'], 'manifest_record': cls.row,
                                          'input_line': 1, 'input_row_sha256': roi.digest(cls.row)},
                                         roi.file_identity(cls.manifest), cls.model_ref, dict(roi.DEFAULT_PARAMS),
                                         '/opt/ms_tools/ffprobe', '/usr/local/bin/ffmpeg')
        cls.original_root = cls.root / 'original'
        cls.original_root.mkdir()
        cls.original = roi.ensure_prepared(cls.original_root, cls.prepared)
        cls.original_manifest = roi.extract_source(cls.original, cls.prepared, RecordedShapeStub())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(prefix='case-', dir=self.root)
        self.case = Path(self.temporary_case.name)
        self.local = self.case / 'local'
        self.local.mkdir()
        self.shared = self.case / 'shared'
        self.shared.mkdir()
        self.final = self.shared / self.row['id']

    def tearDown(self):
        self.temporary_case.cleanup()

    def extract_local(self):
        adapter = io.LocalROI(roi, self.local)
        self.assertEqual(adapter.ensure_prepared(self.shared, self.prepared), self.final)
        self.assertFalse(self.final.exists())
        return adapter.extract_source(self.final, self.prepared, RecordedShapeStub())

    def copied_original(self):
        import shutil
        directory = self.local / self.row['id']
        shutil.copytree(self.original, directory)
        # Copied test bundle references remain original until we explicitly rebind output-only refs.
        mapped = io.remap_paths(self.original_manifest, self.original, directory)
        (directory / 'roi_manifest.json').write_bytes(roi.json_bytes(mapped))
        receipt = json.loads((directory / 'extract_receipt.json').read_bytes())
        receipt['roi_manifest_sha256'] = roi.sha256(directory / 'roi_manifest.json')
        (directory / 'extract_receipt.json').write_bytes(roi.json_bytes(receipt))
        return directory

    def test_actual_native_decode_pts_tail32_masks_affine_and_hashes(self):
        result = self.extract_local()
        self.assertEqual(self.prepared['probe']['decoded_video_frames'], 81)
        self.assertEqual(len(result['selected_frames']), 32)
        self.assertEqual([result['selected_frames'][i]['frame_index'] for i in (0, -1)], [0, 80])
        self.assertEqual(result['source'], self.prepared['source'])
        self.assertEqual(result['source']['path'], str(self.video))
        self.assertEqual((self.final / 'source_manifest.json').read_bytes(), (self.original / 'source_manifest.json').read_bytes())
        expected = io.remap_paths(self.original_manifest, self.original, self.final)
        for key in ('selected_frames', 'hand_observations', 'crops', 'binding_sha256', 'probe', 'model_ref'):
            self.assertEqual(result[key], expected[key])
        with np.load(self.original_manifest['raw']['path'], allow_pickle=False) as old, np.load(result['raw']['path'], allow_pickle=False) as new:
            self.assertEqual(old.files, new.files)
            for key in old.files:
                np.testing.assert_array_equal(old[key], new[key], err_msg=key)
            self.assertFalse(new['tracked'][5])
            self.assertFalse(new['valid'][5].any())
            self.assertTrue(np.isnan(new['xy'][5]).all())
            self.assertFalse(new['valid'][7, 92:134].any())
            self.assertFalse(new['valid'][9, 95])
            self.assertAlmostEqual(float(new['score'][9, 95]), .3)
        self.assertGreater(len(result['crops']), 0)
        for crop in result['crops']:
            x0, y0, x1, y1 = crop['crop_xyxy']
            image = cv2.imread(crop['path'])
            self.assertEqual(image.shape[:2], (y1 - y0, x1 - x0))
            self.assertFalse(crop['resized'])
            self.assertFalse(crop['rotated'])
            self.assertFalse(crop['mirrored'])
            self.assertEqual(crop['ROI_to_source_affine'], [[1., 0., float(x0)], [0., 1., float(y0)]])
            self.assertEqual(roi.sha256(crop['path']), crop['sha256'])
            self.assertTrue(Path(crop['path']).is_relative_to(self.final))
        receipt = json.loads((self.final / 'io_publish_receipt.json').read_bytes())
        self.assertFalse(receipt['source_media_staged'])
        self.assertFalse(receipt['source_manifest_rewritten'])
        self.assertEqual(receipt['raw_sha256_unchanged'], result['raw']['sha256'])
        self.assertEqual(list(self.local.iterdir()), [])
        evidence = roi.select_roi_evidence(result, 0., 5.0625)
        self.assertTrue(evidence['same_frame_two_hands_is_one_temporal_observation'])
        self.assertTrue(all(o['status'] == 'unknown' for o in result['hand_observations'] if o['frame_index'] == result['selected_frames'][7]['frame_index']))

    def test_existing_completed_is_verified_and_reused_without_stub_call(self):
        first = self.extract_local()
        stub = RecordedShapeStub()
        adapter = io.LocalROI(roi, self.local)
        adapter.ensure_prepared(self.shared, self.prepared)
        self.assertEqual(adapter.extract_source(self.final, self.prepared, stub), first)
        self.assertEqual(stub.calls, 0)

    def test_partial_shared_is_preserved_and_rejected(self):
        self.final.mkdir()
        marker = self.final / 'unknown_user_file'
        marker.write_text('preserve me')
        with self.assertRaises((OSError, ValueError)):
            io.LocalROI(roi, self.local).ensure_prepared(self.shared, self.prepared)
        self.assertEqual(marker.read_text(), 'preserve me')

    def test_incomplete_local_never_published(self):
        directory = roi.ensure_prepared(self.local, self.prepared)
        with self.assertRaises(ValueError):
            io.publish_roi(roi, directory, self.final, self.prepared, 0.)
        self.assertFalse(self.final.exists())

    def test_corrupt_raw_rejected_before_publish(self):
        directory = self.copied_original()
        with (directory / 'raw_wholebody134.npz').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'content SHA'):
            io.publish_roi(roi, directory, self.final, self.prepared, 0.)
        self.assertFalse(self.final.exists())

    def test_shared_copy_failure_cleans_only_own_stage(self):
        directory = self.copied_original()
        unrelated = self.shared / '.unrelated-stage'
        unrelated.mkdir()
        marker = unrelated / 'retain.txt'
        marker.write_text('untouched')
        real_sha = roi.sha256
        def corrupt_stage(path):
            return 'bad' if '.io-v2-' in str(path) else real_sha(path)
        with patch.object(roi, 'sha256', side_effect=corrupt_stage):
            with self.assertRaisesRegex(ValueError, 'Shared stage content SHA mismatch'):
                io.publish_roi(roi, directory, self.final, self.prepared, 0.)
        self.assertFalse(self.final.exists())
        self.assertEqual(marker.read_text(), 'untouched')
        self.assertEqual(list(self.shared.iterdir()), [unrelated])

    def test_rename_noreplace_never_overwrites_empty_existing(self):
        stage = self.shared / 'stage'
        stage.mkdir()
        (stage / 'asset').write_text('new')
        self.final.mkdir()
        with self.assertRaises(FileExistsError):
            io.rename_noreplace(stage, self.final)
        self.assertTrue((stage / 'asset').exists())
        self.assertEqual(list(self.final.iterdir()), [])

    def test_atomic_target_absent_until_copy_verified(self):
        directory = self.copied_original()
        real_rename = io.rename_noreplace
        called = []
        def checked_rename(stage, final):
            self.assertFalse(final.exists())
            self.assertEqual(stage.parent, final.parent)
            self.assertTrue((stage / 'extract_receipt.json').exists())
            self.assertTrue((stage / 'io_publish_receipt.json').exists())
            called.append(True)
            return real_rename(stage, final)
        with patch.object(io, 'rename_noreplace', side_effect=checked_rename):
            io.publish_roi(roi, directory, self.final, self.prepared, 0.)
        self.assertEqual(called, [True])
        roi.validate_completed(self.final, self.prepared)

    def test_local_cleanup_preserves_unknown_subtree(self):
        owned_root = self.local / 'explicitly_owned_test_root'
        owned_root.mkdir()
        directory = owned_root / self.row['id']
        directory.mkdir()
        (directory / 'source_manifest.json').write_bytes(roi.json_bytes(self.prepared))
        unknown = directory / 'unknown_local_directory'
        unknown.mkdir()
        marker = unknown / 'retain.txt'
        marker.write_text('user work')
        self.assertFalse(io.cleanup_local_generated(owned_root, self.prepared))
        self.assertEqual(marker.read_text(), 'user work')
        self.assertTrue((directory / 'source_manifest.json').exists())

    def test_explicit_selection_disjoint_shards_and_unknown_id(self):
        pipeline = io.load_pipeline()
        rows = [{'id': 'sid' + str(i), 'source_ordinal': i, 'split': 'train', 'batch_id': 'batch'} for i in range(495, 1505)]
        path = self.case / 'batch.jsonl'
        path.write_bytes(b''.join(pipeline.raw_json(row) for row in rows))
        summary_path = self.case / 'summary.json'
        pipeline.save(summary_path, {'batches': [{**pipeline.ref(path), 'split': 'train', 'batch_id': 'batch', 'count': len(rows)}]})
        args = SimpleNamespace(ids=None, source_ids_file=None, ordinal_start=500, ordinal_end=1500,
                               manifest=str(summary_path), shard=0, nshard=2, limit=0)
        first = io.select_ids(pipeline, args)
        args.shard = 1
        second = io.select_ids(pipeline, args)
        self.assertEqual((len(first), len(second)), (500, 500))
        self.assertFalse(set(first) & set(second))
        self.assertEqual(set(first) | set(second), {'sid' + str(i) for i in range(500, 1500)})
        args.ordinal_start = args.ordinal_end = None
        with self.assertRaisesRegex(ValueError, 'Explicit IDs'):
            io.select_ids(pipeline, args)
        args.ids = ['absent_sid']
        with self.assertRaisesRegex(ValueError, 'absent from verified manifest'):
            io.select_ids(pipeline, args)


if __name__ == '__main__':
    unittest.main()
