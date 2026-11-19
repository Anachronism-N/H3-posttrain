"""Actual CephFS CPU tests: exclusive mkdir/hardlinks, final receipt commit, native ROI.

All output is unique CPU-only test fixture directories, never production queues.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
import unittest

sys.dont_write_bytecode = True
import test_curated_producer_io_v2 as old
sys.path.insert(0, str(old.RT))
import curated_producer_io_v3 as io


class CephLastReceiptTests(old.LocalPublishTests):
    def setUp(self):
        super().setUp()
        self.ceph_temp = tempfile.TemporaryDirectory(prefix='cpu-ceph-io-v3-', dir=old.RT / 'smalltests')
        self.shared = Path(self.ceph_temp.name)
        self.final = self.shared / self.row['id']
        self.assertNotEqual(self.shared.stat().st_dev, self.local.stat().st_dev)

    def tearDown(self):
        self.ceph_temp.cleanup()  # Only this test's unique, explicitly owned fixture subtree.
        super().tearDown()

    def extract_local(self):
        with patch.object(old.io, 'publish_roi', io.publish_roi):
            return super().extract_local()

    def test_atomic_target_absent_until_copy_verified(self):
        directory = self.copied_original()
        real_link = os.link
        seen = []
        def checked_link(source, target, *args, **kwargs):
            relative = Path(target).relative_to(self.final).as_posix()
            self.assertTrue(self.final.exists())
            if relative == 'extract_receipt.json':
                self.assertTrue((self.final / 'roi_manifest.json').exists())
                self.assertTrue((self.final / 'io_publish_ownership.json').exists())
                for asset in old.roi.validate_completed(directory, self.prepared)['crops']:
                    self.assertTrue((self.final / 'crops' / Path(asset['path']).name).exists())
            elif relative not in ('io_publish_receipt.json',):
                self.assertFalse((self.final / 'extract_receipt.json').exists())
            seen.append(relative)
            return real_link(source, target, *args, **kwargs)
        with patch.object(io.os, 'link', side_effect=checked_link):
            io.publish_roi(old.roi, directory, self.final, self.prepared, 0.)
        self.assertEqual(seen[0], 'io_publish_ownership.json')
        self.assertEqual(seen[-3:], ['roi_manifest.json', 'extract_receipt.json', 'io_publish_receipt.json'])
        old.roi.validate_completed(self.final, self.prepared)
        receipt = json.loads((self.final / 'io_publish_receipt.json').read_bytes())
        self.assertFalse(receipt['whole_directory_atomic'])
        self.assertTrue(receipt['complete_roi_validated'])
        self.assertGreaterEqual(receipt['shared_publish_seconds'], receipt['shared_copy_verify_seconds'])

    def test_rename_noreplace_never_overwrites_empty_existing(self):
        # V3 never calls rename; even an empty existing final dir must remain untouched.
        directory = self.copied_original()
        self.final.mkdir()
        with self.assertRaises(FileExistsError):
            io.publish_roi(old.roi, directory, self.final, self.prepared, 0.)
        self.assertEqual(list(self.final.iterdir()), [])

    def test_shared_copy_failure_cleans_only_own_stage(self):
        directory = self.copied_original()
        unrelated = self.shared / '.unknown-stage'
        unrelated.mkdir()
        marker = unrelated / 'retain.txt'
        marker.write_text('untouched')
        real_sha = old.roi.sha256
        def corrupt_stage(path):
            return 'bad' if '.io-v3-' in str(path) else real_sha(path)
        with patch.object(old.roi, 'sha256', side_effect=corrupt_stage):
            with self.assertRaisesRegex(ValueError, 'Shared stage content SHA mismatch'):
                io.publish_roi(old.roi, directory, self.final, self.prepared, 0.)
        self.assertFalse(self.final.exists())
        self.assertEqual(marker.read_text(), 'untouched')
        self.assertEqual(list(self.shared.iterdir()), [unrelated])

    def test_link_failure_preserves_owned_partial_final_and_no_extraction_receipt(self):
        directory = self.copied_original()
        real_link = os.link
        def failed_link(source, target, *args, **kwargs):
            if Path(target).name == 'raw_wholebody134.npz':
                raise OSError('explicit test link failure')
            return real_link(source, target, *args, **kwargs)
        with patch.object(io.os, 'link', side_effect=failed_link):
            with self.assertRaisesRegex(OSError, 'explicit test link failure'):
                io.publish_roi(old.roi, directory, self.final, self.prepared, 0.)
        self.assertTrue(self.final.exists())
        ownership = json.loads((self.final / 'io_publish_ownership.json').read_bytes())
        self.assertEqual(ownership['source_binding_sha256'], self.prepared['binding_sha256'])
        self.assertFalse((self.final / 'extract_receipt.json').exists())
        self.assertEqual(list(self.shared.iterdir()), [self.final])
        with self.assertRaises((OSError, ValueError)):
            old.io.LocalROI(old.roi, self.local).ensure_prepared(self.shared, self.prepared)
        self.assertTrue((self.final / 'io_publish_ownership.json').exists())

    def test_ceph_mkdir_and_hardlink_reject_replacement(self):
        stage = self.shared / 'cpu_stage'
        stage.mkdir()
        source = stage / 'source'
        with source.open('xb') as stream:
            stream.write(b'CPU test source bytes')
            stream.flush()
            os.fsync(stream.fileno())
        final = self.shared / 'cpu_final'
        final.mkdir(exist_ok=False)
        target = final / 'target'
        os.link(source, target)
        old.io.fsync_directory(final)
        old.io.fsync_directory(self.shared)
        self.assertEqual(old.roi.sha256(source), old.roi.sha256(target))
        self.assertEqual(source.stat().st_ino, target.stat().st_ino)
        with self.assertRaises(FileExistsError):
            os.link(source, target)
        with self.assertRaises(FileExistsError):
            final.mkdir(exist_ok=False)
        self.assertEqual(target.read_bytes(), b'CPU test source bytes')


if __name__ == '__main__':
    unittest.main()
