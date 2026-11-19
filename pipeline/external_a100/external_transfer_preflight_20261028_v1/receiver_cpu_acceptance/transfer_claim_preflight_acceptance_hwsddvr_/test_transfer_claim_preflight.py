"""Synthetic temporary claim domains only; no real source or model access."""
import copy
import errno
import hashlib
import json
import multiprocessing
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transfer_claim_preflight import ConsumptionUncertain, consume_original_claim, verify_boundary


def die_after_create(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os._exit(7)


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_transfer_', dir='/tmp')
        self.root = Path(self.tmp.name)
        (self.root / 'train').mkdir()
        self.row = {'split': 'train', 'id': 'SYNTHETIC_ONLY'}
        self.record = {'source_row': self.row, 'automatic_retry': False, 'worker': 'a100-extra-01', 'epoch': 'synthetic_new'}
        self.claim = self.root / 'train/SYNTHETIC_ONLY.json'

    def tearDown(self):
        self.tmp.cleanup()

    def take(self, **kwargs):
        return consume_original_claim(self.root, self.row, self.record, **kwargs)

    def test_exact_readback(self):
        result = self.take()
        self.assertEqual(result['claim_sha256'], hashlib.sha256(self.claim.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.claim.read_bytes()), self.record)

    def test_two_consumers_one_winner(self):
        def attempt(n):
            record = dict(self.record, worker=['a100-extra-01', 'a100-extra-02'][n])
            try:
                consume_original_claim(self.root, self.row, record)
                return True
            except FileExistsError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(attempt, [0, 1])), 1)

    def test_new_epoch_does_not_reset_source_budget(self):
        self.take()
        original = self.claim.read_bytes()
        self.record['epoch'] = 'synthetic_another'
        with self.assertRaises(FileExistsError):
            self.take()
        self.assertEqual(self.claim.read_bytes(), original)

    def test_partial_claim_retained(self):
        self.claim.write_bytes(b'{')
        with self.assertRaises(FileExistsError):
            self.take()
        self.assertEqual(self.claim.read_bytes(), b'{')

    def test_process_death_zero_byte_claim_retained(self):
        process = multiprocessing.get_context('fork').Process(target=die_after_create, args=(str(self.claim),))
        process.start(); process.join(5)
        self.assertEqual(process.exitcode, 7)
        with self.assertRaises(FileExistsError):
            self.take()
        self.assertEqual(self.claim.read_bytes(), b'')

    def test_transient_write_failure_not_retried(self):
        calls = []
        def fail(fd, raw):
            calls.append(1)
            raise OSError(errno.ENOTCONN, 'synthetic transport failure')
        with self.assertRaises(ConsumptionUncertain):
            self.take(write=fail)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.claim.read_bytes(), b'')
        with self.assertRaises(FileExistsError):
            self.take()

    def test_directory_fsync_uncertain_no_replay(self):
        calls = []
        def sync(fd):
            calls.append(1)
            if len(calls) == 2:
                raise OSError(errno.EIO, 'synthetic post-create directory sync failure')
            os.fsync(fd)
        with self.assertRaises(ConsumptionUncertain):
            self.take(fsync=sync)
        self.assertEqual(json.loads(self.claim.read_bytes()), self.record)
        with self.assertRaises(FileExistsError):
            self.take()

    def test_short_writes_fully_preserved(self):
        result = self.take(write=lambda fd, raw: os.write(fd, raw[:3]))
        self.assertEqual(json.loads(self.claim.read_bytes()), self.record)
        self.assertTrue(result['created_exactly_once'])

    def test_source_path_traversal_rejected(self):
        self.row['id'] = '../unsafe'
        with self.assertRaises(ValueError):
            self.take()

    def test_split_symlink_rejected(self):
        other = self.root / 'other'; other.mkdir()
        (self.root / 'train').rmdir(); (self.root / 'train').symlink_to(other, target_is_directory=True)
        with self.assertRaises(OSError):
            self.take()
        self.assertFalse((other / 'SYNTHETIC_ONLY.json').exists())


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.row = {'split': 'train', 'id': 'SYNTHETIC_ONLY'}
        self.original = {'a100-extra-03': [self.row]}
        self.manifest = {'predecessor_epoch': 'synthetic_old', 'next_epoch': 'synthetic_new',
            'original_claim_root': '/synthetic/domain/claims', 'automatic_retry': False, 'old_queue_append': False,
            'source_count': 1, 'rows': [{'original_worker': 'a100-extra-03', 'original_assignment_index': 0,
            'source_row': self.row, 'receiver': 'a100-extra-01', 'claim_absence_verified_after_retirement': True}]}
        self.release = {'predecessor_epoch': 'synthetic_old', 'original_claim_root': '/synthetic/domain/claims',
            'donor_assignment_sha256': {'a100-extra-03': 'a' * 64}, 'durable_enforcement_receipt_verified': True,
            'enforcement_evidence_ref': {'path': '/synthetic/receipt', 'sha256': 'b' * 64}}

    def check(self, **changes):
        m = json.dumps(self.manifest).encode(); r = json.dumps(self.release).encode()
        trusted = {'manifest_sha256': hashlib.sha256(m).hexdigest(), 'release_sha256': hashlib.sha256(r).hexdigest(),
            'predecessor_epoch': 'synthetic_old', 'next_epoch': 'synthetic_new',
            'original_claim_root': '/synthetic/domain/claims', 'donor_assignment_sha256': {'a100-extra-03': 'a' * 64}}
        trusted.update(changes)
        return verify_boundary(m, r, trusted, self.original)

    def test_success_is_structural_not_production_permission(self):
        self.assertFalse(self.check()['production_permission'])

    def test_missing_independent_expected_boundary_denied(self):
        with self.assertRaises(PermissionError):
            verify_boundary(b'{}', b'{}', None, self.original)

    def test_wrong_release_sha_denied(self):
        with self.assertRaises(PermissionError):
            self.check(release_sha256='0' * 64)

    def test_unenforced_marker_denied(self):
        self.release['durable_enforcement_receipt_verified'] = False
        with self.assertRaises(PermissionError):
            self.check()

    def test_old_epoch_reuse_denied(self):
        self.manifest['next_epoch'] = 'synthetic_old'
        with self.assertRaises(PermissionError):
            self.check(next_epoch='synthetic_old')

    def test_old_queue_append_denied(self):
        self.manifest['old_queue_append'] = True
        with self.assertRaises(PermissionError):
            self.check()

    def test_new_claim_namespace_denied(self):
        self.manifest['original_claim_root'] = '/synthetic/replacement'
        with self.assertRaises(PermissionError):
            self.check()

    def test_duplicate_source_denied(self):
        self.manifest['rows'].append(copy.deepcopy(self.manifest['rows'][0])); self.manifest['source_count'] = 2
        with self.assertRaises(PermissionError):
            self.check()

    def test_claim_absence_before_retirement_denied(self):
        self.manifest['rows'][0]['claim_absence_verified_after_retirement'] = False
        with self.assertRaises(PermissionError):
            self.check()

    def test_original_source_row_changed_denied(self):
        self.manifest['rows'][0]['source_row'] = {'split': 'train', 'id': 'DIFFERENT'}
        with self.assertRaises(PermissionError):
            self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
