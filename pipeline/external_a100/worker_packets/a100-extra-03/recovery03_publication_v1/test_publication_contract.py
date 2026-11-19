"""CPU fixtures; only temporary local files written, never shared outputs/claims/models."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import recover_publication as r
from frozen_deep_contract import validate
sys.path.insert(0, str(r.L / 'unified_code_snapshot_v1'))
import ms_fullframe_pose2d_draft_v2 as rules

class PublicationContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='publication-cpu-fixture-')
        self.root = Path(self.tmp.name)
        self.gpu = 0
        self.row = {'id': 'Fixture', 'split': 'train', 'video_path': '/fixture/Fixture.mp4',
                    'audio_path': '/fixture/Fixture.wav', 'meta_path': '/fixture/Fixture.json', 'source_ordinal': 1}
        self.progress = {'pid': 99999999, 'start_ticks': 10, 'run_id': r.RUNS[0]}
        self.receipt = {'assignment_index': 14080, 'source_row': self.row}
        self.claim = {'edition': r.P.name, 'worker': r.WORKER, 'gpu': 0, 'assignment_sha256': 'a'*64,
            'assignment_index': 14080, 'source_row': self.row, 'split': 'train', 'id': 'Fixture',
            'pid': 99999999, 'owner_pid': 99999999, 'start_ticks': 10, 'run_id': r.RUNS[0], 'automatic_retry': False}
        self.files = ('pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json', 'receipt.json')
        self.local = self.root / 'local'
        self.payloads = {'cases/train/Fixture/' + n: (n + '\n').encode() for n in self.files}
        for relative, raw in self.payloads.items():
            path = self.local / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.archive = r.build_archive(self.local, self.payloads, self.files)
        self.target = self.root / 'shared-fixture'
    def tearDown(self):
        self.tmp.cleanup()
    def test_complete_member_order_and_payload_bytes(self):
        self.assertEqual(r.archive_contract(self.archive, self.payloads), 5)
        with tarfile.open(fileobj=io.BytesIO(self.archive), mode='r:') as pack:
            self.assertEqual(pack.getnames(), list(self.payloads))
    def test_archive_wrong_sha_bytes(self):
        altered = dict(self.payloads)
        altered['cases/train/Fixture/pose.npz'] = b'wrong'
        with self.assertRaises(ValueError): r.archive_contract(self.archive, altered)
    def test_archive_missing_member(self):
        missing = dict(self.payloads)
        missing['cases/train/Fixture/missing.json'] = b'x'
        with self.assertRaises(ValueError): r.archive_contract(self.archive, missing)
    def test_archive_duplicate_member(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as pack:
            for _ in range(2):
                info = tarfile.TarInfo('cases/train/Fixture/pose.npz'); info.size = 1
                pack.addfile(info, io.BytesIO(b'x'))
        with self.assertRaises(ValueError): r.archive_contract(buffer.getvalue(), self.payloads)
    def test_archive_truncated_footer(self):
        with self.assertRaises(ValueError): r.archive_contract(self.archive[:-1], self.payloads)
    def test_original_partial_tar_verified_without_rewrite(self):
        raw = self.archive[:2000]
        result = r.partial_archive_report(raw, self.payloads)
        self.assertFalse(result['complete_archive'])
        self.assertTrue(result['original_tar_untouched'])
        self.assertGreaterEqual(result['complete_members_verified'], 1)
    def test_original_partial_tar_different_member_rejected(self):
        changed = dict(self.payloads)
        changed['cases/train/Fixture/pose.npz'] = b'changed'
        with self.assertRaises(ValueError): r.partial_archive_report(self.archive, changed)
    def test_existing_shared_exact_bytes(self):
        for relative, raw in self.payloads.items():
            path = self.target / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        (self.target / 'batch.tar').write_bytes(self.archive)
        self.assertEqual(len(r.verify_shared(self.target, self.payloads, self.archive)), 6)
    def test_existing_shared_different_value(self):
        path = self.target / 'cases/train/Fixture/pose.npz'; path.parent.mkdir(parents=True); path.write_bytes(b'wrong')
        with self.assertRaises(ValueError): r.verify_shared(self.target, self.payloads, self.archive)
    def test_existing_shared_archive_conflict(self):
        self.target.mkdir(); (self.target / 'batch.tar').write_bytes(b'conflict')
        with self.assertRaises(ValueError): r.verify_shared(self.target, self.payloads, self.archive)
    def test_existing_shared_unknown_file(self):
        self.target.mkdir(); (self.target / 'unknown').write_bytes(b'x')
        with self.assertRaises(ValueError): r.verify_shared(self.target, self.payloads, self.archive)
    def test_claim_original_owner(self):
        r.verify_claim(self.claim, self.receipt, self.progress, 0, 'a'*64)
    def test_claim_owner_pid_mismatch(self):
        self.claim['owner_pid'] += 1
        with self.assertRaises(PermissionError): r.verify_claim(self.claim, self.receipt, self.progress, 0, 'a'*64)
    def test_claim_run_mismatch(self):
        self.claim['run_id'] = 'foreign'
        with self.assertRaises(PermissionError): r.verify_claim(self.claim, self.receipt, self.progress, 0, 'a'*64)
    def test_claim_original_row_mismatch(self):
        self.claim['source_row'] = {**self.row, 'id': 'Different'}
        with self.assertRaises(PermissionError): r.verify_claim(self.claim, self.receipt, self.progress, 0, 'a'*64)
    def test_claim_original_index_mismatch(self):
        self.claim['assignment_index'] += 4
        with self.assertRaises(PermissionError): r.verify_claim(self.claim, self.receipt, self.progress, 0, 'a'*64)
    def test_original_progress_bytes_change(self):
        path = self.root / 'progress.json'; path.write_bytes(r.encoded({'status': 'running'}))
        evidence = {}; r.remember(path, evidence)
        path.write_bytes(r.encoded({'status': 'changed'}))
        with self.assertRaises(PermissionError): r.recheck(evidence)
    def test_original_missing_file(self):
        (self.local / 'cases/train/Fixture/pose.npz').unlink()
        with self.assertRaises(FileNotFoundError): r.build_archive(self.local, self.payloads, self.files)
    def test_symlink_source_rejected(self):
        source = self.root / 'source'; source.write_bytes(b'x'); link = self.root / 'link'; link.symlink_to(source)
        with self.assertRaises(ValueError): r.read_evidence(link)
    def test_duplicate_source_in_other_commit(self):
        common = self.root / 'publication-fixture'
        commit = common / 'results' / r.WORKER / 'batches/gpu0/other/commit.json'
        commit.parent.mkdir(parents=True); commit.write_bytes(r.encoded({'edition': common.name, 'worker': r.WORKER,
            'receipts': [{'source_row': self.row, 'status': 'extracted_rule_draft_unapproved'}]}))
        with patch.object(r, 'P', common):
            with self.assertRaises(ValueError): r.commit_index(set(), {}, {('train', 'fixture')})
    def test_candidate_write_once_different_value(self):
        with patch.object(r, 'CANDIDATE', self.root):
            path = self.root / 'candidate'; r.write_once(path, b'first')
            with self.assertRaises(ValueError): r.write_once(path, b'second')
            self.assertEqual(path.read_bytes(), b'first')
    def test_historical_receipt_absence_fails_before_publication(self):
        with self.assertRaises(PermissionError): r.publish({'original_receipt_output_hash_manifest_present': False}, {}, b'',
            (None, None, None, None, None, None, rules, None, None, None))
    def test_full_original_audit_ast_unchanged(self):
        r.verify_frozen_contract((r.L / 'audit_rollover_output_packages_once_v2.py').read_bytes())
    def test_strict_json_duplicate_partial_nonfinite(self):
        for raw in (b'{"a":1,"a":2}\n', b'{"a":1}', b'{"a":NaN}\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): r.strict(raw)

class DeepCPUFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Existing immutable output bytes are only read as a fixture template; no media/model access.
        local = r.L / 'unified_runs' / r.P.name / r.RUNS[0] / (r.RUNS[0] + '-batch176')
        case = sorted((local / 'cases').glob('*/*'))[0]
        cls.relative = str(case.relative_to(local))
        cls.payloads = {cls.relative + '/' + name: r.read_evidence(case / name)[0] for name in (
            'pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json', 'receipt.json')}
        cls.receipt = r.strict(cls.payloads[cls.relative + '/receipt.json'])
        cls.row = cls.receipt['source_row']
        cls.target = r.P / 'results' / r.WORKER / 'batches/gpu0' / (r.RUNS[0] + '-batch176')
    def test_real_complete_case_passes_original_deep_contract(self):
        result = validate(self.payloads, self.target, self.receipt, self.row, rules, r.strict)
        self.assertTrue(result['actual_artifacts_deep_verified'])
    def mutate(self, name, change):
        payloads = dict(self.payloads); value = r.strict(payloads[self.relative + '/' + name]); change(value)
        payloads[self.relative + '/' + name] = r.encoded(value)
        return payloads
    def test_npz_manifest_sha_mismatch(self):
        payloads = self.mutate('fullframe_manifest.json', lambda x: x.update(npz_sha256='0'*64))
        with self.assertRaises(ValueError): validate(payloads, self.target, self.receipt, self.row, rules, r.strict)
    def test_approval_flags_mismatch(self):
        payloads = self.mutate('draft.json', lambda x: x.update(training_eligible=True))
        with self.assertRaises(ValueError): validate(payloads, self.target, self.receipt, self.row, rules, r.strict)
    def test_rule_support_invalid_frame(self):
        payloads = self.mutate('draft.json', lambda x: x['events'][0]['support'].update(frame_indices=[999999]))
        with self.assertRaises(ValueError): validate(payloads, self.target, self.receipt, self.row, rules, r.strict)
    def test_beat_config_mismatch(self):
        payloads = self.mutate('beats.json', lambda x: x.update(dbn=True))
        with self.assertRaises(ValueError): validate(payloads, self.target, self.receipt, self.row, rules, r.strict)
    def test_original_receipt_source_sha_mismatch(self):
        receipt = copy.deepcopy(self.receipt); receipt['copy_evidence']['input_sha256']['video_path'] = '0'*64
        with self.assertRaises(ValueError): r.verify_original_receipt(self.payloads, self.relative, receipt, self.row)
    def test_historical_output_manifest_sha_mismatch(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['output_files'] = {name: {'sha256': r.sha(self.payloads[self.relative + '/' + name]),
            'bytes': len(self.payloads[self.relative + '/' + name])} for name in ('pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json')}
        receipt['output_files']['draft.json']['sha256'] = '0'*64
        with self.assertRaises(ValueError): r.verify_original_receipt(self.payloads, self.relative, receipt, self.row)

if __name__ == '__main__': unittest.main(verbosity=2)
