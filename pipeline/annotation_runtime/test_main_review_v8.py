"""Three CPU groups: physical/index, origin/history, resource/code contracts. No model/signal."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import main_review_watch_v8 as w
import main_review_resource_manager_v8 as manager
import build_main_review_v8_fresh as builder
c = w.module('physical_fixture_original_cache', RT / 'curated_terminal_reconciliation_v1.py',
    '1ef0825d120775c25f628ed1c1374ac008de41b19f1397dfdf6f8b834038d2a8')


class Members:
    descriptors = {'train-b000000': {'split': 'train'}}
    def batch(self, bid):
        return {'first': {'id': 'first'}, 'second': {'id': 'second'}}


def encoded(sid):
    return (json.dumps({'id': sid, 'split': 'train', 'batch_id': 'train-b000000', 'status': 'failed'}) + '\n').encode()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stream = self.root / 'completed.jsonl'

    def tearDown(self):
        self.tmp.cleanup()

    def physical_proof(self):
        self.stream.write_bytes(b'x' * w.OFFSET + w.RAW)
        raw = self.root / 'completed.prefix.raw'
        raw.write_bytes(self.stream.read_bytes())
        current = [self.stream.stat().st_dev, self.stream.stat().st_ino]
        historic = [current[0] + 1, current[1]]
        prefix = {'path': str(self.stream), 'input_identity': historic,
            'byte_length': raw.stat().st_size, 'prefix_sha256': w.sha(raw)}
        bridge = {'path': str(self.stream), 'original_prefix': prefix,
            'raw_prefix_ref': {'path': str(raw), 'sha256': w.sha(raw)},
            'historical_capture_identity': historic, 'fresh_current_node_identity': current,
            'current_node_hostname': os.uname().nodename,
            'cross_mount_device_rebound_content_verified': True, 'prefix_content_SHA_verified': True}
        # Physical-only fixture; does not construct/admit a full Preparation or claim source/quality proof.
        proof = object.__new__(w.Preparation)
        proof.production = self.root
        proof.snapshot = {'original_journal_prefix': prefix, 'terminal_index_prefixes': {},
            'committed_prefix_sha256': hashlib.sha256(b'x' * w.OFFSET).hexdigest()}
        proof.binding = {'mount_bindings': {str(self.stream): bridge}}
        proof.cache = c.ByteCache()
        return proof


class PhysicalIndex(Fixture):
    def test_exact_physical_five_bytes_three_snapshots_without_sources(self):
        proof = self.physical_proof()
        with patch.object(w, 'prefix_specs', return_value={str(self.stream): proof.snapshot['original_journal_prefix']}):
            proof.check_physical(w.OFFSET, w.RAW)
        self.assertEqual(self.stream.read_bytes()[w.OFFSET:], w.RAW)
        self.assertEqual(hashlib.sha256(w.RAW).hexdigest(), w.RAW_SHA)
        self.assertNotEqual(w.RAW, b'\n')

    def test_wrong_offset_or_exact_bytes_fail_closed(self):
        proof = self.physical_proof()
        for offset, raw in ((w.OFFSET - 1, w.RAW), (w.OFFSET, b'te"x\n'), (w.OFFSET, b'\n')):
            with self.assertRaises(ValueError):
                proof.check_physical(offset, raw)

    def test_wrong_prefix_and_mount_inode_fail_closed(self):
        proof = self.physical_proof()
        prefixes = {str(self.stream): proof.snapshot['original_journal_prefix']}
        with patch.object(w, 'prefix_specs', return_value=prefixes):
            proof.snapshot['committed_prefix_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                proof.check_physical(w.OFFSET, w.RAW)
            proof.binding['mount_bindings'][str(self.stream)]['fresh_current_node_identity'][1] += 1
            with self.assertRaises(ValueError):
                proof.check_physical(w.OFFSET, w.RAW)

    def test_unknown_new_badline_blocks_and_does_not_skip(self):
        first = encoded('first')
        for bad in (b'\n', b'{bad}\n', w.RAW):
            self.stream.write_bytes(first + bad + encoded('second'))
            index = w.RawIndex(self.root, Members())
            index.scan()
            self.assertEqual(index.indexed_offset, len(first))
            self.assertEqual(index.pending['reason'], 'unknown_malformed_line')
            self.assertIs(index.pending['skipped'], False)
            self.assertNotIn('second', index.raw['train-b000000'])

    def test_exact_offset_requires_explicit_frozen_proof(self):
        self.stream.write_bytes(b'x' * w.OFFSET + w.RAW)
        index = w.RawIndex(self.root, Members())
        index.indexed_offset = w.OFFSET
        index.indexed_sha = hashlib.sha256(b'x' * w.OFFSET).hexdigest()
        index.scan()
        self.assertEqual(index.indexed_offset, w.OFFSET)
        self.assertTrue(index.pending)
        self.assertFalse(index.raw)

    def test_five_byte_index_is_zero_encoded_rows_not_emptyline(self):
        proof = self.physical_proof()
        incident = object.__new__(w.Incident)
        incident.preparation = proof
        incident.fragment_row_offsets = []
        index = w.RawIndex(self.root, Members(), incident=incident)
        index.indexed_offset = w.OFFSET
        index.indexed_sha = hashlib.sha256(b'x' * w.OFFSET).hexdigest()
        with patch.object(w, 'prefix_specs', return_value={str(self.stream): proof.snapshot['original_journal_prefix']}):
            index.scan()
        self.assertIsNone(index.pending)
        self.assertEqual(index.indexed_offset, w.OFFSET + 5)
        self.assertEqual(index.lines, [(w.OFFSET, w.OFFSET + 5, [])])
        self.assertFalse(index.journal_keys)
        self.assertFalse(index.raw)
        self.assertFalse(index.completed)
        self.assertFalse(index.empty_row_offsets)
        self.assertEqual(incident.fragment_row_offsets, [w.OFFSET])

    def test_changed_exact_badline_does_not_commit_fragment(self):
        proof = self.physical_proof()
        incident = object.__new__(w.Incident)
        incident.preparation = proof
        incident.fragment_row_offsets = []
        self.stream.write_bytes(b'x' * w.OFFSET + b'te"x\n')
        index = w.RawIndex(self.root, Members(), incident=incident)
        index.indexed_offset = w.OFFSET
        index.indexed_sha = hashlib.sha256(b'x' * w.OFFSET).hexdigest()
        index.scan()
        self.assertEqual(index.indexed_offset, w.OFFSET)
        self.assertTrue(index.pending)
        self.assertFalse(index.journal_keys)
        self.assertFalse(index.raw)

    def test_same_snapshot_revision_and_old_prefix_mutation_fail_closed(self):
        self.stream.write_bytes(encoded('first') + encoded('first').replace(b'failed', b'other!'))
        index = w.RawIndex(self.root, Members())
        index.scan()
        self.assertEqual(index.indexed_offset, len(encoded('first')))
        self.assertTrue(index.pending)
        self.stream.write_bytes(self.stream.read_bytes().replace(b'first', b'FIRST'))
        with self.assertRaises(ValueError):
            index.scan()


class OriginHistory(Fixture):
    def test_original_authoritative_member_bytes_sha_fails(self):
        member = self.root / 'manifest.jsonl'
        member.write_bytes((json.dumps({'id': 'first', 'batch_id': 'train-b000000', 'split': 'train'}) + '\n').encode())
        membership = object.__new__(w.Membership)
        membership.descriptors = {'train-b000000': {'path': str(member), 'sha256': w.sha(member), 'count': 1, 'split': 'train'}}
        membership.rows = {}
        member.write_bytes(member.read_bytes().replace(b'first', b'other'))
        with self.assertRaises(ValueError):
            membership.batch('train-b000000')

    def test_reconciliation_origin_null_offsets_not_journal_count(self):
        source = inspect.getsource(w.RawIndex.reconcile_offset)
        self.assertIn("'byte_begin': None, 'byte_end': None, 'raw_sha256': None", source)
        self.assertIn("'origin': 'explicit_terminal_reconciliation'", source)
        self.assertIn("'fragment_source_association': 'unknown'", source)
        self.assertNotIn('journal_keys.add', source)
        self.assertNotIn('journal_keys.update', source)
        self.assertNotIn('self.lines.append', source)
        self.assertIn('terminal_record(self.production, hint, self.membership)', source)
        self.assertIn("sum(r['data_complete_candidate']", (RT / 'main_review_watch_v8.py').read_text())

    def test_real_four_record_origin_commit_keeps_original_denominator(self):
        production = w.ROOT / 'curated222939_production_v1'
        root = w.ROOT / 'curated222939_repair_v1/collector-offset2053650-reconciled-v1'
        snapshot = json.loads((root / 'snapshot.json').read_text())
        membership = w.Membership(w.ROOT / 'curated222939_inputs_v1/manifest.summary.json')
        # Commit-layer test with real immutable record/terminal bytes; full Preparation deep gate is a separate real CPU proof.
        proof = object.__new__(w.Preparation)
        proof.values, proof.records, proof.cache = {}, [], c.ByteCache()
        for key in snapshot['missing_source_keys']:
            record, ref = w.recovery.document(root / 'records' / (key + '.json'))
            self.assertIsNone(record['original_byte_offset'])
            self.assertIs(record['quality_approved'], False)
            proof.values[key] = record
            proof.records.append((key, ref, record))
        incident = object.__new__(w.Incident)
        incident.preparation, incident.fragment_row_offsets = proof, [w.OFFSET]
        index = w.RawIndex(production, membership, incident=incident)
        stream = production / 'completed.jsonl'
        index.identity = [stream.stat().st_dev, stream.stat().st_ino]
        index.indexed_offset = w.OFFSET + 5
        index.indexed_sha = w.recovery.prefix_sha(stream, index.indexed_offset, index.identity)
        self.assertEqual(index.reconcile_offset(), {'train-b000005', 'train-b000006'})
        self.assertEqual(membership.summary['total'], 222939)
        self.assertEqual(len(membership.batch('train-b000005')), 500)
        self.assertEqual(len(membership.batch('train-b000006')), 500)
        self.assertEqual(len(index.restored_sources), 4)
        self.assertEqual(sum(len(r) for r in index.raw.values()), 4)
        self.assertFalse(index.journal_keys)
        self.assertFalse(index.lines)
        for rows in index.raw.values():
            for value in rows.values():
                self.assertEqual(value['origin'], 'explicit_terminal_reconciliation')
                self.assertIsNone(value['byte_begin'])
                self.assertIsNone(value['byte_end'])
                self.assertIsNone(value['raw_sha256'])
        self.assertEqual(sum(r['label_status'] == 'completed' for rows in index.completed.values() for r in rows.values()), 2)

    def test_full_source_denominators_and_sampling_policy_are_unchanged(self):
        source = builder.source
        self.assertIn("'expected_batch_count'] = 448", source)
        self.assertIn("'selected_source_count'] == 10000", source)
        self.assertIn("'source_range'] == {'start': 21000, 'end': 31000}", source)
        original_main = inspect.getsource(w.v7.main)
        self.assertIn('sample_count=25, risk_count=5', original_main)
        self.assertIn('222939', original_main)
        self.assertIn('10000', original_main)
        self.assertIs(w.retained_selection, w.v7.retained_selection)
        self.assertIs(w.install_history_selection, w.v7.install_history_selection)
        self.assertIs(w.reviewer_from_verified_local_model, w.v7.reviewer_from_verified_local_model)
        self.assertIn("proof['native_cache_reused_source_count'] == 80", inspect.getsource(builder.v7.cpu_smoke))
        self.assertIn('== 7', inspect.getsource(builder.v7.cpu_smoke))

    def test_real_existing_469_scores_and_old_membership_read_only(self):
        root = w.ROOT / 'curated_independent_reviews_v1'
        paths = sorted(root.glob('train-b*/summary.json'))
        before = {str(p): w.sha(p) for p in paths}
        values = [json.loads(p.read_text()) for p in paths]
        reviewed = sum(v['sampled_reviewed_denominator'] for v in values)
        self.assertGreaterEqual(reviewed, 469)
        first = json.loads((root / 'train-b000000/summary.json').read_text())
        self.assertGreaterEqual(first['sampled_reviewed_denominator'], 80)
        self.assertGreaterEqual(first['confirmed_error_count'], 7)
        membership = root / 'train-b000000/batch.membership.json'
        before[str(membership)] = w.sha(membership)
        policy = json.loads(membership.read_text())['policy']
        self.assertTrue(policy)
        self.assertEqual(before, {path: w.sha(path) for path in before})

    def test_fragment_not_reclassified_as_blank_or_terminal_identity(self):
        source = inspect.getsource(w.Incident.expand)
        self.assertIn('return []', source)
        self.assertNotIn('sid', source)
        self.assertNotIn('return self.preparation.records', source)
        self.assertIn("'bytes_contained_complete_JSON_sources': 0", inspect.getsource(w.capture_offset_binding))


class ResourceContracts(unittest.TestCase):
    def test_frozen_original_files_unchanged(self):
        for name, digest in {'main_review_watch_v7.py': w.V7_SHA,
                'main_review_resource_manager_v7.py': manager.OLD_MANAGER_SHA,
                'build_main_review_v7_fresh.py': builder.BUILDER_V7_SHA,
                'resource_manager.py': manager.BASE_MANAGER_SHA}.items():
            self.assertEqual(hashlib.sha256((RT / name).read_bytes()).hexdigest(), digest)

    def test_builder_captures_resources_after_slow_cpu_model_and_history(self):
        source = builder.source
        self.assertLess(source.index('proof = cpu_smoke'), source.index('uuid = base.uuid_for(7)'))
        self.assertLess(source.index('wrapper.check_model(cfg)'), source.index('uuid = base.uuid_for(7)'))
        self.assertIn('base.check_lease_unchanged(reserve, lease)', source)
        self.assertIn("base.check_holder(expected, base.identity(expected['pid'])", source)
        self.assertNotIn('time.time() -', source)
        self.assertNotIn('os.utime', source)

    def test_no_gpu_signal_or_actual_start_in_new_builder(self):
        tree = ast.parse(builder.source)
        called = {getattr(n.func, 'attr', '') for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertFalse({'kill', 'pidfd_send_signal', 'Popen', 'release_holder'} & called)
        self.assertIn("'GPU_model_loaded': False", builder.source)
        self.assertIn("'holder_release_performed': False", builder.source)

    def test_original_pidfd_transaction_retained_and_immediate_current_gate(self):
        source = inspect.getsource(manager.release_holder)
        for token in ('leasefd', 'lease_inode', 'FLOCK', 'fdinfo', 'check_holder', 'headroom', '_release(entry)'):
            self.assertIn(token, source)
        self.assertIs(manager.base.main, manager.v7.base.main)
        self.assertEqual(manager.base.AUTHORIZED_PAIRS, {'N5': ((7,),)})
        self.assertIn('signal.pidfd_send_signal', inspect.getsource(manager._release))
        self.assertNotIn('os.kill', source)

    def test_wrong_node_release_authorization_rejected_before_gpu_query(self):
        for cfg in ({'node': 'N2', 'gpus': [7], 'release_authorized': True, 'source_prepared': True},
                {'node': 'N5', 'gpus': [7], 'release_authorized': False, 'source_prepared': True}):
            with self.assertRaises(RuntimeError):
                manager._validate(cfg)

    def test_new_dependency_and_proof_sha_gate_retained(self):
        source = inspect.getsource(manager.validate)
        self.assertIn("cfg['source_sha256'].get(r['path']) == r['sha256']", source)
        self.assertIn('Preparation(', source)
        self.assertIn('source_byte_offsets', inspect.getsource(w.Preparation.__init__))
        self.assertNotIn('recover_line(', inspect.getsource(w.Preparation.__init__))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=('physical', 'history', 'resource'), required=True)
    a = ap.parse_args()
    group = {'physical': PhysicalIndex, 'history': OriginHistory, 'resource': ResourceContracts}[a.group]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(group))
    sys.exit(not result.wasSuccessful())
