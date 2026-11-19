"""Tests use real immutable terminals and byte-for-byte corrupt production line."""
import copy
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRODUCTION = ROOT/'curated222939_production_v1'
SUMMARY = ROOT/'curated222939_inputs_v1/manifest.summary.json'
SPEC = importlib.util.spec_from_file_location('shadow_recovery_tested', HERE/'completed_shadow_recovery_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
KEYS = ['train/MVuipekTKQDQprcC', 'train/ytykxzHnSDAecSEP']

def writer(path, number):
    for i in range(80):
        m.safe_append(path, {'worker': number, 'i': i, 'padding': 'z'*9000})

class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.identity = m.source_snapshot(PRODUCTION/'completed.jsonl')
        cls.before = cls.raw[:532233]
        cls.bad = cls.raw[532233:533173]
        assert len(cls.bad) == 940
        assert m.sha_bytes(cls.bad) == 'd74abe92289ada7d68da9f97c5444e177ae91b94ee76dd8be955a6556137e280'
        cls.rows = [m.load(PRODUCTION/'sources'/key/'consumer.terminal.json') for key in KEYS]
        cls.norm = b''.join(m.enc(r) for r in cls.rows)
        cls.tail = m.enc(cls.rows[1])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='shadow-test-', dir=HERE)
        self.shadow = Path(self.tmp.name)/'shadow'
        self.data = self.before+self.bad+self.tail
        self.source_patch = patch.object(m, 'source_snapshot', side_effect=lambda path: (self.data, self.identity))
        self.source_patch.start()
        m.make_plan(PRODUCTION, self.shadow, SUMMARY, 532233, KEYS)

    def tearDown(self):
        self.source_patch.stop()
        self.tmp.cleanup()

    def test_real_splice_immutable_terminal_derivation(self):
        m.sync(PRODUCTION, self.shadow)
        actual = (self.shadow/'completed.shadow.jsonl').read_bytes()
        self.assertEqual(actual, self.before+self.norm+self.tail)
        plan = m.load(self.shadow/'recovery.plan.json')
        self.assertEqual(plan['terminal_prefix_length'], 247)
        self.assertEqual(plan['batch_member_count'], 500)
        self.assertEqual(m.checked(plan['bad_raw_ref']), self.bad)
        old_sha = m.sha_bytes(actual)
        m.sync(PRODUCTION, self.shadow)
        self.assertEqual(m.sha_bytes((self.shadow/'completed.shadow.jsonl').read_bytes()), old_sha)

    def test_halfline_incremental_append_not_consumed(self):
        self.data += self.tail[:50]
        m.sync(PRODUCTION, self.shadow)
        first = m.load(self.shadow/'shadow.state.json')
        self.assertEqual(first['source_offset'], len(self.data)-50)
        self.data += self.tail[50:80]
        m.sync(PRODUCTION, self.shadow)
        second = m.load(self.shadow/'shadow.state.json')
        self.assertEqual(first['source_offset'], second['source_offset'])
        self.data += self.tail[80:]
        m.sync(PRODUCTION, self.shadow)
        self.assertEqual((self.shadow/'completed.shadow.jsonl').read_bytes(), self.before+self.norm+self.tail*2)

    def test_crash_partial_write_and_postwrite_prestate_resume(self):
        def crash(shadow, state, txn):
            payload = m.checked(txn['payload_ref'])
            (shadow/'completed.shadow.jsonl').write_bytes(payload[:12345])
            raise RuntimeError('simulated crash after partial write')
        with patch.object(m, 'apply_transaction', side_effect=crash):
            with self.assertRaises(RuntimeError):
                m.sync(PRODUCTION, self.shadow)
        m.sync(PRODUCTION, self.shadow)
        expected = self.before+self.norm+self.tail
        self.assertEqual((self.shadow/'completed.shadow.jsonl').read_bytes(), expected)
        self.data += self.tail
        original_replace = m.replace
        def no_checkpoint(path, value):
            if path.name == 'shadow.state.json':
                raise RuntimeError('simulated crash after fsync before state')
            return original_replace(path, value)
        with patch.object(m, 'replace', side_effect=no_checkpoint):
            with self.assertRaises(RuntimeError):
                m.sync(PRODUCTION, self.shadow)
        m.sync(PRODUCTION, self.shadow)
        self.assertEqual((self.shadow/'completed.shadow.jsonl').read_bytes(), expected+self.tail)

    def test_plan_original_prefix_change_refused_before_first_commit(self):
        self.data = b'X'+self.data[1:]
        with self.assertRaises(m.Changed):
            m.sync(PRODUCTION, self.shadow)
        self.assertFalse((self.shadow/'completed.shadow.jsonl').exists())

    def test_source_content_identity_and_truncation_refused(self):
        m.sync(PRODUCTION, self.shadow)
        original = self.data
        for changed in (b'X'+original[1:], original[:-1]):
            self.data = changed
            with self.assertRaises(m.Changed):
                m.sync(PRODUCTION, self.shadow)
        self.data = original
        with patch.object(m, 'host', return_value={'hostname': 'other', 'boot_id': 'other'}):
            with self.assertRaises(m.Changed):
                m.sync(PRODUCTION, self.shadow)
        with patch.object(m, 'source_snapshot', return_value=(original, [0, 0])):
            with self.assertRaises(m.Changed):
                m.sync(PRODUCTION, self.shadow)

    def test_unknown_badline_and_missing_terminal_never_skipped(self):
        self.data += b'unknown broken JSON\n'+self.tail
        with self.assertRaises(m.Changed):
            m.sync(PRODUCTION, self.shadow)
        state = m.load(self.shadow/'shadow.state.json')
        self.assertEqual(state['source_offset'], len(self.before+self.bad+self.tail))
        for diagnostic in (self.shadow/'diagnostics').glob('*.json'):
            d = m.load(diagnostic)
            self.assertTrue(d['pending'])
            self.assertFalse(d['skipped'])
        original = m.terminal_evidence
        def unavailable(production, row):
            if m.safe_key(row) == KEYS[0]:
                raise FileNotFoundError('terminal unavailable')
            return original(production, row)
        second = Path(self.tmp.name)/'missing'
        m.make_plan(PRODUCTION, second, SUMMARY, 532233, KEYS)
        with patch.object(m, 'terminal_evidence', side_effect=unavailable):
            with self.assertRaises(m.Changed):
                m.sync(PRODUCTION, second)
        self.assertLessEqual(m.load(second/'shadow.state.json')['source_offset'], 532233)

    def test_migration_exact_prefix_seen_and_state_sha(self):
        m.sync(PRODUCTION, self.shadow)
        out = Path(self.tmp.name)/'repair'
        out.mkdir()
        seen = {m.safe_key(r): m.sha_bytes(m.enc(r)) for r in map(json.loads, self.before.splitlines())}
        state = {'input_path': str(PRODUCTION/'completed.jsonl'), 'input_identity': self.identity,
                 'offset': 532233, 'seen': seen, 'counts': {'keep': 777}, 'stage_counts': {'keep': 11}}
        m.once(out/'collector.state.json', state)
        expected = m.sha_bytes((out/'collector.state.json').read_bytes())
        with self.assertRaises(m.Changed):
            m.migrate(PRODUCTION, self.shadow, out, '0'*64)
        receipt = m.migrate(PRODUCTION, self.shadow, out, expected)
        new = m.load(out/'collector.state.json')
        self.assertEqual(new['seen'], state['seen'])
        self.assertEqual(new['counts'], state['counts'])
        self.assertEqual(new['stage_counts'], state['stage_counts'])
        self.assertEqual(receipt['identical_prefix_sha256'], m.sha_bytes(self.before))
        self.assertEqual(receipt['sources_seen'], len(seen))
        self.assertEqual(m.checked(receipt['original_state_ref']), m.enc(state))

    def test_shared_filesystem_two_process_writer_and_stale_lock(self):
        path = Path(self.tmp.name)/'parallel.jsonl'
        processes = [multiprocessing.Process(target=writer, args=(path, n)) for n in range(2)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(90)
            self.assertEqual(p.exitcode, 0)
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        self.assertEqual(len(rows), 160)
        self.assertEqual({(r['worker'], r['i']) for r in rows}, {(n, i) for n in range(2) for i in range(80)})
        lock = path.with_name(path.name+'.writer-lock')
        lock.mkdir()
        before = path.read_bytes()
        with self.assertRaises(m.Changed):
            m.safe_append(path, {'must_not_write': True}, timeout=0)
        self.assertEqual(path.read_bytes(), before)
        lock.rmdir()

    def test_frozen_collector_minimal_parameterization(self):
        module = m.collector_module(HERE/'source_repair_gpu_v2/curated_repair_queue_v1.py')
        self.assertIn('completed_stream', module.collect.__code__.co_varnames)
        out = Path(self.tmp.name)/'fresh-collector'
        queue = Path(self.tmp.name)/'only-complete.jsonl'
        queue.write_bytes(self.norm)
        result = module.collect(PRODUCTION, out, 'repair-20261025-v1', queue)
        self.assertEqual(result['sources_seen'], 2)
        self.assertEqual(result['new_queued_sources'], 0)
        result = module.collect(PRODUCTION, out, 'repair-20261025-v1', queue)
        self.assertEqual(result['new_queued_sources'], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
