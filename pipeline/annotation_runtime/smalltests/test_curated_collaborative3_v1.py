"""CPU-only enabled-feature tests; no model load, signals, or GPU lease admission."""
import argparse
import copy
import fcntl
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curated_collaborative3_plan_v1 as p
import curated_collaborative3_build_v1 as b
import curated_collaborative3_consumer_v1 as w
import curated_collaborative3_resource_manager_v1 as m


class Collab3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.capture = self.root / 'capture.json'
        holders = [{'gpu': g, 'pid': 900000 + g, 'uuid': 'test-gpu-%d' % g} for g in (2, 3)]
        p.write(self.capture, {'node': 'N2', 'mode': 'consumer', 'gpus': [2, 3],
                'holders': holders, 'leases': [None, None], 'signals_sent': False,
                'captured_unix_ns': time.time_ns()})
        self.cfg = b.construct('n2c0', p.ref(self.capture))
        self.safety = [patch.object(m.base, 'release_holder', side_effect=AssertionError('signal in CPU test')),
                       patch.object(m.base, 'acquire_lease', side_effect=AssertionError('GPU lease in CPU test')),
                       patch.object(m.c, 'check_model', side_effect=AssertionError('model check/load not requested'))]
        for guard in self.safety:
            guard.start()

    def tearDown(self):
        for guard in reversed(self.safety):
            guard.stop()
        self.tmp.cleanup()

    def validate(self, cfg):
        with patch.object(m.c.subprocess, 'check_output', return_value=p.HOSTS['N2']), \
             patch.object(m, 'verify_capture'), patch.object(m.old, 'readonly_lock_inventory', return_value=[]), \
             patch.object(m.worker, 'configuration', return_value=p.load(b.EVIDENCE)['new_configuration']):
            m.validate(cfg)

    def test_enabled_exact_config(self):
        self.validate(self.cfg)

    def test_all_members_unique_range_partition(self):
        plan = p.verify_plan(p.load(p.PLAN))
        all_rows = p.rows_from_manifest(p.load(p.MANIFEST))[0]
        for mode in ('producer', 'consumer'):
            rows = [r for label, member in p.MEMBERS.items() if member[1] == mode
                    for r in p.checked(plan['selected_sources'][label])]
            self.assertEqual(sorted(r['source_ordinal'] for r in rows), list(range(32000, 62000)))
            self.assertEqual(len({r['id'] for r in rows}), 30000)
        self.assertEqual(len(all_rows), 30000)
        occupied = {(n, gpu) for n, _, gpus, _, _ in p.MEMBERS.values() for gpu in gpus}
        self.assertEqual(len(occupied), 22)
        self.assertNotIn(('N7', 6), occupied)
        self.assertNotIn(('N7', 7), occupied)

    def test_range_mutation_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['source_range']['start'] = 31000
        with self.assertRaises(RuntimeError):
            self.validate(cfg)

    def test_member_mutation_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['collab3_member'] = 'n2c1'
        with self.assertRaises(RuntimeError):
            self.validate(cfg)

    def test_different_physical_gpu_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['gpus'] = [0, 1]
        with self.assertRaises(RuntimeError):
            self.validate(cfg)

    def test_wrong_node_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['node'] = 'N3'
        with self.assertRaises((RuntimeError, KeyError)):
            self.validate(cfg)

    def test_existing_managed_queue_refused(self):
        directory = self.root / 'queue'
        directory.mkdir()
        cfg = copy.deepcopy(self.cfg)
        cfg['queue'] = str(directory)
        with patch.object(b, 'construct', return_value=cfg), patch.object(p, 'verify_plan', return_value=p.load(p.PLAN)):
            with self.assertRaises(RuntimeError):
                self.validate(cfg)

    def test_repeat_dispatch_refused(self):
        launch = self.root / 'launch.json'
        launch.write_text('{}')
        with patch.object(p, 'paths', return_value={'launch': launch, 'queue': self.root / 'newqueue', 'out': p.paths('n2c0')['out']}):
            with self.assertRaisesRegex(RuntimeError, 'Repeated member dispatch'):
                b.build('n2c0', self.capture)

    def test_wrong_source_sha_refused(self):
        path = self.root / 'source.py'
        path.write_text('frozen')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            p.checked({'path': str(path), 'sha256': '0' * 64})
        cfgpath = self.root / 'cfg.json'
        p.write(cfgpath, self.cfg)
        with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
            m.base.check_sources(self.cfg, str(cfgpath), '0' * 64, p.sha(p.RT / 'resource_manager.py'))

    def test_selected_source_membership_refused(self):
        reference = self.root / 'wrong_selected.json'
        rows = p.checked(self.cfg['selected_sources'])
        rows[0]['source_ordinal'] += 1
        p.write(reference, rows)
        cfg = copy.deepcopy(self.cfg)
        cfg['selected_sources'] = p.ref(reference)
        with self.assertRaisesRegex(RuntimeError, 'reference changed'):
            m.selected_rows(cfg)

    def test_active_lease_and_nanosecond_change_refused(self):
        lease = self.root / 'lease'
        lease.touch()
        with self.assertRaisesRegex(RuntimeError, 'active lease'):
            m.base.lease_snapshot(lease)
        old_ns = time.time_ns() - 1300 * 10**9
        os.utime(lease, ns=(old_ns, old_ns))
        old = m.base.lease_snapshot(lease)
        os.utime(lease, ns=(old_ns, old_ns + 1))
        with self.assertRaisesRegex(RuntimeError, 'lease changed'):
            m.base.check_lease_unchanged(lease, old)

    def test_fresh_capture_stale_refused(self):
        value = p.load(self.capture)
        value['captured_unix_ns'] -= 1201 * 10**9
        stale = self.root / 'stale.json'
        p.write(stale, value)
        with self.assertRaisesRegex(RuntimeError, '20 minutes'):
            b.construct('n2c0', p.ref(stale))

    def test_real_kernel_flock_conflict(self):
        lock = self.root / 'lock'
        fd1 = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        fd2 = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd2)
            os.close(fd1)

    def test_160gib_negative_headroom(self):
        path = self.root / 'cfg.json'
        p.write(path, self.cfg)
        args = SimpleNamespace(config=str(path), config_sha=p.sha(path),
                manager_sha=p.sha(p.RT / 'resource_manager.py'), report=str(self.root / 'report.json'))
        with patch.object(p, 'paths', return_value={'launch': path}), patch.object(m, 'validate'), \
             patch.object(m.base, 'check_sources'), patch.object(m.c, 'check_model'), \
             patch.object(m.old, 'readonly_lock_inventory', return_value=[]), \
             patch.object(os, 'pidfd_open', side_effect=lambda pid: os.open('/dev/null', os.O_RDONLY)), \
             patch.object(m.select, 'select', return_value=([], [], [])), \
             patch.object(m.base, 'headroom', return_value={'headroom': 160 * 2**30 - 1}):
            with self.assertRaisesRegex(RuntimeError, '160GiB'):
                m.readonly_admission(args)
        self.assertFalse(Path(args.report).exists())

    def test_optimized_consumer_context_is_enabled_and_restored(self):
        pipeline = p.pipeline()
        args = pipeline.parser().parse_args(["consumer", "--manifest", "/fixture", "--out", str(self.root),
               "--model", "/fixture", "--ms-source", str(p.RT / "source_curated_consumer_v1")])
        bar, _, _ = pipeline.modules(args.repo, args.ms_source)
        from test_curated_transaction_cache_v1 import FakeModel, request
        model = FakeModel()
        native_process, native_transactions = bar.process_source, bar.transactions
        def process(row, raw, root, *remaining):
            item = {"root": root / "window" / "attempt", "request": request(),
                    "binding": {"source": row["id"]}, "validator": lambda value: value}
            return bar.transactions([item], lambda: (model, None), args)
        bar.process_source = process
        fixture_process = bar.process_source
        try:
            with patch.object(bar.core, "structured_sampling", return_value=None), patch("sys.stdout", new=io.StringIO()):
                with w.optimization(pipeline, args):
                    self.assertIsNot(bar.transactions, native_transactions)
                    self.assertIsNot(bar.process_source, fixture_process)
                    result = bar.process_source({"id": "fixture"}, b"{}", self.root / "visual", args, {}, None, {})
                    self.assertEqual(next(iter(result.values()))["status"], "candidate")
                    self.assertEqual(len(model.calls), 1)
                    self.assertIsNone(bar.transactions.__self__.source_boundary)
            self.assertIs(bar.transactions, native_transactions)
            self.assertIs(bar.process_source, fixture_process)
        finally:
            bar.process_source = native_process

    def test_optimization_exception_restores_native_modules(self):
        pipeline = p.pipeline()
        args = pipeline.parser().parse_args(["consumer", "--manifest", "/fixture", "--out", str(self.root),
               "--model", "/fixture", "--ms-source", str(p.RT / "source_curated_consumer_v1")])
        bar, _, _ = pipeline.modules(args.repo, args.ms_source)
        before = (bar.process_source, bar.transactions, pipeline.save)
        with self.assertRaisesRegex(RuntimeError, "fixture stopped"):
            with w.optimization(pipeline, args):
                raise RuntimeError("fixture stopped")
        self.assertEqual(before, (bar.process_source, bar.transactions, pipeline.save))

    def test_four_ready_streams_isolated_terminals_zero_models(self):
        pipeline = p.pipeline()
        w.install(pipeline)
        source = self.root / 'producer'
        output = self.root / 'consumer'
        pipeline.SOURCE_OUT = source
        pipeline.load_style_reuse = lambda paths: {}
        configuration = p.load(b.EVIDENCE)['new_configuration']
        bar = SimpleNamespace(code_identity=lambda: {}, params=lambda args, code: configuration,
              runtime_loader=lambda *a: lambda: (_ for _ in ()).throw(AssertionError('model call')), __file__=str(p.RT/'source_curated_consumer_v1/ms_bar_annotation.py'))
        pipeline.modules = lambda *a: (bar, None, SimpleNamespace(digest=b.digest))
        ids = []
        for index, ordinal in enumerate((32004, 32013, 32022, 32031)):
            sid = 'cpu_source_%d' % index
            ids.append(sid)
            row = {'id': sid, 'split': 'train', 'batch_id': 'train-b000064', 'source_ordinal': ordinal}
            terminal = {**row, 'status': 'failed', 'manifest_record': row}
            pth = source / 'sources/train' / sid / 'producer.terminal.json'
            pipeline.save(pth, terminal)
            pipeline.append(source / 'ready' / ('producer-%04d.jsonl' % (ordinal % 4)), terminal)
        manifest = self.root / 'summary.json'
        p.write(manifest, {'total': 222939, 'counts': {'train': 200568, 'val': 11110, 'test': 11261},
                'batches': [{'batch_id': 'train-b000064', 'count': 500}]})
        args = SimpleNamespace(manifest=str(manifest), tp=2, batch_size=1, max_num_seqs=1,
                shard=0, nshard=9, poll_seconds=0.001, repo=None, ms_source=None, out=str(output),
                ids=ids, limit=0, watch=False, style_reuse=[], stage='run')
        result = pipeline.run_consumer(args)
        self.assertEqual(result['queue_sources_consumed'], 4)
        completed = [json.loads(line) for line in (output/'completed.jsonl').read_text().splitlines()]
        self.assertEqual({v['id'] for v in completed}, set(ids))
        self.assertFalse((source/'completed.jsonl').exists())
        self.assertFalse(any((source/'sources/train'/sid/'consumer.terminal.json').exists() for sid in ids))
        for sid in ids:
            self.assertTrue((output/'sources/train'/sid/'consumer.terminal.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
