"""Only new recovery behavior; CPU-only fixtures never acquire resources."""
import copy
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curated_collaborative3_n2c1_recovery_v1 as r


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.guards = [patch.object(r.base, 'release_holder', side_effect=AssertionError('CPU signal forbidden')),
                       patch.object(r.base, 'acquire_lease', side_effect=AssertionError('CPU GPU lease forbidden')),
                       patch.object(r.manager.c, 'check_model', side_effect=AssertionError('CPU model check forbidden'))]
        for guard in self.guards:
            guard.start()

    def tearDown(self):
        for guard in reversed(self.guards):
            guard.stop()
        self.tmp.cleanup()

    def capture(self):
        original = r.p.load(r.ORIGINAL)
        holders = [r.p.load(Path(original['queue']) / ('holder_restored_gpu%d.json' % g))['holder_identity']
                   for g in (4, 5)]
        for g, owner in zip((4, 5), holders):
            owner.update(gpu=g, uuid=r.p.load(Path(original['queue']) /
                         ('holder_restored_gpu%d.json' % g))['uuid'])
        path = self.root / 'capture.json'
        r.p.write(path, {'node': 'N2', 'mode': 'consumer', 'gpus': [4, 5], 'holders': holders,
                  'leases': [None, None], 'signals_sent': False, 'captured_unix_ns': time.time_ns()})
        return path

    def test_original_params_and_out_exactly_preserved(self):
        original = r.p.load(r.ORIGINAL)
        cfg = r.construct(r.p.ref(self.capture()))
        for key in ('out', 'selected_sources', 'targetcount', 'source_range', 'source_selection',
                    'consumer_params_sha256', 'consumer_params_evidence', 'env', 'modelref'):
            self.assertEqual(cfg[key], original[key])
        self.assertEqual(cfg['argv'][:3] + cfg['argv'][4:], original['argv'][:3] + original['argv'][4:])
        self.assertNotEqual(cfg['queue'], original['queue'])
        self.assertEqual(cfg['native_consumer_params_sha256'],
                         'fd8868d106d718df50c470d79aea47fb0fc8218516178cc772de7db302b0034e')

    def test_native_digest_differs_from_legacy_evidence_digest(self):
        configuration = r.p.load(r.builder.EVIDENCE)['new_configuration']
        pipeline = r.p.pipeline()
        args = pipeline.parser().parse_args(['consumer', '--manifest', '/fixture', '--out', str(self.root),
                       '--model', '/fixture', '--ms-source', str(r.p.RT / 'source_curated_consumer_v1')])
        _, _, visual = pipeline.modules(args.repo, args.ms_source)
        self.assertEqual(r.native_digest(configuration), visual.digest(configuration))
        self.assertNotEqual(r.native_digest(configuration), r.builder.digest(configuration))

    def test_stale_capture_refused(self):
        path = self.capture()
        value = r.p.load(path)
        value['captured_unix_ns'] -= 1201 * 10**9
        stale = self.root / 'stale.json'
        r.p.write(stale, value)
        with self.assertRaisesRegex(RuntimeError, '20 minutes'):
            r.construct(r.p.ref(stale))

    def test_prior_safe_reference_mutation_refused(self):
        cfg = r.construct(r.p.ref(self.capture()))
        cfg['recovery_references'][3]['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'references changed'):
            r.prior_gate(cfg)

    def test_existing_launch_refuses_repeated_build(self):
        launch = self.root / 'launch.json'
        launch.write_text('{}')
        with patch.object(r, 'LAUNCH', launch), patch.object(r, 'QUEUE', self.root / 'queue'):
            with self.assertRaisesRegex(RuntimeError, 'Repeated recovery'):
                r.build(self.root / 'missing-capture')

    def test_existing_queue_refuses_repeated_build(self):
        queue = self.root / 'queue'
        queue.mkdir()
        with patch.object(r, 'LAUNCH', self.root / 'launch.json'), patch.object(r, 'QUEUE', queue):
            with self.assertRaisesRegex(RuntimeError, 'Repeated recovery'):
                r.build(self.root / 'missing-capture')

    def consumer_fixture(self, invalid=False, all_resumed=False):
        pipeline = r.p.pipeline()
        r.install_consumer(pipeline)
        source, output = self.root / 'producer', self.root / 'consumer'
        pipeline.SOURCE_OUT = source
        pipeline.load_style_reuse = lambda paths: {}
        configuration = r.p.load(r.builder.EVIDENCE)['new_configuration']
        bar = SimpleNamespace(code_identity=lambda: {}, params=lambda *a: configuration,
              runtime_loader=lambda *a: lambda: (_ for _ in ()).throw(AssertionError('CPU model load forbidden')),
              process_source=lambda *a: (_ for _ in ()).throw(AssertionError('terminal source reprocessed')),
              __file__=str(r.p.RT / 'source_curated_consumer_v1/ms_bar_annotation.py'))
        pipeline.modules = lambda *a: (bar, None, SimpleNamespace(digest=r.native_digest))
        rows = [{'id': 'cpu_resume', 'split': 'train', 'batch_id': 'fixture', 'source_ordinal': 32005},
                {'id': 'cpu_new', 'split': 'train', 'batch_id': 'fixture', 'source_ordinal': 32014}]
        queues = []
        for row in rows:
            terminal = {**row, 'status': 'failed', 'manifest_record': row}
            pipeline.save(source / 'sources/train' / row['id'] / 'producer.terminal.json', terminal)
            queue = source / 'ready' / ('producer-%04d.jsonl' % (row['source_ordinal'] % 4))
            pipeline.append(queue, terminal)
            queues.append(queue)
        for i, row in enumerate(rows):
            if i == 0 or all_resumed:
                terminal = {**row, 'status': 'producer_failed', 'params_sha256': r.native_digest(configuration)}
                pipeline.save(output / 'sources/train' / row['id'] / 'consumer.terminal.json', terminal)
                pipeline.append(output / 'completed.jsonl', terminal)
        before = (output / 'completed.jsonl').read_bytes()
        manifest = self.root / 'manifest.json'
        r.p.write(manifest, {'total': 222939, 'counts': {}, 'batches': [{'batch_id': 'fixture', 'count': 500}]})
        args = SimpleNamespace(manifest=str(manifest), tp=2, batch_size=1, max_num_seqs=1,
               shard=1, nshard=9, poll_seconds=0.001, repo=None, ms_source=None, out=str(output),
               ids=[v['id'] for v in rows], limit=0, watch=True, style_reuse=[], stage='run')
        if invalid:
            good = queues[0].read_bytes()
            queues[0].write_bytes(b'{invalid}\n')
        else:
            good = None
        return pipeline, args, queues[0], good, output, before

    def test_existing_terminals_not_reprocessed_or_appended(self):
        pipeline, args, queue, good, output, before = self.consumer_fixture(all_resumed=True)
        result = pipeline.run_consumer(args)
        self.assertEqual(result['queue_sources_consumed'], 2)
        self.assertEqual(result['counts'], {'resumed_terminal': 2})
        self.assertEqual((output / 'completed.jsonl').read_bytes(), before)

    def test_invalid_ready_rewinds_then_recovers_without_skipping(self):
        pipeline, args, queue, good, output, before = self.consumer_fixture(invalid=True)
        sleeps = []
        def stabilize(seconds):
            sleeps.append(seconds)
            queue.write_bytes(good)
        with patch.object(pipeline.time, 'sleep', side_effect=stabilize):
            result = pipeline.run_consumer(args)
        self.assertTrue(sleeps)
        self.assertEqual(result['queue_sources_consumed'], 2)
        self.assertEqual(result['counts'], {'resumed_terminal': 1, 'producer_failed': 1})
        completed = [json.loads(v) for v in (output / 'completed.jsonl').read_text().splitlines()]
        self.assertEqual([v['id'] for v in completed].count('cpu_resume'), 1)
        self.assertEqual([v['id'] for v in completed].count('cpu_new'), 1)

    def test_persistent_invalid_ready_stops_without_false_completion(self):
        pipeline, args, queue, good, output, before = self.consumer_fixture(invalid=True)
        with patch.object(pipeline.time, 'sleep'), self.assertRaisesRegex(pipeline.SharedFailure, 'five reads'):
            pipeline.run_consumer(args)
        self.assertFalse(list((output / 'runs').glob('*/completed.json')))
        self.assertEqual(queue.read_bytes(), b'{invalid}\n')

    def test_native_summary_accepts_resumed_finite_completion(self):
        pipeline, args, queue, good, output, before = self.consumer_fixture(all_resumed=True)
        pipeline.run_consumer(args)
        cfg = {'mode': 'consumer', 'consumer_params_sha256': 'legacy',
               'native_consumer_params_sha256': r.native_digest(r.p.load(r.builder.EVIDENCE)['new_configuration']),
               'source_selection': {'shard': 1}, 'targetcount': 2,
               'producer': r.p.ref(r.ENTRY), 'collab3_source_plan': r.p.ref(r.p.PLAN),
               'plan': {'sha256': 'fixture'}}
        rows = [{'id': sid, 'split': 'train', 'batch_id': 'fixture'} for sid in args.ids]
        with patch.object(r.manager, 'selected_rows', return_value=rows):
            summary = r.summaries(cfg, output)
        self.assertEqual(next(iter(summary.values()))['summary']['processed_source_count'], 2)
        self.assertEqual((output / 'completed.jsonl').read_bytes(), before)

    def test_native_160gib_budget_not_relaxed(self):
        path = self.root / 'config.json'
        r.p.write(path, {'holders': [{'pid': 1}, {'pid': 2}], 'gpus': [4, 5]})
        args = SimpleNamespace(config=str(path), config_sha='fixture', manager_sha='fixture',
                               report=str(self.root / 'readonly.json'))
        with patch.object(r, 'LAUNCH', path), patch.object(r, 'validate'), \
             patch.object(r.base, 'check_sources'), patch.object(r.manager.c, 'check_model'), \
             patch.object(r.manager.old, 'readonly_lock_inventory', return_value=[]), \
             patch.object(r.os, 'pidfd_open', side_effect=lambda pid: r.os.open('/dev/null', r.os.O_RDONLY)), \
             patch.object(r.select, 'select', return_value=([], [], [])), \
             patch.object(r.base, 'headroom', return_value={'headroom': 160 * 2**30 - 1}):
            with self.assertRaisesRegex(RuntimeError, '160GiB'):
                r.readonly_admission(args)
        self.assertFalse(Path(args.report).exists())

    def test_summary_uses_native_digest_without_rewriting_legacy_config(self):
        cfg = {'consumer_params_sha256': 'legacy', 'native_consumer_params_sha256': 'native'}
        with patch.object(r.manager, 'summaries', return_value={'ok': True}) as call:
            self.assertEqual(r.summaries(cfg, self.root), {'ok': True})
        self.assertEqual(call.call_args.args[0]['consumer_params_sha256'], 'native')
        self.assertEqual(cfg['consumer_params_sha256'], 'legacy')


if __name__ == '__main__':
    unittest.main(verbosity=2)
