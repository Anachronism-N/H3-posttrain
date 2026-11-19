"""CPU-only enabled-feature tests; no model load, signals, or GPU lease admission."""
import argparse
import copy
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
import curated_collaborative4_plan_v1 as p
import curated_collaborative4_build_v1 as b
import curated_collaborative4_consumer_v1 as w
import curated_collaborative4_resource_manager_v1 as m


class Collab4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.capture = self.root / 'capture.json'
        holders = [{'gpu': g, 'pid': 900000 + g, 'uuid': 'test-gpu-%d' % g} for g in (4, 5)]
        p.write(self.capture, {'node': 'N1', 'mode': 'consumer', 'gpus': [4, 5],
                'holders': holders, 'leases': [None, None], 'signals_sent': False,
                'captured_unix_ns': time.time_ns()})
        self.cfg = b.construct('n1c0', p.ref(self.capture))
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
        with patch.object(m.c.subprocess, 'check_output', return_value=p.HOSTS['N1']), \
             patch.object(m, 'verify_capture'), patch.object(m, 'verify_reclaim'), patch.object(m.old, 'readonly_lock_inventory', return_value=[]), \
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
            self.assertEqual(sorted(r['source_ordinal'] for r in rows), list(range(62000, 82000)))
            self.assertEqual(len({r['id'] for r in rows}), 20000)
        self.assertEqual(len(all_rows), 20000)
        occupied = {(n, gpu) for n, _, gpus, _, _ in p.MEMBERS.values() for gpu in gpus}
        self.assertEqual(len(occupied), 9)
        self.assertNotIn(('N7', 6), occupied)
        self.assertNotIn(('N7', 7), occupied)

    def test_range_mutation_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['source_range']['start'] = 31000
        with self.assertRaises(RuntimeError):
            self.validate(cfg)

    def test_member_mutation_refused(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['collab4_member'] = 'n1c1'
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
        with patch.object(p, 'paths', return_value={'launch': launch, 'queue': self.root / 'newqueue', 'out': p.paths('n1c0')['out']}):
            with self.assertRaisesRegex(RuntimeError, 'Repeated member dispatch'):
                b.build('n1c0', self.capture)

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
            b.construct('n1c0', p.ref(stale))

    def test_active_kernel_lock_readonly_refused(self):
        lock = self.root / 'lock'
        lock.touch(mode=0o600)
        st = lock.stat()
        key = '%02x:%02x:%d' % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
        original_read = Path.read_text
        original_path = m.old.Path
        def path(value):
            return lock if str(value) == '/tmp/ms12k_gpu_4.lock' else original_path(value)
        def read(value, *args, **kwargs):
            return '1: FLOCK ADVISORY WRITE 123 ' + key + ' 0 EOF' if str(value) == '/proc/locks' else original_read(value, *args, **kwargs)
        with patch.object(m.old, 'Path', side_effect=path), patch.object(Path, 'read_text', read):
            with self.assertRaisesRegex(RuntimeError, 'Active kernel GPU lock'):
                m.old.readonly_lock_inventory([4])

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

    def test_single_ready_stream_isolated_terminals_zero_models(self):
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
        for index, ordinal in enumerate((62000, 62004, 62008, 62012)):
            sid = 'cpu_source_%d' % index
            ids.append(sid)
            row = {'id': sid, 'split': 'train', 'batch_id': 'train-b000124', 'source_ordinal': ordinal}
            terminal = {**row, 'status': 'failed', 'manifest_record': row}
            pth = source / 'sources/train' / sid / 'producer.terminal.json'
            pipeline.save(pth, terminal)
            pipeline.append(source / 'ready' / 'producer-0000.jsonl', terminal)
        manifest = self.root / 'summary.json'
        p.write(manifest, {'total': 222939, 'counts': {'train': 200568, 'val': 11110, 'test': 11261},
                'batches': [{'batch_id': 'train-b000124', 'count': 500}]})
        args = SimpleNamespace(manifest=str(manifest), tp=2, batch_size=1, max_num_seqs=1,
                shard=0, nshard=4, poll_seconds=0.001, repo=None, ms_source=None, out=str(output),
                ids=ids, limit=0, watch=False, style_reuse=[], stage='run')
        result = pipeline.run_consumer(args)
        self.assertEqual(result['queue_sources_consumed'], 4)
        completed = [json.loads(line) for line in (output/'completed.jsonl').read_text().splitlines()]
        self.assertEqual({v['id'] for v in completed}, set(ids))
        self.assertFalse((source/'completed.jsonl').exists())
        self.assertFalse(any((source/'sources/train'/sid/'consumer.terminal.json').exists() for sid in ids))
        for sid in ids:
            self.assertTrue((output/'sources/train'/sid/'consumer.terminal.json').exists())


    def test_previous_reservation_overlap_refused(self):
        plan = p.load(p.PLAN)
        previous = p.checked
        def checked(reference):
            value = previous(reference)
            if reference == plan['previous_reservations'][-1]:
                value['source_range']['end'] = 62001
            return value
        with patch.object(p, 'checked', side_effect=checked):
            with self.assertRaisesRegex(RuntimeError, 'overlaps'):
                p.verify_plan(plan, verify_sources=False)

    def test_original_batch_membership_refused(self):
        plan = p.load(p.PLAN)
        plan['selected_original_batches'][0]['count'] = 499
        with self.assertRaisesRegex(RuntimeError, 'descriptor membership'):
            p.verify_plan(plan)

    def test_empty_gpu_bootstrap_capture_refused(self):
        args = SimpleNamespace(node='N1', mode='consumer', gpus=[4, 5], out=str(self.root/'empty.json'))
        with patch.object(m.old.subprocess, 'check_output', return_value=p.HOSTS['N1']),              patch.object(m.base, 'uuid_for', return_value='empty-gpu'),              patch.object(m.base, 'apps', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'Exactly one authorized holder'):
                m.old.capture_current(args)
        self.assertFalse(Path(args.out).exists())

    def holder(self, gpu=4):
        old = next(x for x in p.load(p.RECLAIM['N1'])['before_targets'] if x['gpu'] == gpu)
        return dict(gpu=gpu, uuid=old['uuid'], pid=900000+gpu, ppid=800000,
                    uid=os.getuid(), argv=['/opt/venv_eval/bin/python', '-u', str(m.base.HOLDER)],
                    start_ticks=old['start_ticks']+100000, cgroup='cpu-fixture',
                    CVD=['CUDA_VISIBLE_DEVICES='+str(gpu)], exe='/usr/bin/python3.11',
                    sig_ign=0, sig_blk=0, state='S')

    def test_unknown_compute_holder_identity_and_term_refused(self):
        holder = self.holder()
        current = copy.deepcopy(holder)
        with self.assertRaisesRegex(RuntimeError, 'nonholder compute'):
            m.base.check_holder(holder, current, 4, holder['uuid'],
                [{'pid': holder['pid'], 'uuid': holder['uuid']}, {'pid': 999999, 'uuid': holder['uuid']}])
        for key, value in (('uid', os.getuid()+1), ('start_ticks', holder['start_ticks']+1),
                           ('CVD', ['CUDA_VISIBLE_DEVICES=5'])):
            current = {**holder, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'identity changed'):
                m.base.check_holder(holder, current, 4, holder['uuid'],
                                    [{'pid': holder['pid'], 'uuid': holder['uuid']}])
        current = {**holder, 'sig_ign': 1 << 14}
        with self.assertRaisesRegex(RuntimeError, 'SIGTERM ignored'):
            m.base.check_holder(holder, current, 4, holder['uuid'],
                                [{'pid': holder['pid'], 'uuid': holder['uuid']}])

    def test_new_bootstrap_reclaim_identity_and_owner_absence(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['holders'] = [self.holder(4), self.holder(5)]
        with patch.object(m.base, 'identity', side_effect=FileNotFoundError):
            m.verify_reclaim(cfg)
        bad = copy.deepcopy(cfg)
        bad['holders'][0]['uuid'] = 'wrong-gpu'
        with self.assertRaisesRegex(RuntimeError, 'reclaimed GPU'):
            m.verify_reclaim(bad)
        old = p.load(p.RECLAIM['N1'])['before_managers'][0]
        with patch.object(m.base, 'identity', return_value={**old, 'state':'S'}):
            with self.assertRaisesRegex(RuntimeError, 'remains alive'):
                m.verify_reclaim(cfg)

    def test_fresh_holder_parent_binding_refused(self):
        capture = p.load(self.capture)
        capture['holders'] = [self.holder(4), self.holder(5)]
        path = self.root/'parents.json'
        p.write(path, capture)
        cfg = {**self.cfg, 'holder_capture':p.ref(path), 'holders':capture['holders']}
        def identity(pid):
            holder = next(x for x in cfg['holders'] if x['pid'] == pid)
            return {**holder, 'ppid':holder['ppid']+1}
        with patch.object(m.base, 'lease_snapshot', return_value=None),              patch.object(m.base, 'identity', side_effect=identity),              patch.object(m.base, 'uuid_for', side_effect=lambda gpu:self.holder(gpu)['uuid']),              patch.object(m.base, 'apps', return_value=[{'pid':x['pid'], 'uuid':x['uuid']} for x in cfg['holders']]):
            with self.assertRaisesRegex(RuntimeError, 'parent identity changed'):
                m.verify_capture(cfg)

    def test_first_pair_build_returns_safe_cli_without_dispatch(self):
        launch = self.root/'firstpair.json'
        out = p.paths('n1c0')['out']
        with patch.object(p, 'paths', return_value={'launch':launch, 'queue':self.root/'newqueue', 'out':out}):
            value = b.build('n1c0', self.capture)
        self.assertFalse(value['gpu_started'])
        self.assertFalse(value['signals_sent'])
        self.assertFalse(value['leases_created'])
        self.assertIn('--readonly-admission', value['readonly_command'])
        self.assertNotIn('--readonly-admission', value['command'])
        self.assertEqual(value['gpus'], [4, 5])
        self.assertEqual(value['targetcount'], 5000)
        self.assertEqual(p.sha(launch), value['launch']['sha256'])

    def test_producer_enabled_single_ready_routing(self):
        import curated_collaborative4_producer_v1 as producer
        import curated_producer_io_v4 as native
        events = []
        pipeline = SimpleNamespace(append=lambda path,row:events.append((Path(path),row)),
                                   save=lambda path,row:events.append((Path(path),row)))
        argv = ['--member','n5p0','producer','--manifest',str(p.MANIFEST),'--out',str(p.OUT),
                '--shard','0','--nshard','1','--limit','0','--ordinal-start','62000','--ordinal-end','82000']
        def run(remaining):
            loaded = native.base.load_pipeline()
            loaded.append(p.OUT/'ready.jsonl', {'source_ordinal':62000})
            loaded.save(self.root/'protocol.json', {'mode':'producer'})
            return {'fixture':True}
        original = native.base.load_pipeline
        try:
            with patch.object(native.base, 'load_pipeline', return_value=pipeline), patch.object(native, 'main', side_effect=run):
                producer.main(argv)
        finally:
            native.base.load_pipeline = original
        self.assertEqual(events[0][0], p.OUT/'ready/producer-0000.jsonl')
        self.assertEqual(events[1][1]['collab4_member'], 'n5p0')
        self.assertEqual(events[1][1]['collab4_source_plan'], p.ref(p.PLAN))


    def test_actual_native_consumer_params_keep_model_batch_images_sampling(self):
        pipeline = p.pipeline()
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--member')
        parser.add_argument('--selected-sources')
        _, remaining = parser.parse_known_args(self.cfg['argv'][4:])
        args = pipeline.parser().parse_args(remaining)
        actual = w.configuration(pipeline, args)
        self.assertEqual(actual, p.load(b.EVIDENCE)['new_configuration'])
        self.assertEqual(b.digest(actual), self.cfg['consumer_params_sha256'])
        self.assertEqual(args.tp, 2)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.max_num_seqs, 2)
        old = p.load(b.EVIDENCE)['original_configuration']
        self.assertEqual({k:v for k,v in actual.items() if k != 'hand_roi_root'},
                         {k:v for k,v in old.items() if k != 'hand_roi_root'})

    def test_first_producer_build_cpu_reference_only(self):
        capture = self.root/'producer_capture.json'
        p.write(capture, {'node':'N5', 'mode':'producer', 'gpus':[0],
                         'holders':[{'gpu':0, 'pid':900000, 'uuid':'cpu-fixture-gpu0'}],
                         'leases':[None], 'signals_sent':False, 'captured_unix_ns':time.time_ns()})
        node_roi = p.RT/'curated_collaborative4_roi_N5_v1.json'
        original_roi = p.load(b.ROI_TEMPLATE)
        original_load, original_ref = p.load, p.ref
        def load(path):
            return copy.deepcopy(original_roi) if Path(path) == node_roi else original_load(path)
        def ref(path):
            return {'path':str(node_roi), 'sha256':p.sha(b.ROI_TEMPLATE)} if Path(path) == node_roi else original_ref(path)
        launch = self.root/'firstproducer.json'
        with patch.object(p, 'load', side_effect=load), patch.object(p, 'ref', side_effect=ref),              patch.object(p, 'paths', return_value={'launch':launch, 'queue':self.root/'newproducerqueue', 'out':p.OUT}):
            value = b.build('n5p0', capture)
        cfg = original_load(launch)
        self.assertEqual(cfg['targetcount'], 20000)
        self.assertEqual(cfg['source_selection'], {'shard':0, 'nshard':1, 'limit':0})
        self.assertEqual(cfg['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(m.base.argv_option(cfg, '--ordinal-start'), '62000')
        self.assertEqual(m.base.argv_option(cfg, '--ordinal-end'), '82000')
        self.assertEqual(m.base.argv_option(cfg, '--roi-model-reference'), str(node_roi))
        self.assertFalse(value['signals_sent'])
        self.assertFalse(value['gpu_started'])
        self.assertFalse(value['leases_created'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
