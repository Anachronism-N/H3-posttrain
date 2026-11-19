"""CPU/tmp tests only: no SSH, signal delivery, GPU queries or model copies."""
import copy
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import caption_producer_cache_handoff_v1 as hand
import caption_producer_cached_worker_v1 as worker
import caption_producer_cached_manager_v1 as manager


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.tmp.cleanup)
        self.source = self.root / 'shared'
        self.source.mkdir()
        for name, value in (('B', self.root), ('SOURCE_ROOT', self.source)):
            self.stack.enter_context(patch.object(hand, name, value))

    def file(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return hand.ref(path)

    def fixture(self):
        original = [{'id': 'source%d' % i, 'split': 'train', 'batch_id': 'train-b%06d' % (i//500),
                     'source_ordinal': i} for i in range(32000, 62000, 4)]
        selected = self.file('oldselected.json', original)
        old = {'argv': ['/opt/venv_eval/bin/python', '-B', '-u', 'oldworker.py', 'producer',
                        '--member', 'n2p0', '--manifest', 'originalsummary.json', '--out', str(self.source),
                        '--repo', 'frozenrepo', '--shard', '0', '--nshard', '4', '--limit', '0',
                        '--stage', 'run', '--device', '0', '--beat-device', 'cuda',
                        '--ordinal-start', '32000', '--ordinal-end', '62000'],
               'selected_sources': selected, 'queue': str(self.root / 'oldqueue')}
        oldref = self.file('oldcfg.json', old)
        self.stack.enter_context(patch.object(hand, 'OLD_CONFIG', Path(oldref['path'])))
        self.stack.enter_context(patch.object(hand, 'OLD_SHA', oldref['sha256']))
        rows = original[:2]
        remaining = self.file('remaining.json', rows)
        ids = self.root / 'ids.txt'
        ids.write_text(''.join(r['id'] + chr(10) for r in rows))
        parent = {'pid': 100, 'ppid': 90, 'argv': ['oldmanager.py']}
        child = {'pid': 101, 'ppid': 100, 'argv': old['argv']}
        admission = self.file('oldadmission.json', {'config_sha256': hand.OLD_SHA,
                              'manager_sha256': hand.BASE_SHA, 'manager_identity': parent})
        live = self.file('oldlive.json', {'manager_identity': parent, 'child_identity': child, 'argv': old['argv']})
        ready = self.source / 'ready' / 'producer-0000.jsonl'
        ready.parent.mkdir()
        ready.write_bytes(json.dumps(dict(original[2], status='ready')).encode() + bytes([10]))
        claim = self.file('claim.json', {'old_config': oldref, 'no_retry_after_failure': True})
        proof = {'schema': 'caption_producer_cache_handoff_v1', 'old_config': oldref,
                 'source_root': str(self.source), 'remaining_meta': remaining, 'ids_file': hand.ref(ids),
                 'source_owner_absence': True, 'old_gpu_tree_absent': True, 'sole_ready_writer': 'n2p0',
                 'old_config_resume_forbidden': True, 'old_selected': selected,
                 'excluded_started_meta': original[2:], 'old_admission': admission, 'old_live': live,
                 'old_parent': parent, 'old_child': child, 'old_tree_pids': [101],
                 'claim': claim, 'old_resource_safe': self.file('safe.json', {}),
                 'restored_holder': self.file('restore.json', {}), 'old_ready_prefix': hand.ready_prefix(ready)}
        cfg = {'old_config': oldref, 'node': 'N2', 'gpus': [0], 'mode': 'producer',
               'source_root': str(self.source), 'input_count': 222939, 'env': {'CUDA_VISIBLE_DEVICES': '0'},
               'phase': hand.PHASE, 'out': str(self.root / hand.PHASE),
               'queue': str(self.root / 'queues' / hand.PHASE),
               'source_selection': {'shard': 0, 'nshard': 4, 'limit': 0},
               'old_config_resume_forbidden': True, 'source_handoff': self.file('proof.json', proof),
               'remaining_meta': remaining, 'ids_file': hand.ref(ids), 'plan': remaining, 'targetcount': 2,
               'native_argv': hand.native_args(old, ids), 'producer': hand.ref(worker.__file__),
               'resource_manager': hand.ref(manager.__file__),
               'cache_code': hand.ref(hand.RT / 'caption_producer_stage_cache_v1.py'),
               'handoff_code': hand.ref(hand.__file__)}
        return cfg, rows, proof

    def test_exact_remaining_contract(self):
        cfg, rows, proof = self.fixture()
        self.assertEqual(worker.contract(cfg), (rows, proof))
        self.assertEqual(manager.selected(cfg), ([r['id'] for r in rows], 222939))

    def test_fixed_node_gpu_cvd_and_selection_reject(self):
        cfg, _, _ = self.fixture()
        for key, value in [('node', 'N7'), ('gpus', [1]), ('input_count', 2),
                           ('source_selection', {'shard': 1, 'nshard': 4, 'limit': 0})]:
            wrong = {**cfg, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                worker.contract(wrong)
        cfg['env']['CUDA_VISIBLE_DEVICES'] = '1'
        with self.assertRaises(RuntimeError):
            worker.contract(cfg)

    def test_wrong_actual_node_rejected(self):
        with patch.object(hand.subprocess, 'check_output', return_value='unrelated-address'):
            with self.assertRaises(RuntimeError):
                hand.fixed_node({'node': 'N2', 'gpus': [0], 'mode': 'producer'})

    def test_wrong_id_split_batch_ordinal_and_modulo_reject(self):
        original = [{'id': 'good', 'split': 'train', 'batch_id': 'train-b000064', 'source_ordinal': 32000}]
        for key, value in [('id', 'other'), ('split', 'validation'), ('batch_id', 'train-b000065'),
                           ('source_ordinal', 32001), ('source_ordinal', 64000)]:
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                hand.validate_meta([{**original[0], key: value}], original)
        with self.assertRaises(RuntimeError):
            hand.validate_meta(original + original, original)

    def test_native_args_no_budget_or_roi_changes(self):
        cfg, _, _ = self.fixture()
        cfg['native_argv'][cfg['native_argv'].index('--beat-device') + 1] = 'cpu'
        with self.assertRaises(RuntimeError):
            worker.contract(cfg)

    def test_all_started_and_roi_partial_excluded_unready_retained(self):
        rows = [{'id': 'a', 'split': 'train'}, {'id': 'b', 'split': 'train'},
                {'id': 'c', 'split': 'train'}, {'id': 'd', 'split': 'train'}]
        a = self.source / 'sources' / 'train' / 'a'
        a.mkdir(parents=True)
        (a / 'producer.terminal.json').write_text('{"status":"failed"}')
        b = self.source / 'sources' / 'train' / 'b'
        b.mkdir()
        (b / 'producer.prepare.json').write_text('{}')
        c = self.source / 'roi' / 'c'
        c.mkdir(parents=True)
        remaining, started = hand.split_remaining(rows, self.source)
        self.assertEqual(remaining, rows[3:])
        self.assertEqual(started, rows[:3])
        self.assertTrue((b / 'producer.prepare.json').exists())
        self.assertTrue(c.exists())

    def test_prefix_complete_lines_only_append_progress_never_overwrites(self):
        path = self.root / 'ready.jsonl'
        initial = b'{}' + bytes([10]) + b'{"partial":'
        path.write_bytes(initial)
        value = hand.ready_prefix(path)
        self.assertEqual(value['complete_bytes'], 3)
        self.assertEqual(value['complete_lines'], 1)
        self.assertGreater(value['trailing_bytes'], 0)
        path.write_bytes(initial + b'1}' + bytes([10]))
        hand.verify_prefix(value)
        self.assertEqual(path.read_bytes()[:len(initial)], initial)
        path.write_bytes(b'{ ' + bytes([10]))
        with self.assertRaises(RuntimeError):
            hand.verify_prefix(value)

    def test_failed_claim_no_retry(self):
        path = self.root / 'claim.json'
        hand.write_once(path, {'failed_attempt': True})
        with self.assertRaises(FileExistsError):
            hand.write_once(path, {'retry': True})
        self.assertEqual(hand.load(path), {'failed_attempt': True})

    def test_remaining_cannot_drop_unstarted_or_add_excluded(self):
        cfg, rows, proof = self.fixture()
        proof['excluded_started_meta'] = proof['excluded_started_meta'][1:]
        cfg['source_handoff'] = self.file('badproof.json', proof)
        with self.assertRaises(RuntimeError):
            worker.contract(cfg)

    def test_native_sources_check_and_producer_model_hook(self):
        cfg, _, _ = self.fixture()
        launch = self.file('newcfg.json', cfg)
        cfg.update(argv=['python', cfg['producer']['path']], source_sha256={})
        launch = self.file('newcfg.json', cfg)
        manager.base.check_sources(cfg, launch['path'], launch['sha256'], hand.BASE_SHA)
        self.assertEqual(manager.base.__file__, str(hand.RT / 'resource_manager.py'))
        self.assertEqual(hand.sha(manager.base.__file__), hand.BASE_SHA)
        weight = self.root / 'tiny_roi.onnx'
        weight.write_bytes(b'cpu fake ROI fixture')
        model = self.file('model.json', {'sha256': 'ROIidentity', 'files': {'pose': hand.ref(weight)}})
        cfg['modelref'] = {**model, 'model_identity_sha256': 'ROIidentity'}
        manager.base.check_model(cfg)
        self.assertIs(manager.producer_model_check.__globals__['base'], manager.base)
        self.assertEqual(manager.base.required_budget(cfg), 160 * 2**30)
        weight.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            manager.base.check_model(cfg)

    def test_real_selected_hook_rejects_wrong_member(self):
        cfg, rows, _ = self.fixture()
        rows[0]['source_ordinal'] = 32001
        cfg['remaining_meta'] = self.file('badremaining.json', rows)
        with self.assertRaises(RuntimeError):
            manager.base.selected_sources(cfg)

    def test_old_roi_gpu_app_is_child_tree_not_holder_even_small_memory(self):
        queue = self.root / 'oldqueue'
        queue.mkdir()
        parent = {'pid': 3000, 'ppid': 2999, 'uid': 0, 'state': 'R', 'start_ticks': 1,
                  'argv': ['python', 'oldproducer', str(hand.RT / 'curated_collaborative3_resource_manager_v1.py'),
                           '--config', str(hand.OLD_CONFIG), '--config-sha', hand.OLD_SHA, '--manager-sha', hand.BASE_SHA], 'CVD': [], 'exe': 'python',
                  'cgroup': 'owned', 'sig_ign': 0, 'sig_blk': 0}
        child = {**parent, 'pid': 3001, 'ppid': 3000, 'start_ticks': 2,
                 'argv': ['python', 'oldproducer'], 'CVD': ['CUDA_VISIBLE_DEVICES=0']}
        cfg = {'queue': str(queue), 'producer': {'producer': True}, 'out': str(self.source),
               'argv': child['argv'], 'phase': 'old', 'holders': [{'uuid': 'old-uuid'}]}
        admission = {'config_sha256': hand.OLD_SHA, 'manager_sha256': hand.BASE_SHA,
                     'node': 'N2', 'gpus': [0], 'producer': cfg['producer'], 'out': cfg['out'],
                     'manager_identity': parent}
        (queue / 'admission.json').write_text(json.dumps(admission))
        (queue / 'live.json').write_text(json.dumps({'manager_identity': parent, 'child_identity': child,
                                                     'argv': cfg['argv'], 'phase': 'old'}))
        base = SimpleNamespace(IDENTITY_KEYS=('pid','uid','argv','start_ticks','cgroup','CVD','exe'),
                identity=lambda pid: parent if pid == 3000 else child,
                uuid_for=lambda gpu: 'old-uuid',
                apps=lambda: [{'uuid':'old-uuid','pid':3001,'used_MiB':'1340'}])
        original_lstat = Path.lstat
        lease = SimpleNamespace(st_mode=0o100600, st_uid=0, st_size=0, st_mtime=hand.time.time())
        def lstat(path, *a, **k):
            return lease if str(path) == '/tmp/od_reserve_0' else original_lstat(path, *a, **k)
        with (patch.object(hand, 'lock_state'), patch.object(hand, 'proc_inventory', return_value={3001:child}),
                patch.object(hand, 'ref', return_value={'fixture':True}), patch.object(Path, 'lstat', lstat),
                patch.object(hand.os, 'getuid', return_value=0)):
            self.assertEqual(hand.preflight_old(base, cfg)['old_tree_pids'], [3001])
            base.apps = lambda: [{'uuid':'old-uuid','pid':9999,'used_MiB':'1340'}]
            with self.assertRaises(RuntimeError):
                hand.preflight_old(base, cfg)

    def test_new_validate_uses_real_contract_with_hardware_only_mocked(self):
        cfg, _, proof = self.fixture()
        old = hand.checked(cfg['old_config'])
        for key in ('manifest_summary', 'original_full_manifest', 'modelref'):
            old[key] = {'path': key, 'sha256': key}
        cfg['old_config'] = self.file('oldcfg.json', old)
        self.stack.enter_context(patch.object(hand, 'OLD_SHA', cfg['old_config']['sha256']))
        proof['old_config'] = cfg['old_config']
        admission = hand.checked(proof['old_admission'])
        admission['config_sha256'] = hand.OLD_SHA
        proof['old_admission'] = self.file('oldadmission.json', admission)
        proof['claim'] = self.file('claim.json', {'old_config': cfg['old_config'], 'no_retry_after_failure': True})
        cfg['source_handoff'] = self.file('proof.json', proof)
        cfg.update(release_authorized=True, holders=[{'gpu': 0}], cwd=str(self.root),
                   argv=old['argv'][:3] + [cfg['producer']['path'], '--config',
                       str(hand.RT / (hand.PHASE + '_launch.json'))])
        cfg.update({k: old[k] for k in ('manifest_summary', 'original_full_manifest', 'modelref')})
        capture = {'captured_unix_ns': hand.time.time_ns(), 'node': 'N2', 'gpus': [0],
                   'mode': 'producer', 'holders': cfg['holders'], 'signals_sent': False, 'leases': [None]}
        cfg['holder_capture'] = self.file('capture.json', capture)
        with patch.object(hand, 'fixed_node'), patch.object(hand, 'safe_gate',
                return_value=(cfg['holders'][0], None, {'headroom': 160 * 2**30})):
            manager.base.validate_pair(cfg)
            (self.source / 'roi' / 'source32000').mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                manager.base.validate_pair(cfg)

    def test_parent_gate_binds_native_argv_live_identity_uid_cgroup_and_sole_gpu(self):
        cfg, _, proof = self.fixture()
        cfg['argv'] = ['/opt/venv_eval/bin/python', '-B', '-u', cfg['producer']['path'], '--config', 'config']
        queue = Path(cfg['queue'])
        queue.mkdir(parents=True)
        config = self.root / 'boundlaunch.json'
        config.write_text(json.dumps(cfg))
        parent = {'pid': 1000, 'ppid': 999, 'uid': 0, 'state': 'S', 'start_ticks': 7,
                  'cgroup': 'owned-cgroup', 'CVD': [], 'exe': '/usr/bin/python3.11',
                  'argv': cfg['argv'][:3] + [cfg['resource_manager']['path'], '--config', str(config),
                       '--config-sha', hand.sha(config), '--manager-sha', hand.BASE_SHA]}
        child = {**parent, 'pid': 1001, 'ppid': 1000, 'start_ticks': 8, 'argv': cfg['argv'],
                 'CVD': ['CUDA_VISIBLE_DEVICES=0']}
        admission = {'manager_identity': parent, 'node': 'N2', 'gpus': [0],
                     'config_sha256': hand.sha(config), 'manager_sha256': hand.BASE_SHA,
                     'producer': cfg['producer'], 'plan': cfg['plan'], 'out': cfg['out'], 'targetcount': 2}
        (queue / 'admission.json').write_text(json.dumps(admission))
        live = {'manager_identity': parent, 'child_identity': child, 'argv': cfg['argv']}
        (queue / 'live.json').write_text(json.dumps(live))
        proof['gpu_uuid'] = 'fixture-uuid'
        base = SimpleNamespace(IDENTITY_KEYS=('pid','uid','argv','start_ticks','cgroup','CVD','exe'),
                identity=lambda pid: parent if pid == 1000 else child,
                uuid_for=lambda gpu: 'fixture-uuid', apps=lambda: [])
        original_lstat = Path.lstat
        lease = SimpleNamespace(st_mode=0o100600, st_uid=0, st_size=0, st_mtime=hand.time.time())
        def lstat(path, *args, **kwargs):
            return lease if str(path) == '/tmp/od_reserve_0' else original_lstat(path, *args, **kwargs)
        with (patch.object(hand, 'fixed_node'), patch.object(hand, 'old_owner_absence'),
                patch.object(hand, 'lock_state'), patch.object(Path, 'lstat', lstat),
                patch.object(worker.os, 'getpid', return_value=1001),
                patch.object(worker.os, 'getppid', return_value=1000),
                patch.object(worker.os, 'getuid', return_value=0),
                patch.dict(worker.os.environ, {'CUDA_VISIBLE_DEVICES':'0'})):
            self.assertEqual(worker.parent_gate(base, cfg, config, proof)['gpu_uuid'], 'fixture-uuid')
            for key, value in [('ppid', 2000), ('uid', 9), ('cgroup', 'foreign'),
                               ('CVD', ['CUDA_VISIBLE_DEVICES=1'])]:
                original = child[key]
                child[key] = value
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    worker.parent_gate(base, cfg, config, proof)
                child[key] = original
            base.apps = lambda: [{'uuid': 'fixture-uuid', 'pid': 99}]
            with self.assertRaises(RuntimeError):
                worker.parent_gate(base, cfg, config, proof)

    def fake_run(self, fail=False):
        metadata = {'id': 's32000', 'split': 'train', 'batch_id': 'train-b000064', 'source_ordinal': 32000}
        events, saves, appends = [], [], []
        execution = self.root / 'execution'
        execution.mkdir()
        roi = SimpleNamespace(load_cuda_model=lambda *a, **k: 'model')
        pipeline = SimpleNamespace(modules=lambda repo: ('bar', roi, 'visual'),
                                   append=lambda path, value: appends.append((str(path), value)),
                                   save=lambda path, value: saves.append((str(path), value)),
                                   iter_rows=lambda *a, **k: iter([(metadata, {})]), run_producer=None)
        native_rows, native_append, native_save, native_modules = pipeline.iter_rows, pipeline.append, pipeline.save, pipeline.modules
        native_publish = lambda *a: 'original'
        publisher = lambda *a: 'v4'
        io2 = SimpleNamespace(load_pipeline=lambda path=None: pipeline, publish_roi=native_publish,
                              LocalROI='original-local', FROZEN='frozen')
        loader0 = io2.load_pipeline
        class Cache:
            def __init__(self, p, bar, r, visual, out, emit):
                self.p = p
            def run_wrapper(self, original):
                return original
            @contextmanager
            def installed(self, publishers):
                self_test.assertIs(publishers[0], io2)
                self_test.assertIs(io2.publish_roi, publisher)
                events.append('installed')
                old_rows, old_append, old_publish = self.p.iter_rows, self.p.append, io2.publish_roi
                def cached_rows(*a, **k):
                    events.append('rows-cached')
                    yield from old_rows(*a, **k)
                def cached_append(*a, **k):
                    events.append('append-cached')
                    return old_append(*a, **k)
                self.p.iter_rows, self.p.append = cached_rows, cached_append
                io2.publish_roi = lambda *a: old_publish(*a)
                try:
                    yield self
                finally:
                    self.p.iter_rows, self.p.append, io2.publish_roi = old_rows, old_append, old_publish
                    events.append('restored')
        self_test = self
        def io_main(args):
            oldloader, oldpub, oldlocal = io2.load_pipeline, io2.publish_roi, io2.LocalROI
            io2.publish_roi, io2.LocalROI = publisher, 'v4local'
            try:
                p = io2.load_pipeline()
                previous_rows, previous_append = p.iter_rows, p.append
                def installed_rows(*args, **kwargs):
                    yield from previous_rows(*args, **kwargs)
                p.iter_rows = installed_rows
                p.modules = lambda repo: ('bar', SimpleNamespace(roi=roi), 'visual')
                events.append('captured')
                try:
                    return p.run_producer(SimpleNamespace(repo='realrepo'))
                finally:
                    p.iter_rows, p.append = previous_rows, previous_append
            finally:
                io2.load_pipeline, io2.publish_roi, io2.LocalROI = oldloader, oldpub, oldlocal
        def frozen_run(args):
            run = self.source / 'runs' / 'p0-cpu-fixture'
            pipeline.save(run / 'protocol.json', {'pipeline_code': {'native': True}, 'native_field': 17})
            list(pipeline.iter_rows())
            bad = {**metadata, 'id': 'other'} if fail else metadata
            pipeline.append(self.source / 'ready.jsonl', {**bad, 'status': 'ready'})
            return {'run_root': str(run), 'counts': {'ready': 1}}
        pipeline.run_producer = frozen_run
        io4 = SimpleNamespace(base=io2, publish_roi=publisher, main=io_main)
        cfg = {'native_argv': ['--repo', 'realrepo'], 'source_root': str(self.source),
               'cache_code': {'cache': True}, 'producer': {'worker': True},
               'source_handoff': {'handoff': True}, 'remaining_meta': {'remaining': True},
               'handoff_code': {'handoff_code': True}}
        with patch.object(hand, 'native_base', return_value=SimpleNamespace()):
            if fail:
                with self.assertRaises(RuntimeError):
                    worker.run_cached(cfg, [metadata], {}, io4, Cache, lambda r: None, execution)
            else:
                result = worker.run_cached(cfg, [metadata], {}, io4, Cache, lambda r: None, execution)
                self.assertEqual(result['counts'], {'ready': 1})
                self.assertEqual(appends[0][0], str(self.source / 'ready' / 'producer-0000.jsonl'))
                self.assertEqual(saves[0][1]['native_field'], 17)
                self.assertIn('caption_producer_cache_code', saves[0][1])
        self.assertIs(io2.load_pipeline, loader0)
        self.assertIs(io2.publish_roi, native_publish)
        self.assertEqual(io2.LocalROI, 'original-local')
        self.assertIs(pipeline.iter_rows, native_rows)
        self.assertIs(pipeline.append, native_append)
        self.assertIs(pipeline.save, native_save)
        self.assertIs(pipeline.modules, native_modules)
        self.assertLess(events.index('captured'), events.index('installed'))
        self.assertIn('rows-cached', events)
        self.assertEqual(events[-1], 'restored')

    def test_run_hook_after_io2_capture_actual_publisher_and_restore(self):
        self.fake_run()

    def test_native_parent_created_execution_is_reused_without_second_mkdir(self):
        cfg, rows, proof = self.fixture()
        self.stack.enter_context(patch.object(hand, 'RT', self.root))
        config = self.root / (hand.PHASE + '_launch.json')
        config.write_text(json.dumps(cfg))
        execution = Path(cfg['out'])
        execution.mkdir()
        fake_io = SimpleNamespace()
        def finish(*args):
            hand.write_once(execution / 'native_run.json', dict(run_root='cpu_fixture'))
            return dict(run_root='cpu_fixture', counts={'ready': len(rows)})
        with patch.object(hand, 'load', return_value=cfg), patch.object(worker, 'contract', return_value=(rows, proof)), \
             patch.object(hand, 'native_base', return_value=SimpleNamespace()), patch.object(worker, 'parent_gate', return_value={}), \
             patch.object(hand, 'split_remaining', return_value=(rows, [])), patch.object(worker, 'run_cached', side_effect=finish):
            result = worker.main(['--config', str(config)])
        self.assertEqual(result['summary']['processed_source_count'], len(rows))
        self.assertTrue((execution / 'producer_owner_claim.json').exists())

    def test_handoff_low_headroom_opens_no_pidfd_or_claim(self):
        cfg = dict(node='N2', gpus=[0], mode='producer')
        base = SimpleNamespace(SAFETY=80 * 2**30, LOAD=80 * 2**30, headroom=lambda: dict(headroom=159 * 2**30))
        with patch.object(hand, 'old_config', return_value=cfg), patch.object(hand, 'fixed_node'), \
             patch.object(hand, 'native_base', return_value=base), patch.object(hand, 'write_once') as writing, \
             patch.object(hand.os, 'pidfd_open') as opening:
            with self.assertRaisesRegex(RuntimeError, 'before parking'):
                hand.prepare(SimpleNamespace(wait_seconds=5))
            writing.assert_not_called()
            opening.assert_not_called()

    def test_old_pid_reuse_is_not_mistaken_for_old_source_owner(self):
        old = dict(pid=101, start_ticks=1, argv=['oldproducer'])
        current = dict(pid=101, start_ticks=2, argv=['unrelated'], state='S')
        base = SimpleNamespace(apps=lambda: [])
        proof = dict(old_child=old, old_tree_pids=[101], old_tree_identities=[old], gpu_uuid='oldgpu')
        with patch.object(hand, 'proc_inventory', return_value={101: current}):
            hand.old_owner_absence(base, proof)
            current['start_ticks'] = 1
            with self.assertRaises(RuntimeError):
                hand.old_owner_absence(base, proof)

    def test_wrong_ready_source_refused_and_all_functions_restored(self):
        self.fake_run(fail=True)


if __name__ == '__main__':
    unittest.main()
