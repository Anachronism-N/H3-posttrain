"""CPU contract tests only: fixture proofs/resources are never launch artifacts."""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module('salvage_current_contract', RT / 'prepare_salvage_current_resource_restore_v1.py')
native = module('salvage_original_primitives', RT / 'resource_manager.py')


class ProofOrderTests(unittest.TestCase):
    def test_full_proof_uses_original_replay_and_never_refreshes_resources(self):
        expected = {'processed_source_count': 662, 'real_audit_receipt_count': 110,
                    'model_calls_performed': 0, 'state_migrated': False, 'original_attempt_budgets_unchanged': True}
        calls = []
        worker = NS(w=NS(Manifest=lambda path: calls.append('manifest') or path), pilot=NS(MANIFEST='/fixture/manifest'))
        prepare = NS(worker=worker, resume=NS(resume_state_proof=lambda *a: calls.append('full_original_proof') or copy.deepcopy(expected)))
        fake_review = NS()
        with patch.dict(sys.modules, {'ms_batch_review': fake_review}), patch.object(m, 'checked', return_value=expected), \
                patch.object(m, 'fresh_current', side_effect=AssertionError('resource called during proof')):
            self.assertEqual(m.full_proof(prepare, {}), expected)
        self.assertEqual(calls, ['manifest', 'full_original_proof'])
        prepare.resume.resume_state_proof = lambda *a: dict(expected, real_audit_receipt_count=109)
        with patch.dict(sys.modules, {'ms_batch_review': fake_review}), patch.object(m, 'checked', return_value=expected):
            with self.assertRaisesRegex(RuntimeError, 'proof differs'):
                m.full_proof(prepare, {})

    def test_long_legal_evidence_precedes_fresh_capture_and_plan_copy_is_samebytes(self):
        with tempfile.TemporaryDirectory(prefix='salvage_current_order_') as tmp:
            root = Path(tmp)
            old_plan_path = root / 'old-plan.json'
            old_plan_path.write_bytes(b'{"fixture":true}\n')
            old_ref = {'path': str(old_plan_path), 'sha256': hashlib.sha256(old_plan_path.read_bytes()).hexdigest()}
            events, written = [], {}
            clock = [0]
            old = {'argv': ['/python', '/frozen/worker', '--launch', '/old/launch', '--launch-sha', 'oldsha', '--model', '/same/model', '--watch'],
                   'env': {'CUDA_VISIBLE_DEVICES': '3'}, 'out': '/same/out', 'producer': {'path': '/frozen/worker', 'sha256': 'same'},
                   'modelref': {'path': '/same/modelref', 'sha256': 'same'}, 'worker_param': 91}
            proof = {'processed_source_count': 662, 'real_audit_receipt_count': 110}
            plan = {'resume_state_ref': {'path': '/original/proof', 'sha256': 'original-proof-sha'}}
            def slow_evidence(*args, **kwargs):
                events.append('complete_full_proof_source_and_model')
                clock[0] = 3600 * 10**9
                return old, plan, proof, {'receipt_refs': {}}, {}
            def fresh(*args):
                events.append('actual_fresh')
                return {'holders': [{'pid': 41}], 'lease_snapshot_ns': {'/lease': None}, 'captured_unix_ns': clock[0]}
            def write(path, value):
                written[str(path)] = copy.deepcopy(value)
                Path(path).write_text(json.dumps(value))
            def validate(cfg):
                events.append('original_validate')
                self.assertLessEqual(clock[0] - written[cfg['holder_capture']['path']]['captured_unix_ns'], 1200 * 10**9)
            prepare = NS(worker=NS(write_once=write), manager=NS(QUEUE_PREFIX='fixture-', validate=validate))
            with patch.object(m, 'RT', root), patch.object(m, 'OLD_PLAN', old_ref), \
                    patch.object(m, 'evidence', side_effect=slow_evidence), patch.object(m, 'fresh_current', side_effect=fresh), \
                    patch.object(m.time, 'time_ns', side_effect=lambda: clock[0]):
                result = m.build(prepare, root / 'fixture-config.json', {'path': '/safe', 'sha256': 'safe'})
            cfg = written[str(root / 'fixture-config.json')]
            self.assertEqual(events, ['complete_full_proof_source_and_model', 'actual_fresh', 'original_validate'])
            self.assertEqual(Path(cfg['plan']['path']).read_bytes(), old_plan_path.read_bytes())
            self.assertEqual(cfg['plan']['sha256'], old_ref['sha256'])
            for key in ('env', 'out', 'producer', 'modelref', 'worker_param'):
                self.assertEqual(cfg[key], old[key])
            self.assertEqual(cfg['resume_state_ref'], plan['resume_state_ref'])
            self.assertFalse(result['signals_sent'])
            self.assertEqual(result['model_calls_performed'], 0)
            self.assertEqual(Path(result['command'][3]), m.SELF)
            self.assertEqual(result['command'][-4], '--execution-sha')

    def test_old_SHA_cannot_be_resigned_and_missing_safe_default_denies(self):
        with self.assertRaisesRegex(RuntimeError, 're-signed'):
            m.merge({'/old/source': 'original'}, {'/old/source': 'replacement'})
        with self.assertRaisesRegex(RuntimeError, 'actual previous safe'):
            m.safe_chain(NS(base=native), {'path': '/mock/config', 'sha256': 'mock'})

    def test_process_cache_miss_fullhash_and_hit_stat_identity(self):
        with tempfile.TemporaryDirectory(prefix='salvage_current_hash_') as tmp:
            path = Path(tmp) / 'source'
            path.write_bytes(b'original')
            m._cache.clear()
            self.assertEqual(m.content_sha(path), hashlib.sha256(b'original').hexdigest())
            with patch.object(Path, 'open', side_effect=AssertionError('cache hit must only stat')):
                self.assertEqual(m.content_sha(path), hashlib.sha256(b'original').hexdigest())
            path.write_bytes(b'changed')
            self.assertEqual(m.content_sha(path), hashlib.sha256(b'changed').hexdigest())


class CurrentResourceTests(unittest.TestCase):
    def setUp(self):
        self.holder = {'pid': 71, 'uid': os.getuid(), 'argv': list(native.RESTORE_ARGV),
                       'start_ticks': 100, 'cgroup': '/fixture', 'CVD': ['CUDA_VISIBLE_DEVICES=3'],
                       'exe': '/fixture/python', 'sig_ign': 0, 'sig_blk': 0}
        self.base = NS(identity=lambda pid: dict(self.holder), uuid_for=lambda gpu: 'fixture-uuid',
                       apps=lambda: [{'pid': 71, 'uuid': 'fixture-uuid'}], check_holder=native.check_holder,
                       lease_snapshot=lambda path: None, headroom=lambda: {'headroom': m.BUDGET},
                       required_budget=native.required_budget, RESTORE_ARGV=native.RESTORE_ARGV)
        self.prepare = NS(base=self.base, subprocess=NS(check_output=lambda *a, **kw: '28.216.19.225'), worker=NS(OUT=Path('/fixture/out')))
        self.prior = {'restored': {'holder_identity': dict(self.holder), 'uuid': 'fixture-uuid'}}

    def capture(self):
        with patch.object(m, 'probe_lock', return_value={'fixture': 1}), \
                patch.object(m.os, 'pidfd_open', return_value=10001), patch.object(m.os, 'close'), \
                patch.object(m.select, 'select', return_value=([], [], [])):
            return m.fresh_current(self.prepare, self.prior)

    def test_true_fresh_after_long_history_has_current_time_and_both_native_probes(self):
        with patch.object(m.time, 'time_ns', return_value=9_000 * 10**9), patch.object(m, 'probe_lock', return_value={'fixture': 1}) as probe, \
                patch.object(m.os, 'pidfd_open', return_value=10001), patch.object(m.os, 'close'), \
                patch.object(m.select, 'select', return_value=([], [], [])):
            value = m.fresh_current(self.prepare, self.prior)
        self.assertEqual(value['captured_unix_ns'], 9_000 * 10**9)
        self.assertEqual(set(value['native_lock_stats']), {str(m.GPU_LOCK), '/fixture/out/reviewer.lock'})
        self.assertEqual(probe.call_count, 4)
        self.assertFalse(value['lease_modified'])
        self.assertEqual(value['holders'][0]['pid'], 71)

    def test_current_PID_start_UUID_unknown_and_normal_TERM_all_default_deny(self):
        for field, changed in [('pid', 72), ('start_ticks', 101), ('sig_ign', 1 << (signal.SIGTERM - 1)),
                               ('sig_blk', 1 << (signal.SIGTERM - 1))]:
            with self.subTest(field=field):
                previous = self.holder[field]
                self.holder[field] = changed
                with self.assertRaises(RuntimeError):
                    self.capture()
                self.holder[field] = previous
        self.base.uuid_for = lambda gpu: 'changed-uuid'
        with self.assertRaises(RuntimeError):
            self.capture()
        self.base.uuid_for = lambda gpu: 'fixture-uuid'
        self.base.apps = lambda: [{'pid': 71, 'uuid': 'fixture-uuid'}, {'pid': 999, 'uuid': 'fixture-uuid'}]
        with self.assertRaisesRegex(RuntimeError, 'nonholder'):
            self.capture()

    def test_unknown_active_lease_changed_ns_low_headroom_and_dead_FD_deny(self):
        self.base.lease_snapshot = lambda path: (_ for _ in ()).throw(RuntimeError('active or unknown lease'))
        with self.assertRaisesRegex(RuntimeError, 'lease'):
            self.capture()
        values = [None, NS(**dict(zip(m.STAT_KEYS, [1, 2, os.getuid(), 0o100600, 0, 3, 4])))]
        self.base.lease_snapshot = lambda path: values.pop(0)
        with self.assertRaisesRegex(RuntimeError, 'lease changed'):
            self.capture()
        self.base.lease_snapshot = lambda path: None
        self.base.headroom = lambda: {'headroom': m.BUDGET - 1}
        with self.assertRaisesRegex(RuntimeError, 'headroom'):
            self.capture()
        self.base.headroom = lambda: {'headroom': m.BUDGET}
        with patch.object(m, 'probe_lock', return_value={}), patch.object(m.os, 'pidfd_open', return_value=10001), \
                patch.object(m.os, 'close'), patch.object(m.select, 'select', return_value=([10001], [], [])):
            with self.assertRaisesRegex(RuntimeError, 'FD exited'):
                m.fresh_current(self.prepare, self.prior)

    def test_admission_sources_and_model_finish_before_fresh_lease_and_identity_recheck(self):
        with tempfile.TemporaryDirectory(prefix='salvage_current_admission_') as tmp:
            rt = Path(tmp)
            old = {'argv': ['/python', '/frozen/worker', '--launch', '/old/plan', '--launch-sha', 'oldsha'],
                   'env': {'CUDA_VISIBLE_DEVICES': '3'}, 'out': '/same/out'}
            plan = {'resume_state_ref': {'path': '/frozen/proof', 'sha256': 'sameproof'}}
            holder = dict(self.holder, gpu=3, uuid='fixture-uuid', state='R')
            fresh = {'holders': [dict(holder, state='S')], 'lease_snapshot_ns': {str(m.LEASE): None},
                     'captured_unix_ns': __import__('time').time_ns()}
            historical = {'holders': [holder], 'lease_snapshot_ns': fresh['lease_snapshot_ns'],
                          'signals_sent': False, 'lease_modified': False, 'captured_unix_ns': 0}
            cfg = dict(old, holders=[holder], source_sha256={}, plan={'path': '/copy/plan', 'sha256': 'sameplan'},
                       holder_capture={'path': '/immutable/history', 'sha256': 'samehistory'},
                       current_resource_restore_ref=m.ref(m.SELF), current_resource_restore_test_ref=m.ref(m.TEST),
                       from_safe_config_ref={'path': '/actual/safe', 'sha256': 'same-safe'}, from_safe_receipt_refs={},
                       resume_state_ref=plan['resume_state_ref'], lease_snapshot_ns=fresh['lease_snapshot_ns'])
            cfg['argv'] = ['/python', '/frozen/worker', '--launch', '/copy/plan', '--launch-sha', 'sameplan']
            seen = []
            def value(reference):
                if reference == m.OLD_CONFIG:
                    return old
                if reference == cfg['plan']:
                    return plan
                if reference == cfg['holder_capture']:
                    return historical
                raise AssertionError(reference)
            def evidence(*args, **kwargs):
                seen.append('complete_proof_sources_local_model')
                self.assertTrue(kwargs['local_model'])
                return old, plan, {}, {'receipt_refs': {}}, {}
            def capture(*args):
                seen.append('actual_fresh')
                return copy.deepcopy(fresh)
            def validate(view):
                seen.append('original_validate_actual_capture')
                self.assertNotEqual(view['holder_capture'], cfg['holder_capture'])
                self.assertEqual(view['holders'][0]['state'], 'S')
            prepare = NS(base=self.base, manager=NS(check_sources=lambda *a: seen.append('original_full_sources'), validate=validate),
                         worker=NS(write_once=lambda path, val: Path(path).write_text(json.dumps(val))))
            with patch.object(m, 'RT', rt), patch.object(m, 'checked', side_effect=value), \
                    patch.object(m, 'evidence', side_effect=evidence), patch.object(m, 'fresh_current', side_effect=capture):
                m.admission(prepare, cfg, '/fixture/config', 'cfgsha', m.BASE_SHA)
                self.assertEqual(seen, ['original_full_sources', 'complete_proof_sources_local_model',
                                        'actual_fresh', 'original_validate_actual_capture'])
                self.assertEqual(historical['captured_unix_ns'], 0)
                self.assertEqual(cfg['holders'][0]['state'], 'R')
                fresh['lease_snapshot_ns'] = {str(m.LEASE): {'st_mtime_ns': 99}}
                with self.assertRaisesRegex(RuntimeError, 'lease changed'):
                    m.admission(prepare, cfg, '/fixture/config', 'cfgsha', m.BASE_SHA)
                prepare.manager.check_sources = lambda *a: (_ for _ in ()).throw(RuntimeError('sourcefail'))
                seen.clear()
                with self.assertRaisesRegex(RuntimeError, 'sourcefail'):
                    m.admission(prepare, cfg, '/fixture/config', 'cfgsha', m.BASE_SHA)
                self.assertEqual(seen, [])

    def test_original_native_20min_TTL_rejects_old_accepts_actual_fresh_without_extension(self):
        import re
        import time
        path = RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
        with tempfile.TemporaryDirectory(prefix='salvage_current_native_validate_') as tmp:
            rt = Path(tmp) / 'runtime'
            rt.mkdir()
            worker = NS(PHASE='fixture-watch', OUT=Path(tmp) / 'out', MODEL='/fixture/model',
                        MODEL_REF='/fixture/modelref', pilot=NS(MANIFEST='/fixture/manifest'))
            holder = dict(self.holder, gpu=3, uuid='fixture-uuid')
            snap = {'node': 'N6', 'gpus': [3], 'holders': [holder], 'signals_sent': False,
                    'captured_unix_ns': 0, 'lease': None}
            worker.checked = lambda reference: snap
            base = NS(**vars(self.base), argv_option=native.argv_option)
            namespace = {'RT': rt, 'worker': worker, 'base': base, 'PHASE': worker.PHASE, 'Path': Path,
                         'QUEUE_PREFIX': 'fixture-', 're': re, 'time': time, 'subprocess': self.prepare.subprocess}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
            launch = {'path': str(rt / 'plan.json'), 'sha256': 'fixture-plan'}
            args = ['/python', str(rt / 'curated_clause_salvage_review_watch_v5.py')]
            for key, value in {'--out': worker.OUT, '--manifest': worker.pilot.MANIFEST, '--launch': launch['path'],
                               '--launch-sha': launch['sha256'], '--model': worker.MODEL, '--model-reference': worker.MODEL_REF,
                               '--annotator-model-reference': rt.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json',
                               '--ms-source': rt / 'source_curated_review_v2'}.items():
                args.extend([key, str(value)])
            args.append('--watch')
            cfg = {'node': 'N6', 'gpus': [3], 'release_authorized': True, 'phase': worker.PHASE, 'semantic_approval': False,
                   'queue': str(rt.parent / 'queues' / 'fixture-123-abcdef123456'), 'out': str(worker.OUT),
                   'env': {'CUDA_VISIBLE_DEVICES': '3'}, 'reuse_existing_results': True, 'holders': [holder],
                   'argv': args, 'plan': launch, 'producer': {'path': str(rt / 'curated_clause_salvage_review_watch_v5.py')},
                   'cwd': str(rt / 'source_curated_review_v2'), 'source_selection': {'shard': 0, 'nshard': 1, 'limit': 0},
                   'holder_capture': {'path': '/fixture/old-capture'}}
            historical = copy.deepcopy(snap)
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                namespace['validate'](cfg)
            fresh = self.capture()
            snap = {**fresh}
            namespace['validate'](cfg)
            self.assertEqual(historical['captured_unix_ns'], 0)
            self.assertEqual(cfg['holder_capture']['path'], '/fixture/old-capture')

    def test_real_CPU_native_FLOCK_busy_and_symlink_denied_without_creation(self):
        with tempfile.TemporaryDirectory(prefix='salvage_current_lock_') as tmp:
            path = Path(tmp) / 'native.lock'
            path.touch()
            before = path.read_bytes(), path.stat().st_mtime_ns
            self.assertIsInstance(m.probe_lock(path), dict)
            with path.open('rb') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(m.flock_owned(os.getpid(), path))
                self.assertTrue(m.fd_owned(os.getpid(), path.stat().st_dev, path.stat().st_ino))
                with self.assertRaises(BlockingIOError):
                    m.probe_lock(path)
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            link = Path(tmp) / 'symlink'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                m.probe_lock(link)
            with self.assertRaises(FileNotFoundError):
                m.probe_lock(Path(tmp) / 'missing')


class NativeReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='salvage_current_release_')
        root = Path(self.tmp.name)
        self.lock = root / 'gpu.lock'
        self.lock.touch()
        self.lease = root / 'lease'
        self.lease.touch()
        self.cfg = {'holders': [{'pid': 41, 'gpu': 3}], 'lease_snapshot_ns': {str(self.lease): None}}
        self.leasefd = os.open(self.lease, os.O_RDONLY)
        s = self.lease.stat()
        self.entry = {'gpu': 3, 'expected': self.cfg['holders'][0], 'reserve': self.lease,
                      'leasefd': self.leasefd, 'lease_inode': (s.st_dev, s.st_ino), 'fd': 10001, 'old': None}
        self.parent = {'pid': os.getpid(), 'argv': ['fixture-exact-SELF']}
        self.base = NS(identity=lambda pid: self.parent, headroom=lambda: {'headroom': m.BUDGET},
                       required_budget=native.required_budget, uuid_for=lambda gpu: 'uuid', apps=lambda: [],
                       check_holder=lambda *a: None)
        self.prepare = NS(base=self.base, worker=NS(OUT=root))

    def tearDown(self):
        os.close(self.leasefd)
        self.tmp.cleanup()

    def call(self, **overrides):
        defaults = {'flock_owned': True, 'fd_owned': True, 'probe_lock': {}, 'select': ([], [], [])}
        defaults.update(overrides)
        with patch.object(m, 'GPU_LOCK', self.lock), patch.object(m, 'LEASE', self.lease), patch.object(m, 'launch_command', return_value=['fixture-exact-SELF']), \
                patch.object(m, 'flock_owned', return_value=defaults['flock_owned']), \
                patch.object(m, 'fd_owned', return_value=defaults['fd_owned']), \
                patch.object(m, 'probe_lock', return_value=defaults['probe_lock']), \
                patch.object(m.select, 'select', return_value=defaults['select']):
            m.before_release(self.prepare, self.cfg, self.entry, '/fixture/config')

    def test_last_moment_FLOCK_FD_lease_headroom_holder_checks(self):
        self.call()
        with self.assertRaisesRegex(RuntimeError, 'FLOCK'):
            self.call(flock_owned=False)
        with self.assertRaisesRegex(RuntimeError, 'lock FD'):
            self.call(fd_owned=False)
        with self.assertRaisesRegex(RuntimeError, 'pidfd'):
            self.call(select=([10001], [], []))
        self.base.headroom = lambda: {'headroom': m.BUDGET - 1}
        with self.assertRaisesRegex(RuntimeError, 'headroom'):
            self.call()
        self.base.headroom = lambda: {'headroom': m.BUDGET}
        self.entry['old'] = self.lease.stat()
        with self.assertRaisesRegex(RuntimeError, 'lease changed after'):
            self.call()
        self.entry['old'] = None
        with self.assertRaisesRegex(RuntimeError, 'integer/ns'):
            m.verify_lease(dict(zip(m.STAT_KEYS, [1, 2, False, 0o100600, 0, 3, 4])))
        self.entry['lease_inode'] = (123, 456)
        with self.assertRaisesRegex(RuntimeError, 'lease FD'):
            self.call()

    def test_native_signals_and_false_completion_logic_remain_original(self):
        text = m.SELF.read_text()
        self.assertIn('native_release(entry)', text)
        self.assertIn('base.main()', text)
        self.assertNotIn('pidfd_send_signal(', text)
        self.assertNotIn('stage_summaries =', text)
        self.assertNotIn('check_sources =', text)
        self.assertNotIn('check_model =', text)
        self.assertNotIn('1200 * 10**9', text)
        self.assertNotIn('require_unreviewed(', text)
        for name, digest in m.PINNED.items():
            self.assertEqual(hashlib.sha256((RT / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
