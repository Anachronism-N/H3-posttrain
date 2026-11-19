"""CPU-only fixed3 admission tests; no model, signals or GPU lease acquisition."""
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import time
import types
import unittest
from unittest.mock import patch, Mock

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed3_managed_cpu_tests', RT / 'curated_coarse_fixed3_managed_primary_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def ref(path):
    return {'path': str(path), 'sha256': 'a' * 64}


def ident(pid, ppid=1, argv=None, cvd=None):
    return dict(pid=pid, uid=123, ppid=ppid, start_ticks=100 + pid,
                state='S', argv=argv or ['native'], cgroup='test-cgroup',
                CVD=cvd or [], exe='/usr/bin/python3.11')


class KernelOwnershipTests(unittest.TestCase):
    def test_real_parent_fd_and_kernel_flock(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'cpu-test.lock'
            with path.open('xb') as stream:
                st = os.fstat(stream.fileno())
                self.assertTrue(m.c.fd_inode_owned(os.getpid(), st.st_dev, st.st_ino))
                self.assertFalse(m.c.proc_lock_owned(os.getpid(), str(path)))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(m.c.proc_lock_owned(os.getpid(), str(path)))
                fcntl.flock(stream, fcntl.LOCK_UN)
                self.assertFalse(m.c.proc_lock_owned(os.getpid(), str(path)))
            self.assertFalse(m.c.fd_inode_owned(os.getpid(), st.st_dev, st.st_ino))

    def test_external_cache_rehashes_replaced_inode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'cpu-input'
            path.write_bytes(b'old')
            with path.open('rb') as original:
                old_inode = os.fstat(original.fileno()).st_ino
                first = m.c.sha(path)
                self.assertEqual(first, m.c.sha(path))
                replacement = Path(root) / 'replacement'
                replacement.write_bytes(b'new')
                os.replace(replacement, path)
                self.assertNotEqual(old_inode, path.stat().st_ino)
                self.assertNotEqual(first, m.c.sha(path))

    def test_private_outputs_always_read_current_bytes(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m, 'OUT', Path(root)):
            path = Path(root) / 'new-result.json'
            path.write_bytes(b'old')
            first = m.c.sha(path)
            path.write_bytes(b'new')
            self.assertNotEqual(first, m.c.sha(path))


class PairTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time_ns()
        self.st = types.SimpleNamespace(st_dev=1, st_ino=2, st_uid=123,
                    st_mode=stat.S_IFREG | 0o600, st_size=0, st_mtime_ns=self.now,
                    st_ctime_ns=self.now)
        self.leases = [{'gpu': gpu, 'uuid': 'uuid-%d' % gpu,
                        'path': '/tmp/od_reserve_%d' % gpu,
                        'stat': m.c.lease_value(self.st)} for gpu in m.PAIR]
        self.base = types.SimpleNamespace(uuid_for=lambda gpu: 'uuid-%d' % gpu)
        self.patches = [patch.object(Path, 'lstat', return_value=self.st),
                        patch.object(m.os, 'getuid', return_value=123),
                        patch.object(m.c, 'fd_inode_owned', return_value=True),
                        patch.object(m.c, 'proc_lock_owned', return_value=True)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def verify(self):
        return m.owned_pair(self.base, 10, self.leases, ['uuid-0', 'uuid-1'])

    def test_current_pair_accepted(self):
        self.verify()

    def test_static_flags_cannot_replace_owned_fd(self):
        m.c.fd_inode_owned.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'FD'):
            self.verify()

    def test_kernel_flock_required(self):
        m.c.proc_lock_owned.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'FLOCK'):
            self.verify()

    def test_replaced_inode_denied(self):
        self.leases[0]['stat']['st_ino'] = 999
        with self.assertRaisesRegex(RuntimeError, 'inode'):
            self.verify()

    def test_expired_and_future_heartbeat_denied(self):
        for delta in (-61 * 10**9, 60 * 10**9):
            self.st.st_mtime_ns = self.now + delta
            with self.assertRaisesRegex(RuntimeError, 'heartbeat'):
                self.verify()

    def test_uuid_swap_and_duplicate_denied(self):
        self.leases[0]['uuid'] = 'uuid-1'
        with self.assertRaises(RuntimeError):
            self.verify()
        with self.assertRaises(RuntimeError):
            m.owned_pair(self.base, 10, self.leases, ['uuid-0', 'uuid-0'])

    def test_lease_symlink_content_owner_denied(self):
        for key, value in [('st_mode', stat.S_IFLNK | 0o600), ('st_size', 1), ('st_uid', 9)]:
            old = getattr(self.st, key)
            setattr(self.st, key, value)
            with self.assertRaises(RuntimeError):
                self.verify()
            setattr(self.st, key, old)


class ParentTests(unittest.TestCase):
    def setUp(self):
        self.parent = ident(10, argv=['manager'])
        self.own = ident(11, ppid=10, argv=['driver'], cvd=['CUDA_VISIBLE_DEVICES=0,1'])
        self.keys = ('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe')
        self.base = types.SimpleNamespace(IDENTITY_KEYS=self.keys,
                    identity=lambda pid: self.parent if pid == 10 else self.own,
                    headroom=lambda: {'headroom': 160 * 2**30}, apps=lambda: [])
        self.cfg = {'source_sha256': {}, 'producer': ref(m.SELF), 'plan': ref('/plan'),
                    'modelref': {'model_identity_sha256': 'model'}, 'argv': ['driver'], 'env': {}}
        self.contract = {'manifest_ref': ref('/manifest'), 'manifest': {'jobs': [{'job_ref': ref('/job%d' % n)} for n in range(3)]},
                         'jobs': [{'refs': {'original_canonical_job': ref('/original%d' % n)}} for n in range(3)]}
        self.admission = {'config_sha256': 'a' * 64, 'manager_sha256': m.BASE_SHA,
                    'source_sha256': {}, 'producer': self.cfg['producer'], 'plan': self.cfg['plan'],
                    'model_identity_sha256': 'model', 'node': 'N6', 'gpus': [0, 1], 'out': str(m.OUT),
                    'targetcount': 3, 'phase': m.PHASE, 'manager_identity': copy.deepcopy(self.parent),
                    'holders': [{'gpu': n, 'uuid': 'uuid-%d' % n} for n in range(2)]}
        self.proof = {'schema': 'fixed3_actual_leases_held_v1', 'resource_admission_ref': ref(m.QUEUE / 'admission.json'),
                      'config_ref': ref(m.LAUNCH), 'node': 'N6', 'gpus': [0, 1], 'manager_identity': copy.deepcopy(self.parent),
                      'leases': [{'path': '/lease%d' % n} for n in range(2)]}
        def checked(reference):
            return self.admission if reference['path'].endswith('admission.json') else self.proof
        self.patches = [patch.object(m, 'check_node'), patch.object(m, 'verify_config'),
                patch.object(m.c, 'ref', side_effect=ref), patch.object(m.c, 'checked', side_effect=checked),
                patch.object(m, 'manager_command', return_value=['manager']), patch.object(m, 'owned_pair'),
                patch.object(m.os, 'getpid', return_value=11), patch.object(m.os, 'getppid', return_value=10),
                patch.object(m.os, 'getuid', return_value=123),
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', FIXED3_N6_CONFIG=str(m.LAUNCH))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def verify(self, tracker=None, compute=False):
        return m.verify_actual_parent(self.cfg, self.base, self.contract, tracker, compute)

    def test_current_parent_binds_three_original_refs_and_model(self):
        admission = self.verify()
        self.assertEqual(len(admission['source_job_refs']), 3)
        self.assertEqual(len(admission['original_source_job_refs']), 3)
        self.assertEqual(admission['model_identity_sha256'], 'model')
        m.owned_pair.assert_called_once()

    def test_pid_reuse_and_changed_exe_cgroup_cvd_denied(self):
        for key, value in [('start_ticks', 999), ('exe', '/other'), ('cgroup', 'other'), ('CVD', ['CUDA_VISIBLE_DEVICES=2,3'])]:
            old = self.parent[key]
            self.parent[key] = value
            with self.assertRaisesRegex(RuntimeError, 'PID'):
                self.verify()
            self.parent[key] = old

    def test_indirect_child_wrong_argv_and_cvd_denied(self):
        for key, value in [('ppid', 9), ('argv', ['different-driver']), ('CVD', [])]:
            old = self.own[key]
            self.own[key] = value
            with self.assertRaisesRegex(RuntimeError, 'direct'):
                self.verify()
            self.own[key] = old

    def test_current_headroom_not_historical_flags(self):
        self.base.headroom = lambda: {'headroom': 159 * 2**30}
        with self.assertRaisesRegex(RuntimeError, 'headroom'):
            self.verify()
        self.base.headroom = lambda: {'headroom': 79 * 2**30}
        with self.assertRaisesRegex(RuntimeError, 'headroom'):
            self.verify(Mock(), True)

    def test_unknown_gpu_app_before_load_denied_and_after_load_tracked(self):
        self.base.apps = lambda: [{'uuid': 'uuid-0', 'pid': 55}]
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            self.verify()
        tracker = Mock()
        self.verify(tracker, True)
        tracker.verify.assert_called_once_with(55)
        tracker.verify.side_effect = RuntimeError('unknown descendant')
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            self.verify(tracker, True)

    def test_other_gpu_apps_are_not_touched(self):
        self.base.apps = lambda: [{'uuid': 'other-gpu', 'pid': 66}]
        tracker = Mock()
        self.verify(tracker, True)
        tracker.verify.assert_not_called()

    def test_admission_model_source_and_revision_scope_fail_closed(self):
        for key, value in [('model_identity_sha256', 'other'), ('source_sha256', {'bad': 'bad'}), ('gpus', [2, 3]), ('out', '/wrong')]:
            old = self.admission[key]
            self.admission[key] = value
            with self.assertRaisesRegex(RuntimeError, 'admission'):
                self.verify()
            self.admission[key] = old


class PreflightTests(unittest.TestCase):
    def test_original_preflight_ttl_not_extended(self):
        contract = {'manifest_ref': ref('/manifest'), 'freeze_ref': ref('/freeze'), 'bindings': {'/source': 'sha'}}
        proof = {'schema': 'fixed3_primary_cpu_preflight_v1', 'original_config_ref': {'path': str(m.OLD), 'sha256': m.OLD_SHA},
                 'jobs_manifest_ref': contract['manifest_ref'], 'source_freeze_ref': contract['freeze_ref'],
                 'source_sha256': contract['bindings'], 'signals_sent': False, 'captured_unix_ns': time.time_ns()}
        with patch.object(m.c, 'ref', side_effect=ref), patch.object(m.c, 'checked', return_value=proof):
            self.assertIs(m.preflight_binding(contract), proof)
            proof['captured_unix_ns'] -= 1201 * 10**9
            with self.assertRaisesRegex(RuntimeError, 'expired'):
                m.preflight_binding(contract)
            proof['captured_unix_ns'] = time.time_ns() + 60 * 10**9
            with self.assertRaisesRegex(RuntimeError, 'expired'):
                m.preflight_binding(contract)

    def test_node_n6_only(self):
        with patch.object(m.subprocess, 'check_output', return_value='28.216.19.137'):
            with self.assertRaisesRegex(RuntimeError, 'N6'):
                m.check_node()


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.old = {'node': 'N6', 'gpus': [0, 1], 'holders': ['old'], 'lease_snapshot_ns': {},
                    'queue': '/original-resource-queue', 'out': '/original-primary-ledger',
                    'global_revision': 'repair-20261025-v1', 'argv': ['watcher', '--watch', '--out', '/original-primary-ledger'],
                    'env': {'CUDA_VISIBLE_DEVICES': '0,1', 'HF_HUB_OFFLINE': '1'}, 'cwd': '/original-cwd',
                    'source_sha256': {'/original-review-queue-source': 'frozen'}}
        self.cfg = copy.deepcopy(self.old)
        self.cfg.update(queue=str(m.RESTORE_QUEUE), holders=['fresh'], fixed3_primary_restore=True,
                        original_primary_config_ref=ref(m.OLD), fixed3_launch_ref=ref(m.LAUNCH))

    def test_restore_preserves_original_worker_not_old_resource_receipt_queue(self):
        with patch.object(m.c, 'ref', side_effect=ref):
            m.verify_restore_config(self.cfg, self.old)
        self.assertEqual(self.cfg['argv'], self.old['argv'])
        self.assertEqual(self.cfg['env'], self.old['env'])
        self.assertNotEqual(self.cfg['queue'], self.old['queue'])

    def test_restore_changed_argv_env_out_revision_old_queue_binding_denied(self):
        with patch.object(m.c, 'ref', side_effect=ref):
            for key, value in [('argv', ['different-watch']), ('env', {}), ('out', '/new-ledger'),
                               ('global_revision', 'new-budget'), ('cwd', '/new-cwd')]:
                changed = copy.deepcopy(self.cfg)
                changed[key] = value
                with self.assertRaisesRegex(RuntimeError, 'preserve'):
                    m.verify_restore_config(changed, self.old)
            self.cfg['source_sha256'].clear()
            with self.assertRaisesRegex(RuntimeError, 'binding'):
                m.verify_restore_config(self.cfg, self.old)

    def test_resources_safe_not_full_semantic_completion(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            queue = Path(root)
            holders = [{'gpu': n, 'uuid': 'uuid-%d' % n, 'identity': {'pid': n + 20}} for n in range(2)]
            admission = queue / 'admission.json'
            admission.write_text(json.dumps({'holders': holders}))
            base = types.SimpleNamespace(RESTORE_ARGV=['normal-holder'])
            for n, holder in enumerate(holders):
                (queue / ('holder_restored_gpu%d.json' % n)).write_text(json.dumps({'gpu': n, 'uuid': holder['uuid'],
                        'original_identity': holder['identity'], 'restore_argv': base.RESTORE_ARGV,
                        'holder_identity': {'pid': n + 30}}))
            safe = {'all_holders_safe': True, 'locks_released': True, 'leases_restored': True,
                    'managed_parent_wait': True, 'unknown_jobs_not_signaled': True}
            (queue / 'resources_safe.json').write_text(json.dumps(safe))
            result = m.restored_evidence(base, queue, m.c.ref(admission))
            self.assertEqual(len(result['previous_pairs']), 2)
            self.assertTrue(result['original_completion_not_claimed'])
            self.assertFalse((queue / 'worker_exit.json').exists())
            safe['locks_released'] = False
            (queue / 'resources_safe.json').write_text(json.dumps(safe))
            with self.assertRaisesRegex(RuntimeError, 'resources_safe'):
                m.restored_evidence(base, queue, m.c.ref(admission))


class ModelBudgetTests(unittest.TestCase):
    def setUp(self):
        self.previews, jobs, refs = {}, [], []
        for n, (sid, count) in enumerate([('mrDP-test', 13), ('Ecd-test', 10), ('FSj-test', 14)]):
            job = {'id': sid, 'source_key': 'train/' + sid, 'target_window_id': 'w%d' % n}
            jobs.append(job)
            refs.append(ref('/job%d' % n))
            self.previews[job['source_key']] = {'messages': [{'role': 'system'}, {'content': [{'type': 'image_url'}] * count}],
                                                'json_schema': {'n': n}, 'sampling': {'max_tokens': 10}}
        self.runtime = types.SimpleNamespace(contract={'manifest': {'jobs': [{'job_ref': r} for r in refs]}, 'jobs': jobs},
                    visual_calls=0, seen_jobs=set(), claims={}, verify_current_admission=Mock(),
                    loaded=(None, object), worker=types.SimpleNamespace(budget_path=lambda job: Path('/budget') / job['id'],
                        bar=types.SimpleNamespace(core=types.SimpleNamespace(structured_sampling=lambda cls, schema, tokens: {'schema': schema, 'tokens': tokens}))))
        for job in jobs:
            claim = ref(self.runtime.worker.budget_path(job))
            self.runtime.claims[claim['path']] = claim
        def load(path):
            return next(value for key, value in self.previews.items() if key in str(path))
        self.model = Mock()
        self.model._ms_bar_model_sha256 = 'model'
        self.wrapper = m.FiniteModel(self.model, self.runtime)
        self.patches = [patch.object(m.c, 'load', side_effect=load), patch.object(m.c, 'ref', side_effect=ref)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def call(self, n=0):
        p = list(self.previews.values())[n]
        return self.wrapper.chat([p['messages']], [{'schema': p['json_schema'], 'tokens': 10}])

    def test_exact_13_10_14_visual_three_total(self):
        for n in range(3):
            self.call(n)
        self.assertEqual(self.runtime.visual_calls, 3)
        self.assertEqual(self.model.chat.call_count, 3)
        self.assertEqual(self.runtime.verify_current_admission.call_count, 3)
        with self.assertRaisesRegex(RuntimeError, 'three total'):
            self.call()

    def test_ambiguous_dispatch_spent_never_retried(self):
        self.model.chat.side_effect = RuntimeError('ambiguous native failure')
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            self.call()
        self.assertEqual(self.runtime.visual_calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            self.call()
        self.assertEqual(self.model.chat.call_count, 1)

    def test_text_and_changed_sampling_never_dispatch(self):
        p = list(self.previews.values())[0]
        with self.assertRaisesRegex(RuntimeError, 'preview'):
            self.wrapper.chat([[{'role': 'user', 'content': 'text-only'}]], [{}])
        with self.assertRaisesRegex(RuntimeError, 'sampling'):
            self.wrapper.chat([p['messages']], [{'schema': p['json_schema'], 'tokens': 99}])
        self.model.chat.assert_not_called()
        self.assertEqual(self.runtime.visual_calls, 0)

    def test_actual_claim_and_current_admission_required(self):
        self.runtime.claims.clear()
        with self.assertRaisesRegex(RuntimeError, 'claim'):
            self.call()
        self.runtime.verify_current_admission.side_effect = RuntimeError('current resource rejected')
        with self.assertRaisesRegex(RuntimeError, 'resource'):
            self.call()
        self.model.chat.assert_not_called()

    def test_current_all_source_changes_abort_before_resource_check(self):
        r = object.__new__(m.FiniteRuntime)
        r.contract = self.runtime.contract
        r.worker = object()
        r.claims = {}
        r.cfg, r.base, r.tracker, r.model_started = {}, object(), object(), False
        refs = [h['job_ref'] for h in r.contract['manifest']['jobs']]
        with patch.object(m, 'frozen', return_value={'changed': 'other source'}), patch.object(m, 'verify_actual_parent') as parent:
            with self.assertRaisesRegex(RuntimeError, 'complete fixed3'):
                r.verify_current_admission(refs)
            parent.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
