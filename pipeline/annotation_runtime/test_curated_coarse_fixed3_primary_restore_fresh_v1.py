"""CPU fixtures only: no real model load, GPU reserve or signal."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import curated_coarse_fixed3_primary_restore_fresh_v1 as f
m, c = f.m, f.c
spec = importlib.util.spec_from_file_location('fresh_restore_original_cpu_tests', f.v.V1_TEST)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
t.m = m
for name in ('KernelOwnershipTests', 'PairTests', 'ParentTests', 'ModelBudgetTests'):
    globals()[name] = getattr(t, name)


class HistoricalTests(unittest.TestCase):
    def fixture(self):
        bindings = {'/input%d' % i: 'a' * 64 for i in range(136)}
        self.contract = {'bindings': {**bindings, **{p: 'b' * 64 for p in f.v.BRIDGE_KEYS}},
                         'manifest_ref': t.ref('/manifest'), 'freeze_ref': t.ref('/freeze')}
        self.proof = dict(schema='fixed3_primary_cpu_preflight_v1', source_sha256=bindings,
            original_config_ref={'path': str(m.OLD), 'sha256': m.OLD_SHA},
            jobs_manifest_ref=self.contract['manifest_ref'], source_freeze_ref=self.contract['freeze_ref'],
            signals_sent=False, captured_unix_ns=time.time_ns() - 7200 * 10**9)
        patches = [patch.object(f.v, 'execution_bridge'), patch.object(c, 'checked', return_value=self.proof),
                   patch.object(c, 'sha', return_value=f.V2_TEST_SHA),
                   patch.object(m, 'preflight_binding', side_effect=AssertionError('expired authority forbidden'))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_expired_history_contents_kept_without_authorization_reissue(self):
        self.fixture()
        before = copy.deepcopy(self.proof)
        self.assertEqual(f.historical_binding(self.contract), before)
        self.assertEqual(self.proof, before)
        m.preflight_binding.assert_not_called()

    def test_historical_source_difference_rejected(self):
        self.fixture()
        self.contract['bindings']['/input0'] = 'c' * 64
        with self.assertRaisesRegex(RuntimeError, '136-source'):
            f.historical_binding(self.contract)

    def test_missing_original_source_rejected(self):
        self.fixture()
        self.contract['bindings'].pop('/input0')
        with self.assertRaisesRegex(RuntimeError, '136-source'):
            f.historical_binding(self.contract)

    def test_old_queue_safe_never_bypasses_existing_finite_no_safe(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m, 'QUEUE', Path(root)), patch.object(m, 'original_config') as old:
            with self.assertRaisesRegex(RuntimeError, 'finite queue actual resources_safe mandatory'):
                f.previous(Mock())
            old.assert_not_called()

    def test_finite_actual_safe_gate_not_mocked_away(self):
        with tempfile.TemporaryDirectory() as root:
            queue = Path(root)
            admission = queue / 'admission.json'
            admission.write_text(json.dumps({'holders': []}))
            (queue / 'resources_safe.json').write_text(json.dumps({'all_holders_safe': False}))
            with patch.object(m, 'QUEUE', queue), patch.object(c, 'checked', side_effect=lambda ref: json.loads(Path(ref['path']).read_text()) if ref['path'] == str(admission) else {}):
                with self.assertRaisesRegex(RuntimeError, 'actual original managed resources_safe'):
                    f.previous(Mock())


class RestoreConfigTests(unittest.TestCase):
    def fixture(self):
        self.old = {'queue': '/original-review-resource', 'holders': [], 'lease_snapshot_ns': {},
                    'source_sha256': {'/original-source': 'a' * 64}, 'argv': ['frozen-worker', '--watch'],
                    'env': {'CUDA_VISIBLE_DEVICES': '0,1'}, 'cwd': '/original-cwd', 'out': '/original-out',
                    'global_revision': 'original-revision', 'plan': t.ref('/original-plan'), 'producer': t.ref('/worker')}
        self.cfg = copy.deepcopy(self.old)
        self.cfg.update(queue=str(m.RESTORE_QUEUE), fixed3_primary_restore=True,
                        original_primary_config_ref=t.ref(m.OLD), fixed3_launch_ref=t.ref(m.LAUNCH),
                        execution_bridge={'bridge': 'original'}, fresh_resource_authority='actual-current-after-finite-safe-v1',
                        holder_capture=t.ref(f.CAPTURE), previous_resources_ref=t.ref(f.PRIOR))
        self.cfg['source_sha256'].update({str(f.SELF): 'a' * 64, str(f.TEST): 'a' * 64})
        self.capture = {'schema': 'fixed3_primary_actual_current_restore_capture_v1'}
        patches = [patch.object(f.v, 'execution_bridge', return_value={'bridge': 'original'}),
                   patch.object(c, 'ref', side_effect=t.ref), patch.object(c, 'sha', return_value='a' * 64),
                   patch.object(c, 'checked', return_value=self.capture)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_original_primary_config_preserved(self):
        self.fixture()
        f.verify_restore(self.cfg, self.old)

    def test_changed_original_argv_env_cwd_out_revision_or_reviewplan_rejected(self):
        self.fixture()
        for key in ('argv', 'env', 'cwd', 'out', 'global_revision', 'plan', 'producer'):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = 'changed'
                with self.assertRaisesRegex(RuntimeError, 'preserve original worker'):
                    f.verify_restore(cfg, self.old)

    def test_historical_capture_not_actual_current_authority(self):
        self.fixture()
        self.capture['schema'] = 'fixed3_primary_cpu_preflight_v1'
        with self.assertRaisesRegex(RuntimeError, 'actual current capture'):
            f.verify_restore(self.cfg, self.old)


class NativeFreshRestoreTests(unittest.TestCase):
    def fixture(self, expired=False, wronglease=False, unknown=False, badcfg=False):
        self.base = c.base_module()
        self.events = []
        holders = []
        for gpu in (0, 1):
            h = t.ident(100 + gpu, argv=list(self.base.RESTORE_ARGV), cvd=['CUDA_VISIBLE_DEVICES=%d' % gpu])
            h.update(gpu=gpu, uuid='uuid-%d' % gpu, sig_ign=0, sig_blk=0)
            holders.append(h)
        self.cfg = {'node': 'N6', 'gpus': [0, 1], 'phase': 'curated_repair_resident_v2', 'targetcount': 222939,
                    'holders': holders, 'argv': ['original-native-worker'], 'env': {'CUDA_VISIBLE_DEVICES': '0,1'},
                    'cwd': str(f.RT), 'release_authorized': not badcfg, 'fixed3_primary_restore': True,
                    'lease_snapshot_ns': {'/tmp/od_reserve_0': None, '/tmp/od_reserve_1': None},
                    'previous_resources_ref': t.ref('/prior'), 'holder_capture': t.ref('/capture')}
        if wronglease:
            self.cfg['lease_snapshot_ns']['/tmp/od_reserve_0'] = {k: 1 for k in c.LEASE_KEYS}
        previous = {'previous_pairs': [{'restored': {'holder_identity': h, 'uuid': h['uuid']}} for h in holders],
                    'finite_launch_ref': t.ref('/finite')}
        capture = dict(holders=holders, lease_snapshot_ns=self.cfg['lease_snapshot_ns'], signals_sent=False,
                       schema='fixed3_primary_actual_current_restore_capture_v1',
                       captured_unix_ns=time.time_ns() - (7200 * 10**9 if expired else 0))
        def checked(ref):
            return previous if ref['path'] == '/prior' else {'lease_snapshot_ns': {'/tmp/od_reserve_0': None, '/tmp/od_reserve_1': None}} if ref['path'] == '/finite' else capture
        def lstat(path):
            self.events.append('lease:' + str(path))
            self.assertIn('config', self.events)
            self.assertIn('model', self.events)
            raise FileNotFoundError(str(path))
        def repair_validate(cfg):
            self.base_original_validate(cfg)
            self.events.append('config')
        self.base_original_validate = self.base.validate_pair
        self.base.AUTHORIZED_PAIRS = {'N6': ((0, 1),)}
        self.acquire, self.release = Mock(), Mock()
        self.base.acquire_lease, self.base.release_holder = self.acquire, self.release
        self.base.main = lambda: self.base.validate_pair(self.cfg)
        fake_stat = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=123, st_size=0)
        apps = [{'pid': h['pid'], 'uuid': h['uuid']} for h in holders]
        if unknown:
            apps.append({'pid': 999, 'uuid': 'uuid-0'})
        patches = [patch.object(c, 'base_module', return_value=self.base),
            patch.object(m, 'private_module', return_value=types.SimpleNamespace(validate=repair_validate)),
            patch.object(m, 'check_node'), patch.object(c, 'check_node'), patch.object(m, 'worker_module', return_value=Mock()),
            patch.object(m, 'frozen', return_value={'jobs': []}), patch.object(m, 'verify_restore_config'),
            patch.object(m, 'original_config', return_value={'queue': '/original'}),
            patch.object(m, 'primary_restore_previous', return_value=previous),
            patch.object(c, 'verify_local_model', side_effect=lambda *a: self.events.append('model')),
            patch.object(c, 'checked', side_effect=checked), patch.object(c, 'load', return_value=self.cfg),
            patch.object(self.base, 'identity', side_effect=lambda pid: holders[pid - 100]),
            patch.object(self.base, 'uuid_for', side_effect=lambda gpu: 'uuid-%d' % gpu),
            patch.object(self.base, 'apps', return_value=apps), patch.object(self.base, 'headroom', return_value={'headroom': 160 * 2**30}),
            patch.object(c.os, 'open', return_value=500), patch.object(c.os, 'fstat', return_value=fake_stat),
            patch.object(c.os, 'close'), patch.object(c.os, 'getuid', return_value=123),
            patch.object(c.os, 'pidfd_open', return_value=501), patch.object(c.fcntl, 'flock'),
            patch.object(c.select, 'select', return_value=([], [], [])), patch.object(Path, 'lstat', autospec=True, side_effect=lstat),
            patch.object(sys, 'argv', [str(f.SELF), '--config', str(m.RESTORE_LAUNCH)]),
            patch.object(m, 'preflight_binding', side_effect=AssertionError('historical expired authorization invoked'))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_real_base_nested_validate_freshpair_lease_snapshot_positive(self):
        self.fixture()
        self.assertIs(c.fresh_pair, f.fresh_pair)
        f.run_manager()
        self.assertEqual([x for x in self.events if x.startswith('lease:')], ['lease:/tmp/od_reserve_0', 'lease:/tmp/od_reserve_1'] * 2)
        self.acquire.assert_not_called()
        self.release.assert_not_called()
        m.preflight_binding.assert_not_called()

    def test_expired_actual_fresh_capture_rejected_before_lease_read(self):
        self.fixture(expired=True)
        with self.assertRaisesRegex(RuntimeError, 'capture expired'):
            f.run_manager()
        self.assertFalse(any(x.startswith('lease:') for x in self.events))
        self.acquire.assert_not_called()

    def test_changed_exact_integer_ns_lease_rejected_before_acquire(self):
        self.fixture(wronglease=True)
        with self.assertRaisesRegex(RuntimeError, 'lease snapshot changed'):
            f.run_manager()
        self.acquire.assert_not_called()

    def test_unknown_compute_never_signaled(self):
        self.fixture(unknown=True)
        with self.assertRaisesRegex(RuntimeError, 'nonholder compute processes'):
            f.run_manager()
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_invalid_original_authorization_cannot_reach_lease(self):
        self.fixture(badcfg=True)
        with self.assertRaisesRegex(RuntimeError, 'SIGTERM authorization'):
            f.run_manager()
        self.assertFalse(any(x.startswith('lease:') for x in self.events))

    def test_wrong_old_scope_config_path_denied(self):
        self.fixture()
        with patch.object(sys, 'argv', [str(f.SELF), '--config', str(m.LAUNCH)]):
            with self.assertRaisesRegex(RuntimeError, 'independent primary restore config'):
                f.run_manager()
        self.acquire.assert_not_called()


class PartialTransactionTests(unittest.TestCase):
    def fixture(self, badbudget=False, partial=False):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.budget = self.out / 'budget.json'
        self.job = {'source_key': 'source', 'target_window_id': 'w00000', 'original_result_sha256': 'a' * 64,
                    'refs': {'original_canonical_job': t.ref('/original')}}
        self.claim = dict(source_key='source', original_result_sha256='a' * 64, global_revision=m.REVISION,
                          extra_source_attempt=1, maximum_extra_source_attempts=1)
        self.budget.write_text(json.dumps({**self.claim, 'extra_source_attempt': 2} if badbudget else self.claim))
        self.root = self.out / 'revisions' / m.REVISION / 'source'
        self.root.mkdir(parents=True)
        for name, value in [('canonical_budget_claim.json', self.claim), ('scope_job.json', self.job), ('original_job.json', {'original': True})]:
            (self.root / name).write_text(json.dumps(value))
        if partial:
            (self.root / 'partial.dispatch.json').write_text(json.dumps(dict(status='partial_ambiguous_dispatch_spent',
                current_job_ref=t.ref('/scope'), canonical_budget_claim=self.claim, error='ambiguous', semantic_approval=False)))
        self.worker = types.SimpleNamespace(budget_path=lambda job: self.budget, q=types.SimpleNamespace(FLAGS=('semantic_approval',)))
        self.contract = {'jobs': [self.job], 'manifest': {'jobs': [{'job_ref': t.ref('/scope')}]},
                         'consumer': {'modelref': {'model_identity_sha256': 'model-original'}}}
        patches = [patch.object(m, 'OUT', self.out), patch.object(c, 'checked', return_value={'original': True}),
                   patch.object(m, 'verify_completed_transactions')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_original_partial_before_request_safe_for_restore_not_quality(self):
        self.fixture(partial=True)
        refs = f.transactions(self.worker, self.contract)
        self.assertTrue(any(Path(r['path']).name == 'partial.dispatch.json' for r in refs))
        self.assertEqual(json.loads(self.budget.read_text()), self.claim)

    def test_wrong_original_budget_rejected(self):
        self.fixture(badbudget=True)
        with self.assertRaisesRegex(RuntimeError, 'canonical budget changed'):
            f.transactions(self.worker, self.contract)

    def test_changed_scope_job_rejected(self):
        self.fixture()
        (self.root / 'scope_job.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'partial transaction/budget changed'):
            f.transactions(self.worker, self.contract)

    def test_ambiguous_claim_changed_rejected(self):
        self.fixture(partial=True)
        path = self.root / 'partial.dispatch.json'
        value = json.loads(path.read_text())
        value['canonical_budget_claim'] = {}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'ambiguous failure transaction changed'):
            f.transactions(self.worker, self.contract)

    def test_extra_window_attempt_rejected(self):
        self.fixture()
        (self.root / 'windows/w00001/attempt-000002').mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'unexpected source attempt/window'):
            f.transactions(self.worker, self.contract)

    def test_partial_binding_without_request_rejected(self):
        self.fixture()
        attempt = self.root / 'windows/w00000/attempt-000001'
        attempt.mkdir(parents=True)
        (attempt / 'binding.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'lacks original request'):
            f.transactions(self.worker, self.contract)


class RestoredLeaseChainTests(unittest.TestCase):
    def fixture(self):
        self.old = {k: 1 for k in c.LEASE_KEYS}
        self.current = dict(self.old, st_ctime_ns=2)
        self.prior = {'finite_launch_ref': t.ref('/finite')}
        self.fresh = {'lease_snapshot_ns': {'/tmp/od_reserve_0': self.current, '/tmp/od_reserve_1': self.current}}
        self.finite = {'lease_snapshot_ns': {'/tmp/od_reserve_0': self.old, '/tmp/od_reserve_1': self.old}}
        patches = [patch.object(c, 'checked', return_value=self.finite), patch.object(f, '_native_fresh', return_value=self.fresh)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_restored_lease_ctime_may_advance_but_mtime_exactly_original(self):
        self.fixture()
        capture = f.fresh_pair(Mock(), self.prior)
        self.assertEqual(capture['schema'], 'fixed3_primary_actual_current_restore_capture_v1')
        self.assertEqual(self.current['st_mtime_ns'], self.old['st_mtime_ns'])

    def test_changed_restored_mtime_inode_or_type_rejected(self):
        self.fixture()
        for key in ('st_mtime_ns', 'st_ino', 'st_dev'):
            with self.subTest(key=key):
                self.current[key] += 1
                with self.assertRaisesRegex(RuntimeError, 'leases not actually restored'):
                    f.fresh_pair(Mock(), self.prior)
                self.current[key] -= 1

    def test_float_ns_never_accepted(self):
        self.fixture()
        self.current['st_mtime_ns'] = 1.0
        with self.assertRaisesRegex(RuntimeError, 'exact integer nanosecond'):
            f.fresh_pair(Mock(), self.prior)


if __name__ == '__main__':
    unittest.main()
