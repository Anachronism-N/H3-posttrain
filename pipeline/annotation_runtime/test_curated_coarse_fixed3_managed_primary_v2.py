"""CPU-only v2 initializer, identity bridge and beforequeue-safe restoration tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

import curated_coarse_fixed3_managed_primary_v2 as v
m = v.m
spec = importlib.util.spec_from_file_location('frozen_fixed3_v1_tests_for_v2', v.V1_TEST)
original_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_tests)
original_tests.m = m
for name in ('KernelOwnershipTests', 'PairTests', 'ParentTests', 'ModelBudgetTests'):
    globals()[name] = getattr(original_tests, name)
ref = original_tests.ref


class InitializerTests(unittest.TestCase):
    def setup_manager(self, wrong_snapshot=False, fresh_error=False, initial_reject=False):
        self.base = types.SimpleNamespace(validate_pair=Mock(), lease_snapshot=Mock(return_value=None),
                    refresh_leases=Mock(), release_holder=Mock(), acquire_lease=Mock(), main=None,
                    check_holder=Mock(), apps=Mock(return_value=[]))
        self.acquire, self.release = self.base.acquire_lease, self.base.release_holder
        contract = {'jobs': ['job']}
        previous = {'safe': True}
        self.cfg = {'holders': [{'gpu': 0, 'uuid': 'u0'}, {'gpu': 1, 'uuid': 'u1'}],
                    'lease_snapshot_ns': {'/tmp/od_reserve_0': None, '/tmp/od_reserve_1': None},
                    'previous_resources_ref': ref('/previous'), 'holder_capture': ref('/capture'),
                    'primary_preflight_ref': ref('/preflight')}
        capture = {'holders': self.cfg['holders'], 'lease_snapshot_ns': self.cfg['lease_snapshot_ns'],
                   'signals_sent': False, 'captured_unix_ns': time.time_ns()}
        def checked(reference):
            if reference['path'] == '/previous':
                return previous
            if reference['path'] == '/preflight':
                return {'original_admission_ref': ref('/old-admission')}
            return capture
        def fresh(base, prior):
            # Exact previously failing nested callback before base.main creates queue.
            for path in self.cfg['lease_snapshot_ns']:
                base.lease_snapshot(Path(path))
            if fresh_error:
                raise RuntimeError('fresh current resource rejected')
            values = dict(self.cfg['lease_snapshot_ns'])
            if wrong_snapshot:
                values['/tmp/od_reserve_0'] = {'changed': True}
            return {'holders': self.cfg['holders'], 'lease_snapshot_ns': values}
        def main():
            self.base.validate_pair(self.cfg)
        self.base.main = main
        patches = [patch.object(m.c, 'base_module', return_value=self.base),
            patch.object(m, 'private_module', return_value=types.SimpleNamespace(validate=Mock())),
            patch.object(m, 'check_node'), patch.object(m, 'worker_module', return_value=Mock()),
            patch.object(m, 'frozen', return_value=contract), patch.object(m, 'verify_config',
                side_effect=RuntimeError('config gate rejected') if initial_reject else None),
            patch.object(m, 'preflight_binding'), patch.object(m, 'original_config', return_value={'queue': '/old'}),
            patch.object(m, 'restored_evidence', return_value=previous), patch.object(m.c, 'verify_local_model'),
            patch.object(m.c, 'require_unreserved'), patch.object(m.c, 'checked', side_effect=checked),
            patch.object(m.c, 'fresh_pair', side_effect=fresh),
            patch.object(sys, 'argv', [str(v.SELF), '--config', str(m.LAUNCH)])]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_fresh_nested_snapshot_uses_initialized_local_config(self):
        self.setup_manager()
        m.run_manager()
        self.assertEqual(self.base.lease_snapshot.__name__, 'snapshot')
        self.assertEqual(self.base.check_holder.call_count, 2)
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_changed_fresh_snapshot_still_denied_before_acquire(self):
        self.setup_manager(wrong_snapshot=True)
        with self.assertRaisesRegex(RuntimeError, 'nanosecond'):
            m.run_manager()
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_fresh_current_resource_rejection_not_bypassed(self):
        self.setup_manager(fresh_error=True)
        with self.assertRaisesRegex(RuntimeError, 'current resource'):
            m.run_manager()
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_config_gate_still_precedes_initialized_snapshot(self):
        self.setup_manager(initial_reject=True)
        with self.assertRaisesRegex(RuntimeError, 'config gate'):
            m.run_manager()
        m.c.fresh_pair.assert_not_called()
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_wrong_v2_config_path_denied_before_fresh(self):
        self.setup_manager()
        with patch.object(sys, 'argv', [str(v.SELF), '--config', str(v.OLD_LAUNCH)]):
            with self.assertRaisesRegex(RuntimeError, 'config path'):
                m.run_manager()
        m.c.fresh_pair.assert_not_called()


class NativeNestedValidationTests(unittest.TestCase):
    def setup_native(self, missing_path=False, wrong_old_lease=False, invalid_cfg=False):
        self.base = m.c.base_module()
        self.original_fresh = m.c.fresh_pair
        self.events = []
        self.original_snapshot = self.base.lease_snapshot
        self.acquire, self.release = Mock(), Mock()
        self.base.acquire_lease, self.base.release_holder = self.acquire, self.release
        holders = []
        for gpu in (0, 1):
            holder = original_tests.ident(100 + gpu, argv=list(self.base.RESTORE_ARGV),
                        cvd=['CUDA_VISIBLE_DEVICES=%d' % gpu])
            holder.update(gpu=gpu, uuid='uuid-%d' % gpu, sig_ign=0, sig_blk=0)
            holders.append(holder)
        self.cfg = {'node': 'N6', 'gpus': [0, 1], 'phase': m.PHASE, 'targetcount': 3,
                    'argv': ['native-v2-driver'], 'env': {'CUDA_VISIBLE_DEVICES': '0,1'}, 'cwd': str(v.RT),
                    'release_authorized': not invalid_cfg, 'holders': holders,
                    'lease_snapshot_ns': {'/tmp/od_reserve_0': None, '/tmp/od_reserve_1': None},
                    'previous_resources_ref': ref('/previous'), 'holder_capture': ref('/capture'),
                    'primary_preflight_ref': ref('/preflight')}
        if missing_path:
            self.cfg['lease_snapshot_ns']['/tmp/od_reserve_2'] = self.cfg['lease_snapshot_ns'].pop('/tmp/od_reserve_1')
        if wrong_old_lease:
            self.cfg['lease_snapshot_ns']['/tmp/od_reserve_0'] = {k: 1 for k in m.c.LEASE_KEYS}
        previous = {'previous_pairs': [{'restored': {'holder_identity': holder, 'uuid': holder['uuid']}} for holder in holders]}
        capture = {'holders': holders, 'lease_snapshot_ns': self.cfg['lease_snapshot_ns'],
                   'signals_sent': False, 'captured_unix_ns': time.time_ns()}
        def checked(reference):
            if reference['path'] == '/previous':
                return previous
            if reference['path'] == '/preflight':
                return {'original_admission_ref': ref('/old-admission')}
            self.events.append('capture-validated')
            return capture
        def lstat(path):
            # Real original_snapshot must execute, but reads only mocked proc/FS boundary.
            self.events.append('lease-lstat:' + str(path))
            self.assertIn('complete-config-validated', self.events)
            self.assertIn('model-validated', self.events)
            self.assertIn('capture-validated', self.events)
            raise FileNotFoundError(str(path))
        def config_gate(cfg, contract):
            self.assertEqual(cfg['node'], 'N6')
            self.assertEqual(cfg['gpus'], [0, 1])
            self.events.append('complete-config-validated')
        def main():
            self.assertIs(m.c.fresh_pair, self.original_fresh)
            self.base.validate_pair(self.cfg)  # Actual private validate closure.
        self.base.main = main
        fake_stat = types.SimpleNamespace(st_mode=__import__('stat').S_IFREG | 0o600, st_uid=123, st_size=0)
        patches = [patch.object(m.c, 'base_module', return_value=self.base),
            patch.object(m, 'private_module', return_value=types.SimpleNamespace(validate=Mock())),
            patch.object(m, 'check_node'), patch.object(m.c, 'check_node'),
            patch.object(m, 'worker_module', return_value=Mock()), patch.object(m, 'frozen', return_value={'jobs': ['job']}),
            patch.object(m, 'verify_config', side_effect=config_gate), patch.object(m, 'preflight_binding'),
            patch.object(m, 'original_config', return_value={'queue': '/old'}),
            patch.object(m, 'restored_evidence', return_value=previous),
            patch.object(m.c, 'verify_local_model', side_effect=lambda *a: self.events.append('model-validated')),
            patch.object(m.c, 'require_unreserved'), patch.object(m.c, 'checked', side_effect=checked),
            patch.object(self.base, 'identity', side_effect=lambda pid: holders[pid - 100]),
            patch.object(self.base, 'uuid_for', side_effect=lambda gpu: 'uuid-%d' % gpu),
            patch.object(self.base, 'apps', return_value=[{'pid': h['pid'], 'uuid': h['uuid']} for h in holders]),
            patch.object(self.base, 'headroom', return_value={'headroom': 160 * 2**30}),
            patch.object(m.c.os, 'open', return_value=500), patch.object(m.c.os, 'fstat', return_value=fake_stat),
            patch.object(m.c.os, 'close'), patch.object(m.c.os, 'getuid', return_value=123),
            patch.object(m.c.os, 'pidfd_open', return_value=501), patch.object(m.c.fcntl, 'flock'),
            patch.object(m.c.select, 'select', return_value=([], [], [])), patch.object(Path, 'lstat', autospec=True, side_effect=lstat),
            patch.object(sys, 'argv', [str(v.SELF), '--config', str(m.LAUNCH)])]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_real_validate_freshpair_snapshot_order_positive(self):
        self.setup_native()
        m.run_manager()
        # original c.fresh_pair calls both patched original lease snapshots twice.
        leases = [e for e in self.events if e.startswith('lease-lstat:')]
        self.assertEqual(leases, ['lease-lstat:/tmp/od_reserve_0', 'lease-lstat:/tmp/od_reserve_1'] * 2)
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_real_nested_unvalidated_cfg_cannot_reach_snapshot(self):
        self.setup_native(invalid_cfg=True)
        with self.assertRaisesRegex(RuntimeError, 'authorization'):
            m.run_manager()
        self.assertFalse(any(e.startswith('lease-lstat:') for e in self.events))
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_real_nested_missing_exact_leasepath_denied(self):
        self.setup_native(missing_path=True)
        with self.assertRaisesRegex(RuntimeError, 'captures required'):
            m.run_manager()
        self.acquire.assert_not_called()
        self.release.assert_not_called()

    def test_real_nested_original_lease_mismatch_denied(self):
        self.setup_native(wrong_old_lease=True)
        with self.assertRaisesRegex(RuntimeError, 'snapshot changed'):
            m.run_manager()
        self.acquire.assert_not_called()
        self.release.assert_not_called()


class BridgeTests(unittest.TestCase):
    def contract(self):
        proof = json.loads(v.exact_bytes(v.PREFLIGHT, v.PREFLIGHT_SHA))
        contract = {'manifest_ref': proof['jobs_manifest_ref'], 'freeze_ref': proof['source_freeze_ref'],
                    'bindings': dict(proof['source_sha256'])}
        contract['bindings'].update(v.PINNED)
        contract['bindings'].update({str(v.SELF): m.c.sha(v.SELF), str(v.TEST): m.c.sha(v.TEST)})
        return proof, contract

    def test_only_new_execution_identity_bridge_preserves_original_bindings(self):
        proof, contract = self.contract()
        # Isolate TTL with an unchanged original observation time, never save/re-sign.
        with patch.object(m.time, 'time_ns', return_value=proof['captured_unix_ns']):
            result = v.preflight_binding(contract)
        self.assertEqual(result, proof)
        self.assertEqual(len(proof['source_sha256']), 136)
        self.assertEqual(len(contract['bindings']), 140)

    def test_changed_original_data_or_model_binding_denied(self):
        proof, contract = self.contract()
        key = next(k for k in proof['source_sha256'] if k not in (str(v.V1), str(v.V1_TEST)))
        contract['bindings'][key] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, '136'):
            v.preflight_binding(contract)

    def test_original_preflight_expired_remains_denied(self):
        proof, contract = self.contract()
        with patch.object(m.time, 'time_ns', return_value=proof['captured_unix_ns'] + 1201 * 10**9):
            with self.assertRaisesRegex(RuntimeError, 'expired'):
                v.preflight_binding(contract)

    def test_missing_or_modified_code_config_bridge_denied(self):
        with self.assertRaisesRegex(RuntimeError, 'bridge'):
            v.verify_config({}, {})
        bridge = v.execution_bridge()
        changed = copy.deepcopy(bridge)
        changed['v2_adapter_ref']['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'bridge'):
            v.verify_config({'execution_bridge': changed}, {})

    def test_frozen_v1_bytes_and_explicit_patch_cardinality(self):
        self.assertEqual(v.exact_bytes(v.V1, v.V1_SHA), v.V1.read_bytes())
        self.assertEqual(v.exact_bytes(v.V1_TEST, v.V1_TEST_SHA), v.V1_TEST.read_bytes())
        source = v.transformed_source()
        self.assertIn("active.update(cfg=cfg, contract=contract)\n        fresh = c.fresh_pair", source)
        self.assertNotEqual(m.SELF, v.V1)
        self.assertTrue(str(m.LAUNCH).endswith('_v2.json'))


class OriginalSafeRestoreTests(unittest.TestCase):
    def setup_route(self, v2_queue=False, v1_queue=False):
        old = {'queue': '/original-primary-resource'}
        failed = {'node': 'N6', 'gpus': [0, 1], 'global_revision': m.REVISION, 'out': str(m.OUT)}
        proof = {'original_admission_ref': ref('/original-admission')}
        def checked(reference):
            return failed if reference['path'] == str(v.OLD_LAUNCH) else proof
        def exists(path):
            if path == m.QUEUE:
                return v2_queue
            if path == v.OLD_QUEUE:
                return v1_queue
            return False
        patches = [patch.object(v, 'frozen', return_value={'jobs': ['original-job']}),
                   patch.object(v, 'preflight_binding'), patch.object(Path, 'exists', autospec=True, side_effect=exists),
                   patch.object(m.c, 'checked', side_effect=checked), patch.object(m.c, 'ref', side_effect=ref),
                   patch.object(m.c, 'require_unreserved'), patch.object(m, 'original_config', return_value=old),
                   patch.object(m, 'restored_evidence', return_value={'exact_original_safe': True})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_beforequeue_failure_restores_primary_from_exact_original_safe(self):
        self.setup_route()
        result = v.primary_restore_previous(Mock())
        self.assertTrue(result['exact_original_safe'])
        self.assertEqual(m.restored_evidence.call_args.args[1], Path('/original-primary-resource'))
        m.c.require_unreserved.assert_called_once()
        self.assertEqual(v.restore_launch_reference(), {'path': str(v.OLD_LAUNCH), 'sha256': v.OLD_LAUNCH_SHA})

    def test_v2_queue_started_never_falls_back_to_original_safe(self):
        self.setup_route(v2_queue=True)
        v.primary_restore_previous(Mock())
        self.assertEqual(m.restored_evidence.call_args.args[1], m.QUEUE)
        m.restored_evidence.side_effect = RuntimeError('v2 not resources_safe')
        with self.assertRaisesRegex(RuntimeError, 'resources_safe'):
            v.primary_restore_previous(Mock())

    def test_existing_v1_queue_prohibits_original_safe_fallback(self):
        self.setup_route(v1_queue=True)
        with self.assertRaisesRegex(RuntimeError, 'fallback prohibited'):
            v.primary_restore_previous(Mock())
        m.restored_evidence.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
