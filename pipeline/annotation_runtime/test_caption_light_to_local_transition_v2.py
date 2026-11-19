"""CPU-only v4 receipts, five-thread budget, immutable handoff and native checks."""
import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import caption_light_to_local_transition_v2 as t
import test_caption_light_to_local_transition_v1 as old_tests


class RealV4Tests(unittest.TestCase):
    def setUp(self):
        self.old = t.frozen.load(t.OLD_CONFIG)
        self.scope = t.lane_functions(self.old, t.OLD_CONFIG)
        queue = Path(self.old['queue'])
        self.admission = t.frozen.load(queue / 'admission.json')
        self.live = t.frozen.load(queue / 'live.json')

    def test_current_v4_receipts_read_only_no_signal(self):
        with patch.object(t.frozen.signal, 'pidfd_send_signal', autospec=True) as signal, \
             patch.object(t.frozen.os, 'pidfd_open', autospec=True) as opening:
            parent, child = self.scope['check_old_receipts'](
                self.old, t.OLD_CONFIG, self.admission, self.live)
        self.assertEqual(parent['argv'][2], self.old['producer']['path'])
        self.assertEqual(child['argv'], self.old['argv'])
        self.assertEqual(self.scope['LIGHT_MANAGER'], Path(self.old['producer']['path']))
        opening.assert_not_called()
        signal.assert_not_called()

    def test_actual_bindings_include_v4_v1_wrapper_and_local(self):
        bindings = self.scope['code_bindings'](self.old)
        for path in (Path(self.old['producer']['path']), t.WRAPPER, t.frozen.SELF, t.m.SELF):
            self.assertEqual(bindings[str(path)], t.m.base.sha(path))
        self.assertEqual(bindings[str(t.frozen.SELF)], t.FROZEN_SHA)
        self.assertEqual(bindings[str(t.m.SELF)], t.LOCAL_SHA)

    def test_frozen_module_identity_and_globals_unchanged(self):
        self.assertEqual(t.frozen.LIGHT_MANAGER.name, 'caption_two_stage_live_manager_v5.py')
        self.assertEqual(t.frozen.SELF, Path(t.frozen.__file__).resolve())
        self.assertEqual(self.scope['SELF'], t.frozen.SELF)
        self.assertEqual(self.scope['__file__'], t.frozen.__file__)
        self.assertEqual(self.scope['run'].__code__, t.frozen.run.__code__)
        self.assertIsNot(self.scope['run'].__globals__, t.frozen.run.__globals__)
        self.assertEqual(t.m.base.sha(t.m.base.__file__), t.m.BASE_SHA)
        self.assertEqual(t.m.base.sha(t.frozen.SELF), t.FROZEN_SHA)

    def test_wrong_path_sha_version_and_lane_rejected(self):
        for producer in (
            dict(path=str(t.RT / 'unknown_manager.py'), sha256=self.old['producer']['sha256']),
            dict(self.old['producer'], sha256='0' * 64),
            dict(path=str(t.RT / 'caption_two_stage_live_manager_v5.py'),
                 sha256=t.OLD_PRODUCERS[str(t.RT / 'caption_two_stage_live_manager_v5.py')]),
        ):
            old = dict(self.old, producer=producer)
            with self.subTest(producer=producer), self.assertRaises(ValueError):
                t.old_manager(old, t.OLD_CONFIG)
        for key, value in (('gpus', [4, 5]), ('node', 'N3'), ('phase', 'other'),
                           ('queue', str(t.RT)), ('draft_out', str(t.RT))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Wrong fixed'):
                t.old_manager(dict(self.old, **{key: value}), t.OLD_CONFIG)
        with self.assertRaisesRegex(ValueError, 'Wrong fixed'):
            t.old_manager(self.old, t.RT / 'other_launch.json')
        with self.assertRaisesRegex(ValueError, 'fixed new'):
            t.run(t.OLD_CONFIG, 'caption_two_stage_local_other_v1')

    def test_native_receipt_wrong_parent_producer_and_worker_rejected(self):
        for kind in ('parent', 'worker', 'admission', 'resource'):
            old, admission, live = map(copy.deepcopy, (self.old, self.admission, self.live))
            if kind == 'parent':
                admission['manager_identity']['argv'][2] = str(t.frozen.LIGHT_MANAGER)
            elif kind == 'worker':
                live['child_identity']['ppid'] += 1
            elif kind == 'admission':
                admission['producer'] = t.m.ref(t.frozen.LIGHT_MANAGER)
            else:
                old['resource_manager_ref'] = t.m.ref(t.frozen.LIGHT_MANAGER)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.scope['check_old_receipts'](old, t.OLD_CONFIG, admission, live)

    def test_restored_holders_checked_before_quiescent_parent_wait(self):
        self.scope.update(safe_proof=Mock(),
                          restored_owners=Mock(side_effect=ValueError('unsafe restored holders')),
                          current_process=Mock())
        with self.assertRaisesRegex(ValueError, 'unsafe restored holders'):
            self.scope['safe_parent_wait'](self.old, {})
        self.scope['restored_owners'].assert_called_once_with(self.old)
        self.scope['current_process'].assert_not_called()

    def test_budget_gate_precedes_native_park(self):
        with patch.object(t.frozen.shutil, 'disk_usage', autospec=True,
                          return_value=types.SimpleNamespace(free=t.frozen.MIN_DISK)), \
             patch.object(t.m.base, 'headroom', autospec=True,
                          return_value=dict(headroom=t.frozen.MIN_HEADROOM - 1)), \
             patch.object(t.frozen.signal, 'pidfd_send_signal', autospec=True) as signal:
            with self.assertRaisesRegex(ValueError, '160 GiB'):
                self.scope['park_old'](self.old, {}, {}, [], t.RT / 'unused_tree.json')
        signal.assert_not_called()

    def test_writeonce_claim_records_wrapper_and_frozen_base(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(t, 'RT', Path(directory)):
            path = Path(directory) / (t.OLD_PHASE + '_light_to_local_handoff_claim.json')
            self.scope['write_once'](path, dict(transition=t.m.ref(t.frozen.SELF)))
            value = t.frozen.load(path)
            self.assertEqual(value['transition'], t.m.ref(t.WRAPPER))
            self.assertEqual(value['frozen_transition'], t.m.ref(t.frozen.SELF))
            with self.assertRaisesRegex(ValueError, 'Refusing changed existing output'):
                self.scope['write_once'](path, dict(transition=t.m.ref(t.frozen.SELF), changed=True))

    def test_failed_claim_cannot_retry_or_park(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = types.ModuleType('cpu_local_manager')
            manager.__dict__.update(vars(t.m), BAR=root)
            self.scope.update(RT=root, m=manager, check_node=lambda cfg: None,
                              check_old_receipts=lambda *args: ({}, {}),
                              code_bindings=lambda cfg: {}, capacity=lambda: {})
            park = Mock(side_effect=ValueError('Copy concurrency budget changed'))
            self.scope['park_old'] = park
            with self.assertRaisesRegex(ValueError, 'Copy concurrency budget changed'):
                self.scope['run'](t.OLD_CONFIG, t.NEW_PHASE, True)
            claim = root / (t.OLD_PHASE + '_light_to_local_handoff_claim.json')
            failed = root / (t.NEW_PHASE + '_handoff_failed.json')
            self.assertTrue(claim.exists() and failed.exists())
            self.assertTrue(t.frozen.load(failed)['locks_leases_budgets_not_deleted'])
            self.assertTrue(t.frozen.load(failed)['original_config_resume_forbidden'])
            park.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'already claimed'):
                self.scope['run'](t.OLD_CONFIG, t.NEW_PHASE, True)
            park.assert_called_once()

    def test_final_check_rejects_wrong_budget_before_resources(self):
        self.scope.update(check_files=Mock(), old_tree_absent=Mock(), safe_parent_wait=Mock())
        with self.assertRaisesRegex(ValueError, 'Copy concurrency budget'):
            self.scope['final_check']({}, Path('/unused'), [], [], dict(total_copy_threads=4))
        self.scope['safe_parent_wait'].assert_not_called()


class NativeV2Tests(old_tests.NativeInterfaceTests):
    def setUp(self):
        super().setUp()
        for gpu, owner in zip((2, 3), self.owners):
            owner.update(gpu=gpu, uuid='cpu-gpu-' + str(gpu),
                         CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)])
            t.frozen.write_once(self.queue / ('holder_restored_gpu%d.json' % gpu),
                dict(gpu=gpu, uuid=owner['uuid'], holder_identity=owner))
        self.apps.return_value = [dict(uuid=owner['uuid'], pid=owner['pid'], used_MiB='1')
                                 for owner in self.owners]
        self.cfg['gpus'] = [2, 3]
        self.cfg['env']['CUDA_VISIBLE_DEVICES'] = '2,3'

    def test_final_caller_real_v4_receipts_and_native_new_validators(self):
        old = t.frozen.load(t.OLD_CONFIG)
        self.cfg['previous_config'] = t.m.ref(t.OLD_CONFIG)
        self.cfg['source_sha256'][str(t.OLD_CONFIG)] = self.cfg['previous_config']['sha256']
        self.cfg['draft_out'] = str(t.RT.parent / t.NEW_PHASE)
        path = Path(self.cfg['config_path'])
        t.frozen.write_once(path, self.cfg)
        budget = dict(existing_copy_threads=4, added_copy_threads=1, total_copy_threads=5,
            existing_node_copy_threads=2, added_node_copy_threads=1, total_node_copy_threads=3,
            source_batch_size=3, max_inflight_batches=2)
        with patch.object(t.m, 'BAR', t.RT.parent), patch.object(t.frozen, 'RT', t.RT):
            scope = t.lane_functions(old, t.OLD_CONFIG)
            scope.update(capacity=Mock(return_value=budget), safe_parent_wait=Mock())
            original = t.m.base.selected_sources
            scope['final_check'](self.cfg, path, [], [], budget)
            scope['safe_parent_wait'].assert_called_once()
            self.assertIs(t.m.base.selected_sources, original)
            self.assertEqual(self.cfg['producer'], t.m.ref(t.m.SELF))

    def test_real_local_validate_and_real_check_sources_native_identity(self):
        producer = t.m.ref(t.m.SELF)
        self.assertEqual(self.cfg['producer'], producer)
        self.assertEqual(self.cfg['resource_manager_ref'], producer)
        self.assertNotEqual(producer['path'], str(t.WRAPPER))
        path = Path(self.cfg['config_path'])
        t.m.validate(self.cfg)
        t.frozen.write_once(path, self.cfg)
        original = t.m.base.selected_sources
        t.check_native_sources(self.cfg, path)
        self.assertIs(t.m.base.selected_sources, original)
        self.assertEqual(t.m.base.sha(t.m.base.__file__), t.m.BASE_SHA)

    def test_capacity_has_actual_four_thread_fixed_budget(self):
        with patch.object(t.frozen.shutil, 'disk_usage', autospec=True,
                          return_value=types.SimpleNamespace(free=t.frozen.MIN_DISK)), \
             patch.object(t.m.base, 'headroom', autospec=True,
                          return_value=dict(headroom=t.frozen.MIN_HEADROOM)):
            budget = t.capacity()
        t.check_copy_budget(budget)
        self.assertEqual(budget['total_copy_threads'], 5)
        self.assertEqual(budget['total_node_copy_threads'], 3)
        self.assertEqual(budget['added_copy_threads'], 1)
        self.assertEqual(budget['max_inflight_batches'], 2)

    def test_each_copy_budget_field_tamper_rejected(self):
        budget = dict(existing_copy_threads=4, added_copy_threads=1, total_copy_threads=5,
            existing_node_copy_threads=2, added_node_copy_threads=1, total_node_copy_threads=3,
            source_batch_size=3, max_inflight_batches=2)
        for key in budget:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Copy concurrency'):
                t.check_copy_budget(dict(budget, **{key: budget[key] + 1}))

    def test_real_check_sources_wrapper_sha_cannot_replace_base_sha(self):
        path = Path(self.cfg['config_path'])
        t.frozen.write_once(path, self.cfg)
        with self.assertRaisesRegex(RuntimeError, 'manager or launch config SHA'):
            t.m.base.check_sources(self.cfg, str(path), t.m.base.sha(path), t.m.base.sha(t.WRAPPER))


if __name__ == '__main__':
    unittest.main()
