"""Original budget continuation CPU checks: isolated old fixture, no GPU/model/exec."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import ms_native_original_budget_dispatch_v1 as d
import ms_unified_native_continuous_manager_v2 as v2
import test_ms_native_refill_dispatch_v1 as old
import test_ms_native_tail_dispatch_v1 as tail


class OriginalBudgetTests(tail.TailTests):
    def setUp(self):
        old.RefillTests.setUp(self)
        for module in (old, tail):
            redirect = patch.object(module, 'd', d)
            redirect.start()
            self.addCleanup(redirect.stop)
        self.lockroot = self.root / 'locks'
        self.lockroot.mkdir()
        local_path = patch.object(d, 'Path', side_effect=lambda value: self.lockroot if str(value) == '/tmp' else Path(value))
        local_path.start()
        self.addCleanup(local_path.stop)
        self.job.update(converted_config_path=str(self.rt / 'new_budgetreuse_config.json'),
                        converted_ticket_path=str(self.rt / 'new_budgetreuse_ticket.json'))
        self.jp.write_text(json.dumps(self.job))

    def checked(self, reference):
        path = Path(reference['path'])
        if not path.is_absolute() or str(path.resolve(strict=True)) != str(path):
            raise PermissionError('Stable absolute reference required')
        if self.sha(path) != reference['sha256']:
            raise PermissionError('Frozen reference changed: ' + str(path))
        return json.loads(path.read_text())

    def prepare_launch(self, count=100):
        tail.TailTests.prepare_launch(self, count)
        self.manager.checked = self.checked
        self.cfg.update(node='N3', gpus=[0])
        self.previous.write_text(json.dumps(self.cfg))
        self.previous_ticket = Path(self.cfg['ticket']['path'])
        self.current_ticket = dict(schema='ms_unified_novlm_ticket_v3', authorization='native_parent',
                                   edition='old', ticket_id='remaining_valtest', sources=[], batch_sources=20,
                                   gpu_index=0, max_cache_bytes=8 * 2**30, local_cache_root='/tmp',
                                   beat_checkpoint='/frozen/current/beat.ckpt', scail_root='/frozen/current/scail',
                                   ffmpeg_path='/frozen/current/ffmpeg', ffprobe_path='/frozen/current/ffprobe')
        self.previous_ticket.write_text(json.dumps(self.current_ticket))
        self.cfg['ticket'] = self.ref(self.previous_ticket)
        self.previous.write_text(json.dumps(self.cfg))
        phase = 'ms_unified_native_v1_nvml100_n3g0_20261027'
        self.job['phase'] = phase
        self.cp = self.rt / (phase + '_budgetreuse_config.json')
        self.tp = self.rt / (phase + '_budgetreuse_ticket.json')
        self.original_cp = self.rt / (phase + '_config.json')
        self.original_tp = self.rt / (phase + '_ticket.json')
        self.original_out = self.root / phase
        self.original_queue = self.root / 'queues' / phase
        self.original_claims = self.original_out / phase / 'claims'
        self.original_ticket = dict(schema='ms_unified_novlm_ticket_v2', authorization='native_parent',
                                    edition=phase, ticket_id='n3g0_nvml100', output_root=str(self.original_out),
                                    gpu_index=0, sources=self.rows, beat_checkpoint='/stale/beat.ckpt',
                                    ffmpeg_path='/stale/ffmpeg', local_cache_root='/stale/cache')
        self.original_tp.write_text(json.dumps(self.original_ticket))
        self.original_cfg = dict(schema='ms_unified_native_fleet_config_v1', config_path=str(self.original_cp),
                                 node='N3', gpus=[0], phase=phase, out=str(self.original_out), queue=str(self.original_queue),
                                 input_count=count, targetcount=count, selected_source_ids=[row['id'] for row in self.rows],
                                 ticket=self.ref(self.original_tp), holders=[dict(pid=999, gpu=0, uuid='expired')],
                                 source_sha256={'/expired/old_source.py': 'stale'}, producer={'path': '/expired/old_manager'})
        self.original_cp.write_text(json.dumps(self.original_cfg))
        self.job.update(original_ticket_ref=self.ref(self.original_tp), original_config_ref=self.ref(self.original_cp),
                        converted_config_path=str(self.cp), converted_ticket_path=str(self.tp))
        self.jp.write_text(json.dumps(self.job))
        self.original_cp_bytes = self.original_cp.read_bytes()
        self.original_tp_bytes = self.original_tp.read_bytes()
        self.previous_bytes = self.previous.read_bytes()
        self.previous_ticket_bytes = self.previous_ticket.read_bytes()
        self.plan_bytes = self.plan.read_bytes()

    def refresh_original_refs(self):
        self.original_tp.write_text(json.dumps(self.original_ticket))
        self.original_cfg['ticket'] = self.ref(self.original_tp)
        self.original_cp.write_text(json.dumps(self.original_cfg))
        self.job.update(original_ticket_ref=self.ref(self.original_tp), original_config_ref=self.ref(self.original_cp))
        self.jp.write_text(json.dumps(self.job))

    def launch(self):
        with patch.object(d.os, 'execv', side_effect=tail.ExecIntercept()) as execute:
            with self.assertRaises(tail.ExecIntercept):
                self.run_main()
        return execute

    def assert_tail_launch(self, count):
        self.prepare_launch(count)
        events = []
        self.manager.validate.side_effect = lambda cfg: events.append('validate')
        self.manager.check_model.side_effect = lambda cfg: events.append('model')
        real_update = d.update_registry
        def handoff(job):
            events.append('registry')
            return real_update(job)
        def execute(*args):
            events.append('exec')
            raise tail.ExecIntercept()
        with patch.object(d, 'update_registry', side_effect=handoff), patch.object(d.os, 'execv', side_effect=execute) as intercepted:
            with self.assertRaises(tail.ExecIntercept):
                self.run_main()
        self.assertEqual(events, ['validate', 'model', 'registry', 'exec'])
        cfg = json.loads(self.cp.read_text())
        ticket = json.loads(self.tp.read_text())
        self.assertEqual(cfg['phase'], self.original_cfg['phase'])
        self.assertEqual(cfg['out'], self.original_cfg['out'])
        self.assertEqual(cfg['queue'], self.original_cfg['queue'])
        self.assertEqual(cfg['config_path'], str(self.cp))
        self.assertEqual(cfg['targetcount'], count)
        self.assertEqual(cfg['input_count'], count)
        self.assertEqual(cfg['selected_source_ids'], [row['id'] for row in self.rows])
        self.assertEqual(cfg['holders'], [self.holder])
        self.assertEqual(cfg['modelref'], self.cfg['modelref'])
        self.assertEqual(cfg['argv'], [sys.executable, '-B', str(self.manager.SELF), 'worker', '--config', str(self.cp)])
        self.assertEqual(cfg['producer'], self.ref(self.manager.SELF))
        self.assertEqual(cfg['resource_manager_ref'], self.ref(self.manager.SELF))
        self.assertEqual(cfg['core_ref'], self.ref(self.manager.CORE))
        self.assertEqual(ticket['schema'], 'ms_unified_novlm_ticket_v3')
        self.assertEqual(ticket['batch_sources'], 20)
        self.assertEqual(ticket['sources'], self.original_ticket['sources'])
        for key in ['ticket_id', 'edition', 'output_root']:
            self.assertEqual(ticket[key], self.original_ticket[key])
        for key in ['beat_checkpoint', 'scail_root', 'ffmpeg_path', 'ffprobe_path', 'max_cache_bytes', 'local_cache_root']:
            self.assertEqual(ticket[key], self.current_ticket[key])
        self.assertEqual(Path(ticket['output_root']) / ticket['edition'] / 'claims', self.original_claims)
        self.assertEqual(ticket['source_budget'], dict(schema='ms_unified_original_source_budget_v1',
                         mode='unconsumed_original_namespace', original_ticket_ref=self.job['original_ticket_ref'],
                         original_config_ref=self.job['original_config_ref'], source_count=count,
                         claim_namespace=str(self.original_claims)))
        self.assertEqual(ticket['stop_file'], str(self.original_claims.parent / 'STOP'))
        refs = [cfg[k] for k in ['ticket', 'plan', 'core_ref', 'modelref', 'resource_previous_safe']] + cfg['holder_restored_refs']
        expected = self.manager.dependencies() | {r['path'] for r in refs}
        self.assertEqual(cfg['source_sha256'], {path: self.sha(path) for path in expected})
        self.assertEqual(cfg['plan'], dict(path=str(self.plan), sha256=self.job['plan_sha256']))
        self.base.check_holder.assert_called_once_with(self.holder, {'pid': 101, 'start_ticks': 1}, 0, 'fixture-uuid', [])
        self.assertFalse(self.original_out.exists())
        self.assertFalse(self.original_queue.exists())
        self.assertEqual(self.original_cp.read_bytes(), self.original_cp_bytes)
        self.assertEqual(self.original_tp.read_bytes(), self.original_tp_bytes)
        self.assertEqual(self.previous.read_bytes(), self.previous_bytes)
        self.assertEqual(self.previous_ticket.read_bytes(), self.previous_ticket_bytes)
        self.assertEqual(self.plan.read_bytes(), self.plan_bytes)
        registry = json.loads(self.registry.read_text())
        self.assertEqual(registry['jobs'][0]['config_path'], str(self.cp))
        self.assertNotEqual(registry['jobs'][0]['config_path'], str(self.original_cp))
        self.assertEqual(registry['jobs'][0]['task_handle'], 'actualhandle')
        self.assertEqual(registry['jobs'][1], dict(node='N2', gpu=2, config_path='untouched'))
        intercepted.assert_called_once_with(sys.executable, [sys.executable, '-B', str(self.manager.SELF), '--config', str(self.cp),
                    '--config-sha', self.sha(self.cp), '--manager-sha', self.manager.BASE_SHA])

    def test_imports_v2_manager(self):
        self.assertIs(d.manager, v2)

    def test_original_100_preserves_budget_and_frozen_current_resources(self):
        self.assert_tail_launch(100)

    def test_existing_config_ticket_or_queue_never_restarts(self):
        self.prepare_launch()
        self.cp.write_text('existing')
        self.assert_launch_denied('already claimed; never restart')
        self.assertEqual(self.cp.read_text(), 'existing')
        self.assertFalse(self.tp.exists())

    def test_original_ticket_sha_changed(self):
        self.prepare_launch()
        self.original_tp.write_text(self.original_tp.read_text() + ' ')
        self.assert_launch_denied('Frozen reference changed', PermissionError)
        self.assertFalse(self.tp.exists())

    def test_original_config_sha_changed(self):
        self.prepare_launch()
        self.original_cp.write_text(self.original_cp.read_text() + ' ')
        self.assert_launch_denied('Frozen reference changed', PermissionError)
        self.assertFalse(self.tp.exists())

    def test_original_sources_must_equal_plan_even_same_ids(self):
        self.prepare_launch()
        self.original_ticket['sources'] = [dict(row) for row in self.rows]
        self.original_ticket['sources'][0]['video_path'] = '/changed/source_0000.mp4'
        self.refresh_original_refs()
        self.assert_launch_denied('exactly equal selected plan')

    def test_original_source_order_must_not_change(self):
        self.prepare_launch()
        self.original_ticket['sources'] = list(reversed(self.rows))
        self.refresh_original_refs()
        self.assert_launch_denied('exactly equal selected plan')

    def test_original_config_count_and_source_identity(self):
        self.prepare_launch()
        self.original_cfg['targetcount'] -= 1
        self.refresh_original_refs()
        self.assert_launch_denied('exactly equal selected plan')

    def test_original_split_must_be_train(self):
        self.prepare_launch()
        self.rows[0]['split'] = 'val'
        self.plan.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.job['plan_sha256'] = self.sha(self.plan)
        self.refresh_original_refs()
        self.assert_launch_denied('requires train split')

    def test_original_schema_v2_required(self):
        self.prepare_launch()
        self.original_ticket['schema'] = 'ms_unified_novlm_ticket_v3'
        self.refresh_original_refs()
        self.assert_launch_denied('schema-v2 ticket required')

    def test_original_edition_cannot_move(self):
        self.prepare_launch()
        self.original_ticket['edition'] += '_new'
        self.refresh_original_refs()
        self.assert_launch_denied('namespace differs')

    def test_original_output_or_queue_existence_denies_even_empty(self):
        for key in ['original_out', 'original_queue']:
            with self.subTest(key=key):
                self.prepare_launch()
                path = getattr(self, key)
                path.mkdir()
                self.assert_launch_denied('Original output/queue already exists')
                self.assertTrue(path.is_dir())
                self.assertFalse(self.tp.exists())
                path.rmdir()

    def test_original_ticket_claim_never_retries_or_erases(self):
        self.prepare_launch()
        path = self.original_claims / 'tickets' / (self.original_ticket['ticket_id'] + '.json')
        path.parent.mkdir(parents=True)
        path.write_text('claimed')
        self.assert_launch_denied('already consumed; never retry claims')
        self.assertEqual(path.read_text(), 'claimed')
        self.assertFalse(self.tp.exists())

    def test_original_source_claim_never_retries_or_erases(self):
        self.prepare_launch()
        path = self.original_claims / 'sources' / (self.rows[0]['id'] + '.json')
        path.parent.mkdir(parents=True)
        path.write_text('consumed')
        self.assert_launch_denied('already consumed; never retry claims')
        self.assertEqual(path.read_text(), 'consumed')
        self.assertFalse(self.tp.exists())

    def test_converted_ticket_exists_no_overwrite(self):
        self.prepare_launch()
        self.tp.write_text('existing converted ticket')
        self.assert_launch_denied('already claimed; never restart')
        self.assertEqual(self.tp.read_text(), 'existing converted ticket')

    def test_exclusive_config_write_race_cannot_handoff(self):
        self.prepare_launch()
        original_write = d.write_new
        def race(path, value):
            if Path(path) == self.cp:
                self.cp.write_text('other dispatcher')
            return original_write(path, value)
        with patch.object(d, 'write_new', side_effect=race):
            self.assert_launch_denied('File exists', FileExistsError)
        self.assertEqual(self.cp.read_text(), 'other dispatcher')
        self.assertEqual(self.original_cp.read_bytes(), self.original_cp_bytes)

    def test_converted_path_cannot_overwrite_original_config(self):
        self.prepare_launch()
        self.job['converted_config_path'] = str(self.original_cp)
        self.jp.write_text(json.dumps(self.job))
        self.assert_launch_denied('independent converted paths')
        self.assertEqual(self.original_cp.read_bytes(), self.original_cp_bytes)

    def test_previous_config_must_not_be_original(self):
        self.prepare_launch()
        self.job['previous_config'] = str(self.original_cp)
        with patch.object(d, 'RT', self.rt), patch.object(d, 'B', self.root), patch.object(d, 'manager', self.manager):
            with self.assertRaisesRegex(RuntimeError, 'original config cannot regain slot'):
                d.original_budget(self.job, self.rows)

    def test_registry_current_predecessor_not_original_config(self):
        self.prepare_launch()
        self.registry.write_text(json.dumps({'jobs': [dict(node='N3', gpu=0, config_path=str(self.original_cp))]}))
        self.before_registry = self.registry.read_bytes()
        self.assert_launch_denied('already handed')

    def test_failure_even_with_resources_safe_denied(self):
        self.prepare_launch()
        (self.q / 'failure.json').write_text('{}')
        self.assert_launch_denied('Previous worker failed')
        self.base.check_holder.assert_not_called()

    def test_frozen_current_ticket_checked_no_model_or_old_fallback(self):
        self.prepare_launch()
        self.previous_ticket.write_text(self.previous_ticket.read_text() + ' ')
        self.assert_launch_denied('Frozen reference changed', PermissionError)
        self.assertFalse(self.tp.exists())

    def test_previous_slot_identity_not_reassigned(self):
        self.prepare_launch()
        self.cfg['gpus'] = [2]
        self.previous.write_text(json.dumps(self.cfg))
        self.assert_launch_denied('Current predecessor slot/output identity differs')
        self.assertFalse(self.tp.exists())

    def use_real_native_validation(self, headroom):
        self.prepare_launch()
        native = self.rt / 'resource_manager.py'
        native.write_text('fixture frozen native lifecycle')
        self.base.__file__ = str(native)
        self.base.required_budget = unittest.mock.Mock(return_value=160 * v2.GIB)
        self.base.headroom = unittest.mock.Mock(return_value={'headroom': headroom})
        self.base.lease_snapshot = unittest.mock.Mock()
        self.manager.dependencies.return_value.add(str(native))
        mounts = self.rt / 'fixture_mounts'
        mounts.write_text('fixture / ext4 rw 0 0')
        for name, value in [('RT', self.rt), ('ROOT', self.root), ('SELF', self.manager.SELF),
                            ('CORE', self.manager.CORE), ('base', self.base), ('PYTHON', sys.executable),
                            ('BASE_SHA', self.sha(native)), ('dependencies', self.manager.dependencies)]:
            redirect = patch.object(v2, name, value)
            redirect.start()
            self.addCleanup(redirect.stop)
        for target, name, value in [(v2.h.subprocess, 'check_output', unittest.mock.Mock(return_value=v2.HOSTS['N3'])),
                                    (v2.h, 'lock_snapshot', self.manager.h.lock_snapshot),
                                    (v2.shutil, 'disk_usage', unittest.mock.Mock(return_value=type('Disk', (), {'free': 32 * v2.GIB})())),
                                    (v2, 'Path', unittest.mock.Mock(side_effect=lambda value: mounts if str(value) == '/proc/mounts' else Path(value)))]:
            redirect = patch.object(target, name, value)
            redirect.start()
            self.addCleanup(redirect.stop)
        for key in ['ffprobe_path', 'ffmpeg_path']:
            executable = self.rt / key
            executable.write_text('fixture executable, never run')
            executable.chmod(0o700)
            self.current_ticket[key] = str(executable)
        self.current_ticket['scail_root'] = str(v2.SCAIL)
        self.previous_ticket.write_text(json.dumps(self.current_ticket))
        self.cfg.update(schema=v2.SCHEMA, release_authorized=True, cwd=str(self.rt),
                        env={'CUDA_VISIBLE_DEVICES': '0', 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1',
                             'MKL_NUM_THREADS': '1', 'PYTHONPATH': v2.DEPS}, ticket=self.ref(self.previous_ticket))
        self.previous.write_text(json.dumps(self.cfg))
        safe_path = self.q / 'resources_safe.json'
        safe_path.write_text(json.dumps({key: True for key in ['all_holders_safe', 'locks_released', 'leases_restored',
                                                              'managed_parent_wait', 'unknown_jobs_not_signaled']}))
        self.manager.validate.side_effect = v2.validate

    def test_real_native_v2_validation_accepts_independent_conversion(self):
        self.use_real_native_validation(160 * v2.GIB)
        self.launch()
        cfg = json.loads(self.cp.read_text())
        self.assertEqual(cfg['out'], str(self.original_out))
        self.assertEqual(cfg['config_path'], str(self.cp))
        self.base.required_budget.assert_called_once_with(cfg)
        self.base.lease_snapshot.assert_called_once()
        self.manager.check_model.assert_called_once()
        self.assertFalse(self.original_out.exists())
        self.assertFalse(self.original_queue.exists())

    def test_real_native_160_gib_gate_not_lowered_for_original_budget(self):
        self.use_real_native_validation(160 * v2.GIB - 1)
        self.assert_launch_denied('Original 160 GiB native RAM/cgroup gate not met', PermissionError)
        self.manager.check_model.assert_not_called()
        self.assertEqual(self.original_cp.read_bytes(), self.original_cp_bytes)
        self.assertEqual(self.original_tp.read_bytes(), self.original_tp_bytes)
        self.assertFalse(self.original_out.exists())
        self.assertFalse(self.original_queue.exists())

    def test_source_gate_failure_leaves_old_budget_files_unchanged(self):
        self.prepare_launch()
        self.manager.validate.side_effect = PermissionError('native resource gate blocked')
        self.assert_launch_denied('native resource gate blocked', PermissionError)
        self.assertEqual(self.original_cp.read_bytes(), self.original_cp_bytes)
        self.assertEqual(self.original_tp.read_bytes(), self.original_tp_bytes)
        self.assertFalse(self.original_out.exists())
        self.assertFalse(self.original_queue.exists())


if __name__ == '__main__':
    unittest.main()
