"""Tail dispatch CPU checks; old safety fixture, isolated files, no GPUs/models/signals."""
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import ms_native_tail_dispatch_v1 as d
import ms_unified_native_continuous_manager_v2 as v2
import test_ms_native_refill_dispatch_v1 as old


class ExecIntercept(Exception):
    pass


class TailTests(old.RefillTests):
    def setUp(self):
        super().setUp()
        fixture_dispatch = patch.object(old, 'd', d)
        fixture_dispatch.start()
        self.addCleanup(fixture_dispatch.stop)
        self.lockroot = self.root / 'locks'
        self.lockroot.mkdir()
        local_path = patch.object(d, 'Path', side_effect=lambda value: self.lockroot if str(value) == '/tmp' else Path(value))
        local_path.start()
        self.addCleanup(local_path.stop)

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def ref(self, path):
        return dict(path=str(Path(path).resolve()), sha256=self.sha(path))

    def prepare_launch(self, count=783):
        self.safe()
        self.holder = dict(pid=201, start_ticks=2, gpu=0, uuid='fixture-uuid')
        restored = dict(holder_identity={k: self.holder[k] for k in self.base.IDENTITY_KEYS}, uuid=self.holder['uuid'])
        (self.q / 'holder_restored_gpu0.json').write_text(json.dumps(restored))
        self.base.sha = self.sha
        self.base.uuid_for = Mock(return_value=self.holder['uuid'])
        self.base.apps = Mock(return_value=[])
        self.manager.h = types.SimpleNamespace(lock_snapshot=Mock(return_value={'held': False}))
        self.manager.PYTHON = sys.executable
        self.manager.SELF = self.rt / 'ms_unified_native_continuous_manager_v2.py'
        self.manager.SELF.write_text('fixture manager')
        self.manager.CORE = self.rt / 'fixture_core.py'
        self.manager.CORE.write_text('fixture core')
        self.source = self.rt / 'fixture_dependency.py'
        self.source.write_text('fixture dependency')
        self.manager.BASE_SHA = 'fixture-base-sha'
        self.manager.ref = self.ref
        self.manager.dependencies = Mock(return_value={str(self.manager.SELF), str(self.manager.CORE), str(self.source)})
        self.manager.validate = Mock()
        self.manager.check_model = Mock()
        model = self.rt / 'model.json'
        model.write_text('{}')
        ticket = self.rt / 'old_ticket.json'
        ticket.write_text(json.dumps(dict(ticket_id='old_refill1000', batch_sources=20, edition='old', sources=[])))
        self.cfg.update(ticket=self.ref(ticket), modelref=self.ref(model))
        self.previous.write_text(json.dumps(self.cfg))
        self.rows = [dict(id='source_%04d' % i, split='train', video_path='/fixture/source_%04d.mp4' % i,
                          audio_path='/fixture/source_%04d.wav' % i) for i in range(count)]
        self.plan = self.rt / 'tail_plan.jsonl'
        self.plan.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.job.update(phase='ms_unified_native_v1_tail_n3g0_20261027', plan=str(self.plan),
                        plan_sha256=self.sha(self.plan), source_count=count)
        self.jp.write_text(json.dumps(self.job))
        self.registry = self.rt / 'ms_novlm_continuous_jobs_registry_v1.json'
        self.registry.write_text(json.dumps({'jobs': [dict(node='N3', gpu=0, config_path=str(self.previous)),
                                                     dict(node='N2', gpu=2, config_path='untouched')]}))
        self.before_registry = self.registry.read_bytes()
        self.cp = self.rt / (self.job['phase'] + '_config.json')
        self.tp = self.rt / (self.job['phase'] + '_ticket.json')

    def assert_launch_denied(self, regex, exception=RuntimeError):
        with patch.object(d.os, 'execv', side_effect=AssertionError('spawn forbidden')) as execute:
            with self.assertRaisesRegex(exception, regex):
                self.run_main()
            execute.assert_not_called()
        self.assertEqual(self.registry.read_bytes(), self.before_registry)

    def assert_tail_launch(self, count):
        self.prepare_launch(count)
        events = []
        self.manager.validate.side_effect = lambda cfg: events.append('validate')
        self.manager.check_model.side_effect = lambda cfg: events.append('model')
        real_registry_update = d.update_registry
        def handoff(job):
            events.append('registry')
            return real_registry_update(job)
        def intercept(*args):
            events.append('exec')
            raise ExecIntercept()
        with patch.object(d, 'update_registry', side_effect=handoff), patch.object(d.os, 'execv', side_effect=intercept) as execute:
            with self.assertRaises(ExecIntercept):
                self.run_main()
        cfg = json.loads(self.cp.read_text())
        ticket = json.loads(self.tp.read_text())
        self.assertEqual(events, ['validate', 'model', 'registry', 'exec'])
        self.assertEqual(cfg['targetcount'], count)
        self.assertEqual(cfg['input_count'], count)
        self.assertEqual(cfg['selected_source_ids'], [row['id'] for row in self.rows])
        self.assertEqual(ticket['sources'], self.rows)
        self.assertEqual(ticket['batch_sources'], 20)
        self.assertEqual(count % ticket['batch_sources'], count - 780)
        self.assertEqual(ticket['ticket_id'], self.job['phase'])
        self.assertNotIn('refill1000', ticket['ticket_id'])
        self.assertEqual(ticket['edition'], self.job['phase'])
        self.assertEqual(cfg['holders'], [self.holder])
        self.assertEqual(cfg['argv'], [sys.executable, '-B', str(self.manager.SELF), 'worker', '--config', str(self.cp)])
        self.assertEqual(cfg['producer'], self.ref(self.manager.SELF))
        self.assertEqual(cfg['resource_manager_ref'], self.ref(self.manager.SELF))
        refs = [cfg[k] for k in ['ticket', 'plan', 'core_ref', 'modelref', 'resource_previous_safe']] + cfg['holder_restored_refs']
        expected = self.manager.dependencies() | {r['path'] for r in refs}
        self.assertEqual(cfg['source_sha256'], {p: self.sha(p) for p in expected})
        self.base.check_holder.assert_called_once_with(self.holder, {'pid': 101, 'start_ticks': 1}, 0, 'fixture-uuid', [])
        execute.assert_called_once_with(sys.executable, [sys.executable, '-B', str(self.manager.SELF), '--config', str(self.cp),
                                                        '--config-sha', self.sha(self.cp), '--manager-sha', self.manager.BASE_SHA])
        registry = json.loads(self.registry.read_text())
        self.assertEqual(registry['jobs'][0]['config_path'], str(self.cp))
        self.assertEqual(registry['jobs'][0]['task_handle'], 'actualhandle')
        self.assertEqual(registry['jobs'][1], dict(node='N2', gpu=2, config_path='untouched'))

    def test_imports_v2_manager(self):
        self.assertIs(d.manager, v2)
        self.assertEqual(Path(d.manager.SELF).name, 'ms_unified_native_continuous_manager_v2.py')

    def test_tail_783_keeps_20_batch_and_three_source_last_batch(self):
        self.assert_tail_launch(783)

    def test_tail_784_keeps_20_batch_and_four_source_last_batch(self):
        self.assert_tail_launch(784)

    def test_all_three_resources_safe_flags_required(self):
        self.prepare_launch()
        for flag in ['all_holders_safe', 'locks_released', 'leases_restored']:
            with self.subTest(flag=flag):
                self.safe()
                safe_path = self.q / 'resources_safe.json'
                safe = json.loads(safe_path.read_text())
                safe[flag] = False
                safe_path.write_text(json.dumps(safe))
                self.assert_launch_denied('Predecessor not safe')
        self.base.check_holder.assert_not_called()

    def test_finite_completion_exit_and_drain_all_required(self):
        self.prepare_launch()
        cases = [('exit_code', 1), ('status', 'failed'), ('processed', 2),
                 ('allprefetch_drained', False), ('uncommitted_prefetch', 1)]
        for key, value in cases:
            with self.subTest(key=key):
                summary = self.safe()
                if key == 'exit_code':
                    (self.q / 'worker_exit.json').write_text(json.dumps({key: value}))
                else:
                    summary[key] = value
                    (self.out / 'summary.json').write_text(json.dumps(summary))
                self.assert_launch_denied('finish cleanly')
        self.base.check_holder.assert_not_called()

    def test_archive_only_outcome_flags_required(self):
        self.prepare_launch()
        for flag in ['vlm_called', 'quality_approved']:
            with self.subTest(flag=flag):
                summary = self.safe()
                summary[flag] = True
                (self.out / 'summary.json').write_text(json.dumps(summary))
                self.assert_launch_denied('scope differs')
        self.base.check_holder.assert_not_called()

    def test_holder_identity_check_fail_closed(self):
        self.prepare_launch()
        self.base.check_holder.side_effect = PermissionError('exact holder identity differs')
        self.assert_launch_denied('holder identity differs', PermissionError)
        self.manager.validate.assert_not_called()

    def test_gpu_flock_already_owned_fail_closed(self):
        self.prepare_launch()
        self.manager.h.lock_snapshot.return_value = {'held': True}
        self.assert_launch_denied('owned by another job')
        self.manager.validate.assert_not_called()

    def test_plan_sha_change_fail_closed(self):
        self.prepare_launch()
        self.plan.write_text(self.plan.read_text() + ' ')
        self.assert_launch_denied('plan changed')
        self.assertFalse(self.tp.exists())

    def test_source_count_mismatch_fail_closed(self):
        self.prepare_launch()
        self.job['source_count'] = 784
        self.jp.write_text(json.dumps(self.job))
        self.assert_launch_denied('source count differs')
        self.assertFalse(self.tp.exists())

    def test_duplicate_sources_fail_closed(self):
        self.prepare_launch()
        self.rows[-1] = self.rows[0]
        self.plan.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.job['plan_sha256'] = self.sha(self.plan)
        self.jp.write_text(json.dumps(self.job))
        self.assert_launch_denied('source count differs')
        self.assertFalse(self.tp.exists())

    def test_existing_config_ticket_or_queue_never_restarts(self):
        self.prepare_launch()
        queue = self.root / 'queues' / self.job['phase']
        for path in [self.cp, self.tp, queue]:
            with self.subTest(path=path.name):
                if path == queue:
                    path.mkdir()
                else:
                    path.write_text('existing')
                self.assert_launch_denied('already claimed; never restart')
                if path == queue:
                    path.rmdir()
                else:
                    self.assertEqual(path.read_text(), 'existing')
                    path.unlink()
        self.manager.validate.assert_not_called()

    def test_exclusive_write_never_overwrites(self):
        self.prepare_launch()
        self.tp.write_text('existing ticket')
        with self.assertRaises(FileExistsError):
            d.write_new(self.tp, {'changed': True})
        self.assertEqual(self.tp.read_text(), 'existing ticket')

    def test_model_gate_failure_cannot_handoff_or_exec(self):
        self.prepare_launch()
        self.manager.check_model.side_effect = PermissionError('model source changed')
        self.assert_launch_denied('model source changed', PermissionError)
        self.manager.validate.assert_called_once()

    def test_validation_source_sha_failure_cannot_handoff_or_exec(self):
        self.prepare_launch()
        self.manager.validate.side_effect = PermissionError('Frozen reference changed')
        self.assert_launch_denied('Frozen reference changed', PermissionError)
        self.manager.check_model.assert_not_called()

    def test_registry_cas_failure_cannot_exec(self):
        self.prepare_launch()
        self.registry.write_text(json.dumps({'jobs': [dict(node='N3', gpu=0, config_path='other')] }))
        self.before_registry = self.registry.read_bytes()
        self.assert_launch_denied('already handed')
        self.manager.check_model.assert_called_once()

    def test_registry_requires_exactly_one_owned_slot(self):
        self.prepare_launch()
        for jobs in [[], [dict(node='N3', gpu=0, config_path=str(self.previous))] * 2]:
            with self.subTest(count=len(jobs)):
                self.registry.write_text(json.dumps({'jobs': jobs}))
                original = self.registry.read_bytes()
                with patch.object(d, 'RT', self.rt):
                    with self.assertRaisesRegex(RuntimeError, 'exactly one owned slot'):
                        d.update_registry(self.job)
                self.assertEqual(self.registry.read_bytes(), original)

    def test_registry_replace_is_atomic_and_directory_fsynced(self):
        self.prepare_launch()
        real_replace = d.os.replace
        real_fsync = d.os.fsync
        calls = []
        def replace(source, destination):
            self.assertEqual(Path(source).parent, self.rt)
            self.assertEqual(Path(destination), self.registry)
            self.assertEqual(self.registry.read_bytes(), self.before_registry)
            json.loads(Path(source).read_text())
            calls.append('replace')
            return real_replace(source, destination)
        def fsync(fd):
            calls.append('fsync')
            return real_fsync(fd)
        with patch.object(d, 'RT', self.rt), patch.object(d.os, 'replace', side_effect=replace), patch.object(d.os, 'fsync', side_effect=fsync):
            d.update_registry(self.job)
        self.assertEqual(calls, ['fsync', 'replace', 'fsync'])

    def test_wait_has_7200_second_deadline_and_no_spawn(self):
        self.prepare_launch()
        (self.q / 'resources_safe.json').unlink()
        (self.q / 'live.json').write_text(json.dumps({'manager_identity': {'pid': 101, 'start_ticks': 1}}))
        with patch.object(d.time, 'monotonic', side_effect=[0, 7201]), patch.object(d.time, 'sleep') as sleep:
            self.assert_launch_denied('safety wait exceeded bound')
            sleep.assert_not_called()
        self.base.check_holder.assert_not_called()

    def test_single_slot_nonblocking_dispatch_flock(self):
        self.prepare_launch()
        real_flock = d.fcntl.flock
        lock_path = self.lockroot / 'ms-refill-N3-0.lock'
        with lock_path.open('a+b') as held:
            real_flock(held, d.fcntl.LOCK_EX | d.fcntl.LOCK_NB)
            try:
                with patch.object(d.fcntl, 'flock', wraps=real_flock) as flock:
                    self.assert_launch_denied('Resource temporarily unavailable', BlockingIOError)
                    self.assertEqual(flock.call_args.args[1], d.fcntl.LOCK_EX | d.fcntl.LOCK_NB)
            finally:
                real_flock(held, d.fcntl.LOCK_UN)
        self.base.check_holder.assert_not_called()

    def test_unknown_node_gpu_and_host_denied(self):
        self.prepare_launch()
        for changes in [dict(node='UNKNOWN'), dict(gpu=9), dict(host='unknown')]:
            with self.subTest(changes=changes):
                value = dict(self.job, **changes)
                self.jp.write_text(json.dumps(value))
                self.assert_launch_denied('Unowned slot')
        self.base.check_holder.assert_not_called()

    def test_job_must_be_absolute_in_owned_runtime(self):
        self.prepare_launch()
        for job_path in ['job.json', str(self.root / 'job.json')]:
            with self.subTest(job_path=job_path), patch.object(d, 'RT', self.rt), patch.object(sys, 'argv', ['watch', '--job', job_path]):
                with self.assertRaisesRegex(RuntimeError, 'Only root-owned refill job'):
                    d.main()
        self.base.check_holder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
