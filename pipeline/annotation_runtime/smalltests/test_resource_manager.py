import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('bar_resource_manager_tests', Path(__file__).resolve().parents[1] / 'resource_manager.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def expected_holder(gpu=2, pid=100, uuid='test-uuid-2'):
    return dict(gpu=gpu, uuid=uuid, pid=pid, uid=os.getuid(), argv=['/old/python', '-u', str(m.HOLDER)],
                start_ticks=123, cgroup='0::/unit-test', CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)],
                exe='/old/python', sig_ign=0, sig_blk=0)


def config():
    return dict(node='N3', gpus=[2, 3], release_authorized=True,
                holders=[expected_holder(), expected_holder(3, 101, 'test-uuid-3')],
                targetcount=3, phase='pilot3', argv=['/python', '/producer.py'],
                env={'CUDA_VISIBLE_DEVICES': '2,3'}, cwd='/tmp')


class IdentityAndAdmissionTests(unittest.TestCase):
    def test_current_cpu_identity(self):
        value = m.identity(os.getpid())
        self.assertEqual(value['pid'], os.getpid())
        self.assertEqual(value['uid'], os.getuid())
        self.assertGreater(value['start_ticks'], 0)
        self.assertIn('sig_ign', value)

    def test_authorized_pairs_only(self):
        cfg = config()
        m.validate_pair(cfg)
        for node, pair in [('N1', [4, 5]), ('N5', [2, 3]), ('N8', [1, 3]), ('N3', [0, 1])]:
            with self.subTest(node=node, pair=pair), self.assertRaises(RuntimeError):
                m.validate_pair(dict(cfg, node=node, gpus=pair))
        with self.assertRaises(RuntimeError):
            m.validate_pair(dict(cfg, release_authorized=False))
        with self.assertRaises(RuntimeError):
            m.validate_pair(dict(cfg, env={'CUDA_VISIBLE_DEVICES': '0,1'}))

    def test_fresh_budget_two_cold_loads_and_gate(self):
        self.assertEqual(m.required_budget(config()), 160 * 2**30)
        self.assertEqual(m.required_budget(dict(config(), node='N1')), 160 * 2**30)
        self.assertEqual(m.required_budget(dict(config(), node='N1', previous_generation_gate={'queue': 'x'})), 160 * 2**30)

    def test_every_identity_key_and_unique_compute(self):
        expected = expected_holder()
        running = [dict(uuid=expected['uuid'], pid=expected['pid'])]
        m.check_holder(expected, expected.copy(), 2, expected['uuid'], running)
        for key in m.IDENTITY_KEYS:
            changed = expected.copy()
            changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.check_holder(expected, changed, 2, expected['uuid'], running)
        with self.assertRaises(RuntimeError):
            m.check_holder(expected, expected.copy(), 2, 'different-uuid', running)
        with self.assertRaises(RuntimeError):
            m.check_holder(expected, expected.copy(), 2, expected['uuid'], running + [dict(uuid=expected['uuid'], pid=999)])
        for field in ('sig_ign', 'sig_blk'):
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                m.check_holder(expected, dict(expected, **{field: 1 << (signal.SIGTERM - 1)}), 2, expected['uuid'], running)

    def test_release_only_exact_pidfd_and_mark_before_signal(self):
        e = {'released': False, 'fd': 77}
        def check(fd, sig):
            self.assertTrue(e['released'])
            self.assertEqual((fd, sig), (77, signal.SIGTERM))
        with mock.patch.object(m.signal, 'pidfd_send_signal', side_effect=check) as sender:
            m.release_holder(e)
            sender.assert_called_once_with(77, signal.SIGTERM)
        e = {'released': False, 'fd': 77}
        with mock.patch.object(m.signal, 'pidfd_send_signal', side_effect=OSError('ambiguous signal')):
            with self.assertRaises(OSError):
                m.release_holder(e)
        self.assertTrue(e['released'])

    def test_interruption_raises_into_finally(self):
        with self.assertRaises(InterruptedError):
            m.interrupted(signal.SIGTERM, None)


class LeaseTests(unittest.TestCase):
    def test_absent_stale_fresh_nonempty_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lease'
            self.assertIsNone(m.lease_snapshot(path))
            path.touch()
            with self.assertRaises(RuntimeError):
                m.lease_snapshot(path)
            os.utime(path, (time.time() - 1300,) * 2)
            old = m.lease_snapshot(path)
            m.check_lease_unchanged(path, old)
            path.write_text('other-owner-content')
            with self.assertRaises(RuntimeError):
                m.lease_snapshot(path)
            with self.assertRaises(RuntimeError):
                m.check_lease_unchanged(path, old)
            link = Path(tmp) / 'link'
            link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                m.lease_snapshot(link)
            with self.assertRaises(OSError):
                os.open(link, os.O_RDWR | os.O_NOFOLLOW)

    def test_pair_heartbeat_and_replaced_lease_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = []
            for gpu in (2, 3):
                path = Path(tmp) / str(gpu)
                path.touch()
                os.utime(path, (time.time() - 1300,) * 2)
                e = dict(gpu=gpu, reserve=path, old=m.lease_snapshot(path), leasefd=None)
                m.acquire_lease(e)
                entries.append(e)
            try:
                m.refresh_leases(entries)
                self.assertTrue(all(time.time() - e['reserve'].stat().st_mtime < 2 for e in entries))
                moved = Path(tmp) / 'old'
                entries[0]['reserve'].rename(moved)
                entries[0]['reserve'].touch()
                failures = []
                m.refresh_leases(entries, lambda e, err: failures.append((e['gpu'], err)))
                self.assertEqual([gpu for gpu, _ in failures], [2])
                self.assertLess(time.time() - entries[1]['reserve'].stat().st_mtime, 2)
            finally:
                for e in entries:
                    os.close(e['leasefd'])

    def test_atomic_absent_lease_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = dict(gpu=2, reserve=Path(tmp) / 'new', old=None, leasefd=None)
            m.acquire_lease(e)
            try:
                st = e['reserve'].stat()
                self.assertTrue(stat.S_ISREG(st.st_mode))
                self.assertEqual(st.st_size, 0)
                self.assertEqual(e['lease_inode'], (st.st_dev, st.st_ino))
            finally:
                os.close(e['leasefd'])


class RestoreAndReceiptTests(unittest.TestCase):
    def test_local_restore_drops_shared_python_vars(self):
        e = dict(gpu=3, original_env={'PYTHONHOME': 'old', 'PYTHONPATH': 'old', 'CONDA_PREFIX': 'old',
                                    'CUDA_VISIBLE_DEVICES': '3', 'OTHER': 'keep'})
        m.configure_local_restore(e, config())
        self.assertEqual(e['restore_argv'], ['/opt/venv_eval/bin/python', '-u', str(m.HOLDER)])
        self.assertEqual(e['restore_cwd'], '/tmp')
        self.assertEqual(e['restore_env'], {'CUDA_VISIBLE_DEVICES': '3', 'OTHER': 'keep',
                                          'OCCUPY_GPU': '0', 'OMP_NUM_THREADS': '2'})

    def test_unknown_compute_keeps_restore_pending_without_spawn_or_signal(self):
        e = dict(gpu=2, uuid='u2', fd=77, released=True)
        receipts = []
        with mock.patch.object(m.select, 'select', return_value=([77], [], [])), \
             mock.patch.object(m, 'apps', return_value=[dict(uuid='u2', pid=999)]), \
             mock.patch.object(m.subprocess, 'Popen') as spawn, \
             mock.patch.object(m.signal, 'pidfd_send_signal') as sender:
            self.assertFalse(m.restore_step(e, lambda n, v: receipts.append((n, v)), []))
            spawn.assert_not_called()
            sender.assert_not_called()
        self.assertTrue(receipts[0][1]['all_pair_leases_heartbeat'])
        self.assertTrue(receipts[0][1]['lock_and_lease_retained'])

    def test_restore_error_becomes_blocked_no_duplicate_retry(self):
        e = dict(gpu=2, uuid='u2', fd=77, released=True)
        receipts = []
        with mock.patch.object(m.select, 'select', return_value=([77], [], [])), \
             mock.patch.object(m, 'apps', side_effect=OSError('GPU query failed')):
            self.assertFalse(m.restore_step(e, lambda n, v: receipts.append((n, v)), []))
        self.assertEqual(e['restore_state'], 'blocked')
        with mock.patch.object(m, 'apps') as query:
            self.assertFalse(m.restore_step(e, lambda n, v: receipts.append((n, v)), []))
            query.assert_not_called()
        self.assertEqual(len(receipts), 1)

    def test_stdout_brokenpipe_not_resource_failure(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.print', side_effect=BrokenPipeError()):
            m.save_receipt(Path(tmp), 'receipt.json', {'safe': True}, 'pilot')
            self.assertEqual(json.loads((Path(tmp) / 'receipt.json').read_text()), {'safe': True})

    def test_actual_cpu_child_term_defaults_despite_parent_ignore(self):
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        child = None
        fd = None
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            child = subprocess.Popen([sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(60)'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     preexec_fn=m.reset_holder_signals)
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            status = dict(line.split(':', 1) for line in (Path('/proc') / str(child.pid) / 'status').read_text().splitlines() if ':' in line)
            ignored = int(status['SigIgn'].strip(), 16)
            self.assertFalse(ignored & (1 << (signal.SIGTERM - 1)))
            self.assertFalse(ignored & (1 << (signal.SIGINT - 1)))
            fd = os.pidfd_open(child.pid)
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            if child is not None:
                child.stdout.close()
                child.stderr.close()
            if fd is not None:
                os.close(fd)


class StageBindingTests(unittest.TestCase):
    def test_frozen_producer_plan_and_exact_target_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            producer, plan, launch = root / 'producer.py', root / 'plan.jsonl', root / 'launch.json'
            producer.write_text('pass\n')
            plan.write_text('{"id":"a"}\n{"id":"b"}\n{"id":"c"}\n')
            cfg = dict(config(), argv=[sys.executable, str(producer)], source_sha256={},
                       producer={'path': str(producer), 'sha256': m.sha(producer)},
                       plan={'path': str(plan), 'sha256': m.sha(plan)})
            launch.write_text(json.dumps(cfg))
            m.check_sources(cfg, launch, m.sha(launch), m.sha(m.__file__))
            wrong = dict(cfg, targetcount=4)
            launch.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(RuntimeError, 'selected source count'):
                m.check_sources(wrong, launch, m.sha(launch), m.sha(m.__file__))
            producer.write_text('changed\n')
            with self.assertRaisesRegex(RuntimeError, 'source/input changed'):
                m.check_sources(cfg, launch, m.sha(launch), m.sha(m.__file__))

    def test_full_plan_selection_limit_and_sharding_match_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / 'plan.jsonl'
            plan.write_text(''.join(json.dumps({'id': 'src%05d' % i}) + '\n' for i in range(12000)))
            cfg = dict(config(), plan={'path': str(plan)}, input_count=12000,
                       source_selection={'shard': 0, 'nshard': 8, 'limit': 0},
                       argv=[sys.executable, '/producer.py', '--plan', str(plan), '--shard', '0', '--nshard', '8'])
            selected, total = m.selected_sources(cfg)
            self.assertEqual((len(selected), total), (1500, 12000))
            self.assertEqual(selected[:3], ['src00000', 'src00008', 'src00016'])
            cfg['source_selection']['limit'] = 3
            cfg['argv'] += ['--limit', '3']
            self.assertEqual(m.selected_sources(cfg), (['src00000', 'src00008', 'src00016'], 12000))
            cfg['source_selection']['limit'] = 50
            with self.assertRaisesRegex(RuntimeError, 'source_selection'):
                m.selected_sources(cfg)

    def test_summary_is_bound_to_plan_stage_and_selected_ids_not_old_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.jsonl'
            plan.write_text(''.join(json.dumps({'id': 'src%d' % i}) + '\n' for i in range(50)))
            cfg = dict(config(), plan={'path': str(plan), 'sha256': m.sha(plan)}, input_count=50,
                       source_selection={'shard': 0, 'nshard': 1, 'limit': 3},
                       argv=[sys.executable, '/producer.py', '--plan', str(plan), '--stage', 'run', '--limit', '3'])
            old = root / 'snapshots' / 'old'
            new = root / 'snapshots' / 'new'
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            stale = dict(stage='run', plan_sha256=m.sha(plan), total_source_count=50, selected_source_count=50)
            (old / 'summary.json').write_text(json.dumps({'summary': stale}))
            good = dict(stale, selected_source_count=3)
            selected = {'selected_ids': ['src0', 'src1', 'src2']}
            (new / 'summary.json').write_text(json.dumps({'summary': good, 'selection': selected}))
            matched = m.stage_summaries(cfg, root)
            self.assertEqual(list(matched), [str(new / 'summary.json')])
            selected['selected_ids'] = ['src1', 'src2', 'src3']
            (new / 'summary.json').write_text(json.dumps({'summary': good, 'selection': selected}))
            self.assertEqual(m.stage_summaries(cfg, root), {})

    def test_old_generation_in_reused_out_cannot_admit_new_stage(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m, 'EXPERIMENT', Path(tmp)):
            root = Path(tmp)
            queue, out = root / 'queues' / 'first', root / 'first-out'
            queue.mkdir(parents=True)
            out.mkdir()
            load, generation = out / 'load.receipt.json', out / 'generation.json'
            load.write_text('{}')
            generation.write_text(json.dumps({'model_call_performed': True}))
            os.utime(load, (time.time() - 60,) * 2)
            os.utime(generation, (time.time() - 60,) * 2)
            admission = queue / 'admission.json'
            admission.write_text(json.dumps(dict(node='N1', gpus=[0, 1], out=str(out),
                                                model_identity_sha256='model', config_sha256='stage-sha')))
            cfg = dict(config(), node='N1', gpus=[2, 3], modelref={'model_identity_sha256': 'model'},
                       previous_generation_gate={'queue': str(queue), 'out': str(out), 'config_sha256': 'stage-sha'})
            saved = []
            with mock.patch.object(m.time, 'monotonic', side_effect=[0, 3601]):
                with self.assertRaisesRegex(RuntimeError, 'gate timeout'):
                    m.wait_generation_gate(cfg, lambda n, v: saved.append((n, v)))
            self.assertEqual(saved, [])
            stamp = admission.stat().st_mtime_ns + 1000000
            os.utime(load, ns=(stamp, stamp))
            os.utime(generation, ns=(stamp, stamp))
            m.wait_generation_gate(cfg, lambda n, v: saved.append((n, v)))
            self.assertEqual(saved[0][0], 'previous_generation_gate.json')
            bad = dict(cfg, previous_generation_gate=dict(cfg['previous_generation_gate'], config_sha256='other-stage'))
            with self.assertRaisesRegex(RuntimeError, 'config mismatch'):
                m.wait_generation_gate(bad, lambda n, v: None)

    def test_restored_holder_rechecked_until_both_pair_members_safe(self):
        holder = mock.Mock()
        holder.poll.return_value = 1
        e = dict(gpu=2, released=True, restored_holder=holder, restore_state='safe')
        saved = []
        self.assertFalse(m.restore_step(e, lambda n, v: saved.append((n, v)), []))
        self.assertEqual(e['restore_state'], 'blocked')
        self.assertEqual(saved[0][0], 'restore_failure_gpu2.json')



if __name__ == '__main__':
    unittest.main(verbosity=2)
