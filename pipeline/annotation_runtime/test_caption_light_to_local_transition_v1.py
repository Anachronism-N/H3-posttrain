"""CPU tests use real local/base validators; only hardware boundaries are mocked."""
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import caption_light_to_local_transition_v1 as t


class SourceIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(id='prepared', split='train'), dict(id='raw', split='val'),
                     dict(id='claimed', split='test'), dict(id='failed', split='train'),
                     dict(id='done', split='val'), dict(id='fresh', split='test')]
        self.plan = self.root / 'plan.json'
        t.write_once(self.plan, self.rows)
        self.old = dict(plan=t.m.ref(self.plan), draft_out=str(self.root / 'drafts'),
                        out=str(self.root / 'wrong_execution'), targetcount=len(self.rows))

    def mark(self, category, split, sid, artifact):
        path = Path(self.old['draft_out']) / category / split / sid / artifact
        t.write_once(path, dict(id=sid, split=split))

    def test_prepared_raw_claim_failure_terminal_all_excluded_one_index(self):
        self.mark('sources', 'train', 'prepared', 'prepared.draft.json')
        self.mark('sources', 'val', 'raw', 'windows/w0/result.json')
        self.mark('sources', 'test', 'claimed', 'windows/w0/claim.json')
        self.mark('drafts', 'train', 'failed', 'draft.json')
        self.mark('drafts', 'val', 'done', 'draft.json')
        with patch.object(t.os, 'scandir', wraps=os.scandir) as scan:
            remaining, excluded, index = t.remaining_sources(self.old)
        self.assertEqual(remaining, [self.rows[-1]])
        self.assertEqual(set(map(tuple, excluded)), {t.row_key(row) for row in self.rows[:-1]})
        self.assertEqual(index['sources']['test'], ['claimed'])
        detail = t.exclusion_details(index)
        self.assertEqual(set(map(tuple, detail['excluded_started_without_draft'])),
                         {('prepared', 'train'), ('raw', 'val'), ('claimed', 'test')})
        self.assertEqual(set(map(tuple, detail['excluded_draft_keys'])), {('failed', 'train'), ('done', 'val')})
        self.assertTrue(detail['budget_not_reset'])
        paths = [str(call.args[0]) for call in scan.call_args_list]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(self.old['out'] not in path for path in paths))
        self.assertEqual(t.load(self.plan), self.rows)

    def test_empty_started_directory_is_excluded(self):
        (Path(self.old['draft_out']) / 'sources/train/prepared').mkdir(parents=True)
        remaining, excluded, _ = t.remaining_sources(self.old)
        self.assertNotIn(self.rows[0], remaining)
        self.assertIn(('prepared', 'train'), excluded)

    def test_unknown_source_rejected(self):
        self.mark('sources', 'train', 'unknown', 'claim.json')
        with self.assertRaisesRegex(ValueError, 'Unknown source'):
            t.remaining_sources(self.old)

    def test_unknown_split_rejected(self):
        self.mark('drafts', 'unknown', 'fresh', 'draft.json')
        with self.assertRaisesRegex(ValueError, 'Unknown source split'):
            t.remaining_sources(self.old)

    def test_symlink_source_rejected(self):
        directory = Path(self.old['draft_out']) / 'sources/train'
        directory.mkdir(parents=True)
        (directory / 'prepared').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unsafe source'):
            t.remaining_sources(self.old)


class ProcessSafetyTests(unittest.TestCase):
    def test_real_current_cpu_process_refuses_old_alive(self):
        current = t.m.base.identity(os.getpid())
        with self.assertRaisesRegex(ValueError, 'still alive'):
            t.old_tree_absent([current])

    def test_same_pid_changed_identity_not_mistaken_for_absent(self):
        current = t.m.base.identity(os.getpid())
        original = dict(current, argv=['changed'])
        with self.assertRaisesRegex(ValueError, 'still alive'):
            t.old_tree_absent([original])

    def test_pid_reuse_new_start_is_absent(self):
        current = t.m.base.identity(os.getpid())
        self.assertTrue(t.old_tree_absent([dict(current, start_ticks=current['start_ticks'] - 1)]))

    def test_incomplete_safe_refuses_each_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in t.SAFE_KEYS:
                path = Path(directory) / (key + '.json')
                t.write_once(path, dict.fromkeys(t.SAFE_KEYS, True) | {key: False})
                with self.assertRaisesRegex(ValueError, 'Incomplete native safe'):
                    t.safe_proof(path)
            missing = Path(directory) / 'missing.json'
            t.write_once(missing, dict(all_holders_safe=True))
            with self.assertRaisesRegex(ValueError, 'Incomplete native safe'):
                t.safe_proof(missing)

    def test_already_safe_but_real_old_alive_no_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            t.write_once(queue / 'resources_safe.json', dict.fromkeys(t.SAFE_KEYS, True))
            current = t.m.base.identity(os.getpid())
            with patch.object(t, 'term_exact', autospec=True) as term:
                with self.assertRaisesRegex(ValueError, 'still alive'):
                    t.park_old(dict(queue=str(queue)), current, current, [], queue / 'tree.json')
            term.assert_not_called()

    def test_proof_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            t.write_once(path, dict(original=True))
            proof = t.file_identity(path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after capture'):
                t.check_files([proof])

    def test_native_safe_real_parent_wait_allows_reclaimed_child(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            t.write_once(queue / 'resources_safe.json', dict.fromkeys(t.SAFE_KEYS, True))
            # A real CPU parent waits on its normally exiting child; no real signal is sent.
            code = "import subprocess,sys; c=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(1)']); print('ready',flush=True); c.wait()"
            parent_process = subprocess.Popen([sys.executable, '-B', '-c', code], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(parent_process.stdout.readline().strip(), 'ready')
                parent = t.m.base.identity(parent_process.pid)
                deadline = time.monotonic() + 0.5
                while (Path('/proc') / str(parent_process.pid) / 'wchan').read_text().strip() != 'do_wait':
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.005)
                parent = t.m.base.identity(parent_process.pid)
                old_child = dict(t.m.base.identity(os.getpid()), start_ticks=0)
                old = dict(queue=str(queue), gpus=[])
                with patch.object(t.m.base, 'apps', autospec=True, return_value=[]), \
                     patch.object(t, 'term_exact', autospec=True) as signaling:
                    expected, safe = t.park_old(old, parent, old_child, [], queue / 'tree.json')
                self.assertEqual(expected, [old_child])
                self.assertEqual(safe, t.m.ref(queue / 'resources_safe.json'))
                self.assertIsNone(parent_process.poll())
                signaling.assert_not_called()
            finally:
                parent_process.wait(timeout=5)
                parent_process.stdout.close()

    def test_safe_changed_parent_refused_without_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            t.write_once(queue / 'resources_safe.json', dict.fromkeys(t.SAFE_KEYS, True))
            parent = t.m.base.identity(os.getpid())
            changed_expected = dict(parent, argv=['wrong old parent'])
            with patch.object(t, 'term_exact', autospec=True) as signaling, \
                 patch.object(t.m.base, 'apps', autospec=True, return_value=[]):
                with self.assertRaisesRegex(ValueError, 'identity/PPID changed'):
                    t.safe_parent_wait(dict(queue=str(queue), gpus=[]), changed_expected)
            signaling.assert_not_called()

    def test_expired_deadline_opens_no_pidfd(self):
        with patch.object(t.os, 'pidfd_open', autospec=True) as opening:
            with self.assertRaisesRegex(ValueError, 'deadline expired'):
                t.term_exact(t.m.base.identity(os.getpid()), 0)
        opening.assert_not_called()

    def test_pidfd_identity_change_refuses_signal_and_closes_fd(self):
        current = t.m.base.identity(os.getpid())
        with patch.object(t.os, 'pidfd_open', autospec=True, return_value=123) as opening, \
             patch.object(t.os, 'close', autospec=True) as closing, \
             patch.object(t.signal, 'pidfd_send_signal', autospec=True) as signaling:
            with self.assertRaisesRegex(ValueError, 'identity/PPID changed'):
                t.term_exact(dict(current, start_ticks=current['start_ticks'] - 1), t.time.monotonic() + 1)
        opening.assert_called_once_with(current['pid'])
        closing.assert_called_once_with(123)
        signaling.assert_not_called()

    def test_pidfd_only_term_no_kill(self):
        current = t.m.base.identity(os.getpid())
        current['sig_blk'] = current['sig_ign'] = 0
        with patch.object(t.os, 'pidfd_open', autospec=True, return_value=123), \
             patch.object(t.os, 'close', autospec=True), \
             patch.object(t.m.base, 'identity', autospec=True, return_value=current), \
             patch.object(t.signal, 'pidfd_send_signal', autospec=True) as signaling:
            t.term_exact(current, t.time.monotonic() + 1)
        signaling.assert_called_once_with(123, t.signal.SIGTERM)

    def test_shared_transition_lock_excludes_different_new_names(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(t, 'RT', Path(directory)):
            with t.handoff_lock('old_shared'):
                with self.assertRaises(BlockingIOError):
                    with t.handoff_lock('old_shared'):
                        self.fail('Concurrent handoff entered')
            self.assertTrue((Path(directory) / 'old_shared_light_to_local_handoff.lock').exists())


class NativeInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rt = self.root / 'runtime'
        self.rt.mkdir()
        self.queue = self.root / 'queues/old'
        self.queue.mkdir(parents=True)
        self.safe = self.queue / 'resources_safe.json'
        t.write_once(self.safe, dict.fromkeys(t.SAFE_KEYS, True))
        self.rows = [dict(id='fresh', split='train')]
        self.plan = self.rt / 'remaining.json'
        t.write_once(self.plan, self.rows)
        self.handoff = self.rt / 'handoff.json'
        t.write_once(self.handoff, dict(remaining_sources=self.rows, old_worker_tree_absent=True,
                                      original_config_resume_forbidden=True))
        self.old_path = self.rt / 'old.json'
        self.old = dict(node='N4', gpus=[4, 5], queue=str(self.queue), draft_out=str(self.root / 'old'),
                        source_root=str(self.root / 'source'), modelref=dict(model_identity_sha256='model'),
                        env=dict(CUDA_VISIBLE_DEVICES='4,5', HF_HUB_OFFLINE='1'))
        t.write_once(self.old_path, self.old)
        self.owners = []
        for gpu in (4, 5):
            item = dict(pid=800000 + gpu, uid=os.getuid(), ppid=1, argv=list(t.m.base.RESTORE_ARGV),
                        start_ticks=1000 + gpu, cgroup='test-cpu', CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)],
                        exe='/cpu/python', sig_ign=0, sig_blk=0, gpu=gpu, uuid='cpu-gpu-' + str(gpu))
            self.owners.append(item)
            t.write_once(self.queue / ('holder_restored_gpu%d.json' % gpu),
                         dict(gpu=gpu, uuid=item['uuid'], holder_identity=item))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(t, 'RT', self.rt))
        self.stack.enter_context(patch.object(t.m, 'BAR', self.root))
        self.stack.enter_context(patch.object(t.m.h.subprocess, 'check_output', autospec=True,
                                             return_value=t.m.HOSTS['N4']))
        self.identity = self.stack.enter_context(patch.object(t.m.base, 'identity', autospec=True,
            side_effect=lambda pid: next(copy.deepcopy(owner) for owner in self.owners if owner['pid'] == pid)))
        self.stack.enter_context(patch.object(t.m.base, 'uuid_for', autospec=True,
                                             side_effect=lambda gpu: 'cpu-gpu-' + str(gpu)))
        self.apps = self.stack.enter_context(patch.object(t.m.base, 'apps', autospec=True,
            return_value=[dict(uuid=owner['uuid'], pid=owner['pid'], used_MiB='1') for owner in self.owners]))
        self.lock = self.stack.enter_context(patch.object(t.m.h, 'lock_snapshot', autospec=True,
                                                         return_value=dict(exists=True, held=[])))
        self.cfg = t.build_config(self.old, self.old_path, 'caption_two_stage_local_cpu_v1', self.owners,
                                 self.plan, self.handoff, t.m.ref(self.safe), {str(t.SELF): t.m.base.sha(t.SELF)})

    def test_real_local_validate_and_real_check_sources_native_identity(self):
        producer = t.m.ref(t.m.SELF)
        self.assertEqual(self.cfg['producer'], producer)
        self.assertEqual(self.cfg['resource_manager_ref'], producer)
        self.assertNotEqual(self.cfg['producer']['path'], str(t.SELF))
        path = Path(self.cfg['config_path'])
        self.assertEqual(self.cfg['argv'], [t.PYTHON, '-B', str(t.m.SELF), 'worker', '--config', str(path)])
        self.assertEqual(t.manager_argv(path, t.m.SELF, 'digest')[-1], t.m.BASE_SHA)
        self.assertEqual(self.cfg['source_root'], self.old['source_root'])
        self.assertEqual(self.cfg['modelref'], self.old['modelref'])
        t.m.validate(self.cfg)
        t.write_once(path, self.cfg)
        original = t.m.base.selected_sources
        try:
            t.m.base.selected_sources = t.m.selected
            t.m.base.check_sources(self.cfg, str(path), t.m.base.sha(path), t.m.BASE_SHA)
        finally:
            t.m.base.selected_sources = original

    def test_transition_as_producer_is_rejected_by_real_validate(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['producer'] = cfg['resource_manager_ref'] = t.m.ref(t.SELF)
        with self.assertRaisesRegex(ValueError, 'Separate native worker identity'):
            t.m.validate(cfg)

    def test_fresh_holder_identity_changed_real_validate_refuses(self):
        self.identity.side_effect = lambda pid: dict(next(owner for owner in self.owners if owner['pid'] == pid), start_ticks=0)
        with self.assertRaisesRegex(RuntimeError, 'holder identity changed'):
            t.m.validate(self.cfg)

    def test_additional_gpu_app_real_check_holder_refuses(self):
        self.apps.return_value.append(dict(uuid=self.owners[0]['uuid'], pid=987654, used_MiB='1'))
        with self.assertRaisesRegex(RuntimeError, 'nonholder compute processes'):
            t.m.validate(self.cfg)

    def test_held_native_gpu_lock_refuses(self):
        self.lock.return_value = dict(held=['kernel owner'])
        with self.assertRaisesRegex(ValueError, 'Active GPU lock'):
            t.m.validate(self.cfg)

    def test_safe_incomplete_real_validate_refuses(self):
        self.safe.write_bytes(t.m.local.light.encoded(dict.fromkeys(t.SAFE_KEYS, True) | {'leases_restored': False}))
        self.cfg['resource_previous_safe'] = t.m.ref(self.safe)
        with self.assertRaisesRegex(ValueError, 'predecessor resources not safe'):
            t.m.validate(self.cfg)

    def test_no_handoff_or_unallowed_plan_real_validate_refuses(self):
        self.handoff.write_bytes(t.m.local.light.encoded(dict(remaining_sources=[], old_worker_tree_absent=True,
                                                           original_config_resume_forbidden=True)))
        self.cfg['source_handoff'] = t.m.ref(self.handoff)
        with self.assertRaisesRegex(ValueError, 'outside handed-off'):
            t.m.validate(self.cfg)

    def test_capacity_has_actual_four_thread_fixed_budget(self):
        with patch.object(t.shutil, 'disk_usage', autospec=True, return_value=type('Disk', (), {'free': t.MIN_DISK})()), \
             patch.object(t.m.base, 'headroom', autospec=True, return_value=dict(headroom=t.MIN_HEADROOM)):
            result = t.capacity()
        self.assertEqual(result['total_copy_threads'], 4)
        self.assertEqual(result['max_inflight_batches'], 2)
        self.assertEqual(result['source_batch_size'], 3)

    def test_low_disk_and_headroom_refuse(self):
        with patch.object(t.shutil, 'disk_usage', autospec=True, return_value=type('Disk', (), {'free': t.MIN_DISK - 1})()):
            with self.assertRaisesRegex(ValueError, '8 GiB'):
                t.capacity()
        with patch.object(t.shutil, 'disk_usage', autospec=True, return_value=type('Disk', (), {'free': t.MIN_DISK})()), \
             patch.object(t.m.base, 'headroom', autospec=True, return_value=dict(headroom=t.MIN_HEADROOM - 1)):
            with self.assertRaisesRegex(ValueError, '160 GiB'):
                t.capacity()


if __name__ == '__main__':
    unittest.main()
