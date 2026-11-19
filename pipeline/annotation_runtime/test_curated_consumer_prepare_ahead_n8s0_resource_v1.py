"""Three pure-CPU groups. All GPU/process/lock/lease operations are mocked."""
import copy
import fcntl
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('test_n8s0_prefetch', RT / 'curated_consumer_prepare_ahead_n8s0_resource_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def identity(pid, argv, cvd=None, ppid=9):
    return dict(pid=pid, uid=os.getuid(), argv=list(argv), start_ticks=10000+pid, cgroup='fixture',
                CVD=[] if cvd is None else [cvd], exe='/usr/bin/python3.11', ppid=ppid,
                state='S', sig_ign=0, sig_blk=0)


class N8PrepareAheadTests(unittest.TestCase):
    def fixture(self, root):
        queue = root / 'queues' / 'original'
        queue.mkdir(parents=True)
        old = m.original_config(0)[0]
        old['queue'] = str(queue)
        oldref = {'path': str(root/'old.json'), 'sha256': 'f'*64}
        manager = identity(123, ['/opt/venv_eval/bin/python', '-B', '-u',
            str(root/'runtime'/'curated_consumer_resource_manager.py'), '--config', oldref['path'],
            '--config-sha', oldref['sha256'], '--manager-sha', m.BASE_SHA])
        child = identity(456, old['argv'], 'CUDA_VISIBLE_DEVICES=4,5', 123)
        old['holders'] = [dict(identity(600+g, ['original-holder'], ppid=1),
                              gpu=g, uuid='GPU-fixture-'+str(g)) for g in m.GPUS]
        admission = {key:old[key] for key in ('phase', 'node', 'gpus', 'out', 'producer', 'plan', 'targetcount', 'source_sha256')}
        admission.update(config_sha256=oldref['sha256'], manager_sha256=m.BASE_SHA,
            manager_identity=manager, model_identity_sha256=old['modelref']['model_identity_sha256'],
            holders=[{'gpu':g, 'uuid':'GPU-fixture-'+str(g), 'identity':identity(600+g, ['original-holder'], ppid=1)} for g in m.GPUS])
        start = time.time_ns()-10**9
        def save(name, value, offset):
            path = queue/name
            path.write_text(json.dumps(value))
            os.utime(path, ns=(start+offset, start+offset))
            return m.ref(path)
        ar = save('admission.json', admission, 0)
        lr = save('live.json', dict(phase=old['phase'], argv=old['argv'], targetcount=old['targetcount'],
                                  child_identity=child, manager_identity=manager), 1)
        outcome = save('failure.json', dict(phase=old['phase'], worker_started=True,
            unknown_jobs_not_signaled=True, partial_outputs_preserved=True,
            error="InterruptedError('manager received signal 15')"), 2)
        restores, restore_refs = [], []
        for g, admitted in zip(m.GPUS, admission['holders']):
            row = {'gpu':g, 'uuid':admitted['uuid'], 'original_identity':admitted['identity'],
                   'holder_identity':identity(700+g, m.c.base.RESTORE_ARGV, 'CUDA_VISIBLE_DEVICES='+str(g), 123),
                   'managed_parent_wait':True, 'restore_argv':list(m.c.base.RESTORE_ARGV)}
            restores.append(row)
            restore_refs.append(save('holder_restored_gpu%d.json'%g, row, 3+g))
        safe = save('resources_safe.json', dict(phase=old['phase'], all_holders_safe=True,
            locks_released=True, leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True), 10)
        transition = dict(schema='curated_consumer_prefetch_n8s0_transition_v1', node='N8', shard=0,
            reason='intentional_reschedule', old_config_ref=oldref, admission_ref=ar, live_ref=lr,
            outcome_ref=outcome, safe_ref=safe, restored_holder_refs=restore_refs,
            old_manager_identity=manager, checked_unix_ns=start+11)
        path = root/'transition.json'
        path.write_text(json.dumps(transition))
        return old, oldref, manager, child, restores, transition, m.ref(path), save

    def capture_mocks(self, restores, budget=200*2**30, extra_app=False, ignored_term=False, wrong_host=False):
        events = []
        st = SimpleNamespace(st_dev=1, st_ino=2, st_uid=os.getuid(), st_mode=stat.S_IFREG|0o600,
                             st_size=0, st_mtime_ns=time.time_ns()-m.TTL_NS-10**9)
        holders = {r['holder_identity']['pid']:copy.deepcopy(r['holder_identity']) for r in restores}
        if ignored_term:
            next(iter(holders.values()))['sig_ign'] = 1<<(signal.SIGTERM-1)
        def apps():
            events.append('apps')
            result = [{'uuid':r['uuid'], 'pid':r['holder_identity']['pid']} for r in restores]
            if extra_app:
                result.append({'uuid':restores[0]['uuid'], 'pid':9999})
            return result
        contexts = [patch.object(m.subprocess, 'check_output', return_value=m.c.HOSTS['N1' if wrong_host else 'N8']),
            patch.object(Path, 'lstat', return_value=st), patch.object(Path, 'read_text', return_value='boot-fixture'),
            patch.object(m.os, 'open', return_value=800), patch.object(m.os, 'fstat', return_value=st),
            patch.object(m.os, 'close'), patch.object(m.fcntl, 'flock', side_effect=lambda *a:events.append('lock')),
            patch.object(m, 'lease_snapshot', return_value=st),
            patch.object(m.c.base, 'uuid_for', side_effect=lambda gpu:'GPU-fixture-'+str(gpu)),
            patch.object(m.c.base, 'apps', side_effect=apps), patch.object(m.c.base, 'identity', side_effect=lambda pid:holders[pid]),
            patch.object(m.os, 'pidfd_open', return_value=801), patch.object(m.select, 'select', return_value=([], [], [])),
            patch.object(m.c.base, 'headroom', return_value={'headroom':budget})]
        return contexts, events, st

    def test_1_same_native_and_cpu_source_model_first(self):
        cfg = m.plan_config(0)
        old, oldref = m.original_config(0)
        m.validate_semantics(cfg)
        self.assertEqual(set(cfg), set(old)|m.NEW_FIELDS)
        for key in set(old)-m.CHANGED:
            self.assertEqual(cfg[key], old[key])
        self.assertEqual(cfg['argv'][:3]+cfg['argv'][4:], old['argv'][:3]+old['argv'][4:])
        self.assertEqual((cfg['node'], cfg['gpus'], cfg['targetcount'], cfg['input_count']), ('N8', [4, 5], 55735, 222939))
        self.assertEqual(cfg['resource_manager_ref'], m.ref(m.SELF))
        self.assertEqual(cfg['preparation_ref'], m.ref(m.SELF))
        for shard in (1, 2, 3, False):
            with self.subTest(shard=shard), self.assertRaises(RuntimeError):
                m.plan_config(shard)
        for key, value in [('node', 'N1'), ('gpus', [0,1]), ('argv', cfg['argv']+['--batch-size', '1']),
                           ('targetcount', 1), ('out', '/tmp/other'), ('unexpected', True)]:
            altered = copy.deepcopy(cfg)
            altered[key] = value
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                m.validate_semantics(altered)
        resigned = copy.deepcopy(cfg)
        resigned['source_sha256'][next(iter(old['source_sha256']))] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'Original source'):
            m.validate_semantics(resigned)
        digests = dict(cfg['source_sha256'])
        digests.update({cfg[k]['path']:cfg[k]['sha256'] for k in ('original_full_manifest', 'manifest_summary', 'modelref')})
        real_sha = m.c.base.sha
        events = []
        with patch.object(m, 'original_check_sources', side_effect=lambda *a:events.append('source')) as sources, \
             patch.object(m.c, 'check_model', side_effect=lambda c:events.append('model')) as model, \
             patch.object(m.c.base, 'sha', side_effect=lambda p:digests.get(str(p)) or real_sha(p)), \
             patch.object(m, 'fresh_capture', side_effect=AssertionError('real capture forbidden')), \
             patch.object(m.os, 'open', side_effect=AssertionError('real lock access forbidden')), \
             patch.object(m.c.base, 'apps', side_effect=AssertionError('real GPU access forbidden')), \
             patch.object(m.subprocess, 'check_output', side_effect=AssertionError('host access forbidden')):
            m.check_plan(cfg)
            self.assertEqual(events, ['source', 'model'])
            sources.assert_called_once_with(cfg, oldref['path'], oldref['sha256'], m.BASE_SHA)
            model.assert_called_once_with(cfg)
        with patch.object(m.c, 'original_model') as native_model, \
             patch.object(m.c.base, 'sha', return_value=cfg['modelref']['sha256']), \
             patch.object(Path, 'read_text', return_value='{}'):
            m.c.check_model(cfg)
            native_model.assert_called_once_with(cfg)
            self.assertEqual(native_model.call_args.args[0]['node'], 'N8')
        # Exercise the real base model inventory/stage gate against tiny CPU files.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for i in range(25):
                name = ('config.json', 'model.safetensors.index.json')[i] if i < 2 else 'part%d.bin'%i
                path = root/name
                path.write_bytes(b'{}')
                files[name] = {'stat':{'size':2}, 'sha256':hashlib.sha256(b'{}').hexdigest()}
            model_reference = root/'reference.json'
            # Keep reference/marker outside the mapped model inventory.
            inventory = root/'model'
            inventory.mkdir()
            for name in files:
                (root/name).rename(inventory/name)
            model_reference.write_text(json.dumps({'model_identity':{'sha256':cfg['modelref']['model_identity_sha256'],
                'root':str(inventory), 'files':files}}))
            model_cfg = copy.deepcopy(cfg)
            model_cfg['modelref'].update(path=str(model_reference), sha256=m.c.base.sha(model_reference))
            marker = root/'marker.json'
            def local_path(p):
                if str(p) == '/opt/models/Qwen3-VL-32B-Instruct':return inventory
                if str(p) == str(inventory)+'.stage_receipt.json':return marker
                return Path(p)
            with patch.object(m.c.base, 'Path', side_effect=local_path):
                with self.assertRaises(FileNotFoundError):m.c.original_model(model_cfg)
                marker.write_text(json.dumps({'complete':True, 'model_identity_sha256':model_cfg['modelref']['model_identity_sha256']}))
                m.c.original_model(model_cfg)
                marker.write_text(json.dumps({'complete':False, 'model_identity_sha256':model_cfg['modelref']['model_identity_sha256']}))
                with self.assertRaisesRegex(RuntimeError, 'staged model'):m.c.original_model(model_cfg)
        args = SimpleNamespace(out=str(m.RT/(m.PHASE+'_never.json')), transition='/unused', transition_sha='0'*64)
        with patch.object(m, 'check_plan', side_effect=RuntimeError('changed source')), \
             patch.object(m, 'original_validate') as host, patch.object(m, 'handoff_gate') as gate, \
             patch.object(m, 'fresh_capture') as capture:
            with self.assertRaisesRegex(RuntimeError, 'changed source'):m.build(args)
            host.assert_not_called()
            gate.assert_not_called()
            capture.assert_not_called()
        with patch.object(m.c.base, 'main') as native_main:
            m.main(['--config', '/unused', '--config-sha', 'a'*64, '--manager-sha', m.BASE_SHA,
                    '--self-sha', real_sha(m.SELF), '--ahead-sha', m.AHEAD_SHA])
            native_main.assert_called_once()
        # Pure in-memory save_once fixture; no real launch is created.
        output = str(m.RT/(m.PHASE+'_cpu_fixture_never_written.json'))
        stream = MagicMock()
        stream.__enter__.return_value = stream
        stream.fileno.return_value = 987
        mockcapture = {'holders':cfg['holders']}
        with patch.object(m, 'plan_config', return_value=copy.deepcopy(cfg)), \
             patch.object(m, 'check_plan'), patch.object(m, 'original_validate'), \
             patch.object(m, 'handoff_gate', return_value=([], [])), \
             patch.object(m, 'fresh_capture', return_value=mockcapture), patch.object(m, 'validate'), \
             patch.object(Path, 'open', return_value=stream) as save_once, patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'is_symlink', return_value=False), patch.object(m.os, 'fsync') as fsync, \
             patch.object(m.c.base, 'sha', return_value='a'*64):
            result = m.build(SimpleNamespace(out=output, transition='/fixture/transition', transition_sha='b'*64))
            save_once.assert_called_once_with('x')
            stream.flush.assert_called_once()
            fsync.assert_called_once_with(987)
            argv = result['manager_argv']
            self.assertEqual(argv[3], str(m.SELF))
            self.assertEqual(argv[argv.index('--self-sha')+1], 'a'*64)
            self.assertEqual(argv[argv.index('--ahead-sha')+1], m.AHEAD_SHA)
            self.assertEqual(result['resource_manager_ref']['path'], str(m.SELF))
            for refused_out in ('/tmp/not-runtime.json', output+'suffix', output.replace(m.PHASE+'_', 'curated_consumer_cache_s2_')):
                with self.subTest(output=refused_out), self.assertRaises(RuntimeError):
                    m.build(SimpleNamespace(out=refused_out, transition='/fixture/transition', transition_sha='b'*64))
        with patch.object(m, 'plan_config', return_value=cfg), patch.object(m, 'check_plan'), \
             patch.object(m, 'original_validate'), patch.object(Path, 'exists', return_value=True), \
             patch.object(m, 'handoff_gate') as gate:
            with self.assertRaisesRegex(RuntimeError, 'Unique absolute'):m.build(SimpleNamespace(out=output))
            gate.assert_not_called()

    def test_2_real_fixture_handoff_and_rejection(self):
        cfg = m.plan_config(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, oldref, manager, child, restores, transition, reference, save = self.fixture(root)
            def current(pid):
                if pid == manager['pid']:return manager
                raise FileNotFoundError()
            with patch.object(m, 'RT', root/'runtime'), patch.object(m, 'original_config', return_value=(old, oldref)), \
                 patch.object(m.c.base, 'identity', side_effect=current):
                got, refs = m.handoff_gate(cfg, reference, require_parent=True)
                self.assertEqual(got, restores)
                self.assertIn(transition['outcome_ref'], refs)
                self.assertFalse(any(Path(r['path']).name=='worker_exit.json' for r in refs))
                with patch.object(m.c.base, 'identity', side_effect=lambda p:manager if p==123 else child):
                    with self.assertRaisesRegex(RuntimeError, 'worker still alive'):m.handoff_gate(cfg, reference, require_parent=True)
                with patch.object(m.c.base, 'identity', side_effect=FileNotFoundError()):
                    with self.assertRaisesRegex(RuntimeError, 'original manager'):m.handoff_gate(cfg, reference, require_parent=True)
                    m.handoff_gate(cfg, reference)  # Parent may normally exit after handoff.
                for field, value in [('safe_ref', transition['live_ref']), ('old_manager_identity', child), ('node', 'N1'), ('worker_exit_claimed', True)]:
                    bad = copy.deepcopy(transition)
                    bad[field] = value
                    path = root/'bad-transition.json'
                    path.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(RuntimeError):m.handoff_gate(cfg, m.ref(path), require_parent=True)
                fake = copy.deepcopy(transition)
                fake['outcome_ref'] = save('worker_exit.json', dict(phase=old['phase'], exit_code=0), 2)
                path = root/'fake-exit.json'
                path.write_text(json.dumps(fake))
                with self.assertRaisesRegex(RuntimeError, 'outcome identity'):m.handoff_gate(cfg, m.ref(path), require_parent=True)
                complete_exit = {key:old[key] for key in ('phase', 'targetcount', 'producer', 'plan', 'source_selection', 'input_count')}
                complete_exit.update(exit_code=15, source_unchanged=True, summaries={})
                fake['outcome_ref'] = save('worker_exit.json', complete_exit, 2)
                path.write_text(json.dumps(fake))
                m.handoff_gate(cfg, m.ref(path), require_parent=True)  # No fictitious full-shard zero requirement.
                for field, value in [('ppid', 999), ('CVD', ['CUDA_VISIBLE_DEVICES=5']), ('sig_ign', 1<<14), ('exe', '/different')]:
                    badrow = copy.deepcopy(restores[0])
                    badrow['holder_identity'][field] = value
                    bad = copy.deepcopy(transition)
                    bad['restored_holder_refs'][0] = save('holder_restored_gpu4.json', badrow, 7)
                    path = root/'bad-restore.json'
                    path.write_text(json.dumps(bad))
                    with self.subTest(restore_field=field), self.assertRaisesRegex(RuntimeError, 'restored-holder provenance'):
                        m.handoff_gate(cfg, m.ref(path), require_parent=True)
                transition['restored_holder_refs'][0] = save('holder_restored_gpu4.json', restores[0], 7)
                path = root/'transition.json'
                path.write_text(json.dumps(transition))
                reference = m.ref(path)
                _, refs = m.handoff_gate(cfg, reference, require_parent=True)
                cfg['handoff'] = {'transition_ref':reference, 'receipt_refs':refs}
                for r in refs:cfg['source_sha256'][r['path']] = r['sha256']
                m.handoff_gate(cfg, reference, require_binding=True)
                cfg['source_sha256'][transition['safe_ref']['path']] = '0'*64
                with self.assertRaisesRegex(RuntimeError, 'not source-bound'):m.handoff_gate(cfg, reference, require_binding=True)
                safe_path = Path(transition['safe_ref']['path'])
                safe_path.rename(root/'missing-safe.json')
                with self.assertRaises(FileNotFoundError):m.handoff_gate(cfg, reference)

    def test_3_exact_lock_lease_capture_native_admission(self):
        cfg = m.plan_config(0)
        restores = [{'gpu':g, 'uuid':'GPU-fixture-'+str(g), 'holder_identity':identity(700+g,
            m.c.base.RESTORE_ARGV, 'CUDA_VISIBLE_DEVICES='+str(g), 123)} for g in m.GPUS]
        contexts, events, st = self.capture_mocks(restores)
        with ExitStack() as stack:
            for context in contexts:stack.enter_context(context)
            captured = m.fresh_capture(cfg, restores)
            self.assertEqual(events[:2], ['lock', 'lock'])
            self.assertEqual(m.os.open.call_count, 2)
            for call in m.os.open.call_args_list:
                self.assertFalse(call.args[1] & os.O_CREAT)
            self.assertEqual(m.os.close.call_count, 4)
            self.assertEqual(captured['locks_leases'][0]['lease']['mtime_ns'], st.st_mtime_ns)
            cfg['holders'] = captured['holders']
            cfg['fresh_capture'] = captured
            cfg['handoff'] = {'transition_ref': {'path':'/fixture/transition', 'sha256':'f'*64}}
            m.compare_capture(cfg, captured, captured)
            for field in ('device', 'inode', 'uid', 'mode', 'size', 'mtime_ns'):
                changed = copy.deepcopy(captured)
                changed['locks_leases'][0]['lease'][field] += 1
                with self.subTest(ns_field=field), self.assertRaises(RuntimeError):m.compare_capture(cfg, captured, changed)
            for mutate in (lambda c:c['locks_leases'][0]['lock'].pop('inode'),
                           lambda c:c['locks_leases'][0]['lock'].update(mtime_ns=float(st.st_mtime_ns)),
                           lambda c:c.update(captured_unix_ns=time.time_ns()-m.TTL_NS-1),
                           lambda c:c['headroom'].update(headroom=159*2**30)):
                changed = copy.deepcopy(captured)
                mutate(changed)
                with self.assertRaises(RuntimeError):m.compare_capture(cfg, changed, changed)
            with patch.object(m, 'handoff_gate') as gate, patch.object(m, 'original_configure_restore') as native_restore, \
                 patch.object(m, 'original_acquire_lease') as native_acquire, patch.object(m, 'original_release_holder') as native_release:
                entry = dict(gpu=4, old=st, reserve=Path('/tmp/od_reserve_4'), leasefd=800)
                m.configure_restore(entry, cfg)
                m.acquire_lease(entry)
                heartbeat = copy.copy(st)
                heartbeat.st_mtime_ns = time.time_ns()
                with patch.object(Path, 'lstat', new=lambda p:heartbeat if p == entry['reserve'] else st), \
                     patch.object(m.os, 'fstat', return_value=heartbeat):
                    m.release_holder(entry)
                native_restore.assert_called_once_with(entry, cfg)
                native_acquire.assert_called_once_with(entry)
                native_release.assert_called_once_with(entry)
                self.assertEqual(gate.call_count, 3)
                changedst = copy.copy(st)
                changedst.st_mtime_ns += 1
                with patch.object(Path, 'lstat', return_value=changedst):
                    for action in (lambda:m.configure_restore(entry, cfg), lambda:m.acquire_lease(entry), lambda:m.release_holder(entry)):
                        with self.assertRaisesRegex(RuntimeError, 'lock inode/nanoseconds'):action()
                expired = copy.deepcopy(cfg)
                expired['fresh_capture']['captured_unix_ns'] = time.time_ns()-m.TTL_NS-1
                with self.assertRaisesRegex(RuntimeError, 'expired'):m.configure_restore(entry, expired)
        for options in ({'extra_app':True}, {'ignored_term':True}, {'budget':159*2**30}, {'wrong_host':True}):
            with self.subTest(options=options), ExitStack() as stack:
                contexts, _, _ = self.capture_mocks(restores, **options)
                for context in contexts:stack.enter_context(context)
                with self.assertRaises(RuntimeError):m.fresh_capture(cfg, restores)
        # Real native acquire/refresh on two isolated CPU lease FDs. Never touch
        # production /tmp GPU locks, run a holder or deliver a signal.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries, lockfds = [], []
            try:
                for gpu in m.GPUS:
                    reserve = root/('lease%d'%gpu)
                    reserve.write_bytes(b'')
                    stale = time.time_ns()-m.TTL_NS-10**9
                    os.utime(reserve, ns=(stale, stale))
                    lockpath = root/('lock%d'%gpu)
                    lockfd = os.open(lockpath, os.O_RDWR|os.O_CREAT|os.O_EXCL, 0o600)
                    lockfds.append(lockfd)
                    fcntl.flock(lockfd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    entries.append(dict(gpu=gpu, reserve=reserve, old=m.lease_snapshot(reserve),
                        leasefd=None, ahead_handoff_config=cfg))
                def captured_entry(entry, config):
                    return {'gpu':entry['gpu'], 'lease':m.stat_value(entry['old'])}
                real_utime = os.utime
                heartbeat_ns = [time.time_ns()-5*10**9]
                def timestamp_fd(path, times=None, **kwargs):
                    if times is None and not kwargs:
                        # Real FD timestamp writes with a controlled CPU clock;
                        # no sleeps or filesystem timestamp-resolution assumption.
                        heartbeat_ns[0] += 10**9
                        return real_utime(path, ns=(heartbeat_ns[0], heartbeat_ns[0]))
                    return real_utime(path, times, **kwargs)
                with patch.object(m, 'entry_capture', side_effect=captured_entry), \
                     patch.object(m.c.base.os, 'utime', side_effect=timestamp_fd), \
                     patch.object(m, 'original_release_holder') as native_release:
                    m.acquire_lease(entries[0])  # Calls the real frozen native acquire.
                    first = entries[0]['ahead_owned_lease_stat']['mtime_ns']
                    m.acquire_lease(entries[1])
                    # Native refresh is legal after either lease is acquired.
                    earlier = time.time_ns()-10**9
                    os.utime(entries[0]['leasefd'], ns=(earlier, earlier))
                    m.c.base.refresh_leases(entries)
                    refreshed = os.fstat(entries[0]['leasefd']).st_mtime_ns
                    self.assertNotEqual(refreshed, first)
                    self.assertNotEqual(refreshed, earlier)
                    for entry in entries:
                        # The original pair lock FDs remain owned throughout.
                        contender = os.open(root/('lock%d'%entry['gpu']), os.O_RDWR)
                        try:
                            with self.assertRaises(BlockingIOError):fcntl.flock(contender, fcntl.LOCK_EX|fcntl.LOCK_NB)
                        finally:
                            os.close(contender)
                        m.release_holder(entry)
                    self.assertEqual(native_release.call_count, 2)
                    victim = entries[0]
                    victim['reserve'].rename(root/'retained-original-lease')
                    victim['reserve'].write_bytes(b'')
                    with self.assertRaisesRegex(RuntimeError, 'Owned lease identity'):
                        m.release_holder(victim)
                    self.assertEqual(native_release.call_count, 2)
                    victim = entries[1]
                    past = time.time_ns()-61*10**9
                    os.utime(victim['leasefd'], ns=(past, past))
                    with self.assertRaisesRegex(RuntimeError, 'current heartbeat'):
                        m.release_holder(victim)
                    self.assertEqual(native_release.call_count, 2)
            finally:
                for entry in entries:
                    if entry['leasefd'] is not None:os.close(entry['leasefd'])
                for fd in lockfds:os.close(fd)
        with patch.object(Path, 'lstat', side_effect=FileNotFoundError()):
            self.assertIsNone(m.lease_snapshot(Path('/fixture')))
        lease = SimpleNamespace(st_dev=1, st_ino=2, st_uid=os.getuid(), st_mode=stat.S_IFREG|0o600,
                                st_size=0, st_mtime_ns=time.time_ns()-m.TTL_NS-10**9)
        with patch.object(Path, 'lstat', return_value=lease):
            self.assertIs(m.lease_snapshot(Path('/fixture')), lease)
            for field, value in [('st_size', 1), ('st_uid', os.getuid()+1), ('st_mode', stat.S_IFLNK|0o600), ('st_mtime_ns', time.time_ns())]:
                badlease = copy.copy(lease)
                setattr(badlease, field, value)
                with patch.object(Path, 'lstat', return_value=badlease), self.subTest(lease_field=field), self.assertRaises(RuntimeError):
                    m.lease_snapshot(Path('/fixture'))


if __name__ == '__main__':
    unittest.main()
