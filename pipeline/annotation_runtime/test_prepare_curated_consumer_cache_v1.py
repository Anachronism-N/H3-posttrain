"""Pure CPU fixtures: no real GPU, process signals, BAR launch, or lease mutation."""
import copy
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
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('test_cache_prepare', RT / 'prepare_curated_consumer_cache_v1.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
m = prepare.load_manager(__import__('hashlib').sha256(prepare.MANAGER.read_bytes()).hexdigest())


def identity(pid, argv, cvd=None, ppid=9):
    return dict(pid=pid, uid=os.getuid(), argv=argv, start_ticks=10000+pid, cgroup='fixture',
                CVD=[] if cvd is None else [cvd], exe='/usr/bin/python3.11', ppid=ppid,
                state='S', sig_ign=0, sig_blk=0)


class CachePreparationTests(unittest.TestCase):
    def test_plan_preserves_all_native_parameters(self):
        for shard in (2, 3):
            cfg = m.plan_config(shard)
            m.validate_semantics(cfg)
            old, _ = m.original_config(shard)
            for key in set(old) - m.CHANGED:
                self.assertEqual(cfg[key], old[key])
            self.assertEqual(cfg['argv'][:3], old['argv'][:3])
            self.assertEqual(cfg['argv'][4:], old['argv'][4:])
            self.assertIsNone(cfg['handoff'])
            self.assertIsNone(cfg['fresh_capture'])

    def test_semantic_changes_and_source_rebinding_rejected(self):
        original = m.plan_config(2)
        mutations = [('env', {'CUDA_VISIBLE_DEVICES':'0,1'}), ('targetcount', 3),
                     ('out', '/tmp/other'), ('source_selection', {'shard':2,'nshard':4,'limit':3}),
                     ('argv', original['argv']+['--batch-size','1']), ('unexpected', True)]
        for key, value in mutations:
            cfg = copy.deepcopy(original)
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.validate_semantics(cfg)
        cfg = copy.deepcopy(original)
        first = next(iter(m.original_config(2)[0]['source_sha256']))
        cfg['source_sha256'][first] = '0'*64
        with self.assertRaises(RuntimeError):
            m.validate_semantics(cfg)

    def test_check_plan_never_touches_gpu_or_locks(self):
        cfg = m.plan_config(2)
        digests = dict(cfg['source_sha256'])
        digests.update({cfg[key]['path']:cfg[key]['sha256'] for key in ('modelref','manifest_summary','original_full_manifest')})
        real_sha = m.c.base.sha
        with patch.object(m, 'original_check_sources') as sources, patch.object(m.c, 'check_model') as model, \
             patch.object(m.c.base, 'sha', side_effect=lambda p: digests.get(str(p)) or real_sha(p)), \
             patch.object(m, 'fresh_capture', side_effect=AssertionError('GPU capture forbidden')), \
             patch.object(m.os, 'open', side_effect=AssertionError('Lock access forbidden')), \
             patch.object(m.c.base, 'apps', side_effect=AssertionError('GPU access forbidden')):
            m.check_plan(cfg)
            sources.assert_called_once()
            model.assert_called_once_with(cfg)

    def test_source_validation_failure_precedes_build_capture(self):
        args = SimpleNamespace(shard=2, out='/tmp/no-launch.json', transition='/tmp/unused', transition_sha='0'*64)
        with patch.object(m, 'check_plan', side_effect=RuntimeError('changed source')), \
             patch.object(m, 'handoff_gate') as gate, patch.object(m, 'fresh_capture') as capture:
            with self.assertRaisesRegex(RuntimeError, 'changed source'):
                prepare.build(args, m)
            gate.assert_not_called()
            capture.assert_not_called()

    def fixture(self, root):
        queue = root / 'queues' / 'original'
        queue.mkdir(parents=True)
        old = m.original_config(2)[0]
        old['queue'] = str(queue)
        oldref = {'path':str(root/'old.json'),'sha256':'f'*64}
        manager_identity = identity(123, ['/opt/venv_eval/bin/python','-B','-u',
            str(m.RT/'curated_consumer_resource_manager.py'),'--config',oldref['path'],
            '--config-sha',oldref['sha256'],'--manager-sha',m.BASE_SHA])
        child = identity(456, old['argv'], 'CUDA_VISIBLE_DEVICES=0,1', 123)
        admission = {key:old[key] for key in ('phase','node','gpus','out','producer','plan','targetcount','source_sha256')}
        old['holders'] = [dict(identity(600+g,['old-holder'],ppid=1),gpu=g,uuid='GPU-fixture-'+str(g)) for g in old['gpus']]
        admission.update(config_sha256=oldref['sha256'], manager_sha256=m.BASE_SHA,
            manager_identity=manager_identity, model_identity_sha256=old['modelref']['model_identity_sha256'],
            holders=[{'gpu':g,'uuid':'GPU-fixture-'+str(g),'identity':identity(600+g,['old-holder'],ppid=1)} for g in old['gpus']])
        now = time.time_ns()-10**9
        def save(name, value, offset):
            p = queue/name
            p.write_text(json.dumps(value))
            os.utime(p, ns=(now+offset,now+offset))
            return m.ref(p)
        ar = save('admission.json', admission, 0)
        lr = save('live.json', {'phase':old['phase'],'argv':old['argv'],'child_identity':child,'manager_identity':manager_identity}, 1)
        outcome = save('failure.json', {'phase':old['phase'],'worker_started':True,'unknown_jobs_not_signaled':True,
            'error':'InterruptedError intentional stop'}, 2)
        restored = []
        restore_refs = []
        for g, admitted in zip(old['gpus'],admission['holders']):
            row = {'gpu':g,'uuid':admitted['uuid'],'original_identity':admitted['identity'],
                   'holder_identity':identity(700+g,m.c.base.RESTORE_ARGV,'CUDA_VISIBLE_DEVICES='+str(g),123),
                   'managed_parent_wait':True,'restore_argv':m.c.base.RESTORE_ARGV}
            restored.append(row)
            restore_refs.append(save('holder_restored_gpu%d.json'%g,row,3+g))
        sr = save('resources_safe.json', dict(phase=old['phase'],all_holders_safe=True,
            locks_released=True,leases_restored=True,managed_parent_wait=True,unknown_jobs_not_signaled=True),6)
        transition = dict(schema='curated_consumer_cache_transition_v1',node='N1',shard=2,
            reason='intentional_reschedule',worker_exit_claimed=False,original_consumer_config_ref=oldref,
            admission_ref=ar,live_ref=lr,outcome_ref=outcome,resources_safe_ref=sr,
            restored_holder_refs=restore_refs,old_manager_identity=manager_identity,checked_unix_ns=now+7)
        p = root/'transition.json'
        p.write_text(json.dumps(transition))
        return old, oldref, manager_identity, child, restored, transition, m.ref(p)

    def test_handoff_accepts_real_safe_intentional_stop_not_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            old, oldref, manager, child, restores, transition, reference = self.fixture(Path(directory))
            cfg = m.plan_config(2)
            def current(pid):
                if pid == manager['pid']:
                    current_manager=copy.deepcopy(manager)
                    current_manager['ppid']=1
                    return current_manager
                raise FileNotFoundError()
            with patch.object(m, 'original_config', return_value=(old,oldref)), \
                 patch.object(m.c.base, 'identity', side_effect=current), \
                 patch.object(Path, 'is_relative_to', return_value=True):
                result, refs = m.handoff_gate(cfg,reference,require_parent=True)
                self.assertEqual(result,restores)
                self.assertIn(transition['outcome_ref'],refs)
                self.assertNotIn('worker_exit.json',[Path(r['path']).name for r in refs])

    def test_handoff_rejects_live_child_fake_exit_or_wrong_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            old, oldref, manager, child, restores, transition, reference = self.fixture(Path(directory))
            cfg = m.plan_config(2)
            with patch.object(m,'original_config',return_value=(old,oldref)), \
                 patch.object(Path,'is_relative_to',return_value=True):
                with patch.object(m.c.base,'identity',side_effect=lambda pid: manager if pid==manager['pid'] else child):
                    with self.assertRaisesRegex(RuntimeError,'worker still alive'):
                        m.handoff_gate(cfg,reference,require_parent=True)
                def current(pid):
                    if pid==manager['pid']:return manager
                    raise FileNotFoundError()
                with patch.object(m.c.base,'identity',side_effect=current):
                    for key,value in [('worker_exit_claimed',True),('resources_safe_ref',transition['live_ref'])]:
                        changed = copy.deepcopy(transition)
                        changed[key]=value
                        p=Path(directory)/'bad.json';p.write_text(json.dumps(changed))
                        with self.subTest(key=key), self.assertRaises(RuntimeError):
                            m.handoff_gate(cfg,m.ref(p),require_parent=True)
                    wrong=copy.deepcopy(restores[0]);wrong['holder_identity']['ppid']=999
                    p=Path(transition['restored_holder_refs'][0]['path']);p.write_text(json.dumps(wrong))
                    changed=copy.deepcopy(transition);changed['restored_holder_refs'][0]=m.ref(p)
                    q=Path(directory)/'badrestore.json';q.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(RuntimeError,'restored-holder provenance'):
                        m.handoff_gate(cfg,m.ref(q),require_parent=True)

    def test_exact_parent_required_before_build_but_may_exit_after_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            old,oldref,manager,child,restores,transition,reference=self.fixture(Path(directory))
            cfg=m.plan_config(2)
            with patch.object(m,'original_config',return_value=(old,oldref)), \
                 patch.object(Path,'is_relative_to',return_value=True), \
                 patch.object(m.c.base,'identity',side_effect=FileNotFoundError()):
                with self.assertRaisesRegex(RuntimeError,'original manager'):
                    m.handoff_gate(cfg,reference,require_parent=True)
                restored,refs=m.handoff_gate(cfg,reference)
                cfg['handoff']={'transition_ref':reference,'receipt_refs':refs}
                for r in refs:cfg['source_sha256'][r['path']]=r['sha256']
                m.handoff_gate(cfg,reference,require_binding=True)
                cfg['source_sha256'][transition['resources_safe_ref']['path']]='0'*64
                with self.assertRaisesRegex(RuntimeError,'not source-bound'):
                    m.handoff_gate(cfg,reference,require_binding=True)

    def capture_mocks(self, cfg, restores, budget=None, extra_app=False, ignored_term=False):
        captures = []
        lockstat = SimpleNamespace(st_dev=1,st_ino=2,st_uid=os.getuid(),st_mode=stat.S_IFREG|0o600,
                                   st_size=0,st_mtime_ns=1792911234567890123)
        holders={r['holder_identity']['pid']:copy.deepcopy(r['holder_identity']) for r in restores}
        if ignored_term:
            next(iter(holders.values()))['sig_ign']=1<<(signal.SIGTERM-1)
        def apps():
            captures.append('apps')
            result=[{'uuid':r['uuid'],'pid':r['holder_identity']['pid']} for r in restores]
            if extra_app:result.append({'uuid':restores[0]['uuid'],'pid':9999})
            return result
        return [patch.object(m.subprocess,'check_output',return_value=m.c.HOSTS['N1']),
                patch.object(Path,'lstat',return_value=lockstat),patch.object(Path,'read_text',return_value='boot-fixture'),
                patch.object(m.os,'open',return_value=800),patch.object(m.os,'fstat',return_value=lockstat),
                patch.object(m.os,'close'),patch.object(m.fcntl,'flock',side_effect=lambda *a: captures.append('lock')),
                patch.object(m.c.base,'lease_snapshot',return_value=None),
                patch.object(m.c.base,'uuid_for',side_effect=lambda gpu:'GPU-fixture-'+str(gpu)),
                patch.object(m.c.base,'apps',side_effect=apps),
                patch.object(m.c.base,'identity',side_effect=lambda pid:holders[pid]),
                patch.object(m.os,'pidfd_open',return_value=801),patch.object(m.select,'select',return_value=([],[],[])),
                patch.object(m.c.base,'headroom',return_value={'headroom':budget if budget is not None else 200*2**30})],captures

    def test_fresh_capture_checks_all_pair_locks_before_apps(self):
        cfg=m.plan_config(2)
        restores=[{'gpu':g,'uuid':'GPU-fixture-'+str(g),'holder_identity':identity(700+g,
            m.c.base.RESTORE_ARGV,'CUDA_VISIBLE_DEVICES='+str(g),123)} for g in cfg['gpus']]
        contexts,events=self.capture_mocks(cfg,restores)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in contexts:stack.enter_context(context)
            captured=m.fresh_capture(cfg,restores)
            self.assertEqual(events[:2],['lock','lock'])
            self.assertEqual(captured['locks_leases'][0]['lock']['mtime_ns'],1792911234567890123)
            self.assertIs(type(captured['locks_leases'][0]['lock']['mtime_ns']),int)
            cfg['holders']=captured['holders']
            m.compare_capture(cfg,captured,captured)
            changed=copy.deepcopy(captured);changed['locks_leases'][0]['lock']['mtime_ns']+=1
            with self.assertRaises(RuntimeError):m.compare_capture(cfg,captured,changed)
            changed=copy.deepcopy(captured);changed['captured_unix_ns']-=1201*10**9
            with self.assertRaises(RuntimeError):m.compare_capture(cfg,changed,captured)
            changed=copy.deepcopy(captured);changed['locks_leases'][0]['lock']['mtime_ns']=float(changed['locks_leases'][0]['lock']['mtime_ns'])
            with self.assertRaises(RuntimeError):m.compare_capture(cfg,changed,changed)

    def test_fresh_capture_rejects_unknown_compute_ignored_term_and_low_memory(self):
        cfg=m.plan_config(2)
        restores=[{'gpu':g,'uuid':'GPU-fixture-'+str(g),'holder_identity':identity(700+g,
            m.c.base.RESTORE_ARGV,'CUDA_VISIBLE_DEVICES='+str(g),123)} for g in cfg['gpus']]
        from contextlib import ExitStack
        for options in ({'extra_app':True},{'ignored_term':True},{'budget':159*2**30}):
            with self.subTest(options=options),ExitStack() as stack:
                contexts,_=self.capture_mocks(cfg,restores,**options)
                for context in contexts:stack.enter_context(context)
                with self.assertRaises(RuntimeError):m.fresh_capture(cfg,restores)

    def test_native_admission_rejects_one_ns_lease_change(self):
        cfg=m.plan_config(2)
        captured={'device':1,'inode':2,'uid':os.getuid(),'mode':stat.S_IFREG|0o600,'size':0,
                  'mtime_ns':1792911234567890123}
        cfg['fresh_capture']={'locks_leases':[{'gpu':0,'lock':captured,'lease':captured}]}
        lease=SimpleNamespace(st_dev=1,st_ino=2,st_uid=os.getuid(),st_mode=stat.S_IFREG|0o600,
                              st_size=0,st_mtime_ns=captured['mtime_ns']+1)
        lock=SimpleNamespace(**{k:v for k,v in lease.__dict__.items()});lock.st_mtime_ns-=1
        entry={'gpu':0,'cache_handoff_config':cfg,'reserve':Path('/tmp/fixture-not-real')}
        with patch.object(Path,'lstat',return_value=lock),patch.object(m.c.base,'lease_snapshot',return_value=lease), \
             patch.object(m,'original_acquire_lease') as acquire:
            with self.assertRaisesRegex(RuntimeError,'lease changed'):
                m.acquire_lease(entry)
            acquire.assert_not_called()

    def test_cli_check_plan_disallows_output(self):
        with self.assertRaises(SystemExit):
            prepare.main(['--check-plan','--shard','2','--self-sha','unused','--cache-manager-sha','unused','--out','unused'])


if __name__ == '__main__':
    unittest.main()
