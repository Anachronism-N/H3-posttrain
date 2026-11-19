"""Fresh allocation CPU tests: no real GPU, no existing-process signals."""
from contextlib import ExitStack
import copy
import fcntl
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import ms_vlm_fresh_empty_pair_handoff_v1 as m
v=m.v

class FreshPairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='fresh-pair-cpu-',dir='/tmp')
        self.root=Path(self.tmp.name);self.rt=self.root/'runtime';self.rt.mkdir();self.reserves=self.root/'reserves';self.reserves.mkdir()
        self.stack=ExitStack();self.addCleanup(self.tmp.cleanup);self.addCleanup(self.stack.close)
        self.script=self.rt/'fresh.py';self.script.write_bytes(b'frozen cpu fixture')
        for name,value in [('RT',self.rt),('ROOT',self.root),('TMP',self.reserves),('SELF',self.script)]:self.stack.enter_context(patch.object(m,name,value))
        self.phase='ms_vlm_fresh_seed_n1g01_cpu'
        self.jobpath=self.rt/(self.phase+'_job.json')
        self.job=dict(schema='ms_vlm_fresh_empty_pair_job_v1',node='N1',gpus=[0,1],phase=self.phase,fresh_pair_authorized=True,may_signal_existing_processes=False,producer_ref=v.ref(self.script),holder_ref=v.ref(v.base.HOLDER))
        self.children=[];self.foreign=[];self.uuid_change=False;self.uuid_calls=0
        self.stack.enter_context(patch.object(m.subprocess,'check_output',return_value=m.HOST))
        self.headroom=self.stack.enter_context(patch.object(v,'actual_controller_headroom',return_value=dict(headroom=160*2**30)))
        self.stack.enter_context(patch.object(v.base,'uuid_for',side_effect=self.uuid))
        self.stack.enter_context(patch.object(v.base,'apps',side_effect=self.apps))
        self.stack.enter_context(patch.object(v.base,'identity',side_effect=self.identity))
        self.Popen=self.stack.enter_context(patch.object(m.subprocess,'Popen',side_effect=self.spawn))
        self.signal=self.stack.enter_context(patch.object(v.base.signal,'pidfd_send_signal',side_effect=AssertionError('No existing process signal')))
        self.kill=self.stack.enter_context(patch.object(os,'kill',side_effect=AssertionError('No existing process signal')))
        self.model=self.stack.enter_context(patch.object(v,'runtime',side_effect=AssertionError('No VLM model')))
        self.freeze()

    def freeze(self):
        self.jobpath.write_bytes(v.h.bench.encoded(self.job));self.jobref=v.ref(self.jobpath)

    def uuid(self,g):
        self.uuid_calls+=1
        return ('Changed-' if self.uuid_change and self.uuid_calls>2 else 'GPU-')+str(g)

    def identity(self,pid):
        c=next((x for x in self.children if x.pid==pid),None);gpu=c.gpu if c else 0
        return dict(pid=pid,uid=os.getuid(),argv=list(v.base.RESTORE_ARGV),start_ticks=pid*10,cgroup='cpu-fixture',CVD=['CUDA_VISIBLE_DEVICES='+str(gpu)],exe='cpu-python',sig_ign=0,sig_blk=0,state='S')

    def apps(self):
        return [dict(uuid='GPU-'+str(c.gpu),pid=c.pid,used_MiB='1') for c in self.children if c.poll() is None]+self.foreign

    def spawn(self,argv,env,cwd,stdout,stderr,preexec_fn):
        outer=self
        class Child:
            def __init__(self):self.pid=100000+len(outer.children);self.gpu=int(env['CUDA_VISIBLE_DEVICES']);self.finished=False;self.terminated=False
            def poll(self):return 0 if self.finished else None
            def wait(self):self.finished=True;return 0
            def terminate(self):self.terminated=True;self.finished=True
        self.assertEqual(argv,v.base.RESTORE_ARGV);self.assertEqual(cwd,'/tmp');self.assertEqual(env['OCCUPY_GPU'],'0')
        c=Child();self.children.append(c);return c

    def run_provision(self):m.provision(str(self.jobpath),self.jobref['sha256'])

    def blocked(self):
        with self.assertRaises((ValueError,RuntimeError,FileExistsError,BlockingIOError)):self.run_provision()
        self.Popen.assert_not_called();self.signal.assert_not_called();self.kill.assert_not_called();self.model.assert_not_called()

    def test_real_kernel_lock_and_stale_lease_ownership_handoff(self):
        for gpu in (0,1):
            p=self.reserves/('od_reserve_%d'%gpu);p.write_bytes(b'');os.utime(p,ns=(1_000_000_000,1_000_000_000))
        before=[(self.reserves/('od_reserve_%d'%g)).stat().st_mtime_ns for g in (0,1)]
        self.run_provision();q=self.root/'queues'/self.phase;s=v.load(q/'resources_safe.json')
        self.assertTrue(all(s[k] is True for k in ('all_holders_safe','locks_released','leases_restored','managed_parent_wait','unknown_jobs_not_signaled')))
        self.assertTrue(s['fresh_owned_pair_handoff']);self.assertEqual(self.Popen.call_count,2)
        for gpu in (0,1):
            r=v.load(q/('holder_restored_gpu%d.json'%gpu));self.assertTrue(r['created_from_verified_empty_pair']);self.assertFalse(r['previous_task_holder_restored'])
        self.assertEqual(before,[(self.reserves/('od_reserve_%d'%g)).stat().st_mtime_ns for g in (0,1)])
        self.assertFalse(any(self.root.rglob('*.claim.json')));self.signal.assert_not_called();self.kill.assert_not_called();self.model.assert_not_called()

    def test_new_lease_removed_after_handoff(self):
        self.run_provision();self.assertFalse(any(self.reserves.glob('od_reserve_*')))

    def test_busy_gpu_never_signaled(self):
        self.foreign=[dict(uuid='GPU-0',pid=42,used_MiB='12')];self.blocked()

    def test_current_live_lease_rejected(self):
        p=self.reserves/'od_reserve_0';p.write_bytes(b'');before=p.stat().st_mtime_ns;self.blocked();self.assertEqual(p.stat().st_mtime_ns,before)

    def test_real_competing_flock_is_not_released(self):
        p=self.reserves/'ms12k_gpu_0.lock'
        with p.open('a+b') as fd:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);self.blocked();fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_wrong_node(self):self.job['node']='N2';self.freeze();self.blocked()
    def test_wrong_pair(self):self.job['gpus']=[0,2];self.freeze();self.blocked()
    def test_wrong_phase_pair_binding(self):self.job['gpus']=[2,3];self.freeze();self.blocked()
    def test_missing_authorization(self):self.job['fresh_pair_authorized']=False;self.freeze();self.blocked()
    def test_existing_signals_forbidden(self):self.job['may_signal_existing_processes']=True;self.freeze();self.blocked()
    def test_wrong_physical_host(self):
        with patch.object(m.subprocess,'check_output',return_value='unrelated'):self.blocked()
    def test_budget_one_byte_below160_gib(self):self.headroom.return_value=dict(headroom=160*2**30-1);self.blocked()
    def test_changed_job_sha(self):self.jobref['sha256']='0'*64;self.blocked()
    def test_changed_source_sha(self):self.script.write_bytes(b'changed');self.blocked()
    def test_wrong_holder_script(self):self.job['holder_ref']=self.job['producer_ref'];self.freeze();self.blocked()
    def test_physical_uuid_changed_before_any_holder(self):self.uuid_change=True;self.blocked()
    def test_lease_symlink_rejected(self):
        p=self.reserves/'external';p.write_bytes(b'');(self.reserves/'od_reserve_0').symlink_to(p);self.blocked()
    def test_nonempty_lock_rejected(self):(self.reserves/'ms12k_gpu_0.lock').write_bytes(b'foreign');self.blocked()
    def test_duplicate_queue_rejected(self):
        q=self.root/'queues'/self.phase;q.mkdir(parents=True);self.blocked()

if __name__=='__main__':unittest.main(verbosity=2)
