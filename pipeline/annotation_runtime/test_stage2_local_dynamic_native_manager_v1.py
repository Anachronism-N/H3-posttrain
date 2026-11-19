"""Local native admission validates original ownership without touching any shared I/O."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import stage2_local_dynamic_native_manager_v1 as m

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='native-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.cfg=dict(schema='stage2_local_dynamic_native_config_v1',node='N3',gpus=[0,1],phase='local_dynamic_test',local_root='/root/stage2_test',out='/root/stage2_test/out',queue='/root/stage2_test/queues/test',holders=[dict(gpu=i,pid=100+i,uuid='GPU-'+str(i)) for i in range(2)],release_authorized=True,new_stage_authorized=True,targetcount=64)
        self.stack.enter_context(patch.object(m,'local_path',side_effect=lambda x:Path(x)))
        self.stack.enter_context(patch.object(m.b.subprocess,'check_output',return_value=m.HOSTS['N3']))
        self.memory=self.stack.enter_context(patch.object(m,'headroom',return_value=dict(headroom=160*2**30)))
        self.disk=self.stack.enter_context(patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':8*2**30})()))
        self.ident=self.stack.enter_context(patch.object(m.b,'identity',side_effect=lambda pid:dict(pid=pid)))
        self.owner=self.stack.enter_context(patch.object(m.b,'check_holder'))
        self.stack.enter_context(patch.object(m.b,'uuid_for',side_effect=lambda gpu:'GPU-'+str(gpu)))
        self.stack.enter_context(patch.object(m.b,'apps',return_value=[]))
        self.lease=self.stack.enter_context(patch.object(m.b,'lease_snapshot',return_value=None))
        self.stack.enter_context(patch.object(m.Path,'exists',return_value=False))
        self.signal=self.stack.enter_context(patch.object(m.b.signal,'pidfd_send_signal',side_effect=AssertionError('Validation must not signal')))
    def test_exact_pair_160gib_tmp8gib_pass_without_signals(self):m.validate(self.cfg);self.assertEqual(self.owner.call_count,2);self.assertEqual(self.lease.call_count,2);self.signal.assert_not_called()
    def test_memory1byte_below_refused(self):self.memory.return_value=dict(headroom=160*2**30-1);self.reject()
    def test_tmp1byte_below_refused(self):self.disk.return_value=type('Disk',(),{'free':8*2**30-1})();self.reject()
    def test_unknown_node_refused(self):self.cfg['node']='foreign';self.reject()
    def test_invalid_pair_refused(self):self.cfg['gpus']=[0,2];self.reject()
    def test_duplicate_holder_pid_refused(self):self.cfg['holders'][1]['pid']=100;self.reject()
    def test_wrong_holder_order_refused(self):self.cfg['holders'].reverse();self.reject()
    def test_new_stage_permission_required(self):self.cfg['new_stage_authorized']=False;self.reject()
    def test_holder_release_permission_required(self):self.cfg['release_authorized']=False;self.reject()
    def test_existing_queue_never_restarted(self):
        with patch.object(m.Path,'exists',return_value=True):self.reject()
    def test_actual_wrong_node_refused(self):
        with patch.object(m.b.subprocess,'check_output',return_value='wrong'):self.reject()
    def test_unowned_lifecycle_root_refused(self):self.cfg['out']='/root/unrelated';self.reject()
    def reject(self):
        with self.assertRaises(ValueError):m.validate(self.cfg)
        self.signal.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
