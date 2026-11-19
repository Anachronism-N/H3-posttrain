"""Refill safety gates and actual registry handoff; no GPU/models/signals."""
import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import ms_native_refill_dispatch_v1 as d

class RefillTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.rt=self.root/'runtime';self.rt.mkdir();self.q=self.root/'queues/old';self.q.mkdir(parents=True);self.out=self.root/'old';self.out.mkdir()
        self.cfg={'queue':str(self.q),'out':str(self.out),'phase':'old','targetcount':3,'selected_source_ids':['a','b','c']}
        self.previous=self.rt/'oldcfg.json';self.previous.write_text(json.dumps(self.cfg))
        self.job={'node':'N3','gpu':0,'host':'known','previous_config':str(self.previous),'phase':'new','dispatcher_task_handle':'actualhandle'}
        self.jp=self.rt/'job.json';self.job['job_path']=str(self.jp);self.jp.write_text(json.dumps(self.job))
        self.base=types.SimpleNamespace(IDENTITY_KEYS=['pid','start_ticks'],identity=Mock(return_value={'pid':101,'start_ticks':1}),check_holder=Mock())
        self.manager=types.SimpleNamespace(HOSTS={'N3':'known'},ALLOWED_GPUS={'N3':[0]},base=self.base)
    def run_main(self):
        with patch.object(d,'RT',self.rt),patch.object(d,'B',self.root),patch.object(d,'manager',self.manager),patch.object(sys,'argv',['watch','--job',str(self.jp)]):d.main()
    def safe(self):
        (self.q/'resources_safe.json').write_text(json.dumps({k:True for k in ['all_holders_safe','locks_released','leases_restored']}))
        (self.q/'worker_exit.json').write_text(json.dumps({'exit_code':0}))
        summary={'status':'completed','processed':3,'selected_source_ids':['a','b','c'],'allprefetch_drained':True,'uncommitted_prefetch':0,'vlm_called':False,'quality_approved':False}
        (self.out/'summary.json').write_text(json.dumps(summary));return summary
    def test_failed_previous_never_starts_next(self):
        (self.q/'failure.json').write_text('{}')
        with patch.object(d.os,'execv',side_effect=AssertionError('spawn forbidden')):
            with self.assertRaisesRegex(RuntimeError,'failed'):self.run_main()
        self.base.check_holder.assert_not_called()
    def test_not_drained_summary_denied_before_holder_release(self):
        value=self.safe();value['uncommitted_prefetch']=1;(self.out/'summary.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError,'finish cleanly'):self.run_main()
        self.base.check_holder.assert_not_called()
    def test_old_scope_mismatch_denied(self):
        v=self.safe();v['selected_source_ids']=['wrong'];(self.out/'summary.json').write_text(json.dumps(v))
        with self.assertRaisesRegex(RuntimeError,'scope differs'):self.run_main()
        self.base.check_holder.assert_not_called()
    def test_manager_PID_reuse_stops_wait(self):
        (self.q/'live.json').write_text(json.dumps({'manager_identity':{'pid':101,'start_ticks':9}}))
        with self.assertRaisesRegex(RuntimeError,'manager changed'):self.run_main()
        self.base.check_holder.assert_not_called()
    def test_actual_registry_slot_replaced_only_once(self):
        path=self.rt/'ms_novlm_continuous_jobs_registry_v1.json';path.write_text(json.dumps({'jobs':[{'node':'N3','gpu':0,'config_path':str(self.previous)}]}))
        with patch.object(d,'RT',self.rt):d.update_registry(self.job)
        v=json.loads(path.read_text());self.assertEqual(v['jobs'][0]['task_handle'],'actualhandle');self.assertEqual(v['jobs'][0]['phase'],'new')
        with patch.object(d,'RT',self.rt):
            with self.assertRaisesRegex(RuntimeError,'already handed'):d.update_registry(self.job)

if __name__=='__main__':unittest.main()
