"""Publication optimized safe dispatcher with older pending allocations retained."""
import unittest
import test_ms_vlm_batch_safe_dispatch_v4 as old
import ms_vlm_batch_safe_dispatch_v5 as d
old.d=d
old.original.d=d
old.original.original.d=d
old.original.original.original.d=d
old.original.original.original.v=d.v

class PublicationDispatchTests(old.FastDiskDispatchTests):
    def test_n1_other_version_future_pending_keeps_reserved_sources(self):
        phase='ms_fullframe_vlm_postprocess_batch_v8_future_n1g01'
        p=self.rt/'other_plan.jsonl';self.put_plan(p,[dict(id='OtherUnclaimed')])
        r=d.v.load(d.REGISTRY);r['jobs'].append(dict(node='N1',gpus=[0,1],phase=phase,config_path=str(self.rt/(phase+'_config.json')),expected_running=False,sources=1,batch_sources=16,plan_ref=d.v.ref(p),worker_status='future_pending'))
        self.put(d.REGISTRY,r);self.run_dispatch();self.exec.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)
