"""Independent optimized-caller safe dispatch with full inherited CPU guard tests."""
import unittest
import test_ms_vlm_batch_safe_dispatch_v2 as original
import ms_vlm_batch_safe_dispatch_v3 as d
original.d=d
original.original.d=d
original.original.v=d.v

class OptimizedDispatchTests(original.ProductionDispatchTests):
    def test_parallel_caller_sixteen_batch_budget(self):
        self.rows=self.make_rows(32)
        self.put_plan(self.plan,self.rows)
        self.job.update(source_count=32,batch_sources=16)
        self.freeze()
        self.run_dispatch()
        cfg=d.v.load(self.rt/(self.phase+"_config.json"))
        self.assertEqual(cfg["targetcount"],32)
        self.assertEqual(cfg["batch_sources"],16)
        self.exec.assert_called_once()

    def test_parallel_caller_thirty_two_batch_budget(self):
        self.rows=self.make_rows(64)
        self.put_plan(self.plan,self.rows)
        self.job.update(source_count=64,batch_sources=32)
        self.freeze()
        self.run_dispatch()
        self.assertEqual(d.v.load(self.rt/(self.phase+"_config.json"))["batch_sources"],32)

    def test_wrong_batch(self):
        self.job.update(batch_sources=64)
        self.freeze()
        self.assert_blocked()

if __name__=="__main__":unittest.main(verbosity=2)
