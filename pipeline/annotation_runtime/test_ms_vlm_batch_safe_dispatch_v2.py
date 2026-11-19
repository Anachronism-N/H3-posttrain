"""Independent long-ticket CPU regression, retaining the original safety suite."""
import test_ms_vlm_batch_safe_dispatch_v1 as original
import ms_vlm_batch_safe_dispatch_v2 as d
import unittest
original.d = d

class ProductionDispatchTests(original.DispatchTests):
    def test_production_128_sources_use_batch_eight_and_one_model_lifecycle(self):
        self.rows = self.make_rows(128)
        self.put_plan(self.plan, self.rows)
        self.job.update(source_count=128, batch_sources=8)
        self.freeze()
        self.run_dispatch()
        cfg=d.v.load(self.rt/(self.phase+"_config.json"))
        self.assertEqual(cfg["targetcount"],128)
        self.assertEqual(cfg["batch_sources"],8)
        self.assertEqual(cfg["selected_source_ids"],[r["id"] for r in self.rows])
        self.exec.assert_called_once()
        self.model.assert_called_once()
        self.replay_mock.assert_called_once()
        for action in self.forbidden: action.assert_not_called()

    def test_production_short_tail_is_allowed(self):
        self.rows = self.make_rows(11)
        self.put_plan(self.plan,self.rows)
        self.job.update(source_count=11,batch_sources=8)
        self.freeze()
        self.run_dispatch()
        self.assertEqual(d.v.load(self.rt/(self.phase+"_config.json"))["targetcount"],11)

    def test_source_budget_over_bound_rejected(self):
        self.job.update(source_count=10001)
        self.freeze()
        self.assert_blocked()

if __name__=="__main__": unittest.main(verbosity=2)
