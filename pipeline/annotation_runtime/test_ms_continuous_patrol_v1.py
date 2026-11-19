import unittest
from unittest.mock import patch
import ms_continuous_patrol_v1 as m

class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.now=1000;self.s=m.State(clock=lambda:self.now)
        self.job={'node':'N6','gpu':2,'phase':'owned','host':m.HOSTS['N6']}
    def obs(self,index,sid,status='extracted_rule_draft_unapproved'):
        return {'job':self.job,'worker_alive':True,'status':'running','progress':{'processed':20*index,'source_count':1000},'progress_age_s':5,'warnings':[],'batches':[{'index':index,'records':[{'id':sid,'split':'train','status':status,'draft_status':'partial'}]}]}
    def test_baseline_and_duplicate_ID_not_counted_twice(self):
        self.assertEqual(self.s.sample([self.obs(1,'old')])['new_unique_sources'],0)
        self.now+=60;r=self.s.sample([self.obs(2,'new')]);self.assertEqual(r['new_unique_sources'],1)
        self.now+=60;r=self.s.sample([self.obs(3,'new')]);self.assertEqual(r['new_unique_sources'],1)
    def test_failures_not_counted_as_success(self):
        self.s.sample([self.obs(1,'old')]);self.now+=60
        r=self.s.sample([self.obs(2,'fail','failed_sealed_no_retry')]);self.assertEqual(r['rolling_15min']['success_sources'],0)
    def test_bad_batch_does_not_advance_cursor(self):
        self.s.sample([self.obs(1,'old')]);self.now+=60
        x=self.obs(3,'gap');r=self.s.sample([x]);self.assertEqual(self.s.cursors['owned'],1);self.assertTrue(r['rolling_15min']['measurement_incomplete'])
    def test_missing_observation_keeps_wall_time(self):
        self.s.sample([self.obs(1,'old')]);self.now+=100
        r=self.s.sample([{'job':self.job,'warnings':['ssh error'],'measurement_incomplete':True}]);self.assertEqual(r['rolling_15min']['seconds'],100);self.assertIsNone(r['rolling_15min']['sources_per_hour'])
    def test_completed_not_idle_alarm(self):
        self.s.sample([self.obs(1,'old')]);self.now+=700
        x=self.obs(1,'old');x['batches']=[];x['status']='completed';x['worker_alive']=False
        r=self.s.sample([x]);self.assertEqual(r['jobs'][0]['alarms'],[])
    def test_recent_progress_not_idle(self):
        self.s.sample([self.obs(1,'old')]);self.now+=700
        r=self.s.sample([self.obs(2,'new')]);self.assertEqual(r['jobs'][0]['alarms'],[])
    def test_unapproved_host_rejected(self):
        with self.assertRaises(ValueError):m.collect({'jobs':[dict(self.job,host='unknown')]},self.s)

if __name__=='__main__':unittest.main()
