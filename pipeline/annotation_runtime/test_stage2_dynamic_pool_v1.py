"""Durable SQLite dynamic claims, real concurrent connections, no models."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
import stage2_dynamic_pool_v1 as m

class DynamicPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dynamic-pool-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'pool.db';self.rows=[dict(id='Source%03d'%i,split='train',video_path='/declared/Source%03d.mp4'%i) for i in range(65)];self.pool=m.DynamicPool.create(self.p,self.rows,[['train','Source000']]);self.addCleanup(self.pool.close)
    def test_three_to_eight_first_occupies_then_returns_bound_rows(self):
        r=self.pool.claim('N1-g01-worker','request0001',8);self.assertEqual(r['source_count'],8);self.assertEqual(r['rows'],self.rows[1:9]);self.assertFalse(r['automatic_retry']);self.assertEqual(self.pool.progress(),{'assigned':8,'unused':56})
    def test_excluded_prior_source_never_issued(self):
        r=self.pool.claim('N1-g01-worker','request0001',8);self.assertNotIn('Source000',[x['id'] for x in r['rows']])
    def test_existing_request_is_not_an_automatic_retry(self):
        self.pool.claim('N1-g01-worker','request0001',8)
        with self.assertRaises(ValueError):self.pool.claim('N1-g01-worker','request0001',8)
        self.assertEqual(self.pool.progress()['assigned'],8)
    def test_reopen_after_crash_keeps_claimed_sources_unavailable(self):
        r=self.pool.claim('N1-g01-worker','request0001',8);other=m.DynamicPool(self.p)
        try:r2=other.claim('N2-g23-worker','request0002',8);self.assertFalse({x['id'] for x in r['rows']} & {x['id'] for x in r2['rows']})
        finally:other.close()
    def test_concurrent_workers_each_source_at_most_once(self):
        def claim(i):
            p=m.DynamicPool(self.p)
            try:return p.claim('N%d-g01-worker'%(1+i%8),'request%04d'%i,8)
            finally:p.close()
        with ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(claim,range(12)))
        keys=[x['id'] for r in rows if r for x in r['rows']];self.assertEqual(len(keys),64);self.assertEqual(len(set(keys)),64);self.assertEqual(self.pool.progress(),{'assigned':64})
    def test_failed_source_remains_permanently_sealed(self):
        r=self.pool.claim('N1-g01-worker','request0001',3);self.pool.terminal('N1-g01-worker','request0001','train',r['rows'][0]['id'],'failed_sealed_no_retry')
        r2=self.pool.claim('N2-g23-worker','request0002',8);self.assertNotIn(r['rows'][0]['id'],[x['id'] for x in r2['rows']]);self.assertEqual(self.pool.progress()['failed_sealed_no_retry'],1)
    def test_terminal_status_cannot_be_changed_or_regenerated(self):
        r=self.pool.claim('N1-g01-worker','request0001',3);id_=r['rows'][0]['id'];self.pool.terminal('N1-g01-worker','request0001','train',id_,'failed_sealed_no_retry')
        with self.assertRaises(ValueError):self.pool.terminal('N1-g01-worker','request0001','train',id_,'published_unapproved')
    def test_other_worker_cannot_close_claim(self):
        r=self.pool.claim('N1-g01-worker','request0001',3)
        with self.assertRaises(ValueError):self.pool.terminal('N2-g23-worker','request0001','train',r['rows'][0]['id'],'published_unapproved')
    def test_wrong_split_cannot_close_source(self):
        r=self.pool.claim('N1-g01-worker','request0001',3)
        with self.assertRaises(ValueError):self.pool.terminal('N1-g01-worker','request0001','test',r['rows'][0]['id'],'published_unapproved')
    def test_recovery_inspection_does_not_authorize_models(self):
        self.pool.claim('N1-g01-worker','request0001',3);r=self.pool.inspect_claim('N1-g01-worker','request0001');self.assertFalse(r['may_run_models']);self.assertTrue(r['metadata_only_recovery'])
    def test_unknown_claim_cannot_be_recovered_as_unused(self):
        with self.assertRaises(ValueError):self.pool.inspect_claim('N1-g01-worker','unknown0001')
    def test_invalid_batch_sizes_rejected_before_assignment(self):
        for n in (True,0,1,2,9,32):
            with self.assertRaises(ValueError):self.pool.claim('N1-g01-worker','request0001',n)
        self.assertEqual(self.pool.progress(),{'unused':64})
    def test_invalid_owner_refused(self):
        for owner in ('N0-g01-worker','N1-g00-worker','N1-g10-worker','N1-g07-worker','../../foreign'):
            with self.assertRaises(ValueError):self.pool.claim(owner,'request0001',8)
    def test_empty_pool_returns_none_without_reissuing(self):
        for i in range(8):self.pool.claim('N1-g01-worker','request%04d'%i,8)
        self.assertIsNone(self.pool.claim('N2-g23-worker','request9999',8))
    def test_short_tail_keeps_original_sources_and_once_budget(self):
        for i in range(7):self.pool.claim('N1-g01-worker','request%04d'%i,8)
        self.pool.claim('N1-g01-worker','request0008',3);r=self.pool.claim('N1-g01-worker','request0009',8);self.assertEqual(r['source_count'],5)
    def test_casefold_duplicates_forbidden(self):
        p=Path(self.tmp.name)/'other.db'
        with self.assertRaises(ValueError):m.DynamicPool.create(p,[self.rows[0],dict(self.rows[0],id='source000')],[])
        self.assertFalse(p.exists())
    def test_existing_pool_never_reset(self):
        with self.assertRaises(FileExistsError):m.DynamicPool.create(self.p,self.rows,[])
    def test_unknown_terminal_state_refused(self):
        r=self.pool.claim('N1-g01-worker','request0001',3)
        with self.assertRaises(ValueError):self.pool.terminal('N1-g01-worker','request0001','train',r['rows'][0]['id'],'approved_for_training')
if __name__=='__main__':unittest.main(verbosity=2)
