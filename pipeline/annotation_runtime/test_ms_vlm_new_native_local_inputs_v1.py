"""Synthetic selection contracts; no production data mutation or models."""
from contextlib import ExitStack
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import ms_vlm_new_native_local_inputs_v1 as m

class NewSupplySelectionTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close);self.rows=[dict(id='New%04d'%i,split='train',video_path='/synthetic/New%04d.mp4'%i) for i in range(1000)]
        self.slot='n3g0';self.phase='ms_unified_native_v1_stream1000_'+self.slot+'_20261027';self.ids=[r['id'] for r in self.rows]
        self.cfg=dict(phase=self.phase,out='/synthetic/out',targetcount=1000,input_count=1000,selected_source_ids=self.ids,ticket=dict(path='/synthetic/ticket',sha256='a'*64),plan=dict(path='/synthetic/plan',sha256='b'*64))
        self.ticket=dict(edition=self.phase,ticket_id='Synthetic',sources=self.rows)
        self.summary=dict(edition=self.phase,status='completed',processed=1000,allprefetch_drained=True,uncommitted_prefetch=0,vlm_called=False,selected_source_ids=self.ids,all_outcomes=[dict(id=r['id'],split=r['split'],batch_index=1,status=m.a.SUCCESS) for r in self.rows],completed_batches=[dict(commit_path='/synthetic/commit',commit_sha256='c'*64,batch_index=1)])
        self.commit=dict(schema='ms_unified_novlm_batch_commit_v3',edition=self.phase,parent_ticket_id='Synthetic',batch_index=1,archive_index={r['id']:dict(source_row=r) for r in self.rows},receipts=[dict(source_row=r,status=m.a.SUCCESS) for r in self.rows])
        self.registry=dict(jobs=[])
        self.stack.enter_context(patch.object(m,'SLOTS',(self.slot,)))
        def load(path):
            s=str(path)
            if s.endswith('ms_vlm_postprocess_jobs_registry_v1.json'):return copy.deepcopy(self.registry)
            if s.endswith('_config.json'):return copy.deepcopy(self.cfg)
            if s=='/synthetic/out/summary.json':return copy.deepcopy(self.summary)
            raise AssertionError(s)
        self.stack.enter_context(patch.object(m.v,'load',side_effect=load))
        def checked(ref):
            if ref['path']=='/synthetic/ticket':return copy.deepcopy(self.ticket)
            if ref['path'].endswith('source_selection_proof.json'):return dict(source_keys=[['train','Reserved0000']])
            raise AssertionError(ref)
        self.stack.enter_context(patch.object(m.v,'checked',side_effect=checked))
        self.stack.enter_context(patch.object(m.v,'frozen_commit',return_value=self.commit))
        self.stack.enter_context(patch.object(m.v,'ref',side_effect=lambda p:dict(path=str(p),sha256='c'*64)))
        self.stack.enter_context(patch.object(m.v.base,'sha',return_value='b'*64))
        self.stack.enter_context(patch.object(Path,'read_bytes',side_effect=lambda:b'\n'.join(m.v.h.bench.encoded(r) for r in self.rows)))
        self.stack.enter_context(patch.object(Path,'exists',return_value=True))
        self.eligible=self.stack.enter_context(patch.object(m.a,'eligible'))
        self.stack.enter_context(patch.object(m.v,'runtime',side_effect=AssertionError('No GPU model')))
    def test_new_sources_preserve_selection_order(self):
        batches,keys,refs=m.select(20);self.assertEqual(keys,[['train',i] for i in self.ids[:20]]);self.assertEqual(batches[0]['source_ids'],self.ids[:20]);self.assertEqual(self.eligible.call_count,20)
    def test_excludes_all_registered_prior_stage_ids(self):
        self.registry['jobs']=[dict(config_path='/synthetic/registered_config.json')];self.cfg['selected_source_ids']=self.ids
        with self.assertRaises(ValueError):m.select(1)
    def test_no_retry_of_failed_receipts(self):
        self.commit['receipts'][0]['status']='failed_sealed_no_retry';self.summary['all_outcomes'][0]['status']='failed_sealed_no_retry';_,keys,_=m.select(1);self.assertEqual(keys,[['train','New0001']])
    def test_val_test_not_silently_made_train(self):
        self.rows[0]['split']='val';self.commit['receipts'][0]['source_row']['split']='val';self.summary['all_outcomes'][0]['split']='val';_,keys,_=m.select(1);self.assertEqual(keys,[['train','New0001']])
    def test_uncommitted_prefetch_rejected(self):self.summary['uncommitted_prefetch']=1;self.reject()
    def test_incomplete_summary_rejected(self):self.summary['processed']=999;self.reject()
    def test_wrong_ticket_binding_rejected(self):self.ticket['edition']='wrong';self.reject()
    def test_summary_source_list_mismatch_rejected(self):self.summary['selected_source_ids']=self.ids[:-1];self.reject()
    def test_missing_terminal_source_rejected(self):self.summary['all_outcomes']=self.summary['all_outcomes'][:-1];self.reject()
    def test_duplicate_terminal_source_rejected(self):self.summary['all_outcomes'][-1]=self.summary['all_outcomes'][0];self.reject()
    def test_source_row_changed_from_ticket_rejected(self):self.commit['receipts'][0]['source_row']=dict(self.rows[0],video_path='/wrong');self.reject()
    def test_insufficient_source_pool_does_not_synthesize(self):
        with self.assertRaises(ValueError):m.select(1001)
    def test_invalid_budget_before_any_write(self):
        for count in (0,-1,True,5001):
            with self.assertRaises(ValueError):m.run(count,'/root/ms_vlm_fastcache/unused','/synthetic/shared')
    def reject(self):
        with self.assertRaises(ValueError):m.select(1)

if __name__=='__main__':unittest.main(verbosity=2)
