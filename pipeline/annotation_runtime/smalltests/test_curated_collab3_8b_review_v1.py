import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import curated_collab3_8b_review_watch_v1 as watch
import curated_collab3_8b_review_manager_v1 as manager
import curated_collab3_review_adapter_v1 as aggregation
import test_curated_collab3_review_adapter_v1 as fixtures


class FakeNative:
    @staticmethod
    def review_queue(plan,results,representative_limit=3):
        items=[]
        for sid,result in results.items():
            if result['status']=='confirmed_error':
                items.append(dict(source_id=sid,status=result['status'],errors=result['errors']))
        return dict(items=items,optional_uncertain_items=[],all_error_source_ids=[x['source_id'] for x in items])


class FakeReviewer:
    def __init__(self,*args,**kwargs):
        self.calls=[]
        self.code={'sha256':'native-code'}
        self.model='/local-independent-8b'
    def identity(self):return {'sha256':'independent8B'}
    def __call__(self,row,root,prepare_only=False):
        self.calls.append((row['id'],row['label_sha256']))
        path=Path(root)/'review/receipt.json'
        aggregation.atomic(path,dict(source_id=row['id'],label_sha256=row['label_sha256']))
        return dict(source_id=row['id'],status='confirmed_error',errors=[dict(error_type='hallucination',frame_indices=[1,2])],
                    label_sha256=row['label_payload_sha256'],review_receipt=aggregation.reference(path))


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.AdapterTests.setUpClass()
    @classmethod
    def tearDownClass(cls):fixtures.AdapterTests.tearDownClass()
    def setUp(self):
        self.fixture=fixtures.AdapterTests('test_live_append_dedup_and_exact_original_sampling');self.fixture.setUp()
        self.f=self.fixture
        self.root=self.f.out
        self.native=FakeNative();self.reviewer=FakeReviewer()
        self.rows=self.f.originals[:500]
        selection=self.f.review.sampling_plan(self.rows,'train-b000064')['random_source_ids']
        self.even=next(r for r in self.rows if r['id'] in selection and r['source_ordinal']%2==0)
        self.odd=next(r for r in self.rows if r['id'] in selection and r['source_ordinal']%2==1)
    def tearDown(self):self.fixture.tearDown()
    def watcher(self,gpu=6):
        return watch.Watcher(self.f.adapter,self.root,self.f.work/('review-gpu%d'%gpu),gpu,self.reviewer,self.native)
    def ready(self,row):
        self.f.consume(row);self.f.adapter.poll(self.root)

    def test_both_cards_share500policy_but_source_owners_never_overlap(self):
        self.ready(self.even);self.ready(self.odd)
        left=self.watcher(6);right=self.watcher(7)
        lv=left.poll(run=True);rv=right.poll(run=True)
        self.assertEqual(len(self.reviewer.calls),2)
        self.assertEqual({sid for sid,_ in self.reviewer.calls},{self.even['id'],self.odd['id']})
        self.assertEqual(lv['individually_reviewed'],1);self.assertEqual(rv['individually_reviewed'],1)
        for lane in (left,right):
            summary=lane.state['batches']['train-b000064']['summary']
            self.assertEqual(summary['source_count'],500);self.assertEqual(summary['sampling_denominator'],500)
            self.assertEqual(len(summary['sampling']['random_source_ids']),25)
            self.assertEqual(len(summary['sampling']['risk_source_ids']),5)
            self.assertFalse(summary['full_batch_review_complete']);self.assertFalse(summary['training_eligible'])
            self.assertFalse(summary['all_source_labels_completed'])
        self.assertEqual(left.state['batches']['train-b000064']['summary']['sampling'],right.state['batches']['train-b000064']['summary']['sampling'])

    def test_live_selected_arrival_without_waiting_all500(self):
        self.f.produce();self.f.adapter.poll(self.root)
        lane=self.watcher();self.assertEqual(lane.poll(run=True)['individually_reviewed'],0)
        self.assertEqual(self.reviewer.calls,[])
        self.ready(self.even)
        self.assertEqual(lane.poll(run=True)['individually_reviewed'],1)
        self.assertEqual(len(self.reviewer.calls),1)

    def test_dedup_native_receipt_and_restart_never_redispatch(self):
        self.ready(self.even);lane=self.watcher();lane.poll(run=True)
        lane.poll(run=True);self.assertEqual(len(self.reviewer.calls),1)
        self.ready(self.odd);lane.poll(run=True)
        self.assertEqual(len(self.reviewer.calls),1)
        state=json.loads((lane.out/'state.json').read_bytes())
        restored=watch.Watcher(self.f.adapter,self.root,lane.out,6,self.reviewer,self.native,state)
        restored.poll(run=True);self.assertEqual(len(self.reviewer.calls),1)

    def test_label_revision_does_not_reuse_old_error_score(self):
        self.ready(self.even);lane=self.watcher();lane.poll(run=True)
        self.f.consume(self.even,text='new revision step');self.f.adapter.poll(self.root)
        lane.poll(run=True)
        self.assertEqual(len(self.reviewer.calls),2)
        self.assertNotEqual(self.reviewer.calls[0][1],self.reviewer.calls[1][1])
        self.assertEqual(len(lane.state['receipts']),2)
        web=json.loads((lane.out/'batches/train-b000064/review_web.json').read_bytes())
        item=web['items'][0]
        self.assertEqual(item['label_file_ref']['sha256'],self.reviewer.calls[-1][1])
        self.assertFalse(item['human_review_completed']);self.assertFalse(item['source_quality_pass'])

    def test_ai_error_kept_with_exact_native_rgb_roi_revision(self):
        self.ready(self.even);lane=self.watcher();value=lane.poll(run=True)
        self.assertEqual(value['AI_error_candidate_count'],1)
        entry=next(iter(lane.state['receipts'].values()))
        record=json.loads(Path(entry['result_ref']['path']).read_bytes())
        self.assertTrue(record['AI_errors_are_candidates_not_human_truth'])
        self.assertIn('raw_rgb_manifest',record['source_binding']);self.assertIn('roi_ref',record['source_binding'])
        self.assertFalse(record['training_eligible'])

    def test_stale_label_before_submit_does_not_call_model(self):
        self.ready(self.even);lane=self.watcher()
        self.f.consume(self.even,text='changed without refreshed input')
        with self.assertRaises(ValueError):lane.poll(run=True)
        self.assertEqual(self.reviewer.calls,[])

    def test_immutable_reference_cache_not_rehashing_old_csv_each_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original.csv';path.write_bytes(b'original immutable CSV')
            reference=aggregation.reference(path);calls=[]
            stat={'size':path.stat().st_size}
            def file_identity(p):calls.append(str(p));return dict(**reference,stat=stat)
            native=SimpleNamespace(batch=SimpleNamespace(file_identity=file_identity,snapshot=lambda p:stat))
            watch.install_reference_cache(native)
            native.source_references(dict(source_refs={'csv':reference}))
            native.source_references(dict(source_refs={'csv':reference}))
            self.assertEqual(len(calls),1)
            native.batch.snapshot=lambda p:{'size':99999}
            with self.assertRaises(ValueError):native.source_references(dict(source_refs={'csv':reference}))

    def test_native_receipt_orphan_cpu_adoption_never_model_dispatch(self):
        self.ready(self.even);lane=self.watcher()
        manifest=json.loads((self.root/'batches/train-b000064/review-required.json').read_bytes())
        sid=self.even['id'];binding=manifest['exact_label_bindings'][sid]
        source=next(r for r in manifest['records'] if r['id']==sid)
        root=lane.out/'versions'/binding['exact_version_review_key']/'native'
        evidence=dict(binding=dict(source_id=sid,label_file_ref=binding['label_file_ref'],
                                   label_sha256=binding['label_payload_sha256'],input_row_sha256=self.f.review.batch.digest(source)),
                      frames=[],hand_rois=[])
        aggregation.atomic(root/'evidence.json',evidence)
        aggregation.atomic(root/'evidence.receipt.json',{'sha256':aggregation.sha(root/'evidence.json')})
        native_binding=dict(model_sha256='independent8B',code_sha256='native-code',label_file_ref=binding['label_file_ref'],
                            label_sha256=binding['label_payload_sha256'],evidence_sha256=aggregation.sha(root/'evidence.json'))
        aggregation.atomic(root/'review/receipt.json',dict(binding=native_binding))
        native=SimpleNamespace(read_json=lambda p:aggregation.strict(Path(p).read_bytes()),batch=self.f.review.batch,
                               review_request=lambda *args:{'request':'cpu-reconstructed'},validate_review=lambda v,e:v,
                               review_queue=self.native.review_queue,
                               core=SimpleNamespace(transaction=lambda *args:dict(status='candidate',json=dict(status='confirmed_error',errors=[]))))
        lane.native=native
        lane.poll(run=True)
        self.assertEqual(self.reviewer.calls,[])
        self.assertEqual(len(lane.state['receipts']),1)

    def test_cpu_cli_does_not_import_or_load_native_model(self):
        self.ready(self.even)
        out=self.f.work/'cpu-cli'
        args=['--gpu','6','--plan',str(self.f.plan),'--plan-sha256',aggregation.sha(self.f.plan),
              '--input-root',str(self.root),'--out',str(out),'--model','/not-staged',
              '--model-reference','/not-staged-ref','--annotator-model-reference','/unused-teacher',
              '--ms-source',str(RT/'source_curated_review_v2')]
        with patch.object(watch,'native_review',side_effect=AssertionError('native/model import forbidden')):
            value=watch.main(args)
        self.assertEqual(value['individually_reviewed'],0)
        self.assertEqual(value['exact_version_receipt_count'],0)
        self.assertFalse(value['training_eligible'])

    def test_cross_lane_direct_claim_rejected(self):
        self.ready(self.odd)
        manifest=json.loads((self.root/'batches/train-b000064/review-required.json').read_bytes())
        row=next(r for r in manifest['records'] if r['id']==self.odd['id'])
        ref=aggregation.reference(self.root/'batches/train-b000064/review-required.json')
        with self.assertRaises(ValueError):self.watcher(6).audit(ref,row,manifest['exact_label_bindings'][row['id']],True)
        self.assertEqual(self.reviewer.calls,[])

    def test_partial_unknown_never_training_approved(self):
        notice,stream=self.f.consume(self.even,complete=False,publish=False)
        path=Path(notice['joint_candidate_ref']['path']);label=json.loads(path.read_bytes())
        label['label']['dance_style']='unknown';label['style_status']='unknown'
        notice['joint_candidate_ref']=self.f.put(path,label)
        self.f.put(stream.parent/'sources/train'/self.even['id']/'consumer.terminal.json',notice)
        self.f.append(stream,notice);self.f.adapter.poll(self.root)
        lane=self.watcher(6);value=lane.poll(run=True)
        summary=lane.state['batches']['train-b000064']['summary']
        self.assertFalse(summary['all_source_labels_completed']);self.assertFalse(summary['training_eligible'])
        self.assertFalse(summary['source_quality_pass']);self.assertFalse(summary['full_batch_review_complete'])
        self.assertFalse(value['training_eligible'])

    def test_initial_two_plus_one_budget_then_explicit_full_policy_resume(self):
        selected=self.f.review.sampling_plan(self.rows,'train-b000064')['random_source_ids']
        ready=[r for r in self.rows if r['id'] in selected][:10]
        for row in ready:self.f.consume(row)
        self.f.adapter.poll(self.root)
        lanes=[]
        for gpu,limit in ((6,2),(7,1)):
            grant=self.f.work/('continue%d.json'%gpu)
            lane=watch.Watcher(self.f.adapter,self.root,self.f.work/('limited%d'%gpu),gpu,self.reviewer,self.native,
                               initial_source_review_limit=limit,continuation_authority=grant)
            lane.poll(run=True);lanes.append((lane,grant))
            self.assertEqual(len(lane.state['initial_review_sources']),limit)
            self.assertTrue(lane.state['batches']['train-b000064']['pending_review_budget'])
        self.assertEqual(len(self.reviewer.calls),3)
        for lane,grant in lanes:
            aggregation.atomic(grant,dict(schema='curated_collab3_review_continuation_v1',node='N7',gpu=lane.gpu,
                               plan_ref=self.f.adapter.plan_ref,authorized=True,fixed_full500_random25_risk5=True))
            lane.poll(run=True)
            self.assertEqual(lane.state['batches']['train-b000064']['summary']['sampling_denominator'],500)
        self.assertEqual(len(self.reviewer.calls),10)
        for lane,_ in lanes:lane.poll(run=True)
        self.assertEqual(len(self.reviewer.calls),10)

    def test_zero_model_cpu_prepare_mode(self):
        self.ready(self.even);lane=self.watcher()
        import fcntl,subprocess
        with patch.object(os,'kill',side_effect=AssertionError('signal forbidden')), \
             patch.object(subprocess,'Popen',side_effect=AssertionError('model/process forbidden')), \
             patch.object(fcntl,'flock',side_effect=AssertionError('lock forbidden')):
            value=lane.poll(run=False)
        self.assertEqual(value['individually_reviewed'],0);self.assertEqual(self.reviewer.calls,[])
        self.assertFalse(value['training_eligible'])

    def test_crash_after_result_before_state_adopts_without_call(self):
        self.ready(self.even);lane=self.watcher();lane.poll(run=True)
        lane.state['receipts'].clear();lane.state['batches'].clear();lane.input_stat=None
        lane.poll(run=True)
        self.assertEqual(len(self.reviewer.calls),1)
        self.assertEqual(len(lane.state['receipts']),1)


class ManagerSafetyTests(unittest.TestCase):
    def owner(self,gpu=6):
        return dict(pid=10001,uid=os.getuid(),argv=['/opt/venv_eval/bin/python','-u',str(manager.base.HOLDER)],
                    start_ticks=123,cgroup='test',CVD=['CUDA_VISIBLE_DEVICES=%d'%gpu],exe='/usr/bin/python3',
                    sig_ign=0,sig_blk=0,gpu=gpu,uuid='GPU-exact-%d'%gpu)

    def test_holder_uid_start_uuid_term_and_sole_app_are_enforced(self):
        original=self.owner();current=copy.deepcopy(original)
        running=[dict(pid=original['pid'],uuid=original['uuid'])]
        manager.base.check_holder(original,current,6,original['uuid'],running)
        for field in ('uid','argv','start_ticks','CVD'):
            modified=copy.deepcopy(current);modified[field]='changed'
            with self.assertRaises(RuntimeError):manager.base.check_holder(original,modified,6,original['uuid'],running)
        import signal
        modified=copy.deepcopy(current);modified['sig_ign']=1<<(signal.SIGTERM-1)
        with self.assertRaises(RuntimeError):manager.base.check_holder(original,modified,6,original['uuid'],running)
        with self.assertRaises(RuntimeError):manager.base.check_holder(original,current,6,'GPU-wrong',running)
        with self.assertRaises(RuntimeError):manager.base.check_holder(original,current,6,original['uuid'],running+running)

    def test_real_node_not_fake_alias(self):
        with patch.object(manager.subprocess,'check_output',return_value='wrong-physical-interface'):
            with self.assertRaises(ValueError):manager.node()

    def test_capture_source_model_checks_precede_fresh_holder(self):
        events=[];owner=self.owner();budget={'headroom':manager.base.SAFETY+manager.base.LOAD}
        def event(name,value=None):
            def func(*args,**kwargs):events.append(name);return value
            return func
        with patch.object(manager,'sources',side_effect=event('sources',{})), \
             patch.object(manager,'model_preflight',side_effect=event('model',{})), \
             patch.object(manager,'node',side_effect=event('node',{})), \
             patch.object(manager,'lock_inventory',return_value={'active_lock':False}), \
             patch.object(manager.base,'lease_snapshot',return_value=None), \
             patch.object(manager.base,'uuid_for',return_value=owner['uuid']), \
             patch.object(manager.base,'apps',return_value=[dict(uuid=owner['uuid'],pid=owner['pid'])]), \
             patch.object(manager.base,'identity',side_effect=event('holder',owner)), \
             patch.object(manager.base,'headroom',return_value=budget):
            value=manager.capture(6,dict(path='model',sha256='8b'),dict(path='teacher',sha256='32b'))
        self.assertLess(events.index('sources'),events.index('holder'));self.assertLess(events.index('model'),events.index('holder'))
        self.assertFalse(value['signals_sent']);self.assertFalse(value['lease_modified']);self.assertEqual(value['gpus'],[6])

    def test_capture_low_headroom_zero_release(self):
        owner=self.owner(7)
        with patch.object(manager,'sources',return_value={}),patch.object(manager,'model_preflight',return_value={}), \
             patch.object(manager,'node',return_value={}),patch.object(manager,'lock_inventory',return_value={}), \
             patch.object(manager.base,'lease_snapshot',return_value=None),patch.object(manager.base,'uuid_for',return_value=owner['uuid']), \
             patch.object(manager.base,'apps',return_value=[dict(uuid=owner['uuid'],pid=owner['pid'])]), \
             patch.object(manager.base,'identity',return_value=owner),patch.object(manager.base,'headroom',return_value={'headroom':0}), \
             patch.object(manager.base,'release_holder',side_effect=AssertionError('release forbidden')):
            with self.assertRaises(ValueError):manager.capture(7,{}, {})

    def test_checkpoint_alias_does_not_create_independence(self):
        left=dict(root='/model/teacher',files={'one.safetensors':{'sha256':'same-actual-weight'}})
        right=dict(root='/model/reviewer8b-alias',files={'one.safetensors':{'sha256':'same-actual-weight'}})
        self.assertEqual(manager.checkpoint(left),manager.checkpoint(right))
        right['files']['one.safetensors']['sha256']='distinct-actual-weight'
        self.assertNotEqual(manager.checkpoint(left),manager.checkpoint(right))

    def test_capture_sole_gpu_app_failure_zero_signal(self):
        with patch.object(manager,'sources',return_value={}),patch.object(manager,'model_preflight',return_value={}), \
             patch.object(manager,'node',return_value={}),patch.object(manager,'lock_inventory',return_value={}), \
             patch.object(manager.base,'lease_snapshot',return_value=None),patch.object(manager.base,'uuid_for',return_value='UUID'), \
             patch.object(manager.base,'apps',return_value=[]),patch.object(os,'kill',side_effect=AssertionError('signal forbidden')):
            with self.assertRaises(ValueError):manager.capture(7,{}, {})

    def test_flock_acquire_rechecks_exact_captured_lease(self):
        capture=dict(gpus=[6],leases=[None])
        with patch.object(manager,'ADMITTED_CAPTURE',capture),patch.object(manager,'original_acquire_lease') as actual:
            manager.acquire_captured_lease(dict(gpu=6,old=None));actual.assert_called_once()
            with self.assertRaises(ValueError):manager.acquire_captured_lease(dict(gpu=7,old=None))
        changed=SimpleNamespace(st_dev=1,st_ino=2,st_uid=0,st_mode=3,st_size=0,st_mtime_ns=4)
        with patch.object(manager,'ADMITTED_CAPTURE',capture),patch.object(manager,'original_acquire_lease') as actual:
            with self.assertRaises(ValueError):manager.acquire_captured_lease(dict(gpu=6,old=changed))
            actual.assert_not_called()

    def test_missing_local_stage_and_wrong_node_fail_before_gpu(self):
        with patch.object(manager.base,'uuid_for',side_effect=AssertionError('GPU query forbidden')):
            with self.assertRaises((ValueError,FileNotFoundError)):manager.capture(6,dict(path='/missing/stage.json',sha256='bad'),{})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stage.json';aggregation.atomic(path,dict(node='N6',complete=True,full_copy_SHA_verified=True))
            with self.assertRaises(ValueError):manager.model_preflight(aggregation.reference(path),{})

    def test_explicit_authorization_cannot_be_omitted(self):
        with self.assertRaises(ValueError):manager.config(6,{}, {},{},False)
        with self.assertRaises(ValueError):manager.validate(dict(node='N6',gpus=[6],release_authorized=True))
        with self.assertRaises(ValueError):manager.validate(dict(node='N7',gpus=[6],release_authorized=False))


if __name__=='__main__':unittest.main()
