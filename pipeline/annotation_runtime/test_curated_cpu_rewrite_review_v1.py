"""Real fixed-three CPU evidence tests and explicitly synthetic transaction tests.

No GPU/model loading; synthetic uncertain responses only exercise receipt logic,
never assert semantic correctness of the real labels.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import curated_cpu_rewrite_review_v1 as r


class SyntheticReviewer(r.old.ManifestReviewer):
    """No model runtime: writes test-only synthetic uncertain receipt artifacts."""
    calls = 0

    def runtime(self):
        raise AssertionError('A real model runtime must never be invoked in CPU tests')

    def __call__(self, row, root, *, prepare_only=False):
        self.calls += 1
        root = Path(root)
        evidence = r.audit.read_json(root / 'evidence.json')
        identity = self.identity()
        transaction = root / 'review'
        request = r.audit.review_request(row, evidence, root, self.model)
        binding = {**evidence['binding'], 'model_sha256':identity['sha256'], 'code_sha256':self.code['sha256'],
                   'annotator_model_sha256':self._annotator_identity['sha256'],
                   'annotator_model_reference':self._annotator_reference_ref,
                   'checkpoint_content_independence_verified':True,
                   'reviewer_checkpoint_sha256':r.audit.checkpoint_identity(identity),
                   'evidence_sha256':r.batch.sha256(root / 'evidence.json'), 'test_simulation_only':True}
        raw = dict(status='uncertain', errors=[], uncertainties=[dict(claimed_text='TEST ONLY',
                   actual_visual_evidence='Synthetic uncertain protocol fixture; not a video judgment',
                   frame_indices=[evidence['frames'][0]['index']],error_type='boundary',severity='minor',evidence_kind='ambiguous')],
                   coverage_note='Synthetic transaction test only; zero actual model calls')
        checked = r.audit.validate_review(raw,evidence)
        transaction.mkdir(parents=True)
        for name,value in [('request.json',request),('binding.json',binding),('response.raw.txt',raw),
                           ('response.json',checked),('final.json',dict(status='candidate',json=checked))]:
            r.core.save_json(transaction / name,value)
        generation = dict(request_sha256=r.batch.sha256(transaction / 'request.json'),
                          raw_response_sha256=r.batch.sha256(transaction / 'response.raw.txt'),
                          max_tokens=request['sampling']['max_tokens'],model_call_performed=True,
                          structured_output_schema_sha256=r.batch.digest(request['json_schema']),
                          finish_reason='stop',completion_tokens=42,model_identity_sha256=identity['sha256'],
                          test_simulation_only=True,actual_model_calls=0)
        r.q.save_once(transaction / 'generation.json',generation)
        receipt = dict(status='candidate',binding=binding,
                       files_sha256={name:r.batch.sha256(transaction / name) for name in r.old.SIX_FILES},
                       semantic_approval=False,human_review_completed=False,test_simulation_only=True)
        r.q.save_once(transaction / 'receipt.json',receipt)
        return checked


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cpu-rewrite-review-tests-')
        cls.out = Path(cls.temp.name) / 'real-pilot3'
        cls.summary = r.execute(cls.out,prepare_only=True)
        cls.rows,cls.population = r.load_population()
        cls.ledger = [json.loads(line) for line in (cls.out / 'independent.ledger.jsonl').read_bytes().splitlines()]
        cls.originals = []
        cls.visuals = []
        cls.jobs = []
        cls.windows = []
        for candidate in cls.rows:
            job = r.q.checked(candidate['original_job_ref'])
            original = r.q.checked(candidate['original_visual_result_ref'])
            cls.jobs.append(job)
            cls.originals.append(original)
            cls.visuals.append(r.q.checked(candidate['visual_result_ref']))
            cls.windows.append(r.cpu.scope(job,original))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_three_prepare_native_rgb_and_roi(self):
        self.assertEqual(self.summary['statuses'],{'prepared':3})
        self.assertEqual(self.summary['native_source_frames'],48)
        self.assertEqual(self.summary['native_hand_rois'],6)
        self.assertEqual(self.summary['actual_VLM_calls'],0)
        self.assertEqual(self.summary['sampled_reviewed_denominator'],0)
        self.assertFalse(self.summary['independent_sample_review_completed'])
        self.assertFalse(self.summary['GPU_model_load_performed'])
        self.assertEqual([x['evidence_frames'] for x in self.ledger],[16,16,16])
        self.assertEqual([x['native_hand_rois'] for x in self.ledger],[0,6,0])
        self.assertTrue(all(self.summary[k] is False for k in r.NO_APPROVAL))
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('vllm',sys.modules)

    def test_real_resume_cpu_no_dispatch(self):
        before = [(x['evidence_ref']['sha256'],x['input_row_sha256']) for x in self.ledger]
        repeated = r.execute(self.out,prepare_only=True)
        self.assertEqual(repeated['statuses'],{'prepared':3})
        after = [json.loads(line) for line in (self.out / 'independent.ledger.jsonl').read_bytes().splitlines()]
        self.assertEqual(before,[(x['evidence_ref']['sha256'],x['input_row_sha256']) for x in after])
        self.assertEqual(repeated['actual_VLM_calls'],0)

    def test_jqo_rejected_hand_event_keeps_admitted_torso(self):
        original = self.originals[1]
        fact = next(f for f in original['facts'] if f['fact_id']=='w00001:e2_c1')
        self.assertIn('hand open',fact['source_clause']['original_description'])
        self.assertEqual(fact['raw_text'],'The torso is upright.')
        self.assertIn(fact,original['facts'])
        self.assertEqual(len(r.cpu.scope(self.jobs[1],original)),3)

    def test_wrong_candidate_and_sha_zero_dispatch(self):
        candidate = copy.deepcopy(self.rows[0])
        candidate['label']['timed_motion_script'] += ' changed'
        with self.assertRaisesRegex(ValueError,'candidate bytes/source'):
            r.validate_source(candidate,self.population['candidate_refs'][0])
        wrong = dict(self.population['candidate_refs'][0],sha256='0'*64)
        with self.assertRaises(r.q.SourceChanged):
            r.validate_source(self.rows[0],wrong)

    def test_raw_fact_frame_part_mutations_rejected(self):
        for field,value in [('raw_text','Invented dancing'),('frame_indices',[999999]),('parts',['head'])]:
            visual = copy.deepcopy(self.visuals[0])
            visual['facts'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'Facts/raw_text'):
                r.validate_claim(self.rows[0],visual,self.originals[0],self.windows[0],self.rows[0]['style_result'])

    def test_caption_ids_and_missing_window_rejected(self):
        for field,value in [('caption','invented'),('used_fact_ids',[])]:
            visual = copy.deepcopy(self.visuals[0])
            visual['final']['json']['windows'][0][field] = value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'caption must concatenate'):
                r.validate_claim(self.rows[0],visual,self.originals[0],self.windows[0],self.rows[0]['style_result'])
        original = copy.deepcopy(self.originals[0])
        original['window_observations'].pop()
        with self.assertRaises(r.cpu.Pending):
            r.cpu.scope(self.jobs[0],original)

    def test_actual_admitted_shape_and_uncertain_exclusion(self):
        for description in ('The dancer holds a fist.','The torso is perhaps tilted.'):
            original = copy.deepcopy(self.originals[0])
            event = original['window_observations'][0]['json']['events'][0]
            event['description'] = description
            event['source_clause'].update(raw_source_clause=description,original_description=description)
            original['facts'] = r.original_bar.facts_from_observations(self.windows[0],original['window_observations'])
            with self.subTest(description=description),self.assertRaises(r.cpu.Pending):
                r.cpu.scope(self.jobs[0],original)

    def test_original_job_wrong_root_and_split(self):
        candidate = self.rows[0]
        for field,value in [('production','/tmp/wrong-production'),('queue_root','/tmp/wrong-queue'),('split','val')]:
            job = copy.deepcopy(self.jobs[0])
            job[field] = value
            checked = r.q.checked
            def wrong_job(ref):
                return job if ref == candidate['original_job_ref'] else checked(ref)
            with self.subTest(field=field),patch.object(r.q,'checked',side_effect=wrong_job),self.assertRaisesRegex(ValueError,'original job/root'):
                r.validate_source(candidate,self.population['candidate_refs'][0])

    def test_budget_and_approval_flag_rejected(self):
        for field,value in [('attempt_budget_consumed',True),('model_calls',1),('extra_source_attempts',1),('training_eligible',True)]:
            row = copy.deepcopy(self.rows[0])
            row[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                r.validate_claim(row,self.visuals[0],self.originals[0],self.windows[0],row['style_result'])

    def test_three_cannot_claim_original_full_population(self):
        for field,value in [('population_source_count',500),('population_kind','complete448'),('independent_batch_id','train-b000000')]:
            population = copy.deepcopy(self.population)
            population[field] = value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'500/448'):
                r.validate_population(self.rows,population)
        row = copy.deepcopy(self.rows[0])
        row['split'] = 'val'
        with self.assertRaises(ValueError):
            r.validate_population([row]+self.rows[1:],self.population)

    def test_output_protected_roots(self):
        for path in [r.cpu.PRODUCTION,r.CANDIDATES,r.cpu.QUEUE,r.RT,r.BAR,r.OUT/'nested']:
            with self.subTest(path=path),self.assertRaises(ValueError):
                r.check_output(path)

    def test_original_six_receipt_sha_rejected(self):
        observation = self.originals[0]['window_observations'][0]
        with tempfile.TemporaryDirectory(dir=self.temp.name) as tmp:
            attempt = Path(tmp) / 'attempt-0'
            shutil.copytree(observation['attempt_path'],attempt)
            r.cpu.receipt_inventory(attempt,'candidate')
            receipt = r.q.load(attempt / 'receipt.json')
            receipt['files_sha256']['response.raw.txt'] = '0'*64
            (attempt / 'receipt.json').write_bytes(r.q.encoded(receipt))
            with self.assertRaises(r.q.SourceChanged):
                r.cpu.receipt_inventory(attempt,'candidate')

    def source_review_fixture(self):
        candidate = self.rows[1]
        ledger = self.ledger[1]
        verified = dict(probe=ledger['probe'],source_hint=ledger['source_hint'],origin_root=ledger['origin_root'])
        row = r.review_row(candidate,self.population['candidate_refs'][1],r.q.identity(self.out/'frozen.population.json'),verified)
        root = Path(ledger['evidence_ref']['path']).parent
        return row,root,r.audit.read_json(root/'evidence.json')

    def test_native_frame_and_roi_geometry_mutation(self):
        row,root,evidence = self.source_review_fixture()
        for mutate in ('frame','roi'):
            changed = copy.deepcopy(evidence)
            if mutate=='frame':
                changed['frames'][0]['index'] = 999999
            else:
                changed['hand_rois'][0]['crop_xyxy'][0] += 1
            with self.subTest(mutate=mutate),self.assertRaises((ValueError,IndexError)):
                r.verify_native_evidence(changed,row,root,evidence['probe'])

    def test_semantic_only_native_request(self):
        row,root,evidence = self.source_review_fixture()
        request = r.audit.review_request(row,evidence,root,'/tmp/test-only-no-model')
        content = request['messages'][1]['content']
        claim = json.loads(content[0]['text'])
        self.assertEqual(claim['label_to_test'],row['label'])
        self.assertEqual(sum(x['type']=='image_url' for x in content),22)
        text = '\n'.join(x['text'] for x in content if x['type']=='text')
        for forbidden in ('producer_terminal','original_final','model_calls','original_description','source_hint'):
            self.assertNotIn(forbidden,text)

    def test_partial_and_failed_transactions_never_dispatch(self):
        class NeverDispatch:
            calls = 0
            def __call__(self,*args,**kwargs):
                self.calls += 1
                raise AssertionError('Partial transaction must not dispatch')
        for name in ['request.json','response.raw.txt','binding.json','generation.json','final.json','response.json','failed-receipt']:
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=self.temp.name) as tmp:
                root = Path(tmp)
                transaction = root/'review'
                transaction.mkdir()
                if name=='failed-receipt':
                    r.q.save_once(transaction/'receipt.json',dict(status='failed'))
                else:
                    (transaction/name).write_bytes(b'partial')
                reviewer = NeverDispatch()
                with self.assertRaisesRegex(ValueError,'never retry'):
                    r.review_transaction(root,{}, {},reviewer)
                self.assertEqual(reviewer.calls,0)

    def test_synthetic_six_receipt_verified_resume_zero_dispatch(self):
        row,original_root,evidence = self.source_review_fixture()
        with tempfile.TemporaryDirectory(dir=self.temp.name) as tmp:
            tmp = Path(tmp)
            model = tmp/'test-only-independent-8b'
            teacher = tmp/'test-only-teacher'
            for path,weights in [(model,b'test-only-independent'),(teacher,b'test-only-teacher')]:
                path.mkdir()
                (path/'config.json').write_text('{}')
                (path/'weights.bin').write_bytes(weights)
            teacher_identity = r.batch.model_identity(teacher)
            teacher_ref = tmp/'teacher.reference.json'
            r.q.save_once(teacher_ref,dict(model_identity=teacher_identity))
            row = copy.deepcopy(row)
            row['annotator_model_sha256'] = teacher_identity['sha256']
            root = tmp/'synthetic-test-only'
            root.mkdir()
            evidence = copy.deepcopy(evidence)
            evidence['binding']['input_row_sha256'] = r.batch.digest(row)
            for frame in evidence['frames']:
                target = root/frame['file']
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original_root/frame['file'],target)
            r.q.save_once(root/'evidence.json',evidence)
            with r.review_library():
                reviewer = SyntheticReviewer(model,tmp/'model-out',annotator_model_reference=teacher_ref)
                checked,receipt,calls = r.review_transaction(root,row,evidence,reviewer)
                self.assertEqual(checked['status'],'uncertain')
                self.assertFalse(checked['semantic_approval'])
                self.assertEqual(calls,1)
                resumed,receipt2,calls2 = r.review_transaction(root,row,evidence,reviewer)
                self.assertEqual(checked,resumed)
                self.assertEqual(receipt['sha256'],receipt2['sha256'])
                self.assertEqual(calls2,0)
                self.assertEqual(reviewer.calls,1)
                broken = r.q.load(root/'review/receipt.json')
                broken['files_sha256']['request.json'] = '0'*64
                (root/'review/receipt.json').write_bytes(r.q.encoded(broken))
                with self.assertRaisesRegex(ValueError,'artifact SHA'):
                    r.review_transaction(root,row,evidence,reviewer)
                self.assertEqual(reviewer.calls,1)
                self.assertNotIn('vllm',sys.modules)

    def test_shared_stable_sha_stat_cache_only(self):
        with tempfile.TemporaryDirectory(dir=self.temp.name) as tmp:
            path = Path(tmp)/'immutable.plan.json'
            path.write_bytes(b'plan')
            ref = r.q.identity(path)
            r.cpu.register_shared(ref)
            self.assertEqual(r.cpu.cached_sha(path),ref['sha256'])
            path.write_bytes(b'changed')
            with self.assertRaises(r.q.SourceChanged):
                r.cpu.cached_sha(path)


if __name__=='__main__':
    unittest.main(verbosity=2)
