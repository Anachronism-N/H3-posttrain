"""Real-source CPU replay only: no model runtime, no signals, old transactions untouched."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('clause_watch_v2_test', RT / 'curated_clause_salvage_review_watch_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


class StrictReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = m.w.Manifest(m.pilot.MANIFEST)
        cls.previous = m.previous_reviews()
        cls.snapshot = m.checked(dict(path=str(m.RESUME), sha256=m.RESUME_SHA))
        cls.state = m.load(m.OUT / 'state.json')
        with m.STREAM.open('rb') as f:
            f.seek(cls.snapshot['cursor']['offset'])
            cls.bad = json.loads(f.readline())
        cls.notices = [json.loads(line) for line in m.STREAM.read_bytes().splitlines()]
        cls.false = next(n for n in cls.notices if not n['need_real_review'] and n['source_key'] not in cls.state['sources'])

    def watcher(self, tmp, notices):
        stream = Path(tmp) / 'input.jsonl'
        stream.write_text(''.join(json.dumps(n) + '\n' for n in notices))
        no_call = Mock(side_effect=AssertionError('CPU replay forbids new VLM calls'))
        watch = m.Watcher(Path(tmp) / 'out', self.manifest, no_call, review, stream=stream,
            gate=lambda:None, previous=self.previous)
        return watch, no_call

    def test_bad_source_isolated_then_next_pending_no_call(self):
        original = copy.deepcopy(self.bad)
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, [self.bad, self.false])
            summary = watch.poll(drain=True)
            self.assertTrue(summary['input_drain'])
            self.assertEqual(summary['processed_source_count'], 2)
            self.assertEqual(summary['newsource_error_isolated'], 1)
            self.assertEqual(summary['sampled_sources_reviewed'], 0)
            self.assertEqual(summary['remaining_pending_count'], 2)
            tx = m.checked(watch.state['sources'][self.bad['source_key']]['transaction_ref'])
            row = m.checked(tx['input_ref']); status = m.checked(tx['summary_ref'])
            self.assertEqual(tx['status'], 'source_eligibility_disagreement')
            self.assertTrue(row['review_provenance']['original_notice_flags']['need_real_review'])
            self.assertFalse(row['review_provenance']['review_delta']['review_required'])
            self.assertEqual(m.checked(status['original_notice_ref']), original)
            dispute = m.checked(status['eligibility_disagreement_ref'])
            self.assertTrue(dispute['oldflag']); self.assertFalse(dispute['newflag'])
            self.assertIn('coarse_only_no_named_hand_shape_or_finger_detail',
                dispute['independent_review_eligibility']['rejected_new_facts'][0]['reasons'])
            for flag in m.FLAGS:
                self.assertFalse(status[flag])
            no_call.assert_not_called()
            saved = {key:m.sha(value['transaction_ref']['path']) for key,value in watch.state['sources'].items()}
            watch.process(self.bad)
            self.assertEqual(saved, {key:m.sha(value['transaction_ref']['path']) for key,value in watch.state['sources'].items()})
            resumed = m.Watcher(watch.out,self.manifest,no_call,review,stream=watch.cursor.path,gate=lambda:None,previous=self.previous)
            self.assertEqual(resumed.poll(drain=True)['processed_source_count'],2)
            no_call.assert_not_called()
        self.assertEqual(self.bad, original)

    def test_true_sha_id_and_original_flag_mutation_fatal(self):
        for key in ('id', 'need_real_review'):
            changed=copy.deepcopy(self.bad)
            changed[key] = 'other' if key=='id' else False
            with self.assertRaises(ValueError):
                m.bind_v2.bind_notice(changed,self.manifest,m.w)
        changed=copy.deepcopy(self.bad); changed['joint_candidate_ref']['sha256']='0'*64
        changed['result_ref']=changed['joint_candidate_ref']
        with self.assertRaisesRegex(ValueError,'SHA changed'):
            m.bind_v2.bind_notice(changed,self.manifest,m.w)
        with tempfile.TemporaryDirectory() as tmp:
            watch,no_call=self.watcher(tmp,[changed,self.false])
            with self.assertRaises(ValueError): watch.poll(drain=True)
            self.assertEqual(watch.state['cursor']['offset'],0)
            self.assertEqual(watch.state['sources'],{})
            no_call.assert_not_called()

    def test_shape_uncertainty_and_own_event_frames(self):
        fact=dict(fact_id='w:e1',raw_text='Both hands move outward.',parts=['left_hand','right_hand'],
            frame_indices=[0,1],source_clause=dict(raw_source_clause='Both hands move outward.',original_description='Both hands move outward.'))
        event=dict(id='e1',description=fact['raw_text'],parts=fact['parts'],frame_indices=fact['frame_indices'],source_clause=fact['source_clause'])
        observation=dict(window=dict(window_id='w'),status='candidate',json=dict(events=[event],part_support=[
            dict(part='left_hand',frame_indices=[0,1]),dict(part='right_hand',frame_indices=[0,1])]))
        self.assertEqual(m.bind_v2.fact_check(fact,observation,m.w),['missing_checked_original_raw_evidence'])
        for text in ('Both hands form open hand shapes.','Both hands possibly move outward.','Both fingers extend.'):
            f=copy.deepcopy(fact);o=copy.deepcopy(observation)
            f['raw_text']=text;f['source_clause'].update(raw_source_clause=text,original_description=text)
            o['json']['events'][0].update(description=text,source_clause=f['source_clause'])
            self.assertTrue(m.bind_v2.fact_check(f,o,m.w))
        observation['json']['part_support'][1]['frame_indices']=[2,3]
        self.assertIn('event_required_part_lacks_own_two_visible_frames',m.bind_v2.fact_check(fact,observation,m.w))

    def test_full_prefix_resume_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stream';path.write_bytes(b'{"a":1}\n{"b":2}\n')
            state={};cursor=m.AppendCursor(path,state)
            for _ in range(2): row,line=cursor.next();cursor.commit(line)
            path.write_bytes(b'{"a":9}\n{"b":2}\n')
            with self.assertRaisesRegex(ValueError,'Full committed.*prefix changed'):
                m.AppendCursor(path,copy.deepcopy(state))

    def test_reuse_two_real_pilot_refs_no_new_call(self):
        notices=[n for n in self.notices if n['source_key'] in self.previous]
        with tempfile.TemporaryDirectory() as tmp:
            watch,no_call=self.watcher(tmp,notices)
            legacy=m.module('legacy_pilot_replay',RT/'curated_clause_salvage_review_watch_v1.py',m.sha(RT/'curated_clause_salvage_review_watch_v1.py'))
            previous_watch=legacy.Watcher(watch.out,self.manifest,no_call,review,stream=watch.cursor.path,gate=lambda:None,previous=self.previous)
            previous_watch.poll(drain=True)
            watch=m.Watcher(watch.out,self.manifest,no_call,review,stream=watch.cursor.path,gate=lambda:None,previous=self.previous)
            summary=watch.poll(drain=True)
            self.assertEqual(summary['sampled_sources_reviewed'],2)
            self.assertEqual(summary['imported_previous_review_count'],2)
            no_call.assert_not_called()

    def test_original111_45_and_byteprefix_preflight_no_write(self):
        original_state_sha=m.sha(m.OUT/'state.json')
        proof=m.resume_preflight(m.OUT,self.manifest,review)
        self.assertEqual(proof['original_sources_verified'],111)
        self.assertEqual(proof['original_real_audits_verified'],45)
        self.assertEqual(proof['previous_pilot_refs_verified'],2)
        self.assertEqual(m.sha(m.OUT/'state.json'),original_state_sha)
        no_call=Mock(side_effect=AssertionError('original committed source must not review again'))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir()
            state=copy.deepcopy(self.state)
            (out/'state.json').write_text(json.dumps(state))
            watcher=m.Watcher(out,self.manifest,no_call,review,stream=m.STREAM,gate=lambda:None,previous=self.previous)
            for notice in self.notices:
                if notice['source_key'] in self.state['sources']: watcher.process(notice)
            self.assertEqual(watcher.state['sources'],self.state['sources'])
            no_call.assert_not_called()

    def test_bad_source_continues_to_real_eligible_source_only(self):
        eligible=None
        for notice in self.notices:
            if notice['source_key'] in self.state['sources'] or not notice['need_real_review']: continue
            row=m.bind_v2.bind_notice(notice,self.manifest,m.w)
            if row['review_provenance']['review_delta']['review_required'] and not row['review_provenance']['eligibility_disagreement']:
                eligible=notice;break
        self.assertIsNotNone(eligible,'Need one actual checked new eligible source')
        with tempfile.TemporaryDirectory() as tmp:
            watch,no_call=self.watcher(tmp,[self.bad,eligible])
            no_call.side_effect=RuntimeError('eligible-only mock audit reached')
            with self.assertRaisesRegex(RuntimeError,'eligible-only mock audit reached'): watch.poll(drain=True)
            no_call.assert_called_once()
            self.assertEqual(list(watch.state['sources']),[self.bad['source_key']])
            self.assertEqual(watch.state['sources'][self.bad['source_key']]['status'],'source_eligibility_disagreement')

    def test_pure_rewrite_blank_unknown_window_not_eligible(self):
        fact=dict(raw_text='The torso sways.')
        repair=dict(facts=[fact],final=dict(status='failed'),window_observations=[dict(status='candidate')])
        visual=dict(facts=[fact],window_observations=[dict(status='candidate',window=dict(window_id='w'))],
            final=dict(status='candidate',json=dict(windows=[dict(caption='',used_fact_ids=[])])),final_timed_caption='[w] unknown')
        result=m.bind_v2.review_delta(visual,repair,dict(label=dict(timed_motion_script='old')),m.w)
        self.assertFalse(result['pure_rewrite_failure']);self.assertFalse(result['review_required'])

    def test_completed_recovery_never_yields(self):
        self.assertIsNone(m.recovery_yield())

if __name__=='__main__': unittest.main(verbosity=2)
