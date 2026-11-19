"""Final binder only; five CPU checks, original files untouched, no VLM runtime."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('final_bind',RT/'curated_clause_salvage_review_bind_v2.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class FinalBinderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=m.support();cls.manifest=cls.w.Manifest(m.MANIFEST)
        cls.state=m.load(RT.parent/'curated_clause_salvage_reviews_v1/state.json')
        with Path(cls.state['input_stream']).open('rb') as f:
            f.seek(cls.state['cursor']['offset']);cls.bad=json.loads(f.readline())
        cls.candidate=cls.w.checked(cls.bad['joint_candidate_ref'])
        cls.visual=cls.w.checked(cls.candidate['visual_result_ref'])
        cls.repair=cls.w.checked(cls.candidate['original_repair_result_ref'])
        cls.oldrepair=cls.w.checked(cls.candidate['original_repair_joint_result_ref'])

    def test_real_nextbad_false_original_flag_and_sha_preserved(self):
        notice=copy.deepcopy(self.bad)
        row=m.bind_notice(notice,self.manifest,self.w)
        delta=row['review_provenance']['review_delta']
        self.assertFalse(delta['review_required'])
        self.assertEqual(delta['pending_status'],'declared_handshape_unsafe')
        self.assertTrue(row['review_provenance']['original_notice_flags']['need_real_review'])
        self.assertTrue(row['review_provenance']['eligibility_disagreement']['oldflag'])
        self.assertFalse(row['review_provenance']['eligibility_disagreement']['newflag'])
        self.assertEqual(row['label_sha256'],self.bad['joint_candidate_ref']['sha256'])
        self.assertEqual(notice,self.bad)

    def test_shape_bare_tokens_but_not_V_arms(self):
        for text in ('Both hands form V.','The right hand makes OK.','Both hands form heart.','The left fist closes.','Both fingers extend.','Both hands maintain an open hand posture.'):
            self.assertTrue(m.named_hand_detail(text),text)
        for text in ('Both arms form a V shape.','Both arms are arranged in V.','The torso sways.'):
            self.assertFalse(m.named_hand_detail(text),text)

    def test_body_plus_detail_entire_source_pending(self):
        visual=copy.deepcopy(self.visual)
        fake=copy.deepcopy(visual['facts'][0]);fake.update(fact_id='wextra:e1',window_id='wextra',raw_text='The torso sways.',parts=['torso'])
        visual['facts'].append(fake)
        visual['window_observations'].append(dict(window=dict(window_id='wextra'),status='candidate'))
        def gate(fact,observation,w):
            return [] if fact['fact_id']=='wextra:e1' else ['coarse_only_no_named_hand_shape_or_finger_detail']
        with patch.object(m,'fact_check',side_effect=gate):
            delta=m.review_delta(visual,self.repair,self.oldrepair,self.w)
        self.assertFalse(delta['review_required'])
        self.assertEqual(delta['pending_status'],'declared_handshape_unsafe')
        self.assertEqual([f['fact_id'] for f in delta['added_supported_body_facts']],['wextra:e1'])

    def test_actual_valid_body_metadata_disagreement_does_not_veto(self):
        eligible=None
        with Path(self.state['input_stream']).open('rb') as f:
            f.seek(self.state['cursor']['offset'])
            for line in f:
                if not line.endswith(b'\n'):break
                notice=json.loads(line)
                row=m.bind_notice(notice,self.manifest,self.w)
                if row['review_provenance']['review_delta']['review_required']:
                    eligible=(notice,row);break
        self.assertIsNotNone(eligible,'actual eligible body candidate required')
        notice,row=eligible
        independent=copy.deepcopy(row['review_provenance']['review_delta'])
        independent['reason']='independent_verified_coarse_body_delta'
        with patch.object(m,'review_delta',return_value=independent):
            changed=m.bind_notice(notice,self.manifest,self.w)
        self.assertTrue(changed['review_provenance']['review_delta']['review_required'])
        self.assertTrue(changed['review_provenance']['eligibility_disagreement'])
        self.assertEqual(changed['need_real_review'],notice['need_real_review'])

    def test_SHA_id_declaredparts_and_ownframes_failclosed(self):
        changed=copy.deepcopy(self.bad);changed['id']='other'
        with self.assertRaises(ValueError):m.bind_notice(changed,self.manifest,self.w)
        changed=copy.deepcopy(self.bad);changed['joint_candidate_ref']['sha256']='0'*64;changed['result_ref']=changed['joint_candidate_ref']
        with self.assertRaises(ValueError):m.bind_notice(changed,self.manifest,self.w)
        fact=next(f for f in self.visual['facts'] if f not in self.repair['facts'])
        observation=copy.deepcopy(next(o for o in self.visual['window_observations'] if o['window']['window_id']==fact['window_id']))
        for support in observation['json']['part_support']:
            if support['part'] in fact['parts']:support['frame_indices']=[]
        with self.assertRaisesRegex(ValueError,'own two visible'):m.fact_check(fact,observation,self.w)

if __name__=='__main__':unittest.main(verbosity=2)
