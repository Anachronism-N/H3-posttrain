"""CPU request-only tests; synthetic pixels, no media/GPU or dispatch."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
spec = importlib.util.spec_from_file_location('motion', RT / 'caption_motion_grounded_request_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MotionGroundedRequestTests(unittest.TestCase):
    def setUp(self):
        self.bar = m.compact.native()
        self.manifest = dict(window=dict(window_id='w00001', start=0.0, end=3.0),
            frames=[dict(index=74, time=0.2), dict(index=121, time=1.2)])
        self.request = dict(model='/existing/frozen/model', messages=[dict(role='system', content=self.bar.SYSTEM),
            dict(role='user', content=[dict(type='text', text='Source frames 74,121; original ROI label is unreviewed'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,Ym9keQ==')),
                dict(type='text', text='left_hand ROI, same global frame 74, NOT second temporal frame'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,Uk9J'))])],
            sampling=dict(temperature=0.0, seed=0, max_tokens=1800),
            json_schema=self.bar.observation_schema([74, 121]),
            prompt_sha256=hashlib.sha256(self.bar.SYSTEM.encode()).hexdigest())

    def versions(self):
        return [self.request, m.compact.compact_request(self.request, self.manifest)]

    def test_native_and_compact_preserve_full_user_pixels_roi_schema_model_budget(self):
        for old in self.versions():
            with self.subTest(schema=old['json_schema'].get('title')):
                before = copy.deepcopy(old)
                new = m.prepare_request(old, self.manifest)
                self.assertEqual(old, before)
                for key in ('model', 'sampling', 'json_schema'):
                    self.assertEqual(new[key], old[key])
                self.assertEqual(new['sampling']['max_tokens'], 1800)
                self.assertEqual(new['messages'][1], old['messages'][1])
                self.assertEqual(new['messages'][0]['content'], old['messages'][0]['content'] + m.CONTRACT)
                self.assertEqual(new['prompt_sha256'], hashlib.sha256(new['messages'][0]['content'].encode()).hexdigest())
                for key in ('caption_recipe', 'generation_recipe', 'fact_source_recipe'):
                    self.assertEqual(new[key], m.RECIPE)
                self.assertEqual(new['response_protocol_recipe'], m.compact.RECIPE if 'caption_recipe' in old else self.bar.OUTPUT_SCHEMA)
                for key in ('native_visual_model_protocol', 'quality_approved', 'training_eligible',
                            'semantic_approval', 'human_review_completed'):
                    self.assertFalse(new[key])

    def test_wrong_old_request_schema_prompt_budget_identity_and_reprepare_rejected(self):
        for old in self.versions():
            for kind in ('schema', 'prompt', 'digest', 'budget', 'frames', 'layout', 'extra'):
                bad = copy.deepcopy(old)
                if kind == 'schema': bad['json_schema']['properties']['events']['maxItems'] = 99
                elif kind == 'prompt': bad['messages'][0]['content'] += ' changed'
                elif kind == 'digest': bad['prompt_sha256'] = 'wrong'
                elif kind == 'budget': bad['sampling']['max_tokens'] = 1000
                elif kind == 'frames': bad['json_schema']['properties']['events']['items']['properties']['frame_indices']['items']['enum'] = [0, 1]
                elif kind == 'layout': bad['messages'] = []
                else: bad['previous_attempt'] = 'attempt-0'
                with self.subTest(kind=kind, compact='caption_recipe' in old), self.assertRaises(ValueError):
                    m.prepare_request(bad, self.manifest)
            with self.assertRaises(ValueError):
                m.prepare_request(m.prepare_request(old, self.manifest), self.manifest)
        for user in ({}, dict(role='assistant', content=[]), dict(role='user', content='not-RGB-layout')):
            bad = copy.deepcopy(self.request)
            bad['messages'][1] = user
            with self.subTest(user=user), self.assertRaises(ValueError):
                m.prepare_request(bad, self.manifest)
        bad = self.versions()[1]
        bad['caption_recipe'] = 'other_recipe'
        with self.assertRaises(ValueError):
            m.prepare_request(bad, self.manifest)

    def test_stable_derived_identity_binds_new_prompt_original_request_and_protocol(self):
        for old in self.versions():
            new = m.prepare_request(old, self.manifest)
            again = m.prepare_request(dict(reversed(list(old.items()))), self.manifest)
            self.assertEqual(new['recipe_identity_sha256'], again['recipe_identity_sha256'])
            self.assertEqual(new['recipe_identity']['source_request_sha256'], m.digest(old))
            self.assertEqual(new['recipe_identity']['prompt_sha256'], new['prompt_sha256'])
            self.assertEqual(new['recipe_identity_sha256'], m.digest(new['recipe_identity']))
            for key in ('model', 'user'):
                changed = copy.deepcopy(old)
                if key == 'model': changed['model'] += '-another-model'
                else: changed['messages'][1]['content'][0]['text'] += 'new-source-context'
                self.assertNotEqual(m.prepare_request(changed, self.manifest)['recipe_identity_sha256'], new['recipe_identity_sha256'])
        self.assertNotEqual(m.prepare_request(self.versions()[0], self.manifest)['recipe_identity_sha256'],
                            m.prepare_request(self.versions()[1], self.manifest)['recipe_identity_sha256'])

    def test_contract_motion_endpoint_side_and_evidence_separation_explicit(self):
        text = m.CONTRACT
        for required in ('continuous motion/change', 'not a string of static poses',
                         'Do NOT claim throughout, always, or until frame X',
                         'Descriptions contain motion only: no frame references',
                         'Cite evidence in frame_indices', 'put unresolved visibility, continuity or side details in uncertainties',
                         'side-neutral one arm', 'TWO DISTINCT original BODY-context frames',
                         'Screen left/right is NOT anatomical', 'ROI part labels/proposals are NOT anatomical-side ground truth',
                         'own two distinct global evidence frames', 'uncertain anatomical support slot',
                         'V-shaped forearm is NOT a finger shape', 'Empty support means unknown, NOT immobility',
                         'Never reuse/relabel an old attempt or old admitted facts'):
            with self.subTest(required=required):
                self.assertIn(required, ' '.join(text.split()))

    def test_frozen_compact_and_gates_unchanged_adapter_only_prepares_requests(self):
        expected = {'caption_compact_visual_v1.py': '971c74a024ac40e7138e7a2d774ca906579464fcf8e02538a104ad8bc524b3cd',
            'smalltests/test_caption_compact_visual_v1.py': '8260c1794cd4e99b98ad050a829d46ae5040d9aa972ef82c2a43bc3837dde87b'}
        for path, sha in expected.items():
            self.assertEqual(hashlib.sha256((RT / path).read_bytes()).hexdigest(), sha)
        self.assertLessEqual(len((RT / 'caption_motion_grounded_request_v1.py').read_text().splitlines()), 100)
        self.assertFalse(hasattr(m, 'expand_and_filter'))
        value = dict(events=[dict(description='One arm sweeps outward.', parts=['left_arm'], frame_indices=[74, 121])],
            part_support=[dict(frame_indices=[74, 121] if p == 'left_arm' else []) for p in self.bar.PARTS],
            uncertainties=['Anatomical side needs independent body context review.'], abstain=False)
        before = m.compact.expand_and_filter(value, self.manifest)
        m.prepare_request(self.request, self.manifest)
        self.assertEqual(before, m.compact.expand_and_filter(value, self.manifest))
        with self.assertRaisesRegex(ValueError, 'raw 2D mask'):
            m.compact.expand_and_filter(value, self.manifest, {'left_arm': {74}})


if __name__ == '__main__':
    unittest.main()
