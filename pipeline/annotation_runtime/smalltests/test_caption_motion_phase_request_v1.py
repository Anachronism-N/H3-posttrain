"""CPU-only phase-request contracts; synthetic pixels, no GPU/media or old-output edits."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_motion_phase_request_v1 as p


class MotionPhaseRequestTests(unittest.TestCase):
    def setUp(self):
        self.bar = p.compact.native()
        self.manifest = dict(window=dict(window_id='w00001', start=0.0, end=3.0),
                             frames=[dict(index=74, time=0.2), dict(index=121, time=1.2)])
        self.request = dict(model='/existing/frozen/model',
            messages=[dict(role='system', content=self.bar.SYSTEM), dict(role='user', content=[
                dict(type='text', text='Source frames 74,121; original left_hand ROI is unreviewed'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,Ym9keQ==')),
                dict(type='text', text='ROI is same global frame 74, not another time observation'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,Uk9J'))])],
            sampling=dict(temperature=0.0, seed=0, max_tokens=1800),
            json_schema=self.bar.observation_schema([74, 121]),
            prompt_sha256=hashlib.sha256(self.bar.SYSTEM.encode()).hexdigest())

    def test_original_pixels_roi_schema_model_budget_unchanged_new_prompt_only(self):
        before = copy.deepcopy(self.request)
        result = p.prepare_request(self.request, self.manifest)
        self.assertEqual(self.request, before)
        self.assertEqual(result['messages'][1], before['messages'][1])
        for key in ('model', 'json_schema', 'sampling'):
            self.assertEqual(result[key], before[key])
        self.assertEqual(result['messages'][0]['content'], self.bar.SYSTEM + p.CONTRACT)
        self.assertEqual(result['prompt_sha256'], hashlib.sha256((self.bar.SYSTEM + p.CONTRACT).encode()).hexdigest())
        self.assertEqual(result['response_protocol_recipe'], self.bar.OUTPUT_SCHEMA)
        for key in ('caption_recipe', 'generation_recipe', 'fact_source_recipe'):
            self.assertEqual(result[key], p.RECIPE)
        for key in ('semantic_approval', 'human_review_completed', 'quality_approved', 'training_eligible',
                    'native_visual_model_protocol'):
            self.assertFalse(result[key])

    def test_stable_content_refs_bind_original_request_manifest_schema_and_new_prompt(self):
        first = p.prepare_request(self.request, self.manifest)
        reordered = p.prepare_request(dict(reversed(list(self.request.items()))), self.manifest)
        self.assertEqual(first['recipe_identity_sha256'], reordered['recipe_identity_sha256'])
        identity = first['recipe_identity']
        self.assertEqual(first['recipe_identity_sha256'], p.digest(identity))
        self.assertEqual(identity['source_request_ref'], {'sha256': p.digest(self.request)})
        self.assertEqual(identity['source_window_manifest_ref'], {'sha256': p.digest(self.manifest)})
        self.assertEqual(identity['response_schema_ref'], {'sha256': p.digest(self.request['json_schema'])})
        self.assertEqual(identity['new_system_prompt_ref'], {'sha256': first['prompt_sha256']})
        changed = copy.deepcopy(self.request);changed['model'] += '-different'
        self.assertNotEqual(first['recipe_identity_sha256'], p.prepare_request(changed, self.manifest)['recipe_identity_sha256'])

    def test_wrong_original_request_and_default_previous_recipe_reprepare_denied(self):
        for kind in ('keys', 'system', 'digest', 'schema', 'budget', 'role', 'messages'):
            bad = copy.deepcopy(self.request)
            if kind == 'keys': bad['old_attempt'] = 'attempt-0'
            elif kind == 'system': bad['messages'][0]['content'] += ' prior prompt'
            elif kind == 'digest': bad['prompt_sha256'] = 'wrong'
            elif kind == 'schema': bad['json_schema']['properties']['events']['maxItems'] = 99
            elif kind == 'budget': bad['sampling']['max_tokens'] = 900
            elif kind == 'role': bad['messages'][1]['role'] = 'assistant'
            else: bad['messages'] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.prepare_request(bad, self.manifest)
        for bad in (None, {}, p.prepare_request(self.request, self.manifest),
                    p.compact.compact_request(self.request, self.manifest)):
            with self.subTest(previous=type(bad).__name__), self.assertRaises(ValueError):
                p.prepare_request(bad, self.manifest)

    def test_explicit_phase_and_finehand_contract_without_oldtext_repair_or_gate_change(self):
        text = ' '.join(p.CONTRACT.split())
        for phrase in ('ONE complete sentence describing ONE observed motion phase', '40 English words',
                       'adjacent supplied BODY-context frames', 'SAME phase may share',
                       'Do NOT append hands opening, palms, fingers, foot contact, gaze',
                       'SEPARATE event', 'TWO DISTINCT source frames', 'minimal necessary supported parts',
                       'never declare that same detail unknown while asserting a definite hand shape',
                       'neutral one arm/another arm', 'metadata still requires independently supported enum entries',
                       'always/throughout/steady/minimal change', 'placeholders, NOT descriptions to emit',
                       'never automatic splitting of old raw text, deletion of old facts'):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        self.assertLessEqual(len((RT / 'caption_motion_phase_request_v1.py').read_text().splitlines()), 100)
        value = dict(events=[dict(id='e1', description='The torso bends sideways.', parts=['torso'], frame_indices=[74, 121]),
                             dict(id='e2', description='The left hand opens the fingers.', parts=['left_hand'], frame_indices=[74, 121])],
            part_support=[dict(part=part, frame_indices=[74, 121] if part == 'torso' else [], reason='unreviewed')
                          for part in self.bar.PARTS], uncertainties=['Fine hand shape unknown.'], abstain=False)
        checked = self.bar.filter_supported_observation(value, self.manifest)
        p.prepare_request(self.request, self.manifest)
        self.assertEqual(checked, self.bar.filter_supported_observation(value, self.manifest))
        self.assertEqual([e['description'] for e in checked['events']], ['The torso bends sideways.'])
        self.assertEqual(value['events'][1]['description'], 'The left hand opens the fingers.')
        self.assertFalse(hasattr(p, 'expand_and_filter'))


if __name__ == '__main__':
    unittest.main()
