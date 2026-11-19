"""CPU-only compact protocol parity; no models, GPUs, capture or production writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

RT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('caption_compact_visual_v1', RT / 'caption_compact_visual_v1.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CompactVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bar = c.native()

    def setUp(self):
        self.indices = [101, 205, 309, 413]
        self.manifest = dict(window=dict(window_id='w00001', start=2.0, end=6.0),
                             frames=[dict(index=i, time=2.0 + n) for n, i in enumerate(self.indices)])
        support = {'head': self.indices, 'torso': self.indices, 'left_arm': [101, 205],
                   'right_arm': [205, 309], 'left_hand': [101, 205], 'right_hand': [205, 309]}
        descriptions = [
            ('The dancer bends the torso sideways while the left arm swings outward.', ['torso', 'left_arm'], [101, 205]),
            ('The left hand spreads the fingers and curls them twice.', ['left_hand'], [101, 205]),
            ('The right hand rotates at the wrist with the thumb extended.', ['right_hand'], [205, 309]),
            ('The left hand spreads the fingers and curls them twice.', ['left_hand'], [101, 205]),
            ('The head tilts and the torso rebounds.', ['head', 'torso'], [309, 413])]
        self.value = dict(events=[dict(id='e' + str(i), description=text, parts=parts, frame_indices=frames)
                                  for i, (text, parts, frames) in enumerate(descriptions, 1)],
                          part_support=[dict(part=p, frame_indices=list(support.get(p, [])),
                            reason='visible in cited original frames' if p in support else 'unknown; cropped out')
                                        for p in self.bar.PARTS],
                          uncertainties=['Footwork is unknown because feet are cropped out.',
                              'The right hand shape is indistinguishable at frame 413.'], abstain=False)

    def compact(self, value):
        by_part = {p['part']: p for p in value['part_support']}
        return dict(events=[{k: copy.deepcopy(v) for k, v in e.items() if k != 'id'} for e in value['events']],
                    part_support=[dict(frame_indices=copy.deepcopy(by_part[p]['frame_indices'])) for p in self.bar.PARTS],
                    uncertainties=copy.deepcopy(value['uncertainties']), abstain=value['abstain'])

    def parity(self, value, forbidden=None):
        original = self.bar.filter_supported_observation(value, self.manifest, forbidden)
        compact = self.compact(value)
        before = copy.deepcopy(compact)
        result = c.expand_and_filter(compact, self.manifest, forbidden)
        self.assertEqual(compact, before)
        self.assertEqual(original['events'], result['json']['events'])
        self.assertEqual(original['support_filter_audit'], result['json']['support_filter_audit'])
        for key in ('uncertainties', 'abstain'):
            self.assertEqual(original[key], result['json'][key])
        self.assertEqual([(p['part'], p['frame_indices']) for p in original['part_support']],
                         [(p['part'], p['frame_indices']) for p in result['json']['part_support']])
        return result, original

    def test_01_schema_fixed_order_and_global_indices(self):
        schema = c.observation_schema(self.indices)
        self.assertEqual(schema['title'], c.RECIPE)
        support = schema['properties']['part_support']
        self.assertEqual((support['minItems'], support['maxItems'], len(support['prefixItems'])), (10, 10, 10))
        for row in support['prefixItems']:
            self.assertEqual(row['required'], ['frame_indices'])
            self.assertEqual(set(row['properties']), {'frame_indices'})
            self.assertEqual(row['properties']['frame_indices']['items']['enum'], self.indices)
        event = schema['properties']['events']['items']
        self.assertEqual(set(event['properties']), {'description', 'parts', 'frame_indices'})
        self.assertEqual(event['properties']['parts']['items']['enum'], list(self.bar.PARTS))
        self.assertEqual(event['properties']['frame_indices']['minItems'], 2)
        for indices in ([101], [101, 101], list(range(17))):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                c.observation_schema(indices)

    def test_02_legal_combined_repeat_and_left_right_hand_facts_parity(self):
        self.bar.validate_observation(self.value, self.manifest)
        result, original = self.parity(self.value)
        self.assertEqual(len(result['json']['events']), 5)
        self.assertEqual(result['json']['events'][1]['description'], result['json']['events'][3]['description'])
        windows = [self.manifest['window']]
        expected = self.bar.facts_from_observations(windows, [dict(status='candidate', json=original)])
        actual = self.bar.facts_from_observations(windows, [result])
        self.assertEqual(actual, expected)
        new_facts = c.facts_from_observations(windows, [result])
        self.assertEqual([f['text'] for f in new_facts], [f['text'] for f in expected])
        for old, new in zip(expected, new_facts):
            self.assertEqual(new['source'], c.RECIPE)
            self.assertFalse(new['native_visual_model_protocol'])
            for key in ('fact_id', 'raw_text', 'parts', 'frame_indices', 'source_clause'):
                self.assertEqual(old[key], new[key])

    def test_03_empty_abstain_unknown_not_stationary_or_visibility_groundtruth(self):
        self.value.update(events=[], abstain=True)
        for row in self.value['part_support']:
            row.update(frame_indices=[], reason='anatomical side/visibility unknown')
        self.bar.validate_observation(self.value, self.manifest)
        result, _ = self.parity(self.value)
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(c.facts_from_observations([self.manifest['window']], [result]), [])
        self.assertTrue(all(p['reason'] == 'derivedtag:empty_support_unknown' for p in result['json']['part_support']))
        self.assertFalse(result['derivation']['support_reason_is_VLM_original_reason'])
        self.assertFalse(result['derivation']['support_reason_is_visibility_ground_truth'])
        self.assertTrue(result['derivation']['unknown_support_is_not_stationary'])
        for flag in ('semantic_approval', 'human_review_completed', 'training_eligible', 'quality_approved'):
            self.assertFalse(result[flag])

    def test_04_full_native_clause_filter_preserves_reliable_hand_not_unsupported_feet(self):
        self.value['events'] = [dict(id='e1', description='The left hand opens the fingers. Both feet step forward.',
                                    parts=['left_hand', 'left_foot', 'right_foot'], frame_indices=[101, 205])]
        result, _ = self.parity(self.value)
        self.assertEqual(result['json']['events'][0]['description'], 'The left hand opens the fingers.')
        self.assertEqual(result['expanded_native_value']['events'][0]['description'], self.value['events'][0]['description'])
        self.assertEqual(len(result['json']['support_filter_audit']['rejected_events']), 1)
        self.assertTrue(result['derivation']['original_complete_support_filter_called'])

    def test_05_duplicate_nonchronological_bool_outside_event_indices_native_rejected(self):
        for frames in ([101, 101], [205, 101], [True, 205], [101, 999], [101]):
            with self.subTest(frames=frames):
                self.value['events'] = [dict(id='e1', description='The left hand opens.',
                                            parts=['left_hand'], frame_indices=frames)]
                result, _ = self.parity(self.value)
                self.assertEqual(result['status'], 'abstain')
                self.assertTrue(result['json']['support_filter_audit']['rejected_events'])

    def test_06_duplicate_or_unknown_event_parts_and_wrong_side_native_rejected(self):
        for text, parts in [('The left hand opens.', ['left_hand', 'left_hand']),
                            ('The left hand opens.', ['not_a_part']),
                            ('The right hand opens.', ['left_hand']),
                            ('Both hands open.', ['left_hand']),
                            ('The left hand opens.', ['left_arm'])]:
            with self.subTest(text=text, parts=parts):
                self.value['events'] = [dict(id='e1', description=text, parts=parts, frame_indices=[101, 205])]
                result, _ = self.parity(self.value)
                self.assertEqual(result['status'], 'abstain')

    def test_07_each_required_part_needs_two_own_distinct_frames(self):
        self.value['events'] = [dict(id='e1', description='The torso bends while the left hand opens.',
                                    parts=['torso', 'left_hand'], frame_indices=[101, 309])]
        result, _ = self.parity(self.value)
        self.assertEqual(result['status'], 'abstain')
        self.assertIn('own two visible evidence frames', result['json']['support_filter_audit']['rejected_events'][0]['reason'])

    def test_08_mask_denies_support_or_event_and_half_open_manifest_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'raw 2D mask'):
            c.expand_and_filter(self.compact(self.value), self.manifest, {'left_hand': {205}})
        self.value['events'] = [dict(id='e1', description='The left hand opens.',
                                    parts=['left_hand'], frame_indices=[101, 205, 309])]
        result, _ = self.parity(self.value, {'left_hand': {309}})
        self.assertEqual(result['status'], 'abstain')
        self.assertIn('raw 2D mask', result['json']['support_filter_audit']['rejected_events'][0]['reason'])
        self.manifest['frames'][-1]['time'] = 6.0
        with self.assertRaisesRegex(ValueError, 'half-open window'):
            c.expand_and_filter(self.compact(self.value), self.manifest)

    def test_09_bad_support_and_schema_or_duplicate_json_keys_not_repaired(self):
        compact = self.compact(self.value)
        for frames in ([101, 101], [205, 101], [101, 999], [True, 205]):
            bad = copy.deepcopy(compact)
            bad['part_support'][0]['frame_indices'] = frames
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                c.expand_and_filter(bad, self.manifest)
        bad_values = []
        bad = copy.deepcopy(compact); bad['extra'] = 1; bad_values.append(bad)
        bad = copy.deepcopy(compact); bad['events'][0]['id'] = 'e1'; bad_values.append(bad)
        bad = copy.deepcopy(compact); bad['part_support'][0]['part'] = 'head'; bad_values.append(bad)
        bad = copy.deepcopy(compact); bad['part_support'].pop(); bad_values.append(bad)
        bad = copy.deepcopy(compact); bad['abstain'] = 1; bad_values.append(bad)
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.expand_and_filter(bad, self.manifest)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            c.expand_and_filter('{"events":[],"events":[]}', self.manifest)
        with self.assertRaisesRegex(ValueError, 'compact recipe results'):
            c.facts_from_observations([self.manifest['window']], [dict(status='candidate', json=self.value)])

    def test_10_request_only_adds_contract_schema_identity_keeps_original_pixel_bytes_budget(self):
        request = dict(model='/existing/model', messages=[dict(role='system', content=self.bar.SYSTEM),
            dict(role='user', content=[dict(type='text', text='Global frames: 101,205,309,413; original ROI metadata'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,YWJj'))])],
            sampling=dict(temperature=0.0, seed=0, max_tokens=1800),
            json_schema=self.bar.observation_schema(self.indices),
            prompt_sha256=hashlib.sha256(self.bar.SYSTEM.encode()).hexdigest())
        before = copy.deepcopy(request)
        new = c.compact_request(request, self.manifest)
        self.assertEqual(request, before)
        self.assertEqual(new['model'], request['model'])
        self.assertEqual(new['messages'][1], request['messages'][1])
        self.assertEqual(new['sampling'], request['sampling'])
        self.assertTrue(new['messages'][0]['content'].startswith(self.bar.SYSTEM))
        self.assertEqual(new['caption_recipe'], c.RECIPE)
        self.assertFalse(new['native_visual_model_protocol'])
        self.assertEqual(new['prompt_sha256'], hashlib.sha256(new['messages'][0]['content'].encode()).hexdigest())
        for mutate in ('sampling', 'json_schema', 'system'):
            bad = copy.deepcopy(request)
            if mutate == 'sampling': bad['sampling']['max_tokens'] = 1200
            elif mutate == 'json_schema': bad['json_schema'] = c.observation_schema(self.indices)
            else: bad['messages'][0]['content'] += 'changed'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                c.compact_request(bad, self.manifest)
        with self.assertRaises(ValueError):
            c.compact_request(new, self.manifest)

    def test_11_byte_savings_not_token_or_gpu_claim_and_new_id_order_explicit(self):
        wire = lambda v: json.dumps(v, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        compact = self.compact(self.value)
        self.assertLess(len(wire(compact)), len(wire(self.value)))
        result = c.expand_and_filter(compact, self.manifest)
        self.assertEqual([e['id'] for e in result['expanded_native_value']['events']], ['e1', 'e2', 'e3', 'e4', 'e5'])
        self.assertEqual([e['description'] for e in result['expanded_native_value']['events']],
                         [e['description'] for e in self.value['events']])
        self.assertIn('response_array_order_not_measured_times', result['derivation']['event_ids'])
        self.assertFalse(result['native_visual_model_protocol'])
        self.assertTrue(all(p['reason'].startswith('derivedtag:') for p in result['json']['part_support']))


if __name__ == '__main__':
    unittest.main()
