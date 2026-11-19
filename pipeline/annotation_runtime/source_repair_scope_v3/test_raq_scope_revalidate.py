import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/source_curated_consumer_v1'))
import ms_bar_annotation as bar
import raq_scope_revalidate as scope

CASE = ROOT / 'legacy_raq_repair_v1/revisions/raq-demo-20261025-v1/train/RAQoxqRxCfMvTHSx'
RAW = json.loads((CASE / 'windows/w00001/response.raw.txt').read_text())
JOB = json.loads((CASE / 'original_job.json').read_text())
MANIFEST = json.loads(Path(JOB['refs']['demo_manifest']['path']).read_text())
BODY = RAW['events'][0]['description'].split('. ')[0] + '.'


class ScopeTests(unittest.TestCase):
    def run_value(self, raw=None, manifest=None):
        original = copy.deepcopy(RAW if raw is None else raw)
        result = scope.validate_reobserve(original, MANIFEST if manifest is None else manifest, {}, bar)
        self.assertEqual(original, RAW if raw is None else raw)
        return result

    def mixed(self, detail, hand='left_hand'):
        raw = copy.deepcopy(RAW)
        raw['events'] = [{'id': 'e1', 'description': BODY + ' ' + detail,
                          'parts': ['left_arm', 'right_arm', hand],
                          'frame_indices': [26, 28, 36, 38]}]
        for support in raw['part_support']:
            if support['part'] == hand:
                support.update(frame_indices=[36, 38], reason='visible')
        return raw

    def test_frozen_validator_reproduces_false_rejection(self):
        queue = scope.import_frozen_queue(ROOT)
        with self.assertRaisesRegex(ValueError, 'Named finger shape requires two distinct actual native ROI frames'):
            queue.validate_reobserve(copy.deepcopy(RAW), MANIFEST, {}, bar)

    def test_real_raw_preserves_only_verbatim_body_and_original_refs(self):
        result = self.run_value()
        self.assertEqual([e['description'] for e in result['events']], [BODY])
        event = result['events'][0]
        self.assertEqual(event['id'], 'e1_c0')
        self.assertEqual(event['parts'], ['left_arm', 'right_arm'])
        self.assertEqual(event['frame_indices'], RAW['events'][0]['frame_indices'])
        self.assertEqual(event['source_clause']['raw_source_clause'], BODY)
        self.assertEqual(len(result['scope_filter_audit']['internal_detail_clauses']), 2)
        self.assertEqual(result['part_support'], RAW['part_support'])
        self.assertFalse(result['abstain'])

    def test_negative_shape_does_not_require_empty_right_roi(self):
        result = self.run_value(self.mixed('No clear V/OK/heart/fist shape is distinguishable.', 'right_hand'))
        self.assertEqual([e['description'] for e in result['events']], [BODY])
        self.assertEqual(MANIFEST['hand_roi_evidence']['right_hand']['crops'], [])

    def test_possibly_is_not_deleted_to_create_a_fact(self):
        detail = 'The left hand possibly forms a fist.'
        result = self.run_value(self.mixed(detail))
        self.assertEqual([e['description'] for e in result['events']], [BODY])
        self.assertIn(detail, result['uncertainties'])
        self.assertFalse(result['scope_filter_audit']['uncertainty_tokens_removed_to_strengthen_claim'])

    def test_negated_shape_contraction_is_not_affirmative(self):
        detail = "The left hand doesn't form a fist."
        result = self.run_value(self.mixed(detail))
        self.assertEqual([e['description'] for e in result['events']], [BODY])
        self.assertIn(detail, result['uncertainties'])

    def test_true_right_shape_without_roi_is_rejected_not_body(self):
        result = self.run_value(self.mixed('The right hand forms a clear fist.', 'right_hand'))
        self.assertEqual([e['description'] for e in result['events']], [BODY])
        self.assertTrue(any(e.get('strict_named_shape_gate') for e in result['support_filter_audit']['rejected_events']))

    def test_affirmative_left_shape_two_native_frames_is_structurally_supported(self):
        result = self.run_value(self.mixed('The left hand forms a clear fist.'))
        self.assertEqual(len(result['events']), 2)
        self.assertIn('fist', result['events'][1]['description'])
        self.assertTrue(result['scope_filter_audit']['filter_is_not_semantic_truth_review'])

    def test_affirmative_shape_one_native_frame_still_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest['hand_roi_evidence']['left_hand']['crops'] = manifest['hand_roi_evidence']['left_hand']['crops'][:1]
        result = self.run_value(self.mixed('The left hand forms a clear fist.'), manifest)
        self.assertEqual([e['description'] for e in result['events']], [BODY])

    def test_duplicate_crop_is_not_two_native_frames(self):
        manifest = copy.deepcopy(MANIFEST)
        crop = manifest['hand_roi_evidence']['left_hand']['crops'][0]
        manifest['hand_roi_evidence']['left_hand']['crops'] = [copy.deepcopy(crop), copy.deepcopy(crop)]
        result = self.run_value(self.mixed('The left hand forms a clear fist.'), manifest)
        self.assertEqual([e['description'] for e in result['events']], [BODY])

    def test_both_arms_needs_own_right_arm_two_frames(self):
        raw = copy.deepcopy(RAW)
        next(p for p in raw['part_support'] if p['part'] == 'right_arm')['frame_indices'] = [26]
        result = self.run_value(raw)
        self.assertEqual(result['events'], [])
        self.assertTrue(result['abstain'])

    def test_mask_is_never_bypassed(self):
        with self.assertRaisesRegex(ValueError, 'raw 2D mask forbids'):
            scope.validate_reobserve(copy.deepcopy(RAW), MANIFEST, {'right_arm': {26}}, bar)


if __name__ == '__main__':
    unittest.main(verbosity=2)
