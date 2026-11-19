"""Focused CPU-only regressions for immutable clause salvage and review manifests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('salvage_test', RT / 'curated_clause_salvage_offline_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(events):
    return {'events': events, 'part_support': [
        {'part': p, 'frame_indices': [0, 1], 'reason': 'Clearly visible across two frames.'} for p in m.bar.PARTS],
        'uncertainties': [], 'abstain': not events}


def event(eid, text, parts, refs=(0, 1)):
    return {'id': eid, 'description': text, 'parts': parts, 'frame_indices': list(refs)}


def manifest(native=(0, 1)):
    return {'window': {'window_id': 'w00000', 'start': 0.0, 'end': 2.0},
        'frames': [{'index': i, 'time': float(i)} for i in (0, 1)],
        'hand_roi_evidence': {p: {'crops': [{'frame_index': i, 'resized': False} for i in native]}
                              for p in ('left_hand', 'right_hand')}}


class ClauseSalvageRegression(unittest.TestCase):
    def test_mixed_statement_drops_only_uncertain_clause_without_comma_split(self):
        raw = fixture([event('e1', 'The right arm lifts, with the elbow bent. Possibly the right hand forms a heart.',
                             ['right_arm', 'right_hand'])])
        before = copy.deepcopy(raw)
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['description'] for e in value['events']], ['The right arm lifts, with the elbow bent.'])
        self.assertEqual(value['events'][0]['id'], 'e1_c0')
        self.assertEqual(value['events'][0]['parts'], ['right_arm'])
        self.assertEqual(value['events'][0]['frame_indices'], [0, 1])
        self.assertEqual(raw, before)
        self.assertIn('Possibly the right hand forms a heart.', value['uncertainties'])

    def test_true_finger_missing_native_support_drops_only_that_clause(self):
        raw = fixture([event('e1', 'The right hand forms a V-sign. The torso leans forward.', ['right_hand', 'torso']),
                       event('e2', 'The right foot steps forward.', ['right_foot'])])
        value = m.salvage_clauses(raw, manifest((0,)), {})
        self.assertEqual([e['description'] for e in value['events']],
                         ['The torso leans forward.', 'The right foot steps forward.'])
        self.assertEqual([e['id'] for e in value['events']], ['e1_c1', 'e2'])
        self.assertTrue(any('V-sign' in n['raw_source_clause'] for n in value['scope_filter_audit']['rejected_or_internal_clauses']))

    def test_true_finger_requires_support_intersection_not_just_two_crops(self):
        raw = fixture([event('e1', 'The right hand forms a fist.', ['right_hand']),
                       event('e2', 'The torso turns right.', ['torso'])])
        hand = next(p for p in raw['part_support'] if p['part'] == 'right_hand')
        hand['frame_indices'] = [0]
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['id'] for e in value['events']], ['e2'])
        self.assertEqual(hand['frame_indices'], [0])

    def test_coarse_only_does_not_reapprove_fingers_even_with_two_native_frames(self):
        raw = fixture([event('e1', 'The right hand forms a fist.', ['right_hand']),
                       event('e2', 'The torso turns left.', ['torso'])])
        raw['uncertainties'] = ['No clear finger shapes are visible.']
        before = copy.deepcopy(raw)
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['id'] for e in value['events']], ['e2'])
        self.assertEqual(raw, before)
        self.assertIn('No clear finger shapes are visible.', value['uncertainties'])

    def test_unnamed_finger_detail_does_not_discard_independent_body_motion(self):
        raw = fixture([event('e1', 'The fingers are interlaced or touching.', ['right_hand']),
                       event('e2', 'The right arm lifts.', ['right_arm'])])
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['id'] for e in value['events']], ['e2'])

    def test_arm_v_cannot_become_finger_shape_and_other_body_survives(self):
        raw = fixture([event('e1', 'Both arms raise into a V-shape.', ['left_arm', 'right_arm']),
                       event('e2', 'The torso leans left.', ['torso'])])
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['id'] for e in value['events']], ['e2'])
        self.assertFalse(any('hand' in p for e in value['events'] for p in e['parts']))

    def test_duplicate_event_frames_and_missing_left_arm_are_not_repaired(self):
        raw = fixture([event('e1', 'Both arms extend forward.', ['right_arm']),
                       event('e2', 'The right hand opens.', ['right_hand'], (0, 0)),
                       event('e3', 'The torso turns left.', ['torso'])])
        value = m.salvage_clauses(raw, manifest(), {})
        self.assertEqual([e['id'] for e in value['events']], ['e3'])
        self.assertEqual(raw['events'][0]['parts'], ['right_arm'])
        self.assertEqual(raw['events'][1]['frame_indices'], [0, 0])
        bad_manifest = manifest(); bad_manifest['frames'].append(copy.deepcopy(bad_manifest['frames'][0]))
        with self.assertRaises(ValueError):
            m.salvage_clauses(raw, bad_manifest, {})

    def test_empty_abstention_and_missing_actual_frames_stay_truthful(self):
        raw = fixture([])
        self.assertTrue(m.salvage_clauses(raw, manifest(), {})['abstain'])
        bad = manifest(); bad['frames'] = bad['frames'][:1]
        for support in raw['part_support']:
            support['frame_indices'] = [0]
        value = m.salvage_clauses(raw, bad, {})
        self.assertEqual(value['events'], [])
        body = m.offline.concatenate_facts([bad['window']], [])
        self.assertEqual(body['windows'][0], {'window_id': 'w00000', 'caption': '',
                                             'used_fact_ids': [], 'used_boundary_ids': []})

    def test_rewrite_joins_raw_verbatim_exact_ids_no_boundaries(self):
        raw = fixture([event('e1', 'The torso turns left.', ['torso']),
                       event('e2', 'The right arm lifts.', ['right_arm'])])
        value = m.salvage_clauses(raw, manifest(), {})
        obs = [{'window': manifest()['window'], 'status': 'candidate', 'json': value}]
        facts = m.bar.facts_from_observations([manifest()['window']], obs)
        body = m.offline.concatenate_facts([manifest()['window']], facts)
        self.assertEqual(body['windows'][0]['caption'], ' '.join(e['description'] for e in value['events']))
        self.assertEqual(body['windows'][0]['used_fact_ids'], ['w00000:e1', 'w00000:e2'])
        self.assertEqual(body['windows'][0]['used_boundary_ids'], [])
        m.bar.check_supported_rewrite(body, [manifest()['window']], facts, [])

    def attempt_fixture(self, root):
        job = {'global_revision': 'repair-test', 'id': 'test'}
        jobpath = root / 'job.json'; m.q.save_once(jobpath, job)
        binding = {'source_binding': {'id': 'test'}}
        receipt_binding = {**binding, 'repair_revision': job['global_revision'],
                           'window': manifest()['window'], 'frames_sha256': m.bar.batch.digest(manifest()['frames'])}
        request = {'sampling': {'max_tokens': 100}, 'json_schema': {}, 'repair_provenance': {
            'job_ref': m.q.identity(jobpath), 'global_revision': 'repair-test', 'automatic_reference_fill': False}}
        m.q.save_once(root / 'request.json', request)
        m.q.save_once(root / 'response.raw.txt', fixture([event('e1', 'The torso turns.', ['torso'])]))
        m.q.save_once(root / 'binding.json', receipt_binding)
        m.q.save_once(root / 'final.json', {'status': 'failed'})
        gen = {'request_sha256': m.q.sha(root / 'request.json'), 'raw_response_sha256': m.q.sha(root / 'response.raw.txt'),
               'max_tokens': 100, 'model_call_performed': True, 'finish_reason': 'stop', 'completion_tokens': 50,
               'structured_output_schema_sha256': m.bar.core.digest({}), 'model_identity_sha256': 'test-model'}
        m.q.save_once(root / 'generation.json', gen)
        m.q.save_once(root / 'receipt.json', {'status': 'failed', 'binding': receipt_binding,
            'files_sha256': {name: m.q.sha(root / name) for name in
                            ('request.json', 'response.raw.txt', 'generation.json', 'binding.json', 'final.json')}})
        m.q.save_once(root / 'load.json', {'model_identity_sha256': 'test-model'})
        return job, binding, m.q.identity(root / 'load.json')

    def test_raw_or_request_sha_tamper_is_rejected(self):
        for name in ('response.raw.txt', 'request.json'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); job, binding, load = self.attempt_fixture(root)
                value, evidence = m.checked_attempt(root, job, manifest(), binding, load)
                self.assertIsInstance(value, dict)
                self.assertTrue(evidence['actual_frame_SHA_verified'])
                with (root / name).open('ab') as stream:
                    stream.write(b' ')
                with self.assertRaises(m.q.SourceChanged):
                    m.checked_attempt(root, job, manifest(), binding, load)

    def test_incomplete_artifacts_do_not_produce_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value, audit = m.checked_attempt(root, {}, manifest(), {}, {})
            self.assertIsNone(value)
            self.assertEqual(audit['status'], 'not_salvaged')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); job, binding, load = self.attempt_fixture(root)
            with patch.object(m.bar.core, 'validate_generation', side_effect=ValueError('completion truncated')):
                value, audit = m.checked_attempt(root, job, manifest(), binding, load)
            self.assertIsNone(value)
            self.assertEqual(audit['reason'], 'completion truncated')

    def test_immutable_output_budget_flags_and_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'production'; source.mkdir()
            original = source / 'original.json'; m.q.save_once(original, {'status': 'failed'})
            budget = source / 'budget.json'; m.q.save_once(budget, {'consumed': True})
            before = [m.q.sha(p) for p in (original, budget)]
            target = root / 'independent/result.json'
            m.q.save_once(target, m.FLAGS); m.q.save_once(target, m.FLAGS)
            self.assertFalse(m.q.load(target)['training_eligible'])
            self.assertFalse(m.q.load(target)['attempt_budget_consumed'])
            self.assertEqual(m.q.load(target)['model_calls'], 0)
            with self.assertRaises(m.q.SourceChanged):
                m.q.save_once(target, {**m.FLAGS, 'model_calls': 1})
            with self.assertRaises(ValueError):
                m.isolated(source / 'nested', root / 'repair', source)
            self.assertEqual(before, [m.q.sha(p) for p in (original, budget)])

    def test_review_manifest_crash_resume_is_exactly_once_and_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'review_required.jsonl'
            row = {'source_key': 'train/test', 'global_revision': m.DEFAULT_REVISION, **m.FLAGS}
            end = m.append_checkpointed(path, row, 0)
            self.assertEqual(m.append_checkpointed(path, row, 0), end)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with self.assertRaises(m.q.SourceChanged):
                m.append_checkpointed(path, {**row, 'source_key': 'train/changed'}, 0)
            self.assertNotIn('G2', path.read_text())


if __name__ == '__main__':
    unittest.main()
