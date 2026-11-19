"""Offline facts derivation regressions; real sources only read, optional writes isolated."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('offline_test', RT / 'curated_rewrite_facts_offline_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
OUT = RT.parent / 'curated222939_repair_v1'
REV = 'offline-facts-20261025-v1'
KEY = 'train/RPItfOOsPbLXBhXC'


class OfflineFactsRegression(unittest.TestCase):
    def setUp(self):
        self.ref = m.q.identity(OUT / 'jobs' / (KEY + '.json'))
        self.job = m.q.checked(self.ref)
        self.original = m.q.checked(self.job['refs']['visual_result'])

    def test_actual_failed_preserved_no_dispatch_no_budget(self):
        paths = [Path(self.job['refs']['visual_result']['path']), OUT / 'worker.state.json']
        budget = OUT / 'attempt-budget' / (KEY + '.json')
        before = {str(p): m.q.sha(p) if p.exists() else None for p in paths + [budget]}
        with patch.object(m.q, 'repair_visual', side_effect=AssertionError('no VLM')):
            ref, joint = m.derive(self.ref, REV)
        self.assertIsNone(ref)
        self.assertTrue(joint['joint_complete'])
        self.assertEqual(joint['original_final'], self.original['final'])
        self.assertEqual(joint['original_final']['status'], 'failed')
        self.assertEqual(joint['extra_source_attempts'], 0)
        self.assertFalse(joint['model_rewrite_performed'])
        self.assertFalse(joint['attempt_budget_consumed'])
        self.assertFalse(joint['training_eligible'])
        self.assertEqual(before, {str(p): m.q.sha(p) if p.exists() else None for p in paths + [budget]})

    def test_verbatim_raw_text_exact_fact_ids_and_no_boundaries(self):
        job, row, style, visual, provenance, original_joint = m.candidate(self.ref, REV)
        for item in visual['final']['json']['windows']:
            facts = [f for f in self.original['facts'] if f['window_id'] == item['window_id']]
            self.assertEqual(item['caption'], ' '.join(f['raw_text'] for f in facts))
            self.assertEqual(item['used_fact_ids'], [f['fact_id'] for f in facts])
            self.assertEqual(item['used_boundary_ids'], [])
        m.bar.check_supported_rewrite(visual['final']['json'],
                                     [o['window'] for o in self.original['window_observations']], self.original['facts'], [])

    def test_empty_window_stays_empty_and_joint_incomplete(self):
        result = list(m.candidate(self.ref, REV))
        visual = copy.deepcopy(result[3])
        windows = [o['window'] for o in self.original['window_observations']]
        facts = [f for f in self.original['facts'] if f['window_id'] != windows[0]['window_id']]
        body = m.concatenate_facts(windows, facts)
        self.assertEqual(body['windows'][0]['caption'], '')
        self.assertEqual(body['windows'][0]['used_fact_ids'], [])
        self.assertEqual(body['windows'][0]['used_boundary_ids'], [])
        visual['facts'] = facts
        visual['final']['json'] = body
        visual['coverage']['full_window_caption'] = False
        visual['final_timed_caption'] = m.bar.core.timed_text({'windows': windows}, body['windows'])
        result[3] = visual
        with patch.object(m, 'candidate', return_value=tuple(result)):
            _, joint = m.derive(self.ref, REV)
        self.assertFalse(joint['joint_complete'])
        self.assertFalse(joint['all_windows_nonempty'])
        self.assertEqual(joint['target_text'], '')

    def test_reobserve_and_wrong_job_sha_rejected(self):
        ref = m.q.identity(OUT / 'jobs/train/YHdOyRCqljUFfUJV.json')
        with self.assertRaises(ValueError):
            m.derive(ref, REV)
        with self.assertRaises(ValueError):
            m.derive({**self.ref, 'sha256': '0' * 64}, REV)

    def test_admitted_fact_tamper_rejected(self):
        visual_ref = self.job['refs']['visual_result']
        original_checked = m.q.checked
        changed = copy.deepcopy(self.original)
        changed['facts'][0]['raw_text'] += ' invented action'
        with patch.object(m.q, 'checked', side_effect=lambda ref: changed if ref == visual_ref else original_checked(ref)):
            with self.assertRaises(ValueError):
                m.derive(self.ref, REV)

    def test_write_isolated_and_idempotent_no_original_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, joint = m.derive(self.ref, REV, tmp)
            again, second = m.derive(self.ref, REV, tmp)
            self.assertEqual(ref, again)
            self.assertEqual(joint, second)
            self.assertEqual(m.q.checked(ref), joint)
            self.assertEqual(m.q.checked(joint['visual_result_ref'])['original_final'], self.original['final'])
        with self.assertRaises(ValueError):
            m.derive(self.ref, REV, OUT / 'offline-bad-overlap')


if __name__ == '__main__':
    unittest.main(verbosity=2)
