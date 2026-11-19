"""Reference metric arithmetic only; no GPU or model quality claims."""
import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
m = load('tested_reference_F1', ROOT / 'scripts_omnidance/exp/ms/music_planner_reference_f1.py')
ev = load('reference_word_parser', ROOT / 'scripts_omnidance/exp/ms/music_planner_eval.py')
class ReferenceF1CPU(unittest.TestCase):
    def test_multiset_overlap_precision_recall_harmonic(self):
        result = m.multiset_f1(['step', 'turn'], ['step', 'step', 'turn'])
        self.assertEqual(result['overlap'], 2)
        self.assertEqual(result['precision'], 1)
        self.assertAlmostEqual(result['recall'], 2 / 3)
        self.assertAlmostEqual(result['f1'], .8)
    def test_set_f1_does_not_count_duplicate_category(self):
        result = m.set_f1(['step', 'step', 'turn'], ['step', 'bend'])
        self.assertEqual((result['precision'], result['recall'], result['f1']), (.5, .5, .5))
    def test_zero_division_no_positive_labels_is_zero(self):
        self.assertEqual(m.multiset_f1([], [])['f1'], 0)
        self.assertEqual(m.set_f1([], ['step'])['f1'], 0)
        self.assertEqual(m.set_f1(['step'], [])['f1'], 0)
    def test_ngrams_and_lcs_order(self):
        self.assertEqual(m.ngrams(['a', 'b', 'c'], 2), [('a', 'b'), ('b', 'c')])
        self.assertEqual(m.lcs_length(['raise', 'arms', 'step'], ['step', 'raise', 'arms']), 2)
    def test_timestamp_format_removed_and_same_content_exact(self):
        a = '[0-2s] Step left. [2-5.0625s] Raise arms.'
        b = '[0.000-2.000s] Step left. [2.000-5.062s] Raise arms.'
        result = m.compare_text(a, b, ev)
        self.assertEqual(result['rouge1']['f1'], 1)
        self.assertEqual(result['rougeL']['f1'], 1)
    def test_swapped_windows_can_match_bag_but_not_window_action(self):
        a = '[0-2s] Raise arms. [2-5.0625s] Step left.'
        b = '[0-2s] Step left. [2-5.0625s] Raise arms.'
        result = m.compare_text(a, b, ev)
        self.assertEqual(result['rouge1']['f1'], 1)
        self.assertLess(result['rougeL']['f1'], 1)
        self.assertEqual([window['action_categories']['f1'] for window in result['windows']], [0, 0])
    def test_missing_predicted_windows_zero_overlap(self):
        result = m.compare_text('Step left, raise arms.', '[0-2s] Step left. [2-5.0625s] Raise arms.', ev)
        self.assertTrue(all(w['action_categories']['f1'] == 0 for w in result['windows']))
    def test_macro_micro_different_supports(self):
        result = m.aggregate([m.precision_recall_f1(1, 1, 1), m.precision_recall_f1(0, 9, 9)])
        self.assertEqual(result['macro']['f1'], .5)
        self.assertAlmostEqual(result['micro']['f1'], .1)
if __name__ == '__main__':
    unittest.main()
