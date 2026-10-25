"""Pure JSON text-gate tests; no GPU or final-test data consumption."""
import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
gate = load('tested_gate', ROOT / 'scripts_omnidance/exp/ms/music_planner_text_gate.py')
ev = load('tested_metrics', ROOT / 'scripts_omnidance/exp/ms/music_planner_eval.py')
IDS = [f'case-{i}' for i in range(32)]
GOOD = '[0-2s] The dancer steps left and raises both arms. [2-5.0625s] The dancer pivots right and bends the knees.'
def rows():
    result = []
    for sid in IDS:
        for condition in gate.CONDITIONS:
            text = GOOD + ' ' + sid + ' ' + condition
            native = dict(raw_text=text, new_token_ids=[1, 2], eos_token_ids=[2], native_logits_all_finite=True,
                          actual_forward_count=2, finite_native_logits_forward_count=2)
            result.append(dict(id=sid, condition=condition, raw_text=text, new_token_ids=[1, 2],
                               eos_token_ids=[2], stop_reason='eos', failure=None, native_output=native))
    return result
def text(row, value):
    row['raw_text'] = value
    row['native_output']['raw_text'] = value
def truncate(row):
    row['new_token_ids'] = [1] * 2048
    row['native_output']['new_token_ids'] = row['new_token_ids']
    row['stop_reason'] = 'budget_truncation'
class TextGateCPU(unittest.TestCase):
    def test_complete_format_is_not_semantic_quality(self):
        result = gate.analyze_records(rows(), IDS, ev, 'moss')
        self.assertTrue(result['text_gate_passed'])
        self.assertFalse(result['semantic_choreography_quality_accepted'])
        self.assertIsNone(result['dance_style_accuracy'])
    def test_one_vs_two_truncations(self):
        r = rows(); truncate(r[0])
        self.assertTrue(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
        truncate(r[3])
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_one_vs_two_invalid_windows(self):
        r = rows(); text(r[0], 'The dancer moves without time windows.')
        self.assertTrue(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
        text(r[3], 'The dancer steps without time windows.')
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_severe_sentence_repeat_threshold(self):
        r = rows(); repeat = '[0-2s] Step left. Step left. Step left. [2-5.0625s] Raise arms.'
        text(r[0], repeat)
        self.assertTrue(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
        text(r[3], repeat)
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_earlier_EOS_does_not_mean_natural_stop(self):
        r = rows(); r[0]['new_token_ids'] = [2, 1, 2]; r[0]['native_output']['new_token_ids'] = [2, 1, 2]
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_missing_or_nonfinite_or_inconsistent_MOSS_proof(self):
        for bad in ({'native_logits_all_finite': False}, {'actual_forward_count': 0}, {'finite_native_logits_forward_count': 1}):
            r = rows(); r[0]['native_output'].update(bad)
            self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_incomplete_interface_failure_empty(self):
        r = rows()
        self.assertFalse(gate.analyze_records(r[:-1], IDS, ev, 'moss')['text_gate_passed'])
        r[0]['failure'] = {'message': 'real error'}
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
        r = rows(); text(r[0], '')
        self.assertFalse(gate.analyze_records(r, IDS, ev, 'moss')['text_gate_passed'])
    def test_audio_change_is_not_synchronization(self):
        r = rows()
        for row in r: text(row, GOOD + row['id'])
        result = gate.analyze_records(r, IDS, ev, 'moss')
        self.assertEqual(result['music_dependency']['mismatched']['raw_identical_fraction'], 1)
        self.assertTrue(result['music_dependency_change_is_not_synchronization_score'])
        self.assertFalse(result['semantic_choreography_quality_accepted'])
if __name__ == '__main__':
    unittest.main()
