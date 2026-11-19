"""Synthetic CPU tests only: no model loading, GPU generation, or final report."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts_omnidance/exp/ms/music_planner_final_text_report.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load('final_text_report_test', SCRIPT)
metric = load('final_reference_metric_test', SCRIPT.with_name('music_planner_reference_f1.py'))
evaluator = load('final_text_evaluator_test', SCRIPT.with_name('music_planner_eval.py'))
GOOD = '[0-2s] The dancer steps and lifts arms.\n[2-5.0625s] The dancer turns and bends knees.'


def source(sid):
    return dict(id=sid, source_group_id=sid, audio_sha256='audio-' + sid,
        audio_path='/private/' + sid + '.wav', audio_window_seconds=[0.0, 5.0625],
        target_text=GOOD, target_windows=[{'text': 'first'}, {'text': 'second'}])


def record(row):
    return dict(id=row['id'], source_group_id=row['source_group_id'], condition='original',
        audio_source_id=row['id'], audio_source_group_id=row['source_group_id'],
        audio_sha256=row['audio_sha256'], audio_window_seconds=row['audio_window_seconds'],
        audio_transform='none', audio_ref=dict(path=row['audio_path'], sha256=row['audio_sha256']),
        reference_target=row['target_text'], reference_target_windows=row['target_windows'],
        raw_text=GOOD, failure=None, metrics=evaluator.text_metrics(GOOD))


def selection_fixture():
    selected = dict(training_launch={'path': '/launch', 'sha256': 'launch'},
        full_complete={'path': '/full', 'sha256': 'full'},
        resources_safe={'path': '/safe', 'sha256': 'safe'},
        checkpoint_selection={'path': '/best', 'sha256': 'best'},
        checkpoint_marker={'path': '/checkpoint/trainer_state.json', 'sha256': 'marker'},
        fingerprint='family-fingerprint')
    protocol_ref = {'path': '/protocol', 'sha256': 'protocol'}
    selection_ref = {'path': '/final_selection', 'sha256': 'selection'}
    cfg = dict(selected, job='test', family='music_flamingo', out='/first_test_out',
        protocol=protocol_ref, final_selection=selection_ref)
    claim = dict(family='music_flamingo', out=cfg['out'], automatic_retry=False,
        test_used_for_selection=False, checkpoint_selection=selected['checkpoint_selection'],
        checkpoint_marker=selected['checkpoint_marker'], final_selection=selection_ref)
    identity = dict(fingerprint=selected['fingerprint'], trainer_state=selected['checkpoint_marker'],
                    training_config={'family': 'music_flamingo'})
    manifest = dict(family='music_flamingo', checkpoint_identity=identity,
        checkpoint='/checkpoint', complete=True, attempted=500, test_used_for_checkpoint_selection=False)
    return cfg, claim, manifest, identity, selected, protocol_ref, selection_ref


class FinalTextReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = [source('source-%03d' % i) for i in range(500)]
        cls.records = [record(row) for row in cls.sources]

    def test_same_500_sources_accepted(self):
        report.validate_rows(self.records, self.sources, evaluator)

    def test_duplicate_order_reference_failure_and_audio_swap_rejected(self):
        mutations = (
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            lambda rows: rows.reverse(),
            lambda rows: rows[0].__setitem__('reference_target', 'swapped weak reference'),
            lambda rows: rows[0].__setitem__('reference_target_windows', []),
            lambda rows: rows[0].__setitem__('failure', {'type': 'native_failure'}),
            lambda rows: rows[0].__setitem__('audio_source_id', 'donor'),
            lambda rows: rows.pop(),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                rows = copy.deepcopy(self.records)
                mutate(rows)
                with self.assertRaises(ValueError):
                    report.validate_rows(rows, self.sources, evaluator)

    def test_existing_output_directories_are_not_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            handoff = Path(directory)
            sub = handoff / 'manual_recovery_v1'
            (sub / 'music_flamingo_test_queue').mkdir(parents=True)
            (sub / 'music_flamingo_test_out').mkdir()
            with self.assertRaises(FileNotFoundError):
                report.validate_family(report.Inputs(), handoff, 'music_flamingo',
                    self.sources, {}, {}, {}, {}, evaluator, None)

    def test_duplicate_reference_source_group_rejected(self):
        sources = copy.deepcopy(self.sources)
        sources[1]['source_group_id'] = sources[0]['source_group_id']
        with self.assertRaises(ValueError):
            report.validate_rows(self.records, sources, evaluator)

    def test_incomplete_exit_and_resource_receipts_rejected(self):
        complete = dict(technical_collection_complete=True, complete_native_GPU_outputs_verified=True)
        exited = {'exit_code': 0}
        safe = dict(all_holders_safe=True, locks_released=True, leases_restored=True,
            managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=1)
        report.validate_completion(complete, exited, safe, False)
        for target, key, value in ((complete, 'technical_collection_complete', False),
            (complete, 'complete_native_GPU_outputs_verified', False), (exited, 'exit_code', 1),
            (exited, 'exit_code', False), (safe, 'all_holders_safe', False),
            (safe, 'locks_released', False), (safe, 'admitted_gpu_count', 8)):
            args = [copy.deepcopy(complete), copy.deepcopy(exited), copy.deepcopy(safe)]
            args[[complete, exited, safe].index(target)][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                report.validate_completion(*args, parallel=False)
        with self.assertRaises(ValueError):
            report.validate_completion(complete, exited, dict(safe, admitted_gpu_count=8), True)
        parallel_complete = dict(complete, worker_exit_code=0, all_eight_real_GPU_manifests_verified=True,
                                 world_size=8, source_count=500, each_source_once=True)
        report.validate_completion(parallel_complete, exited, dict(safe, admitted_gpu_count=8), True)

    def test_different_selection_checkpoint_family_and_claim_rejected(self):
        args = selection_fixture()
        report.validate_selection('music_flamingo', *args)
        for position, key, value in ((0, 'fingerprint', 'new-best'),
            (0, 'checkpoint_selection', {'path': '/other-best', 'sha256': 'other'}),
            (1, 'out', '/retry'), (1, 'final_selection', {'path': '/new-selection', 'sha256': 'new'}),
            (2, 'checkpoint', '/checkpoint-other'), (2, 'family', 'moss'), (2, 'complete', False)):
            changed = copy.deepcopy(args)
            changed[position][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.validate_selection('music_flamingo', *changed)

    def test_input_sha_changed_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / 'raw.jsonl'
            raw.write_text('{}\n')
            inputs = report.Inputs()
            expected = dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            inputs.read(raw, expected)
            with self.assertRaises(ValueError):
                inputs.read(raw, dict(expected, sha256='0' * 64))
            out = directory / 'report.json'
            report.save_report(out, {'synthetic': True}, inputs)
            before = out.read_bytes()
            with self.assertRaises(FileExistsError):
                report.save_report(out, {'synthetic': False}, inputs)
            self.assertEqual(out.read_bytes(), before)
            raw.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                report.save_report(directory / 'second.json', {}, inputs)
            self.assertFalse((directory / 'second.json').exists())

    def test_aggregate_formula_and_window_source_pairing(self):
        def case(sid, reversed_windows=False):
            value = metric.compare_text(GOOD, GOOD, evaluator)
            for channel in ('token', 'action_categories', 'body_categories'):
                windows = [metric.precision_recall_f1(1, 1, 1), metric.precision_recall_f1(0, 1, 1)]
                if reversed_windows:
                    windows.reverse()
                for index in range(2):
                    value['windows'][index][channel] = windows[index]
            return {'id': sid, 'agreement': value}
        cases = [case('a'), case('b', True)]
        for indices in ([0, 0], [1, 1], [0, 1]):
            stats = report.bootstrap_stat(cases, 'window_token', indices, metric)
            self.assertEqual(stats['count'], 4)
            self.assertEqual(stats['macro']['f1'], 0.5)
            self.assertEqual(stats['micro']['f1'], 0.5)
        result = report.paired_bootstrap({'first': cases, 'second': copy.deepcopy(cases)}, metric,
                                        replicates=5000, seed=42)
        self.assertEqual(result, report.paired_bootstrap({'first': cases, 'second': copy.deepcopy(cases)},
                                                        metric, replicates=5000, seed=42))
        for channel in ('window_token', 'window_action_categories', 'window_body_categories'):
            for level in ('macro', 'micro'):
                self.assertEqual(result['family_intervals']['first'][channel][level]['f1'],
                                 {'low': 0.5, 'high': 0.5})
        for channels in result['paired_differences'].values():
            for levels in channels.values():
                for values in levels.values():
                    for interval in values.values():
                        self.assertEqual(interval, {'low': 0.0, 'high': 0.0})
        with self.assertRaises(ValueError):
            report.paired_bootstrap({'first': cases, 'second': list(reversed(cases))}, metric, replicates=2)

    def test_exact_full_prediction_duplicate_case_fraction(self):
        texts = [GOOD, GOOD, GOOD, GOOD + ' ', GOOD + ' ', GOOD + '\n']
        cases = [dict(agreement=metric.compare_text(text, GOOD, evaluator),
            prediction_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
            stop_reason='eos', empty_raw=False, diagnostics=evaluator.text_metrics(text)) for text in texts]
        diagnostics = report.summarize_cases(cases, metric)['diagnostics']
        self.assertEqual(diagnostics['unique_full_prediction_count'], 3)
        self.assertEqual(diagnostics['exact_full_prediction_duplicate_case_fraction'], 5 / 6)
        self.assertIn('no normalization', diagnostics['full_prediction_exact_match_definition'])

    def test_compare_text_and_aggregate_not_reimplemented(self):
        value = metric.compare_text(GOOD, GOOD, evaluator)
        self.assertEqual(value['rouge1']['f1'], 1.0)
        invalid = metric.compare_text('No valid timing.', GOOD, evaluator)
        self.assertTrue(all(w['token']['f1'] == 0 for w in invalid['windows']))
        self.assertEqual(report.bootstrap_stat([{'agreement': value}], 'rougeL', [0], metric),
                         metric.aggregate([value['rougeL']]))


if __name__ == '__main__':
    unittest.main()
