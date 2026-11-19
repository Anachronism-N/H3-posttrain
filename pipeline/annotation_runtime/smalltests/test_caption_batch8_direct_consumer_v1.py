"""Native rolling-ready CPU contracts; synthetic media/model only, no resources."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_batch8_direct_consumer_v1 as consumer
from test_caption_batch8_direct_full_pilot_v1 import FullFixture, FakeLLM, bar, pipeline


class ReadyFixture(FullFixture):
    def __init__(self, base, empty=False):
        super().__init__(base, empty_window=empty)
        self.args.producer_root = str(self.original)
        self.args.selected_sources = {sid: 90000 + i for i, sid in enumerate(self.sources)}
        self.args.style_reuse = None
        records = []
        manifest_path = Path(self.entries[0]['row']['source_refs']['manifest']['path'])
        for i, entry in enumerate(self.entries):
            entry['row']['manifest_record']['source_ordinal'] = self.args.selected_sources[self.sources[i]]
            records.append(entry['row']['manifest_record'])
        manifest_path.write_bytes(b''.join(pipeline.raw_json(r) for r in records))
        for entry in self.entries:
            source = pipeline.source_dir(self.original, entry['row']['manifest_record'])
            plan = entry['row']
            plan['source_refs']['manifest'] = pipeline.ref(manifest_path)
            plan_path = source / 'bar_plan.json'
            plan_path.write_bytes(pipeline.raw_json(plan))
            terminal_path = source / 'producer.terminal.json'
            terminal = pipeline.load(terminal_path)
            terminal['plan_ref'] = pipeline.ref(plan_path)
            terminal_path.write_bytes(pipeline.raw_json(terminal))
        self.assignment_path = Path(base) / 'exclusive-assignment.json'
        self.assignment = dict(schema='caption_direct_lane_assignment_v1', exclusive_source_owner=True,
            producer_root=str(self.original.resolve()), out=str(self.out.resolve()),
            selected_sources=self.args.selected_sources, mode='new_unassigned')
        self.assign(self.assignment)
        self.events = []
        self.llm = FakeLLM()
        self.runtime = consumer.Runtime(self.runtime_provider, self.events.append)
        self.refs = {}
        self.styles_by_id = dict(zip(self.sources, self.styles))

    def runtime_provider(self):
        return self.llm, object

    def assign(self, value):
        self.assignment_path.write_bytes(pipeline.raw_json(value))
        self.args.assignment_ref = pipeline.ref(self.assignment_path)

    def run_batch(self, entries=None, style_lookup=None):
        entries = entries or list(consumer.iter_ready(self.args, self.runtime, set()))
        return consumer.run_ready_batch(entries, self.args, self.runtime, self.refs,
                                        style_lookup or (lambda e: self.styles_by_id[e['row']['id']]))


class ReadyBatchTests(unittest.TestCase):
    def test_three_sources_eight_windows_native_batch8_and_direct_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            original = {p: p.read_bytes() for p in f.original.rglob('*') if p.is_file()}
            with f.context(), f.runtime.scope(f.args):
                result = f.run_batch()
                self.assertEqual([v['status'] for v in result], ['joint_candidate'] * 3)
            self.assertEqual([len(c) for c in f.llm.calls], [8])
            self.assertEqual(len(f.decoder_calls), 3)
            self.assertEqual({p: p.read_bytes() for p in original}, original)
            self.assertEqual(len(list(f.out.glob('visual/sources/*/windows/*/attempt-0/receipt.json'))), 8)
            self.assertFalse(list(f.out.rglob('rewrite')))
            self.assertFalse(list(f.out.rglob('boundaries')))
            self.assertEqual(len([e for e in f.events if e['event'] == 'native_transactions_timing']), 7)
            self.assertGreaterEqual(len([e for e in f.events if e['event'] == 'native_core_transaction' and
                                         e['receipt_resume_full_validation'] is True]), 8)
            for value in result:
                joint = pipeline.load(value['joint_candidate_ref']['path'])
                self.assertTrue(joint['target_text'].startswith('Dance style: '))
                self.assertEqual(joint['consumer_recipe'], consumer.RECIPE)
                self.assertTrue(all(not joint[key] for key in consumer.FLAGS))
            self.assertFalse(bar._fast_source_active)
            self.assertFalse(bar._transaction_cache_v1_active)

    def test_same_thread_cpu_prefetch_reused_no_extra_decode_or_model_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                for entry in entries:
                    consumer.prepare_ready(entry, f.args, f.runtime, f.refs)
                self.assertEqual(f.llm.calls, [])
                self.assertEqual(len(f.decoder_calls), 3)
                result = f.run_batch(entries)
            self.assertEqual(len(f.decoder_calls), 3)
            self.assertEqual([len(c) for c in f.llm.calls], [8])
            self.assertTrue(all(r['joint_complete'] for r in result))

    def test_missing_style_uses_original_independent_rgb_annotator_not_music(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            import ms_dance_style as style
            from types import SimpleNamespace as NS
            class WithStyle(FakeLLM):
                def chat(self, messages, sampling, use_tqdm=False):
                    if messages[0][0]['content'] == bar.SYSTEM:
                        return super().chat(messages, sampling, use_tqdm)
                    if not all(m[0]['content'] == style.SYSTEM for m in messages):
                        raise AssertionError('Only native independent RGB style allowed')
                    self.calls.append(copy.deepcopy(messages))
                    value = dict(dance_style='unknown', frame_indices=[], reason='CPU uncertainty', abstain=True)
                    return [NS(outputs=[NS(text=json.dumps(value), token_ids=[7, 8], finish_reason='stop', stop_reason=None)],
                               prompt_token_ids=[1, 2, 3]) for m in messages]
            f.llm = WithStyle()
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                result = consumer.run_ready_batch(entries, f.args, f.runtime, f.refs, lambda e: None)
            self.assertEqual([len(c) for c in f.llm.calls], [8, 1, 1, 1])
            self.assertTrue(all(r['status'] == 'partial_candidate' for r in result))
            self.assertTrue(all(pipeline.load(r['joint_candidate_ref']['path'])['style_result']['source'] == 'independent_RGB' for r in result))

    def test_bad_source_only_source_error_others_commit_no_window_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                entries[1]['row']['binding']['video_sha256'] = 'f' * 64
                result = f.run_batch(entries)
            self.assertEqual([v['status'] for v in result], ['joint_candidate', 'source_error', 'joint_candidate'])
            self.assertEqual([len(c) for c in f.llm.calls], [5])
            self.assertFalse(list(f.out.rglob('attempt-1')))

    def test_failed_window_and_unknown_style_remain_partial_not_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            f.llm = FakeLLM(bad_source='source0')
            f.styles_by_id['source1'] = dict(id='source1', status='unknown', dance_style='unknown')
            with f.context(), f.runtime.scope(f.args):
                result = f.run_batch()
            self.assertEqual([v['status'] for v in result], ['partial_candidate', 'partial_candidate', 'joint_candidate'])
            self.assertFalse(result[0]['joint_complete'])
            self.assertFalse(result[1]['joint_complete'])
            self.assertEqual([len(c) for c in f.llm.calls], [8])

    def test_authorized_old_complete_terminal_skipped_no_prepare_or_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            old = f.original / 'old-terminal.json'
            old.write_bytes(pipeline.raw_json(dict(id='source0', joint_complete=True)))
            f.assign(dict(f.assignment, mode='handoff', old_consumer_stopped=True,
                          old_terminals={'source0': pipeline.ref(old)}))
            seen = set()
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, seen))
                self.assertEqual([e['row']['id'] for e in entries], ['source1', 'source2'])
                self.assertEqual(seen, {'source0'})
                result = f.run_batch(entries)
            self.assertEqual(len(result), 2)
            self.assertEqual([len(c) for c in f.llm.calls], [5])
            self.assertFalse((f.out / 'visual/sources/source0').exists())

    def test_selection_mismatch_no_output_and_default_unassigned_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            f.args.selected_sources['source0'] += 100
            with f.context(), self.assertRaisesRegex(ValueError, 'exclusive selected'):
                with f.runtime.scope(f.args):
                    self.fail('must reject before scope')
            self.assertFalse(f.out.exists())
            with self.assertRaisesRegex(ValueError, 'runtime.scope'):
                f.run_batch()

    def test_old_recipe_out_hot_resume_duplicate_source_and_terminal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    f.run_batch([entries[0], entries[0]])
                entries[0]['binding']['params_sha256'] = 'old-recipe'
                with self.assertRaisesRegex(ValueError, 'hot-resume'):
                    f.run_batch(entries)
                entries[0]['binding']['params_sha256'] = bar.batch.digest(f.runtime.configuration)
                result = f.run_batch(entries)
                with self.assertRaisesRegex(ValueError, 'already committed'):
                    f.run_batch(entries)
                seen = {r['id'] for r in result}
                self.assertEqual(list(consumer.iter_ready(f.args, f.runtime, seen)), [])
            with f.context(), self.assertRaisesRegex(ValueError, 'fresh isolated'):
                with f.runtime.scope(f.args):
                    pass
            self.assertEqual([len(c) for c in f.llm.calls], [8])

    def test_source_boundary_before_native_result_and_terminal_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            original_timed = bar.core.timed_text
            changed = []
            def mutate(*args, **kwargs):
                value = original_timed(*args, **kwargs)
                if not changed:
                    changed.append(True)
                    Path(f.entries[0]['row']['manifest_record']['video_path']).write_bytes(b'changed CPU source')
                return value
            with f.context(), f.runtime.scope(f.args), patch.object(bar.core, 'timed_text', side_effect=mutate):
                result = f.run_batch()
            self.assertEqual(result[0]['status'], 'source_error')
            self.assertFalse(list((f.out / 'visual/sources/source0/results').glob('*.json')))
            self.assertFalse((f.out / 'sources/train/source0/consumer.terminal.json').exists())
            self.assertEqual([len(c) for c in f.llm.calls], [8])

    def test_failed_receipts_stay_failed_on_final_retry_with_cached_native_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            f.llm = FakeLLM(bad_source='source0')
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                def style_failure(entry):
                    if entry['row']['id'] == 'source0':
                        raise ValueError('final publication unavailable')
                    return f.styles_by_id[entry['row']['id']]
                first = f.run_batch(entries, style_failure)
                self.assertEqual(first[0]['status'], 'source_error')
                receipts = {p: p.read_bytes() for p in (f.out / 'visual/sources/source0').glob('windows/*/attempt-0/receipt.json')}
                clean_llm = FakeLLM()
                f.llm = clean_llm
                second = f.run_batch(entries[:1])
                self.assertEqual(second[0]['status'], 'partial_candidate')
                self.assertFalse(second[0]['joint_complete'])
                self.assertEqual(clean_llm.calls, [])
                self.assertEqual({p: p.read_bytes() for p in receipts}, receipts)
                self.assertTrue(all(pipeline.load(p)['status'] == 'failed' for p in receipts))
            self.assertFalse(list(f.out.rglob('attempt-1')))

    def test_final_source_failure_reuses_committed_window_receipts_no_redispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ReadyFixture(tmp)
            with f.context(), f.runtime.scope(f.args):
                entries = list(consumer.iter_ready(f.args, f.runtime, set()))
                def style_failure(entry):
                    if entry['row']['id'] == 'source0':
                        raise ValueError('source final failure')
                    return f.styles_by_id[entry['row']['id']]
                result = f.run_batch(entries, style_failure)
                self.assertEqual(result[0]['status'], 'source_error')
                second = f.run_batch(entries[:1])
                self.assertEqual(second[0]['status'], 'joint_candidate')
            self.assertEqual([len(c) for c in f.llm.calls], [8])
            self.assertFalse(list(f.out.rglob('attempt-1')))


if __name__ == '__main__':
    unittest.main()
