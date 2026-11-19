"""Only v2 check-de-duplication differences; native fixtures and receipts are real CPU paths."""
import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_batch8_production_consumer_v2 as consumer
import caption_direct_facts_v1 as direct
from test_caption_batch8_production_consumer_v1 import ProductionFixture, FakeLLM
from curated_fast_source_v1 import SourceAssets
bar, pipeline = direct.native()


class Fixture(ProductionFixture):
    def run(self, llm=None):
        llm = llm or FakeLLM()
        with self.context():
            report = consumer.run_consumer(self.args, self.runtime(llm), lambda args, runtime: True)
        return report, llm

    def visual(self, sid):
        terminal = pipeline.load(self.out / 'sources/train' / sid / 'consumer.terminal.json')
        joint = self.joint(terminal)
        return pipeline.load(joint['visual_result_ref']['path']), joint


class V2Checks(unittest.TestCase):
    def test_six_sources_two_groups_check_lane_twice_source_once_and_native_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, masked=True)
            verifies, instances, borders, caches = Counter(), {}, Counter(), []
            original_verify, original_border = consumer.Inputs.verify, SourceAssets.verify_boundary
            original_stage = consumer.cross.stage_sources
            def verify(inputs):
                verifies[id(inputs)] += 1
                instances[id(inputs)] = inputs
                return original_verify(inputs)
            def border(cache):
                borders[cache.root.name] += 1
                return original_border(cache)
            def stage(*args, **kwargs):
                caches.append(id(args[3]))
                return original_stage(*args, **kwargs)
            with patch.object(consumer.Inputs, 'verify', verify), patch.object(SourceAssets, 'verify_boundary', border),                     patch.object(consumer.cross, 'stage_sources', side_effect=stage),                     patch.object(bar, 'facts_from_observations', wraps=bar.facts_from_observations) as facts,                     patch.object(direct, 'direct_final', wraps=direct.direct_final) as final, \
                    patch.object(bar, 'filter_supported_observation', wraps=bar.filter_supported_observation) as support:
                report, llm = f.run()
            identity = [i for i, obj in instances.items() if Path(consumer.__file__).resolve() in obj.evidence.cache]
            assignment = [i for i, obj in instances.items() if Path(f.source_plan).resolve() in obj.evidence.cache and i not in identity]
            sources = [i for i, obj in instances.items() if any(p.name == 'producer.terminal.json' for p in obj.evidence.cache) and i not in assignment]
            self.assertEqual(len(identity), 1)
            self.assertEqual(len(assignment), 1)
            self.assertEqual((verifies[identity[0]], verifies[assignment[0]]), (2, 2))
            self.assertEqual(len(sources), 6)
            self.assertEqual([verifies[i] for i in sources], [1] * 6)
            self.assertEqual(borders, Counter({'source%d' % i: 1 for i in range(6)}))
            self.assertEqual((facts.call_count, final.call_count), (6, 6))
            self.assertEqual(caches[0], caches[1])
            self.assertEqual([len(c) for c in llm.calls], [8, 8])
            events = [e for e in report['transaction_cache_events'] if e['event'] == 'native_core_transaction']
            self.assertEqual(len(events), 32)  # Native prepare + finish, no middleware resume replay.
            self.assertEqual(support.call_count, 16)
            self.assertTrue(all(e['receipt_resume_full_validation'] is False for e in events))
            self.assertEqual(report['terminal_count'], 6)
            self.assertEqual({p: p.read_bytes() for p in f.original_bytes}, f.original_bytes)

    def test_candidate_text_and_native_support_equal_v1_without_second_semantic_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'v1').mkdir()
            (root / 'v2').mkdir()
            old, new = ProductionFixture(root / 'v1', masked=True), Fixture(root / 'v2', masked=True)
            old_report, _ = old.run()
            new_report, _ = new.run()
            for i in range(6):
                sid = 'source%d' % i
                old_terminal = pipeline.load(old.out / 'sources/train' / sid / 'consumer.terminal.json')
                old_joint = old.joint(old_terminal)
                old_visual = pipeline.load(old_joint['visual_result_ref']['path'])
                visual, joint = new.visual(sid)
                for key in ('facts', 'final', 'final_timed_caption', 'coverage', 'status'):
                    self.assertEqual(visual[key], old_visual[key])
                self.assertEqual([o['json'] for o in visual['window_observations']],
                                 [o['json'] for o in old_visual['window_observations']])
                self.assertEqual(joint['target_text'], old_joint['target_text'])
                for flag in direct.FLAGS:
                    self.assertFalse(joint[flag])
                    self.assertFalse(new_report[flag])
                self.assertEqual(joint['batch_review_status'], 'pending_independent_review')
            self.assertEqual(new_report['counts'], old_report['counts'])

    def test_commit_before_mutation_refuses_args_source_rgb_and_receipt_not_failed_terminal(self):
        for kind in ('args', 'source', 'rgb', 'receipt', 'generation', 'params_identity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp, groups=1)
                original = consumer.cross.resume_source
                def resume(*args, **kwargs):
                    visual = original(*args, **kwargs)
                    if args[1] == 0:
                        root = args[0].entries[0]['root']
                        if kind == 'args':
                            f.args.visual_max_tokens += 1
                        elif kind == 'params_identity':
                            old_params = bar.params
                            bar.params = lambda *a, **kw: old_params(*a, **kw)
                        else:
                            if kind == 'source':
                                path = Path(f.entries[0]['manifest_record']['video_path'])
                            elif kind == 'rgb':
                                prepared = pipeline.load(root / 'prepared.json')
                                path = root / 'rgb' / prepared['windows'][0]['frames'][0]['file']
                            else:
                                path = root / 'windows/w00000/attempt-0' / (kind + '.json')
                            path.write_bytes(b'CPU changed immutable input')
                    return visual
                with patch.object(consumer.cross, 'resume_source', side_effect=resume):
                    with self.assertRaises(ValueError):
                        f.run()
                self.assertFalse(list(f.out.glob('sources/train/*/consumer.terminal.json')))
                self.assertFalse(list(f.out.glob('sources/train/*/joint-results/*.json')))
                self.assertFalse((f.out / 'completed.jsonl').exists())

    def test_real_restart_full_native_receipts_no_resubmit_and_orphan_notice_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f, llm = Fixture(tmp, groups=1), FakeLLM()
            original = pipeline.append
            stopped = False
            def append(path, value):
                nonlocal stopped
                if Path(path) == f.out / 'completed.jsonl' and not stopped:
                    stopped = True
                    raise OSError('CPU interruption after native source terminal')
                return original(path, value)
            with patch.object(pipeline, 'append', side_effect=append):
                with self.assertRaisesRegex(OSError, 'interruption'):
                    f.run(llm)
            report, _ = f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])
            events = [e for e in report['transaction_cache_events'] if e['event'] == 'native_core_transaction']
            self.assertEqual(len(events), 5)
            self.assertTrue(all(e['receipt_resume_full_validation'] is True for e in events))
            self.assertEqual(report['recovered_completed_notices'], 1)
            self.assertEqual(report['terminal_count'], 3)
            before = (f.out / 'completed.jsonl').read_bytes()
            repeated, _ = f.run(llm)
            self.assertEqual(repeated['transaction_cache_events'], [])
            self.assertEqual(before, (f.out / 'completed.jsonl').read_bytes())
            self.assertEqual([len(c) for c in llm.calls], [8])

    def test_changed_restart_receipt_full_gate_refuses_without_new_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            f, llm = Fixture(tmp, groups=1), FakeLLM()
            with patch.object(consumer.cross, 'resume_source', side_effect=RuntimeError('CPU interrupted before source result')):
                with self.assertRaises(RuntimeError):
                    f.run(llm)
            receipt = f.out / 'visual/sources/source0/windows/w00000/attempt-0/receipt.json'
            receipt.write_bytes(b'corrupt native receipt')
            with self.assertRaises(ValueError):
                f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])
            self.assertFalse(list(f.out.glob('sources/train/*/consumer.terminal.json')))

    def test_handoff_permission_is_not_replaced_by_resource_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, groups=1)
            del f.args.handoff_ref
            llm = FakeLLM()
            with self.assertRaisesRegex(ValueError, 'handoff_ref'):
                f.run(llm)
            self.assertEqual(llm.calls, [])
            self.assertFalse(f.out.exists())


if __name__ == '__main__':
    unittest.main()
