"""CPU tests; exactly three fixed real sources, never modifies production or budgets."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curated_rewrite_only_cpu_batch_v1 as cpu

IDS = ['wqGYgdMiTDqyyVuS', 'jqoZXbNoNchzsYOd', 'SDjBFHHLfNrWZyQw']


class CPURewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notices, cls.prefix = cpu.stable_prefix(500)
        cls.references = {notice['source_key'].split('/')[1]: notice['job_ref']
                          for notice in cls.notices if notice['source_key'].split('/')[1] in IDS}
        cls.job = cpu.q.checked(cls.references[IDS[0]])
        cls.original = cpu.q.checked(cls.job['refs']['visual_result'])
        cls.prepared = cpu.q.checked(cls.job['refs']['prepared'])
        cls.observation = cls.original['window_observations'][0]
        cls.manifest = cls.prepared['windows'][0]
        cls.raw = cpu.q.load(Path(cls.observation['attempt_path']) / 'response.raw.txt')
        cls.fact = cls.original['facts'][0]

    def reject_scope(self, visual, reason, job=None):
        with self.assertRaises(cpu.Pending) as caught:
            cpu.scope(job or self.job, visual)
        self.assertEqual(caught.exception.reason, reason)

    def test_01_three_real_sources_zero_dispatch_budget_and_idempotency(self):
        budget_paths = [cpu.QUEUE / 'attempt-budget/train' / (sid + '.json') for sid in IDS]
        before = {str(path): path.read_bytes() if path.exists() else None for path in budget_paths}
        with tempfile.TemporaryDirectory(prefix='rewrite-only-cpu-three-real-') as temp:
            with (patch.object(cpu.q, 'reserve_attempt', side_effect=AssertionError('No budget reservation')),
                  patch.object(cpu.bar, 'transactions', side_effect=AssertionError('No model calls'))):
                report = cpu.once(500, IDS, temp)
                files = {str(path.relative_to(temp)): path.read_bytes() for path in Path(temp).rglob('*') if path.is_file()}
                second = cpu.once(500, IDS, temp)
            self.assertEqual(report, second)
            self.assertEqual(files, {str(path.relative_to(temp)): path.read_bytes() for path in Path(temp).rglob('*') if path.is_file()})
            self.assertEqual(report['source_dedup_count'], 3)
            self.assertEqual(report['model_calls'], 0)
            self.assertFalse(report['attempt_budget_consumed'])
            self.assertFalse(report['original_population_quality_approved'])
            self.assertEqual(report['joint_complete_count'], 3)
            self.assertEqual(report['pending_reason_counts'], {})
            for notice in report['candidates']:
                joint = cpu.q.checked(notice['result_ref'])
                visual = cpu.q.checked(notice['visual_result_ref'])
                original_job = cpu.q.checked(notice['original_job_ref'])
                original = cpu.q.checked(original_job['refs']['visual_result'])
                audit = cpu.q.checked(notice['evidence_manifest_ref'])
                self.assertTrue(joint['joint_complete'])
                self.assertEqual(visual['facts'], original['facts'])
                self.assertEqual(visual['window_observations'], original['window_observations'])
                self.assertFalse(joint['semantic_approval'])
                self.assertFalse(joint['training_eligible'])
                for evidence in audit['original_window_evidence']:
                    names = {Path(reference['path']).name for reference in evidence['original_attempt_refs']}
                    self.assertTrue({'receipt.json', 'request.json', 'binding.json', 'response.raw.txt',
                                     'generation.json', 'response.json', 'final.json'} <= names)
                    self.assertTrue(evidence['per_part_two_distinct_actual_frames_checked'])
                for window in visual['final']['json']['windows']:
                    local = [fact for fact in original['facts'] if fact['window_id'] == window['window_id']]
                    self.assertEqual(window['caption'], ' '.join(fact['raw_text'] for fact in local))
                    self.assertEqual(window['used_fact_ids'], [fact['fact_id'] for fact in local])
            manifest = Path(temp) / 'revisions' / cpu.REVISION / 'review-required.manifest.json'
            manifest.write_bytes(manifest.read_bytes() + b' ')
            with self.assertRaises(cpu.q.SourceChanged):
                cpu.once(500, IDS, temp)
        after = {str(path): path.read_bytes() if path.exists() else None for path in budget_paths}
        self.assertEqual(before, after)
        print(json.dumps({'fixed_real_ids': IDS, 'complete': report['joint_complete_count'],
              'typed_pending': report['pending_reason_counts'], 'model_calls': 0, 'budget_unchanged': True}, ensure_ascii=False))

    def test_unadmitted_context_detail_does_not_suppress_coarse_clause(self):
        job = cpu.q.checked(self.references['jqoZXbNoNchzsYOd'])
        visual = cpu.q.checked(job['refs']['visual_result'])
        facts = visual['facts']
        self.assertTrue(all(not cpu.v5.named_hand_detail(f['raw_text']) for f in facts))
        raw_events = []
        for observation in visual['window_observations']:
            raw = cpu.q.load(Path(observation['attempt_path']) / 'response.raw.txt')
            raw_events.extend(raw['events'])
        self.assertTrue(any(cpu.v5.named_hand_detail(e['description']) for e in raw_events))
        self.assertEqual(len(cpu.scope(job, visual)), 3)

    def test_raw_facts_parts_frame_mismatch(self):
        for field, value in [('raw_text', 'The dancer invents a turn.'), ('parts', ['head']), ('frame_indices', [999999])]:
            with self.subTest(field=field):
                visual = copy.deepcopy(self.original)
                visual['facts'][0][field] = value
                self.reject_scope(visual, 'facts_observation_mismatch')

    def test_shape_uncertain_coherent_facts_remain_whole_source_pending(self):
        for text, reason in [('The dancer closes both hands into fists.', 'excluded_fine_hand_source'),
                             ('The dancer possibly extends the right leg.', 'pending_uncertain_source')]:
            visual = copy.deepcopy(self.original)
            event = visual['window_observations'][0]['json']['events'][0]
            event['description'] = text
            event['source_clause']['raw_source_clause'] = text
            event['source_clause']['original_description'] = text
            visual['facts'] = cpu.bar.facts_from_observations([o['window'] for o in visual['window_observations']],
                                                           visual['window_observations'])
            self.reject_scope(visual, reason)

    def test_incomplete_window_and_nonrewrite_and_notfailed(self):
        visual = copy.deepcopy(self.original)
        visual['window_observations'][0]['json']['events'] = []
        self.reject_scope(visual, 'incomplete_original_windows')
        visual = copy.deepcopy(self.original)
        visual['window_observations'] = visual['window_observations'][:-1]
        self.reject_scope(visual, 'incomplete_original_windows')
        job = copy.deepcopy(self.job)
        job['stages'] = ['rewrite', 'reobserve']
        self.reject_scope(self.original, 'excluded_non_rewrite_scope', job)
        visual = copy.deepcopy(self.original)
        visual['final']['status'] = 'candidate'
        self.reject_scope(visual, 'not_actual_failed_final')
        job = copy.deepcopy(self.job)
        job['id'] = cpu.w.RAQ_ID
        self.reject_scope(self.original, 'excluded_non_rewrite_scope', job)

    def test_exact_frozen_filter_catches_modified_raw_text_parts_and_frames(self):
        forbidden = {part: set(values) for part, values in self.prepared['forbidden'].items()}
        cpu.v5.replay_fact(self.fact, self.observation, self.raw, self.manifest, forbidden, cpu.bar, [], isolatable=False)
        for field, value in [('raw_text', 'The dancer moves somewhere else.'), ('parts', ['head']), ('frame_indices', [0])]:
            fact = copy.deepcopy(self.fact)
            fact[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                cpu.v5.replay_fact(fact, self.observation, self.raw, self.manifest, forbidden, cpu.bar, [], isolatable=False)
        raw = copy.deepcopy(self.raw)
        for support in raw['part_support']:
            if support['part'] in self.fact['parts']:
                support['frame_indices'] = support['frame_indices'][:1]
        with self.assertRaises(ValueError):
            cpu.v5.replay_fact(self.fact, self.observation, raw, self.manifest, forbidden, cpu.bar, [], isolatable=False)

    def test_SHA_receipt_missing_files_and_write_once_refusal(self):
        with tempfile.TemporaryDirectory(prefix='rewrite-only-cpu-negatives-') as temp:
            path = Path(temp) / 'artifact.json'
            cpu.q.save_once(path, {'value': 1})
            reference = cpu.q.identity(path)
            cpu.q.save_once(path, {'value': 1})
            with self.assertRaises(cpu.q.SourceChanged):
                cpu.q.save_once(path, {'value': 2})
            path.write_bytes(b'{"value":2}\n')
            with self.assertRaises(cpu.q.SourceChanged):
                cpu.q.checked(reference)
            receipt = Path(temp) / 'receipt.json'
            receipt.write_text(json.dumps({'files_sha256': {}, 'status': 'candidate'}))
            with self.assertRaises(cpu.Pending) as caught:
                cpu.receipt_inventory(Path(temp), 'candidate')
            self.assertEqual(caught.exception.reason, 'missing_original_evidence')

    def test_shared_cache_checks_full_SHA_then_stat_and_expected_SHA(self):
        with tempfile.TemporaryDirectory(prefix='rewrite-only-cpu-cache-') as temp:
            path = Path(temp) / 'plan.json'
            path.write_bytes(b'{"plan":1}\n')
            reference = cpu.q.identity(path)
            cpu.register_shared(reference)
            self.assertEqual(cpu.cached_sha(path), reference['sha256'])
            with self.assertRaises(cpu.q.SourceChanged):
                cpu.register_shared({**reference, 'sha256': '0' * 64})
            previous_mtime = path.stat().st_mtime_ns
            path.write_bytes(b'{"plan":2}\n')
            os.utime(path, ns=(path.stat().st_atime_ns, previous_mtime + 1_000_000_000))
            with self.assertRaises(cpu.q.SourceChanged):
                cpu.cached_sha(path)
            cpu._CACHE.pop(str(path.resolve()))

    def test_output_never_production_queue_or_runtime_and_prefix_bounded(self):
        for out in (cpu.QUEUE, cpu.PRODUCTION, cpu.RT, cpu.QUEUE / 'forbidden-test'):
            with self.subTest(out=out), self.assertRaises(ValueError):
                cpu.once(3, IDS, out)
        for limit in (0, 501):
            with self.assertRaises(cpu.Pending):
                cpu.stable_prefix(limit)


if __name__ == '__main__':
    unittest.main(verbosity=2)
