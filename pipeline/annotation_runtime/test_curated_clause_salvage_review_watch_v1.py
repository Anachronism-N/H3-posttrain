"""CPU-only targeted regression checks; never create a model runtime or send GPU signals."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('salvage_watch_test_worker', RT / 'curated_clause_salvage_review_watch_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review
spec = importlib.util.spec_from_file_location('salvage_watch_test_manager', RT / 'curated_clause_salvage_review_watch_resource_manager_v1.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class CursorTests(unittest.TestCase):
    def test_half_line_never_consumed_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            path.write_bytes(b'{"a":')
            state = {}
            cursor = m.AppendCursor(path, state)
            self.assertIsNone(cursor.next())
            self.assertEqual(state.get('offset', 0), 0)
            with path.open('ab') as f:
                f.write(b'1}\n')
            row, line = cursor.next()
            self.assertEqual(row, {'a': 1})
            cursor.commit(line)
            self.assertIsNone(m.AppendCursor(path, copy.deepcopy(state)).next())

    def test_partial_rewrite_truncate_replace_anchor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            path.write_bytes(b'{"a":')
            cursor = m.AppendCursor(path, {})
            cursor.next()
            path.write_bytes(b'{"b":')
            with self.assertRaisesRegex(ValueError, 'prefix changed'):
                cursor.next()
            path.write_bytes(b'{"a":1}\n')
            cursor = m.AppendCursor(path, {})
            row, line = cursor.next(); cursor.commit(line)
            saved = copy.deepcopy(cursor.state)
            path.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                cursor.next()
            path.write_bytes(b'{"a":1}\n')
            path.with_suffix('.new').write_bytes(path.read_bytes())
            path.with_suffix('.new').replace(path)
            with self.assertRaisesRegex(ValueError, 'replaced'):
                m.AppendCursor(path, saved).next()
            cursor = m.AppendCursor(path, {})
            row, line = cursor.next(); cursor.commit(line)
            path.write_bytes(b'{"b":1}\n')
            with self.assertRaisesRegex(ValueError, 'anchor changed'):
                cursor.next()


class WatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = m.w.Manifest(m.pilot.MANIFEST)
        cls.previous = m.previous_reviews()
        cls.notices = {}
        # Existing small candidate stream only; never scan the222939 source population.
        for line in m.STREAM.read_bytes().splitlines():
            notice = json.loads(line)
            if notice['source_key'] in cls.previous or not notice['need_real_review']:
                cls.notices[notice['source_key']] = notice
        cls.pilot_notices = [cls.notices[key] for key in cls.previous]

    def watcher(self, tmp, notices, gate=lambda: None, previous=None):
        stream = Path(tmp) / 'input.jsonl'
        stream.write_text(''.join(json.dumps(n) + '\n' for n in notices))
        out = Path(tmp) / 'reviews'
        out.mkdir()
        no_call = unittest.mock.Mock(side_effect=AssertionError('MODEL CALL FORBIDDEN IN CPU TEST'))
        watch = m.Watcher(out, self.manifest, no_call, review, stream=stream, gate=gate,
            previous=self.previous if previous is None else previous)
        return watch, no_call

    def test_real_two_pilot_imports_exact_receipt_and_no_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, self.pilot_notices)
            result = watch.poll(drain=True)
            self.assertEqual(result['sampled_sources_reviewed'], 2)
            self.assertEqual(result['imported_previous_review_count'], 2)
            self.assertTrue(result['input_drain'])
            self.assertFalse(result['allreviewed'])
            no_call.assert_not_called()
            for source in watch.state['sources'].values():
                transaction = m.checked(source['transaction_ref'])
                entry = json.loads(Path(transaction['ledger_ref']['path']).read_bytes())
                self.assertTrue(entry['imported_previous_review_ref'])
                self.assertFalse(entry['model_call_performed'])
                self.assertFalse(entry['semantic_approval'])
                self.assertFalse(entry['training_eligible'])
            resumed = m.Watcher(watch.out, self.manifest, no_call, review, stream=watch.cursor.path,
                gate=lambda: None, previous=self.previous)
            again = resumed.poll(drain=True)
            self.assertEqual(again['processed_source_count'], 2)
            no_call.assert_not_called()

    def test_false_delta_stays_pending_without_review(self):
        notice = next(n for n in self.notices.values() if not n['need_real_review'])
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, [notice])
            result = watch.poll(drain=True)
            self.assertEqual(result['no_new_supported_facts_count'], 1)
            self.assertEqual(result['sampled_sources_reviewed'], 0)
            self.assertEqual(result['remaining_pending_count'], 1)
            no_call.assert_not_called()
            transaction = m.checked(next(iter(watch.state['sources'].values()))['transaction_ref'])
            entry = json.loads(Path(transaction['ledger_ref']['path']).read_bytes())
            self.assertEqual(entry['independent_review_status'], 'pending')
            self.assertFalse(entry['review_performed'])
            self.assertFalse(entry['human_review_completed'])
            summary = m.checked(transaction['summary_ref'])
            plan = m.load(Path(transaction['summary_ref']['path']).parent / 'batch.plan.json')
            self.assertEqual(plan['batch_id'], summary['batch_id'])
            self.assertEqual(plan['split'], entry['split'])
            self.assertEqual(review.batch.digest(plan['records'][0]), entry['input_row_sha256'])
            self.assertFalse(plan['semantic_review_completed'])

    def test_false_producer_delta_rejected_independently(self):
        notice = copy.deepcopy(self.pilot_notices[0]); notice['need_real_review'] = False
        with self.assertRaisesRegex(ValueError, 'eligibility differs'):
            m.pilot.bind_notice(notice, self.manifest, m.w)

    def test_prior_label_sha_and_source_mutation_rejected(self):
        notice = self.pilot_notices[0]
        row = review.normalized_record(m.pilot.bind_notice(notice, self.manifest, m.w))
        entry = copy.deepcopy(self.previous[notice['source_key']][0])
        for key in ('label_sha256', 'semantic_label_sha256', 'input_row_sha256', 'source_id'):
            broken = copy.deepcopy(entry); broken[key] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                m.verify_audit(row, broken, review)
        entry['review_receipt']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'reference SHA changed'):
            m.verify_audit(row, entry, review)

    def test_recovery_notready_uses_strict_gate(self):
        with patch.object(m.recovery, 'ready_records', side_effect=m.recovery.NotReady('not ready')):
            self.assertIsNone(m.recovery_yield())
        with patch.object(m.recovery, 'ready_records', side_effect=ValueError('changed SHA')):
            with self.assertRaisesRegex(ValueError, 'changed SHA'):
                m.recovery_yield()

    def test_midpoll_yield_is_after_transaction_not_approval(self):
        calls = iter([None, dict(reason='reserved_recovery_yield', recovery_source_ids=['a','b','c'],
            recovery_ready_refs=[], reserved_queue=str(m.RECOVERY_QUEUE), recovery_admission_absent=True,
            other_tasks_signaled=False)])
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, self.pilot_notices, gate=lambda: next(calls))
            result = watch.poll(drain=True)
            self.assertTrue(result['reserved_recovery_yield'])
            self.assertFalse(result['input_drain'])
            self.assertFalse(result['source_records_consumed'])
            self.assertEqual(result['processed_source_count'], 1)
            self.assertGreater(result['remaining_unconsumed_stream_bytes'], 0)
            for key in m.FLAGS:
                self.assertFalse(result[key])
            no_call.assert_not_called()

    def test_early_yield_keeps_current_stream_unconsumed(self):
        evidence = dict(reason='reserved_recovery_yield', recovery_source_ids=['a','b','c'],
            recovery_ready_refs=[], reserved_queue=str(m.RECOVERY_QUEUE), recovery_admission_absent=True,
            other_tasks_signaled=False)
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, self.pilot_notices, gate=lambda: evidence)
            value = watch.poll(drain=True)
            self.assertEqual(value['processed_source_count'], 0)
            self.assertGreater(value['remaining_unconsumed_stream_bytes'], 0)
            self.assertTrue(value['reserved_recovery_yield'])
            no_call.assert_not_called()

    def test_manager_rejects_mock_yield_without_actual_ready_evidence(self):
        evidence = dict(reason='reserved_recovery_yield', recovery_source_ids=['a','b','c'],
            recovery_ready_refs=[], reserved_queue=str(m.RECOVERY_QUEUE), recovery_admission_absent=True,
            other_tasks_signaled=False)
        with tempfile.TemporaryDirectory() as tmp:
            watch, no_call = self.watcher(tmp, self.pilot_notices, gate=lambda: evidence)
            value = watch.poll()
            with patch.object(manager.worker.recovery, 'ready_records',
                    side_effect=manager.worker.recovery.NotReady('actual three not ready')):
                with self.assertRaisesRegex(manager.worker.recovery.NotReady, 'not ready'):
                    manager.summaries({'plan': None}, watch.out)
            for key in m.FLAGS:
                self.assertFalse(value[key])
            no_call.assert_not_called()

    def test_incomplete_label_cannot_be_approved(self):
        row = m.pilot.bind_notice(self.pilot_notices[0], self.manifest, m.w)
        row['joint_complete'] = False; row['label_status'] = 'partial'
        with tempfile.TemporaryDirectory() as tmp:
            summary = review.review_batch_ready([row], tmp,
                lambda row, root, **kwargs: {'source_id': row['id'], 'status': 'label_incomplete'},
                batch_id='test', expected_source_ids=[row['id']], sample_count=1, risk_count=0,
                expansion_counts=())
            summary = m.w.gate_summary(summary, edition=m.EDITION, provenance=row['review_provenance'])
            self.assertFalse(summary['acceptance']['eligible_for_training'])
            self.assertFalse(summary['acceptance']['all_source_labels_completed'])
            self.assertEqual(summary['state'], 'pending_label_or_review')
            self.assertFalse(summary['semantic_approval'])
            self.assertFalse(summary['training_eligible'])

    def test_claim_prompt_omits_generation_reasons(self):
        row = {'label': {'dance_style': 'dance', 'timed_motion_script': 'body moves'},
            'review_reason': 'SHOULD_NOT_ENTER_PROMPT', 'generator_reasoning': 'SHOULD_NOT_ENTER_PROMPT'}
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'frame.jpg'; image.write_bytes(b'CPU request fixture')
            evidence = {'frames': [dict(index=i, time=float(i), source_timestamp=float(i), file='frame.jpg')
                for i in (0, 1)], 'hand_rois': []}
            request = review.review_request(row, evidence, tmp, m.MODEL)
            self.assertNotIn('SHOULD_NOT_ENTER_PROMPT', json.dumps(request))
            self.assertIn('label_to_test', json.dumps(request))


if __name__ == '__main__':
    unittest.main(verbosity=2)
