"""Actual-source CPU regressions; temporary outputs only, never construct a model."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, RT / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


worker = module('curated_clause_salvage_review_watch_v4')
resume = module('curated_clause_salvage_review_resume_validate_v4')
binder = worker.bind_v2
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


class SourceIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = worker.w.Manifest(worker.pilot.MANIFEST)
        cls.state = worker.load(worker.OUT / 'state.json')
        cls.boundaries = []
        with worker.STREAM.open('rb') as stream:
            stream.seek(cls.state['cursor']['offset'])
            for _ in range(3):
                start = stream.tell()
                raw = stream.readline()
                cls.boundaries.append((json.loads(raw), raw, start))
        cls.bad, cls.other_invalid, cls.legal = [item[0] for item in cls.boundaries]
        assert cls.bad['source_key'] == 'train/XnYJAqQnRPmARioI'
        assert cls.other_invalid['source_key'] == 'train/KhfvQfWrHlPRsXTt'
        assert cls.legal['source_key'] == 'train/sbNGfqszAuWipMNs'
        cls.visual = worker.checked(cls.bad['visual_result_ref'])
        cls.observation = cls.visual['window_observations'][0]
        cls.fact = next(f for f in cls.visual['facts'] if f['fact_id'] == 'w00000:e2_c0')
        cls.proof = resume.resume_state_proof(worker, review, cls.manifest)
        print(json.dumps(dict(actual_processed=cls.proof['processed_source_count'],
            actual_real_receipts=cls.proof['real_audit_receipt_count'],
            phases=cls.proof['transaction_schemas'], verified_files=len(cls.proof['verified_file_inventory']),
            model_calls_performed=0)), flush=True)

    def no_model(self):
        return Mock(side_effect=AssertionError('NO CPU TEST MAY CALL A MODEL'))

    def test_actual_typed_scope_violation_and_legal_binding(self):
        with self.assertRaises(binder.SourceEvidenceViolation) as raised:
            binder.bind_notice(self.bad, self.manifest, worker.w)
        error = raised.exception
        self.assertEqual(error.details['source_key'], self.bad['source_key'])
        self.assertEqual(len(error.details['violations']), 1)
        violation = error.details['violations'][0]
        self.assertEqual(violation['undeclared_parts'], ['head'])
        self.assertEqual(violation['foreign_frame_indices'], [])
        self.assertEqual(violation['original_parts'], ['right_arm', 'right_hand'])
        self.assertEqual(violation['candidate_frame_indices'], [10, 14])
        self.assertFalse(error.delta['review_required'])
        self.assertEqual(error.row['label'], worker.checked(self.bad['joint_candidate_ref'])['label'])
        row = binder.bind_notice(self.legal, self.manifest, worker.w)
        self.assertFalse(row['review_provenance']['review_delta']['review_required'])

    def test_real_raw_hash_and_receipt_binding_corruption_not_isolatable(self):
        raw = self.observation['source_evidence']['source_artifact_refs']['response.raw.txt']['path']
        actual_sha = binder.sha
        with patch.object(binder, 'sha', side_effect=lambda p:'0'*64 if str(p) == raw else actual_sha(p)):
            with self.assertRaises(ValueError) as raised:
                binder.bind_notice(self.bad, self.manifest, worker.w)
        self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)
        actual_checked = worker.w.checked
        receipt_ref = self.observation['source_evidence']['source_artifact_refs']['receipt.json']
        def changed_receipt(reference):
            value = actual_checked(reference)
            if reference == receipt_ref:
                value = copy.deepcopy(value)
                value['binding']['window']['start'] = 999
            return value
        with patch.object(worker.w, 'checked', side_effect=changed_receipt):
            with self.assertRaises(ValueError) as raised:
                binder.bind_notice(self.bad, self.manifest, worker.w)
        self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)

    def test_new_fact_scope_change_after_checked_observation_fails_closed(self):
        fact = copy.deepcopy(self.fact)
        fact['parts'].append('left_leg')
        with self.assertRaisesRegex(ValueError, 'differs from checked visual event') as raised:
            binder.fact_check(fact, self.observation, worker.w, source_visual=self.visual)
        self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)
        fact = copy.deepcopy(self.fact)
        fact['frame_indices'].append(9999)
        with self.assertRaisesRegex(ValueError, 'differs from checked visual event'):
            binder.fact_check(fact, self.observation, worker.w, source_visual=self.visual)

    def test_selected_actual_legal_source_continues_after_immutable_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, stream = Path(tmp) / 'out', Path(tmp) / 'selected_actual_sources.jsonl'
            # Exact source rows selected from physical positions 0 and 2; not a production skip.
            stream.write_bytes(self.boundaries[0][1] + self.boundaries[2][1])
            model = self.no_model()
            watch = worker.Watcher(out, self.manifest, model, review, stream=stream, previous={}, gate=lambda:None)
            result = watch.poll(drain=True)
            self.assertEqual(result['processed_source_count'], 2)
            self.assertEqual(result['sampled_sources_reviewed'], 0)
            self.assertEqual(watch.state['cursor']['offset'], stream.stat().st_size)
            source = watch.state['sources'][self.bad['source_key']]
            transaction = worker.checked(source['transaction_ref'])
            entry, summary = worker.checked(transaction['ledger_ref']), worker.checked(transaction['summary_ref'])
            for value in (source, transaction, entry, summary):
                self.assertFalse(value['label_complete'])
            for value in (transaction, entry, summary):
                self.assertFalse(value['review_performed'])
                self.assertEqual(value['model_calls_performed'], 0)
                self.assertFalse(value['training_eligible'])
                self.assertFalse(value['semantic_approval'])
            self.assertEqual(entry['status'], 'source_evidence_violation')
            self.assertIsNone(entry['review_receipt'])
            self.assertIsNone(entry['review_final_ref'])
            model.assert_not_called()
            resumed = worker.Watcher(out, self.manifest, model, review, stream=stream, previous={}, gate=lambda:None)
            resumed.process(self.bad)
            with patch.object(worker, 'OUT', out), patch.object(worker, 'STREAM', stream):
                proof = resume.resume_state_proof(worker, review, self.manifest)
            self.assertEqual(proof['processed_source_count'], 2)
            self.assertEqual(proof['real_audit_receipt_count'], 0)
            original_bytes = Path(source['transaction_ref']['path']).read_bytes()
            changed = worker.load(source['transaction_ref']['path'])
            changed['status'] = 'pass'
            Path(source['transaction_ref']['path']).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                resumed.process(self.bad)
            Path(source['transaction_ref']['path']).write_bytes(original_bytes)
            # Same source/raw scope must still reproduce; a validator silently dropping it is rejected.
            with patch.object(binder, 'bind_notice', return_value=worker.checked(transaction['input_ref'])):
                with self.assertRaisesRegex(ValueError, 'no longer has checked part/frame violation'):
                    resumed.process(self.bad)
            model.assert_not_called()

    def test_actual_next_other_violation_is_not_swallowed_and_cursor_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, stream = Path(tmp) / 'out', Path(tmp) / 'actual_adjacent.jsonl'
            stream.write_bytes(self.boundaries[0][1] + self.boundaries[1][1])
            model = self.no_model()
            watch = worker.Watcher(out, self.manifest, model, review, stream=stream, previous={}, gate=lambda:None)
            with self.assertRaisesRegex(ValueError, 'Pure facts differ from checked original repair/visual facts'):
                watch.poll(drain=True)
            saved = worker.load(out / 'state.json')
            self.assertEqual(set(saved['sources']), {self.bad['source_key']})
            self.assertEqual(saved['cursor']['offset'], len(self.boundaries[0][1]))
            self.assertNotIn(self.other_invalid['source_key'], saved['sources'])
            model.assert_not_called()

    def test_full_real_mixed_resume_preserves_first164_51_and_reuses_all_phases(self):
        proof = self.proof
        self.assertGreaterEqual(proof['processed_source_count'], 178)
        self.assertGreaterEqual(proof['real_audit_receipt_count'], 53)
        self.assertEqual(sum(r['review_receipt'] is not None for r in proof['committed_records'][:164]), 51)
        self.assertEqual(set(proof['transaction_schemas']), {worker.LEGACY_PHASE, worker.V2_PHASE, worker.V3_PHASE})
        self.assertFalse(proof['state_migrated'])
        self.assertEqual(proof['model_calls_performed'], 0)
        with worker.STREAM.open('rb') as stream:
            notices = {n['source_key']:n for n in map(json.loads, stream.read(self.state['cursor']['offset']).splitlines())}
        selected = {}
        for key, source in self.state['sources'].items():
            transaction = worker.checked(source['transaction_ref'])
            if source['status'] in ('pass', 'confirmed_error', 'uncertain'):
                selected.setdefault(transaction['schema'], (key, source))
        model = self.no_model()
        with tempfile.TemporaryDirectory() as tmp:
            watch = worker.Watcher(Path(tmp) / 'out', self.manifest, model, review,
                stream=Path(tmp) / 'unused.jsonl', previous={}, gate=lambda:None)
            for key, source in selected.values():
                watch.state['sources'][key] = source
                watch.process(notices[key])
            model.assert_not_called()

    def test_manager_allows_actual_contiguous_pending_growth_not_baseline_mutation(self):
        manager = module('curated_clause_salvage_review_watch_resource_manager_v4')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, queue = root / 'out', root / 'queue'
            out.mkdir(); queue.mkdir()
            state_path = out / 'state.json'
            state_path.write_text(json.dumps(self.state))
            proof = copy.deepcopy(self.proof)
            proof['original_state_ref'] = worker.ref(state_path)
            proof_path = root / 'proof.json'; proof_path.write_text(json.dumps(proof))
            plan = dict(schema=manager.worker.PHASE, total_source_count=222939, expected_batch_count=448,
                input_stream=str(worker.STREAM), resume_state_ref=worker.ref(proof_path))
            plan_path = root / 'plan.json'; plan_path.write_text(json.dumps(plan))
            cfg = dict(source_sha256={}, producer=worker.ref(Path(manager.worker.__file__)),
                plan=worker.ref(plan_path), resume_state_ref=worker.ref(proof_path), queue=str(queue),
                input_count=222939, expected_batch_count=448, targetcount=0, dynamic_source_watch=True)
            config_path = root / 'config.json'; config_path.write_text(json.dumps(cfg))
            config_sha = manager.base.sha(config_path)
            with patch.object(manager.worker, 'OUT', out):
                manager.check_sources(cfg, config_path, config_sha, manager.BASE_MANAGER_SHA)
                model = self.no_model()
                watch = worker.Watcher(out, self.manifest, model, review, stream=worker.STREAM, previous={}, gate=lambda:None)
                watch.poll(limit=1)
                self.assertEqual(len(watch.state['sources']), len(self.state['sources']) + 1)
                self.assertEqual(watch.state['cursor']['offset'], self.boundaries[1][2])
                (queue / 'live.json').write_text('{}')
                manager.check_sources(cfg, config_path, config_sha, manager.BASE_MANAGER_SHA)
                grown = worker.load(state_path)
                removed = proof['committed_records'][0]['source_key']
                grown['sources'].pop(removed)
                state_path.write_text(json.dumps(grown))
                with self.assertRaisesRegex(RuntimeError, 'source removed/replaced'):
                    manager.check_sources(cfg, config_path, config_sha, manager.BASE_MANAGER_SHA)
                model.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
