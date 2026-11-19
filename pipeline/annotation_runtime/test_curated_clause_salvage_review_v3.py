"""Five targeted CPU regressions; immutable actual sources, temporary outputs only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

RT = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, RT / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

worker = module('curated_clause_salvage_review_watch_v3')
resume = module('curated_clause_salvage_review_resume_validate_v3')
binder = worker.bind_v2
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


class PureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = worker.w.Manifest(worker.pilot.MANIFEST)
        cls.state = worker.load(worker.OUT / 'state.json')
        with worker.STREAM.open('rb') as stream:
            stream.seek(cls.state['cursor']['offset'])
            cls.notices = [json.loads(stream.readline()) for _ in range(3)]
        assert cls.notices[0]['source_key'] == 'train/aTDEGuZtIifWgYRJ'
        cls.candidate = worker.checked(cls.notices[0]['joint_candidate_ref'])
        cls.original = worker.checked(cls.candidate['original_visual_result_ref'])
        cls.observation = cls.original['window_observations'][0]
        cls.fact = cls.original['facts'][0]

    def test_real_original_pure_without_new_wrapper_is_reviewable(self):
        self.assertFalse(self.observation.get('source_evidence'))
        row = binder.bind_notice(self.notices[0], self.manifest, worker.w)
        delta = row['review_provenance']['review_delta']
        self.assertTrue(delta['review_required'])
        self.assertTrue(delta['pure_rewrite_failure'])
        self.assertEqual(len(delta['original_attempt_refs']), 8)
        self.assertFalse(delta['missing_original_evidence'])
        self.assertEqual(worker.sha(row['label_path']), self.notices[0]['joint_candidate_ref']['sha256'])

    def test_original_receipt_artifact_and_binding_changes_fail_closed(self):
        actual_sha = binder.sha
        request = str(Path(self.observation['attempt_path']) / 'request.json')
        with patch.object(binder, 'sha', side_effect=lambda p: '0'*64 if str(p)==request else actual_sha(p)):
            with self.assertRaisesRegex(ValueError, 'artifact SHA changed'):
                binder.original_attempt_check(self.fact, self.observation, self.original, worker.w)
        actual_load = binder.load
        def changed_receipt(path):
            value = actual_load(path)
            if Path(path).name == 'receipt.json':
                value = copy.deepcopy(value)
                value['binding']['window']['start'] = 999
            return value
        with patch.object(binder, 'load', side_effect=changed_receipt):
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                binder.original_attempt_check(self.fact, self.observation, self.original, worker.w)

    def test_added_fact_still_requires_new_raw_evidence(self):
        with self.assertRaisesRegex(ValueError, 'New fact missing checked original raw evidence'):
            binder.fact_check(self.fact, self.observation, worker.w)

    def test_case_missing_original_evidence_pending_and_next_source_continues(self):
        missing = Path(self.observation['attempt_path']) / 'receipt.json'
        actual_is_file = Path.is_file
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            stream = Path(tmp) / 'stream.jsonl'
            stream.write_bytes(bytes([10]).join(json.dumps(n).encode() for n in self.notices[:2]) + bytes([10]))
            reviewer = Mock(side_effect=lambda row, root, **kw: {'source_id':row['id'], 'status':'label_incomplete'})
            watch = worker.Watcher(out, self.manifest, reviewer, review, stream=stream, previous={}, gate=lambda:None)
            with patch.object(Path, 'is_file', autospec=True, side_effect=lambda p:False if p==missing else actual_is_file(p)):
                summary = watch.poll(drain=True)
            self.assertEqual(summary['processed_source_count'], 2)
            first = worker.checked(watch.state['sources'][self.notices[0]['source_key']]['transaction_ref'])
            self.assertEqual(first['status'], 'source_pending_missing_original_evidence')
            self.assertFalse(first['review_required'])
            self.assertIsNone(worker.checked(first['ledger_ref'])['review_receipt'])
            self.assertFalse(first['training_eligible'])
            reviewer.assert_called_once()
            self.assertEqual(reviewer.call_args.args[0]['id'], self.notices[1]['id'])

    def check_manager_postworker_allows_only_baseline_preserving_state_growth(self, selected):
        manager = module('curated_clause_salvage_review_watch_resource_manager_v3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, queue = root / 'out', root / 'queue'
            out.mkdir(); queue.mkdir()
            state_path = out / 'state.json'
            state_path.write_text(json.dumps(self.state))
            initial_ref = worker.ref(state_path)
            records = []
            for key, source in selected.values():
                transaction = worker.checked(source['transaction_ref'])
                records.append({'source_key':key, 'status':source['status'], 'transaction_ref':source['transaction_ref'],
                    'input_ref':transaction['input_ref'], 'ledger_ref':transaction['ledger_ref'],
                    'summary_ref':transaction['summary_ref'], 'label_result_ref':transaction['identity']['label_result_ref']})
            proof = {'original_state_ref':initial_ref, 'original_cursor':self.state['cursor'],
                'consumed_prefix_bytes':self.state['cursor']['offset'],
                'consumed_prefix_sha256':self.state['cursor']['committed_prefix_sha256'],
                'processed_source_count':len(self.state['sources']), 'state_migrated':False,
                'model_calls_performed':0, 'committed_records':records, 'queued_boundary_records':[]}
            proof_path = root / 'proof.json'; proof_path.write_text(json.dumps(proof))
            plan = {'schema':worker.PHASE, 'total_source_count':222939, 'expected_batch_count':448,
                'input_stream':str(worker.STREAM), 'resume_state_ref':worker.ref(proof_path)}
            plan_path = root / 'plan.json'; plan_path.write_text(json.dumps(plan))
            cfg = {'source_sha256':{}, 'producer':worker.ref(Path(worker.__file__)), 'plan':worker.ref(plan_path),
                'resume_state_ref':worker.ref(proof_path), 'queue':str(queue), 'input_count':222939,
                'expected_batch_count':448, 'targetcount':0, 'dynamic_source_watch':True}
            config_path = root / 'config.json'; config_path.write_text(json.dumps(cfg))
            manager_sha, config_sha = manager.base.sha(manager.base.__file__), manager.base.sha(config_path)
            with patch.object(manager.worker, 'OUT', out), patch.object(manager.worker.AppendCursor, 'next', return_value=None):
                manager.check_sources(cfg, config_path, config_sha, manager_sha)
                grown = copy.deepcopy(self.state)
                grown['sources']['train/CPU_ONLY_ADDED'] = {'status':'pending'}
                state_path.write_text(json.dumps(grown))
                (queue / 'live.json').write_text('{}')
                manager.check_sources(cfg, config_path, config_sha, manager_sha)
                grown['sources'].pop(records[0]['source_key'])
                state_path.write_text(json.dumps(grown))
                with self.assertRaisesRegex(RuntimeError, 'source removed/replaced'):
                    manager.check_sources(cfg, config_path, config_sha, manager_sha)

    def test_mixed_committed_state_reuse_checks_input_and_never_calls_model(self):
        with worker.STREAM.open('rb') as stream:
            consumed = stream.read(self.state['cursor']['offset'])
        notices = {json.loads(line)['source_key']:json.loads(line) for line in consumed.splitlines()}
        selected = {}
        for key, source in self.state['sources'].items():
            transaction = worker.checked(source['transaction_ref'])
            if source['status'] in ('pass','confirmed_error','uncertain'):
                selected.setdefault(transaction['schema'], (key, source))
        self.assertEqual(set(selected), {worker.LEGACY_PHASE, worker.V2_PHASE})
        self.check_manager_postworker_allows_only_baseline_preserving_state_growth(selected)
        no_call = Mock(side_effect=AssertionError('COMMITTED REVIEW MUST NOT CALL MODEL'))
        with tempfile.TemporaryDirectory() as tmp:
            watch = worker.Watcher(Path(tmp) / 'out', self.manifest, no_call, review,
                stream=Path(tmp) / 'unused', previous={}, gate=lambda:None)
            for key, source in selected.values():
                watch.state['sources'][key] = source
                watch.process(notices[key])
                transaction = worker.checked(source['transaction_ref'])
                bad = copy.deepcopy(transaction)
                bad['identity']['input_row_sha256'] = '0'*64
                actual_checked = worker.checked
                with patch.object(worker, 'checked', side_effect=lambda ref:bad if ref==source['transaction_ref'] else actual_checked(ref)):
                    with self.assertRaisesRegex(ValueError, 'identity changed'):
                        watch.process(notices[key])
            no_call.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
