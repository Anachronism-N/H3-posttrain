"""Actual N6 evidence replay; /tmp-only lifecycle, no model or semantic mock pass."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, RT / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


worker = module('curated_clause_salvage_review_watch_v5')
resume = module('curated_clause_salvage_review_resume_validate_v5')
binder = worker.bind_v2
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert '28.216.19.225' in subprocess.check_output(['hostname', '-I'], text=True).split()
        cls.state_bytes = (worker.OUT / 'state.json').read_bytes()
        cls.state = json.loads(cls.state_bytes)
        cls.manifest = worker.w.Manifest(worker.pilot.MANIFEST)
        cls.boundaries = []
        with worker.STREAM.open('rb') as stream:
            stream.seek(cls.state['cursor']['offset'])
            for _ in range(3):
                start = stream.tell()
                raw = stream.readline()
                cls.boundaries.append((json.loads(raw), raw, start))
        assert [n['source_key'] for n, _, _ in cls.boundaries] == [
            'train/XnYJAqQnRPmARioI', 'train/KhfvQfWrHlPRsXTt', 'train/sbNGfqszAuWipMNs']
        resume.install_sha_cache(worker)
        cls.proof = resume.resume_state_proof(worker, review, cls.manifest)
        cls.rows = [binder.bind_notice(n, cls.manifest, worker.w) for n, _, _ in cls.boundaries]
        cls.visual = worker.checked(cls.boundaries[0][0]['visual_result_ref'])
        cls.observation = cls.visual['window_observations'][0]
        cls.fact = next(f for f in cls.visual['facts'] if f['fact_id'] == 'w00000:e2_c0')
        cls.repair = worker.checked(cls.visual['original_repair_result_ref'])
        cls.bar, cls.queue, _, _, cls.prepared, cls.refs = binder.prepared_context(cls.visual, worker.w)
        cls.window = cls.prepared['windows'][0]
        cls.forbidden = {p:set(v) for p,v in cls.prepared['forbidden'].items()}
        cls.raw = worker.checked(cls.observation['source_evidence']['source_artifact_refs']['response.raw.txt'])
        cls.tmp = tempfile.TemporaryDirectory(prefix='salvage_v5_cpu_', dir='/tmp')
        Path(cls.tmp.name, 'original_resume_proof.json').write_text(json.dumps(cls.proof, ensure_ascii=False))
        print(json.dumps(dict(cpu_proof_dir=cls.tmp.name, processed_sources=cls.proof['processed_source_count'],
            real_audit_receipts=cls.proof['real_audit_receipt_count'], verified_files=len(cls.proof['verified_file_inventory']),
            prefix_bytes=cls.proof['consumed_prefix_bytes'], prefix_sha=cls.proof['consumed_prefix_sha256'],
            boundary_records=cls.proof['queued_boundary_records'], actual_model_calls=0)), flush=True)

    @classmethod
    def tearDownClass(cls):
        assert (worker.OUT / 'state.json').read_bytes() == cls.state_bytes
        # Retain /tmp CPU proof only; no production artifacts are written.
        cls.tmp._finalizer.detach()

    def test_original_178_53_proof_and_three_exact_boundary_bytes(self):
        self.assertEqual(self.proof['processed_source_count'], 178)
        self.assertEqual(self.proof['real_audit_receipt_count'], 53)
        self.assertEqual(self.proof['consumed_prefix_bytes'], 486344)
        self.assertEqual(self.proof['consumed_prefix_sha256'], '0afde461643db5ed79707fd58baffbc22625e1ac1ae217b475f58b49c2d76666')
        self.assertEqual(self.state['cursor']['identity'], [1048634, 9895974728173])
        self.assertEqual(len(self.proof['queued_boundary_records']), 3)
        offset = 486344
        for notice, raw, start in self.boundaries:
            self.assertEqual(start, offset)
            boundary = next(b for b in self.proof['queued_boundary_records'] if b['source_key'] == notice['source_key'])
            self.assertEqual(boundary['byte_offset'], start)
            self.assertEqual(boundary['byte_length'], len(raw))
            self.assertEqual(boundary['sha256'], hashlib.sha256(raw).hexdigest())
            offset += len(raw)
        self.assertEqual(offset, 496484)

    def test_XnY_exact_filter_head_exception_and_closed_grip_full_clause_pending(self):
        replay = self.bar.filter_supported_observation(self.raw, self.window, self.forbidden)
        event = next(e for e in replay['events'] if e['id'] == 'e2_c0')
        derived = next(e for e in self.observation['json']['events'] if e['id'] == 'e2_c0')
        self.assertEqual(event, derived)
        self.assertEqual(event['parts'], ['head', 'right_arm', 'right_hand'])
        self.assertEqual(event['frame_indices'], [10,14])
        self.assertIn('with the hand remaining in a closed grip', self.fact['raw_text'])
        delta = self.rows[0]['review_provenance']['review_delta']
        self.assertFalse(delta['review_required'])
        self.assertEqual(delta['pending_status'], 'declared_handshape_unsafe')
        self.assertEqual([f['fact_id'] for f in delta['added_supported_body_facts']], ['w00000:e2_c1'])
        self.assertEqual(delta['rejected_new_facts'][0]['fact'], self.fact)
        self.assertEqual(self.rows[0]['label'], worker.checked(self.boundaries[0][0]['joint_candidate_ref'])['label'])

    def test_Khfv_existing_successful_primary_repair_and_main_both_replayed(self):
        visual = worker.checked(self.boundaries[1][0]['visual_result_ref'])
        repair = worker.checked(visual['original_repair_result_ref'])
        main = worker.checked(visual['original_visual_result_ref'])
        self.assertEqual(visual['facts'], repair['facts'])
        self.assertNotEqual(visual['facts'], main['facts'])
        self.assertEqual([f['fact_id'] for f in visual['facts']], ['w00000:e1_c1','w00001:e2_c0'])
        self.assertTrue(self.rows[1]['review_provenance']['review_delta']['review_required'])
        self.assertTrue(self.rows[1]['review_provenance']['review_delta']['pure_rewrite_failure'])
        refs = self.rows[1]['review_provenance']['review_delta']['original_attempt_refs']
        self.assertTrue(any('/curated222939_repair_v1/revisions/' in r['path'] and r['path'].endswith('/receipt.json') for r in refs))
        self.assertTrue(any('/curated222939_production_v1/visual/' in r['path'] and r['path'].endswith('/receipt.json') for r in refs))
        self.assertFalse(self.rows[1]['review_provenance']['review_delta'].get('semantic_approval', False))
        self.assertNotIn('source_evidence', visual['window_observations'][0])

    def test_replay_rejects_insufficient_head_support_extra_arm_and_foreign_frame(self):
        raw = copy.deepcopy(self.raw)
        head = next(p for p in raw['part_support'] if p['part'] == 'head')
        head['frame_indices'] = [10]
        with self.assertRaises(binder.SourceEvidenceViolation):
            binder.replay_fact(self.fact, self.observation, raw, self.window, self.forbidden, self.bar, self.refs, isolatable=True)
        for field, value in [('parts','left_arm'), ('frame_indices',9999)]:
            fact, observation = copy.deepcopy(self.fact), copy.deepcopy(self.observation)
            fact[field].append(value)
            event = next(e for e in observation['json']['events'] if e['id'] == 'e2_c0')
            event[field].append(value)
            with self.assertRaises(binder.SourceEvidenceViolation):
                binder.replay_fact(fact, observation, self.raw, self.window, self.forbidden, self.bar, self.refs, isolatable=True)

    def test_raw_and_receipt_SHA_corruption_failclosed_not_typedpending(self):
        for name in ['response.raw.txt','receipt.json']:
            observation = copy.deepcopy(self.observation)
            observation['source_evidence']['source_artifact_refs'][name]['sha256'] = '0'*64
            with self.assertRaises(ValueError) as raised:
                binder.primary_repair_attempt_check(self.fact, observation, self.visual, self.repair, worker.w, salvaged=True)
            self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)

    def test_unknown_root_and_foreign_primary_repair_path_failclosed(self):
        observation = copy.deepcopy(self.observation)
        for reference in observation['source_evidence']['source_artifact_refs'].values():
            reference['path'] = '/tmp/foreign-repair/' + Path(reference['path']).name
        with self.assertRaises(ValueError) as raised:
            binder.primary_repair_attempt_check(self.fact, observation, self.visual, self.repair, worker.w, salvaged=True)
        self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)
        visual = worker.checked(self.boundaries[1][0]['visual_result_ref'])
        repair = worker.checked(visual['original_repair_result_ref'])
        observation = copy.deepcopy(visual['window_observations'][0])
        observation['attempt_path'] = '/tmp/nonexistent/attempt-0'
        with self.assertRaises(ValueError):
            binder.primary_repair_attempt_check(visual['facts'][0], observation, visual, repair, worker.w, salvaged=False)

    def test_job_and_binding_corruption_failclosed_not_typedpending(self):
        artifacts = self.observation['source_evidence']['source_artifact_refs']
        checked = worker.w.checked
        for name in ['request.json','binding.json']:
            target = artifacts[name]
            def corrupt(reference):
                value = checked(reference)
                if reference == target:
                    value = copy.deepcopy(value)
                    if name == 'request.json':
                        value['repair_provenance']['job_ref']['path'] = '/tmp/fake-dispatch/job.json'
                    else:
                        value['repair_revision'] = 'foreign-revision'
                return value
            with patch.object(worker.w, 'checked', side_effect=corrupt):
                with self.assertRaises(ValueError) as raised:
                    binder.primary_repair_attempt_check(self.fact, self.observation, self.visual, self.repair, worker.w, salvaged=True)
            self.assertNotIsInstance(raised.exception, binder.SourceEvidenceViolation)

    def test_three_physical_notices_lifecycle_no_model_no_mock_pass(self):
        out = Path(self.tmp.name) / 'lifecycle'
        out.mkdir()
        (out / 'state.json').write_bytes(self.state_bytes)
        calls = []
        def cpu_pending(row, root, **kwargs):
            calls.append(row['id'])
            # Only the CPU test callback: never fabricate a receipt or a semantic verdict.
            return dict(source_id=row['id'], status='cpu_only_pending_independent_risk_review',
                label_sha256=review.batch.digest(review.label_payload(row)), review_receipt=None,
                model_call_performed=False, semantic_approval=False, human_review_completed=False, training_eligible=False)
        watch = worker.Watcher(out, self.manifest, cpu_pending, review, stream=worker.STREAM,
            previous={}, gate=lambda:None)
        summary = watch.poll(limit=3)
        self.assertEqual(calls, ['KhfvQfWrHlPRsXTt'])
        self.assertEqual(summary['processed_source_count'], 181)
        self.assertEqual(summary['sampled_sources_reviewed'], 53)
        self.assertEqual(watch.state['cursor']['offset'], 496484)
        for notice, _, start in self.boundaries:
            source = watch.state['sources'][notice['source_key']]
            transaction = worker.checked(source['transaction_ref'])
            ledger = worker.checked(transaction['ledger_ref'])
            self.assertIsNone(ledger['review_receipt'])
            self.assertNotIn(ledger['status'], ('pass','confirmed_error','uncertain'))
            self.assertEqual(ledger['review_provenance']['original_notice_stream_ref']['byte_offset'], start)
            self.assertFalse(ledger['training_eligible'])
            self.assertFalse(ledger['semantic_approval'])
        for key, old in self.state['sources'].items():
            self.assertEqual(watch.state['sources'][key], old)
        replay = worker.Watcher(out, self.manifest, lambda *a,**k:self.fail('Replay cannot call model'), review,
            stream=worker.STREAM, previous={}, gate=lambda:None)
        for notice, _, _ in self.boundaries:
            replay.process(notice)
        key = self.boundaries[0][0]['source_key']
        transaction = worker.checked(watch.state['sources'][key]['transaction_ref'])
        path = Path(transaction['input_ref']['path'])
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            replay.process(self.boundaries[0][0])
        path.write_bytes(original)
        print(json.dumps(dict(cpu_lifecycle_sources=181, unchanged_real_reviews=53, next_offset=496484,
            pending_callback_sources=calls, actual_model_calls=0, semantic_mock_passes=0)), flush=True)

    def test_z_resource_manager_source_inventory_excludes_mutable_state_and_base_is_unchanged(self):
        text = (RT / 'prepare_curated_clause_salvage_review_watch_v5.py').read_text()
        self.assertIn("if Path(reference['path']) != worker.OUT / 'state.json'", text)
        self.assertEqual(worker.sha(RT/'resource_manager.py'), '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6')
        for k in ('V3_PHASE','V2_PHASE','LEGACY_PHASE'):
            self.assertIn(getattr(worker,k), self.proof['transaction_schemas'])
        manager = module('curated_clause_salvage_review_watch_resource_manager_v5')
        out = Path(self.tmp.name) / 'lifecycle'
        state = worker.load(out / 'state.json')
        proof = copy.deepcopy(self.proof)
        proof['original_state_ref'] = dict(path=str(out / 'state.json'), sha256=hashlib.sha256(self.state_bytes).hexdigest())
        proof_path = Path(self.tmp.name) / 'resource_resume_proof.json'
        proof_path.write_text(json.dumps(proof))
        launch_path = Path(self.tmp.name) / 'resource_launch.json'
        launch_path.write_text(json.dumps(dict(schema=worker.PHASE, resume_state_ref=worker.ref(proof_path),
            total_source_count=222939, expected_batch_count=448, input_stream=str(worker.STREAM))))
        queue = Path(self.tmp.name) / 'resource_queue'
        queue.mkdir()
        (queue / 'live.json').write_text('{}')
        cfg = dict(source_sha256={}, producer=worker.ref(Path(worker.__file__)), plan=worker.ref(launch_path),
            resume_state_ref=worker.ref(proof_path), queue=str(queue), input_count=222939,
            expected_batch_count=448, targetcount=0, dynamic_source_watch=True)
        config_path = Path(self.tmp.name) / 'resource_config.json'
        config_path.write_text(json.dumps(cfg))
        with patch.object(manager, 'worker', worker), patch.object(worker, 'OUT', out):
            manager.check_sources(cfg, config_path, worker.sha(config_path), manager.BASE_MANAGER_SHA)
            actual_load = worker.load
            key = next(iter(self.state['sources']))
            for mutation in ('delete', 'replace', 'completeness'):
                changed = copy.deepcopy(state)
                if mutation == 'delete':
                    del changed['sources'][key]
                elif mutation == 'replace':
                    changed['sources'][key]['transaction_ref']['sha256'] = '0'*64
                else:
                    changed['sources'][key]['label_complete'] = not changed['sources'][key]['label_complete']
                def changed_state(path):
                    return changed if Path(path) == out / 'state.json' else actual_load(path)
                with patch.object(worker, 'load', side_effect=changed_state):
                    with self.assertRaises(RuntimeError):
                        manager.check_sources(cfg, config_path, worker.sha(config_path), manager.BASE_MANAGER_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
