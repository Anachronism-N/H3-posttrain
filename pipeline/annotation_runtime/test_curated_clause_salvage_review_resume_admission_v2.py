"""Focused CPU tests; temporary outputs only, no model construction or GPU signals."""
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
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

worker = module('curated_clause_salvage_review_watch_strict_resume_v2')
manager = module('curated_clause_salvage_review_watch_resource_manager_v2')
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = worker.w.Manifest(worker.pilot.MANIFEST)
        state = worker.load(worker.OUT / 'state.json')
        cls.notices = []
        with worker.STREAM.open('rb') as f:
            f.seek(state['cursor']['offset'])
            for _ in range(3):
                cls.notices.append(json.loads(f.readline()))

    def test_three_next_notices_pending_without_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = root / 'stream.jsonl'
            stream.write_bytes(bytes([10]).join(json.dumps(n).encode() for n in self.notices) + bytes([10]))
            no_call = Mock(side_effect=AssertionError('CPU MODEL CALL FORBIDDEN'))
            watch = worker.Watcher(root / 'out', self.manifest, no_call, review,
                stream=stream, previous={}, gate=lambda: None)
            summary = watch.poll(drain=True)
            self.assertEqual(summary['processed_source_count'], 3)
            self.assertEqual(summary['sampled_sources_reviewed'], 0)
            self.assertEqual(summary['remaining_pending_count'], 3)
            first = worker.checked(watch.state['sources'][self.notices[0]['source_key']]['transaction_ref'])
            self.assertEqual(first['status'], 'declared_handshape_unsafe')
            for source in watch.state['sources'].values():
                transaction = worker.checked(source['transaction_ref'])
                entry = worker.checked(transaction['ledger_ref'])
                self.assertFalse(entry['review_performed'])
                self.assertIsNone(entry['review_receipt'])
                for key in worker.FLAGS:
                    self.assertFalse(entry[key])
            no_call.assert_not_called()

    def test_real_coarse_body_still_requires_review(self):
        with worker.STREAM.open('rb') as stream:
            for line in stream:
                notice = json.loads(line)
                if notice['source_key'] == 'train/ifcFqjkjSgJuQeGR':
                    break
            else:
                self.fail('Known immutable coarse-body source disappeared')
        row = worker.bind_v2.bind_notice(notice, self.manifest, worker.w)
        self.assertTrue(row['review_provenance']['review_delta']['review_required'])
        self.assertEqual(len(row['review_provenance']['review_delta']['added_supported_body_facts']), 1)
        row['review_provenance']['eligibility_disagreement'] = {'reason':'CPU_metadata_fixture', 'review_performed':False, 'semantic_approval':False, 'human_review_completed':False, 'training_eligible':False}
        with tempfile.TemporaryDirectory() as tmp:
            fake_audit = Mock(return_value={'source_id':row['id'], 'status':'label_incomplete'})
            watch = worker.Watcher(Path(tmp) / 'out', self.manifest, fake_audit, review,
                stream=Path(tmp) / 'unused', previous={}, gate=lambda:None)
            with patch.object(worker.bind_v2, 'bind_notice', return_value=row):
                watch.process(notice)
            fake_audit.assert_called_once()
            transaction = worker.checked(watch.state['sources'][notice['source_key']]['transaction_ref'])
            self.assertTrue(transaction['review_required'])
            self.assertFalse(transaction['training_eligible'])


    def test_notice_declaration_mutation_fails_closed(self):
        notice = copy.deepcopy(self.notices[0])
        notice['need_real_review'] = not notice['need_real_review']
        with self.assertRaisesRegex(ValueError, 'declaration changed'):
            worker.bind_v2.bind_notice(notice, self.manifest, worker.w)

    def test_raw_candidate_sha_mutation_fails_closed(self):
        notice = copy.deepcopy(self.notices[0])
        notice['joint_candidate_ref']['sha256'] = '0' * 64
        notice['result_ref'] = copy.deepcopy(notice['joint_candidate_ref'])
        with self.assertRaisesRegex(ValueError, 'SHA changed'):
            worker.bind_v2.bind_notice(notice, self.manifest, worker.w)

    def test_committed_prefix_rewrite_detected_beyond_last_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stream'
            path.write_bytes(b'1234567890')
            state = {'offset':10, 'committed_prefix_sha256': worker.hashlib.sha256(b'1234567890').hexdigest()}
            path.write_bytes(b'0234567890')
            with self.assertRaisesRegex(ValueError, 'Full committed.*prefix changed'):
                worker.AppendCursor(path, state)

    def test_no_implicit_gpu_release_authorization(self):
        with self.assertRaisesRegex(RuntimeError, 'authorization required'):
            manager.validate(dict(node='N6',gpus=[3],release_authorized=False,
                phase=worker.PHASE,semantic_approval=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
