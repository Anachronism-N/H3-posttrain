"""Small directed recovery3 review/wait tests; no model or GPU use."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


w = module('recovery3_test_worker', RT / 'curated_io_recovery_review3_v1.py')
waiter = module('recovery3_test_waiter', RT / 'curated_io_recovery_review3_when_ready_v1.py')
manager = module('recovery3_test_manager', RT / 'curated_io_recovery_review3_resource_manager_v1.py')


class Review3Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, value in [('ROOT', self.root), ('RECOVERY', self.root / 'recovery'),
                            ('SELECTED', self.root / 'selected.json'), ('REQUESTS', self.root / 'requests.json'),
                            ('MANIFEST', self.root / 'manifest.json')]:
            previous = getattr(w, name)
            setattr(w, name, value)
            self.addCleanup(setattr, w, name, previous)
        self.originals = []
        for i in range(3):
            row = dict(id=f'original{i}', source_id=f'original{i}', split='train', batch_id='train-b000001',
                       source_ordinal=500+i, video_path='/video', audio_path='/audio')
            oldref = self.artifact(self.root / 'old' / row['id'] / 'producer.terminal.json', dict(**row, status='failed'))
            self.originals.append((row, dict(**row, old_terminal_ref=oldref)))
        for p in (w.SELECTED, w.REQUESTS, w.MANIFEST):
            self.artifact(p, dict(test_input=True))
        previous = w.source_rows
        w.source_rows = lambda: self.originals
        self.addCleanup(setattr, w, 'source_rows', previous)

    def artifact(self, path, value, addressed=False):
        raw = json.dumps(value).encode()
        if addressed:
            import hashlib
            path = path / (hashlib.sha256(raw).hexdigest() + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return w.ref(path)

    def terminal(self, i, *, failed=False, partial=False, edition=None):
        row, request = self.originals[i]
        root = w.RECOVERY / 'sources/train' / row['id']
        self.artifact(root / 'producer.terminal.json', dict(**row, status='ready'))
        candidate = dict(**row, label=dict(dance_style='unknown' if partial else 'hiphop',
                         timed_motion_script='[0-2] Both arms rise.'), joint_complete=not partial,
                         all_windows_nonempty=True, style_status='unknown' if partial else 'candidate',
                         visual_status='candidate', edition=edition or w.EDITION,
                         source_refs={}, recovery_revision_requests_ref=w.ref(w.REQUESTS),
                         old_failed_producer_terminal_ref=request['old_terminal_ref'])
        reference = self.artifact(root / 'joint-results', candidate, addressed=True)
        terminal = dict(id=row['id'], split='train', batch_id=row['batch_id'],
                        status='failed' if failed else 'incomplete_candidate' if partial else 'joint_candidate')
        if not failed:
            terminal['joint_candidate_ref'] = reference
        self.artifact(root / 'consumer.terminal.json', terminal)
        with (w.RECOVERY / 'completed.jsonl').open('a') as f:
            f.write(json.dumps(terminal) + '\n')
        return terminal

    def test_wait_exact_third_terminal_not_full_batch500(self):
        self.terminal(0); self.terminal(1)
        with self.assertRaisesRegex(w.NotReady, 'third'):
            w.ready_records()
        self.terminal(2)
        records, refs = w.ready_records()
        self.assertEqual([r['source_ordinal'] for r in records], [500, 501, 502])
        self.assertTrue(all(r['review_provenance']['original_batch_source_count'] == 500 for r in records))
        self.assertEqual(len(records), 3)
        self.assertTrue(refs)

    def test_wrong_label_edition_rejected(self):
        self.terminal(0, edition='other'); self.terminal(1); self.terminal(2)
        with self.assertRaisesRegex(ValueError, 'edition'):
            w.ready_records()

    def test_old_failed_ref_mismatch_rejected(self):
        notice = self.terminal(0); self.terminal(1); self.terminal(2)
        self.originals[0][1]['old_terminal_ref']['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'oldfailed'):
            w.ready_records()

    def test_content_SHA_changed_rejected(self):
        notice = self.terminal(0); self.terminal(1); self.terminal(2)
        Path(notice['joint_candidate_ref']['path']).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            w.ready_records()

    def test_unknown_extra_source_stream_rejected(self):
        self.terminal(0); self.terminal(1); self.terminal(2)
        with (w.RECOVERY / 'completed.jsonl').open('a') as f:
            f.write(json.dumps(dict(id='unknown')) + '\n')
        with self.assertRaisesRegex(ValueError, 'Unknown source'):
            w.ready_records()

    def test_every_reviewable_source_risk_audited_partial_failed_never_approved(self):
        self.terminal(0); self.terminal(1, partial=True); self.terminal(2, failed=True)
        rows, _ = w.ready_records()
        calls = []
        def fake(row, root, **kwargs):
            calls.append(row['id'])
            return dict(source_id=row['id'], status='pass', errors=[], uncertainties=[],
                        label_sha256=review.batch.digest(review.label_payload(row)),
                        review_receipt=dict(path=str(root / 'review/receipt.json'), sha256='synthetic'))
        out = self.root / 'reviews'
        result = w.review_records(rows, out, fake, review)
        self.assertEqual(calls, ['original0', 'original1'])
        self.assertEqual(result['selected_terminal_count'], 3)
        self.assertEqual(result['sampled_sources_reviewed'], 2)
        self.assertEqual(result['original_batch_source_count'], 500)
        self.assertFalse(result['original_batch_review_complete'])
        self.assertFalse(result['training_eligible'])
        self.assertEqual(result['label_incomplete_source_ids'], ['original1', 'original2'])
        records = [json.loads(Path(ref['path']).read_text()) for ref in result['review_record_refs']]
        self.assertTrue(all(r['label_edition'] == w.EDITION and not r['human_review_completed'] for r in records))

    def test_manager_refuses_G2_or_unapproved_G3(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU3'):
            manager.validate(dict(node='N6', gpus=[2], release_authorized=True))
        with self.assertRaisesRegex(RuntimeError, 'GPU3'):
            manager.validate(dict(node='N6', gpus=[3], release_authorized=False))

    def test_fresh_capture_unknown_active_or_active_lease_waits_without_signals(self):
        base = SimpleNamespace(lease_snapshot=lambda p: None, uuid_for=lambda g: 'uuid3',
            apps=lambda: [dict(uuid='uuid3', pid=12)], identity=lambda p: dict(pid=p),
            check_holder=lambda *args: (_ for _ in ()).throw(RuntimeError('not project holder')))
        m = SimpleNamespace(base=base, worker=w)
        with patch.object(waiter.subprocess, 'check_output', return_value='28.216.19.225'),              patch.object(waiter.Path, 'exists', return_value=False), patch.object(waiter.Path, 'is_symlink', return_value=False):
            with self.assertRaisesRegex(w.NotReady, 'not project holder'):
                waiter.fresh_capture(m)
            base.lease_snapshot = lambda p: (_ for _ in ()).throw(RuntimeError('active lease'))
            with self.assertRaisesRegex(w.NotReady, 'active lease'):
                waiter.fresh_capture(m)


if __name__ == '__main__':
    unittest.main()
