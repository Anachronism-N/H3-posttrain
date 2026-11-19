"""Targeted frozen-review tests: no model, RGB decoding or GPU allocation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review
spec = importlib.util.spec_from_file_location('review_extension_test', RT / 'curated_review_extension_watch_v1.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class ExtensionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id=f's{i:03d}', split='train', batch_id='train-b000040',
                          video_path='/video', audio_path='/audio', original_caption='original') for i in range(40)]
        self.members = {r['id']: r for r in self.rows}
        self.manifest = SimpleNamespace(path=self.root / 'manifest.json', sha256='original-manifest-SHA',
            members=lambda bid: self.members, descriptors={'train-b000040': dict(batch_id='train-b000040', count=40)})
        self.collab = self.root / 'curated222939_collaborative_v1'
        self.repairs = self.root / 'curated222939_repair_v1'
        self.out = self.root / 'curated_independent_reviews_extension_v1'
        self.calls = []
        self.errors = set()
        self.watcher = w.Watcher(self.manifest, self.collab, self.repairs, self.out, self.fake_review, review)

    def fake_review(self, row, root, **kwargs):
        self.calls.append((row['id'], str(root), copy.deepcopy(row)))
        status = 'confirmed_error' if row['id'] in self.errors else 'pass'
        errors = [dict(claimed_text='raises hands', actual_visual_evidence='visibly lowers hands',
                       frame_indices=[0, 1], error_type='hands', severity='major', evidence_kind='direct_contradiction')] if status == 'confirmed_error' else []
        return dict(source_id=row['id'], status=status, errors=errors, uncertainties=[],
                    label_sha256=review.batch.digest(review.label_payload(row)),
                    review_receipt=dict(path=str(Path(root) / 'review/receipt.json'), sha256='synthetic-receipt'))

    def artifact(self, path, value, addressed=False):
        data = json.dumps(value).encode()
        if addressed:
            path = path / (__import__('hashlib').sha256(data).hexdigest() + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return dict(path=str(path), sha256=w.sha(path))

    def candidate(self, row, unknown=False):
        return {**copy.deepcopy(row), 'label': dict(dance_style='unknown' if unknown else 'hiphop',
                    timed_motion_script='[0-2] The dancer raises both hands.'),
                'joint_complete': not unknown, 'all_windows_nonempty': True,
                'style_status': 'unknown' if unknown else 'candidate', 'visual_status': 'candidate',
                'source_refs': {}, 'annotator_model_sha256': 'annotator-test-SHA'}

    def collaborative_notice(self, row):
        ref = self.artifact(self.collab / 'sources' / row['split'] / row['id'] / 'joint-results',
                            self.candidate(row), addressed=True)
        return dict(id=row['id'], split=row['split'], batch_id=row['batch_id'], status='joint_candidate', joint_candidate_ref=ref)

    def repair_notice(self, row, unknown=False):
        candidate = self.candidate(row, unknown)
        original = self.artifact(self.root / 'curated222939_production_v1/sources' / row['split'] / row['id'] / 'joint-results',
                                 self.candidate(row), addressed=True)
        candidate.update(global_revision='repair-20261025-v1', risk_review_case=True, original_result_ref=original)
        ref = self.artifact(self.repairs / 'joint/revisions/repair-20261025-v1' / row['split'] / row['id'] / 'result.json', candidate)
        return dict(id=row['id'], source_key=row['split'] + '/' + row['id'], batch_id=row['batch_id'],
                    global_revision='repair-20261025-v1', joint_result_ref=ref, result_ref=ref)

    def append(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')

    def test_fixed_random_membership_with_pending_labels_and_expansion(self):
        pending = [{**r, 'label_status': 'pending'} for r in self.rows]
        complete = [self.candidate(r) for r in self.rows]
        first = review.sampling_plan(pending, 'train-b000040')
        last = review.sampling_plan(complete, 'train-b000040')
        self.assertEqual(first['random_source_ids'], last['random_source_ids'])
        self.errors = set(last['selected_source_ids'][:5])
        self.append(self.collab / 'completed.jsonl', [self.collaborative_notice(r) for r in self.rows])
        progress = self.watcher.poll()
        result = next(iter(self.watcher.batch_summaries.values()))
        self.assertEqual([r['random_target'] for r in result['rounds']], [25, 40])
        self.assertEqual(progress['collaborative_sampled_reviewed'], 40)
        self.assertFalse(result['acceptance']['eligible_for_training'])
        self.assertFalse(result['human_review_completed'])
        self.assertEqual(len(last['risk_source_ids']), 5)
        self.assertEqual(result['label_edition'], 'collaborative-v1')

    def test_every_joint_revision_risk_reviewed_and_not_old_score_reused(self):
        self.append(self.collab / 'completed.jsonl', [self.collaborative_notice(self.rows[0])])
        self.append(self.repairs / 'joint/review-required.jsonl', [self.repair_notice(self.rows[0]), self.repair_notice(self.rows[1])])
        progress = self.watcher.poll()
        self.assertEqual(progress['joint_revision_notice_count'], 2)
        self.assertEqual(progress['joint_revision_reviewed_count'], 2)
        for v in self.watcher.repair_summaries.values():
            self.assertTrue(v['review_provenance']['every_joint_revision_is_risk_reviewed'])
            self.assertFalse(v['review_provenance']['original_review_scores_reused'])
        repair = [c for c in self.calls if 'joint-revisions' in c[1]]
        self.assertEqual({c[0] for c in repair}, {'s000', 's001'})
        self.assertTrue(all(c[2]['review_provenance']['label_edition'] == 'repair-20261025-v1' for c in repair))
        before = len(self.calls)
        self.watcher.poll()
        self.assertEqual(len(self.calls), before)
        collab_root = review.batch.digest(self.watcher.completed['train-b000040']['s000'])
        repair_root = review.batch.digest(repair[0][2])
        self.assertNotEqual(collab_root, repair_root)

    def test_unknown_partial_can_be_audited_but_never_passed(self):
        self.append(self.repairs / 'joint/review-required.jsonl', [self.repair_notice(self.rows[0], unknown=True)])
        progress = self.watcher.poll()
        result = next(iter(self.watcher.repair_summaries.values()))
        self.assertEqual(progress['joint_revision_reviewed_count'], 1)
        self.assertEqual(progress['joint_revision_label_incomplete_count'], 1)
        self.assertFalse(result['algorithmic_checks_satisfied'])
        self.assertFalse(result['acceptance']['eligible_for_training'])
        self.assertEqual(result['state'], 'pending_label_or_review')

    def test_source_mismatch_and_changed_label_SHA_rejected(self):
        notice = self.collaborative_notice(self.rows[0])
        ref = notice['joint_candidate_ref']
        Path(ref['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.watcher.collaborative_row(notice)
        row = self.candidate(self.rows[0])
        row['video_path'] = '/different-original-video'
        with self.assertRaisesRegex(ValueError, 'media mismatch'):
            w.bound_row(self.rows[0], row, dict(path='/label', sha256='test'),
                        edition='test', origin_root=self.collab, stream_path='/stream')

    def test_visual_only_joint_stream_or_revision_mismatch_rejected(self):
        notice = self.repair_notice(self.rows[0])
        notice['joint_result_ref']['path'] = str(self.repairs / 'revisions/repair-20261025-v1/train/s000/result.json')
        with self.assertRaisesRegex(ValueError, 'JOINT revision'):
            self.watcher.repair_row(notice)
        notice = self.repair_notice(self.rows[0])
        notice['global_revision'] = 'other-edition'
        with self.assertRaisesRegex(ValueError, 'edition'):
            self.watcher.repair_row(notice)

    def extra_notice(self):
        sid, edition = 'RAQoxqRxCfMvTHSx', 'raq-scope-offline-20261025-v3'
        original = self.candidate(dict(id=sid, split='train', batch_id='independent-pilot25-v1', video_path='/video', audio_path='/audio'))
        original['label']['timed_motion_script'] = '[0-1] original w0\n[1-2] original w1\n[2-3] original w2'
        pilot = self.root / 'runtime/independent_review_pilot25_rows_v1.jsonl'
        self.append(pilot, [original])
        source = self.artifact(self.watcher.extra_root / 'joint-revisions' / edition / 'train' / sid / 'source_record.json', original)
        old = self.artifact(self.root / 'old-visual.json', dict(manifest_record=original))
        visual = self.artifact(self.root / 'legacy_raq_repair_v1/revisions' / edition / 'train' / sid / 'result.json', dict(id=sid, global_revision=edition))
        roi = self.artifact(self.root / 'original-roi.json', dict(test_roi=True))
        candidate = copy.deepcopy(original)
        candidate['label']['timed_motion_script'] = '[0-1] original w0\n[1-2] circles both arms near face then extends outward\n[2-3] original w2'
        candidate.update(global_revision=edition, extra_visual_revision_ref=visual, original_visual_result_ref=old,
            original_source_record_ref=source, original_pilot_rows_ref=dict(path=str(pilot), sha256=w.sha(pilot)),
            source_row_sha256=review.batch.digest(original), new_model_calls=0, model_rewrite_performed=False,
            hand_roi=dict(available=True, **roi))
        ref = self.artifact(self.watcher.extra_root / 'joint-revisions' / edition / 'train' / sid / 'result.json', candidate)
        return dict(id=sid, split='train', source_key='train/' + sid, batch_id=original['batch_id'],
                    global_revision=edition, result_ref=ref, source_record_ref=source, visual_revision_ref=visual, risk_review_case=True)

    def test_extra_RAQ_risk_audit_and_fixed_review_ledger(self):
        notice = self.extra_notice()
        self.append(self.watcher.extra.path, [notice])
        progress = self.watcher.poll()
        self.assertEqual(progress['extra_visual_revision_notice_count'], 1)
        self.assertEqual(progress['extra_visual_revision_reviewed_count'], 1)
        result = next(iter(self.watcher.extra_summaries.values()))
        ref = result['review_records_ref']
        self.assertEqual(w.sha(ref['path']), ref['sha256'])
        records = [json.loads(l) for l in Path(ref['path']).read_text().splitlines()]
        self.assertEqual(records[0]['source_id'], notice['id'])
        self.assertEqual(records[0]['split'], 'train')
        self.assertEqual(records[0]['label_sha256'], notice['result_ref']['sha256'])
        self.assertEqual(records[0]['label_edition'], notice['global_revision'])
        self.assertFalse(records[0]['human_review_completed'])
        self.assertFalse(result['acceptance']['eligible_for_training'])
        self.assertIsNone(result['human_resolution_stream'])

    def test_extra_RAQ_changed_original_style_rejected(self):
        notice = self.extra_notice()
        path = Path(notice['result_ref']['path'])
        value = json.loads(path.read_text())
        value['label']['dance_style'] = 'different'
        self.artifact(path, value)
        notice['result_ref']['sha256'] = w.sha(path)
        with self.assertRaisesRegex(ValueError, 'preserve original pilot style'):
            self.watcher.extra_row(notice)

    def test_stream_partial_lines_and_truncation(self):
        path = self.root / 'stream.jsonl'
        path.write_bytes(b'{}\n{"id":')
        stream = w.Stream(path)
        self.assertEqual(list(stream.rows()), [{}])
        self.assertEqual(stream.offset, 3)
        with path.open('ab') as f:
            f.write(b'1}\n')
        self.assertEqual(list(stream.rows()), [dict(id=1)])
        path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            list(stream.rows())

    def test_duplicate_notice_with_changed_label_edition_rejected(self):
        notice = self.collaborative_notice(self.rows[0])
        self.watcher.collaborative_row(notice)
        changed = copy.deepcopy(notice)
        changed['status'] = 'different_status'
        with self.assertRaisesRegex(ValueError, 'edition'):
            self.watcher.collaborative_row(changed)


if __name__ == '__main__':
    unittest.main()
