#!/usr/bin/env python3
"""Real completed-import/base-overlay tests, with source-bound negative mutations."""
import copy
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ms_curated_oldbar_overlay_v1 as o


class OldbarOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='oldbar_overlay_real13_v1_', dir='/tmp')
        cls.root = Path(cls.temp.name)
        cls.e = o.Evidence()
        cls.summary = cls.e.document({'path': str(o.BAR / 'curated222939_inputs_v1/manifest.summary.json'), 'sha256': o.SUMMARY_SHA})
        cls.base_progress, cls.base_rows, cls.base_refs = o.read_base(o.BAR / 'curated222939_overlay_v2', cls.e, cls.summary)
        cls.all_imports, cls.diagnostics = o.completed_imports(o.BAR / 'oldbar_explicit_import_v1', cls.e)
        ordered = sorted(cls.all_imports.items(), key=lambda item: o.stat(item[1]['disposition_ref']['path'])[3])
        cls.imports = dict(ordered[:13])
        assert len(cls.imports) == 13
        cls.e.prepare_csv([r['candidate'] for r in cls.all_imports.values()], cls.summary)
        cls.members = {key: o.validate_candidate(record, cls.e, cls.summary) for key, record in cls.imports.items()}
        cls.reviews = o.Reviews(o.BAR / 'curated_oldbar3_independent_reviews_v1', cls.e)
        cls.progress, cls.rows = o.derive(cls.base_progress, cls.base_rows, cls.imports, cls.members,
                                         cls.reviews, cls.summary, cls.diagnostics)
        cls.frozen_base = cls.root / 'base'
        cls.frozen_base.mkdir()
        (cls.frozen_base / 'progress.json').write_bytes(o.raw(cls.base_progress))
        (cls.frozen_base / 'candidates.jsonl').write_bytes(b''.join(o.raw(r) for r in cls.base_rows))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_real_thirteen_dedup_netnew_and_batch_membership(self):
        counts = self.progress['oldbar_incremental_counts']
        before = {o.key(r): r for r in self.base_rows}
        base_complete = {k for k, r in before.items() if r['state'] == 'complete_candidate'}
        self.assertEqual(counts['old_import_verified_unique'], 13)
        self.assertEqual(counts['old_import_net_complete_unique'], len(self.imports.keys() - base_complete))
        self.assertEqual(counts['old_import_new_source_unique'], len(self.imports.keys() - before.keys()))
        self.assertEqual(counts['old_import_overlap_base_unique'], len(self.imports.keys() & before.keys()))
        self.assertEqual(len({o.key(r) for r in self.rows}), len(self.rows))
        for row in self.rows:
            if o.key(row) in self.imports:
                self.assertEqual(row['batch_id'], self.members[o.key(row)]['batch_id'])
                old = [a for a in row['alternatives'] if a['origin'] == 'oldbar_verified_import']
                self.assertEqual(len(old), 1)
                self.assertNotEqual(old[0]['import_batch_id'], row['batch_id'])
                self.assertEqual(old[0]['original_curated_batch_id'], row['batch_id'])
        self.assertEqual(self.progress['counts']['complete_candidate'] - self.base_progress['counts']['complete_candidate'], counts['old_import_net_complete_unique'])

    def test_02_real_existing_same_source_complete_selection_preserved(self):
        base = {o.key(r): r for r in self.base_rows}
        overlap = next((k for k in self.all_imports if k in base and base[k]['state'] == 'complete_candidate'), None)
        self.assertIsNotNone(overlap, 'live real pool must contain already-complete source overlap')
        record = self.all_imports[overlap]
        member = o.validate_candidate(record, self.e, self.summary)
        progress, rows = o.derive(self.base_progress, self.base_rows, {overlap: record}, {overlap: member}, self.reviews, self.summary, [])
        selected = next(r for r in rows if o.key(r) == overlap)
        self.assertEqual(selected['selection'], base[overlap]['selection'])
        self.assertEqual(selected['failed_attempts'], base[overlap]['failed_attempts'])
        self.assertEqual(progress['oldbar_incremental_counts']['old_import_net_complete_unique'], 0)
        self.assertEqual(progress['oldbar_incremental_counts']['old_import_new_source_unique'], 0)
        self.assertEqual(progress['oldbar_incremental_counts']['old_import_overlap_base_unique'], 1)
        self.assertEqual(len(selected['alternatives']), len(base[overlap]['alternatives']) + 1)

    def test_03_base_partial_shows_complete_old_but_aierror_preserved(self):
        identity, record = next(iter(self.imports.items()))
        member = self.members[identity]
        old_label = {'dance_style': 'k-pop', 'timed_motion_script': 'partial prior caption'}
        failure = {'kind': 'failed', 'origin': 'production', 'reason': 'realistic_prior_failure'}
        review = {'status': 'confirmed_error', 'independent': True, 'errors': ['prior unsupported hand claim']}
        partial = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
            'state': 'partial_candidate', 'selection': {'kind': 'partial_candidate', 'label': old_label, 'reviews': [review]},
            'alternatives': [{'kind': 'partial_candidate', 'label': old_label, 'reviews': [review], 'pendinghuman': True}],
            'failed_attempts': [failure], 'ready_alternatives': [], 'pendinghuman': True}
        p, rows = o.derive(self.base_progress, [partial], {identity: record}, {identity: member}, self.reviews, self.summary, [])
        row = rows[0]
        self.assertEqual(row['state'], 'complete_candidate')
        self.assertEqual(row['base_state'], 'partial_candidate')
        self.assertEqual(row['base_selection'], partial['selection'])
        self.assertEqual(row['failed_attempts'], [failure])
        self.assertEqual(row['alternatives'][0], partial['alternatives'][0])
        self.assertTrue(row['pendinghuman'])
        self.assertTrue(row['approval_pending'])
        self.assertFalse(row['training_eligible'])

    def test_04_three_real_passes_matched_exact_semantics_not_approved(self):
        self.assertEqual(self.reviews.diagnostics, [])
        counts = self.progress['oldbar_incremental_counts']
        self.assertEqual(counts['old_import_individually_reviewed_unique'], 3)
        self.assertEqual(counts['old_import_sampled_pass_unique'], 3)
        reviewed = [r for r in self.rows if r.get('oldbar_import_present') and r['oldbar_individual_review_status'] == 'reviewed_exact_semantic_payload']
        self.assertEqual(len(reviewed), 3)
        for row in reviewed:
            alternative = next(a for a in row['alternatives'] if a['origin'] == 'oldbar_verified_import')
            review = alternative['reviews'][0]
            self.assertEqual(review['manifest_ref']['sha256'], o.PILOT_SHA)
            self.assertEqual(review['native_body_frames'], 16)
            self.assertEqual(review['native_hand_rois'], 6)
            self.assertFalse(review['current_review_source_used_to_validate_old_receipt'])
            self.assertFalse(row['training_eligible'])
            self.assertFalse(row['human_approved'])
        self.assertFalse(self.progress['all_batches_reviewed'])
        self.assertFalse(self.progress['oldbar_subset_original_batches_review_completed'])
        unreviewed = [r for r in self.rows if r.get('oldbar_import_present') and r['oldbar_individual_review_status'] == 'not_individually_reviewed']
        self.assertEqual(len(unreviewed), 10)

    def test_05_changed_semantic_label_cannot_reuse_three_passes(self):
        record = next(r for r in self.imports.values() if self.reviews.match(r['candidate']))
        candidate = copy.deepcopy(record['candidate'])
        candidate['label']['timed_motion_script'] += ' changed motion'
        self.assertEqual(self.reviews.match(candidate), [])

    def test_06_wrong_candidate_sha_rejected(self):
        record = copy.deepcopy(next(iter(self.imports.values())))
        record['candidate_ref']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'content SHA mismatch'):
            o.validate_candidate(record, self.e, self.summary)

    def test_07_wrong_current_media_binding_rejected(self):
        record = copy.deepcopy(next(iter(self.imports.values())))
        record['candidate']['source_refs']['import_provenance']['current_video']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'content SHA mismatch'):
            o.validate_candidate(record, self.e, self.summary)
        path = self.root / 'wrong-media'
        path.write_bytes(b'not original media')
        with self.assertRaisesRegex(ValueError, 'content SHA mismatch'):
            self.e.check({'path': str(path), 'sha256': next(iter(self.imports.values()))['candidate']['source_refs']['import_provenance']['current_video']['sha256']}, strong=True)

    def test_08_wrong_curated_split_or_original_batch_rejected(self):
        record = copy.deepcopy(next(iter(self.imports.values())))
        record['candidate']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'member/line SHA differs'):
            o.validate_candidate(record, self.e, self.summary)
        record = copy.deepcopy(next(iter(self.imports.values())))
        record['candidate']['original_curated_batch_id'] = 'test-b000000'
        with self.assertRaisesRegex(ValueError, 'descriptor differs'):
            o.validate_candidate(record, self.e, self.summary)

    def test_09_base_change_fail_closed(self):
        original_stat = o.stat
        calls = Counter()
        def changed(path):
            state = original_stat(path)
            calls[str(path)] += 1
            if Path(path).name == 'progress.json' and calls[str(path)] > 1:
                state[3] += 1
            return state
        with patch.object(o, 'stat', changed):
            with self.assertRaisesRegex(ValueError, 'pending_unstable_base_snapshot'):
                o.read_base(self.frozen_base, self.e, self.summary)
        wrong = copy.deepcopy(self.base_progress)
        wrong['counts']['complete_candidate'] += 1
        root = self.root / 'inconsistent-base'
        root.mkdir()
        (root / 'progress.json').write_bytes(o.raw(wrong))
        (root / 'candidates.jsonl').write_bytes((self.frozen_base / 'candidates.jsonl').read_bytes())
        with self.assertRaisesRegex(ValueError, 'cross_snapshot_count_mismatch'):
            o.read_base(root, self.e, self.summary)

    def test_10_invalid_journal_and_failed_history_preserved(self):
        self.assertEqual(self.progress['invalid_records'], self.base_progress['invalid_records'])
        self.assertEqual(self.progress['journal_recovery_scope'], self.base_progress['journal_recovery_scope'])
        self.assertEqual(self.progress['recovery_requests'], self.base_progress['recovery_requests'])
        self.assertEqual(self.progress['counts']['pendinghuman'], self.base_progress['counts']['pendinghuman'])
        self.assertEqual(self.progress['counts']['confirmed_AI_error'], self.base_progress['counts']['confirmed_AI_error'])
        base = {o.key(r): r for r in self.base_rows}
        for row in self.rows:
            if o.key(row) in base:
                self.assertEqual(row['failed_attempts'], base[o.key(row)]['failed_attempts'])
                self.assertEqual(row['alternatives'][:len(base[o.key(row)]['alternatives'])], base[o.key(row)]['alternatives'])

    def test_11_all_global_split_and_448_batch_denominators(self):
        self.assertEqual(sum(self.progress['counts'][s] for s in o.STATES), self.summary['total'])
        self.assertEqual(len(self.progress['batches']), 448)
        self.assertEqual(sum(b['expected'] for b in self.progress['batches']), self.summary['total'])
        for split, expected in self.summary['counts'].items():
            self.assertEqual(sum(self.progress['split_counts'][split][s] for s in o.STATES), expected)
        for batch in self.progress['batches']:
            self.assertEqual(sum(batch['counts'][s] for s in o.STATES), batch['expected'])
            self.assertFalse(batch['oldbar_subset_approves_original_batch'])
        self.assertEqual(sum(b['counts']['complete_candidate'] for b in self.progress['batches']), self.progress['counts']['complete_candidate'])
        self.assertEqual(self.progress['counts']['ready_unique'], self.base_progress['counts']['ready_unique'])
        self.assertEqual(self.progress['counts']['terminal_unique'], self.base_progress['counts']['terminal_unique'])

    def test_12_candidate_without_finished_disposition_is_pending(self):
        root = self.root / 'candidate-before-disposition'
        (root / 'candidates').mkdir(parents=True)
        record = next(iter(self.imports.values()))
        (root / 'candidates' / (record['candidate']['id'] + '.json')).write_bytes(Path(record['candidate_ref']['path']).read_bytes())
        found, diagnostics = o.completed_imports(root, self.e)
        self.assertEqual(found, {})
        self.assertEqual(diagnostics[0]['state'], 'pending')
        self.assertEqual(diagnostics[0]['reason'], 'candidate_written_but_completed_disposition_missing')

    def test_13_local_sha_cache_rehashes_changed_files(self):
        path = self.root / 'cached-evidence'
        path.write_bytes(b'original')
        ref = {'path': str(path), 'sha256': o.sha_bytes(path.read_bytes())}
        self.e.check(ref)
        self.e.check(ref)
        path.write_bytes(b'changed size')
        with self.assertRaisesRegex(ValueError, 'content SHA mismatch'):
            self.e.check(ref)

    def test_14_real_thirteen_once_output_and_incremental_cache(self):
        out = self.root / 'derived'
        args = o.parser().parse_args(['--base-root', str(self.frozen_base), '--out', str(out), '--once'])
        original = o.completed_imports
        def thirteen(root, evidence):
            found, diagnostics = original(root, evidence)
            return {k: found[k] for k in self.imports}, diagnostics
        with patch.object(o, 'completed_imports', thirteen), contextlib.redirect_stdout(io.StringIO()):
            first = o.run(args)
            second = o.run(args)
        self.assertEqual(first['oldbar_incremental_counts'], second['oldbar_incremental_counts'])
        self.assertEqual(first['oldbar_incremental_counts']['old_import_verified_unique'], 13)
        rows = [json.loads(s) for s in (out / 'candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), len({o.key(r) for r in rows}))
        self.assertEqual(o.sha_bytes((out / 'candidates.jsonl').read_bytes()), second['candidate_export_sha256'])
        cache = Path(second['state_cache']['path'])
        self.assertTrue(cache.is_relative_to(Path('/tmp')))
        self.assertFalse(second['state_cache']['authority'])


    def test_15_legal_base_update_keeps_verified_fixed_snapshot_not_latest(self):
        out = self.root / 'derived'
        before = (out / 'candidates.jsonl').read_bytes()
        progress_before = (out / 'progress.json').read_bytes()
        base_before = (self.frozen_base / 'progress.json').read_bytes()
        args = o.parser().parse_args(['--base-root', str(self.frozen_base), '--out', str(out), '--once'])
        original = o.derive
        def changing(*values):
            result = original(*values)
            changed = copy.deepcopy(self.base_progress)
            changed['updated_unix'] += 1
            (self.frozen_base / 'progress.json').write_bytes(o.raw(changed))
            return result
        try:
            with patch.object(o, 'derive', changing), patch.object(o, 'completed_imports', return_value=(self.imports, [])), contextlib.redirect_stdout(io.StringIO()):
                result = o.run(args)
            self.assertTrue(result['base_snapshot_only'])
            self.assertFalse(result['latest_live_base_claimed'])
            self.assertEqual(result['base_counts'], self.base_progress['counts'])
            self.assertEqual(result['oldbar_incremental_counts']['old_import_verified_unique'], 13)
            self.assertEqual((out / 'candidates.jsonl').read_bytes(), before)
            snapshot = result['frozen_base_snapshot']
            self.assertIn('read_atUTC', snapshot)
            self.assertIn('updatedUTC', snapshot)
            progress_ref = next(r for r in snapshot['immutable_refs'] if Path(r['path']).name == 'progress.json')
            self.assertEqual(Path(progress_ref['path']).read_bytes(), base_before)
            self.assertNotEqual((self.frozen_base / 'progress.json').read_bytes(), base_before)
        finally:
            (self.frozen_base / 'progress.json').write_bytes(base_before)


    def test_16_immutable_snapshot_tamper_refused_and_output_unchanged(self):
        out = self.root / 'derived'
        before = (out / 'candidates.jsonl').read_bytes()
        progress_before = (out / 'progress.json').read_bytes()
        args = o.parser().parse_args(['--base-root', str(self.frozen_base), '--out', str(out), '--once'])
        original = o.derive
        modified = []
        def tampering(*values):
            result = original(*values)
            path = next((out / 'snapshots').glob('*/candidates.jsonl'))
            old = path.read_bytes()
            path.write_bytes(old + b'changed snapshot bytes')
            modified.append((path, old))
            return result
        try:
            with patch.object(o, 'derive', tampering), patch.object(o, 'completed_imports', return_value=(self.imports, [])), contextlib.redirect_stdout(io.StringIO()):
                result = o.run(args)
            self.assertEqual(result['state'], 'pending_stable_base_snapshot')
            self.assertIn('immutable_base_snapshot_changed', result['reason'])
            self.assertEqual((out / 'candidates.jsonl').read_bytes(), before)
            self.assertEqual((out / 'progress.json').read_bytes(), progress_before)
        finally:
            for path, old in modified:
                path.write_bytes(old)

    def test_17_same_global_counts_wrong_split_or_batch_snapshot_rejected(self):
        for dimension in ('split', 'batch'):
            root = self.root / ('wrong-' + dimension)
            root.mkdir()
            progress = copy.deepcopy(self.base_progress)
            if dimension == 'split':
                progress['split_counts']['train']['complete_candidate'] += 1
            else:
                progress['batches'][0]['counts']['complete_candidate'] += 1
            (root / 'progress.json').write_bytes(o.raw(progress))
            (root / 'candidates.jsonl').write_bytes((self.frozen_base / 'candidates.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'pending_snapshot_mismatch'):
                o.read_base(root, self.e, self.summary)

    def test_18_real_rejected_raq_is_alternative_only_never_borrows_new_pass(self):
        identity = 'train/RAQoxqRxCfMvTHSx'
        record = self.all_imports[identity]
        member = o.validate_candidate(record, self.e, self.summary)
        self.assertEqual(self.reviews.match(record['candidate']), [])
        progress, rows = o.derive(self.base_progress, [], {identity: record}, {identity: member}, self.reviews, self.summary, [])
        source = rows[0]
        self.assertIsNone(source['selection'])
        self.assertEqual(source['state'], 'missing')
        self.assertTrue(source['known_source_review_risk']['oldbar_original_rejected'])
        self.assertTrue(source['pendinghuman'])
        self.assertEqual(source['alternatives'][0]['reviews'], [])
        self.assertEqual(source['alternatives'][0]['individual_review_status'], 'not_individually_reviewed')
        self.assertEqual(progress['oldbar_incremental_counts']['old_import_net_complete_unique'], 0)
        self.assertEqual(sum(progress['counts'][s] for s in o.STATES), self.summary['total'])
        new_selection = {'kind': 'complete_candidate', 'label': {'dance_style': 'k-pop',
            'timed_motion_script': 'new scope-v3 bilateral arm repaired semantic payload'},
            'reviews': [{'status': 'pass', 'independent': True}], 'reviewed': True, 'sampled_pass': True}
        prior = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
            'state': 'complete_candidate', 'selection': new_selection, 'alternatives': [new_selection],
            'ready_alternatives': [], 'failed_attempts': [], 'pendinghuman': True}
        _, rows = o.derive(self.base_progress, [prior], {identity: record}, {identity: member}, self.reviews, self.summary, [])
        self.assertEqual(rows[0]['selection'], new_selection)
        self.assertEqual(rows[0]['alternatives'][1]['reviews'], [])
        self.assertTrue(rows[0]['pendinghuman'])
        self.assertFalse(rows[0]['training_eligible'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
