import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ms_curated_overlay as m


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.refs = {'csv': {'path': '/source/train.csv', 'sha256': 'csv', 'csv_line': 2, 'raw_row_sha256': 'row'}}
        self.rows = [{'source_id': 'a', 'id': 'a', 'split': 'train', 'batch_id': 'train-b0', 'source_ordinal': 0, 'source_refs': self.refs},
                     {'source_id': 'b', 'id': 'b', 'split': 'train', 'batch_id': 'train-b0', 'source_ordinal': 1, 'source_refs': self.refs},
                     {'source_id': 'a', 'id': 'a', 'split': 'val', 'batch_id': 'val-b0', 'source_ordinal': 2, 'source_refs': self.refs}]
        batches = []
        for bid, rows in [('train-b0', self.rows[:2]), ('val-b0', self.rows[2:])]:
            ref = self.write(self.root / 'inputs' / (bid + '.jsonl'), rows, lines=True)
            batches.append({'batch_id': bid, 'split': rows[0]['split'], 'count': len(rows), **ref})
        self.summary = self.root / 'inputs/manifest.summary.json'
        self.write(self.summary, {'total': 3, 'counts': {'train': 2, 'val': 1}, 'batches': batches})
        self.overlay = m.Overlay(self.summary)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, value, lines=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b''.join((json.dumps(row) + '\n').encode() for row in value) if lines else m.encoded(value)
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

    def stream(self, rows, kind='candidate', origin='production', name='completed.jsonl'):
        path = self.root / origin / name
        self.write(path, rows, lines=True)
        self.overlay.stream(path, origin, kind)
        return path

    def candidate(self, index=0, name='base', completed=True, revision=False):
        result = {**self.rows[index], 'joint_complete': completed, 'all_windows_nonempty': completed,
                  'visual_status': 'candidate', 'style_status': 'resolved',
                  'label': {'dance_style': 'pop', 'timed_motion_script': name}}
        if revision:
            result['global_revision'] = 'edition-v2'
            result['original_result_ref'] = self.base_ref
        ref = self.write(self.root / (name + str(index) + '.json'), result)
        if name == 'base':
            self.base_ref = ref
        notice = {**self.rows[index], 'joint_candidate_ref': ref}
        if revision:
            notice = {'id': self.rows[index]['id'], 'source_key': m.identity(self.rows[index]),
                      'batch_id': self.rows[index]['batch_id'], 'global_revision': 'edition-v2', 'result_ref': ref}
        return notice, result

    def audit(self, notice, result, status='pass'):
        ref = notice.get('joint_candidate_ref', notice.get('result_ref'))
        input_sha = 'review-input'
        binding = {'source_id': result['id'], 'split': result['split'], 'label_sha256': m.digest(m.payload(result)),
                   'label_file_ref': ref, 'input_row_sha256': input_sha,
                   'checkpoint_content_independence_verified': True}
        final_ref = self.write(self.root / 'review_refs' / (status + '-final.json'),
            {'status': 'candidate', 'format_valid': True, 'json': {'status': status}})
        receipt_ref = self.write(self.root / 'review_refs' / (status + '-receipt.json'),
            {'binding': binding, 'files_sha256': {'final.json': final_ref['sha256']}})
        return {**self.rows[0], 'original_batch_id': result['batch_id'], 'label_path': ref['path'],
                'label_sha256': ref['sha256'], 'semantic_label_sha256': binding['label_sha256'],
                'input_row_sha256': input_sha, 'review_receipt': receipt_ref, 'review_final_ref': final_ref, 'status': status}

    def ledger(self, directory, row):
        ledger_ref = self.write(directory / 'reviews.jsonl', [row], lines=True)
        self.write(directory / 'summary.json', {'review_records_ref': ledger_ref})
        self.overlay.review_batch(directory, row['original_batch_id'])

    def clause_candidate(self, completed=True, fact=None):
        self.edition = 'clause-salvage-20261025-v1'
        self.overlay.clause_salvage_root = self.root / 'curated222939_clause_salvage_v1'
        self.clause_path = self.overlay.clause_salvage_root / 'revisions' / self.edition / 'train/a/result.json'
        self.visual_path = self.clause_path.with_name('visual_result.json')
        raw_visual = self.write(self.root / 'raw-visual.json', {'id': 'a', 'source_refs': {'original_csv': self.refs['csv']}})
        style_ref = self.write(self.root / 'style.json', {**self.rows[0]})
        base, original = self.candidate()
        original.update(visual_result_ref=raw_visual, style_result_ref=style_ref,
                        source_binding={'source_binding': {'id': 'a', 'video_sha256': 'scene-a'}})
        self.base_ref = base['joint_candidate_ref'] = self.write(self.root / 'base0.json', original)
        job_ref = self.write(self.root / 'job.json', {**self.rows[0], 'original_result_ref': self.base_ref})
        raw_repair_ref = self.write(self.root / 'raw-repair.json', {'id': 'a'})
        old_ref = self.write(self.root / 'old-repair-joint.json', {**original,
            'original_result_ref': self.base_ref, 'repair_result_ref': raw_repair_ref,
            'original_visual_result_ref': raw_visual, 'original_job_ref': job_ref})
        flags = dict(VLM_dispatch_performed=False, attempt_budget_consumed=False, model_call_performed=False,
                     model_rewrite_performed=False, original_raw_or_terminal_modified=False,
                     model_calls=0, extra_source_attempts=0, **m.FLAGS)
        fact = fact or {'id': 'w0:e0', 'fact_id': 'w0:e0', 'window_id': 'w0', 'ai_weak_candidate': True,
                'raw_text': 'Raises both arms.', 'text': 'Raises both arms.', 'source_clause': {'raw_source_clause': 'Raises both arms.',
                    'original_description': 'Raises both arms. The hand is uncertain.',
                    'source_text_is_verbatim_substring': True}, **m.FLAGS}
        audit = {'window_id': fact['window_id'], 'actual_frame_SHA_verified': True,
                 'scope_filter_audit': {'actual_part_support_gate_unchanged': True,
                    'native_named_shape_gate_preserved': True, 'filter_is_not_semantic_truth_review': True,
                    'affirmative_text_modified': False, 'comma_splitting_added': False,
                    'frame_references_added': False, 'uncertainty_tokens_removed_to_strengthen_claim': False},
                 'support_filter_audit': {'raw_response_changed': False, 'remaining_reliable_parts_preserved': True,
                                          'filter_is_not_semantic_truth_review': True}}
        shared = {**self.rows[0], **flags, 'source_key': 'train/a', 'global_revision': self.edition,
                  'new_safe_fact_ids': [fact['fact_id']], 'new_safe_facts': [fact],
                  'original_result_ref': self.base_ref, 'original_visual_result_ref': raw_visual,
                  'original_repair_joint_result_ref': old_ref, 'original_repair_result_ref': raw_repair_ref,
                  'original_job_ref': job_ref, 'original_source_binding': original['source_binding']['source_binding'],
                  'style_label_changed': False}
        self.clause_visual = {**shared, 'facts': [fact], 'window_derivation_audit': [audit],
                              'final_timed_caption': fact['raw_text']}
        visual_ref = self.write(self.visual_path, self.clause_visual)
        result = {**original, **shared, 'visual_result_ref': visual_ref,
                  'joint_complete': completed, 'all_windows_nonempty': completed,
                  'label': {'dance_style': 'pop', 'timed_motion_script': fact['raw_text']}}
        notice = {**shared, 'joint_complete': completed, 'visual_result_ref': visual_ref,
                  'new_safe_fact_refs': [{'fact_id': fact['fact_id'], 'visual_result_ref': visual_ref}]}
        return self.clause_notice(notice, result), result, base, original

    def clause_notice(self, notice, result):
        ref = self.write(self.clause_path, result)
        return {**notice, 'result_ref': ref, 'joint_candidate_ref': ref}

    def clause_ledger(self, root, notice, result, status='pass'):
        audit_bid = 'train-' + hashlib.sha256(b'a').hexdigest()[:24]
        directory = root / self.edition / audit_bid
        input_row = {**result, 'label_path': notice['result_ref']['path'],
                     'label_sha256': notice['result_ref']['sha256']}
        row = self.audit(notice, result, status)
        row['label_edition'] = self.edition
        row['input_row_sha256'] = m.digest(input_row)
        receipt = json.loads(Path(row['review_receipt']['path']).read_bytes())
        receipt['binding']['input_row_sha256'] = row['input_row_sha256']
        row['review_receipt'] = self.write(directory / 'receipt.json', receipt)
        ledger_ref = self.write(directory / 'reviews.jsonl', [row], lines=True)
        self.write(directory / 'batch.plan.json', {'batch_id': audit_bid, 'split': 'train', 'records': [input_row]})
        self.write(directory / 'summary.json', {'batch_id': audit_bid, 'original_batch_id': 'train-b0',
                   'label_edition': self.edition, 'review_records_ref': ledger_ref, 'statuses': {'pass': 99}})
        self.overlay.review_batch(directory, 'train-b0', audit_bid, clause_edition=self.edition)
        return directory, row

    def test_clause_main_joint_required_and_original_visual_ref_separate(self):
        notice, result, base, _ = self.clause_candidate()
        self.stream([notice, notice], kind='clause_revision', origin='clause_salvage')
        status, rows = self.overlay.snapshot()
        candidate = rows[0]['selection']
        self.assertEqual(candidate['original_joint_candidate_ref'], base['joint_candidate_ref'])
        self.assertEqual(candidate['original_result_ref'], base['joint_candidate_ref'])
        self.assertEqual(candidate['original_visual_result_ref'], result['original_visual_result_ref'])
        self.assertNotEqual(candidate['original_joint_candidate_ref'], candidate['original_visual_result_ref'])
        self.assertEqual(candidate['original_job_ref'], result['original_job_ref'])
        self.assertEqual(candidate['original_source_binding'], result['original_source_binding'])
        self.assertEqual(len(rows[0]['alternatives']), 1)
        self.assertEqual(status['counts']['sampled_pass'], 0)
        wrong_result = {**result, 'original_result_ref': result['original_visual_result_ref']}
        with self.assertRaisesRegex(ValueError, 'JOINT/VISUAL'):
            self.overlay.normalize(self.clause_notice(notice, wrong_result), 'clause_salvage', 'clause_revision')
        old = self.overlay.checked(result['original_repair_joint_result_ref']).copy()
        old['original_result_ref'] = result['original_visual_result_ref']
        result['original_repair_joint_result_ref'] = self.write(self.root / 'fake-joint.json', old)
        with self.assertRaisesRegex(ValueError, 'main JOINT'):
            self.overlay.normalize(self.clause_notice(notice, result), 'clause_salvage', 'clause_revision')

    def test_clause_rejects_ref_identity_style_scene_and_repair_mismatches(self):
        for field, value in [('original_repair_result_ref', self.write(self.root / 'wrong-repair.json', {'id': 'a'})),
                             ('original_source_binding', {'id': 'a', 'video_sha256': 'other-scene'}),
                             ('original_job_ref', self.write(self.root / 'wrong-job.json', {'id': 'a'})),
                             ('id', 'b'), ('split', 'val'), ('batch_id', 'val-b0'),
                             ('source_refs', {'csv': {'path': 'wrong'}}), ('style_label_changed', True)]:
            with self.subTest(field=field):
                notice, result, _, _ = self.clause_candidate()
                result[field] = value
                with self.assertRaises(ValueError):
                    self.overlay.normalize(self.clause_notice(notice, result), 'clause_salvage', 'clause_revision')
        notice, result, _, _ = self.clause_candidate()
        notice['result_ref'] = self.base_ref
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.overlay.normalize(notice, 'clause_salvage', 'clause_revision')

    def test_clause_zero_vlm_budget_and_named_shape_gates_required(self):
        for field in ('VLM_dispatch_performed', 'attempt_budget_consumed', 'model_call_performed', 'formal_training'):
            with self.subTest(field=field):
                notice, result, _, _ = self.clause_candidate()
                result[field] = True
                with self.assertRaisesRegex(ValueError, 'flags'):
                    self.overlay.normalize(self.clause_notice(notice, result), 'clause_salvage', 'clause_revision')
        for field, value in [('native_named_shape_gate_preserved', False),
                             ('uncertainty_tokens_removed_to_strengthen_claim', True)]:
            with self.subTest(field=field):
                notice, result, _, _ = self.clause_candidate()
                self.clause_visual['window_derivation_audit'][0]['scope_filter_audit'][field] = value
                result['visual_result_ref'] = notice['visual_result_ref'] = self.write(self.visual_path, self.clause_visual)
                notice['new_safe_fact_refs'][0]['visual_result_ref'] = result['visual_result_ref']
                with self.assertRaisesRegex(ValueError, 'named-shape'):
                    self.overlay.normalize(self.clause_notice(notice, result), 'clause_salvage', 'clause_revision')

    def test_clause_real_neutral_pronoun_fixture_preserves_verbatim_caption(self):
        raw = 'The dancer moves her right arm forward and slightly upward, with the hand blurred in motion, near the chest area.'
        fact = {'id': 'w00001:e1_c0', 'fact_id': 'w00001:e1_c0', 'window_id': 'w00001',
                'ai_weak_candidate': True, 'frame_indices': [38, 40], 'parts': ['right_arm', 'right_hand'],
                'raw_text': raw,
                'text': "The dancer moves the dancer's right arm forward and slightly upward, with the hand blurred in motion, near the chest area.",
                'source_clause': {'raw_source_clause': raw,
                    'original_description': raw + ' The hand appears to be open or loosely curled, but finger details are not distinguishable.',
                    'raw_clause_index': 0, 'raw_event_id': 'e1', 'source_text_is_verbatim_substring': True}, **m.FLAGS}
        notice, result, _, _ = self.clause_candidate(fact=fact)
        self.assertNotEqual(fact['text'], fact['raw_text'])
        candidate = self.overlay.normalize(notice, 'clause_salvage', 'clause_revision')
        self.assertEqual(candidate['kind'], 'complete_candidate')
        self.assertEqual(candidate['label']['timed_motion_script'], raw)
        self.assertEqual(candidate['new_safe_facts'][0]['frame_indices'], [38, 40])

    def test_clause_changed_raw_text_rejected_even_with_matching_fact_copies(self):
        notice, result, _, _ = self.clause_candidate()
        result['new_safe_facts'][0]['raw_text'] = 'Invented motion.'
        result['visual_result_ref'] = notice['visual_result_ref'] = self.write(self.visual_path, self.clause_visual)
        notice['new_safe_fact_refs'][0]['visual_result_ref'] = result['visual_result_ref']
        self.assertEqual(self.clause_visual['facts'][0], result['new_safe_facts'][0])
        self.assertEqual(notice['new_safe_facts'], result['new_safe_facts'])
        with self.assertRaisesRegex(ValueError, 'safe fact/verbatim'):
            self.overlay.normalize(self.clause_notice(notice, result), 'clause_salvage', 'clause_revision')

    def test_clause_unreviewed_or_partial_cannot_replace_complete_or_clear_old_error(self):
        notice, result, base, original = self.clause_candidate(completed=False)
        self.stream([base])
        self.ledger(self.root / 'review/train-b0', self.audit(base, original, 'confirmed_error'))
        self.stream([notice], kind='clause_revision', origin='clause_salvage')
        status, rows = self.overlay.snapshot()
        self.assertEqual(rows[0]['selection']['joint_candidate_ref'], self.base_ref)
        self.assertTrue(rows[0]['pendinghuman'])
        self.assertEqual(status['counts']['confirmed_AI_error'], 1)
        self.assertFalse(rows[0]['alternatives'][1]['sampled_pass'])
        notice, result, _, _ = self.clause_candidate()
        self.stream([notice], kind='clause_revision', origin='clause_salvage')
        self.assertEqual(self.overlay.snapshot()[1][0]['selection']['joint_candidate_ref'], self.base_ref)
        self.assertTrue(self.overlay.snapshot()[1][0]['pendinghuman'])

    def test_clause_review_only_ledger_status_and_bound_input_can_grant_sampled_pass(self):
        notice, result, _, _ = self.clause_candidate()
        self.stream([notice], kind='clause_revision', origin='clause_salvage')
        directory, row = self.clause_ledger(self.root / 'clause-reviews', notice, result)
        self.assertEqual(self.overlay.snapshot()[0]['counts']['sampled_pass'], 1)
        row['input_row_sha256'] = 'tampered-input'
        ref = self.write(directory / 'reviews.jsonl', [row], lines=True)
        summary = json.loads((directory / 'summary.json').read_bytes())
        summary['review_records_ref'] = ref
        self.write(directory / 'summary.json', summary)
        audit_bid = directory.name
        self.overlay.review_batch(directory, 'train-b0', audit_bid, clause_edition=self.edition)
        self.assertEqual(self.overlay.snapshot()[0]['counts']['sampled_pass'], 0)
        self.assertEqual(self.overlay.stream_status[str(directory / 'summary.json')]['state'], 'pending_review_binding')
        summary.pop('review_records_ref')
        self.write(directory / 'summary.json', summary)
        self.overlay.review_batch(directory, 'train-b0', audit_bid, clause_edition=self.edition)
        self.assertEqual(self.overlay.snapshot()[0]['counts']['sampled_pass'], 0)

    def test_clause_default_roots_and_incremental_pending_with_old_cache(self):
        old = m.Overlay(self.summary, {'streams': {}, 'members': {}})
        self.assertEqual(old.state['review_batches'], {})
        args = m.defaults(m.parser().parse_args(['--root', str(self.root), '--manifest-summary', str(self.summary),
            '--state-cache', str(self.root / 'clause.cache.json')]))
        self.assertEqual(args.clause_salvage, self.root / 'curated222939_clause_salvage_v1')
        self.assertEqual(args.clause_reviews, self.root / 'curated_clause_salvage_reviews_v1')
        self.assertEqual(args.clause_pilot_reviews, self.root / 'curated_clause_salvage_reviews_pilot5_v1')
        status = m.run(args)
        self.assertEqual(status['stream_status'][str(args.clause_salvage / 'review_required.jsonl')]['state'], 'pending_missing')
        notice, result, _, _ = self.clause_candidate()
        self.write(args.clause_salvage / 'review_required.jsonl', [notice, notice], lines=True)
        self.clause_ledger(args.clause_reviews, notice, result)
        status = m.run(args)
        self.assertEqual(status['clause_salvage_review_scope']['source_count_with_joint_candidates'], 1)
        self.assertEqual(status['clause_salvage_review_scope']['sampled_pass'], 1)
        self.assertEqual(m.run(args)['counts'], status['counts'])
        self.assertFalse(status['all_batches_reviewed'])
        self.assertFalse(status['training_eligible'])

    @unittest.skipUnless((m.ROOT / 'curated_clause_salvage_reviews_pilot5_v1/run.summary.json').exists(),
                         'real pilot evidence unavailable')
    def test_clause_real_pilot_two_ledgers_read_only_smoke(self):
        overlay = m.Overlay(m.ROOT / 'curated222939_inputs_v1/manifest.summary.json')
        overlay.clause_salvage_root = m.ROOT / 'curated222939_clause_salvage_v1'
        source_rows = [json.loads(line) for line in (overlay.clause_salvage_root / 'review_required.jsonl').read_bytes().splitlines()[:5]]
        normalized = [overlay.normalize(row, 'clause_salvage', 'clause_revision') for row in source_rows]
        overlay.state['streams']['read-only-smoke'] = {'records': normalized}
        pilot = json.loads((m.ROOT / 'curated_clause_salvage_reviews_pilot5_v1/run.summary.json').read_bytes())
        for ledger in pilot['review_record_refs']:
            directory = Path(ledger['path']).parent
            summary = json.loads((directory / 'summary.json').read_bytes())
            self.assertEqual(summary['review_records_ref'], ledger)
            overlay.review_batch(directory, summary['original_batch_id'], directory.name,
                                 clause_edition='clause-salvage-20261025-v1')
        status, rows = overlay.snapshot()
        self.assertEqual([r['kind'] for r in normalized], ['partial_candidate', 'partial_candidate',
                         'complete_candidate', 'partial_candidate', 'complete_candidate'])
        self.assertEqual(status['counts']['sampled_pass'], 2)
        self.assertEqual(status['counts']['reviewed'], 2)
        self.assertEqual((status['expected_sources'], status['expected_batches']), (222939, 448))
        self.assertFalse(status['all_sources_joint_complete'])
        self.assertFalse(status['all_batches_joint_complete'])
        for row in rows:
            if row['state'] == 'partial_candidate':
                self.assertFalse(row['selection']['sampled_pass'])
        for flag in m.FLAGS:
            self.assertFalse(status[flag])

    def journal_fixture(self):
        module = self.overlay.journal_module()
        self.overlay.journal_production_root = self.root / 'production'
        self.overlay.journal_recovery_root = self.root / 'journal'
        badraw = b'broken journal interleaving\n'
        bad_sha = hashlib.sha256(badraw).hexdigest()
        rule = {'sha256': bad_sha, 'size': len(badraw), 'split': 'train',
                'batch_id': 'train-b0', 'ids': ['a', 'b']}
        rules = patch.object(module, 'KNOWN', {0: rule})
        rules.start()
        self.addCleanup(rules.stop)
        verifier = patch.object(module, 'Evidence')
        mock = verifier.start()
        self.addCleanup(verifier.stop)
        source = self.overlay.journal_production_root / 'completed.jsonl'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(badraw)
        bad_path = self.overlay.journal_recovery_root / 'badlines' / ('0-' + bad_sha + '.raw')
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_bytes(badraw)
        bad_ref = {'path': str(bad_path), 'sha256': bad_sha}
        records, actual = [], {}
        for index in range(2):
            candidate, _ = self.candidate(index)
            terminal = {**self.rows[index], 'joint_candidate_ref': candidate['joint_candidate_ref'],
                        'status': 'joint_candidate', 'joint_complete': True,
                        'visual_status': 'candidate', 'style_status': 'resolved', 'params_sha256': 'immutable-params'}
            terminal_path = self.overlay.journal_production_root / 'sources' / m.identity(terminal) / 'consumer.terminal.json'
            terminal_path.parent.mkdir(parents=True, exist_ok=True)
            terminal_path.write_bytes(module.q.encoded(terminal))
            terminal_ref = {'path': str(terminal_path), 'sha256': module.q.digest(terminal)}
            evidence = {'consumer_terminal_ref': terminal_ref,
                        'joint_result_ref': terminal['joint_candidate_ref'], 'params_sha256': terminal['params_sha256']}
            record = {'schema': module.SCHEMA, 'source_key': m.identity(terminal), 'hint': terminal,
                      'completed_row_sha256': module.q.digest(terminal), 'evidence': evidence,
                      'badline_ref': bad_ref, 'source_input_path': str(source),
                      'source_input_byte_begin': 0, 'source_input_byte_end': len(badraw),
                      'recovered_terminal_index': index, 'recovered_terminal_count': 2,
                      'stable_badline_snapshots': [{'identity': [1, 2, len(badraw), 4, 5], 'line_sha256': bad_sha}] * 3,
                      'recovery_policy': 'exact_offset_SHA_then_authoritative_manifest_and_immutable_terminal_chain', **m.FLAGS}
            records.append(record)
            actual[terminal['id']] = (terminal, json.loads(json.dumps(evidence)))
            self.write_journal_hint(record)
        mock.return_value.terminal.side_effect = lambda sid, split, batch: actual[sid]
        stream = self.overlay.journal_recovery_root / 'derived-hints.jsonl'
        self.write(stream, records, lines=True)
        return records, stream, source, mock.return_value.terminal

    def write_journal_hint(self, record):
        module = self.overlay.journal_module()
        path = self.overlay.journal_recovery_root / 'hints' / (record['source_key'] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(module.q.encoded(record))
        hint_ref = {'path': str(path), 'sha256': module.q.digest(record)}
        self.write(self.overlay.journal_recovery_root / 'announced' / (record['source_key'] + '.json'), {'hint_ref': hint_ref})
        return hint_ref

    def test_journal_two_recovered_count_dedup_preserve_invalid_and_incremental(self):
        records, stream, source, terminal_call = self.journal_fixture()
        with source.open('ab') as out:
            out.write((json.dumps(records[0]['hint']) + '\n').encode())
        self.overlay.stream(source, 'production', 'candidate')
        before = self.overlay.snapshot()[0]
        self.assertEqual(before['counts']['complete_candidate'], 1)
        self.overlay.stream(stream, 'journal_recovered', 'journal_recovery')
        status, rows = self.overlay.snapshot()
        self.assertEqual(status['counts']['complete_candidate'], 2)
        self.assertEqual(status['counts']['missing'], 1)
        self.assertEqual(len(status['invalid_records']), 1)
        scope = status['journal_recovery_scope']
        self.assertTrue(scope['badline_known'])
        self.assertTrue(scope['fully_bound_diagnostic'])
        self.assertEqual(scope['recovered_source_count'], 2)
        self.assertEqual(status['invalid_records'][0], before['invalid_records'][0])
        self.assertTrue(all(len(row['alternatives']) == 1 for row in rows))
        a, b = rows
        self.assertEqual(a['selection']['journal_recoveries'][0]['origin'], 'journal_recovered')
        self.assertEqual(b['selection']['origin'], 'journal_recovered')
        recovery = b['selection']['journal_recovery']
        self.assertEqual(recovery['original_ref']['sha256'], records[0]['badline_ref']['sha256'])
        self.assertEqual(recovery['receipt']['hint_ref'], recovery['hint_ref'])
        self.assertFalse(status['all_sources_joint_complete'])
        self.assertEqual(terminal_call.call_count, 2)
        self.overlay.stream(source, 'production', 'candidate')
        self.overlay.stream(stream, 'journal_recovered', 'journal_recovery')
        self.assertEqual(terminal_call.call_count, 2)
        self.assertEqual(self.overlay.stream_status[str(source)]['new_records'], 0)
        self.assertEqual(self.overlay.stream_status[str(stream)]['new_records'], 0)
        self.assertEqual(self.overlay.snapshot()[0]['counts'], status['counts'])

    def test_journal_changed_hint_terminal_or_joint_sha_rejected(self):
        records, stream, source, _ = self.journal_fixture()
        record = records[0]
        record['evidence']['consumer_terminal_ref']['sha256'] = 'not-original'
        self.write_journal_hint(record)
        self.write(stream, [record], lines=True)
        self.overlay.stream(source, 'production', 'candidate')
        self.overlay.stream(stream, 'journal_recovered', 'journal_recovery')
        status, _ = self.overlay.snapshot()
        self.assertEqual(status['counts']['complete_candidate'], 0)
        self.assertTrue(status['journal_recovery_scope']['pending_diagnostic'])
        self.assertIn('SHA mismatch', status['invalid_records'][1]['reason'])
        self.assertEqual(len(status['invalid_records']), 2)

    def test_journal_artifact_content_changes_rejected(self):
        for target in ('hint', 'receipt', 'terminal', 'joint', 'badline', 'original'):
            with self.subTest(target=target):
                self.overlay = m.Overlay(self.summary)
                records, stream, source, _ = self.journal_fixture()
                record = records[0]
                if target == 'hint':
                    path = self.overlay.journal_recovery_root / 'hints' / (record['source_key'] + '.json')
                elif target == 'receipt':
                    path = self.overlay.journal_recovery_root / 'announced' / (record['source_key'] + '.json')
                    self.write(path, {'hint_ref': {'path': 'wrong', 'sha256': 'changed'}})
                elif target == 'terminal':
                    path = Path(record['evidence']['consumer_terminal_ref']['path'])
                elif target == 'joint':
                    path = Path(record['evidence']['joint_result_ref']['path'])
                elif target == 'badline':
                    path = Path(record['badline_ref']['path'])
                else:
                    path = source
                if target != 'receipt':
                    path.write_bytes(path.read_bytes() + b' ')
                if target == 'original':
                    source.write_bytes(b'x' + source.read_bytes()[1:])
                with self.assertRaisesRegex(ValueError, 'SHA|badline|changed'):
                    self.overlay.journal_candidate(record)

    def test_journal_binding_change_and_incomplete_actual_joint_not_complete(self):
        records, stream, source, _ = self.journal_fixture()
        record = records[0]
        record['evidence']['params_sha256'] = 'other-params'
        self.write_journal_hint(record)
        with self.assertRaisesRegex(ValueError, 'params/binding'):
            self.overlay.journal_candidate(record)
        record['evidence']['params_sha256'] = 'immutable-params'
        ref = record['hint']['joint_candidate_ref']
        result = json.loads(Path(ref['path']).read_bytes())
        result['all_windows_nonempty'] = False
        new_ref = self.write(Path(ref['path']), result)
        record['hint']['joint_candidate_ref'] = record['evidence']['joint_result_ref'] = new_ref
        terminal_path = Path(record['evidence']['consumer_terminal_ref']['path'])
        module = self.overlay.journal_module()
        terminal_path.write_bytes(module.q.encoded(record['hint']))
        record['completed_row_sha256'] = record['evidence']['consumer_terminal_ref']['sha256'] = module.q.digest(record['hint'])
        self.overlay.journal_verifier.terminal.side_effect = lambda *a: (record['hint'], record['evidence'])
        self.write_journal_hint(record)
        with self.assertRaisesRegex(ValueError, 'fully bound complete'):
            self.overlay.journal_candidate(record)

    def test_journal_unknown_bad_pending_even_with_complete_population(self):
        notices = [self.candidate(i)[0] for i in range(3)]
        path = self.stream(notices)
        with path.open('ab') as out:
            out.write(b'unknown broken journal\n')
        self.overlay.stream(path, 'production', 'candidate')
        status, _ = self.overlay.snapshot()
        self.assertEqual(status['counts']['complete_candidate'], 3)
        self.assertFalse(status['all_sources_joint_complete'])
        self.assertFalse(status['all_batches_joint_complete'])
        self.assertTrue(status['journal_recovery_scope']['pending_diagnostic'])
        self.assertFalse(status['journal_recovery_scope']['badline_known'])
        self.assertEqual(status['journal_recovery_scope']['recovered_source_count'], 0)

    def test_journal_only_one_hint_not_fully_bound(self):
        records, stream, source, _ = self.journal_fixture()
        self.write(stream, records[:1], lines=True)
        self.overlay.stream(source, 'production', 'candidate')
        self.overlay.stream(stream, 'journal_recovered', 'journal_recovery')
        scope = self.overlay.snapshot()[0]['journal_recovery_scope']
        self.assertEqual(scope['recovered_source_count'], 1)
        self.assertTrue(scope['pending_diagnostic'])
        self.assertFalse(scope['fully_bound_diagnostic'])

    def test_v2_default_missing_is_pending_and_additional_cli_appends(self):
        args = m.defaults(m.parser().parse_args(['--root', str(self.root), '--manifest-summary', str(self.summary),
            '--state-cache', str(self.root / 'v2.cache.json')]))
        v2 = self.root / 'curated222939_collaborative_v2'
        self.assertIn(v2, args.additional_production)
        self.assertEqual(args.journal_recovery_root, self.root / 'curated222939_repair_v1/collector-recovery-v1')
        self.assertEqual(args.collaborative_v2_review_root, self.root / 'curated_independent_reviews_collaborative_v2')
        status = m.run(args)
        self.assertEqual(status['counts']['complete_candidate'], 0)
        self.assertEqual(status['counts']['missing'], 3)
        self.assertEqual(status['stream_status'][str(v2 / 'completed.jsonl')]['state'], 'pending_missing')
        explicit = m.defaults(m.parser().parse_args(['--root', str(self.root), '--additional-production',
            str(self.root / 'one'), '--additional-production', str(self.root / 'two')]))
        self.assertEqual(explicit.additional_production, [self.root / 'one', self.root / 'two'])

    def test_v2_different_out_same_source_single_denominator_and_old_error_preserved(self):
        old, original = self.candidate()
        new, result = self.candidate(name='v2')
        result['source_binding'] = {'params_sha256': 'v2-params', 'source_binding': {'hand_roi_root': 'independent-out'}}
        new['joint_candidate_ref'] = self.write(self.root / 'v2-result.json', result)
        self.stream([old])
        self.ledger(self.root / 'review/train-b0', self.audit(old, original, 'confirmed_error'))
        v2origin = 'additional:' + str(self.root / 'curated222939_collaborative_v2')
        self.stream([new, new], origin=v2origin)
        status, rows = self.overlay.snapshot()
        self.assertEqual(status['counts']['complete_candidate'], 1)
        self.assertEqual(status['counts']['missing'], 2)
        self.assertEqual(status['counts']['confirmed_AI_error'], 1)
        self.assertTrue(rows[0]['pendinghuman'])
        self.assertEqual(len(rows[0]['alternatives']), 2)
        self.assertEqual(rows[0]['selection']['joint_candidate_ref'], old['joint_candidate_ref'])
        other = next(r for r in rows[0]['alternatives'] if r['origin'] == v2origin)
        self.assertNotEqual(other['joint_candidate_ref'], rows[0]['selection']['joint_candidate_ref'])
        self.assertEqual(other['source_binding'], result['source_binding'])
        self.assertFalse(other['reviewed'])
        self.assertFalse(other['sampled_pass'])

    def test_v2_optional_reviews_only_fixed_active_source_members(self):
        args = m.defaults(m.parser().parse_args(['--root', str(self.root), '--manifest-summary', str(self.summary),
            '--state-cache', str(self.root / 'v2-reviews.cache.json'), '--collaborative-v2-review-root',
            str(self.root / 'future-v2-review')]))
        v2 = self.root / 'curated222939_collaborative_v2'
        notice, result = self.candidate(name='v2')
        self.write(v2 / 'completed.jsonl', [notice], lines=True)
        status = m.run(args)
        self.assertEqual(status['counts']['reviewed'], 0)
        self.assertEqual(status['stream_status'][str(args.collaborative_v2_review_root / 'train-b0/summary.json')]['state'],
                         'pending_missing_review')
        self.assertNotIn(str(args.collaborative_v2_review_root / 'val-b0/summary.json'), status['stream_status'])
        row = self.audit(notice, result)
        self.ledger(args.collaborative_v2_review_root / 'train-b0', row)
        status = m.run(args)
        self.assertEqual(status['counts']['sampled_pass'], 1)
        self.assertFalse(status['all_batches_reviewed'])
        self.assertFalse(status['training_approved'])

    def test_ready_is_not_joint_and_original_split_kept(self):
        self.stream([self.rows[0], self.rows[0], self.rows[2]], kind='ready', name='ready.jsonl')
        status, candidates = self.overlay.snapshot()
        self.assertEqual(status['counts']['ready_unique'], 2)
        self.assertEqual(status['counts']['ready_not_joint'], 2)
        self.assertEqual(status['counts']['missing'], 1)
        self.assertEqual(len(candidates), 2)
        self.assertFalse(status['all_sources_joint_complete'])

    def test_duplicate_failed_ready_hints_are_not_ready_or_extra_sources(self):
        failed = {**self.rows[0], 'status': 'failed'}
        self.stream([failed, failed], kind='ready', name='ready.jsonl')
        status, candidates = self.overlay.snapshot()
        self.assertEqual(status['counts']['failed'], 1)
        self.assertEqual(status['counts']['ready_unique'], 0)
        self.assertEqual(status['counts']['missing'], 2)
        self.assertEqual(len(candidates[0]['failed_attempts']), 1)

    def test_complete_reviewed_original_not_replaced_by_partial_or_unreviewed_repair(self):
        base, result = self.candidate()
        partial, _ = self.candidate(name='partial', completed=False)
        repair, _ = self.candidate(name='repair', revision=True)
        self.stream([base, partial])
        self.stream([repair], kind='joint_revision', origin='joint_repair')
        self.ledger(self.root / 'review/train-b0', self.audit(base, result))
        status, candidates = self.overlay.snapshot()
        self.assertEqual(candidates[0]['selection']['joint_candidate_ref'], self.base_ref)
        self.assertEqual(len(candidates[0]['alternatives']), 3)
        self.assertEqual(status['counts']['sampled_pass'], 1)
        self.assertFalse(status['semantic_approval'])

    def test_half_line_and_same_inode_truncate_regrow(self):
        path = self.root / 'ready.jsonl'
        row = (json.dumps(self.rows[0]) + '\n').encode()
        path.write_bytes(row[:20])
        self.overlay.stream(path, 'production', 'ready')
        self.assertEqual(self.overlay.snapshot()[0]['counts']['ready_unique'], 0)
        path.write_bytes(row)
        self.overlay.stream(path, 'production', 'ready')
        self.assertEqual(self.overlay.snapshot()[0]['counts']['ready_unique'], 1)
        path.write_text(json.dumps(self.rows[2]) + '\n' + ' ' * 500)
        self.overlay.stream(path, 'production', 'ready')
        self.assertTrue(self.overlay.stream_status[str(path)]['reset_after_truncation'])
        self.assertEqual(self.overlay.snapshot()[0]['split_counts']['val']['ready_not_joint'], 1)

    def test_transient_read_failure_is_revalidated_without_stream_change(self):
        notice, _ = self.candidate()
        path = self.root / "completed.jsonl"
        self.write(path, [notice], lines=True)
        with patch.object(self.overlay, "normalize", side_effect=OSError("temporary read failure")):
            self.overlay.stream(path, "production", "candidate")
        self.assertEqual(len(self.overlay.snapshot()[0]["invalid_records"]), 1)
        self.overlay.stream(path, "production", "candidate")
        status, _ = self.overlay.snapshot()
        self.assertEqual(status["counts"]["complete_candidate"], 1)
        self.assertEqual(status["invalid_records"], [])

    def test_missing_stream_retracts_cache_entries(self):
        path = self.stream([self.rows[0]], kind='ready')
        path.unlink()
        self.overlay.stream(path, 'production', 'ready')
        self.assertEqual(self.overlay.snapshot()[0]['counts']['missing'], 3)

    def test_tampered_label_or_csv_binding_rejected(self):
        notice, result = self.candidate()
        result['source_refs'] = {'csv': {'path': 'wrong'}}
        notice['joint_candidate_ref'] = self.write(self.root / 'wrong.json', result)
        self.stream([notice])
        self.assertEqual(self.overlay.snapshot()[0]['counts']['complete_candidate'], 0)
        self.assertIn('CSV/row', self.overlay.snapshot()[0]['invalid_records'][0]['reason'])
        notice['joint_candidate_ref']['sha256'] = 'bad'
        self.stream([notice])
        self.assertIn('SHA mismatch', self.overlay.snapshot()[0]['invalid_records'][0]['reason'])

    def test_ledger_error_keeps_pendinghuman_no_human_approval(self):
        notice, result = self.candidate()
        self.stream([notice])
        self.ledger(self.root / 'review/train-b0', self.audit(notice, result, 'confirmed_error'))
        status, candidates = self.overlay.snapshot()
        self.assertEqual(status['counts']['pendinghuman'], 1)
        self.assertEqual(status['counts']['confirmed_AI_error'], 1)
        self.assertTrue(candidates[0]['pendinghuman'])
        self.assertFalse(candidates[0]['human_review_completed'])

    def test_complete_population_does_not_grant_training_approval(self):
        rows = [self.candidate(i)[0] for i in range(3)]
        self.stream(rows)
        status, _ = self.overlay.snapshot()
        self.assertTrue(status['all_sources_joint_complete'])
        self.assertTrue(status['all_batches_joint_complete'])
        for flag in m.FLAGS:
            self.assertFalse(status[flag])

    def test_rebuildable_local_cache_and_only_active_batch_paths(self):
        notice, _ = self.candidate()
        self.write(self.root / 'curated222939_production_v1/completed.jsonl', [notice], lines=True)
        self.write(self.root / 'curated222939_production_v1/batches/train-b0/terminal_index.jsonl',
                   [{'id': 'a', 'split': 'train'}], lines=True)
        args = m.defaults(m.parser().parse_args(['--root', str(self.root), '--manifest-summary', str(self.summary), '--state-cache', str(self.root / 'local.cache.json')]))
        visited = []
        original = m.Overlay.stream
        def stream(obj, path, *a, **kw):
            visited.append(str(path))
            return original(obj, path, *a, **kw)
        with patch.object(m.Overlay, 'stream', stream):
            status = m.run(args)
        self.assertEqual(status['counts']['complete_candidate'], 1)
        self.assertEqual(status['counts']['terminal_unique'], 1)
        self.assertEqual(status['stream_status'][str(args.recovery_reviews)]['state'], 'conditional_pending_consumer_candidates')
        self.assertFalse(any('/val-b0/' in p for p in visited))
        self.assertFalse((args.out / 'state.sqlite3').exists())
        self.assertEqual(sorted(p.name for p in args.out.iterdir()), ['candidates.jsonl', 'progress.json'])
        first = status['counts']
        self.assertEqual(m.run(args)['counts'], first)
        args.state_cache = self.root / 'fresh.cache.json'
        self.assertEqual(m.run(args)['counts'], first)
        args.state_cache = Path('/apdcephfs/shared/cache.json')
        with self.assertRaisesRegex(ValueError, 'local'):
            m.run(args)

    def test_legacy_review_uses_frozen_fixed_path_and_digest(self):
        notice, result = self.candidate()
        self.stream([notice])
        row = {**result, 'label_path': notice['joint_candidate_ref']['path'],
               'label_sha256': m.digest(m.payload(result)), 'annotator_model_sha256': 'annotator'}
        directory = self.root / 'legacy/train-b0'
        self.write(directory / 'batch.plan.json', {'batch_id': 'train-b0', 'split': 'train', 'records': [row]})
        self.write(directory / 'summary.json', {'reviewed_source_ids': ['a']})
        edition = m.digest({'label': m.payload(row), 'label_path': row['label_path'],
            'input_row_sha256': m.digest(row), 'annotator_model_sha256': 'annotator'})[:16]
        fixed = directory / 'sources' / hashlib.sha256(b'a').hexdigest()[:24] / edition / 'review'
        final_ref = self.write(fixed / 'final.json', {'status': 'candidate', 'format_valid': True, 'json': {'status': 'pass'}})
        self.write(fixed / 'receipt.json', {'binding': {'source_id': 'a', 'split': 'train',
            'label_sha256': m.digest(m.payload(row)), 'label_file_ref': notice['joint_candidate_ref'],
            'input_row_sha256': m.digest(row), 'checkpoint_content_independence_verified': True},
            'files_sha256': {'final.json': final_ref['sha256']}})
        self.overlay.review_batch(directory, 'train-b0')
        self.assertEqual(self.overlay.snapshot()[0]['counts']['sampled_pass'], 1)

    def test_snapshot_repeated_announcements_stable_and_review_ledger_tamper_pending(self):
        notice, result = self.candidate()
        self.stream([notice, notice])
        row = self.audit(notice, result)
        directory = self.root / 'review/train-b0'
        self.ledger(directory, row)
        for _ in range(3):
            self.assertEqual(len(self.overlay.snapshot()[1][0]['alternatives']), 1)
        row['semantic_label_sha256'] = 'bad'
        self.ledger(directory, row)
        status, candidates = self.overlay.snapshot()
        self.assertEqual(status['counts']['reviewed'], 0)
        self.assertFalse(candidates[0]['human_review_completed'])
        self.assertEqual(self.overlay.stream_status[str(directory / 'summary.json')]['state'], 'pending_review_binding')

    def test_io_recovery_edition_and_refs_separate_from_global_revision(self):
        notice, result = self.candidate(name='io-recovery')
        failed_ref = self.write(self.root / 'old.json', {**self.rows[0], 'status': 'failed'})
        requests_ref = self.write(self.root / 'requests.json', {'requests': [
            {**self.rows[0], 'old_terminal_ref': failed_ref}]})
        result.update(edition='io-failure-recovery500-v1', old_failed_producer_terminal_ref=failed_ref,
                      recovery_revision_requests_ref=requests_ref)
        notice['joint_candidate_ref'] = self.write(self.root / 'io-result.json', result)
        self.stream([notice], origin='io_recovery')
        status, rows = self.overlay.snapshot()
        candidate = rows[0]['selection']
        self.assertEqual(candidate['edition'], 'io-failure-recovery500-v1')
        self.assertIsNone(candidate['global_revision'])
        self.assertEqual(candidate['old_failed_producer_terminal_ref'], failed_ref)
        self.assertEqual(candidate['recovery_revision_requests_ref'], requests_ref)
        self.assertFalse(candidate['training_approved'])
        result['recovery_revision_requests_ref'] = self.write(self.root / 'bad-requests.json', {'requests': []})
        notice['joint_candidate_ref'] = self.write(self.root / 'io-bad.json', result)
        self.stream([notice], origin='io_recovery')
        self.assertIn('request binding mismatch', self.overlay.snapshot()[0]['invalid_records'][0]['reason'])

    def test_recovery_source_audit_does_not_mark_original_batch_review_complete(self):
        notice, result = self.candidate(name='recovery-ledger')
        result['edition'] = 'io-failure-recovery500-v1'
        notice['joint_candidate_ref'] = self.write(self.root / 'recovery-result.json', result)
        args = m.defaults(m.parser().parse_args(['--root', str(self.root), '--manifest-summary', str(self.summary),
            '--state-cache', str(self.root / 'recovery-local.cache.json')]))
        self.write(args.io_recovery / 'completed.jsonl', [notice], lines=True)
        directory = args.recovery_reviews / result['edition'] / ('train-' + hashlib.sha256(b'a').hexdigest()[:24])
        audit_row = self.audit(notice, result)
        ledger_ref = self.write(directory / 'reviews.jsonl', [audit_row], lines=True)
        self.write(directory / 'summary.json', {'review_records_ref': ledger_ref,
            'original_batch_review_complete': False})
        status = m.run(args)
        self.assertEqual(status['counts']['sampled_pass'], 1)
        self.assertEqual(status['io_recovery_review_scope']['source_count_with_joint_candidates'], 1)
        self.assertFalse(status['io_recovery_review_scope']['original_batch_review_complete'])
        self.assertFalse(status['all_batches_reviewed'])
        self.assertFalse(status['training_approved'])
        self.assertFalse(status['human_review_completed'])

    def test_recovery_original_failed_ref_preserved_and_ready_not_joint(self):
        member = self.rows[0]
        old_ref = self.write(self.root / 'old_failed.json', {**member, 'status': 'failed'})
        mapping = self.root / 'mapping.json'
        request = {**member, 'old_terminal_ref': old_ref, 'revision_request': {'new_out': 'separate-out', 'computed_output_exists': False}}
        self.write(mapping, {'requests': [request]})
        self.overlay.recoveries(mapping)
        self.stream([{**member, 'status': 'ready'}], kind='ready', origin='io_recovery')
        status, candidates = self.overlay.snapshot()
        self.assertEqual(status['counts']['complete_candidate'], 0)
        self.assertEqual(status['counts']['failed'], 0)
        self.assertEqual(status['counts']['ready_not_joint'], 1)
        self.assertEqual(status['counts']['sources_with_failed_attempts'], 1)
        self.assertEqual(status['counts']['ready_unique'], 1)
        self.assertEqual(candidates[0]['failed_attempts'][0]['producer_terminal_ref'], old_ref)
        self.assertEqual(status['recovery_requests'][0]['overlay_status'], 'explicit_producer_alternative_pending_not_joint')


if __name__ == '__main__':
    unittest.main()
