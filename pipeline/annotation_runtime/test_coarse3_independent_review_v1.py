"""CPU contract tests. No GPU load, signal, frozen-source or production writes."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import coarse3_independent_review_v1 as m


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = m.libraries()

    def label(self, complete=True):
        label = {'dance_style': 'Hip hop', 'timed_motion_script': '[0-2] The torso leans left.'}
        candidate = {'label': label, 'joint_complete': complete, 'all_windows_nonempty': complete,
                     'target_text': 'Dance style: Hip hop' + chr(10) + label['timed_motion_script'] if complete else ''}
        visual = {'final': {'json': {'windows': [{'caption': 'The torso leans left.'},
                                               {'caption': 'The head turns.' if complete else ''}]}}}
        return candidate, visual

    def test_real_libraries_are_isolated(self):
        before = dict(m.library_modules())
        self.assertIn('source_curated_consumer_v1', self.lib.worker.bar.__file__)
        with self.lib.review_library():
            self.assertIn('source_curated_review_v2', sys.modules['ms_bar_annotation'].__file__)
            self.assertIs(sys.modules['ms_batch_review'], self.lib.audit)
        self.assertEqual(before, m.library_modules())
        self.assertNotIn('vllm', sys.modules)

    def test_run_default_denied_before_any_library_or_output(self):
        with patch.object(m, 'libraries', side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError, 'verified caller admission'):
                m.execute('/tmp/not-created-coarse3', prepare_only=False, reviewer=object())

    def test_cli_run_denied(self):
        with patch.object(m, 'execute', side_effect=AssertionError('must not execute')):
            with self.assertRaises(SystemExit) as error:
                m.main(['--run'])
            self.assertEqual(error.exception.code, 2)

    def test_output_protected(self):
        for out in (m.RT, m.BAR, self.lib.worker.OUT, self.lib.worker.PRIMARY,
                    self.lib.recovery.OUT, m.BAR / 'coarse3_independent_reviews_v1'):
            with self.subTest(out=out), self.assertRaises(ValueError):
                m.check_output(out, self.lib)

    def test_only_complete_recovery_labels_are_reviewable(self):
        c, v = self.label(True)
        self.assertEqual(m.label_scope(self.lib.audit, c, v), 'complete')
        c, v = self.label(False)
        with self.assertRaisesRegex(ValueError, 'Only complete recovered'):
            m.label_scope(self.lib.audit, c, v)

    def test_unknown_empty_and_time_markers_not_pass(self):
        for style, motion in (('unknown', 'Moves.'), ('  ', 'Moves.'), ('Hip hop', ''), ('uncertain', 'Moves.')):
            c, v = self.label(False)
            c['label'].update(dance_style=style, timed_motion_script=motion)
            with self.subTest(style=style, motion=motion), self.assertRaises(ValueError):
                m.label_scope(self.lib.audit, c, v)
        c, v = self.label(False)
        v['final']['json']['windows'] = [{'caption': ''}]
        with self.assertRaisesRegex(ValueError, 'Time markers'):
            m.label_scope(self.lib.audit, c, v)

    def test_partial_cannot_claim_complete_target(self):
        c, v = self.label(False)
        c['target_text'] = 'forged training target'
        with self.assertRaisesRegex(ValueError, 'Only complete recovered'):
            m.label_scope(self.lib.audit, c, v)

    def test_three_pass_does_not_approve_original(self):
        rows = [{'source_id': sid, 'status': 'pass', 'label_scope': 'complete',
                 'original_membership': {'batch_id': 'original', 'original_member_count': 500}} for i, sid in enumerate(m.IDS)]
        summary = m.aggregate(rows, prepare_only=False, population_ref={'path': 'fixture', 'sha256': 'x'})
        self.assertEqual(summary['sampled_reviewed_denominator'], 3)
        self.assertEqual(summary['scope_denominators']['complete']['reviewed_denominator'], 3)
        self.assertNotIn('partial_nonempty', summary['scope_denominators'])
        self.assertTrue(summary['independent_sample_review_completed'])
        for flag in m.NO_APPROVAL:
            self.assertIs(summary[flag], False)
        self.assertFalse(summary['statistical_population_error_estimate'])

    def test_admission_rejects_stale_other_candidate_or_unverified(self):
        context = {'caller_pid': 10, 'candidate_ref': {'path': '/fixture', 'sha256': 'a'}}
        values = ({}, {'admission_verified': False, 'binding': context},
                  {'admission_verified': True, 'binding': {'caller_pid': 9}},
                  {'admission_verified': True, 'binding': context})
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.verified_admission(self.lib, lambda c: value, context)

    def test_verified_admission_requires_actual_reference_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            path.write_text('{}')
            ref = self.lib.q.identity(path)
            context = {'caller_pid': 10}
            good = {'admission_verified': True, 'binding': context,
                    'resource_admission_ref': ref, 'current_lease_refs': [ref]}
            self.assertEqual(m.verified_admission(self.lib, lambda c: good, context), good)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                m.verified_admission(self.lib, lambda c: good, context)

    def transaction(self, root, *, coarse=True, body=None):
        bar = self.lib.worker.bar
        manifest = {'window': {'window_id': 'w00000', 'start': 0., 'end': 2.},
                    'frames': [{'index': 0, 'time': 0.}, {'index': 1, 'time': 1.}]}
        raw = body or {'events': [{'id': 'e1', 'description': 'The torso leans left.', 'parts': ['torso'], 'frame_indices': [0, 1]}],
            'part_support': [{'part': p, 'frame_indices': [0, 1] if p == 'torso' else [], 'reason': 'Visible' if p == 'torso' else 'Unknown'}
                             for p in bar.PARTS], 'uncertainties': [], 'abstain': False}
        response = self.lib.worker.validate_coarse(raw, manifest, {}) if coarse else bar.filter_supported_observation(raw, manifest, {})
        request = {'model': '/fixture-annotator', 'sampling': {'max_tokens': 100}, 'json_schema': {'type': 'object'}}
        binding = {'source_id': 'fixture', 'window': manifest['window']}
        for name, value in (('request.json', request), ('binding.json', binding), ('response.json', response),
                            ('final.json', {'status': 'candidate', 'json': response, **bar.core.FLAGS})):
            (root / name).write_bytes(self.lib.q.encoded(value))
        (root / 'response.raw.txt').write_bytes(self.lib.q.encoded(raw))
        generation = {'request_sha256': self.lib.q.sha(root / 'request.json'), 'raw_response_sha256': self.lib.q.sha(root / 'response.raw.txt'),
                      'max_tokens': 100, 'model_call_performed': True, 'model_identity_sha256': 'model',
                      'structured_output_schema_sha256': bar.core.digest(request['json_schema']),
                      'finish_reason': 'stop', 'completion_tokens': 20}
        (root / 'generation.json').write_bytes(self.lib.q.encoded(generation))
        receipt = {'status': 'candidate', 'files_sha256': {n: self.lib.q.sha(root / n) for n in m.SIX},
                   'binding': binding, 'semantic_approval': False, 'human_review_completed': False}
        (root / 'receipt.json').write_bytes(self.lib.q.encoded(receipt))
        return request, binding, manifest

    def replay(self, root, args, coarse=True):
        request, binding, manifest = args
        return m.receipt_replay(self.lib, root, request, binding, 'model', manifest, {}, coarse=coarse)

    def test_actual_six_file_native_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final, refs = self.replay(root, self.transaction(root))
            self.assertEqual(final['status'], 'candidate')
            self.assertEqual(len(refs), 7)

    def test_raw_tamper_and_missing_receipt_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.transaction(root)
            (root / 'response.raw.txt').write_text('{}')
            with self.assertRaises(ValueError):
                self.replay(root, args)
            args = self.transaction(root)
            receipt = self.lib.q.load(root / 'receipt.json')
            receipt['files_sha256'].pop('response.json')
            (root / 'receipt.json').write_bytes(self.lib.q.encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'six-file'):
                self.replay(root, args)

    def test_request_generation_schema_and_filter_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for artifact, mutate in (
                ('request.json', lambda v: v['sampling'].update(max_tokens=101)),
                ('generation.json', lambda v: v.update(structured_output_schema_sha256='wrong')),
                ('response.json', lambda v: v['events'][0].update(description='The torso turns.'))):
                args = self.transaction(root)
                value = self.lib.q.load(root / artifact)
                mutate(value)
                (root / artifact).write_bytes(self.lib.q.encoded(value))
                receipt = self.lib.q.load(root / 'receipt.json')
                receipt['files_sha256'][artifact] = self.lib.q.sha(root / artifact)
                (root / 'receipt.json').write_bytes(self.lib.q.encoded(receipt))
                with self.subTest(artifact=artifact), self.assertRaises(ValueError):
                    self.replay(root, args)

    def test_original_filter_keeps_valid_clauses_despite_rejected_raw_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bar = self.lib.worker.bar
            raw = {'events': [{'id': 'e1', 'description': 'The torso leans left.', 'parts': ['torso'], 'frame_indices': [0, 1]},
                              {'id': 'e2', 'description': 'The left hand makes a fist.', 'parts': ['left_hand'], 'frame_indices': [0, 1]}],
                   'part_support': [{'part': p, 'frame_indices': [0, 1] if p == 'torso' else [], 'reason': 'Checked'} for p in bar.PARTS],
                   'uncertainties': [], 'abstain': False}
            args = self.transaction(root, coarse=False, body=raw)
            final, _ = self.replay(root, args, coarse=False)
            self.assertEqual(len(final['json']['events']), 1)

    def failed_transaction(self, root, valid_raw=False):
        args = self.transaction(root)
        bar = self.lib.worker.bar
        raw = self.lib.q.load(root / 'response.raw.txt')
        if not valid_raw:
            raw['events'][0].update(description='The dancer lifts both arms.', parts=['right_arm', 'left_arm'])
            for support in raw['part_support']:
                support['frame_indices'] = [0, 1] if support['part'] in ('left_arm', 'right_arm') else []
        try:
            self.lib.worker.validate_coarse(raw, args[2], {})
        except ValueError as error:
            message = type(error).__name__ + ': ' + str(error)
        else:
            message = 'ValueError: deliberately wrong fixture failure'
        (root / 'response.raw.txt').write_bytes(self.lib.q.encoded(raw))
        generation = self.lib.q.load(root / 'generation.json')
        generation['raw_response_sha256'] = self.lib.q.sha(root / 'response.raw.txt')
        (root / 'generation.json').write_bytes(self.lib.q.encoded(generation))
        (root / 'response.json').unlink()
        (root / 'final.json').write_bytes(self.lib.q.encoded({'status': 'failed', 'error': message, 'format_valid': False, **bar.core.FLAGS}))
        receipt = self.lib.q.load(root / 'receipt.json')
        receipt.update(status='failed', error=message, files_sha256={n:self.lib.q.sha(root / n) for n in m.SIX - {'response.json'}})
        (root / 'receipt.json').write_bytes(self.lib.q.encoded(receipt))
        return args

    def test_failed_source_replays_exact_error_without_fabricating_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, binding, manifest = self.failed_transaction(root)
            final, refs = m.receipt_replay(self.lib, root, request, binding, 'model', manifest, {}, coarse=True, allow_failed=True)
            self.assertEqual(final['status'], 'failed')
            self.assertEqual(len(refs), 6)
            self.assertFalse((root / 'response.json').exists())
            receipt = self.lib.q.load(root / 'receipt.json')
            receipt['error'] = 'different failure'
            (root / 'receipt.json').write_bytes(self.lib.q.encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'exact preserved native error'):
                m.receipt_replay(self.lib, root, request, binding, 'model', manifest, {}, coarse=True, allow_failed=True)

    def test_failed_receipt_that_now_validates_never_upgrades(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, binding, manifest = self.failed_transaction(root, valid_raw=True)
            with self.assertRaisesRegex(ValueError, 'unexpectedly validates'):
                m.receipt_replay(self.lib, root, request, binding, 'model', manifest, {}, coarse=True, allow_failed=True)

    def test_semantic_request_has_real_rgb_same_frame_roi_no_prior(self):
        with tempfile.TemporaryDirectory() as directory, self.lib.review_library():
            root = Path(directory)
            rgb, crop = root / 'rgb.jpg', root / 'roi.jpg'
            rgb.write_bytes(b'actual-rgb-fixture')
            crop.write_bytes(b'actual-native-roi-fixture')
            row = {'id': 'fixture', 'label': {'dance_style': 'Hip hop', 'timed_motion_script': 'The left hand closes.'},
                   'old_errors': 'DO_NOT_USE_PRIOR', 'original_caption': 'DO_NOT_USE_PRIOR', 'score': 1}
            evidence = {'frames': [{'index': 2, 'time': .2, 'source_timestamp': 1.2, 'file': 'rgb.jpg'},
                                   {'index': 3, 'time': .3, 'source_timestamp': 1.3, 'file': 'rgb.jpg'}],
                        'hand_rois': [{'frame_index': 2, 'part': 'left_hand', 'crop_xyxy': [0, 0, 4, 4],
                                       'crop_pixel_size': [4, 4], 'source_frame_size': [10, 10], 'path': str(crop),
                                       'sha256': self.lib.q.sha(crop)}]}
            request = self.lib.audit.review_request(row, evidence, root, '/model-fixture')
            text = json.dumps(request)
            self.assertNotIn('DO_NOT_USE_PRIOR', text)
            content = request['messages'][1]['content']
            payloads = [base64.b64decode(c['image_url']['url'].split(',')[1]) for c in content if c['type'] == 'image_url']
            self.assertEqual(payloads, [rgb.read_bytes(), crop.read_bytes(), rgb.read_bytes()])
            self.assertEqual(json.loads(content[0]['text'])['label_to_test'], row['label'])

    def test_missing_candidate_or_terminal_explicit_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            q = SimpleNamespace(checked=lambda ref: {'id': m.IDS[0], 'source_key': 'train/' + m.IDS[0]})
            worker = SimpleNamespace(OUT=Path(directory), REVISION='fixture')
            recovery = SimpleNamespace(OUT=Path(directory) / 'recovery', REVISION='fixture')
            lib = SimpleNamespace(q=q, worker=worker, recovery=recovery)
            with patch.object(m, 'verify_recovery_freeze'):
                candidate, value = m.validate_candidate(lib, {}, {'job_ref': {}, 'source_key': 'train/' + m.IDS[0]}, ffprobe='never-called')
            self.assertIsNone(candidate)
            self.assertEqual(value['status'], 'pending')
            self.assertEqual(value['reason'], 'recovery_candidate_or_original_terminal_not_landed')

    def test_original_membership_requires_full_denominator_and_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / 'train.csv'
            csv_path.write_text('id,value' + chr(10) + 'fixture,original' + chr(10))
            csv_ref = {**self.lib.q.identity(csv_path), 'csv_line': 2,
                       'raw_row_sha256': self.lib.worker.pipeline.digest({'header':['id','value'], 'values':['fixture','original']})}
            row = {'id': 'fixture', 'split': 'train', 'batch_id': 'original-batch', 'source_refs': {'csv':csv_ref}}
            path = root / 'members.jsonl'
            members = [row] + [{'id':'other' + str(i), 'split':'train', 'batch_id':'original-batch'} for i in range(499)]
            path.write_bytes(b''.join(self.lib.q.encoded(r) for r in members))
            q = SimpleNamespace(verify_bytes=self.lib.q.verify_bytes, digest=self.lib.q.digest,
                                verify_original_csv=lambda ref, split:self.lib.q.verify_bytes(ref))
            lib = SimpleNamespace(q=q, worker=self.lib.worker)
            plan = {'source_refs':{'manifest':self.lib.q.identity(path), 'original_csv':csv_ref}, 'manifest_record':row}
            result = m.verify_membership(lib, row, plan)
            self.assertEqual(result['original_member_count'], 500)
            bad = copy.deepcopy(plan)
            bad['source_refs']['original_csv'] = {**bad['source_refs']['original_csv'], 'raw_row_sha256':'forged'}
            with self.assertRaisesRegex(ValueError, 'CSV row digest'):
                m.verify_membership(lib, row, bad)
            path.write_bytes(b''.join(self.lib.q.encoded(r) for r in members[:3]))
            plan['source_refs']['manifest'] = self.lib.q.identity(path)
            with self.assertRaisesRegex(ValueError, '500/448'):
                m.verify_membership(lib, row, plan)

    def test_new_namespace_and_master_recovery_freeze(self):
        self.assertEqual(m.OUT.name, 'coarse3_filter_recovery_independent_reviews_v1')
        self.assertEqual(m.SCHEMA, 'coarse3_filter_recovery_independent_review_v1')
        m.verify_recovery_freeze(self.lib)
        self.assertEqual(self.lib.c.sha(self.lib.recovery.__file__), m.RECOVERY_REFS[0]['sha256'])
        self.assertNotIn('vllm', sys.modules)

    def recovery_fixture(self, root):
        q = self.lib.q
        recovery = SimpleNamespace(OUT=root, REVISION='recovery-fixture', SCHEMA='recovery-fixture')
        lib = SimpleNamespace(q=q, recovery=recovery)
        hint = {'source_key':'train/' + m.IDS[0], 'job_ref':{'path':'frozen-original-job', 'sha256':'sha'}}
        proof = {'schema':recovery.SCHEMA, 'global_revision':recovery.REVISION, 'source_key':hint['source_key'],
                 'original_job_ref':hint['job_ref'], 'recovery_code_ref':m.RECOVERY_REFS[0],
                 'original_observation_model_calls':1, 'new_visual_requests':0, 'new_text_model_requests':0,
                 'extra_source_attempts':0, 'canonical_budget_unchanged':True, 'new_label_review_required':True, **q.FLAGS}
        visual = {'status':'candidate', 'facts':[{'fact_id':'e1_c0'}], **proof}
        joint = {'status':'filter_recovery_joint_candidate', 'joint_complete':True,
                 'all_windows_nonempty':True, 'label':{'dance_style':'Hip hop', 'timed_motion_script':'Moves.'},
                 'training_eligible':False, **proof}
        dest = root / 'revisions' / recovery.REVISION / hint['source_key']
        dest.mkdir(parents=True)
        (dest / 'visual_result.json').write_bytes(q.encoded(visual))
        visual_ref = q.identity(dest / 'visual_result.json')
        (dest / 'result.json').write_bytes(q.encoded({**joint,'visual_result_ref':visual_ref}))
        (dest / 'derivation.receipt.json').write_bytes(q.encoded(proof))
        marker = root / 'review-required' / (hint['source_key'] + '.json')
        marker.parent.mkdir(parents=True)
        marker.write_bytes(q.encoded({'source_key':hint['source_key'], 'status':joint['status'], 'joint_complete':True,
                                     'result_ref':q.identity(dest / 'result.json'), 'new_model_calls':0,
                                     'new_label_review_required':True, **q.FLAGS}))
        return lib, hint, visual, joint, proof

    def test_recovery_full_dictionary_replay_binds_actual_visual_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.recovery_fixture(Path(directory))
            candidate, result_ref, visual_ref, proof_ref, marker_ref = m.recovery_bundle(*args)
            self.assertEqual(candidate['visual_result_ref'], visual_ref)
            self.assertEqual(candidate['original_observation_model_calls'], 1)
            self.assertEqual(candidate['new_visual_requests'], 0)
            self.assertNotIn('visual_result_ref', args[3])
            lib, hint, visual, joint, proof = args
            for altered in ({**joint,'joint_complete':False}, {**joint,'target_text':'forged'}):
                with self.assertRaisesRegex(ValueError, 'Full frozen recovery'):
                    m.recovery_bundle(lib, hint, visual, altered, proof)

    def test_recovery_code_and_original_call_identity_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            lib, hint, visual, joint, proof = self.recovery_fixture(Path(directory))
            for key, value in (('original_observation_model_calls',0), ('new_visual_requests',1),
                               ('recovery_code_ref',{'path':m.RECOVERY_REFS[0]['path'],'sha256':'forged'})):
                with self.subTest(key=key):
                    bad_proof = {**proof,key:value}
                    bad_visual, bad_joint = {**visual,key:value}, {**joint,key:value}
                    dest = lib.recovery.OUT / 'revisions' / lib.recovery.REVISION / hint['source_key']
                    (dest / 'visual_result.json').write_bytes(lib.q.encoded(bad_visual))
                    (dest / 'derivation.receipt.json').write_bytes(lib.q.encoded(bad_proof))
                    (dest / 'result.json').write_bytes(lib.q.encoded({**bad_joint,'visual_result_ref':lib.q.identity(dest / 'visual_result.json')}))
                    with self.assertRaisesRegex(ValueError, 'Frozen recovery code/source'):
                        m.recovery_bundle(lib,hint,bad_visual,bad_joint,bad_proof)

    def test_generation_uses_native_batch_digest_not_queue_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request, binding, manifest = self.transaction(root)
            schema_hash = self.lib.worker.bar.batch.digest(request['json_schema'])
            self.assertNotEqual(schema_hash, self.lib.q.digest(request['json_schema']))
            generation = self.lib.q.load(root / 'generation.json')
            self.assertEqual(generation['structured_output_schema_sha256'], schema_hash)
            self.assertEqual(self.lib.worker.bar.core.validate_generation(root,request), generation)

    def test_partial_transaction_never_redispatched(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / 'source'
            (root / 'review').mkdir(parents=True)
            (root / 'review/request.json').write_text('{}')
            reviewer = unittest.mock.Mock()
            with self.assertRaisesRegex(ValueError, 'never erase or redispatch'):
                m.review_transaction(self.lib, out, {'id': 'fixture', 'split': 'train'}, root, {}, reviewer,
                                     lambda context: self.fail('must not call admission'), {})
            reviewer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
