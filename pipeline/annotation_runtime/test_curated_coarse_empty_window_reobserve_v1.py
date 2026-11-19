"""Real three-source CPU evidence tests; synthetic body events are NOT semantic proof."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PATH = Path(__file__).with_name('curated_coarse_empty_window_reobserve_v1.py')
spec = importlib.util.spec_from_file_location('empty_coarse_test_subject', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RealSourceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(m.bar, 'runtime_loader', side_effect=AssertionError('NO MODEL LOAD')), \
                patch.object(m.bar, 'transactions', side_effect=AssertionError('NO VLM CALL')):
            cls.receipt = m.prepare()
        cls.sources = {}
        for hint in m.q.load(m.OUT / 'jobs.manifest.json')['jobs']:
            job = m.q.checked(hint['job_ref'])
            cls.sources[job['id']] = (job, hint['job_ref'], *m.verify_job(job, probe=True))

    def synthetic_body(self, job, prepared):
        """Protocol-only fixture: refs are selected for tests, not a visual truth claim."""
        manifest = m.verify_window(job, prepared)
        refs = [f['index'] for f in manifest['frames'] if f['index'] not in prepared['forbidden'].get('torso', [])][:2]
        self.assertEqual(len(refs), 2)
        value = {'events': [{'id': 'e1', 'description': 'The torso turns left.', 'parts': ['torso'],
                             'frame_indices': refs}],
                 'part_support': [{'part': p, 'frame_indices': refs if p == 'torso' else [],
                                   'reason': 'synthetic fixture only, not semantic proof'} for p in m.bar.PARTS],
                 'uncertainties': ['Handshape, closedgrip and palm openness remain unknown metadata.'], 'abstain': False}
        return value, manifest

    def test_real_three_source_cpu_prepare_no_calls_no_budget(self):
        self.assertEqual(set(self.sources), set(m.TARGETS))
        self.assertEqual(self.receipt['source_count'], 3)
        for name in ('model_loads', 'visual_model_calls', 'text_model_calls', 'budget_reservations_performed'):
            self.assertEqual(self.receipt[name], 0)
        self.assertFalse(self.receipt['GPU_dispatch_performed'])
        self.assertFalse(self.receipt['GPU_resource_prepared'])
        self.assertFalse(self.receipt['launch_authorized'])
        for check in self.receipt['checks']:
            job = self.sources[check['id']][0]
            self.assertEqual(check['primary_budget_reserved_ref'], m.check_budget(job))
            self.assertEqual(check['sample_fps'], 8.0)

    def test_all_actual_pts_rgb_roi_bytes_and_budget_are_original(self):
        for sid, (job, ref, original, joint, prepared, plan) in self.sources.items():
            with self.subTest(sid=sid):
                request = m.make_request(job, ref, prepared, m.worker_args())
                native = m.q.load(next(Path(o['attempt_path']) / 'request.json' for o in original['window_observations']
                                      if o['window']['window_id'] == job['target_window_id']))
                self.assertEqual(request['model'], native['model'])
                self.assertEqual(request['sampling'], native['sampling'])
                self.assertEqual(request['sampling']['max_tokens'], 1800)
                # Header and strict prompt/schema differ; every body/ROI image and PTS text is native.
                self.assertEqual(request['messages'][1]['content'][1:], native['messages'][1]['content'][1:])
                images = [base64.b64decode(c['image_url']['url'].split(',', 1)[1], validate=True)
                          for c in request['messages'][1]['content'] if c['type'] == 'image_url']
                self.assertEqual(len(images), len(m.TARGETS[sid][3]) + m.TARGETS[sid][4])
                self.assertLessEqual(len(images), 16)
                check = next(c for c in self.receipt['checks'] if c['id'] == sid)
                self.assertEqual([{'sha256': m.q.hashlib.sha256(b).hexdigest(), 'bytes': len(b)} for b in images],
                                 check['image_bytes_and_sha256'])
                manifest = m.verify_window(job, prepared)
                self.assertEqual([f['index'] for f in manifest['frames']], m.TARGETS[sid][3])
                for f in manifest['frames']:
                    self.assertEqual(f['time'], prepared['probe']['times'][f['index']])
                    self.assertEqual(f['source_timestamp'], prepared['probe']['source_timestamps'][f['index']])
                schema = copy.deepcopy(native['json_schema'])
                parts = schema['properties']['events']['items']['properties']['parts']
                parts['items']['enum'], parts['maxItems'] = list(m.COARSE_PARTS), 6
                self.assertEqual(request['json_schema'], schema)
                header = json.loads(request['messages'][1]['content'][0]['text'])
                self.assertEqual(header['part_support_parts'], list(m.bar.PARTS))
                self.assertTrue(header['ROI_inventory_is_not_hand_visibility_frame_set'])

    def test_original_complete_job_reference_and_facts_style_are_frozen(self):
        for sid, (job, ref, original, joint, prepared, plan) in self.sources.items():
            canonical = m.q.checked(job['refs']['original_canonical_job'])
            self.assertEqual(job['original_result_sha256'], canonical['original_result_sha256'])
            self.assertEqual(job['source_binding'], canonical['source_binding'])
            self.assertEqual(job['original_source_window_diagnostics'], canonical['window_diagnostics'])
            self.assertEqual(m.q.checked(job['refs']['original_deferred_terminal'])['status'], 'deferred')
            self.assertEqual(original['facts'], m.bar.facts_from_observations(
                [v['window'] for v in prepared['windows']], original['window_observations']))
            self.assertEqual(joint['style_result'], m.q.checked(job['refs']['style_result']))
            self.assertEqual(job['maximum_reobserve_requests'], 1)
            self.assertEqual(job['maximum_rewrite_requests'], 0)

    def test_synthetic_body_unknown_hands_do_not_suppress_torso_not_semantic_proof(self):
        for job, ref, original, joint, prepared, plan in self.sources.values():
            value, manifest = self.synthetic_body(job, prepared)
            before = copy.deepcopy(value)
            filtered = m.validate_coarse(value, manifest, prepared['forbidden'])
            self.assertEqual(value, before)
            self.assertEqual(len(filtered['events']), 1)
            self.assertEqual(filtered['events'][0]['parts'], ['torso'])
            self.assertEqual(filtered['part_support'], before['part_support'])
            self.assertTrue(filtered['support_filter_audit']['filter_is_not_semantic_truth_review'])

    def test_pack_synthetic_target_only_and_verbatim_no_text_calls_not_semantic_proof(self):
        for job, ref, original, old_joint, prepared, plan in self.sources.values():
            value, manifest = self.synthetic_body(job, prepared)
            observed = m.validate_coarse(value, manifest, prepared['forbidden'])
            observations = copy.deepcopy(original['window_observations'])
            for i, old in enumerate(observations):
                if old['window']['window_id'] == job['target_window_id']:
                    observations[i] = {'status': 'candidate', 'json': observed, 'window': old['window']}
            visual, joint = m.package(job, observations, original, old_joint, prepared, plan)
            self.assertEqual(joint['style_result'], old_joint['style_result'])
            self.assertEqual(joint['dance_style_label'], old_joint['dance_style_label'])
            self.assertFalse(joint['training_approved'])
            self.assertFalse(joint['human_review_completed'])
            self.assertFalse(visual['final']['model_rewrite_performed'])
            target = next(v for v in visual['final']['json']['windows'] if v['window_id'] == job['target_window_id'])
            self.assertEqual(target['caption'], 'The torso turns left.')
            self.assertEqual(target['used_fact_ids'], [job['target_window_id'] + ':e1'])
            for old, new in zip(original['window_observations'], visual['window_observations']):
                if old['window']['window_id'] != job['target_window_id']:
                    self.assertEqual(new, old)
            for old, new in zip(original['final']['json']['windows'], visual['final']['json']['windows']):
                if old['window_id'] != job['target_window_id']:
                    self.assertEqual(new, old)

    def test_empty_or_failed_observation_stays_partial_no_stillness(self):
        for job, ref, original, old_joint, prepared, plan in self.sources.values():
            for status in ('abstain', 'failed'):
                observations = copy.deepcopy(original['window_observations'])
                if status == 'failed':
                    target = next(o for o in observations if o['window']['window_id'] == job['target_window_id'])
                    target.update(status='failed', json=None)
                visual, joint = m.package(job, observations, original, old_joint, prepared, plan)
                self.assertFalse(joint['joint_complete'])
                self.assertEqual(joint['target_text'], '')
                self.assertEqual(next(v for v in visual['final']['json']['windows']
                                      if v['window_id'] == job['target_window_id'])['caption'], '')
                self.assertEqual(joint['style_result'], old_joint['style_result'])

    def test_strict_bad_parts_frames_support_mask_and_hand_details(self):
        job, ref, original, joint, prepared, plan = next(iter(self.sources.values()))
        value, manifest = self.synthetic_body(job, prepared)
        bads = []
        for text in ('The torso turns with a closedgrip.', 'The torso moves with an open handshape.',
                     'The torso twists while palms open.', 'The torso turns with a V sign.',
                     'The torso turns with an OK sign.', 'The torso turns with a heart sign.',
                     'The torso turns with a fist.', 'The torso remains still.',
                     'The performer moves.', 'The torso turns; the torso bends.', 'The torso turns'):
            bad = copy.deepcopy(value)
            bad['events'][0]['description'] = text
            bads.append(bad)
        bad = copy.deepcopy(value); bad['events'][0]['parts'] = ['left_hand']; bads.append(bad)
        bad = copy.deepcopy(value); bad['events'][0]['frame_indices'] = [999, 1000]; bads.append(bad)
        bad = copy.deepcopy(value); bad['part_support'][1]['frame_indices'] = []; bads.append(bad)
        bad = copy.deepcopy(value); bad['part_support'] = bad['part_support'][:-1]; bads.append(bad)
        bad = copy.deepcopy(value); bad['part_support'].reverse(); bads.append(bad)
        bad = copy.deepcopy(value); bad['events'][0]['frame_indices'] *= 2; bads.append(bad)
        bad = copy.deepcopy(value); bad['events'][0]['parts'] = ['head', 'torso']; bads.append(bad)
        for bad in bads:
            with self.subTest(body=bad['events']):
                with self.assertRaises(ValueError):
                    m.validate_coarse(bad, manifest, prepared['forbidden'])
        forbidden = copy.deepcopy(prepared['forbidden'])
        forbidden['torso'] = value['events'][0]['frame_indices']
        with self.assertRaises(ValueError):
            m.validate_coarse(value, manifest, forbidden)

    def test_changed_manifest_frames_pts_mask_and_media_binding_rejected(self):
        job, ref, original, joint, prepared, plan = next(iter(self.sources.values()))
        for change in ('frames', 'mask', 'pts'):
            bad = copy.deepcopy(prepared)
            if change == 'frames':
                target = next(w for w in bad['windows'] if w['window']['window_id'] == job['target_window_id'])
                target['frames'][0]['index'] = 999
            elif change == 'mask':
                bad['raw2d_mask'] = {'forged': True}
            else:
                bad['probe']['times'][m.TARGETS[job['id']][3][0]] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.verify_window(job, bad)
        bad = copy.deepcopy(prepared); del bad['raw2d_mask']
        with self.assertRaises((ValueError, KeyError)):
            m.verify_window(job, bad)
        badjob = copy.deepcopy(job); badjob['source_binding']['video_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            m.verify_job(badjob)
        badjob = copy.deepcopy(job); badjob['target_manifest_ref']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            m.verify_job(badjob)

    def test_pack_rejects_forged_target_part_refs_after_filter(self):
        job, ref, original, joint, prepared, plan = next(iter(self.sources.values()))
        value, manifest = self.synthetic_body(job, prepared)
        observed = m.validate_coarse(value, manifest, prepared['forbidden'])
        observed['events'][0]['frame_indices'] = [999, 1000]
        observations = copy.deepcopy(original['window_observations'])
        target = next(o for o in observations if o['window']['window_id'] == job['target_window_id'])
        target.update(status='candidate', json=observed)
        with self.assertRaises(ValueError):
            m.package(job, observations, original, joint, prepared, plan)

    def test_missing_current_admission_callback_refuses_flags_only(self):
        job, ref, *_ = next(iter(self.sources.values()))
        class FlagsOnly:
            launch_authorized = True
            resource_manager_verified_and_leases_held = True
            def __call__(self):
                raise AssertionError('Must not load')
        with self.assertRaises(ValueError):
            m.caller_admission(FlagsOnly(), ref)
        class BadReceipt:
            def __call__(self):
                raise AssertionError('Must not load')
            def verify_current_admission(self, refs):
                return {'launch_authorized': True, 'fresh_pair_safety_verified': True}
        with self.assertRaises(ValueError):
            m.caller_admission(BadReceipt(), ref)

    def test_worker_test_dependencies_and_prepare_command_freeze(self):
        frozen = m.q.load(m.OUT / 'source.freeze.json')
        self.assertEqual(frozen, m.source_freeze())
        self.assertEqual(frozen['worker_ref'], m.q.identity(PATH))
        self.assertFalse(frozen['launch_authorized'])
        self.assertFalse(frozen['GPU_resource_prepared'])
        self.assertEqual(frozen['prepare_only_argv'][-1], '--prepare-only')

    def test_wrong_request_reference_sha_rejected(self):
        check = self.receipt['checks'][0]
        ref = {**check['request_preview_ref'], 'sha256': '0' * 64}
        with self.assertRaises(ValueError):
            m.q.checked(ref)

    def test_changed_untargeted_caption_facts_and_style_not_rewritten(self):
        job, ref, original, joint, prepared, plan = next(iter(self.sources.values()))
        bad = copy.deepcopy(original['window_observations'])
        old = next(o for o in bad if o['window']['window_id'] != job['target_window_id'])
        old['json']['uncertainties'].append('forged modification')
        with self.assertRaises(ValueError):
            m.package(job, bad, original, joint, prepared, plan)

    def test_changed_worker_model_sampling_budget_rejected(self):
        for attr, value in [('visual_max_tokens', 1801), ('sample_fps', 16.), ('tp', 1),
                            ('retry_failed', True), ('model', '/tmp/other-model')]:
            args = m.worker_args(); setattr(args, attr, value)
            with self.subTest(attr=attr), self.assertRaises(ValueError):
                m.verify_args(args)

    def test_unadmitted_callable_runtime_rejected_without_call_or_budget(self):
        job, ref, *_ = next(iter(self.sources.values()))
        calls = []
        before = m.check_budget(job)
        with self.assertRaises(ValueError):
            m.claim_once(job, ref, lambda: calls.append('MODEL MUST NOT LOAD'))
        self.assertEqual(calls, [])
        self.assertEqual(m.check_budget(job), before)


class FakeCanonicalBudgetOnly(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'fakecanon'
        self.root.mkdir()
        self.job = {'id': 'synthetic', 'split': 'train', 'source_key': 'train/synthetic', 'refs': {},
                    'window_diagnostics': [], 'original_result_sha256': 'a' * 64,
                    'queue_root': str(self.root), 'primary_budget_root': str(self.root)}
        result_path = Path(self.tmp.name) / 'synthetic_result.json'
        m.q.save_once(result_path, {'synthetic_fixture_only': True})
        self.job['original_result_ref'] = m.q.identity(result_path)
        self.refpath = Path(self.tmp.name) / 'synthetic_job.json'
        m.q.save_once(self.refpath, self.job)
        self.ref = m.q.identity(self.refpath)
        self.primary = patch.object(m, 'PRIMARY', self.root)
        self.primary.start()
        self.verify = patch.object(m, 'verify_job', return_value=None)
        self.verify.start()
        self.admission = patch.object(m, 'caller_admission', return_value={'synthetic': True})
        self.admission.start()

    def tearDown(self):
        self.admission.stop(); self.verify.stop(); self.primary.stop(); self.tmp.cleanup()

    def test_fakecanon_load_failure_does_not_reserve(self):
        def failed():
            raise RuntimeError('synthetic shared load failure')
        with self.assertRaises(RuntimeError):
            m.claim_once(self.job, self.ref, failed)
        self.assertFalse(m.budget_path(self.job).exists())

    def test_fakecanon_original_single_attempt_across_revisions(self):
        calls = []
        m.claim_once(self.job, self.ref, lambda: calls.append('synthetic runtime only'))
        self.assertEqual(calls, ['synthetic runtime only'])
        claim = m.q.load(m.budget_path(self.job))
        self.assertEqual(claim['source_key'], 'train/synthetic')
        self.assertEqual(claim['original_result_sha256'], 'a' * 64)
        self.assertEqual(claim['maximum_extra_source_attempts'], 1)
        with self.assertRaises(ValueError):
            m.claim_once(self.job, self.ref, lambda: calls.append('FORBIDDEN SECOND CALL'))
        with self.assertRaises(ValueError):
            m.q.reserve_attempt(self.job, self.root, 'other-edition')
        self.assertEqual(len(calls), 1)

    def test_fakecanon_wrong_budget_sha_and_count_rejected(self):
        for override in ({'original_result_sha256': 'b' * 64}, {'maximum_extra_source_attempts': 2},
                         {'source_key': 'train/other'}):
            with tempfile.TemporaryDirectory() as td, patch.object(m, 'PRIMARY', Path(td)):
                job = {**self.job, 'queue_root': td, 'primary_budget_root': td}
                claim = {'source_key': job['source_key'], 'original_result_sha256': 'a' * 64,
                         'extra_source_attempt': 1, 'maximum_extra_source_attempts': 1, **override}
                m.q.save_once(m.budget_path(job), claim)
                with self.assertRaises(ValueError):
                    m.check_budget(job)

    def test_fakecanon_cannot_borrow_legacy_or_unknown_root(self):
        for path in (self.root / 'legacy-demo', self.root / 'unknown-shared-scope'):
            bad = {**self.job, 'primary_budget_root': str(path)}
            with self.assertRaises(ValueError):
                m.budget_path(bad)
        with self.assertRaises(ValueError):
            m.q.reserve_attempt(self.job, self.root / 'other-edition', 'other')

    def test_fakecanon_load_then_reverify_failure_no_reservation(self):
        self.verify.stop()
        with patch.object(m, 'verify_job', side_effect=[None, ValueError('source changed after shared load')]):
            with self.assertRaises(ValueError):
                m.claim_once(self.job, self.ref, lambda: None)
        self.assertFalse(m.budget_path(self.job).exists())
        self.verify.start()

    def test_fakecanon_concurrent_claim_once_only_one_source_reservation(self):
        barrier = threading.Barrier(2)
        results = []
        def reserve():
            try:
                m.claim_once(self.job, self.ref, lambda: barrier.wait(timeout=10))
                results.append('reserved')
            except ValueError:
                results.append('denied')
        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=15)
        self.assertCountEqual(results, ['reserved', 'denied'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
