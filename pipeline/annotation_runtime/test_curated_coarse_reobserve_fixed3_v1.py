"""Three CPU-only groups: native pixels/raw, exact packages, source-budget lifecycle."""
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import curated_coarse_reobserve_fixed3_v1 as w


def body_raw(manifest, forbidden=None, extra_part=True):
    frames = [frame['index'] for frame in manifest['frames']]
    forbidden = forbidden or {}
    support = [{'part': part, 'frame_indices': [i for i in frames if i not in forbidden.get(part, [])]
                if part in w.COARSE_PARTS else [], 'reason': 'Original RGB visible body; fine detail unknown.'}
               for part in w.bar.PARTS]
    refs = [i for i in frames if all(i not in forbidden.get(p, []) for p in ('left_arm', 'right_arm', 'torso'))][:2]
    return {'events': [{'id': 'e1', 'description': 'The dancer raises both arms outward.',
        'parts': ['left_arm', 'right_arm'] + (['torso'] if extra_part else []), 'frame_indices': refs}],
        'part_support': support, 'uncertainties': ['Fine hand detail unknown.'], 'abstain': False}


class NativePixelsAndRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = []
        for sid in w.TARGETS:
            ref, original_job = w.original_source(sid)
            w.q.verify_job(original_job)
            job = w.build_job(sid, {'path': str(w.OUT / 'protocol.json'), 'sha256': 'fixture-not-written'})
            prepared = w.q.checked(job['refs']['prepared'])
            original = w.q.checked(job['refs']['visual_result'])
            joint = w.q.checked(job['refs']['joint_result'])
            plan = w.q.checked(job['refs']['plan_ref'])
            manifest = w.verify_window(job, prepared)
            cls.records.append((job, original, joint, prepared, plan, manifest))

    def test_native_three_image_order_bytes_schema_sampling_and_stages(self):
        for job, original, joint, prepared, plan, manifest in self.records:
            with self.subTest(id=job['id']):
                args = w.worker_args()
                request = w.make_request(job, {'path': '/fixture/job.json', 'sha256': 'x'}, prepared, args)
                native = w.bar.visual_request(args, Path(job['visual_root']) / 'rgb', manifest,
                                              {p: set(v) for p, v in prepared['forbidden'].items()})
                original_request = w.q.load(Path(next(o for o in original['window_observations']
                    if o['window']['window_id'] == job['target_window_id'])['attempt_path']) / 'request.json')
                self.assertEqual(native, original_request)
                self.assertEqual(request['messages'][1]['content'][1:], native['messages'][1]['content'][1:])
                self.assertEqual(request['sampling'], native['sampling'])
                self.assertEqual(sum(item['type'] == 'image_url' for item in request['messages'][1]['content']),
                                 len(w.TARGETS[job['id']][3]) + w.TARGETS[job['id']][4])
                header = json.loads(request['messages'][1]['content'][0]['text'])
                self.assertNotIn('hand_ROI_support', header)
                self.assertEqual(header['available_crop_inventory_not_visibility_or_handtruth'],
                                 json.loads(native['messages'][1]['content'][0]['text'])['hand_ROI_support'])
                self.assertEqual(header['part_support_parts'], list(w.bar.PARTS))
                self.assertEqual(header['allowed_event_parts'], list(w.COARSE_PARTS))
                schema = copy.deepcopy(native['json_schema'])
                parts = schema['properties']['events']['items']['properties']['parts']
                parts['items']['enum'], parts['maxItems'] = list(w.COARSE_PARTS), 6
                self.assertEqual(request['json_schema'], schema)
                self.assertIn('EXPLICIT OVERRIDE', request['messages'][0]['content'])
                self.assertEqual(job['stages'], w.original_source(job['id'])[1]['stages'])
                self.assertEqual(job['global_revision'], w.ORIGINAL_REVISION)
                self.assertEqual(job['execution_plan']['text_model_requests'], 0)
                self.assertIsNone(w.check_budget(job))

    def test_original_full_validation_then_subset_not_equal_with_exact_raw_id(self):
        for job, original, joint, prepared, plan, manifest in self.records:
            with self.subTest(id=job['id']):
                raw = body_raw(manifest, prepared['forbidden'])
                original_raw = copy.deepcopy(raw)
                filtered = w.validate_coarse(raw, manifest, prepared['forbidden'])
                self.assertEqual(raw, original_raw)
                self.assertEqual(filtered['events'][0]['id'], 'e1_c0')
                self.assertEqual(filtered['events'][0]['parts'], ['left_arm', 'right_arm'])
                self.assertEqual(filtered['events'][0]['frame_indices'], raw['events'][0]['frame_indices'])
                self.assertEqual(filtered['events'][0]['description'], raw['events'][0]['description'])
                self.assertEqual(filtered['events'][0]['source_clause']['raw_event_id'], 'e1')
                with self.assertRaisesRegex(ValueError, 'observation JSON keys|event keys|event ID'):
                    w.validate_coarse(filtered, manifest, prepared['forbidden'])

    def test_fine_static_visibility_whole_sentence_forbidden_and_added_refs_rejected(self):
        manifest = self.records[0][-1]
        forbidden = self.records[0][3]['forbidden']
        raw = body_raw(manifest, forbidden)
        bad_texts = ['The dancer raises both arms with fingers forming a V sign.',
                     'The dancer keeps both arms stationary.', 'Both arms are visible.',
                     'The dancer raises both arms. The dancer lowers both arms.',
                     'The dancer raises both arms; then lowers both arms.']
        for text in bad_texts:
            bad = copy.deepcopy(raw)
            bad['events'][0]['description'] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                w.validate_coarse(bad, manifest, forbidden)
        bad = copy.deepcopy(raw)
        bad['events'][0]['frame_indices'] = [99998, 99999]
        with self.assertRaises(ValueError):
            w.validate_coarse(bad, manifest, forbidden)
        bad = copy.deepcopy(raw)
        bad['part_support'][2]['frame_indices'] = []
        with self.assertRaises(ValueError):
            w.validate_coarse(bad, manifest, forbidden)
        for change in ('part', 'refs', 'text', 'raw_id'):
            normalized = w.bar.filter_supported_observation(raw, manifest, forbidden)
            if change == 'part':
                normalized['events'][0]['parts'].append('head')
            elif change == 'refs':
                normalized['events'][0]['frame_indices'].reverse()
            elif change == 'text':
                normalized['events'][0]['description'] += ' Invented.'
            else:
                normalized['events'][0]['source_clause']['raw_clause_index'] = 1
            with patch.object(w.bar, 'filter_supported_observation', return_value=normalized):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    w.validate_coarse(raw, manifest, forbidden)
        bad = copy.deepcopy(raw)
        with self.assertRaises(ValueError):
            w.validate_coarse(bad, manifest, {'left_arm': bad['events'][0]['frame_indices']})


    def test_real_admission_contract_fixed3_original_refs_live_two_lease_refs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(w, 'OUT', Path(temp) / 'scope'), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}):
            refs = []
            for job, *_ in self.records:
                path = w.OUT / 'jobs' / (job['source_key'] + '.json')
                w.q.save_once(path, job)
                refs.append(w.q.identity(path))
            w.q.save_once(w.OUT / 'jobs.manifest.json', {'fixture': True})
            resource = w.OUT / 'cpu_admission.json'
            w.q.save_once(resource, {'CPU_fixture_not_live_resource': True})
            leases = []
            for i in (0, 1):
                lease = w.OUT / ('cpu_lease' + str(i) + '.json')
                w.q.save_once(lease, {'CPU_fixture_not_actual_reserve': i})
                leases.append(w.q.identity(lease))
            admission = {'jobs_manifest_ref': w.q.identity(w.OUT / 'jobs.manifest.json'), 'source_job_refs': refs,
                'original_source_job_refs': [w.q.checked(ref)['refs']['original_canonical_job'] for ref in refs],
                'out': str(w.OUT), 'global_revision': w.REVISION, 'caller_pid': os.getpid(),
                'tensor_parallel_size': 2, 'gpus': [0, 1], 'resource_admission_ref': w.q.identity(resource),
                'current_lease_refs': leases}
            def runtime():
                raise AssertionError('Admission must never load models')
            runtime.verify_current_admission = lambda supplied: copy.deepcopy(admission)
            self.assertEqual(w.caller_admission(runtime, refs), admission)
            for field, value in [('gpus', [1, 0]), ('original_source_job_refs', []), ('caller_pid', -1),
                                 ('source_job_refs', refs[:2]), ('current_lease_refs', [leases[0], leases[0]])]:
                altered = {**admission, field: value}
                runtime.verify_current_admission = lambda supplied, a=altered: copy.deepcopy(a)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    w.caller_admission(runtime, refs)

    def test_prepare_isolated_out_zero_claim_load_and_budget_SHA_unchanged(self):
        original_verify = w.verify_job
        before = {sid: w.check_budget({**w.original_source(sid)[1], 'primary_budget_root': str(w.PRIMARY)}) for sid in w.TARGETS}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.object(w, 'OUT', Path(temp) / 'prepared'))
            stack.enter_context(patch.object(w, 'verify_job', side_effect=lambda job, probe=False: original_verify(job, probe=False)))
            reserve = stack.enter_context(patch.object(w.q, 'reserve_attempt', side_effect=AssertionError('CPU must never reserve')))
            loader = stack.enter_context(patch.object(w.bar, 'runtime_loader', side_effect=AssertionError('CPU must never construct model runtime')))
            receipt = w.prepare()
            self.assertEqual(receipt['source_count'], 3)
            self.assertEqual(receipt['budget_reservations_performed'], 0)
            self.assertEqual(receipt['model_loads'], 0)
            self.assertEqual(receipt['visual_model_calls'], 0)
            self.assertEqual(receipt['text_model_calls'], 0)
            self.assertEqual(receipt['budget_before'], receipt['budget_after'])
            self.assertEqual(receipt['budget_before'], before)
            self.assertFalse((w.OUT / 'ready.jsonl').exists())
            self.assertFalse((w.OUT / 'state.json').exists())
            self.assertFalse((w.OUT / 'attempt-budget').exists())
            self.assertEqual(w.q.load(w.OUT / 'source.freeze.json'), w.source_freeze())
            self.assertEqual([c['actual_image_count'] for c in receipt['checks']], [13, 10, 14])
            reserve.assert_not_called()
            loader.assert_not_called()
        after = {sid: w.check_budget({**w.original_source(sid)[1], 'primary_budget_root': str(w.PRIMARY)}) for sid in w.TARGETS}
        self.assertEqual(before, after)


class ExactPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NativePixelsAndRawTests.setUpClass()
        cls.records = NativePixelsAndRawTests.records

    def test_three_originals_non_target_facts_captions_boundaries_style_and_neutral_fallback(self):
        for job, original, joint, prepared, plan, manifest in self.records:
            with self.subTest(id=job['id']):
                raw = body_raw(manifest, prepared['forbidden'])
                filtered = w.validate_coarse(raw, manifest, prepared['forbidden'])
                observations = copy.deepcopy(original['window_observations'])
                target = next(i for i, o in enumerate(observations) if o['window']['window_id'] == job['target_window_id'])
                observations[target] = {'window': observations[target]['window'], 'status': 'candidate', 'json': filtered}
                visual, candidate = w.package(job, observations, original, joint, prepared, plan, raw)
                self.assertEqual(candidate['style_result'], joint['style_result'])
                self.assertEqual(candidate['source_refs'], joint['source_refs'])
                self.assertEqual(candidate['source_binding'], joint['source_binding'])
                self.assertEqual(visual['original_final'], original['final'])
                self.assertEqual(visual['boundary_observations'], original['boundary_observations'])
                old_items = {i['window_id']: i for i in original['final'].get('json', {}).get('windows', [])}
                for old, new in zip(original['window_observations'], visual['window_observations']):
                    if old['window']['window_id'] != job['target_window_id']:
                        self.assertEqual(new, old)
                for item in visual['final']['json']['windows']:
                    wid = item['window_id']
                    facts = [f for f in visual['facts'] if f['window_id'] == wid]
                    if wid == job['target_window_id']:
                        self.assertEqual(item['caption'], ' '.join(f['text'] for f in facts))
                        self.assertEqual(item['used_fact_ids'], [f['fact_id'] for f in facts])
                        self.assertEqual(item['used_boundary_ids'], [])
                    else:
                        self.assertEqual(facts, [f for f in original['facts'] if f['window_id'] == wid])
                        if old_items:
                            self.assertEqual(item, old_items[wid])
                        else:
                            self.assertEqual(item['caption'], ' '.join(f['text'] for f in facts))
                            self.assertTrue(all(f['text'] == w.bar.core.neutral_fact_text(f['raw_text']) for f in facts))
                if job['id'] == 'mrDPKCMnBHBbcIpO':
                    self.assertTrue(visual['final']['json']['windows'][1]['used_boundary_ids'])
                if job['id'] == 'FSjBuXNyaWYFrpXj':
                    self.assertEqual(visual['final']['derivation'], 'existing_neutral_fact_text_no_original_final_caption')
                    self.assertEqual(visual['original_final']['status'], 'failed')
                self.assertFalse(candidate['training_approved'])
                with self.assertRaisesRegex(ValueError, 'original strict raw target'):
                    w.package(job, observations, original, joint, prepared, plan)
                changed = copy.deepcopy(observations)
                other = next(i for i in range(len(changed)) if i != target)
                changed[other]['status'] = 'failed'
                with self.assertRaises(w.q.SourceChanged):
                    w.package(job, changed, original, joint, prepared, plan, raw)

    def test_new_empty_remains_partial_no_text_or_completion_promise(self):
        job, original, joint, prepared, plan, manifest = self.records[0]
        raw = body_raw(manifest, prepared['forbidden'])
        raw.update(events=[], abstain=True)
        filtered = w.validate_coarse(raw, manifest, prepared['forbidden'])
        observations = copy.deepcopy(original['window_observations'])
        target = next(i for i, o in enumerate(observations) if o['window']['window_id'] == job['target_window_id'])
        observations[target] = {'window': observations[target]['window'], 'status': 'abstain', 'json': filtered}
        visual, candidate = w.package(job, observations, original, joint, prepared, plan, raw)
        self.assertFalse(candidate['joint_complete'])
        self.assertFalse(candidate['target_text'])
        self.assertFalse(visual['final']['model_call_performed'])


class Sampling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeModel:
    _ms_bar_model_sha256 = 'cpu-fixture-model'
    def __init__(self, raw, fail=False):
        self.raw, self.fail, self.calls = raw, fail, 0
    def get_tokenizer(self):
        return types.SimpleNamespace(eos_token_id=9)
    def chat(self, messages, sampling, use_tqdm=False):
        self.calls += len(messages)
        if self.fail:
            raise RuntimeError('CPU simulated shared dispatch failure')
        completion = types.SimpleNamespace(text=json.dumps(self.raw), token_ids=[1, 9], finish_reason='stop', stop_reason=None)
        return [types.SimpleNamespace(outputs=[completion], prompt_token_ids=[1]) for _ in messages]


class SourceBudgetLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(w, 'OUT', root / 'out'))
        self.stack.enter_context(patch.object(w, 'PRIMARY', root / 'primary'))
        self.evidence, self.canonicals, self.reservations = {}, {}, []
        self.args = w.worker_args()
        fake_sampling_module = types.ModuleType('vllm.sampling_params')
        fake_sampling_module.StructuredOutputsParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
        self.stack.enter_context(patch.dict(sys.modules, {'vllm': types.ModuleType('vllm'),
                                                         'vllm.sampling_params': fake_sampling_module}))
        self.model = FakeModel(None)
        self.runtime_loads = 0
        self.admissions = 0
        self.stack.enter_context(patch.object(w, 'protocol', return_value={
            'native_source_sha256': {}, 'model_identity_sha256': self.model._ms_bar_model_sha256}))
        self.stack.enter_context(patch.object(w, 'source_freeze', return_value={}))
        w.q.save_once(w.OUT / 'source.freeze.json', {})
        refs = []
        for sid, spec in w.TARGETS.items():
            target = {'window_id': spec[0], 'start': 1., 'end': 2.}
            other = {'window_id': 'untargeted', 'start': 0., 'end': 1.}
            manifest = {'window': target, 'frames': [{'index': 2, 'time': 1.1}, {'index': 3, 'time': 1.2}]}
            other_manifest = {'window': other, 'frames': [{'index': 0, 'time': .1}, {'index': 1, 'time': .2}]}
            raw = body_raw(manifest)
            self.model.raw = raw
            old_raw = body_raw(other_manifest, extra_part=False)
            old = w.validate_coarse(old_raw, other_manifest, {})
            empty = w.bar.empty_observation('No admitted motion events.')
            obs = [{'window': other, 'status': 'candidate', 'json': old}, {'window': target, 'status': 'abstain', 'json': empty}]
            facts = w.bar.facts_from_observations([other, target], obs)
            binding = {'source_binding': {'id': sid}}
            visual = {'id': sid, 'binding': binding, 'window_observations': obs, 'facts': facts,
                'original_windows': [other, target], 'boundary_observations': [], 'coverage': {},
                'final': {'status': 'failed'}, 'status': 'failed'}
            row = {'id': sid, 'split': 'train', 'batch_id': 'cpu', 'video_path': '/fixture/video',
                   'audio_path': '/fixture/audio', 'source_refs': {'fixture': True}}
            style = {'dance_style': 'fixture_style', 'status': 'resolved', 'original_payload': [1, 2]}
            joint = w.pipeline.joint_candidate(row, visual, style)
            canonical = {'id': sid, 'source_key': 'train/' + sid, 'split': 'train', 'batch_id': 'cpu',
                'stages': ['reobserve', 'rewrite'], 'failures': [{'stage': 'rewrite', 'reason': 'original_failure'}],
                'global_revision': w.ORIGINAL_REVISION, 'queue_root': str(w.PRIMARY),
                'maximum_extra_source_attempts': 1, 'maximum_reobserve_requests': 2, 'maximum_rewrite_requests': 1,
                'original_result_sha256': sid, 'original_result_ref': {'path': '/fixture/result', 'sha256': sid},
                'source_binding': {'id': sid}, 'refs': {}, 'visual_root': '/fixture/visual',
                'window_diagnostics': [{'window_id': spec[0], 'body_status': 'present', 'status': 'abstain',
                                        'accepted_event_count': 0, 'manifest_ref': {'path': '/fixture/manifest', 'sha256': sid}}]}
            cp = root / 'canonical' / (sid + '.json')
            w.q.save_once(cp, canonical)
            cref = w.q.identity(cp)
            self.canonicals[sid] = canonical
            job = copy.deepcopy(canonical)
            job['refs'].update(original_canonical_job=cref, visual_result={'path': '/fixture/visual', 'sha256': sid})
            job.update(primary_budget_root=str(w.PRIMARY), target_window_id=spec[0],
                ledger_bindings={'original_canonical_job_ref': cref},
                execution_plan={'visual_model_requests': 1, 'text_model_requests': 0,
                                'original_maximum_reobserve_requests': 2, 'original_maximum_rewrite_requests': 1})
            jp = w.OUT / 'jobs' / (job['source_key'] + '.json')
            w.q.save_once(jp, job)
            jref = w.q.identity(jp)
            refs.append({'source_key': job['source_key'], 'job_ref': jref})
            prepared = {'windows': [other_manifest, manifest], 'forbidden': {}}
            plan = {'manifest_record': row}
            self.evidence[sid] = (visual, joint, prepared, plan)
        w.q.save_once(w.OUT / 'jobs.manifest.json', {'schema': w.SCHEMA, 'global_revision': w.REVISION,
                                                    'source_count': 3, 'jobs': refs})
        self.refs = [item['job_ref'] for item in refs]
        self.stack.enter_context(patch.object(w, 'verify_job', side_effect=lambda job, probe=False: copy.deepcopy(self.evidence[job['id']])))
        self.stack.enter_context(patch.object(w, 'verify_window', side_effect=lambda job, prepared: prepared['windows'][1]))
        def request(job, ref, prepared, args):
            raw_schema = w.bar.observation_schema([2, 3])
            return {'model': args.model, 'messages': [{'role': 'system', 'content': w.COARSE_RULES},
                {'role': 'user', 'content': [{'type': 'text', 'text': '{}'}]}],
                'sampling': {'temperature': 0., 'seed': 0, 'max_tokens': args.visual_max_tokens},
                'json_schema': raw_schema, 'prompt_sha256': w.q.hashlib.sha256(w.COARSE_RULES.encode()).hexdigest()}
        self.stack.enter_context(patch.object(w, 'make_request', side_effect=request))
        for ref in self.refs:
            job = w.q.checked(ref)
            pp = w.OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json'
            w.q.save_once(pp, request(job, ref, self.evidence[job['id']][2], self.args))
        def reserve(canonical, out, revision):
            self.assertEqual(canonical, self.canonicals[canonical['id']])
            self.assertEqual(out, w.PRIMARY)
            self.assertEqual(revision, w.REVISION)
            claim = {'source_key': canonical['source_key'], 'original_result_sha256': canonical['original_result_sha256'],
                'global_revision': revision, 'extra_source_attempt': 1, 'maximum_extra_source_attempts': 1}
            w.q.save_once(out / 'attempt-budget' / (canonical['source_key'] + '.json'), claim)
            self.reservations.append(canonical['id'])
            return claim
        self.stack.enter_context(patch.object(w.q, 'reserve_attempt', side_effect=reserve))
        def runtime():
            self.runtime_loads += 1
            return self.model, Sampling
        runtime.verify_current_admission = lambda refs: self.admit(refs)
        self.runtime = runtime
        self.stack.enter_context(patch.object(w, 'caller_admission', side_effect=lambda runtime, refs: runtime.verify_current_admission(refs)))

    def admit(self, refs):
        self.assertEqual(refs, self.refs)
        self.admissions += 1
        return {'fixture': True}

    def test_exact_three_windows_original_visual_max_and_once_budget_receipts(self):
        summary = w.execute_scope(self.args, self.runtime)
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(len(self.reservations), 3)
        self.assertEqual(self.runtime_loads, 1)
        self.assertEqual(summary['new_visual_requests'], 3)
        self.assertEqual(summary['new_text_model_requests'], 0)
        self.assertEqual(summary['target_window_success_count'], 3)
        for ref in self.refs:
            job = w.q.checked(ref)
            self.assertEqual(job['maximum_reobserve_requests'], 2)
            self.assertEqual(job['execution_plan']['visual_model_requests'], 1)
            terminal = w.q.load(w.OUT / 'worker-terminals' / (job['source_key'] + '.json'))
            candidate = w.q.checked(terminal['joint_result_ref'])
            self.assertEqual(candidate['original_failure_metadata'], self.canonicals[job['id']]['failures'])
            self.assertFalse(candidate['new_label_review_required'] is False)
            self.assertEqual(len(w.q.checked(candidate['transaction_receipt_ref'])['files_sha256']), 6)
        again = w.execute_scope(self.args, self.runtime)
        self.assertEqual(again['new_visual_requests'], 0)
        self.assertEqual(self.model.calls, 3)
        self.assertEqual(self.runtime_loads, 1)
        self.assertTrue(all(row['status'] == 'pending_budget_already_reserved' for row in again['sources']))

    def test_cli_run_default_denied_without_prepare_or_runtime(self):
        with patch.object(sys, 'argv', ['fixed3', '--run']), patch.object(w, 'prepare') as prepare:
            with self.assertRaises(SystemExit) as result:
                w.main()
            self.assertEqual(result.exception.code, 2)
            prepare.assert_not_called()
        self.assertEqual(self.reservations, [])
        self.assertEqual(self.runtime_loads, 0)
        self.assertEqual(self.model.calls, 0)

    def test_shared_load_failure_zero_claim_and_default_denial(self):
        def broken():
            raise RuntimeError('CPU shared load failure')
        broken.verify_current_admission = self.runtime.verify_current_admission
        with self.assertRaisesRegex(RuntimeError, 'load failure'):
            w.execute_scope(self.args, broken)
        self.assertEqual(self.reservations, [])
        self.assertEqual(self.model.calls, 0)
        self.assertFalse((w.PRIMARY / 'attempt-budget').exists())
        with self.assertRaisesRegex(ValueError, 'default execution denied'):
            w.execute_scope(self.args)

    def test_ambiguous_dispatch_spent_no_retry_and_later_not_reserved(self):
        with patch.object(w.bar, 'transactions', side_effect=RuntimeError('CPU simulated ambiguous dispatch')):
            summary = w.execute_scope(self.args, self.runtime)
        first = w.q.checked(self.refs[0])
        partial = w.q.load(w.OUT / 'revisions' / w.REVISION / first['source_key'] / 'partial.dispatch.json')
        self.assertEqual(partial['status'], 'partial_ambiguous_dispatch_spent')
        self.assertEqual(self.reservations, [first['id']])
        self.assertEqual(summary['new_visual_requests'], 1)
        self.assertTrue(all(row['status'] == 'not_reserved_after_shared_failure' for row in summary['sources'][1:]))
        for ref in self.refs[1:]:
            self.assertFalse(w.budget_path(w.q.checked(ref)).exists())

    def test_shared_dispatch_failure_persist_partial_and_later_unreserved(self):
        self.model.fail = True
        summary = w.execute_scope(self.args, self.runtime)
        self.assertEqual(self.reservations, [list(w.TARGETS)[0]])
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(summary['new_visual_requests'], 1)
        first = w.q.checked(self.refs[0])
        terminal = w.q.load(w.OUT / 'worker-terminals' / (first['source_key'] + '.json'))
        self.assertTrue(terminal['shared_dispatch_failure'])
        self.assertFalse(terminal['target_window_success'])
        self.assertFalse(terminal['joint_complete'])
        self.assertTrue(Path(terminal['joint_result_ref']['path']).exists())
        for ref in self.refs[1:]:
            job = w.q.checked(ref)
            self.assertFalse(w.budget_path(job).exists())
        self.assertTrue(all(row['status'] == 'not_reserved_after_shared_failure' for row in summary['sources'][1:]))


if __name__ == '__main__':
    unittest.main()
