"""Targeted CPU schema, provenance and source-budget checks; no GPU model load."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('io_coarse_test', RT / 'curated_io_visual_repair_v1.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class CoarseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = w.q.load(w.OUT / 'jobs.manifest.json')['jobs']
        cls.job = w.q.checked(cls.jobs[0]['job_ref'])
        cls.prepared = w.q.checked(cls.job['refs']['prepared'])
        cls.manifest = w.q.checked(cls.job['target_manifest_refs']['w00000'])

    def raw(self):
        value = w.bar.empty_observation('not inferred from crop inventory')
        value['events'] = [{'id': 'e1', 'description': 'The left arm rises outward.',
                            'parts': ['left_arm'], 'frame_indices': [0, 2]}]
        for item in value['part_support']:
            if item['part'] == 'left_arm':
                item['frame_indices'] = [0, 2]
                item['reason'] = 'visually clear in original body context'
        value['abstain'] = False
        return value

    def test_request_same_images_and_schema(self):
        args = w.worker_args()
        for hint in self.jobs:
            job = w.q.checked(hint['job_ref'])
            prepared = w.q.checked(job['refs']['prepared'])
            forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
            for wid in w.TARGETS[job['id']]:
                manifest = w.verify_window(job, wid, prepared)
                old = w.bar.visual_request(args, Path(job['visual_root']) / 'rgb', manifest, forbidden)
                new = w.make_request(job, wid, manifest, prepared, args, hint['job_ref'])
                self.assertEqual(old['messages'][1], new['messages'][1])
                self.assertEqual(old['json_schema'], new['json_schema'])
                self.assertEqual(new['repair_provenance']['current_job_source_binding'], job['source_binding'])
                self.assertEqual(new['repair_provenance']['manifest_ref'], job['target_manifest_refs'][wid])
                self.assertLessEqual(sum(v['type'] == 'image_url' for v in new['messages'][1]['content']), 16)

    def test_correct_coarse_support_without_hand_inventory(self):
        result = w.validate_coarse(self.raw(), self.manifest, {})
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['parts'], ['left_arm'])
        self.assertEqual(result['events'][0]['frame_indices'], [0, 2])

    def test_fine_hand_rejected_not_trimmed(self):
        raw = self.raw()
        raw['events'][0]['description'] = 'The left arm rises with a clenched fist.'
        with self.assertRaises(ValueError):
            w.validate_coarse(raw, self.manifest, {})

    def test_multiple_sentences_rejected(self):
        raw = self.raw()
        raw['events'][0]['description'] = 'The left arm rises. The left arm lowers.'
        with self.assertRaises(ValueError):
            w.validate_coarse(raw, self.manifest, {})

    def test_unknown_event_and_frame_ids_rejected(self):
        for change in ({'id': 'e99'}, {'frame_indices': [0, 999]}, {'frame_indices': [0, 0]}):
            raw = self.raw()
            raw['events'][0].update(change)
            with self.assertRaises(ValueError):
                w.validate_coarse(raw, self.manifest, {})

    def test_part_intersection_not_autofilled(self):
        raw = self.raw()
        for part in raw['part_support']:
            if part['part'] == 'left_arm':
                part['frame_indices'] = [0, 4]
        with self.assertRaises(ValueError):
            w.validate_coarse(raw, self.manifest, {})

    def test_unknown_source_rejected_before_io(self):
        job = copy.deepcopy(self.job)
        job['id'] = 'UnknownSource'
        with self.assertRaises(ValueError):
            w.verify_job(job)

    def test_unknown_roi_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['hand_roi_evidence']['left_hand']['crops'][0]['sha256'] = '0' * 64
        prepared = copy.deepcopy(self.prepared)
        prepared['windows'][0]['hand_roi_evidence'] = manifest['hand_roi_evidence']
        real_checked = w.q.checked
        def checked(ref):
            if ref == self.job['target_manifest_refs']['w00000']:
                return manifest
            return real_checked(ref)
        with patch.object(w.q, 'checked', checked), self.assertRaisesRegex(ValueError, 'Unknown or changed ROI crop'):
            w.verify_window(self.job, 'w00000', prepared)

    def budget_job(self, tmp):
        job = copy.deepcopy(self.job)
        job['primary_budget_root'] = str(tmp)
        refpath = Path(tmp) / 'independent-job.json'
        w.q.save_once(refpath, job)
        return job, w.q.identity(refpath)

    def test_duplicate_budget_refuses_runtime_second_time(self):
        with tempfile.TemporaryDirectory(prefix='io-coarse-budget-') as tmp:
            job, ref = self.budget_job(tmp)
            calls = []
            runtime = lambda: calls.append('load')
            claim = w.reserve_primary_attempt(job, ref, runtime, tmp, lambda job: None)
            self.assertEqual(claim['original_result_sha256'], job['original_result_sha256'])
            self.assertEqual(calls, ['load'])
            with self.assertRaisesRegex(ValueError, 'already reserved'):
                w.reserve_primary_attempt(job, ref, runtime, tmp, lambda job: None)
            self.assertEqual(calls, ['load'])

    def test_shared_load_failure_does_not_reserve(self):
        with tempfile.TemporaryDirectory(prefix='io-coarse-load-') as tmp:
            job, ref = self.budget_job(tmp)
            def failed():
                raise RuntimeError('shared load failure')
            with self.assertRaisesRegex(RuntimeError, 'shared load failure'):
                w.reserve_primary_attempt(job, ref, failed, tmp, lambda job: None)
            self.assertFalse((Path(tmp) / 'attempt-budget' / (job['source_key'] + '.json')).exists())

    def test_new_out_cannot_bypass_primary_budget(self):
        with tempfile.TemporaryDirectory(prefix='io-coarse-wrong-root-') as tmp:
            job, ref = self.budget_job(tmp)
            with self.assertRaisesRegex(ValueError, 'original primary budget root'):
                w.reserve_primary_attempt(job, ref, lambda: self.fail('must not load'), Path(tmp) / 'other', lambda job: None)

    def test_empty_stays_empty_and_Ub_front_frozen(self):
        for hint in self.jobs:
            job = w.q.checked(hint['job_ref'])
            original = w.q.checked(job['refs']['visual_result'])
            prepared = w.q.checked(job['refs']['prepared'])
            windows = [m['window'] for m in prepared['windows']]
            body = w.concatenate(windows, original['facts'], original, set(w.TARGETS[job['id']]))
            for item in body['windows']:
                if item['window_id'] in w.TARGETS[job['id']]:
                    self.assertEqual(item['caption'], '')
                    self.assertEqual(item['used_fact_ids'], [])
            if job['id'] == 'UbjwIZuNtSoLcUaG':
                old = original['final']['json']['windows'][0]
                self.assertEqual(body['windows'][0], old)

    def test_packaged_partial_labels_remain_unapproved(self):
        for hint in self.jobs:
            job = w.q.checked(hint['job_ref'])
            original = w.q.checked(job['refs']['visual_result'])
            old_joint = w.q.checked(job['refs']['joint_result'])
            prepared = w.q.checked(job['refs']['prepared'])
            plan = w.q.checked(job['refs']['plan_ref'])
            with tempfile.TemporaryDirectory(prefix='io-coarse-package-') as tmp:
                out, primary = Path(tmp) / 'out', Path(tmp) / 'primary'
                claim = {'schema': w.SCHEMA, 'extra_source_attempt': 1}
                w.q.save_once(primary / 'attempt-budget' / (job['source_key'] + '.json'), claim)
                with patch.object(w, 'OUT', out), patch.object(w, 'PRIMARY', primary), patch.object(w, 'verify_job'):
                    ref, joint = w.package(job, hint['job_ref'], copy.deepcopy(original['window_observations']),
                                           original, old_joint, prepared, plan, claim)
                self.assertFalse(joint['joint_complete'])
                self.assertFalse(joint['labels_complete'])
                self.assertFalse(joint['training_approved'])
                self.assertFalse(joint['training_eligible'])
                self.assertEqual(joint['target_text'], '')
                self.assertEqual(joint['style_result'], old_joint['style_result'])
                visual = w.q.checked(joint['visual_result_ref'])
                self.assertEqual(visual['window_observations'], original['window_observations'])
                self.assertEqual(visual['facts'], original['facts'])

    def test_safe_template_cannot_launch(self):
        value = w.q.load(RT / 'curated_io_visual_repair_safe_template_v1.json')
        self.assertFalse(value['release_authorized'])
        self.assertFalse(value['launch_authorized'])
        self.assertEqual(value['holders'], [])
        self.assertEqual(value['gpus'], [])
        self.assertEqual(value['tensor_parallel_size'], 2)
        self.assertEqual(value['required_GPU_count'], 2)


def native_compile():
    # Native xgrammar shared-library parser only: no tokenizer, LLM or CUDA call.
    import xgrammar
    rows = []
    for hint in w.q.load(w.OUT / 'jobs.manifest.json')['jobs']:
        job = w.q.checked(hint['job_ref'])
        for wid in w.TARGETS[job['id']]:
            req = w.q.load(w.OUT / 'request-previews' / job['source_key'] / wid / 'request.json')
            schema = json.dumps(req['json_schema'])
            grammar = xgrammar.Grammar.from_json_schema(schema, any_whitespace=False, strict_mode=True)
            rows.append({'id': job['id'], 'window_id': wid,
                'schema_sha256': w.q.digest(req['json_schema']),
                'native_XGrammar_schema_compiled': grammar is not None})
    w.q.save_once(w.OUT / 'cpu.native_xgrammar.receipt.json', {
        'schema': w.SCHEMA, 'checks': rows, 'GPU_dispatch_performed': False,
        'LLM_or_tokenizer_initialized': False, **w.q.FLAGS})
    print(json.dumps(rows))


if __name__ == '__main__':
    if '--native-compile' in __import__('sys').argv:
        native_compile()
    else:
        unittest.main()
