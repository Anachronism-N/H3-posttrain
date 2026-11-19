import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import curated_coarse_empty_filter_recovery_v1 as r


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = r.c.worker_module()
        cls.job = r.c.load(cls.worker.OUT / 'jobs/train/AuuitdcBJyGOnWaH.json')
        cls.prepared = r.c.checked(cls.job['refs']['prepared'])
        cls.manifest = cls.worker.verify_window(cls.job, cls.prepared)
        cls.attempt = cls.worker.OUT / 'revisions' / cls.worker.REVISION / 'train/AuuitdcBJyGOnWaH/windows/w00000/attempt-000001'
        cls.raw = cls.worker.bar.core.strict_json((cls.attempt / 'response.raw.txt').read_bytes())

    def test_actual_native_normalization_failure(self):
        with self.assertRaisesRegex(ValueError, r.NORMALIZATION_ERROR):
            self.worker.validate_coarse(self.raw, self.manifest, self.prepared['forbidden'])
        value = r.replay_raw(self.worker, self.raw, self.manifest, self.prepared['forbidden'])
        self.assertEqual(len(value['events']), 2)
        self.assertEqual(value['events'][0]['parts'], ['left_arm', 'right_arm'])
        self.assertEqual(value['events'][0]['description'], self.raw['events'][0]['description'])
        self.assertEqual(value['events'][0]['frame_indices'], self.raw['events'][0]['frame_indices'])

    def test_foreign_frame_rejected(self):
        value = copy.deepcopy(self.raw)
        value['events'][0]['frame_indices'][0] = 999999
        with self.assertRaises(ValueError):
            r.replay_raw(self.worker, value, self.manifest, self.prepared['forbidden'])

    def test_insufficient_support_rejected(self):
        value = copy.deepcopy(self.raw)
        for item in value['part_support']:
            if item['part'] == 'left_arm':
                item['frame_indices'] = []
        with self.assertRaises(ValueError):
            r.replay_raw(self.worker, value, self.manifest, self.prepared['forbidden'])

    def test_fine_detail_rejected(self):
        value = copy.deepcopy(self.raw)
        value['events'][0]['description'] = 'The dancer raises both arms with fingers forming a V sign.'
        with self.assertRaises(ValueError):
            r.replay_raw(self.worker, value, self.manifest, self.prepared['forbidden'])

    def test_other_validator_error_not_bypassed(self):
        with patch.object(self.worker, 'validate_coarse', side_effect=ValueError('different_failure')):
            with self.assertRaisesRegex(ValueError, 'different_failure'):
                r.replay_raw(self.worker, self.raw, self.manifest, self.prepared['forbidden'])

    def test_added_part_never_allowed(self):
        filtered = self.worker.bar.filter_supported_observation(self.raw, self.manifest, self.prepared['forbidden'])
        filtered['events'][0]['parts'].append('head')
        fake = types.SimpleNamespace(validate_coarse=lambda *args: filtered)
        with self.assertRaisesRegex(ValueError, 'No added part'):
            r.replay_raw(fake, self.raw, self.manifest, self.prepared['forbidden'])

    def test_altered_text_never_allowed(self):
        filtered = self.worker.bar.filter_supported_observation(self.raw, self.manifest, self.prepared['forbidden'])
        filtered['events'][0]['description'] += ' Invented.'
        fake = types.SimpleNamespace(validate_coarse=lambda *args: filtered)
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            r.replay_raw(fake, self.raw, self.manifest, self.prepared['forbidden'])

    def test_exact_second_pass_and_restore(self):
        value = r.replay_raw(self.worker, self.raw, self.manifest, self.prepared['forbidden'])
        validator = self.worker.validate_coarse
        def package(*args):
            self.assertEqual(self.worker.validate_coarse(r.projected(value), self.manifest, self.prepared['forbidden']), value)
            bad = r.projected(value)
            bad['events'][0]['frame_indices'] = [0, 1]
            with self.assertRaisesRegex(ValueError, 'second pass'):
                self.worker.validate_coarse(bad, self.manifest, self.prepared['forbidden'])
            raise ValueError('package_failure')
        with patch.object(self.worker, 'package', side_effect=package):
            with self.assertRaisesRegex(ValueError, 'package_failure'):
                r.package_replayed(self.worker, self.job, [], {}, {}, self.prepared, {}, value)
        self.assertIs(self.worker.validate_coarse, validator)

    def test_original_generation_uses_native_digest(self):
        request = r.c.load(self.attempt / 'request.json')
        generation = self.worker.bar.core.validate_generation(self.attempt, request)
        self.assertEqual(generation['structured_output_schema_sha256'], self.worker.bar.batch.digest(request['json_schema']))
        self.assertNotEqual(generation['structured_output_schema_sha256'], self.worker.q.digest(request['json_schema']))


if __name__ == '__main__':
    unittest.main()
