"""Collector mechanics are synthetic; motion accuracy is never inferred from these tests."""
import importlib.util
import copy
from types import SimpleNamespace
from unittest import mock
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
MODULE_PATH = Path(__file__).resolve().parents[1] / 'curated_repair_queue_v1.py'
spec = importlib.util.spec_from_file_location('repair_queue', MODULE_PATH)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
EXP = MODULE_PATH.parents[1]
PRODUCTION = EXP / 'curated222939_production_v1'
FROZEN_BAR = EXP / 'runtime/source_curated_consumer_v1'


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.production = self.base / 'production'
        self.out = self.base / 'repair'
        self.production.mkdir()
        (self.production / 'completed.jsonl').write_bytes(b'')

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(queue.encoded(value))
        return queue.identity(path)

    def row(self, sid='case1', complete=False, raw_body=b'', final_error=None, style_unknown=False):
        directory = self.production / 'sources/train' / sid
        producer = {'status': 'ready', 'source_binding': {'id': sid, 'video_sha256': 'bound-video'}}
        producer['plan_ref'] = self.write_json(directory / 'bar_plan.json', {'id': sid})
        producer['roi_ref'] = self.write_json(self.production / 'roi' / sid / 'roi_manifest.json', {'id': sid})
        self.write_json(directory / 'producer.terminal.json', producer)
        visual_root = self.production / 'visual/shard-0000/sources' / sid
        attempt = visual_root / 'windows/w00000/attempt-0'
        self.write_json(attempt / 'request.json', {'test': 'collector only, not motion truth'})
        if raw_body is not None:
            (attempt / 'response.raw.txt').write_bytes(raw_body)
        observation = {'window': {'window_id': 'w00000'}, 'status': 'candidate' if final_error else 'abstain',
                       'json': {'events': [{'id': 'e1'}] if final_error else []}, 'attempt_path': str(attempt)}
        final = {'status': 'failed', 'error': final_error} if final_error else {'status': 'abstain'}
        visual = {'window_observations': [observation], 'final': final}
        visual_ref = self.write_json(visual_root / 'results/original.json', visual)
        joint = {'visual_result_ref': visual_ref, 'all_windows_nonempty': complete,
                 'dance_style_label': 'unknown' if style_unknown else 'dance',
                 'style_result': {'status': 'unknown' if style_unknown else 'resolved'}}
        joint_ref = self.write_json(directory / 'joint-results/original.json', joint)
        row = {'id': sid, 'split': 'train', 'batch_id': 'train-b000000',
               'status': 'joint_candidate' if complete else 'incomplete_candidate',
               'joint_complete': complete, 'joint_candidate_ref': joint_ref}
        self.write_json(directory / 'consumer.terminal.json', row)
        return row

    def feed(self, row, newline=True):
        with (self.production / 'completed.jsonl').open('ab') as stream:
            data = queue.encoded(row)
            stream.write(data if newline else data[:-1])

    def collect(self, revision='r1'):
        return queue.collect(self.production, self.out, revision)

    def job(self, sid='case1'):
        return queue.load(self.out / 'jobs/train' / (sid + '.json'))

    def test_incremental_duplicate_complete_and_revision(self):
        first = self.row()
        self.feed(first)
        self.feed(self.row('complete1', complete=True))
        result = self.collect()
        self.assertEqual(result['counts'], {'incomplete_or_failed': 1, 'joint_complete': 1})
        self.assertEqual(result['new_queued_sources'], 1)
        self.feed(first)
        self.assertEqual(self.collect('r2')['new_queued_sources'], 0)
        self.assertEqual(self.job()['global_revision'], 'r1')
        self.assertEqual(len((self.out / 'ready.jsonl').read_bytes().splitlines()), 1)

    def test_eof_partial_never_processed_even_valid_json(self):
        self.feed(self.row(), newline=False)
        result = self.collect()
        self.assertEqual(result['sources_seen'], 0)
        self.assertEqual(result['input_byte_offset'], 0)
        with (self.production / 'completed.jsonl').open('ab') as stream:
            stream.write(b'\n')
        self.assertEqual(self.collect()['sources_seen'], 1)

    def test_nested_source_refs_cross_mount_device_mismatch_does_not_fake_stat(self):
        path = self.base / 'artifact.json'
        ref = self.write_json(path, {'source': 'bound contents'})
        reference = {'path': ref['path'], 'sha256': ref['sha256'],
                     'stat': {'device': path.stat().st_dev + 7, 'inode': path.stat().st_ino}}
        original = copy.deepcopy(reference)
        queue.verify_references_by_content({'files': {'video_path': reference}, 'refs': [reference]})
        self.assertEqual(reference, original)
        self.assertNotEqual(reference['stat']['device'], path.stat().st_dev)
        path.write_bytes(b'changed actual contents')
        with self.assertRaises(queue.SourceChanged):
            queue.verify_references_by_content({'refs': [reference]})

    def test_hash_rejects_actual_metadata_change_during_read(self):
        path = self.base / 'hash.bin'
        path.write_bytes(b'original')
        before = path.stat()
        changed = SimpleNamespace(st_dev=before.st_dev, st_ino=before.st_ino,
            st_size=before.st_size + 1, st_mtime_ns=before.st_mtime_ns,
            st_ctime_ns=before.st_ctime_ns)
        with mock.patch.object(queue.os, 'fstat', side_effect=[before, changed]):
            with self.assertRaisesRegex(queue.SourceChanged, 'changed while hashing'):
                queue.sha(path)

    def test_demo_extra_image_sha_change_rejected_before_budget(self):
        self.feed(self.row())
        self.collect()
        job = self.job()
        image = self.base / 'actual-frame.jpg'
        image.write_bytes(b'original-pixels')
        manifest = self.base / 'demo-manifest.json'
        job['refs']['demo_manifest'] = self.write_json(manifest, {
            'frames': [{'file': str(image), 'sha256': queue.sha(image)}]})
        queue.verify_job(job)
        image.write_bytes(b'changed-pixels')
        with self.assertRaises(queue.SourceChanged):
            queue.reserve_attempt(job, self.out, 'r1')
        self.assertFalse((self.out / 'attempt-budget/train/case1.json').exists())

    def test_body_absent_empty_present_malformed_distinguished(self):
        raw = queue.encoded({'events': [], 'part_support': []})
        for sid, body in [('absent', None), ('empty', b''), ('present', raw), ('malformed', b'{')]:
            self.feed(self.row(sid, raw_body=body))
        self.collect()
        for status in ('absent', 'empty', 'present', 'malformed'):
            self.assertEqual(self.job(status)['window_diagnostics'][0]['body_status'], status)

    def test_reference_protocol_vs_insufficient_support(self):
        raw = queue.encoded({'events': [{'id': 'e1', 'parts': ['left_hand', 'right_hand'],
                                         'frame_indices': [0, 10]}],
            'part_support': [{'part': 'left_hand', 'frame_indices': [0, 2]},
                             {'part': 'right_hand', 'frame_indices': [0]}]})
        self.feed(self.row(raw_body=raw))
        self.collect()
        deficits = self.job()['window_diagnostics'][0]['reference_deficits']
        self.assertTrue(deficits[0]['technical_reference_mismatch_possible'])
        self.assertTrue(deficits[1]['current_part_support_insufficient'])
        self.assertFalse(self.job()['repair_completed'])

    def test_original_result_sha_changed_refused(self):
        row = self.row()
        Path(row['joint_candidate_ref']['path']).write_bytes(b'{}\n')
        self.feed(row)
        with self.assertRaises(queue.SourceChanged):
            self.collect()

    def test_changed_completed_source_revision_refused(self):
        row = self.row()
        self.feed(row)
        self.collect()
        changed = {**row, 'status': 'failed'}
        self.feed(changed)
        with self.assertRaises(queue.SourceChanged):
            self.collect('r2')

    def test_one_attempt_across_revisions_and_source_sha_check(self):
        row = self.row(raw_body=b'{')
        self.feed(row)
        self.collect()
        job = self.job()
        queue.reserve_attempt(job, self.out, 'r1')
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            queue.reserve_attempt(job, self.out, 'r2')
        with self.assertRaisesRegex(ValueError, 'original queue root'):
            queue.reserve_attempt(job, self.base / 'different', 'r2')
        Path(row['joint_candidate_ref']['path']).write_bytes(b'{}\n')
        with self.assertRaises(queue.SourceChanged):
            queue.reserve_attempt(job, self.out, 'r3')

    def test_existing_raw_sha_change_refused_before_attempt(self):
        self.feed(self.row(raw_body=b'{'))
        self.collect()
        job = self.job()
        Path(job['window_diagnostics'][0]['raw_ref']['path']).write_bytes(b'changed')
        with self.assertRaises(queue.SourceChanged):
            queue.reserve_attempt(job, self.out, 'r1')
        self.assertFalse((self.out / 'attempt-budget/train/case1.json').exists())

    def test_previously_absent_original_body_appeared_refused(self):
        self.feed(self.row(raw_body=None))
        self.collect()
        job = self.job()
        Path(job['window_diagnostics'][0]['attempt_path'], 'response.raw.txt').write_bytes(b'{}')
        with self.assertRaisesRegex(queue.SourceChanged, 'absent original response'):
            queue.reserve_attempt(job, self.out, 'r1')

    def test_malformed_complete_stream_line_refused_not_silently_skipped(self):
        (self.production / 'completed.jsonl').write_bytes(b'{\n')
        with self.assertRaises(ValueError):
            self.collect()
        self.assertFalse((self.out / 'jobs').exists())

    def test_failed_rewrite_and_unknown_style_are_separate(self):
        self.feed(self.row(final_error='body part absent', style_unknown=True))
        self.collect()
        self.assertEqual(self.job()['stages'], ['rewrite', 'styleunknown'])
        self.assertEqual(self.job()['maximum_reobserve_requests'], 0)
        self.assertEqual(self.job()['maximum_rewrite_requests'], 1)

    def test_media_failure_without_joint(self):
        directory = self.production / 'sources/train/media1'
        producer_ref = self.write_json(directory / 'producer.terminal.json',
                                       {'status': 'failed', 'error': 'actual media missing'})
        row = {'id': 'media1', 'split': 'train', 'status': 'producer_failed',
               'producer_terminal_ref': producer_ref}
        self.write_json(directory / 'consumer.terminal.json', row)
        self.feed(row)
        self.collect()
        self.assertEqual(self.job('media1')['stage'], 'media')

    def test_input_truncation_and_bad_complete_line_refused(self):
        self.feed(self.row())
        self.collect()
        (self.production / 'completed.jsonl').write_bytes(b'')
        with self.assertRaises(queue.SourceChanged):
            self.collect()


@unittest.skipUnless(PRODUCTION.is_dir() and FROZEN_BAR.is_dir(), 'live evidence not installed')
class RealEvidenceTests(unittest.TestCase):
    """Read-only checks of real failures, not claims of VLM motion accuracy."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(FROZEN_BAR))
        import ms_bar_annotation
        cls.bar = ms_bar_annotation
        cls.rows = {}
        with (PRODUCTION / 'completed.jsonl').open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                row = json.loads(line)
                cls.rows[row['id']] = row

    def case(self, sid):
        return queue.classify(self.rows[sid], PRODUCTION)

    def test_actual_raq_demo_body_budget_original_pixels_and_checklist(self):
        out = EXP / 'legacy_raq_repair_v1'
        job = queue.load(out / 'jobs/train/RAQoxqRxCfMvTHSx.json')
        queue.verify_job(job)
        manifest = queue.checked(job['refs']['demo_manifest'])
        self.assertEqual([f['index'] for f in manifest['frames']],
                         [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 52, 55])
        self.assertEqual(manifest['sampling_policy']['actual_body_images'], 14)
        self.assertEqual(manifest['sampling_policy']['actual_ROI_images'], 2)
        self.assertTrue(all(1.58 <= f['time'] < 3.44 for f in manifest['frames']))
        old = queue.load(job['window_diagnostics'][0]['manifest_ref']['path'])
        by_index = {f['index']: f for f in manifest['frames']}
        for frame in old['frames']:
            self.assertEqual(by_index[frame['index']]['sha256'], frame['sha256'])
        preview = queue.load(Path(job['refs']['demo_manifest']['path']).with_name('request.preview.json'))
        self.assertEqual(sum(x['type'] == 'image_url' for x in preview['messages'][1]['content']), 16)
        instruction = json.loads(preview['messages'][1]['content'][-1]['text'])
        self.assertEqual(instruction['original_event_ids'], ['e1'])
        self.assertEqual(instruction['original_raw_ref'], job['window_diagnostics'][0]['raw_ref'])
        self.assertIn('Coarse hand/wrist displacement', instruction['support_policy'])
        self.assertFalse((out / 'attempt-budget').exists())

    def test_actual_raq_cross_mount_content_pts_validation_does_not_modify_stat(self):
        root = EXP / 'full12000_stable_roi_v1/shard0/sources/RAQoxqRxCfMvTHSx'
        original = (root / 'prepared.json').read_bytes()
        prepared = queue.verify_prepared_content(root, queue.load(root / 'plan.json'), self.bar, '/opt/ms_tools/ffprobe')
        self.assertEqual(prepared['binding']['source_binding']['id'], 'RAQoxqRxCfMvTHSx')
        self.assertEqual(original, (root / 'prepared.json').read_bytes())

    def test_uncertain_shape_in_fact_rejected_without_modifying_raw(self):
        root = EXP / 'full12000_stable_roi_v1/shard0/sources/RAQoxqRxCfMvTHSx'
        raw = queue.load(root / 'windows/w00001/attempt-0/response.raw.txt')
        old_raw = copy.deepcopy(raw)
        manifest = queue.load(root / 'windows/w00001/manifest.json')
        with self.assertRaisesRegex(ValueError, 'Uncertain detail'):
            queue.validate_reobserve(raw, manifest, {}, self.bar)
        self.assertEqual(raw, old_raw)

    def test_finger_shape_needs_roi_two_distinct_frames_not_head_or_body(self):
        manifest = {'hand_roi_evidence': {'right_hand': {'crops': [{'frame_index': 0}]}}}
        value = {'events': [{'description': 'The right hand forms a V-sign.',
                            'parts': ['right_hand'], 'frame_indices': [0, 2]}]}
        with self.assertRaisesRegex(ValueError, 'two distinct actual native ROI'):
            queue.validate_reobserve(value, manifest, {}, self.bar)

    def test_untargeted_window_exact_fields_are_frozen(self):
        frozen = {'w00000': {'window_id': 'w00000', 'caption': 'original',
                             'used_fact_ids': ['w00000:e1'], 'used_boundary_ids': []}}
        value = {'windows': [{**frozen['w00000'], 'caption': 'rephrased'}], 'issues': []}
        bar = SimpleNamespace(check_supported_rewrite=lambda *args: None)
        with self.assertRaisesRegex(ValueError, 'Untargeted legacy demo window'):
            queue.validate_repair_rewrite(value, [], [], [], bar, frozen)

    def test_real_four_risks_retained_for_independent_review(self):
        for sid in ('YHdOyRCqljUFfUJV', 'aBSengdTPwixaKcN', 'OAXBZGQJSmIWjJSD', 'jBZpnklmWcwxZUgZ'):
            job = self.case(sid)
            queue.verify_job(job)
            self.assertTrue(job['risk_review_case'])
            self.assertEqual(job['independent_review_status'], 'required_after_extra_attempt')
            self.assertFalse(job['repair_completed'])

    def test_actual_reference_deficits_still_rejected_not_autofilled(self):
        job = self.case('YHdOyRCqljUFfUJV')
        diagnostic = job['window_diagnostics'][0]
        self.assertEqual(diagnostic['raw_event_count'], 2)
        self.assertEqual(diagnostic['accepted_event_count'], 0)
        raw = queue.load(diagnostic['raw_ref']['path'])
        manifest = queue.load(diagnostic['manifest_ref']['path'])
        prepared = queue.load(job['refs']['prepared']['path'])
        forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
        with self.assertRaisesRegex(ValueError, 'two visible evidence frames'):
            self.bar.validate_observation(raw, manifest, forbidden)
        filtered = self.bar.filter_supported_observation(raw, manifest, forbidden)
        self.assertTrue(filtered['abstain'])
        self.assertEqual(filtered['events'], [])

    def test_actual_rewrite_introduced_part_and_empty_window_failures(self):
        for sid, error in [('OAXBZGQJSmIWjJSD', 'body part absent'),
                           ('jBZpnklmWcwxZUgZ', 'empty/partial window')]:
            job = self.case(sid)
            rewrite = next(f for f in job['failures'] if f['stage'] == 'rewrite')
            self.assertIn(error, rewrite['reason'])
            visual = queue.checked(job['refs']['visual_result'])
            request = queue.load(Path(rewrite['attempt_path']) / 'request.json')
            supplied = json.loads(request['messages'][1]['content'])
            raw = queue.load(Path(rewrite['attempt_path']) / 'response.raw.txt')
            windows = [o['window'] for o in visual['window_observations']]
            with self.assertRaises(ValueError):
                self.bar.check_supported_rewrite(raw, windows, visual['facts'], supplied['boundary_relations'])


if __name__ == '__main__':
    unittest.main()
