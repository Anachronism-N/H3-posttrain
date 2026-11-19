#!/usr/bin/env python3
"""Real three-source evidence integration tests; no model/GPU/original writes."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import oldbar_explicit_import_v1 as imp

SIDS = ['hwrlCvOCYUKpljlV', 'KMcaVVwQmswUfTuP', 'qKgmlBaDdOeqvUZZ']


class RealOldbarImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='oldbar_explicit_real3_v1_', dir='/tmp')
        cls.root = Path(cls.temp.name)
        cls.importer = imp.Importer(SIDS, cohort='oldbar3-import-v1')
        cls.targets = {r['id']: r for r in [json.loads(line) for line in
            (imp.RT / 'oldbar3_explicit_alternative_candidates_v1.jsonl').read_text().splitlines() if line.strip()]}
        cls.candidates = {sid: cls.importer.import_one(sid) for sid in SIDS}
        cls.results = {sid: imp.load(cls.targets[sid]['source_refs']['import_provenance']['old_completed_record']['path']) for sid in SIDS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def evidence(self, result, current=None):
        sid = SIDS[0]
        plan, _, raw_line = self.importer.plans[sid]
        row = current or self.importer.current_row(sid)[0]
        return self.importer.verify_evidence(result, plan, raw_line, Path(self.targets[sid]['source_hint']), row)

    def test_01_real_three_happy_paths_exact_targets_windows_and_roi(self):
        for sid in SIDS:
            with self.subTest(sid=sid):
                candidate, target = self.candidates[sid], self.targets[sid]
                self.assertEqual(candidate['target_text'], target['target_text'])
                self.assertEqual(candidate['label'], target['label'])
                self.assertEqual(candidate['source_binding'], target['source_binding'])
                self.assertEqual(candidate['hand_roi'], target['hand_roi'])
                self.assertEqual(candidate['origin_root'], target['origin_root'])
                self.assertEqual(candidate['source_hint'], target['source_hint'])
                provenance = candidate['source_refs']['import_provenance']
                self.assertEqual(provenance['old_original_windows'], target['source_refs']['import_provenance']['old_original_windows'])
                self.assertEqual(provenance['old_normalized_final_windows'], target['source_refs']['import_provenance']['old_normalized_final_windows'])
                self.assertEqual(provenance['current_curated_batch'], target['source_refs']['import_provenance']['current_curated_batch'])
                self.assertEqual(provenance['old_generation_receipts'], target['source_refs']['import_provenance']['old_generation_receipts'])
                self.assertEqual(candidate['population_status'], target['population_status'])
                self.assertEqual(candidate['label_status'], target['label_status'])
                self.assertEqual(len(provenance['old_generation_receipts']), len(target['source_refs']['import_provenance']['old_generation_receipts']))
                for receipt in provenance['old_generation_receipts']:
                    self.assertEqual(set(receipt['artifacts']), imp.ARTIFACTS)
                self.assertEqual(candidate['style_result'], target['style_result'])
                self.assertEqual(candidate['original_curated_batch_id'], target['original_curated_batch_id'])
                self.assertNotEqual(candidate['batch_id'], candidate['original_curated_batch_id'])
                self.assertTrue(provenance['current_media_content_SHA_matches_old_MS'])
                self.assertEqual(provenance['full_content_media_identity_verified_sources'], 1)
                for flag, value in imp.FLAGS.items():
                    self.assertIs(candidate[flag], value)
        self.assertIn('hand_roi_gpu_fixed3_v3', self.candidates[SIDS[0]]['hand_roi']['path'])
        self.assertEqual(sum('hand_roi_full12000_v1' in c['hand_roi']['path'] for c in self.candidates.values()), 1)
        self.assertEqual(sum('hand_roi_gpu_fixed3_v3' in c['hand_roi']['path'] for c in self.candidates.values()), 2)

    def test_02_wrong_media_content_full_sha_rejected(self):
        video = self.root / 'wrong.mp4'
        original = Path(self.candidates[SIDS[0]]['video_path']).read_bytes()
        video.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        expected = self.candidates[SIDS[0]]['source_binding']['source_binding']['video_sha256']
        with self.assertRaisesRegex(ValueError, 'content SHA mismatch'):
            self.importer.strong(video, expected)
        row = copy.deepcopy(self.importer.current_row(SIDS[0])[0])
        row['video_path'] = str(video)
        with self.assertRaisesRegex(ValueError, 'current media path differs'):
            self.evidence(self.results[SIDS[0]], row)

    def test_03_wrong_roi_root_rejected(self):
        result = copy.deepcopy(self.results[SIDS[0]])
        result['hand_roi']['path'] = str(imp.BAR / 'hand_roi_full12000_v1' / SIDS[0] / 'roi_manifest.json')
        with self.assertRaisesRegex(ValueError, 'wrong original ROI reference'):
            self.evidence(result)

    def test_04_raw_fact_mutation_rejected_after_refilter(self):
        result = copy.deepcopy(self.results[SIDS[0]])
        result['facts'][0]['raw_text'] += ' altered motion'
        with self.assertRaisesRegex(ValueError, 're-filtered facts differ'):
            self.evidence(result)

    def test_05_self_consistent_raw_receipt_mutation_not_accepted(self):
        result = self.results[SIDS[0]]
        observation = result['window_observations'][0]
        source = Path(observation['attempt_path'])
        target = self.root / 'changed-raw-phase'
        target.mkdir()
        for name in imp.ARTIFACTS | {'receipt.json'}:
            shutil.copyfile(source / name, target / name)
        parsed = self.importer.bar.parse_response(self.importer.bar.batch.json_bytes((source / 'response.raw.txt').read_text()))
        parsed['events'][0]['description'] = 'The dancer moves both arms outward.'
        (target / 'response.raw.txt').write_bytes(json.dumps(parsed).encode())
        generation = imp.load(target / 'generation.json')
        generation['raw_response_sha256'] = self.importer.files.sha(target / 'response.raw.txt')
        (target / 'generation.json').write_bytes(imp.raw(generation))
        receipt = imp.load(target / 'receipt.json')
        for name in ('response.raw.txt', 'generation.json'):
            receipt['files_sha256'][name] = self.importer.files.sha(target / name)
        (target / 'receipt.json').write_bytes(imp.raw(receipt))
        prepared = imp.load(Path(self.targets[SIDS[0]]['source_hint']) / 'prepared.json')
        manifest = prepared['windows'][0]
        with self.assertRaisesRegex(ValueError, 'raw/filter/final differs'):
            self.importer.phase('window', target, receipt['binding'],
                lambda v: self.importer.bar.filter_supported_observation(v, manifest,
                    {p: set(v) for p, v in prepared['forbidden'].items()}))

    def test_06_missing_window_rejected(self):
        result = copy.deepcopy(self.results[SIDS[0]])
        result['window_observations'].pop()
        self.assertFalse(self.importer.structural(result))
        with self.assertRaisesRegex(ValueError, 'missing original window'):
            self.evidence(result)
        result = copy.deepcopy(self.results[SIDS[0]])
        result['final']['json']['windows'][0]['caption'] = ''
        self.assertFalse(self.importer.structural(result))

    def test_07_historical_style_sha_upgrade_rejected(self):
        candidate = copy.deepcopy(self.candidates[SIDS[0]])
        meta = candidate['style_result']['metadata']
        meta['historical_video_sha256'] = meta['current_video_sha256']
        meta['historical_video_binding_verified'] = True
        meta['historical_video_binding'] = 'sha256_verified'
        with self.assertRaisesRegex(ValueError, 'must remain weak'):
            imp.validate_candidate_payload(candidate)
        candidate = copy.deepcopy(self.candidates[SIDS[0]])
        candidate['source_refs']['import_provenance']['historical_video_sha256'] = candidate['style_result']['metadata']['current_video_sha256']
        with self.assertRaisesRegex(ValueError, 'must remain weak'):
            imp.validate_candidate_payload(candidate)

    def test_08_same_source_idempotence_and_changed_bytes_refused(self):
        path = self.root / 'immutable' / (SIDS[0] + '.json')
        candidate = self.candidates[SIDS[0]]
        self.assertTrue(imp.immutable(path, candidate))
        self.assertFalse(imp.immutable(path, copy.deepcopy(candidate)))
        before = path.read_bytes()
        changed = copy.deepcopy(candidate)
        changed['source_hint'] += '/wrong'
        with self.assertRaisesRegex(ValueError, 'different candidate bytes refused'):
            imp.immutable(path, changed)
        self.assertEqual(before, path.read_bytes())

    def test_09_shared_full_sha_cache_rejects_stat_change(self):
        path = self.root / 'shared-cache'
        path.write_bytes(b'first')
        files = imp.StableFiles()
        self.assertEqual(files.sha(path, True), files.sha(path, True))
        path.write_bytes(b'changed-size')
        with self.assertRaisesRegex(ValueError, 'shared input changed'):
            files.sha(path, True)

    def test_10_cli_real_single_pass_export_receipt_and_resume(self):
        out = self.root / 'cli-real'
        args = ['--ids', ','.join(SIDS), '--limit', '3', '--out', str(out), '--once', '--cohort', 'oldbar3-import-v1']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(imp.main(args), 0)
        exported = [json.loads(line) for line in (out / 'candidates.jsonl').read_text().splitlines()]
        self.assertEqual(len(exported), 3)
        for candidate in exported:
            self.assertEqual(candidate['target_text'], self.targets[candidate['id']]['target_text'])
        before = (out / 'candidates.jsonl').read_bytes()
        receipt = imp.load(imp.load(out / 'latest.receipt.json')['path'])
        self.assertEqual(receipt['verified_this_pass_count'], 3)
        self.assertEqual(receipt['status_counts'], {'imported': 3})
        self.assertEqual(receipt['candidate_export_ref']['sha256'], imp.bytes_sha(before))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(imp.main(args), 0)
        self.assertEqual(before, (out / 'candidates.jsonl').read_bytes())
        receipt = imp.load(imp.load(out / 'latest.receipt.json')['path'])
        self.assertEqual(receipt['status_counts'], {'already_imported': 3})
        self.assertEqual(receipt['verified_this_pass_count'], 3)
        original_files = {p.name: p.read_bytes() for p in (out / 'candidates').glob('*.json')}
        changed = args[:-1] + ['other-import-v1']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(imp.main(changed), 1)
        receipt = imp.load(imp.load(out / 'latest.receipt.json')['path'])
        self.assertEqual(receipt['status_counts'], {'excluded': 3})
        self.assertEqual(receipt['verified_this_pass_count'], 0)
        self.assertEqual((out / 'candidates.jsonl').read_bytes(), b'')
        self.assertEqual(original_files, {p.name: p.read_bytes() for p in (out / 'candidates').glob('*.json')})

    def test_11_per_source_pending_and_excluded_do_not_abort(self):
        out = self.root / 'per-source-errors'
        original = imp.Importer.import_one
        def controlled(importer, sid):
            if sid == SIDS[0]:
                raise imp.Pending('deliberately_missing_evidence')
            if sid == SIDS[1]:
                raise ValueError('deliberately_wrong_evidence')
            return original(importer, sid)
        with patch.object(imp.Importer, 'import_one', controlled), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(imp.main(['--ids', ','.join(SIDS), '--out', str(out)]), 1)
        receipt = imp.load(imp.load(out / 'latest.receipt.json')['path'])
        self.assertEqual(receipt['status_counts'], {'pending': 1, 'imported': 1, 'excluded': 1})
        self.assertEqual(len(receipt['dispositions']), 3)
        self.assertEqual(receipt['verified_this_pass_count'], 1)


    def test_12_independent_reviewer_real_source_interoperability(self):
        path = self.root / 'independent-interop.jsonl'
        path.write_bytes(b''.join(imp.raw(self.candidates[sid]) for sid in SIDS))
        program = chr(10).join([
            'import sys,json',
            'sys.path.insert(0, sys.argv[1])',
            'import oldbar_explicit_review_v1 as review',
            'rows=[json.loads(s) for s in open(sys.argv[2]).read().splitlines()]',
            'refs=review.References()',
            'for row in rows:',
            ' review.validate_claim(row)',
            ' probe=review.validate_source(row,refs,ffprobe="/opt/ms_tools/ffprobe")',
            ' assert len(probe["times"]) == row["source_binding"]["source_binding"]["decoded_video_frames"]',
            ' print("INDEPENDENT_INTEROP_OK",row["id"])',
            'print("INDEPENDENT_INTEROP_PASS",len(rows),"GPU_MODEL_CALLS",0)',
        ])
        env = dict(os.environ, MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages')
        result = subprocess.run(['/opt/venv_eval/bin/python', '-B', '-c', program, str(imp.RT), str(path)],
                                env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('INDEPENDENT_INTEROP_OK'), 3)
        self.assertIn('INDEPENDENT_INTEROP_PASS 3 GPU_MODEL_CALLS 0', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
