"""Directed CPU tests only; synthetic tests are not real VLM reviews."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


w = module('clause_pilot5_test_worker', RT / 'curated_clause_salvage_review_pilot5_v1.py')
builder = module('clause_pilot5_test_builder', RT / 'curated_clause_salvage_review_pilot5_build_v1.py')
manager = module('clause_pilot5_test_manager', RT / 'curated_clause_salvage_review_pilot5_resource_manager_v1.py')


class Pilot5Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bound_case(self):
        root = self.root
        original = dict(id='realcase', split='train', batch_id='train-b000000',
                        source_ordinal=0, video_path='/original/video', audio_path='/original/audio')
        def artifact(path, value, addressed=False):
            raw = json.dumps(value, ensure_ascii=False).encode()
            if addressed:
                import hashlib
                path = path / (hashlib.sha256(raw).hexdigest() + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            return w.ref(path)
        main = dict(**original, joint_complete=False, label=dict(dance_style='unknown', timed_motion_script='[0-2] Old.'),
                    source_refs={}, style_result=dict(status='unknown'))
        main_ref = artifact(root / 'curated222939_production_v1/sources/train/realcase/joint-results', main, True)
        repair_ref = artifact(root / 'curated222939_repair_v1/revisions/repair-20261025-v1/train/realcase/result.json', dict(facts=[], final=dict(status='failed'), window_observations=[dict(status='candidate')]))
        repair = dict(**original, original_result_ref=main_ref, repair_result_ref=repair_ref)
        oldrepair_ref = artifact(root / 'curated222939_repair_v1/joint/revisions/repair-20261025-v1/train/realcase/result.json', repair)
        visual = dict(window_observations=[dict(status='candidate')], final_timed_caption='[0-2] Both arms rise.',
            facts=[dict(window_id='w0', fact_id='f0', raw_text='Both arms rise.', parts=['left_arm', 'right_arm'], frame_indices=[56, 80])],
            final=dict(derivation='verbatim_admitted_fact_concatenation', json=dict(windows=[
                dict(window_id='w0', caption='Both arms rise.', used_fact_ids=['f0'], used_boundary_ids=[])])))
        visual_ref = artifact(root / 'candidates/revisions' / w.EDITION / 'train/realcase/visual_result.json', visual)
        other_ref = artifact(root / 'lineage.json', {})
        candidate = dict(**original, global_revision=w.EDITION, risk_review_case=True, style_label_changed=False,
            model_calls=0, attempt_budget_consumed=False, joint_complete=False, source_refs={},
            label=dict(dance_style='unknown', timed_motion_script=visual['final_timed_caption']),
            style_result=main['style_result'], original_repair_joint_result_ref=oldrepair_ref,
            original_repair_result_ref=repair_ref, visual_result_ref=visual_ref,
            original_visual_result_ref=other_ref, original_worker_terminal_ref=other_ref, original_job_ref=other_ref)
        candidate.update(need_real_review=True, review_reason='new_safe_coarse_body_facts',
            pure_rewrite_failure=False, new_safe_facts=visual['facts'], new_safe_fact_ids=['f0'])
        new_ref = artifact(root / 'candidates/revisions' / w.EDITION / 'train/realcase/result.json', candidate)
        notice = dict(id='realcase', source_key='train/realcase', split='train', batch_id='train-b000000',
            global_revision=w.EDITION, risk_review_case=True, result_ref=new_ref, joint_result_ref=new_ref,
            joint_candidate_ref=new_ref, visual_result_ref=visual_ref, need_real_review=True,
            review_reason='new_safe_coarse_body_facts', pure_rewrite_failure=False,
            new_safe_facts=visual['facts'], new_safe_fact_ids=['f0'],
            new_safe_fact_refs=[dict(visual_result_ref=visual_ref, fact_id='f0')])
        manifest = SimpleNamespace(members=lambda bid: {'realcase': original},
                                   descriptors={'train-b000000': dict(count=500)})
        return notice, candidate, manifest

    def test_bind_real_reference_chain_and_unknown_keeps_partial(self):
        notice, candidate, manifest = self.bound_case()
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'CANDIDATES', self.root / 'candidates'):
            row = w.bind_notice(notice, manifest, w.support())
        self.assertEqual(row['label_status'], 'partial')
        self.assertFalse(review.label_complete(row))
        self.assertEqual(row['review_provenance']['original_batch_source_count'], 500)
        self.assertFalse(row['review_provenance']['original_review_scores_reused'])
        self.assertEqual(row['review_provenance']['label_result_ref'], notice['joint_result_ref'])

    def test_bind_style_or_verbatim_text_change_refused(self):
        for field in ('dance_style', 'timed_motion_script'):
            notice, candidate, manifest = self.bound_case()
            candidate['label'][field] += ' changed'
            p = Path(notice['result_ref']['path'])
            p.write_text(json.dumps(candidate))
            changed = w.ref(p)
            notice.update(result_ref=changed, joint_result_ref=changed, joint_candidate_ref=changed)
            with patch.object(w, 'ROOT', self.root), patch.object(w, 'CANDIDATES', self.root / 'candidates'):
                with self.assertRaisesRegex(ValueError, 'style changed|verbatim'):
                    w.bind_notice(notice, manifest, w.support())

    def test_bound_candidate_file_SHA_change_refused(self):
        notice, candidate, manifest = self.bound_case()
        Path(notice['result_ref']['path']).write_text('{}')
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'CANDIDATES', self.root / 'candidates'):
            with self.assertRaisesRegex(ValueError, 'SHA'):
                w.bind_notice(notice, manifest, w.support())

    def test_delta_only_added_coarsebody_or_pure_rewrite_requires_review(self):
        fact = dict(raw_text='Both arms move outward.', parts=['left_arm', 'right_arm'], frame_indices=[56, 80])
        visual = dict(window_observations=[dict(status='candidate')], facts=[fact], final_timed_caption='[0-2] Both arms move outward.')
        old = dict(facts=[], final=dict(status='failed'), window_observations=[dict(status='candidate')])
        joint = dict(label=dict(timed_motion_script=''))
        self.assertTrue(w.review_delta(visual, old, joint)['review_required'])
        unchanged = dict(facts=[fact], final=dict(status='candidate'), window_observations=[dict(status='candidate')])
        self.assertFalse(w.review_delta(visual, unchanged, joint)['review_required'])
        unchanged['final']['status'] = 'failed'
        self.assertEqual(w.review_delta(visual, unchanged, joint)['reason'], 'pure_rewrite_failure_original_admitted_facts')
        visual['facts'][0]['raw_text'] = 'The hands form fists.'
        with self.assertRaisesRegex(ValueError, 'hand shapes'):
            w.review_delta(visual, old, joint)

    def test_single_partial_changed_case_not_force_five_calls(self):
        notice, candidate, manifest = self.bound_case()
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'CANDIDATES', self.root / 'candidates'):
            row = w.bind_notice(notice, manifest, w.support())
        selected = self.root / 'selected-one.json'
        selected.write_text(json.dumps(dict(candidate_source_count=2, cases=[
            dict(review_required=True), dict(review_required=False, source_key='train/unchanged',
                notice=dict(id='unchanged', batch_id='train-b000000'), label_result_ref=dict(path='/unchanged',sha256='test'))])))
        calls = []
        def fake(r, root, **kwargs):
            calls.append(r['id'])
            return dict(source_id=r['id'], status='pass', errors=[], uncertainties=[])
        result = w.review_records([row], self.root / 'one-out', fake, review, w.ref(selected))
        self.assertEqual(calls, ['realcase'])
        self.assertEqual(result['selected_source_count'], 1)
        self.assertEqual(result['candidate_source_count'], 2)
        self.assertEqual(result['no_new_supported_facts_count'], 1)
        self.assertEqual(result['unreviewed_candidates'][0]['status'], 'no_new_supported_facts')
        self.assertEqual(result['unreviewed_candidates'][0]['independent_review_status'], 'pending')
        self.assertFalse(result['training_eligible'])
        self.assertEqual(result['label_incomplete_source_ids'], ['realcase'])

    def test_producer_eligibility_mismatch_refused(self):
        notice, candidate, manifest = self.bound_case()
        notice['need_real_review'] = False
        with patch.object(w, 'ROOT', self.root), patch.object(w, 'CANDIDATES', self.root / 'candidates'):
            with self.assertRaisesRegex(ValueError, 'eligibility'):
                w.bind_notice(notice, manifest, w.support())

    def test_Gh_old_failed_window_to_empty_abstain_not_pure_rewrite(self):
        fact = dict(raw_text='An existing body motion.', parts=['torso'], frame_indices=[1, 2])
        old = dict(facts=[fact], final=dict(status='failed'), window_observations=[dict(status='candidate'), dict(status='failed')])
        visual = dict(facts=[fact], final_timed_caption='[0-2] An existing body motion.\n[2-4] ',
                      window_observations=[dict(status='candidate'), dict(status='abstain')])
        delta = w.review_delta(visual, old, dict(label=dict(timed_motion_script='')))
        self.assertFalse(delta['review_required'])
        self.assertFalse(delta['pure_rewrite_failure'])
        self.assertEqual(delta['reason'], 'no_new_supported_facts')

    def test_uq_old_prepared_or_failed_window_never_pure_rewrite(self):
        fact = dict(raw_text='An existing arm motion.', parts=['left_arm'], frame_indices=[1, 2])
        for status in ('failed', 'prepared'):
            old = dict(facts=[fact], final=dict(status='failed'), window_observations=[dict(status=status)])
            visual = dict(facts=[fact], final_timed_caption='[0-2] An existing arm motion.',
                          window_observations=[dict(status='abstain')])
            self.assertFalse(w.review_delta(visual, old, dict(label=dict(timed_motion_script='')))['review_required'])

    def test_real_manifest_shape_unchanged(self):
        self.assertEqual(w.sha(w.MANIFEST), w.MANIFEST_SHA)
        m = w.support().Manifest(w.MANIFEST)
        self.assertEqual(m.summary['total'], 222939)
        self.assertEqual(len(m.summary['batches']), 448)
        self.assertEqual(sum(b['count'] for b in m.summary['batches']), 222939)

    def test_manager_rejects_G2_G4_and_unapproved_G3(self):
        for g, authorized in ((2, True), (4, True), (3, False)):
            with self.assertRaisesRegex(RuntimeError, 'GPU3'):
                manager.validate(dict(node='N6', gpus=[g], release_authorized=authorized))

    def test_fresh_capture_unknown_active_and_active_lease_no_signals(self):
        base = SimpleNamespace(lease_snapshot=lambda p: None, uuid_for=lambda g: 'uuid3',
            apps=lambda: [dict(uuid='uuid3', pid=12)], identity=lambda p: dict(pid=p),
            check_holder=lambda *a: (_ for _ in ()).throw(RuntimeError('unknown active; not project holder')))
        m = SimpleNamespace(base=base, worker=w)
        with patch.object(builder.subprocess, 'check_output', return_value='28.216.19.225'),              patch.object(builder.Path, 'exists', return_value=False),              patch.object(builder.Path, 'is_symlink', return_value=False):
            with self.assertRaisesRegex(w.NotReady, 'unknown active'):
                builder.fresh_capture(m)
            base.lease_snapshot = lambda p: (_ for _ in ()).throw(RuntimeError('active lease'))
            with self.assertRaisesRegex(w.NotReady, 'active lease'):
                builder.fresh_capture(m)

    def test_partial_unknown_audited_but_never_admitted(self):
        records = []
        for i in range(5):
            records.append(dict(id=f'case{i}', source_id=f'case{i}', split='train', batch_id='train-b000000',
                video_path='/original/video', audio_path='/original/audio',
                label_path=f'/new-clause/label-{i}.json', label_sha256=f'newsha{i}',
                label=dict(dance_style='unknown' if i == 1 else 'hiphop',
                           timed_motion_script='[0-2] Both arms rise.'),
                joint_complete=i != 1, label_status='partial' if i == 1 else 'completed',
                style_status='unknown' if i == 1 else 'candidate', visual_status='candidate', source_refs={},
                review_provenance=dict(label_edition=w.EDITION, original_batch_source_count=500,
                    label_result_ref=dict(path=f'/new-clause/label-{i}.json', sha256=f'newsha{i}'),
                    original_joint_result_ref=None, original_roi_root='/original/roi',
                    review_delta=dict(review_required=True, reason='new_safe_coarse_body_facts'))))
        calls = []
        def fake(row, root, **kwargs):
            calls.append(row['id'])
            return dict(source_id=row['id'], status='pass', errors=[], uncertainties=[],
                label_sha256=review.batch.digest(review.label_payload(row)))
        selected = self.root / 'selected.json'
        selected.write_text(json.dumps(dict(candidate_source_count=5, cases=[dict(review_required=True) for _ in records])))
        result = w.review_records(records, self.root / 'out', fake, review, w.ref(selected))
        self.assertEqual(calls, [f'case{i}' for i in range(5)])
        self.assertEqual(result['sampled_sources_reviewed'], 5)
        self.assertEqual(result['label_incomplete_source_ids'], ['case1'])
        self.assertFalse(result['training_eligible'])
        self.assertFalse(result['all_batches_reviewed'])
        self.assertTrue(result['fixed_pilot_case_not_population_sample'])
        self.assertFalse(result['statistical_population_error_estimate'])
        for r in result['review_record_refs']:
            ledger = json.loads(Path(r['path']).read_text())
            self.assertEqual(ledger['original_batch_id'], 'train-b000000')
            self.assertFalse(ledger['training_eligible'])
            self.assertFalse(ledger['review_provenance'].get('original_review_scores_reused', False))

    def test_newSHA_changes_review_revision_not_oldscore(self):
        old = dict(label=dict(dance_style='hiphop', timed_motion_script='[0-2] Both arms rise.'),
                   label_path='/repair/result.json', input_row_sha256='old')
        new = copy.deepcopy(old)
        new.update(label_path='/clause-salvage/result.json', input_row_sha256='new')
        self.assertNotEqual(review.batch.digest(old), review.batch.digest(new))

    def test_missing_main_candidates_no_placeholder_config(self):
        with patch.object(builder, 'SOURCE', self.root / 'source.json'),              patch.object(builder, 'SELECTED', self.root / 'selection.json'),              patch.object(w, 'CANDIDATES', self.root / 'no_candidates'),              patch.object(builder, 'module', return_value=SimpleNamespace(worker=w)):
            with self.assertRaisesRegex(w.NotReady, 'No real main'):
                builder.freeze(SimpleNamespace(source_key=None))
            self.assertFalse((self.root / 'source.json').exists())
            self.assertFalse((self.root / 'selection.json').exists())

    def test_unknown_style_not_label_complete(self):
        row = dict(label=dict(dance_style='unknown', timed_motion_script='[0-2] Both arms rise.'),
                   joint_complete=False, label_status='partial')
        self.assertFalse(review.label_complete(row))
        self.assertTrue(review.label_reviewable(row))


if __name__ == '__main__':
    unittest.main()
