"""Fixed-three complete filter-recovery 8B audit. CPU prepare owns no GPU.

Run is a Python caller-owned entry: a verified live-admission callback is
mandatory. No signals, resource manager, production terminals or training writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
OUT = BAR / 'coarse3_filter_recovery_independent_reviews_v1'
SCHEMA = 'coarse3_filter_recovery_independent_review_v1'
RECOVERY_REFS = [
    {'path': str(RT / 'curated_coarse_empty_filter_recovery_v1.py'),
     'sha256': 'f9085da43857bc3f05204dd1557674d0d7f645a740f7e260420748dca8acd957'},
    {'path': str(RT / 'test_curated_coarse_empty_filter_recovery_v1.py'),
     'sha256': 'b24f318ff8bf198f58a6d67fda4e5719e3757350ada2fc48eebcb5c8250c3df2'}]
IDS = ('AuuitdcBJyGOnWaH', 'AzPomlvsANsevIFb', 'DxNCjtNsxcazNpGb')
LAUNCH_SHA = '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806'
OLD_SHA = 'f5507fe6b42f9d5bda22346d369bebf76c1455d14dea1234433ef16869cd23ef'
LIB_SHA = '98cdb5cbc1089ccbbbdfc71de572dbe08787796e2ae13cc247369f07c9821dba'
TEACHER = BAR.parent / '32b_replica_stage_receipts/source_reference_v1.json'
SIX = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
NO_APPROVAL = {k: False for k in ('training_eligible', 'eligible_for_training', 'training_approved',
    'semantic_approval', 'human_review_completed', 'human_approved', 'formal_approval', 'formal_training',
    'training_entry_updated', 'original_curated_batch_completed', 'original_curated_batches_completed',
    'original_population_quality_approved', 'full_batch_semantic_pass_claimed')}
_LIBS = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def library_modules():
    return {k: v for k, v in sys.modules.items() if k.startswith(('ms_', 'motionscript_v2'))}


class Libraries:
    def __init__(self):
        self.c = module('coarse3_source_contract', RT / 'prepare_curated_coarse_empty_n5_v1.py')
        self.worker = self.c.worker_module()
        self.q = self.worker.q
        verify_recovery_freeze(self)
        self.recovery = module('coarse3_frozen_filter_recovery', RT / 'curated_coarse_empty_filter_recovery_v1.py')
        need(self.c.sha(RT / 'oldbar_explicit_review_v1.py') == OLD_SHA, 'Frozen oldbar reviewer changed')
        lib_ref = {'path': str(RT / 'source_curated_review_v2/freeze.receipt.json'), 'sha256': LIB_SHA}
        freeze = self.c.checked(lib_ref)
        for path, digest in freeze['files'].items():
            need(Path(path).resolve().is_relative_to(RT / 'source_curated_review_v2')
                 and self.c.sha(path) == digest, 'Frozen independent review library changed')
        previous, paths = library_modules(), list(sys.path)
        for name in previous:
            sys.modules.pop(name, None)
        try:
            self.old = module('coarse3_real_independent_review', RT / 'oldbar_explicit_review_v1.py')
            self.review_modules, self.review_path = library_modules(), list(sys.path)
        finally:
            for name in library_modules():
                sys.modules.pop(name, None)
            sys.modules.update(previous)
            sys.path[:] = paths
        self.audit, self.batch = self.old.audit, self.old.batch

    @contextmanager
    def review_library(self):
        previous, paths = library_modules(), list(sys.path)
        for name in previous:
            sys.modules.pop(name, None)
        sys.modules.update(self.review_modules)
        sys.path[:] = self.review_path
        try:
            yield
        finally:
            self.review_modules.update(library_modules())
            for name in library_modules():
                sys.modules.pop(name, None)
            sys.modules.update(previous)
            sys.path[:] = paths


def libraries():
    global _LIBS
    if _LIBS is None:
        _LIBS = Libraries()
    return _LIBS


def check_output(out, lib):
    out = Path(out).resolve()
    need(out == OUT or out.is_relative_to(Path('/tmp')), 'Only isolated default output or /tmp allowed')
    protected = (RT, lib.worker.OUT, lib.recovery.OUT, BAR / 'coarse3_independent_reviews_v1',
                 lib.worker.PRIMARY, BAR / 'curated222939_production_v1')
    need(all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out) for p in protected),
         'Audit output overlaps frozen input/production/runtime')
    return out


def source_contract(lib):
    c, worker = lib.c, lib.worker
    verify_recovery_freeze(lib)
    launch_ref = {'path': str(c.LAUNCH), 'sha256': LAUNCH_SHA}
    cfg = c.checked(launch_ref)
    # These are historical source/config gates only: never validate_pair,
    # verify_actual_parent, check_node, lease/holder/app checks or require_unreserved.
    contract = c.frozen(worker, probe=False)
    c.verify_config(cfg, contract)
    need(lib.q.load(worker.OUT / 'source.freeze.json') == worker.source_freeze(), 'Frozen worker/test/protocol changed')
    need([j['id'] for j in contract['jobs']] == list(IDS), 'Exactly three frozen source jobs required')
    return cfg, contract, launch_ref


def receipt_replay(lib, attempt, request, binding, model_sha, manifest, forbidden, *, coarse, allow_failed=False):
    q, bar = lib.q, lib.worker.bar
    attempt = Path(attempt).resolve()
    receipt = q.load(attempt / 'receipt.json')
    failed = allow_failed and receipt.get('status') == 'failed'
    names = SIX - {'response.json'} if failed else SIX
    need((failed or receipt.get('status') == 'candidate') and set(receipt.get('files_sha256', {})) == names,
         'Source requires actual six-file candidate or preserved five-file failed receipt')
    need(not failed or not (attempt / 'response.json').exists(), 'Failed source cannot acquire a fabricated response.json')
    need(receipt.get('semantic_approval') is False and receipt.get('human_review_completed') is False,
         'Source transaction cannot claim semantic/human approval')
    for name, digest in receipt['files_sha256'].items():
        q.verify_bytes({'path': str(attempt / name), 'sha256': digest})
    need(q.load(attempt / 'request.json') == request, 'Actual source request differs from native frozen reconstruction')
    need(q.load(attempt / 'binding.json') == receipt['binding'] == binding, 'Actual source/ROI/frame/job receipt binding changed')
    generation = bar.core.validate_generation(attempt, request)
    need(generation.get('model_call_performed') is True and generation['model_identity_sha256'] == model_sha,
         'Actual annotator checkpoint generation binding changed')
    final = q.load(attempt / 'final.json')
    need(all(final.get(k) is False for k in bar.core.FLAGS), 'Source final cannot claim approval')
    try:
        raw = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
        response = lib.worker.validate_coarse(raw, manifest, forbidden) if coarse else bar.filter_supported_observation(raw, manifest, forbidden)
    except (ValueError, KeyError, TypeError) as error:
        need(failed, 'Candidate native raw/schema/filter replay failed: ' + str(error))
        message = type(error).__name__ + ': ' + str(error)
        need(final == {'status': 'failed', 'error': message, 'format_valid': False, **bar.core.FLAGS}
             and receipt.get('error') == message, 'Failed raw/schema/filter must reproduce the exact preserved native error/final')
    else:
        need(not failed, 'Failed receipt raw unexpectedly validates; preserve pending, never reinterpret as candidate')
        need(final.get('status') == 'candidate' and final.get('json') == response == q.load(attempt / 'response.json'),
             'Raw/schema/filter/response/final replay differs')
    refs = [q.identity(attempt / 'receipt.json')] + [q.identity(attempt / n) for n in sorted(names)]
    return final, refs


def verify_membership(lib, job, plan):
    q = lib.q
    ref = plan['source_refs']['manifest']
    q.verify_bytes(ref)
    members = [lib.worker.bar.core.strict_json(line) for line in Path(ref['path']).read_bytes().splitlines() if line.strip()]
    keys = [(r['id'], r['split']) for r in members]
    need(len(members) in (500, 448) and len(set(keys)) == len(keys)
         and all(r['batch_id'] == job['batch_id'] and r['split'] == job['split'] for r in members),
         'Original full 500/448 member denominator required, never pilot3 as original batch')
    found = [r for r in members if r['id'] == job['id']]
    need(found == [plan['manifest_record']], 'Original full-cohort source row differs from plan')
    ref_csv = plan['source_refs']['original_csv']
    q.verify_original_csv(ref_csv, job['split'])
    with Path(ref_csv['path']).open('r', newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = ref_csv.get('csv_line', ref_csv.get('csv_record_number', 0) + 1)
        values = next((v for n, v in enumerate(reader, 2) if n == index), None)
    need(values is not None, 'Original CSV referenced row missing')
    csv_ref = found[0]['source_refs']['csv']
    digest = lib.worker.pipeline.digest({'header': header, 'values': values})
    need(digest == ref_csv['raw_row_sha256'] == csv_ref['raw_row_sha256'], 'Original CSV row digest differs')
    return {'batch_id': job['batch_id'], 'split': job['split'], 'source_id': job['id'],
            'original_member_count': len(members), 'membership_ref': ref,
            'membership_ids_sha256': q.digest(keys), 'original_csv_ref': ref_csv}


def verify_recovery_freeze(lib):
    for ref in RECOVERY_REFS:
        lib.q.verify_bytes(ref)


def recovery_bundle(lib, hint, visual, joint, proof):
    q, recovery = lib.q, lib.recovery
    root = recovery.OUT / 'revisions' / recovery.REVISION / hint['source_key']
    result_ref, visual_ref = q.identity(root / 'result.json'), q.identity(root / 'visual_result.json')
    proof_ref = q.identity(root / 'derivation.receipt.json')
    candidate = q.checked(result_ref)
    expected = {**joint, 'visual_result_ref': visual_ref}
    need(q.checked(visual_ref) == visual and candidate == expected and q.checked(proof_ref) == proof,
         'Full frozen recovery visual/joint/proof replay differs from current bytes')
    need(proof.get('recovery_code_ref') == RECOVERY_REFS[0]
         and proof.get('schema') == recovery.SCHEMA and proof.get('global_revision') == recovery.REVISION
         and proof.get('source_key') == hint['source_key'] and proof.get('original_job_ref') == hint['job_ref']
         and proof.get('original_observation_model_calls') == 1
         and type(proof.get('original_observation_model_calls')) is int
         and proof.get('new_visual_requests') == 0 and proof.get('new_text_model_requests') == 0
         and proof.get('extra_source_attempts') == 0 and proof.get('canonical_budget_unchanged') is True
         and proof.get('new_label_review_required') is True,
         'Frozen recovery code/source/original-one-call/zero-new-model/budget proof changed')
    need(candidate.get('status') == 'filter_recovery_joint_candidate' and candidate.get('joint_complete') is True
         and candidate.get('all_windows_nonempty') is True
         and all(candidate.get(k) is False and visual.get(k) is False and proof.get(k) is False for k in lib.q.FLAGS)
         and candidate.get('training_eligible') is False, 'Only complete unapproved recovery candidates are reviewable')
    marker_ref = q.identity(recovery.OUT / 'review-required' / (hint['source_key'] + '.json'))
    marker = q.checked(marker_ref)
    need(marker.get('source_key') == hint['source_key'] and marker.get('status') == candidate['status']
         and marker.get('joint_complete') is True and marker.get('result_ref') == result_ref
         and marker.get('new_model_calls') == 0 and marker.get('new_label_review_required') is True
         and all(marker.get(k) is False for k in q.FLAGS), 'Actual recovered candidate review marker changed')
    return candidate, result_ref, visual_ref, proof_ref, marker_ref


def validate_candidate(lib, cfg, hint, *, ffprobe):
    q, worker, recovery = lib.q, lib.worker, lib.recovery
    verify_recovery_freeze(lib)
    job = q.checked(hint['job_ref'])
    need(job['id'] in IDS and job['source_key'] == 'train/' + job['id'] == hint['source_key'], 'Fixed original source identity changed')
    root = recovery.OUT / 'revisions' / recovery.REVISION / job['source_key']
    terminal_path = worker.OUT / 'worker-terminals' / (job['source_key'] + '.json')
    required = [terminal_path, root / 'result.json', root / 'visual_result.json', root / 'derivation.receipt.json',
                recovery.OUT / 'review-required' / (job['source_key'] + '.json')]
    if not all(p.exists() for p in required):
        return None, {'status': 'pending', 'reason': 'recovery_candidate_or_original_terminal_not_landed', 'source_id': job['id']}
    value, status = recovery.derive(worker, cfg, hint)
    if value is None:
        return None, {'status': 'pending', 'reason': status['status'], 'source_id': job['id']}
    visual, joint, proof = value
    candidate, result_ref, visual_ref, proof_ref, marker_ref = recovery_bundle(lib, hint, visual, joint, proof)
    original, old_joint, prepared, plan = worker.verify_job(job, probe=False)
    q.verify_original_csv(plan['source_refs']['original_csv'], job['split'])
    q.verify_references_by_content(plan['source_refs'])
    need(q.verify_prepared_content(job['visual_root'], plan, worker.bar, ffprobe) == prepared,
         'CPU media/RGB/ROI/actual PTS evidence differs')
    mapping = verify_membership(lib, job, plan)
    need(q.checked(job['refs']['style_result']) == old_joint['style_result'] == candidate['style_result'],
         'Original unchanged style reference differs')
    metadata = old_joint['style_result'].get('metadata', {})
    q.verify_references_by_content(metadata.get('source_refs', {}))
    need(metadata.get('video_sha256') == plan['binding']['video_sha256'], 'Original style video binding changed')
    budget_ref = worker.check_budget(job)
    need(budget_ref is not None and budget_ref == proof['primary_budget_ref']
         and q.checked(budget_ref) == {'source_key':job['source_key'], 'original_result_sha256':job['original_result_sha256'],
             'global_revision':worker.REVISION, 'extra_source_attempt':1, 'maximum_extra_source_attempts':1},
         'Original canonical one-attempt source/SHA/revision reservation changed')
    parent_root = worker.OUT / 'revisions' / worker.REVISION / job['source_key']
    need(q.load(parent_root / 'original_job.json') == job, 'Original persisted coarse job differs')
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    original_model = q.checked(job['refs']['model_identity_ref'])
    evidence_refs = list(copy.deepcopy(proof['original_transaction_refs']))
    for observation in original['window_observations']:
        wid = observation['window']['window_id']
        attempt = Path(observation['attempt_path']).resolve()
        need(attempt.parent == Path(job['visual_root']) / 'windows' / wid, 'Original attempt outside exact source/window')
        manifest = q.load(Path(job['visual_root']) / 'windows' / wid / 'manifest.json')
        need(manifest['window'] == observation['window'] and manifest['binding'] == prepared['binding'],
             'Original window/source binding changed')
        request = worker.bar.visual_request(worker.worker_args(), Path(job['visual_root']) / 'rgb', manifest, forbidden)
        final, refs = receipt_replay(lib, attempt, request, {**original['binding'], 'window':observation['window'],
            'frames_sha256':worker.bar.batch.digest(manifest['frames'])}, original_model['sha256'], manifest, forbidden, coarse=False)
        need(final['json'] == observation['json'] and observation['status'] == ('abstain' if final['json']['abstain'] else 'candidate'),
             'Original admitted/abstain observation differs from six-file replay')
        evidence_refs.extend(refs)
    for ref in evidence_refs:
        q.verify_bytes(ref)
    need(worker.check_budget(job) == budget_ref, 'Canonical budget SHA changed during independent source verification')
    verify_recovery_freeze(lib)
    label_kind = label_scope(lib.audit, candidate, visual)
    return candidate, {'status':'verified', 'label_scope':label_kind, 'candidate_ref':result_ref,
        'visual_result_ref':visual_ref, 'recovery_proof_ref':proof_ref, 'recovery_marker_ref':marker_ref,
        'recovery_code_ref':RECOVERY_REFS[0], 'recovery_test_ref':RECOVERY_REFS[1],
        'terminal_ref':q.identity(terminal_path), 'original_coarse_result_ref':proof['original_coarse_result_ref'],
        'original_observation_model_calls':1, 'recovery_new_model_calls':0, 'recovery_new_budget_reservations':0,
        'source_job_ref':hint['job_ref'], 'original_membership':mapping, 'probe':prepared['probe'], 'source_hint':job['visual_root'],
        'origin_root':str(Path(job['visual_root']).parent.parent), 'source_transaction_refs':evidence_refs,
        'canonical_budget_ref':budget_ref, 'actual_media_SHA_PTS_RGB_ROI_checked':True,
        'raw_schema_filter_and_complete_package_replayed':True}


def label_scope(audit, candidate, visual):
    label = candidate.get('label')
    need(isinstance(label, dict) and set(label) == {'dance_style', 'timed_motion_script'}
         and all(isinstance(v, str) and v.strip() for v in label.values())
         and label['dance_style'].strip().lower() not in ('unknown', 'uncertain', 'abstain')
         and audit.label_reviewable(candidate), 'Empty/unknown/unreviewable label stays pending, not pass')
    windows = visual.get('final', {}).get('json', {}).get('windows', [])
    need(windows and any(w.get('caption', '').strip() for w in windows), 'Time markers alone are not motion claims')
    need(candidate.get('joint_complete') is True and candidate.get('all_windows_nonempty') is True
         and all(w.get('caption', '').strip() for w in windows)
         and candidate.get('target_text') == 'Dance style: ' + label['dance_style'] + '\n' + label['timed_motion_script'],
         'Only complete recovered labels are reviewable; partial/empty source stays pending')
    return 'complete'


def review_row(lib, candidate, verified, population_ref):
    return {'id': candidate['id'], 'source_id': candidate['id'], 'split': candidate['split'],
        'video_path': candidate['video_path'], 'audio_path': candidate['audio_path'],
        'label': copy.deepcopy(candidate['label']), 'label_status': 'completed',
        'joint_complete': candidate['joint_complete'], 'all_windows_nonempty': candidate['all_windows_nonempty'],
        'source_hint': verified['source_hint'], 'origin_root': verified['origin_root'],
        'hand_roi': candidate['hand_roi'], 'annotator_model_sha256': candidate['annotator_model_sha256'],
        'sourceRefs': {'candidate': verified['candidate_ref'], 'terminal': verified['terminal_ref'],
                       'original_job': verified['source_job_ref'], 'population': population_ref,
                       'recovery_proof': verified['recovery_proof_ref'], 'recovery_code': verified['recovery_code_ref'],
                       'recovery_test': verified['recovery_test_ref']},
        'explicit_review_binding': {'schema': SCHEMA, 'label_scope': verified['label_scope'],
            'candidate_sha256': lib.batch.digest(candidate), 'label_sha256': lib.batch.digest(candidate['label']),
            'target_text_sha256': hashlib.sha256(candidate['target_text'].encode()).hexdigest(),
            'old_errors_are_not_prior': True, 'original_membership': verified['original_membership']}}


def verify_evidence(lib, evidence, row, root, probe):
    lib.old.verify_evidence(evidence, row, root, probe)
    need([f['index'] for f in evidence['frames']] == evidence['binding']['frame_indices']
         and len(evidence['frames']) == min(16, len(probe['times'])), 'Native RGB sample count/indices changed')
    roi = lib.q.load(row['hand_roi']['path'])
    need(all(c in roi['crops'] for c in evidence['hand_rois']), 'ROI geometry/source differs from original native inventory')
    need(evidence['image_budget'] == {'source_frames': len(evidence['frames']), 'native_hand_rois': len(evidence['hand_rois']),
         'max_source_frames': 16, 'max_ROIs': 6}, 'Native independent image budget changed')


def verified_admission(lib, callback, context):
    """Callback checks its own live locks/leases/model descendants, never old N5 history.

    It returns admission_verified=True, an unchanged binding=context,
    resource_admission_ref and current_lease_refs with verified actual file SHA.
    This library does not acquire, release, signal or invent resource ownership.
    """
    need(callable(callback), 'Run denied: caller-owned verified admission callback required')
    value = callback(copy.deepcopy(context))
    need(isinstance(value, dict) and value.get('admission_verified') is True and value.get('binding') == context,
         'Caller live admission must bind this PID/output/model/source/candidate/job manifest')
    need(value.get('resource_admission_ref') and value.get('current_lease_refs'), 'Caller current resource/lease evidence required')
    lib.q.verify_bytes(value['resource_admission_ref'])
    for ref in value['current_lease_refs']:
        lib.q.verify_bytes(ref)
    return value


def make_reviewer(lib, model, out, *, model_reference, annotator_model_reference=TEACHER, **kwargs):
    out = check_output(out, lib)
    class IndependentReviewer(lib.old.ManifestReviewer):
        def __init__(self):
            super().__init__(model, out, model_reference=model_reference,
                             annotator_model_reference=annotator_model_reference, **kwargs)
            paths = [Path(v['path']) for v in self.code['files'].values()] + [Path(__file__).resolve()]
            paths += [Path(ref['path']) for ref in RECOVERY_REFS]
            self.code = lib.batch.identity_tree(RT, paths)
            self.admission_callback, self.admission_context = None, None
            self._coarse3_caller_guard = True

        def identity(self):
            value = super().identity()
            need(Path(self.annotator_model_reference).resolve() == TEACHER, 'Exact teacher checkpoint content reference required')
            config = json.loads((Path(self.model) / 'config.json').read_bytes())
            need('8b' in config.get('_name_or_path', '').lower() or '8b' in Path(self.model).name.lower(),
                 'Scheduled independent reviewer must be 8B')
            return value

        def runtime(self):
            need(self.admission_context is not None, 'Run denied before model load without caller admission')
            verified_admission(lib, self.admission_callback, self.admission_context)
            value = super().runtime()
            verified_admission(lib, self.admission_callback, self.admission_context)
            return value
    with lib.review_library():
        return IndependentReviewer()


def review_transaction(lib, out, row, root, evidence, reviewer, callback, context):
    transaction = root / 'review'
    claim_path = out / 'review-attempts' / (row['split'] + '-' + row['id'] + '.json')
    claim = {'schema': SCHEMA, 'source_id': row['id'], 'input_row_sha256': lib.batch.digest(row),
             'root': str(root), 'maximum_review_calls_per_source': 1, 'maximum_total_review_calls': 3, **NO_APPROVAL}
    if (transaction / 'receipt.json').exists():
        need(lib.q.load(claim_path) == claim, 'Prior review must retain exact one-attempt source claim')
        need(lib.q.load(transaction / 'receipt.json').get('status') == 'candidate', 'Failed review remains pending; no retry')
        calls = 0
    else:
        need(not claim_path.exists() and (not transaction.exists() or not any(transaction.iterdir())),
             'Partial/previous review remains pending; never erase or redispatch')
        reviewer.admission_callback, reviewer.admission_context = callback, context
        verified_admission(lib, callback, context)
        lib.q.save_once(claim_path, claim)
        reviewer._coarse3_dispatches = getattr(reviewer, '_coarse3_dispatches', 0) + 1
        result = reviewer(row, root, prepare_only=False)
        need(result.get('status') in ('pass', 'confirmed_error', 'uncertain'), 'Independent actual review transaction incomplete')
        calls = 1
    checked, receipt_ref = lib.old.verify_review_receipt(root, row, evidence, reviewer)
    need(lib.q.load(transaction / 'receipt.json').get('semantic_approval') is False
         and lib.q.load(transaction / 'receipt.json').get('human_review_completed') is False,
         'Review cannot grant human/semantic approval')
    verified_admission(lib, callback, context)
    return checked, receipt_ref, calls


def aggregate(results, *, prepare_only, population_ref, reviewer=None):
    reviewed = [r for r in results if r['status'] in ('pass', 'confirmed_error', 'uncertain')]
    by_scope = {scope: {'prepared_source_count': sum(r.get('label_scope') == scope and r['status'] == 'prepared' for r in results),
        'reviewed_denominator': sum(r.get('label_scope') == scope for r in reviewed),
        'confirmed_error_count': sum(r.get('label_scope') == scope and r['status'] == 'confirmed_error' for r in reviewed)}
        for scope in ('complete',)}
    return {'schema': SCHEMA, 'population_ref': population_ref, 'population_kind': 'fixed-three-independent-pilot',
        'population_source_count': 3, 'maximum_review_calls': 3, 'selected_source_ids': list(IDS),
        'sampled_reviewed_denominator': len(reviewed), 'scope_denominators': by_scope,
        'original_curated_batch_mapping': {r['source_id']: r.get('original_membership') for r in results},
        'statuses': dict(Counter(r['status'] for r in results)),
        'pending_source_ids': [r['source_id'] for r in results if r['status'] == 'pending'],
        'confirmed_error_count': sum(r['status'] == 'confirmed_error' for r in reviewed),
        'review_dispatches': sum(r.get('review_dispatches', 0) for r in results),
        'verified_original_observation_model_calls': sum(r.get('original_observation_model_calls', 0) for r in results),
        'recovery_new_model_calls': 0, 'recovery_new_budget_reservations': 0,
        'actual_VLM_calls': None if any(r.get('actual_VLM_calls') is None for r in results)
                            else sum(r.get('actual_VLM_calls', 0) for r in results),
        'independent_sample_review_completed': not prepare_only and len(reviewed) == 3,
        'state': 'prepared_pending_independent_review' if prepare_only and all(r['status'] == 'prepared' for r in results)
                 else 'sample_review_completed_pending_human' if len(reviewed) == 3 else 'pending',
        'GPU_model_load_performed': bool(not prepare_only and getattr(reviewer, '_loaded', None) is not None),
        'statistical_population_error_estimate': False, 'old_scores_reused': False,
        'confirmed_errors_are_AI_candidates_not_ground_truth': True, 'original_attempt_budget_consumed': False,
        'original_worker_reservation_modified': False, **NO_APPROVAL}


def execute(out=OUT, *, prepare_only=True, reviewer=None, admission_callback=None,
            ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg'):
    need(prepare_only or (reviewer is not None and callable(admission_callback)),
         'Run denied: externally scheduled reviewer AND verified caller admission callback required')
    lib = libraries()
    need(prepare_only or (isinstance(reviewer, lib.audit.LocalReviewer)
         and getattr(reviewer, '_coarse3_caller_guard', False)), 'Run requires make_reviewer caller-guarded real LocalReviewer')
    out = check_output(out, lib)
    need(prepare_only or Path(reviewer.out).resolve() == out, 'Reviewer model/identity outputs must use this independent output')
    cfg, contract, launch_ref = source_contract(lib)
    population = {'schema': SCHEMA, 'source_ids': list(IDS), 'population_source_count': 3,
        'maximum_review_calls': 3, 'source_launch_ref': launch_ref, 'jobs_manifest_ref': contract['manifest_ref'],
        'source_job_refs': [h['job_ref'] for h in contract['manifest']['jobs']],
        'recovery_code_ref': RECOVERY_REFS[0], 'recovery_test_ref': RECOVERY_REFS[1],
        'recovery_output_root': str(lib.recovery.OUT), 'recovery_revision': lib.recovery.REVISION,
        'reviewable_source_scope': 'complete_recovered_labels_only', 'not_original_batch_membership': True, **NO_APPROVAL}
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'independent.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lib.q.save_once(out / 'frozen.population.json', population)
        population_ref = lib.q.identity(out / 'frozen.population.json')
        results = []
        for hint, sid in zip(contract['manifest']['jobs'], IDS):
            base = {'source_id': sid, 'status': 'pending', 'review_dispatches': 0, 'actual_VLM_calls': 0,
                    'AI_candidate_not_ground_truth': True, **NO_APPROVAL}
            before = getattr(reviewer, '_coarse3_dispatches', 0)
            try:
                source_job = lib.q.checked(hint['job_ref'])
                source_plan = lib.q.checked(source_job['refs']['plan_ref'])
                base['original_membership'] = verify_membership(lib, source_job, source_plan)
                candidate, verified = validate_candidate(lib, cfg, hint, ffprobe=ffprobe)
                if candidate is None:
                    results.append({**base, **verified})
                    continue
                base.update({k: v for k, v in verified.items() if k not in ('status', 'probe')})
                row = review_row(lib, candidate, verified, population_ref)
                root = out / 'sources' / ('train-' + sid) / lib.batch.digest(row)[:24]
                with lib.review_library():
                    evidence = lib.audit.prepare_evidence(row, root, ffprobe=ffprobe, ffmpeg=ffmpeg, hand_roi_root=row['hand_roi']['path'])
                    verify_evidence(lib, evidence, row, root, verified['probe'])
                    request_model = '/CPU-preview-only/no-checkpoint-loaded'
                    request = lib.audit.review_request(row, evidence, root, request_model)
                    lib.q.save_once(root / 'independent.request.preview.json', request)
                    lib.q.save_once(root / 'input.row.json', row)
                    base.update(evidence_ref=lib.q.identity(root / 'evidence.json'), input_row_sha256=lib.batch.digest(row),
                                native_source_frames=len(evidence['frames']), native_hand_rois=len(evidence['hand_rois']))
                    if prepare_only:
                        result = {**base, 'status': 'prepared', 'GPU_model_load_performed': False}
                        lib.q.save_once(root / 'independent.prepare.receipt.json', result)
                    else:
                        identity = reviewer.identity()
                        context = {'schema': SCHEMA, 'caller_pid': os.getpid(), 'out': str(out), 'population_ref': population_ref,
                            'jobs_manifest_ref': population['jobs_manifest_ref'],
                            'source_job_refs': population['source_job_refs'], 'candidate_ref': verified['candidate_ref'],
                            'input_row_sha256': lib.batch.digest(row), 'model_sha256': identity['sha256'],
                            'tensor_parallel_size': 1, 'maximum_review_calls': 3}
                        checked, receipt, calls = review_transaction(lib, out, row, root, evidence, reviewer, admission_callback, context)
                        result = {**base, **checked, 'review_receipt': receipt, 'review_dispatches': calls, 'actual_VLM_calls': calls, **NO_APPROVAL}
                        lib.q.save_once(root / 'independent.review.receipt.json',
                                        {k: v for k, v in result.items() if k not in ('actual_VLM_calls', 'review_dispatches')})
                validate_candidate(lib, cfg, hint, ffprobe=ffprobe)
                results.append(result)
            except Exception as error:
                calls = getattr(reviewer, '_coarse3_dispatches', 0) - before
                results.append({**base, 'status': 'pending', 'error': type(error).__name__ + ': ' + str(error),
                                'review_dispatches': calls, 'actual_VLM_calls': None if calls else 0})
        summary = aggregate(results, prepare_only=prepare_only, population_ref=population_ref, reviewer=reviewer)
        lib.batch.atomic_write(out / 'independent.ledger.jsonl', b''.join(lib.q.encoded(r) for r in results))
        lib.batch.atomic_write(out / 'independent.summary.json', lib.batch.json_bytes(summary))
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare', '--prepare-only', dest='prepare', action='store_true')
    stage.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--ffprobe', default='/opt/ms_tools/ffprobe')
    parser.add_argument('--ffmpeg', default='/usr/local/bin/ffmpeg')
    args = parser.parse_args(argv)
    if args.run:
        parser.error('--run denied: use execute(prepare_only=False, reviewer=make_reviewer(...), admission_callback=caller_verified_callback) from the resource-owning caller')
    report = execute(args.out, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report['pending_source_ids'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
