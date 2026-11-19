"""Offline clause-scoped replay; never dispatches or reserves a model attempt."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

SHAPE = re.compile(r'\b(?:V(?:[- ](?:shape|sign))?|OK|heart|fists?)\b', re.I)
UNCERTAIN = re.compile(r'\b(?:uncertain|unclear|unconfirmed|possibly|perhaps|maybe)\b|\bloose\s+V[- ]shape', re.I)
NEGATED_DETAIL = re.compile(r"\b(?:no|not|never|cannot)\b|\b\w+n['’]t\b", re.I)
HAND_DETAIL = re.compile(r'\b(?:hands?|fingers?|shape|detail|visible|visibility|distinguishable)\b', re.I)
FLAGS = {'semantic_approval': False, 'human_review_completed': False,
         'formal_approval': False, 'formal_training': False,
         'training_entry_updated': False, 'training_eligible': False}


def internal_detail(text):
    return bool(UNCERTAIN.search(text) or
                (NEGATED_DETAIL.search(text) and (HAND_DETAIL.search(text) or SHAPE.search(text))))


def named_shape_error(event, value, manifest):
    if not SHAPE.search(event['description']):
        return None
    hands = [p for p in event['parts'] if p in ('left_hand', 'right_hand')]
    if not hands:
        return 'Named finger shape requires its actual hand evidence'
    support = {p['part']: p for p in value['part_support']}
    evidence = manifest.get('hand_roi_evidence') or {}
    refs = set(event['frame_indices'])
    for part in hands:
        # A crop never establishes clarity; an explicitly unclear support cannot authorize a shape.
        if internal_detail(support[part]['reason']):
            return 'Named finger shape has explicitly uncertain hand support'
        crops = evidence.get(part, {}).get('crops', [])
        native = {c['frame_index'] for c in crops if not c.get('resized', False)}
        if len(native & refs & set(support[part]['frame_indices'])) < 2:
            return 'Named finger shape requires two distinct actual native ROI frames'
    return None


def validate_reobserve(value, manifest, forbidden, bar):
    # Let the frozen filter own all clause IDs, part selection, and actual frame support.
    result = bar.filter_supported_observation(value, manifest, forbidden)
    internal = []
    for event in value['events']:
        if not isinstance(event, dict) or not isinstance(event.get('description'), str):
            continue
        description = event['description']
        for index, clause in enumerate(re.split(r'(?<=[.!?;])\s+', description)):
            if internal_detail(clause):
                internal.append({'raw_event_id': event.get('id'), 'raw_clause_index': index,
                    'raw_source_clause': clause, 'original_description': description,
                    'reason': 'uncertain_or_negated_detail_not_an_affirmative_motion_fact',
                    'text_preserved_verbatim': True, 'stored_as_metadata_not_motion_fact': True})
    retained = []
    for event in result['events']:
        if internal_detail(event['description']):
            continue
        error = named_shape_error(event, value, manifest)
        if error:
            result['support_filter_audit']['rejected_events'].append({
                **event['source_clause'], 'reason': error, 'strict_named_shape_gate': True})
            continue
        retained.append(event)
    result.update(events=retained, abstain=not retained)
    result['support_filter_audit']['admitted_event_count'] = len(retained)
    result['scope_filter_audit'] = {'internal_detail_clauses': internal,
        'uncertainty_tokens_removed_to_strengthen_claim': False,
        'affirmative_text_modified': False, 'frame_references_added': False,
        'actual_part_support_gate_unchanged': True,
        'native_named_shape_gate_preserved': True,
        'filter_is_not_semantic_truth_review': True}
    for note in internal:
        if note['raw_source_clause'] not in result['uncertainties']:
            result['uncertainties'].append(note['raw_source_clause'])
    return result


def import_frozen_queue(root):
    path = root / 'runtime/source_repair_gpu_v2/curated_repair_queue_v1.py'
    spec = importlib.util.spec_from_file_location('frozen_repair_queue', path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    return queue


def derive(repair_root, revision, ms_source):
    repair_root = Path(repair_root).resolve()
    # The explicitly supplied original revision path must have this fixed layout.
    queue_root = repair_root.parents[3]
    root = queue_root.parent
    queue = import_frozen_queue(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe offline revision')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(ms_source).resolve()))
    import ms_bar_annotation as bar
    if Path(bar.__file__).resolve().parent != Path(ms_source).resolve():
        raise ValueError('Frozen bar import path differs')
    original_revision = repair_root.parents[1].name
    load_receipt = queue_root / 'gpu-runtime' / original_revision / 'load.receipt.json'
    code = queue.load(load_receipt)['code_sha256']
    for path, expected in code.items():
        queue.verify_bytes({'path': path, 'sha256': expected})
    if bar.code_identity() != code:
        raise ValueError('Frozen bar identity differs from original GPU receipt')
    job = queue.load(repair_root / 'original_job.json')
    queue.verify_job(job)
    prepared = queue.checked(job['refs']['prepared'])
    manifest = queue.checked(job['refs']['demo_manifest'])
    raw_path = repair_root / 'windows/w00001/response.raw.txt'
    receipt_path = raw_path.with_name('receipt.json')
    request_path = raw_path.with_name('request.json')
    generation_path = raw_path.with_name('generation.json')
    receipt = queue.load(receipt_path)
    generation = queue.load(generation_path)
    for name, expected in receipt['files_sha256'].items():
        queue.verify_bytes({'path': raw_path.parent / name, 'sha256': expected})
    if generation['raw_response_sha256'] != queue.sha(raw_path):
        raise ValueError('Original generation raw SHA differs')
    if generation['request_sha256'] != queue.sha(request_path):
        raise ValueError('Original generation request SHA differs')
    if receipt['binding']['source_binding'] != job['source_binding']:
        raise ValueError('Original response source binding differs')
    if receipt['binding']['window'] != manifest['window']:
        raise ValueError('Original response window binding differs')
    if receipt['binding']['frames_sha256'] != bar.batch.digest(manifest['frames']):
        raise ValueError('Original response actual frame binding differs')
    queue.verify_references_by_content(manifest.get('hand_roi_evidence', {}))
    raw = queue.load(raw_path)
    before = {str(p): queue.sha(p) for p in repair_root.rglob('*') if p.is_file() and p.name != '.lock'}
    budget_path = queue_root / 'attempt-budget' / (job['source_key'] + '.json')
    budget_ref = queue.identity(budget_path)
    observations = copy.deepcopy(queue.load(repair_root / 'result.json')['window_observations'])
    original = queue.checked(job['original_result_ref'])
    windows = [m['window'] for m in prepared['windows']]
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    value = validate_reobserve(raw, manifest, forbidden, bar)
    target = next(i for i, w in enumerate(windows) if w['window_id'] == 'w00001')
    new_root = queue_root / 'revisions' / revision / job['source_key']
    provenance = {'source_response_ref': queue.identity(raw_path),
        'source_request_ref': queue.identity(request_path),
        'source_receipt_ref': queue.identity(receipt_path),
        'source_generation_ref': queue.identity(generation_path),
        'original_failed_result_ref': queue.identity(repair_root / 'result.json'),
        'original_result_ref': job['original_result_ref'],
        'evidence_manifest_ref': job['refs']['demo_manifest'],
        'original_gpu_load_receipt_ref': queue.identity(load_receipt),
        'frozen_bar_code_sha256': code,
        'frozen_queue_code_ref': queue.identity(Path(queue.__file__)),
        'new_scope_code_ref': queue.identity(Path(__file__)),
        'offline_replay_only': True, 'new_model_calls': 0,
        'model_rewrite_performed': False, 'reobservation_performed': False,
        'budget_ref_unchanged': budget_ref, 'style_label_changed': False}
    observations[target] = {'status': 'abstain' if value['abstain'] else 'candidate',
        'json': value, 'window': windows[target], 'format_valid': True,
        'model_call_performed': False, 'source_model_call_performed_previously': True,
        'attempt_path': str(new_root / 'windows/w00001'),
        'offline_provenance': provenance, **FLAGS}
    facts = bar.facts_from_observations(windows, observations)
    boundary_results = copy.deepcopy(queue.load(repair_root / 'result.json').get('boundary_observations', []))
    boundaries = [r for b in boundary_results if b['status'] in ('candidate', 'unknown')
                  for r in b.get('json', {}).get('relations', [])]
    deterministic = copy.deepcopy(original['final']['json'])
    local = [f for f in facts if f['window_id'] == 'w00001']
    for item in deterministic['windows']:
        if item['window_id'] == 'w00001':
            item.update(caption=' '.join(f['raw_text'] for f in local),
                        used_fact_ids=[f['fact_id'] for f in local], used_boundary_ids=[])
    frozen = {w['window_id']: copy.deepcopy(w) for w in original['final']['json']['windows']
              if w['window_id'] != 'w00001'}
    validation_error = None
    try:
        queue.validate_repair_rewrite(deterministic, windows, facts, boundaries, bar, frozen)
    except (ValueError, TypeError, KeyError) as error:
        validation_error = str(error)
    for i, window in enumerate(windows):
        if window['window_id'] != 'w00001':
            assert observations[i] == original['window_observations'][i]
    prior_rewrites = sorted(str(p) for p in (queue_root / 'revisions').glob(
        '*/' + job['source_key'] + '/rewrite/**/*') if p.is_file())
    budget = {'maximum_extra_source_attempts': 1, 'extra_source_attempt_already_reserved': True,
        'extra_reobserve_calls_already_used': 1, 'remaining_reobserve_calls': 0,
        'extra_rewrite_artifacts': prior_rewrites,
        'remaining_rewrite_calls': 0 if prior_rewrites else 1,
        'new_model_calls': 0, 'budget_reserved_or_modified_by_offline_replay': False,
        'original_budget_ref': budget_ref}
    final = {'status': 'deterministic_derived_candidate' if validation_error is None else 'failed',
        'json': deterministic, 'format_valid': validation_error is None,
        'offline_supported_rewrite_validation_completed': validation_error is None,
        'model_call_performed': False, 'model_rewrite_performed': False,
        'construction': 'only_target_window_verbatim_admitted_fact_concatenation',
        **FLAGS}
    if validation_error:
        final['error'] = validation_error
    result = {'schema': 'curated_repair_scope_offline_v1', 'id': job['id'],
        'global_revision': revision, 'original_job_ref': queue.identity(repair_root / 'original_job.json'),
        'original_result_ref': job['original_result_ref'], 'original_source_binding': job['source_binding'],
        'window_observations': observations, 'facts': facts,
        'boundary_observations': boundary_results, 'final': final,
        'all_windows_nonempty': all(w['caption'].strip() for w in deterministic['windows']),
        'structural_repair_candidate_complete': False, 'repair_completed': False,
        'scope_supported_body_validation_passed': bool(local),
        'style_label_changed': False, 'independent_review_status': 'pending_review_not_approved',
        'untouched_windows_AI_error_claim': 'notConfirmed_do_not_automatically_modify',
        'original_raw_or_terminal_modified': False, 'offline_provenance': provenance,
        'budget_audit': budget, **FLAGS}
    for path, expected in before.items():
        queue.verify_bytes({'path': path, 'sha256': expected})
    queue.verify_bytes(budget_ref)
    queue.verify_job(job)
    queue.save_once(new_root / 'windows/w00001/final.json', value)
    queue.save_once(new_root / 'windows/w00001/receipt.offline.json', {
        'status': observations[target]['status'], 'provenance': provenance,
        'derived_final_sha256': queue.sha(new_root / 'windows/w00001/final.json'), **FLAGS})
    queue.save_once(new_root / 'budget.audit.json', budget)
    queue.save_once(new_root / 'result.json', result)
    return {'result_ref': queue.identity(new_root / 'result.json'),
            'scope_supported_body_validation_passed': bool(local),
            'deterministic_rewrite_validation_error': validation_error,
            'target_facts': local, 'budget_audit': budget, 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-root', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--ms-source', required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.repair_root, args.revision, args.ms_source), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
