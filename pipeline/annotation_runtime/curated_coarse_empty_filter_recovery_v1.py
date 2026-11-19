"""Zero-model recovery of exact frozen coarse observations; original failures stay intact."""
import argparse
import copy
import json
from pathlib import Path
import prepare_curated_coarse_empty_n5_v1 as c

OUT = c.BAR / 'curated_coarse_empty_filter_recovery_v1'
REVISION = 'coarse-filter-recovery-20261025-v1'
SCHEMA = 'curated_coarse_empty_filter_recovery_v1'
LAUNCH_SHA = '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806'
NORMALIZATION_ERROR = 'No synthesized/repaired part references permitted'
SIX = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}


def need(value, message):
    if not value:
        raise ValueError(message)


def replay_raw(worker, raw, manifest, forbidden):
    # Every original raw/schema/coarse/support guard runs before the one bad equality.
    try:
        filtered = worker.validate_coarse(raw, manifest, forbidden)
    except ValueError as error:
        if str(error) != NORMALIZATION_ERROR:
            raise
        filtered = worker.bar.filter_supported_observation(raw, manifest, forbidden)
    originals = {e['id']: e for e in raw['events']}
    for event in filtered['events']:
        clause = event['source_clause']
        original = originals[clause['raw_event_id']]
        need(clause['raw_clause_index'] == 0 and event['description'] == original['description']
             == clause['raw_source_clause'] == clause['original_description'],
             'Only complete verbatim original coarse sentences may be recovered')
        need(event['frame_indices'] == original['frame_indices']
             and set(event['parts']) <= set(original['parts']),
             'No added part or synthesized/repaired frame reference permitted')
    return filtered


def projected(value):
    return {'events': [{k: copy.deepcopy(e[k]) for k in ('id', 'description', 'parts', 'frame_indices')}
                       for e in value['events']],
            **{k: copy.deepcopy(value[k]) for k in ('part_support', 'uncertainties', 'abstain')}}


def package_replayed(worker, job, observations, original, old_joint, prepared, plan, filtered):
    expected = projected(filtered)
    manifest = worker.verify_window(job, prepared)
    old_validator = worker.validate_coarse

    def exact_prevalidated(value, current_manifest, current_forbidden):
        need(value == expected and current_manifest == manifest
             and current_forbidden == prepared['forbidden'],
             'Normalized second pass differs from this exact already-replayed raw source')
        return copy.deepcopy(filtered)

    worker.validate_coarse = exact_prevalidated
    try:
        return worker.package(job, observations, original, old_joint, prepared, plan)
    finally:
        worker.validate_coarse = old_validator


def derive(worker, cfg, hint):
    q, bar = worker.q, worker.bar
    job = c.checked(hint['job_ref'])
    need(job['source_key'] == hint['source_key'] and job['id'] in worker.TARGETS, 'Fixed source job differs')
    root = worker.OUT / 'revisions' / worker.REVISION / job['source_key']
    terminal_path = worker.OUT / 'worker-terminals' / (job['source_key'] + '.json')
    if not terminal_path.exists():
        return None, {'source_key': job['source_key'], 'status': 'pending_original_terminal'}
    original, old_joint, prepared, plan = worker.verify_job(job, probe=True)
    terminal_ref, parent_ref = c.ref(terminal_path), c.ref(root / 'result.json')
    terminal, parent = c.checked(terminal_ref), c.checked(parent_ref)
    need(terminal['joint_result_ref'] == parent_ref and terminal['original_job_ref'] == hint['job_ref']
         and terminal['global_revision'] == worker.REVISION and terminal['source_key'] == job['source_key']
         and parent['original_job_ref'] == hint['job_ref'] and parent['new_visual_requests'] == 1
         and parent['new_text_model_requests'] == 0 and parent['style_result'] == old_joint['style_result'],
         'Actual original coarse terminal/result provenance changed')
    budget_ref = c.ref(worker.budget_path(job))
    budget = c.checked(budget_ref)
    need(budget == c.load(root / 'canonical_budget_claim.json') and budget['source_key'] == job['source_key']
         and budget['global_revision'] == worker.REVISION and budget['extra_source_attempt'] == 1
         and budget['maximum_extra_source_attempts'] == 1
         and budget['original_result_sha256'] == job['original_result_sha256'],
         'Actual unchanged canonical budget claim differs')
    attempt = root / 'windows' / job['target_window_id'] / 'attempt-000001'
    request = worker.make_request(job, hint['job_ref'], prepared, worker.worker_args())
    preview_ref = c.ref(worker.OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
    receipt = c.load(attempt / 'receipt.json')
    manifest = worker.verify_window(job, prepared)
    expected_binding = {**original['binding'], 'repair_revision': worker.REVISION, 'current_job_ref': hint['job_ref'],
        'original_canonical_job_ref': job['refs']['original_canonical_job'], 'ledger_bindings': job['ledger_bindings'],
        'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames']),
        'request_preview_ref': preview_ref}
    need(c.load(attempt / 'request.json') == c.checked(preview_ref) == request
         and c.load(attempt / 'binding.json') == receipt['binding'] == expected_binding,
         'Actual request/preview/frame/job binding differs')
    files = receipt['files_sha256']
    need(set(files) <= SIX and set(files) >= SIX - {'response.json'}
         and set(files) == {name for name in SIX if (attempt / name).exists()},
         'Actual original successful/failed transaction inventory differs')
    for name, digest in files.items():
        need(c.sha(attempt / name) == digest, 'Original transaction SHA changed: ' + name)
    final = c.load(attempt / 'final.json')
    need(all(receipt.get(k) is False and final.get(k) is False for k in bar.core.FLAGS),
         'Original transaction must not claim approval')
    generation = bar.core.validate_generation(attempt, request)
    need(generation['model_identity_sha256'] == cfg['modelref']['model_identity_sha256'],
         'Actual original annotator checkpoint differs')
    raw = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
    filtered = replay_raw(worker, raw, manifest, prepared['forbidden'])
    if receipt['status'] == 'failed':
        need(final['status'] == 'failed' and final.get('error') == 'ValueError: ' + NORMALIZATION_ERROR
             and receipt.get('error') == final['error'], 'Only proved part-normalization failure is recoverable')
    else:
        need(receipt['status'] == final['status'] == 'candidate' and set(files) == SIX
             and final['json'] == c.load(attempt / 'response.json') == filtered,
             'Successful original candidate must exactly replay')
    observations = copy.deepcopy(original['window_observations'])
    target = next(i for i, o in enumerate(observations) if o['window']['window_id'] == job['target_window_id'])
    observations[target] = {'status': 'abstain' if filtered['abstain'] else 'candidate', 'json': filtered,
        'window': observations[target]['window'], 'attempt_path': str(attempt),
        'original_failed_transaction_preserved': receipt['status'] == 'failed',
        'derivation': 'exact_original_raw_native_filter_replay_no_added_references', **q.FLAGS}
    visual, joint = package_replayed(worker, job, observations, original, old_joint, prepared, plan, filtered)
    refs = [hint['job_ref'], terminal_ref, parent_ref, parent['visual_result_ref'], preview_ref, budget_ref,
            c.ref(attempt / 'receipt.json')] + [c.ref(attempt / name) for name in sorted(files)]
    proof = {'schema': SCHEMA, 'global_revision': REVISION, 'source_key': job['source_key'],
        'original_coarse_revision': worker.REVISION, 'original_job_ref': hint['job_ref'],
        'original_coarse_result_ref': parent_ref, 'primary_budget_ref': budget_ref,
        'original_observation_model_calls': 1, 'new_visual_requests': 0, 'new_text_model_requests': 0,
        'extra_source_attempts': 0, 'canonical_budget_unchanged': True, 'style_label_changed': False,
        'original_raw_or_terminal_modified': False, 'original_failed_transaction_preserved': receipt['status'] == 'failed',
        'new_label_review_required': True, 'independent_review_status': 'pending_independent_vlm_review',
        'derivation': 'exact_original_raw_native_filter_replay_no_added_references',
        'original_transaction_refs': refs, 'recovery_code_ref': c.ref(__file__), **q.FLAGS}
    visual.update(proof)
    joint.update(proof)
    joint['status'] = 'filter_recovery_joint_candidate' if joint['joint_complete'] else 'filter_recovery_incomplete_candidate'
    worker.verify_job(job, probe=True)
    for reference in refs:
        c.checked(reference) if Path(reference['path']).suffix == '.json' else need(c.sha(reference['path']) == reference['sha256'], 'Original artifact changed')
    need(c.ref(worker.budget_path(job)) == budget_ref, 'Canonical budget changed during zero-model derivation')
    return (visual, joint, proof), {'source_key': job['source_key'], 'status': joint['status'],
        'joint_complete': joint['joint_complete'], 'new_model_calls': 0}


def run():
    worker = c.worker_module()
    cfg = c.checked({'path': str(c.LAUNCH), 'sha256': LAUNCH_SHA})
    contract = c.frozen(worker, probe=False)
    c.verify_config(cfg, contract)
    rows = []
    for hint in contract['manifest']['jobs']:
        value, status = derive(worker, cfg, hint)
        if value is not None:
            visual, joint, proof = value
            root = OUT / 'revisions' / REVISION / hint['source_key']
            worker.q.save_once(root / 'derivation.receipt.json', proof)
            worker.q.save_once(root / 'visual_result.json', visual)
            joint['visual_result_ref'] = c.ref(root / 'visual_result.json')
            worker.q.save_once(root / 'result.json', joint)
            status['result_ref'] = c.ref(root / 'result.json')
            worker.q.save_once(OUT / 'review-required' / (hint['source_key'] + '.json'),
                {**status, 'new_label_review_required': True, **worker.q.FLAGS})
        rows.append(status)
    summary = {'schema': SCHEMA, 'source_count': 3, 'sources': rows, 'new_model_calls': 0,
               'new_budget_reservations': 0, 'original_population_completed': False, **worker.q.FLAGS}
    worker.q.save_once(OUT / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False), flush=True)
