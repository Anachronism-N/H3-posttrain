"""Exact N5G01 finite-three wrapper; original pidfd/restore manager is unchanged."""
import os
from pathlib import Path
import select
import stat
import time
import prepare_curated_coarse_empty_n5_v1 as c

base = c.base_module()
original_validate = base.validate_pair
original_snapshot = base.lease_snapshot
original_refresh = base.refresh_leases
active_config = None
proof_written = False


def selected(cfg):
    manifest = c.checked(cfg['jobs_manifest_ref'])
    if len(manifest.get('jobs', [])) != 3:
        raise RuntimeError('complete three-source manifest required')
    return [hint['source_key'] for hint in manifest['jobs']], 222939


def validate(cfg):
    global active_config
    worker = c.worker_module()
    contract = c.frozen(worker, probe=True)
    c.verify_config(cfg, contract)
    c.check_node()
    original_validate(cfg)
    c.verify_local_model(cfg, worker)
    evidence = c.checked(cfg['previous_resources_ref'])
    if evidence != c.prior_safe(base):
        raise RuntimeError('original ownedmanager safe/restored evidence changed')
    capture = c.checked(cfg['holder_capture'])
    if (capture['holders'] != cfg['holders'] or capture['lease_snapshot_ns'] != cfg['lease_snapshot_ns']
            or capture.get('signals_sent') is not False
            or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9):
        raise RuntimeError('immutable fresh CPU capture expired/changed')
    if capture['budget']['headroom'] < 160 * 2**30:
        raise RuntimeError('captured headroom under 160GiB')
    c.require_unreserved(contract['jobs'])
    active_config = cfg
    fresh = c.fresh_pair(base, evidence)
    if fresh['lease_snapshot_ns'] != cfg['lease_snapshot_ns']:
        raise RuntimeError('exact old lease snapshot changed since launch')
    for current, expected in zip(fresh['holders'], cfg['holders']):
        base.check_holder(expected, current, current['gpu'], current['uuid'], base.apps())


def lease_snapshot(path):
    value = original_snapshot(path)
    if active_config is None or str(path) not in active_config['lease_snapshot_ns']:
        raise RuntimeError('both exact fresh pair lease snapshots required')
    expected = active_config['lease_snapshot_ns'][str(path)]
    c.verify_lease_value(expected)
    if c.lease_value(value) != expected:
        raise RuntimeError('full integer-nanosecond lease snapshot changed')
    return value


def refresh_leases(entries, on_error=None):
    global proof_written
    original_refresh(entries, on_error)
    if proof_written or active_config is None or len(entries) != 2 or not all(e['released'] and e['leasefd'] is not None for e in entries):
        return
    if not all(select.select([e['fd']], [], [], 0)[0] for e in entries):
        return
    if [app for app in base.apps() if app['uuid'] in {e['uuid'] for e in entries}]:
        return
    pid = os.getpid()
    proof = {'schema': 'coarse_empty_n5_actual_leases_held_v1', 'manager_identity': base.identity(pid),
             'config_ref': c.ref(c.LAUNCH), 'resource_admission_ref': c.ref(c.QUEUE / 'admission.json'),
             'node': 'N5', 'gpus': c.PAIR, 'leases': [], 'observed_unix_ns': time.time_ns(),
             'proof_is_historical_live_fd_kernel_locks_and_heartbeat_are_current_authority': True}
    for e in entries:
        fd_stat, path_stat = os.fstat(e['leasefd']), e['reserve'].lstat()
        if (not stat.S_ISREG(path_stat.st_mode) or path_stat.st_uid != os.getuid() or path_stat.st_size != 0
                or (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
                or not c.fd_inode_owned(pid, path_stat.st_dev, path_stat.st_ino)
                or not c.proc_lock_owned(pid, '/tmp/ms12k_gpu_%d.lock' % e['gpu'])):
            raise RuntimeError('actual two parent lease descriptors/kernel FLOCK ownership required')
        proof['leases'].append({'gpu': e['gpu'], 'path': str(e['reserve']), 'stat': c.lease_value(path_stat),
                                'uuid': e['uuid']})
    c.save_once(c.QUEUE / 'actual_leases_held.json', proof)
    proof_written = True


def verify_result(cfg, out, hint, terminal):
    job = c.checked(hint['job_ref'])
    root = out / 'revisions' / c.REVISION / job['source_key']
    if terminal['joint_result_ref']['path'] != str(root / 'result.json'):
        raise RuntimeError('candidate result outside exact current source revision')
    result = c.checked(terminal['joint_result_ref'])
    if (result.get('new_text_model_requests') != 0 or result.get('new_visual_requests') != 1
            or result.get('original_job_ref') != hint['job_ref']
            or result.get('new_label_review_required') is not True or result.get('training_eligible') is not False
            or result.get('style_label_changed') is not False):
        raise RuntimeError('candidate budget/revision/review/style binding changed')
    if result['visual_result_ref']['path'] != str(root / 'visual_result.json'):
        raise RuntimeError('current source visual-result reference changed')
    visual = c.checked(result['visual_result_ref'])
    worker = c.worker_module()
    original, old_joint, prepared, plan = worker.verify_job(job, probe=False)
    if result['style_result'] != old_joint['style_result'] or result['dance_style_label'] != old_joint['dance_style_label']:
        raise RuntimeError('original style modified by current stage')
    if [o for o in visual['window_observations'] if o['window']['window_id'] != job['target_window_id']] != [o for o in original['window_observations'] if o['window']['window_id'] != job['target_window_id']]:
        raise RuntimeError('original untargeted windows modified by current stage')
    attempt = root / 'windows' / job['target_window_id'] / 'attempt-000001'
    request = c.load(attempt / 'request.json')
    preview = c.load(out / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
    receipt = c.load(attempt / 'receipt.json')
    if request != preview or receipt['binding']['current_job_ref'] != hint['job_ref']:
        raise RuntimeError('actual request/receipt differs from CPU frozen job preview')
    names = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
    observed = receipt.get('files_sha256', {})
    if not set(observed) <= names or any(c.sha(attempt / name) != digest for name, digest in observed.items()):
        raise RuntimeError('actual six-file transaction receipt SHA changed')
    if receipt.get('status') == 'candidate' and set(observed) != names:
        raise RuntimeError('candidate requires all six current transaction files')
    if (attempt / 'generation.json').exists():
        generation = c.load(attempt / 'generation.json')
        if (generation.get('model_call_performed') is not True or generation.get('request_sha256') != c.sha(attempt / 'request.json')
                or generation.get('raw_response_sha256') != c.sha(attempt / 'response.raw.txt')
                or generation.get('model_identity_sha256') != cfg['modelref']['model_identity_sha256']
                or generation.get('max_tokens') != request['sampling']['max_tokens']
                or generation.get('structured_output_schema_sha256') != worker.q.digest(request['json_schema'])):
            raise RuntimeError('actual new generation model/request/schema/raw-response binding changed')
    elif receipt.get('status') != 'failed':
        raise RuntimeError('nonfailed current source requires new generation proof')
    marker = c.load(out / 'joint/review-required' / (hint['source_key'] + '.json'))
    if marker.get('result_ref') != terminal['joint_result_ref'] or marker.get('new_label_review_required') is not True:
        raise RuntimeError('new candidate review-required marker missing')


def summaries(cfg, out):
    # Semantic completion is not the condition for safe resource restoration.
    report_path = out / 'finite.stage.summary.json'
    report = c.load(report_path)
    manifest = c.checked(cfg['jobs_manifest_ref'])
    refs = [h['job_ref'] for h in manifest['jobs']]
    if (report.get('global_revision') != c.REVISION or report.get('jobs_manifest_ref') != cfg['jobs_manifest_ref']
            or report.get('source_job_refs') != refs or report.get('source_count') != 3
            or report.get('terminal_count') != 3 or report.get('text_model_calls') != 0
            or type(report.get('visual_model_calls')) is not int or not 0 <= report['visual_model_calls'] <= 3
            or len(report.get('terminals', [])) != 3):
        raise RuntimeError('current finite three-terminal zero-text summary required')
    for hint, terminal_ref in zip(manifest['jobs'], report['terminals']):
        expected = out / 'worker-terminals' / (hint['source_key'] + '.json')
        if terminal_ref['path'] != str(expected):
            raise RuntimeError('terminal outside current independent source')
        terminal = c.checked(terminal_ref)
        if (terminal.get('source_key') != hint['source_key'] or terminal.get('global_revision') != c.REVISION
                or terminal.get('original_job_ref') != hint['job_ref']
                or terminal.get('status') not in ('joint_repair_candidate', 'incomplete_repair_candidate',
                    'failed_preserved_no_redispatch', 'not_dispatched_after_shared_failure')
                or terminal.get('semantic_approval') is not False or terminal.get('training_eligible') is not False):
            raise RuntimeError('actual source terminal/ref/edition/review binding changed')
        if terminal.get('joint_result_ref'):
            verify_result(cfg, out, hint, terminal)
    for value in report.get('evidence_refs', []):
        if not Path(value['path']).resolve().is_relative_to(out):
            raise RuntimeError('current-stage result evidence outside new output')
        if c.sha(value['path']) != value['sha256']:
            raise RuntimeError('current-stage evidence SHA changed')
    return {str(report_path): {'summary': {'stage': 'run', 'selected_source_count': 3,
                'processed_source_count': 3, 'total_source_count': 222939, 'plan_sha256': cfg['plan']['sha256']},
                'actual_finite_result': report, 'semantic_approval': False, 'partial_labels_preserved': True}}


base.AUTHORIZED_PAIRS = {'N5': ((0, 1),)}
base.validate_pair = validate
base.selected_sources = selected
base.lease_snapshot = lease_snapshot
base.refresh_leases = refresh_leases
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
