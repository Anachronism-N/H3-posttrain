"""Independent N3G01 wrapper of the unchanged pidfd/heartbeat/restore manager."""
import json
import os
from pathlib import Path
import select
import time
import coarse6when_safe_common_v1 as c

base = c.base_module()
original_validate = base.validate_pair
original_snapshot = base.lease_snapshot
original_refresh = base.refresh_leases
active_config = None
proof_written = False


def selected(cfg):
    manifest = c.checked(cfg['jobs_manifest_ref'])
    return [row['source_key'] for row in manifest['jobs']], 222939


def lease_snapshot(path):
    value = original_snapshot(path)
    if active_config is None or str(path) not in active_config['lease_snapshot_ns']:
        raise RuntimeError('exact fresh pair lease evidence required')
    expected = active_config['lease_snapshot_ns'][str(path)]
    c.verify_lease_value(expected)
    if c.lease_value(value) != expected:
        raise RuntimeError('lease differs from full integer nanosecond fresh snapshot')
    return value


def validate(cfg):
    global active_config
    if cfg['node'] != 'N3' or cfg['gpus'] != c.PAIR or cfg['phase'] != 'coarse6when_safe_n3g01_v1' or cfg['targetcount'] != 2 or cfg['input_count'] != 222939:
        raise RuntimeError('manager only admits exact N3 physical0,1 two sources')
    original_validate(cfg)
    if Path(cfg['queue']).resolve() != c.QUEUE or Path(cfg['out']).resolve() != c.OUT or cfg['global_revision'] != c.REVISION:
        raise RuntimeError('independent output/queue/edition changed')
    scope = c.checked(cfg['scope_ref'])
    if cfg['source_sha256'].get(str(c.MANAGER)) != c.sha(c.MANAGER) or scope['source_sha256'].get(str(c.MANAGER)) != c.sha(c.MANAGER):
        raise RuntimeError('exact new manager source binding required')
    template, previous, consumer, manifest, jobs, bindings = c.frozen()
    if cfg['modelref'] != consumer['modelref'] or cfg['plan'] != consumer['plan'] or cfg['template_ref'] != c.ref(c.TEMPLATE) or cfg['jobs_manifest_ref'] != template['jobs_manifest_ref'] or cfg['protocol_ref'] != template['protocol_ref'] or cfg['producer'] != c.ref(c.DRIVER):
        raise RuntimeError('frozen model/source/driver contract changed')
    if cfg['argv'] != ['/opt/venv_vllm/bin/python', '-B', '-u', str(c.DRIVER), '--launch', str(c.LAUNCH)] or cfg['cwd'] != consumer['cwd'] or cfg['env'].get('COARSE6_MANAGER_CONFIG') != str(c.LAUNCH):
        raise RuntimeError('only exact admission thin driver allowed')
    if any(cfg['source_sha256'].get(path) != digest for path, digest in bindings.items()) or any(cfg['source_sha256'].get(path) != digest for path, digest in scope['source_sha256'].items()):
        raise RuntimeError('complete source freeze required')
    capture = c.checked(cfg['holder_capture'])
    evidence = c.checked(cfg['previous_complete_ref'])
    fresh_previous = c.previous_completed(previous)
    if evidence != fresh_previous:
        raise RuntimeError('previous completion/restored-holder evidence changed')
    active_config = cfg
    fresh = c.fresh_pair(base, evidence)
    if capture['holders'] != cfg['holders'] or capture['lease_snapshot_ns'] != cfg['lease_snapshot_ns'] or capture['signals_sent'] is not False or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9:
        raise RuntimeError('fresh holder capture expired/changed')
    for current, expected in zip(fresh['holders'], cfg['holders']):
        base.check_holder(expected, current, current['gpu'], current['uuid'], base.apps())
    if fresh['lease_snapshot_ns'] != cfg['lease_snapshot_ns']:
        raise RuntimeError('lease changed since immutable launch')
    c.require_unreserved(jobs)
    active_config = cfg


def refresh_leases(entries, on_error=None):
    global proof_written
    original_refresh(entries, on_error)
    if proof_written or active_config is None or len(entries) != 2 or not all(e['released'] and e['leasefd'] is not None for e in entries):
        return
    if not all(select.select([e['fd']], [], [], 0)[0] for e in entries):
        return
    if [a for a in base.apps() if a['uuid'] in {e['uuid'] for e in entries}]:
        return
    proof = {'schema': 'coarse6_actual_manager_leases_held_v1', 'manager_identity': base.identity(os.getpid()),
             'resource_admission_ref': c.ref(c.QUEUE / 'admission.json'), 'config_ref': c.ref(c.LAUNCH),
             'node': 'N3', 'gpus': c.PAIR, 'leases': [], 'observed_unix_ns': time.time_ns()}
    for e in entries:
        s = os.fstat(e['leasefd'])
        now = e['reserve'].lstat()
        if (s.st_dev, s.st_ino) != (now.st_dev, now.st_ino) or not c.proc_lock_owned(os.getpid(), '/tmp/ms12k_gpu_%d.lock' % e['gpu']):
            raise RuntimeError('actual base-manager lock/lease ownership missing')
        proof['leases'].append({'gpu': e['gpu'], 'path': str(e['reserve']), 'stat': c.lease_value(now)})
    c.save_once(c.QUEUE / 'actual_leases_held.json', proof)
    proof_written = True


def summaries(cfg, out):
    manifest = c.checked(cfg['jobs_manifest_ref'])
    values = {}
    for hint in manifest['jobs']:
        path = out / 'announced' / (hint['source_key'] + '.json')
        value = c.load(path)
        if value.get('source_key') != hint['source_key'] or value.get('global_revision') != c.REVISION:
            raise RuntimeError('actual two-source terminal edition changed')
        if value.get('semantic_approval') is not False or value.get('training_eligible') is not False:
            raise RuntimeError('unreviewed partial results must not be approved')
        if value.get('joint_result_ref'):
            c.checked(value['joint_result_ref'])
            review = c.load(out / 'joint/review-required' / (hint['source_key'] + '.json'))
            if review.get('new_label_review_required') is not True or review.get('independent_review_status') != 'pending_independent_vlm_review' or review.get('result_ref') != value['joint_result_ref']:
                raise RuntimeError('independent review-required marker missing')
        values[str(path)] = {'summary': {'stage': 'run', 'selected_source_count': 2, 'processed_source_count': 2,
                              'total_source_count': 222939, 'plan_sha256': cfg['plan']['sha256']},
                             'actual_terminal': value, 'semantic_approval': False,
                             'independent_review_required': True, 'partial_labels_preserved': True}
    return values


base.AUTHORIZED_PAIRS = {'N3': ((0, 1),)}
base.validate_pair = validate
base.selected_sources = selected
base.lease_snapshot = lease_snapshot
base.refresh_leases = refresh_leases
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
