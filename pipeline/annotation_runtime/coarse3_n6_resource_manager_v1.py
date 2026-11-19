"""Single N6 GPU3 wrapper of the unchanged pidfd/FLOCK/restore manager."""
from collections import Counter
import os
from pathlib import Path
import select
import stat
import time
import prepare_coarse3_n6_v1 as c

base = c.base_module()
original_snapshot, original_refresh = base.lease_snapshot, base.refresh_leases
active_config = None
proof_written = False


def selected(cfg):
    plan = c.checked(cfg['plan'])
    c.need([s['source_id'] for s in plan['sources']] == list(c.IDS), 'complete fixed-three plan required')
    return list(c.IDS), 222939


def validate(cfg):
    global active_config
    plan = c.source_plan()
    c.verify_config(cfg, plan)
    c.check_node()
    c.verify_local_model(cfg)
    prior = c.prior_safe(cfg['transition_ref'], base)
    capture = c.checked(cfg['holder_capture'])
    c.need(capture.get('signals_sent') is False and 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9,
           'fresh read-only holder capture expired')
    c.need(capture['holders'] == cfg['holders'] and capture['lease_snapshot_ns'] == cfg['lease_snapshot_ns']
           and capture['budget']['headroom'] >= 160 * 2**30, 'fresh holder snapshot/headroom changed')
    c.require_unreviewed(plan)
    active_config = cfg
    fresh = c.fresh_holder(base, prior)
    c.need(fresh['lease_snapshot_ns'] == cfg['lease_snapshot_ns'], 'captured exact lease changed')
    base.check_holder(cfg['holders'][0], fresh['holders'][0], 3, fresh['holders'][0]['uuid'], base.apps())
    active_config = cfg


def lease_snapshot(path):
    value = original_snapshot(path)
    c.need(active_config is not None and str(path) == '/tmp/od_reserve_3', 'only exact captured GPU3 lease allowed')
    expected = active_config['lease_snapshot_ns'][str(path)]
    c.verify_lease_value(expected)
    c.need(c.lease_value(value) == expected, 'full nanosecond lease snapshot changed')
    return value


def refresh_leases(entries, on_error=None):
    global proof_written
    original_refresh(entries, on_error)
    if proof_written or active_config is None or len(entries) != 1:
        return
    e = entries[0]
    if not e['released'] or e['leasefd'] is None or not select.select([e['fd']], [], [], 0)[0]:
        return
    if any(app['uuid'] == e['uuid'] for app in base.apps()):
        return
    pid = os.getpid()
    s, fs = e['reserve'].lstat(), os.fstat(e['leasefd'])
    c.need(e['gpu'] == 3 and stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_size == 0
           and (s.st_dev, s.st_ino) == (fs.st_dev, fs.st_ino)
           and c.fd_inode_owned(pid, s.st_dev, s.st_ino) and c.proc_lock_owned(pid, '/tmp/ms12k_gpu_3.lock'),
           'actual parent lease descriptor and kernel FLOCK ownership required')
    launch_ref = c.ref(active_config['launch_path'])
    queue = Path(active_config['queue'])
    c.save_once(queue / 'actual_leases_held.json', {
        'schema': 'coarse3_n6_actual_leases_held_v1', 'manager_identity': base.identity(pid),
        'config_ref': launch_ref, 'resource_admission_ref': c.ref(queue / 'admission.json'),
        'node': 'N6', 'gpus': [3], 'observed_unix_ns': time.time_ns(),
        'leases': [{'gpu': 3, 'path': str(e['reserve']), 'stat': c.lease_value(s), 'uuid': e['uuid']}],
        'historical_proof_current_FD_kernel_FLOCK_and_heartbeat_required': True})
    proof_written = True


def summaries(cfg, out):
    report_path = out / 'finite.stage.summary.json'
    report = c.load(report_path)
    plan = c.checked(cfg['plan'])
    m = c.worker_module()
    lib = m.libraries()
    summary = c.checked(report['independent_summary_ref'])
    ledger_ref = report['independent_ledger_ref']
    c.need(ledger_ref['path'] == str(out / 'independent.ledger.jsonl') and c.sha(ledger_ref['path']) == ledger_ref['sha256'],
           'new current independent ledger SHA required')
    rows = [lib.q.strict_json(line) if hasattr(lib.q, 'strict_json') else c.json.loads(line)
            for line in Path(ledger_ref['path']).read_bytes().splitlines() if line.strip()]
    c.need(report.get('schema') == 'coarse3_n6_finite_stage_summary_v1'
           and report.get('launch_config_ref') == c.ref(cfg['launch_path'])
           and report.get('population_ref') == plan['population_ref']
           and report.get('source_count') == 3 and report.get('terminal_count') == 3
           and report.get('semantic_approval') is False and report.get('training_eligible') is False
           and len(rows) == 3 and [r['source_id'] for r in rows] == list(c.IDS), 'finite exact three-terminal current summary required')
    c.need(report['independent_summary_ref']['path'] == str(out / 'independent.summary.json')
           and summary['population_ref'] == plan['population_ref'] and summary['population_source_count'] == 3
           and summary['maximum_review_calls'] == 3 and type(summary['review_dispatches']) is int
           and 0 <= summary['review_dispatches'] <= 3 and all(summary.get(k) is False for k in m.NO_APPROVAL),
           'finite budget/no-quality-approval binding changed')
    c.need(summary['statuses'] == dict(Counter(r['status'] for r in rows))
           and summary['review_dispatches'] == sum(r['review_dispatches'] for r in rows)
           and report['review_dispatches'] == summary['review_dispatches']
           and summary['sampled_reviewed_denominator'] == sum(r['status'] != 'pending' for r in rows)
           and summary['independent_sample_review_completed'] is all(r['status'] != 'pending' for r in rows),
           'finite semantic denominator/status/dispatch counters changed')
    identity = c.verify_local_model(cfg)
    reviewer = m.make_reviewer(lib, c.MODEL, out, model_reference=c.MODEL_REF,
                               ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    with lib.review_library():
        c.need(reviewer.identity()['sha256'] == identity['sha256'], 'actual reviewer checkpoint changed')
        for source, row in zip(plan['sources'], rows):
            root = Path(source['root'])
            c.need(row['status'] in ('pass', 'confirmed_error', 'uncertain', 'pending')
                   and all(row.get(k) is False for k in m.NO_APPROVAL), 'invalid terminal or automatic approval')
            c.need(type(row.get('review_dispatches')) is int and row['review_dispatches'] in (0, 1), 'source review retry detected')
            if row['status'] == 'pending':
                continue  # A failed/partial review is preserved pending, never retried.
            prepared = c.load(root / 'independent.prepare.receipt.json')
            for key in ('candidate_ref', 'source_job_ref', 'recovery_proof_ref', 'recovery_code_ref',
                        'source_transaction_refs', 'canonical_budget_ref', 'input_row_sha256', 'evidence_ref'):
                c.need(row.get(key) == prepared.get(key), 'terminal original-six-file/recovery/label evidence binding changed: ' + key)
            input_row, evidence = c.load(root / 'input.row.json'), c.load(root / 'evidence.json')
            m.verify_evidence(lib, evidence, input_row, root, evidence['probe'])
            checked, receipt = lib.old.verify_review_receipt(root, input_row, evidence, reviewer)
            c.need(receipt == row['review_receipt'] and checked['status'] == row['status'], 'current six-file score/ledger mismatch')
    for reference in report['score_refs']:
        c.need(Path(reference['path']).resolve().is_relative_to(out) and c.sha(reference['path']) == reference['sha256'], 'current score SHA changed')
    c.need(report['score_refs'] == [r['review_receipt'] for r in rows if r['status'] != 'pending'], 'exact new scores only; old scores excluded')
    c.prior_safe(cfg['transition_ref'], base)
    return {str(report_path): {'summary': {'stage': 'run', 'selected_source_count': 3,
            'processed_source_count': 3, 'total_source_count': 222939, 'plan_sha256': cfg['plan']['sha256']},
            'actual_finite_result': report, 'semantic_approval': False, 'partial_outputs_preserved': True}}


base.AUTHORIZED_PAIRS = {'N6': ((3,),)}
base.validate_pair = validate
base.selected_sources = selected
base.lease_snapshot = lease_snapshot
base.refresh_leases = refresh_leases
base.check_model = c.verify_local_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
