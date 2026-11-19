"""Register one evidence-bound missing incomplete terminal without changing journal cursor."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
BAR = RT.parent
SID = 'PULHoYjXDehOgnaw'
KEY = 'train/' + SID
TERMINAL_SHA = '8d39b4b8191322b2869deb2c392edcb97d9de933ca69dca7c4abcefa927463c2'


def module(name, path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen pending recovery dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = module('pending_missing_original_evidence_v2', RT / 'curated_repair_collector_recovery_v1.py',
    'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd')
q = module('pending_missing_current_classifier_v2', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py',
    '0ff1f304db79170095c97cc44e3d6e7dff38a4fd658691ad502ba49ae5c60548')


def recover(register=False):
    production = BAR / 'curated222939_production_v1'
    out = BAR / 'curated222939_repair_v1'
    terminal_path = production / 'sources' / KEY / 'consumer.terminal.json'
    if q.sha(terminal_path) != TERMINAL_SHA:
        raise q.SourceChanged('Exact pending terminal changed')
    row = q.load(terminal_path)
    if row['id'] != SID or row['split'] != 'train' or row['status'] != 'incomplete_candidate' or row['joint_complete'] is not False:
        raise q.SourceChanged('Pending source must remain an incomplete original candidate')
    journal = production / 'completed.jsonl'
    st = journal.stat()
    identity = [st.st_dev, st.st_ino]
    with journal.open('rb') as stream:
        raw = stream.read(st.st_size)
    end = raw.rfind(b'\n') + 1
    raw = raw[:end]
    if SID.encode() in raw:
        raise q.SourceChanged('Pending source is already present in this journal snapshot')
    if r.prefix_sha(journal, end, identity) != hashlib.sha256(raw).hexdigest():
        raise q.SourceChanged('Original source prefix changed during absence check')
    evidence = r.Evidence(production, BAR / 'curated222939_inputs_v1/manifest.summary.json')
    verified, refs = evidence.terminal(SID, 'train', row['batch_id'])
    r.same(verified, row, 'Original pending terminal changed after full evidence verification')
    job = q.classify(row, production)
    job.update(global_revision='repair-20261025-v1', queue_root=str(out))
    q.verify_job(job)
    request = dict(schema='pending_terminal_journal_recovery_v2', source_key=KEY,
        original_terminal_ref=dict(path=str(terminal_path), sha256=TERMINAL_SHA),
        hint=row, original_evidence=refs,
        original_journal_prefix=dict(path=str(journal), identity=identity, bytes=end,
            prefix_sha256=hashlib.sha256(raw).hexdigest()),
        absent_from_verified_prefix=True, original_byte_offset=None,
        data_complete_candidate=False, original_journal_modified=False,
        original_cursor_modified=False, retry_budget_reset=False,
        independent_review_required=True, **q.FLAGS)
    if not register:
        return dict(source_key=KEY, status='source_preflight_passed', stages=job['stages'],
            complete_candidate=False, retry_reserved=False, model_calls=0, **q.FLAGS)
    # Serialize canonical registration with the original collector. This source
    # is not inserted into its seen-state, because it has no original byte row.
    with (out / 'collector.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = out / 'collector.recovery.state.json'
        state_raw = state_path.read_bytes()
        root = out / 'pending-terminal-recovery-v2' / KEY
        request_path = root / ('request-' + q.digest(request) + '.json')
        q.save_once(request_path, request)
        job_path = out / 'jobs' / (KEY + '.json')
        if job_path.exists():
            existing = q.load(job_path)
            r.same(existing['completed_row_sha256'], job['completed_row_sha256'], 'Existing canonical source job changed')
            r.same(existing['original_result_ref'], job['original_result_ref'], 'Existing original candidate changed')
            job = existing
            created = False
        else:
            job['pending_terminal_recovery_request_ref'] = q.identity(request_path)
            q.save_once(job_path, job)
            created = True
        notice_path = out / 'announced' / (KEY + '.json')
        if not notice_path.exists():
            q.append(out / 'ready.jsonl', dict(source_key=KEY, job_ref=q.identity(job_path),
                stage=job['stage'], stages=job['stages'], global_revision=job['global_revision']))
            q.save_once(notice_path, dict(job_ref=q.identity(job_path)))
        r.same(state_path.read_bytes(), state_raw, 'Original collector cursor/state changed during registration')
        receipt = dict(schema='pending_terminal_journal_registration_v2', source_key=KEY,
            request_ref=q.identity(request_path), canonical_job_ref=q.identity(job_path),
            original_collector_state_sha256=hashlib.sha256(state_raw).hexdigest(),
            collector_state_or_cursor_modified=False, original_journal_modified=False,
            complete_candidate=False, retry_reserved_by_registration=False,
            retry_budget_reset=False, model_calls=0, **q.FLAGS)
        q.save_once(root / ('registered-' + q.digest(receipt) + '.json'), receipt)
        return dict(source_key=KEY, status='queued_original_pending_source', new_job_created=created,
            stages=job['stages'], complete_candidate=False, model_calls=0, **q.FLAGS)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--register', action='store_true')
    args = p.parse_args()
    print(json.dumps(recover(args.register), ensure_ascii=False), flush=True)
