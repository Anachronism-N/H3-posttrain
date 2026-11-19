"""Read-only old-state/source/receipt validation; no model calls or state migration."""
import copy
import hashlib
import json
from pathlib import Path


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str) and 'byte_offset' not in value:
            yield value
        else:
            for item in value.values():
                yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def verify_committed_source(worker, key, source, review, manifest, notice):
    need, checked, sha = worker.need, worker.checked, worker.sha
    transaction = checked(source['transaction_ref'])
    row = checked(transaction['input_ref'])
    summary = checked(transaction['summary_ref'])
    entry = checked(transaction['ledger_ref'])
    identity = transaction['identity']
    need(transaction['schema'] in (worker.PHASE, worker.V3_PHASE, worker.V2_PHASE, worker.LEGACY_PHASE),
         'Committed transaction phase changed')
    need(identity['source_key'] == key and key == row['split'] + '/' + row['id']
         and row == review.normalized_record(row)
         and identity['input_row_sha256'] == review.batch.digest(row)
         and identity['semantic_label_sha256'] == review.batch.digest(review.label_payload(row))
         and identity['label_result_ref'] == worker.ref(row['label_path'])
         and source['status'] == transaction['status'] == entry['status']
         and source['review_required'] is transaction['review_required']
         and source['label_complete'] is transaction['label_complete']
         and source['imported_previous_review_ref'] == transaction['imported_previous_review_ref'],
         'Committed original input/label/state identity changed')
    need(worker.digest(notice) == identity['notice_sha256']
         and notice['joint_candidate_ref'] == identity['label_result_ref'],
         'Committed source notice/reference changed')
    original = manifest.members(row['batch_id']).get(row['id'])
    need(original == row['review_provenance']['original_manifest_record'], 'Committed full-manifest member changed')
    binding = row.get('source_binding', {}).get('source_binding', {})
    for kind in ('video', 'audio'):
        need(sha(row[kind + '_path']) == binding[kind + '_sha256'], 'Committed actual source media SHA changed')
    candidate = checked(identity['label_result_ref'])
    need(candidate['label'] == row['label'] and candidate.get('model_calls') == 0
         and candidate.get('attempt_budget_consumed') is False
         and candidate.get('style_label_changed') is False, 'Committed zero-VLM source/label changed')
    for reference in references([row['source_refs'], row['review_provenance']]):
        need(sha(reference['path']) == reference['sha256'], 'Committed original source reference changed')
    for value in (transaction, summary, entry):
        need(all(value.get(k) is False for k in ('semantic_approval', 'human_review_completed', 'training_eligible')),
             'Committed historical audit acquired approval')
    if entry['status'] == 'source_evidence_violation':
        for value in (source, transaction, summary, entry):
            need(value.get('label_complete') is False, 'Isolated invalid source acquired label completeness')
        for value in (transaction, summary, entry):
            need(value.get('review_performed') is False and value.get('model_calls_performed') == 0,
                 'Isolated invalid source acquired a model review')
        need(source['review_required'] is False and transaction['review_required'] is False
             and entry.get('review_receipt') is None and entry.get('review_final_ref') is None
             and summary['sampled_reviewed_denominator'] == 0,
             'Isolated invalid source acquired audit receipt or denominator')
        violation_ref = transaction['source_evidence_violation_ref']
        need(violation_ref == summary['source_evidence_violation_ref'], 'Isolation evidence reference changed')
        violation = checked(violation_ref)
        try:
            worker.bind_v2.bind_notice(notice, manifest, worker.w)
        except worker.bind_v2.SourceEvidenceViolation as error:
            need(error.details == violation == row['review_provenance']['source_evidence_violation'],
                 'Original source evidence violation no longer reproduces')
            expected = review.normalized_record(error.row)
            expected['source_refs']['source_evidence_violation_checked_original_artifacts'] = [
                item['checked_original_artifact_refs'] for item in violation['violations']]
            for field in ('original_notice_ref', 'original_notice_stream_ref'):
                expected['review_provenance'][field] = row['review_provenance'][field]
            disagreement = expected['review_provenance'].get('eligibility_disagreement')
            if disagreement:
                disagreement['original_notice_ref'] = row['review_provenance']['original_notice_ref']
            need(expected == row, 'Full isolated source input differs from checked original candidate')
        else:
            need(False, 'Isolated source no longer has checked part/frame violation')
        audit = None
    else:
        audit = worker.verify_audit(row, entry, review)
    if entry['status'] in ('pass', 'confirmed_error', 'uncertain'):
        need(audit is not None, 'Committed real audit receipt missing')
    return dict(source_key=key, transaction_ref=source['transaction_ref'],
        input_ref=transaction['input_ref'], label_result_ref=identity['label_result_ref'],
        summary_ref=transaction['summary_ref'], ledger_ref=transaction['ledger_ref'],
        review_receipt=entry.get('review_receipt'), status=entry['status'], model_call_performed=False)


def _resume_state_proof(worker, review, manifest):
    need = worker.need
    original_ref = worker.ref(worker.OUT / 'state.json')
    state = worker.checked(original_ref)
    need(state['schema'] in (worker.LEGACY_PHASE, worker.V2_PHASE, worker.V3_PHASE, worker.PHASE) and state['global_revision'] == worker.EDITION
         and state['input_stream'] == str(worker.STREAM) and state['total_source_count'] == 222939
         and state['expected_batch_count'] == 448
         and all(state.get(k) is False for k in worker.FLAGS), 'Exact old state edition/scope/flags required')
    cursor = worker.AppendCursor(worker.STREAM, copy.deepcopy(state['cursor']))
    cursor.next()
    offset = state['cursor']['offset']
    with worker.STREAM.open('rb') as stream:
        prefix = stream.read(offset)
    need(len(prefix) == offset and (not prefix or prefix.endswith(bytes([10])))
         and state['cursor'].get('committed_prefix_sha256', hashlib.sha256(prefix).hexdigest()) == hashlib.sha256(prefix).hexdigest(),
         'Consumed stream prefix changed/incomplete')
    seen, records = {}, []
    for line in prefix.splitlines():
        notice = json.loads(line)
        key = notice['source_key']
        need(key in state['sources'], 'Consumed notice lacks committed state transaction')
        if key in seen:
            need(seen[key] == worker.digest(notice), 'Consumed duplicate source notice changed')
        else:
            records.append(verify_committed_source(worker, key, state['sources'][key], review, manifest, notice))
        seen[key] = worker.digest(notice)
    need(set(seen) == set(state['sources']), 'Old seen-state/prefix source sets differ')
    need(worker.sha(original_ref['path']) == original_ref['sha256'], 'Old state changed during resume proof')
    receipt_count = sum(r['review_receipt'] is not None for r in records)
    next_refs = []
    with worker.STREAM.open('rb') as stream:
        stream.seek(offset)
        for _ in range(2):
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(bytes([10])):
                break
            notice = json.loads(line)
            for reference in (notice['joint_candidate_ref'], notice['visual_result_ref']):
                worker.checked(reference)
            next_refs.append(dict(path=str(worker.STREAM), byte_offset=start, byte_length=len(line),
                sha256=hashlib.sha256(line).hexdigest(), source_key=notice['source_key'],
                joint_candidate_ref=notice['joint_candidate_ref'], visual_result_ref=notice['visual_result_ref']))
    return dict(schema='curated_clause_salvage_review_resume_proof_v4', queued_boundary_records=next_refs, original_state_ref=original_ref,
        persistent_state_schema=state['schema'], transaction_schemas=sorted({worker.checked(v['transaction_ref'])['schema'] for v in state['sources'].values()}),
        original_cursor=copy.deepcopy(state['cursor']), consumed_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        consumed_prefix_bytes=offset, consumed_notice_count=len(prefix.splitlines()),
        processed_source_count=len(records), real_audit_receipt_count=receipt_count,
        committed_records=records, all_original_receipt_files_SHA_verified=True,
        all_current_label_and_original_source_SHA_verified=True, state_migrated=False,
        model_calls_performed=0, original_attempt_budgets_unchanged=True, **worker.FLAGS)


def resume_state_proof(worker, review, manifest):
    """Hash each unchanged file once, retaining exact stat+SHA inventory for admission."""
    actual_sha = worker.sha
    files = {}
    def identity(path):
        st = Path(path).stat()
        return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    def cached_sha(path):
        path = str(Path(path).resolve())
        before = identity(path)
        if path in files:
            worker.need(before == files[path]['stat'], 'Source changed during full SHA resume verification')
            return files[path]['sha256']
        digest = actual_sha(path)
        worker.need(identity(path) == before, 'Source changed while hashing resume evidence')
        files[path] = dict(path=path, sha256=digest, stat=before)
        return digest
    original_bind_sha, original_support_sha = worker.bind_v2.sha, worker.w.sha
    worker.sha = worker.bind_v2.sha = worker.w.sha = cached_sha
    try:
        proof = _resume_state_proof(worker, review, manifest)
        for path, reference in files.items():
            worker.need(identity(path) == reference['stat'], 'Verified resume file changed before proof commit')
        proof['verified_file_inventory'] = [files[path] for path in sorted(files)]
        return proof
    finally:
        worker.sha, worker.bind_v2.sha, worker.w.sha = actual_sha, original_bind_sha, original_support_sha


def install_sha_cache(worker):
    actual_sha, files = worker.sha, {}
    def cached_sha(path):
        path = str(Path(path).resolve())
        if Path(path).parent == worker.OUT and Path(path).name in ('state.json', 'progress.json', 'run.summary.json'):
            return actual_sha(path)
        st = Path(path).stat()
        identity = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        if path in files:
            worker.need(identity == files[path][0], 'Previously SHA-verified source changed during resident review')
            return files[path][1]
        value = actual_sha(path)
        after = Path(path).stat()
        worker.need(identity == [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns],
            'Source changed during resident content hashing')
        files[path] = (identity, value)
        return value
    worker.sha = cached_sha
    worker.bind_v2.sha = cached_sha
    worker.w.sha = cached_sha
    return cached_sha
