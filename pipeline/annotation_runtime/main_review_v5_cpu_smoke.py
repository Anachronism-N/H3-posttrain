"""Real original normalized ledger + native full-batch cache resume, no GPU/model."""
import hashlib
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
ROOT = RT.parent


def run(worker, binding, reader, cache, out):
    sys.path.insert(0, str(RT / 'source_curated_review_v2'))
    import ms_batch_review as review
    review.batch.sha256 = cache.sha
    original_out = ROOT / 'curated_independent_reviews_v1'
    bid = 'train-b000000'
    originals = reader.membership.batch(bid)
    raw_rows = [reader.completed[bid].get(sid, {**row, 'label_status': 'pending'}) for sid, row in originals.items()]
    rows = [review.normalized_record(r) for r in raw_rows]
    retained, errors = worker.retained_selection(review, original_out, bid, rows)
    worker.need(len(rows) == 500 and len(retained) >= 3 and errors, 'Full real batch and old AI errors required')
    original_membership = original_out / bid / 'batch.membership.json'
    membership_value = json.loads(original_membership.read_text())
    baseline = review.sampling_plan(rows, bid, sample_count=25, risk_count=5)
    worker.need(baseline['policy'] == membership_value['policy'], 'Original frozen sampling policy differs')
    worker.install_history_selection(review, original_out)
    selection = review.sampling_plan(rows, bid, sample_count=25, risk_count=5)
    worker.need(selection['policy'] == membership_value['policy'] and set(errors) <= set(selection['selected_source_ids']), 'History changed immutable policy or dropped old errors')
    pending_rows = [review.normalized_record({**originals[review.source_id(row)], 'label_status': 'pending'}) if review.source_id(row) == errors[0] else row for row in rows]
    try:
        worker.retained_selection(review, original_out, bid, pending_rows)
    except ValueError:
        pass
    else:
        raise AssertionError('Historical error pending must fail closed')
    reviewer = worker.reviewer_from_verified_local_model(review, '/opt/models/Qwen3-VL-8B-Instruct', original_out,
        binding['annotator_model_ref']['path'], binding['model_stage_ref'])
    reviewer.hand_roi_root = str(reader.production / 'roi')
    def forbidden_runtime():
        raise AssertionError('CPU full batch must not load VLM or issue new generation')
    reviewer.runtime = forbidden_runtime
    protected = {}
    for name in ('summary.json', 'batch.plan.json', 'batch.membership.json', 'review_web.json'):
        p = original_out / bid / name
        if p.exists():
            protected[str(p)] = worker.sha(p)
    for p in (original_out / bid / 'isolation').glob('*.json'):
        protected[str(p)] = worker.sha(p)
    old = json.loads((original_out / bid / 'summary.json').read_text())
    # Preserve every old reviewed terminal; other full members remain real pending
    # in the CPU-only sandbox so there is no new model invocation.
    cpu_rows = [row if review.source_id(row) in retained else review.normalized_record({**originals[review.source_id(row)], 'label_status': 'pending'}) for row in rows]
    test_out = Path(out)
    root = test_out / bid
    root.mkdir(parents=True, exist_ok=False)
    (root / 'batch.membership.json').write_bytes(original_membership.read_bytes())
    before_membership = worker.sha(root / 'batch.membership.json')
    calls = []
    receipts = {}
    def cached(row, source_root, *, prepare_only=False):
        sid = review.source_id(row)
        worker.need(sid in retained and not prepare_only, 'Only existing real native transactions may run in smoke')
        actual = original_out / bid / 'sources' / Path(source_root).relative_to(root / 'sources')
        receipt_path = actual / 'review/receipt.json'
        receipt_before = worker.sha(receipt_path)
        value = reviewer(row, actual)
        worker.need(value['review_receipt']['sha256'] == receipt_before and worker.sha(receipt_path) == receipt_before, 'Existing score receipt mutated')
        calls.append(sid)
        receipts[sid] = {'source_id': sid, 'status': value['status'], 'receipt_sha256': receipt_before,
                         'input_row_sha256': review.batch.digest(row), 'label_sha256': value['label_sha256']}
        return value
    result = review.review_batch_ready(cpu_rows, test_out, cached, batch_id=bid,
        expected_source_ids=list(originals), sample_count=25, risk_count=5)
    worker.need(worker.sha(root / 'batch.membership.json') == before_membership, 'Native full batch mutated copied immutable membership')
    worker.need(set(errors) <= set(result['acceptance']['unresolved_confirmed_error_source_ids']), 'Native full batch lost old AI errors')
    worker.need(result['source_count'] == 500 and result['acceptance']['eligible_for_training'] is False, 'CPU pending batch must not approve training')
    for path, digest in protected.items():
        worker.need(worker.sha(path) == digest, 'CPU proof changed original persistent review output')
    sample = list(dict.fromkeys(errors[:1] + sorted(receipts)))[:3]
    worker.need(len(sample) == 3 and all(sid in receipts for sid in sample), 'Three real existing receipt scores required')
    negative_root = test_out / 'stream_negative'
    negative_root.mkdir()
    stream = negative_root / 'completed.jsonl'
    stream.write_bytes(b'{"partial":')
    negative = worker.NormalizedReader(negative_root, reader.membership)
    worker.need(not negative.poll() and negative.offset == 0 and negative.pending, 'Halfline cursor must not advance')
    stream.write_bytes(b'{"partial":\n')
    worker.need(not negative.poll() and negative.offset == 0 and negative.pending, 'Unknown malformed line must not advance')
    verified = {str(p): digest for p, (_, digest) in cache.files.items() if p.endswith('/train.csv') or p.endswith('/val.csv') or p.endswith('/test.csv')}
    worker.need(verified, 'This process must actually SHA verify source CSV content, not skip it')
    return {'schema': 'main_review_v5_cpu_smoke_v1', 'source_count': 500,
        'full_rows_sha256': review.batch.digest(rows), 'full_native_batch_path_verified': True,
        'immutable_original_membership_policy_unchanged': True, 'original_outputs_unchanged': True,
        'retained_historical_receipt_count': len(retained), 'retained_historical_error_count': len(errors),
        'native_cache_reused_source_count': len(set(calls)), 'native_candidate_error_count': result['confirmed_error_count'],
        'previous_summary_error_count': old['confirmed_error_count'], 'three_case_receipt_reuses': [receipts[sid] for sid in sample],
        'historical_missing_terminal_fails_closed': True, 'halfline_and_unknownbad_offset_retained': True,
        'normalized_source_offset': reader.offset, 'normalized_source_prefix_sha256': reader.prefix_sha,
        'recovered_badline_offsets': reader.recovered_offsets, 'normalized_source_pending': reader.pending,
        'same_process_actual_source_csv_SHA_verified': verified, 'CPU_sandbox_out': str(test_out),
        'sample_count': 3, 'VLM_model_loaded': False, 'new_VLM_calls': 0, 'training_approved': False,
        'worker_sha256': worker.sha(RT / 'main_review_watch_v5.py'),
        'history_helper_sha256': worker.sha(RT / 'main_review_history_v1.py')}
