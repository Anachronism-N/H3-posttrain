"""Preserve fixed-batch immutable review history without changing frozen reviewer code."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def retained_selection(review, out, batch_id, records):
    root = Path(out) / batch_id
    by_id = {review.source_id(r): r for r in records}
    retained, errors = set(), set()
    summary = root / 'summary.json'
    if summary.exists():
        old = json.loads(summary.read_text())
        retained.update(old['reviewed_source_ids'])
        errors.update(old['acceptance']['unresolved_confirmed_error_source_ids'])
    for p in (root / 'isolation').glob('*.json'):
        v = json.loads(p.read_text())
        errors.add(v['source_id'])
        retained.add(v['source_id'])
        if sha(v['review_receipt']['path']) != v['review_receipt']['sha256']:
            raise ValueError('historical isolation receipt changed')
    bound = set()
    for p in (root / 'sources').glob('*/*/review/receipt.json'):
        receipt = json.loads(p.read_text())
        if receipt['status'] != 'candidate':
            continue
        binding = receipt['binding']
        sid = binding['source_id']
        if sid not in by_id:
            raise ValueError('historical receipt outside fixed batch membership')
        row = by_id[sid]
        if binding['input_row_sha256'] != review.batch.digest(row):
            raise ValueError('historical input revision requires explicit repair: ' + sid)
        if binding['label_sha256'] != review.batch.digest(review.label_payload(row)):
            raise ValueError('historical label revision requires explicit repair: ' + sid)
        ref = binding['label_file_ref']
        if not ref or ref['path'] != row.get('label_path') or sha(ref['path']) != ref['sha256']:
            raise ValueError('historical label reference changed: ' + sid)
        for name, digest in receipt['files_sha256'].items():
            path = p.parent / name
            if not path.resolve().is_relative_to(p.parent.resolve()) or sha(path) != digest:
                raise ValueError('historical transaction artifact changed: ' + sid)
        final = json.loads((p.parent / 'final.json').read_text())
        if final['status'] != receipt['status'] or final['json']['status'] not in ('pass', 'confirmed_error', 'uncertain'):
            raise ValueError('historical final/receipt status mismatch: ' + sid)
        if final['json']['status'] == 'confirmed_error':
            errors.add(sid)
        bound.add(sid)
        retained.add(sid)
    if not retained <= bound:
        raise ValueError('historical review receipt missing; preserve old summaries and fail closed')
    if not errors <= retained or not all(review.label_reviewable(by_id[sid]) for sid in errors):
        raise ValueError('historical errors cannot disappear while labels pending')
    return sorted(retained), sorted(errors)


def install_history_selection(review, out):
    original = review.sampling_plan
    def sampling(records, batch_id, **kwargs):
        plan = original(records, batch_id, **kwargs)
        retained, errors = retained_selection(review, out, batch_id, records)
        baseline = list(plan['risk_source_ids'])
        plan['risk_source_ids'] = list(dict.fromkeys(baseline + retained))
        plan['selected_source_ids'] = list(dict.fromkeys(plan['random_source_ids'] + plan['risk_source_ids']))
        plan['historical_selection'] = {'retained_source_ids': retained, 'error_source_ids': errors,
            'baseline_risk_source_ids': baseline,
            'extra_source_ids': [sid for sid in retained if sid not in plan['random_source_ids'] + baseline]}
        return plan
    review.sampling_plan = sampling
