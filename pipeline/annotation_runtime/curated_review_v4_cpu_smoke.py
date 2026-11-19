"""Three real receipt reuses from a full 500-member source batch; no VLM runtime."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
ROOT = RT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--completed-stream')
    a = ap.parse_args()
    sys.path.insert(0, str(RT / 'source_curated_review_v2'))
    import ms_batch_review as review
    from main_review_history_v1 import retained_selection
    worker_sha_start = hashlib.sha256((RT / 'curated_review_watch_v4.py').read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('watch4', RT / 'curated_review_watch_v4.py')
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    spec = importlib.util.spec_from_file_location('resource4', RT / 'curated_review_resource_manager_v4.py')
    resource = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resource)
    cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
    for path, expected in cfg['source_sha256'].items():
        assert watch.sha(path) == expected, 'frozen source changed: ' + path
    resource.check_model(cfg)
    manifest = json.loads((ROOT / 'curated222939_inputs_v1/manifest.summary.json').read_text())
    membership = watch.Membership(ROOT / 'curated222939_inputs_v1/manifest.summary.json')
    descriptor = next(b for b in manifest['batches'] if b['batch_id'] == 'train-b000000')
    assert watch.sha(descriptor['path']) == descriptor['sha256']
    originals = {r['id']: r for r in (json.loads(l) for l in Path(descriptor['path']).read_text().splitlines() if l)}
    assert len(originals) == 500
    stream = Path(a.completed_stream) if a.completed_stream else ROOT / 'curated222939_production_v1/completed.jsonl'
    completed = {}
    bad_offset = None
    with stream.open('rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line or not line.endswith(b'\n'):
                break
            try:
                terminal = json.loads(line)
            except json.JSONDecodeError:
                assert a.completed_stream is None and offset == 532233, 'unexpected broken smoke source'
                bad_offset = offset
                break
            if terminal['batch_id'] == descriptor['batch_id']:
                row = watch.terminal_record(ROOT / 'curated222939_production_v1', terminal, membership)
                sid = terminal['id']
                assert sid not in completed or completed[sid] == row
                completed[sid] = row
    rows = [review.normalized_record(completed.get(sid, {**original, 'label_status': 'pending'})) for sid, original in sorted(originals.items())]
    out = ROOT / 'curated_independent_reviews_v1'
    retained, errors = retained_selection(review, out, descriptor['batch_id'], rows)
    assert retained and errors
    actual_retained, actual_errors = watch.retained_selection(review, out, descriptor['batch_id'], rows)
    assert set(retained) <= set(actual_retained) and set(errors) <= set(actual_errors), 'worker dropped historical receipt/errors'
    missing = [review.normalized_record({**originals[review.source_id(row)], 'label_status': 'pending'}) if review.source_id(row) == errors[0] else row for row in rows]
    try:
        watch.retained_selection(review, out, descriptor['batch_id'], missing)
    except ValueError:
        pass
    else:
        raise AssertionError('worker must fail closed if a historical error has no current terminal/label')
    watch.install_history_selection(review, out)
    selection = review.sampling_plan(rows, descriptor['batch_id'], sample_count=25, risk_count=5)
    assert set(errors) <= set(selection['risk_source_ids'])
    reviewer = review.LocalReviewer('/opt/models/Qwen3-VL-8B-Instruct', out,
        annotator_model_reference=cfg['argv'][cfg['argv'].index('--annotator-model-reference') + 1],
        hand_roi_root=str(ROOT / 'curated222939_production_v1/roi'), ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    reviewer._identity = json.loads((out / 'reviewer.model.identity.json').read_text())
    stage = json.loads(Path(cfg['modelref']['path']).read_text())
    assert set(reviewer._identity['files']) == set(stage['files'])
    assert all(reviewer._identity['files'][name]['sha256'] == value['sha256'] for name, value in stage['files'].items())
    reviewer._annotator_identity, reviewer._annotator_reference_ref = review.read_model_reference(reviewer.annotator_model_reference, permit_attested=True)
    reviewer.annotator_model_sha256 = reviewer._annotator_identity['sha256']
    assert review.checkpoint_identity(reviewer._identity) != review.checkpoint_identity(reviewer._annotator_identity)
    def forbidden_runtime():
        raise AssertionError('CPU smoke must not request any VLM/model load')
    reviewer.runtime = forbidden_runtime
    by_id = {review.source_id(r): r for r in rows}
    sample_ids = list(dict.fromkeys(errors[:1] + retained))[:3]
    assert len(sample_ids) == 3
    results = []
    for sid in sample_ids:
        row = by_id[sid]
        revision = review.batch.digest({'label': review.label_payload(row), 'label_path': row.get('label_path'),
                                      'input_row_sha256': review.batch.digest(row), 'annotator_model_sha256': row.get('annotator_model_sha256')})[:16]
        source_root = out / descriptor['batch_id'] / 'sources' / hashlib.sha256(sid.encode()).hexdigest()[:24] / revision
        receipt = source_root / 'review/receipt.json'
        before = watch.sha(receipt)
        evidence = review.read_json(source_root / 'evidence.json')
        old_receipt = review.read_json(receipt)
        binding = old_receipt['binding']
        assert binding['input_row_sha256'] == review.batch.digest(row)
        assert binding['label_sha256'] == review.batch.digest(review.label_payload(row))
        assert binding['code_sha256'] == reviewer.code['sha256'] and binding['model_sha256'] == reviewer._identity['sha256']
        assert binding['annotator_model_sha256'] == reviewer.annotator_model_sha256
        assert binding['evidence_sha256'] == watch.sha(source_root / 'evidence.json')
        assert review.read_json(source_root / 'evidence.receipt.json')['sha256'] == binding['evidence_sha256']
        review.batch.verify_source(evidence['source_identity'])
        for ref in [binding['label_file_ref']] + binding['source_refs']:
            assert review.batch.snapshot(Path(ref['path'])) == ref['stat'], 'attested source/label reference stat changed'
        for frame in evidence['frames']:
            assert watch.sha(source_root / frame['file']) == frame['sha256']
        for crop in evidence['hand_rois']:
            assert watch.sha(crop['path']) == crop['sha256']
        request = review.review_request(row, evidence, source_root, reviewer.model)
        from types import SimpleNamespace
        result = review.bar.transactions([{'root': source_root / 'review', 'request': request,
            'validator': lambda v: review.validate_review(v, evidence), 'binding': binding}], forbidden_runtime,
            SimpleNamespace(stage='run', batch_size=1))[str(source_root / 'review')]
        assert result['status'] == 'candidate' and watch.sha(receipt) == before
        value = {**result['json'], 'review_receipt': {'sha256': before}, 'label_sha256': binding['label_sha256']}
        assert value['status'] in ('pass', 'confirmed_error', 'uncertain')
        results.append({'source_id': sid, 'status': value['status'], 'receipt_sha256': before,
                        'label_sha256': value['label_sha256'], 'input_row_sha256': review.batch.digest(row)})
    acceptance = review.batch_acceptance({'sampling': selection, 'records': rows}, {}, expansion_complete=False)
    assert acceptance['eligible_for_training'] is False
    assert watch.sha(RT / 'curated_review_watch_v4.py') == worker_sha_start, 'worker changed during CPU smoke'
    value = {'schema': 'curated_review_v4_cpu_smoke_v1', 'source_count': len(rows), 'full_rows_sha256': review.batch.digest(rows),
             'terminal_source_count': len(completed), 'valid_source_stream': str(stream), 'known_bad_line_offset_not_repaired_or_skipped': bad_offset,
             'retained_historical_receipt_count': len(retained), 'retained_historical_error_count': len(errors),
             'actual_worker_history_selection_verified': True, 'historical_missing_terminal_fails_closed': True,
             'sample_count': 3, 'receipt_reuses': results, 'VLM_model_loaded': False, 'new_VLM_calls': 0,
             'native_transaction_reused_from_current_row_request_binding': True,
             'provenance_csv_SHA_rehashed_in_smoke': False, 'attested_source_reference_stats_verified': True,
             'training_approved': False, 'old_summary_web_isolations_not_cleared': True,
             'worker_sha256': watch.sha(RT / 'curated_review_watch_v4.py'),
             'history_helper_sha256': watch.sha(RT / 'main_review_history_v1.py')}
    with Path(a.receipt).open('x') as f:
        json.dump(value, f, indent=2)
    print(json.dumps(value), flush=True)

if __name__ == '__main__':
    main()
