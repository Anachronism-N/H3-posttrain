"""Finite independent risk review of exactly three IO recovery editions."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
ROOT = RT.parent
RECOVERY = ROOT / 'curated222939_io_failure_recovery_v1'
OUT = ROOT / 'curated_io_failure_recovery_reviews3_v1'
EDITION = 'io-failure-recovery500-v1'
SELECTED = RT / 'curated_io_failure_recovery500_selected_v1.json'
SELECTED_SHA = '4a3d1cffa04c1f4c0f2a1432c80acef5846fcac7a1a2533e8458095e205b620e'
REQUESTS = RT / 'curated_io_failure_recovery500_revision_requests_v1.json'
REQUESTS_SHA = '0fc3aea2520d29a18ccf8153a8125a8dcd80301abfbeae106b2a20adee59a527'
MANIFEST = ROOT / 'curated222939_inputs_v1/manifest.summary.json'
MANIFEST_SHA = 'a1eb129cf8557b4adeb4e11e00ace334ecb1575e43a587d665e33f09410e2bfd'
SUPPORT = RT / 'curated_review_extension_watch_v1.py'
SUPPORT_SHA = '410cd5384a36664bedd2c6eebd4bd4ffe5024653510219307cdaa8e75bc72f23'
KEYS = ('id', 'split', 'batch_id', 'source_ordinal')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NotReady(RuntimeError):
    pass


def support():
    need(sha(SUPPORT) == SUPPORT_SHA, 'Frozen G2 review support changed')
    spec = importlib.util.spec_from_file_location('recovery_review3_support', SUPPORT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def source_rows():
    need(sha(SELECTED) == SELECTED_SHA and sha(REQUESTS) == REQUESTS_SHA
         and sha(MANIFEST) == MANIFEST_SHA, 'Recovery selected/requests/original manifest changed')
    selected, requests = load(SELECTED), load(REQUESTS)
    need(len(selected) == 3 and [r['source_ordinal'] for r in selected] == [500, 501, 502]
         and len({r['id'] for r in selected}) == 3 and requests['source_count_unique'] == 3
         and [{k: r[k] for k in KEYS} for r in requests['requests']] == selected,
         'Exactly three original unique recovery sources required')
    w = support()
    manifest = w.Manifest(MANIFEST)
    members = manifest.members('train-b000001')
    need(manifest.descriptors['train-b000001']['count'] == 500, 'Original batch500 denominator changed')
    originals = []
    for selected_row, request in zip(selected, requests['requests']):
        original = members[selected_row['id']]
        need({k: original[k] for k in KEYS} == selected_row, 'Original source ordinal/member differs')
        oldref = request['old_terminal_ref']
        need(Path(oldref['path']).resolve() == ROOT / 'curated222939_production_v1/sources' /
             original['split'] / original['id'] / 'producer.terminal.json', 'Old failed producer path differs')
        old = w.checked(oldref)
        need(old['status'] == 'failed' and {k: old[k] for k in KEYS} == selected_row
             and request['revision_request']['new_out'] == str(RECOVERY)
             and request['revision_request']['old_terminal_preserved'] is True,
             'Old failed producer SHA/status or recovery request differs')
        originals.append((original, request))
    return originals


def ready_records():
    """Use exact terminal paths plus completed stream, never wait for batch500."""
    w = support()
    originals = source_rows()
    stream = RECOVERY / 'completed.jsonl'
    if not stream.exists():
        raise NotReady('Waiting recovery completed stream and all three consumer terminals')
    raw = stream.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise NotReady('Waiting complete recovery stream line')
    notices = {}
    for line in raw.splitlines():
        if not line:
            continue
        value = json.loads(line)
        need(value['id'] in {r['id'] for r, _ in originals}, 'Unknown source in recovery stream')
        if value['id'] in notices:
            need(notices[value['id']] == value, 'Recovery duplicate terminal changed')
        notices[value['id']] = value
    refs = [ref(SELECTED), ref(REQUESTS), ref(MANIFEST), ref(stream)]
    records = []
    for original, request in originals:
        sid = original['id']
        directory = RECOVERY / 'sources' / original['split'] / sid
        consumer, producer = directory / 'consumer.terminal.json', directory / 'producer.terminal.json'
        if not consumer.exists() or not producer.exists() or sid not in notices:
            raise NotReady('Waiting exact third recovery consumer terminal: ' + sid)
        terminal, source = load(consumer), load(producer)
        need(terminal == notices[sid] and all(terminal.get(k) == original[k] for k in ('id', 'split', 'batch_id')),
             'Consumer terminal/completed/member binding differs')
        need(all(source.get(k) == original[k] for k in KEYS) and source['status'] in ('ready', 'failed'),
             'Recovered producer source identity differs')
        references = dict(recovery_revision_requests=ref(REQUESTS),
            old_failed_producer_terminal=copy.deepcopy(request['old_terminal_ref']),
            recovered_producer_terminal=ref(producer), recovered_consumer_terminal=ref(consumer))
        refs.extend(references.values())
        joint = terminal.get('joint_candidate_ref')
        if joint:
            need(source['status'] == 'ready' and Path(joint['path']).resolve() == directory / 'joint-results' /
                 (joint['sha256'] + '.json'), 'Recovery content-addressed JOINT path differs')
            candidate = w.checked(joint)
            need(candidate.get('edition') == EDITION and candidate.get('recovery_revision_requests_ref') == ref(REQUESTS)
                 and candidate.get('old_failed_producer_terminal_ref') == request['old_terminal_ref']
                 and terminal['status'] in ('joint_candidate', 'incomplete_candidate')
                 and (terminal['status'] == 'joint_candidate') == candidate['joint_complete'],
                 'Recovery label edition/requests/oldfailed/complete status differs')
            row = w.bound_row(original, candidate, joint, edition=EDITION, origin_root=RECOVERY, stream_path=stream)
            refs.append(joint)
        else:
            need(terminal['status'] in ('failed', 'producer_failed'), 'Missing JOINT label for successful recovery terminal')
            row = {**copy.deepcopy(original), 'label_status': 'failed', 'consumer_status': terminal['status'],
                'label': dict(dance_style='unknown', timed_motion_script=''),
                'review_provenance': dict(label_edition=EDITION, original_roi_root=str(RECOVERY / 'roi'),
                    original_manifest_record=copy.deepcopy(original), input_stream=str(stream), **w.FLAGS)}
        row['source_refs'] = {**(row.get('source_refs') or {}), 'io_recovery_review_lineage': references}
        row['review_provenance'].update(recovery_revision_requests_ref=ref(REQUESTS),
            old_failed_producer_terminal_ref=request['old_terminal_ref'],
            recovered_consumer_terminal_ref=ref(consumer), original_batch_source_count=500,
            old_review_scores_reused=False)
        records.append(row)
    need(len(records) == len(notices) == 3, 'Exactly three recovery terminals must complete before review')
    need(sha(stream) == hashlib.sha256(raw).hexdigest(), 'Recovery stream changed during gate')
    return records, refs


def review_records(records, out, reviewer, review):
    w = support()
    summaries, audits = [], {}
    def audit(row, root, **kwargs):
        result = reviewer(row, root, **kwargs)
        audits[review.batch.digest(row)] = copy.deepcopy(result)
        return result
    for row in records:
        bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
        value = review.review_batch_ready([row], out / EDITION, audit, batch_id=bid,
            expected_source_ids=[row['id']], sample_count=1, risk_count=0)
        value = w.gate_summary(value, edition=EDITION, provenance=row['review_provenance'])
        value.update(original_batch_id=row['batch_id'], original_batch_source_count=500,
            original_batch_review_complete=False, total_source_count=222939, every_recovery_source_is_risk_reviewed=True)
        w.publish(out / EDITION / bid, value, [row], review, audits)
        summaries.append(value)
    value = dict(schema='curated_io_recovery_review3_v1', label_edition=EDITION,
        total_source_count=222939, original_batch_source_count=500, selected_terminal_count=len(records),
        selected_source_ids=[r['id'] for r in records], processed_source_count=len(summaries),
        sampled_sources_reviewed=sum(s['sampled_reviewed_denominator'] for s in summaries),
        AI_error_candidate_count=sum(s['confirmed_error_count'] for s in summaries),
        label_incomplete_source_ids=[r['id'] for r in records if not review.label_complete(r)],
        review_record_refs=[s['review_records_ref'] for s in summaries],
        original_batch_review_complete=False, all_batches_reviewed=False, source_records_consumed=True,
        human_resolution_stream=None, **w.FLAGS)
    w.atomic(out / 'run.summary.json', value)
    w.atomic(out / 'progress.json', value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-reference', required=True)
    p.add_argument('--annotator-model-reference', required=True)
    p.add_argument('--ms-source', required=True)
    a = p.parse_args()
    need(Path(a.out).resolve() == OUT and Path(a.ms_source).resolve() == RT / 'source_curated_review_v2',
         'Only independent three-source recovery output/frozen reviewer allowed')
    rows, _ = ready_records()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'reviewer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, a.ms_source)
        import ms_batch_review as review
        local = review.LocalReviewer(a.model, OUT, model_reference=a.model_reference,
            annotator_model_reference=a.annotator_model_reference, hand_roi_root=str(RECOVERY / 'roi'),
            ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
        print(json.dumps(review_records(rows, OUT, local, review), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
