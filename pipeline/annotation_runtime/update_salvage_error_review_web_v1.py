"""Read-only CPU index of committed clause-salvage AI audits; no label approval."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'curated_clause_salvage_reviews_v1'
WEB = ROOT.parent.parent / 'review_web/salvage_quality_v1'
EDITION = 'clause-salvage-20261025-v1'
SCHEMA = 'curated_salvage_error_quality_web_v1'
FLAGS = dict(semantic_approval=False, human_review_completed=False,
    human_approval_automatically_granted=False, training_eligible=False,
    formal_approval=False, formal_training=False, training_entry_updated=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    # Same json_bytes format as the frozen ms_optimize/ms_visual_batch reviewer.
    return sha((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


class Reader:
    def __init__(self):
        self.cache = {}
        self.asset_cache = {}

    def raw(self, path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (str(path), stamp)
        if key not in self.cache:
            raw = path.read_bytes()
            after = path.stat()
            need(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'reference changed during read: ' + str(path))
            self.cache[key] = (raw, sha(raw))
        raw, actual = self.cache[key]
        need(expected is None or actual == expected, 'SHA mismatch: ' + str(path))
        return raw, dict(path=str(path), sha256=actual)

    def file(self, path, expected):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = (str(path), stamp)
        if key not in self.asset_cache:
            hasher = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    hasher.update(block)
            after = path.stat()
            need(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'asset changed during hash: ' + str(path))
            self.asset_cache[key] = hasher.hexdigest()
        need(self.asset_cache[key] == expected, 'asset SHA mismatch: ' + str(path))

    def checked(self, reference):
        need(Path(reference['path']).is_absolute(), 'absolute reference required')
        return json.loads(self.raw(reference['path'], reference['sha256'])[0])


def snapshot():
    state_raw = (OUT / 'state.json').read_bytes()
    progress_raw = (OUT / 'progress.json').read_bytes()
    state, progress = json.loads(state_raw), json.loads(progress_raw)
    need((OUT / 'state.json').read_bytes() == state_raw and
         (OUT / 'progress.json').read_bytes() == progress_raw, 'mutable snapshot changed')
    need(progress['state_ref']['sha256'] == sha(state_raw), 'state/progress publication in flight')
    need(state['global_revision'] == progress['global_revision'] == EDITION, 'edition mismatch')
    need(progress['processed_source_count'] == len(state['sources']), 'snapshot count mismatch')
    return state, progress, state_raw, progress_raw


def same_snapshot(state_raw, progress_raw):
    return ((OUT / 'state.json').read_bytes() == state_raw and
            (OUT / 'progress.json').read_bytes() == progress_raw)


def verify_source(key, source, reader):
    need(re.fullmatch(r'(train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key), 'unsafe source key')
    split, sid = key.split('/')
    case = OUT / EDITION / (split + '-' + hashlib.sha256(sid.encode()).hexdigest()[:24])
    need(Path(source['transaction_ref']['path']).resolve() == case / 'transaction.json', 'transaction outside exact edition/source')
    tx = reader.checked(source['transaction_ref'])
    need(tx['identity']['source_key'] == key and tx['status'] == source['status'], 'transaction state mismatch')
    for name in ('summary_ref', 'ledger_ref', 'input_ref'):
        need(Path(tx[name]['path']).resolve().parent == case, 'transaction child escaped')
    summary = reader.checked(tx['summary_ref'])
    raw, _ = reader.raw(tx['ledger_ref']['path'], tx['ledger_ref']['sha256'])
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    need(len(entries) == 1, 'source ledger scope mismatch')
    entry = entries[0]
    row = reader.checked(tx['input_ref'])
    ident = tx['identity']
    label = reader.checked(ident['label_result_ref'])
    need(label['source_key'] == key and label['global_revision'] == EDITION, 'current label edition mismatch')
    need(entry['source_id'] == sid and entry['split'] == split and entry['label_edition'] == EDITION and
         entry['status'] == tx['status'] and entry['input_row_sha256'] == ident['input_row_sha256'] == digest(row),
         'ledger source/input mismatch')
    need(row['label'] == label['label'] and entry['semantic_label_sha256'] == ident['semantic_label_sha256'] == digest(label['label']) and
         entry['label_path'] == ident['label_result_ref']['path'] and entry['label_sha256'] == ident['label_result_ref']['sha256'],
         'current label/semantic SHA mismatch')
    need(summary['review_records_ref'] == tx['ledger_ref'] and summary['label_result_ref'] == ident['label_result_ref'], 'summary binding mismatch')
    for value in (tx, summary, entry):
        for flag in ('semantic_approval', 'human_review_completed', 'training_eligible'):
            need(value.get(flag) is False, 'unexpected approval, excluded from unapproved candidate index')
    return case, tx, summary, entry, row, label


def candidate(key, source, reader):
    case, tx, summary, entry, row, label = verify_source(key, source, reader)
    need(entry['status'] in ('confirmed_error', 'uncertain'), 'not an error/uncertain terminal')
    receipt = reader.checked(entry['review_receipt'])
    binding = receipt['binding']
    review_dir = Path(entry['review_receipt']['path']).resolve().parent
    need(receipt['status'] == 'candidate' and receipt['semantic_approval'] is False and
         receipt['human_review_completed'] is False, 'real nonapproval receipt required')
    for name, expected in receipt['files_sha256'].items():
        path = (review_dir / name).resolve()
        need(path.parent == review_dir, 'receipt file escaped')
        reader.raw(path, expected)
    need(binding['source_id'] == entry['source_id'] and binding['split'] == entry['split'] and
         binding['input_row_sha256'] == entry['input_row_sha256'] and
         binding['label_sha256'] == entry['semantic_label_sha256'] and
         all(binding['label_file_ref'][k] == tx['identity']['label_result_ref'][k] for k in ('path', 'sha256')), 'receipt binding mismatch')
    final = reader.checked(entry['review_final_ref'])
    need(final['status'] == 'candidate' and final['json']['status'] == entry['status'] and
         final['json']['errors'] == entry['errors'] and final['json']['uncertainties'] == entry['uncertainties'], 'ledger/final mismatch')
    evidence_root = review_dir.parent
    evidence_raw, evidence_ref = reader.raw(evidence_root / 'evidence.json', binding['evidence_sha256'])
    evidence = json.loads(evidence_raw)
    need(evidence['binding']['input_row_sha256'] == entry['input_row_sha256'] and
         evidence['binding']['label_sha256'] == entry['semantic_label_sha256'] and
         evidence['binding']['video_sha256'] == binding['video_sha256'], 'evidence binding mismatch')
    reader.file(row['video_path'], binding['video_sha256'])
    web_raw, web_ref = reader.raw(case / 'review_web.json')
    queue = json.loads(web_raw)
    # Older committed transactions do not include a web_ref. Bind the real web
    # payload to SHA-checked ledger/label/receipt and re-read its SHA before publish.
    items = queue.get('items', []) + queue.get('optional_uncertain_items', [])
    matches = [v for v in items if v['source_id'] == entry['source_id']]
    need(len(matches) == 1, 'review_web source scope mismatch')
    item = copy.deepcopy(matches[0])
    need(item['status'] == entry['status'] and item['errors'] == entry['errors'] and
         item['new_label'] == label['label'] and item['label_sha256'] == entry['semantic_label_sha256'] and
         item['label_file_ref'] == tx['identity']['label_result_ref'] and item['review_receipt'] == entry['review_receipt'],
         'review_web differs from committed ledger/current label')
    old_ref = row['review_provenance']['original_joint_result_ref']
    old = reader.checked(old_ref)
    need(old['id'] == entry['source_id'] and old['split'] == entry['split'] and old['video_path'] == row['video_path'], 'old label lineage mismatch')
    frames = []
    for f in evidence['frames']:
        f = copy.deepcopy(f)
        f['path'] = str((evidence_root / f['file']).resolve())
        need(Path(f['path']).is_relative_to(evidence_root), 'frame path escaped')
        reader.file(f['path'], f['sha256'])
        frames.append(f)
    rois = copy.deepcopy(evidence['hand_rois'])
    for roi in rois:
        reader.file(roi['path'], roi['sha256'])
    need(sha((case / 'review_web.json').read_bytes()) == web_ref['sha256'], 'review_web changed during index construction')
    item.update(source_key=key, old_label=old['label'], old_label_ref=old_ref,
        current_label_ref=tx['identity']['label_result_ref'], current_semantic_label_sha256=entry['semantic_label_sha256'],
        review_delta=row['review_provenance']['review_delta'], original_batch_id=entry['original_batch_id'],
        original_batch_source_count=summary['original_batch_source_count'], original_batch_review_complete=False,
        uncertainties=entry['uncertainties'], evidence_frames=frames, native_hand_rois=rois,
        evidence_metadata={k:v for k,v in evidence.items() if k not in ('frames','hand_rois','source_identity','binding','probe')},
        verification_refs=dict(transaction=source['transaction_ref'], summary=tx['summary_ref'], ledger=tx['ledger_ref'],
            input=tx['input_ref'], review_web=web_ref, evidence=evidence_ref, receipt=entry['review_receipt']), **FLAGS)
    return item


def build(state, progress, state_raw, progress_raw, reader):
    errors, uncertain, deferred = [], [], []
    for key, source in sorted(state['sources'].items()):
        if source['status'] not in ('confirmed_error', 'uncertain'):
            continue
        try:
            item = candidate(key, source, reader)
            if source['status'] == 'confirmed_error':
                errors.append(item)
            elif any(v.get('severity') == 'major' for v in item['uncertainties']):
                uncertain.append(item)
        except (ValueError, KeyError, OSError, TypeError) as error:
            deferred.append(dict(source_key=key, reason=str(error), status='deferred_retry_not_completed'))
    selected, slots = [], collections.Counter()
    for item in errors:
        types = sorted({e['error_type'] for e in item['errors']})
        admitted = [t for t in types if slots[t] < 3]
        if admitted:
            slots.update(admitted)
            item['representative_error_types'] = admitted
            item['errors'] = [e for e in item['errors'] if e['error_type'] in admitted]
            selected.append(item)
    counts = collections.Counter(v['status'] for v in state['sources'].values())
    asof = dict(label_edition=EDITION, state_sha256=sha(state_raw), progress_sha256=sha(progress_raw),
                committed_source_count=len(state['sources']), cursor_offset=state['cursor']['offset'])
    return dict(schema=SCHEMA, label_edition=EDITION, updated_utc=time.strftime('%FT%TZ', time.gmtime()),
        asofversion=asof, total_sources=state['total_source_count'], expected_batches=state['expected_batch_count'],
        progress={k:progress[k] for k in ('processed_source_count','sampled_sources_reviewed','AI_error_candidate_count',
            'no_new_supported_facts_count','remaining_population_not_claimed_reviewed')}, status_counts=dict(counts),
        items=selected, optional_major_uncertain_items=uncertain[:3], verified_error_candidate_count=len(errors),
        representative_limit_per_error_type=3, representative_counts_by_error_type=dict(slots),
        deferred_retry_sources=deferred, no_new_facts_are_not_human_todos=True,
        original_batch_review_complete=False, all_batches_reviewed=False, statistical_population_error_estimate=False,
        source_addressing='state.sources.transaction_ref_only_no_population_scan', **FLAGS)


def refresh(reader):
    for _ in range(6):
        try:
            state, progress, state_raw, progress_raw = snapshot()
            value = build(state, progress, state_raw, progress_raw, reader)
            if not same_snapshot(state_raw, progress_raw):
                time.sleep(0.2)
                continue
            WEB.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=WEB, delete=False) as f:
                json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.write('\n'); f.flush(); os.fsync(f.fileno())
                temporary = Path(f.name)
            temporary.replace(WEB / 'index.json')
            print(json.dumps(dict(index=str(WEB/'index.json'), asofversion=value['asofversion'],
                representatives=len(value['items']), deferred_retry=len(value['deferred_retry_sources']))), flush=True)
            return True
        except (ValueError, KeyError, OSError, TypeError) as error:
            last_error = str(error)
            time.sleep(0.2)
    print(json.dumps(dict(status='snapshot_busy_retry_next_poll_previous_index_retained',
                          reason=locals().get('last_error', 'mutable snapshot changed'))), flush=True)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args()
    need(args.poll_seconds > 0, 'positive poll interval required')
    reader = Reader()
    while True:
        ok = refresh(reader)
        if not args.watch:
            raise SystemExit(0 if ok else 1)
        time.sleep(args.poll_seconds)
