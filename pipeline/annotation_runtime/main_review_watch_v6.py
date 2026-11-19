"""Finite raw-byte index, batch-lazy terminal verification, independent editions."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_main_review_v5_for_v6', RT / 'main_review_watch_v5.py')
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)
ROOT = RT.parent
need, sha, atomic = v5.need, v5.sha, v5.atomic
Membership, Incident, ByteCache = v5.Membership, v5.Incident, v5.ByteCache
terminal_record = v5.terminal_record
retained_selection, install_history_selection = v5.retained_selection, v5.install_history_selection
reviewer_from_verified_local_model = v5.reviewer_from_verified_local_model
NormalizedReader = v5.NormalizedReader
recovery = v5.recovery


class RawIndex:
    def __init__(self, production, membership, incident=None, allowed_batches=None, binding=None):
        self.production = Path(production).resolve()
        self.stream = self.production / 'completed.jsonl'
        self.membership, self.incident, self.binding = membership, incident, binding
        self.allowed_batches = set(allowed_batches) if allowed_batches is not None else set(membership.descriptors)
        self.identity = None
        self.indexed_offset = 0
        self.indexed_sha = hashlib.sha256(b'').hexdigest()
        self.raw = collections.defaultdict(dict)
        self.lines = []
        self.completed = collections.defaultdict(dict)
        self.verified_keys = set()
        self.pending = None
        self.batch_pending = {}
        self.recovered_offsets = []
        self.offset = 0
        self.prefix_sha = hashlib.sha256(b'').hexdigest()
        self.line_frontier = 0

    def check_prefix(self):
        if self.identity is None:
            return
        recovery.original_stream(self.stream, self.identity, self.indexed_offset)
        need(recovery.prefix_sha(self.stream, self.indexed_offset, self.identity) == self.indexed_sha, 'Previously indexed original bytes changed')
        if self.binding:
            need(self.binding['path'] == str(self.stream) and self.binding['identity'] == self.identity, 'Prepared original identity differs')
            need(recovery.prefix_sha(self.stream, self.binding['bytes'], self.identity) == self.binding['sha256'], 'Prepared original prefix changed')

    def scan(self):
        self.pending = None
        dirty = set()
        if not self.stream.exists():
            self.pending = {'offset': self.indexed_offset, 'reason': 'stream_not_published', 'skipped': False}
            return dirty
        stat = self.stream.stat()
        identity = [stat.st_dev, stat.st_ino]
        if self.identity is None:
            self.identity = identity
        need(self.identity == identity and stat.st_size >= self.indexed_offset, 'Original stream replaced/truncated')
        self.check_prefix()
        # A finite snapshot; appends are handled in the next tick.
        limit = stat.st_size
        with self.stream.open('rb') as f:
            need([os.fstat(f.fileno()).st_dev, os.fstat(f.fileno()).st_ino] == identity, 'Opened source identity differs')
            f.seek(self.indexed_offset)
            data = f.read(limit - self.indexed_offset)
        begin = self.indexed_offset
        consumed = bytearray()
        for line in data.splitlines(keepends=True):
            if not line.endswith(b'\n'):
                self.pending = {'offset': begin, 'reason': 'partial_original_line', 'skipped': False}
                break
            try:
                values = [json.loads(line)]
            except (ValueError, UnicodeError):
                if self.incident is None or begin not in recovery.KNOWN:
                    self.pending = {'offset': begin, 'reason': 'unknown_malformed_line', 'raw_sha256': hashlib.sha256(line).hexdigest(), 'skipped': False}
                    break
                recovery.stable_badline(self.stream, begin, line, identity, .05)
                values = self.incident.expand(begin, line)
            staged = []
            for value in values:
                need(isinstance(value, dict) and all(k in value for k in ('id', 'split', 'batch_id', 'status')), 'Raw terminal schema differs')
                sid, split, bid = value['id'], value['split'], value['batch_id']
                need(isinstance(sid, str) and sid and bid in self.allowed_batches, 'Raw source outside admitted edition')
                originals = self.membership.batch(bid)
                need(sid in originals and split == self.membership.descriptors[bid]['split'], 'Raw hint outside full authoritative membership')
                old = self.raw[bid].get(sid)
                need(old is None or old['hint'] == value, 'Raw source revision changed without explicit repair')
                staged.append((bid, sid, value, old))
            for bid, sid, value, old in staged:
                if old is None:
                    self.raw[bid][sid] = {'hint': value, 'byte_begin': begin, 'byte_end': begin + len(line), 'raw_sha256': hashlib.sha256(line).hexdigest()}
                    dirty.add(bid)
            self.lines.append((begin, begin + len(line), [(b, s) for b, s, _, _ in staged]))
            if len(values) == 2 and begin not in self.recovered_offsets:
                self.recovered_offsets.append(begin)
            consumed.extend(line)
            begin += len(line)
        # Never replace the initially read byte hash with a later changed prefix.
        h = recovery.prefix_hash(self.stream, self.indexed_offset, identity)
        need(h.hexdigest() == self.indexed_sha, 'Previously indexed bytes changed during scan')
        h.update(consumed)
        need(recovery.prefix_sha(self.stream, begin, identity) == h.hexdigest(), 'Read raw snapshot changed before index commit')
        self.indexed_offset, self.indexed_sha = begin, h.hexdigest()
        return dirty

    def verify(self, bid, ids=None):
        self.check_prefix()
        wanted = list(self.raw[bid]) if ids is None else list(ids)
        staged = []
        for sid in wanted:
            need(sid in self.raw[bid], 'Historical receipt source has no current raw terminal: ' + sid)
            value = self.raw[bid][sid]['hint']
            try:
                row = terminal_record(self.production, value, self.membership)
            except FileNotFoundError as error:
                self.batch_pending[bid] = {'source_id': sid, 'reason': 'terminal_or_evidence_not_published', 'detail': str(error)}
                return False
            old = self.completed[bid].get(sid)
            need(old is None or old == row, 'Verified source revision changed without explicit repair')
            staged.append((sid, row))
        self.check_prefix()
        for sid, row in staged:
            self.completed[bid][sid] = row
            self.verified_keys.add((bid, sid))
        self.batch_pending.pop(bid, None)
        while self.line_frontier < len(self.lines):
            _, end, keys = self.lines[self.line_frontier]
            if not all(k in self.verified_keys for k in keys):
                break
            self.offset = end
            self.line_frontier += 1
        self.prefix_sha = recovery.prefix_sha(self.stream, self.offset, self.identity)
        return True


def historical_ids(out, bid):
    root = Path(out) / bid
    ids = set()
    summary = root / 'summary.json'
    if summary.exists():
        old = json.loads(summary.read_text())
        ids.update(old['reviewed_source_ids'])
        ids.update(old['acceptance']['unresolved_confirmed_error_source_ids'])
    for p in (root / 'isolation').glob('*.json'):
        ids.add(json.loads(p.read_text())['source_id'])
    for p in (root / 'sources').glob('*/*/review/receipt.json'):
        v = json.loads(p.read_text())
        if v['status'] == 'candidate':
            ids.add(v['binding']['source_id'])
    need(ids, 'Original historical batch receipts required')
    return sorted(ids)


def progress(index, out, edition, bids, total):
    value = v5.context_progress(index, out, edition, bids, total)
    value['schema'] = 'curated_independent_review_watch_v6'
    value['all_source_labels_completed'] = value['all_source_labels_completed'] and not index.pending and not index.batch_pending
    value['indexed_raw_source_count'] = sum(len(r) for r in index.raw.values())
    value['indexed_raw_byte_offset'] = index.indexed_offset
    value['indexed_raw_prefix_sha256'] = index.indexed_sha
    value['verified_terminal_source_count'] = sum(len(r) for r in index.completed.values())
    value['actually_reviewed_source_count'] = value['sampled_sources_reviewed']
    value['unverified_raw_source_count'] = value['indexed_raw_source_count'] - value['verified_terminal_source_count']
    value['batch_pending_evidence'] = index.batch_pending
    value['raw_index_is_not_terminal_approval'] = True
    return value


def main():
    ap = argparse.ArgumentParser()
    for name in ('manifest', 'results-root', 'out', 'model', 'ms-source', 'annotator-model-reference', 'source-binding'):
        ap.add_argument('--' + name, required=True)
    for name in ('additional-results-root', 'additional-out', 'additional-manifest'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--poll-seconds', type=float, default=60)
    a = ap.parse_args()
    need(a.poll_seconds > 0, 'Positive poll interval required')
    binding = json.loads(Path(a.source_binding).read_text())
    cache = ByteCache()
    recovery.q.sha = cache.sha
    membership = Membership(a.manifest)
    incident = Incident(a.results_root, a.manifest, binding['incident'])
    main_index = RawIndex(a.results_root, membership, incident=incident, binding=binding['main_prefix'])
    main_index.scan()
    need(recovery.BAD_OFFSET in main_index.recovered_offsets, 'Known incident not fully indexed from real evidence')
    need(main_index.verify('train-b000000', historical_ids(a.out, 'train-b000000')), 'Historical evidence not fully published')
    sys.path.insert(0, a.ms_source)
    import ms_batch_review as review
    review.batch.sha256 = cache.sha
    reviewer = reviewer_from_verified_local_model(review, a.model, a.out, a.annotator_model_reference, binding['model_stage_ref'])
    extra = v5.checked(binding['additional_edition']['manifest'])
    need(extra['original_summary'] == membership.ref and extra['selected_source_count'] == 10000 and extra['source_range'] == {'start': 21000, 'end': 31000}, 'Additional authoritative range differs')
    bids = {b['batch_id'] for b in extra['selected_batches']}
    need(len(bids) == 20 and all(membership.descriptors[b['batch_id']] == b for b in extra['selected_batches']), 'Additional original full batch descriptors differ')
    need(binding['additional_edition']['results_root'] == a.additional_results_root and binding['additional_edition']['out'] == a.additional_out and binding['additional_edition']['manifest']['path'] == a.additional_manifest, 'Additional edition binding differs')
    extra_index = RawIndex(a.additional_results_root, membership, allowed_batches=bids)
    contexts = [(main_index, Path(a.out), 'main-production-v1', set(membership.descriptors), 222939),
                (extra_index, Path(a.additional_out), 'collaborative-v2', bids, 10000)]
    pending_dirty = {id(main_index): set(main_index.raw), id(extra_index): set()}
    baseline = review.sampling_plan
    while True:
        values = []
        for index, out, edition, expected, total in contexts:
            out.mkdir(parents=True, exist_ok=True)
            pending_dirty[id(index)] |= index.scan()
            review.sampling_plan = baseline
            install_history_selection(review, out)
            reviewer.hand_roi_root = str(index.production / 'roi')
            for bid in sorted(pending_dirty[id(index)]):
                if not index.verify(bid):
                    continue
                originals = membership.batch(bid)
                rows = [index.completed[bid].get(sid, {**row, 'label_status': 'pending'}) for sid, row in originals.items()]
                retained_selection(review, out, bid, [review.normalized_record(r) for r in rows])
                result = review.review_batch_ready(rows, out, reviewer, batch_id=bid, expected_source_ids=list(originals), sample_count=25, risk_count=5)
                print(json.dumps({'edition': edition, 'batch': bid, 'verified_terminal_sources': len(index.completed[bid]),
                    'sources_reviewed': result['sampled_reviewed_denominator'], 'AI_error_candidates': result['confirmed_error_count'],
                    'state': result['state'], 'eligible_for_training': result['acceptance']['eligible_for_training']}), flush=True)
                pending_dirty[id(index)].remove(bid)
            value = progress(index, out, edition, expected, total)
            value['source_binding'] = a.source_binding
            atomic(out / 'progress.json', value)
            if value['all_batches_reviewed'] and value['all_source_labels_completed'] and not index.pending and not index.batch_pending:
                atomic(out / 'run.summary.json', value)
            values.append(value)
        if all(v['all_batches_reviewed'] and v['all_source_labels_completed'] and not v['pending_stream_record'] and not v['batch_pending_evidence'] for v in values):
            break
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
