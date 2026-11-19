"""Original byte-cursor review, exact incident recovery, two independent editions."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('verified_collector_recovery_v5', RT / 'curated_repair_collector_recovery_v1.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
from main_review_history_v1 import retained_selection, install_history_selection


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return recovery.q.sha(path)


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    tmp.replace(path)


class ByteCache:
    """Only reuse a SHA after this process verified its bytes and unchanged stat."""
    def __init__(self):
        self.files = {}

    def sha(self, path):
        path = Path(path).resolve()
        before = recovery.snapshot(path.stat())
        key = str(path)
        if key in self.files:
            identity, digest = self.files[key]
            need(before == identity, 'Previously SHA-verified file changed: ' + key)
            return digest
        h = hashlib.sha256()
        with path.open('rb') as f:
            need(recovery.snapshot(os.fstat(f.fileno())) == before, 'File replaced before hashing')
            for block in iter(lambda: f.read(8388608), b''):
                h.update(block)
            need(recovery.snapshot(os.fstat(f.fileno())) == before, 'File changed while hashing')
        need(recovery.snapshot(path.stat()) == before, 'File replaced after hashing')
        self.files[key] = (before, h.hexdigest())
        return h.hexdigest()


def checked(ref):
    need(sha(ref['path']) == ref['sha256'], 'Immutable evidence SHA changed: ' + ref['path'])
    return recovery.document(ref['path'], ref['sha256'])[0]


class Membership:
    def __init__(self, manifest):
        self.summary, self.ref = recovery.document(manifest)
        need(self.summary['total'] == 222939 and self.summary['batch_size'] == 500 and len(self.summary['batches']) == 448, 'Full authoritative 222939/448 manifest required')
        self.descriptors = {b['batch_id']: b for b in self.summary['batches']}
        need(len(self.descriptors) == 448 and sum(b['count'] for b in self.descriptors.values()) == 222939, 'Authoritative batch coverage differs')
        self.rows = {}

    def batch(self, bid):
        b = self.descriptors[bid]
        if bid not in self.rows:
            raw, ref = recovery.stable_read(b['path'])
            need(ref['sha256'] == b['sha256'], 'Authoritative full member SHA changed')
            rows = [json.loads(line) for line in raw.splitlines()]
            need(len(rows) == b['count'] and all(r['batch_id'] == bid and r['split'] == b['split'] for r in rows), 'Full batch split/member count differs')
            self.rows[bid] = {r['id']: r for r in rows}
            need(len(self.rows[bid]) == b['count'], 'Duplicate full batch member')
        return self.rows[bid]


def terminal_record(production, value, membership):
    production = Path(production).resolve()
    bid, sid, split = value['batch_id'], value['id'], value['split']
    original = membership.batch(bid)
    need(sid in original and split == membership.descriptors[bid]['split'], 'Terminal outside full authoritative membership')
    key = recovery.q.safe_key(value)
    directory = production / 'sources' / key
    terminal, _ = recovery.document(directory / 'consumer.terminal.json')
    need(terminal == value, 'Completed row differs from immutable original terminal')
    if value.get('joint_candidate_ref'):
        ref = value['joint_candidate_ref']
        path = Path(ref['path']).resolve()
        need(path.parent == directory / 'joint-results' and path.name == ref['sha256'] + '.json', 'Joint ref outside edition/source/content-SHA namespace')
        row = checked(ref)
        recovery.source_identity(row, sid, split, bid)
        need(type(row['joint_complete']) is bool and all(row.get(k) == value.get(k) for k in ('joint_complete', 'style_status', 'visual_status')), 'Joint terminal statuses differ')
        need(row['source_refs'] == original[sid]['source_refs'], 'Joint original label source refs differ')
        for name, folder in (('style_result_ref', 'style-results'),):
            if row.get(name):
                p = Path(row[name]['path']).resolve()
                need(p.parent == directory / folder and p.name == row[name]['sha256'] + '.json', 'Cross-edition source ref')
                checked(row[name])
        if row.get('visual_result_ref'):
            need(Path(row['visual_result_ref']['path']).resolve().is_relative_to(production), 'Visual ref crosses edition')
            checked(row['visual_result_ref'])
        row['label_path'], row['label_sha256'] = ref['path'], ref['sha256']
        row['label_status'] = 'completed' if row['joint_complete'] else 'partial'
        return row
    return {**original[sid], 'label_status': 'failed', 'label': {'dance_style': 'unknown', 'timed_motion_script': ''}, 'consumer_status': value['status']}


class Incident:
    def __init__(self, production, manifest, binding):
        self.production = Path(production).resolve()
        self.binding = binding
        self.evidence = recovery.Evidence(production, manifest)
        self.rows = None

    def expand(self, begin, line):
        rule = recovery.KNOWN.get(begin)
        need(rule is not None and hashlib.sha256(line).hexdigest() == rule['sha256'] and len(line) == rule['size'], 'Unknown malformed original line; do not advance cursor')
        need(self.binding['original_byte_begin'] == begin and self.binding['original_byte_end'] == begin + len(line), 'Wrong source incident span')
        raw = Path(self.binding['badline_ref']['path'])
        need(sha(raw) == self.binding['badline_ref']['sha256'] and raw.read_bytes() == line, 'Original badline evidence differs')
        if self.rows is None:
            rows = []
            for index, ref in enumerate(self.binding['hint_refs']):
                hint = checked(ref)
                sid = rule['ids'][index]
                row, refs = self.evidence.terminal(sid, rule['split'], rule['batch_id'])
                need(hint['schema'] == recovery.SCHEMA and hint['hint'] == row and hint['evidence'] == refs, 'Recovered immutable terminal evidence chain differs')
                need(hint['badline_ref']['path'] == str(raw) and hint['badline_ref']['sha256'] == self.binding['badline_ref']['sha256'], 'Recovered hint badline binding differs')
                need(hint['source_input_path'] == str(self.production / 'completed.jsonl') and hint['source_input_byte_begin'] == begin and hint['source_input_byte_end'] == begin + len(line), 'Recovered hint source byte cursor differs')
                need(hint['recovered_terminal_index'] == index and hint['recovered_terminal_count'] == 2 and hint['completed_row_sha256'] == recovery.q.digest(row), 'Recovered hint identity differs')
                rows.append(row)
            self.rows = rows
        boundary = len(line) - len(recovery.q.encoded(self.rows[1]))
        need(boundary > 0 and line[boundary:] == recovery.q.encoded(self.rows[1]) and line[:boundary] == recovery.q.encoded(self.rows[0])[:boundary], 'Two real serialized terminals do not prove exact splice')
        return self.rows


class NormalizedReader:
    def __init__(self, production, membership, incident=None, allowed_batches=None, binding=None):
        self.production = Path(production).resolve()
        self.stream = self.production / 'completed.jsonl'
        self.membership, self.incident = membership, incident
        self.allowed_batches = set(allowed_batches) if allowed_batches is not None else set(membership.descriptors)
        self.offset = 0
        self.identity = None
        self.prefix_sha = hashlib.sha256(b'').hexdigest()
        self.completed = collections.defaultdict(dict)
        self.pending = None
        self.recovered_offsets = []
        self.binding = binding

    def poll(self):
        self.pending = None
        dirty = set()
        if not self.stream.exists():
            self.pending = {'offset': self.offset, 'reason': 'edition stream not published', 'skipped': False}
            return dirty
        st = self.stream.stat()
        identity = [st.st_dev, st.st_ino]
        if self.identity is None:
            self.identity = identity
        need(identity == self.identity and st.st_size >= self.offset, 'Original stream replaced or truncated')
        if self.binding:
            need(identity == self.binding['identity'] and st.st_size >= self.binding['bytes'], 'Prepared original input identity changed')
            need(recovery.prefix_sha(self.stream, self.binding['bytes'], identity) == self.binding['sha256'], 'Prepared original source prefix changed')
        need(recovery.prefix_sha(self.stream, self.offset, identity) == self.prefix_sha, 'Previously consumed original prefix changed')
        h = recovery.prefix_hash(self.stream, self.offset, identity)
        with self.stream.open('rb') as f:
            need([os.fstat(f.fileno()).st_dev, os.fstat(f.fileno()).st_ino] == identity, 'Opened source inode changed')
            f.seek(self.offset)
            while True:
                begin = f.tell()
                line = f.readline()
                if not line or not line.endswith(b'\n'):
                    if line:
                        self.pending = {'offset': begin, 'reason': 'partial original line', 'skipped': False}
                    break
                try:
                    try:
                        values = [json.loads(line)]
                    except (ValueError, UnicodeError):
                        need(self.incident is not None, 'Unreviewed malformed edition stream')
                        recovery.stable_badline(self.stream, begin, line, identity, .05)
                        values = self.incident.expand(begin, line)
                    staged = []
                    for value in values:
                        bid, sid = value['batch_id'], value['id']
                        need(bid in self.allowed_batches, 'Completed source outside admitted edition batch range')
                        row = terminal_record(self.production, value, self.membership)
                        old = self.completed[bid].get(sid)
                        need(old is None or old == row, 'Source revision changed without explicit repair')
                        staged.append((bid, sid, row, old))
                except (ValueError, UnicodeError, OSError, KeyError, TypeError, recovery.StreamBusy) as error:
                    self.pending = {'offset': begin, 'reason': str(error), 'raw_sha256': hashlib.sha256(line).hexdigest(), 'skipped': False}
                    break
                for bid, sid, row, old in staged:
                    if old is None:
                        self.completed[bid][sid] = row
                        dirty.add(bid)
                if len(values) == 2 and begin not in self.recovered_offsets:
                    self.recovered_offsets.append(begin)
                h.update(line)
                self.offset = f.tell()
        need(recovery.prefix_sha(self.stream, self.offset, identity) == h.hexdigest(), 'Original bytes changed during scan')
        self.prefix_sha = h.hexdigest()
        return dirty


def reviewer_from_verified_local_model(review, model, out, annotator_ref, stage_ref):
    stage = checked(stage_ref)
    identity = json.loads((Path(out) / 'reviewer.model.identity.json').read_text())
    need(identity['root'] == model and set(identity['files']) == set(stage['files']), 'Original reviewer identity inventory differs')
    for name, f in stage['files'].items():
        s = Path(f['path']).stat()
        need((s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev) == (f['bytes'], f['mtime_ns'], f['inode'], f['device']), 'Local verified reviewer model stat changed')
        need(identity['files'][name]['sha256'] == f['sha256'], 'Original reviewer model SHA differs from full-copy evidence')
    content = {name: {'path': f['path'], 'sha256': f['sha256']} for name, f in identity['files'].items()}
    need(review.batch.digest({'root': model, 'files': content}) == identity['sha256'], 'Reviewer identity digest differs')
    reviewer = review.LocalReviewer(model, out, annotator_model_reference=annotator_ref,
        hand_roi_root=None, ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    reviewer._identity = identity
    reviewer._annotator_identity, reviewer._annotator_reference_ref = review.read_model_reference(annotator_ref, permit_attested=True)
    reviewer.annotator_model_sha256 = reviewer._annotator_identity['sha256']
    need(review.checkpoint_identity(identity) != review.checkpoint_identity(reviewer._annotator_identity), 'Reviewer equals annotator checkpoint')
    return reviewer


def context_progress(reader, out, edition, expected_batches, expected_count):
    statuses = []
    for bid in sorted(reader.completed):
        path = Path(out) / bid / 'summary.json'
        if path.exists():
            value = json.loads(path.read_text())
            need(value['source_count'] == len(reader.membership.batch(bid)), 'Summary full membership changed')
            statuses.append(value)
    terminal = sum(len(r) for r in reader.completed.values())
    complete = sum(s['acceptance']['review_complete'] for s in statuses)
    return {'schema': 'curated_independent_review_watch_v5', 'edition': edition,
        'total_source_count': expected_count, 'global_authoritative_source_count': 222939,
        'expected_batch_count': len(expected_batches), 'batch_review_count': len(statuses),
        'review_complete_batch_count': complete, 'terminal_source_count': terminal,
        'sampled_sources_reviewed': sum(s['sampled_reviewed_denominator'] for s in statuses),
        'AI_error_candidate_count': sum(s['confirmed_error_count'] for s in statuses),
        'awaiting_human_or_label_completion_batches': [s['batch_id'] for s in statuses if not s['acceptance']['eligible_for_training']],
        'all_batches_reviewed': complete == len(expected_batches),
        'all_source_labels_completed': terminal == expected_count and len(statuses) == len(expected_batches) and all(s['acceptance']['all_source_labels_completed'] for s in statuses),
        'consumer_stream_offset': reader.offset, 'consumer_stream_identity': reader.identity,
        'consumer_stream_prefix_sha256': reader.prefix_sha, 'completed_stream': str(reader.stream),
        'pending_stream_record': reader.pending, 'recovered_badline_offsets': reader.recovered_offsets,
        'one_resident_reviewer': True, 'malformed_records_skipped': False,
        'human_approval_automatically_granted': False, 'training_entry_updated': False}


def main():
    ap = argparse.ArgumentParser()
    for flag in ('manifest', 'results-root', 'out', 'model', 'ms-source', 'annotator-model-reference', 'source-binding'):
        ap.add_argument('--' + flag, required=True)
    ap.add_argument('--additional-results-root')
    ap.add_argument('--additional-out')
    ap.add_argument('--additional-manifest')
    ap.add_argument('--poll-seconds', type=float, default=60)
    a = ap.parse_args()
    need(a.poll_seconds > 0, 'Positive poll interval required')
    binding = json.loads(Path(a.source_binding).read_text())
    cache = ByteCache()
    recovery.q.sha = cache.sha
    membership = Membership(a.manifest)
    incident = Incident(a.results_root, a.manifest, binding['incident'])
    reader = NormalizedReader(a.results_root, membership, incident=incident, binding=binding['main_prefix'])
    # Verify original source/incident before any VLM call or output refresh.
    reader.poll()
    need(recovery.BAD_OFFSET in reader.recovered_offsets, 'Real known splice not fully recovered')
    sys.path.insert(0, a.ms_source)
    import ms_batch_review as review
    review.batch.sha256 = cache.sha
    reviewer = reviewer_from_verified_local_model(review, a.model, a.out, a.annotator_model_reference, binding['model_stage_ref'])
    contexts = [(reader, Path(a.out), 'main-production-v1', set(membership.descriptors), 222939)]
    if any((a.additional_results_root, a.additional_out, a.additional_manifest)):
        need(all((a.additional_results_root, a.additional_out, a.additional_manifest)), 'Complete independent edition arguments required')
        extra = json.loads(Path(a.additional_manifest).read_text())
        need(extra['original_summary'] == membership.ref and extra['selected_source_count'] == 10000 and extra['source_range'] == {'start': 21000, 'end': 31000}, 'Only admitted actual collaborative v2 10000 edition allowed')
        bids = {b['batch_id'] for b in extra['selected_batches']}
        need(len(bids) == 20 and all(membership.descriptors[b['batch_id']] == b for b in extra['selected_batches']), 'Additional full original batches differ')
        extra_reader = NormalizedReader(a.additional_results_root, membership, allowed_batches=bids)
        contexts.append((extra_reader, Path(a.additional_out), 'collaborative-v2', bids, 10000))
    base_sampling = review.sampling_plan
    pending_dirty = {id(current): set(current.completed) for current, _, _, _, _ in contexts}
    while True:
        values = []
        for current, out, edition, bids, count in contexts:
            out.mkdir(parents=True, exist_ok=True)
            pending_dirty[id(current)] |= current.poll()
            dirty = pending_dirty[id(current)]
            if not current.pending:
                review.sampling_plan = base_sampling
                install_history_selection(review, out)
                reviewer.hand_roi_root = str(current.production / 'roi')
                for bid in sorted(dirty):
                    originals = membership.batch(bid)
                    rows = [current.completed[bid].get(sid, {**r, 'label_status': 'pending'}) for sid, r in originals.items()]
                    normalized = [review.normalized_record(r) for r in rows]
                    retained_selection(review, out, bid, normalized)
                    result = review.review_batch_ready(rows, out, reviewer, batch_id=bid, expected_source_ids=list(originals), sample_count=25, risk_count=5)
                    print(json.dumps({'edition': edition, 'batch': bid, 'labels_terminal': len(current.completed[bid]),
                        'source_count': len(originals), 'sources_reviewed': result['sampled_reviewed_denominator'],
                        'AI_error_candidates': result['confirmed_error_count'], 'state': result['state'],
                        'eligible_for_training': result['acceptance']['eligible_for_training']}), flush=True)
                pending_dirty[id(current)] = set()
            value = context_progress(current, out, edition, bids, count)
            value['source_binding'] = a.source_binding
            atomic(out / 'progress.json', value)
            if value['all_batches_reviewed'] and value['all_source_labels_completed'] and not current.pending:
                atomic(out / 'run.summary.json', value)
            values.append(value)
        if all(v['all_batches_reviewed'] and v['all_source_labels_completed'] and not v['pending_stream_record'] for v in values):
            break
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
