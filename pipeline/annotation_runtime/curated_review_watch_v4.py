"""Resume the full fixed-membership reviewer from a terminal-verified shadow journal."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('review_shadow_evidence', RT/'completed_shadow_recovery_v1.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return evidence.sha_bytes(evidence.read_stable(path))


def prepared_stream(stream, production, plan_ref=None):
    stream, production = Path(stream).resolve(), Path(production).resolve()
    need(stream.name == 'completed.shadow.jsonl' and not stream.is_relative_to(production), 'Independent shadow stream required')
    shadow = stream.parent
    actual_ref = evidence.ref(shadow/'recovery.plan.json')
    need(plan_ref is None or actual_ref == plan_ref, 'Immutable recovery plan changed')
    plan = json.loads(evidence.checked(actual_ref))
    need(plan['schema'] == evidence.SCHEMA and plan['source_path'] == str(production/'completed.jsonl'), 'Shadow recovery source lineage differs')
    state = evidence.load(shadow/'shadow.state.json')
    need(state.get('recovery_plan_ref') == actual_ref and state['source_path'] == plan['source_path'] and state['source_identity'] == plan['source_identity'], 'Shadow has not committed the bound recovery plan')
    need(state['source_offset'] >= plan['bad_offset'] + plan['bad_raw_length'], 'Demonstrated malformed line not yet recovered')
    for name in ('manifest_summary_ref', 'batch_manifest_ref', 'terminal_index_ref', 'bad_raw_ref'):
        evidence.checked(plan[name])
    for terminal in plan['terminal_evidence']:
        row = json.loads(evidence.checked(terminal['terminal_ref']))
        need(evidence.terminal_evidence(production, row) == terminal, 'Recovered terminal evidence differs')
    return actual_ref, state


class Membership:
    def __init__(self, manifest):
        self.summary = evidence.load(manifest)
        need(self.summary['total'] == 222939 and self.summary['batch_size'] == 500 and len(self.summary['batches']) == 448, 'Authoritative full curated manifest required')
        self.descriptors = {b['batch_id']: b for b in self.summary['batches']}
        need(len(self.descriptors) == 448 and sum(b['count'] for b in self.descriptors.values()) == 222939, 'Manifest coverage differs')
        self.by_split = collections.Counter()
        for b in self.descriptors.values():
            need(b['split'] in ('train', 'val', 'test') and 0 < b['count'] <= 500, 'Invalid fixed batch descriptor')
            self.by_split[b['split']] += b['count']
        self.rows = {}
        self.source_batches = {}

    def batch(self, bid):
        need(bid in self.descriptors, 'Unknown batch in completed journal')
        if bid not in self.rows:
            b = self.descriptors[bid]
            rows = [json.loads(line) for line in evidence.checked({'path': b['path'], 'sha256': b['sha256']}).splitlines()]
            need(len(rows) == b['count'] and all(r['batch_id'] == bid and r['split'] == b['split'] for r in rows), 'Full fixed batch membership/split differs')
            members = {r['id']: r for r in rows}
            need(len(members) == len(rows), 'Duplicate source in fixed batch')
            for sid in members:
                need(sid not in self.source_batches or self.source_batches[sid] == bid, 'Source reused across fixed batches')
                self.source_batches[sid] = bid
            self.rows[bid] = members
        return self.rows[bid]


def attach_terminal(value, originals, descriptor):
    need(value['id'] in originals and value['batch_id'] == descriptor['batch_id'] and value['split'] == descriptor['split'], 'Source not in authoritative fixed batch/split')
    if value.get('joint_candidate_ref'):
        ref = value['joint_candidate_ref']
        row = json.loads(evidence.checked(ref))
        need(row['id'] == value['id'] and row['batch_id'] == value['batch_id'] and row['split'] == value['split'] and type(row['joint_complete']) is bool, 'Joint candidate source identity/status differs')
        row['label_path'] = ref['path']
        row['label_sha256'] = ref['sha256']
        row['label_status'] = 'completed' if row['joint_complete'] else 'partial'
        return row
    return {**originals[value['id']], 'label_status': 'failed', 'label': {'dance_style': 'unknown', 'timed_motion_script': ''}, 'consumer_status': value['status']}


sys.path.insert(0, str(RT))
from main_review_history_v1 import retained_selection, install_history_selection


def terminal_record(production, value, membership):
    need(isinstance(value, dict), 'Completed journal record must be object')
    bid = value['batch_id']
    originals = membership.batch(bid)
    need(value['id'] in originals and value['split'] == membership.descriptors[bid]['split'], 'Source not in authoritative fixed batch/split')
    evidence.terminal_evidence(Path(production).resolve(), value)
    if value.get('joint_candidate_ref'):
        ref = value['joint_candidate_ref']
        row = json.loads(evidence.checked(ref))
        need(row['id'] == value['id'] and row['batch_id'] == bid and row['split'] == value['split'] and type(row['joint_complete']) is bool, 'Joint candidate source identity/status differs')
        row['label_path'] = ref['path']
        row['label_sha256'] = ref['sha256']
        row['label_status'] = 'completed' if row['joint_complete'] else 'partial'
        return row
    return {**originals[value['id']], 'label_status': 'failed', 'label': {'dance_style': 'unknown', 'timed_motion_script': ''}, 'consumer_status': value['status']}


class VerifiedReader:
    def __init__(self, stream, production, membership, plan_ref):
        self.stream, self.production = Path(stream).resolve(), Path(production).resolve()
        self.membership, self.plan_ref = membership, plan_ref
        self.offset = 0
        self.identity = None
        self.completed = collections.defaultdict(dict)
        self.committed_offset = 0
        self.committed_sha = None
        self.pending = None

    def prepared(self):
        return prepared_stream(self.stream, self.production, self.plan_ref)

    def poll(self):
        dirty = set()
        self.pending = None
        try:
            _, state = self.prepared()
            with self.stream.open('rb') as stream:
                stat = self.stream.stat()
                identity = (stat.st_dev, stat.st_ino)
                if self.identity is None:
                    self.identity = identity
                need(identity == self.identity and stat.st_size >= self.offset, 'Verified stream replaced/truncated')
                committed = state['shadow_offset']
                need(committed >= self.committed_offset and stat.st_size >= committed, 'Committed shadow prefix truncated')
                data = stream.read(committed)
                need(len(data) == committed and evidence.sha_bytes(data) == state['shadow_sha256'], 'Shadow prefix differs from committed state')
                if self.committed_offset:
                    need(evidence.sha_bytes(data[:self.committed_offset]) == self.committed_sha, 'Previously verified shadow bytes changed')
                self.committed_offset, self.committed_sha = committed, state['shadow_sha256']
            while self.offset < len(data):
                pos = self.offset
                end = data.find(bytes([10]), pos)
                if end < 0:
                    self.pending = {'offset': pos, 'reason': 'incomplete committed line', 'skipped': False}
                    break
                raw = data[pos:end+1]
                try:
                    value = json.loads(raw)
                    row = terminal_record(self.production, value, self.membership)
                    bid, sid = value['batch_id'], value['id']
                    old = self.completed[bid].get(sid)
                    need(old is None or old == row, 'Source revision changed without explicit repair')
                except (ValueError, UnicodeError, OSError, KeyError, TypeError) as error:
                    self.pending = {'offset': pos, 'reason': str(error), 'raw_sha256': evidence.sha_bytes(raw), 'skipped': False}
                    break
                if old is None:
                    self.completed[bid][sid] = row
                    dirty.add(bid)
                self.offset = end+1
            if stat.st_size > committed and self.pending is None:
                self.pending = {'offset': self.offset, 'reason': 'uncommitted journal tail', 'skipped': False}
            if state.get('source_offset', 0) < state.get('source_observed_size', 0) and self.pending is None:
                self.pending = {'offset': self.offset, 'upstream_source_offset': state['source_offset'],
                    'reason': 'upstream source record not terminal-verified/committed; keep pending', 'skipped': False}
        except (ValueError, UnicodeError, OSError, KeyError, TypeError) as error:
            self.pending = {'offset': self.offset, 'reason': str(error), 'skipped': False}
        return dirty


class AdditionalReader(VerifiedReader):
    def __init__(self, stream, production, membership):
        super().__init__(stream, production, membership, None)
        need(self.stream == self.production/'completed.jsonl', 'Additional stream must belong to its distinct production root')

    def prepared(self):
        data, identity = evidence.source_snapshot(self.stream)
        if self.identity is not None:
            need(tuple(identity) == self.identity, 'Additional stream replaced')
        # Every record is still verified against this edition's immutable source terminal.
        return None, {'shadow_offset': len(data), 'shadow_sha256': evidence.sha_bytes(data)}


def progress(membership, reader, statuses):
    terminal = sum(len(rows) for rows in reader.completed.values())
    labels = sum(s['acceptance']['review_complete'] for s in statuses)
    all_source = reader.pending is None and terminal == 222939 and len(statuses) == 448 and all(s['acceptance']['all_source_labels_completed'] for s in statuses)
    return {'schema': 'curated_independent_review_watch_v4', 'incremental_labels_reviewed_without_waiting_full_batch': True,
        'total_source_count': 222939, 'expected_batch_count': 448, 'source_count_by_split': dict(membership.by_split),
        'batch_review_count': len(statuses), 'review_complete_batch_count': labels, 'terminal_source_count': terminal,
        'sampled_sources_reviewed': sum(s['sampled_reviewed_denominator'] for s in statuses),
        'AI_error_candidate_count': sum(s['confirmed_error_count'] for s in statuses),
        'awaiting_human_or_label_completion_batches': [s['batch_id'] for s in statuses if not s['acceptance']['eligible_for_training']],
        'one_resident_reviewer': True, 'all_batches_reviewed': labels == 448,
        'all_source_labels_completed': all_source, 'human_approval_automatically_granted': False,
        'consumer_stream_offset': reader.offset, 'consumer_stream_identity': reader.identity,
        'completed_stream': str(reader.stream), 'recovery_plan_ref': reader.plan_ref, 'source_prepared': True,
        'committed_stream_offset': reader.committed_offset, 'pending_stream_record': reader.pending,
        'malformed_records_skipped': False, 'training_entry_updated': False}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('manifest', 'results-root', 'out', 'model', 'ms-source', 'annotator-model-reference', 'completed-stream', 'recovery-plan', 'recovery-plan-sha'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--additional-results-root')
    p.add_argument('--additional-completed-stream')
    p.add_argument('--additional-out')
    args = p.parse_args(argv)
    need(args.poll_seconds > 0, 'Positive poll interval required')
    plan_ref = {'path': str(Path(args.recovery_plan).resolve()), 'sha256': args.recovery_plan_sha}
    prepared_stream(args.completed_stream, args.results_root, plan_ref)
    membership = Membership(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.ms_source)
    import ms_batch_review as review
    reviewer = review.LocalReviewer(args.model, out, annotator_model_reference=args.annotator_model_reference,
        hand_roi_root=str(Path(args.results_root)/'roi'), ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    sampling_plan = review.sampling_plan
    reader = VerifiedReader(args.completed_stream, args.results_root, membership, plan_ref)
    contexts = [(reader, out, 'original')]
    extra = (args.additional_results_root, args.additional_completed_stream, args.additional_out)
    need(not any(extra) or all(extra), 'Additional edition root/stream/output must be supplied together')
    if all(extra):
        extra_root, extra_stream, extra_out = map(lambda p: Path(p).resolve(), extra)
        original_root = Path(args.results_root).resolve()
        need(extra_root.is_relative_to(ROOT) and extra_out.is_relative_to(ROOT)
            and not extra_root.is_relative_to(original_root) and not original_root.is_relative_to(extra_root)
            and not extra_out.is_relative_to(out.resolve()) and not out.resolve().is_relative_to(extra_out)
            and not extra_out.is_relative_to(extra_root) and not extra_root.is_relative_to(extra_out), 'New edition requires independent production and review namespaces')
        contexts.append((AdditionalReader(extra_stream, extra_root, membership), extra_out, 'additional'))
    while True:
        complete = []
        for context_reader, context_out, edition in contexts:
            review.sampling_plan = sampling_plan
            install_history_selection(review, context_out)
            reviewer.hand_roi_root = str(context_reader.production/'roi')
            dirty = context_reader.poll()
            for bid in sorted(dirty):
                originals = membership.batch(bid)
                rows = [context_reader.completed[bid].get(sid, {**row, 'label_status': 'pending'}) for sid, row in originals.items()]
                # History is checked before any mutable batch summary/plan publication.
                retained_selection(review, context_out, bid, [review.normalized_record(row) for row in rows])
                result = review.review_batch_ready(rows, context_out, reviewer, batch_id=bid,
                    expected_source_ids=list(originals), sample_count=25, risk_count=5)
                print(json.dumps({'batch': bid, 'edition': edition, 'labels_terminal': len(context_reader.completed[bid]), 'source_count': len(originals),
                    'sources_reviewed': result['sampled_reviewed_denominator'], 'AI_error_candidates': result['confirmed_error_count'],
                    'state': result['state'], 'eligible_for_training': result['acceptance']['eligible_for_training']}), flush=True)
            statuses = []
            for bid in sorted(context_reader.completed):
                path = context_out/bid/'summary.json'
                if path.exists():
                    value = evidence.load(path)
                    need(value['source_count'] == len(membership.batch(bid)), 'Review summary full membership changed')
                    statuses.append(value)
            value = progress(membership, context_reader, statuses)
            value.update(review_edition=edition, label_results_root=str(context_reader.production), review_out=str(context_out),
                original_main_labels_replaced=False, resident_reviewer_shared_across_editions=True)
            context_out.mkdir(parents=True, exist_ok=True)
            evidence.replace(context_out/'progress.json', value)
            done = value['all_batches_reviewed'] and value['all_source_labels_completed'] and not context_reader.pending
            complete.append(done)
            if done:
                evidence.replace(context_out/'run.summary.json', value)
        if all(complete):
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
