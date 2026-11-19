"""Independent collaborative batches plus every explicit JOINT repair revision.

The frozen reviewer is imported unchanged; outputs and label editions are new.
"""
from __future__ import annotations
import argparse
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import os
import time

SCHEMA = 'curated_review_extension_watch_v1'
FLAGS = dict(semantic_approval=False, human_review_completed=False,
             human_approval_automatically_granted=False, training_eligible=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    tmp.replace(path)


def checked(ref):
    path = Path(ref['path'])
    need(path.is_absolute() and sha(path) == ref['sha256'], 'label/source reference SHA changed')
    value = json.loads(path.read_text())
    need(sha(path) == ref['sha256'], 'reference changed while reading')
    return value


class Stream:
    def __init__(self, path):
        self.path, self.offset, self.identity = Path(path), 0, None

    def rows(self):
        if not self.path.exists():
            return
        st = self.path.stat()
        identity = [st.st_dev, st.st_ino]
        need(self.identity in (None, identity) and st.st_size >= self.offset,
             'append stream replaced/truncated: ' + str(self.path))
        self.identity = identity
        with self.path.open('rb') as f:
            f.seek(self.offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset = pos
                    break
                yield json.loads(line)
                self.offset = f.tell()


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = sha(path)
        self.summary = json.loads(self.path.read_text())
        need(self.summary['total'] == 222939 and len(self.summary['batches']) == 448,
             'original curated manifest must remain 222939/448')
        self.descriptors = {v['batch_id']: v for v in self.summary['batches']}
        need(len(self.descriptors) == 448, 'duplicate original batch ID')
        self.cache = {}

    def members(self, bid):
        need(bid in self.descriptors, 'unknown original batch')
        if bid not in self.cache:
            b = self.descriptors[bid]
            need(sha(b['path']) == b['sha256'], 'original batch manifest changed')
            rows = [json.loads(l) for l in Path(b['path']).read_text().splitlines() if l]
            need(len(rows) == b['count'] and all(r['batch_id'] == bid and r['split'] == b['split'] for r in rows),
                 'batch membership/split/count mismatch')
            self.cache[bid] = {r['id']: r for r in rows}
            need(len(self.cache[bid]) == len(rows), 'duplicate fixed source member')
        return self.cache[bid]


def bound_row(original, candidate, ref, *, edition, origin_root, stream_path, original_joint_ref=None):
    need(candidate['id'] == original['id'] and candidate['batch_id'] == original['batch_id']
         and candidate['split'] == original['split'], 'joint label original member mismatch')
    for key in ('video_path', 'audio_path'):
        need(candidate.get(key) == original[key], 'joint label original media mismatch: ' + key)
    need(isinstance(candidate.get('joint_complete'), bool) and isinstance(candidate.get('label'), dict),
         'true JOINT candidate with both semantic components required')
    row = {**copy.deepcopy(original), **copy.deepcopy(candidate)}
    row.update(label_path=ref['path'], label_sha256=ref['sha256'],
               label_status='completed' if candidate['joint_complete'] else 'partial')
    row['review_provenance'] = dict(label_edition=edition, label_result_ref=copy.deepcopy(ref),
        original_manifest_record=copy.deepcopy(original), original_joint_result_ref=copy.deepcopy(original_joint_ref),
        original_roi_root=str(Path(origin_root) / 'roi'), input_stream=str(stream_path),
        original_review_scores_reused=False, **FLAGS)
    refs = copy.deepcopy(row.get('source_refs') or {})
    need(isinstance(refs, dict), 'explicit source_refs object required')
    refs['independent_review_lineage'] = dict(joint_label=copy.deepcopy(ref))
    if original_joint_ref:
        old = checked(original_joint_ref)
        need(all(old.get(k) == original[k] for k in ('id', 'split', 'batch_id', 'video_path', 'audio_path')),
             'original JOINT label lineage differs from fixed original source')
        refs['independent_review_lineage']['original_joint_label'] = copy.deepcopy(original_joint_ref)
    row['source_refs'] = refs
    return row


def gate_summary(summary, *, edition, provenance):
    value = copy.deepcopy(summary)
    value['algorithmic_checks_satisfied'] = bool(value['acceptance']['eligible_for_training'])
    value['acceptance']['eligible_for_training'] = False
    value.update(schema_extension=SCHEMA, label_edition=edition, review_provenance=provenance,
                 state='pending_label_or_review' if not value['acceptance']['review_complete']
                 or not value['acceptance']['all_source_labels_completed'] else 'AI_audit_done_not_human_approved', **FLAGS)
    return value


def publish(root, summary, rows, review, audit_results):
    by_id = {r.get('source_id', r['id']): r for r in rows}
    web = root / 'review_web.json'
    if web.exists():
        value = json.loads(web.read_text())
        value.update(label_edition=summary['label_edition'], **FLAGS)
        for item in value['items'] + value['optional_uncertain_items']:
            provenance = by_id[item['source_id']]['review_provenance']
            item.update(label_edition=summary['label_edition'],
                        label_file_ref=provenance.get('label_result_ref'),
                        original_joint_result_ref=provenance.get('original_joint_result_ref'), **FLAGS)
        atomic(web, value)
    ledger = []
    for sid in summary['selected_source_ids']:
        row = review.normalized_record(by_id[sid])
        digest = review.batch.digest(row)
        result = audit_results.get(digest, dict(source_id=sid, status='label_incomplete'))
        provenance = row.get('review_provenance', {})
        receipt = result.get('review_receipt')
        final_path = Path(receipt['path']).with_name('final.json') if receipt else None
        ledger.append(dict(schema='curated_independent_review_record_v1', source_id=sid, id=sid,
            split=row['split'], original_batch_id=row['batch_id'], label_edition=summary['label_edition'],
            label_path=row.get('label_path'), label_sha256=row.get('label_sha256'),
            semantic_label_sha256=result.get('label_sha256'), input_row_sha256=digest,
            status=result['status'], review_receipt=receipt,
            review_final_ref=dict(path=str(final_path), sha256=sha(final_path)) if final_path and final_path.exists() else None,
            source_refs=copy.deepcopy(row.get('source_refs')), source_binding=copy.deepcopy(row.get('source_binding')),
            review_provenance=copy.deepcopy(provenance), errors=result.get('errors', []),
            uncertainties=result.get('uncertainties', []), label_complete=review.label_complete(row), **FLAGS))
    path = root / 'reviews.jsonl'
    with tempfile.NamedTemporaryFile('w', dir=root, delete=False) as f:
        for item in ledger:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno()); temporary = Path(f.name)
    temporary.replace(path)
    summary['review_records_ref'] = dict(path=str(path), sha256=sha(path))
    summary['human_resolution_stream'] = None
    atomic(root / 'summary.json', summary)


class Watcher:
    def __init__(self, manifest, collaborative_root, repair_root, out, reviewer, review,
                 revision='repair-20261025-v1', extra_root=None):
        self.manifest, self.review, self.reviewer = manifest, review, reviewer
        self.collaborative_root, self.repair_root, self.out = map(Path, (collaborative_root, repair_root, out))
        self.revision = revision
        need(re.fullmatch(r'[A-Za-z0-9_-]+', revision), 'unsafe label edition')
        self.collaborative = Stream(self.collaborative_root / 'completed.jsonl')
        self.repairs = Stream(self.repair_root / 'joint' / 'review-required.jsonl')
        self.extra_root = Path(extra_root) if extra_root else self.repair_root.parent / 'curated_review_extra_visual_v1'
        self.extra = Stream(self.extra_root / 'review-required.jsonl')
        self.extra_notices, self.extra_summaries = {}, {}
        self.completed = collections.defaultdict(dict)
        self.collaborative_notices, self.repair_notices, self.repair_summaries = {}, {}, {}
        self.batch_summaries, self.audit_results = {}, {}

    def audit(self, row, root, **kwargs):
        result = self.reviewer(row, root, **kwargs)
        self.audit_results[self.review.batch.digest(row)] = copy.deepcopy(result)
        return result

    def collaborative_row(self, notice):
        bid, sid = notice['batch_id'], notice['id']
        original = self.manifest.members(bid).get(sid)
        need(original is not None and notice['split'] == original['split'], 'collaborative source is outside original batch')
        key = original['split'] + '/' + sid
        if key in self.collaborative_notices:
            need(self.collaborative_notices[key] == notice, 'collaborative source changed without explicit edition')
            return None
        ref = notice.get('joint_candidate_ref')
        if ref:
            need(Path(ref['path']).resolve() == self.collaborative_root.resolve() / 'sources' / key / 'joint-results' / (ref['sha256'] + '.json'),
                 'collaborative label reference outside its exact content-addressed source result')
            row = bound_row(original, checked(ref), ref, edition='collaborative-v1',
                            origin_root=self.collaborative_root, stream_path=self.collaborative.path)
        else:
            row = {**copy.deepcopy(original), 'label_status': 'failed', 'consumer_status': notice['status'],
                   'label': {'dance_style': 'unknown', 'timed_motion_script': ''},
                   'review_provenance': dict(label_edition='collaborative-v1', input_stream=str(self.collaborative.path), **FLAGS)}
        self.collaborative_notices[key] = copy.deepcopy(notice)
        self.completed[bid][sid] = row
        return bid

    def repair_row(self, notice):
        key, bid = notice['source_key'], notice['batch_id']
        need(re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key), 'unsafe JOINT repair source key')
        need(notice['global_revision'] == self.revision, 'unexpected JOINT repair label edition')
        original = self.manifest.members(bid).get(notice['id'])
        need(original is not None and key == original['split'] + '/' + original['id'], 'JOINT repair source/member mismatch')
        ref = notice['joint_result_ref']
        need(notice['result_ref'] == ref and Path(ref['path']).resolve() ==
             self.repair_root.resolve() / 'joint' / 'revisions' / self.revision / key / 'result.json',
             'only exact JOINT revision result is reviewable; visual-only stream is forbidden')
        identity = self.revision + '/' + key
        if identity in self.repair_notices:
            need(self.repair_notices[identity] == notice, 'JOINT repair edition changed')
            return None
        candidate = checked(ref)
        need(candidate['global_revision'] == self.revision and candidate.get('risk_review_case') is True,
             'explicit JOINT repair risk/edition required')
        original_ref = candidate['original_result_ref']
        need(Path(original_ref['path']).resolve() == self.repair_root.parent.resolve() /
             'curated222939_production_v1' / 'sources' / key / 'joint-results' / (original_ref['sha256'] + '.json'),
             'JOINT revision original label lineage outside exact original production source')
        row = bound_row(original, candidate, ref, edition=self.revision,
                        origin_root=self.repair_root.parent / 'curated222939_production_v1',
                        stream_path=self.repairs.path, original_joint_ref=original_ref)
        self.repair_notices[identity] = copy.deepcopy(notice)
        return identity, row

    def extra_row(self, notice):
        edition, key = notice['global_revision'], notice['source_key']
        need(edition == 'raq-scope-offline-20261025-v3' and key == 'train/RAQoxqRxCfMvTHSx'
             and notice['id'] == 'RAQoxqRxCfMvTHSx' and notice.get('risk_review_case') is True,
             'only explicitly authorized RAQ offline visual edition allowed')
        root = self.extra_root.resolve() / 'joint-revisions' / edition / key
        ref, source_ref = notice['result_ref'], notice['source_record_ref']
        need(Path(ref['path']).resolve() == root / 'result.json' and Path(source_ref['path']).resolve() == root / 'source_record.json',
             'extra visual joint/source reference outside explicit edition')
        identity = edition + '/' + key
        if identity in self.extra_notices:
            need(self.extra_notices[identity] == notice, 'extra visual label edition changed')
            return None
        candidate, original = checked(ref), checked(source_ref)
        pilot_ref = candidate['original_pilot_rows_ref']
        need(Path(pilot_ref['path']).resolve() == self.extra_root.parent.resolve() / 'runtime/independent_review_pilot25_rows_v1.jsonl'
             and sha(pilot_ref['path']) == pilot_ref['sha256'], 'explicit original pilot source SHA required')
        pilot = [json.loads(l) for l in Path(pilot_ref['path']).read_text().splitlines() if l]
        prior = [r for r in pilot if r['id'] == original['id']]
        need(prior == [original] and candidate['source_row_sha256'] == self.review.batch.digest(original),
             'extra source record differs from original pilot source')
        old_label, new_label = self.review.label_payload(original), self.review.label_payload(candidate)
        need(old_label['dance_style'] == new_label['dance_style'], 'extra visual revision must preserve original pilot style')
        oldlines, newlines = old_label['timed_motion_script'].splitlines(), new_label['timed_motion_script'].splitlines()
        need(len(oldlines) == len(newlines) == 3 and oldlines[0] == newlines[0] and oldlines[2] == newlines[2],
             'unchanged RAQ w0/w2 must remain verbatim and not reuse old review scores')
        need(candidate['global_revision'] == edition and candidate['extra_visual_revision_ref'] == notice['visual_revision_ref']
             and candidate['original_source_record_ref'] == source_ref
             and candidate.get('new_model_calls') == 0 and candidate.get('model_rewrite_performed') is False,
             'explicit original source and zero-new-VLM extra visual lineage required')
        need(Path(notice['visual_revision_ref']['path']).resolve() == self.extra_root.parent.resolve() /
             'legacy_raq_repair_v1/revisions' / edition / key / 'result.json', 'extra offline visual path differs')
        visual = checked(notice['visual_revision_ref'])
        need(visual['id'] == original['id'] and visual['global_revision'] == edition,
             'offline visual candidate/source identity mismatch')
        old = checked(candidate['original_visual_result_ref'])
        need(all(old['manifest_record'][k] == original[k] for k in ('id', 'split', 'video_path', 'audio_path')),
             'original RAQ visual/media lineage differs')
        row = bound_row(original, candidate, ref, edition=edition, origin_root=self.extra_root,
                        stream_path=self.extra.path)
        roi = row.get('hand_roi', {})
        need(roi.get('available') is True and Path(roi['path']).is_absolute(), 'original RAQ native ROI reference required')
        need(sha(roi['path']) == roi['sha256'], 'original RAQ ROI changed')
        row['review_provenance'].update(original_roi_root=roi['path'],
            original_visual_result_ref=candidate['original_visual_result_ref'],
            extra_visual_revision_ref=notice['visual_revision_ref'], unchanged_windows_old_review_scores_reused=False)
        self.extra_notices[identity] = copy.deepcopy(notice)
        return identity, row

    def poll(self):
        dirty = set()
        for notice in self.collaborative.rows():
            bid = self.collaborative_row(notice)
            if bid:
                dirty.add(bid)
        # A single model instance is shared, but every edition has a new receipt tree.
        for notice in self.repairs.rows():
            entry = self.repair_row(notice)
            if entry is None:
                continue
            identity, row = entry
            local = self.out / 'joint-revisions' / self.revision
            bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
            result = self.review.review_batch_ready([row], local, self.audit, batch_id=bid,
                expected_source_ids=[row['id']], sample_count=1, risk_count=0)
            provenance = dict(source_key=identity, label_result_ref=row['review_provenance']['label_result_ref'],
                original_joint_result_ref=row['review_provenance']['original_joint_result_ref'],
                fixed_original_batch_id=row['batch_id'], every_joint_revision_is_risk_reviewed=True,
                original_review_scores_reused=False)
            result = gate_summary(result, edition=self.revision, provenance=provenance)
            publish(local / bid, result, [row], self.review, self.audit_results)
            self.repair_summaries[identity] = result
        for notice in self.extra.rows():
            entry = self.extra_row(notice)
            if entry is None:
                continue
            identity, row = entry
            edition = row['review_provenance']['label_edition']
            local = self.out / 'extra-visual-revisions' / edition
            bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
            result = self.review.review_batch_ready([row], local, self.audit, batch_id=bid,
                expected_source_ids=[row['id']], sample_count=1, risk_count=0)
            result = gate_summary(result, edition=edition, provenance=dict(source_key=identity,
                label_result_ref=row['review_provenance']['label_result_ref'],
                original_visual_result_ref=row['original_visual_result_ref'],
                extra_visual_revision_ref=row['extra_visual_revision_ref'], every_extra_visual_revision_is_risk_reviewed=True,
                original_review_scores_reused=False, unchanged_windows_old_review_scores_reused=False))
            publish(local / bid, result, [row], self.review, self.audit_results)
            self.extra_summaries[identity] = result
        for bid in sorted(dirty):
            members = self.manifest.members(bid)
            rows = [self.completed[bid].get(sid, {**copy.deepcopy(r), 'label_status': 'pending'})
                    for sid, r in members.items()]
            local = self.out / 'collaborative-v1'
            result = self.review.review_batch_ready(rows, local, self.audit, batch_id=bid,
                expected_source_ids=list(members), sample_count=25, risk_count=5)
            result = gate_summary(result, edition='collaborative-v1', provenance=dict(
                original_manifest=dict(path=str(self.manifest.path), sha256=self.manifest.sha256),
                fixed_batch_manifest=copy.deepcopy(self.manifest.descriptors[bid]),
                completed_stream=str(self.collaborative.path), original_review_scores_reused=False))
            publish(local / bid, result, rows, self.review, self.audit_results)
            self.batch_summaries[bid] = result
        value = dict(schema=SCHEMA, total_source_count=222939, expected_batch_count=448,
            original_manifest=dict(path=str(self.manifest.path), sha256=self.manifest.sha256),
            collaborative_terminal_source_count=len(self.collaborative_notices),
            collaborative_review_batch_count=len(self.batch_summaries),
            collaborative_sampled_reviewed=sum(v['sampled_reviewed_denominator'] for v in self.batch_summaries.values()),
            joint_revision_notice_count=len(self.repair_notices),
            joint_revision_reviewed_count=sum(v['sampled_reviewed_denominator'] for v in self.repair_summaries.values()),
            joint_revision_label_incomplete_count=sum(not v['acceptance']['all_source_labels_completed'] for v in self.repair_summaries.values()),
            extra_visual_revision_notice_count=len(self.extra_notices),
            extra_visual_revision_reviewed_count=sum(v['sampled_reviewed_denominator'] for v in self.extra_summaries.values()),
            extra_visual_revision_stream_offset=self.extra.offset,
            AI_error_candidate_count=sum(v['confirmed_error_count'] for v in list(self.batch_summaries.values()) + list(self.repair_summaries.values()) + list(self.extra_summaries.values())),
            collaborative_stream_offset=self.collaborative.offset, joint_revision_stream_offset=self.repairs.offset,
            one_resident_reviewer=True, old_review_outputs_modified=False, original_review_scores_reused=False,
            subset_not_full_curated_coverage=True, all_batches_reviewed=False, global_revision=self.revision, **FLAGS)
        atomic(self.out / 'progress.json', value)
        return value


class EditionReviewer:
    def __init__(self, local):
        self.local = local

    def __call__(self, row, root, **kwargs):
        self.local.hand_roi_root = row['review_provenance']['original_roi_root']
        return self.local(row, root, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--collaborative-root', required=True)
    p.add_argument('--repair-root', required=True)
    p.add_argument('--extra-visual-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-reference', required=True)
    p.add_argument('--ms-source', required=True)
    p.add_argument('--annotator-model-reference', required=True)
    p.add_argument('--global-revision', default='repair-20261025-v1')
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    need(a.poll_seconds > 0, 'positive poll interval required')
    out = Path(a.out).resolve()
    need(out.name == 'curated_independent_reviews_extension_v1', 'dedicated new review output required')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'watcher.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, a.ms_source)
        import ms_batch_review as review
        local = review.LocalReviewer(a.model, out, model_reference=a.model_reference,
            annotator_model_reference=a.annotator_model_reference,
            ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
        watcher = Watcher(Manifest(a.manifest), a.collaborative_root, a.repair_root, out,
                          EditionReviewer(local), review, a.global_revision, a.extra_visual_root)
        while True:
            print(json.dumps(watcher.poll(), ensure_ascii=False), flush=True)
            if not a.watch:
                break
            time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
