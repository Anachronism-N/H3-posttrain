"""CPU-only derived nine-consumer aggregation and exact-version review inputs."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
import tempfile
import time

RT = Path(__file__).resolve().parent
PLAN_SHA = 'f4345234020b1286d7723fcff4dcf0f0fbe78f03bbd88b66514e716878e9bcc6'
SCHEMA = 'curated_collab3_review_input_v1'
FLAGS = dict(semantic_approval=False, human_review_completed=False, training_eligible=False,
             formal_approval=False, formal_training=False, source_quality_pass=False)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_sampling(source):
    source = Path(source).resolve()
    namespace = dict(json=json, hashlib=hashlib, random=random, re=re, Path=Path,
                     read_json=lambda path: strict(Path(path).read_bytes()))
    refs = []
    # Compile only the unchanged pure functions, not scipy/model/import side effects.
    for filename, names in (
            ('ms_optimize.py', {'json_bytes'}),
            ('ms_visual_batch.py', {'digest'}),
            ('ms_batch_review.py', {'need', 'source_id', 'label_payload', 'label_complete',
                                    'label_reviewable', 'risk_flags', 'sampling_plan', 'POLICY'})):
        path = source / filename
        tree = ast.parse(path.read_bytes(), filename=str(path))
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.FunctionDef) and node.name in names) or
                 (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
        found = {node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id for node in nodes}
        need(found == names, 'frozen pure sampling function inventory changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        refs.append(reference(path))
        if filename == 'ms_visual_batch.py':
            namespace['batch'] = SimpleNamespace(digest=namespace['digest'])
    return SimpleNamespace(**namespace, code_refs=refs)


class Adapter:
    def __init__(self, plan_path, expected_plan_sha, review, state=None, max_lines_per_stream=64):
        need(type(max_lines_per_stream) is int and max_lines_per_stream > 0, 'positive bounded stream increment required')
        self.max_lines_per_stream = max_lines_per_stream
        self.plan_ref = {'path': str(Path(plan_path).resolve()), 'sha256': expected_plan_sha}
        self.boundary = {}
        self.documents = {}
        self.plan = self.checked(self.plan_ref)
        p = self.plan
        need(p['source_range'] == {'start': 32000, 'end': 62000} and p['selected_count'] == 30000,
             'fixed original 32000:62000 reservation changed')
        need(p['input_count'] == 222939 and p['full_batch_count'] == 448, 'full denominator changed')
        self.root = Path(p['out']).resolve()
        self.review = review
        self.sampling_code_refs = review.code_refs
        self.manifest = self.checked(p['manifest_summary'])
        need(self.manifest['total'] == 222939 and len(self.manifest['batches']) == 448 and
             sum(b['count'] for b in self.manifest['batches']) == 222939,
             'original full manifest denominator changed')
        self.batches = {b['batch_id']: b for b in p['selected_original_batches']}
        need(list(self.batches) == ['train-b%06d' % i for i in range(64, 124)] and
             all(b['count'] == 500 and b['split'] == 'train' and b['first_source_ordinal'] == 32000 + i * 500
                 for i, b in enumerate(self.batches.values())), 'original 60x500 batches changed')
        all_batches = {b['batch_id']: b for b in self.manifest['batches']}
        need(all(all_batches[k] == b for k, b in self.batches.items()), 'plan/full manifest batch binding changed')
        self.streams = {}
        self.consumers = {}
        selected = {}
        for member, assignment in p['members'].items():
            mode, shard, nshard = (assignment[k] for k in ('mode', 'shard', 'nshard'))
            need(mode in ('producer', 'consumer') and nshard == (4 if mode == 'producer' else 9),
                 'member shard protocol changed')
            expected_node = ('N2' if shard < (2 if mode == 'producer' else 3)
                             else 'N7' if mode == 'producer' or shard >= 7 else 'N4')
            need(member == expected_node.lower() + ('p' if mode == 'producer' else 'c') + str(shard)
                 and assignment['node'] == expected_node, 'member/node identity changed')
            values = self.checked(p['selected_sources'][member])
            keys = set()
            for row in values:
                key = self.key(row)
                need(key not in keys and row['source_ordinal'] % nshard == shard and
                     32000 <= row['source_ordinal'] < 62000, 'member selected source binding changed')
                keys.add(key)
                minimal = {k: row[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')}
                need(selected.get(key, minimal) == minimal, 'cross-member source binding mismatch')
                selected[key] = minimal
            if mode == 'consumer':
                physical = self.root / 'consumers' / ('shard-%04d' % shard)
                self.consumers[shard] = (member, physical)
                path = physical / 'completed.jsonl'
            else:
                physical = self.root
                path = self.root / 'ready' / ('producer-%04d.jsonl' % shard)
            self.streams[mode + ':' + str(shard)] = dict(member=member, mode=mode, shard=shard,
                                                       nshard=nshard, path=str(path), result_root=str(physical), keys=keys)
        need(len(self.streams) == 13 and set(self.consumers) == set(range(9)) and len(selected) == 30000,
             'exact 4 producer/9 consumer source assignment required')
        need(sorted(r['source_ordinal'] for r in selected.values()) == list(range(32000, 62000)),
             'original ordinals missing or duplicated')
        for info in self.streams.values():
            need(info['keys'] == {key for key, row in selected.items()
                                 if row['source_ordinal'] % info['nshard'] == info['shard']},
                 'member selected list is not exact modulo assignment')
        self.selected = selected
        self.state = state if state is not None else {}
        need(self.state.get('schema', SCHEMA) == SCHEMA and
             self.state.get('plan_ref', self.plan_ref) == self.plan_ref, 'state/plan revision mismatch')
        need(self.state.get('sampling_code_refs', self.sampling_code_refs) == self.sampling_code_refs,
             'sampling code changed; require explicit new review adapter state')
        self.state.update(schema=SCHEMA, plan_ref=self.plan_ref, sampling_code_refs=self.sampling_code_refs)
        for key in ('streams', 'ready', 'terminal', 'batches'):
            self.state.setdefault(key, {})
        self.members = {}
        self.dirty = set()
        self.boundary.clear()

    @staticmethod
    def key(row):
        sid = row.get('id', row.get('source_id'))
        need(isinstance(sid, str) and sid and '/' not in sid and sid not in ('.', '..') and
             row.get('split') == 'train', 'explicit original train source identity required')
        need(row.get('source_id', sid) == sid and row.get('source_key', 'train/' + sid) == 'train/' + sid,
             'source-key/id mismatch')
        return 'train/' + sid

    def checked(self, ref, expected=None):
        need(isinstance(ref, dict) and isinstance(ref.get('path'), str) and isinstance(ref.get('sha256'), str),
             'explicit ref path/SHA required')
        path = Path(ref['path']).resolve()
        need(expected is None or path == Path(expected).resolve(), 'cross-source/physical-root ref rejected')
        key = (str(path), ref['sha256'])
        if key not in self.documents:
            raw = path.read_bytes()
            need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'reference SHA mismatch: ' + str(path))
            self.documents[key] = strict(raw)
        self.boundary[str(path)] = ref['sha256']
        return self.documents[key]

    def member(self, notice):
        key = self.key(notice)
        need(key in self.selected, 'source outside frozen reservation')
        wanted = self.selected[key]
        need(notice.get('batch_id') == wanted['batch_id'] and
             notice.get('source_ordinal', wanted['source_ordinal']) == wanted['source_ordinal'],
             'notice original batch/ordinal mismatch')
        bid = wanted['batch_id']
        if bid not in self.members:
            descriptor = self.batches[bid]
            raw = Path(descriptor['path']).read_bytes()
            need(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'original member SHA mismatch')
            rows = [strict(line) for line in raw.splitlines()]
            need(len(rows) == 500, 'original 500-source membership required')
            members = {}
            for offset, row in enumerate(rows):
                sourcekey = self.key(row)
                need(sourcekey not in members and row['batch_id'] == bid and
                     row['source_ordinal'] == descriptor['first_source_ordinal'] + offset and
                     self.selected[sourcekey] == {k: row[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')},
                     'original member/source ordinal binding mismatch')
                members[sourcekey] = row
            self.members[bid] = members
        self.boundary[self.batches[bid]['path']] = self.batches[bid]['sha256']
        return self.members[bid][key]

    def bind(self, document, original):
        for field in ('id', 'source_id', 'split', 'batch_id', 'source_ordinal'):
            if field in document:
                need(document[field] == original.get(field, original['id']), 'cross-source member field: ' + field)
        need(document.get('id', document.get('source_id')) == original['id'], 'source identity missing')
        if 'manifest_record' in document:
            need(document['manifest_record'] == original, 'exact original manifest member changed')
        if 'source_refs' in document:
            refs = document['source_refs']
            need(refs.get('csv', refs.get('original_csv')) == original['source_refs']['csv'],
                 'original CSV/raw-row lineage changed')
            if refs.get('manifest'):
                descriptor = self.batches[original['batch_id']]
                need(all(refs['manifest'][k] == descriptor[k] for k in ('path', 'sha256')),
                     'original batch manifest lineage changed')
        for flag in FLAGS:
            need(document.get(flag) is not True, 'approval unexpectedly granted upstream')

    def producer(self, original, notice=None):
        directory = self.root / 'sources' / 'train' / original['id']
        ref = reference(directory / 'producer.terminal.json')
        terminal = self.checked(ref)
        self.bind(terminal, original)
        if notice is not None:
            need(notice == terminal, 'ready notice/current producer terminal mismatch')
        result = {'terminal_ref': ref, 'status': terminal['status']}
        if terminal['status'] == 'failed':
            return result
        need(terminal['status'] == 'ready', 'producer terminal not ready or failed')
        plan = self.checked(terminal['plan_ref'], directory / 'bar_plan.json')
        self.bind(plan, original)
        need(plan['manifest_record'] == original and terminal['source_binding'] == plan['binding'] and
             plan['binding']['id'] == original['id'], 'producer plan/source binding mismatch')
        roi = self.checked(terminal['roi_ref'], self.root / 'roi' / original['id'] / 'roi_manifest.json')
        need(roi['id'] == original['id'] and roi['video_sha256'] == plan['binding']['video_sha256'] and
             roi['source']['path'] == original['video_path'], 'ROI/source media binding mismatch')
        media_ref = reference(directory / 'actual_media.json')
        media = self.checked(media_ref)
        source = media['source_identity']
        for field, sha_field in (('video_path', 'video_sha256'), ('audio_path', 'audio_sha256')):
            identity = source['files'][field]
            need(identity['path'] == original[field] and identity['sha256'] == plan['binding'][sha_field],
                 'actual source media path/SHA binding mismatch')
            # Only newly admitted sources are hashed; no all-media polling or decode.
            need(sha(identity['path']) == identity['sha256'], 'actual source media SHA mismatch')
            self.boundary[identity['path']] = identity['sha256']
        result.update(plan_ref=terminal['plan_ref'], roi_ref=terminal['roi_ref'], actual_media_ref=media_ref,
                      source_binding=plan['binding'], source_identity=source)
        return result

    def accept(self, info, notice):
        original = self.member(notice)
        key = self.key(original)
        need(key in info['keys'], 'wrong producer/consumer member modulo assignment')
        self.bind(notice, original)
        current = self.state['ready' if info['mode'] == 'producer' else 'terminal'].get(key)
        notice_sha = digest(notice)
        if current and current['notice_sha256'] == notice_sha:
            return False
        producer = self.producer(original, notice if info['mode'] == 'producer' else None)
        if info['mode'] == 'producer':
            result = producer
        else:
            physical = Path(info['result_root'])
            path = physical / 'sources' / 'train' / original['id'] / 'consumer.terminal.json'
            terminal_ref = reference(path)
            terminal = self.checked(terminal_ref)
            need(terminal == notice, 'completed notice/current consumer terminal mismatch')
            self.bind(terminal, original)
            result = dict(status=terminal['status'], terminal_ref=terminal_ref, producer=producer,
                          physical_result_root=str(physical), member=info['member'], label_status='failed')
            if terminal.get('producer_terminal_ref'):
                need(terminal['producer_terminal_ref'] == producer['terminal_ref'], 'producer terminal linkage mismatch')
            if terminal.get('joint_candidate_ref'):
                ref = terminal['joint_candidate_ref']
                path = Path(ref['path']).resolve()
                need(path.parent == physical / 'sources' / 'train' / original['id'] / 'joint-results',
                     'cross-source/physical-root joint candidate rejected')
                label = self.checked(ref)
                self.bind(label, original)
                need(producer['status'] == 'ready' and label['source_binding']['source_binding'] == producer['source_binding']
                     and label['source_binding']['plan_sha256'] == producer['plan_ref']['sha256'] and
                     label['source_identity'] == producer['source_identity'] and
                     label['video_path'] == original['video_path'] and label['audio_path'] == original['audio_path'],
                     'joint label/media/producer-plan mismatch')
                need(label['joint_complete'] == terminal['joint_complete'], 'terminal/joint completeness mismatch')
                for field, parent in (('visual_result_ref', physical / 'visual' / ('shard-%04d' % info['shard']) / 'sources' / original['id']),
                                      ('style_result_ref', physical / 'sources' / 'train' / original['id'] / 'style-results')):
                    artifact = label.get(field)
                    need(isinstance(artifact, dict) and Path(artifact['path']).resolve().is_relative_to(parent),
                         'explicit source-bound visual/style result ref required')
                    document = self.checked(artifact)
                    self.bind(document, original)
                rgb_ref = label.get('raw_rgb_manifest')
                if rgb_ref is not None:
                    rgb = self.checked(rgb_ref, physical/'visual'/('shard-%04d'%info['shard'])/'sources'/original['id']/'rgb/manifest.json')
                    need(rgb['binding']['source_binding'] == producer['source_binding'] and
                         rgb['video_sha256'] == producer['source_binding']['video_sha256'] and
                         all(all(rgb['source_identity']['files'][field][k] == producer['source_identity']['files'][field][k]
                                 for k in ('path', 'sha256')) for field in ('video_path', 'audio_path')),
                         'RGB/source media binding mismatch')
                hand = label.get('hand_roi')
                if hand and hand.get('available'):
                    need(all(hand[k] == producer['roi_ref'][k] for k in ('path', 'sha256')),
                         'joint hand ROI differs from exact producer ROI')
                payload = self.review.label_payload(label)
                complete = label['joint_complete'] is True and label['all_windows_nonempty'] is True and self.review.label_complete(label)
                result.update(joint_candidate_ref=ref, label=payload, label_revision=ref['sha256'],
                              label_payload_sha256=self.review.batch.digest(payload), candidate_complete=complete,
                              label_status='completed' if complete else 'partial',
                              style_status=label.get('style_status'), visual_status=label.get('visual_status'),
                              source_refs=label.get('source_refs'), raw_rgb_manifest=label.get('raw_rgb_manifest'),
                              hand_roi=label.get('hand_roi'), visual_result_ref=label['visual_result_ref'],
                              style_result_ref=label['style_result_ref'], annotator_model_sha256=label.get('annotator_model_sha256'))
            else:
                need(terminal['status'] in ('failed', 'producer_failed'), 'missing joint ref for candidate terminal')
        result['notice_sha256'] = notice_sha
        self.state['ready' if info['mode'] == 'producer' else 'terminal'][key] = result
        self.dirty.add(original['batch_id'])
        return True

    def read_stream(self, stream_id, info):
        cursor = self.state['streams'].setdefault(stream_id, {'offset': 0})
        path = Path(info['path'])
        if not path.exists():
            cursor.update(status='missing', error=None)
            return
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino]
        if cursor.get('identity', identity) != identity or stat.st_size < cursor['offset']:
            cursor.update(status='blocked', error='stream replaced or truncated')
            return
        cursor['identity'] = identity
        cursor['snapshot_end'] = stat.st_size
        increment_lines = 0
        with path.open('rb') as stream:
            stream.seek(cursor['offset'])
            while True:
                start = stream.tell()
                if start >= stat.st_size:
                    cursor.update(status='caught_up', error=None)
                    break
                if increment_lines >= self.max_lines_per_stream:
                    cursor.update(status='pending_backlog', error='bounded increment; unread snapshot bytes retained')
                    break
                line = stream.readline(stat.st_size - start)
                if not line:
                    cursor.update(status='caught_up', error=None)
                    break
                if not line.endswith(b'\n'):
                    cursor.update(status='pending_tail', error='unfinished JSONL tail; cursor retained')
                    break
                try:
                    notice = strict(line)
                    need(isinstance(notice, dict), 'JSONL record must be object')
                    self.accept(info, notice)
                except FileNotFoundError as error:
                    cursor.update(status='pending_reference', error=str(error))
                    break
                except (ValueError, KeyError, TypeError, OSError) as error:
                    cursor.update(status='blocked', error=type(error).__name__ + ': ' + str(error))
                    break
                cursor['offset'] = stream.tell()
                cursor['accepted_lines'] = cursor.get('accepted_lines', 0) + 1
                increment_lines += 1
                need(cursor['offset'] > start, 'cursor did not advance')

    def review_input(self, bid):
        descriptor = self.batches[bid]
        first = next(row for row in self.selected.values() if row['batch_id'] == bid)
        self.member(first)
        originals = self.members[bid]
        records = []
        bindings = {}
        for key, original in originals.items():
            sid = original['id']
            terminal = self.state['terminal'].get(key)
            shard = original['source_ordinal'] % 9
            member, physical = self.consumers[shard]
            row = {k: original[k] for k in ('id', 'source_id', 'split', 'batch_id', 'source_ordinal', 'video_path', 'audio_path', 'source_refs')}
            row.update(source_key=key, physical_result_root=str(physical), consumer_member=member,
                       hand_roi_root=str(self.root / 'roi'), label_status='pending',
                       individual_review_status='not_individually_reviewed', **FLAGS)
            if terminal:
                row['label_status'] = terminal['label_status']
                if terminal.get('joint_candidate_ref'):
                    for name in ('label', 'label_revision', 'label_payload_sha256', 'source_refs', 'raw_rgb_manifest',
                                 'hand_roi', 'visual_result_ref', 'style_result_ref', 'style_status', 'visual_status',
                                 'annotator_model_sha256'):
                        row[name] = terminal[name]
                    row.update(label_path=terminal['joint_candidate_ref']['path'], label_sha256=terminal['joint_candidate_ref']['sha256'],
                               joint_complete=terminal['candidate_complete'], all_windows_nonempty=terminal['candidate_complete'])
                    binding = dict(source_key=key, source_ordinal=original['source_ordinal'], original_member_sha256=digest(original),
                                   original_batch_ref=descriptor, consumer_member=member, physical_result_root=str(physical),
                                   consumer_terminal_ref=terminal['terminal_ref'], label_file_ref=terminal['joint_candidate_ref'],
                                   label_revision=terminal['label_revision'], label_payload_sha256=terminal['label_payload_sha256'],
                                   **terminal['producer'])
                    binding.update(raw_rgb_manifest=terminal['raw_rgb_manifest'], hand_roi=terminal['hand_roi'],
                                   visual_result_ref=terminal['visual_result_ref'], style_result_ref=terminal['style_result_ref'],
                                   original_source_RGB='fresh_original_video_decode_required_at_review_submission',
                                   annotator_RGB_is_not_semantic_truth=True, evidence_ready=False,
                                   exact_version_review_key=digest({'source_key': key, 'label_file_ref': terminal['joint_candidate_ref']}), **FLAGS)
                    bindings[sid] = binding
            records.append(row)
        sampling = self.review.sampling_plan(records, bid, seed=0, sample_count=25, risk_count=5)
        selected = set(sampling['selected_source_ids'])
        for row in records:
            row['selected_for_review'] = row['id'] in selected
            if row['selected_for_review']:
                row['individual_review_status'] = 'review_required' if self.review.label_reviewable(row) else 'pending_label'
        return dict(schema=SCHEMA, batch_id=bid, split='train', source_ids=[r['id'] for r in records],
                    source_count=500, total_source_count=222939, total_batch_count=448, plan_ref=self.plan_ref,
                    records=records, exact_label_bindings=bindings, sampling=sampling, sampling_code_refs=self.sampling_code_refs,
                    sampling_denominator=500, selected_ready_count=sum(r['selected_for_review'] and self.review.label_reviewable(r) for r in records),
                    independent_reviewer_requirement='local 8B checkpoint identity distinct from 32B annotator; not yet loaded',
                    AI_errors_are_human_review_candidates=True, human_web_queue_required=True,
                    prior_source_id_only_rejections_not_applicable_to_new_revision=True,
                    model_calls=0, extra_source_attempts=0, attempt_budget_consumed=False, signals_sent=0, **FLAGS)

    def submission_binding(self, input_ref, source_id):
        """CPU preflight for a sampled exact version; never loads or calls a model."""
        self.boundary.clear()
        manifest = self.checked(input_ref)
        need(manifest['schema'] == SCHEMA and manifest['plan_ref'] == self.plan_ref and
             manifest['sampling_code_refs'] == self.sampling_code_refs and manifest['source_count'] == 500 and
             manifest['sampling_denominator'] == 500 and len(manifest['records']) == 500,
             'explicit current-plan full-batch review input required')
        need(manifest['sampling'] == self.review.sampling_plan(manifest['records'], manifest['batch_id'],
             seed=0, sample_count=25, risk_count=5), 'review input sampling changed')
        need(source_id in manifest['sampling']['selected_source_ids'], 'source is not sampled for this snapshot')
        rows = {r['id']: r for r in manifest['records']}
        row = rows[source_id]
        need(self.review.label_reviewable(row), 'sampled label still pending')
        original = self.member(row)
        need(set(rows) == {r['id'] for r in self.members[original['batch_id']].values()},
             'full original sampling denominator membership changed')
        binding = manifest['exact_label_bindings'][source_id]
        need(binding['original_member_sha256'] == digest(original) and binding['source_key'] == self.key(original),
             'review exact original source binding changed')
        member, physical = self.consumers[original['source_ordinal'] % 9]
        need(binding['consumer_member'] == member and binding['physical_result_root'] == str(physical) and
             row['physical_result_root'] == str(physical), 'review consumer physical-root binding changed')
        terminal = self.checked(binding['consumer_terminal_ref'], physical/'sources/train'/source_id/'consumer.terminal.json')
        self.bind(terminal, original)
        need(terminal.get('joint_candidate_ref') == binding['label_file_ref'] and
             row['label_sha256'] == binding['label_revision'] == binding['label_file_ref']['sha256'] and
             row['label_path'] == binding['label_file_ref']['path'], 'stale label revision at review submission')
        label = self.checked(binding['label_file_ref'])
        self.bind(label, original)
        need(self.review.label_payload(label) == row['label'] and
             self.review.batch.digest(row['label']) == binding['label_payload_sha256'], 'review label payload mismatch')
        for field in ('raw_rgb_manifest', 'hand_roi', 'visual_result_ref', 'style_result_ref'):
            need(row.get(field) == binding.get(field) == label.get(field), 'review RGB/ROI/result ref mismatch')
            artifact = label.get(field)
            if artifact and artifact.get('path') and artifact.get('sha256'):
                self.checked(artifact)
        producer = self.producer(original)
        need(all(binding[k] == producer[k] for k in producer), 'review source/ROI/plan changed since snapshot')
        self.boundary[self.plan_ref['path']] = self.plan_ref['sha256']
        for path, wanted in self.boundary.items():
            need(sha(path) == wanted, 'review submission SHA mismatch: ' + path)
        return dict(row=row, binding=binding, exact_input_ref=input_ref,
                    review_evidence_namespace=binding['exact_version_review_key'],
                    independent_8B_identity_verification_still_required=True, source_RGB_decode_still_required=True,
                    AI_errors_are_human_review_candidates=True, model_calls=0, signals_sent=0, **FLAGS)

    def poll(self, out):
        self.dirty.clear()
        self.boundary.clear()
        for stream_id, info in self.streams.items():
            self.read_stream(stream_id, info)
        manifests = {bid: self.review_input(bid) for bid in sorted(self.dirty)}
        # Admission/commit boundaries recheck only this increment, plus the two fixed descriptors.
        self.boundary[self.plan_ref['path']] = self.plan_ref['sha256']
        self.boundary[self.plan['manifest_summary']['path']] = self.plan['manifest_summary']['sha256']
        for path, wanted in self.boundary.items():
            need(sha(path) == wanted, 'commit-boundary SHA mismatch: ' + path)
        for bid, value in manifests.items():
            path = Path(out) / 'batches' / bid / 'review-required.json'
            atomic(path, value)
            self.state['batches'][bid] = dict(input_ref=reference(path), source_count=500,
                                            selected_ready_count=value['selected_ready_count'])
        counts = Counter(v['label_status'] for v in self.state['terminal'].values())
        value = dict(schema=SCHEMA, source='13 physical append streams + exact current terminal and result refs',
                     plan_ref=self.plan_ref, full_source_denominator=222939, full_batch_denominator=448,
                     reserved_source_denominator=30000, reserved_batch_denominator=60,
                     producer_ready_sources=sum(v['status'] == 'ready' for v in self.state['ready'].values()),
                     producer_failed_sources=sum(v['status'] == 'failed' for v in self.state['ready'].values()),
                     terminal_sources=len(self.state['terminal']), candidate_complete_sources=counts['completed'],
                     candidate_partial_sources=counts['partial'], failed_terminal_sources=counts['failed'],
                     missing_terminal_sources=30000-len(self.state['terminal']),
                     terminal_sources_without_ready_notice=sum(k not in self.state['ready'] for k in self.state['terminal']),
                     review_required_batch_count=len(self.state['batches']), individually_reviewed_sources=0,
                     blocked_streams={k:v for k,v in self.state['streams'].items() if v['status'] == 'blocked'},
                     pending_streams={k:v for k,v in self.state['streams'].items() if v['status'].startswith('pending')},
                     streams=self.state['streams'], batches=self.state['batches'],
                     physical_source_result_root_map={k:{'member':self.consumers[r['source_ordinal'] % 9][0],
                         'result_root':str(self.consumers[r['source_ordinal'] % 9][1])} for k,r in self.selected.items()
                         if k in self.state['terminal']},
                     model_calls=0, extra_source_attempts=0, attempt_budget_consumed=False, signals_sent=0,
                     resource_locks_acquired=0, evidence_prepared=False, **FLAGS)
        atomic(Path(out) / 'state.json', self.state)
        atomic(Path(out) / 'progress.json', value)
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(RT / 'curated_collaborative3_source_plan_v1.json'))
    parser.add_argument('--plan-sha256', default=PLAN_SHA)
    parser.add_argument('--ms-source', default=str(RT / 'source_curated_review_v2'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--enable-feature', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--max-lines-per-stream', type=int, default=64)
    args = parser.parse_args(argv)
    need(args.enable_feature and args.poll_seconds > 0, 'explicit CPU feature enable and positive poll required')
    out = Path(args.out).resolve()
    plan = strict(Path(args.plan).read_bytes())
    production = Path(plan['out']).resolve()
    need(out != production and not out.is_relative_to(production), 'derived output cannot modify frozen production tree')
    statepath = out / 'state.json'
    state = strict(statepath.read_bytes()) if statepath.exists() else None
    review = load_sampling(args.ms_source)
    adapter = Adapter(args.plan, args.plan_sha256, review, state, max_lines_per_stream=args.max_lines_per_stream)
    while True:
        value = adapter.poll(out)
        print(json.dumps({k:value[k] for k in ('terminal_sources','candidate_complete_sources','candidate_partial_sources',
               'producer_ready_sources','review_required_batch_count','model_calls','signals_sent')}, sort_keys=True), flush=True)
        if args.once:
            return value
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
