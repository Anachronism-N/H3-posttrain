#!/usr/bin/env python3
"""Small derived curated progress/candidate overlay. No producer or audit files are changed."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
FLAGS = dict(semantic_approval=False, human_review_completed=False, formal_approval=False,
             formal_training=False, training_eligible=False, training_approved=False,
             full_pool_updated=False, training_entry_updated=False)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def digest(value):
    # Frozen reviewer hashes indent=2 JSON plus newline, without sorting keys.
    return hashlib.sha256(encoded(value)).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def identity(row):
    sid, split = row.get('source_id', row.get('id')), row.get('split')
    if not split and row.get('source_key'):
        split, _, key_id = row['source_key'].partition('/')
        if key_id != sid:
            raise ValueError('source_key/id mismatch')
    if not isinstance(sid, str) or not sid or split not in ('train', 'val', 'test'):
        raise ValueError('explicit source_id + original split required')
    return split + '/' + sid


def payload(row):
    label = row.get('label')
    if not isinstance(label, dict):
        return None
    return {'dance_style': label.get('dance_style', ''),
            'timed_motion_script': label.get('timed_motion_script', '')}


def refs_match(want, got):
    a = (want or {}).get('csv', {})
    b = (got or {}).get('csv', {})
    return bool(a and all(a.get(k) == b.get(k)
                         for k in ('path', 'sha256', 'raw_row_sha256', 'csv_line')))


def complete(row):
    label = payload(row)
    pending = {'failed', 'partial', 'pending', 'prepared', 'needs_style', 'needs_rgb', 'unknown', 'abstain'}
    return bool(row.get('joint_complete') is True and row.get('all_windows_nonempty') is True
        and label and isinstance(label['dance_style'], str)
        and label['dance_style'].strip().lower() not in ('', 'unknown') and label['timed_motion_script']
        and all(str(row.get(k, '')).lower() not in pending for k in ('visual_status', 'style_status', 'label_status')))


class Overlay:
    def __init__(self, summary_path, state=None):
        self.summary_path = Path(summary_path)
        raw = self.summary_path.read_bytes()
        self.manifest_sha = hashlib.sha256(raw).hexdigest()
        self.summary = json.loads(raw)
        self.batches = {b['batch_id']: b for b in self.summary['batches']}
        if len(self.batches) != len(self.summary['batches']) or sum(b['count'] for b in self.batches.values()) != self.summary['total']:
            raise ValueError('authoritative batch membership/count mismatch')
        self.state = state or {}
        for name in ('streams', 'members', 'review_batches'):
            self.state.setdefault(name, {})
        if self.state.get('manifest_sha256', self.manifest_sha) != self.manifest_sha:
            raise ValueError('manifest changed; use a fresh rebuildable local state cache')
        self.state['manifest_sha256'] = self.manifest_sha
        self.stream_status, self.dirty, self.documents = {}, set(), {}
        if self.state.get('cache_schema') != 'curated_overlay_local_derived_v3':
            self.state['streams'] = {}
        self.state['cache_schema'] = 'curated_overlay_local_derived_v3'
        self.recovery_requests = []
        self.journal_verifier = None

    def checked(self, ref):
        if not isinstance(ref, dict) or not ref.get('path') or not ref.get('sha256'):
            raise ValueError('explicit file path/SHA required')
        key = (ref['path'], ref['sha256'])
        if key not in self.documents:
            raw = Path(ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('reference SHA mismatch: ' + ref['path'])
            self.documents[key] = json.loads(raw)
        return self.documents[key]

    def member(self, row):
        key, bid = identity(row), row.get('batch_id', row.get('original_batch_id'))
        if bid not in self.batches or not key.startswith(self.batches[bid]['split'] + '/'):
            raise ValueError('unknown original batch or split mismatch')
        if bid not in self.state['members']:
            info = self.batches[bid]
            raw = Path(info['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise ValueError('authoritative batch SHA changed')
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            members = {}
            for record in rows:
                member_key = identity(record)
                if member_key in members or record['batch_id'] != bid or record['split'] != info['split']:
                    raise ValueError('duplicate/wrong original batch member')
                members[member_key] = {k: record[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')}
            if len(members) != info['count']:
                raise ValueError('original batch count mismatch')
            self.state['members'][bid] = members
        if key not in self.state['members'][bid]:
            raise ValueError('source outside fixed original batch')
        return self.state['members'][bid][key]

    def clause_original(self, row, result, ref, member):
        edition = 'clause-salvage-20261025-v1'
        root = getattr(self, 'clause_salvage_root', ROOT / 'curated222939_clause_salvage_v1')
        expected = root / 'revisions' / edition / member['split'] / member['source_id'] / 'result.json'
        if (row.get('result_ref') != ref or row.get('joint_candidate_ref') != ref
                or Path(ref['path']).resolve() != expected.resolve()
                or row.get('global_revision') != edition or result.get('global_revision') != edition
                or row.get('source_key') != identity(member) or result.get('source_key') != identity(member)):
            raise ValueError('clause canonical candidate/edition/source-key mismatch')

        def bound(document, name):
            for record in (document, document.get('manifest_record', {})):
                for field, want in (('id', member['source_id']), ('source_id', member['source_id']),
                                    ('split', member['split']), ('batch_id', member['batch_id']),
                                    ('source_key', identity(member))):
                    if field in record and record[field] != want:
                        raise ValueError('clause lineage source/batch mismatch: ' + name)
                refs = record.get('source_refs')
                if refs:
                    refs = {'csv': refs['original_csv']} if 'original_csv' in refs else refs
                    if not refs_match(member['source_refs'], refs):
                        raise ValueError('clause lineage CSV/row mismatch: ' + name)
            if (document.get('id', document.get('source_id')) != member['source_id']
                    and document.get('source_key') != identity(member)):
                raise ValueError('clause lineage source identity missing: ' + name)

        bound(row, 'notice')
        bound(result, 'result')
        old = self.checked(result.get('original_repair_joint_result_ref'))
        bound(old, 'original_repair_joint_result_ref')
        original_joint_ref = old.get('original_result_ref')
        original = self.checked(original_joint_ref)
        bound(original, 'original_joint_candidate_ref')
        if payload(original) is None or identity(original) != identity(member):
            raise ValueError('clause original main JOINT required, not raw VISUAL')
        if old.get('repair_result_ref') != result.get('original_repair_result_ref'):
            raise ValueError('clause original repair result linkage mismatch')
        visual_ref = result.get('original_visual_result_ref')
        if (visual_ref != original.get('visual_result_ref') or visual_ref != old.get('original_visual_result_ref')
                or result.get('original_result_ref') != original_joint_ref
                or result.get('original_job_ref') != old.get('original_job_ref')
                or result.get('style_result_ref') != original.get('style_result_ref')
                or result.get('original_source_binding') != original.get('source_binding', {}).get('source_binding')):
            raise ValueError('clause original JOINT/VISUAL/job/style/scene linkage mismatch')
        for name, value in result.items():
            if name.startswith('original_') and name.endswith('_ref'):
                document = self.checked(value)
                if name != 'original_gpu_load_receipt_ref':
                    bound(document, name)
        job = self.checked(result['original_job_ref'])
        if job.get('original_result_ref') != original_joint_ref:
            raise ValueError('clause collector job original JOINT mismatch')
        if row.get('source_refs') and not refs_match(member['source_refs'], row['source_refs']):
            raise ValueError('clause notice CSV/row mismatch')
        for name, value in row.items():
            if name.startswith('original_') and name.endswith('_ref') and result.get(name) != value:
                raise ValueError('clause notice/result lineage mismatch: ' + name)
        if row.get('visual_result_ref') != result.get('visual_result_ref'):
            raise ValueError('clause notice/new visual reference mismatch')
        visual = self.checked(result.get('visual_result_ref'))
        bound(visual, 'visual_result_ref')
        for name in ('original_result_ref', 'original_visual_result_ref', 'original_repair_joint_result_ref',
                     'original_repair_result_ref', 'original_job_ref'):
            if visual.get(name) != result.get(name):
                raise ValueError('clause new visual/original lineage mismatch: ' + name)
        if Path(result['visual_result_ref']['path']).resolve() != expected.with_name('visual_result.json').resolve():
            raise ValueError('clause canonical visual path mismatch')
        for document in (row, result, visual):
            if (document.get('global_revision') != edition
                    or any(document.get(k) is not False for k in ('VLM_dispatch_performed',
                        'attempt_budget_consumed', 'model_call_performed', 'model_rewrite_performed',
                        'original_raw_or_terminal_modified'))
                    or document.get('model_calls') != 0 or document.get('extra_source_attempts') != 0
                    or any(document.get(k) is True for k in FLAGS)):
                raise ValueError('clause zero-VLM/budget/approval flags mismatch')
        if (result.get('style_label_changed') is not False or visual.get('style_label_changed') is not False
                or payload(result)['dance_style'] != payload(original)['dance_style']
                or payload(result)['timed_motion_script'] != visual.get('final_timed_caption')):
            raise ValueError('clause unchanged style/visual label payload mismatch')
        ids, facts = result.get('new_safe_fact_ids'), result.get('new_safe_facts')
        if (not isinstance(ids, list) or not isinstance(facts, list) or len(set(ids)) != len(ids)
                or ids != [f.get('fact_id') for f in facts]
                or any(document.get('new_safe_fact_ids') != ids or document.get('new_safe_facts') != facts
                       for document in (row, visual))):
            raise ValueError('clause new core fact IDs/payload mismatch')
        admitted = {f.get('fact_id'): f for f in visual.get('facts', [])}
        evidence_windows = {a.get('window_id') for a in visual.get('window_derivation_audit', [])}
        for fact in facts:
            clause = fact.get('source_clause', {})
            if (fact.get('window_id') not in evidence_windows
                    or admitted.get(fact.get('fact_id')) != fact or not fact.get('text')
                    or not fact.get('raw_text') or fact['raw_text'] != clause.get('raw_source_clause')
                    or fact['raw_text'] not in clause.get('original_description', '')
                    or clause.get('source_text_is_verbatim_substring') is not True
                    or fact.get('ai_weak_candidate') is not True
                    or any(fact.get(k) is True for k in FLAGS)):
                raise ValueError('clause safe fact/verbatim/weak-candidate mismatch')
        if row.get('new_safe_fact_refs') != [{'fact_id': fid, 'visual_result_ref': result['visual_result_ref']} for fid in ids]:
            raise ValueError('clause new fact visual reference mismatch')
        for audit in visual.get('window_derivation_audit', []):
            scope, support = audit.get('scope_filter_audit', {}), audit.get('support_filter_audit', {})
            if (audit.get('actual_frame_SHA_verified') is not True
                    or any(scope.get(k) is not True for k in ('actual_part_support_gate_unchanged',
                        'native_named_shape_gate_preserved', 'filter_is_not_semantic_truth_review'))
                    or any(scope.get(k) is not False for k in ('affirmative_text_modified', 'comma_splitting_added',
                        'frame_references_added', 'uncertainty_tokens_removed_to_strengthen_claim'))
                    or support.get('raw_response_changed') is not False
                    or support.get('remaining_reliable_parts_preserved') is not True
                    or support.get('filter_is_not_semantic_truth_review') is not True):
                raise ValueError('clause evidence/named-shape gates mismatch')
        return original_joint_ref

    def journal_module(self):
        if not hasattr(self, '_journal_module'):
            path = Path(__file__).with_name('curated_repair_collector_recovery_v1.py')
            spec = importlib.util.spec_from_file_location('overlay_journal_evidence', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._journal_module = module
        return self._journal_module

    def journal_candidate(self, notice):
        module = self.journal_module()
        root = Path(self.journal_recovery_root).resolve()
        production = Path(self.journal_production_root).resolve()
        row = notice['hint']
        key = identity(row)
        begin, end = notice['source_input_byte_begin'], notice['source_input_byte_end']
        rule = module.KNOWN.get(begin)
        if (not rule or notice.get('schema') != module.SCHEMA
                or notice.get('recovery_policy') != 'exact_offset_SHA_then_authoritative_manifest_and_immutable_terminal_chain'
                or notice.get('source_key') != key or row.get('split') != rule['split']
                or row.get('batch_id') != rule['batch_id'] or row.get('id') not in rule['ids']
                or notice.get('recovered_terminal_count') != len(rule['ids'])
                or notice.get('recovered_terminal_index') != rule['ids'].index(row['id'])
                or end != begin + rule['size']
                or notice.get('source_input_path') != str(production / 'completed.jsonl')):
            raise ValueError('journal recovery unknown bad line/source/terminal scope')
        if any(document.get(flag) is True for document in (notice, row) for flag in FLAGS):
            raise ValueError('journal recovery cannot grant approval')
        hint_ref = {'path': str(root / 'hints' / (key + '.json')), 'sha256': module.q.digest(notice)}
        if self.checked(hint_ref) != notice:
            raise ValueError('journal immutable source hint mismatch')
        receipt, receipt_ref = module.document(root / 'announced' / (key + '.json'))
        if receipt.get('hint_ref') != hint_ref:
            raise ValueError('journal hint/announcement receipt SHA mismatch')
        bad_ref = {'path': str(root / 'badlines' / (str(begin) + '-' + rule['sha256'] + '.raw')),
                   'sha256': rule['sha256']}
        badraw, actual_bad_ref = module.stable_read(bad_ref['path'])
        if notice.get('badline_ref') != bad_ref or actual_bad_ref != bad_ref or len(badraw) != rule['size']:
            raise ValueError('journal original badline SHA/bytes mismatch')
        with (production / 'completed.jsonl').open('rb') as stream:
            stream.seek(begin)
            original_badraw = stream.read(rule['size'])
        if original_badraw != badraw:
            raise ValueError('journal original completed badline changed')
        snapshots = notice.get('stable_badline_snapshots', [])
        if (len(snapshots) != 3 or any(s.get('line_sha256') != rule['sha256'] for s in snapshots)
                or any(s.get('identity') != snapshots[0].get('identity') for s in snapshots)
                or len(snapshots[0].get('identity', [])) != 5):
            raise ValueError('journal stable badline diagnostic mismatch')
        member = self.member(row)
        evidence = notice['evidence']
        terminal = self.checked(evidence['consumer_terminal_ref'])
        if (terminal != row or evidence['consumer_terminal_ref']['sha256'] != notice.get('completed_row_sha256')
                or module.q.digest(terminal) != notice.get('completed_row_sha256')
                or terminal.get('joint_candidate_ref') != evidence.get('joint_result_ref')):
            raise ValueError('journal hint/immutable consumer terminal/JOINT SHA mismatch')
        if self.journal_verifier is None:
            self.journal_verifier = module.Evidence(production, self.summary_path)
        actual_terminal, actual_evidence = self.journal_verifier.terminal(
            member['source_id'], member['split'], member['batch_id'])
        if actual_terminal != terminal or actual_evidence != evidence:
            raise ValueError('journal authoritative manifest/sourceRefs/params/binding evidence mismatch')
        candidate = self.normalize(terminal, 'journal_recovered', 'candidate')
        if candidate['kind'] != 'complete_candidate' or terminal.get('status') != 'joint_candidate':
            raise ValueError('journal recovery requires fully bound complete consumer terminal')
        return {**candidate, 'journal_recovery': {
            'original_ref': {'path': str(production / 'completed.jsonl'), 'byte_begin': begin,
                             'byte_end': end, 'sha256': rule['sha256']},
            'badline_ref': bad_ref, 'hint_ref': hint_ref, 'receipt_ref': receipt_ref,
            'receipt': receipt, 'evidence': evidence,
            'stable_badline_snapshots': snapshots, 'fully_bound': True}}

    def normalize(self, row, origin, kind):
        if kind == 'journal_recovery':
            return self.journal_candidate(row)
        member = self.member(row)
        if kind == 'ready' and row.get('status') == 'failed':
            return {**member, 'origin': origin, 'kind': 'failed', 'producer_notice': row}
        if kind in ('ready', 'terminal'):
            return {**member, 'origin': origin, 'kind': kind,
                    'producer_notice': row if kind == 'ready' else None}
        ref = row.get('joint_candidate_ref', row.get('joint_result_ref', row.get('result_ref')))
        if not ref:
            return {**member, 'origin': origin, 'kind': 'failed', 'producer_notice': row}
        result = self.checked(ref)
        if identity(result) != identity(member) or result.get('batch_id') != member['batch_id']:
            raise ValueError('joint result original source/batch mismatch')
        if not refs_match(member['source_refs'], result.get('source_refs')):
            raise ValueError('joint result CSV/row binding mismatch')
        if result.get('old_failed_producer_terminal_ref') or result.get('recovery_revision_requests_ref'):
            failed_ref = result.get('old_failed_producer_terminal_ref')
            failed = self.checked(failed_ref)
            requests = self.checked(result.get('recovery_revision_requests_ref'))
            if (identity(failed) != identity(member) or failed.get('status') != 'failed'
                    or failed.get('batch_id') != member['batch_id']
                    or not refs_match(member['source_refs'], failed.get('manifest_record', failed).get('source_refs'))
                    or not any(identity(r) == identity(member) and r.get('old_terminal_ref') == failed_ref
                               for r in requests.get('requests', []))):
                raise ValueError('IO recovery original failed terminal/request binding mismatch')
        original_ref = result.get('original_result_ref', result.get('original_joint_candidate_ref'))
        revision = kind in ('joint_revision', 'clause_revision')
        original_joint_ref = original_ref
        if kind == 'clause_revision':
            original_joint_ref = self.clause_original(row, result, ref, member)
        original = None
        if revision:
            original = self.checked(original_joint_ref)
            if identity(original) != identity(member) or not refs_match(member['source_refs'], original.get('source_refs')):
                raise ValueError('repair original joint source binding mismatch')
            if row.get('global_revision') != result.get('global_revision'):
                raise ValueError('repair label edition mismatch')
            for name in ('original_job_ref', 'original_visual_result_ref', 'repair_result_ref'):
                if result.get(name):
                    bound = self.checked(result[name])
                    if bound.get('id', bound.get('source_id')) != member['source_id']:
                        raise ValueError('repair lineage source mismatch: ' + name)
        label = payload(result)
        if row.get('joint_complete') is False and complete(result):
            raise ValueError('notice/result joint completeness mismatch')
        return {**member, 'origin': origin, 'kind': 'complete_candidate' if complete(result) else 'partial_candidate',
                'revision': revision, 'global_revision': result.get('global_revision'),
                'edition': result.get('edition'),
                'recovery_revision_requests_ref': result.get('recovery_revision_requests_ref'),
                'old_failed_producer_terminal_ref': result.get('old_failed_producer_terminal_ref'),
                'joint_candidate_ref': {'path': ref['path'], 'sha256': ref['sha256']},
                'label': label, 'semantic_label_sha256': digest(label) if label else None,
                'source_binding': result.get('source_binding'),
                'original_joint_candidate_ref': original_joint_ref,
                **({'original_result_ref': original_ref,
                    'clause_salvage': True, 'new_safe_fact_ids': result['new_safe_fact_ids'],
                    'new_safe_facts': result['new_safe_facts'], 'new_safe_fact_refs': row['new_safe_fact_refs'],
                    'window_derivation_audit': result.get('window_derivation_audit'),
                    **{k: v for k, v in result.items() if k.startswith('original_') and k.endswith('_ref')},
                    'VLM_dispatch_performed': False, 'attempt_budget_consumed': False, 'model_calls': 0}
                   if kind == 'clause_revision' else {}),
                'original_semantic_label_sha256': digest(payload(original)) if original else None,
                **{k: result.get(k) for k in ('original_job_ref', 'original_visual_result_ref',
                    'repair_result_ref', 'original_source_binding', 'visual_result_ref', 'style_result_ref')}, **FLAGS}

    def stream(self, path, origin, kind, batch_id=None):
        path, records = Path(path), []
        cache = self.state['streams'].get(str(path), {})
        if not path.exists():
            self.state['streams'].pop(str(path), None)
            self.stream_status[str(path)] = {'state': 'pending_missing', 'new_records': 0}
            return
        with path.open('rb') as stream:
            st = os.fstat(stream.fileno())
            offset = cache.get('offset', 0)
            stream.seek(0)
            head = stream.read(min(offset, 64)).hex()
            stream.seek(max(0, offset - 128))
            tail = stream.read(min(offset, 128)).hex()
            reset = bool(cache and ((st.st_dev, st.st_ino) != tuple(cache['file_identity'])
                         or st.st_size < offset or head != cache['head'] or tail != cache['tail']
                         or any(r.get('kind') == 'invalid' and r.get('invalid_kind') != 'journal_decode'
                                for r in cache['records'])))
            if reset:
                offset = 0
                self.dirty.update(r['batch_id'] for r in cache['records'] if r.get('batch_id'))
            else:
                records = cache.get('records', [])
            stream.seek(offset)
            new_count, partial = 0, False
            while True:
                pos = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    partial = True
                    break
                if line.strip():
                    try:
                        row = json.loads(line)
                        if batch_id is not None:
                            row = {**row, 'batch_id': batch_id}
                        record = self.normalize(row, origin, kind)
                        self.dirty.add(record['batch_id'])
                    except (ValueError, KeyError, OSError, TypeError) as error:
                        record = {'kind': 'invalid', 'path': str(path), 'position': pos, 'reason': str(error),
                                  'origin': origin, 'stream_kind': kind,
                                  'invalid_kind': 'journal_decode' if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)) else 'reference_binding',
                                  'byte_end': pos + len(line), 'raw_size': len(line),
                                  'raw_sha256': hashlib.sha256(line).hexdigest()}
                    records.append(record)
                    new_count += 1
                offset = stream.tell()
            stream.seek(0)
            head = stream.read(min(offset, 64)).hex()
            stream.seek(max(0, offset - 128))
            tail = stream.read(min(offset, 128)).hex()
        self.state['streams'][str(path)] = {'file_identity': [st.st_dev, st.st_ino], 'offset': offset,
                                           'head': head, 'tail': tail, 'records': records}
        self.stream_status[str(path)] = {'state': 'partial_line_pending' if partial else 'caught_up',
            'new_records': new_count, 'cursor': offset, 'reset_after_truncation': reset}

    def active_batches(self):
        return {r['batch_id'] for s in self.state['streams'].values() for r in s['records'] if r.get('batch_id')}

    def audit(self, receipt_ref, final_ref, label_ref, member, semantic_sha, input_sha=None):
        receipt, final, label = self.checked(receipt_ref), self.checked(final_ref), self.checked(label_ref)
        binding = receipt.get('binding', {})
        if (binding.get('source_id') != member['source_id'] or binding.get('split') != member['split']
                or binding.get('label_sha256') != semantic_sha or digest(payload(label)) != semantic_sha
                or not refs_match(member['source_refs'], label.get('source_refs'))
                or identity(label) != identity(member)):
            raise ValueError('independent review source/semantic-label SHA mismatch')
        if input_sha and binding.get('input_row_sha256') != input_sha:
            raise ValueError('review input-row SHA mismatch')
        bound_ref = binding.get('label_file_ref', {})
        if (bound_ref.get('path'), bound_ref.get('sha256')) != (label_ref['path'], label_ref['sha256']):
            raise ValueError('review joint file SHA mismatch')
        if receipt.get('files_sha256', {}).get('final.json') != final_ref['sha256']:
            raise ValueError('review final/receipt SHA mismatch')
        status = final.get('json', {}).get('status')
        if final.get('status') != 'candidate' or final.get('format_valid') is not True:
            return None
        if status not in ('pass', 'uncertain', 'confirmed_error'):
            return None
        return {**member, 'status': status, 'semantic_label_sha256': semantic_sha,
            'joint_candidate_ref': {'path': label_ref['path'], 'sha256': label_ref['sha256']},
            'review_receipt': receipt_ref, 'review_final_ref': final_ref,
            'independent': binding.get('checkpoint_content_independence_verified') is True, **FLAGS}

    def review_batch(self, directory, bid, audit_bid=None, clause_edition=None):
        directory = Path(directory)
        summary_path = directory / 'summary.json'
        if not summary_path.exists():
            self.state['review_batches'].pop(str(directory), None)
            self.stream_status[str(summary_path)] = {'state': 'pending_missing_review'}
            return
        try:
            raw = summary_path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            cached = self.state['review_batches'].get(str(directory))
            if (cached and cached['summary_sha256'] == sha
                    and (not clause_edition or cached.get('clause_edition') == clause_edition)):
                return
            summary, reviews = json.loads(raw), []
            ledger_ref = summary.get('review_records_ref')
            inputs = {}
            if clause_edition:
                if (not ledger_ref or summary.get('label_edition') != clause_edition
                        or summary.get('batch_id') != audit_bid or summary.get('original_batch_id') != bid):
                    raise ValueError('clause review explicit ledger/edition/batch mismatch')
                plan = json.loads((directory / 'batch.plan.json').read_bytes())
                if plan['batch_id'] != audit_bid or plan['split'] != self.batches[bid]['split']:
                    raise ValueError('clause review fixed input plan mismatch')
                inputs = {identity(r): r for r in plan['records']}
                if len(inputs) != len(plan['records']):
                    raise ValueError('clause review duplicate input source')
            if ledger_ref:
                ledger_raw = Path(ledger_ref['path']).read_bytes()
                if hashlib.sha256(ledger_raw).hexdigest() != ledger_ref['sha256'] or (ledger_raw and not ledger_raw.endswith(b'\n')):
                    raise ValueError('review ledger SHA/complete-line mismatch')
                for line in ledger_raw.splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get('status') not in ('pass', 'confirmed_error', 'uncertain') or not row.get('review_receipt'):
                        continue
                    member = self.member({**row, 'batch_id': row['original_batch_id']})
                    if member['batch_id'] != bid:
                        raise ValueError('review ledger original batch mismatch')
                    if clause_edition:
                        input_row = inputs[identity(member)]
                        label_ref = {'path': row['label_path'], 'sha256': row['label_sha256']}
                        label = self.checked(label_ref)
                        canonical = self.clause_salvage_root / 'revisions' / clause_edition / member['split'] / member['source_id'] / 'result.json'
                        if (row.get('label_edition') != clause_edition
                                or label.get('global_revision') != clause_edition
                                or Path(row['label_path']).resolve() != canonical.resolve()
                                or digest(input_row) != row['input_row_sha256']
                                or input_row.get('label_path') != row['label_path']
                                or input_row.get('label_sha256') != row['label_sha256']
                                or digest(payload(input_row)) != row['semantic_label_sha256']
                                or not refs_match(member['source_refs'], row.get('source_refs'))):
                            raise ValueError('clause review edition/input-row/label binding mismatch')
                    audit = self.audit(row['review_receipt'], row['review_final_ref'],
                        {'path': row['label_path'], 'sha256': row['label_sha256']}, member,
                        row['semantic_label_sha256'], row['input_row_sha256'])
                    if audit:
                        if clause_edition and audit['status'] != row['status']:
                            raise ValueError('clause review ledger/final status mismatch')
                        reviews.append(audit)
            else:
                # Legacy review: fixed plan + source hash + edition hash, never walk source dirs.
                plan = json.loads((directory / 'batch.plan.json').read_bytes())
                if plan['batch_id'] != (audit_bid or bid) or plan['split'] != self.batches[bid]['split']:
                    raise ValueError('legacy review fixed batch mismatch')
                rows = {identity(r): r for r in plan['records']}
                for sid in summary.get('reviewed_source_ids', []):
                    row = rows[self.batches[bid]['split'] + '/' + sid]
                    member = self.member(row)
                    revision = digest({'label': payload(row), 'label_path': row.get('label_path'),
                        'input_row_sha256': digest(row), 'annotator_model_sha256': row.get('annotator_model_sha256')})[:16]
                    path = directory / 'sources' / hashlib.sha256(sid.encode()).hexdigest()[:24] / revision / 'review'
                    receipt_raw = (path / 'receipt.json').read_bytes()
                    receipt = json.loads(receipt_raw)
                    audit = self.audit({'path': str(path / 'receipt.json'), 'sha256': hashlib.sha256(receipt_raw).hexdigest()},
                        {'path': str(path / 'final.json'), 'sha256': receipt['files_sha256']['final.json']},
                        receipt['binding']['label_file_ref'], member, digest(payload(row)), digest(row))
                    if audit:
                        reviews.append(audit)
            self.state['review_batches'][str(directory)] = {'summary_sha256': sha, 'records': reviews,
                                                          'clause_edition': clause_edition}
        except (ValueError, KeyError, OSError, TypeError) as error:
            self.state['review_batches'].pop(str(directory), None)
            self.stream_status[str(summary_path)] = {'state': 'pending_review_binding', 'reason': str(error)}

    def recoveries(self, mapping):
        if not Path(mapping).exists():
            self.stream_status[str(mapping)] = {'state': 'pending_missing_recovery_mapping'}
            return
        raw = Path(mapping).read_bytes()
        document = json.loads(raw)
        for request in document['requests']:
            member = self.member(request)
            old = self.checked(request['old_terminal_ref'])
            if (identity(old) != identity(member) or old.get('status') != 'failed'
                    or old.get('batch_id') != member['batch_id']
                    or not refs_match(member['source_refs'], old.get('manifest_record', old).get('source_refs'))):
                raise ValueError('recovery request original failed source/batch/CSV binding mismatch')
            self.recovery_requests.append({**member, **request,
                'mapping_ref': {'path': str(mapping), 'sha256': hashlib.sha256(raw).hexdigest()},
                'overlay_status': 'explicit_producer_alternative_pending_not_joint', **FLAGS})

    def snapshot(self):
        grouped = defaultdict(lambda: {'alternatives': [], 'ready': {}, 'terminal': False, 'failures': []})
        invalid = []
        for stream in self.state['streams'].values():
            for row in stream['records']:
                if row['kind'] == 'invalid':
                    invalid.append(row)
                    continue
                source = grouped[identity(row)]
                source['member'] = {k: row[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')}
                if row['kind'] == 'ready':
                    source['ready'][row['origin']] = row
                elif row['kind'] == 'terminal':
                    source['terminal'] = True
                elif row['kind'] == 'failed':
                    if row not in source['failures']:
                        source['failures'].append(row)
                else:
                    existing = next((a for a in source['alternatives']
                                     if a['joint_candidate_ref'] == row['joint_candidate_ref']), None)
                    if existing is None:
                        source['alternatives'].append(row)
                    elif row.get('journal_recovery'):
                        recoveries = existing.setdefault('journal_recoveries', [])
                        recovery = {'origin': row['origin'], **row['journal_recovery']}
                        if recovery not in recoveries:
                            recoveries.append(recovery)
        for request in self.recovery_requests:
            source = grouped[identity(request)]
            source['member'] = {k: request[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')}
            failure = {**source['member'], 'origin': 'production', 'kind': 'failed',
                       'producer_terminal_ref': request['old_terminal_ref']}
            source['failures'].append(failure)
        reviews = defaultdict(list)
        for batch in self.state['review_batches'].values():
            for row in batch['records']:
                reviews[identity(row)].append(row)
        counts = Counter({k: 0 for k in ('complete_candidate', 'partial_candidate', 'failed', 'ready_not_joint',
            'terminal_without_result', 'missing', 'ready_unique', 'terminal_unique', 'reviewed', 'sampled_pass',
            'pendinghuman', 'confirmed_AI_error', 'sources_with_failed_attempts')})
        split_counts, batch_counts = defaultdict(Counter), defaultdict(Counter)
        export = []
        for key, source in grouped.items():
            alternatives = source['alternatives']
            for candidate in alternatives:
                matched = [r for r in reviews[key] if r['joint_candidate_ref'] == candidate['joint_candidate_ref']
                    and r['semantic_label_sha256'] == candidate['semantic_label_sha256'] and r['independent']]
                candidate['reviews'] = matched
                candidate['reviewed'] = bool(matched)
                candidate['sampled_pass'] = any(r['status'] == 'pass' for r in matched) and (
                    not candidate.get('clause_salvage') or candidate['kind'] == 'complete_candidate')
                candidate['pendinghuman'] = any(r['status'] in ('confirmed_error', 'uncertain') for r in matched)
            # Repairs remain explicit alternatives; never replace a complete original label automatically.
            base = [r for r in alternatives if not r['revision']]
            order = {'collaborative': 0, 'io_recovery': 1, 'production': 2}
            base.sort(key=lambda r: (r['kind'] != 'complete_candidate', not r['reviewed'], order.get(r['origin'], 3)))
            selected = base[0] if base else (alternatives[0] if alternatives else None)
            failed_origins = {r['origin'] for r in source['failures']}
            new_ready_alternative = any(origin not in failed_origins for origin in source['ready'])
            state = selected['kind'] if selected else ('ready_not_joint' if new_ready_alternative else
                'failed' if source['failures'] else 'ready_not_joint' if source['ready'] else 'terminal_without_result')
            counts['sources_with_failed_attempts'] += bool(source['failures'])
            counts[state] += 1
            counts['ready_unique'] += bool(source['ready'])
            counts['terminal_unique'] += source['terminal']
            candidate_refs = {r['joint_candidate_ref']['sha256'] for r in alternatives}
            bound = [r for r in reviews[key] if r['joint_candidate_ref']['sha256'] in candidate_refs]
            pending = any(r['independent'] and r['status'] in ('confirmed_error', 'uncertain') for r in bound)
            counts['pendinghuman'] += pending
            counts['confirmed_AI_error'] += any(r['independent'] and r['status'] == 'confirmed_error' for r in bound)
            if selected:
                counts['reviewed'] += selected['reviewed']
                counts['sampled_pass'] += selected['sampled_pass']
            member = source['member']
            export.append({**member, 'state': state, 'selection': selected, 'alternatives': alternatives,
                'ready_alternatives': list(source['ready'].values()), 'failed_attempts': source['failures'],
                'pendinghuman': pending, 'overlay_kind': 'candidate_not_training_approved', **FLAGS})
            split_counts[member['split']][state] += 1
            batch_counts[member['batch_id']][state] += 1
        counts['missing'] = self.summary['total'] - len(grouped)
        for split, total in self.summary['counts'].items():
            split_counts[split]['missing'] = total - sum(split_counts[split].values())
        batches = []
        for bid, info in self.batches.items():
            counter = batch_counts[bid]
            counter['missing'] = info['count'] - sum(counter.values())
            batches.append({'batch_id': bid, 'split': info['split'], 'expected': info['count'],
                'counts': dict(counter), 'all_joint_complete': counter['complete_candidate'] == info['count']})
        recovered = [r for stream in self.state['streams'].values() for r in stream['records']
                     if r.get('origin') == 'journal_recovered' and r.get('journal_recovery')]
        journal_invalid = [r for r in invalid if r.get('stream_kind') == 'candidate'
                           and r.get('invalid_kind') == 'journal_decode']
        diagnostics = []
        for record in journal_invalid:
            rule = self.journal_module().KNOWN.get(record['position'])
            known = bool(rule and record.get('origin') == 'production'
                         and record.get('raw_sha256') == rule['sha256']
                         and record.get('raw_size') == rule['size'])
            bound = {identity(r) for r in recovered
                     if r['journal_recovery']['original_ref'] == {
                         'path': record['path'], 'byte_begin': record['position'],
                         'byte_end': record['byte_end'], 'sha256': record['raw_sha256']}}
            expected = {rule['split'] + '/' + sid for sid in rule['ids']} if known else set()
            diagnostics.append({'original_invalid_record': record, 'badline_known': known,
                'recovered_source_count': len(bound), 'fully_bound_diagnostic': known and bound == expected,
                'expected_source_keys': sorted(expected), 'recovered_source_keys': sorted(bound)})
        rejected_hints = [r for r in invalid if r.get('stream_kind') == 'journal_recovery']
        partial_journals = [path for path, status in self.stream_status.items()
                            if status.get('state') == 'partial_line_pending'
                            and (Path(path).name in ('completed.jsonl', 'derived-hints.jsonl'))]
        pending_journal = bool(rejected_hints or partial_journals
                               or any(not d['fully_bound_diagnostic'] for d in diagnostics))
        journal_scope = {'badline_known': bool(diagnostics) and all(d['badline_known'] for d in diagnostics),
            'recovered_source_count': len({identity(r) for r in recovered}),
            'fully_bound_diagnostic': bool(diagnostics) and not pending_journal,
            'pending_diagnostic': pending_journal, 'badline_diagnostics': diagnostics,
            'rejected_hint_records': rejected_hints, 'partial_line_pending_streams': partial_journals,
            'original_invalid_records_preserved': True,
            'derived_hint_stream_incremental': True}
        progress = {'schema': 'curated_candidate_overlay_v2', 'updated_unix': time.time(),
            'authoritative_manifest': {'path': str(self.summary_path), 'sha256': self.manifest_sha},
            'expected_sources': self.summary['total'], 'expected_batches': len(self.batches),
            'counts': dict(counts), 'split_counts': {k: dict(v) for k,v in split_counts.items()}, 'batches': batches,
            'all_sources_joint_complete': counts['complete_candidate'] == self.summary['total'] and not pending_journal,
            'all_batches_joint_complete': all(b['all_joint_complete'] for b in batches) and not pending_journal,
            'journal_recovery_scope': journal_scope,
            'reviewed_is_sampled_not_population_semantic_approval': True,
            'human_resolution_stream': None, 'all_batches_reviewed': False,
            'full_batch_semantic_pass_claimed': False, 'human_approval_automatically_granted': False,
            'recovery_requests': self.recovery_requests,
            'stream_status': self.stream_status, 'invalid_records': invalid,
            'active_batch_count': len(self.active_batches()), 'dirty_batch_count': len(self.dirty),
            'cache_derived_rebuildable_not_authoritative': True,
            'references_verified_on_new_records_not_full_media_rescan': True,
            'selection_policy': 'reviewed_complete_baseline_then_collaborative_io_main; joint_repairs_remain_alternatives', **FLAGS}
        export.sort(key=lambda r: r['source_ordinal'])
        return progress, export


def run(args):
    out, cache = Path(args.out).resolve(), Path(args.state_cache).resolve()
    if not (cache == Path('/tmp') or Path('/tmp') in cache.parents or Path('/opt') in cache.parents):
        raise ValueError('state cache must be local under /tmp or /opt, never shared Ceph')
    if cache == out or cache in out.parents or out in cache.parents:
        raise ValueError('local cache must be disjoint from shared JSON output')
    protected = [Path(args.manifest_summary).parent, args.production, args.collaborative, args.io_recovery,
                 args.repair, args.review, args.review_extension, args.recovery_reviews,
                 args.clause_salvage, args.clause_reviews, args.clause_pilot_reviews, args.review30000]
    protected.extend(args.additional_production)
    protected.append(args.journal_recovery_root)
    if args.collaborative_v2_review_root is not None:
        protected.append(args.collaborative_v2_review_root)
    for source in map(lambda p: Path(p).resolve(), protected):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('overlay output must be disjoint from input/old progress directories')
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(cache.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(cache.read_bytes()) if cache.exists() else None
        overlay = Overlay(args.manifest_summary, state)
        overlay.clause_salvage_root = Path(args.clause_salvage)
        overlay.journal_recovery_root = Path(args.journal_recovery_root)
        overlay.journal_production_root = Path(args.production)
        overlay.recoveries(args.recovery_requests)
        producers = [('production', Path(args.production)), ('collaborative', Path(args.collaborative)),
                     ('io_recovery', Path(args.io_recovery))]
        producers.extend(('additional:' + str(Path(root)), Path(root)) for root in args.additional_production)
        for origin, root in producers:
            overlay.stream(root / 'ready.jsonl', origin, 'ready')
            overlay.stream(root / 'completed.jsonl', origin, 'candidate')
        overlay.stream(Path(args.journal_recovery_root) / 'derived-hints.jsonl',
                       'journal_recovered', 'journal_recovery')
        overlay.stream(Path(args.repair) / 'joint/review-required.jsonl', 'joint_repair', 'joint_revision')
        overlay.stream(Path(args.clause_salvage) / 'review_required.jsonl', 'clause_salvage', 'clause_revision')
        # Only batches whose sources appeared in a stream are addressed; 448 descriptors stay in-memory.
        active = overlay.active_batches()
        for origin, root in producers:
            origin_active = {r['batch_id'] for s in overlay.state['streams'].values() for r in s['records']
                             if r.get('batch_id') and (r.get('origin') == origin
                                 or origin == 'production' and r.get('origin') == 'journal_recovered')}
            for bid in sorted(origin_active):
                path = root / 'batches' / bid / 'terminal_index.jsonl'
                overlay.stream(path, origin, 'terminal', batch_id=bid)
        for bid in sorted(active):
            overlay.review_batch(Path(args.review) / bid, bid)
            overlay.review_batch(Path(args.review_extension) / 'collaborative-v1' / bid, bid)
            overlay.review_batch(Path(args.review30000) / bid, bid)
        if args.collaborative_v2_review_root is not None:
            v2_origin = 'additional:' + str(args.root / 'curated222939_collaborative_v2')
            v2_active = {r['batch_id'] for s in overlay.state['streams'].values() for r in s['records']
                         if r.get('origin') == v2_origin and r.get('batch_id')}
            for bid in sorted(v2_active):
                overlay.review_batch(Path(args.collaborative_v2_review_root) / bid, bid)
        revisions = {(r['global_revision'], r['split'], r['source_id'], r['batch_id'])
            for s in overlay.state['streams'].values() for r in s['records']
            if r.get('revision') and not r.get('clause_salvage') and r.get('global_revision')}
        for edition, split, sid, bid in revisions:
            if Path(edition).name != edition or edition in ('.', '..'):
                raise ValueError('unsafe revision edition')
            audit_bid = split + '-' + hashlib.sha256(sid.encode()).hexdigest()[:24]
            overlay.review_batch(Path(args.review_extension) / 'joint-revisions' / edition / audit_bid, bid, audit_bid)
        recovery_candidates = {(r['edition'], r['split'], r['source_id'], r['batch_id'])
            for s in overlay.state['streams'].values() for r in s['records']
            if r.get('origin') == 'io_recovery' and r.get('joint_candidate_ref') and r.get('edition')}
        recovery_review_root = Path(args.recovery_reviews)
        if not recovery_candidates:
            overlay.stream_status[str(recovery_review_root)] = {'state': 'conditional_pending_consumer_candidates'}
        for edition, split, sid, bid in recovery_candidates:
            if Path(edition).name != edition or edition in ('.', '..'):
                raise ValueError('unsafe IO recovery edition')
            audit_bid = split + '-' + hashlib.sha256(sid.encode()).hexdigest()[:24]
            overlay.review_batch(recovery_review_root / edition / audit_bid, bid, audit_bid)
        clause_candidates = {(r['global_revision'], r['split'], r['source_id'], r['batch_id'])
            for s in overlay.state['streams'].values() for r in s['records'] if r.get('clause_salvage')}
        for edition, split, sid, bid in sorted(clause_candidates):
            if Path(edition).name != edition or edition in ('.', '..'):
                raise ValueError('unsafe clause revision edition')
            audit_bid = split + '-' + hashlib.sha256(sid.encode()).hexdigest()[:24]
            for root in (args.clause_reviews, args.clause_pilot_reviews):
                overlay.review_batch(Path(root) / edition / audit_bid, bid, audit_bid, clause_edition=edition)
        progress, candidates = overlay.snapshot()
        clause_alternatives = [r for source in candidates for r in source['alternatives'] if r.get('clause_salvage')]
        progress['clause_salvage_review_scope'] = {
            'source_count_with_joint_candidates': len(clause_candidates),
            'complete_candidate': sum(r['kind'] == 'complete_candidate' for r in clause_alternatives),
            'partial_candidate': sum(r['kind'] == 'partial_candidate' for r in clause_alternatives),
            'reviewed': sum(r['reviewed'] for r in clause_alternatives),
            'sampled_pass': sum(r['sampled_pass'] for r in clause_alternatives),
            'review_roots': [str(args.clause_reviews), str(args.clause_pilot_reviews)],
            'original_batch_review_complete': False, 'conditional_source_subset_not_full_original_batch': True}
        progress['io_recovery_review_scope'] = {'source_count_with_joint_candidates': len(recovery_candidates),
            'review_root': str(recovery_review_root), 'original_batch_review_complete': False,
            'conditional_source_subset_not_full_original_batch': True}
        progress['state_cache'] = {'path': str(cache), 'local_only': True, 'rebuildable': True, 'authority': False}
        atomic(out / 'candidates.jsonl', b''.join((json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n').encode() for r in candidates))
        atomic(out / 'progress.json', encoded(progress))
        atomic(cache, encoded(overlay.state))
        return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--collaborative-v2-review-root', type=Path,
                   help='Optional future v2 review root; only fixed active original batches are read')
    p.add_argument('--additional-production', action='append', type=Path,
                   help='Independent production root, deduplicated against original source/split membership')
    for name in ('out', 'state-cache', 'manifest-summary', 'production', 'collaborative', 'io-recovery',
                 'repair', 'review', 'review-extension', 'recovery-requests', 'recovery-reviews',
                 'clause-salvage', 'clause-reviews', 'clause-pilot-reviews', 'review30000', 'journal-recovery-root'):
        p.add_argument('--' + name, type=Path)
    return p


def defaults(args):
    if args.additional_production is None:
        args.additional_production = [args.root / 'curated222939_collaborative_v2']
    for name, suffix in {'out': 'curated222939_overlay_v2', 'manifest_summary': 'curated222939_inputs_v1/manifest.summary.json',
        'production': 'curated222939_production_v1', 'collaborative': 'curated222939_collaborative_v1',
        'io_recovery': 'curated222939_io_failure_recovery_v1', 'repair': 'curated222939_repair_v1',
        'review': 'curated_independent_reviews_v1', 'review_extension': 'curated_independent_reviews_extension_v1',
        'recovery_requests': 'runtime/curated_io_failure_recovery500_revision_requests_v1.json',
        'recovery_reviews': 'curated_io_failure_recovery_reviews3_v1',
        'clause_salvage': 'curated222939_clause_salvage_v1',
        'clause_reviews': 'curated_clause_salvage_reviews_v1',
        'clause_pilot_reviews': 'curated_clause_salvage_reviews_pilot5_v1',
        'review30000': 'curated_independent_reviews30000_v1',
        'collaborative_v2_review_root': 'curated_independent_reviews_collaborative_v2',
        'journal_recovery_root': 'curated222939_repair_v1/collector-recovery-v1'}.items():
        if getattr(args, name) is None:
            setattr(args, name, args.root / suffix)
    if args.state_cache is None:
        tag = hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:16]
        args.state_cache = Path('/tmp') / ('curated_overlay_' + tag + '.cache.json')
    return args


if __name__ == '__main__':
    result = run(defaults(parser().parse_args()))
    print(json.dumps({k: result[k] for k in ('counts', 'expected_sources', 'expected_batches', 'training_approved')}))
