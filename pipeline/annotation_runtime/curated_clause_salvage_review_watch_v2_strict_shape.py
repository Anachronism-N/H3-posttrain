"""Independent append-only clause audits; naturally yield to reserved recovery review."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
ROOT = RT.parent
OUT = ROOT / 'curated_clause_salvage_reviews_v1'
STREAM = ROOT / 'curated222939_clause_salvage_v1/review_required.jsonl'
PHASE = 'curated_clause_salvage_review_watch_v2'
LEGACY_PHASE = 'curated_clause_salvage_review_watch_v1'
EDITION = 'clause-salvage-20261025-v1'
MODEL = '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'
MODEL_REF = RT / 'reviewer8b_stage_N6_extension_v2.json'
MODEL_REF_SHA = '5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'
BIND = RT / 'curated_clause_salvage_review_bind_v2_strict_shape.py'
BIND_SHA = '197f5bc77028606c7fdcee2a9e95d7e11543ee4406fa26271d319128312fdd1d'
RESUME = RT / 'curated_clause_salvage_review_resume_snapshot_v2.json'
RESUME_SHA = '8f86ea6d5a8f18dc31a118d75a3522fd2f53ff3dc17139021072fa0757cc8516'
PILOT = RT / 'curated_clause_salvage_review_pilot5_v1.py'
PILOT_SHA = '14079c81cd586780bc058bb4e29efdbb5b81bd14cafd9f26c98a560a36580045'
RECOVERY = RT / 'curated_io_recovery_review3_v1.py'
RECOVERY_SHA = '04852ce45069616a42f088ecd339a45fdfc53ba10beaf1b4f1a8d391b0f92be8'
RECOVERY_QUEUE = ROOT / 'queues/curated_io_recovery_review3_n6g3_v1'
PILOT_SUMMARY = ROOT / 'curated_clause_salvage_reviews_pilot5_v1/run.summary.json'
PILOT_SUMMARY_SHA = '2ca9238809fbc51303b7ba42ee4749baf7226624908d2263901060a9f6c70b05'
FLAGS = dict(semantic_approval=False, human_review_completed=False,
    human_approval_automatically_granted=False, training_eligible=False,
    formal_approval=False, formal_training=False, training_entry_updated=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'Immutable reference SHA changed')
    value = load(reference['path'])
    need(sha(reference['path']) == reference['sha256'], 'Reference changed during read')
    return value


def module(name, path, expected):
    need(sha(path) == expected, 'Frozen dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


pilot = module('salvage_watch_pilot', PILOT, PILOT_SHA)
bind_v2 = module('salvage_watch_bind_v2', BIND, BIND_SHA)
w = pilot.support()
recovery = module('salvage_watch_recovery', RECOVERY, RECOVERY_SHA)


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(load(path) == value, 'Write-once source transaction changed')
        return
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


class AppendCursor:
    def __init__(self, path, state):
        self.path, self.state = Path(path), state
        self.state.setdefault('offset', 0)
        self.prefix = hashlib.sha256()
        if self.path.exists():
            with self.path.open('rb') as f:
                remaining = self.state['offset']
                while remaining:
                    block = f.read(min(8 * 1024 * 1024, remaining))
                    need(bool(block), 'Committed append stream prefix truncated')
                    self.prefix.update(block); remaining -= len(block)
            if self.state.get('committed_prefix_sha256'):
                need(self.prefix.hexdigest() == self.state['committed_prefix_sha256'],
                     'Full committed append stream prefix changed')

    def next(self):
        s = self.state
        if not self.path.exists():
            need(s.get('identity') is None, 'Previously observed stream disappeared')
            return None
        with self.path.open('rb') as f:
            st = os.fstat(f.fileno())
            identity = [st.st_dev, st.st_ino]
            need(s.get('identity') in (None, identity) and st.st_size >= s.get('observed_size', 0)
                 and st.st_size >= s.get('offset', 0), 'Append stream replaced/truncated')
            s['identity'] = identity
            if s.get('anchor_length', 0):
                f.seek(s['anchor_start'])
                need(hashlib.sha256(f.read(s['anchor_length'])).hexdigest() == s['anchor_sha256'],
                     'Committed append stream anchor changed')
            if s.get('partial_length', 0):
                f.seek(s['offset'])
                need(hashlib.sha256(f.read(s['partial_length'])).hexdigest() == s['partial_sha256'],
                     'Incomplete append stream prefix changed')
            f.seek(s.get('offset', 0))
            line = f.readline()
            s['observed_size'] = st.st_size
            if not line or not line.endswith(b'\n'):
                s['partial_length'] = len(line)
                s['partial_sha256'] = hashlib.sha256(line).hexdigest()
                return None
            value = json.loads(line)
            need(isinstance(value, dict), 'Manifest line must be an object')
            return value, line

    def commit(self, line):
        s = self.state
        self.prefix.update(line)
        s['committed_prefix_sha256'] = self.prefix.hexdigest()
        s.update(anchor_start=s.get('offset', 0), anchor_length=len(line),
                 anchor_sha256=hashlib.sha256(line).hexdigest(),
                 offset=s.get('offset', 0) + len(line), partial_length=0, partial_sha256=None)


def recovery_yield():
    safe_path, exit_path = RECOVERY_QUEUE / 'resources_safe.json', RECOVERY_QUEUE / 'worker_exit.json'
    if safe_path.exists() and exit_path.exists():
        safe = checked(dict(path=str(safe_path), sha256='e2d3c0426734ec809d11f026a64a340fe8e194e0e0bcfec7ff3819e7e84c8302'))
        exited = checked(dict(path=str(exit_path), sha256='5f133b233024f84a771df48899a8a2fb203ebe4208c142031e3384340ca42fa3'))
        if all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored')) and exited['exit_code'] == 0:
            return None
    try:
        rows, references = recovery.ready_records()
    except recovery.NotReady:
        return None
    need(len(rows) == 3, 'Recovery ready gate returned wrong source count')
    if (RECOVERY_QUEUE / 'admission.json').exists():
        return None
    return dict(reason='reserved_recovery_yield', recovery_ready_refs=references,
        recovery_source_ids=[r['id'] for r in rows], reserved_queue=str(RECOVERY_QUEUE),
        recovery_admission_absent=True, other_tasks_signaled=False)


def verify_audit(row, entry, review):
    """Verify current source, semantic/input SHA, full receipt and original native evidence."""
    row = review.normalized_record(row)
    need(entry['source_id'] == row['id'] and entry['split'] == row['split']
         and entry['original_batch_id'] == row['batch_id'] and entry['label_edition'] == EDITION
         and entry['label_path'] == row['label_path'] and entry['label_sha256'] == row['label_sha256']
         and entry['semantic_label_sha256'] == review.batch.digest(review.label_payload(row))
         and entry['input_row_sha256'] == review.batch.digest(row)
         and entry['source_refs'] == row['source_refs']
         and entry.get('source_binding') == row.get('source_binding'), 'Review source/label/semantic/input binding changed')
    for key in ('semantic_approval', 'human_review_completed', 'training_eligible'):
        need(entry.get(key) is False, 'Imported audit must remain nonapproval')
    if not entry.get('review_receipt'):
        need(entry['status'] not in ('pass', 'confirmed_error', 'uncertain'), 'Real audit status requires receipt')
        return None
    receipt = checked(entry['review_receipt'])
    transaction = Path(entry['review_receipt']['path']).parent
    need(receipt['status'] == 'candidate' and receipt.get('semantic_approval') is False
         and receipt.get('human_review_completed') is False, 'Real independent nonapproval receipt required')
    required = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
    need(required <= set(receipt['files_sha256']), 'Incomplete real receipt files')
    for name, expected in receipt['files_sha256'].items():
        path = (transaction / name).resolve()
        need(path.parent == transaction.resolve() and sha(path) == expected, 'Receipt file changed or escaped')
    binding = receipt['binding']
    need(load(transaction / 'binding.json') == binding, 'Receipt and binding file differ')
    need(binding['source_id'] == row['id'] and binding['split'] == row['split']
         and binding['input_row_sha256'] == entry['input_row_sha256']
         and binding['label_sha256'] == entry['semantic_label_sha256']
         and binding['label_file_ref']['path'] == row['label_path']
         and binding['label_file_ref']['sha256'] == row['label_sha256'], 'Receipt label/input binding differs')
    need(sha(MODEL_REF) == MODEL_REF_SHA, 'Staged reviewer receipt changed')
    model = load(MODEL_REF)['model_identity']
    need(binding['code_sha256'] == review.code_identity()['sha256'], 'Frozen reviewer code differs from receipt')
    need(binding['model_sha256'] == model['sha256']
         and binding['annotator_model_sha256'] != model['sha256']
         and binding['checkpoint_content_independence_verified'] is True, 'Independent reviewer checkpoint required')
    for reference in [binding['label_file_ref'], binding['annotator_model_reference']] + binding['source_refs']:
        need(sha(reference['path']) == reference['sha256'], 'Receipt source file SHA changed')
    evidence_root = transaction.parent
    evidence_path = evidence_root / 'evidence.json'
    need(sha(evidence_path) == binding['evidence_sha256'], 'Original evidence changed')
    evidence = load(evidence_path)
    need(evidence['binding']['input_row_sha256'] == entry['input_row_sha256']
         and evidence['binding']['label_sha256'] == entry['semantic_label_sha256']
         and evidence['binding']['video_sha256'] == binding['video_sha256'], 'Evidence source binding differs')
    for asset in evidence['source_identity']['files'].values():
        if asset:
            need(sha(asset['path']) == asset['sha256'], 'Original media content SHA changed')
    need(evidence['source_identity']['files']['video_path']['path'] == row['video_path']
         and sha(row['video_path']) == binding['video_sha256'], 'Original video content changed')
    for frame in evidence['frames']:
        need(sha(evidence_root / frame['file']) == frame['sha256'], 'Original RGB frame changed')
    for crop in evidence['hand_rois']:
        need(sha(crop['path']) == crop['sha256'], 'Native ROI changed')
    final = checked(entry['review_final_ref'])
    need(Path(entry['review_final_ref']['path']).resolve() == transaction / 'final.json'
         and final['status'] == 'candidate' and final['json']['status'] == entry['status'], 'Final review/ledger status differs')
    need(entry['label_complete'] is review.label_complete(row), 'Ledger label completeness differs')
    return {**copy.deepcopy(final['json']), 'source_id': row['id'], 'transaction_status': 'candidate',
        'review_receipt': entry['review_receipt'], 'label_sha256': entry['semantic_label_sha256'],
        'evidence': evidence, 'evidence_root': str(evidence_root)}


def previous_reviews():
    need(sha(PILOT_SUMMARY) == PILOT_SUMMARY_SHA, 'Frozen pilot summary changed')
    summary = load(PILOT_SUMMARY)
    values = {}
    need(summary['sampled_sources_reviewed'] == 2 and summary['selected_source_count'] == 2,
         'Exactly two explicit previous real pilot audits required')
    for reference in summary['review_record_refs']:
        need(sha(reference['path']) == reference['sha256'], 'Frozen previous ledger changed')
        lines = Path(reference['path']).read_bytes().splitlines()
        need(len(lines) == 1, 'Previous source ledger scope changed')
        entry = json.loads(lines[0])
        need(entry['status'] == 'pass' and entry['label_complete'] is True, 'Previous completed real pass required')
        key = entry['split'] + '/' + entry['source_id']
        need(key not in values, 'Duplicate previous pilot source')
        values[key] = (entry, reference)
    return values


class Watcher:
    def __init__(self, out, manifest, reviewer, review, *, stream=STREAM, gate=recovery_yield,
                 previous=None, launch_ref=None):
        self.out, self.manifest, self.reviewer, self.review = Path(out), manifest, reviewer, review
        self.gate = gate
        self.previous = previous_reviews() if previous is None else previous
        self.launch_ref = launch_ref
        self.state_path = self.out / 'state.json'
        self.state = load(self.state_path) if self.state_path.exists() else dict(
            schema=PHASE, global_revision=EDITION, input_stream=str(stream),
            total_source_count=222939, expected_batch_count=448, cursor={}, sources={}, **FLAGS)
        need(self.state['schema'] in (PHASE, LEGACY_PHASE) and self.state['global_revision'] == EDITION
             and self.state['input_stream'] == str(stream), 'Incremental state belongs to other input/edition')
        self.cursor = AppendCursor(stream, self.state['cursor'])
        self.yielded = False
        self.yield_evidence = None

    def save(self):
        w.atomic(self.state_path, self.state)

    def process(self, notice):
        key = notice['source_key']
        existing = self.state['sources'].get(key)
        legacy = existing and checked(existing['transaction_ref'])['schema'] == LEGACY_PHASE
        row = self.review.normalized_record((pilot if legacy else bind_v2).bind_notice(notice, self.manifest, w))
        disagreement = row['review_provenance'].get('eligibility_disagreement')
        if existing and not legacy:
            stored = checked(checked(existing['transaction_ref'])['input_ref'])
            for field in ('original_notice_ref', 'original_notice_stream_ref'):
                row['review_provenance'][field] = stored['review_provenance'][field]
            if disagreement:
                disagreement['original_notice_ref'] = row['review_provenance']['original_notice_ref']
                row['review_provenance']['review_delta'] = {**row['review_provenance']['review_delta'], 'review_required': False}
        if not existing:
            bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
            notice_path = self.out / EDITION / bid / 'original_notice.json'
            write_once(notice_path, notice)
            row['review_provenance']['original_notice_ref'] = ref(notice_path)
            row['review_provenance']['original_notice_stream_ref'] = getattr(self, 'current_notice_ref', None)
            if disagreement:
                disagreement['original_notice_ref'] = ref(notice_path)
                row['review_provenance']['review_delta'] = {**row['review_provenance']['review_delta'], 'review_required': False}
            elif key in self.previous:
                # Preserve exact previous request/input bytes when explicitly reusing its real receipt.
                row = self.review.normalized_record(pilot.bind_notice(notice, self.manifest, w))
        key = row['split'] + '/' + row['id']
        identity = dict(source_key=key, notice_sha256=digest(notice),
            label_result_ref=ref(row['label_path']), semantic_label_sha256=self.review.batch.digest(self.review.label_payload(row)),
            input_row_sha256=self.review.batch.digest(row))
        if key in self.state['sources']:
            transaction = checked(self.state['sources'][key]['transaction_ref'])
            need(transaction['identity'] == identity, 'Source label changed within one edition')
            checked(transaction['summary_ref'])
            need(sha(transaction['ledger_ref']['path']) == transaction['ledger_ref']['sha256'], 'Committed source ledger changed')
            return
        bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
        case = self.out / EDITION / bid
        transaction_path = case / 'transaction.json'
        if transaction_path.exists():
            transaction = load(transaction_path)
            need(transaction['identity'] == identity, 'Existing source transaction label/input changed')
            checked(transaction['summary_ref'])
            need(sha(transaction['ledger_ref']['path']) == transaction['ledger_ref']['sha256'], 'Existing source ledger changed')
        else:
            case.mkdir(parents=True, exist_ok=True)
            write_once(case / 'input.json', row)
            required = row['review_provenance']['review_delta']['review_required'] and not disagreement
            pending_status = 'source_eligibility_disagreement' if disagreement else 'no_new_supported_facts'
            if disagreement:
                write_once(case / 'eligibility_disagreement.json', disagreement)
            imported = None
            if not required:
                write_once(case / 'batch.plan.json', dict(schema=self.review.SCHEMA,
                    batch_id=bid, split=row['split'], source_ids=[row['id']], source_count=1,
                    records=[row], semantic_review_completed=False, human_review_completed=False))
                entry = dict(schema='curated_independent_review_record_v1', source_id=row['id'], id=row['id'],
                    split=row['split'], original_batch_id=row['batch_id'], label_edition=EDITION,
                    label_path=row['label_path'], label_sha256=row['label_sha256'],
                    semantic_label_sha256=identity['semantic_label_sha256'], input_row_sha256=identity['input_row_sha256'],
                    status=pending_status, independent_review_status='pending', review_performed=False,
                    review_receipt=None, review_final_ref=None, label_complete=self.review.label_complete(row),
                    source_refs=row['source_refs'], source_binding=row.get('source_binding'),
                    review_provenance=row['review_provenance'], errors=[], uncertainties=[], **FLAGS)
                with (case / 'reviews.jsonl').open('x') as f:
                    f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
                summary = dict(label_edition=EDITION, sampled_reviewed_denominator=0, confirmed_error_count=0,
                    independent_review_status='pending', review_performed=False, status=pending_status,
                    review_records_ref=ref(case / 'reviews.jsonl'), **FLAGS)
            else:
                prior = self.previous.get(key)
                if prior and prior[0]['label_sha256'] == row['label_sha256'] and self.review.label_complete(row):
                    result = verify_audit(row, prior[0], self.review)
                    imported = prior[1]
                    need(result is not None, 'Previous real audit receipt missing')
                else:
                    result = None
                audits = {}
                def audit(normalized, root, **kwargs):
                    value = result if result is not None else self.reviewer(normalized, root, **kwargs)
                    audits[self.review.batch.digest(normalized)] = copy.deepcopy(value)
                    return value
                summary = self.review.review_batch_ready([row], self.out / EDITION, audit, batch_id=bid,
                    expected_source_ids=[row['id']], sample_count=1, risk_count=0, expansion_counts=())
                summary = w.gate_summary(summary, edition=EDITION, provenance=row['review_provenance'])
                if imported:
                    summary['imported_previous_review_ref'] = imported
                w.publish(case, summary, [row], self.review, audits)
                entry = json.loads((case / 'reviews.jsonl').read_bytes())
                if entry['status'] in ('pass', 'confirmed_error', 'uncertain'):
                    verify_audit(row, entry, self.review)
                if imported:
                    entry.update(imported_previous_review_ref=imported, previous_real_audit_explicitly_imported=True,
                                 model_call_performed=False, **FLAGS)
                    self.review.batch.atomic_write(case / 'reviews.jsonl',
                        (json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n').encode())
                    summary['review_records_ref'] = ref(case / 'reviews.jsonl')
            summary.update(schema_extension=PHASE, source_key=key, batch_id=bid,
                original_batch_id=row['batch_id'], original_batch_source_count=row['review_provenance']['original_batch_source_count'],
                original_batch_review_complete=False, all_batches_reviewed=False,
                total_source_count=222939, expected_batch_count=448,
                source_selected_for_risk_review=required,
                original_notice_ref=ref(case / 'original_notice.json'),
                original_notice_sha256=digest(notice),
                original_notice_stream_ref=getattr(self, 'current_notice_ref', None),
                independent_review_eligibility=row['review_provenance'].get('independent_review_eligibility'),
                eligibility_disagreement_ref=ref(case / 'eligibility_disagreement.json') if disagreement else None,
                newsource_error_isolated=bool(disagreement),
                every_selected_case_is_risk_reviewed=required and summary['sampled_reviewed_denominator'] == 1,
                statistical_population_error_estimate=False,
                label_complete=self.review.label_complete(row), input_row_sha256=identity['input_row_sha256'],
                label_result_ref=identity['label_result_ref'], semantic_label_sha256=identity['semantic_label_sha256'], **FLAGS)
            w.atomic(case / 'summary.json', summary)
            transaction = dict(schema=PHASE, identity=identity, review_required=required,
                status=entry['status'], label_complete=self.review.label_complete(row),
                imported_previous_review_ref=imported, summary_ref=ref(case / 'summary.json'),
                ledger_ref=summary['review_records_ref'], input_ref=ref(case / 'input.json'), **FLAGS)
            write_once(transaction_path, transaction)
        self.state['sources'][key] = dict(transaction_ref=ref(transaction_path), status=transaction['status'],
            label_complete=transaction['label_complete'], review_required=transaction['review_required'],
            imported_previous_review_ref=transaction['imported_previous_review_ref'])

    def summary(self, drained=False):
        sources = self.state['sources']
        reviewed = [key for key, v in sources.items() if v['status'] in ('pass', 'confirmed_error', 'uncertain')]
        pending = [key for key, v in sources.items() if key not in reviewed or not v['label_complete']]
        value = dict(schema=PHASE, global_revision=EDITION, label_edition=EDITION,
            total_source_count=222939, expected_batch_count=448, processed_source_count=len(sources),
            sampled_sources_reviewed=len(reviewed), AI_error_candidate_count=sum(v['status'] == 'confirmed_error' for v in sources.values()),
            newsource_error_isolated=sum(v['status'] == 'source_eligibility_disagreement' for v in sources.values()),
            newsource_error_isolated_not_counted_as_reviewed=True,
            no_new_supported_facts_count=sum(v['status'] == 'no_new_supported_facts' for v in sources.values()),
            imported_previous_review_count=sum(v['imported_previous_review_ref'] is not None for v in sources.values()),
            remaining_pending=pending, remaining_pending_count=len(pending),
            remaining_unconsumed_stream_bytes=max(0, self.state['cursor'].get('observed_size', 0) - self.state['cursor'].get('offset', 0)),
            remaining_population_not_claimed_reviewed=222939 - len(reviewed),
            reserved_recovery_yield=self.yielded, recovery_yield_evidence=self.yield_evidence,
            input_drain=drained, input_drain_scope='current_complete_append_stream_snapshot_only' if drained else None,
            allreviewed=False, all_batches_reviewed=False, original_batch_review_complete=False,
            source_records_consumed=drained, statistical_population_error_estimate=False,
            state_ref=ref(self.state_path), launch_ref=self.launch_ref, human_resolution_stream=None, **FLAGS)
        w.atomic(self.out / 'progress.json', value)
        if self.yielded or drained:
            w.atomic(self.out / 'run.summary.json', value)
        return value

    def poll(self, *, drain=False, limit=16):
        # Gate is always read-only and consulted before reading any new audit work.
        self.yield_evidence = self.gate()
        if self.yield_evidence:
            self.yielded = True
            # Observe only the next byte range, without binding or auditing another source.
            self.cursor.next()
            self.save()
            return self.summary()
        at_eof = False
        for _ in range(limit):
            item = self.cursor.next()
            if item is None:
                at_eof = True
                break
            notice, line = item
            self.current_notice_ref = dict(path=str(self.cursor.path), byte_offset=self.state['cursor']['offset'],
                byte_length=len(line), sha256=hashlib.sha256(line).hexdigest())
            self.process(notice)
            self.cursor.commit(line)
            self.save()
            # Any in-flight audit transaction has now durably committed; yield naturally.
            self.yield_evidence = self.gate()
            if self.yield_evidence:
                self.yielded = True
                break
        self.save()
        complete_eof = at_eof and not self.state['cursor'].get('partial_length', 0)
        return self.summary(drained=drain and complete_eof and not self.yielded)


def resume_preflight(out, manifest, review, snapshot=None):
    snapshot = checked(dict(path=str(RESUME), sha256=RESUME_SHA)) if snapshot is None else snapshot
    state_path = Path(out) / 'state.json'
    state = load(state_path)
    need(state['schema'] in (PHASE, LEGACY_PHASE) and state['global_revision'] == EDITION
         and state['input_stream'] == snapshot['input_stream'], 'Resume state source/edition changed')
    baseline = {key:state['sources'][key] for key in snapshot['committed_source_keys']}
    need(digest(baseline) == snapshot['committed_sources_sha256'], 'Original committed source mapping changed')
    need(state['cursor']['offset'] >= snapshot['committed_prefix_bytes']
         and state['cursor']['identity'] == snapshot['cursor']['identity'], 'Resume cursor identity/offset changed')
    with Path(state['input_stream']).open('rb') as f:
        st = os.fstat(f.fileno())
        need([st.st_dev, st.st_ino] == state['cursor']['identity'], 'N6 append stream identity drift')
        prefix = f.read(snapshot['committed_prefix_bytes'])
    need(len(prefix) == snapshot['committed_prefix_bytes'] and hashlib.sha256(prefix).hexdigest() ==
         snapshot['committed_prefix_sha256'], 'Original111 byte prefix changed')
    cursor = AppendCursor(state['input_stream'], copy.deepcopy(state['cursor']))
    cursor.next()  # Verify actual lastcommit anchor, incomplete prefix, device/inode and size before VLM.
    notices = {json.loads(line)['source_key']:json.loads(line) for line in prefix.splitlines()}
    audited = 0
    for key, source in baseline.items():
        transaction = checked(source['transaction_ref'])
        checked(transaction['summary_ref']); row = checked(transaction['input_ref'])
        need(sha(transaction['ledger_ref']['path']) == transaction['ledger_ref']['sha256'], 'Original source ledger changed')
        notice = notices[key]
        current = review.normalized_record(pilot.bind_notice(notice, manifest, w))
        need(current == row and transaction['identity']['notice_sha256'] == digest(notice),
             'Original111 source/input/notice binding changed')
        entry = json.loads(Path(transaction['ledger_ref']['path']).read_bytes())
        if source['status'] in ('pass', 'confirmed_error', 'uncertain'):
            verify_audit(current, entry, review); audited += 1
    need(audited == snapshot['reviewed_source_count'], 'Original45 real audits count changed')
    previous = previous_reviews()
    need(len(previous) == 2, 'Exactly2 pilot receipt references required')
    return dict(schema='clause_review_resume_preflight_v2', original_sources_verified=len(baseline),
        original_real_audits_verified=audited, previous_pilot_refs_verified=len(previous),
        byteprefix_sha256=snapshot['committed_prefix_sha256'], original_cursor=copy.deepcopy(snapshot['cursor']),
        current_cursor=copy.deepcopy(state['cursor']), identity_n6_no_drift=True,
        model_calls=0, original_transactions_modified=False, **FLAGS)


def main():
    p = argparse.ArgumentParser()
    for option in ('out', 'manifest', 'model', 'model-reference', 'annotator-model-reference', 'ms-source', 'launch', 'launch-sha'):
        p.add_argument('--' + option, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=20)
    a = p.parse_args()
    need(Path(a.out).resolve() == OUT and Path(a.manifest).resolve() == pilot.MANIFEST
         and sha(a.manifest) == pilot.MANIFEST_SHA and a.model == MODEL
         and Path(a.model_reference).resolve() == MODEL_REF and sha(MODEL_REF) == MODEL_REF_SHA
         and Path(a.ms_source).resolve() == RT / 'source_curated_review_v2'
         and a.poll_seconds > 0, 'Dedicated frozen reviewer/output/model/full-manifest required')
    launch_ref = dict(path=str(Path(a.launch).resolve()), sha256=a.launch_sha)
    launch = checked(launch_ref)
    need(launch['schema'] == PHASE and launch['total_source_count'] == 222939
         and launch['expected_batch_count'] == 448, 'Full denominator launch plan required')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'reviewer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, a.ms_source)
        import ms_batch_review as review
        preflight = resume_preflight(OUT, w.Manifest(a.manifest), review)
        write_once(Path(a.launch).with_suffix('.resume_preflight.json'), preflight)
        local = review.LocalReviewer(a.model, OUT, model_reference=a.model_reference,
            annotator_model_reference=a.annotator_model_reference,
            ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
        watcher = Watcher(OUT, w.Manifest(a.manifest), w.EditionReviewer(local), review, launch_ref=launch_ref)
        while True:
            value = watcher.poll(drain=not a.watch)
            print(json.dumps(value, ensure_ascii=False), flush=True)
            if watcher.yielded or value['input_drain']:
                break
            time.sleep(a.poll_seconds)


if __name__ == '__main__':
    main()
