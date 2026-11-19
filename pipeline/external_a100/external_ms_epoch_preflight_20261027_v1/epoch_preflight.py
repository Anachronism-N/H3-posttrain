"""Pure CPU, in-memory transition evidence preflight; never a launch permission.

validate_transition_contract(contract, evidence_blobs, trusted_expected, now_unix)
accepts a JSON object or strict UTF-8 JSON bytes. evidence_blobs maps canonical
absolute reference paths to caller-supplied bytes; paths are labels, never opened.
trusted_expected is an independently supplied boundary, NOT a contract field.

Required contract keys: schema, identity, refs, code_refs, approval_refs.
Required trusted keys: identity, refs, code_refs, approval_refs,
original_assignment_ref, source_claim_namespace, lock_identity, budgets,
exact_source_count. refs has the twelve names in REF_NAMES. The original
assignment blob is a bounded, independently anchored *complete lane projection*,
not a claim that the full on-disk original manifest was read or verified. Its
original_assignment_ref binds the original manifest digest from the boundary.
Release claim lists are complete bounded original-own-lane projections within
the unchanged source-claim namespace; they are not a full global claim scan.
Old ticket/queue/progress/lease are explicit small binding-evidence envelopes;
this schema is NOT an extension accepted by the old exact-20000/two-ticket runner.

Three separately anchored approval documents bind retirement/no-resume, exact
source budget, and resource/code scope to this transition. An authorized boolean,
self-signed document, or a claimed signature is never an independent anchor.
The caller is responsible for independently obtaining these trusted inputs;
there is no authentication/signature system here. All successful outputs remain
structural only and require fresh local checks by a separate future system.
"""

import hashlib
import json
import math
import re

SCHEMA = 'ms_epoch_transition_preflight_v1'
MAX_BLOB_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_ROWS = 8192
MAX_REFS = 64
MAX_INTEGER = 2**63 - 1
GIB = 2**30
REQUIRED_CODE_NAMES = {
    "ms_unified_novlm_v1.py", "ms_fullframe_pose2d_draft_v1.py",
    "ms_fullframe_pose2d_draft_v2.py", "ms_pose2d_draft.py", "dwpose_full_extract.py",
    "unified_lane_batchrpc_next_v2.py", "unified_access_transport_batchrpc_next_v2.py",
    "batchrpc_publication_next_v2.py", "beatthis_rejection_next_v2.py",
}
REF_NAMES = (
    'frozen_manifest', 'independent_acceptance', 'old_ticket', 'old_assignment',
    'old_progress', 'old_lease', 'original_queue', 'retirement', 'source_release',
    'resource_snapshot', 'next_ticket', 'next_queue',
)
APPROVAL_NAMES = ('retirement_no_resume', 'exact_source_budget', 'resource_code_scope')
IDENTITY_KEYS = {
    'transition_id', 'predecessor_epoch', 'next_epoch', 'old_run_id', 'next_run_id',
    'worker', 'physical_gpu_index', 'gpu_uuid', 'host', 'boot_id', 'old_pid',
    'old_start_ticks',
}
FRESH_CHECKS = (
    'local_original_physical_lock_FD_dev_inode_and_exclusive_ownership',
    'local_exact_old_PID_start_ticks_host_boot_exit_and_no_foreign_lease',
    'local_preserved_old_lease_and_progress_SHA_and_no_newepoch_lease_or_rollover_progress',
    'local_physical_GPU_index_UUID_compute_graphics_PIDs_empty_and_memory_zero',
    'local_original_claim_namespace_quiet_no_prefetch_and_exact_suffix_unclaimed',
    'local_node_effective_headroom_and_disk_budget_once_not_per_GPU',
)


class _Invalid(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Invalid(reason)


def _equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _int(value, minimum=0, maximum=MAX_INTEGER):
    return type(value) is int and minimum <= value <= maximum


def _text(value):
    return type(value) is str and 0 < len(value) <= 512 and '\0' not in value


def _canonical(value):
    if not _text(value) or not value.startswith('/') or value.startswith('//'):
        return False
    parts = value.split('/')[1:]
    return bool(parts) and all(part not in ('', '.', '..') for part in parts)


def _exact(value, keys, reason):
    _require(type(value) is dict and set(value) == set(keys), reason)


def _safe_tree(value, depth=0, counter=None):
    if counter is None:
        counter = [0]
    counter[0] += 1
    _require(depth <= 24 and counter[0] <= 400000, 'json_structure_bound_exceeded')
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str and '\0' not in key, 'invalid_json_key')
            _safe_tree(item, depth + 1, counter)
    elif type(value) is list:
        for item in value:
            _safe_tree(item, depth + 1, counter)
    elif type(value) is str:
        _require('\0' not in value, 'json_NUL_rejected')
    elif type(value) is int:
        _require(-MAX_INTEGER <= value <= MAX_INTEGER, 'json_integer_bound_exceeded')
    elif type(value) is float:
        _require(math.isfinite(value), 'json_nonfinite_rejected')
    else:
        _require(value is None or type(value) is bool, 'unsupported_json_type')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate_json_key')
        result[key] = value
    return result


def _bad_constant(value):
    raise _Invalid('json_nonfinite_rejected')


def _json(value):
    if type(value) is bytes:
        _require(0 < len(value) <= MAX_BLOB_BYTES and b'\0' not in value,
                 'json_blob_size_or_NUL_rejected')
        try:
            value = json.loads(value.decode('utf-8'), object_pairs_hook=_pairs,
                               parse_constant=_bad_constant)
        except (UnicodeError, ValueError, RecursionError) as error:
            if isinstance(error, _Invalid):
                raise
            raise _Invalid('invalid_strict_UTF8_JSON') from error
    _safe_tree(value)
    _require(type(value) is dict, 'JSON_object_required')
    return value


def _ref(ref, metadata_only=False):
    _exact(ref, ('path', 'sha256', 'bytes'), 'ref_exact_path_sha256_bytes_required')
    _require(_canonical(ref['path']), 'ref_path_not_absolute_lexically_canonical')
    _require(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
             'ref_SHA256_invalid')
    _require(_int(ref['bytes'], 1, MAX_INTEGER if metadata_only else MAX_BLOB_BYTES),
             'ref_bytes_bound_or_type_invalid')
    return ref


def _read(ref, blobs, as_json=True):
    ref = _ref(ref)
    _require(ref['path'] in blobs, 'evidence_blob_missing:' + ref['path'])
    raw = blobs[ref['path']]
    _require(type(raw) is bytes and len(raw) == ref['bytes'], 'evidence_bytes_mismatch')
    _require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'evidence_SHA256_mismatch')
    return _json(raw) if as_json else raw


def _identity(identity):
    _exact(identity, IDENTITY_KEYS, 'identity_exact_fields_required')
    for key in IDENTITY_KEYS - {'physical_gpu_index', 'old_pid', 'old_start_ticks'}:
        _require(_text(identity[key]), 'identity_text_invalid:' + key)
    _require(_int(identity['physical_gpu_index'], 0, 3), 'only_physical_GPU_0_to_3')
    _require(_int(identity['old_pid'], 1) and _int(identity['old_start_ticks'], 1),
             'exact_old_PID_start_required')
    _require(identity['predecessor_epoch'] != identity['next_epoch'], 'epoch_reuse_forbidden')
    _require(identity['old_run_id'] != identity['next_run_id'], 'run_reuse_forbidden')


def _bound(doc, schema, identity):
    _require(doc.get('schema') == schema, 'evidence_schema_mismatch:' + schema)
    _require(_equal(doc.get('identity'), identity), 'evidence_identity_mismatch:' + schema)


def _lock(value):
    _exact(value, ('path', 'dev', 'inode'), 'original_lock_identity_required')
    _require(_canonical(value['path']) and _int(value['dev']) and _int(value['inode'], 1),
             'lock_dev_inode_path_type_invalid')


def _budgets(value):
    _exact(value, ('effective_headroom_bytes', 'local_disk_free_bytes', 'accounting'),
           'independent_resource_budgets_required')
    _require(_int(value['effective_headroom_bytes'], 32 * GIB)
             and _int(value['local_disk_free_bytes'], 8 * GIB)
             and value['accounting'] == 'node_total_once', 'resource_budget_type_or_accounting_invalid')


def _key(value):
    _require(type(value) is list and len(value) == 2, 'source_key_requires_split_id_pair')
    _require(all(type(s) is str and re.fullmatch('[A-Za-z0-9_-]+', s) for s in value),
             'source_key_invalid')
    return tuple(value)


def _keys(values):
    _require(type(values) is list and len(values) <= MAX_ROWS, 'source_set_bound_or_type')
    keys = [_key(value) for value in values]
    _require(len(set(keys)) == len(keys)
             and len({tuple(s.casefold() for s in k) for k in keys}) == len(keys),
             'duplicate_casefold_source_key')
    return keys


def _rows(rows, gpu):
    _require(type(rows) is list and 0 < len(rows) <= MAX_ROWS, 'bounded_lane_rows_required')
    indices, keys = [], []
    for item in rows:
        _exact(item, ('assignment_index', 'source_row'), 'original_index_and_six_field_row_required')
        index, row = item['assignment_index'], item['source_row']
        _require(_int(index, 0, MAX_ROWS * 4 - 1) and index % 4 == gpu,
                 'original_assignment_index_wrong_lane_or_type')
        _exact(row, ('id', 'split', 'video_path', 'audio_path', 'meta_path', 'source_ordinal'),
               'original_six_field_source_row_required')
        key = _key([row['split'], row['id']])
        _require(_int(row['source_ordinal'], 1), 'original_source_ordinal_integer_required')
        for name in ('video_path', 'audio_path', 'meta_path'):
            _require(_canonical(row[name]), 'source_path_not_canonical')
        for name in ('video_path', 'audio_path'):
            stem = row[name].rsplit('/', 1)[-1].rsplit('.', 1)[0]
            _require(stem == row['id'], 'media_source_id_mismatch')
        indices.append(index)
        keys.append(key)
    _require(indices == list(range(gpu, gpu + 4 * len(rows), 4)),
             'lane_projection_hole_reordered_or_reindexed')
    _require(len(set(keys)) == len(keys), 'duplicate_original_source_key')
    ordinals = [item["source_row"]["source_ordinal"] for item in rows]
    _require(len(set(ordinals)) == len(ordinals), "duplicate_original_source_ordinal")
    media = [item['source_row'][name] for item in rows for name in ('video_path', 'audio_path')]
    _require(len(set(media)) == len(media), 'duplicate_original_media_path')
    return keys


def _candidate(c, t, d, blobs):
    refs, identity = c['refs'], c['identity']
    manifest, acceptance = d['frozen_manifest'], d['independent_acceptance']
    _require(manifest.get('schema') == 'ms_batchrpc_next_frozen_manifest_v3'
             and manifest.get('deploy_ready') is False, 'candidate_frozen_manifest_v3_required')
    _require(acceptance.get('schema') == 'ms_batchrpc_independent_acceptance_v3'
             and acceptance.get('status') == 'independent_cpu_verified_not_deployed'
             and acceptance.get('deploy_ready') is False, 'independent_v3_acceptance_required')
    # Existing independent v3 receipt has an immutable legacy two-field manifest reference.
    _require(_equal(acceptance.get('manifest_ref'), {k: refs['frozen_manifest'][k] for k in ('path', 'sha256')}),
             'independent_receipt_manifest_binding_mismatch')
    report = acceptance.get('report')
    _require(type(report) is dict and report.get('success') is True
             and report.get('independent_v3') is True
             and report.get("code_and_frozen_proofs_unchanged") is True
             and type(report.get("forbidden_callbacks")) is dict
             and all(type(report["forbidden_callbacks"].get(k)) is int
                     and report["forbidden_callbacks"][k] == 0
                     for k in ("gpu", "model", "claim", "copy", "extract"))
             and _int(report.get('tests_run'), 1)
             and all(type(report.get(k)) is int and report[k] == 0 for k in ('failures', 'errors', 'skipped')),
             'independent_CPU_suite_not_successful')
    codes = c['code_refs']
    _require(type(codes) is list and 0 < len(codes) <= 16 and _equal(codes, t['code_refs']),
             'independent_exact_code_scope_mismatch')
    basenames = []
    for ref in codes:
        _read(ref, blobs, as_json=False)
        basename = ref['path'].rsplit('/', 1)[-1]
        basenames.append(basename)
        entry = manifest.get('files', {}).get(basename, {})
        _require(entry.get('sha256') == ref['sha256'] and type(entry.get('bytes')) is int
                 and entry['bytes'] == ref['bytes'], 'frozen_candidate_code_binding_mismatch')
        _require(report.get('actual_code_hashes', {}).get(basename) == ref['sha256'],
                 'independent_receipt_code_binding_mismatch')
    _require(len(set(basenames)) == len(basenames), 'duplicate_candidate_code_basename')
    _require(set(basenames) == REQUIRED_CODE_NAMES, "complete_five_core_and_four_wrapper_scope_required")
    ticket, queue = d['next_ticket'], d['next_queue']
    _bound(ticket, 'ms_epoch_candidate_ticket_preflight_v1', identity)
    _bound(queue, 'ms_epoch_candidate_queue_preflight_v1', identity)
    _require(_equal(ticket.get('source_release_ref'), refs['source_release'])
             and _equal(ticket.get('original_assignment_ref'), t['original_assignment_ref'])
             and _equal(ticket.get('code_refs'), codes)
             and ticket.get('source_claim_namespace') == t['source_claim_namespace'],
             'next_ticket_source_code_namespace_binding_mismatch')
    _require(type(ticket.get('source_count')) is int and ticket['source_count'] == t['exact_source_count']
             and ticket.get('automatic_retry') is False
             and _equal(ticket.get('allowed_gpu_indices'), [identity['physical_gpu_index']]),
             'next_ticket_exact_budget_no_retry_or_GPU_scope_mismatch')
    _require(_equal(queue.get('tickets'), [refs['next_ticket']]) and queue.get('finite') is True
             and queue.get('append_existing_queue') is False,
             'new_queue_exact_ticket_binding_no_append_required')
    _require(refs['next_queue']['path'] != refs['original_queue']['path']
             and refs['next_ticket']['path'] != refs['old_ticket']['path'],
             'old_queue_or_ticket_path_reuse_forbidden')


def _predecessor(c, t, d, blobs):
    refs, identity = c['refs'], c['identity']
    ticket, lane, progress, lease, queue = [d[k] for k in
        ('old_ticket', 'old_assignment', 'old_progress', 'old_lease', 'original_queue')]
    for doc, schema in (
        (ticket, 'ms_epoch_original_ticket_binding_v1'),
        (lane, 'ms_epoch_original_lane_projection_v1'),
        (progress, 'ms_epoch_original_progress_binding_v1'),
        (lease, 'ms_epoch_original_lease_binding_v1'),
        (queue, 'ms_epoch_original_queue_binding_v1'),
    ):
        _bound(doc, schema, identity)
    _ref(t['original_assignment_ref'], metadata_only=True)
    _require(_equal(ticket.get('original_assignment_ref'), t['original_assignment_ref'])
             and _equal(lane.get('original_assignment_ref'), t['original_assignment_ref'])
             and _equal(ticket.get('lane_projection_ref'), refs['old_assignment'])
             and ticket.get('source_claim_namespace') == t['source_claim_namespace'],
             'old_ticket_original_manifest_lane_namespace_binding_mismatch')
    _require(_equal(progress.get('old_ticket_ref'), refs['old_ticket'])
             and _equal(progress.get('lane_projection_ref'), refs['old_assignment'])
             and _equal(progress.get('old_lease_ref'), refs['old_lease'])
             and _equal(lease.get('old_ticket_ref'), refs['old_ticket'])
             and _equal(lease.get('lane_projection_ref'), refs['old_assignment']),
             'old_progress_lease_ticket_assignment_binding_mismatch')
    _require(_equal(lease.get('lock_identity'), t['lock_identity']), 'old_lock_inode_or_dev_changed')
    _require(lease.get('status') == 'released_sealed_no_retry', 'old_lease_not_released_no_retry')
    _require(queue.get('finite') is True and queue.get('append_allowed') is False
             and queue.get('newepoch_lease_present') is False
             and queue.get('rollover_progress_present') is False,
             'old_queue_not_finite_or_prior_rollover_exists')
    entries = queue.get('tickets')
    _require(type(entries) is list and len(entries) == 2 and _equal(entries[-1], refs['old_ticket']),
             'original_exact_two_ticket_queue_required')
    for ref in entries:
        _read(ref, blobs)
    _require(entries[0]['path'] != entries[1]['path'], 'duplicate_original_queue_ticket')
    keys = _rows(lane.get('rows'), identity['physical_gpu_index'])
    _require(type(ticket.get('original_source_count')) is int
             and ticket['original_source_count'] == len(keys) * 4,
             'complete_original_lane_projection_count_mismatch')
    _require(_int(progress.get('processed'), 0, len(keys))
             and _int(progress.get('final_boundary_unix'), 1), 'old_processed_or_boundary_invalid')


def _retirement(c, t, d):
    proof, progress, refs = d['retirement'], d['old_progress'], c['refs']
    _bound(proof, 'ms_epoch_retirement_proof_v1', c['identity'])
    _require(_equal(proof.get('old_progress_ref'), refs['old_progress'])
             and _equal(proof.get('old_lease_ref'), refs['old_lease'])
             and _equal(proof.get('original_queue_ref'), refs['original_queue']),
             'retirement_predecessor_ref_binding_mismatch')
    _require(proof.get('alive') is False and type(proof.get('uncommitted_prefetch')) is int
             and proof['uncommitted_prefetch'] == 0 and proof.get('no_resume') is True,
             'exact_old_process_exit_and_prefetch_drain_required')
    names = ('final_boundary_unix', 'prefetch_drained_unix', 'retired_unix',
             'exit_observed_unix', 'pid_sample_unix', 'claims_closed_unix')
    times = [proof.get(k) for k in names]
    _require(all(_int(value, 1) for value in times) and times == sorted(times)
             and times[0] == progress['final_boundary_unix'],
             'retirement_sample_order_after_final_boundary_required')
    mode = proof.get('mode')
    _require(mode in ('finite_queue_naturally_exhausted', 'operator_confirmed_retired_no_resume'),
             'explicit_retirement_or_natural_exhaustion_required_not_status_failed')
    if mode == 'finite_queue_naturally_exhausted':
        _require(type(proof.get('consumed_ticket_count')) is int
                 and proof['consumed_ticket_count'] == 2
                 and progress['processed'] == len(d['old_assignment']['rows']),
                 'natural_exhaustion_not_complete')
    else:
        _require(proof.get('operator_confirmed_retired_no_resume') is True,
                 'operator_retirement_no_resume_not_confirmed')


def _release(c, t, d):
    release, refs = d['source_release'], c['refs']
    _bound(release, 'ms_epoch_source_release_proof_v1', c['identity'])
    _require(_equal(release.get('old_ticket_ref'), refs['old_ticket'])
             and _equal(release.get('lane_projection_ref'), refs['old_assignment'])
             and _equal(release.get('old_progress_ref'), refs['old_progress'])
             and _equal(release.get('retirement_ref'), refs['retirement']), 'release_predecessor_refs_mismatch')
    _require(release.get('source_claim_namespace') == t['source_claim_namespace']
             and d['next_ticket'].get('source_claim_namespace') == t['source_claim_namespace'],
             'original_source_claim_namespace_reset_forbidden')
    _require(release.get('no_claim_reset') is True and release.get('no_auto_retry') is True
             and release.get('claim_listing_complete') is True
             and release.get('publication_listing_complete') is True
             and type(release.get('uncommitted_prefetch')) is int
             and release['uncommitted_prefetch'] == 0, 'release_quiet_complete_no_reset_no_retry_required')
    _require(type(release.get('claims_closed_unix')) is int
             and release['claims_closed_unix'] == d['retirement']['claims_closed_unix'],
             'release_claim_close_boundary_mismatch')
    rows = d['old_assignment']['rows']
    keys = _rows(rows, c['identity']['physical_gpu_index'])
    processed = d['old_progress']['processed']
    outcomes = release.get('processed_prefix_outcomes')
    _require(type(outcomes) is list and len(outcomes) == processed, 'processed_outcomes_not_exact_prefix')
    categories = {k: [] for k in ('published', 'sealed', 'unknown', 'existing_claim')}
    for pos, outcome in enumerate(outcomes):
        _exact(outcome, ('key', 'assignment_index', 'status'), 'prefix_outcome_fields_invalid')
        _require(_key(outcome['key']) == keys[pos]
                 and type(outcome['assignment_index']) is int
                 and outcome['assignment_index'] == rows[pos]['assignment_index']
                 and outcome['status'] in categories, 'processed_outcomes_not_exact_prefix')
        categories[outcome['status']].append(keys[pos])
    claims = set(_keys(release.get('claims')))
    _require(claims == set(keys[:processed]), 'missing_claim_or_orphan_claim')
    for name, expected in categories.items():
        _require(set(_keys(release.get(name))) == set(expected), 'prefix_outcome_set_mismatch:' + name)
    suffix = release.get('released_rows')
    _require(type(suffix) is list and _equal(suffix, rows[processed:]),
             'release_not_exact_original_ordered_lane_suffix_hole_reindexed_or_reordered')
    _require(len(suffix) == t['exact_source_count'], 'explicit_exact_source_budget_mismatch')
    suffix_keys = set(keys[processed:])
    all_existing = set(_keys(release.get('all_existing_claims')))
    _require(not suffix_keys & all_existing, 'released_suffix_already_claimed')
    _require(all_existing == claims, 'missing_or_orphan_original_lane_existing_claim')
    _require(type(release.get('other_allocation_intersection')) is int
             and release['other_allocation_intersection'] == 0
             and not suffix_keys & set(_keys(release.get('other_allocation_keys'))),
             'released_suffix_intersects_other_allocation')


def _resources(c, t, d, now):
    snapshot, retired = d['resource_snapshot'], d['retirement']
    _bound(snapshot, 'ms_epoch_resource_snapshot_v1', c['identity'])
    captured = snapshot.get('captured_unix')
    _require(all(_int(retired.get(k), 1) and 0 <= now - retired[k] <= 120
                 for k in ("pid_sample_unix", "claims_closed_unix")),
             "retirement_PID_or_claim_snapshot_stale_or_future")
    _require(_int(captured, 1) and 0 <= now - captured <= 120
             and captured >= retired['claims_closed_unix'],
             'resource_snapshot_stale_future_or_before_retirement_exit_claimclose')
    _require(_equal(snapshot.get('lock_identity'), t['lock_identity']), 'original_lock_inode_or_dev_changed')
    _require(snapshot.get('preserved_old_lease_sha256') == c['refs']['old_lease']['sha256'],
             'preserved_old_lease_SHA_changed')
    _require(snapshot.get('newepoch_lease_present') is False
             and snapshot.get('rollover_progress_present') is False,
             'existing_newepoch_lease_or_rollover_progress_forbidden')
    _require(snapshot.get('compute_pids') == [] and snapshot.get('graphics_pids') == []
             and type(snapshot.get('memory_used_mib')) is int and snapshot['memory_used_mib'] == 0,
             'foreign_compute_graphics_PID_or_nonzero_GPU_memory')
    _require(snapshot.get('accounting') == 'node_total_once', 'node_headroom_must_not_be_counted_per_GPU')
    _require(snapshot.get("local_filesystem") in ("ext4", "xfs", "tmpfs", "overlay"),
             "explicit_local_filesystem_required")
    own = snapshot.get("own_cache_bytes")
    retained = snapshot.get("retained_four_lane_cache_bytes")
    _require(_int(own, 0, 8 * GIB) and _int(retained, 0, 32 * GIB),
             "local_cache_budget_exceeded_or_missing")
    disk_floor = 8 * GIB + max(0, 32 * GIB - retained)
    _require(t["budgets"]["local_disk_free_bytes"] >= disk_floor,
             "independent_disk_budget_below_original_reserve_formula")
    for key in ('effective_headroom_bytes', 'local_disk_free_bytes'):
        _require(_int(snapshot.get(key)) and snapshot[key] >= t['budgets'][key],
                 'independent_resource_budget_not_met:' + key)


def _approvals(c, t, d, blobs, errors):
    approvals = c.get('approval_refs')
    expected = t.get('approval_refs')
    if type(expected) is not dict or set(expected) != set(APPROVAL_NAMES):
        return
    if type(approvals) is not dict:
        errors.append('pending_explicit_approval:contract_three_approval_refs_missing')
        return
    if set(approvals) != set(APPROVAL_NAMES):
        errors.append('pending_explicit_approval:contract_missing='
                      + ','.join(sorted(set(APPROVAL_NAMES) - set(approvals)))
                      + ';unexpected=' + ','.join(sorted(set(approvals) - set(APPROVAL_NAMES))))
        return
    for purpose in APPROVAL_NAMES:
        try:
            _ref(expected[purpose])
            _require(_equal(approvals[purpose], expected[purpose]),
                     'pending_explicit_approval:independent_approval_anchor_mismatch:' + purpose)
            approval = _read(approvals[purpose], blobs)
            _bound(approval, 'ms_epoch_transition_explicit_approval_v1', c['identity'])
            _require(approval.get('purpose') == purpose and approval.get('explicit_approval') is True,
                     'pending_explicit_approval:explicit_purpose_missing:' + purpose)
            if purpose == 'retirement_no_resume':
                _require(_equal(approval.get('retirement_ref'), c['refs']['retirement'])
                         and approval.get('no_resume') is True, 'approval_retirement_scope_mismatch')
            elif purpose == 'exact_source_budget':
                _require(_equal(approval.get('source_release_ref'), c['refs']['source_release'])
                         and type(approval.get('exact_source_count')) is int
                         and approval['exact_source_count'] == t['exact_source_count']
                         and approval.get('source_claim_namespace') == t['source_claim_namespace'],
                         'approval_exact_source_scope_mismatch')
            else:
                _require(_equal(approval.get('resource_snapshot_ref'), c['refs']['resource_snapshot'])
                         and _equal(approval.get('next_ticket_ref'), c['refs']['next_ticket'])
                         and _equal(approval.get('next_queue_ref'), c['refs']['next_queue'])
                         and _equal(approval.get('budgets'), t['budgets'])
                         and _equal(approval.get('code_refs'), c['code_refs']), 'approval_resource_code_scope_mismatch')
        except (_Invalid, KeyError, TypeError, AttributeError, RecursionError) as error:
            errors.append('pending_explicit_approval:' + purpose + ':' + str(error))


def validate_transition_contract(contract, evidence_blobs, trusted_expected, now_unix):
    """Return structural diagnostics only; no I/O, process, GPU, clock or state access."""
    result = {'structural_valid': False, 'blocked_reasons': [],
              'local_fresh_checks_required': list(FRESH_CHECKS), 'deploy_ready': False,
              'may_create_claims': False, 'full_original_assignment_verified': False,
              'actual_physical_lease_verified': False, 'actual_process_exit_verified': False,
              'full_commit_claim_artifacts_verified': False}
    errors = result['blocked_reasons']
    try:
        c = _json(contract)
        t = _json(trusted_expected)
        expected_approvals = t.get('approval_refs')
        if type(expected_approvals) is not dict:
            errors.append('pending_explicit_approval:independent_three_approval_refs_missing')
        elif set(expected_approvals) != set(APPROVAL_NAMES):
            missing = sorted(set(APPROVAL_NAMES) - set(expected_approvals))
            unexpected = sorted(set(expected_approvals) - set(APPROVAL_NAMES))
            errors.append('pending_explicit_approval:missing=' + ','.join(missing)
                          + ';unexpected=' + ','.join(unexpected))
        _require(_int(now_unix, 1), 'explicit_now_unix_integer_required')
        _require(type(evidence_blobs) is dict and len(evidence_blobs) <= MAX_REFS,
                 'bounded_evidence_blob_mapping_required')
        _require(all(_canonical(path) and type(raw) is bytes and len(raw) <= MAX_BLOB_BYTES
                     for path, raw in evidence_blobs.items()), 'evidence_mapping_key_or_bytes_invalid')
        _require(sum(len(raw) for raw in evidence_blobs.values()) <= MAX_TOTAL_BYTES,
                 'total_evidence_bytes_bound_exceeded')
        required = {'schema', 'identity', 'refs', 'code_refs', 'approval_refs'}
        # Unknown top-level keys (including authorized/signature flags) cannot add authority.
        _require(set(c) <= required, 'unsupported_contract_fields_no_self_authorization')
        _require(c.get('schema') == SCHEMA, 'transition_preflight_schema_required')
        _require(type(t.get('identity')) is dict, 'independent_identity_anchor_required')
        _identity(t['identity'])
        _require(_equal(c.get('identity'), t['identity']), 'independent_identity_binding_mismatch')
        _identity(c['identity'])
        _require(type(t.get('refs')) is dict and set(t['refs']) == set(REF_NAMES),
                 'independent_evidence_refs_required')
        _require(type(c.get('refs')) is dict and set(c['refs']) == set(REF_NAMES),
                 'contract_evidence_refs_required')
        _require(_canonical(t.get('source_claim_namespace')), 'independent_original_claim_namespace_required')
        _lock(t.get('lock_identity'))
        _budgets(t.get('budgets'))
        _require(_int(t.get('exact_source_count'), 0, MAX_ROWS), 'independent_exact_source_budget_required')
        _require(type(t.get('code_refs')) is list, 'independent_code_refs_required')
        d = {}
        for name in REF_NAMES:
            try:
                _ref(t['refs'][name])
                _require(_equal(c['refs'][name], t['refs'][name]), 'independent_ref_binding_mismatch')
                d[name] = _read(c['refs'][name], evidence_blobs)
            except (_Invalid, KeyError, TypeError, AttributeError, RecursionError) as error:
                errors.append(name + ':' + str(error))
        _approvals(c, t, d, evidence_blobs, errors)
        stages = (
            ('candidate', ('frozen_manifest', 'independent_acceptance', 'next_ticket', 'next_queue'),
             lambda: _candidate(c, t, d, evidence_blobs)),
            ('predecessor', ('old_ticket', 'old_assignment', 'old_progress', 'old_lease', 'original_queue'),
             lambda: _predecessor(c, t, d, evidence_blobs)),
            ('retirement', ('retirement', 'old_progress', 'old_assignment'), lambda: _retirement(c, t, d)),
            ('source_release', ('source_release', 'old_assignment', 'old_progress', 'retirement', 'next_ticket'),
             lambda: _release(c, t, d)),
            ('resources', ('resource_snapshot', 'retirement'), lambda: _resources(c, t, d, now_unix)),
        )
        for stage, needed, check in stages:
            if not all(name in d for name in needed):
                errors.append(stage + ':missing_validated_dependency')
                continue
            try:
                check()
            except (_Invalid, KeyError, TypeError, AttributeError, IndexError, RecursionError) as error:
                errors.append(stage + ':' + str(error))
    except (_Invalid, KeyError, TypeError, AttributeError, IndexError, RecursionError) as error:
        errors.append('input:' + str(error))
    result['structural_valid'] = not errors
    result["status"] = "blocked" if errors else "structurally_valid_not_admitted"
    return result
