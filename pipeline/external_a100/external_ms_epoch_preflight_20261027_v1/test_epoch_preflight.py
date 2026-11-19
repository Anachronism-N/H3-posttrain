"""Synthetic CPU-only tests. No production contract, media or state is loaded."""

import copy
import hashlib
import json
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import epoch_preflight as preflight

NOW = 1800000000
GIB = 2**30
BASE = '/synthetic_only_not_production'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fixture(mutate=None, gpu=0, processed=2, mode='operator_confirmed_retired_no_resume',
            code_names=None, original_manifest_bytes=11189750, memory_budget=32 * GIB,
            disk_budget=40 * GIB):
    """Construct independent synthetic anchors, never take expectations from a contract."""
    blobs, refs = {}, {}
    identity = {
        'transition_id': 'synthetic-transition-1', 'predecessor_epoch': 'synthetic-old',
        'next_epoch': 'synthetic-next', 'old_run_id': 'synthetic-old-run',
        'next_run_id': 'synthetic-next-run', 'worker': 'synthetic-worker',
        'physical_gpu_index': gpu, 'gpu_uuid': 'GPU-synthetic-0', 'host': 'synthetic-host',
        'boot_id': 'synthetic-boot', 'old_pid': 1001, 'old_start_ticks': 8888,
    }
    namespace = BASE + '/original_shared_claims'
    lock = {'path': BASE + '/physical.lock', 'dev': 9, 'inode': 17}
    budgets = {'effective_headroom_bytes': memory_budget, 'local_disk_free_bytes': disk_budget,
               'accounting': 'node_total_once'}
    origin = {'path': BASE + '/not_read_original_manifest.jsonl',
              'sha256': '7' * 64, 'bytes': original_manifest_bytes}
    rows = [{'assignment_index': gpu + 4 * i, 'source_row': {
        'id': 'source_' + str(i), 'split': 'train', 'video_path': BASE + '/source_' + str(i) + '.mp4',
        'audio_path': BASE + '/source_' + str(i) + '.wav', 'meta_path': BASE + '/source_' + str(i) + '.json',
        'source_ordinal': 10000 + i,
    }} for i in range(4)]
    keys = [['train', 'source_' + str(i)] for i in range(4)]

    def put(name, doc, schema=None):
        doc = copy.deepcopy(doc)
        if schema:
            doc = {'schema': schema, 'identity': copy.deepcopy(identity), **doc}
        if mutate:
            mutate(name, doc)
        raw = encoded(doc)
        path = BASE + '/' + name + '.json'
        ref = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        blobs[path] = raw
        refs[name] = ref
        return ref

    codes = []
    code_files = {}
    for name in sorted(preflight.REQUIRED_CODE_NAMES if code_names is None else code_names):
        code_raw = ('# synthetic inert fixture: ' + name).encode()
        code = {'path': BASE + '/' + name, 'sha256': hashlib.sha256(code_raw).hexdigest(),
                'bytes': len(code_raw)}
        blobs[code['path']] = code_raw
        codes.append(code)
        code_files[name] = {'sha256': code['sha256'], 'bytes': code['bytes']}
    put('frozen_manifest', {'schema': 'ms_batchrpc_next_frozen_manifest_v3', 'deploy_ready': False,
        'files': code_files})
    put('independent_acceptance', {'schema': 'ms_batchrpc_independent_acceptance_v3',
        'status': 'independent_cpu_verified_not_deployed', 'deploy_ready': False,
        'manifest_ref': {k: refs['frozen_manifest'][k] for k in ('path', 'sha256')},
        'report': {'independent_v3': True, 'success': True, 'tests_run': 1, 'failures': 0,
                   'errors': 0, 'skipped': 0, 'code_and_frozen_proofs_unchanged': True,
                   'forbidden_callbacks': {k: 0 for k in ('gpu', 'model', 'claim', 'copy', 'extract')},
                   'actual_code_hashes': {n: v['sha256'] for n, v in code_files.items()}}})
    put('old_assignment', {'original_assignment_ref': origin, 'rows': rows},
        'ms_epoch_original_lane_projection_v1')
    put('old_ticket', {'original_assignment_ref': origin, 'lane_projection_ref': refs['old_assignment'],
        'source_claim_namespace': namespace, 'original_source_count': 16}, 'ms_epoch_original_ticket_binding_v1')
    put('old_lease', {'old_ticket_ref': refs['old_ticket'], 'lane_projection_ref': refs['old_assignment'],
        'lock_identity': lock, 'status': 'released_sealed_no_retry'}, 'ms_epoch_original_lease_binding_v1')
    put('old_progress', {'old_ticket_ref': refs['old_ticket'], 'lane_projection_ref': refs['old_assignment'],
        'old_lease_ref': refs['old_lease'], 'processed': processed, 'final_boundary_unix': NOW - 60,
        'status': 'failed_sealed_no_retry'}, 'ms_epoch_original_progress_binding_v1')
    earlier = put('prior_consumed_ticket', {'synthetic_prior_consumed': True})
    put('original_queue', {'tickets': [earlier, refs['old_ticket']], 'finite': True, 'append_allowed': False,
        'newepoch_lease_present': False, 'rollover_progress_present': False}, 'ms_epoch_original_queue_binding_v1')
    put('retirement', {'old_progress_ref': refs['old_progress'], 'old_lease_ref': refs['old_lease'],
        'original_queue_ref': refs['original_queue'], 'mode': mode, 'alive': False, 'no_resume': True,
        'uncommitted_prefetch': 0, 'final_boundary_unix': NOW - 60, 'prefetch_drained_unix': NOW - 55,
        'retired_unix': NOW - 50, 'exit_observed_unix': NOW - 45, 'pid_sample_unix': NOW - 40,
        'claims_closed_unix': NOW - 35, 'operator_confirmed_retired_no_resume': True,
        'consumed_ticket_count': 2}, 'ms_epoch_retirement_proof_v1')
    statuses = ['published', 'sealed', 'unknown', 'existing_claim']
    outcomes = [{'key': keys[i], 'assignment_index': rows[i]['assignment_index'],
                 'status': statuses[i]} for i in range(processed)]
    sets = {status: [keys[i] for i in range(processed) if statuses[i] == status] for status in statuses}
    put('source_release', {'old_ticket_ref': refs['old_ticket'], 'lane_projection_ref': refs['old_assignment'],
        'old_progress_ref': refs['old_progress'], 'retirement_ref': refs['retirement'],
        'source_claim_namespace': namespace, 'no_claim_reset': True, 'no_auto_retry': True,
        'claim_listing_complete': True, 'publication_listing_complete': True, 'uncommitted_prefetch': 0,
        'claims_closed_unix': NOW - 35, 'processed_prefix_outcomes': outcomes,
        'claims': keys[:processed], **sets, 'released_rows': rows[processed:],
        'all_existing_claims': keys[:processed], 'other_allocation_intersection': 0,
        'other_allocation_keys': []}, 'ms_epoch_source_release_proof_v1')
    put('resource_snapshot', {'captured_unix': NOW - 10, 'lock_identity': lock,
        'preserved_old_lease_sha256': refs['old_lease']['sha256'], 'newepoch_lease_present': False,
        'rollover_progress_present': False, 'compute_pids': [], 'graphics_pids': [], 'memory_used_mib': 0,
        'effective_headroom_bytes': 32 * GIB, 'local_disk_free_bytes': 40 * GIB,
        'accounting': 'node_total_once', 'local_filesystem': 'ext4',
        'own_cache_bytes': 0, 'retained_four_lane_cache_bytes': 0}, 'ms_epoch_resource_snapshot_v1')
    put('next_ticket', {'source_release_ref': refs['source_release'], 'original_assignment_ref': origin,
        'source_claim_namespace': namespace, 'code_refs': codes, 'source_count': 4 - processed,
        'automatic_retry': False, 'allowed_gpu_indices': [gpu]}, 'ms_epoch_candidate_ticket_preflight_v1')
    put('next_queue', {'tickets': [refs['next_ticket']], 'finite': True, 'append_existing_queue': False},
        'ms_epoch_candidate_queue_preflight_v1')
    approvals = {}
    for purpose in preflight.APPROVAL_NAMES:
        scope = {'purpose': purpose, 'explicit_approval': True}
        if purpose == 'retirement_no_resume':
            scope.update(retirement_ref=refs['retirement'], no_resume=True)
        elif purpose == 'exact_source_budget':
            scope.update(source_release_ref=refs['source_release'], exact_source_count=4 - processed,
                         source_claim_namespace=namespace)
        else:
            scope.update(resource_snapshot_ref=refs['resource_snapshot'], next_ticket_ref=refs['next_ticket'],
                         next_queue_ref=refs['next_queue'], budgets=budgets, code_refs=codes)
        approvals[purpose] = put('approval_' + purpose, scope, 'ms_epoch_transition_explicit_approval_v1')
    anchors = {name: copy.deepcopy(refs[name]) for name in preflight.REF_NAMES}
    trusted = {'identity': copy.deepcopy(identity), 'refs': anchors, 'code_refs': copy.deepcopy(codes),
        'approval_refs': copy.deepcopy(approvals), 'original_assignment_ref': copy.deepcopy(origin),
        'source_claim_namespace': namespace, 'lock_identity': copy.deepcopy(lock),
        'budgets': copy.deepcopy(budgets), 'exact_source_count': 4 - processed}
    contract = {'schema': preflight.SCHEMA, 'identity': copy.deepcopy(identity),
        'refs': {name: copy.deepcopy(refs[name]) for name in preflight.REF_NAMES},
        'code_refs': copy.deepcopy(codes), 'approval_refs': copy.deepcopy(approvals)}
    return contract, blobs, trusted


class PreflightTests(unittest.TestCase):
    def result(self, triple=None, now=NOW):
        return preflight.validate_transition_contract(*(triple or fixture()), now)

    def blocked(self, triple, reason=None):
        result = self.result(triple)
        self.assertFalse(result['structural_valid'], result)
        self.assertIs(result['deploy_ready'], False)
        self.assertEqual(result['local_fresh_checks_required'], list(preflight.FRESH_CHECKS))
        self.assertTrue(result['blocked_reasons'])
        if reason:
            self.assertIn(reason, '\n'.join(result['blocked_reasons']))
        return result

    def changed(self, name, action, reason=None, **kwargs):
        def mutate(doc_name, doc):
            if doc_name == name:
                action(doc)
        return self.blocked(fixture(mutate, **kwargs), reason)

    def test_synthetic_operator_positive_structure_only(self):
        result = self.result()
        self.assertTrue(result['structural_valid'], result)
        self.assertEqual(result['blocked_reasons'], [])
        self.assertIs(result['deploy_ready'], False)
        self.assertEqual(len(result['local_fresh_checks_required']), 6)

    def test_each_GPU_0_3_independently_bound(self):
        for gpu in range(4):
            with self.subTest(gpu=gpu):
                result = self.result(fixture(gpu=gpu))
                self.assertTrue(result['structural_valid'], result)
                self.assertFalse(result['deploy_ready'])

    def test_natural_exhaustion_positive_zero_release_only(self):
        result = self.result(fixture(processed=4, mode='finite_queue_naturally_exhausted'))
        self.assertTrue(result['structural_valid'], result)
        self.assertFalse(result['deploy_ready'])

    def test_strict_bytes_input_positive(self):
        c, b, t = fixture()
        result = self.result((encoded(c), b, encoded(t)))
        self.assertTrue(result['structural_valid'], result)

    def test_no_independent_authorization(self):
        c, b, t = fixture()
        del t['approval_refs']
        self.blocked((c, b, t), 'pending_explicit_approval')

    def test_each_approval_required(self):
        for purpose in preflight.APPROVAL_NAMES:
            with self.subTest(purpose=purpose):
                c, b, t = fixture()
                del t['approval_refs'][purpose]
                self.blocked((c, b, t), 'pending_explicit_approval')

    def test_contract_authorized_bool_never_authorizes(self):
        c, b, t = fixture()
        c['authorized'] = True
        t.pop('approval_refs')
        result = self.blocked((c, b, t), 'no_self_authorization')
        self.assertIn('pending_explicit_approval', '\n'.join(result['blocked_reasons']))

    def test_only_authorized_boolean(self):
        self.blocked(({'schema': preflight.SCHEMA, 'authorized': True}, {}, {}))

    def test_self_signed_without_anchor(self):
        c, b, t = fixture()
        c['signature'] = 'self-issued-synthetic'
        t['approval_refs'] = {}
        self.blocked((c, b, t), 'no_self_authorization')

    def test_wrong_approval_anchor(self):
        c, b, t = fixture()
        c['approval_refs']['exact_source_budget']['sha256'] = '0' * 64
        self.blocked((c, b, t), 'pending_explicit_approval')

    def test_approval_scope_semantics(self):
        self.changed('approval_exact_source_budget', lambda d: d.update(exact_source_count=3),
                     'approval_exact_source_scope')

    def test_approval_resource_budget_semantics(self):
        self.changed('approval_resource_code_scope',
                     lambda d: d['budgets'].update(effective_headroom_bytes=1),
                     'approval_resource_code_scope')

    def test_approval_retirement_not_boolean(self):
        self.changed('approval_retirement_no_resume', lambda d: d.update(no_resume=1),
                     'approval_retirement_scope')

    def test_hash_wrong(self):
        c, b, t = fixture()
        b[c['refs']['old_progress']['path']] += b' '
        self.blocked((c, b, t), 'bytes_mismatch')

    def test_hash_wrong_same_length(self):
        c, b, t = fixture()
        path = c['refs']['old_lease']['path']
        b[path] = b'X' + b[path][1:]
        self.blocked((c, b, t), 'SHA256_mismatch')

    def test_independent_ref_anchor_missing(self):
        c, b, t = fixture()
        t['refs'].pop('old_lease')
        self.blocked((c, b, t), 'independent_evidence_refs')

    def test_missing_blob(self):
        c, b, t = fixture()
        del b[c['refs']['retirement']['path']]
        self.blocked((c, b, t), 'evidence_blob_missing')

    def test_caller_replaced_evidence_without_anchor(self):
        c, b, t = fixture()
        c['refs']['old_progress'] = copy.deepcopy(c['refs']['old_lease'])
        self.blocked((c, b, t), 'independent_ref_binding_mismatch')

    def test_ref_extra_fields_disallowed(self):
        c, b, t = fixture()
        t['refs']['old_lease']['authorized'] = True
        c['refs']['old_lease']['authorized'] = True
        self.blocked((c, b, t), 'ref_exact')

    def test_ref_bytes_bool_not_int(self):
        c, b, t = fixture()
        c['refs']['old_lease']['bytes'] = True
        t['refs']['old_lease']['bytes'] = True
        self.blocked((c, b, t), 'ref_bytes')

    def test_ref_path_canonical(self):
        for path in ('relative.json', '/a/../b', '/a/./b', '/a//b', '//a/b', '/a/b/', '/a/\0b'):
            with self.subTest(path=path):
                c, b, t = fixture()
                c['refs']['old_lease']['path'] = path
                t['refs']['old_lease']['path'] = path
                self.blocked((c, b, t))

    def test_duplicate_contract_JSON_keys(self):
        c, b, t = fixture()
        raw = encoded(c)
        raw = b'{"schema":"ignored",' + raw[1:]
        self.blocked((raw, b, t), 'duplicate_json_key')

    def test_duplicate_blob_JSON_keys(self):
        c, b, t = fixture()
        name = 'old_lease'
        path = c['refs'][name]['path']
        raw = b'{"schema":"ignored",' + b[path][1:]
        b[path] = raw
        ref = {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        c['refs'][name] = copy.deepcopy(ref)
        t['refs'][name] = copy.deepcopy(ref)
        self.blocked((c, b, t), 'duplicate_json_key')

    def test_JSON_NUL_NaN_Infinity_invalid_UTF8(self):
        c, b, t = fixture()
        for raw in (b'{"schema":"a\x00b"}', b'{"x":NaN}', b'{"x":Infinity}',
                    b'{"x":-Infinity}', b'{"x":1e999}', b'{"x":"\xff"}'):
            with self.subTest(raw=raw):
                self.blocked((raw, b, t))

    def test_object_nonfinite_and_NUL(self):
        for value in (float('nan'), float('inf'), 'abc\0def'):
            c, b, t = fixture()
            c['illegal'] = value
            self.blocked((c, b, t))

    def test_bool_as_now_or_budget(self):
        c, b, t = fixture()
        result = preflight.validate_transition_contract(c, b, t, True)
        self.assertFalse(result['structural_valid'])
        t['budgets']['effective_headroom_bytes'] = True
        self.blocked((c, b, t), 'resource_budget_type')

    def test_nested_GPU_bool_not_int(self):
        self.changed('next_ticket', lambda d: d.update(allowed_gpu_indices=[False]), 'GPU_scope')

    def test_nested_identity_bool_not_int(self):
        self.changed('retirement', lambda d: d['identity'].update(physical_gpu_index=False),
                     'evidence_identity')

    def test_nested_suffix_bool_index_not_int(self):
        self.changed('source_release', lambda d: d['released_rows'][0].update(assignment_index=True),
                     'exact_original_ordered')

    def test_deep_structure_bounded(self):
        c, b, t = fixture()
        nested = []
        for _ in range(30):
            nested = [nested]
        c['extra'] = nested
        self.blocked((c, b, t), 'structure_bound')

    def test_old_1719_snapshot_stale(self):
        self.changed('resource_snapshot', lambda d: d.update(captured_unix=NOW - 3600), 'snapshot_stale')

    def test_freshness_exact_120_seconds(self):
        def mutate(name, doc):
            if name == 'retirement':
                doc['pid_sample_unix'] = NOW - 10
                doc['claims_closed_unix'] = NOW - 10
            elif name == 'source_release':
                doc['claims_closed_unix'] = NOW - 10
        c, b, t = fixture(mutate)
        result = preflight.validate_transition_contract(c, b, t, NOW + 110)
        self.assertTrue(result['structural_valid'], result)
        result = preflight.validate_transition_contract(c, b, t, NOW + 111)
        self.assertFalse(result['structural_valid'])

    def test_future_snapshot(self):
        self.changed('resource_snapshot', lambda d: d.update(captured_unix=NOW + 1), 'snapshot_stale')

    def test_resource_capture_before_claimclose(self):
        self.changed('resource_snapshot', lambda d: d.update(captured_unix=NOW - 36), 'before_retirement')

    def test_queue_append_rejected(self):
        self.changed('next_queue', lambda d: d.update(append_existing_queue=True), 'no_append')

    def test_queue_ticket_binding(self):
        self.changed('next_queue', lambda d: d['tickets'][0].update(sha256='0' * 64), 'ticket_binding')

    def test_original_queue_exact_two(self):
        self.changed('original_queue', lambda d: d['tickets'].append(copy.deepcopy(d['tickets'][0])),
                     'exact_two_ticket')

    def test_newepoch_or_rollover_blocks(self):
        for name in ('original_queue', 'resource_snapshot'):
            for field in ('newepoch_lease_present', 'rollover_progress_present'):
                with self.subTest(name=name, field=field):
                    self.changed(name, lambda d, field=field: d.update({field: True}))

    def test_host_boot_PID_start_mismatch(self):
        for field, value in (('host', 'other-host'), ('boot_id', 'other-boot'),
                             ('old_pid', 2002), ('old_start_ticks', 9999)):
            with self.subTest(field=field):
                self.changed('retirement', lambda d, field=field, value=value: d['identity'].update({field: value}),
                             'identity_mismatch')

    def test_old_PID_alive(self):
        self.changed('retirement', lambda d: d.update(alive=True), 'old_process_exit')

    def test_status_failed_not_retirement(self):
        self.changed('retirement', lambda d: d.update(mode='failed'), 'not_status_failed')

    def test_retirement_missing_no_resume(self):
        self.changed('retirement', lambda d: d.pop('no_resume'), 'prefetch_drain')

    def test_uncommitted_prefetch(self):
        self.changed('retirement', lambda d: d.update(uncommitted_prefetch=1), 'prefetch_drain')

    def test_retirement_sampling_order(self):
        self.changed('retirement', lambda d: d.update(pid_sample_unix=NOW - 70), 'sample_order')

    def test_natural_exhaustion_not_finished(self):
        self.blocked(fixture(mode='finite_queue_naturally_exhausted'), 'natural_exhaustion_not_complete')

    def test_old_lease_SHA_changed(self):
        self.changed('resource_snapshot', lambda d: d.update(preserved_old_lease_sha256='0' * 64),
                     'old_lease_SHA_changed')

    def test_original_lock_inode_dev_changed(self):
        for name in ('old_lease', 'resource_snapshot'):
            for field in ('dev', 'inode'):
                with self.subTest(name=name, field=field):
                    self.changed(name, lambda d, field=field: d['lock_identity'].update({field: 99}),
                                 'inode_or_dev_changed')

    def test_missing_claim_and_orphan(self):
        self.changed('source_release', lambda d: d['claims'].pop(), 'missing_claim_or_orphan')
        self.changed('source_release', lambda d: d['claims'].append(['train', 'orphan']),
                     'missing_claim_or_orphan')

    def test_orphan_existing_lane_claim(self):
        self.changed('source_release', lambda d: d['all_existing_claims'].append(['train', 'orphan']),
                     'orphan_original_lane_existing_claim')

    def test_missing_publication(self):
        self.changed('source_release', lambda d: d.update(published=[]), 'prefix_outcome_set')

    def test_incomplete_claim_or_publication_listing(self):
        for field in ('claim_listing_complete', 'publication_listing_complete'):
            with self.subTest(field=field):
                self.changed('source_release', lambda d, field=field: d.update({field: False}),
                             'quiet_complete')

    def test_suffix_claimed(self):
        self.changed('source_release', lambda d: d['all_existing_claims'].append(['train', 'source_2']),
                     'suffix_already_claimed')

    def test_suffix_unknown_or_sealed_contamination(self):
        for field in ('unknown', 'sealed', 'existing_claim'):
            with self.subTest(field=field):
                self.changed('source_release', lambda d, field=field: d[field].append(['train', 'source_2']),
                             'prefix_outcome_set')

    def test_suffix_hole_reordered_reindexed(self):
        actions = (lambda d: d['released_rows'].pop(0), lambda d: d['released_rows'].reverse(),
                   lambda d: d['released_rows'][0].update(assignment_index=0))
        for action in actions:
            with self.subTest(action=action):
                self.changed('source_release', action, 'exact_original_ordered_lane_suffix')

    def test_projection_hole_reordered_reindexed(self):
        actions = (lambda d: d['rows'].pop(0), lambda d: d['rows'].reverse(),
                   lambda d: d['rows'][1].update(assignment_index=12))
        for action in actions:
            with self.subTest(action=action):
                self.changed('old_assignment', action, 'projection_hole_reordered_or_reindexed')

    def test_six_fields_original_ordinal_preserved(self):
        self.changed('old_assignment', lambda d: d['rows'][0]['source_row'].update(source_ordinal=True),
                     'source_ordinal_integer')
        self.changed('source_release', lambda d: d['released_rows'][0]['source_row'].update(source_ordinal=0),
                     'exact_original_ordered')

    def test_processed_prefix_outcome_hole(self):
        self.changed('source_release', lambda d: d['processed_prefix_outcomes'].reverse(), 'not_exact_prefix')

    def test_claim_namespace_reset(self):
        self.changed('source_release', lambda d: d.update(source_claim_namespace=BASE + '/next_epoch/claims'),
                     'namespace_reset')
        self.changed('next_ticket', lambda d: d.update(source_claim_namespace=BASE + '/next_epoch/claims'),
                     'namespace_binding')

    def test_claim_reset_or_auto_retry(self):
        for field in ('no_claim_reset', 'no_auto_retry'):
            with self.subTest(field=field):
                self.changed('source_release', lambda d, field=field: d.update({field: False}), 'no_reset_no_retry')
        self.changed('next_ticket', lambda d: d.update(automatic_retry=True), 'no_retry')

    def test_other_allocation_intersection(self):
        self.changed('source_release', lambda d: d.update(other_allocation_intersection=1), 'other_allocation')
        self.changed('source_release', lambda d: d['other_allocation_keys'].append(['train', 'source_2']),
                     'other_allocation')

    def test_casefold_duplicate_and_alias(self):
        self.changed('source_release', lambda d: d['claims'].append(['TRAIN', 'SOURCE_0']), 'duplicate_casefold')
        self.changed('source_release', lambda d: d['all_existing_claims'].append(['TRAIN', 'SOURCE_2']))

    def test_GPU_4_to_7(self):
        for gpu in range(4, 8):
            with self.subTest(gpu=gpu):
                self.blocked(fixture(gpu=gpu), 'GPU_0_to_3')

    def test_foreign_GPU_UUID(self):
        self.changed('resource_snapshot', lambda d: d['identity'].update(gpu_uuid='GPU-foreign'),
                     'identity_mismatch')

    def test_foreign_compute_graphics_PID_and_memory(self):
        for field, value in (('compute_pids', [7]), ('graphics_pids', [8]), ('memory_used_mib', 1)):
            with self.subTest(field=field):
                self.changed('resource_snapshot', lambda d, field=field, value=value: d.update({field: value}),
                             'foreign_compute_graphics')

    def test_low_independent_budgets(self):
        for field in ('effective_headroom_bytes', 'local_disk_free_bytes'):
            with self.subTest(field=field):
                self.changed('resource_snapshot', lambda d, field=field: d.update({field: 1}),
                             'resource_budget_not_met')

    def test_external_budget_not_replaced_by_main_160GiB(self):
        result = self.result()
        self.assertTrue(result['structural_valid'], result)
        c, b, t = fixture()
        t['budgets']['effective_headroom_bytes'] = 160 * GIB
        self.blocked((c, b, t), 'resource_budget_not_met')

    def test_node_headroom_not_per_GPU(self):
        self.changed('resource_snapshot', lambda d: d.update(accounting='per_GPU'), 'not_be_counted_per_GPU')

    def test_missing_independent_resource_budget(self):
        c, b, t = fixture()
        t.pop('budgets')
        self.blocked((c, b, t), 'independent_resource_budgets')

    def test_old_new_epoch_run_not_reused(self):
        for field, other in (('next_epoch', 'predecessor_epoch'), ('next_run_id', 'old_run_id')):
            with self.subTest(field=field):
                c, b, t = fixture()
                c['identity'][field] = c['identity'][other]
                t['identity'][field] = t['identity'][other]
                self.blocked((c, b, t), 'reuse_forbidden')

    def test_wrong_independent_acceptance(self):
        self.changed('independent_acceptance', lambda d: d['report'].update(success=False), 'not_successful')
        self.changed('independent_acceptance', lambda d: d['manifest_ref'].update(sha256='0' * 64),
                     'manifest_binding')

    def test_candidate_code_bytes_bound(self):
        c, b, t = fixture()
        path = c['code_refs'][0]['path']
        b[path] = b'X' + b[path][1:]
        self.blocked((c, b, t), 'SHA256_mismatch')

    def test_no_mutation_of_inputs(self):
        triple = fixture()
        before = copy.deepcopy(triple)
        self.assertTrue(self.result(triple)['structural_valid'])
        self.assertEqual(triple, before)

    def test_function_no_IO_system_calls_and_deterministic(self):
        triple = fixture()
        with ExitStack() as stack:
            for target in ('builtins.open', 'os.open', 'os.stat', 'os.lstat', 'os.listdir', 'os.scandir',
                           'os.system', 'os.fork', 'os.execve', 'os.kill', 'subprocess.run',
                           'subprocess.Popen', 'socket.socket', 'time.time', 'time.monotonic'):
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden:' + target)))
            forbidden, active = [], [True]
            def audit(event, args):
                if active[0] and event in ('open', 'os.listdir', 'os.scandir', 'os.system', 'os.fork', 'os.exec',
                                          'subprocess.Popen', 'socket.__new__', 'socket.connect'):
                    forbidden.append(event)
                    raise AssertionError('audit_forbidden:' + event)
            sys.addaudithook(audit)
            try:
                first = self.result(triple)
                second = self.result(triple)
                self.assertTrue(first['structural_valid'], first)
                self.assertEqual(first, second)
                self.assertEqual(forbidden, [])
            finally:
                # Python audit hooks cannot be removed; deactivate only after both real calls.
                active[0] = False


    def test_prefix_identity_case_change_is_not_equal(self):
        self.changed('source_release', lambda d: d['processed_prefix_outcomes'][0]['key'].__setitem__(1, 'SOURCE_0'),
                     'processed_outcomes_not_exact_prefix')

    def test_fresh_GPU_cannot_hide_old_claim_closure(self):
        def mutate(name, doc):
            if name == 'old_progress':
                doc['final_boundary_unix'] -= 10000
            if name == 'retirement':
                for k in ('final_boundary_unix', 'prefetch_drained_unix', 'retired_unix',
                          'exit_observed_unix', 'pid_sample_unix', 'claims_closed_unix'):
                    doc[k] -= 10000
            if name == 'source_release':
                doc['claims_closed_unix'] -= 10000
        self.blocked(fixture(mutate), 'claim_snapshot_stale')

    def test_CPU_proof_must_include_frozen_source_invariance(self):
        self.changed('independent_acceptance', lambda d: d['report'].update(code_and_frozen_proofs_unchanged=False),
                     'not_successful')

    def test_CPU_proof_cannot_have_model_or_claim_calls(self):
        for k in ('gpu', 'model', 'claim', 'copy', 'extract'):
            self.changed('independent_acceptance', lambda d, k=k: d['report']['forbidden_callbacks'].__setitem__(k, 1),
                         'not_successful')

    def test_complete_core_wrapper_scope_cannot_be_shrunk(self):
        for missing in preflight.REQUIRED_CODE_NAMES:
            self.blocked(fixture(code_names=preflight.REQUIRED_CODE_NAMES - {missing}), 'complete_five_core')

    def test_original_manifest_metadata_can_exceed_small_blob_bound(self):
        c, b, t = fixture(original_manifest_bytes=11189750)
        self.assertNotIn(t['original_assignment_ref']['path'], b)
        result = self.result((c, b, t))
        self.assertTrue(result['structural_valid'], result)
        self.assertFalse(result['full_original_assignment_verified'])

    def test_original_manifest_size_mismatch_is_rejected(self):
        self.changed('old_assignment', lambda d: d['original_assignment_ref'].update(bytes=11189749),
                     'manifest_lane_namespace_binding')

    def test_independently_approved_budget_cannot_lower_resource_floors(self):
        self.blocked(fixture(memory_budget=1, disk_budget=1), 'resource_budget_type')
        self.blocked(fixture(disk_budget=8 * GIB), 'reserve_formula')

    def test_original_ordinal_zero_or_duplicate_is_rejected(self):
        self.changed('old_assignment', lambda d: d['rows'][0]['source_row'].update(source_ordinal=0),
                     'source_ordinal_integer')
        self.changed('old_assignment', lambda d: d['rows'][1]['source_row'].update(source_ordinal=10000),
                     'duplicate_original_source_ordinal')

    def test_local_filesystem_and_cache_contract_is_required(self):
        for fs in ('ceph', 'nfs', 'fuse.ceph', 'unknown'):
            self.changed('resource_snapshot', lambda d, fs=fs: d.update(local_filesystem=fs),
                         'local_filesystem_required')
        self.changed('resource_snapshot', lambda d: d.update(own_cache_bytes=8 * GIB + 1),
                     'cache_budget')
        self.changed('resource_snapshot', lambda d: d.update(retained_four_lane_cache_bytes=32 * GIB + 1),
                     'cache_budget')

    def test_structural_success_never_claims_actual_exit_or_lease_verification(self):
        result = self.result()
        self.assertEqual(result['status'], 'structurally_valid_not_admitted')
        for k in ('deploy_ready', 'may_create_claims', 'full_original_assignment_verified',
                  'actual_process_exit_verified', 'actual_physical_lease_verified',
                  'full_commit_claim_artifacts_verified'):
            self.assertIs(result[k], False)
        self.assertTrue(result['local_fresh_checks_required'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
