"""CPU tests; real byte/evidence checks are separate from mock resource regressions."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import main_review_watch_v7 as w
import main_review_resource_manager_v7 as manager
import build_main_review_v7_fresh as builder


class Members:
    descriptors = {'train-b000000': {'split': 'train'}, 'train-b000001': {'split': 'train'}}
    def batch(self, bid):
        return {'first': {'id': 'first'}, 'second': {'id': 'second'}}


def row(sid, batch='train-b000000'):
    return {'id': sid, 'split': 'train', 'batch_id': batch, 'status': 'joint_candidate'}


def encoded(value):
    return (json.dumps(value) + '\n').encode()


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stream = self.root / 'completed.jsonl'
        self.stream.write_bytes(encoded(row('first')) + encoded(row('second', 'train-b000001')))
        self.index = w.RawIndex(self.root, Members())

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_reconciliation_admission_preserves_original_scope(self):
        membership = w.Membership(w.ROOT / 'curated222939_inputs_v1/manifest.summary.json')
        index = w.RawIndex(w.ROOT / 'curated222939_production_v1', membership)
        scope = no_reconciliation_scope(index)
        self.assertEqual(scope['main_original_source_count'], 222939)
        self.assertEqual(scope['main_expected_batch_count'], 448)
        self.assertEqual(scope['additional_original_source_count'], 10000)
        self.assertEqual(scope['additional_expected_batch_count'], 20)
        self.assertEqual(scope['reconciliation_restored_source_count'], 0)
        self.assertFalse(scope['all_original_batches_approved'])

    def test_fixed_mount_binding_never_refreshes_changed_inode_or_path(self):
        historic = [9999, self.stream.stat().st_ino]
        raw = self.root / 'snapshot.raw'
        raw.write_bytes(self.stream.read_bytes())
        digest = w.sha(raw)
        prefix = {'path': str(self.stream), 'input_identity': historic, 'byte_length': raw.stat().st_size, 'prefix_sha256': digest}
        raw_ref = {'path': str(raw), 'sha256': digest}
        binding = {'path': str(self.stream), 'original_prefix': prefix, 'raw_prefix_ref': raw_ref,
                   'historical_capture_identity': historic,
                   'fresh_current_node_identity': [self.stream.stat().st_dev, self.stream.stat().st_ino],
                   'current_node_hostname': os.uname().nodename, 'cross_mount_device_rebound_content_verified': True,
                   'prefix_content_SHA_verified': True}
        w.verify_mount_binding(binding, prefix, raw_ref, str(self.stream))
        wrong = copy.deepcopy(binding); wrong['fresh_current_node_identity'][1] += 1
        with self.assertRaises(ValueError):
            w.verify_mount_binding(wrong, prefix, raw_ref, str(self.stream))
        wrong = copy.deepcopy(binding); wrong['path'] = '/tmp/foreign/completed.jsonl'
        with self.assertRaises(ValueError):
            w.verify_mount_binding(wrong, prefix, raw_ref, str(self.stream))
        wrong = copy.deepcopy(binding); wrong['cross_mount_device_rebound_content_verified'] = False
        with self.assertRaises(ValueError):
            w.verify_mount_binding(wrong, prefix, raw_ref, str(self.stream))
        replacement = self.root / 'replacement'
        replacement.write_bytes(self.stream.read_bytes())
        replacement.replace(self.stream)
        with self.assertRaises(ValueError):
            w.verify_mount_binding(binding, prefix, raw_ref, str(self.stream))

    def test_builder_worker_api(self):
        tree = ast.parse((RT / 'build_main_review_v7_fresh.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'worker':
                self.assertTrue(hasattr(w, node.attr), 'Missing worker API: ' + node.attr)

    def test_raw_is_not_terminal_approval(self):
        with patch.object(w, 'terminal_record', side_effect=AssertionError('Raw scan must not read consumer terminal')):
            self.assertEqual(self.index.scan(), {'train-b000000', 'train-b000001'})
        self.assertEqual(self.index.offset, 0)
        self.assertFalse(self.index.completed)
        self.assertGreater(self.index.indexed_offset, 0)

    def test_contiguous_verified_watermark(self):
        self.index.scan()
        with patch.object(w, 'terminal_record', side_effect=lambda p, v, m: v):
            self.index.verify('train-b000001')
            self.assertEqual(self.index.offset, 0)
            self.index.verify('train-b000000')
        self.assertEqual(self.index.offset, self.index.indexed_offset)

    def test_unknown_blank_and_badline_stop_without_skip(self):
        for bad in (b'\n', b'{bad}\n'):
            self.stream.write_bytes(encoded(row('first')) + bad + encoded(row('second')))
            index = w.RawIndex(self.root, Members())
            index.scan()
            self.assertEqual(index.indexed_offset, len(encoded(row('first'))))
            self.assertEqual(index.pending['reason'], 'unknown_malformed_line')
            self.assertIs(index.pending['skipped'], False)
            self.assertNotIn('second', index.raw['train-b000000'])

    def test_half_line_remains_at_original_boundary(self):
        first, second = encoded(row('first')), encoded(row('second'))
        self.stream.write_bytes(first + second[:-1])
        self.index.scan()
        self.assertEqual(self.index.indexed_offset, len(first))
        with self.stream.open('ab') as f:
            f.write(b'\n')
        self.index.scan()
        self.assertIn('second', self.index.raw['train-b000000'])

    def test_previously_consumed_prefix_change_rejected(self):
        self.index.scan()
        self.stream.write_bytes(self.stream.read_bytes().replace(b'first', b'FIRST'))
        with self.assertRaises(ValueError):
            self.index.scan()

    def test_foreign_split_source_and_batch_fail_closed(self):
        for changed in ({**row('first'), 'split': 'val'}, row('foreign'), row('first', 'train-b000999')):
            self.stream.write_bytes(encoded(changed))
            index = w.RawIndex(self.root, Members())
            index.scan()
            self.assertEqual(index.indexed_offset, 0)
            self.assertTrue(index.pending)
            self.assertFalse(index.completed)

    def test_same_snapshot_and_append_same_sid_dedup(self):
        self.stream.write_bytes(encoded(row('first')) * 2)
        self.index.scan()
        self.assertEqual(len(self.index.raw['train-b000000']), 1)
        with self.stream.open('ab') as f:
            f.write(encoded(row('first')))
        self.index.scan()
        self.assertEqual(len(self.index.raw['train-b000000']), 1)
        before = self.index.indexed_offset
        with self.stream.open('ab') as f:
            f.write(encoded({**row('first'), 'status': 'failed'}))
        self.index.scan()
        self.assertEqual(self.index.indexed_offset, before)
        self.assertTrue(self.index.pending)

    def test_same_snapshot_conflicting_revision_is_not_committed(self):
        self.stream.write_bytes(encoded(row('first')) + encoded({**row('first'), 'status': 'failed'}))
        self.index.scan()
        self.assertEqual(self.index.indexed_offset, len(encoded(row('first'))))
        self.assertEqual(self.index.raw['train-b000000']['first']['hint'], row('first'))
        self.assertTrue(self.index.pending)

    def test_changed_read_snapshot_never_commits_raw(self):
        original = w.recovery.prefix_sha
        with patch.object(w.recovery, 'prefix_sha', side_effect=lambda p, e, i: '0' * 64 if e else original(p, e, i)):
            with self.assertRaises(ValueError):
                self.index.scan()
        self.assertEqual(self.index.indexed_offset, 0)
        self.assertFalse(self.index.raw)

    def test_missing_evidence_is_pending_source_sha_failure_is_fatal(self):
        self.index.scan()
        with patch.object(w, 'terminal_record', side_effect=FileNotFoundError('not published')):
            self.assertFalse(self.index.verify('train-b000000'))
        with patch.object(w, 'terminal_record', side_effect=ValueError('source SHA changed')):
            with self.assertRaises(ValueError):
                self.index.verify('train-b000000')
        self.assertEqual(self.index.offset, 0)

    def test_pending_never_approves_population(self):
        base = {'all_source_labels_completed': True, 'sampled_sources_reviewed': 0}
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            self.index.pending = {'offset': 0, 'reason': 'unknown'}
            value = w.progress(self.index, self.root, 'test', set(), 0)
            self.assertFalse(value['all_source_labels_completed'])
            self.assertFalse(value['full_population_or_quality_inferred_from_recovery'])

    def test_reconciliation_is_not_physical_raw_population(self):
        self.index.raw['train-b000000']['first'] = {'hint': row('first'), 'byte_begin': None, 'origin': 'explicit_terminal_reconciliation'}
        self.index.completed['train-b000000']['first'] = row('first')
        self.index.verified_keys.add(('train-b000000', 'first'))
        self.index.restored_sources['train/first'] = {'original_byte_offset': None}
        base = {'all_source_labels_completed': False, 'sampled_sources_reviewed': 0}
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            value = w.progress(self.index, self.root, 'test', set(), 0)
        self.assertEqual(value['indexed_raw_source_count'], 0)
        self.assertEqual(value['unverified_raw_source_count'], 0)
        self.assertEqual(value['admitted_terminal_hint_count'], 1)
        self.assertEqual(value['reconciliation_restored_source_count'], 1)
        self.index.scan()
        with patch.object(w.v5, 'context_progress', return_value=base.copy()):
            value = w.progress(self.index, self.root, 'test', set(), 0)
        self.assertEqual(value['indexed_raw_source_count'], 2)
        self.assertEqual(value['verified_journal_source_count'], 1)
        self.assertEqual(value['unverified_raw_source_count'], 1)
        self.assertEqual(value['admitted_terminal_hint_count'], 2)

    def test_sha_cache_requires_actual_first_hash_and_immutable_stat(self):
        cache = w.ByteCache()
        digest = cache.sha(self.stream)
        self.assertEqual(digest, w.sha(self.stream))
        self.stream.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            cache.sha(self.stream)

    def test_mock_holder_contract_not_gpu_release(self):
        cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
        expected = copy.deepcopy(cfg['holders'][0])
        current = copy.deepcopy(expected)
        running = [{'uuid': expected['uuid'], 'pid': expected['pid']}]
        manager.base.check_holder(expected, current, 7, expected['uuid'], running)
        for key in manager.base.IDENTITY_KEYS:
            wrong = copy.deepcopy(current)
            wrong[key] = -1 if isinstance(wrong[key], int) else ['foreign'] if isinstance(wrong[key], list) else 'foreign'
            with self.assertRaises(RuntimeError):
                manager.base.check_holder(expected, wrong, 7, expected['uuid'], running)
        for field in ('sig_ign', 'sig_blk'):
            wrong = copy.deepcopy(current)
            wrong[field] |= 1 << (signal.SIGTERM - 1)
            with self.assertRaises(RuntimeError):
                manager.base.check_holder(expected, wrong, 7, expected['uuid'], running)
        with self.assertRaises(RuntimeError):
            manager.base.check_holder(expected, current, 7, 'foreignUUID', running)
        with self.assertRaises(RuntimeError):
            manager.base.check_holder(expected, current, 7, expected['uuid'], running + [{'uuid': expected['uuid'], 'pid': -1}])
        wrong = copy.deepcopy(current)
        wrong['argv'][-1] = str(RT / 'main_review_watch_v5.py')
        not_holder = copy.deepcopy(wrong)
        with self.assertRaises(RuntimeError):
            manager.base.check_holder(not_holder, wrong, 7, expected['uuid'], running)
        tree = ast.parse((RT / 'build_main_review_v7_fresh.py').read_text())
        invoked = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({'release_holder', 'kill', 'pidfd_send_signal', 'Popen', 'capture'} & invoked)
        self.assertEqual(manager.BASE_MANAGER_SHA, '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6')

    def test_lease_and_160GiB_budget_contract(self):
        self.assertEqual(manager.base.required_budget({}), 160 * 2**30)
        lease = self.root / 'lease'
        self.assertIsNone(manager.base.lease_snapshot(lease))
        lease.touch()
        with self.assertRaises(RuntimeError):
            manager.base.lease_snapshot(lease)
        os.utime(lease, (time.time() - 1800, time.time() - 1800))
        snapshot = manager.base.lease_snapshot(lease)
        manager.base.check_lease_unchanged(lease, snapshot)
        lease.write_bytes(b'foreign')
        with self.assertRaises(RuntimeError):
            manager.base.check_lease_unchanged(lease, snapshot)
        with self.assertRaises(RuntimeError):
            manager.base.lease_snapshot(lease)

    def test_manager_frozen_sourcebinding_and_no_mutable_output_authority(self):
        cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
        cfg['argv'][cfg['argv'].index(str(RT / 'curated_review_watch_v3.py'))] = str(RT / 'main_review_watch_v7.py')
        cfg.update(producer={'path': str(RT / 'main_review_watch_v7.py'), 'sha256': w.sha(RT / 'main_review_watch_v7.py')},
                   phase='main_review_watch_v7', queue=str(w.ROOT / 'queues/main_review_watch_v7-unit'),
                   source_prepared=True, expected_batch_count=448, semantic_approval=False, release_authorized=True)
        cfg['argv'].extend(['--additional-results-root', str(w.ROOT / 'curated222939_collaborative_v2'),
                            '--additional-out', str(w.ROOT / 'curated_independent_reviews_collaborative_v2'),
                            '--additional-manifest', str(RT / 'curated_collaborative21000_auto_review_manifest_v2.json')])
        stream = w.ROOT / 'curated222939_production_v1/completed.jsonl'
        stat = stream.stat()
        binding = {'schema': 'main_review_v7_original_stream_binding_v1',
                   'validation_mode': 'finite_raw_index_batch_lazy', 'remaining_raw_sources_are_unverified': True,
                   'main_prefix': {'path': str(stream), 'identity': [stat.st_dev, stat.st_ino], 'bytes': 0, 'sha256': w.recovery.prefix_sha(stream, 0, [stat.st_dev, stat.st_ino])},
                   'empty_incident': {'original_byte_begin': w.EMPTY_OFFSET, 'original_byte_end': w.EMPTY_OFFSET + 1,
                                     'receipt_ref': w.recovery.document(w.ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json')[1]}}
        empty = binding['empty_incident']
        historic = w.checked(empty['receipt_ref'])['stable_badline_snapshots'][0]['identity'][:2]
        empty.update(collector_snapshot_identity=historic, historical_capture_identity=historic,
                     fresh_current_node_identity=binding['main_prefix']['identity'], current_verified_prefix=binding['main_prefix'],
                     cross_mount_device_rebound_content_verified=stat.st_dev != historic[0])
        path = self.root / 'binding.json'
        path.write_text(json.dumps(binding))
        cfg['source_binding'] = {'path': str(path), 'sha256': w.sha(path)}
        cfg['source_sha256'][str(path)] = cfg['source_binding']['sha256']
        empty_ref = binding['empty_incident']['receipt_ref']
        cfg['source_sha256'][empty_ref['path']] = empty_ref['sha256']
        cfg['argv'].extend(['--source-binding', str(path)])
        with patch.object(manager.subprocess, 'check_output', return_value='28.216.19.137'):
            manager.validate(cfg)
            for field, value in (('release_authorized', False), ('phase', 'main_review_watch_v5'), ('gpus', [6]),
                                 ('queue', '/tmp/main_review_watch_v7-foreign')):
                wrong = copy.deepcopy(cfg); wrong[field] = value
                with self.assertRaises(RuntimeError):
                    manager.validate(wrong)
            wrong = copy.deepcopy(cfg)
            wrong['source_sha256'][str(w.ROOT / 'curated_independent_reviews_v1/train-b000000/summary.json')] = '0' * 64
            with self.assertRaises(RuntimeError):
                manager.validate(wrong)
            wrong = copy.deepcopy(cfg); wrong['source_binding']['sha256'] = '0' * 64
            with self.assertRaises(RuntimeError):
                manager.validate(wrong)
            wrong = copy.deepcopy(cfg); wrong['argv'].extend(['--reconciliation-manifest', '/tmp/foreign.json'])
            with self.assertRaises(RuntimeError):
                manager.validate(wrong)


def real_reader():
    cache = w.ByteCache()
    w.recovery.q.sha = cache.sha
    manifest = w.ROOT / 'curated222939_inputs_v1/manifest.summary.json'
    membership = w.Membership(manifest)
    production = w.ROOT / 'curated222939_production_v1'
    hints = [w.recovery.document(w.ROOT / ('curated222939_repair_v1/collector-recovery-v1/hints/train/' + sid + '.json')) for sid in w.recovery.KNOWN[w.recovery.BAD_OFFSET]['ids']]
    bad_ref = hints[0][0]['badline_ref']
    incident_binding = {'original_byte_begin': w.recovery.BAD_OFFSET, 'original_byte_end': w.recovery.BAD_OFFSET + 940,
        'badline_ref': bad_ref, 'hint_refs': [ref for _, ref in hints]}
    stream = production / 'completed.jsonl'
    s = stream.stat()
    identity = [s.st_dev, s.st_ino]
    with stream.open('rb') as f:
        raw = f.read(s.st_size)
    end = raw.rfind(b'\n') + 1
    prefix = {'path': str(stream), 'identity': identity, 'bytes': end, 'sha256': __import__('hashlib').sha256(raw[:end]).hexdigest()}
    empty = {'original_byte_begin': w.EMPTY_OFFSET, 'original_byte_end': w.EMPTY_OFFSET + 1,
        'source_identity': identity,
        'collector_snapshot_identity': w.recovery.document(w.ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json')[0]['stable_badline_snapshots'][0]['identity'][:2],
        'receipt_ref': w.recovery.document(w.ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json')[1]}
    empty.update(historical_capture_identity=empty['collector_snapshot_identity'], fresh_current_node_identity=identity,
                 current_verified_prefix=prefix, cross_mount_device_rebound_content_verified=identity[0] != empty['collector_snapshot_identity'][0])
    incident = w.Incident(production, manifest, incident_binding, empty)
    reader = w.RawIndex(production, membership, incident=incident, binding=prefix)
    reader.scan()
    for _ in range(5):
        if not reader.pending or not any(x in reader.pending['reason'] for x in ('not stable', 'Concurrent stream write')):
            break
        time.sleep(.2)
        reader.scan()
    w.need(reader.pending is None or reader.pending['reason'] == 'partial_original_line', 'Actual original stream blocked: ' + str(reader.pending))
    w.need(reader.recovered_offsets == [w.recovery.BAD_OFFSET] and reader.empty_row_offsets == [w.EMPTY_OFFSET], 'Real exact two-source/empty row proof missing')
    cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
    binding = {'main_prefix': prefix, 'incident': incident_binding, 'empty_incident': empty,
        'manifest_ref': membership.ref, 'model_stage_ref': cfg['modelref'],
        'annotator_model_ref': {'path': cfg['argv'][cfg['argv'].index('--annotator-model-reference') + 1]}}
    return reader, binding, cache


def must_reject(action, name):
    try:
        action()
    except (ValueError, KeyError, TypeError, OSError):
        return name
    raise AssertionError('Negative accepted: ' + name)


def no_reconciliation_scope(reader):
    cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
    annotator = cfg['argv'][cfg['argv'].index('--annotator-model-reference') + 1]
    extra_ref = w.recovery.document(RT / 'curated_collaborative21000_auto_review_manifest_v2.json')[1]
    args = argparse.Namespace(results_root=str(reader.production), annotator_model_reference=annotator, reconciliation_manifest=None,
        additional_results_root=str(w.ROOT / 'curated222939_collaborative_v2'),
        additional_out=str(w.ROOT / 'curated_independent_reviews_collaborative_v2'), additional_manifest=extra_ref['path'])
    binding = {'schema': 'main_review_v7_original_stream_binding_v1', 'manifest_ref': reader.membership.ref,
        'validation_mode': 'finite_raw_index_batch_lazy', 'remaining_raw_sources_are_unverified': True,
        'main_prefix': {'path': str(reader.stream)}, 'annotator_model_ref': {'path': annotator, 'sha256': w.sha(annotator)},
        'additional_edition': {'results_root': args.additional_results_root, 'out': args.additional_out, 'manifest': extra_ref}}
    bids = w.validate_binding(binding, args, reader.membership)
    extra = w.RawIndex(args.additional_results_root, reader.membership, allowed_batches=bids)
    with tempfile.TemporaryDirectory() as tmp:
        main_progress = w.progress(reader, tmp, 'main-production-v1', set(reader.membership.descriptors), 222939)
        extra_progress = w.progress(extra, tmp, 'collaborative-v2', bids, 10000)
    w.need(reader.reconciliation_manifest_ref is None and not reader.restored_sources and not extra.restored_sources,
           'No-reconciliation startup falsely claims restored sources')
    w.need((main_progress['total_source_count'], main_progress['expected_batch_count'],
            extra_progress['total_source_count'], extra_progress['expected_batch_count']) == (222939, 448, 10000, 20),
           'No-reconciliation startup altered authoritative edition population')
    w.need(not main_progress['all_source_labels_completed'] and not main_progress['all_batches_reviewed'],
           'No-reconciliation source scan falsely approves full population')
    return {'reconciliation_manifest_enabled': False, 'reconciliation_restored_source_count': 0,
            'main_original_source_count': main_progress['total_source_count'], 'main_expected_batch_count': main_progress['expected_batch_count'],
            'additional_original_source_count': extra_progress['total_source_count'], 'additional_expected_batch_count': extra_progress['expected_batch_count'],
            'all_original_batches_approved': False}


def stage1_checks(cpu_proof=None):
    reader, binding, cache = real_reader()
    result = no_reconciliation_scope(reader)
    result.update(schema='main_review_v7_stage1_actual_CPU_v1', source_binding_empty_incident=binding['empty_incident'],
        original_prefix=binding['main_prefix'], indexed_raw_byte_offset=reader.indexed_offset,
        indexed_raw_source_count=len(reader.journal_keys), physical_verified_cursor=reader.offset,
        recovered_badline_offsets=reader.recovered_offsets, empty_row_offsets=reader.empty_row_offsets,
        empty_encoded_source_count=0, empty_reconstructed_source_count=0, GPU_model_loaded=False, new_VLM_calls=0,
        original_logs_scores_state_budget_modified=False, worker_sha256=w.sha(RT / 'main_review_watch_v7.py'))
    negatives = []
    for field, change, name in (
            ('fresh_current_node_identity', lambda v: [v[0], v[1] + 1], 'wrong_current_source_inode'),
            ('historical_capture_identity', lambda v: [v[0], v[1] + 1], 'wrong_historical_source_inode'),
            ('cross_mount_device_rebound_content_verified', lambda v: not v, 'missing_cross_mount_content_proof')):
        wrong = copy.deepcopy(binding['empty_incident']); wrong[field] = change(wrong[field])
        bad = w.Incident(reader.production, reader.membership.ref['path'], binding['incident'], wrong)
        negatives.append(must_reject(lambda: bad.expand(w.EMPTY_OFFSET, b'\n'), name))
    wrong = copy.deepcopy(binding['empty_incident']); wrong['current_verified_prefix']['path'] = '/tmp/foreign/completed.jsonl'
    bad = w.Incident(reader.production, reader.membership.ref['path'], binding['incident'], wrong)
    negatives.append(must_reject(lambda: bad.expand(w.EMPTY_OFFSET, b'\n'), 'wrong_original_source_path'))
    negatives.append(must_reject(lambda: reader.incident.expand(w.EMPTY_OFFSET + 1, b'\n'), 'unknown_new_blank'))
    negatives.append(must_reject(lambda: reader.incident.expand(w.EMPTY_OFFSET, b' \n'), 'wrong_empty_raw_SHA'))
    result['negative_rejections'] = negatives
    if cpu_proof:
        path = Path(cpu_proof).resolve()
        w.need(path.is_relative_to(Path('/tmp')), 'Stage1 test receipt must stay in /tmp')
        builder.save_new(path, result)
    return result


def actual_checks(reconciliation_manifest=None, cached80=False, cpu_proof=None):
    reader, binding, cache = real_reader()
    negatives = []
    incident = reader.incident
    negatives.append(must_reject(lambda: incident.expand(w.EMPTY_OFFSET + 1, b'\n'), 'unknown_new_blank'))
    negatives.append(must_reject(lambda: incident.expand(w.EMPTY_OFFSET, b' \n'), 'wrong_empty_raw_SHA'))
    bad_empty = copy.deepcopy(binding['empty_incident'])
    bad_empty['receipt_ref']['sha256'] = '0' * 64
    bad = w.Incident(reader.production, reader.membership.ref['path'], binding['incident'], bad_empty)
    negatives.append(must_reject(lambda: bad.expand(w.EMPTY_OFFSET, b'\n'), 'wrong_empty_receipt_SHA'))
    bad_empty = copy.deepcopy(binding['empty_incident'])
    bad_empty['original_byte_end'] += 1
    bad = w.Incident(reader.production, reader.membership.ref['path'], binding['incident'], bad_empty)
    negatives.append(must_reject(lambda: bad.expand(w.EMPTY_OFFSET, b'\n'), 'wrong_empty_origin_span'))
    bad = w.Incident(w.ROOT / 'curated222939_collaborative_v2', reader.membership.ref['path'], binding['incident'], binding['empty_incident'])
    negatives.append(must_reject(lambda: bad.expand(w.EMPTY_OFFSET, b'\n'), 'foreign_empty_root'))
    evidence = w.recovery.Evidence(reader.production, reader.membership.ref['path'])
    four = []
    for key, bid in sorted(w.EXPECTED_RECONCILIATION.items()):
        split, sid = key.split('/')
        terminal, refs = evidence.terminal(sid, split, bid)
        normalized = w.terminal_record(reader.production, terminal, reader.membership)
        w.need(len(reader.membership.batch(bid)) == 500 and terminal['joint_complete'] is True and normalized['label_status'] == 'completed', 'Actual four-source full500/JOINT proof failed')
        w.need(sid not in reader.raw[bid], 'Expected missing journal source is no longer absent; reconsider frozen snapshot')
        four.append({'source_key': key, 'batch_id': bid, 'consumer_terminal_ref': refs['consumer_terminal_ref'], 'joint_result_ref': refs['joint_result_ref'],
                     'label_sha256': normalized['label_sha256'], 'all_source_refs_SHA_verified': True})
    result = {'schema': 'main_review_v7_actual_CPU_v1', 'original_prefix': binding['main_prefix'],
        'indexed_raw_source_count': sum(len(v) for v in reader.raw.values()), 'physical_cursor_before_reconciliation': reader.offset,
        'old_splice_offsets': reader.recovered_offsets, 'empty_row_offsets': reader.empty_row_offsets,
        'empty_encoded_sources': 0, 'empty_reconstructed_sources': 0, 'empty_receipt_identity_binding': binding['empty_incident'],
        'actual_four_terminal_evidence': four,
        'reconciliation_manifest_admitted': False, 'negative_rejections': negatives, 'GPU_model_loaded': False,
        'new_VLM_calls': 0, 'all_448_batches_approved': False, 'training_approved': False}
    if reconciliation_manifest:
        reader.reconciliation_mount_bindings = w.capture_reconciliation_mount_bindings(reconciliation_manifest)
        before = reader.offset, reader.indexed_offset, reader.indexed_sha, len(reader.lines), len(reader.journal_keys)
        reader.reconcile(reconciliation_manifest)
        w.need(before == (reader.offset, reader.indexed_offset, reader.indexed_sha, len(reader.lines), len(reader.journal_keys)), 'Four restored sources changed original physical cursor')
        w.need(len(reader.restored_sources) == 4 and all(v['original_byte_offset'] is None for v in reader.restored_sources.values()), 'False original offset assigned')
        counts = sum(len(v) for v in reader.raw.values())
        reader.reconcile(reconciliation_manifest)
        w.need(counts == sum(len(v) for v in reader.raw.values()), 'Same candidate reconciliation did not dedup')
        admission = w.checked(reader.reconciliation_manifest_ref)
        record_ref = admission['record_refs'][0]
        original_checked = w.checked
        original = original_checked(record_ref)
        variants = []
        changed = copy.deepcopy(original); changed['original_byte_offset'] = w.EMPTY_OFFSET; variants.append(('false_blank_original_offset', changed))
        changed = copy.deepcopy(original); changed['hint']['split'] = 'val'; variants.append(('foreign_split', changed))
        changed = copy.deepcopy(original); changed['hint']['id'] = 'foreignSource'; variants.append(('foreign_source', changed))
        changed = copy.deepcopy(original); changed['completed_row_sha256'] = '0' * 64; variants.append(('candidate_SHA', changed))
        changed = copy.deepcopy(original); changed['current_original_manifest']['original_csv_ref']['sha256'] = '0' * 64; variants.append(('CSV_identity', changed))
        changed = copy.deepcopy(original); changed['original_consumer_terminal_ref']['path'] = '/tmp/foreign.terminal.json'; variants.append(('foreign_terminal_root', changed))
        changed = copy.deepcopy(original); changed['quality_approved'] = True; variants.append(('false_quality_approval', changed))
        for name, changed in variants:
            with patch.object(w, 'checked', side_effect=lambda ref, v=changed: v if ref == record_ref else original_checked(ref)):
                negatives.append(must_reject(lambda: reader.reconcile(reconciliation_manifest), name))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'completed.jsonl'
            sid = sorted(w.EXPECTED_RECONCILIATION)[0].split('/')[1]
            bid = w.EXPECTED_RECONCILIATION['train/' + sid]
            hint = reader.raw[bid][sid]['hint']
            p.write_bytes(w.recovery.q.encoded(hint))
            appended = w.RawIndex(tmp, reader.membership, allowed_batches={bid})
            appended.raw[bid][sid] = copy.deepcopy(reader.raw[bid][sid])
            appended.completed[bid][sid] = copy.deepcopy(reader.completed[bid][sid])
            appended.verified_keys.add((bid, sid))
            appended.scan()
            w.need(len(appended.raw[bid]) == 1, 'Appended real same-SID candidate counted twice')
            with p.open('ab') as stream:
                stream.write(w.recovery.q.encoded({**hint, 'status': 'failed'}))
            end = appended.indexed_offset
            appended.scan()
            w.need(appended.pending and appended.indexed_offset == end, 'Changed appended candidate silently replaced label')
        result.update(reconciliation_manifest_admitted=True, reconciliation_manifest_ref=reader.reconciliation_manifest_ref,
                      restored_sources=reader.restored_sources, actual_restored_unique_sources=4,
                      physical_cursor_after_reconciliation=reader.offset, real_candidate_append_dedup=True)
    if cached80:
        w.need(reader.verify('train-b000000', w.historical_ids(w.ROOT / 'curated_independent_reviews_v1', 'train-b000000')), 'Actual old score evidence not published')
        out = Path(tempfile.mkdtemp(prefix='main-review-v7-actual80-'))
        proof = builder.cpu_smoke(w, binding, reader, cache, out)
        result['actual80_cached_native_proof'] = proof
        result['CPU_sandbox_out'] = str(out)
    result['actual_original_CSV_SHA_verified'] = {p: digest for p, (_, digest) in cache.files.items() if p.endswith('/train.csv')}
    w.need(result['actual_original_CSV_SHA_verified'], 'Actual CSV never SHA verified')
    if cpu_proof:
        path = Path(cpu_proof).resolve()
        w.need(path.is_relative_to(Path('/tmp')), 'Actual test receipt must stay in /tmp')
        builder.save_new(path, result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--actual', action='store_true')
    ap.add_argument('--stage1', action='store_true')
    ap.add_argument('--reconciliation-manifest')
    ap.add_argument('--cached80', action='store_true')
    ap.add_argument('--cpu-proof')
    a, rest = ap.parse_known_args()
    if a.stage1:
        w.need(not a.reconciliation_manifest and not a.cached80, 'Stage1 scope does not run reconciliation/80score media replay')
        print(json.dumps(stage1_checks(a.cpu_proof), ensure_ascii=False), flush=True)
    elif a.actual:
        print(json.dumps(actual_checks(a.reconciliation_manifest, a.cached80, a.cpu_proof), ensure_ascii=False), flush=True)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
