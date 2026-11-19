"""CPU exact-history, identity, and unchanged source/resource contracts."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import main_review_watch_v9 as w
import main_review_resource_manager_v9 as manager
import build_main_review_v9_fresh as builder
h = w.historical


class ExactHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = h.capture()

    def test_two_exact_content_addressed_preimages(self):
        self.assertEqual(self.binding['source_count'], 2)
        for sid, entry in self.binding['entries'].items():
            expected = h.EXPECTED[sid][1]
            raw = Path(entry['recovered_receipt_ref']['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            self.assertEqual(Path(entry['recovered_receipt_ref']['path']).name, expected + '.json')
            value = json.loads(raw)
            self.assertEqual(value['actual_load_ref'], entry['actual_load_ref'])
            self.assertIs(entry['new_load_used_to_prove_old_call'], False)
            self.assertIs(entry['retry_budget_reset'], False)
            self.assertIs(entry['quality_approved'], False)
            self.assertEqual(value['model_identity_sha256'], entry['historical_model_inventory']['sha256'])
            self.assertEqual(len(entry['historical_model_inventory']['files']), 25)

    def test_current_receipts_are_not_old_receipt_preimages(self):
        for sid, entry in self.binding['entries'].items():
            current, ref = w.recovery.document(entry['historical_load_receipt_ref']['path'])
            self.assertNotEqual(ref['sha256'], entry['historical_load_receipt_ref']['sha256'])
            self.assertNotEqual(current['actual_load_ref']['path'], entry['actual_load_ref']['path'])
            old = json.loads(Path(entry['recovered_receipt_ref']['path']).read_bytes())
            self.assertEqual({k: v for k, v in current.items() if k != 'actual_load_ref'},
                             {k: v for k, v in old.items() if k != 'actual_load_ref'})

    def test_only_original_pid_loads_reconstruct_old_hashes(self):
        for sid, entry in self.binding['entries'].items():
            actual, actual_ref = w.recovery.document(entry['actual_load_ref']['path'])
            encode = h.native_bytes(actual)
            self.assertEqual(hashlib.sha256(h.compose(actual_ref, actual, encode)).hexdigest(), h.EXPECTED[sid][1])
            current = json.loads(Path(entry['historical_load_receipt_ref']['path']).read_bytes())
            self.assertNotEqual(hashlib.sha256(h.compose(current['actual_load_ref'], actual, encode)).hexdigest(), h.EXPECTED[sid][1])

    def test_missing_or_wrong_sha_bridge_fails_closed(self):
        for name in ('historical_load_receipt_ref', 'recovered_receipt_ref', 'actual_load_ref', 'model_identity_ref'):
            wrong = copy.deepcopy(self.binding)
            first = next(iter(wrong['entries'].values()))
            first[name]['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                h.validate(wrong)
        wrong = copy.deepcopy(self.binding)
        del wrong['entries']['ytykxzHnSDAecSEP']
        with self.assertRaises(ValueError):
            h.validate(wrong)
        wrong = copy.deepcopy(self.binding)
        next(iter(wrong['entries'].values()))['historical_model_inventory']['files'].popitem()
        with self.assertRaises(ValueError):
            h.validate(wrong)

    def test_no_exact_existing_candidate_means_pending_not_fallback(self):
        with patch.object(h, 'compose', return_value=b'not-an-exact-historical-receipt'):
            with self.assertRaisesRegex(ValueError, 'retain pending'):
                h.capture()

    def test_foreign_hint_admission_fails_before_new_incident(self):
        original = {'hint_refs': [e['original_hint_ref'] for e in self.binding['entries'].values()]}
        original['historical_load_receipts'] = self.binding
        bad = copy.deepcopy(original)
        bad['hint_refs'].reverse()
        with self.assertRaisesRegex(ValueError, 'foreign source hints'):
            w.Incident(w.ROOT / 'curated222939_production_v1',
                       w.ROOT / 'curated222939_inputs_v1/manifest.summary.json', bad)


class OriginalChain(unittest.TestCase):
    def test_real_two_full_terminal_chains_keep_exact_old_hint_refs(self):
        binding = h.capture()
        production = w.ROOT / 'curated222939_production_v1'
        manifest = w.ROOT / 'curated222939_inputs_v1/manifest.summary.json'
        cache = w.ByteCache()
        prior = w.recovery.q.sha
        w.recovery.q.sha = cache.sha
        try:
            evidence = h.Evidence(production, manifest, binding)
            rows = []
            before = {}
            for sid, entry in binding['entries'].items():
                hint, ref = w.recovery.document(entry['original_hint_ref']['path'])
                before[ref['path']] = ref['sha256']
                current_path = entry['historical_load_receipt_ref']['path']
                before[current_path] = cache.sha(current_path)
                row, refs = evidence.terminal(sid, 'train', 'train-b000001')
                self.assertEqual(row, hint['hint'])
                self.assertEqual(refs, hint['evidence'])
                self.assertEqual(refs['load_receipt_ref'], entry['historical_load_receipt_ref'])
                rows.append(row)
            rule = w.recovery.KNOWN[w.recovery.BAD_OFFSET]
            old_hint = w.recovery.document(next(iter(binding['entries'].values()))['original_hint_ref']['path'])[0]
            incident_binding = {'original_byte_begin': w.recovery.BAD_OFFSET,
                'original_byte_end': w.recovery.BAD_OFFSET + rule['size'], 'badline_ref': old_hint['badline_ref'],
                'hint_refs': [e['original_hint_ref'] for e in binding['entries'].values()],
                'historical_load_receipts': binding}
            incident = w.Incident(production, manifest, incident_binding)
            raw = Path(old_hint['badline_ref']['path']).read_bytes()
            self.assertEqual(incident.expand(w.recovery.BAD_OFFSET, raw), rows)
            self.assertEqual(before, {path: cache.sha(path) for path in before})
        finally:
            w.recovery.q.sha = prior

    def test_real_old_four_reconciliation_plus_original_two_keep_all_refs(self):
        old_two = h.capture()
        old_four = h.capture_reconciliation()
        production = w.ROOT / 'curated222939_production_v1'
        manifest = w.ROOT / 'curated222939_inputs_v1/manifest.summary.json'
        cache = w.ByteCache()
        prior = w.recovery.q.sha
        w.recovery.q.sha = cache.sha
        try:
            original = json.loads((RT / 'main_review_watch_v7-20261025_main_reconciled_v7_1792911886550195090.source_binding.json').read_bytes())
            admission = w.checked(old_four['admission_ref'])
            snapshot = w.checked(admission['snapshot_ref'])
            prefix = snapshot['original_journal_prefix']
            stream = production / 'completed.jsonl'
            identity = [stream.stat().st_dev, stream.stat().st_ino]
            empty = copy.deepcopy(original['empty_incident'])
            current_prefix = {'path': str(stream), 'identity': identity,
                              'bytes': prefix['byte_length'], 'sha256': prefix['prefix_sha256']}
            empty.update(source_identity=identity, fresh_current_node_identity=identity,
                current_verified_prefix=current_prefix,
                cross_mount_device_rebound_content_verified=identity[0] != empty['collector_snapshot_identity'][0])
            incident_binding = copy.deepcopy(original['incident'])
            incident_binding['historical_load_receipts'] = old_two
            incident_binding['historical_reconciliation_load_receipts'] = old_four
            incident = w.Incident(production, manifest, incident_binding, empty)
            membership = w.Membership(manifest)
            index = w.RawIndex(production, membership, incident=incident)
            index.identity, index.indexed_offset, index.indexed_sha = identity, prefix['byte_length'], prefix['prefix_sha256']
            index.reconciliation_mount_bindings = w.capture_reconciliation_mount_bindings(h.RECONCILIATION_MANIFEST)
            refs = h.immutable_refs(old_two) + h.immutable_refs(old_four)
            before = {r['path']: cache.sha(r['path']) for r in refs}
            current_receipts = {e['historical_load_receipt_ref']['path'] for b in (old_two, old_four) for e in b['entries'].values()}
            before.update({p: cache.sha(p) for p in current_receipts})
            self.assertEqual(index.reconcile(h.RECONCILIATION_MANIFEST, old_four['admission_ref']),
                             {'train-b000001', 'train-b000003'})
            self.assertEqual(set(index.restored_sources), set(w.v8.v7.EXPECTED_RECONCILIATION))
            self.assertEqual(len(index.restored_sources), 4)
            self.assertEqual(len(incident.rows), 2)
            self.assertFalse(index.journal_keys)
            for bid, rows in index.raw.items():
                self.assertEqual(len(membership.batch(bid)), 500)
                for value in rows.values():
                    self.assertIsNone(value['byte_begin'])
                    self.assertIsNone(value['byte_end'])
                    self.assertIsNone(value['raw_sha256'])
                    self.assertEqual(value['origin'], 'explicit_terminal_reconciliation')
            self.assertEqual(before, {p: cache.sha(p) for p in before})
            self.assertIs(w.recovery.Evidence, h.recovery.Evidence)
            self.assertIsNot(w.recovery.Evidence, h.ReconciliationEvidence)
        finally:
            w.recovery.q.sha = prior

    def test_old_four_exact_and_foreign_source_fail_closed(self):
        binding = h.capture_reconciliation()
        evidence = h.ReconciliationEvidence(w.ROOT / 'curated222939_production_v1',
            w.ROOT / 'curated222939_inputs_v1/manifest.summary.json', binding,
            h.RECONCILIATION_MANIFEST, binding['admission_ref'])
        with self.assertRaisesRegex(ValueError, 'Foreign source'):
            evidence.terminal('MVuipekTKQDQprcC', 'train', 'train-b000001')
        with self.assertRaisesRegex(ValueError, 'active original source'):
            evidence.historical_receipt(next(iter(binding['entries'].values()))['historical_load_receipt_ref']['path'])
        wrong = copy.deepcopy(binding)
        wrong['admission_ref']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            h.validate_reconciliation(wrong)
        with patch.object(h, 'compose', return_value=b'not-exact'):
            with self.assertRaisesRegex(ValueError, 'retain pending'):
                h.capture_reconciliation()


class Contracts(unittest.TestCase):
    def test_original_native_cpu_helper_interfaces_are_available(self):
        self.assertIs(w.NormalizedReader, w.v8.NormalizedReader)
        for name in ("need", "sha", "retained_selection", "install_history_selection",
                     "reviewer_from_verified_local_model", "NormalizedReader"):
            self.assertTrue(callable(getattr(w, name)))

    def test_v7_and_v8_execution_files_stay_frozen(self):
        frozen = {'main_review_watch_v7.py': w.V7_SHA, 'main_review_watch_v8.py': w.V8_SHA,
            'main_review_resource_manager_v7.py': manager.OLD_MANAGER_SHA,
            'main_review_resource_manager_v8.py': manager.MANAGER_V8_SHA,
            'build_main_review_v7_fresh.py': builder.BUILDER_V7_SHA,
            'build_main_review_v8_fresh.py': builder.BUILDER_V8_SHA,
            'test_main_review_v8.py': '615de6ff8e70bd63b7eb562237029bc8fb9a73be701b9382dced5e021275f71c'}
        for name, digest in frozen.items():
            self.assertEqual(hashlib.sha256((RT / name).read_bytes()).hexdigest(), digest)

    def test_native_history_four_plus_four_and_five_bytes_unchanged(self):
        self.assertTrue(issubclass(w.RawIndex, w.v8.RawIndex))
        self.assertIs(w.RawIndex.scan, w.v8.RawIndex.scan)
        self.assertIs(w.RawIndex.reconcile_offset, w.v8.RawIndex.reconcile_offset)
        self.assertIs(w.retained_selection, w.v8.retained_selection)
        self.assertIs(w.install_history_selection, w.v8.install_history_selection)
        self.assertEqual(w.RAW, bytes.fromhex('7465227d0a'))
        self.assertEqual(len(w.RAW), 5)
        self.assertIs(w.capture_offset_binding, w.v8.capture_offset_binding)
        self.assertIn('reader.reconcile(a.reconciliation_manifest', builder.source)
        self.assertIn("binding['offset2053650']", builder.source)
        self.assertIn("'expected_batch_count'] = 448", builder.source)
        self.assertIn("'selected_source_count'] == 10000", builder.source)
        self.assertIn("'source_range'] == {'start': 21000, 'end': 31000}", builder.source)

    def test_builder_and_manager_new_identity_globals_match(self):
        self.assertIs(builder.main.__globals__['cpu_smoke'], builder.cpu_smoke)
        self.assertEqual(Path(manager.main.__globals__['__file__']).name, 'main_review_resource_manager_v9.py')
        self.assertEqual(Path(manager._offset_validate.__globals__['__file__']).name, 'main_review_resource_manager_v9.py')
        self.assertIs(manager.base.validate_pair, manager.validate)
        self.assertIn("RT / 'main_review_watch_v9.py'", builder.source)
        self.assertIn("RT / 'main_review_resource_manager_v9.py'", builder.source)
        self.assertIn("RT / 'build_main_review_v9_fresh.py'", builder.source)
        self.assertIn("RT / 'test_main_review_v9.py'", builder.source)

    def test_slow_source_proof_precedes_actualfresh_resources(self):
        self.assertLess(builder.source.index('capture_historical_load_binding'), builder.source.index('reader.scan()'))
        self.assertLess(builder.source.index('proof = cpu_smoke'), builder.source.index('uuid = base.uuid_for(7)'))
        self.assertIn('base.check_lease_unchanged(reserve, lease)', builder.source)
        self.assertIn('base.check_holder', builder.source)
        self.assertIs(manager.release_holder, manager.v8.release_holder)
        self.assertIs(manager._release, manager.v8._release)
        self.assertEqual(manager.base.AUTHORIZED_PAIRS, {'N5': ((7,),)})

    def test_no_new_gpu_load_signal_or_budget_in_cpu_builder(self):
        calls = {getattr(n.func, 'attr', '') for n in ast.walk(ast.parse(builder.source)) if isinstance(n, ast.Call)}
        self.assertFalse({'kill', 'pidfd_send_signal', 'Popen', 'release_holder'} & calls)
        self.assertIn('160 * 2**30', builder.source)
        self.assertIn("proof['native_cache_reused_source_count'] == 80", (RT / 'build_main_review_v7_fresh.py').read_text())
        self.assertIn("proof['native_candidate_error_count'] == 7", (RT / 'build_main_review_v7_fresh.py').read_text())
        self.assertIn("proof['new_VLM_calls'] == 0", (RT / 'build_main_review_v7_fresh.py').read_text())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=('exact', 'chain', 'contracts'), required=True)
    args = ap.parse_args()
    selected = {'exact': ExactHistory, 'chain': OriginalChain, 'contracts': Contracts}[args.group]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(selected))
    sys.exit(not result.wasSuccessful())
