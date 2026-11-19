"""CPU-only contracts. Every writable fixture lives under the candidate directory."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recover03 as r


class Fixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cpu-fixture-', dir=r.CANDIDATE)
        self.root = Path(self.temporary.name)
        self.candidate = self.root / 'candidate'
        self.candidate.mkdir()
        (self.root / 'logs').mkdir()
        (self.root / 'gpu_locks').mkdir()
        self.modelinfo = {'model_signatures': {}, 'old_cache_identity_sha256': 'a' * 64,
                          'old_verified_models_unchanged': True}
        for gpu, (pid, ticks) in enumerate(r.ORIGINAL_IDENTITIES):
            oldpath = self.root / 'logs' / f'unified_gpu{gpu}_progress.json'
            old = {'worker': r.WORKER, 'physical_gpu_index': gpu, 'pid': 900000000 + gpu,
                   'start_ticks': 1, 'status': 'completed', 'ended_unix': 1,
                   'assignment_sha256': 'b' * 64, 'run_id': f'oldg{gpu}'}
            oldpath.write_bytes(r.base.raw_json(old))
            oldlease = self.root / 'gpu_locks' / f'physical_gpu{gpu}.lease.json'
            oldlease.write_bytes(r.base.raw_json({**old, 'status': 'released_sealed_no_retry'}))
            evidence = {'previous_progress_path': str(oldpath),
                        'previous_progress_sha256': r.base.file_sha(oldpath),
                        'preserved_old_lease_sha256': {str(oldlease): r.base.file_sha(oldlease)},
                        'pid': old['pid'], 'start_ticks': old['start_ticks'],
                        'assignment_sha256': old['assignment_sha256'], 'status': old['status'],
                        **self.modelinfo}
            progress = {'worker': r.WORKER, 'physical_gpu_index': gpu, 'pid': pid,
                        'start_ticks': ticks, 'run_id': f'originalg{gpu}', 'status': 'running',
                        'lane_source_count': 5000, 'assignment_sha256': 'c' * 64,
                        'queue_sha256': 'd' * 64, 'old_consumed_ticket': evidence}
            (self.root / 'logs' / f'rollover_gpu{gpu}_progress.json').write_bytes(r.base.raw_json(progress))
            (self.root / 'gpu_locks' / f'physical_gpu{gpu}.newepoch.lease.json').write_bytes(
                r.base.raw_json({**{k: progress[k] for k in ('worker', 'physical_gpu_index', 'pid',
                     'start_ticks', 'assignment_sha256', 'queue_sha256')},
                     'edition': r.base.EDITION, 'gpu_uuid': f'fixture-gpu{gpu}'}))
            (self.root / 'gpu_locks' / f'physical_gpu{gpu}.lock').touch()
        self.args = SimpleNamespace(root=self.root, candidate=self.candidate, worker=r.WORKER,
            gpu=0, queue_sha='d' * 64, original_queue_sha='d' * 64, assignment_sha='c' * 64,
            expected_gpu_uuid='fixture-gpu0', recovery_run_id='recovery-fixture-g0', relay_host=None,
            previous_progress=self.root / 'logs/unified_gpu0_progress.json')
        self.args.original_progress, self.args.original_seals, self.args.original_leases = r.original_state(self.args)
        self.args.previous_evidence = self.args.original_progress[0]['old_consumed_ticket']
        self.args.executable_seals, self.args.metadata_seals, self.args.retained_claims = {}, {}, {}
        self.helper = SimpleNamespace(memory_contract=lambda root: {
            'effective_headroom_bytes': 64 * r.base.GIB, 'local_disk_free_bytes': 100 * r.base.GIB})
        self.rows = [{'split': 'train', 'id': f'source{i}', 'source_ordinal': i + 1,
                      'video_path': f'/fixture/videos/source{i}.mp4',
                      'audio_path': f'/fixture/audio/source{i}.wav',
                      'meta_path': f'/fixture/meta/source{i}.json'} for i in range(20000)]
        self.ticket = {'claims_root': str(self.root / 'claims'), 'source_count': 20000,
                       'output_root': str(self.root / 'results')}

    def claim_record(self, index):
        row, p = self.rows[index], self.args.original_progress[index % 4]
        return {'edition': r.base.EDITION, 'worker': r.WORKER, 'gpu': index % 4,
                'assignment_sha256': self.args.assignment_sha, 'assignment_index': index,
                'source_row': row, 'split': row['split'], 'id': row['id'], 'pid': p['pid'],
                'owner_pid': p['pid'], 'start_ticks': p['start_ticks'], 'claimed_unix': 1,
                'automatic_retry': False, 'run_id': p['run_id']}

    def write_claim(self, index, record=None):
        row = self.rows[index]
        path = Path(self.ticket['claims_root']) / row['split'] / (row['id'] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(r.base.raw_json(record or self.claim_record(index)))
        return path

    def gate(self):
        fixture = self
        class CPUGate(r.RecoveryGate):
            def gpu(self, fresh):
                return {'gpu_uuid': fixture.args.expected_gpu_uuid, 'memory_used_mib': 0,
                        'process_pids': [], 'physical_gpu_index': fixture.args.gpu}
        return CPUGate(self.args, self.args.assignment_sha, self.helper, self.candidate / 'runs/fixture')

    def close(self):
        self.temporary.cleanup()


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.addCleanup(self.f.close)

    def test_full_scope_claim_filter_and_original_modulo_holes(self):
        for i in (0, 1, 4, 7, 19996, 19999):
            self.f.write_claim(i)
        rows, retained, counts = r.audit_claims(self.f.args, self.f.ticket, self.f.rows)
        self.assertEqual(len(retained), 6)
        self.assertEqual(counts, [3, 1, 0, 2])
        self.assertEqual(len(rows), 4997)
        self.assertEqual(rows[:2], [(8, self.f.rows[8]), (12, self.f.rows[12])])
        self.assertTrue(all(i % 4 == 0 for i, row in rows))
        self.assertNotIn(19996, [i for i, row in rows])

    def test_claim_owner_assignment_and_ticks_fail_closed(self):
        record = self.f.claim_record(0)
        for field, value in [('owner_pid', 1), ('start_ticks', 2), ('assignment_index', 4),
                             ('assignment_sha256', 'e' * 64), ('gpu', True), ('run_id', 'foreign')]:
            with self.subTest(field=field):
                with self.assertRaises(PermissionError):
                    r.validate_claim({**record, field: value}, 0, self.f.rows[0], self.f.args, {})
        wrongrow = {**record['source_row'], 'audio_path': '/other/source.wav'}
        with self.assertRaises(PermissionError):
            r.validate_claim({**record, 'source_row': wrongrow}, 0, self.f.rows[0], self.f.args, {})

    def test_malformed_claim_in_other_lane_rejects_full_scope(self):
        path = self.f.write_claim(1)
        path.write_bytes(b'{"pid":1,"pid":2}')
        with self.assertRaises(ValueError):
            r.audit_claims(self.f.args, self.f.ticket, self.f.rows)

    def test_original_progress_and_lease_pid_ticks_required(self):
        path = self.f.root / 'gpu_locks/physical_gpu2.newepoch.lease.json'
        value = r.base.strict_json(path.read_bytes())
        path.write_bytes(r.base.raw_json({**value, 'start_ticks': value['start_ticks'] + 1}))
        with self.assertRaises(PermissionError):
            r.original_state(self.f.args)

    def test_active_original_owner_rejected(self):
        with patch.object(r.base, 'pid_alive', return_value=True):
            with self.assertRaises(PermissionError):
                r.original_state(self.f.args)

    def test_original_lease_hash_and_inode_change_rejected(self):
        path = self.f.root / 'gpu_locks/physical_gpu0.lease.json'
        raw = path.read_bytes()
        path.write_bytes(raw + b' ')
        with self.assertRaises(PermissionError):
            r.assert_original_state(self.f.args)
        path.write_bytes(raw)
        replacement = path.with_name('replacement')
        replacement.write_bytes(raw)
        os.replace(replacement, path)
        with self.assertRaises(PermissionError):
            r.assert_original_state(self.f.args)

    def test_foreign_original_lease_and_lock_inode_rejected(self):
        lock = self.f.root / 'gpu_locks/physical_gpu0.lock'
        replacement = lock.with_name('replacement.lock')
        replacement.touch()
        os.replace(replacement, lock)
        with self.assertRaises(PermissionError):
            r.assert_original_state(self.f.args)
        foreign = self.f.root / 'gpu_locks/physical_gpu1.foreign.lease.json'
        foreign.write_bytes(b'{}')
        with self.assertRaises(PermissionError):
            r.original_state(self.f.args)

    def test_queue_bytes_and_inode_change_rejected(self):
        queue = self.f.root / 'queue.json'
        queue.write_bytes(b'{"tickets":[]}\n')
        self.f.args.metadata_seals[str(queue)] = r.stable_read(queue)[1]
        queue.write_bytes(b'{"tickets":[1]}\n')
        with self.assertRaises(PermissionError):
            r.assert_original_state(self.f.args)

    def test_recovery_gate_preserves_originals_and_has_one_exclusive_start(self):
        originals = {p: r.require_seal(p, s) for p, s in self.f.args.original_seals.items() if not p.endswith('.lock')}
        with patch.object(r.base, 'verify_models', return_value=self.f.modelinfo):
            with self.f.gate() as gate:
                self.assertTrue(gate.created)
                self.assertTrue(gate.token.exists())
                with self.assertRaises(FileExistsError):
                    self.f.gate().__enter__()
                gate.check()
            with self.assertRaises(FileExistsError):
                self.f.gate().__enter__()
        self.assertTrue(all(Path(p).read_bytes() == raw for p, raw in originals.items()))

    def test_two_concurrent_start_attempts_exactly_one_reserved(self):
        barrier = threading.Barrier(2)
        outcomes = []
        def attempt():
            gate = self.f.gate()
            barrier.wait()
            try:
                gate.__enter__()
                outcomes.append('started')
                gate.__exit__(None, None, None)
            except FileExistsError:
                outcomes.append('rejected')
        with patch.object(r.base, 'verify_models', return_value=self.f.modelinfo):
            threads = [threading.Thread(target=attempt) for _ in range(2)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
        self.assertCountEqual(outcomes, ['started', 'rejected'])

    def test_failed_admission_keeps_exclusive_start_and_does_not_make_claim(self):
        gate = self.f.gate()
        with patch.object(gate, 'gpu', side_effect=PermissionError('fresh GPU occupied')):
            with self.assertRaises(PermissionError):
                gate.__enter__()
        self.assertTrue(gate.token.exists())
        self.assertFalse(gate.lease.exists())
        self.assertIsNone(gate.fd)
        with self.assertRaises(FileExistsError):
            self.f.gate().__enter__()
        self.assertFalse(Path(self.f.ticket['claims_root']).exists())

    def test_recovery_lease_inode_or_hash_change_never_overwritten(self):
        with patch.object(r.base, 'verify_models', return_value=self.f.modelinfo):
            gate = self.f.gate().__enter__()
        raw = gate.lease.read_bytes()
        replacement = gate.lease.with_name('replacement')
        replacement.write_bytes(raw)
        os.replace(replacement, gate.lease)
        with self.assertRaises(PermissionError): gate.check()
        with self.assertRaises(PermissionError): gate.__exit__(None, None, None)
        self.assertEqual(gate.lease.read_bytes(), raw)
        self.assertIsNone(gate.fd)

    def test_recovery_claim_owner_requires_registered_token_and_lease(self):
        with patch.object(r.base, 'verify_models', return_value=self.f.modelinfo):
            with self.f.gate() as gate:
                owner = r.recovery_owner(self.f.args, 0)
                record = self.f.claim_record(0)
                record.update(pid=owner['pid'], owner_pid=owner['pid'], start_ticks=owner['start_ticks'],
                              run_id=owner['run_id'])
                r.validate_claim(record, 0, self.f.rows[0], self.f.args, {0: owner})
                with self.assertRaises(PermissionError):
                    r.validate_claim(record, 0, self.f.rows[0], self.f.args, {})

    def test_fresh_gpu_requires_zero_memory_no_pids_matching_uuid(self):
        args = SimpleNamespace(gpu=0)
        gate = r.base.Gate(SimpleNamespace(root=self.f.root, gpu=0, worker=r.WORKER, queue_sha='d' * 64),
                           'c' * 64, self.f.helper, self.f.root)
        def query(memory, pids, uuid='fixture-gpu0'):
            xml = '<nvidia_smi_log><gpu><uuid>' + uuid + '</uuid><fb_memory_usage><used>' + str(memory) + ' MiB</used></fb_memory_usage><processes>'
            xml += ''.join('<process_info><pid>' + str(p) + '</pid></process_info>' for p in pids)
            xml += '</processes></gpu></nvidia_smi_log>'
            return [SimpleNamespace(stdout=xml), SimpleNamespace(stdout='0, fixture-gpu0')]
        for memory, pids in ((1, []), (5, []), (0, [os.getpid()]), (0, [900000000])):
            with patch.object(r.base.subprocess, 'run', side_effect=query(memory, pids)):
                with self.assertRaises(PermissionError): gate.gpu_once(True)
        with patch.object(r.base.subprocess, 'run', side_effect=query(0, [])):
            self.assertEqual(gate.gpu_once(True)['memory_used_mib'], 0)
        with patch.object(r.base.subprocess, 'run', side_effect=query(0, [], 'other-gpu')):
            with self.assertRaises(PermissionError): gate.gpu_once(True)

    def test_budget_cannot_drop_below_frozen_32gib_and_four_cache_reserve(self):
        result = r.resource_budget(self.f.args, self.f.helper, self.f.candidate)
        self.assertEqual(result['required_effective_headroom_bytes'], 32 * r.base.GIB)
        self.assertEqual(result['required_disk_free_bytes'], 40 * r.base.GIB)
        for memory, disk in ((31 * r.base.GIB, 100 * r.base.GIB), (64 * r.base.GIB, 39 * r.base.GIB)):
            helper = SimpleNamespace(memory_contract=lambda root: {
                'effective_headroom_bytes': memory, 'local_disk_free_bytes': disk})
            with self.assertRaises(PermissionError): r.resource_budget(self.f.args, helper, self.f.candidate)

    def test_lane_uses_only_unclaimed_original_indices_and_new_progress(self):
        self.f.args.unclaimed = [(8, self.f.rows[8]), (20, self.f.rows[20])]
        self.f.args.claim_counts = [4998, 0, 0, 0]
        self.f.args.queue, self.f.args.original_queue = str(self.f.root / 'queue.json'), {}
        lane = r.RecoveryLane(self.f.args, SimpleNamespace(), self.f.ticket, self.f.rows,
                              'c' * 64, {}, self.f.helper)
        self.assertEqual([i for i, row in lane.rows], [8, 20])
        self.assertTrue(lane.work.is_relative_to(self.f.candidate))
        self.assertTrue(lane.progress_path.is_relative_to(self.f.candidate))
        self.assertNotEqual(lane.progress_path, self.f.root / 'logs/rollover_gpu0_progress.json')
        self.assertEqual(lane.progress['counts_scope'], 'this_recovery_increment_only_not_total80k')
        self.assertTrue(r.transport.SAFE.fullmatch(lane.run_id))
        self.assertIs(r.RecoveryLane.enqueue, r.base.Lane.enqueue)
        self.assertIs(r.RecoveryLane.process, r.base.Lane.process)
        self.assertIs(r.RecoveryLane.publish, r.base.Lane.publish)

    def test_frozen_enqueue_competing_claim_has_no_copy_or_inference(self):
        self.f.args.unclaimed, self.f.args.claim_counts = [(8, self.f.rows[8])], [0] * 4
        self.f.args.queue = str(self.f.root / 'queue.json')
        calls = []
        access = SimpleNamespace(stat=lambda path: {'bytes': 1},
                                 claim=lambda path, record: calls.append('claim') or False)
        lane = r.RecoveryLane(self.f.args, access, self.f.ticket, self.f.rows, 'c' * 64, {}, self.f.helper)
        lane.core = SimpleNamespace(copy_source=lambda *args: self.fail('copy of competing claim'))
        executor = SimpleNamespace(submit=lambda *args: self.fail('inference/copy after failed claim'))
        self.assertTrue(lane.enqueue(1, executor))
        self.assertEqual(calls, ['claim'])
        self.assertEqual(len(lane.pending), 0)
        self.assertEqual(lane.progress['unique_processed'], 0)
        self.assertEqual(lane.receipts[0]['status'], 'existing_claim_not_reassigned')

    def test_actual_frozen_transport_xb_claim_never_overwrites_fixture_claim(self):
        common, old = self.f.root / 'common', self.f.root / 'old'
        assignments = common / 'assignments'
        assignments.mkdir(parents=True)
        ticket = {'schema': 'ms_external_unified_assignment_v1', 'worker_id': r.WORKER,
                  'edition': common.name, 'source_count': 20000,
                  'assignment_path': str(assignments / (r.WORKER + '.jsonl')),
                  'claims_root': str(common / 'claims'), 'output_root': str(common / 'results' / r.WORKER),
                  'code_refs': [{'path': '/fixture/core.py', 'sha256': 'a' * 64}]}
        mraw = b''.join(r.base.raw_json(row) for row in self.f.rows)
        ticket['assignment_sha256'] = r.base.digest(mraw)
        traw = r.base.raw_json(ticket)
        (assignments / (r.WORKER + '.jsonl')).write_bytes(mraw)
        (assignments / (r.WORKER + '.ticket.json')).write_bytes(traw)
        queue = {'schema': 'ms_external_authorized_rollover_queue_v1', 'worker_id': r.WORKER,
                 'automatic_rollover_within_preallocated_tickets': True,
                 'no_automatic_retry_sealed_previous_sources': True, 'no_restart_active_previous_job': True,
                 'run_until_preallocated_queue_exhausted': True,
                 'tickets': [{'path': str(old / 'assignments' / (r.WORKER + '.ticket.json')), 'sha256': 'b' * 64},
                             {'path': str(assignments / (r.WORKER + '.ticket.json')), 'sha256': r.base.digest(traw)}]}
        (common / (r.WORKER + '.queue.json')).write_bytes(r.base.raw_json(queue))
        with patch.object(r.transport, 'P', common), patch.object(r.transport, 'PREVIOUS_P', old):
            access = r.transport.Access(r.WORKER)
            access.bind(ticket, self.f.rows, r.base.digest(traw), r.base.digest(mraw))
            row = self.f.rows[0]
            path = common / 'claims' / row['split'] / (row['id'] + '.json')
            record = {**self.f.claim_record(0), 'edition': common.name, 'assignment_sha256': ticket['assignment_sha256']}
            self.assertTrue(access.claim(path, record))
            seal = r.stable_read(path)[1]
            self.assertFalse(access.claim(path, {**record, 'pid': os.getpid()}))
            r.require_seal(path, seal)

    def test_all84k_allocation_disjoint_ids_media_old4000_and_other_workers(self):
        old, common = self.f.root / 'alloc-old', self.f.root / 'alloc-current'
        own = None
        for root, count, prefix in ((old, 1000, 'old'), (common, 20000, 'current')):
            (root / 'assignments').mkdir(parents=True)
            for worker in r.transport.WORKERS:
                rows = [{'split': 'train', 'id': prefix + worker + str(i), 'source_ordinal': i + 1,
                         'video_path': '/fixture/videos/' + prefix + worker + str(i) + '.mp4',
                         'audio_path': '/fixture/audio/' + prefix + worker + str(i) + '.wav',
                         'meta_path': '/fixture/meta/' + prefix + worker + str(i) + '.json'} for i in range(count)]
                traw = r.base.raw_json({'worker_id': worker, 'source_count': count})
                (root / 'assignments' / (worker + '.ticket.json')).write_bytes(traw)
                (root / 'assignments' / (worker + '.jsonl')).write_bytes(
                    b''.join(r.base.raw_json(row) for row in rows))
                if root == common:
                    (root / (worker + '.queue.json')).write_bytes(b'{}\n')
                    if worker == r.WORKER: own = rows
        class FakeMetadataAccess:
            def __init__(self, worker): self.worker = worker
            def bind(self, ticket, rows, tsha, msha):
                if ticket['worker_id'] != self.worker or len(rows) != ticket['source_count']:
                    raise PermissionError('fixture allocation identity changed')
        with patch.object(r.base, 'OLD_COMMON', old), patch.object(r.base, 'COMMON', common), \
                patch.object(r.old_transport, 'Access', FakeMetadataAccess), \
                patch.object(r.transport, 'Access', FakeMetadataAccess), \
                patch.object(r.transport, 'validate_queue', return_value={}):
            self.assertEqual(len(r.verify_disjoint_allocations(self.f.args, own)), 20)
            path = old / 'assignments' / (r.WORKER + '.jsonl')
            raw = path.read_bytes()
            original = [r.base.strict_json(line) for line in raw.split(b'\n')[:-1]]
            changed = [own[0], *original[1:]]
            path.write_bytes(b''.join(r.base.raw_json(row) for row in changed))
            with self.assertRaises(PermissionError): r.verify_disjoint_allocations(self.f.args, own)
            path.write_bytes(raw)
            path = common / 'assignments/a100-extra-04.jsonl'
            original = [r.base.strict_json(line) for line in path.read_bytes().split(b'\n')[:-1]]
            original[0]['video_path'] = own[0]['video_path']
            path.write_bytes(b''.join(r.base.raw_json(row) for row in original))
            with self.assertRaises(PermissionError): r.verify_disjoint_allocations(self.f.args, own)

    def test_real_cli_launch_blocked_before_preflight_or_production_tokens(self):
        with patch.object(r, 'preflight', side_effect=AssertionError('production preflight forbidden')):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    r.main(['--gpu', '0'])
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse((r.CANDIDATE / 'starts').exists())
        self.assertFalse((r.CANDIDATE / 'leases').exists())

    def test_check_only_main_never_claims_locks_runs_or_loads_models(self):
        args = self.f.args
        args.claim_counts, args.unclaimed = [0] * 4, [(0, self.f.rows[0])]
        resource = {'memory_used_mib': 0, 'required_effective_headroom_bytes': 32 * r.base.GIB,
                    'required_disk_free_bytes': 40 * r.base.GIB}
        with patch.object(r, 'fixed_args', return_value=args), patch.object(r, 'preflight',
                return_value=(None, self.f.ticket, self.f.rows, 'c' * 64, {}, self.f.helper, resource)), \
                patch.object(r.RecoveryLane, 'run', side_effect=AssertionError('real run forbidden')), \
                patch.object(r.RecoveryGate, '__enter__', side_effect=AssertionError('lock forbidden')), \
                patch.object(r.transport.Access, 'claim', side_effect=AssertionError('claim forbidden')):
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(r.main(['--gpu', '0', '--check-only']), 0)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'verified_no_models_claims_locks')


if __name__ == '__main__':
    unittest.main(verbosity=2)
