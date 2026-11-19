"""CPU regressions using checked real repair sources; never call the VLM."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parent
ROOT = RT.parent
OUT = ROOT / 'curated222939_repair_v1'
REV = 'repair-20261025-v1'
KEY = 'train/YHdOyRCqljUFfUJV'
FAILED_KEY = 'train/RPItfOOsPbLXBhXC'
sys.dont_write_bytecode = True
sys.path.insert(0, str(RT / 'source_curated_consumer_v1'))
import curated_pipeline as pipeline
import ms_bar_annotation as bar


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


q = module('test_repair_q', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
w = module('test_repair_watch', RT / 'curated_repair_watch_v2.py')


class RepairRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.original_job = q.load(OUT / 'jobs' / (KEY + '.json'))
        self.original_repair = q.load(OUT / 'revisions' / REV / KEY / 'result.json')

    def fixture(self, failed=False, final=None):
        key = FAILED_KEY if failed else KEY
        job = copy.deepcopy(q.load(OUT / 'jobs' / (key + '.json')))
        job['queue_root'] = str(self.out)
        path = self.out / 'jobs' / (key + '.json')
        q.save_once(path, job)
        ref = q.identity(path)
        selected = w.dispatch_job(job, ref)
        dispatch = self.out / 'dispatch-jobs' / (key + '.json')
        q.save_once(dispatch, selected)
        if failed:
            visual = q.checked(job['refs']['visual_result'])
            repair = {k: copy.deepcopy(visual[k]) for k in
                      ('window_observations', 'facts', 'boundary_observations', 'final')}
            repair.update(id=job['id'], global_revision=REV,
                          original_result_ref=job['original_result_ref'],
                          original_source_binding=job['source_binding'], all_windows_nonempty=False)
        else:
            repair = copy.deepcopy(self.original_repair)
        if final is not None:
            repair['final'] = copy.deepcopy(final)
        repair['original_job_ref'] = q.identity(dispatch)
        result = self.out / 'revisions' / REV / key / 'result.json'
        q.save_once(result, repair)
        budget = q.load(OUT / 'attempt-budget' / (KEY + '.json'))
        budget.update(source_key=key, original_result_sha256=job['original_result_sha256'])
        q.save_once(self.out / 'attempt-budget' / (key + '.json'), budget)
        q.append(self.out / 'ready.jsonl', dict(source_key=key, job_ref=ref, global_revision=REV))
        return key, selected, ref, repair

    def package(self, fixture):
        key, selected, ref, repair = fixture
        return w.package_joint(selected, ref, repair, self.out, REV, q, pipeline, bar)

    def test_real_success_reuses_existing_visual_bytes(self):
        fixture = self.fixture()
        ref, joint = self.package(fixture)
        visual = self.out / 'joint/revisions' / REV / KEY / 'visual_result.json'
        before = visual.read_bytes()
        again, second = self.package(fixture)
        self.assertEqual(again, ref)
        self.assertEqual(second, joint)
        self.assertEqual(visual.read_bytes(), before)
        self.assertTrue(joint['joint_complete'])
        existing = q.load(OUT / 'joint/revisions' / REV / KEY / 'visual_result.json')
        self.assertEqual(joint['label']['timed_motion_script'], existing['final_timed_caption'])
        self.assertEqual(len(joint['label']['timed_motion_script']), 747)
        self.assertFalse(joint['training_eligible'])
        self.assertEqual(q.checked(ref), joint)

    def test_real_failed_missing_json_stays_incomplete(self):
        fixture = self.fixture(failed=True)
        before = copy.deepcopy(fixture[3]['final'])
        ref, joint = self.package(fixture)
        self.assertFalse(joint['joint_complete'])
        self.assertFalse(joint['all_windows_nonempty'])
        self.assertEqual(joint['target_text'], '')
        self.assertEqual(joint['original_final'], before)
        self.assertEqual(q.checked(joint['visual_result_ref'])['final'], before)
        self.assertEqual(q.checked(joint['joint_input_ref'])['final']['json'], {})
        self.assertEqual(q.checked(ref), joint)

    def test_null_json_preserves_failed_final_and_only_normalizes_input(self):
        final = {'status': 'failed', 'json': None, 'error': 'original diagnostic',
                 'attempt_path': '/nonexecuted/regression/attempt', 'format_valid': False}
        fixture = self.fixture(failed=True, final=final)
        ref, joint = self.package(fixture)
        visual = q.checked(joint['visual_result_ref'])
        joint_input = q.checked(joint['joint_input_ref'])
        self.assertEqual(visual['final'], final)
        self.assertEqual(joint['original_final'], final)
        self.assertEqual(joint_input['original_final'], final)
        self.assertEqual(joint_input['final'], {**final, 'json': {}})
        self.assertFalse(joint['joint_complete'])
        self.assertFalse(joint['all_windows_nonempty'])
        self.assertEqual(joint['target_text'], '')
        self.assertEqual(fixture[3]['final'], final)

    def test_null_candidate_never_structurally_promoted(self):
        fixture = self.fixture(final={'status': 'candidate', 'json': None})
        ref, joint = self.package(fixture)
        self.assertFalse(joint['joint_complete'])
        self.assertFalse(joint['all_windows_nonempty'])
        self.assertEqual(joint['target_text'], '')

    def test_checked_plan_sha_and_binding_and_identity(self):
        job = self.original_job
        producer = q.checked(job['refs']['producer_terminal'])
        self.assertNotIn('manifest_record', producer)
        row = w.checked_manifest(job, producer, q)
        self.assertEqual(row['id'], job['id'])
        changed = copy.deepcopy(job)
        changed['refs']['plan_ref']['sha256'] = '0' * 64
        with self.assertRaises(q.SourceChanged):
            w.checked_manifest(changed, producer, q)
        for field in ('id', 'split', 'batch_id'):
            with self.subTest(field=field):
                changed = copy.deepcopy(producer)
                changed[field] = 'different'
                with self.assertRaises(q.SourceChanged):
                    w.checked_manifest(job, changed, q)
        changed = copy.deepcopy(producer)
        changed['source_binding']['video_sha256'] = '0' * 64
        with self.assertRaises(q.SourceChanged):
            w.checked_manifest(job, changed, q)

    def test_manifest_identity_mismatch_checked_after_rehash(self):
        producer = q.checked(self.original_job['refs']['producer_terminal'])
        for field in ('id', 'split', 'batch_id'):
            with self.subTest(field=field):
                job = copy.deepcopy(self.original_job)
                plan = q.checked(job['refs']['plan_ref'])
                plan['manifest_record'][field] = 'different'
                path = self.out / (field + '.json')
                q.save_once(path, plan)
                job['refs']['plan_ref'] = q.identity(path)
                altered_producer = {**producer, 'plan_ref': job['refs']['plan_ref']}
                with self.assertRaises(q.SourceChanged):
                    w.checked_manifest(job, altered_producer, q)

    def test_repair_revision_binding_is_strict(self):
        fixture = self.fixture()
        changed = copy.deepcopy(fixture[3])
        changed['global_revision'] = 'other-revision'
        with self.assertRaises(q.SourceChanged):
            w.package_joint(fixture[1], fixture[2], changed, self.out, REV, q, pipeline, bar)
        self.assertFalse((self.out / 'joint/revisions' / REV / KEY / 'result.json').exists())

    def test_recovery_terminal_notice_state_and_budget_idempotent(self):
        fixture = self.fixture()
        immutable = {p: p.read_bytes() for p in self.out.rglob('*.json')}
        def forbidden(*args, **kwargs):
            raise AssertionError('VLM or repair dispatch forbidden')
        with patch.object(q, 'repair_visual', forbidden):
            state, count = w.consume_once(self.out, REV, q, pipeline, bar, forbidden, None,
                                          limit=1, recover_only=True)
            again, repeated = w.consume_once(self.out, REV, q, pipeline, bar, forbidden, None,
                                             limit=1, recover_only=True)
        self.assertEqual((count, repeated), (1, 0))
        self.assertEqual(again, state)
        for path, data in immutable.items():
            self.assertEqual(path.read_bytes(), data)
        terminal = q.load(self.out / 'worker-terminals' / (KEY + '.json'))
        notice = q.load(self.out / 'worker-announced' / (KEY + '.json'))
        joint = q.checked(terminal['joint_result_ref'])
        self.assertEqual(terminal, notice)
        self.assertEqual(terminal['status'], joint['status'])
        self.assertTrue(terminal['joint_complete'])
        self.assertEqual(state['seen'][KEY], fixture[2]['sha256'])
        self.assertEqual(state['offset'], (self.out / 'ready.jsonl').stat().st_size)
        review_path = self.out / 'joint/review-required.jsonl'
        lines = review_path.read_bytes()
        self.assertEqual(len(lines.splitlines()), 1)
        # Simulate crash after append but before the immutable notice was installed.
        (self.out / 'worker-announced' / (KEY + '.json')).unlink()
        w.announce(self.out, KEY, terminal, q)
        self.assertEqual(review_path.read_bytes(), lines)
        self.assertEqual(q.load(self.out / 'worker-announced' / (KEY + '.json')), terminal)

    def test_recovery_missing_budget_never_redispatches(self):
        self.fixture()
        (self.out / 'attempt-budget' / (KEY + '.json')).unlink()
        with patch.object(q, 'repair_visual', side_effect=AssertionError('dispatch forbidden')):
            with self.assertRaises(q.SourceChanged):
                w.consume_once(self.out, REV, q, pipeline, bar, None, None, limit=1, recover_only=True)

    def test_reserved_without_result_is_terminal_not_retry(self):
        self.fixture()
        result = self.out / 'revisions' / REV / KEY / 'result.json'
        result.unlink()
        with patch.object(q, 'repair_visual', side_effect=AssertionError('dispatch forbidden')):
            state, count = w.consume_once(self.out, REV, q, pipeline, bar, None, None,
                                          limit=1, recover_only=True)
        terminal = q.load(self.out / 'worker-terminals' / (KEY + '.json'))
        self.assertEqual(terminal['status'], 'interrupted_attempt')
        self.assertNotIn('joint_result_ref', terminal)
        self.assertEqual(count, 1)

    def test_checked_original_visual_manifest_plan_and_joint_binding(self):
        fixture = self.fixture()
        visual_ref = fixture[1]['refs']['visual_result']
        joint_ref = fixture[1]['refs']['joint_result']
        real_checked = q.checked
        for field in ('manifest_record_sha256', 'full_source_row_sha256', 'plan_sha256'):
            with self.subTest(field=field):
                changed = copy.deepcopy(real_checked(visual_ref))
                changed['binding'][field] = '0' * 64
                with patch.object(q, 'checked', side_effect=lambda ref: changed if ref == visual_ref else real_checked(ref)):
                    with self.assertRaises(q.SourceChanged):
                        self.package(fixture)
        changed = copy.deepcopy(real_checked(joint_ref))
        changed['source_refs'] = {}
        with patch.object(q, 'checked', side_effect=lambda ref: changed if ref == joint_ref else real_checked(ref)):
            with self.assertRaises(q.SourceChanged):
                self.package(fixture)

    def test_failed_retained_json_never_outputs_failed_caption(self):
        final = copy.deepcopy(self.original_repair['final'])
        final['status'] = 'failed'
        final['error'] = 'validator rejected body parts'
        fixture = self.fixture(final=final)
        ref, joint = self.package(fixture)
        self.assertEqual(q.checked(joint['visual_result_ref'])['final'], final)
        self.assertEqual(joint['label']['timed_motion_script'], '')
        self.assertFalse(joint['joint_complete'])
        self.assertFalse(joint['all_windows_nonempty'])

    def cursor_fixture(self):
        import hashlib
        fixture = self.fixture()
        w.consume_once(self.out, REV, q, pipeline, bar, None, None, limit=1, recover_only=True)
        state_path = self.out / 'worker.state.json'
        state = q.load(state_path)
        state['input_identity'][0] += 1
        q.replace_state(state_path, state)
        prefix = (self.out / 'ready.jsonl').read_bytes()[:state['offset']]
        return q.identity(state_path), hashlib.sha256(prefix).hexdigest()

    def test_cursor_rebind_validates_prefix_terminal_and_preserves_budget(self):
        migration = module('test_cursor_rebind', RT / 'curated_repair_cursor_rebind_v2.py')
        state_ref, prefix_sha = self.cursor_fixture()
        budgets = {p: p.read_bytes() for p in (self.out / 'attempt-budget').rglob('*.json')}
        receipt = migration.rebind_state(self.out, REV, state_ref, prefix_sha, 'test-node')
        state = q.load(self.out / 'worker.state.json')
        st = (self.out / 'ready.jsonl').stat()
        self.assertEqual(state['input_identity'], [st.st_dev, st.st_ino])
        self.assertEqual(receipt['validated_prefix_sha256'], prefix_sha)
        self.assertEqual(q.checked(receipt['state_before_ref'])['input_identity'][0], st.st_dev + 1)
        self.assertEqual(q.checked(receipt['state_after_ref']), state)
        self.assertFalse(receipt['budget_changed'])
        for path, before in budgets.items():
            self.assertEqual(path.read_bytes(), before)
        # Seek past the migrated prefix without touching the VLM or consuming a new source.
        _, count = w.consume_once(self.out, REV, q, pipeline, bar, None, None, limit=1, recover_only=True)
        self.assertEqual(count, 0)

    def test_cursor_rebind_rejects_prefix_sha_or_terminal_or_inode_mismatch(self):
        migration = module('test_cursor_rebind_reject', RT / 'curated_repair_cursor_rebind_v2.py')
        state_ref, prefix_sha = self.cursor_fixture()
        with self.assertRaises(q.SourceChanged):
            # Different import uses the same SourceChanged semantics but a separate class.
            try:
                migration.rebind_state(self.out, REV, state_ref, '0' * 64, 'test-node')
            except ValueError as error:
                raise q.SourceChanged(str(error))
        state = q.load(state_ref['path'])
        state['input_identity'][1] += 1
        q.replace_state(Path(state_ref['path']), state)
        with self.assertRaises(ValueError):
            migration.rebind_state(self.out, REV, q.identity(state_ref['path']), prefix_sha, 'test-node')
        state['input_identity'][1] -= 1
        q.replace_state(Path(state_ref['path']), state)
        notice = self.out / 'worker-announced' / (KEY + '.json')
        changed = q.load(notice)
        changed['status'] = 'different'
        q.replace_state(notice, changed)
        with self.assertRaises(ValueError):
            migration.rebind_state(self.out, REV, q.identity(state_ref['path']), prefix_sha, 'test-node')

    def test_minimal_frozen_repair_derivation(self):
        original = (RT / 'source_repair_gpu_v2/curated_repair_queue_v1.py').read_bytes()
        derived = (RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py').read_bytes()
        expected = original.replace(
            b"for w in original.get('final', {}).get('json', {}).get('windows', [])",
            b"for w in (original.get('final', {}).get('json') or {}).get('windows', [])").replace(
            b"    plan = checked(job['refs']['plan_ref'])\n    prepared = verify_prepared_content",
            b"    plan = checked(job['refs']['plan_ref'])\n    verify_original_csv(plan['source_refs']['original_csv'], job['split'])\n    prepared = verify_prepared_content")
        # Outside the exact CSV recursion/cache region, the frozen API stays identical.
        self.assertEqual(derived.split(b'ORIGINAL_CSV_CACHE = {}')[0],
                         expected.split(b'def verify_references_by_content')[0])
        self.assertEqual(derived.split(b'def verify_prepared_content')[1],
                         expected.split(b'def verify_prepared_content')[1])


class OriginalCsvCacheRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'train.csv'
        self.path.write_bytes(b'id,video\na,video-a\n')
        self.q = module('test_csv_q', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
        self.ref = self.q.identity(self.path)

    def test_first_sha_then_recursive_registered_csv_reuses_stat_only(self):
        original_sha = self.q.sha
        calls = []
        def counted(path):
            calls.append(str(path))
            return original_sha(path)
        with patch.object(self.q, 'sha', side_effect=counted):
            self.q.verify_original_csv(self.ref, 'train')
            self.q.verify_original_csv(self.ref, 'train')
            self.q.verify_references_by_content([self.ref, {'original_csv': self.ref}])
        self.assertEqual(calls, [str(self.path)])
        self.assertEqual(len(self.q.ORIGINAL_CSV_CACHE), 1)
        unregistered = self.path.with_name('not-cached.json')
        unregistered.write_bytes(b'{}')
        ref = self.q.identity(unregistered)
        with patch.object(self.q, 'sha', wraps=self.q.sha) as sha:
            self.q.verify_references_by_content([ref, ref])
            self.assertEqual(sha.call_count, 2)

    def test_wrong_sha_before_or_after_cache_never_accepted(self):
        wrong = {**self.ref, 'sha256': '0' * 64}
        with self.assertRaises(ValueError):
            self.q.verify_original_csv(wrong, 'train')
        self.assertEqual(self.q.ORIGINAL_CSV_CACHE, {})
        self.q.verify_original_csv(self.ref, 'train')
        with self.assertRaises(ValueError):
            self.q.verify_references_by_content(wrong)
        with self.assertRaises(ValueError):
            self.q.verify_original_csv(self.ref, 'val')

    def test_stat_change_rejected_without_rehash_and_first_read_change_rejected(self):
        self.q.verify_original_csv(self.ref, 'train')
        self.path.write_bytes(b'id,video\nb,video-b\n')
        with patch.object(self.q, 'sha', side_effect=AssertionError('changed stat must reject before hash')):
            with self.assertRaises(ValueError):
                self.q.verify_original_csv(self.ref, 'train')
        self.q.ORIGINAL_CSV_CACHE.clear()
        self.ref = self.q.identity(self.path)
        first = self.q.original_csv_stat(self.path)
        changed = (*first[:-1], first[-1] + 1)
        with patch.object(self.q, 'original_csv_stat', side_effect=[first, changed]):
            with self.assertRaises(ValueError):
                self.q.verify_original_csv(self.ref, 'train')
        self.assertEqual(self.q.ORIGINAL_CSV_CACHE, {})


class RepairManagerRegression(unittest.TestCase):
    def setUp(self):
        self.manager = module('test_safe_manager_v2', RT / 'curated_repair_resource_manager_v2.py')
        self.config = q.load(RT / 'curated_repair_resident_n6_g01_v2.json')

    def validate(self, cfg):
        with patch.object(self.manager, 'original_validate'), patch.object(
                self.manager.subprocess, 'check_output', return_value='28.216.19.225'):
            self.manager.validate(cfg)

    def test_nanosecond_snapshot_integer_protocol_and_immutable_paths(self):
        self.validate(self.config)
        cfg = copy.deepcopy(self.config)
        cfg['lease_snapshot_ns']['/tmp/od_reserve_0']['st_mtime_ns'] = float(
            cfg['lease_snapshot_ns']['/tmp/od_reserve_0']['st_mtime_ns'])
        with self.assertRaises(RuntimeError):
            self.validate(cfg)
        cfg = copy.deepcopy(self.config)
        cfg['producer']['path'] = str(RT / 'curated_repair_watch_v1.py')
        with self.assertRaises(RuntimeError):
            self.validate(cfg)

    def test_snapshot_full_stat_equality_and_no_missing_lease_fallback(self):
        from types import SimpleNamespace
        self.validate(self.config)
        values = self.config['lease_snapshot_ns']['/tmp/od_reserve_0']
        old = SimpleNamespace(**values)
        with patch.object(self.manager, 'original_lease_snapshot', return_value=old):
            self.assertIs(self.manager.checked_lease_snapshot(Path('/tmp/od_reserve_0')), old)
            self.manager.lease_expectations['/tmp/od_reserve_0']['st_mtime_ns'] += 1
            with self.assertRaises(RuntimeError):
                self.manager.checked_lease_snapshot(Path('/tmp/od_reserve_0'))
        self.manager.lease_expectations = None
        with patch.object(self.manager, 'original_lease_snapshot', return_value=None):
            with self.assertRaises(RuntimeError):
                self.manager.checked_lease_snapshot(Path('/tmp/od_reserve_0'))

    def test_summary_only_accepts_full_drained_v2_same_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            value = dict(schema='curated_repair_watch_v2', global_revision=REV,
                         queue_drained=True, source_count=self.config['targetcount'])
            q.save_once(out / 'resident.summary.json', value)
            self.assertTrue(self.manager.summaries(self.config, out))
            for update in [dict(queue_drained=False), dict(schema='curated_repair_watch_v1'),
                           dict(global_revision='different'), dict(source_count=self.config['targetcount'] + 1)]:
                q.replace_state(out / 'resident.summary.json', {**value, **update})
                with self.assertRaises(RuntimeError):
                    self.manager.summaries(self.config, out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
