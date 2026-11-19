"""CPU-only synthetic regression suite; dependencies are copied to /tmp once."""
import atexit
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
SANDBOX = tempfile.TemporaryDirectory(prefix='caption-pool-cpu-', dir='/tmp')
atexit.register(SANDBOX.cleanup)
RT = Path(SANDBOX.name) / 'runtime'
RT.mkdir()
for name in ('caption_ready_pool_v1', 'caption_pool_live_manager_v1', 'caption_local_live_manager_v1',
             'caption_local_source_bundle_v1', 'caption_two_stage_light_v1', 'caption_direct_facts_v1',
             'visual_batch_fixed3_benchmark_v1', 'bench_manager_driver_v2', 'curated_resource_manager', 'resource_manager'):
    shutil.copyfile(SOURCE / (name + '.py'), RT / (name + '.py'))
sys.path.insert(0, str(RT))
import caption_ready_pool_v1 as p
import caption_pool_live_manager_v1 as m

CODE_REFS = {str(path): p.sha(path) for path in RT.glob('*.py')}


class FakeLLM:
    def __init__(self):
        self.loads = self.calls = 0
        self.roots = []

    def factory(self, cfg):
        self.loads += 1
        return lambda: (self, object)

    def generate(self, entries, args, runtime, out, admission):
        assert admission(args, runtime) is True
        assert args.batch_size == args.max_num_seqs == 8 and args.visual_max_tokens == 1800 and not args.retry_failed
        assert runtime()[0] is self
        self.calls += 1
        self.roots.append(Path(out).parent)
        drafts = []
        for entry in entries:
            row = entry['row']
            draft = dict(id=row['id'], split=row['manifest_record']['split'], status='draft_generated', **p.FLAGS)
            p.exclusive(Path(out) / 'drafts' / draft['split'] / draft['id'] / 'draft.json', draft)
            drafts.append(draft)
        return dict(drafts=drafts, new_model_calls=1)


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='assets-', dir=SANDBOX.name)
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.addCleanup(patch.stopall)
        # writeback needs only its unchanged portable batch gate, not the media/native AST package.
        patch.object(m.local.light, 'bar_module', return_value=types.SimpleNamespace(portable=lambda x: bool(x))).start()

    def fixture(self, count=6, ready_count=None):
        rows = [dict(id='source%03d' % i, split='train', synthetic=True) for i in range(count)]
        source = self.root / 'producer'
        p.mkdir(source / 'ready')
        plan, handoff, manifest = (self.root / name for name in ('plan.json', 'handoff.json', 'manifest.json'))
        p.exclusive(plan, rows)
        p.exclusive(handoff, dict(remaining_sources=rows, old_worker_tree_absent=True, original_config_resume_forbidden=True))
        model = self.root / 'modelref.json'
        p.exclusive(model, dict(synthetic_model=True))
        prior = self.root / 'safe'
        p.exclusive(prior / 'resources_safe.json', dict(all_holders_safe=True, locks_released=True, leases_restored=True))
        cfgs = []
        for worker, node, gpus in [('workerA', 'N4', [0, 1]), ('workerB', 'N6', [2, 3])]:
            holders = []
            for gpu in gpus:
                holder = dict(pid=1000 + gpu, uid=os.getuid(), argv=['python', str(m.base.HOLDER)], start_ticks=11 + gpu,
                    cgroup='synthetic', CVD=['CUDA_VISIBLE_DEVICES=' + str(gpu)], exe='/fake/python',
                    sig_ign=0, sig_blk=0, gpu=gpu, uuid='uuid-' + str(gpu))
                holders.append(holder)
                p.exclusive(prior / ('holder_restored_gpu%d.json' % gpu), dict(uuid=holder['uuid'], holder_identity=holder))
            config_path = self.root / (worker + '.json')
            draft_out = RT.parent / ('caption_two_stage_local_cpu_' + self.root.name + '_' + worker)
            cfg = dict(config_path=str(config_path), phase=worker, pool_worker_id=worker, node=node, gpus=gpus,
                release_authorized=True, env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus))), holders=holders,
                plan=p.ref(plan), input_count=count, targetcount=count, source_handoff=p.ref(handoff),
                source_root=str(source), pool_root=str(self.root / 'pool'), pool_manifest=dict(path=str(manifest), sha256='pending'),
                producer=p.ref(m.SELF), resource_manager_ref=p.ref(m.SELF), source_sha256=dict(CODE_REFS),
                resource_previous_safe=p.ref(prior / 'resources_safe.json'), draft_out=str(draft_out),
                out=str(self.root / (worker + '-out')), queue=str(self.root / (worker + '-queue')),
                modelref=dict(p.ref(model), model_identity_sha256='synthetic-model-identity'),
                argv=['/opt/venv_vllm/bin/python', '-B', str(m.SELF), 'worker', '--config', str(config_path)])
            cfgs.append(cfg)
        value = dict(schema='caption_ready_pool_v1', plan=p.ref(plan), source_handoff=p.ref(handoff),
            source_root=str(source), pool_root=str(self.root / 'pool'), participants=[dict(worker_id=c['pool_worker_id'],
            config_path=c['config_path'], node=c['node'], gpus=c['gpus'], config_identity_sha256=p.config_identity(c)) for c in cfgs])
        p.exclusive(manifest, value)
        for cfg in cfgs:
            cfg['pool_manifest'] = p.ref(manifest)
            cfg['source_sha256'][str(manifest)] = p.sha(manifest)
            p.exclusive(cfg['config_path'], cfg)
        for i, row in enumerate(rows):
            directory = source / 'sources' / row['split'] / row['id']
            media = self.root / ('media%03d.bin' % i)
            media.write_bytes(b'synthetic-media-bytes')
            original = dict(id=row['id'], manifest_record=dict(id=row['id'], split=row['split'], video_path=str(media)),
                            binding=dict(synthetic=True))
            p.exclusive(directory / 'bar_plan.json', original)
            p.exclusive(directory / 'actual_media.json', dict(synthetic=True))
            crop = self.root / ('crop%03d.bin' % i)
            crop.write_bytes(b'synthetic-roi-bytes')
            p.exclusive(source / 'roi' / row['id'] / 'roi_manifest.json', dict(id=row['id'], crops=[dict(path=str(crop))]))
        self.ready(rows[:count if ready_count is None else ready_count], source)
        return cfgs, rows

    def ready(self, rows, source):
        for i, row in enumerate(rows):
            with (source / 'ready' / ('producer-%04d.jsonl' % (i % 4))).open('ab') as stream:
                stream.write(p.encoded(dict(row, status='ready')))

    def proof(self, cfg, before=False):
        return dict(admitted=True, parent_identity=dict(pid=99, synthetic=True), child_identity=dict(pid=os.getpid(),
            uid=os.getuid(), argv=cfg['argv'], start_ticks=42, cgroup='synthetic', CVD=[cfg['env']['CUDA_VISIBLE_DEVICES']], exe=sys.executable))

    def pool(self, cfg, admit=True):
        pool = p.Pool(cfg)
        if admit:
            pool.admit(lambda before: self.proof(cfg, before))
        return pool

    def commit(self, pool, job, status='draft_generated'):
        root = Path(tempfile.mkdtemp(prefix=job['batch_id'] + '-', dir=self.root))
        for meta in job['metas']:
            p.exclusive(root / 'caption_two_stage_results/drafts' / meta['split'] / meta['id'] / 'draft.json',
                        dict(id=meta['id'], split=meta['split'], status=status, **p.FLAGS))
        commit = m.local.writeback(root, pool.cfg['draft_out'], job['batch_id'])
        return root, commit, m.archived_drafts(job, commit)

    def test_two_workers_compete_one_budget_and_one_model_call(self):
        cfgs, rows = self.fixture(count=1)
        pools = [self.pool(c) for c in cfgs]
        barrier = threading.Barrier(2)
        def reserve(pool):
            barrier.wait()
            return pool.reserve()
        with ThreadPoolExecutor(max_workers=2) as threads:
            jobs = list(threads.map(reserve, pools))
        self.assertEqual(sum(job is not None for job in jobs), 1)
        self.assertEqual(len(list((pools[0].root / 'sources').glob('*.json'))), 1)
        winning = jobs.index(next(job for job in jobs if job))
        loser = 1 - winning
        fake = FakeLLM()
        job = jobs[winning]
        root = Path(tempfile.mkdtemp(prefix=job['batch_id'] + '-', dir=self.root))
        entries = [m.local.stage_source(cfgs[winning]['source_root'], meta, root)[0] for meta in job['metas']]
        runtime = fake.factory(cfgs[winning])
        fake.generate(entries, m.args_for(cfgs[winning]), runtime, root / 'caption_two_stage_results',
                      lambda a, r: pools[winning].ownership(job))
        commit = m.local.writeback(root, cfgs[winning]['draft_out'], job['batch_id'])
        pools[winning].publish(job, commit, m.archived_drafts(job, commit))
        self.assertEqual((fake.loads, fake.calls), (1, 1))
        summary = m.run(pools[loser], lambda before=False: self.proof(cfgs[loser], before), fake.factory,
                        generate=fake.generate, poll_seconds=0, local_parent=str(self.root))
        self.assertEqual(summary['processed'], 0)
        self.assertEqual(summary['global_terminal'], 1)
        self.assertEqual((fake.loads, fake.calls), (1, 1))

    def test_source_to_batch_crash_permanently_sealed_zero_calls(self):
        cfgs, rows = self.fixture(count=1)
        first = self.pool(cfgs[0])
        def crash(value):
            raise RuntimeError('source->batch crash')
        with self.assertRaises(RuntimeError):
            first.reserve(after_source=crash)
        other = self.pool(cfgs[1])
        self.assertIsNone(other.reserve())
        worker, progress = other.progress()
        self.assertEqual((progress['reserved_uncommitted'], progress['global_terminal']), (1, 0))
        self.assertFalse(progress['complete'])
        self.assertFalse(list((first.root / 'batches').glob('*/claim.json')))
        self.assertEqual(FakeLLM().calls, 0)

    def test_partial_sourceclaim_is_sealed(self):
        cfgs, rows = self.fixture(count=1)
        pool = self.pool(cfgs[0])
        pool.source_path(p.source_key(rows[0])).write_bytes(b'{partial')
        other = self.pool(cfgs[1])
        self.assertIsNone(other.reserve())
        self.assertEqual(other.progress()[1]['reserved_uncommitted'], 1)

    def test_ready_record_only_rereads_on_invalid_json(self):
        path = self.root / 'ready.jsonl'
        value = dict(id='source000', split='train')
        data = p.encoded(value)
        path.write_bytes(data)
        with patch.object(Path, 'open', wraps=path.open) as opening:
            self.assertEqual(p.ready_record(path, data, 0), (value, len(data)))
            opening.assert_not_called()
        self.assertEqual(p.ready_record(path, b'{invalid}' + bytes([10]), 0), (value, len(data)))
        path.write_bytes(b'{invalid}' + bytes([10]))
        with self.assertRaises(ValueError):
            p.ready_record(path, b'{invalid}' + bytes([10]), 0)
        path.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            p.ready_record(path, b'{invalid}' + bytes([10]), 0)

    def test_claimed_failed_never_reassigned(self):
        cfgs, rows = self.fixture(count=3)
        first, second = [self.pool(c) for c in cfgs]
        job = first.reserve()
        first.fail(job, RuntimeError('model failed'), '/tmp/owned-cache')
        self.assertIsNone(second.reserve())
        worker, progress = first.progress()
        self.assertEqual(worker['failed'], 3)
        self.assertEqual(progress['reserved_uncommitted'], 3)
        self.assertFalse(progress['complete'])

    def test_commit_before_progress_recovered_without_model_or_copy(self):
        cfgs, rows = self.fixture(count=3)
        first = self.pool(cfgs[0])
        job = first.reserve()
        root, commit, drafts = self.commit(first, job)
        # Crash immediately after durable tar receipt; no terminal/progress yet.
        self.assertFalse((Path(cfgs[0]['draft_out']) / 'progress.json').exists())
        second = self.pool(cfgs[1])
        m.recover_commits(second)
        self.assertEqual(second.progress()[1]['global_terminal'], 3)
        fake = FakeLLM()
        summary = m.run(second, lambda before=False: self.proof(cfgs[1], before), fake.factory,
                        generate=fake.generate, poll_seconds=0, local_parent=str(self.root))
        self.assertEqual((fake.loads, fake.calls), (0, 0))
        self.assertEqual(summary['processed'], 0)
        self.assertEqual(first.progress()[0]['committed'], 3)
        self.assertTrue(root.exists())  # Recovery never cleans another worker's cache.
        result = m.summaries(cfgs[1], Path(cfgs[1]['out']))
        self.assertEqual(next(iter(result.values()))['worker_processed'], 0)

    def test_copy_next_overlaps_current_model_with_event(self):
        cfgs, rows = self.fixture(count=6)
        pool = self.pool(cfgs[0])
        copied_next, model_current = threading.Event(), threading.Event()
        fake = FakeLLM()
        first_root = []
        def stage(source, meta, root):
            if not first_root:
                first_root.append(root)
            if root != first_root[0]:
                self.assertTrue(model_current.wait(3))
                copied_next.set()
            return m.local.stage_source(source, meta, root)
        def generate(entries, args, runtime, out, admission):
            if fake.calls == 0:
                model_current.set()
                self.assertTrue(copied_next.wait(3), 'single copy thread did not overlap GPU processing')
            return fake.generate(entries, args, runtime, out, admission)
        summary = m.run(pool, lambda before=False: self.proof(cfgs[0], before), fake.factory, stage_source=stage,
                        generate=generate, poll_seconds=0, local_parent=str(self.root))
        self.assertEqual((fake.loads, fake.calls), (1, 2))
        self.assertEqual(summary['processed'], 6)
        self.assertTrue(summary['UTC'].endswith('Z'))
        self.assertEqual(summary['failed'], 0)
        self.assertTrue(all(not root.exists() for root in fake.roots))
        self.assertEqual(len(list((pool.root / 'sources').glob('*.json'))), 6)

    def test_writeback_failure_preserves_cache_and_claims(self):
        cfgs, rows = self.fixture(count=3)
        first = self.pool(cfgs[0])
        fake = FakeLLM()
        def broken_writeback(*args):
            raise OSError('writeback failed')
        with self.assertRaises(OSError):
            m.run(first, lambda before=False: self.proof(cfgs[0], before), fake.factory, generate=fake.generate,
                  writeback=broken_writeback, poll_seconds=0, local_parent=str(self.root))
        self.assertTrue(fake.roots[0].exists())
        second = self.pool(cfgs[1])
        self.assertIsNone(second.reserve())
        worker, progress = first.progress()
        self.assertEqual((fake.calls, worker['failed'], progress['global_terminal']), (1, 3, 0))
        self.assertFalse(progress['complete'])

    def test_source_id_split_mismatch_commits_failure_without_model(self):
        cfgs, rows = self.fixture(count=1)
        cfg = cfgs[0]
        path = Path(cfg['source_root']) / 'sources/train/source000/bar_plan.json'
        original = p.load(path)
        original['manifest_record']['split'] = 'val'
        p.atomic(path, original)
        pool = self.pool(cfg)
        fake = FakeLLM()
        summary = m.run(pool, lambda before=False: self.proof(cfg, before), fake.factory,
                        generate=fake.generate, poll_seconds=0, local_parent=str(self.root))
        self.assertEqual(fake.calls, 0)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['global_terminal'], 1)
        other = self.pool(cfgs[1])
        self.assertIsNone(other.reserve())

    def test_default_denied_nonparticipant_and_id_split_rejected(self):
        cfgs, rows = self.fixture(count=1)
        pool = self.pool(cfgs[0], admit=False)
        with self.assertRaises(ValueError):
            pool.reserve()
        with self.assertRaises(ValueError):
            pool.admit(lambda before: dict(admitted=False))
        bad = dict(cfgs[0], pool_worker_id='outsider')
        with self.assertRaises(ValueError):
            p.Pool(bad)
        for row in (dict(id='../escape', split='train'), dict(id='valid', split='outside')):
            with self.assertRaises(ValueError):
                p.source_key(row)
        bad = copy.deepcopy(cfgs[0])
        bad['gpus'] = [4, 5]
        with self.assertRaises(ValueError):
            p.check_manifest(bad)

    def test_unready_conservation_and_four_incremental_prefixes(self):
        cfgs, rows = self.fixture(count=5, ready_count=0)
        pool = self.pool(cfgs[0])
        self.assertIsNone(pool.reserve())
        self.assertEqual(pool.progress()[1]['unready'], 5)
        source = Path(cfgs[0]['source_root'])
        path = source / 'ready/producer-0000.jsonl'
        line = p.encoded(dict(rows[0], status='ready'))
        with path.open('ab') as stream:
            stream.write(line[:-1])
        self.assertIsNone(pool.reserve())
        with path.open('ab') as stream:
            stream.write(b'\n')
        self.ready(rows[1:4], source)
        job = pool.reserve()
        self.assertEqual(len(job['metas']), 3)
        worker, progress = pool.progress()
        self.assertEqual(progress['unready'], 1)
        self.assertEqual(progress['N'], progress['reserved'] + progress['ready_unclaimed'] + progress['unready'])
        offsets = dict(pool.offsets)
        pool.progress()
        self.assertEqual(pool.offsets, offsets)
        self.assertFalse(progress['complete'])

    def test_normalized_config_binds_all_other_fields(self):
        cfgs, rows = self.fixture(count=1)
        cfg = cfgs[0]
        baseline = p.config_identity(cfg)
        for key in cfg:
            changed = copy.deepcopy(cfg)
            if key == 'pool_manifest':
                changed[key]['path'] += '.changed'
            elif key == 'source_sha256':
                changed[key]['/synthetic-extra-binding'] = 'changed'
            else:
                changed[key] = dict(changed_field=key)
            with self.subTest(key=key):
                self.assertNotEqual(p.config_identity(changed), baseline)
        changed = dict(cfg, _explicit_cfg_field='changed')
        self.assertNotEqual(p.config_identity(changed), baseline)
        with self.assertRaises(ValueError):
            p.check_manifest(changed)
        runtime = dict(cfg, _config_path=cfg['config_path'], _config_sha=p.sha(cfg['config_path']))
        self.assertEqual(p.config_identity(runtime), baseline)
        p.check_manifest(runtime)

    def test_unadmitted_run_has_zero_copy_load_and_model_calls(self):
        cfgs, rows = self.fixture(count=1)
        pool = self.pool(cfgs[0], admit=False)
        fake = FakeLLM()
        with patch.object(m.local, 'stage_source') as stage:
            with self.assertRaises(ValueError):
                m.run(pool, lambda before=False: self.proof(cfgs[0], before), fake.factory, generate=fake.generate)
            stage.assert_not_called()
        self.assertEqual((fake.loads, fake.calls), (0, 0))
        self.assertFalse(list((pool.root / 'sources').glob('*.json')))

    def test_malformed_batch_does_not_trust_archive_or_complete(self):
        cfgs, rows = self.fixture(count=2)
        first = self.pool(cfgs[0])
        job = first.reserve()
        path = first.batch_path(job['batch_id'])
        variants = []
        empty = copy.deepcopy(job)
        empty.update(metas=[], source_claims=[], reserved_sources=0)
        variants.append(empty)
        truncated = copy.deepcopy(job)
        truncated['source_claims'] = truncated['source_claims'][:1]
        variants.append(truncated)
        duplicated = copy.deepcopy(job)
        duplicated['metas'][1] = duplicated['metas'][0]
        duplicated['source_claims'][1] = duplicated['source_claims'][0]
        variants.append(duplicated)
        foreign = copy.deepcopy(job)
        foreign['owner']['config']['path'] = '/unknown-owner-config.json'
        variants.append(foreign)
        second = self.pool(cfgs[1])
        with patch.object(m.local, 'archive_drafts') as archives:
            for bad in variants:
                p.atomic(path, bad)
                with self.subTest(variant=bad):
                    with self.assertRaises(ValueError):
                        first.ownership(bad)
                    m.recover_commits(second)
                    self.assertFalse(second.progress()[1]['complete'])
            archives.assert_not_called()
        self.assertIsNone(second.reserve())

    def test_real_validate_and_check_sources_with_mock_hardware_edges(self):
        cfgs, rows = self.fixture(count=3)
        cfg = cfgs[0]
        holders = {o['pid']: o for o in cfg['holders']}
        apps = [dict(pid=o['pid'], uuid=o['uuid'], used_MiB='1') for o in cfg['holders']]
        with patch.object(m.h.subprocess, 'check_output', return_value=m.old.HOSTS[cfg['node']]),              patch.object(m.base, 'identity', side_effect=lambda pid: holders[pid]),              patch.object(m.base, 'uuid_for', side_effect=lambda gpu: 'uuid-' + str(gpu)),              patch.object(m.base, 'apps', return_value=apps),              patch.object(m.h, 'lock_snapshot', return_value=dict(held=[])),              patch.object(m.base, 'selected_sources', m.selected):
            m.validate(cfg)  # Real cloned local validate, real check_holder, real SHA checks.
            m.base.check_sources(cfg, cfg['config_path'], p.sha(cfg['config_path']), m.BASE_SHA)
            with self.assertRaises(RuntimeError):
                m.base.check_sources(cfg, cfg['config_path'], p.sha(cfg['config_path']), p.sha(m.SELF))
            bad = copy.deepcopy(cfg)
            bad['release_authorized'] = False
            with self.assertRaises(ValueError):
                m.validate(bad)
            bad = copy.deepcopy(cfg)
            bad['producer'] = p.ref(m.old.SELF)
            with self.assertRaises(ValueError):
                m.validate(bad)
        self.assertEqual(Path(m.base.__file__).resolve(), RT / 'resource_manager.py')
        self.assertEqual(p.sha(m.base.__file__), m.BASE_SHA)
        self.assertEqual(m.base.SAFETY + m.base.LOAD, 160 * 2**30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
