"""CPU-only temporary fixtures. Fake parent injection is NOT a production gate.

No GPU model is imported, no signals/real locks/leases/old pool files are changed.
All source/queue/admission/commit mutations are confined to TemporaryDirectory.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_union_native_v4 as v4


class NativeUnionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='native-union-v4-cpu-test-')
        self.root = Path(self.tmp.name)
        self.rt = self.root / 'runtime'
        self.rt.mkdir()
        self.bar = self.root / 'experiment'
        self.bar.mkdir()
        original_base = Path(v4.base.__file__).read_bytes()
        (self.rt / 'resource_manager.py').write_bytes(original_base)
        self.patches = [patch.object(v4, 'BAR', self.bar), patch.object(v4, 'RT', self.rt),
                        patch.object(v4.base, '__file__', str(self.rt / 'resource_manager.py'))]
        for mocked in self.patches:
            mocked.start()
        self.rows = [dict(id='source%d' % n, split='train', metadata=dict(index=n)) for n in range(9)]
        source = self.bar / 'source'
        source.mkdir()
        (source / 'ready.jsonl').write_text(''.join(json.dumps(dict(r, status='ready')) + '\n' for r in self.rows))
        self.origin_cfgs = [self.make_origin(i, self.rows[i * 4:(i + 1) * 4] if i == 0 else self.rows[4:]) for i in range(2)]
        model = self.root / 'modelref.json'
        self.write(model, dict(model_identity=dict(root=v4.mgr.MODEL, sha256='test-model')))
        safe = self.root / 'prior' / 'resources_safe.json'
        self.write(safe, dict(all_holders_safe=True, locks_released=True, leases_restored=True))
        holders = []
        holderrefs = []
        for gpu in (0, 1):
            identity = v4.base.identity(os.getpid())
            identity.update(pid=1000 + gpu, gpu=gpu, uuid='test-gpu%d' % gpu)
            holders.append(identity)
            path = safe.parent / ('holder_restored_gpu%d.json' % gpu)
            self.write(path, dict(uuid=identity['uuid'], holder_identity=identity))
            holderrefs.append(v4.poolmod.ref(path))
        pool = self.bar / 'caption_union_native_v4_fixture'
        phase = pool.name + '__newworker'
        plan = self.rt / 'caption_union_native_v4_plan.json'
        self.write(plan, self.rows)
        manifest_path = self.rt / 'caption_union_native_v4_manifest.json'
        config_path = self.rt / 'caption_union_native_v4_fixture.json'
        cfg = dict(schema='caption_union_native_v4', node='N4', gpus=[0, 1], tensor_parallel_size=2,
                   recipe=v4.local.light.RECIPE, release_authorized=True, env=dict(CUDA_VISIBLE_DEVICES='0,1'), cwd='/tmp',
                   pool_root=str(pool), pool_worker_id='newworker', phase=phase, queue=str(self.bar / 'queues' / phase),
                   out=str(self.bar / phase), draft_out=str(self.bar / phase / 'archives'),
                   stop_file=str(pool / 'workers/newworker/STOP'), config_path=str(config_path),
                   argv=['/opt/venv_vllm/bin/python', '-B', str(v4.SELF), 'worker', '--config', str(config_path)],
                   producer=v4.poolmod.ref(v4.SELF), resource_manager_ref=v4.poolmod.ref(v4.SELF),
                   origin_configs=[c['config_path'] for c in self.origin_cfgs],
                   origin_config_refs=[v4.poolmod.ref(c['config_path']) for c in self.origin_cfgs],
                   source_root=str(source), plan=v4.poolmod.ref(plan), input_count=len(self.rows), targetcount=len(self.rows),
                   union_manifest=dict(path=str(manifest_path), sha256='pending'),
                   resource_previous_safe=v4.poolmod.ref(safe), holders=holders, holder_restored_refs=holderrefs,
                   modelref=dict(**v4.poolmod.ref(model), model_identity_sha256='test-model'), source_sha256={})
        self.cfg = cfg
        for origin in self.origin_cfgs:
            origin['modelref'] = cfg['modelref']
            self.bind_origin(origin)
        cfg['origin_config_refs'] = [v4.poolmod.ref(c['config_path']) for c in self.origin_cfgs]
        self.rebind()

    def tearDown(self):
        for mocked in reversed(self.patches):
            mocked.stop()
        self.tmp.cleanup()

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(v4.poolmod.encoded(value))

    def make_origin(self, index, rows):
        root = self.bar / ('oldpool%d' % index)
        root.mkdir()
        plan = self.root / ('oldplan%d.json' % index)
        hand = self.root / ('oldhand%d.json' % index)
        manifest_path = self.root / ('oldmanifest%d.json' % index)
        config_path = self.root / ('oldconfig%d.json' % index)
        self.write(plan, rows)
        self.write(hand, dict(remaining_sources=rows, old_worker_tree_absent=True, original_config_resume_forbidden=True))
        cfg = dict(plan=v4.poolmod.ref(plan), source_handoff=v4.poolmod.ref(hand), source_root=str(self.bar / 'source'),
                   pool_root=str(root), pool_worker_id='oldworker%d' % index, node='N4', gpus=[2 + index * 2, 3 + index * 2],
                   config_path=str(config_path), targetcount=len(rows), input_count=len(rows),
                   pool_manifest=dict(path=str(manifest_path), sha256='pending'), source_sha256={})
        self.bind_origin(cfg)
        return cfg

    def bind_origin(self, cfg):
        manifest = dict(schema='caption_ready_pool_v1', plan=cfg['plan'], source_handoff=cfg['source_handoff'],
                        source_root=cfg['source_root'], pool_root=cfg['pool_root'],
                        participants=[dict(worker_id=cfg['pool_worker_id'], node=cfg['node'], gpus=cfg['gpus'],
                                           config_path=cfg['config_path'], config_identity_sha256=v4.poolmod.config_identity(cfg))])
        self.write(cfg['pool_manifest']['path'], manifest)
        cfg['pool_manifest'] = v4.poolmod.ref(cfg['pool_manifest']['path'])
        self.write(cfg['config_path'], cfg)

    def rebind(self, participant_mutator=None):
        cfg = self.cfg
        bindings = {str(v4.SELF): v4.poolmod.sha(v4.SELF), str(self.rt / 'resource_manager.py'): v4.BASE_SHA}
        for module in (v4.oldunion, v4.mgr, v4.mgr.old, v4.poolmod, v4.local, v4.local.light,
                       v4.local.light.native, v4.local.light.schemas, v4.h, v4.h.bench, v4.h.c):
            path = str(Path(module.__file__).resolve())
            bindings[path] = v4.poolmod.sha(path)
        for reference in (*cfg['origin_config_refs'], cfg['plan'], cfg['resource_previous_safe'], cfg['modelref'], *cfg['holder_restored_refs']):
            bindings[reference['path']] = reference['sha256']
        for origin in self.origin_cfgs:
            for reference in (origin['plan'], origin['pool_manifest'], origin['source_handoff']):
                bindings[reference['path']] = reference['sha256']
            bindings.update(origin['source_sha256'])
        cfg['source_sha256'] = bindings
        participant = dict(worker_id=cfg['pool_worker_id'], config_identity_sha256=v4.config_identity(cfg),
                           **{k: cfg[k] for k in ('node', 'gpus', 'config_path', 'queue', 'out', 'draft_out')})
        participants = [participant]
        if participant_mutator:
            participant_mutator(participants)
        manifest = dict(schema='caption_union_native_manifest_v4', participants=participants,
                        **{k: cfg[k] for k in ('plan', 'origin_config_refs', 'source_root', 'pool_root', 'recipe')})
        self.write(cfg['union_manifest']['path'], manifest)
        cfg['union_manifest'] = v4.poolmod.ref(cfg['union_manifest']['path'])
        cfg['source_sha256'][cfg['union_manifest']['path']] = cfg['union_manifest']['sha256']
        self.write(cfg['config_path'], cfg)

    def fake_parent(self):
        cfg = self.cfg
        child = v4.base.identity(os.getpid())
        parent = copy.deepcopy(child)
        parent.update(pid=90001, ppid=90000,
                      argv=['/opt/venv_vllm/bin/python', '-B', str(v4.SELF), '--config', cfg['config_path'],
                            '--config-sha', v4.poolmod.sha(cfg['config_path']), '--manager-sha', v4.BASE_SHA])
        child.update(ppid=parent['pid'], argv=cfg['argv'], CVD=['CUDA_VISIBLE_DEVICES=0,1'])
        queue = Path(cfg['queue'])
        admission = dict(manager_identity=parent, config_sha256=v4.poolmod.sha(cfg['config_path']), manager_sha256=v4.BASE_SHA,
                         producer=cfg['producer'], plan=cfg['plan'], node=cfg['node'], gpus=cfg['gpus'],
                         source_sha256=cfg['source_sha256'], out=cfg['out'], targetcount=cfg['targetcount'],
                         model_identity_sha256=cfg['modelref']['model_identity_sha256'])
        self.write(queue / 'admission.json', admission)
        self.write(queue / 'live.json', dict(child_identity=child, manager_identity=parent))
        def injected_fake_parent(before=False):
            return dict(admitted=True, caller_owned=True, child_identity=child, parent_identity=parent,
                        admission_ref=v4.poolmod.ref(queue / 'admission.json'))
        return injected_fake_parent

    def admitted_pool(self):
        pool = v4.NativeUnionPool(self.cfg, self.fake_parent())
        pool.admit()
        return pool

    def draft_archive(self, pool, job):
        cache = Path(tempfile.mkdtemp(dir=self.root, prefix='batch-cache-'))
        drafts = [dict(id=m['id'], split=m['split'], status='draft_generated', **v4.FLAGS) for m in job['metas']]
        for d in drafts:
            self.write(cache / 'caption_two_stage_results/drafts' / d['split'] / d['id'] / 'draft.json', d)
        commit = v4.local.writeback(cache, self.cfg['draft_out'], job['batch_id'])
        return commit, drafts

    def test_valid_config_and_native_resource_validation(self):
        self.assertEqual(len(v4.check_config(self.cfg)[1]), 9)
        with patch.object(v4.h.subprocess, 'check_output', return_value=v4.HOSTS['N4']),              patch.object(v4.base, 'headroom', return_value=dict(headroom=160 * 2**30)),              patch.object(v4.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 8 * 2**30})()),              patch.object(v4.base, 'apps', return_value=[]), patch.object(v4.base, 'identity', side_effect=lambda pid: next(h for h in self.cfg['holders'] if h['pid'] == pid)),              patch.object(v4.base, 'uuid_for', side_effect=lambda gpu: 'test-gpu%d' % gpu),              patch.object(v4.base, 'check_holder') as holder_check, patch.object(v4.h, 'lock_snapshot', return_value=dict(held=[])),              patch.object(v4.base, 'lease_snapshot', return_value=None):
            v4.validate(self.cfg)
            self.assertEqual(holder_check.call_count, 2)

    def test_wrong_node_pair_and_argv_rejected(self):
        for field, value in [('node', 'N3'), ('gpus', [0, 2]), ('tensor_parallel_size', 4), ('argv', ['python', 'oldworker'])]:
            cfg = copy.deepcopy(self.cfg)
            cfg[field] = value
            with self.assertRaises(ValueError):
                v4.check_config(cfg)

    def test_duplicate_config_rejected(self):
        self.rebind(lambda ps: ps.append(dict(ps[0], worker_id='otherworker')))
        with self.assertRaisesRegex(ValueError, 'Duplicate union'):
            v4.check_config(self.cfg)

    def test_conflicting_origin_metadata_rejected(self):
        origin = self.origin_cfgs[1]
        rows = v4.poolmod.load(origin['plan']['path'])
        rows.append(dict(self.rows[0], metadata=dict(index=-1)))
        self.write(origin['plan']['path'], rows)
        origin['plan'] = v4.poolmod.ref(origin['plan']['path'])
        self.write(origin['source_handoff']['path'], dict(remaining_sources=rows, old_worker_tree_absent=True, original_config_resume_forbidden=True))
        origin['source_handoff'] = v4.poolmod.ref(origin['source_handoff']['path'])
        origin['targetcount'] = origin['input_count'] = len(rows)
        self.bind_origin(origin)
        self.cfg['origin_config_refs'][1] = v4.poolmod.ref(origin['config_path'])
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'metadata conflict'):
            v4.check_config(self.cfg)

    def test_no_native_parent_gate_rejected(self):
        pool = v4.NativeUnionPool(self.cfg)
        with self.assertRaisesRegex(ValueError, 'native parent gate required'):
            pool.admit()
        with self.assertRaises((FileNotFoundError, ValueError)):
            v4.gate(dict(self.cfg, _config_path=self.cfg['config_path'], _config_sha=v4.poolmod.sha(self.cfg['config_path'])), True)

    def test_foreign_owner_rejected_and_claim_not_reissued(self):
        pool = self.admitted_pool()
        job = pool.reserve()
        foreign = copy.deepcopy(job)
        foreign['owner']['worker_id'] = 'not-a-participant'
        with self.assertRaisesRegex(ValueError, 'Foreign/nonparticipant'):
            pool.check_job(foreign)
        self.assertEqual(job['reserved_sources'], 3)
        report = v4.readonly_foreign_owner_report(self.cfg)
        self.assertEqual(len(report['blocked_source_keys']), 3)
        self.assertTrue(report['original_resume_recommended_forbidden'])
        self.assertFalse(report['old_launchers_modified'])
        next_job = pool.reserve()
        self.assertFalse(set(m['id'] for m in next_job['metas']) & set(m['id'] for m in job['metas']))

    def test_existing_claims_all_blocked_never_complete_or_model_retry(self):
        pool = v4.NativeUnionPool(self.cfg)
        for key, readers in pool.origins_by_key.items():
            reader = readers[0]
            self.write(reader.source_path(key), dict(meta=pool.rows[key], pool_sha256=reader.pool_sha, owner=dict(worker_id='old-v3')))
        pool = self.admitted_pool()
        self.assertIsNone(pool.reserve())
        self.assertEqual(pool.progress()[1]['blocked'], 9)
        self.assertFalse(pool.progress()[1]['complete'])
        Path(self.cfg['stop_file']).touch()
        called = []
        summary = v4.run(pool, runtime_factory=lambda cfg: called.append('model'), poll_seconds=0)
        self.assertEqual(called, [])
        self.assertEqual(summary['status'], 'drained_not_complete')
        self.assertFalse(summary['complete'])
        v4.summaries(self.cfg, Path(self.cfg['out']))

    def test_existing_admitted_namespace_denies_restart(self):
        self.admitted_pool()
        with self.assertRaisesRegex(ValueError, 'Previously reserved/admitted'):
            v4.NativeUnionPool(self.cfg, self.fake_parent()).admit()
        with self.assertRaisesRegex(ValueError, 'Existing queue/output/admitted'):
            v4.validate(self.cfg)

    def test_durable_commit_exact_and_idempotent(self):
        pool = self.admitted_pool()
        job = pool.reserve()
        commit, drafts = self.draft_archive(pool, job)
        pool.publish(job, commit, drafts)
        pool.publish(job, commit, drafts)
        self.assertEqual(len(pool.terminals), 3)
        self.assertEqual(pool.counts, dict(draft_generated=3))
        for meta in job['metas']:
            key = v4.poolmod.source_key(meta)
            self.assertEqual(pool.verify_terminal(pool.terminal_path(key))[0], key)
        conflict = copy.deepcopy(commit)
        conflict['archive_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'archive/commit'):
            pool.publish(job, conflict, drafts)
        self.assertFalse(pool.progress()[1]['complete'])
        recovered = v4.readonly_recovery_check(self.cfg, job['batch_id'])
        self.assertEqual(recovered['model_calls'], 0)
        self.assertFalse(recovered['receipts_published'])

    def test_foreign_claim_source_path_or_owner_change_rejected(self):
        pool = self.admitted_pool()
        job = pool.reserve()
        claim = job['source_claims'][0][0]
        value = v4.poolmod.load(claim['path'])
        value['owner']['node'] = 'N6'
        self.write(claim['path'], value)
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            pool.check_job(job)

    def test_hold_before_load_denies_model_creation(self):
        pool = self.admitted_pool()
        real_fake = pool.parent_gate
        def denied(before=False):
            if before:
                raise ValueError('GPU occupied before admitted model load')
            return real_fake(before)
        pool.parent_gate = denied
        called = []
        with self.assertRaisesRegex(ValueError, 'GPU occupied'):
            v4.run(pool, runtime_factory=lambda cfg: called.append('model'))
        self.assertEqual(called, [])
        self.assertEqual(len(pool.failed), 3)

    def test_stop_does_not_discard_prefetched_batch(self):
        pool = self.admitted_pool()
        copied, generated = [], []
        def stage(source_root, meta, cache):
            copied.append(meta['id'])
            return dict(meta=meta), dict(copy_seconds=0)
        def generate(entries, args, runtime, out, admitted):
            self.assertTrue(admitted(args, runtime))
            drafts = []
            for entry in entries:
                meta = entry['meta']
                generated.append(meta['id'])
                draft = dict(id=meta['id'], split=meta['split'], status='draft_generated', **v4.FLAGS)
                self.write(out / 'drafts' / meta['split'] / meta['id'] / 'draft.json', draft)
                drafts.append(draft)
            Path(self.cfg['stop_file']).touch()
            return dict(drafts=drafts, new_model_calls=len(drafts))
        summary = v4.run(pool, runtime_factory=lambda cfg: object(), stage_source=stage, generate=generate,
                         local_parent=str(self.root), poll_seconds=0)
        self.assertEqual(set(copied), set(generated))
        self.assertEqual(len(generated), 6)
        self.assertEqual(summary['processed'], 6)
        self.assertEqual(summary['reserved'], 6)
        self.assertEqual(summary['uncommitted_owned'], 0)
        self.assertEqual(summary['status'], 'drained_not_complete')
        self.assertFalse(summary['complete'])
        self.assertEqual(len(list(Path(self.cfg['draft_out']).glob('*.committed.json'))), 2)
        self.assertEqual(len(list((pool.readers[1].root / 'sources').glob('*.json'))), 2)
        v4.summaries(self.cfg, Path(self.cfg['out']))

    def add_overlap(self):
        origin = self.origin_cfgs[1]
        rows = v4.poolmod.load(origin['plan']['path']) + [self.rows[0]]
        self.write(origin['plan']['path'], rows)
        origin['plan'] = v4.poolmod.ref(origin['plan']['path'])
        self.write(origin['source_handoff']['path'], dict(remaining_sources=rows, old_worker_tree_absent=True, original_config_resume_forbidden=True))
        origin['source_handoff'] = v4.poolmod.ref(origin['source_handoff']['path'])
        origin['targetcount'] = origin['input_count'] = len(rows)
        self.bind_origin(origin)
        self.cfg['origin_config_refs'][1] = v4.poolmod.ref(origin['config_path'])
        self.rebind()

    def test_overlap_claims_every_origin_once(self):
        self.add_overlap()
        pool = self.admitted_pool()
        self.assertEqual(len(pool.rows), 9)
        job = pool.reserve()
        self.assertEqual(len(job['source_claims'][0]), 2)
        pool.check_job(job)
        commit, drafts = self.draft_archive(pool, job)
        pool.publish(job, commit, drafts)
        self.assertIsNone(pool.progress()[1]['global_terminal'])
        self.assertEqual(v4.check_completeness(self.cfg)['global_terminal'], 3)

    def test_partial_cross_origin_claim_race_never_reissues(self):
        self.add_overlap()
        pool = self.admitted_pool()
        key = v4.poolmod.source_key(self.rows[0])
        readers = pool.origins_by_key[key]
        second = readers[1].source_path(key)
        real_exclusive = v4.poolmod.exclusive
        def race(path, value):
            if Path(path) == second:
                self.write(second, dict(meta=self.rows[0], pool_sha256=readers[1].pool_sha, owner=dict(worker_id='racing-old-worker')))
                raise FileExistsError('another worker reserved the second origin')
            return real_exclusive(path, value)
        with patch.object(v4.poolmod, 'exclusive', side_effect=race):
            job = pool.reserve()
        self.assertNotIn(self.rows[0], job['metas'])
        self.assertIn(key, pool.blocked)
        self.assertTrue(readers[0].source_path(key).exists())
        pool.check_job(job)
        self.assertFalse(pool.progress()[1]['complete'])

    def test_full_completion_only_explicit_offline_and_durable(self):
        pool = self.admitted_pool()
        jobs = []
        for _ in range(3):
            job = pool.reserve()
            commit, drafts = self.draft_archive(pool, job)
            pool.publish(job, commit, drafts)
            jobs.append((job, commit))
        self.assertFalse(pool.progress()[1]['complete'])
        self.assertIsNone(pool.progress()[1]['global_terminal'])
        self.assertTrue(v4.check_completeness(self.cfg)['complete'])
        self.assertIsNone(pool.reserve())
        Path(self.cfg['stop_file']).touch()
        summary = v4.run(pool, runtime_factory=lambda cfg: self.fail('drained pool must not load model'))
        self.assertEqual(summary['status'], 'drained_not_complete')
        v4.summaries(self.cfg, Path(self.cfg['out']))
        Path(jobs[0][1]['archive_path']).unlink()
        reread = v4.check_completeness(self.cfg)
        self.assertFalse(reread['complete'])
        self.assertTrue(reread['invalid_receipts'])
        with self.assertRaises(ValueError):
            v4.summaries(self.cfg, Path(self.cfg['out']))

    def test_origin_valid_terminal_counted_without_pool_constructor_or_journal(self):
        origin = self.origin_cfgs[0]
        origin.update(draft_out=str(self.root / 'oldarchives'), argv=['test-original-worker'])
        self.bind_origin(origin)
        self.cfg['origin_config_refs'][0] = v4.poolmod.ref(origin['config_path'])
        self.rebind()
        reader = v4.read_origin(self.cfg['origin_config_refs'][0])
        metas = self.rows[:2]
        owner = dict(worker_id=origin['pool_worker_id'], node=origin['node'], gpus=origin['gpus'],
                     config=v4.poolmod.ref(origin['config_path']), identity=dict(argv=origin['argv']))
        claims = []
        for meta in metas:
            path = reader.source_path(v4.poolmod.source_key(meta))
            self.write(path, dict(pool_sha256=reader.pool_sha, meta=meta, owner=owner, **v4.FLAGS))
            claims.append(v4.poolmod.ref(path))
        batch_id = 'batch-' + 'a' * 32
        job = dict(batch_id=batch_id, pool_sha256=reader.pool_sha, metas=metas, source_claims=claims,
                   reserved_sources=2, owner=owner, **v4.FLAGS)
        self.write(reader.batch_path(batch_id), job)
        self.write(reader.root / 'admitted' / (owner['worker_id'] + '.json'), dict(owner=owner))
        cache = Path(tempfile.mkdtemp(dir=self.root, prefix='old-valid-cache-'))
        for meta in metas:
            self.write(cache / 'caption_two_stage_results/drafts' / meta['split'] / meta['id'] / 'draft.json',
                       dict(id=meta['id'], split=meta['split'], status='draft_generated', **v4.FLAGS))
        commit = v4.local.writeback(cache, origin['draft_out'], batch_id)
        for meta, claim in zip(metas, claims):
            key = v4.poolmod.source_key(meta)
            receipt = dict(id=key[0], split=key[1], status='draft_generated', pool_sha256=reader.pool_sha,
                           batch_id=batch_id, owner=owner, source_claim=claim, batch_claim=v4.poolmod.ref(reader.batch_path(batch_id)),
                           commit=v4.poolmod.ref(Path(commit['archive_path']).with_suffix('.committed.json')),
                           archive_path=commit['archive_path'], archive_member=batch_id + '/caption_two_stage_results/drafts/' + key[1] + '/' + key[0] + '/draft.json', **v4.FLAGS)
            self.write(reader.root / 'terminals' / reader.source_path(key).name, receipt)
        before = sorted(str(p) for p in reader.root.rglob('*'))
        with patch.object(v4.poolmod.Pool, '__init__', side_effect=AssertionError('must never construct original Pool')):
            pool = v4.NativeUnionPool(self.cfg)
            self.assertIsNone(pool.progress()[1]['global_terminal'])
            self.assertEqual(v4.check_completeness(self.cfg)['global_terminal'], 2)
        self.assertEqual(before, sorted(str(p) for p in reader.root.rglob('*')))
        self.assertFalse((reader.root / 'claims.jsonl').exists())
        self.assertFalse((reader.root / 'terminal_index.jsonl').exists())

    def test_gate_denied_before_claim_no_mutation(self):
        pool = self.admitted_pool()
        pool.parent_gate = lambda before=False: dict(admitted=False, caller_owned=False)
        before = sorted(str(p) for p in self.root.rglob('*'))
        with self.assertRaisesRegex(ValueError, 'denied before claiming'):
            pool.reserve()
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))
        self.assertEqual(pool.own_reserved, set())

    def test_production_never_scans_origin_history(self):
        original_glob = Path.glob
        origin_roots = {Path(c['pool_root']) for c in self.origin_cfgs}
        def guarded(path, pattern):
            if any(path.is_relative_to(root) for root in origin_roots):
                self.fail('production must not glob any old source/terminal tree')
            return original_glob(path, pattern)
        with patch.object(Path, 'glob', guarded), patch.object(v4.local, 'archive_drafts', side_effect=AssertionError('historical tar scan')):
            pool = self.admitted_pool()
            self.assertIsNone(pool.progress()[1]['global_terminal'])
            pool.reserve()
            pool.progress()

    def test_torn_own_terminal_is_rechecked_after_fix(self):
        pool = self.admitted_pool()
        job = pool.reserve()
        commit, drafts = self.draft_archive(pool, job)
        pool.publish(job, commit, drafts)
        path = pool.terminal_path(v4.poolmod.source_key(job['metas'][0]))
        original = path.read_bytes()
        path.write_bytes(b'\x00')
        verifier = v4.NativeUnionPool(self.cfg)
        self.assertIn(str(path), verifier.invalid_receipts)
        self.assertNotIn(path, verifier._seen_terminals)
        path.write_bytes(original)
        verifier.refresh()
        self.assertNotIn(str(path), verifier.invalid_receipts)
        self.assertEqual(len(verifier.terminals), 3)

    def test_offline_conflict_excludes_source_and_can_reread_torn_receipt(self):
        # Temporary fixture injects already-verified old evidence to exercise
        # conflict classification, not production resource admission.
        pool = self.admitted_pool()
        job = pool.reserve()
        commit, drafts = self.draft_archive(pool, job)
        pool.publish(job, commit, drafts)
        origin = v4.read_origin(self.cfg['origin_config_refs'][0])
        path = origin.root / 'terminals' / origin.source_path(v4.poolmod.source_key(job['metas'][0])).name
        receipt = dict(id=job['metas'][0]['id'], split='train', status='draft_generated', owner=dict(worker_id='conflicting-old'),
                       batch_claim=v4.poolmod.ref(pool.batch_path(job['batch_id'])),
                       commit=v4.poolmod.ref(Path(commit['archive_path']).with_suffix('.committed.json')))
        self.write(path, receipt)
        with patch.object(v4.poolmod.Pool, 'verify_terminal', return_value=receipt):
            result = v4.check_completeness(self.cfg)
        self.assertEqual(result['global_terminal'], 2)
        self.assertEqual(result['conflicting_source_keys'], [[job['metas'][0]['id'], 'train']] if isinstance(result['conflicting_source_keys'][0], list) else [(job['metas'][0]['id'], 'train')])
        self.assertFalse(result['complete'])
        valid_path = pool.terminal_path(v4.poolmod.source_key(job['metas'][1]))
        original = valid_path.read_bytes()
        valid_path.write_bytes(b'\x00')
        self.assertIn(str(valid_path), v4.check_completeness(self.cfg)['invalid_receipts'])
        valid_path.write_bytes(original)
        self.assertNotIn(str(valid_path), v4.check_completeness(self.cfg)['invalid_receipts'])

    def test_check_only_cli_never_calls_native_main_or_creates_outputs(self):
        args = [str(v4.SELF), '--check-only', '--config', self.cfg['config_path'],
                '--config-sha', v4.poolmod.sha(self.cfg['config_path']), '--manager-sha', v4.BASE_SHA]
        before = sorted(str(p) for p in self.root.rglob('*'))
        with patch.object(sys, 'argv', args), patch.object(v4, 'validate') as validate,              patch.object(v4.base, 'check_model') as modelcheck, patch.object(v4.base, 'main', side_effect=AssertionError('must not enter resource transaction')):
            result = v4.main()
            validate.assert_called_once_with(self.cfg)
            modelcheck.assert_called_once_with(self.cfg)
        self.assertEqual(result['status'], 'check_only_passed')
        self.assertFalse(result['queue_created'])
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_actual_parent_gate_code_with_temporary_kernel_fixture(self):
        # Execute the unchanged production parent_gate code against injected CPU
        # /proc, fd and lock records. No real GPU locks or lease paths are opened.
        fake_parent = self.fake_parent()
        proof = fake_parent()
        parent, child = proof['parent_identity'], proof['child_identity']
        queue = Path(self.cfg['queue'])
        admission = v4.poolmod.load(queue / 'admission.json')
        admission['holders'] = [dict(uuid='test-gpu%d' % gpu) for gpu in self.cfg['gpus']]
        self.write(queue / 'admission.json', admission)
        fake_proc = self.root / 'proc'
        fds = fake_proc / str(parent['pid']) / 'fd'
        fds.mkdir(parents=True)
        lines = []
        names = {}
        for gpu in self.cfg['gpus']:
            for kind, name in [('lock', '/tmp/ms12k_gpu_%d.lock' % gpu), ('lease', '/tmp/od_reserve_%d' % gpu)]:
                target = self.root / (kind + str(gpu))
                target.touch()
                fd = fds / (kind + str(gpu))
                fd.symlink_to(target)
                names[str(fd)] = name
                if kind == 'lock':
                    st = target.stat()
                    lines.append('1: FLOCK ADVISORY WRITE %d %02x:%02x:%d 0 EOF' % (parent['pid'], os.major(st.st_dev), os.minor(st.st_dev), st.st_ino))
        (fake_proc / 'locks').write_text(chr(10).join(lines))
        def remap(path):
            path = str(path)
            if path == '/proc':
                return fake_proc
            if path == '/proc/locks':
                return fake_proc / 'locks'
            for fd, name in names.items():
                if path == name:
                    return Path(fd).resolve()
            return Path(path)
        original_readlink = os.readlink
        fake_os = types.SimpleNamespace(getppid=lambda: parent['pid'], getpid=os.getpid, getuid=os.getuid,
                                       major=os.major, minor=os.minor,
                                       readlink=lambda p: names[str(p)] if str(p) in names else original_readlink(p))
        native_fn = types.FunctionType(v4.h.parent_gate.__code__, dict(v4.h.parent_gate.__globals__, Path=remap, os=fake_os),
                                       'fixture_native_parent_gate', v4.h.parent_gate.__defaults__)
        cfg = dict(self.cfg, _config_path=self.cfg['config_path'], _config_sha=v4.poolmod.sha(self.cfg['config_path']))
        outsider = dict(child, pid=777, ppid=777)
        def identity(pid):
            return parent if pid == parent['pid'] else child if pid == child['pid'] else outsider
        with patch.object(v4.h, 'parent_gate', native_fn), patch.object(v4.base, 'identity', side_effect=identity),              patch.object(v4.base, 'uuid_for', side_effect=lambda g: 'test-gpu%d' % g), patch.object(v4.base, 'apps', return_value=[]) as apps:
            self.assertTrue(v4.gate(cfg, True)['admitted'])
            apps.return_value = [dict(uuid='test-gpu0', pid=child['pid'])]
            with self.assertRaisesRegex(ValueError, 'GPU occupied before'):
                v4.gate(cfg, True)
            self.assertTrue(v4.gate(cfg, False)['admitted'])
            apps.return_value = [dict(uuid='test-gpu0', pid=777)]
            with self.assertRaisesRegex(ValueError, 'outside sole admitted'):
                v4.gate(cfg, False)
            apps.return_value = []
            (fake_proc / 'locks').write_text(chr(10).join(line.replace('WRITE 90001', 'WRITE 90002') for line in lines))
            with self.assertRaisesRegex(ValueError, 'FLOCK fd/owner missing'):
                v4.gate(cfg, False)
            (fake_proc / 'locks').write_text(chr(10).join(lines))
            parent['cgroup'] = 'different-cgroup'
            with self.assertRaisesRegex(ValueError, 'direct native parent identity'):
                v4.gate(cfg, False)

    def test_fake_complete_without_terminals_rejected(self):
        pool = self.admitted_pool()
        worker, progress = pool.progress()
        summary = dict(worker, scope_pool=True, status='complete', complete=True, N=9, global_terminal=9,
                       blocked=0, uncommitted_owned=0)
        self.write(Path(self.cfg['out']) / 'summary.json', summary)
        with self.assertRaises(ValueError):
            v4.summaries(self.cfg, Path(self.cfg['out']))


if __name__ == '__main__':
    unittest.main()
