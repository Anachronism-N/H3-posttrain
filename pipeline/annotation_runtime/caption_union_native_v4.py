"""Independent native union adapter. No old journals, owner repair, retries or launcher.

The unchanged native manager owns GPU locks, lease fds and holder restoration.
STOP is a worker-owned file: stop reserving, finish both in-flight batches, then
write a drained_not_complete summary. A SIGTERM uses the native failure cleanup.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import types
import uuid

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_union_pool_worker_v3 as oldunion

poolmod = oldunion.poolmod
mgr = oldunion.mgr
base, local, h, BAR = mgr.base, mgr.local, mgr.h, mgr.BAR
BASE_SHA = mgr.BASE_SHA
FLAGS = poolmod.FLAGS
need = poolmod.need
HOSTS = {'N4': '28.216.19.143', 'N6': '28.216.19.225'}
PAIRS = ([0, 1], [2, 3], [4, 5], [6, 7])
STATUSES = ('draft_generated', 'draft_partial', 'draft_needs_regeneration')


def checked(reference):
    path = Path(reference['path'])
    need(path.is_absolute() and not path.is_symlink() and str(path.resolve()) == str(path), 'Unsafe frozen reference path')
    need(poolmod.ref(path) == reference, 'Frozen reference SHA differs: ' + str(path))
    return poolmod.load(path)


def config_identity(cfg):
    value = copy.deepcopy(cfg)
    manifest = value['union_manifest']
    value.get('source_sha256', {}).pop(manifest['path'], None)
    manifest.pop('sha256', None)
    for key in ('_config_path', '_config_sha'):
        value.pop(key, None)
    return poolmod.digest(value)


def namespace(cfg):
    root = Path(cfg['pool_root'])
    worker = cfg['pool_worker_id']
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', worker) is not None, 'Invalid union worker identity')
    need(root.parent == BAR and re.fullmatch(r'caption_union_native_v4_[A-Za-z0-9_-]+', root.name) is not None, 'Independent v4 pool namespace required')
    phase = root.name + '__' + worker
    need(cfg['phase'] == phase and cfg['queue'] == str(BAR / 'queues' / phase)
         and cfg['out'] == str(BAR / phase) and cfg['draft_out'] == str(BAR / phase / 'archives')
         and cfg['stop_file'] == str(root / 'workers' / worker / 'STOP'), 'Worker queue/output/STOP namespace differs')
    need(Path(cfg['config_path']).parent == RT and Path(cfg['config_path']).name.startswith('caption_union_native_v4_')
         and Path(cfg['config_path']).suffix == '.json', 'Independent v4 config path required')
    for path in (root, Path(cfg['out']), Path(cfg['queue']), Path(cfg['draft_out'])):
        need(str(path.resolve()) == str(path) and not path.is_symlink(), 'Unsafe union namespace')
    return root / 'workers' / worker


def read_origin(reference):
    cfg = checked(reference)
    need(cfg['config_path'] == reference['path'], 'Origin frozen config path differs')
    manifest, rows = poolmod.check_manifest(cfg)
    # __new__ avoids Pool.__init__, whose journal recovery is NOT read-only.
    reader = poolmod.Pool.__new__(poolmod.Pool)
    reader.cfg, reader.manifest = cfg, manifest
    reader.root, reader.pool_sha = Path(cfg['pool_root']), cfg['plan']['sha256']
    reader.rows = {poolmod.source_key(row): row for row in rows}
    reader.reserved = set(reader.rows)
    return reader


def check_config(cfg):
    need(cfg['schema'] == 'caption_union_native_v4' and cfg['node'] in HOSTS
         and cfg['gpus'] in PAIRS and all(type(g) is int for g in cfg['gpus'])
         and type(cfg['tensor_parallel_size']) is int and cfg['tensor_parallel_size'] == 2, 'Only explicitly authorized N4/N6 TP2 pairs supported')
    need(cfg.get('gpu_memory_utilization', .88) == .88 and cfg['recipe'] == local.light.RECIPE, 'Original local model/sampling recipe required')
    need(cfg['release_authorized'] is True and cfg['env']['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, cfg['gpus']))
         and cfg['cwd'] == '/tmp', 'Explicit native GPU/environment assignment required')
    namespace(cfg)
    need(cfg['argv'] == ['/opt/venv_vllm/bin/python', '-B', str(SELF), 'worker', '--config', cfg['config_path']], 'Exact native union worker argv required')
    need(cfg['producer'] == cfg['resource_manager_ref'] == poolmod.ref(SELF), 'Separate native adapter identity required')
    need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and base.sha(base.__file__) == BASE_SHA
         and base.SAFETY + base.LOAD == 160 * 2**30, 'Original native lifecycle/160GiB boundary differs')
    refs = cfg['origin_config_refs']
    need(len(refs) == 2 and len({r['path'] for r in refs}) == 2 and cfg['origin_configs'] == [r['path'] for r in refs], 'Exactly two distinct immutable origins required')
    readers = [read_origin(r) for r in refs]
    need(len({str(r.root) for r in readers}) == 2 and all(r.cfg['source_root'] == cfg['source_root'] for r in readers), 'Origin root/source conflicts')
    need(Path(cfg['pool_root']) not in {r.root for r in readers}, 'Union cannot reuse origin pool')
    rows, origins = {}, {}
    bindings = dict(cfg['source_sha256'])
    required = {str(SELF), str(RT / 'resource_manager.py')}
    required.update(str(Path(m.__file__).resolve()) for m in
                    (oldunion, mgr, mgr.old, poolmod, local, local.light, local.light.native, local.light.schemas, h, h.bench, h.c))
    for reader, reference in zip(readers, refs):
        for frozen in (reference, reader.cfg['plan'], reader.cfg['pool_manifest'], reader.cfg['source_handoff']):
            checked(frozen)
            required.add(frozen['path'])
            need(bindings.get(frozen['path']) == frozen['sha256'], 'Missing immutable origin/config/plan/manifest/handoff source binding')
        handoff = checked(reader.cfg['source_handoff'])
        allowed = {poolmod.source_key(row): row for row in handoff['remaining_sources']}
        need(handoff['old_worker_tree_absent'] is True and handoff['original_config_resume_forbidden'] is True, 'Origin handoff unsafe')
        for row in reader.rows.values():
            key = poolmod.source_key(row)
            need(allowed.get(key) == row and (key not in rows or rows[key] == row), 'Origin metadata conflict or outside remaining handoff')
            rows[key] = row
            origins.setdefault(key, []).append(reader)
        for path, digest in reader.cfg['source_sha256'].items():
            if Path(path).suffix == '.py':
                need(bindings.get(path) == digest, 'Origin source identity conflict/missing binding')
                required.add(path)
        need(reader.cfg['modelref'] == cfg['modelref'], 'Origin model identity differs')
    merged = checked(cfg['plan'])
    need(isinstance(merged, list) and merged and len(merged) == len(rows)
         and len({poolmod.source_key(r) for r in merged}) == len(merged)
         and {poolmod.source_key(r): r for r in merged} == rows
         and len({r['id'].casefold() for r in merged}) == len(merged), 'Merged plan must exactly deduplicate both origin plans, without metadata conflicts')
    need(type(cfg['targetcount']) is int and cfg['targetcount'] == cfg['input_count'] == len(rows), 'Global union denominator differs')
    manifest = checked(cfg['union_manifest'])
    need(manifest['schema'] == 'caption_union_native_manifest_v4'
         and all(manifest[k] == cfg[k] for k in ('plan', 'origin_config_refs', 'source_root', 'pool_root', 'recipe')), 'Union immutable provenance differs')
    participants = manifest['participants']
    need(participants and len({p['worker_id'] for p in participants}) == len(participants)
         and len({p['config_path'] for p in participants}) == len(participants), 'Duplicate union config/participant')
    occupied = set()
    for participant in participants:
        need(participant['node'] in HOSTS and participant['gpus'] in PAIRS and all(type(g) is int for g in participant['gpus'])
             and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', participant['worker_id']) is not None, 'Invalid union participant node/GPU/identity')
        for gpu in participant['gpus']:
            need((participant['node'], gpu) not in occupied, 'Overlapping union participant GPUs')
            occupied.add((participant['node'], gpu))
        expected = Path(cfg['pool_root']).name + '__' + participant['worker_id']
        need(participant['queue'] == str(BAR / 'queues' / expected) and participant['out'] == str(BAR / expected)
             and participant['draft_out'] == str(BAR / expected / 'archives')
             and Path(participant['config_path']).parent == RT
             and Path(participant['config_path']).name.startswith('caption_union_native_v4_'), 'Participant namespaces differ')
        need(all(participant['worker_id'] != p['worker_id'] and participant['config_path'] != p['config_path']
                 for reader in readers for p in reader.manifest['participants']), 'Union must not impersonate original participant')
    matches = [p for p in participants if p['worker_id'] == cfg['pool_worker_id']]
    need(len(matches) == 1, 'Nonparticipant native union worker')
    participant = matches[0]
    need(all(participant[k] == cfg[k] for k in ('node', 'gpus', 'config_path', 'queue', 'out', 'draft_out'))
         and participant['config_identity_sha256'] == config_identity(cfg), 'Union participant config identity differs')
    need(len(cfg['holder_restored_refs']) == 2 and [r['path'] for r in cfg['holder_restored_refs']]
         == [str(Path(cfg['resource_previous_safe']['path']).parent / ('holder_restored_gpu%d.json' % gpu)) for gpu in cfg['gpus']],
         'Exact restored holder receipt references required')
    for frozen in (cfg['plan'], cfg['union_manifest'], cfg['resource_previous_safe'], cfg['modelref'], *cfg['holder_restored_refs']):
        required.add(frozen['path'])
        need(bindings.get(frozen['path']) == frozen['sha256'], 'Missing native frozen input binding')
    required.update(str(Path(m.__file__).resolve()) for m in list(sys.modules.values())
                    if getattr(m, '__file__', None) and str(Path(m.__file__).resolve()).startswith(str(RT) + '/')
                    and Path(m.__file__).suffix == '.py' and not Path(m.__file__).name.startswith('test_'))
    need(required <= set(bindings), 'All imported runtime sources must be startup SHA-bound')
    for path, digest in bindings.items():
        need(Path(path).is_absolute() and not Path(path).is_symlink() and base.sha(path) == digest, 'Frozen source/input changed: ' + path)
    return manifest, rows, origins, readers


def selected(cfg):
    rows = checked(cfg['plan'])
    return [r['id'] for r in rows], len(rows)


def validate(cfg):
    check_config(cfg)
    need(not Path(cfg['queue']).exists() and not Path(cfg['out']).exists()
         and not namespace(cfg).exists(), 'Existing queue/output/admitted namespace: no automatic restart')
    need(HOSTS[cfg['node']] in h.subprocess.check_output(['hostname', '-I'], text=True).split(), 'Wrong physical execution node')
    need(base.headroom()['headroom'] >= 160 * 2**30 and shutil.disk_usage('/tmp').free >= 8 * 2**30, 'Native 160GiB headroom or 8GiB local staging boundary unavailable')
    safe = checked(cfg['resource_previous_safe'])
    need(all(safe[k] is True for k in ('all_holders_safe', 'locks_released', 'leases_restored')), 'Native predecessor resources not safe')
    need(len(cfg['holders']) == 2 and [o['gpu'] for o in cfg['holders']] == cfg['gpus']
         and len({o['pid'] for o in cfg['holders']}) == len({o['uuid'] for o in cfg['holders']}) == 2, 'Exact distinct fresh ordered holder pair required')
    running = base.apps()
    for gpu, owner in zip(cfg['gpus'], cfg['holders']):
        prior = checked(cfg['holder_restored_refs'][cfg['gpus'].index(gpu)])
        need(prior['uuid'] == owner['uuid'] and all(prior['holder_identity'][k] == owner[k] for k in base.IDENTITY_KEYS), 'Fresh restored holder identity/UUID differs')
        base.check_holder(owner, base.identity(owner['pid']), gpu, base.uuid_for(gpu), running)
        need(not h.lock_snapshot(gpu)['held'], 'Active native GPU lock')
        base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))


def gate(cfg, before=False):
    # Exact existing parent gate code: direct parent argv, UID/cgroup, live child,
    # actual parent-owned flock and lease fds, compute-app subtree are preserved.
    fn = types.FunctionType(h.parent_gate.__code__, dict(h.parent_gate.__globals__, SELF=SELF,
                            base=base, NODE=cfg['node'], GPUS=cfg['gpus'], CVD=cfg['env']['CUDA_VISIBLE_DEVICES']),
                            'native_union_parent_gate', h.parent_gate.__defaults__)
    return fn(cfg, before)


def readonly_foreign_owner_report(cfg):
    _, rows, origins, readers = check_config(cfg)
    blocked, foreign = set(), []
    for key in rows:
        for reader in origins[key]:
            path = reader.source_path(key)
            if not path.exists():
                continue
            blocked.add(key)
            try:
                claim = poolmod.load(path)
                owner = claim['owner']
                matches = [p for p in reader.manifest['participants'] if p['worker_id'] == owner['worker_id']]
                need(len(matches) == 1, 'foreign owner')
                participant = matches[0]
                owner_cfg = checked(owner['config'])
                need(owner['config']['path'] == participant['config_path'] and owner['node'] == participant['node']
                     and owner['gpus'] == participant['gpus']
                     and poolmod.config_identity(owner_cfg) == participant['config_identity_sha256']
                     and owner['identity']['argv'] == owner_cfg['argv']
                     and poolmod.load(reader.root / 'admitted' / (owner['worker_id'] + '.json'))['owner'] == owner
                     and claim == dict(pool_sha256=reader.pool_sha, meta=reader.rows[key], owner=owner, **FLAGS), 'foreign owner/source claim')
            except (OSError, ValueError, KeyError, TypeError):
                foreign.append(str(path))
    return dict(blocked_source_keys=sorted(blocked), foreign_owner_claims=foreign,
                original_resume_recommended_forbidden=bool(foreign), old_launchers_modified=False,
                claims_modified=False, no_automatic_retry=True)


class NativeUnionPool(oldunion.UnionPool):
    def __init__(self, cfg, parent_gate=None):
        self.cfg = cfg
        self.manifest, self.rows, self.origins_by_key, self.readers = check_config(cfg)
        self.root = Path(cfg['pool_root'])
        self.worker_root = namespace(cfg)
        self.pool_sha = cfg['plan']['sha256']
        self.owner, self.parent_gate = None, parent_gate
        self.offsets, self.queue, self.queued = {}, deque(), set()
        self.reserved, self.terminals, self.failed, self.blocked = set(), set(), set(), set()
        self.counts, self.own_reserved, self.valid_global = {}, set(), set()
        self._seen_terminals, self.invalid_receipts = set(), set()
        self.refresh(force=True)
        for path in (self.worker_root / 'batches').glob('*/claim.json'):
            try:
                job = poolmod.load(path)
                self.check_job(job)
                self.own_reserved.update(poolmod.source_key(m) for m in job['metas'])
            except (OSError, ValueError, KeyError, TypeError):
                self.invalid_receipts.add(str(path))
        self.blocked.difference_update(self.own_reserved)

    def batch_path(self, name, worker_id=None):
        need(re.fullmatch(r'batch-[a-f0-9]{32}', name) is not None, 'Invalid batch identity')
        return self.root / 'workers' / (worker_id or self.cfg['pool_worker_id']) / 'batches' / name / 'claim.json'

    def terminal_path(self, key, worker_id=None):
        return self.root / 'workers' / (worker_id or self.cfg['pool_worker_id']) / 'terminals' / (poolmod.digest([self.pool_sha, *key]) + '.json')

    def stopped(self):
        path = Path(self.cfg['stop_file'])
        need(not path.is_symlink(), 'Unsafe STOP symlink')
        return path.exists()

    def admit(self):
        need(callable(self.parent_gate), 'Actual native parent gate required; admission defaults denied')
        proof = self.parent_gate(True)
        need(proof.get('admitted') is True and proof.get('caller_owned') is True, 'Native parent denied')
        identity = proof['child_identity']
        need(identity['pid'] == os.getpid() and identity['uid'] == os.getuid() and identity['argv'] == self.cfg['argv'], 'Actual worker owner identity required')
        need(not self.worker_root.exists(), 'Previously reserved/admitted worker: no automatic restart')
        self.owner = dict(worker_id=self.cfg['pool_worker_id'], node=self.cfg['node'], gpus=self.cfg['gpus'],
                          config=poolmod.ref(self.cfg['config_path']), identity=identity, schema='caption_union_owner_v4')
        poolmod.exclusive(self.worker_root / 'admitted.json', dict(owner=self.owner, parent=proof['parent_identity'],
                          admission=proof['admission_ref'], live=poolmod.ref(Path(self.cfg['queue']) / 'live.json'), **FLAGS))
        return self.owner

    def check_job(self, job):
        need(job['union_pool'] is True and job['pool_sha256'] == self.pool_sha and all(job.get(k) == v for k, v in FLAGS.items()), 'Foreign union batch')
        keys = [poolmod.source_key(m) for m in job['metas']]
        need(0 < len(keys) == len(set(keys)) == len(job['source_claims']) == job['reserved_sources'] <= 3
             and all(self.rows.get(k) == m for k, m in zip(keys, job['metas'])), 'Duplicate/foreign/incomplete batch metadata')
        owner = job['owner']
        matches = [p for p in self.manifest['participants'] if p['worker_id'] == owner['worker_id']]
        need(len(matches) == 1 and owner['schema'] == 'caption_union_owner_v4', 'Foreign/nonparticipant union owner')
        participant = matches[0]
        need(owner['config']['path'] == participant['config_path'] and owner['node'] == participant['node']
             and owner['gpus'] == participant['gpus'], 'Batch owner origin/config/node/GPU differs')
        owner_cfg = checked(owner['config'])
        need(config_identity(owner_cfg) == participant['config_identity_sha256']
             and owner_cfg['union_manifest'] == self.cfg['union_manifest']
             and owner_cfg['plan'] == self.cfg['plan'] and owner_cfg['origin_config_refs'] == self.cfg['origin_config_refs']
             and owner['identity']['argv'] == owner_cfg['argv'], 'Owner frozen config/source identity differs')
        admitted = poolmod.load(self.root / 'workers' / owner['worker_id'] / 'admitted.json')
        need(admitted['owner'] == owner and admitted['admission']['path'] == str(Path(owner_cfg['queue']) / 'admission.json'), 'Actual native admission receipt differs')
        native = checked(admitted['admission'])
        need(native['config_sha256'] == owner['config']['sha256'] and native['manager_sha256'] == BASE_SHA
             and native['producer'] == owner_cfg['producer'] and native['plan'] == owner_cfg['plan']
             and native['node'] == owner['node'] and native['gpus'] == owner['gpus']
             and native['source_sha256'] == owner_cfg['source_sha256'] and native['out'] == owner_cfg['out']
             and native['targetcount'] == owner_cfg['targetcount']
             and native['model_identity_sha256'] == owner_cfg['modelref']['model_identity_sha256']
             and all(native['manager_identity'][k] == admitted['parent'][k] for k in base.IDENTITY_KEYS), 'Native resource admission differs')
        live = checked(admitted['live'])
        need(admitted['live']['path'] == str(Path(owner_cfg['queue']) / 'live.json')
             and all(live['child_identity'][k] == owner['identity'][k] for k in base.IDENTITY_KEYS)
             and owner['identity']['ppid'] == admitted['parent']['pid']
             and owner['identity']['uid'] == admitted['parent']['uid'] == os.getuid()
             and owner['identity']['cgroup'] == admitted['parent']['cgroup']
             and admitted['parent']['argv'] == ['/opt/venv_vllm/bin/python', '-B', str(SELF), '--config', owner_cfg['config_path'],
                 '--config-sha', owner['config']['sha256'], '--manager-sha', BASE_SHA], 'Native parent/live durable identity differs')
        need(poolmod.load(self.batch_path(job['batch_id'], owner['worker_id'])) == job, 'Batch claim identity differs')
        need(job['origins'] == [[str(r.root) for r in self.origins_by_key[k]] for k in keys], 'Batch origin mapping differs')
        for key, meta, refs in zip(keys, job['metas'], job['source_claims']):
            readers = self.origins_by_key[key]
            need(len(refs) == len(readers), 'Incomplete cross-origin mutual exclusion')
            for reader, reference in zip(readers, refs):
                need(reference['path'] == str(reader.source_path(key)) and checked(reference)
                     == dict(pool_sha256=reader.pool_sha, meta=meta, owner=owner, **FLAGS), 'Source claim origin/path/SHA/owner differs')
        return owner_cfg

    def ownership(self, job=None):
        need(self.owner is not None and self.owner['identity']['pid'] == os.getpid(), 'Unadmitted native union owner')
        need(poolmod.load(self.worker_root / 'admitted.json')['owner'] == self.owner, 'Union admitted owner changed')
        if job is not None:
            self.check_job(job)
            need(job['owner'] == self.owner and not (self.batch_path(job['batch_id']).parent / 'failure.json').exists(), 'Batch owner/failure differs')
        return True

    def reserve(self):
        need(callable(self.parent_gate), 'Actual native parent gate required before claiming')
        proof = self.parent_gate(False)
        need(proof.get('admitted') is True and proof.get('caller_owned') is True, 'Native parent denied before claiming')
        self.ownership()
        if self.stopped():
            return None
        self.update_ready()
        chosen, claims = [], []
        while self.queue and len(chosen) < 3:
            if self.stopped():
                break
            key = self.queue.popleft()
            if key in self.reserved or key in self.valid_global:
                continue
            readers = self.origins_by_key[key]
            if any(r.source_path(key).exists() or (r.root / 'terminals' / r.source_path(key).name).exists() for r in readers):
                self.reserved.add(key)
                self.blocked.add(key)
                continue
            refs = []
            try:
                for reader in readers:
                    refs.append(poolmod.exclusive(reader.source_path(key), dict(pool_sha256=reader.pool_sha,
                                meta=self.rows[key], owner=self.owner, **FLAGS)))
            except FileExistsError:
                self.reserved.add(key)
                self.blocked.add(key)
                continue  # Any partial claim remains permanently sealed; never undo/retry.
            self.reserved.add(key)
            self.own_reserved.add(key)
            chosen.append(self.rows[key])
            claims.append(refs)
        if not chosen:
            return None
        name = 'batch-' + uuid.uuid4().hex
        job = dict(batch_id=name, union_pool=True, pool_sha256=self.pool_sha, metas=chosen, source_claims=claims,
                   owner=self.owner, reserved_sources=len(chosen),
                   origins=[[str(r.root) for r in self.origins_by_key[poolmod.source_key(m)]] for m in chosen], **FLAGS)
        poolmod.exclusive(self.batch_path(name), job)
        return job

    def fail(self, job, error, cache=None):
        path = self.batch_path(job['batch_id']).parent / 'failure.json'
        if not path.exists():
            poolmod.exclusive(path, dict(reason=repr(error)[:4000], local_cache_path=str(cache) if cache else None,
                              budget_not_reset=True, no_automatic_retry=True, owner=job['owner'], **FLAGS))
        self.failed.update(poolmod.source_key(m) for m in job['metas'])

    def verify_commit(self, job, commit):
        owner_cfg = self.check_job(job)
        archive = Path(owner_cfg['draft_out']) / (job['batch_id'] + '.tar')
        reference = poolmod.ref(archive.with_suffix('.committed.json'))
        need(checked(reference) == commit and commit['reliable_writeback'] is True and commit['batch_id'] == job['batch_id']
             and commit['archive_path'] == str(archive) and not archive.is_symlink()
             and archive.is_file() and type(commit['archive_bytes']) is int
             and archive.stat().st_size == commit['archive_bytes'], 'Durable exact archive/commit path/size differs')
        drafts = mgr.archived_drafts(job, commit)
        need(all(d['status'] in STATUSES and all(d.get(k) == v for k, v in FLAGS.items()) for d in drafts), 'Archived draft status/review flags differ')
        return reference, drafts

    def receipt(self, job, source_ref, draft, commit, commit_ref):
        key = poolmod.source_key(draft)
        return dict(id=key[0], split=key[1], status=draft['status'], pool_sha256=self.pool_sha,
                    batch_id=job['batch_id'], owner=job['owner'], source_claims=source_ref,
                    batch_claim=poolmod.ref(self.batch_path(job['batch_id'], job['owner']['worker_id'])), commit=commit_ref,
                    origin_config_refs=self.cfg['origin_config_refs'], archive_path=commit['archive_path'],
                    archive_member=job['batch_id'] + '/caption_two_stage_results/drafts/' + key[1] + '/' + key[0] + '/draft.json', **FLAGS)

    def verify_terminal(self, path):
        value = poolmod.load(path)
        key = poolmod.source_key(value)
        need(key in self.rows and path == self.terminal_path(key, value['owner']['worker_id']) and not path.is_symlink(), 'Foreign union terminal path')
        job = checked(value['batch_claim'])
        need(value['batch_claim']['path'] == str(self.batch_path(value['batch_id'], value['owner']['worker_id'])), 'Terminal batch reference path differs')
        commit_ref, drafts = self.verify_commit(job, checked(value['commit']))
        pairs = [(m, s) for m, s in zip(job['metas'], job['source_claims']) if poolmod.source_key(m) == key]
        matching = [d for d in drafts if poolmod.source_key(d) == key]
        need(len(pairs) == len(matching) == 1 and value == self.receipt(job, pairs[0][1], matching[0], checked(commit_ref), commit_ref), 'Conflicting terminal/commit/archive evidence')
        return key, value

    def publish(self, job, commit, drafts, recovery=False):
        need(not recovery, 'No automatic recovery publication; use explicit read-only recovery check')
        self.ownership(job)
        commit_ref, archived = self.verify_commit(job, commit)
        results = {poolmod.source_key(d): d for d in drafts}
        need(len(results) == len(drafts) == len(job['metas']) and results == {poolmod.source_key(d): d for d in archived}, 'Draft/actual archive mapping differs')
        for meta, source_ref in zip(job['metas'], job['source_claims']):
            key = poolmod.source_key(meta)
            draft = results[key]
            receipt = self.receipt(job, source_ref, draft, commit, commit_ref)
            path = self.terminal_path(key)
            if path.exists():
                need(poolmod.load(path) == receipt, 'Conflicting duplicate commit receipt')
            else:
                poolmod.exclusive(path, receipt)
            if key not in self.terminals:
                self.terminals.add(key)
                self.counts[draft['status']] = self.counts.get(draft['status'], 0) + 1
            if draft['status'] == 'draft_needs_regeneration':
                self.failed.add(key)
            self._seen_terminals.add(path)

    def refresh(self, force=False):
        # Only this worker's small namespace, never origins or other workers.
        # Failed/torn receipts remain eligible for a later explicit re-read.
        for path in (self.worker_root / 'terminals').glob('*.json'):
            if path in self._seen_terminals and not force:
                continue
            try:
                key, value = self.verify_terminal(path)
                self._seen_terminals.add(path)
                self.invalid_receipts.discard(str(path))
                if key not in self.terminals:
                    self.terminals.add(key)
                    self.counts[value['status']] = self.counts.get(value['status'], 0) + 1
                if value['status'] == 'draft_needs_regeneration':
                    self.failed.add(key)
            except (OSError, ValueError, KeyError, TypeError):
                self.invalid_receipts.add(str(path))

    def progress(self):
        # Other workers/origin terminal history is intentionally unknown. O_EXCL
        # on ready-key origin claims enforces exclusion without any history scan.
        worker = dict(scope='worker', worker_id=self.cfg['pool_worker_id'], reserved=len(self.own_reserved),
                      committed=len(self.terminals), processed=len(self.terminals), failed=len(self.failed), counts=dict(self.counts),
                      completed_source_keys=sorted(self.terminals), pool_sha256=self.pool_sha, **FLAGS)
        progress = dict(scope_pool=True, N=len(self.rows), global_terminal=None, global_completeness_known=False,
                        complete=False, observed_worker_terminal=len(self.terminals),
                        blocked=len(self.blocked), blocked_source_keys=sorted(self.blocked),
                        invalid_receipts=sorted(self.invalid_receipts), reserved_uncommitted=len(self.own_reserved - self.terminals),
                        pool_sha256=self.pool_sha, **FLAGS)
        return worker, progress


def check_completeness(cfg):
    """Explicit offline archive audit. Never called by production/main/progress."""
    pool = NativeUnionPool(cfg)
    evidence, invalid, conflicts = {}, set(), set()
    def accept(key, receipt, draft, path):
        fingerprint = poolmod.digest(dict(draft=draft, owner=receipt['owner']))
        previous = evidence.get(key)
        if previous is not None and previous[0] != fingerprint:
            conflicts.add(key)
            invalid.update((previous[1], str(path)))
        else:
            evidence[key] = (fingerprint, str(path))
    for reader in pool.readers:
        for path in (reader.root / 'terminals').glob('*.json'):
            try:
                receipt = reader.verify_terminal(poolmod.ref(path))
                job = checked(receipt['batch_claim'])
                drafts = mgr.archived_drafts(job, checked(receipt['commit']))
                key = poolmod.source_key(receipt)
                matching = [d for d in drafts if poolmod.source_key(d) == key]
                need(len(matching) == 1 and matching[0]['status'] == receipt['status']
                     and all(matching[0].get(k) == v for k, v in FLAGS.items()), 'Origin archive draft status/FLAGS differs')
                accept(key, receipt, matching[0], path)
            except (OSError, ValueError, KeyError, TypeError):
                invalid.add(str(path))
    for participant in pool.manifest['participants']:
        for path in (pool.root / 'workers' / participant['worker_id'] / 'terminals').glob('*.json'):
            try:
                key, receipt = pool.verify_terminal(path)
                _, drafts = pool.verify_commit(checked(receipt['batch_claim']), checked(receipt['commit']))
                draft = next(d for d in drafts if poolmod.source_key(d) == key)
                accept(key, receipt, draft, path)
            except (OSError, ValueError, KeyError, TypeError):
                invalid.add(str(path))
    valid = set(evidence) - conflicts
    return dict(N=len(pool.rows), global_terminal=len(valid), global_completeness_known=True,
                complete=len(valid) == len(pool.rows) and not invalid, conflicting_source_keys=sorted(conflicts),
                invalid_receipts=sorted(invalid), model_calls=0, claims_modified=False, offline_only=True)


def readonly_recovery_check(cfg, batch_id):
    pool = NativeUnionPool(cfg)
    job = poolmod.load(pool.batch_path(batch_id))
    owner_cfg = pool.check_job(job)
    commit = poolmod.load(Path(owner_cfg['draft_out']) / (batch_id + '.committed.json'))
    reference, drafts = pool.verify_commit(job, commit)
    return dict(batch_id=batch_id, durable_commit=reference, recoverable_source_keys=sorted(poolmod.source_key(d) for d in drafts),
                model_calls=0, claims_modified=False, receipts_published=False)


def run(pool, runtime_factory=None, stage_source=None, generate=None, writeback=None, poll_seconds=20., local_parent='/tmp'):
    cfg = pool.cfg
    need(callable(pool.parent_gate) and pool.ownership() and pool.parent_gate(False).get('admitted') is True, 'Actual native GPU/pool admission required')
    runtime_factory = runtime_factory or oldunion.model_runtime
    stage_source, generate, writeback = stage_source or local.stage_source, generate or local.light.generate_batch, writeback or local.writeback
    args, runtime = mgr.args_for(cfg), None
    out, started = Path(cfg['draft_out']), time.perf_counter()
    poolmod.mkdir(out)

    def owned(job, before=False):
        need(pool.parent_gate(before).get('admitted') is True and pool.ownership(job), 'Actual GPU/pool gate denied')
        return True

    def copy_batch(job):
        root = None
        try:
            owned(job)
            root = Path(tempfile.mkdtemp(prefix=job['batch_id'] + '-', dir=local_parent))
            poolmod.exclusive(pool.batch_path(job['batch_id']).parent / 'cache.json', dict(local_cache_path=str(root), owner=pool.owner, **FLAGS))
            entries, times, failures = [], [], []
            for meta in job['metas']:
                try:
                    entry, timing = stage_source(cfg['source_root'], meta, root)
                    entries.append(entry)
                    times.append(timing)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    draft = dict(id=meta['id'], split=meta['split'], status='draft_needs_regeneration', reason=repr(error), **FLAGS)
                    poolmod.exclusive(root / 'caption_two_stage_results/drafts' / meta['split'] / meta['id'] / 'draft.json', draft)
                    failures.append(draft)
            return job, root, entries, times, failures
        except BaseException as error:
            pool.fail(job, error, root)
            raise

    def checkpoint():
        worker, progress = pool.progress()
        worker.update(node=cfg['node'], gpus=cfg['gpus'], selected=len(pool.rows), scope_pool=False,
                      UTC=time.strftime('%FT%TZ', time.gmtime()), wall_seconds=time.perf_counter() - started,
                      local_inputs_intermediates_results=True, copy_threads=1, max_inflight_batches=2, source_batch_size=3)
        poolmod.atomic(out / 'progress.json', worker)
        return worker, progress

    def finish(worker, progress, status):
        summary = dict(worker, scope_pool=True, status=status, complete=progress['complete'] if status == 'complete' else False,
                       global_terminal=progress['global_terminal'], N=progress['N'], blocked=progress['blocked'],
                       blocked_source_keys=progress['blocked_source_keys'], invalid_receipts=progress['invalid_receipts'],
                       uncommitted_owned=len(pool.own_reserved - pool.terminals), no_automatic_retry=True)
        need(summary['uncommitted_owned'] == 0, 'Cannot drain while an owned prefetched batch is uncommitted')
        poolmod.exclusive(Path(cfg['out']) / 'summary.json', summary)
        return summary

    with ThreadPoolExecutor(max_workers=1) as copier:
        pending = None
        while True:
            worker_progress, global_progress = checkpoint()
            if pending is None and global_progress['complete']:
                return finish(worker_progress, global_progress, 'complete')
            if pending is None and pool.stopped():
                return finish(worker_progress, global_progress, 'drained_not_complete')
            if pending is None:
                job = pool.reserve()
                if job is None:
                    time.sleep(poll_seconds)
                    continue  # Queue-empty/all-claimed is never completion.
                if runtime is None:
                    try:
                        owned(job, True)
                        runtime = runtime_factory(cfg)
                    except BaseException as error:
                        pool.fail(job, error)
                        raise
                pending = copier.submit(copy_batch, job)
            job, root, entries, times, failures = pending.result()
            pending = None
            try:
                next_job = pool.reserve()
                if next_job is not None:
                    pending = copier.submit(copy_batch, next_job)
                owned(job)
                result = generate(entries, args, runtime, root / 'caption_two_stage_results', lambda a, r: owned(job)) if entries else dict(drafts=[], new_model_calls=0)
                result['drafts'].extend(failures)
                poolmod.exclusive(root / 'source_batch.json', dict(metas=job['metas'], source_claims=job['source_claims'],
                                  owner=pool.owner, copy_timings=times, new_model_calls=result['new_model_calls'], **FLAGS))
                commit = writeback(root, out, job['batch_id'])
                pool.publish(job, commit, mgr.archived_drafts(job, commit))
                checkpoint()
            except BaseException as error:
                pool.fail(job, error, root)
                raise
            shutil.rmtree(root)
            root.with_suffix('.tar').unlink()


def summaries(cfg, out):
    value = poolmod.load(out / 'summary.json')
    pool = NativeUnionPool(cfg)
    _, progress = pool.progress()
    need(value['worker_id'] == cfg['pool_worker_id'] and value['pool_sha256'] == cfg['plan']['sha256']
         and value['N'] == progress['N'] == cfg['targetcount']
         and value['global_terminal'] is None and progress['global_terminal'] is None
         and value['processed'] == value['committed'] == len(pool.terminals)
         and value['counts'] == pool.counts and value['reserved'] == len(pool.own_reserved)
         and value['uncommitted_owned'] == len(pool.own_reserved - pool.terminals) == 0, 'Worker summary/durable union evidence differs')
    need(value['status'] == 'drained_not_complete' and value['complete'] is False
         and Path(cfg['stop_file']).exists(), 'Production accepts only explicit drained STOP; global completeness is unknown')
    return {str(out / 'summary.json'): dict(summary=dict(stage='run', total_source_count=cfg['targetcount'],
            selected_source_count=cfg['targetcount'], processed_source_count=value['processed'], plan_sha256=cfg['plan']['sha256'],
            complete=value['complete'], status=value['status'], scope_pool=True, global_terminal=value['global_terminal'], N=value['N']),
            selection=dict(selected_ids=selected(cfg)[0]), draft_counts=value['counts'], blocked=value['blocked'], **FLAGS)}


def worker(path):
    cfg = poolmod.load(path)
    cfg.update(_config_path=str(Path(path).resolve()), _config_sha=base.sha(path))
    need(cfg['config_path'] == cfg['_config_path'], 'Worker config path differs')
    pool = NativeUnionPool(cfg, lambda before=False: gate(cfg, before))
    pool.admit()
    return run(pool)


def main():
    if '--check-only' in sys.argv or '--check-completeness' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', required=True)
        parser.add_argument('--config-sha', required=True)
        parser.add_argument('--manager-sha', required=True)
        modes = parser.add_mutually_exclusive_group(required=True)
        modes.add_argument('--check-only', action='store_true')
        modes.add_argument('--check-completeness', action='store_true')
        args = parser.parse_args()
        need(args.manager_sha == BASE_SHA and base.sha(base.__file__) == BASE_SHA
             and base.sha(args.config) == args.config_sha, 'Check-only native/config SHA differs')
        cfg = poolmod.load(args.config)
        need(cfg['config_path'] == str(Path(args.config).resolve()), 'Check-only config path differs')
        if args.check_completeness:
            result = check_completeness(cfg)
        else:
            validate(cfg)
            base.selected_sources = selected
            base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
            base.check_model(cfg)
            result = dict(status='check_only_passed', node=cfg['node'], gpus=cfg['gpus'], N=cfg['targetcount'],
                          model_calls=0, claims_modified=False, signals_sent=False, queue_created=False,
                          global_completeness_known=False, complete=False)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        parser = argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--config', required=True)
        return worker(parser.parse_args().config)
    base.validate_pair, base.selected_sources, base.stage_summaries = validate, selected, summaries
    return base.main()


if __name__ == '__main__':
    main()
