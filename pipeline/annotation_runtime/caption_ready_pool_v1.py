"""Immutable ready union, exclusive source/batch budgets and durable terminal receipts."""
from collections import deque
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

FLAGS = dict(training_eligible=False, semantic_approval=False, quality_approved=False,
             needs_visual_revision=True, pixel_quality_review_completed=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdir(path):
    path = Path(path)
    if not path.exists():
        mkdir(path.parent)
        try:
            path.mkdir()
        except FileExistsError:
            pass
        syncdir(path.parent)
    need(path.is_dir() and not path.is_symlink(), 'Unsafe pool directory')


def exclusive(path, value):
    path = Path(path)
    mkdir(path.parent)
    # A partial xb file is itself a permanently consumed budget, never replaced.
    with path.open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    syncdir(path.parent)
    return ref(path)


def atomic(path, value):
    path = Path(path)
    mkdir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    syncdir(path.parent)


def source_key(row):
    sid, split = row['id'], row['split']
    need(isinstance(sid, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sid)
         and split in ('train', 'val', 'test'), 'Nonportable source ID/split')
    return sid, split


def ready_record(path, line, offset):
    try:
        return json.loads(line), offset + len(line)
    except json.JSONDecodeError:
        # A fresh read at the same offset handles an observed transient FUSE view.
        with Path(path).open('rb') as stream:
            stream.seek(offset)
            current = stream.readline()
            need(current.endswith(bytes([10])), 'Ready record reread is incomplete; no source reserved')
            return json.loads(current), stream.tell()


def config_identity(cfg):
    value = copy.deepcopy(cfg)
    path = value['pool_manifest']['path']
    value['pool_manifest'].pop('sha256', None)
    value.get('source_sha256', {}).pop(path, None)
    for key in ('_config_path', '_config_sha'):
        value.pop(key, None)
    return digest(value)


def check_manifest(cfg):
    manifest = load(cfg['pool_manifest']['path'])
    need(ref(cfg['pool_manifest']['path']) == cfg['pool_manifest'], 'Pool manifest SHA differs')
    need(manifest['schema'] == 'caption_ready_pool_v1' and manifest['plan'] == cfg['plan']
         and manifest['source_handoff'] == cfg['source_handoff']
         and manifest['source_root'] == cfg['source_root']
         and str(Path(manifest['pool_root']).resolve()) == cfg['pool_root'], 'Pool immutable binding differs')
    need(ref(cfg['plan']['path']) == cfg['plan'] and ref(cfg['source_handoff']['path']) == cfg['source_handoff'],
         'Pool plan/handoff SHA differs')
    rows = load(cfg['plan']['path'])
    keys = [source_key(row) for row in rows]
    need(isinstance(rows, list) and rows and len(keys) == len(set(keys))
         and len({k[0].casefold() for k in keys}) == len(keys), 'Pool array/portable unique IDs required')
    handoff = load(cfg['source_handoff']['path'])
    allowed = {source_key(row): row for row in handoff['remaining_sources']}
    need(handoff['old_worker_tree_absent'] is True and handoff['original_config_resume_forbidden'] is True
         and all(allowed.get(source_key(row)) == row for row in rows), 'Unsafe union source handoff')
    participants = manifest['participants']
    need(participants and len({p['worker_id'] for p in participants}) == len(participants)
         and len({p['config_path'] for p in participants}) == len(participants), 'Duplicate pool participant')
    occupied = set()
    for p in participants:
        need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', p['worker_id'])
             and p['gpus'] in ([0, 1], [2, 3], [4, 5], [6, 7]), 'Invalid pool participant')
        for gpu in p['gpus']:
            need((p['node'], gpu) not in occupied, 'Overlapping pool GPU participants')
            occupied.add((p['node'], gpu))
    matches = [p for p in participants if p['worker_id'] == cfg['pool_worker_id']]
    need(len(matches) == 1, 'Nonparticipant worker refused')
    participant = matches[0]
    need(participant['node'] == cfg['node'] and participant['gpus'] == cfg['gpus']
         and participant['config_path'] == cfg['config_path']
         and participant['config_identity_sha256'] == config_identity(cfg), 'Participant config/node/GPU differs')
    need(cfg['targetcount'] == cfg.get('input_count', len(rows)) == len(rows), 'Pool eligibility count differs')
    return manifest, rows


class Pool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.manifest, rows = check_manifest(cfg)
        self.root = Path(cfg['pool_root'])
        self.pool_sha = cfg['plan']['sha256']
        self.rows = {source_key(row): row for row in rows}
        self.ready, self.reserved, self.terminals, self.owners = set(), set(), {}, {}
        self.queue, self.queued = deque(), set()
        self.unready_keys, self.ready_unclaimed = set(self.rows), set()
        self.worker_reserved, self.worker_terminal, self.worker_failed, self.worker_counts = {}, {}, {}, {}
        self.offsets = {}
        self.owner = None
        for name in ('sources', 'batches', 'terminals', 'admitted'):
            mkdir(self.root / name)
        source_paths = {self.source_path(key): key for key in self.rows}
        with self.lock():
            for path in (self.root / 'sources').glob('*.json'):
                # Recover claims missing their journal after source -> batch interruption.
                need(path in source_paths, 'Foreign source claim in pool')
                key = source_paths[path]
                self.reserved.add(key)
                self.unready_keys.discard(key)
                try:
                    value = load(path)
                    need(value['pool_sha256'] == self.pool_sha and source_key(value['meta']) == key, 'Foreign claim')
                    self.claim_event(value)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    pass  # Permanently sealed; no receipt means blocked, never done.
            self.update_journal('claims.jsonl', self.claim_event)
            self.update_journal('terminal_index.jsonl', self.terminal_event)
            for path in (self.root / 'terminals').glob('*.json'):
                receipt = self.verify_terminal(ref(path))
                key = source_key(receipt)
                if key not in self.terminals:
                    self.append('terminal_index.jsonl', dict(receipt=ref(path)))
                    self.accept_terminal(receipt)

    @contextmanager
    def lock(self):
        path = self.root / 'pool.lock'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size == 0, 'Unsafe pool FLOCK')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def append(self, name, value):
        path = self.root / name
        if path.exists() and path.stat().st_size:
            with path.open('rb') as stream:
                stream.seek(-1, os.SEEK_END)
                need(stream.read(1) == b'\n', 'Torn pool journal sealed; coordinator required')
        with path.open('ab') as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        syncdir(path.parent)

    def update_journal(self, name, callback):
        path = self.root / name
        if not path.exists():
            return
        offset = self.offsets.get(str(path), 0)
        need(path.stat().st_size >= offset, 'Append-only index shrank')
        with path.open('rb') as stream:
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line:
                    break
                need(line.endswith(b'\n'), 'Torn pool journal sealed; coordinator required')
                callback(json.loads(line))
                self.offsets[str(path)] = stream.tell()

    def claim_event(self, value):
        key = source_key(value['meta'])
        need(key in self.rows and value['pool_sha256'] == self.pool_sha, 'Foreign claim journal')
        self.reserved.add(key)
        self.unready_keys.discard(key)
        self.ready_unclaimed.discard(key)
        self.owners[key] = value['owner']
        worker = value['owner']['worker_id']
        self.worker_reserved.setdefault(worker, set()).add(key)
        if value.get('batch_failed') is True:
            self.worker_failed.setdefault(worker, set()).add(key)

    def terminal_event(self, value):
        receipt = self.verify_terminal(value['receipt'])
        key = source_key(receipt)
        need(key not in self.terminals or self.terminals[key] == receipt, 'Conflicting terminal')
        self.accept_terminal(receipt)

    def accept_terminal(self, receipt):
        key = source_key(receipt)
        if key in self.terminals:
            need(self.terminals[key] == receipt, 'Conflicting terminal receipt')
            return
        self.terminals[key] = receipt
        worker = receipt['owner']['worker_id']
        self.worker_terminal.setdefault(worker, set()).add(key)
        counts = self.worker_counts.setdefault(worker, {})
        counts[receipt['status']] = counts.get(receipt['status'], 0) + 1
        if receipt['status'] == 'draft_needs_regeneration':
            self.worker_failed.setdefault(worker, set()).add(key)

    def source_path(self, key):
        return self.root / 'sources' / (digest([self.pool_sha, *key]) + '.json')

    def batch_path(self, name):
        need(re.fullmatch(r'batch-[a-f0-9]{32}', name), 'Invalid batch identity')
        return self.root / 'batches' / name / 'claim.json'

    def admit(self, gate):
        need(callable(gate), 'Pool defaults to denied admission')
        proof = gate(True)
        need(proof.get('admitted') is True, 'GPU parent gate denied')
        identity = proof['child_identity']
        need(identity['pid'] == os.getpid() and identity['uid'] == os.getuid()
             and identity['argv'] == self.cfg['argv'], 'Actual worker owner required')
        owner = dict(worker_id=self.cfg['pool_worker_id'], node=self.cfg['node'], gpus=self.cfg['gpus'],
                     config=ref(self.cfg['config_path']), identity=identity)
        with self.lock():
            exclusive(self.root / 'admitted' / (self.cfg['pool_worker_id'] + '.json'),
                      dict(owner=owner, parent=proof['parent_identity'], **FLAGS))
        self.owner = owner
        return self.owner

    def check_job(self, job):
        need(isinstance(job, dict) and job['pool_sha256'] == self.pool_sha
             and 0 < len(job['metas']) == len(job['source_claims']) == job['reserved_sources'] <= 3,
             'Incomplete batch budget')
        keys = [source_key(meta) for meta in job['metas']]
        need(len(keys) == len(set(keys)) and all(self.rows.get(key) == meta for key, meta in zip(keys, job['metas'])),
             'Duplicate or foreign batch metadata')
        need(load(self.batch_path(job['batch_id'])) == job, 'Batch claim identity differs')
        owner = job['owner']
        participants = [p for p in self.manifest['participants'] if p['worker_id'] == owner['worker_id']]
        need(len(participants) == 1, 'Nonparticipant batch owner')
        participant = participants[0]
        need(owner['config']['path'] == participant['config_path'] and owner['node'] == participant['node']
             and owner['gpus'] == participant['gpus'] and ref(participant['config_path']) == owner['config'],
             'Batch owner config/node/GPU differs')
        owner_cfg = load(participant['config_path'])
        need(config_identity(owner_cfg) == participant['config_identity_sha256']
             and owner_cfg['pool_worker_id'] == owner['worker_id'] and owner_cfg['node'] == owner['node']
             and owner_cfg['gpus'] == owner['gpus']
             and owner['identity']['argv'] == owner_cfg['argv']
             and load(self.root / 'admitted' / (owner['worker_id'] + '.json'))['owner'] == owner,
             'Batch owner actual admission differs')
        for key, meta, source_ref in zip(keys, job['metas'], job['source_claims']):
            need(source_ref == ref(self.source_path(key))
                 and load(source_ref['path']) == dict(pool_sha256=self.pool_sha, meta=meta, owner=owner, **FLAGS),
                 'Batch source claim differs')
        return owner_cfg

    def ownership(self, job=None):
        need(self.owner is not None, 'Unadmitted pool worker')
        record = load(self.root / 'admitted' / (self.cfg['pool_worker_id'] + '.json'))
        need(record['owner'] == self.owner and self.owner['identity']['pid'] == os.getpid(), 'Pool ownership differs')
        if job is not None:
            self.check_job(job)
            need(job['owner'] == self.owner and not (self.batch_path(job['batch_id']).parent / 'failure.json').exists(),
                 'Batch owner/failure differs')
        return True

    def update_ready(self):
        root = Path(self.cfg['source_root'])
        paths = [root / 'ready' / ('producer-%04d.jsonl' % i) for i in range(4)] if (root / 'ready').is_dir() else [root / 'ready.jsonl']
        for path in paths:
            if not path.exists():
                continue
            offset = self.offsets.get(str(path), 0)
            need(path.stat().st_size >= offset, 'Ready journal shrank')
            with path.open('rb') as stream:
                stream.seek(offset)
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    row, next_offset = ready_record(path, line, position)
                    if next_offset != stream.tell():
                        stream.seek(next_offset)
                    key = source_key(row)
                    if key in self.rows and row.get('status', 'ready') == 'ready':
                        self.ready.add(key)
                        self.unready_keys.discard(key)
                        if key not in self.reserved:
                            self.ready_unclaimed.add(key)
                        if key not in self.queued and key not in self.reserved:
                            self.queue.append(key)
                            self.queued.add(key)
                    self.offsets[str(path)] = stream.tell()

    def reserve(self, after_source=None):
        self.ownership()
        with self.lock():
            self.update_journal('claims.jsonl', self.claim_event)
            self.update_journal('terminal_index.jsonl', self.terminal_event)
            self.update_ready()
            chosen, refs = [], []
            while self.queue and len(chosen) < 3:
                key = self.queue.popleft()
                if key in self.reserved:
                    continue
                path = self.source_path(key)
                value = dict(pool_sha256=self.pool_sha, meta=self.rows[key], owner=self.owner, **FLAGS)
                try:
                    claim_ref = exclusive(path, value)
                except FileExistsError:
                    self.reserved.add(key)
                    self.unready_keys.discard(key)
                    self.ready_unclaimed.discard(key)
                    continue
                self.claim_event(value)
                self.append('claims.jsonl', value)
                chosen.append(self.rows[key])
                refs.append(claim_ref)
                if after_source is not None:
                    after_source(value)
            if not chosen:
                return None
            name = 'batch-' + uuid.uuid4().hex
            job = dict(batch_id=name, pool_sha256=self.pool_sha, metas=chosen, source_claims=refs,
                       owner=self.owner, reserved_sources=len(chosen), **FLAGS)
            exclusive(self.batch_path(name), job)
            self.ownership(job)
            return job

    def fail(self, job, error, cache=None):
        exclusive(self.batch_path(job['batch_id']).parent / 'failure.json',
                  dict(reason=repr(error), local_cache_path=str(cache) if cache else None,
                       budget_not_reset=True, no_automatic_retry=True, owner=self.owner, **FLAGS))
        with self.lock():
            for meta in job['metas']:
                event = dict(pool_sha256=self.pool_sha, meta=meta, owner=job['owner'], batch_failed=True, **FLAGS)
                self.append('claims.jsonl', event)
                self.claim_event(event)

    def verify_terminal(self, reference):
        path = Path(reference['path'])
        need(path.parent == self.root / 'terminals' and not path.is_symlink() and ref(path) == reference, 'Terminal receipt SHA/path differs')
        value = load(path)
        key = source_key(value)
        need(key in self.rows and value['pool_sha256'] == self.pool_sha
             and path == self.root / 'terminals' / self.source_path(key).name
             and all(value.get(k) == v for k, v in FLAGS.items()), 'Terminal identity/FLAGS differs')
        source_ref, batch_ref, commit_ref = (value[k] for k in ('source_claim', 'batch_claim', 'commit'))
        need(source_ref['path'] == str(self.source_path(key)) and batch_ref['path'] == str(self.batch_path(value['batch_id']))
             and all(ref(r['path']) == r for r in (source_ref, batch_ref, commit_ref)), 'Terminal claim/commit refs differ')
        claim, batch, commit = (load(r['path']) for r in (source_ref, batch_ref, commit_ref))
        self.check_job(batch)
        need(claim['owner'] == batch['owner'] == value['owner'] and claim['meta'] == self.rows[key]
             and source_ref in batch['source_claims'] and self.rows[key] in batch['metas']
             and batch['pool_sha256'] == self.pool_sha and commit['reliable_writeback'] is True
             and commit['batch_id'] == batch['batch_id'] == value['batch_id'], 'Terminal owner/batch/writeback differs')
        archive = Path(value['archive_path'])
        worker_cfg = load(value['owner']['config']['path'])
        participants = [p for p in self.manifest['participants'] if p['worker_id'] == value['owner']['worker_id']]
        need(len(participants) == 1 and participants[0]['config_path'] == value['owner']['config']['path']
             and participants[0]['node'] == value['owner']['node'] == worker_cfg['node']
             and participants[0]['gpus'] == value['owner']['gpus'] == worker_cfg['gpus']
             and participants[0]['config_identity_sha256'] == config_identity(worker_cfg)
             and load(self.root / 'admitted' / (value['owner']['worker_id'] + '.json'))['owner'] == value['owner'],
             'Terminal participant/admitted identity differs')
        need(ref(value['owner']['config']['path']) == value['owner']['config']
             and archive == Path(worker_cfg['draft_out']) / (value['batch_id'] + '.tar')
             and commit_ref['path'] == str(archive.with_suffix('.committed.json'))
             and commit['archive_path'] == str(archive) and not archive.is_symlink()
             and archive.stat().st_size == commit['archive_bytes'], 'Terminal archive publication differs')
        expected = value['batch_id'] + '/caption_two_stage_results/drafts/' + key[1] + '/' + key[0] + '/draft.json'
        need(value['archive_member'] == expected and value['status'] in ('draft_generated', 'draft_partial', 'draft_needs_regeneration'), 'Terminal archive member/status differs')
        need(key in self.reserved, 'Terminal without source budget')
        return value

    def publish(self, job, commit, drafts, recovery=False):
        if not recovery:
            self.ownership(job)
        else:
            self.check_job(job)
        need(commit['reliable_writeback'] is True and commit['batch_id'] == job['batch_id'], 'Durable writeback required')
        results = {source_key(d): d for d in drafts}
        need(len(results) == len(drafts) == len(job['metas']) and set(results) == {source_key(r) for r in job['metas']}, 'Exactly one terminal per reserved source required')
        batch_ref = ref(self.batch_path(job['batch_id']))
        commit_ref = ref(Path(commit['archive_path']).with_suffix('.committed.json'))
        with self.lock():
            for meta, source_ref in zip(job['metas'], job['source_claims']):
                key = source_key(meta)
                draft = results[key]
                need(all(draft.get(k) == v for k, v in FLAGS.items()), 'Draft training/review flags changed')
                receipt = dict(id=key[0], split=key[1], status=draft['status'], pool_sha256=self.pool_sha,
                    batch_id=job['batch_id'], owner=job['owner'], source_claim=source_ref, batch_claim=batch_ref,
                    commit=commit_ref, archive_path=commit['archive_path'], archive_member=job['batch_id'] +
                    '/caption_two_stage_results/drafts/' + key[1] + '/' + key[0] + '/draft.json', **FLAGS)
                path = self.root / 'terminals' / self.source_path(key).name
                if path.exists():
                    need(load(path) == receipt, 'Conflicting source terminal receipt')
                    receipt_ref = ref(path)
                else:
                    receipt_ref = exclusive(path, receipt)
                self.verify_terminal(receipt_ref)
                self.append('terminal_index.jsonl', dict(receipt=receipt_ref))
                self.accept_terminal(receipt)

    def progress(self):
        with self.lock():
            self.update_journal('claims.jsonl', self.claim_event)
            self.update_journal('terminal_index.jsonl', self.terminal_event)
            self.update_ready()
            worker_id = self.cfg['pool_worker_id']
            owned = self.worker_reserved.get(worker_id, set())
            committed = self.worker_terminal.get(worker_id, set())
            counts = dict(self.worker_counts.get(worker_id, {}))
            worker = dict(scope='worker', worker_id=self.cfg['pool_worker_id'], reserved=len(owned),
                committed=len(committed), processed=len(committed), failed=len(self.worker_failed.get(worker_id, set())),
                counts=counts, completed_source_keys=sorted(committed), pool_sha256=self.pool_sha, **FLAGS)
            pool = dict(scope_pool=True, N=len(self.rows), global_terminal=len(self.terminals), reserved=len(self.reserved),
                reserved_uncommitted=len(self.reserved) - len(self.terminals), unready=len(self.unready_keys),
                ready_unclaimed=len(self.ready_unclaimed), complete=len(self.terminals) == len(self.rows), pool_sha256=self.pool_sha, **FLAGS)
            atomic(self.root / 'progress.json', pool)
        return worker, pool
