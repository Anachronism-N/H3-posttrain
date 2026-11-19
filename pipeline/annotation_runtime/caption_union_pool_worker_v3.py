"""Union pool v3 worker: one worker draws ready sources from several existing dynamic pools.

Cross-pool mutual exclusion = O_EXCL creation of the ORIGIN pool's own source claim file
(origin_root/sources/<digest(origin_sha,id,split)>.json), the same file the origin workers create.
Origin journals/terminals are never written. Own batches/terminals live under the v3 root.
Ready journals are read tolerantly: an undecodable line (e.g. transient FUSE NUL view) stops that
file for this poll and is re-read later; it never crashes the worker.
Drafts stay unreviewed (FLAGS unchanged); claimed/failed sources are never re-dispatched.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import caption_ready_pool_v1 as poolmod
import caption_pool_live_manager_v2 as mgr

FLAGS = poolmod.FLAGS


class UnionPool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['pool_root'])
        for name in ('batches', 'terminals', 'admitted', 'claims'):
            poolmod.mkdir(self.root / name)
        self.rows, self.origin = {}, {}
        self.origins = []
        for path in cfg['origin_configs']:
            ocfg = poolmod.load(path)
            assert ocfg['source_root'] == cfg['source_root'], 'origin source_root differs'
            sha = ocfg['plan']['sha256']
            assert poolmod.sha(ocfg['plan']['path']) == sha, 'origin plan changed'
            info = dict(name=Path(ocfg['pool_root']).name, root=Path(ocfg['pool_root']), sha=sha)
            self.origins.append(info)
            for row in poolmod.load(ocfg['plan']['path']):
                key = poolmod.source_key(row)
                if key in self.rows:
                    continue
                self.rows[key] = row
                self.origin[key] = info
        self.offsets, self.queue, self.queued = {}, deque(), set()
        self.terminals, self.failed, self.reserved = set(), set(), set()
        self.counts = {}
        self.owner = None
        self._lockfds = []

    def claim_path(self, key):
        o = self.origin[key]
        return o['root'] / 'sources' / (poolmod.digest([o['sha'], *key]) + '.json')

    def batch_path(self, name):
        return self.root / 'batches' / name / 'claim.json'

    def admit(self):
        for gpu in self.cfg['gpus']:
            fd = os.open('/tmp/ms12k_gpu_%d.lock' % gpu, os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Raises if another managed job holds the GPU.
            self._lockfds.append(fd)
        import threading

        def heartbeat():
            while True:
                for gpu in self.cfg['gpus']:
                    try:
                        Path('/tmp/od_reserve_%d' % gpu).touch()
                    except OSError:
                        pass
                time.sleep(30)
        threading.Thread(target=heartbeat, daemon=True).start()
        self.owner = dict(worker_id=self.cfg['pool_worker_id'], node=self.cfg['node'], gpus=self.cfg['gpus'],
                          config=poolmod.ref(self.cfg['config_path']), pid=os.getpid(),
                          host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip())
        poolmod.exclusive(self.root / 'admitted' / (self.cfg['pool_worker_id'] + '-' + uuid.uuid4().hex[:8] + '.json'),
                          dict(owner=self.owner, **FLAGS))

    def ownership(self, job=None):
        assert self.owner is not None and self.owner['pid'] == os.getpid(), 'unadmitted'
        if job is not None:
            assert job['owner'] == self.owner, 'batch owner differs'
            assert not (self.batch_path(job['batch_id']).parent / 'failure.json').exists(), 'batch failed'
        return True

    def update_ready(self):
        root = Path(self.cfg['source_root'])
        paths = sorted((root / 'ready').glob('producer-*.jsonl')) if (root / 'ready').is_dir() else [root / 'ready.jsonl']
        for path in paths:
            offset = self.offsets.get(str(path), 0)
            with path.open('rb') as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line.endswith(b'\n'):
                        break
                    try:
                        row = json.loads(line)
                        key = poolmod.source_key(row)
                    except Exception:
                        break  # Transient/torn view: keep offset, re-read on next poll.
                    offset += len(line)
                    if key in self.rows and row.get('status', 'ready') == 'ready' and key not in self.queued:
                        self.queue.append(key)
                        self.queued.add(key)
            self.offsets[str(path)] = offset

    def reserve(self):
        self.ownership()
        self.update_ready()
        chosen, refs = [], []
        while self.queue and len(chosen) < 3:
            key = self.queue.popleft()
            if key in self.reserved:
                continue
            o = self.origin[key]
            if (o['root'] / 'terminals' / self.claim_path(key).name).exists():
                self.reserved.add(key)
                continue
            value = dict(pool_sha256=o['sha'], meta=self.rows[key], owner=self.owner, **FLAGS)
            try:
                ref = poolmod.exclusive(self.claim_path(key), value)
            except FileExistsError:
                self.reserved.add(key)
                continue
            self.reserved.add(key)
            chosen.append(self.rows[key])
            refs.append(ref)
        if not chosen:
            return None
        name = 'batch-' + uuid.uuid4().hex
        job = dict(batch_id=name, union_pool=True, metas=chosen, source_claims=refs, owner=self.owner,
                   reserved_sources=len(chosen), origins=[self.origin[poolmod.source_key(m)]['name'] for m in chosen], **FLAGS)
        poolmod.exclusive(self.batch_path(name), job)
        return job

    def fail(self, job, error, cache=None):
        poolmod.exclusive(self.batch_path(job['batch_id']).parent / 'failure.json',
                          dict(reason=repr(error)[:4000], local_cache_path=str(cache) if cache else None,
                               budget_not_reset=True, no_automatic_retry=True, owner=self.owner, **FLAGS))
        for meta in job['metas']:
            self.failed.add(poolmod.source_key(meta))

    def publish(self, job, commit, drafts, recovery=False):
        self.ownership(job)
        assert commit['reliable_writeback'] is True and commit['batch_id'] == job['batch_id'], 'durable writeback required'
        results = {poolmod.source_key(d): d for d in drafts}
        assert set(results) == {poolmod.source_key(m) for m in job['metas']}, 'one terminal per source'
        for meta, claim in zip(job['metas'], job['source_claims']):
            key = poolmod.source_key(meta)
            draft = results[key]
            assert all(draft.get(k) == v for k, v in FLAGS.items()), 'draft flags changed'
            receipt = dict(id=key[0], split=key[1], status=draft['status'], origin_pool=self.origin[key]['name'],
                           origin_pool_sha256=self.origin[key]['sha'], batch_id=job['batch_id'], owner=job['owner'],
                           source_claim=claim, archive_path=commit['archive_path'],
                           archive_member=job['batch_id'] + '/caption_two_stage_results/drafts/' + key[1] + '/' + key[0] + '/draft.json',
                           **FLAGS)
            path = self.root / 'terminals' / (self.origin[key]['name'] + '__' + key[0] + '.json')
            poolmod.exclusive(path, receipt)
            with (self.root / 'terminal_index.jsonl').open('ab') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(poolmod.encoded(dict(receipt=str(path))))
                stream.flush()
                os.fsync(stream.fileno())
            self.terminals.add(key)
            self.counts[draft['status']] = self.counts.get(draft['status'], 0) + 1

    def progress(self):
        worker = dict(scope='worker', worker_id=self.cfg['pool_worker_id'], reserved=len(self.terminals) + len(self.failed),
                      committed=len(self.terminals), processed=len(self.terminals), failed=len(self.failed),
                      counts=dict(self.counts), **FLAGS)
        pool = dict(scope_pool=True, N=len(self.rows), global_terminal=len(self.terminals), complete=False,
                    queued_unchecked=len(self.queue), **FLAGS)
        return worker, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = poolmod.load(args.config)
    cfg['config_path'] = str(Path(args.config).resolve())
    pool = UnionPool(cfg)
    pool.admit()
    print(json.dumps(dict(event='union_worker_admitted', worker=cfg['pool_worker_id'], rows=len(pool.rows),
                          origins=[o['name'] for o in pool.origins])), flush=True)
    return mgr.run(pool, lambda before=False: dict(admitted=True), model_runtime, poll_seconds=20.)


def model_runtime(cfg):
    tp = int(cfg.get('tensor_parallel_size', 2))
    if tp == 2 and not cfg.get('gpu_memory_utilization'):
        return mgr.model_runtime(cfg)  # Unchanged 80GB A800 recipe.
    from vllm import LLM, SamplingParams
    llm = LLM(model=mgr.MODEL, dtype='bfloat16', tensor_parallel_size=tp, max_num_seqs=8, max_model_len=16384,
              gpu_memory_utilization=float(cfg.get('gpu_memory_utilization', .88)), limit_mm_per_prompt={'image': 16, 'video': 0},
              mm_processor_kwargs={'max_pixels': 200704},
              structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
              enable_prefix_caching=True, trust_remote_code=False, seed=0)
    llm._ms_bar_model_sha256 = cfg['modelref']['model_identity_sha256']
    return lambda: (llm, SamplingParams)


if __name__ == '__main__':
    main()
