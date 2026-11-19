"""Dynamic union worker with the untouched native resource-manager lifecycle."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_ready_pool_v1 as poolmod
import caption_local_live_manager_v1 as old

base, local, h, BAR = old.base, old.local, old.h, old.BAR
BASE_SHA, MODEL = old.BASE_SHA, old.MODEL


def selected(cfg):
    rows = poolmod.load(cfg['plan']['path'])
    return [r['id'] for r in rows], len(rows)


def validate(cfg):
    local.light.need(Path(base.__file__).resolve() == RT / 'resource_manager.py'
                     and base.sha(base.__file__) == BASE_SHA and base.SAFETY + base.LOAD == 160 * 2**30,
                     'Original lifecycle identity/160GiB boundary differs')
    poolmod.check_manifest(cfg)
    local.light.need(cfg['resource_manager_ref'] == cfg['producer'] == old.ref(SELF)
                     and cfg['source_sha256'].get(str(RT / 'resource_manager.py')) == BASE_SHA
                     and cfg['source_sha256'].get(str(SELF)) == cfg['producer']['sha256']
                     and cfg['source_sha256'].get(str(Path(poolmod.__file__).resolve())) == base.sha(poolmod.__file__),
                     'New pool manager/worker/base source bindings required')
    local.light.need(cfg['config_path'] in cfg['argv'] and 'worker' in cfg['argv'], 'Bound pool worker argv required')
    native_validate = types.FunctionType(old.validate.__code__, dict(old.validate.__globals__, SELF=SELF),
                                         'pool_native_validate', old.validate.__defaults__)
    native_validate(cfg)
    local.light.need(len(cfg['holders']) == 2 and len({o['pid'] for o in cfg['holders']}) == 2
                     and len({o['uuid'] for o in cfg['holders']}) == 2, 'Two distinct fresh holders required')


def gate(cfg, before=False):
    native_gate = types.FunctionType(old.gate.__code__, dict(old.gate.__globals__, SELF=SELF),
                                    'pool_native_gate', old.gate.__defaults__)
    return native_gate(cfg, before)


def archived_drafts(job, commit):
    local.light.need(commit['reliable_writeback'] is True and commit['batch_id'] == job['batch_id'], 'Durable archive required')
    entries = list(local.archive_drafts(commit['archive_path']))
    drafts = []
    for member, draft in entries:
        sid, split = poolmod.source_key(draft)
        expected = job['batch_id'] + '/caption_two_stage_results/drafts/' + split + '/' + sid + '/draft.json'
        local.light.need(member == expected, 'Archive draft source/member mapping differs')
        drafts.append(draft)
    local.light.need(len(drafts) == len(job['metas']) and {poolmod.source_key(d) for d in drafts}
                     == {poolmod.source_key(m) for m in job['metas']}, 'Archive does not cover reserved batch')
    return drafts


def recover_commits(pool):
    # One startup small-receipt scan; no media/weight SHA, no generation or copy.
    for path in (pool.root / 'batches').glob('*/claim.json'):
        try:
            job = poolmod.load(path)
            owner_cfg = pool.check_job(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Partial batch remains sealed, never reissued.
        if all(poolmod.source_key(meta) in pool.terminals for meta in job['metas']):
            continue
        receipt = Path(owner_cfg['draft_out']) / (job['batch_id'] + '.committed.json')
        if receipt.exists():
            commit = poolmod.load(receipt)
            archive = Path(owner_cfg['draft_out']) / (job['batch_id'] + '.tar')
            local.light.need(commit['archive_path'] == str(archive) and not archive.is_symlink()
                             and archive.stat().st_size == commit['archive_bytes'], 'Recovery archive path/size differs')
            pool.publish(job, commit, archived_drafts(job, commit), recovery=True)


def args_for(cfg):
    return argparse.Namespace(stage='run', batch_size=8, max_num_seqs=8, retry_failed=False, model=MODEL,
        visual_max_tokens=1800, hand_roi_root=str(Path(cfg['source_root']) / 'roi'), sample_fps=4., ffmpeg='/usr/local/bin/ffmpeg')


def model_runtime(cfg):
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype='bfloat16', tensor_parallel_size=2, max_num_seqs=8, max_model_len=16384,
        gpu_memory_utilization=.88, limit_mm_per_prompt={'image': 16, 'video': 0},
        mm_processor_kwargs={'max_pixels': 200704}, structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
        enable_prefix_caching=True, trust_remote_code=False, seed=0)
    llm._ms_bar_model_sha256 = cfg['modelref']['model_identity_sha256']
    return lambda: (llm, SamplingParams)


def run(pool, parent_gate, runtime_factory, stage_source=None, generate=None, writeback=None, poll_seconds=10., local_parent='/tmp'):
    cfg = pool.cfg
    local.light.need(callable(parent_gate) and callable(runtime_factory) and pool.ownership(), 'Actual admitted caller required')
    local.light.need(parent_gate(False).get('admitted') is True, 'Actual GPU parent admission required')
    stage_source = stage_source or local.stage_source
    generate = generate or local.light.generate_batch
    writeback = writeback or local.writeback
    args = args_for(cfg)
    runtime = None
    out = Path(cfg['draft_out'])
    poolmod.mkdir(out)
    started = time.perf_counter()

    def owned(job, before=False):
        local.light.need(parent_gate(before).get('admitted') is True and pool.ownership(job), 'GPU/pool gates required')
        return True

    def copy_batch(job):
        root = None
        try:
            owned(job)
            root = Path(tempfile.mkdtemp(prefix=job['batch_id'] + '-', dir=local_parent))
            poolmod.exclusive(pool.batch_path(job['batch_id']).parent / 'cache.json', dict(local_cache_path=str(root), owner=pool.owner, **poolmod.FLAGS))
            entries, times, failures = [], [], []
            for meta in job['metas']:
                try:
                    entry, timing = stage_source(cfg['source_root'], meta, root)
                    entries.append(entry)
                    times.append(timing)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    failure = dict(id=meta['id'], split=meta['split'], status='draft_needs_regeneration', reason=repr(error), **poolmod.FLAGS)
                    poolmod.exclusive(root / 'caption_two_stage_results/drafts' / meta['split'] / meta['id'] / 'draft.json', failure)
                    failures.append(failure)
            return job, root, entries, times, failures
        except BaseException as error:
            pool.fail(job, error, root)
            raise

    def checkpoint():
        worker, global_progress = pool.progress()
        worker.update(selected=len(pool.rows), scope_pool=False, node=cfg['node'], gpus=cfg['gpus'],
                      UTC=time.strftime('%FT%TZ', time.gmtime()),
                      wall_seconds=time.perf_counter() - started, local_inputs_intermediates_results=True,
                      copy_threads=1, max_inflight_batches=2, source_batch_size=3, **poolmod.FLAGS)
        poolmod.atomic(out / 'progress.json', worker)
        return worker, global_progress

    with ThreadPoolExecutor(max_workers=1) as copier:
        pending = None
        while True:
            worker_progress, global_progress = checkpoint()
            if global_progress['complete']:
                local.light.need(pending is None, 'Pool complete with prefetched unfinished batch')
                summary = dict(worker_progress, scope_pool=True, global_terminal=global_progress['global_terminal'], N=global_progress['N'])
                poolmod.exclusive(Path(cfg['out']) / 'summary.json', summary)
                return summary
            if pending is None:
                job = pool.reserve()
                if job is None:
                    time.sleep(poll_seconds)
                    continue  # Empty ready is NOT completion, orphans remain blocked.
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
                if next_job:
                    pending = copier.submit(copy_batch, next_job)
                owned(job)
                result = generate(entries, args, runtime, root / 'caption_two_stage_results',
                                  lambda a, r: owned(job)) if entries else dict(drafts=[], new_model_calls=0)
                result['drafts'].extend(failures)
                poolmod.exclusive(root / 'source_batch.json', dict(metas=job['metas'], source_claims=job['source_claims'],
                    owner=pool.owner, copy_timings=times, new_model_calls=result['new_model_calls'], **poolmod.FLAGS))
                commit = writeback(root, out, job['batch_id'])
                pool.publish(job, commit, archived_drafts(job, commit))
                committed_worker, committed_pool = checkpoint()
                print(json.dumps(dict(event='pool_batch_committed', worker_id=cfg['pool_worker_id'],
                    worker_processed=committed_worker['processed'], global_terminal=committed_pool['global_terminal'],
                    N=committed_pool['N'], UTC=committed_worker['UTC'])), flush=True)
            except BaseException as error:
                pool.fail(job, error, root)
                raise
            # Reclaim ONLY this exact worker-owned cache, after terminal publication.
            shutil.rmtree(root)
            root.with_suffix('.tar').unlink()


def worker(path):
    cfg = poolmod.load(path)
    cfg.update(_config_path=str(Path(path).resolve()), _config_sha=base.sha(path))
    pool = poolmod.Pool(cfg)
    recover_commits(pool)
    pool.admit(lambda before: gate(cfg, before))
    return run(pool, lambda before=False: gate(cfg, before), model_runtime)


def summaries(cfg, out):
    value = poolmod.load(out / 'summary.json')
    pool = poolmod.Pool(cfg)
    worker_progress, progress = pool.progress()
    local.light.need(value['scope_pool'] is True and progress['complete']
                     and value['global_terminal'] == value['N'] == progress['global_terminal'] == cfg['targetcount']
                     and value['worker_id'] == cfg['pool_worker_id'] and value['pool_sha256'] == cfg['plan']['sha256']
                     and value['processed'] == value['committed'] == worker_progress['committed']
                     and value['reserved'] == worker_progress['reserved'] and value['counts'] == worker_progress['counts'],
                     'Pool/global vs actual worker summary differs')
    return {str(out / 'summary.json'): dict(summary=dict(stage='run', total_source_count=cfg['targetcount'],
        selected_source_count=cfg['targetcount'], processed_source_count=value['processed'], plan_sha256=cfg['plan']['sha256'],
        scope_pool=True, global_terminal=value['global_terminal'], N=value['N']), selection=dict(selected_ids=selected(cfg)[0]),
        worker_processed=value['processed'], draft_counts=value['counts'], **poolmod.FLAGS)}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        parser = argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--config', required=True)
        worker(parser.parse_args().config)
        return
    # __file__, check_sources, main, admission SHA and every resource transaction stay native.
    base.validate_pair = validate
    base.selected_sources = selected
    base.stage_summaries = summaries
    base.main()


if __name__ == '__main__':
    main()
