"""Eight-GPU continuation of immutable rich/MS recipe with source-assumed export."""
import argparse
from collections import Counter
import concurrent.futures
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

SCRIPT = Path(__file__).absolute()
HERE = SCRIPT.parent
V3_SHA = '8917ef1d90094d1c392f0dd3097d546371ce62d81ec5fca2b5cfbcb23ed1e682'
V4_SHA = '30cd043149375920c185e5195767f38b5f929e57d826f73cd32ccb09b6f73795'


def module(name, path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = module('frozen_three_gpu', HERE / 'fusion_complete_parallel_n1_v3.py', V3_SHA)
POLICY = module('frozen_source_policy', HERE / 'fusion_source_assumed_full_n1_v4.py', V4_SHA)
PREVIOUS = BASE.OUT
OUT = BASE.MS / 'fusion_parallel_rewrite_v5'
QUEUE = BASE.MS / 'fusion_parallel_queue_v5'
EXPORT = BASE.MS / 'fusion_source_assumed_full6000_v5'
UUIDS = dict(BASE.UUIDS, **{})
UUIDS.update({3: 'GPU-670f50aa-b4bc-51d0-d768-5689bd7a25d8',
    4: 'GPU-391a6df9-2874-dfaa-ac69-e5fc3ccdc554', 5: 'GPU-1f94935b-c4da-79cf-bf7f-14df55219b96',
    6: 'GPU-c946459a-2445-a29f-0862-8bf131dcdf43', 7: 'GPU-8698d9b1-c2bb-6728-4abb-fd2d7e78a5ec'})
OLD_IDENTITIES = [(3014509, 523168040, None), (3021482, 523194759, 0),
    (3021483, 523194794, 1), (3021485, 523194829, 2)]
LEGACY = BASE.load_module('legacy_eight_gpu', BASE.LEGACY, BASE.LEGACY_SHA)
HELPER = LEGACY.helper()


def configure_worker():
    BASE.SCRIPT = SCRIPT
    BASE.OUT = OUT
    BASE.QUEUE = QUEUE
    BASE.UUIDS = UUIDS
    BASE.VERSION = 'rich-ms-fusion6k-same-recipe-eightgpu-v5'


def event(kind, **values):
    row = dict(event=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with (QUEUE / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
    return row


def memory():
    cg = Path('/sys/fs/cgroup/memory')
    values = {line.split()[0].rstrip(':'): int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    return dict(cgroup_limit=int((cg / 'memory.limit_in_bytes').read_text()),
        cgroup_usage=int((cg / 'memory.usage_in_bytes').read_text()),
        cgroup_headroom=int((cg / 'memory.limit_in_bytes').read_text()) - int((cg / 'memory.usage_in_bytes').read_text()),
        host_available=values['MemAvailable'] * 1024)


def snapshot_gpu():
    prior = BASE.UUIDS
    BASE.UUIDS = UUIDS
    try:
        return BASE.gpu_snapshot()
    finally:
        BASE.UUIDS = prior


def boundary(row, data, allowed):
    record = json.loads(data, object_pairs_hook=HELPER.unique_object)
    assert record['id'] == row['id'] and record['original_source_record'] == row
    assert record['split'] == row['split'] and record['source_group'] == row['source_group']
    assert record['source_records_sha256'] == BASE.SOURCE_SHA and record['source_sha256'] in allowed
    material = LEGACY.actual_material(row)
    assert record['actual_text_input'] == material
    assert record['actual_text_input_sha256'] == LEGACY.object_sha(material)
    assert record['raw_source_text_input_sha256'] == row['actual_text_input_sha256']
    assert record['semantic_approval'] is False and record['human_review_completed'] is False and record['window_mapping'] == 'unknown'
    refs = {source['ref'] for source in material['sources']}
    assert 'MS:original' in refs and any(ref.startswith('rich:') for ref in refs)
    target, recovery = POLICY.recover_target(record, HELPER.unique_object)
    assert record['generated_tokens'] == len(record['raw_generated_token_ids']) < LEGACY.MAX_TOKENS
    return record, target, recovery


def source_rows():
    assert BASE.sha(BASE.SOURCE) == BASE.SOURCE_SHA
    rows = [json.loads(line, object_pairs_hook=HELPER.unique_object) for line in BASE.SOURCE.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 6000 and all(r['split'] == 'train' for r in rows)
    return rows


def inspect_existing(item):
    index, row = item
    sid = row['id']
    paths = [PREVIOUS / ('shard' + str(gpu)) / (sid + '.json') for gpu in range(3)] + \
        [PREVIOUS / 'aggregate' / (sid + '.json'), BASE.OLD / (sid + '.json')]
    failures = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            boundary(row, data, {BASE.LEGACY_SHA, V3_SHA})
            return index, row, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), data=data), failures
        except (AssertionError, ValueError, TypeError, KeyError, OSError) as exc:
            failures.append(dict(path=str(path), reason=str(exc), raw_retained=True))
    return index, row, None, failures


def scan(rows):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect_existing, enumerate(rows)))
    reused = sum(result is not None for _, _, result, _ in results)
    tasks = [dict(index=i, id=row['id'], shard=i % 8, reason={'reason': 'missing_complete_body'},
        existing_invalid=failures) for i, row, result, failures in results if result is None]
    report = dict(total=6000, reused=reused, missing=6000-reused, tasks=tasks,
        shard_counts={str(g): sum(t['shard'] == g for t in tasks) for g in UUIDS},
        source_records_sha256=BASE.SOURCE_SHA, new_source_sha256=BASE.sha(SCRIPT),
        recipe_changed=False, original_prompt_sha256=hashlib.sha256(LEGACY.SYSTEM.encode()).hexdigest(),
        old_v3_source_sha256=V3_SHA, legacy_source_sha256=BASE.LEGACY_SHA,
        source_assumed_policy=POLICY.CLEAN.POLICY, legacy_refs_and_keyword_flags_nonblocking=True,
        semantic_review_performed=False, human_review_completed=False, val500_deferred=True)
    return results, report


def stop_exact_old():
    for pid, start, gpu in OLD_IDENTITIES:
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            event('old_exact_already_exited', pid=pid)
            continue
        try:
            identity = BASE.identity(pid)
            argv = [BASE.PYTHON, '-B', '-u', str(HERE / 'fusion_complete_parallel_n1_v3.py')]
            if gpu is not None:
                argv += ['--worker', str(gpu), '--source-sha', V3_SHA]
            assert identity['start'] == start and identity['argv'] == argv and identity['uids'] == [0]*4
            assert identity['cgroup'] == identity['owner_cgroup'] and BASE.identity(pid) == identity
            event('old_exact_SIGTERM', identity=identity)
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            assert select.select([fd], [], [], 30)[0], 'No SIGKILL escalation'
            event('old_exact_exited', pid=pid)
        finally:
            os.close(fd)


def materialize(results, report):
    OUT.mkdir(exist_ok=False)
    (OUT / 'aggregate').mkdir()
    assigned = {g: [] for g in UUIDS}
    lineage = []
    for index, row, result, failures in results:
        if result is None:
            assigned[index % 8].append(row)
        else:
            path = OUT / 'aggregate' / (row['id'] + '.json')
            with path.open('xb') as stream:
                stream.write(result['data'])
            lineage.append(dict(id=row['id'], original_result_path=result['path'],
                original_result_sha256=result['sha256'], snapshot_result_path=str(path), byte_identical=True))
    sets = [set(r['id'] for r in rows) for rows in assigned.values()]
    assert sum(map(len, sets)) == len(set().union(*sets)) == report['missing']
    assert report['reused'] + report['missing'] == 6000
    report['manifest_sha256'] = {}
    for gpu, rows in assigned.items():
        folder = OUT / ('shard' + str(gpu))
        folder.mkdir()
        path = folder / 'tasks.jsonl'
        with path.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        report['manifest_sha256'][str(gpu)] = BASE.sha(path)
    with (OUT / 'reused_lineage.jsonl').open('x', encoding='utf-8') as stream:
        for row in lineage:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    BASE.save(OUT / 'plan.json', report)
    event('eight_shard_materialized', reused=report['reused'], missing=report['missing'], shard_counts=report['shard_counts'])


def admit_all():
    resource = memory()
    # Five extra 70-GiB CPU-loading budgets plus an 80-GiB reserve fit the measured three-model headroom.
    assert resource['cgroup_headroom'] >= 430 * 2**30 and resource['host_available'] >= 600 * 2**30
    state = snapshot_gpu()
    event('eight_gpu_resource_preflight', memory=resource, gpus=state,
        additional_models=5, conservative_additional_loading_bytes=350*2**30, safety_reserve_bytes=80*2**30)
    expected = ['/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python',
        str(BASE.ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py')]
    for gpu, value in state.items():
        reserve = Path('/tmp/od_reserve_' + str(gpu))
        assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid == 0)
        reserve.touch()
        assert len(value['processes']) <= 1
        for proc in value['processes']:
            fd = os.pidfd_open(proc['pid'])
            try:
                identity = BASE.identity(proc['pid'])
                assert identity['uids'] == [0]*4 and identity['argv'] == expected and identity['cuda'] == str(gpu)
                assert identity['cgroup'] == identity['owner_cgroup'] and proc['used'] <= 1100
                assert snapshot_gpu()[gpu] == value and BASE.identity(proc['pid']) == identity
                event('confirmed_project_holder_SIGTERM', gpu=gpu, identity=identity)
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                assert select.select([fd], [], [], 15)[0]
            finally:
                os.close(fd)
    fresh = snapshot_gpu()
    assert all(not value['processes'] and value['used'] <= 1024 and value['total'] >= 80000 for value in fresh.values())
    event('eight_gpu_admission_passed', gpus=fresh, memory=memory())


def export():
    POLICY.CLEAN.self_test()
    rows = source_rows()
    allowed = {BASE.LEGACY_SHA, V3_SHA, BASE.sha(SCRIPT)}
    def inspect(item):
        index, row = item
        paths = [p for p in [OUT / 'aggregate' / (row['id'] + '.json')] +
            [OUT / ('shard' + str(g)) / (row['id'] + '.json') for g in UUIDS] if p.is_file()]
        try:
            assert len(paths) == 1, 'missing_or_multiple_result'
            path = paths[0]
            data = path.read_bytes()
            record, target, recovery = boundary(row, data, allowed)
            return (row, path, data, record, target, recovery), None
        except (AssertionError, ValueError, TypeError, KeyError, OSError) as exc:
            return None, dict(id=row['id'], paths=[str(p) for p in paths], reason=str(exc), raw_preserved=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(inspect, enumerate(rows)))
    unavailable = [error for _, error in results if error]
    if unavailable:
        path = OUT / ('source_assumed_unavailable_' + str(time.time_ns()) + '.json')
        BASE.save(path, unavailable)
        event('source_assumed_final_export_blocked', valid=6000-len(unavailable), unavailable_path=str(path), unavailable=len(unavailable))
        raise RuntimeError('Only listed unavailable IDs need repair, never restart all shards')
    EXPORT.mkdir(exist_ok=False)
    flags = Counter()
    code_counts = Counter()
    changed = recovered = 0
    with (EXPORT / 'captions.jsonl').open('x', encoding='utf-8') as captions, \
         (EXPORT / 'lineage.jsonl').open('x', encoding='utf-8') as lineage:
        for result, _ in results:
            row, path, data, record, target, recovery = result
            body = target['motion_caption']
            cleaned = POLICY.CLEAN.clean(body)
            changed += cleaned != body
            recovered += recovery is not None
            flags.update(record.get('validation_errors', []))
            code_counts.update([record['source_sha256']])
            common = dict(id=row['id'], split=row['split'], source_group=row['source_group'], genre=row['genre'],
                caption_version='rich_ms_fusion_source_assumed_full6000_v5', source_policy=POLICY.CLEAN.POLICY,
                source_action_correctness_assumed=True, semantic_review_performed=False, human_review_completed=False,
                original_source_window_mapping='unknown', faithfulness_status='not_individually_verified',
                source_record_path=str(path), source_record_sha256=hashlib.sha256(data).hexdigest())
            captions.write(json.dumps(dict(common, motion_caption=cleaned), ensure_ascii=False) + '\n')
            lineage.write(json.dumps(dict(common, raw_fused_caption=body, exported_caption=cleaned,
                original_source_record=row, actual_text_input=record['actual_text_input'],
                actual_text_input_sha256=record['actual_text_input_sha256'],
                raw_source_text_input_sha256=record['raw_source_text_input_sha256'],
                raw_rich_text_sha256=row['raw_rich_text_sha256'], raw_MS_text_sha256=row['raw_MS_text_sha256'],
                source_records_sha256=BASE.SOURCE_SHA, generation_source_code_sha256=record['source_sha256'],
                raw_generation_sha256=hashlib.sha256(record['raw_generation'].encode()).hexdigest(),
                raw_generated_token_ids_sha256=LEGACY.object_sha(record['raw_generated_token_ids']),
                original_rule_flags=record.get('validation_errors', []), original_flags_do_not_block_export=True,
                original_completion_status=record.get('completion_status'), original_generated_target=record.get('target'),
                source_refs=target.get('source_refs'), source_refs_status='generated_unverified_sidecar_only',
                actual_bound_source_refs=[s['ref'] for s in record['actual_text_input']['sources']],
                conflicts=target.get('conflicts'), uncertainty=target.get('uncertainty'),
                raw_JSON_recovery=recovery, text_changed=body != cleaned), ensure_ascii=False) + '\n')
        for stream in (captions, lineage):
            stream.flush()
            os.fsync(stream.fileno())
    plan = json.loads((OUT / 'plan.json').read_text())
    report = dict(complete=True, batch6000_complete=True, included=6000, unique_ids=6000, unavailable=0,
        split='train', source_records_sha256=BASE.SOURCE_SHA, source_policy=POLICY.CLEAN.POLICY,
        source_action_correctness_assumed=True, semantic_review_performed=False, human_review_completed=False,
        original_source_window_mapping='unknown', val500_deferred=True, generation_source_code_counts=dict(code_counts),
        source_code_sha256=BASE.sha(SCRIPT), frozen_worker_implementation_sha256=V3_SHA, frozen_policy_sha256=V4_SHA,
        reused_old_complete=plan['reused'], new_sharded=plan['missing'], shard_counts=plan['shard_counts'],
        original_rule_flag_counts=dict(flags), raw_JSON_recovered=recovered, deterministically_cleaned=changed,
        raw_outputs_unchanged=True, recipe_changed=False,
        cleanup='original frozen clean(): pronouns and listed synthetic timing only; body direction/degrees/amplitude/repetitions retained',
        files={name: BASE.sha(EXPORT / name) for name in ('captions.jsonl', 'lineage.jsonl')})
    BASE.save(EXPORT / 'complete.json', report)
    BASE.save(OUT / 'run_complete.json', report)
    event('full6000_source_assumed_export_complete', output=str(EXPORT), report=report)


def manager():
    assert os.getuid() == 0 and not OUT.exists()
    QUEUE.mkdir(exist_ok=True)
    lock = (QUEUE / 'manager.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = source_rows()
    preliminary, report = scan(rows)
    BASE.save(QUEUE / 'preflight_plan.json', report)
    event('CPU_source_assumed_plan_ready', reused=report['reused'], missing=report['missing'], shard_counts=report['shard_counts'])
    if report['missing']:
        before = memory()
        assert before['cgroup_headroom'] >= 430*2**30
        stop_exact_old()
        preliminary, report = scan(rows)
    materialize(preliminary, report)
    if not report['missing']:
        export()
        return
    admit_all()
    configure_worker()
    running = []
    for gpu in UUIDS:
        if report['shard_counts'][str(gpu)] == 0:
            continue
        assert memory()['cgroup_headroom'] >= 120*2**30
        log = (QUEUE / ('shard' + str(gpu) + '.log')).open('xb')
        argv = ['env', 'CUDA_VISIBLE_DEVICES=' + str(gpu), 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1', BASE.PYTHON, '-B', '-u', str(SCRIPT), '--worker', str(gpu), '--source-sha', BASE.sha(SCRIPT)]
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        running.append((gpu, child, log))
        event('eight_gpu_worker_started', gpu=gpu, identity=BASE.identity(child.pid), argv=argv, memory=memory())
        deadline = time.monotonic() + 300
        while not list((OUT / ('shard' + str(gpu))).glob('run_started_*.json')):
            assert child.poll() is None and time.monotonic() < deadline, 'Exact shard loading failed; retain partials'
            assert memory()['cgroup_headroom'] >= 100*2**30
            for active, process, _ in running:
                if process.poll() is None:
                    Path('/tmp/od_reserve_' + str(active)).touch()
            time.sleep(3)
        event('eight_gpu_model_load_resource_verified', gpu=gpu, memory=memory(), gpu_snapshot=snapshot_gpu()[gpu])
    event('all_eight_workers_models_loaded', memory=memory(), gpus=snapshot_gpu())
    while any(child.poll() is None for _, child, _ in running):
        for gpu, child, _ in running:
            if child.poll() is None:
                Path('/tmp/od_reserve_' + str(gpu)).touch()
        time.sleep(60)
    for gpu, child, log in running:
        log.close()
        event('eight_gpu_worker_exited', gpu=gpu, pid=child.pid, exit_code=child.returncode)
    assert all(child.returncode == 0 for _, child, _ in running), 'Only exact failed shard may resume'
    export()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=list(UUIDS))
    parser.add_argument('--source-sha')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        assert args.source_sha == BASE.sha(SCRIPT)
        configure_worker()
        BASE.worker(args.worker, args.source_sha)
    elif args.export:
        export()
    else:
        manager()
