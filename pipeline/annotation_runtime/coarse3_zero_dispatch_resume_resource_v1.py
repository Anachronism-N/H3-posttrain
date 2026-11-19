"""Exact fixed-three zero-dispatch continuation; no general retry or resource framework."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TEST = RT / 'test_coarse3_zero_dispatch_resume_resource_v1.py'
OUT = RT.parent / 'coarse3_filter_recovery_independent_reviews_v1'
IDS = ('AuuitdcBJyGOnWaH', 'AzPomlvsANsevIFb', 'DxNCjtNsxcazNpGb')
PREVIOUS = {'path': str(RT / 'coarse3_n6_launch_current_admission_main_v5.json'),
            'sha256': '5404683fc5f17c8e614ea19a9d85fc7fdb20c4d6cd61877f5008d0da56936e84'}
CACHE = RT / 'coarse3_n6_content_cache_execution_v1.py'
BRIDGE = RT / 'coarse3_review_mount_bridge_v1.py'
CACHE_SHA = '216a65acfd1f68f1cb5274defb62290f6481faf0b947e6860e966476b17d84ba'
BRIDGE_SHA = 'd30adeeeaf38c16900d61a59f0a95a552945036b800fbca5a83af3cfae4300be'
HISTORY_NAMES = ('independent.summary.json', 'independent.ledger.jsonl', 'finite.stage.summary.json')
ERROR = 'ValueError: source/label/frame binding changed on resume'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'Bound content SHA changed')
    return json.loads(Path(reference['path']).read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def snapshot_bytes(out, raw):
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(out) / 'zero-dispatch-history' / (digest + '.bytes')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    need(not path.is_symlink() and path.read_bytes() == raw, 'Immutable history snapshot differs')
    return {'path': str(path), 'sha256': digest}


def no_attempts(out, plan):
    need([s['source_id'] for s in plan['sources']] == list(IDS) and plan['maximum_review_calls'] == 3,
         'Only original fixed three allowed')
    need(not os.path.lexists(Path(out) / 'review-attempts'), 'Any prior claim/attempt directory denies continuation')
    for source in plan['sources']:
        root = Path(source['root'])
        need(root.is_relative_to(Path(out) / 'sources') and not os.path.lexists(root / 'review')
             and not os.path.lexists(root / 'independent.review.receipt.json'),
             'Any partial review/receipt denies continuation')
    need(all(not os.path.lexists(Path(out) / name) for name in
             ('loads', 'reviewer.model.identity.json', 'reviewer.code.identity.json', 'model.runtime.json')),
         'Prior model runtime/load receipt denies continuation')


def zero_history(out, plan, previous_ref, *, preserve=False, history_ref=None):
    no_attempts(out, plan)
    raw = {name: (Path(out) / name).read_bytes() for name in HISTORY_NAMES}
    summary, stage = json.loads(raw[HISTORY_NAMES[0]]), json.loads(raw[HISTORY_NAMES[2]])
    rows = [json.loads(line) for line in raw[HISTORY_NAMES[1]].splitlines() if line.strip()]
    need(summary.get('population_ref') == plan['population_ref'] == stage.get('population_ref')
         and summary.get('selected_source_ids') == list(IDS) and summary.get('pending_source_ids') == list(IDS)
         and summary.get('population_source_count') == summary.get('maximum_review_calls') == 3
         and summary.get('statuses') == {'pending': 3}
         and type(summary.get('review_dispatches')) is int and summary['review_dispatches'] == 0
         and summary.get('actual_VLM_calls') == 0 and summary.get('GPU_model_load_performed') is False
         and summary.get('sampled_reviewed_denominator') == 0
         and summary.get('independent_sample_review_completed') is False
         and summary.get('original_attempt_budget_consumed') is False,
         'Historical summary is not exact zero-dispatch/no-model pending three')
    need(stage.get('launch_config_ref') == previous_ref and stage.get('source_count') == stage.get('terminal_count') == 3
         and type(stage.get('review_dispatches')) is int and stage['review_dispatches'] == 0
         and stage.get('score_refs') == []
         and stage.get('independent_summary_ref') == {'path': str(Path(out) / HISTORY_NAMES[0]), 'sha256': hashlib.sha256(raw[HISTORY_NAMES[0]]).hexdigest()}
         and stage.get('independent_ledger_ref') == {'path': str(Path(out) / HISTORY_NAMES[1]), 'sha256': hashlib.sha256(raw[HISTORY_NAMES[1]]).hexdigest()},
         'Historical finite stage binding/dispatch changed')
    need(len(rows) == 3 and [r.get('source_id') for r in rows] == list(IDS), 'Historical ledger is not exact three')
    for source, row in zip(plan['sources'], rows):
        need(row.get('status') == 'pending' and row.get('error') == ERROR
             and type(row.get('review_dispatches')) is int and row['review_dispatches'] == 0
             and row.get('actual_VLM_calls') == 0 and row.get('label_scope') == 'complete'
             and row.get('candidate_ref') == source['candidate_ref'], 'Historical source has dispatch/claim/different failure')
    for value in (summary, stage, *rows):
        need(value.get('semantic_approval') is False and value.get('training_eligible') is False,
             'Zero model calls are not quality approval')
        for key, marker in value.items():
            if key.startswith('review_') and any(word in key for word in ('budget', 'attempt', 'claim', 'reserved', 'reservation', 'receipt')):
                need(marker is False or marker is None or (type(marker) is int and marker == 0),
                     'Historical review claim/budget/attempt marker denies continuation: ' + key)
    if preserve:
        files = {name: {'original_path': str(Path(out) / name), 'snapshot_ref': snapshot_bytes(out, data)} for name, data in raw.items()}
        history_ref = snapshot_bytes(out, encoded({'schema': 'coarse3_exact_zero_dispatch_history_v1',
            'previous_config_ref': previous_ref, 'population_ref': plan['population_ref'], 'files': files,
            'review_dispatches': 0, 'GPU_model_load_performed': False, 'semantic_approval': False}))
    need(history_ref is not None, 'Durable original-byte history snapshots required')
    history = checked(history_ref)
    need(history['previous_config_ref'] == previous_ref and history['population_ref'] == plan['population_ref']
         and set(history['files']) == set(HISTORY_NAMES), 'History manifest differs')
    for name, data in raw.items():
        entry = history['files'][name]
        need(entry['original_path'] == str(Path(out) / name)
             and checked_bytes(entry['snapshot_ref']) == data, 'Current zero history differs from original preserved bytes')
    return history_ref


def checked_bytes(reference):
    raw = Path(reference['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Historical immutable bytes changed')
    return raw


def load_module(name, path, digest):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest, 'Frozen execution module changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


@contextmanager
def private_import(name, value):
    before = sys.modules.get(name)
    sys.modules[name] = value
    try:
        yield
    finally:
        if before is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = before


def resource_chain(c, resource_ref, previous_ref):
    cfg = c.checked(resource_ref)
    need(cfg.get('node') == 'N6' and cfg.get('gpus') == [3] and cfg.get('modelref') == c.modelref(), 'Prior actual resource scope changed')
    if resource_ref == previous_ref:
        producer, manager = c.ref(CACHE), c.ref(CACHE)
    else:
        producer = {'path': str(RT / 'curated_clause_salvage_review_watch_v5.py'),
                    'sha256': 'c79d1428c0650bce5616d28c96c34abc62677d1798703037d9f92f993cf9b6d2'}
        manager = {'path': str(RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py'),
                   'sha256': '76d48559e7dc89f13f9835d2484757acad9ab89fc3ebf7108f4134ed83bbf983'}
        need(cfg.get('out') == str(c.OLD_OUT), 'Only exact original salvage fresh resource ancestor allowed')
    need(cfg.get('producer') == producer and c.sha(producer['path']) == producer['sha256']
         and c.sha(manager['path']) == manager['sha256'], 'Actual prior frozen producer/manager changed')
    queue = Path(cfg['queue'])
    need(queue.parent == RT.parent / 'queues', 'Prior resource queue outside experiment')
    need((queue / 'resources_safe.json').is_file() and (queue / 'holder_restored_gpu3.json').is_file(),
         'Actual resources_safe/restored proof required; no launch before safety')
    names = ('admission.json', 'live.json', 'holder_restored_gpu3.json', 'resources_safe.json')
    refs = {name: c.ref(queue / name) for name in names}
    admission, live, restored, safe = (c.checked(refs[n]) for n in names)
    if resource_ref == previous_ref:
        refs['actual_leases_held.json'] = c.ref(queue / 'actual_leases_held.json')
        held = c.checked(refs['actual_leases_held.json'])
        need(held.get('schema') == 'coarse3_n6_actual_leases_held_v1'
             and held.get('config_ref') == resource_ref and held.get('gpus') == [3]
             and held.get('resource_admission_ref') == refs['admission.json']
             and all(held['manager_identity'][k] == admission['manager_identity'][k]
                     for k in c.base_module().IDENTITY_KEYS), 'Actual prior held lease admission chain differs')
    need(admission['config_sha256'] == resource_ref['sha256'] and admission['manager_sha256'] == c.BASE_SHA
         and all(admission.get(k) == cfg.get(k) for k in ('source_sha256', 'producer', 'plan', 'out', 'node', 'gpus'))
         and admission['model_identity_sha256'] == c.MODEL_SHA
         and admission['manager_identity']['argv'] == ['/opt/venv_eval/bin/python', '-B', '-u', manager['path'],
             '--config', resource_ref['path'], '--config-sha', resource_ref['sha256'], '--manager-sha', c.BASE_SHA]
         and admission['manager_identity']['uid'] == os.getuid(), 'Actual prior admission/config/manager differs')
    need(all(live['manager_identity'][k] == admission['manager_identity'][k] for k in c.base_module().IDENTITY_KEYS)
         and live['argv'] == cfg['argv'], 'Prior worker/manager live chain differs')
    need(safe.get('phase') == cfg['phase'] and all(safe.get(k) is True for k in
        ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')),
         'Actual resources_safe/restored proof required; no launch before safety')
    need(restored['gpu'] == 3 and restored['uuid'] == admission['holders'][0]['uuid']
         and restored['original_identity'] == admission['holders'][0]['identity']
         and restored['restore_argv'] == c.base_module().RESTORE_ARGV,
         'Actual restored original GPU3 holder chain differs')
    need((queue / 'admission.json').stat().st_mtime_ns <= (queue / 'holder_restored_gpu3.json').stat().st_mtime_ns
         <= (queue / 'resources_safe.json').stat().st_mtime_ns, 'Safe/restored ordering differs')
    return {'restored': restored, 'refs': refs, 'resource_config_ref': resource_ref}


def dispatch_guard(row, reviewer):
    calls = getattr(reviewer, '_coarse3_dispatches', 0)
    need(row.get('id') in IDS and type(calls) is int and 0 <= calls < 3,
         'Only fixed three single-source calls; fourth dispatch denied')


def configured(cfg=None, *, hot=False):
    cache = load_module('zero_dispatch_private_cache', CACHE, CACHE_SHA)
    cache.SELF = SELF
    h = cache.module('zero_dispatch_private_helper', RT / 'prepare_curated_coarse_empty_n5_v1.py')
    h.sha = cache.content_sha
    with private_import('prepare_curated_coarse_empty_n5_v1', h):
        c = cache.module('zero_dispatch_private_prepare', RT / 'prepare_coarse3_n6_v1.py')
    c.sha, c.h.sha = cache.content_sha, cache.content_sha
    c.MANAGER = c.DRIVER = SELF
    bridge = load_module('zero_dispatch_private_mount_bridge', BRIDGE, BRIDGE_SHA)
    prior_cfg = c.checked(PREVIOUS)
    old_plan = c.checked(prior_cfg['plan'])
    need(prior_cfg.get('producer') == c.ref(CACHE) and prior_cfg.get('gpus') == [3]
         and prior_cfg.get('targetcount') == 3, 'Only exact original zero-dispatch config allowed')
    history_ref = cfg['zero_dispatch_history_ref'] if cfg else None
    original_worker = c.worker_module
    def worker():
        m = original_worker()
        lib, _ = bridge.install(m)
        lib.c.sha = cache.content_sha
        original_make, original_transaction = m.make_reviewer, m.review_transaction
        def make(*args, **kwargs):
            reviewer = original_make(*args, **kwargs)
            paths = [Path(v['path']) for v in reviewer.code['files'].values()] + [SELF, BRIDGE, CACHE]
            reviewer.code = lib.batch.identity_tree(RT, paths)
            return reviewer
        def transaction(lib, out, row, root, evidence, reviewer, callback, context):
            dispatch_guard(row, reviewer)
            return original_transaction(lib, out, row, root, evidence, reviewer, callback, context)
        m.make_reviewer, m.review_transaction = make, transaction
        return m
    c.worker_module = worker
    original_plan = c.source_plan
    def source_plan(m=None, *, probe=True):
        current = original_plan(m or worker(), probe=probe)
        for key in old_plan:
            if key == 'source_sha256' or key in ('content_SHA_cache_execution_ref',):
                continue
            if key in current:
                need(current[key] == old_plan[key], 'Original immutable data/source plan differs: ' + key)
        for path, digest in current['source_sha256'].items():
            if path != str(SELF):
                need(old_plan['source_sha256'].get(path) == digest, 'No fresh re-signing original data: ' + path)
        value = copy.deepcopy(old_plan)
        value['source_sha256'].update({str(p): c.sha(p) for p in (SELF, TEST, BRIDGE)})
        value['zero_dispatch_resume_execution_ref'] = c.ref(SELF)
        value['mount_device_bridge_ref'] = c.ref(BRIDGE)
        value['immutable_previous_plan_ref'] = prior_cfg['plan']
        value['zero_dispatch_history_ref'] = history_ref
        if history_ref:
            value['source_sha256'].update(h.references([history_ref, checked(history_ref)['files'], prior_cfg['plan'], PREVIOUS]))
        return value
    c.source_plan = source_plan
    c.require_unreviewed = lambda plan: zero_history(c.OUT, plan, PREVIOUS, history_ref=history_ref)
    original_verify = c.verify_config
    def verify(config, plan, *, verify_history=True):
        need(config.get('previous_zero_dispatch_config_ref') == PREVIOUS and config.get('zero_dispatch_history_ref') == history_ref,
             'Exact zero-dispatch continuation binding missing')
        state_snapshot = config.get('immutable_salvage_ancestor_state_ref')
        need(isinstance(state_snapshot, dict)
             and state_snapshot.get('sha256') == prior_cfg['stream_prefix_proof']['state_ref']['sha256'],
             'Original salvage ancestor content SHA changed')
        prefix = copy.deepcopy(prior_cfg['stream_prefix_proof'])
        prefix['state_ref'] = state_snapshot
        need(config.get('stream_prefix_proof') == prefix, 'Original immutable salvage prefix binding changed')
        return original_verify(config, plan, verify_history=verify_history and not hot)
    c.verify_config = verify
    c.prior_safe = lambda ignored, base=None: resource_chain(c, cfg['resource_ancestor_ref'], PREVIOUS)
    return c, cache, prior_cfg, old_plan


def build(args):
    need({'path': str(Path(args.from_launch).resolve()), 'sha256': args.from_sha} == PREVIOUS,
         'Not the exact historical zero-dispatch launch')
    c, cache, old, old_plan = configured()
    resource_ref = {'path': str(Path(args.resource_from).resolve()), 'sha256': args.resource_sha} if args.resource_from else PREVIOUS
    # Deliberately before snapshot/config creation and any node/lock capture.
    chain = resource_chain(c, resource_ref, PREVIOUS)
    history_ref = zero_history(c.OUT, old_plan, PREVIOUS, preserve=True)
    seed = {'zero_dispatch_history_ref': history_ref, 'resource_ancestor_ref': resource_ref}
    c, cache, old, old_plan = configured(seed)
    plan = c.source_plan()
    c.require_unreviewed(plan)
    c.verify_local_model(old)
    fresh = c.fresh_holder(c.base_module(), chain)
    launch = Path(args.launch).resolve()
    need(launch.parent == RT and launch.name.startswith('coarse3_n6_launch_zero_dispatch_resume_')
         and not os.path.lexists(launch), 'New isolated resume launch required')
    suffix = str(time.time_ns())
    plan_path, capture_path = RT / ('coarse3_zero_dispatch_plan_' + suffix + '.json'), RT / ('coarse3_zero_dispatch_capture_' + suffix + '.json')
    c.save_once(plan_path, plan)
    c.save_once(capture_path, fresh)
    cfg = copy.deepcopy(old)
    cfg.update(launch_path=str(launch), queue=str(RT.parent / 'queues' / ('coarse3_n6_independent_zero_dispatch_resume_' + suffix)),
               argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(SELF), '--launch', str(launch)],
               env=c.driver_env(launch), producer=c.ref(SELF), resource_manager=c.ref(SELF), plan=c.ref(plan_path),
               previous_zero_dispatch_config_ref=PREVIOUS, zero_dispatch_history_ref=history_ref,
               resource_ancestor_ref=resource_ref, resource_safe_refs=chain['refs'], holder_capture=c.ref(capture_path),
               holders=fresh['holders'], lease_snapshot_ns=fresh['lease_snapshot_ns'])
    cfg['source_sha256'].update(plan['source_sha256'])
    state_ref = old['stream_prefix_proof']['state_ref']
    supplied = {'path': str(Path(args.ancestor_state_snapshot).resolve()), 'sha256': state_ref['sha256']} if args.ancestor_state_snapshot else state_ref
    state_snapshot = snapshot_bytes(c.OUT, checked_bytes(supplied))
    need(state_snapshot['sha256'] == state_ref['sha256'], 'Original salvage ancestor state SHA must remain unchanged')
    cfg['stream_prefix_proof']['state_ref'] = state_snapshot
    cfg['immutable_salvage_ancestor_state_ref'] = state_snapshot
    need(cfg['source_sha256'].pop(state_ref['path']) == state_snapshot['sha256'], 'Original live state binding differs')
    cfg['source_sha256'].update(c.h.references([cfg['plan'], cfg['holder_capture'], resource_ref, chain['refs'], state_snapshot]))
    c.verify_config(cfg, plan)
    c.save_once(launch, cfg)
    return {'launch_ref': c.ref(launch), 'wrapper_ref': c.ref(SELF), 'history_ref': history_ref,
            'GPU_model_calls': 0, 'signals_sent': False, 'command': manager_argv(c, cfg)}


def manager_argv(c, cfg):
    return ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config', cfg['launch_path'],
            '--config-sha', c.sha(cfg['launch_path']), '--manager-sha', c.BASE_SHA]


def configured_driver(c, cache, cfg):
    with private_import('prepare_coarse3_n6_v1', c):
        driver = cache.module('zero_dispatch_private_driver', RT / 'coarse3_n6_driver_v1.py')
    original_parent = driver.verify_actual_parent
    def parent(config, base, plan, tracker=None, allow_compute=False):
        proof = original_parent(config, base, plan, tracker, allow_compute)
        budget = base.headroom()
        need(budget['headroom'] >= (base.SAFETY if allow_compute else base.required_budget(config)),
             'Current headroom below unchanged original admission/runtime safety budget')
        need([s['source_id'] for s in plan['sources']] == list(IDS), 'Current source scope changed')
        for source in plan['sources']:
            for reference in [source['source_job_ref'], source['candidate_ref'], *source['prepared_refs']]:
                need(c.sha(reference['path']) == reference['sha256'], 'Current fixed-three job/candidate/input/preview/evidence SHA changed')
            evidence = c.load(source['evidence_ref']['path'])
            for frame in evidence['frames']:
                need(c.sha(Path(source['root']) / frame['file']) == frame['sha256'], 'Current fixed-three native RGB changed')
            for roi in evidence['hand_rois']:
                need(c.sha(roi['path']) == roi['sha256'], 'Current fixed-three native ROI changed')
        return {**proof, 'current_headroom': budget, 'current_fixed_three_SHA_verified': True}
    driver.verify_actual_parent = parent
    original_save = c.save_once
    def save(path, value):
        if Path(path) != c.OUT / 'finite.stage.summary.json':
            return original_save(path, value)
        history = checked(cfg['zero_dispatch_history_ref'])
        expected = history['files']['finite.stage.summary.json']['snapshot_ref']
        need(Path(path).read_bytes() == checked_bytes(expected), 'Original failed finite history changed; no overwrite')
        original_save(Path(cfg['queue']) / 'zero_dispatch_finite_publication.json',
                      {'history_ref': cfg['zero_dispatch_history_ref'], 'launch_config_ref': c.ref(cfg['launch_path'])})
        value = {**value, 'previous_zero_dispatch_history_ref': cfg['zero_dispatch_history_ref'],
                 'previous_config_ref': PREVIOUS}
        c.worker_module().libraries().batch.atomic_write(Path(path), encoded(value))
    c.save_once = save
    return driver


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--config' in argv:
        p = argparse.ArgumentParser()
        p.add_argument('--config', required=True)
        p.add_argument('--config-sha', required=True)
        p.add_argument('--manager-sha', required=True)
        a = p.parse_args(argv)
        cfg = checked({'path': a.config, 'sha256': a.config_sha})
        c, cache, _, _ = configured(cfg)
        need(a.manager_sha == c.BASE_SHA, 'Original base manager SHA required')
        with private_import('prepare_coarse3_n6_v1', c):
            manager = cache.module('zero_dispatch_private_manager', RT / 'coarse3_n6_resource_manager_v1.py')
        manager.base.validate_pair = lambda value: cache.fresh_current_validate(manager, value)
        original_release = manager.base.release_holder
        def release(entry):
            cache.current_before_release(manager, entry)
            original_release(entry)
        manager.base.release_holder = release
        manager.base.main()
    elif '--launch' in argv and '--build' not in argv:
        p = argparse.ArgumentParser()
        p.add_argument('--launch', required=True)
        a = p.parse_args(argv)
        cfg = json.loads(Path(a.launch).read_bytes())
        c, cache, _, _ = configured(cfg, hot=True)
        configured_driver(c, cache, cfg).main()
    else:
        p = argparse.ArgumentParser()
        p.add_argument('--build', action='store_true', required=True)
        p.add_argument('--from-launch', required=True)
        p.add_argument('--from-sha', required=True)
        p.add_argument('--resource-from')
        p.add_argument('--resource-sha')
        p.add_argument('--ancestor-state-snapshot')
        p.add_argument('--launch', required=True)
        a = p.parse_args(argv)
        need(bool(a.resource_from) == bool(a.resource_sha), 'Resource ancestor requires both path and SHA')
        print(json.dumps(build(a), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
