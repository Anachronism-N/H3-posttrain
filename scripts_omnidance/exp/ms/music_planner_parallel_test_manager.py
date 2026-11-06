"""Final test500 only, exact N4/Captioner or N7/MOSS eight-GPU handoff.

Config schema music-planner-parallel-test-manager-v1 retains the v3 receipt fields:
training_launch, full_complete, resources_safe, protocol, checkpoint_selection,
checkpoint_marker, final_selection, fingerprint, queue, out, source_sha256.
Additional explicit {path,sha256} refs: producer, frozen_eval_source,
frozen_adapter_source, generation_patch; generation_patch_entrypoint is family
specific. Require job=test, gpus=list(range(8)), release_authorized=true, captured
holders[8], loading_timeout_seconds and completion_timeout_seconds. Sources are
independently frozen under runtime, never mutable repository defaults. No retry.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import stat
import sys
import time
import uuid
import threading

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
RT = EXP / 'runtime'
FROZEN = RT / 'source_v11'
HANDOFF = EXP / 'eval_handoff_clean_v11'
DATA = EXP.parent / 'music_planner_fixed2s_baseline_20261025_clean_v1'
EVAL_MANAGER = RT / 'planner_eval_manager_v3.py'
EVAL_MANAGER_SHA = 'c387edfd246fd25ab3ac73d00bffe2f41d5f1450cba15df9509b53192ac63c9d'
SCOPE = {'N4': ('28.216.19.143', 'omni_captioner'), 'N7': ('28.216.18.144', 'moss')}
PATCH_ENTRIES = {'moss': 'apply_moss_generation_patch', 'omni_captioner': 'apply_generation_eos_patch'}
LOCK_ROOT = Path('/tmp')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('absolute nonsymlink file required: ' + str(path))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('regular file required: ' + str(path))
    return path


def ref_json(ref):
    if sha(regular(ref['path'])) != ref['sha256']:
        raise RuntimeError('explicit frozen reference SHA changed')
    return json.loads(Path(ref['path']).read_text())


def load_base():
    if sha(regular(EVAL_MANAGER)) != EVAL_MANAGER_SHA:
        raise RuntimeError('frozen v3 helper changed before import')
    spec = importlib.util.spec_from_file_location('_parallel_parent_frozen_v3', EVAL_MANAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_scope(cfg):
    if (cfg.get('schema') != 'music-planner-parallel-test-manager-v1'
            or SCOPE.get(cfg.get('node')) != (cfg.get('host'), cfg.get('family'))
            or cfg.get('job') != 'test' or cfg.get('gpus') != list(range(8))
            or any(type(g) is not int for g in cfg['gpus']) or cfg.get('release_authorized') is not True
            or cfg.get('human_quality_accepted') is True):
        raise RuntimeError('only explicitly authorized N4/Captioner or N7/MOSS all-eight final test admitted')
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    for path in (queue, out):
        if (not path.is_absolute() or path != path.resolve() or path == HANDOFF
                or not path.is_relative_to(HANDOFF) or path.exists() or path.is_symlink()):
            raise RuntimeError('fresh canonical handoff child required')
    if queue == out or queue.is_relative_to(out) or out.is_relative_to(queue):
        raise RuntimeError('queue/output overlap')
    family = cfg['family']
    exact = dict(training_launch=RT / (family + '_launch_clean_v11.json'),
        full_complete=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json',
        resources_safe=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json',
        protocol=HANDOFF / 'protocol.json', final_selection=HANDOFF / 'final_selection.json',
        checkpoint_selection=EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json',
        frozen_eval_source=FROZEN / 'music_planner_eval.py',
        frozen_adapter_source=FROZEN / 'music_planner_adapters.py')
    for key, path in exact.items():
        if cfg[key]['path'] != str(path):
            raise RuntimeError('wrong exact frozen receipt/source path: ' + key)
    for key in ('producer', 'generation_patch'):
        path = Path(cfg[key]['path'])
        if (not path.is_absolute() or path != path.resolve() or not path.is_relative_to(RT)
                or path.is_relative_to(FROZEN)):
            raise RuntimeError('new producer/patch require separately frozen runtime paths')
    if Path(cfg['producer']['path']).name != 'music_planner_parallel_test.py':
        raise RuntimeError('explicit new parallel producer required')
    if cfg['generation_patch_entrypoint'] != PATCH_ENTRIES[family]:
        raise RuntimeError('wrong explicitly pinned inference-only generation patch')
    for key, maximum in (('loading_timeout_seconds', 7200), ('completion_timeout_seconds', 172800)):
        if type(cfg.get(key)) is not int or not 0 < cfg[key] <= maximum:
            raise RuntimeError('explicit bounded load/completion deadline required')
    if cfg['completion_timeout_seconds'] < cfg['loading_timeout_seconds']:
        raise RuntimeError('completion deadline cannot precede loading deadline')
    if 'holders' in cfg:
        if (len(cfg['holders']) != 8 or [h.get('gpu') for h in cfg['holders']] != list(range(8))
                or any(type(h.get('gpu')) is not int for h in cfg['holders'])
                or len({h['pid'] for h in cfg['holders']}) != 8
                or len({h['uuid'] for h in cfg['holders']}) != 8):
            raise RuntimeError('eight unique captured exact holder GPU identities required')


def check_sources(cfg, config_path, config_sha, manager_sha, base):
    if sha(regular(__file__)) != manager_sha or sha(regular(config_path)) != config_sha:
        raise RuntimeError('new manager/config SHA changed')
    required = {**base.PINNED, str(EVAL_MANAGER): EVAL_MANAGER_SHA,
                str(Path(__file__).resolve()): manager_sha}
    for key in ('producer', 'frozen_eval_source', 'frozen_adapter_source', 'generation_patch'):
        ref = cfg[key]
        if key in ('frozen_eval_source', 'frozen_adapter_source') and required.get(ref['path']) != ref['sha256']:
            raise RuntimeError('underlying frozen training/eval SHA differs')
        required[ref['path']] = ref['sha256']
    for path, expected in required.items():
        if cfg['source_sha256'].get(path) != expected:
            raise RuntimeError('missing/conflicting independent source pin: ' + path)
    for path, expected in cfg['source_sha256'].items():
        if sha(regular(path)) != expected:
            raise RuntimeError('frozen source/input changed: ' + path)
    for key in ('training_launch', 'full_complete', 'resources_safe', 'protocol',
                'checkpoint_selection', 'checkpoint_marker', 'final_selection'):
        ref_json(cfg[key])


def check_lock(fd, gpu):
    path = regular(LOCK_ROOT / ('ms12k_gpu_%d.lock' % gpu))
    current, opened = path.lstat(), os.fstat(fd)
    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_size
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
        raise RuntimeError('existing GPU lock owner/type/content/inode changed')


def open_all_locks():
    locks = []
    try:
        for gpu in range(8):
            fd = os.open(regular(LOCK_ROOT / ('ms12k_gpu_%d.lock' % gpu)), os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            check_lock(fd, gpu)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_lock(fd, gpu)
        return locks
    except BaseException:
        for fd in locks:
            os.close(fd)
        raise


def check_all_holders(expected, manager):
    if (len(expected) != 8 or [h['gpu'] for h in expected] != list(range(8))
            or len({h['pid'] for h in expected}) != 8 or len({h['uuid'] for h in expected}) != 8):
        raise RuntimeError('exact eight unique holder scope required')
    for gpu, item in enumerate(expected):
        current = manager.identity(item['pid'])
        if current.get('state') in ('Z', 'X', 'x'):
            raise RuntimeError('holder process is not active')
        manager.check_holder(item, current, gpu, manager.uuid_for(gpu), manager.apps())


def capture_eight(cfg, manager):
    manager.stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    # Native identity checker validates uid/start/argv/CVD/UUID and blocked signals.
    capture = manager.capture_current(cfg['node'], cfg['family'])
    locks = open_all_locks()
    try:
        check_all_holders(capture['holders'], manager)
        for gpu, fd in enumerate(locks):
            check_lock(fd, gpu)
        if 'holders' in cfg:
            check_all_holders(cfg['holders'], manager)
    finally:
        for fd in locks:
            os.close(fd)
    capture.update(signals_sent=False, exact_eight_GPU_authorization=True)
    return capture


def claim_test(cfg, config_sha, source_refs, base):
    path = HANDOFF / ('test_once_' + cfg['family'] + '.json')
    base.new_json(path, dict(schema='music-planner-parallel-test-once-v1', family=cfg['family'],
        config_sha256=config_sha, checkpoint_selection=cfg['checkpoint_selection'],
        checkpoint_marker=cfg['checkpoint_marker'], final_selection=cfg['final_selection'],
        fingerprint=cfg['fingerprint'], source_refs=source_refs, world_size=8,
        out=cfg['out'], claimed_unix=time.time(), pid=os.getpid(), automatic_retry=False,
        test_used_for_selection=False, retry_requires_explicit_diagnosis_and_recorded_authorization=True))
    return path


def worker_argv(cfg, dispatch_ref):
    return ['/opt/venv_music_planner_v1/bin/python', '-m', 'torch.distributed.run',
            '--standalone', '--nnodes=1', '--nproc_per_node=8', '--max_restarts=0',
            cfg['producer']['path'], '--dispatch', dispatch_ref['path'],
            '--dispatch-sha', dispatch_ref['sha256'], '--producer-sha', cfg['producer']['sha256']]


def load_producer(cfg):
    if sha(regular(cfg['producer']['path'])) != cfg['producer']['sha256']:
        raise RuntimeError('explicit parallel producer SHA changed before import')
    spec = importlib.util.spec_from_file_location('_parallel_parent_new_producer', cfg['producer']['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_worker_outputs(cfg, dispatch, producer, ev):
    out = Path(cfg['out'])
    manifest = ref_json(dict(path=str(out / 'manifest.json'), sha256=sha(regular(out / 'manifest.json'))))
    records, byte_lines, receipts, diagnostics = producer.collect_shards(out, dispatch, ev)
    expected = dict(schema='music-planner-parallel-native-test-v1', mode='finaltest_parallel',
        family=cfg['family'], complete=True, world_size=8, rank_count=8,
        invocation_nonce=dispatch['invocation_nonce'], source_ids=dispatch['source_ids'],
        source_ids_sha256=dispatch['source_ids_sha256'], source_refs=dispatch['source_refs'],
        code=dispatch['source_refs']['producer'], frozen_eval_source=dispatch['source_refs']['frozen_eval_source'],
        frozen_adapter_source=dispatch['source_refs']['frozen_adapter_source'],
        checkpoint=dispatch['checkpoint'], checkpoint_identity=dispatch['checkpoint_identity'],
        config=dispatch['config'], model=dispatch['model'], base_identity=dispatch['base_audit']['identity'],
        current_official_base_full_content_SHA_verified=True,
        requested_generation_patch_ref=dispatch['source_refs']['generation_patch'],
        generation_patch_by_rank={str(rank): producer.read_receipt(out / ('loaded-rank%d.json' % rank))['generation_patch'] for rank in range(8)},
        data_refs=dispatch['data_refs'], protocol=dispatch['protocol'], final_selection=dispatch['final_selection'],
        checkpoint_selection=dispatch['checkpoint_selection'], checkpoint_marker=dispatch['checkpoint_marker'],
        decoding=base_decode(), teacher_forced=False, counterfactual=False,
        attempted=500, technical_collection_complete=True, pretrained_GPU_generation_executed=True,
        test_used_for_checkpoint_selection=False, automatic_retry=False,
        human_quality_accepted=False, automatic_quality_ranking=False, winner=None,
        rank_receipts=receipts, diagnostics=diagnostics, diagnostic_count=len(diagnostics),
        diagnostics_do_not_trigger_retry=True,
        actual_forward_count=sum(r['actual_forward_count'] for r in records),
        finite_native_logits_forward_count=sum(r['finite_native_logits_forward_count'] for r in records),
        native_logits_all_finite=True,
        actual_generated_tokens=sum(r['generated_tokens'] for r in records),
        case_GPU_assignment=[dict(id=r['id'], global_index=r['global_index'], rank=r['rank'], device=r['device']) for r in records])
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError('combined new producer manifest is not an actual complete native test500')
    with regular(out / 'raw.jsonl').open('rb') as stream:
        if list(stream) != byte_lines:
            raise RuntimeError('combined raw changed shard record bytes/order')
    if manifest.get('raw') != ev.file_ref(out / 'raw.jsonl') or manifest.get('summary') != ev.file_ref(out / 'summary.json'):
        raise RuntimeError('combined raw/summary provenance mismatch')
    if json.loads((out / 'summary.json').read_text()) != ev.summarize(records):
        raise RuntimeError('summary differs from original frozen summarize function')
    return dict(manifest=ev.file_ref(out / 'manifest.json'), raw=ev.file_ref(out / 'raw.jsonl'),
        complete_native_GPU_outputs_verified=True, technical_collection_complete=True,
        world_size=8, source_count=500, each_source_once=True, diagnostic_count=len(diagnostics),
        diagnostics=diagnostics, human_quality_accepted=False, automatic_retry=False)


def base_decode():
    return dict(seed=42, do_sample=False, max_new_tokens=2048,
        native_training_prefix_unchanged=True, extra_instruction=False,
        raw_output_cleaning=False, GT_length_truncation=False, added_penalties=False)


def release_all(entries, locks, manager, check):
    if len(entries) != 8 or len(locks) != 8 or any(e['leasefd'] is None or 'lease_inode' not in e for e in entries):
        raise RuntimeError('all eight locks and owned heartbeat leases required before any signal')
    manager.refresh_leases(entries)
    check_all_holders([e['expected'] for e in entries], manager)
    for e in entries:
        if select.select([e['fd']], [], [], 0)[0]:
            raise RuntimeError('exact holder exited before signal')
    for gpu, fd in enumerate(locks):
        check_lock(fd, gpu)
    check()
    if manager.headroom()['headroom'] < manager.required_budget({}):
        raise RuntimeError('fresh memory below same160GiB admission before any signal')
    for e in entries:
        manager.release_holder(e)
    for e in entries:
        if not select.select([e['fd']], [], [], 30)[0]:
            raise RuntimeError('authorized exact holder did not exit')


def start_lease_watchdog(entries, manager, interval=5):
    stop = threading.Event()
    errors = []
    def heartbeat():
        while not stop.wait(interval):
            try:
                manager.refresh_leases(entries)
            except BaseException as error:
                errors.append(error)
                break
    thread = threading.Thread(target=heartbeat, name='owned-eval-lease-heartbeat', daemon=True)
    thread.start()
    return stop, errors, thread


def stop_lease_watchdog(watchdog):
    if watchdog is not None:
        watchdog[0].set()
        watchdog[2].join()  # Never close/reuse lease FDs while a heartbeat is in progress.


def manage(cfg, config_path, config_sha, manager_sha):
    if sha(regular(__file__)) != manager_sha or sha(regular(config_path)) != config_sha:
        raise RuntimeError('new manager/config SHA changed before helper import')
    validate_scope(cfg)
    if 'holders' not in cfg:
        raise RuntimeError('explicit independently captured all-eight holders required for run')
    base = load_base()
    check_sources(cfg, config_path, config_sha, manager_sha, base)
    manager, fixed, ev = base.load_helpers()
    manager.stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    launch, result = base.check_completed(cfg, manager)
    protocol = base.check_protocol(cfg, ev)
    # All three independent actual best selections are reverified before any claim.
    base.check_final_selection(cfg, manager, fixed, ev, protocol)
    claim_path = HANDOFF / ('test_once_' + cfg['family'] + '.json')
    if claim_path.exists() or claim_path.is_symlink():
        raise FileExistsError('family test already claimed; no holder released and no retry')
    checkpoint, identity, selected = base.selected_checkpoint(cfg, result, protocol, fixed, ev)
    model = manager.stage.MODEL_BASE / cfg['family']
    audited_base = base.audit_base(model, cfg['family'], identity, fixed, ev, manager)
    producer = load_producer(cfg)
    splits, _ = ev.validate_data(DATA)
    rows = ev.select_rows(splits['test'], 'test', limit=500, seed=42)
    producer.partition(rows, 0)
    sources = {key: ev.file_ref(cfg[key]['path']) for key in
               ('producer', 'frozen_eval_source', 'frozen_adapter_source', 'generation_patch')}
    sources['manager'] = ev.file_ref(__file__)
    sources['frozen_eval_manager'] = ev.file_ref(EVAL_MANAGER)
    capture_eight(cfg, manager)
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    queue.mkdir(parents=True, exist_ok=False)
    logfd = os.open(queue / 'manager.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.close(logfd)
    def save(name, value):
        manager.save_receipt(queue, name, value, 'planner_parallel_final_test')
    watchdog = None
    def check():
        if watchdog is not None and watchdog[1]:
            raise RuntimeError("Owned lease heartbeat failed; no new GPU release") from watchdog[1][0]
        check_sources(cfg, config_path, config_sha, manager_sha, base)
        base.check_completed(cfg, manager)
        base.check_final_selection(cfg, manager, fixed, ev, protocol)
        fixed.checkpoint_selection_identity(cfg['checkpoint_selection']['path'],
            cfg['checkpoint_selection']['sha256'], cfg['checkpoint_marker']['sha256'], identity, cfg['family'], ev)
        for ref in selected['validation_candidates']:
            if ev.file_ref(ref['path']) != ref:
                raise RuntimeError('actual epoch validation candidate changed')
        if ev.file_ref(identity['run_config']['path']) != identity['run_config']:
            raise RuntimeError('original runconfig fingerprint changed')
        if base.checkpoint_snapshot(checkpoint) != selected['stat_snapshot']:
            raise RuntimeError('checkpoint changed after single full-content audit')
        _, digest, _, _, files = manager.stage.read_download(launch['modelref'], cfg['family'])
        marker = json.loads(Path(audited_base['identity']['official_stage']['receipt']['path']).read_text())
        manager.stage.check_staged(marker, digest, cfg['family'], files, model)
        for ref in audited_base['files']:
            if manager.stage.file_stat(Path(ref['path'])) != ref['stat']:
                raise RuntimeError('base changed after single full-content audit')
        if watchdog is not None:
            if watchdog[1]:
                raise RuntimeError('Owned lease heartbeat failed during integrity check') from watchdog[1][0]
            manager.refresh_leases(entries)
    save('integrity_receipt.json', dict(config_sha256=config_sha, manager_sha256=manager_sha,
        source_refs=sources, final_selection=cfg['final_selection'], checkpoint_identity=identity,
        selected_checkpoint=selected, base=audited_base, human_quality_accepted=False, real_GPU_eval_verified=False))
    locks, entries, restored = [], [], []
    owned = dict(child=None, started=False)
    signal.signal(signal.SIGTERM, manager.interrupted)
    signal.signal(signal.SIGINT, manager.interrupted)
    try:
        check()
        locks = open_all_locks()
        check_all_holders(cfg['holders'], manager)
        for gpu, expected in enumerate(cfg['holders']):
            reserve = LOCK_ROOT / ('od_reserve_%d' % gpu)
            old = manager.lease_snapshot(reserve)
            fd = os.pidfd_open(expected['pid'])
            e = dict(gpu=gpu, uuid=expected['uuid'], fd=fd, expected=expected, reserve=reserve,
                old=old, leasefd=None, released=False, original_env=None, original_cwd=None, queue=queue)
            entries.append(e)
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('captured exact holder already exited')
            current = manager.identity(expected['pid'])
            manager.check_holder(expected, current, gpu, manager.uuid_for(gpu), manager.apps())
            proc = Path('/proc') / str(expected['pid'])
            e['identity'] = current
            e['original_env'] = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if x)
            e['original_cwd'] = os.readlink(proc / 'cwd')
            manager.configure_local_restore(e, cfg)
        check()
        if manager.headroom()['headroom'] < manager.required_budget(cfg):
            raise RuntimeError('fresh memory below frozen160GiB admission; no holder released')
        for e in entries:
            manager.acquire_lease(e)
        watchdog = start_lease_watchdog(entries, manager)
        check()
        # Keep claim forever even on partial technical/holder failure. Never reuse test IDs.
        claim = claim_test(cfg, config_sha, sources, base)
        save('test_claim.json', dict(path=str(claim), config_sha256=config_sha, world_size=8))
        release_all(entries, locks, manager, check)
        manager.wait_gpu_cleanup(entries, queue, 'waiting_all_eight_GPU_cleanup')
        check()
        out.mkdir(parents=True, exist_ok=False)
        dispatch = dict(schema='music-planner-parallel-test-dispatch-v1', family=cfg['family'],
            fingerprint=cfg['fingerprint'], world_size=8, invocation_nonce=uuid.uuid4().hex,
            created_unix=time.time(), loading_timeout_seconds=cfg['loading_timeout_seconds'],
            completion_timeout_seconds=cfg['completion_timeout_seconds'], source_refs=sources,
            config=ev.file_ref(config_path), out=str(out), data=str(DATA), model=str(model),
            checkpoint=str(checkpoint), checkpoint_identity=identity,
            checkpoint_stat_snapshot=selected['stat_snapshot'], base_audit=audited_base,
            source_ids=[r['id'] for r in rows], source_ids_sha256=producer.ids_sha([r['id'] for r in rows]),
            data_refs=dict(summary=protocol['data']['summary'], **protocol['data']['splits']),
            protocol=cfg['protocol'], final_selection=cfg['final_selection'],
            checkpoint_selection=cfg['checkpoint_selection'], checkpoint_marker=cfg['checkpoint_marker'],
            generation_patch_entrypoint=cfg['generation_patch_entrypoint'], decoding=base.DECODE,
            test_used_for_checkpoint_selection=False)
        base.new_json(queue / 'dispatch.json', dispatch)
        dispatch_ref = ev.file_ref(queue / 'dispatch.json')
        argv = worker_argv(cfg, dispatch_ref)
        worker = dict(cfg, cwd=str(Path(cfg['producer']['path']).parent),
                      env=dict(launch['env'], CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'))
        save('admission.json', dict(gpus=list(range(8)), holders=[e['identity'] for e in entries],
            config_sha256=config_sha, producer=cfg['producer'], argv=argv, dispatch=dispatch_ref,
            required_memory_bytes=manager.required_budget(cfg), no_training_launched=True))
        check()
        producer.verify_dispatch(dispatch, cfg['producer']['sha256'])
        manager.run_child(worker, argv, 'test', entries, queue, save, owned)
        check()
        evidence = check_worker_outputs(cfg, dispatch, producer, ev)
        save('eval.complete.json', dict(evidence, worker_exit_code=0,
            all_eight_real_GPU_manifests_verified=True, final_selection=cfg['final_selection']))
    except BaseException as error:
        save('failure.json', dict(error=repr(error), worker_started=owned['started'], complete=False,
            raw_and_failed_outputs_preserved=True, unknown_jobs_not_signaled=True, automatic_test_retry=False))
    finally:
        # Native helper retains all locks/heartbeats while safely restoring holders,
        # only terminates this manager's torchrun child, then waits for owned holders.
        stop_lease_watchdog(watchdog)
        manager.finish_resources(entries, locks, queue, save, restored, owned)
    if (queue / 'failure.json').exists():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'run'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--manager-sha', required=True)
    args = parser.parse_args()
    if sha(regular(__file__)) != args.manager_sha:
        raise RuntimeError('new manager SHA changed before helper import')
    cfg = ref_json(dict(path=str(args.config), sha256=args.config_sha))
    validate_scope(cfg)
    base = load_base()
    check_sources(cfg, args.config, args.config_sha, args.manager_sha, base)
    if args.action == 'run':
        manage(cfg, args.config, args.config_sha, args.manager_sha)
    else:
        manager, fixed, ev = base.load_helpers()
        base.check_completed(cfg, manager)
        protocol = base.check_protocol(cfg, ev)
        base.check_final_selection(cfg, manager, fixed, ev, protocol)
        if (HANDOFF / ('test_once_' + cfg['family'] + '.json')).exists():
            raise FileExistsError('family test already claimed; capture refused')
        print(json.dumps(capture_eight(cfg, manager), indent=2))


if __name__ == '__main__':
    main()
