"""One-shot N6G3 preflight/capture; never release holders or execute a model.

--preflight writes nothing. --prepare freezes an isolated fresh capture/config
for a separate explicit native resource-manager command. No waiter/background.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import time
import uuid

RT = Path(__file__).resolve().parent
MANAGER = RT / 'oldbar_explicit_review_resource_manager_v1.py'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_module(args):
    expected = {Path(__file__).resolve(): args.self_sha, MANAGER: args.manager_sha,
        RT / 'oldbar_explicit_review_v1.py': args.worker_sha,
        RT / 'curated_review_extension_resource_manager_v1.py': args.support_sha,
        RT / 'resource_manager.py': BASE_SHA}
    for path, sha in expected.items():
        if digest(path) != sha:
            raise RuntimeError('Frozen executable SHA changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('prepare_oldbar3_resource_manager', MANAGER)
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    return manager


def fresh_capture(manager):
    base = manager.base
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Oldbar3 fresh capture must run on N6')
    lockfd = pidfd = None
    try:
        lock = Path('/tmp/ms12k_gpu_3.lock')
        if lock.exists() or lock.is_symlink():
            lockfd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            st = os.fstat(lockfd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size != 0:
                raise RuntimeError('Unknown GPU3 lock shape/owner/content, left untouched')
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
        gpu_uuid = base.uuid_for(3)
        running = base.apps()
        active = [r for r in running if r['uuid'] == gpu_uuid]
        if len(active) != 1:
            raise RuntimeError('GPU3 needs one exact project holder; unknown processes untouched')
        owner = dict(base.identity(active[0]['pid']), gpu=3, uuid=gpu_uuid)
        base.check_holder(owner, base.identity(owner['pid']), 3, gpu_uuid, running)
        pidfd = os.pidfd_open(owner['pid'])
        if select.select([pidfd], [], [], 0)[0]:
            raise RuntimeError('Captured holder already exited')
        budget = base.headroom()
        if base.required_budget({}) != 160 * 2**30 or budget['headroom'] < base.required_budget({}):
            raise RuntimeError('Original 160GiB admission headroom required, no budget relaxation')
        base.check_holder(owner, base.identity(owner['pid']), 3, base.uuid_for(3), base.apps())
        if manager.lease_value(base.lease_snapshot(Path('/tmp/od_reserve_3'))) != manager.lease_value(lease):
            raise RuntimeError('GPU3 lease changed during capture')
        return dict(node='N6', gpus=[3], holders=[owner], lease=manager.lease_value(lease), headroom=budget,
                    captured_unix_ns=time.time_ns(), signals_sent=False, gpu_started=False, leases_modified=False)
    finally:
        if pidfd is not None:
            os.close(pidfd)
        if lockfd is not None:
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            os.close(lockfd)


def frozen_config(manager):
    base = manager.base
    rows, refs = manager.source_references()
    code = [Path(__file__).resolve(), MANAGER, RT / 'resource_manager.py',
            RT / 'curated_review_extension_resource_manager_v1.py', RT / 'oldbar_explicit_review_v1.py']
    code += list((RT / 'source_curated_review_v2').glob('*.py'))
    code += list((RT / 'source_curated_review_v2' / 'motionscript_v2').glob('*.py'))
    references = list(refs)
    for path in code + [manager.STAGE, manager.ANNOTATOR]:
        references.append(dict(path=str(path), sha256=manager.cached_sha(path)))
    sources = {}
    for ref in references:
        if ref['path'] in sources and sources[ref['path']] != ref['sha256']:
            raise RuntimeError('Conflicting source SHA bindings')
        sources[ref['path']] = ref['sha256']
    model = json.loads(manager.STAGE.read_bytes())['model_identity']
    argv = ['/opt/venv_vllm/bin/python', '-B', '-u', str(RT / 'oldbar_explicit_review_v1.py'),
        '--manifest', str(manager.MANIFEST), '--out', str(manager.OUT), '--run',
        '--model', str(manager.MODEL), '--model-reference', str(manager.STAGE),
        '--annotator-model-reference', str(manager.ANNOTATOR),
        '--ffprobe', '/opt/ms_tools/ffprobe', '--ffmpeg', '/usr/local/bin/ffmpeg']
    cfg = dict(node='N6', gpus=[3], phase=manager.PHASE, input_count=222939, targetcount=3,
        population_count=3, source_selection=dict(shard=0, nshard=1, limit=3),
        plan=dict(path=str(manager.MANIFEST), sha256=manager.worker.PILOT_SHA256),
        producer=dict(path=str(RT / 'oldbar_explicit_review_v1.py'), sha256=manager.cached_sha(RT / 'oldbar_explicit_review_v1.py')),
        resource_manager_ref=dict(path=str(MANAGER), sha256=manager.cached_sha(MANAGER)),
        modelref=dict(path=str(manager.STAGE), sha256=manager.STAGE_SHA, model_identity_sha256=model['sha256']),
        out=str(manager.OUT), queue=str(manager.QUEUE), cwd='/tmp', argv=argv,
        env=dict(CUDA_VISIBLE_DEVICES='3', PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH='',
                 MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages', OMP_NUM_THREADS='2'),
        source_sha256=sources, ready_source_refs=refs, holders=[], holder_capture=None,
        release_authorized=False, semantic_approval=False, training_entry_updated=False,
        reuse_existing_results=True, original_curated_batches_completed=False)
    manager.check_model(cfg)
    return cfg


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(manager, cfg, *, launch_dir):
    if manager.QUEUE.exists():
        raise RuntimeError('Unique oldbar3 queue already exists; inspect resource_safe before any fresh admission')
    # Capture only AFTER potentially slow content hashes; manager enforces freshness.
    capture = fresh_capture(manager)
    token = uuid.uuid4().hex
    root = Path(launch_dir).resolve(strict=True)
    capture_path = root / ('oldbar3_n6g3_capture_' + token + '.json')
    launch = root / ('oldbar3_n6g3_launch_' + token + '.json')
    save_new(capture_path, capture)
    cfg = dict(cfg, holders=capture['holders'], release_authorized=True,
        holder_capture=dict(path=str(capture_path), sha256=digest(capture_path)))
    cfg['source_sha256'][str(capture_path)] = digest(capture_path)
    manager.validate(cfg)
    manager.check_model(cfg)
    save_new(launch, cfg)
    manager.base.check_sources(cfg, launch, digest(launch), BASE_SHA)
    return dict(config=str(launch), config_sha256=digest(launch), resource_manager=str(MANAGER),
                base_manager_sha256=BASE_SHA, signals_sent=False, GPU_model_loaded=False,
                next_command=['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER),
                    '--config', str(launch), '--config-sha', digest(launch), '--manager-sha', BASE_SHA])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    p.add_argument('--self-sha', required=True)
    p.add_argument('--manager-sha', required=True)
    p.add_argument('--worker-sha', required=True)
    p.add_argument('--support-sha', required=True)
    p.add_argument('--launch-dir', default=str(RT))
    args = p.parse_args(argv)
    manager = checked_module(args)
    if manager.QUEUE.exists():
        raise RuntimeError("Unique oldbar3 queue already exists; never prepare or dispatch it twice")
    cfg = frozen_config(manager)
    if args.preflight:
        capture = fresh_capture(manager)
        print(json.dumps(dict(ready=True, read_only=True, capture=capture,
            original_curated_count=222939, oldbar_selected_count=3,
            signals_sent=False, GPU_model_loaded=False), ensure_ascii=False))
    else:
        print(json.dumps(prepare(manager, cfg, launch_dir=args.launch_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
