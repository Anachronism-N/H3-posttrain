"""Read-only fresh N5 GPU7 capture; refuse configs until verified recovery is real."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import uuid

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('fresh_main_review_v4_manager', RT/'curated_review_resource_manager_v4.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
base, worker = manager.base, manager.worker


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--completed-stream', required=True)
    p.add_argument('--authorize-release', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=60)
    args = p.parse_args(argv)
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Fresh capture must run on N5; no signals sent')
    path = Path(args.out).resolve()
    if path.parent != RT or path.exists() or args.poll_seconds <= 0:
        raise RuntimeError('New runtime config and positive polling required')
    old = json.loads((RT/'curated_review_watch_launch_v3.json').read_text())
    safe = json.loads((Path(old['queue'])/'resources_safe.json').read_text())
    if not all(safe.get(name) is True for name in ('all_holders_safe', 'locks_released', 'leases_restored', 'unknown_jobs_not_signaled')):
        raise RuntimeError('Old resource transaction not fully safe')
    stream = Path(args.completed_stream).resolve()
    plan_ref, shadow_state = worker.prepared_stream(stream, manager.ORIGINAL)
    manifest = ROOT/'curated222939_inputs_v1/manifest.summary.json'
    membership = worker.Membership(manifest)
    reader = worker.VerifiedReader(stream, manager.ORIGINAL, membership, plan_ref)
    reader.poll()
    if reader.pending or reader.offset != shadow_state['shadow_offset']:
        raise RuntimeError('Recovered source not yet fully verified; no placeholder config: '+str(reader.pending))
    gpu_uuid = base.uuid_for(7)
    running = base.apps()
    apps = [item for item in running if item['uuid'] == gpu_uuid]
    if len(apps) != 1:
        raise RuntimeError('GPU7 must contain exactly one project holder')
    holder = dict(base.identity(apps[0]['pid']), gpu=7, uuid=gpu_uuid)
    base.check_holder(holder, base.identity(holder['pid']), 7, gpu_uuid, running)
    lease = base.lease_snapshot(Path('/tmp/od_reserve_7'))
    cfg = copy.deepcopy(old)
    suffix = str(time.time_ns())+'-'+uuid.uuid4().hex[:12]
    watcher, wrapper = RT/'curated_review_watch_v4.py', RT/'curated_review_resource_manager_v4.py'
    capture = RT/('curated_review_watch_v4_capture-'+suffix+'.json')
    prepared = RT/('curated_review_watch_v4_source_prepared-'+suffix+'.json')
    worker.evidence.once(capture, {'node': 'N5', 'gpus': [7], 'holders': [holder],
        'captured_unix_ns': time.time_ns(), 'signals_sent': False, 'lease_modified': False,
        'lease_stat': None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns],
        'budget': base.headroom()})
    worker.evidence.once(prepared, {'schema': 'curated_review_watch_v4_source_prepared', 'source_prepared': True,
        'completed_stream': str(stream), 'recovery_plan_ref': plan_ref, 'committed_stream_offset': reader.offset,
        'committed_stream_sha256': reader.committed_sha, 'committed_stream_identity': reader.identity,
        'verified_terminal_source_count': sum(len(rows) for rows in reader.completed.values()),
        'total_source_count': 222939, 'expected_batch_count': 448, 'source_count_by_split': dict(membership.by_split),
        'all_source_labels_completed': False, 'malformed_records_skipped': False, 'GPU_model_load_performed': False})
    cfg.update(holders=[holder], release_authorized=args.authorize_release, phase='curated_review_watch_v4',
        queue=str(ROOT/'queues'/('curated_review_watch_v4-'+suffix)),
        producer={'path': str(watcher), 'sha256': base.sha(watcher)}, expected_batch_count=448,
        completed_stream=str(stream), recovery_plan_ref=plan_ref,
        source_prepared=True, launch_ready=args.authorize_release,
        fresh_holder_capture=worker.evidence.ref(capture), source_prepared_receipt=worker.evidence.ref(prepared),
        training_entry_updated=False)
    sources = {}
    for name, digest in old['source_sha256'].items():
        if base.sha(name) != digest:
            raise RuntimeError('Frozen old reviewer source changed: '+name)
        sources[name] = digest
    for item in (watcher, wrapper, Path(__file__).resolve(), RT/'completed_shadow_recovery_v1.py', capture, prepared,
            Path(plan_ref['path']), base.HOLDER):
        sources[str(item)] = base.sha(item)
    cfg['source_sha256'] = sources
    cfg['argv'][cfg['argv'].index(old['producer']['path'])] = str(watcher)
    cfg['argv'][cfg['argv'].index('--poll-seconds')+1] = str(args.poll_seconds)
    cfg['argv'] += ['--completed-stream', str(stream), '--recovery-plan', plan_ref['path'], '--recovery-plan-sha', plan_ref['sha256']]
    manager.check_model(cfg)
    if args.authorize_release:
        manager.validate(cfg)
    worker.evidence.once(path, cfg)
    print(json.dumps({'config': str(path), 'config_sha': base.sha(path), 'wrapper': str(wrapper),
        'wrapper_sha': base.sha(wrapper), 'base_manager_sha': base.sha(RT/'resource_manager.py'),
        'watcher_sha': base.sha(watcher), 'queue': cfg['queue'], 'source_prepared': True,
        'launch_ready': cfg['launch_ready'], 'source_prepared_receipt': cfg['source_prepared_receipt'],
        'required_headroom_bytes': base.SAFETY+base.LOAD, 'signals_sent': False, 'lease_modified': False}), flush=True)


if __name__ == '__main__':
    main()
