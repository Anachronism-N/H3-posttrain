"""Source-only builder: fresh N5 GPU7 proof/config, never release holders or launch VLM."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import stat
import subprocess
import time

RT = Path(__file__).resolve().parent
ROOT = RT.parent


def load(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--completed-stream', required=True)
    ap.add_argument('--stream-evidence', required=True)
    ap.add_argument('--stream-evidence-sha', required=True)
    ap.add_argument('--cpu-smoke-receipt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--source-only', action='store_true', required=True)
    a = ap.parse_args()
    assert a.tag and all(c.isalnum() or c in '_-' for c in a.tag)
    assert '28.216.19.137' in subprocess.check_output(['hostname', '-I'], text=True).split(), 'must inspect actual N5 node'
    spec = importlib.util.spec_from_file_location('resource_v4_builder', RT / 'curated_review_resource_manager_v4.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    base = wrapper.base
    cfg = load(RT / 'curated_review_watch_launch_v3.json')
    for path, expected in cfg['source_sha256'].items():
        assert base.sha(path) == expected, 'original source SHA changed: ' + path
    wrapper.check_model(cfg)
    evidence = Path(a.stream_evidence).resolve()
    stream = Path(a.completed_stream).resolve()
    assert evidence.is_relative_to(ROOT) and stream.is_relative_to(ROOT) and stream != ROOT / 'curated222939_production_v1/completed.jsonl'
    assert base.sha(evidence) == a.stream_evidence_sha, 'verified repair evidence must be fully SHA bound'
    json.loads(evidence.read_text())
    smoke = Path(a.cpu_smoke_receipt).resolve()
    proof = load(smoke)
    assert proof['source_count'] == 500 and proof['sample_count'] == 3 and proof['VLM_model_loaded'] is False and proof['new_VLM_calls'] == 0
    assert proof.get('actual_worker_history_selection_verified') is True and proof.get('historical_missing_terminal_fails_closed') is True, 'actual worker history preservation not proved'
    worker = RT / 'curated_review_watch_v4.py'
    assert proof['worker_sha256'] == base.sha(worker), 'CPU smoke does not bind current worker'
    assert proof['history_helper_sha256'] == base.sha(RT / 'main_review_history_v1.py'), 'CPU smoke helper SHA differs'
    manifest_path = ROOT / 'curated222939_inputs_v1/manifest.summary.json'
    manifest = load(manifest_path)
    assert manifest['total'] == 222939 and len(manifest['batches']) == 448
    allmembers = {}
    member_bindings = {}
    descriptors = {}
    for b in manifest['batches']:
        assert base.sha(b['path']) == b['sha256'], 'full member SHA changed'
        member_bindings[b['path']] = b['sha256']
        records = [json.loads(l) for l in Path(b['path']).read_text().splitlines() if l]
        assert len(records) == b['count'] and all(r['batch_id'] == b['batch_id'] and r['split'] == b['split'] for r in records)
        for row in records:
            assert row['id'] not in allmembers
            allmembers[row['id']] = row
        descriptors[b['batch_id']] = b
    assert len(allmembers) == 222939
    spec = importlib.util.spec_from_file_location('watch_v4_builder', worker)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    _, shadow_state = watcher.prepared_stream(stream, ROOT / 'curated222939_production_v1', {'path': str(evidence), 'sha256': a.stream_evidence_sha})
    before = stream.stat()
    committed = shadow_state['shadow_offset']
    assert before.st_size >= committed
    with stream.open('rb') as f:
        prefix = f.read(committed)
    assert prefix and prefix.endswith(b'\n') and hashlib.sha256(prefix).hexdigest() == shadow_state['shadow_sha256'], 'committed verified shadow prefix changed'
    after = stream.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino) and after.st_size >= committed
    membership = watcher.Membership(manifest_path)
    terminal_ids = {}
    for line in prefix.splitlines():
        v = json.loads(line)
        original = allmembers[v['id']]
        assert original['batch_id'] == v['batch_id'] and original['split'] == v['split']
        if v['id'] in terminal_ids:
            assert terminal_ids[v['id']] == v, 'verified terminal source revised'
        else:
            terminal_ids[v['id']] = v
        ref = v.get('joint_candidate_ref')
        if ref:
            assert Path(ref['path']).resolve().is_relative_to(ROOT / 'curated222939_production_v1/sources')
        watcher.terminal_record(ROOT / 'curated222939_production_v1', v, membership)
    uuid = base.uuid_for(7)
    running = base.apps()
    gpuapps = [p for p in running if p['uuid'] == uuid]
    assert len(gpuapps) == 1, 'GPU7 must contain exactly one project holder; unknown jobs untouched'
    holder = base.identity(gpuapps[0]['pid'])
    expected = dict(holder, gpu=7, uuid=uuid)
    base.check_holder(expected, holder, 7, uuid, running)
    fd = os.pidfd_open(holder['pid'])
    try:
        assert not select.select([fd], [], [], 0)[0], 'holder exited during source-only check'
    finally:
        os.close(fd)
    reserve = Path('/tmp/od_reserve_7')
    lease = base.lease_snapshot(reserve)
    lock = Path('/tmp/ms12k_gpu_7.lock')
    lock_state = 'absent_base_will_create'
    if lock.exists():
        lockfd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        try:
            s = os.fstat(lockfd)
            assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_size == 0, 'unsafe physical GPU7 lock'
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            lock_state = 'read_only_checked_available'
        finally:
            os.close(lockfd)
    budget = base.headroom()
    assert budget['headroom'] >= 160 * 2**30, 'fresh 160GiB headroom required'
    phase = 'curated_review_watch_v4_' + a.tag
    queue = ROOT / 'queues' / ('curated_review_watch_v4-' + a.tag)
    assert not queue.exists(), 'new immutable queue namespace required'
    binding_path = RT / (phase + '.stream_binding.json')
    config_path = RT / (phase + '.launch.json')
    preflight_path = RT / (phase + '.source_only.json')
    for path in (binding_path, config_path, preflight_path):
        assert not path.exists(), 'fresh immutable artifact already exists'
    prefix_ref = {'schema': 'verified_completed_append_prefix_v1', 'path': str(stream),
                  'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest(),
                  'device': before.st_dev, 'inode': before.st_ino, 'terminal_source_count': len(terminal_ids),
                  'verified_stream_evidence': {'path': str(evidence), 'sha256': a.stream_evidence_sha},
                  'whole_append_stream_not_frozen': True, 'original_completed_not_modified': True}
    save_new(binding_path, prefix_ref)
    cfg['holders'] = [expected]
    cfg['phase'] = 'curated_review_watch_v4'
    cfg['source_prepared'] = True
    cfg['launch_ready'] = True
    cfg['expected_batch_count'] = 448
    cfg['recovery_plan_ref'] = {'path': str(evidence), 'sha256': a.stream_evidence_sha}
    cfg['completed_stream'] = str(stream)
    cfg['queue'] = str(queue)
    cfg['producer'] = {'path': str(worker), 'sha256': base.sha(worker)}
    cfg['source_sha256'].update(member_bindings)
    for path in (worker, RT / 'main_review_history_v1.py', RT / 'completed_shadow_recovery_v1.py', RT / 'curated_review_resource_manager_v4.py', Path(__file__), binding_path, evidence, smoke):
        cfg['source_sha256'][str(path)] = base.sha(path)
    args = cfg['argv']
    args[args.index(str(RT / 'curated_review_watch_v3.py'))] = str(worker)
    args.extend(['--completed-stream', str(stream), '--recovery-plan', str(evidence), '--recovery-plan-sha', a.stream_evidence_sha])
    cfg['completed_stream_prefix_binding'] = str(binding_path)
    cfg['semantic_approval'] = False
    cfg['source_only_preflight_receipt'] = str(preflight_path)
    cfg['original_manager_and_unknown_jobs_not_signaled'] = True
    wrapper.validate(cfg)
    base.check_lease_unchanged(reserve, lease)
    base.check_holder(expected, base.identity(expected['pid']), 7, base.uuid_for(7), base.apps())
    assert base.headroom()['headroom'] >= 160 * 2**30
    save_new(config_path, cfg)
    config_sha, base_sha = base.sha(config_path), base.sha(RT / 'resource_manager.py')
    base.check_sources(cfg, str(config_path), config_sha, base_sha)
    command = ['/opt/venv_eval/bin/python', '-B', '-u', str(RT / 'curated_review_resource_manager_v4.py'),
               '--config', str(config_path), '--config-sha', config_sha, '--manager-sha', base_sha]
    receipt = {'schema': 'curated_review_v4_source_only_v1', 'node': 'N5', 'gpus': [7],
               'created_unix': time.time(), 'holder': expected, 'gpu7_compute_apps': gpuapps,
               'memory': budget, 'required_bytes': 160 * 2**30, 'lock': lock_state,
               'lease': None if lease is None else {'inode': lease.st_ino, 'mtime_ns': lease.st_mtime_ns, 'uid': lease.st_uid, 'bytes': lease.st_size},
               'actual_local_model_inventory_config_index_stats_verified': True, 'model_weights_not_rehashed_no_model_restaging': True,
               'full_member_count': 222939, 'full_batch_count': 448, 'verified_terminal_source_count': len(terminal_ids),
               'verified_stream_prefix': prefix_ref, 'cpu_smoke_receipt': {'path': str(smoke), 'sha256': base.sha(smoke)},
               'config': {'path': str(config_path), 'sha256': config_sha}, 'worker_sha256': cfg['producer']['sha256'],
               'wrapper_sha256': base.sha(RT / 'curated_review_resource_manager_v4.py'), 'base_manager_sha256': base_sha,
               'launch_command': shlex.join(command), 'GPU_model_loaded': False, 'holder_release_performed': False,
               'queue_created': False, 'original_manager_and_unknown_jobs_not_signaled': True}
    save_new(preflight_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
