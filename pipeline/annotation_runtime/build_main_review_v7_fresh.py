"""Build a single immutable v7 source-prepared N5G7 config; never launch/release GPU."""
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


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def cpu_smoke(worker, binding, reader, cache, out):
    path = RT / 'main_review_v6_cpu_smoke.py'
    worker.need(cache.sha(path) == 'cba87aeeda5125493f6665a2ba88d657ec3debb17aa286cfbbcde5eddd0ee219', 'Frozen actual80 CPU helper changed')
    helper = module('frozen_main_v6_cpu_smoke_for_v7', path)
    proof = helper.run(worker, binding, reader, cache, out)
    proof.update(schema='main_review_v7_cpu_smoke_v1', worker_sha256=cache.sha(RT / 'main_review_watch_v7.py'),
        empty_row_offsets=reader.empty_row_offsets, reconciliation_restored_source_count=len(reader.restored_sources),
        recovered_sources_not_quality_approved=True, indexed_raw_source_count=len(reader.journal_keys),
        admitted_terminal_hint_count=sum(len(rows) for rows in reader.raw.values()),
        historical_helper_ref={'path': str(path), 'sha256': cache.sha(path)})
    worker.need(proof['native_cache_reused_source_count'] == 80 and proof['native_candidate_error_count'] == 7
                and proof['retained_historical_error_count'] == 7 and proof['new_VLM_calls'] == 0,
                'Actual original batch0 80 cached/7 retained errors/0 model calls required')
    return proof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-only', action='store_true', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--authorize-release', action='store_true')
    ap.add_argument('--reconciliation-manifest')
    a = ap.parse_args()
    if not a.tag or not all(c.isalnum() or c in '_-' for c in a.tag):
        raise ValueError('New immutable safe tag required')
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Source-only proof must use actual N5 node')
    worker = module('main_watch_v7_builder', RT / 'main_review_watch_v7.py')
    wrapper = module('main_resource_v7_builder', RT / 'main_review_resource_manager_v7.py')
    base = wrapper.base
    cache = worker.ByteCache()
    worker.recovery.q.sha = cache.sha
    cfg = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
    for path, digest in cfg['source_sha256'].items():
        worker.need(cache.sha(path) == digest, 'Original frozen source SHA differs: ' + path)
    wrapper.check_model(cfg)
    manifest_path = ROOT / 'curated222939_inputs_v1/manifest.summary.json'
    membership = worker.Membership(manifest_path)
    production = ROOT / 'curated222939_production_v1'
    repair = ROOT / 'curated222939_repair_v1/collector-recovery-v1'
    rule = worker.recovery.KNOWN[worker.recovery.BAD_OFFSET]
    hint_refs = []
    hints = []
    for sid in rule['ids']:
        path = repair / 'hints/train' / (sid + '.json')
        hint, ref = worker.recovery.document(path)
        hint_refs.append(ref)
        hints.append(hint)
    bad_ref = hints[0]['badline_ref']
    worker.need(all(h['badline_ref'] == bad_ref for h in hints), 'Both recovered hints must share the exact original badline')
    stream = production / 'completed.jsonl'
    before = stream.stat()
    identity = [before.st_dev, before.st_ino]
    with stream.open('rb') as f:
        data = f.read(before.st_size)
        worker.need([os.fstat(f.fileno()).st_dev, os.fstat(f.fileno()).st_ino] == identity, 'Original source inode differs')
    end = data.rfind(b'\n') + 1
    worker.need(end > worker.recovery.BAD_OFFSET + rule['size'], 'Original source has not published evidence span')
    prefix = data[:end]
    binding = {'schema': 'main_review_v7_original_stream_binding_v1',
        'main_prefix': {'path': str(stream), 'identity': identity, 'bytes': end, 'sha256': hashlib.sha256(prefix).hexdigest()},
        'incident': {'original_byte_begin': worker.recovery.BAD_OFFSET,
                     'original_byte_end': worker.recovery.BAD_OFFSET + rule['size'],
                     'badline_ref': bad_ref, 'hint_refs': hint_refs,
                     'recovery_module_ref': {'path': str(RT / 'curated_repair_collector_recovery_v1.py'), 'sha256': cache.sha(RT / 'curated_repair_collector_recovery_v1.py')}},
        'empty_incident': {'original_byte_begin': worker.EMPTY_OFFSET, 'original_byte_end': worker.EMPTY_OFFSET + 1,
            'source_identity': identity,
            'collector_snapshot_identity': worker.recovery.document(ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json')[0]['stable_badline_snapshots'][0]['identity'][:2],
            'receipt_ref': worker.recovery.document(ROOT / 'curated222939_repair_v1/collector-emptyline-v2/1192186.json')[1]},
        'manifest_ref': membership.ref, 'model_stage_ref': cfg['modelref'].copy(),
        'annotator_model_ref': {'path': cfg['argv'][cfg['argv'].index('--annotator-model-reference') + 1],
            'sha256': cache.sha(cfg['argv'][cfg['argv'].index('--annotator-model-reference') + 1])},
        'additional_edition': {'edition': 'collaborative-v2', 'results_root': str(ROOT / 'curated222939_collaborative_v2'),
            'out': str(ROOT / 'curated_independent_reviews_collaborative_v2'),
            'manifest': {'path': str(RT / 'curated_collaborative21000_auto_review_manifest_v2.json'), 'sha256': cache.sha(RT / 'curated_collaborative21000_auto_review_manifest_v2.json')}},
        'source_recovery_writes_performed': False, 'unknown_badlines_never_skipped': True,
        'validation_mode': 'finite_raw_index_batch_lazy', 'historical_scope_batch': 'train-b000000',
        'remaining_raw_sources_are_unverified': True}
    empty = binding['empty_incident']
    empty.update(historical_capture_identity=empty['collector_snapshot_identity'], fresh_current_node_identity=identity,
                 current_verified_prefix=binding['main_prefix'],
                 cross_mount_device_rebound_content_verified=identity[0] != empty['collector_snapshot_identity'][0])
    incident = worker.Incident(production, manifest_path, binding['incident'], empty)
    reader = worker.RawIndex(production, membership, incident=incident, binding=binding['main_prefix'])
    reader.scan()
    if a.reconciliation_manifest:
        binding['reconciliation_manifest_ref'] = worker.recovery.document(a.reconciliation_manifest)[1]
        binding['reconciliation_mount_bindings'] = worker.capture_reconciliation_mount_bindings(a.reconciliation_manifest)
        reader.reconciliation_mount_bindings = binding['reconciliation_mount_bindings']
        reader.reconcile(a.reconciliation_manifest, binding['reconciliation_manifest_ref'])
    worker.need(worker.EMPTY_OFFSET in reader.empty_row_offsets, 'Actual empty row zero-source evidence not proved')
    retained_ids = worker.historical_ids(ROOT / 'curated_independent_reviews_v1', 'train-b000000')
    worker.need(reader.verify('train-b000000', retained_ids), 'Historical current terminal evidence not fully published')
    worker.need(worker.recovery.BAD_OFFSET in reader.recovered_offsets, 'Source-only proof did not recover both known real terminals')
    worker.need(reader.pending is None or reader.pending['reason'] == 'partial_original_line', 'Original stream business/source error blocks prepared config: ' + str(reader.pending))
    extra_manifest = worker.v5.checked(binding['additional_edition']['manifest'])
    worker.need(extra_manifest['original_summary'] == membership.ref and extra_manifest['selected_source_count'] == 10000
                and extra_manifest['source_range'] == {'start': 21000, 'end': 31000}, 'Actual collaborative v2 source range differs')
    bids = {b['batch_id'] for b in extra_manifest['selected_batches']}
    worker.need(len(bids) == 20 and all(membership.descriptors[b['batch_id']] == b for b in extra_manifest['selected_batches']), 'Additional full batch descriptors differ')
    uuid = base.uuid_for(7)
    running = base.apps()
    admitted = [r for r in running if r['uuid'] == uuid]
    worker.need(len(admitted) == 1, 'GPU7 has unknown/extra processes; no release')
    expected = dict(base.identity(admitted[0]['pid']), gpu=7, uuid=uuid)
    base.check_holder(expected, base.identity(expected['pid']), 7, uuid, running)
    fd = os.pidfd_open(expected['pid'])
    try:
        worker.need(not select.select([fd], [], [], 0)[0], 'Holder exited during source-only preflight')
    finally:
        os.close(fd)
    reserve = Path('/tmp/od_reserve_7')
    lease = base.lease_snapshot(reserve)
    lock = Path('/tmp/ms12k_gpu_7.lock')
    lock_state = 'absent_manager_will_create'
    if lock.exists():
        lockfd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        try:
            s = os.fstat(lockfd)
            worker.need(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_size == 0, 'Unsafe GPU7 lock')
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            lock_state = 'available_unchanged'
        finally:
            os.close(lockfd)
    budget = base.headroom()
    worker.need(budget['headroom'] >= 160 * 2**30, 'Fresh 160GiB load+safety budget required')
    stem = 'main_review_watch_v7-' + a.tag
    queue = ROOT / 'queues' / stem
    binding_path = RT / (stem + '.source_binding.json')
    cpu_path = RT / (stem + '.cpu_smoke.json')
    config_path = RT / (stem + '.launch.json')
    preflight_path = RT / (stem + '.source_only.json')
    sandbox = RT / (stem + '.cpu_batch')
    worker.need(not queue.exists() and not sandbox.exists() and all(not p.exists() for p in (binding_path, cpu_path, config_path, preflight_path)), 'Fresh immutable queue/config/test namespace required')
    save_new(binding_path, binding)
    print(json.dumps({'event': 'historical_scope_verified_raw_index_pending', 'original_byte_offset': reader.offset,
        'indexed_raw_byte_offset': reader.indexed_offset, 'indexed_raw_source_count': len(reader.journal_keys),
        'verified_historical_sources': len(retained_ids),
        'terminal_sources': sum(len(x) for x in reader.completed.values()), 'recovered_offsets': reader.recovered_offsets,
        'source_binding': str(binding_path), 'GPU_model_loaded': False}), flush=True)
    proof = cpu_smoke(worker, binding, reader, cache, sandbox)
    save_new(cpu_path, proof)
    worker.need(proof['full_native_batch_path_verified'] and proof['immutable_original_membership_policy_unchanged'], 'Native immutable membership resume not proved')
    print(json.dumps({'event': 'native_full_batch_cpu_pass', 'receipt': str(cpu_path),
        'sources_cache_reused': proof['native_cache_reused_source_count'], 'AI_error_candidates': proof['native_candidate_error_count'],
        'new_VLM_calls': proof['new_VLM_calls']}), flush=True)
    cfg['holders'] = [expected]
    cfg['release_authorized'] = a.authorize_release
    cfg['phase'] = 'main_review_watch_v7'
    cfg['queue'] = str(queue)
    cfg['source_prepared'] = True
    cfg['expected_batch_count'] = 448
    cfg['source_binding'] = {'path': str(binding_path), 'sha256': cache.sha(binding_path)}
    cfg['source_only_preflight_receipt'] = str(preflight_path)
    cfg['producer'] = {'path': str(RT / 'main_review_watch_v7.py'), 'sha256': cache.sha(RT / 'main_review_watch_v7.py')}
    cfg['semantic_approval'] = False
    args = cfg['argv']
    args[args.index(str(RT / 'curated_review_watch_v3.py'))] = cfg['producer']['path']
    args.extend(['--source-binding', str(binding_path), '--additional-results-root', binding['additional_edition']['results_root'],
                 '--additional-out', binding['additional_edition']['out'], '--additional-manifest', binding['additional_edition']['manifest']['path']])
    for path in (RT / 'main_review_watch_v7.py', RT / 'main_review_resource_manager_v7.py', RT / 'build_main_review_v7_fresh.py',
                 RT / 'main_review_v6_cpu_smoke.py', RT / 'main_review_history_v1.py', RT / 'curated_repair_collector_recovery_v1.py',
                 worker.recovery.QUEUE_MODULE, RT / 'main_review_watch_v5.py', RT / 'main_review_watch_v6.py', RT / 'test_main_review_v7.py', binding_path, cpu_path, Path(bad_ref['path']),
                 Path(binding['additional_edition']['manifest']['path'])):
        cfg['source_sha256'][str(path)] = cache.sha(path)
    empty_receipt = worker.checked(binding['empty_incident']['receipt_ref'])
    immutable_refs = hint_refs + [binding['empty_incident']['receipt_ref'], empty_receipt['raw_ref'], empty_receipt['emptyline_handler_ref']]
    if a.reconciliation_manifest:
        cfg['argv'].extend(['--reconciliation-manifest', a.reconciliation_manifest])
        admission = worker.checked(binding['reconciliation_manifest_ref'])
        immutable_refs.extend([binding['reconciliation_manifest_ref'], admission['snapshot_ref'], *admission['record_refs']])
    for ref in immutable_refs:
        cfg['source_sha256'][ref['path']] = ref['sha256']
    # Authoritative summary binds all 448 descriptors; exact member bytes are
    # SHA checked lazily for each consumed batch, not a new 448-file I/O storm.
    for bid in reader.completed:
        b = membership.descriptors[bid]
        cfg['source_sha256'][b['path']] = b['sha256']
    prepared_cfg = dict(cfg, release_authorized=True)
    wrapper.validate(prepared_cfg)
    wrapper.check_model(cfg)
    base.check_lease_unchanged(reserve, lease)
    base.check_holder(expected, base.identity(expected['pid']), 7, base.uuid_for(7), base.apps())
    worker.need(base.headroom()['headroom'] >= 160 * 2**30, 'Fresh budget changed before config freeze')
    save_new(config_path, cfg)
    config_sha, base_sha = cache.sha(config_path), cache.sha(RT / 'resource_manager.py')
    base.check_sources(cfg, str(config_path), config_sha, base_sha)
    command = ['/opt/venv_eval/bin/python', '-B', '-u', str(RT / 'main_review_resource_manager_v7.py'),
               '--config', str(config_path), '--config-sha', config_sha, '--manager-sha', base_sha]
    result = {'schema': 'main_review_v7_source_only_v1', 'node': 'N5', 'gpus': [7], 'created_unix': time.time(),
        'holder': expected, 'gpu7_compute_apps': admitted, 'memory': budget, 'required_bytes': 160 * 2**30,
        'lock': lock_state, 'lease': None if lease is None else {'inode': lease.st_ino, 'mtime_ns': lease.st_mtime_ns, 'uid': lease.st_uid, 'bytes': lease.st_size},
        'config': {'path': str(config_path), 'sha256': config_sha}, 'source_binding': cfg['source_binding'],
        'cpu_smoke_receipt': {'path': str(cpu_path), 'sha256': cache.sha(cpu_path)},
        'worker_sha256': cfg['producer']['sha256'], 'wrapper_sha256': cache.sha(RT / 'main_review_resource_manager_v7.py'),
        'base_manager_sha256': base_sha, 'launch_command': shlex.join(command),
        'actual_local_model_inventory_config_index_stat_verified': True, 'model_weights_not_rehashed_or_restaged': True,
        'original_and_additional_out_separate': True, 'authoritative_original_source_count': 222939,
        'main_expected_batches': 448, 'additional_expected_sources': 10000, 'additional_expected_batches': 20,
        'release_authorized': a.authorize_release, 'holder_release_performed': False, 'GPU_model_loaded': False, 'queue_created': False,
        'validation_mode': 'finite_raw_index_batch_lazy', 'indexed_raw_sources': len(reader.journal_keys),
        'empty_row_offsets': reader.empty_row_offsets, 'reconciliation_restored_source_count': len(reader.restored_sources),
        'recovery_does_not_approve_quality_or_full_population': True,
        'verified_historical_sources': len(retained_ids), 'remaining_raw_sources_are_unverified': True,
        'original_manager_and_unknown_jobs_not_signaled': True}
    save_new(preflight_path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
