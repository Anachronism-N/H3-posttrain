"""CPU evidence gates shared only by the independent coarse6 waiting chain."""
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

RT = Path(__file__).resolve().parent
ROOT = RT.parent
OUT = ROOT / 'curated222939_io_visual_repair_coarse6_v2'
PRIMARY = ROOT / 'curated222939_repair_v1'
QUEUE = ROOT / 'queues/coarse6when_safe_n3g01_v1'
PREVIOUS = RT / 'curated_collaborative20000_n3g01_launch_v1.json'
PREVIOUS_SHA = '226f6a051910e85434f3ac7b25505c8bb48718f3be65eca88b3492a1ab94514d'
TEMPLATE = RT / 'curated_io_visual_repair_coarse6_safe_template_v2.json'
TEMPLATE_SHA = 'd2a5c9642fbb0dfe312af22390e36c0c25954a4f6bdf82157a12f28f8a3d3e27'
WORKER = RT / 'curated_io_visual_repair_coarse6_v2.py'
WORKER_SHA = 'ad8d98cc6b90afd6c6f2c639f273961c045c1df7211e9c84ed8cef9821fa06c5'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
MANAGER = RT / 'coarse6when_safe_manager_v1.py'
DRIVER = RT / 'coarse6when_safe_driver_v1.py'
LAUNCH = RT / 'coarse6when_safe_n3g01_launch_v1.json'
REVISION = 'io-coarse6-reobserve-20261025-v2'
PAIR = [0, 1]
LEASE_KEYS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def load(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    if not Path(reference['path']).is_absolute() or sha(reference['path']) != reference['sha256']:
        raise RuntimeError('immutable reference changed: ' + reference['path'])
    return load(reference['path'])


def save_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def base_module():
    if sha(RT / 'resource_manager.py') != BASE_SHA:
        raise RuntimeError('original base resource manager changed')
    return module('coarse6_original_base', RT / 'resource_manager.py')


def frozen():
    template = checked({'path': str(TEMPLATE), 'sha256': TEMPLATE_SHA})
    if template['launch_authorized'] is not False or template['release_authorized'] is not False or template['gpus'] != [] or template['node'] is not None:
        raise RuntimeError('safe template must remain nonauthorizing')
    if template['worker_ref'] != {'path': str(WORKER), 'sha256': WORKER_SHA} or template['out'] != str(OUT):
        raise RuntimeError('worker/output binding changed')
    previous = checked({'path': str(PREVIOUS), 'sha256': PREVIOUS_SHA})
    consumer = checked(template['consumer_config_ref'])
    manifest = checked(template['jobs_manifest_ref'])
    checked(template['protocol_ref'])
    if manifest['global_revision'] != REVISION or len(manifest['jobs']) != 2 or template['global_revision'] != REVISION:
        raise RuntimeError('exact two-source immutable edition required')
    bindings = {str(TEMPLATE): TEMPLATE_SHA, str(WORKER): WORKER_SHA, str(PREVIOUS): PREVIOUS_SHA}
    bindings.update(consumer['source_sha256'])
    bindings.update(template['source_sha256'])
    for key in ('worker_ref', 'jobs_manifest_ref', 'protocol_ref', 'repair_queue_ref', 'consumer_config_ref'):
        bindings[template[key]['path']] = template[key]['sha256']
    jobs = []
    for hint in manifest['jobs']:
        job = checked(hint['job_ref'])
        bindings[hint['job_ref']['path']] = hint['job_ref']['sha256']
        if job['source_key'] != hint['source_key'] or Path(job['primary_budget_root']).resolve() != PRIMARY or Path(job['queue_root']).resolve() != PRIMARY:
            raise RuntimeError('canonical source budget root changed')
        jobs.append(job)
    keys = [j['source_key'] for j in jobs]
    if keys != ['train/OertkYlRRBiBIQbU', 'train/UbjwIZuNtSoLcUaG']:
        raise RuntimeError('two-source identity changed')
    selected = checked(previous['selected_sources'])
    if len(selected) != 1000 or [r['source_ordinal'] for r in selected] != list(range(20000, 21000)) or set(keys) & {r['split'] + '/' + r['id'] for r in selected}:
        raise RuntimeError('previous range is not exact/disjoint')
    if consumer['modelref'] != previous['modelref']:
        raise RuntimeError('exact original model identity required')
    validation_ref = {'path': str(OUT / 'cpu.validation.receipt.json'), 'sha256': '5b0ec30e2388040f6067fd5d6d47c90da72d866de3faa1059f9b1c3f04473ff9'}
    validation = checked(validation_ref)
    if validation.get('cpu_tests_passed') != 18 or validation.get('actual_visual_model_calls') != 0 or validation.get('actual_text_model_calls') != 0 or validation.get('global_revision') != REVISION:
        raise RuntimeError('original18 CPU validation receipt changed')
    prepare = checked(validation['prepare_receipt_ref'])
    grammar = checked(validation['native_XGrammar_receipt_ref'])
    if [v['actual_images'] for v in prepare['checks']] != [14, 14, 13] or len(grammar['checks']) != 3 or not all(v['native_XGrammar_schema_compiled'] is True for v in grammar['checks']):
        raise RuntimeError('original three actual input/grammar checks changed')
    pending = [validation_ref, validation, prepare, grammar]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                bindings[value['path']] = value['sha256']
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    return template, previous, consumer, manifest, jobs, bindings


def budget_paths(jobs):
    return [PRIMARY / 'attempt-budget' / (job['source_key'] + '.json') for job in jobs]


def require_unreserved(jobs):
    occupied = [str(p) for p in budget_paths(jobs) if os.path.lexists(p)]
    if occupied:
        raise RuntimeError('canonical once-only source budget reserved; preserve remaining_pending, no call/redispatch: ' + ','.join(occupied))


def lease_value(value):
    return None if value is None else {key: getattr(value, key) for key in LEASE_KEYS}


def verify_lease_value(value):
    if value is not None and (set(value) != set(LEASE_KEYS) or any(type(value[k]) is not int for k in LEASE_KEYS)):
        raise RuntimeError('lease requires full exact integer nanosecond/stat values')


def unchanged_lease(base, path, expected):
    verify_lease_value(expected)
    if lease_value(base.lease_snapshot(path)) != expected:
        raise RuntimeError('exact nanosecond lease changed: ' + str(path))


def previous_completed(previous):
    queue = Path(previous['queue'])
    if queue != ROOT / 'queues/curated_collaborative20000_n3g01_v1' or previous['out'] != str(ROOT / 'curated222939_collaborative_v1') or previous['gpus'] != PAIR or previous['node'] != 'N3':
        raise RuntimeError('previous physical pair/output changed')
    if (queue / 'failure.json').exists():
        raise RuntimeError('previous failure present; waiting for coordinator, no success')
    exit_path = queue / 'worker_exit.json'
    if not exit_path.exists():
        raise RuntimeError('previous finite1000 worker still active; worker_exit absent')
    exit_value = load(exit_path)
    if type(exit_value.get('exit_code')) is not int or exit_value['exit_code'] != 0:
        raise RuntimeError('previous worker failed/unknown exit; no success')
    admission = load(queue / 'admission.json')
    if admission['config_sha256'] != PREVIOUS_SHA or admission['model_identity_sha256'] != previous['modelref']['model_identity_sha256'] or admission['source_sha256'] != previous['source_sha256'] or admission['producer'] != previous['producer'] or admission['out'] != previous['out'] or admission['node'] != 'N3' or admission['gpus'] != PAIR:
        raise RuntimeError('previous actual admission identity mismatch')
    if exit_value.get('source_unchanged') is not True or exit_value.get('targetcount') != 1000 or exit_value.get('input_count') != 222939 or exit_value.get('producer') != previous['producer'] or exit_value.get('plan') != previous['plan']:
        raise RuntimeError('previous actual exit provenance/count changed')
    bindings = dict(previous['source_sha256'])
    for key in ('producer', 'plan', 'modelref', 'selected_sources'):
        bindings[previous[key]['path']] = previous[key]['sha256']
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise RuntimeError('previous source/model config changed: ' + path)
    rows = checked(previous['selected_sources'])
    terminals = []
    for row in rows:
        path = Path(previous['out']) / 'sources' / row['split'] / row['id'] / 'consumer.terminal.json'
        value = load(path)
        if any(value.get(k) != row[k] for k in ('id', 'split', 'batch_id')) or value.get('params_sha256') != previous['consumer_params_sha256'] or value.get('status') not in ('joint_candidate', 'incomplete_candidate', 'failed', 'producer_failed'):
            raise RuntimeError('previous exact terminal identity/params/status mismatch')
        if value.get('joint_candidate_ref'):
            checked(value['joint_candidate_ref'])
        terminals.append(ref(path))
    markers = []
    for name, value in exit_value.get('summaries', {}).items():
        path = Path(name)
        if not path.resolve().is_relative_to(Path(previous['out']) / 'runs'):
            raise RuntimeError('previous completion marker outside actual run')
        completed = load(path)
        protocol = load(path.with_name('protocol.json'))
        summary = value.get('summary', {})
        params = hashlib.sha256((json.dumps(protocol['configuration'], ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')).hexdigest()
        if completed.get('selected_total') != 1000 or completed.get('queue_sources_consumed') != 1000 or protocol.get('consumer_wrapper') != previous['producer'] or params != previous['consumer_params_sha256'] or protocol.get('manifest_summary') != previous['manifest_summary'] or any(summary.get(k) != v for k, v in {'selected_source_count': 1000, 'processed_source_count': 1000, 'total_source_count': 222939, 'plan_sha256': previous['plan']['sha256']}.items()) or value.get('selection', {}).get('selected_ids') != [r['id'] for r in rows]:
            raise RuntimeError('previous completion partial/params/provenance mismatch')
        markers.append(ref(path))
    if not markers or len(terminals) != 1000:
        raise RuntimeError('previous finite1000 actual completion missing')
    safe = load(queue / 'resources_safe.json')
    for key in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled'):
        if safe.get(key) is not True:
            raise RuntimeError('previous resources not actually returned: ' + key)
    restored = [load(queue / ('holder_restored_gpu%d.json' % g)) for g in PAIR]
    return {'worker_exit_ref': ref(exit_path), 'resources_safe_ref': ref(queue / 'resources_safe.json'), 'completion_refs': markers,
            'terminal_count': len(terminals), 'terminal_refs': terminals, 'restored': restored}


def fresh_pair(base, evidence):
    if '28.216.19.213' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('CPU waiter must execute locally on physical N3')
    locks, owners, leases = [], [], {}
    try:
        for gpu, restored in zip(PAIR, evidence['restored']):
            lock = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
            fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            s = os.fstat(fd)
            if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size != 0:
                raise RuntimeError('unsafe existing physical lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = Path('/tmp/od_reserve_%d' % gpu)
            leases[str(path)] = lease_value(base.lease_snapshot(path))
            uuid = base.uuid_for(gpu)
            active = [a for a in base.apps() if a['uuid'] == uuid]
            if len(active) != 1:
                raise RuntimeError('pair not exclusively restored project holders; no app touched')
            expected = dict(restored['holder_identity'], gpu=gpu, uuid=restored['uuid'])
            if restored['gpu'] != gpu or restored['restore_argv'] != base.RESTORE_ARGV:
                raise RuntimeError('previous restored holder argv mismatch')
            current = base.identity(active[0]['pid'])
            base.check_holder(expected, current, gpu, uuid, base.apps())
            pidfd = os.pidfd_open(current['pid'])
            try:
                if select.select([pidfd], [], [], 0)[0]:
                    raise RuntimeError('restored holder exited during fresh check')
            finally:
                os.close(pidfd)
            owners.append(dict(current, gpu=gpu, uuid=uuid))
        budget = base.headroom()
        if budget['headroom'] < 160 * 2**30:
            raise RuntimeError('fresh headroom below 160GiB')
        for gpu, owner in zip(PAIR, owners):
            base.check_holder(owner, base.identity(owner['pid']), gpu, base.uuid_for(gpu), base.apps())
            unchanged_lease(base, Path('/tmp/od_reserve_%d' % gpu), leases['/tmp/od_reserve_%d' % gpu])
        return {'holders': owners, 'lease_snapshot_ns': leases, 'budget': budget, 'captured_unix_ns': time.time_ns(), 'signals_sent': False}
    finally:
        for fd in locks:
            os.close(fd)


def proc_lock_owned(pid, path):
    s = Path(path).stat()
    expected = '%02x:%02x:%d' % (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino)
    for line in Path('/proc/locks').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and fields[4] == str(pid) and fields[5] == expected:
            return True
    return False


def fd_inode_owned(pid, dev, ino):
    for path in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            s = path.stat()
            if (s.st_dev, s.st_ino) == (dev, ino):
                return True
        except FileNotFoundError:
            continue
    return False
