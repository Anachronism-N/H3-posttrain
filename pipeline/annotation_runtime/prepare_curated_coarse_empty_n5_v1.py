"""CPU-only immutable N5G01 finite-three builder and shared evidence gates."""
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

RT = Path(__file__).resolve().parent
BAR = RT.parent
OUT = BAR / 'curated_coarse_empty_window_reobserve_v1'
PRIMARY = BAR / 'curated222939_repair_v1'
QUEUE = BAR / 'queues/curated_coarse_empty_n5g01_v1'
LAUNCH = RT / 'curated_coarse_empty_n5g01_launch_v1.json'
MANAGER = RT / 'curated_coarse_empty_n5_resource_manager_v1.py'
DRIVER = RT / 'curated_coarse_empty_n5_driver_v1.py'
WORKER = RT / 'curated_coarse_empty_window_reobserve_v1.py'
WORKER_SHA = '93058e1147a821ffe5032e7907cac6dd284773662903413b333a27409c4dd602'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
CONSUMER = RT / 'curated_consumer_s0_v1_template.json'
CONSUMER_SHA = '7e72d21429d47d9de689c57671ec09ea7d48f6e1ae2b4f48ccafe9cdfa3e849c'
PREVIOUS = [(RT / 'curated_collaborative21000_n5g0_launch_v2.json', '6d782185b694ee7cefd84f219f43213309c6cafebea14936edced8daa35172f9'),
            (RT / 'curated_collaborative21000_n5g1_launch_v2.json', 'b1828e63841ba214bbaef72f899d7c6ffad08a097f90b706b83a904e85b800bf')]
OLD_MANAGER_SHA = '7164c9e8041e022f484efe6bfb65afb8b609e5ad95b3e82e902dfe7d58998656'
REVISION = 'empty-coarse-reobserve-20261025-v1'
PHASE = 'curated_coarse_empty_n5g01_v1'
PAIR = [0, 1]
TARGETS = [('train/AuuitdcBJyGOnWaH', 'w00000'), ('train/AzPomlvsANsevIFb', 'w00001'), ('train/DxNCjtNsxcazNpGb', 'w00001')]
LEASE_KEYS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
_hash_cache = {}


def sha(path):
    path = Path(path)
    before = path.stat()
    key = (str(path.resolve()), before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if key in _hash_cache:
        return _hash_cache[key]
    h = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if any(getattr(before, k) != getattr(s, k) for s in (opened, finished, after) for k in LEASE_KEYS):
        raise RuntimeError('source changed during full SHA: ' + str(path))
    _hash_cache[key] = h.hexdigest()
    return h.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def load(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    if not Path(reference['path']).is_absolute() or sha(reference['path']) != reference['sha256']:
        raise RuntimeError('immutable SHA changed: ' + reference['path'])
    return load(reference['path'])


def save_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def base_module():
    if sha(RT / 'resource_manager.py') != BASE_SHA:
        raise RuntimeError('original base manager SHA changed')
    base = module('empty_n5_original_resource_manager', RT / 'resource_manager.py')
    base.sha = sha  # Same full SHA with stable dev/inode/size/mtime/ctime cache.
    return base


def worker_module():
    if sha(WORKER) != WORKER_SHA:
        raise RuntimeError('new worker not final frozen SHA')
    return module('empty_n5_frozen_worker', WORKER)


def lease_value(value):
    return None if value is None else {key: getattr(value, key) for key in LEASE_KEYS}


def verify_lease_value(value):
    if value is not None and (set(value) != set(LEASE_KEYS) or any(type(value[k]) is not int for k in LEASE_KEYS)):
        raise RuntimeError('exact integer nanosecond lease stat required')


def fd_inode_owned(pid, dev, ino):
    for path in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            s = path.stat()
            if (s.st_dev, s.st_ino) == (dev, ino):
                return True
        except FileNotFoundError:
            pass
    return False


def proc_lock_owned(pid, path):
    s = Path(path).stat()
    expected = '%02x:%02x:%d' % (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino)
    return any(len(v) >= 8 and v[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and v[4] == str(pid) and v[5] == expected
               for v in (line.split() for line in Path('/proc/locks').read_text().splitlines()))


def require_unreserved(jobs):
    for job in jobs:
        if Path(job['primary_budget_root']).resolve() != PRIMARY or Path(job['queue_root']).resolve() != PRIMARY:
            raise RuntimeError('original canonical budget root changed')
        path = PRIMARY / 'attempt-budget' / (job['source_key'] + '.json')
        if os.path.lexists(path):
            raise RuntimeError('canonical budget already reserved across editions; no redispatch: ' + str(path))


def references(value):
    result = {}
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str):
                name, digest = item['path'], item['sha256']
                if name in result and result[name] != digest:
                    raise RuntimeError('conflicting immutable SHA binding')
                result[name] = digest
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return result


def frozen(worker=None, prepare=False, probe=False):
    worker = worker or worker_module()
    consumer = checked({'path': str(CONSUMER), 'sha256': CONSUMER_SHA})
    if prepare:
        worker.prepare()
    protocol_ref = ref(OUT / 'protocol.json')
    protocol = checked(protocol_ref)
    manifest_ref = ref(OUT / 'jobs.manifest.json')
    manifest = checked(manifest_ref)
    evidence_ref = ref(OUT / 'cpu.prepare.evidence.json')
    evidence = checked(evidence_ref)
    if manifest.get('source_count') != 3 or len(manifest.get('jobs', [])) != 3 or manifest.get('global_revision') != REVISION or manifest.get('maximum_total_visual_calls') != 3 or manifest.get('maximum_text_model_calls') != 0:
        raise RuntimeError('complete exact three-job manifest required')
    if protocol != worker.protocol() or protocol['consumer_config_ref'] != ref(CONSUMER):
        raise RuntimeError('frozen native protocol/model parameters changed')
    if any(evidence.get(k) != v for k, v in {'source_count': 3, 'model_loads': 0, 'visual_model_calls': 0, 'text_model_calls': 0, 'budget_reservations_performed': 0, 'GPU_dispatch_performed': False, 'global_revision': REVISION}.items()):
        raise RuntimeError('CPU-only source evidence changed')
    jobs = [checked(hint['job_ref']) for hint in manifest['jobs']]
    if [(j['source_key'], j['target_window_id']) for j in jobs] != TARGETS or [h['source_key'] for h in manifest['jobs']] != [j['source_key'] for j in jobs]:
        raise RuntimeError('fixed three original source/window identities changed')
    if len(evidence.get('checks', [])) != 3 or [c['actual_image_count'] for c in evidence['checks']] != [11, 8, 11]:
        raise RuntimeError('original eight RGB plus 3/0/3 ROI image budget changed')
    bindings = {str(WORKER): WORKER_SHA, str(CONSUMER): CONSUMER_SHA, str(RT / 'resource_manager.py'): BASE_SHA}
    for p in (Path(__file__).resolve(), MANAGER, DRIVER):
        bindings[str(p)] = sha(p)
    for value in (protocol_ref, protocol, manifest_ref, manifest, evidence_ref, evidence, jobs):
        bindings.update(references(value))
    bindings.update(worker.bar.code_identity())
    bindings[str(worker.pipeline.__file__)] = sha(worker.pipeline.__file__)
    for hint, job, check in zip(manifest['jobs'], jobs, evidence['checks']):
        original, joint, prepared, plan = worker.verify_job(job, probe=probe)
        request = worker.make_request(job, hint['job_ref'], prepared, worker.worker_args())
        if check['id'] != job['id'] or check['window_id'] != job['target_window_id'] or checked(check['request_preview_ref']) != request or check['request_digest'] != worker.q.digest(request):
            raise RuntimeError('original strict native request preview/source binding changed')
        if check['original_style_result_digest'] != worker.q.digest(joint['style_result']) or check['original_untargeted_observations_digest'] != worker.q.digest([o for o in original['window_observations'] if o['window']['window_id'] != job['target_window_id']]):
            raise RuntimeError('untargeted original window/style evidence changed')
        bindings.update(references(plan['source_refs']))
    # Reuse the frozen queue helper's already-computed, stat-validated full CSV SHA.
    for entry in worker.q.ORIGINAL_CSV_CACHE.values():
        path = Path(entry['path'])
        current = path.stat()
        if tuple(getattr(current, k) for k in worker.q.CSV_STAT_KEYS) != entry['stat']:
            raise RuntimeError('registered original CSV changed after full SHA')
        key = (str(path.resolve()), current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
        _hash_cache[key] = entry['sha256']
    for name, digest in bindings.items():
        if not Path(name).is_absolute() or sha(name) != digest:
            raise RuntimeError('source SHA changed: ' + name)
    return {'consumer': consumer, 'manifest_ref': manifest_ref, 'manifest': manifest, 'protocol_ref': protocol_ref,
            'evidence_ref': evidence_ref, 'jobs': jobs, 'bindings': bindings}


def verify_config(cfg, contract):
    consumer = contract['consumer']
    if cfg.get('node') != 'N5' or cfg.get('gpus') != PAIR or cfg.get('targetcount') != 3 or cfg.get('input_count') != 222939 or cfg.get('phase') != PHASE or cfg.get('global_revision') != REVISION:
        raise RuntimeError('only N5 physical0,1 finite three admitted')
    if cfg.get('queue') != str(QUEUE) or cfg.get('out') != str(OUT) or cfg.get('reuse_existing_results') is not True:
        raise RuntimeError('new independent queue/output required')
    for key, expected in {'producer': ref(DRIVER), 'modelref': consumer['modelref'], 'plan': consumer['plan'], 'consumer_config_ref': ref(CONSUMER), 'jobs_manifest_ref': contract['manifest_ref'], 'protocol_ref': contract['protocol_ref'], 'cpu_evidence_ref': contract['evidence_ref']}.items():
        if cfg.get(key) != expected:
            raise RuntimeError('frozen config binding changed: ' + key)
    if cfg.get('argv') != ['/opt/venv_vllm/bin/python', '-B', '-u', str(DRIVER), '--launch', str(LAUNCH)] or cfg.get('cwd') != str(RT):
        raise RuntimeError('only direct native finite driver allowed')
    if cfg.get('env') != driver_env(consumer) or cfg.get('release_authorized') is not True:
        raise RuntimeError('exact native offline TP2 environment/authorization required')
    if any(cfg.get('source_sha256', {}).get(p) != h for p, h in contract['bindings'].items()):
        raise RuntimeError('complete source/config SHA freeze missing')
    for p, h in cfg['source_sha256'].items():
        if sha(p) != h:
            raise RuntimeError('launch source SHA changed: ' + p)


def driver_env(consumer):
    env = dict(consumer['env'])
    for key in ('PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX'):
        env.pop(key, None)
    env.update(PYTHONPATH='', PYTHONHOME='', CONDA_PREFIX='',
               CUDA_VISIBLE_DEVICES='0,1', COARSE_EMPTY_N5_CONFIG=str(LAUNCH),
               MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages')
    return env


def prior_safe(base):
    receipts = []
    for gpu, (path, digest) in enumerate(PREVIOUS):
        old = checked({'path': str(path), 'sha256': digest})
        queue = BAR / 'queues' / ('curated_collaborative21000_n5g%d_v2' % gpu)
        if old['node'] != 'N5' or old['gpus'] != [gpu] or old['queue'] != str(queue) or old['resource_manager']['sha256'] != OLD_MANAGER_SHA:
            raise RuntimeError('exact old ownedmanager physical queue/config required')
        if sha(old['resource_manager']['path']) != OLD_MANAGER_SHA:
            raise RuntimeError('original owned manager source changed')
        admission_path, safe_path = queue / 'admission.json', queue / 'resources_safe.json'
        admission, safe = load(admission_path), load(safe_path)
        if admission.get('config_sha256') != digest or admission.get('manager_sha256') != BASE_SHA or admission.get('node') != 'N5' or admission.get('gpus') != [gpu] or admission.get('producer') != old['producer'] or admission.get('plan') != old['plan'] or admission.get('source_sha256') != old['source_sha256'] or admission.get('out') != old['out']:
            raise RuntimeError('old actual ownedmanager admission binding changed')
        identity = admission['manager_identity']
        expected_argv = ['/opt/venv_eval/bin/python', '-B', '-u', old['resource_manager']['path'], '--config', str(path), '--config-sha', digest, '--manager-sha', BASE_SHA]
        if identity['uid'] != os.getuid() or identity['argv'] != expected_argv:
            raise RuntimeError('old actual manager identity/argv changed')
        if safe.get('phase') != old['phase'] or any(safe.get(k) is not True for k in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')) or safe_path.stat().st_mtime_ns < admission_path.stat().st_mtime_ns:
            raise RuntimeError('original resources not actually safely restored')
        restored_path = queue / ('holder_restored_gpu%d.json' % gpu)
        restored = load(restored_path)
        if restored.get('gpu') != gpu or restored.get('restore_argv') != base.RESTORE_ARGV or not admission_path.stat().st_mtime_ns <= restored_path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns:
            raise RuntimeError('exact restored project holder evidence missing')
        if restored['uuid'] != admission['holders'][0]['uuid'] or restored['original_identity'] != admission['holders'][0]['identity']:
            raise RuntimeError('restoration UUID/original identity mismatch')
        history = [ref(path), ref(admission_path), ref(safe_path), ref(restored_path)]
        for name in ('failure.json', 'worker_exit.json'):
            p = queue / name
            if p.exists():
                value = load(p)
                if name == 'worker_exit.json' and (value.get('producer') != old['producer'] or value.get('plan') != old['plan'] or value.get('source_unchanged') is not True or value.get('input_count') != 222939):
                    raise RuntimeError('old worker exit provenance mismatch')
                history.append(ref(p))
        receipts.append({'gpu': gpu, 'restored': restored, 'history_refs': history})
    return {'previous_pairs': receipts, 'prior_producer_completion_not_claimed': True}


def check_node():
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('must execute on physical N5 locally')


def fresh_pair(base, evidence):
    check_node()
    locks, holders, leases = [], [], {}
    try:
        for gpu, row in zip(PAIR, evidence['previous_pairs']):
            lock = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
            fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size != 0:
                raise RuntimeError('unsafe physical pair lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reserve = Path('/tmp/od_reserve_%d' % gpu)
            leases[str(reserve)] = lease_value(base.lease_snapshot(reserve))
            restored = row['restored']
            expected = dict(restored['holder_identity'], gpu=gpu, uuid=restored['uuid'])
            uuid = base.uuid_for(gpu)
            running = base.apps()
            current = base.identity(expected['pid'])
            base.check_holder(expected, current, gpu, uuid, running)
            if current['argv'] != base.RESTORE_ARGV:
                raise RuntimeError('only normal TERM restored local project holder allowed')
            pidfd = os.pidfd_open(expected['pid'])
            try:
                if select.select([pidfd], [], [], 0)[0]:
                    raise RuntimeError('restored holder exited during CPU capture')
            finally:
                os.close(pidfd)
            holders.append(dict(current, gpu=gpu, uuid=uuid))
        budget = base.headroom()
        if budget['headroom'] < 160 * 2**30:
            raise RuntimeError('under 160GiB headroom; no holder released')
        for gpu, owner in zip(PAIR, holders):
            base.check_holder(owner, base.identity(owner['pid']), gpu, base.uuid_for(gpu), base.apps())
            if lease_value(base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != leases['/tmp/od_reserve_%d' % gpu]:
                raise RuntimeError('lease changed during fresh pair check')
        return {'holders': holders, 'lease_snapshot_ns': leases, 'budget': budget, 'captured_unix_ns': time.time_ns(), 'signals_sent': False}
    finally:
        for fd in locks:
            os.close(fd)


def verify_local_model(cfg, worker):
    base_module().check_model(cfg)
    identity = worker.bar.model_reference(worker.worker_args())
    if identity['sha256'] != cfg['modelref']['model_identity_sha256'] or identity['root'] != '/opt/models/Qwen3-VL-32B-Instruct':
        raise RuntimeError('original verified local checkpoint content identity changed')
    if cfg.get('local_model_identity') is not None and identity != cfg['local_model_identity']:
        raise RuntimeError('local checkpoint inventory/stat/content reference changed')
    return identity


def build():
    if LAUNCH.exists() or QUEUE.exists():
        raise RuntimeError('finite edition launch/queue already exists; no redispatch')
    worker = worker_module()
    contract = frozen(worker, prepare=True, probe=True)
    require_unreserved(contract['jobs'])
    consumer = contract['consumer']
    cfg = {'node': 'N5', 'gpus': PAIR, 'phase': PHASE, 'targetcount': 3, 'input_count': 222939,
           'global_revision': REVISION, 'release_authorized': True, 'queue': str(QUEUE), 'out': str(OUT),
           'reuse_existing_results': True, 'cwd': str(RT), 'env': driver_env(consumer),
           'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(DRIVER), '--launch', str(LAUNCH)],
           'producer': ref(DRIVER), 'plan': consumer['plan'], 'modelref': consumer['modelref'],
           'consumer_config_ref': ref(CONSUMER), 'jobs_manifest_ref': contract['manifest_ref'],
           'protocol_ref': contract['protocol_ref'], 'cpu_evidence_ref': contract['evidence_ref'],
           'source_sha256': dict(contract['bindings']), 'semantic_approval': False, 'training_eligible': False,
           'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0}
    cfg['local_model_identity'] = verify_local_model(cfg, worker)
    base = base_module()
    previous = prior_safe(base)
    previous_path = RT / 'curated_coarse_empty_n5_prior_resources_safe_v1.json'
    capture_path = RT / 'curated_coarse_empty_n5_holder_capture_v1.json'
    if previous_path.exists() or capture_path.exists():
        raise RuntimeError('immutable prior builder proof already exists; no overwrite')
    # Hash/source/model checks precede fresh resource observation, never vice versa.
    for name, digest in cfg['source_sha256'].items():
        if sha(name) != digest:
            raise RuntimeError('source changed before fresh capture')
    fresh = fresh_pair(base, previous)
    save_once(previous_path, previous)
    save_once(capture_path, fresh)
    cfg.update(holders=fresh['holders'], lease_snapshot_ns=fresh['lease_snapshot_ns'],
               previous_resources_ref=ref(previous_path), holder_capture=ref(capture_path))
    cfg['source_sha256'].update(references(previous))
    cfg['source_sha256'].update({str(previous_path): sha(previous_path), str(capture_path): sha(capture_path)})
    save_once(LAUNCH, cfg)
    return {'launch_ref': ref(LAUNCH), 'wrapper_ref': ref(MANAGER), 'base_manager_sha256': BASE_SHA,
            'actual_GPU_model_calls': 0, 'signals_sent': False, 'fixed_sources': TARGETS,
            'command': ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER), '--config', str(LAUNCH),
                        '--config-sha', sha(LAUNCH), '--manager-sha', BASE_SHA]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(build(), ensure_ascii=False))


if __name__ == '__main__':
    main()
