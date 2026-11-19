"""Exact N6G01 fixed3 handoff; CPU builders never signal, reserve or load a model."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
SELF = Path(__file__).resolve()
TEST = RT / 'test_curated_coarse_fixed3_managed_primary_v1.py'
OUT = BAR / 'curated_coarse_reobserve_fixed3_v1'
PRIMARY = BAR / 'curated222939_repair_v1'
QUEUE = BAR / 'queues/curated_coarse_fixed3_n6g01_v1'
LAUNCH = RT / 'curated_coarse_fixed3_n6g01_launch_v1.json'
PREFLIGHT = RT / 'curated_coarse_fixed3_primary_preflight_v1.json'
RESTORE_QUEUE = BAR / 'queues/curated_coarse_fixed3_primary_restore_v1'
RESTORE_LAUNCH = RT / 'curated_coarse_fixed3_primary_restore_launch_v1.json'
OLD = RT / 'curated_repair_resident_n6_g01_v2_final.json'
OLD_SHA = '55331847a0e6319b04653dcfe0e524359ab66e849b6ca336e5416333e05e96a1'
REVISION = 'coarse-reobserve-fixed3-20261025-v1'
PHASE = 'curated_coarse_fixed3_n6g01_v1'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
WORKER = RT / 'curated_coarse_reobserve_fixed3_v1.py'
WORKER_SHA = '45f9b6917cc4c53c6cfb40e47456352fa8869b797fbdba6a7790cd59103ea3d5'
WORKER_TEST_SHA = 'c104349c42e75ef250ec8deb5bfab2345b085ccdf337a2c9e75c2198ea074f00'
DEPS = {
    'prepare_curated_coarse_empty_n5_v1.py': '1190df9feed04400391c9d4b7b0df11e9bb40b2ef642754d3dcaaffcdb733319',
    'curated_coarse_empty_n5_driver_v1.py': '2aa225414bf5f61082c65a958b968bcf365e77516342e943fbba54a52595970c',
    'curated_repair_resource_manager_v2.py': '8935d379b22c9fcffb8a331a7b9df52ef467eff7cfa1b086e06d54370f308b16',
    'resource_manager.py': BASE_SHA}
PAIR = [0, 1]


def private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


for name, digest in DEPS.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('original frozen helper changed: ' + name)
c = private_module('fixed3_private_prepare', RT / 'prepare_curated_coarse_empty_n5_v1.py')
d = private_module('fixed3_private_driver', RT / 'curated_coarse_empty_n5_driver_v1.py')
d.c = c
_cached_sha = c.sha


def content_sha(path):
    path = Path(path)
    resolved = path.resolve()
    private_current = (resolved in (SELF, TEST) or resolved.is_relative_to(OUT)
                       or resolved.parent == RT and resolved.name.startswith('curated_coarse_fixed3_'))
    if not private_current:
        # Original external media/source cache: MISS full SHA with four stable
        # observations; HIT requires exact path/dev/inode/size/mtime/ctime key.
        # Same-inode rewrites preserving that complete stat remain its known boundary.
        return _cached_sha(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if any(getattr(before, k) != getattr(s, k) for s in (opened, finished, after) for k in c.LEASE_KEYS):
        raise RuntimeError('source changed during actual full content SHA')
    return digest.hexdigest()


c.sha = content_sha
c.OUT, c.PRIMARY, c.QUEUE, c.LAUNCH = OUT, PRIMARY, QUEUE, LAUNCH
c.MANAGER = c.DRIVER = SELF
c.REVISION = REVISION


def check_node():
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('only physical N6, never another node')


def worker_module():
    if c.sha(WORKER) != WORKER_SHA or c.sha(RT / 'test_curated_coarse_reobserve_fixed3_v1.py') != WORKER_TEST_SHA:
        raise RuntimeError('fixed3 worker/tests are not frozen originals')
    return private_module('fixed3_frozen_worker', WORKER)


def frozen(worker, probe=True):
    if c.sha(WORKER) != WORKER_SHA or c.sha(RT / 'test_curated_coarse_reobserve_fixed3_v1.py') != WORKER_TEST_SHA:
        raise RuntimeError('fixed3 worker/tests changed')
    worker.verify_native_sources()
    manifest, refs = worker.scope_manifest()
    protocol_ref = c.ref(OUT / 'protocol.json')
    protocol = c.checked(protocol_ref)
    evidence_ref = c.ref(OUT / 'cpu.prepare.evidence.json')
    evidence = c.checked(evidence_ref)
    freeze_ref = c.ref(OUT / 'source.freeze.json')
    if protocol != worker.protocol() or c.checked(freeze_ref) != worker.source_freeze():
        raise RuntimeError('complete fixed3 protocol/source freeze changed')
    expected = {'source_count': 3, 'model_loads': 0, 'visual_model_calls': 0,
                'text_model_calls': 0, 'budget_reservations_performed': 0,
                'budget_beforeafter_unchanged': True, 'global_revision': REVISION}
    if any(evidence.get(k) != v for k, v in expected.items()) or [x['actual_image_count'] for x in evidence['checks']] != [13, 10, 14]:
        raise RuntimeError('fixed3 CPU evidence/image population changed')
    jobs = [c.checked(ref) for ref in refs]
    bindings = {str(RT / n): h for n, h in DEPS.items()}
    bindings.update(worker.bar.code_identity())
    bindings[str(worker.pipeline.__file__)] = c.sha(worker.pipeline.__file__)
    original = c.checked({'path': str(OLD), 'sha256': OLD_SHA})
    bindings.update(original['source_sha256'])
    bindings.update({str(WORKER): WORKER_SHA, str(RT / 'test_curated_coarse_reobserve_fixed3_v1.py'): WORKER_TEST_SHA,
                     str(SELF): c.sha(SELF), str(TEST): c.sha(TEST), str(OLD): OLD_SHA})
    for value in (protocol_ref, protocol, c.ref(OUT / 'jobs.manifest.json'), manifest,
                  evidence_ref, evidence, freeze_ref, jobs):
        bindings.update(c.references(value))
    for ref, job, check in zip(refs, jobs, evidence['checks']):
        original, joint, prepared, plan = worker.verify_job(job, probe=probe)
        request = worker.make_request(job, ref, prepared, worker.worker_args())
        if (check['id'] != job['id'] or check['target_window_id'] != job['target_window_id']
                or c.checked(check['request_preview_ref']) != request
                or check['original_job_ref'] != job['refs']['original_canonical_job']
                or check['target_manifest_ref'] != job['target_manifest_ref']
                or check['original_non_target_observations_digest'] != worker.q.digest([o for o in original['window_observations'] if o['window']['window_id'] != job['target_window_id']])
                or check['original_facts_digest'] != worker.q.digest(original['facts'])
                or check['original_style_digest'] != worker.q.digest(joint['style_result'])):
            raise RuntimeError('original complete transaction/preview/style evidence changed')
        bindings.update(c.references(plan['source_refs']))
    for entry in worker.q.ORIGINAL_CSV_CACHE.values():
        p = Path(entry['path'])
        st = p.stat()
        if tuple(getattr(st, k) for k in worker.q.CSV_STAT_KEYS) != entry['stat']:
            raise RuntimeError('original CSV changed after full content scan')
        key = (str(p.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        c._hash_cache[key] = entry['sha256']
    for name, digest in bindings.items():
        if not Path(name).is_absolute() or c.sha(name) != digest:
            raise RuntimeError('frozen source/input changed: ' + name)
    return {'consumer': c.checked(c.ref(c.CONSUMER)), 'manifest': manifest,
            'manifest_ref': c.ref(OUT / 'jobs.manifest.json'), 'protocol_ref': protocol_ref,
            'evidence_ref': evidence_ref, 'freeze_ref': freeze_ref, 'jobs': jobs, 'bindings': bindings}


def original_config():
    cfg = c.checked({'path': str(OLD), 'sha256': OLD_SHA})
    if (cfg['node'] != 'N6' or cfg['gpus'] != PAIR or cfg['out'] != str(PRIMARY)
            or cfg['global_revision'] != 'repair-20261025-v1' or cfg['targetcount'] != 222939):
        raise RuntimeError('original primary identity changed')
    for name, digest in cfg['source_sha256'].items():
        if c.sha(name) != digest:
            raise RuntimeError('original primary frozen source changed')
    return cfg


def manager_command(path):
    return ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config', str(path),
            '--config-sha', c.sha(path), '--manager-sha', BASE_SHA]


def primary_admission(old):
    reference = c.ref(Path(old['queue']) / 'admission.json')
    admission = c.checked(reference)
    expected = {'config_sha256': OLD_SHA, 'manager_sha256': BASE_SHA, 'node': 'N6', 'gpus': PAIR,
                'out': old['out'], 'producer': old['producer'], 'plan': old['plan'],
                'source_sha256': old['source_sha256'], 'model_identity_sha256': old['modelref']['model_identity_sha256']}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise RuntimeError('original actual primary admission changed')
    return reference, admission


def identity_equal(base, expected, current):
    if current['state'] == 'Z' or any(current[k] != expected[k] for k in base.IDENTITY_KEYS):
        raise RuntimeError('actual PID UID/start/cgroup/exe/CVD/argv changed')


def owned_pair(base, pid, leases, uuids, heartbeat=True):
    if len(leases) != 2 or [v['gpu'] for v in leases] != PAIR or len(set(uuids)) != 2:
        raise RuntimeError('two distinct ordered physical owned leases required')
    for value, uuid in zip(leases, uuids):
        gpu = value['gpu']
        path = Path('/tmp/od_reserve_%d' % gpu)
        now = path.lstat()
        old = value['stat']
        c.verify_lease_value(old)
        if (value['path'] != str(path) or value['uuid'] != uuid or base.uuid_for(gpu) != uuid
                or not stat.S_ISREG(now.st_mode) or now.st_uid != os.getuid() or now.st_size != 0
                or (now.st_dev, now.st_ino) != (old['st_dev'], old['st_ino'])
                or heartbeat and not 0 <= time.time_ns() - now.st_mtime_ns <= 60 * 10**9):
            raise RuntimeError('actual owned lease inode/UUID/live heartbeat changed')
        lockpath = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
        lock = lockpath.lstat()
        if (not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.getuid() or lock.st_size != 0
                or not c.fd_inode_owned(pid, now.st_dev, now.st_ino)
                or not c.fd_inode_owned(pid, lock.st_dev, lock.st_ino)
                or not c.proc_lock_owned(pid, str(lockpath))):
            raise RuntimeError('actual parent lease FD/inode/kernel GPU FLOCK not owned')


def preflight():
    check_node()
    if PREFLIGHT.exists():
        raise RuntimeError('original preflight immutable; never extend TTL or re-sign')
    worker, base = worker_module(), c.base_module()
    contract = frozen(worker)
    c.require_unreserved(contract['jobs'])
    old = original_config()
    model = c.verify_local_model(old, worker)
    admission_ref, admission = primary_admission(old)
    live_ref = c.ref(Path(old['queue']) / 'live.json')
    live = c.checked(live_ref)
    manager = base.identity(admission['manager_identity']['pid'])
    identity_equal(base, admission['manager_identity'], manager)
    expected_argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(RT / 'curated_repair_resource_manager_v2.py'),
                     '--config', str(OLD), '--config-sha', OLD_SHA, '--manager-sha', BASE_SHA]
    if manager['argv'] != expected_argv or manager['uid'] != os.getuid() or manager['cgroup'] != base.identity(os.getpid())['cgroup']:
        raise RuntimeError('original exact local managed parent required')
    child = base.identity(live['child_identity']['pid'])
    identity_equal(base, live['child_identity'], child)
    if child['ppid'] != manager['pid'] or child['argv'] != old['argv'] or child['CVD'] != ['CUDA_VISIBLE_DEVICES=0,1']:
        raise RuntimeError('original primary worker direct tree/argv/CVD changed')
    rawenv = dict(v.decode().split('=', 1) for v in (Path('/proc') / str(child['pid']) / 'environ').read_bytes().split(b'\0') if v)
    if any(rawenv.get(k) != v for k, v in old['env'].items()) or os.readlink(Path('/proc') / str(child['pid']) / 'cwd') != old['cwd']:
        raise RuntimeError('actual original configured worker env/cwd changed')
    tracker = d.ModelDescendants(base)
    tracker.own, tracker.known = child, {child['pid']: child}
    uuids = [e['uuid'] for e in admission['holders']]
    leases = [{'gpu': gpu, 'uuid': uuid, 'path': '/tmp/od_reserve_%d' % gpu,
               'stat': c.lease_value(Path('/tmp/od_reserve_%d' % gpu).lstat())} for gpu, uuid in zip(PAIR, uuids)]
    owned_pair(base, manager['pid'], leases, uuids)
    for app in base.apps():
        if app['uuid'] in uuids:
            tracker.verify(app['pid'])
    budget = base.headroom()
    if budget['headroom'] < 160 * 2**30:
        raise RuntimeError('fresh headroom below 160GiB; no primary TERM authorized')
    c.require_unreserved(contract['jobs'])
    identity_equal(base, manager, base.identity(manager['pid']))
    owned_pair(base, manager['pid'], leases, uuids)
    proof = {'schema': 'fixed3_primary_cpu_preflight_v1', 'original_config_ref': c.ref(OLD),
             'original_admission_ref': admission_ref, 'original_live_ref': live_ref,
             'original_manager_identity': manager, 'original_child_identity': child,
             'jobs_manifest_ref': contract['manifest_ref'], 'source_freeze_ref': contract['freeze_ref'],
             'local_model_identity': model, 'budget': budget, 'leases': leases,
             'source_sha256': contract['bindings'], 'captured_unix_ns': time.time_ns(),
             'signals_sent': False, 'model_loads': 0, 'budget_reservations': 0}
    c.save_once(PREFLIGHT, proof)
    return {'preflight_ref': c.ref(PREFLIGHT), 'original_manager_pid': manager['pid'],
            'normal_TERM_only_by_main': True, 'GPU_dispatch_performed': False}


def preflight_binding(contract):
    proof = c.checked(c.ref(PREFLIGHT))
    if (proof.get('schema') != 'fixed3_primary_cpu_preflight_v1'
            or proof['original_config_ref'] != {'path': str(OLD), 'sha256': OLD_SHA}
            or proof['jobs_manifest_ref'] != contract['manifest_ref'] or proof['source_freeze_ref'] != contract['freeze_ref']
            or proof['source_sha256'] != contract['bindings'] or proof['signals_sent'] is not False
            or not 0 <= time.time_ns() - proof['captured_unix_ns'] <= 1200 * 10**9):
        raise RuntimeError('original CPU preflight expired/changed; never extend TTL')
    return proof


def restored_evidence(base, queue, admission_ref):
    admission = c.checked(admission_ref)
    safe_path = Path(queue) / 'resources_safe.json'
    safe = c.load(safe_path)
    if any(safe.get(k) is not True for k in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')) or safe_path.stat().st_mtime_ns < Path(admission_ref['path']).stat().st_mtime_ns:
        raise RuntimeError('actual original managed resources_safe required, not semantic completion')
    rows = []
    for gpu, holder in zip(PAIR, admission['holders']):
        path = Path(queue) / ('holder_restored_gpu%d.json' % gpu)
        value = c.load(path)
        if (value['gpu'] != gpu or value['uuid'] != holder['uuid'] or value['original_identity'] != holder['identity']
                or value['restore_argv'] != base.RESTORE_ARGV
                or not Path(admission_ref['path']).stat().st_mtime_ns <= path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns):
            raise RuntimeError('exact original normal-TERM restored holder receipt required')
        rows.append({'gpu': gpu, 'restored': value, 'history_refs': [c.ref(path)]})
    return {'previous_pairs': rows, 'resource_admission_ref': admission_ref,
            'resources_safe_ref': c.ref(safe_path), 'original_completion_not_claimed': True}


def driver_env(consumer):
    env = dict(consumer['env'])
    env.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONPATH='', PYTHONHOME='', CONDA_PREFIX='',
               FIXED3_N6_CONFIG=str(LAUNCH), MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages')
    return env


def verify_config(cfg, contract):
    expected = {'node': 'N6', 'gpus': PAIR, 'targetcount': 3, 'input_count': 222939,
                'phase': PHASE, 'global_revision': REVISION, 'queue': str(QUEUE), 'out': str(OUT),
                'reuse_existing_results': True, 'release_authorized': True, 'cwd': str(RT),
                'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(SELF), '--driver', '--launch', str(LAUNCH)],
                'env': driver_env(contract['consumer']), 'producer': c.ref(SELF),
                'modelref': contract['consumer']['modelref'], 'plan': contract['consumer']['plan'],
                'jobs_manifest_ref': contract['manifest_ref'], 'protocol_ref': contract['protocol_ref'],
                'cpu_evidence_ref': contract['evidence_ref'], 'source_freeze_ref': contract['freeze_ref'],
                'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise RuntimeError('exact fixed3 config/model/budget/argv/env binding changed')
    if any(cfg.get('source_sha256', {}).get(p) != h for p, h in contract['bindings'].items()):
        raise RuntimeError('complete current frozen source bindings missing')
    for name, digest in cfg['source_sha256'].items():
        if c.sha(name) != digest:
            raise RuntimeError('launch source SHA changed')


def build(restore=False):
    check_node()
    launch, queue = (RESTORE_LAUNCH, RESTORE_QUEUE) if restore else (LAUNCH, QUEUE)
    if launch.exists() or queue.exists():
        raise RuntimeError('immutable launch/queue already exists; no redispatch')
    worker, base = worker_module(), c.base_module()
    contract = frozen(worker)
    old = original_config()
    if restore:
        previous = restored_evidence(base, QUEUE, c.ref(QUEUE / 'admission.json'))
        cfg = copy.deepcopy(old)
        cfg.update(queue=str(queue), fixed3_primary_restore=True, original_primary_config_ref=c.ref(OLD),
                   fixed3_launch_ref=c.ref(LAUNCH), previous_resources_ref=None)
        model = c.verify_local_model(cfg, worker)
    else:
        proof = preflight_binding(contract)
        previous = restored_evidence(base, Path(old['queue']), proof['original_admission_ref'])
        c.require_unreserved(contract['jobs'])
        cfg = {'node': 'N6', 'gpus': PAIR, 'phase': PHASE, 'global_revision': REVISION,
               'targetcount': 3, 'input_count': 222939, 'queue': str(queue), 'out': str(OUT),
               'reuse_existing_results': True, 'release_authorized': True, 'cwd': str(RT),
               'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(SELF), '--driver', '--launch', str(LAUNCH)],
               'env': driver_env(contract['consumer']), 'producer': c.ref(SELF),
               'plan': contract['consumer']['plan'], 'modelref': contract['consumer']['modelref'],
               'jobs_manifest_ref': contract['manifest_ref'], 'protocol_ref': contract['protocol_ref'],
               'cpu_evidence_ref': contract['evidence_ref'], 'source_freeze_ref': contract['freeze_ref'],
               'source_sha256': dict(contract['bindings']), 'primary_preflight_ref': c.ref(PREFLIGHT),
               'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0}
        model = c.verify_local_model(cfg, worker)
    for name, digest in cfg['source_sha256'].items():
        if c.sha(name) != digest:
            raise RuntimeError('source changed before fresh capture')
    # All slow source/model work above; only current resource reads below.
    fresh = c.fresh_pair(base, previous)
    if not restore:
        preflight_binding(contract)
        c.require_unreserved(contract['jobs'])
    capture_path = RT / ('curated_coarse_fixed3_primary_restore_capture_v1.json' if restore else 'curated_coarse_fixed3_n6_capture_v1.json')
    prior_path = RT / ('curated_coarse_fixed3_primary_restore_prior_v1.json' if restore else 'curated_coarse_fixed3_n6_prior_v1.json')
    if capture_path.exists() or prior_path.exists():
        raise RuntimeError('fresh capture/prior evidence immutable; no overwrite')
    c.save_once(prior_path, previous)
    c.save_once(capture_path, fresh)
    cfg.update(holders=fresh['holders'], lease_snapshot_ns=fresh['lease_snapshot_ns'],
               holder_capture=c.ref(capture_path), previous_resources_ref=c.ref(prior_path), local_model_identity=model)
    cfg['source_sha256'].update(c.references(previous))
    cfg['source_sha256'].update({str(prior_path): c.sha(prior_path), str(capture_path): c.sha(capture_path), str(SELF): c.sha(SELF), str(TEST): c.sha(TEST)})
    if restore:
        cfg['source_sha256'].update({str(LAUNCH): c.sha(LAUNCH), str(OLD): OLD_SHA})
    else:
        cfg['source_sha256'][str(PREFLIGHT)] = c.sha(PREFLIGHT)
    c.save_once(launch, cfg)
    return {'launch_ref': c.ref(launch), 'manager_ref': c.ref(SELF), 'command': manager_command(launch),
            'signals_sent': False, 'model_loads': 0, 'GPU_dispatch_performed': False,
            'primary_worker_argv_env_out_revision_preserved': restore}


def verify_actual_parent(cfg, base, contract, tracker=None, allow_compute=False):
    check_node()
    verify_config(cfg, contract)
    resource_ref = c.ref(QUEUE / 'admission.json')
    admission = c.checked(resource_ref)
    proof_ref = c.ref(QUEUE / 'actual_leases_held.json')
    proof = c.checked(proof_ref)
    launch_ref = c.ref(LAUNCH)
    expected = {'config_sha256': launch_ref['sha256'], 'manager_sha256': BASE_SHA,
                'source_sha256': cfg['source_sha256'], 'producer': cfg['producer'], 'plan': cfg['plan'],
                'model_identity_sha256': cfg['modelref']['model_identity_sha256'], 'node': 'N6',
                'gpus': PAIR, 'out': str(OUT), 'targetcount': 3, 'phase': PHASE}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise RuntimeError('actual current ownedparent admission changed')
    if (proof.get('schema') != 'fixed3_actual_leases_held_v1' or proof['resource_admission_ref'] != resource_ref
            or proof['config_ref'] != launch_ref or proof['node'] != 'N6' or proof['gpus'] != PAIR):
        raise RuntimeError('actual held parent proof/config/admission changed')
    pid = os.getppid()
    parent, own = base.identity(pid), base.identity(os.getpid())
    identity_equal(base, admission['manager_identity'], parent)
    identity_equal(base, proof['manager_identity'], parent)
    if (parent['argv'] != manager_command(LAUNCH) or parent['uid'] != os.getuid()
            or parent['cgroup'] != own['cgroup'] or parent['exe'] != own['exe']
            or own['ppid'] != pid or own['argv'] != cfg['argv']
            or own['CVD'] != ['CUDA_VISIBLE_DEVICES=0,1'] or own['start_ticks'] < parent['start_ticks']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or os.environ.get('FIXED3_N6_CONFIG') != str(LAUNCH)):
        raise RuntimeError('exact direct managed caller PIDtree/UID/start/cgroup/exe/CVD required')
    if own['state'] == 'Z':
        raise RuntimeError('caller is not live')
    if any(os.environ.get(k) != v for k, v in cfg['env'].items()):
        raise RuntimeError('actual current configured native worker environment changed')
    uuids = [h['uuid'] for h in admission['holders']]
    if [h['gpu'] for h in admission['holders']] != PAIR:
        raise RuntimeError('actual admission GPU order changed')
    owned_pair(base, pid, proof['leases'], uuids)
    budget = base.headroom()
    if budget['headroom'] < (80 if allow_compute else 160) * 2**30:
        raise RuntimeError('current headroom below live model safety/load requirement')
    for app in base.apps():
        if app['uuid'] not in uuids:
            continue
        if not allow_compute or tracker is None:
            raise RuntimeError('unexpected GPU app before load; no app signaled')
        tracker.verify(app['pid'])
    return {'jobs_manifest_ref': contract['manifest_ref'], 'source_job_refs': [h['job_ref'] for h in contract['manifest']['jobs']],
            'original_source_job_refs': [j['refs']['original_canonical_job'] for j in contract['jobs']],
            'out': str(OUT), 'global_revision': REVISION, 'caller_pid': os.getpid(),
            'tensor_parallel_size': 2, 'gpus': PAIR, 'model_identity_sha256': cfg['modelref']['model_identity_sha256'],
            'resource_admission_ref': resource_ref, 'current_lease_refs': [c.ref(v['path']) for v in proof['leases']],
            'actual_held_leases_ref': proof_ref, 'launch_config_ref': launch_ref, 'budget': budget,
            'verified_unix_ns': time.time_ns()}


def verify_completed_transactions(worker, contract):
    for ref, job in zip([h['job_ref'] for h in contract['manifest']['jobs']], contract['jobs']):
        root = OUT / 'revisions' / REVISION / job['source_key']
        attempt = root / 'windows' / job['target_window_id'] / 'attempt-000001'
        if not (attempt / 'request.json').exists():
            continue
        original, joint, prepared, plan = worker.verify_job(job, probe=True)
        request = worker.make_request(job, ref, prepared, worker.worker_args())
        manifest = worker.verify_window(job, prepared)
        preview_ref = c.ref(OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
        binding = {**original['binding'], 'repair_revision': REVISION, 'current_job_ref': ref,
                   'original_canonical_job_ref': job['refs']['original_canonical_job'], 'ledger_bindings': job['ledger_bindings'],
                   'window': manifest['window'], 'frames_sha256': worker.bar.batch.digest(manifest['frames']),
                   'request_preview_ref': preview_ref}
        if c.load(attempt / 'request.json') != request or c.load(attempt / 'binding.json') != binding:
            raise RuntimeError('actual original request/binding changed')
        if (attempt / 'receipt.json').exists():
            final = c.load(attempt / 'final.json')
            worker.transaction_proof({'root': attempt, 'request': request, 'binding': binding}, final,
                                     contract['consumer']['modelref']['model_identity_sha256'])
        if (root / 'result.json').exists():
            result = c.load(root / 'result.json')
            visual = c.checked(result['visual_result_ref'])
            if (result['current_job_ref'] != ref or result['original_job_ref'] != job['refs']['original_canonical_job']
                    or result['global_revision'] != REVISION or result['new_text_model_requests'] != 0
                    or result['new_visual_requests'] != 1 or result['style_result'] != joint['style_result']
                    or result['dance_style_label'] != joint['dance_style_label']
                    or c.checked(result['primary_budget_ref']) != c.load(root / 'canonical_budget_claim.json')
                    or [o for o in visual['window_observations'] if o['window']['window_id'] != job['target_window_id']]
                       != [o for o in original['window_observations'] if o['window']['window_id'] != job['target_window_id']]):
                raise RuntimeError('current result/actual original canonical transaction changed')
            c.checked(result['transaction_receipt_ref'])


class FiniteRuntime(d.FiniteRuntime):
    def __init__(self, cfg, base, worker, contract):
        super().__init__(cfg, base, worker, contract)
        self.claims = {}

    def verify_current_admission(self, refs):
        if refs != [h['job_ref'] for h in self.contract['manifest']['jobs']] or len(refs) != 3:
            raise RuntimeError('complete current fixed3/original refs required')
        current = frozen(self.worker)
        if current != self.contract:
            raise RuntimeError('complete fixed3/original source transaction changed')
        c.verify_local_model(self.cfg, self.worker)
        if self.loaded is not None and self.loaded[0]._ms_bar_model_sha256 != self.cfg['modelref']['model_identity_sha256']:
            raise RuntimeError('actual loaded model identity changed')
        for job in current['jobs']:
            path = self.worker.budget_path(job)
            if path.exists():
                expected = {'source_key': job['source_key'], 'original_result_sha256': job['original_result_sha256'],
                            'global_revision': REVISION, 'extra_source_attempt': 1, 'maximum_extra_source_attempts': 1}
                value = c.checked(c.ref(path))
                if value != expected or str(path) in self.claims and self.claims[str(path)] != c.ref(path):
                    raise RuntimeError('original canonical attempt transaction changed')
                self.claims[str(path)] = c.ref(path)
            elif str(path) in self.claims:
                raise RuntimeError('canonical attempt disappeared')
        verify_completed_transactions(self.worker, current)
        return verify_actual_parent(self.cfg, self.base, current, self.tracker, self.model_started)

    def __call__(self):
        refs = [h['job_ref'] for h in self.contract['manifest']['jobs']]
        self.verify_current_admission(refs)
        if self.loaded is None:
            if self.model_started:
                raise RuntimeError('ambiguous shared load failed; never retry')
            c.require_unreserved(self.contract['jobs'])
            c.verify_local_model(self.cfg, self.worker)
            self.verify_current_admission(refs)
            root = OUT / 'gpu-runtime' / REVISION
            root.mkdir(parents=True, exist_ok=False)
            self.loader = self.worker.bar.runtime_loader(self.args, root, self.worker.bar.code_identity())
            self.model_started = True
            model, sampling_cls = self.loader()
            if model._ms_bar_model_sha256 != self.cfg['modelref']['model_identity_sha256']:
                raise RuntimeError('actual loaded native model content identity changed')
            self.loaded = (FiniteModel(model, self), sampling_cls)
            self.verify_current_admission(refs)
        return self.loaded


class FiniteModel:
    def __init__(self, model, runtime):
        self.model, self.runtime = model, runtime
        self._ms_bar_model_sha256 = model._ms_bar_model_sha256

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def chat(self, messages, sampling, use_tqdm=False):
        r = self.runtime
        refs = [h['job_ref'] for h in r.contract['manifest']['jobs']]
        r.verify_current_admission(refs)
        if len(messages) != 1 or len(sampling) != 1 or r.visual_calls >= 3:
            raise RuntimeError('only one visual request, three total, zero text calls')
        matches = [(ref, job) for ref, job in zip(refs, r.contract['jobs'])
                   if messages == [c.load(OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')['messages']]]
        if len(matches) != 1 or matches[0][0]['path'] in r.seen_jobs:
            raise RuntimeError('only exact new frozen visual preview, no duplicate source call')
        ref, job = matches[0]
        expected = 13 if job['id'].startswith('mrDP') else 10 if job['id'].startswith('Ecd') else 14
        if sum(v['type'] == 'image_url' for v in messages[0][1]['content']) != expected:
            raise RuntimeError('fixed native 13/10/14 RGB/ROI population changed')
        preview = c.load(OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
        native_sampling = r.worker.bar.core.structured_sampling(r.loaded[1], preview['json_schema'], preview['sampling']['max_tokens'])
        if sampling[0] != native_sampling:
            raise RuntimeError('native sampling/schema/max_tokens changed')
        claim_ref = c.ref(r.worker.budget_path(job))
        if r.claims.get(claim_ref['path']) != claim_ref:
            raise RuntimeError('live original canonical attempt claim required')
        r.seen_jobs.add(ref['path'])
        r.visual_calls += 1
        return self.model.chat(messages, sampling, use_tqdm=use_tqdm)


def run_driver():
    if any(os.environ.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX')):
        raise RuntimeError('shared Python environment injection forbidden')
    cfg, worker, base = c.load(LAUNCH), worker_module(), c.base_module()
    contract = frozen(worker)
    verify_actual_parent(cfg, base, contract)
    c.require_unreserved(contract['jobs'])
    with (OUT / 'managed-fixed3-worker.lock').open('xb') as lock:
        c.fcntl.flock(lock, c.fcntl.LOCK_EX | c.fcntl.LOCK_NB)
        runtime = FiniteRuntime(cfg, base, worker, contract)
        refs = [h['job_ref'] for h in contract['manifest']['jobs']]
        c.save_once(QUEUE / 'actual_worker_admission.json', runtime.verify_current_admission(refs))
        try:
            summary = worker.execute_scope(runtime.args, runtime)
            c.save_once(OUT / 'finite.stage.summary.json', summary)
        finally:
            # Partial transactions are retained even when this final check rejects.
            frozen(worker)
            runtime.verify_current_admission(refs)


def verify_restore_config(cfg, old):
    excluded = {'holders', 'lease_snapshot_ns', 'queue', 'source_sha256'}
    if (any(cfg.get(k) != v for k, v in old.items() if k not in excluded)
            or cfg.get('queue') != str(RESTORE_QUEUE) or cfg.get('fixed3_primary_restore') is not True):
        raise RuntimeError('primary restore must preserve original worker argv/env/cwd/out/revision/old queue')
    if cfg['original_primary_config_ref'] != c.ref(OLD) or cfg['fixed3_launch_ref'] != c.ref(LAUNCH):
        raise RuntimeError('primary restore original transaction ref changed')
    for name, digest in old['source_sha256'].items():
        if cfg['source_sha256'].get(name) != digest:
            raise RuntimeError('primary original frozen source/review queue binding missing')


def run_manager():
    base = c.base_module()
    original_validate, original_snapshot = base.validate_pair, base.lease_snapshot
    original_refresh, original_release, original_acquire = base.refresh_leases, base.release_holder, base.acquire_lease
    active, entries, proof_written = {}, [], False
    repair = private_module('fixed3_private_primary_repair', RT / 'curated_repair_resource_manager_v2.py')

    def validate(cfg):
        check_node()
        expected_launch = RESTORE_LAUNCH if cfg.get('fixed3_primary_restore') is True else LAUNCH
        if sys.argv.count('--config') != 1 or sys.argv[sys.argv.index('--config') + 1] != str(expected_launch):
            raise RuntimeError('only exact immutable managed config path allowed')
        worker = worker_module()
        contract = frozen(worker)
        if cfg.get('fixed3_primary_restore') is True:
            old = original_config()
            verify_restore_config(cfg, old)
            repair.validate(dict(cfg, queue=old['queue']))
            previous = restored_evidence(base, QUEUE, c.ref(QUEUE / 'admission.json'))
        else:
            verify_config(cfg, contract)
            preflight_binding(contract)
            original_validate(cfg)
            c.require_unreserved(contract['jobs'])
            old = original_config()
            previous = restored_evidence(base, Path(old['queue']), c.checked(cfg['primary_preflight_ref'])['original_admission_ref'])
        c.verify_local_model(cfg, worker)
        if c.checked(cfg['previous_resources_ref']) != previous:
            raise RuntimeError('previous actual safe/restored resource evidence changed')
        capture = c.checked(cfg['holder_capture'])
        if (capture['holders'] != cfg['holders'] or capture['lease_snapshot_ns'] != cfg['lease_snapshot_ns']
                or capture['signals_sent'] is not False
                or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9):
            raise RuntimeError('fresh immutable holder/lease capture expired/changed')
        fresh = c.fresh_pair(base, previous)
        if fresh['lease_snapshot_ns'] != cfg['lease_snapshot_ns']:
            raise RuntimeError('exact full nanosecond old lease capture changed')
        for current, expected in zip(fresh['holders'], cfg['holders']):
            base.check_holder(expected, current, current['gpu'], current['uuid'], base.apps())
        active.update(cfg=cfg, contract=contract)

    def snapshot(path):
        value = original_snapshot(path)
        expected = active['cfg']['lease_snapshot_ns'].get(str(path), 'missing')
        if expected == 'missing':
            raise RuntimeError('exact two integer-nanosecond lease captures required')
        c.verify_lease_value(expected)
        if c.lease_value(value) != expected:
            raise RuntimeError('lease snapshot changed, no TTL renewal or re-sign')
        return value

    def acquire(e):
        original_acquire(e)
        entries.append(e)

    def release(e):
        cfg = active['cfg']
        if len(entries) != 2:
            raise RuntimeError('all pair owned lease descriptors required before first release')
        leases = [{'gpu': v['gpu'], 'uuid': v['uuid'], 'path': str(v['reserve']),
                   'stat': c.lease_value(os.fstat(v['leasefd']))} for v in entries]
        owned_pair(base, os.getpid(), leases, [v['uuid'] for v in entries])
        if base.headroom()['headroom'] < 160 * 2**30:
            raise RuntimeError('fresh headroom changed before normal holder TERM')
        running = base.apps()
        for v in entries:
            if not v['released']:
                base.check_holder(v['expected'], base.identity(v['expected']['pid']), v['gpu'], v['uuid'], running)
            elif any(a['uuid'] == v['uuid'] and a['pid'] != v['expected']['pid'] for a in running):
                raise RuntimeError('unknown app appeared on already-released owned pair; not signaled')
        if not cfg.get('fixed3_primary_restore'):
            preflight_binding(active['contract'])
            c.require_unreserved(active['contract']['jobs'])
        original_release(e)

    def refresh(values, on_error=None):
        nonlocal proof_written
        original_refresh(values, on_error)
        if (proof_written or not active or active['cfg'].get('fixed3_primary_restore') or len(values) != 2
                or not all(v['released'] and v['leasefd'] is not None for v in values)
                or not all(select.select([v['fd']], [], [], 0)[0] for v in values)
                or any(a['uuid'] in {v['uuid'] for v in values} for a in base.apps())):
            return
        leases = [{'gpu': v['gpu'], 'uuid': v['uuid'], 'path': str(v['reserve']),
                   'stat': c.lease_value(os.fstat(v['leasefd']))} for v in values]
        owned_pair(base, os.getpid(), leases, [v['uuid'] for v in values])
        c.save_once(QUEUE / 'actual_leases_held.json', {'schema': 'fixed3_actual_leases_held_v1',
                    'manager_identity': base.identity(os.getpid()), 'config_ref': c.ref(LAUNCH),
                    'resource_admission_ref': c.ref(QUEUE / 'admission.json'), 'node': 'N6', 'gpus': PAIR,
                    'leases': leases, 'historical_proof_current_FD_FLOCK_heartbeat_are_authority': True})
        proof_written = True

    def selected(cfg):
        if cfg.get('fixed3_primary_restore'):
            return repair.base.selected_sources(cfg)
        return [h['source_key'] for h in c.checked(cfg['jobs_manifest_ref'])['jobs']], 222939

    def summaries(cfg, out):
        if cfg.get('fixed3_primary_restore'):
            return repair.summaries(cfg, out)
        path = OUT / 'finite.stage.summary.json'
        summary = c.load(path)
        if (summary.get('schema') != 'curated_coarse_reobserve_fixed3_v1' or summary.get('global_revision') != REVISION
                or summary.get('new_text_model_requests') != 0 or not 0 <= summary.get('new_visual_requests', -1) <= 3
                or [row['id'] for row in summary['sources']] != list(worker_module().TARGETS)):
            raise RuntimeError('actual fixed3 zero-text finite summary required')
        worker = worker_module()
        frozen(worker)
        verify_completed_transactions(worker, active['contract'])
        for job in active['contract']['jobs']:
            root = OUT / 'revisions' / REVISION / job['source_key']
            if (root / 'result.json').exists():
                value = c.load(root / 'result.json')
                if value['original_job_ref'] != job['refs']['original_canonical_job'] or value['new_text_model_requests'] != 0:
                    raise RuntimeError('actual fixed3 original transaction/model budget changed')
                c.checked(value['transaction_receipt_ref'])
                original, joint, prepared, plan = worker.verify_job(job, probe=True)
                if value['style_result'] != joint['style_result']:
                    raise RuntimeError('original style changed')
        return {str(path): {'summary': summary, 'semantic_approval': False,
                            'primary_restoration_does_not_require_all_targets_complete': True}}

    base.AUTHORIZED_PAIRS = {'N6': ((0, 1),)}
    base.validate_pair, base.lease_snapshot = validate, snapshot
    base.acquire_lease, base.release_holder, base.refresh_leases = acquire, release, refresh
    base.selected_sources, base.stage_summaries = selected, summaries
    base.main()


c.check_node = check_node


def main():
    # Config arguments are deliberately left to the unchanged original manager.
    if '--config' in sys.argv:
        run_manager()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--build-fixed3', action='store_true')
    modes.add_argument('--build-primary-restore', action='store_true')
    modes.add_argument('--driver', action='store_true')
    parser.add_argument('--launch')
    args = parser.parse_args()
    if args.driver:
        if args.launch != str(LAUNCH):
            raise RuntimeError('only exact immutable fixed3 caller config')
        run_driver()
        return
    result = preflight() if args.preflight_only else build(restore=args.build_primary_restore)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
