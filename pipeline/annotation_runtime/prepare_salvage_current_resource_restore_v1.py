"""Same frozen salvage worker; full evidence first, actual current N6 resource last."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import sys
import time
import uuid

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TEST = RT / 'test_salvage_current_resource_restore_v1.py'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
SAFE_SHA = '8d07eb22484fdf70e183bb1466a6c65b01fa6248a7ddabea8765523601a97438'
SAFE_PATH = RT / 'coarse3_n6_launch_zero_dispatch_resume_main_v1.json'
OLD_CONFIG = {'path': str(RT / 'curated_clause_salvage_review_watch_main_fresh_v5_1792907677082418170.json'),
              'sha256': '49eb26d2a647ca85b481297722519491a2db303a5ba34a7ae85ebc2cd3802d44'}
OLD_PLAN = {'path': str(RT / 'curated_clause_salvage_review_watch_plan_v5-1792933781358213524-c0aeb1bee026.json'),
            'sha256': '15f714dbdde7ecbf6b2df00cd4ea1e2a3554344dbe670d50ec0565b55417b253'}
PINNED = {
    'resource_manager.py': BASE_SHA,
    'prepare_curated_clause_salvage_review_watch_v5.py': '050a97b5d404f4a91a2ad0dc9508aee24924604fa474682e05c28aeeb834bbb3',
    'curated_clause_salvage_review_watch_resource_manager_v5.py': '76d48559e7dc89f13f9835d2484757acad9ab89fc3ebf7108f4134ed83bbf983',
    'curated_clause_salvage_review_watch_v5.py': 'c79d1428c0650bce5616d28c96c34abc62677d1798703037d9f92f993cf9b6d2',
    'curated_clause_salvage_review_resume_validate_v5.py': '54cefaa1ff132118235ef443249dd6127c26b40b1bb58832ea7b4ee8408a6635',
}
STAT_KEYS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
LEASE = Path('/tmp/od_reserve_3')
GPU_LOCK = Path('/tmp/ms12k_gpu_3.lock')
BUDGET = 160 * 2**30
_cache = {}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def fingerprint(st):
    return tuple(getattr(st, k) for k in STAT_KEYS)


def content_sha(path):
    path = Path(path)
    before = path.stat()
    key = (os.path.abspath(os.fspath(path)), *fingerprint(before))
    if key in _cache:
        return _cache[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        finished = os.fstat(stream.fileno())
    need(all(fingerprint(s) == fingerprint(before) for s in (opened, finished, path.stat())),
         'File changed during full SHA: ' + str(path))
    _cache[key] = digest.hexdigest()
    return _cache[key]


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': content_sha(path)}


def checked(reference):
    need(Path(reference['path']).is_absolute() and content_sha(reference['path']) == reference['sha256'],
         'Immutable reference changed: ' + reference['path'])
    value = json.loads(Path(reference['path']).read_bytes())
    need(content_sha(reference['path']) == reference['sha256'], 'Reference changed during read')
    return value


def merge(bindings, added):
    for path, digest in added.items():
        need(path not in bindings or bindings[path] == digest, 'Original SHA binding cannot be re-signed: ' + path)
        bindings[path] = digest


def verify_bindings(bindings):
    for path, digest in bindings.items():
        need(Path(path).is_absolute() and content_sha(path) == digest, 'Original source SHA changed: ' + path)


def original():
    for name, digest in PINNED.items():
        need(content_sha(RT / name) == digest, 'Frozen original execution changed: ' + name)
    path = RT / 'prepare_curated_clause_salvage_review_watch_v5.py'
    spec = importlib.util.spec_from_file_location('salvage_current_private_prepare', path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    # Only the private native hash namespace gets an in-process four-stat cache.
    # No worker parameters, original validators, or external cache seeds change.
    prepare.base.sha = content_sha
    return prepare


def safe_chain(prepare, reference):
    base = prepare.base
    need(reference == {'path': str(SAFE_PATH), 'sha256': SAFE_SHA}, 'Exact actual previous safe config required')
    cfg = checked(reference)
    queue = Path(cfg['queue'])
    need(queue.parent == RT.parent / 'queues' and queue.name == 'coarse3_n6_independent_zero_dispatch_resume_1792931193890113953'
         and cfg['node'] == 'N6' and cfg['gpus'] == [3] and cfg['release_authorized'] is True,
         'Previous actual N6 GPU3 config scope differs')
    refs = {name: ref(queue / name) for name in ('admission.json', 'live.json', 'holder_restored_gpu3.json', 'resources_safe.json')}
    admission, live, restored, safe = (checked(refs[name]) for name in refs)
    expected_manager = ['/opt/venv_eval/bin/python', '-B', '-u', cfg['resource_manager']['path'],
                        '--config', reference['path'], '--config-sha', reference['sha256'], '--manager-sha', BASE_SHA]
    need(admission['config_sha256'] == reference['sha256'] and admission['manager_sha256'] == BASE_SHA
         and admission['manager_identity']['argv'] == expected_manager
         and admission['manager_identity']['uid'] == os.getuid(), 'Actual previous manager/config binding differs')
    need(all(admission[k] == cfg[k] for k in ('node', 'gpus', 'phase', 'targetcount', 'producer', 'plan', 'source_sha256', 'out'))
         and admission['model_identity_sha256'] == cfg['modelref']['model_identity_sha256'], 'Previous admission scope/source identity differs')
    need(all(live['manager_identity'][k] == admission['manager_identity'][k] for k in base.IDENTITY_KEYS)
         and live['argv'] == cfg['argv'], 'Actual previous live provenance differs')
    need(safe['phase'] == cfg['phase'] and all(safe.get(k) is True for k in
         ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')),
         'Previous resources not actually safe/restored')
    need(len(admission['holders']) == 1 and restored['gpu'] == 3
         and restored['uuid'] == admission['holders'][0]['uuid']
         and restored['original_identity'] == admission['holders'][0]['identity']
         and restored['restore_argv'] == base.RESTORE_ARGV
         and restored['holder_identity']['argv'] == base.RESTORE_ARGV
         and restored['managed_parent_wait'] is True, 'Actual restored holder identity differs')
    need((queue / 'admission.json').stat().st_mtime_ns <= (queue / 'holder_restored_gpu3.json').stat().st_mtime_ns
         <= (queue / 'resources_safe.json').stat().st_mtime_ns, 'Actual restoration receipt order differs')
    verify_bindings(cfg['source_sha256'])
    checked(cfg['plan'])
    checked(cfg['resource_manager'])
    return {'config_ref': reference, 'receipt_refs': refs, 'restored': restored}


def full_proof(prepare, expected_ref):
    expected = checked(expected_ref)
    sys.path.insert(0, str(RT / 'source_curated_review_v2'))
    import ms_batch_review as review
    proof = prepare.resume.resume_state_proof(prepare.worker, review, prepare.worker.w.Manifest(prepare.worker.pilot.MANIFEST))
    need(proof == expected, 'Current complete resume proof differs from original immutable proof; no rewrite')
    need(proof['processed_source_count'] == 662 and proof['real_audit_receipt_count'] == 110
         and proof['model_calls_performed'] == 0 and proof['state_migrated'] is False
         and proof['original_attempt_budgets_unchanged'] is True, 'Original662/110 resume scope differs')
    return proof


def evidence(prepare, safe_ref, *, local_model):
    worker = prepare.worker
    old = checked(OLD_CONFIG)
    plan = checked(OLD_PLAN)
    proof = full_proof(prepare, plan['resume_state_ref'])
    prior = safe_chain(prepare, safe_ref)
    bindings = dict(old['source_sha256'])
    merge(bindings, PINNED_PATHS())
    merge(bindings, {OLD_CONFIG['path']: OLD_CONFIG['sha256'], OLD_PLAN['path']: OLD_PLAN['sha256'],
                     plan['resume_state_ref']['path']: plan['resume_state_ref']['sha256']})
    merge(bindings, {r['path']: r['sha256'] for r in prepare.resume.references(proof)
                     if Path(r['path']) != worker.OUT / 'state.json'})
    merge(bindings, {r['path']: r['sha256'] for r in proof['verified_file_inventory']
                     if Path(r['path']) != worker.OUT / 'state.json'})
    merge(bindings, {safe_ref['path']: safe_ref['sha256'],
                     **{r['path']: r['sha256'] for r in prior['receipt_refs'].values()}})
    merge(bindings, {str(SELF): content_sha(SELF), str(TEST): content_sha(TEST)})
    verify_bindings(bindings)
    need(plan['schema'] == worker.PHASE and plan['input_stream'] == str(worker.STREAM)
         and all(plan.get(k) is False for k in worker.FLAGS), 'Frozen dynamic launch scope differs')
    if local_model:
        prepare.manager.check_model(old)
        reviewer = checked(old['modelref'])['model_identity']
        teacher = checked(ref(RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json'))['model_identity']
        weights = lambda identity: sorted(v['sha256'] for n, v in identity['files'].items()
                                          if Path(n).suffix in ('.safetensors', '.bin', '.pt', '.gguf'))
        need(weights(reviewer) and weights(reviewer) != weights(teacher), 'Reviewer actual checkpoint content equals annotator')
    need(prepare.base.required_budget(old) == BUDGET, 'Frozen160GiB native budget changed')
    return old, plan, proof, prior, bindings


def PINNED_PATHS():
    return {str(RT / n): digest for n, digest in PINNED.items()}


def lease_value(st):
    return None if st is None else {k: getattr(st, k) for k in STAT_KEYS}


def verify_lease(value):
    need(value is None or isinstance(value, dict) and set(value) == set(STAT_KEYS)
         and all(type(value[k]) is int for k in STAT_KEYS), 'Exact integer/ns native lease required')


def probe_lock(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, 'Unsafe native lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        now = Path(path).lstat()
        need(fingerprint(now) == fingerprint(st), 'Native lock replaced during read-only probe')
        return lease_value(st)
    finally:
        os.close(fd)


def fresh_current(prepare, prior):
    base = prepare.base
    need('28.216.19.225' in prepare.subprocess.check_output(['hostname', '-I'], text=True).split(), 'Only actual local N6 admitted')
    locks = {str(path): probe_lock(path) for path in (GPU_LOCK, prepare.worker.OUT / 'reviewer.lock')}
    lease = lease_value(base.lease_snapshot(LEASE))
    expected = dict(prior['restored']['holder_identity'], gpu=3, uuid=prior['restored']['uuid'])
    holder = dict(base.identity(expected['pid']), gpu=3, uuid=base.uuid_for(3))
    base.check_holder(expected, holder, 3, holder['uuid'], base.apps())
    fd = os.pidfd_open(holder['pid'])
    try:
        need(not select.select([fd], [], [], 0)[0], 'Current restored holder FD exited')
    finally:
        os.close(fd)
    budget = base.headroom()
    need(budget['headroom'] >= BUDGET and base.required_budget({'gpus': [3]}) == BUDGET, 'Current headroom under160GiB')
    base.check_holder(holder, base.identity(holder['pid']), 3, base.uuid_for(3), base.apps())
    need(lease_value(base.lease_snapshot(LEASE)) == lease, 'Exact lease changed during current capture')
    for path in locks:
        need(probe_lock(Path(path)) == locks[path], 'Native lock changed during current capture')
    return {'node': 'N6', 'gpus': [3], 'holders': [holder], 'captured_unix_ns': time.time_ns(),
            'signals_sent': False, 'lease_modified': False, 'lease': None if lease is None else [lease[k] for k in STAT_KEYS[:-1]],
            'lease_snapshot_ns': {str(LEASE): lease}, 'native_lock_stats': locks, 'budget': budget}


def copy_plan(path):
    checked(OLD_PLAN)
    raw = Path(OLD_PLAN['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == OLD_PLAN['sha256'], 'Original plan bytes changed')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    need(content_sha(path) == OLD_PLAN['sha256'], 'Original plan samebytes copy differs')
    return ref(path)


def build(prepare, out, safe_ref):
    out = Path(out).resolve()
    need(out.parent == RT and not os.path.lexists(out), 'New independent runtime config required')
    old, plan, proof, prior, bindings = evidence(prepare, safe_ref, local_model=True)
    suffix = str(time.time_ns()) + '-' + uuid.uuid4().hex[:12]
    plan_ref = copy_plan(RT / ('salvage_current_resource_plan_' + suffix + '.json'))
    merge(bindings, {plan_ref['path']: plan_ref['sha256']})
    # Every slow source/proof/checkpoint check completes before resource capture.
    verify_bindings(bindings)
    fresh = fresh_current(prepare, prior)
    capture_path = RT / ('salvage_current_resource_capture_' + suffix + '.json')
    prepare.worker.write_once(capture_path, fresh)
    cfg = copy.deepcopy(old)
    cfg.update(holders=fresh['holders'], queue=str(RT.parent / 'queues' / (prepare.manager.QUEUE_PREFIX + suffix)),
               holder_capture=ref(capture_path), plan=plan_ref, resume_state_ref=plan['resume_state_ref'],
               current_resource_restore_ref=ref(SELF), current_resource_restore_test_ref=ref(TEST),
               from_safe_config_ref=safe_ref, from_safe_receipt_refs=prior['receipt_refs'],
               lease_snapshot_ns=fresh['lease_snapshot_ns'], source_sha256=bindings)
    args = list(old['argv'])
    args[args.index('--launch') + 1] = plan_ref['path']
    args[args.index('--launch-sha') + 1] = plan_ref['sha256']
    cfg['argv'] = args
    merge(cfg['source_sha256'], {cfg['holder_capture']['path']: cfg['holder_capture']['sha256']})
    prepare.manager.validate(cfg)  # Unchanged20min TTL; this is genuinely new capture.
    need(0 <= time.time_ns() - fresh['captured_unix_ns'] <= 60 * 10**9, 'Current build capture aged before config commit')
    prepare.worker.write_once(out, cfg)
    return {'config_ref': ref(out), 'wrapper_ref': ref(SELF), 'base_manager_sha': BASE_SHA,
            'processed_source_count': proof['processed_source_count'], 'real_audit_receipt_count': proof['real_audit_receipt_count'],
            'required_headroom_bytes': BUDGET, 'signals_sent': False, 'lease_modified': False, 'model_calls_performed': 0,
            'command': launch_command(out)}


def launch_command(path):
    return ['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config', str(path),
            '--config-sha', content_sha(path), '--manager-sha', BASE_SHA,
            '--execution-sha', content_sha(SELF), '--tests-sha', content_sha(TEST)]


def admission(prepare, cfg, config_path, config_sha, manager_sha):
    need(cfg.get('current_resource_restore_ref') == ref(SELF) and cfg.get('current_resource_restore_test_ref') == ref(TEST),
         'Explicit current resource execution/test identity differs')
    # Original full source validator and original checkpoint gates, then complete
    # current source/receipt replay, then ACTUAL holder/lease/lock observations.
    prepare.manager.check_sources(cfg, config_path, config_sha, manager_sha)
    _, plan, _, prior, bindings = evidence(prepare, cfg['from_safe_config_ref'], local_model=True)
    need(cfg['resume_state_ref'] == plan['resume_state_ref'] and checked(cfg['plan']) == plan, 'Samebytes original launch/proof changed')
    old = checked(OLD_CONFIG)
    changed = {'holders', 'queue', 'holder_capture', 'plan', 'resume_state_ref', 'source_sha256', 'argv'}
    need(all(cfg.get(k) == v for k, v in old.items() if k not in changed), 'Frozen worker/runtime parameters changed')
    expected_argv = list(old['argv'])
    expected_argv[expected_argv.index('--launch') + 1] = cfg['plan']['path']
    expected_argv[expected_argv.index('--launch-sha') + 1] = cfg['plan']['sha256']
    need(cfg['argv'] == expected_argv and cfg['from_safe_receipt_refs'] == prior['receipt_refs'],
         'Frozen worker argv or actual safe receipt bindings changed')
    need(all(cfg['source_sha256'].get(p) == h for p, h in bindings.items()), 'Original source/code SHA dropped or re-signed')
    need(set(cfg['lease_snapshot_ns']) == {str(LEASE)}, 'Only exact GPU3 lease snapshot admitted')
    verify_lease(cfg['lease_snapshot_ns'][str(LEASE)])
    historical = checked(cfg['holder_capture'])
    need(historical['holders'] == cfg['holders'] and historical['lease_snapshot_ns'] == cfg['lease_snapshot_ns']
         and historical['signals_sent'] is False and historical['lease_modified'] is False, 'Historical capture/config binding differs')
    fresh = fresh_current(prepare, prior)
    prepare.base.check_holder(cfg['holders'][0], fresh['holders'][0], 3, fresh['holders'][0]['uuid'], prepare.base.apps())
    need(fresh['lease_snapshot_ns'] == cfg['lease_snapshot_ns'], 'Exact integer/ns lease changed since build')
    path = RT / ('salvage_current_resource_admission_' + str(time.time_ns()) + '.json')
    fresh.update(config_ref={'path': str(Path(config_path).resolve()), 'sha256': config_sha},
                 execution_ref=ref(SELF), tests_ref=ref(TEST), historical_capture_ref=cfg['holder_capture'],
                 source_and_resume_and_local_model_checks_preceded_current_capture=True)
    prepare.worker.write_once(path, fresh)
    # The original gate reads a separately persisted ACTUAL current capture;
    # historical config/capture bytes and its timestamp are never rewritten.
    current_view = dict(cfg, holders=fresh['holders'], holder_capture=ref(path))
    prepare.manager.validate(current_view)
    need(0 <= time.time_ns() - fresh['captured_unix_ns'] <= 60 * 10**9, 'Actual current admission capture aged')
    return fresh


def fd_owned(pid, dev, ino):
    for path in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            st = path.stat()
            if (st.st_dev, st.st_ino) == (dev, ino):
                return True
        except FileNotFoundError:
            pass
    return False


def flock_owned(pid, path):
    st = Path(path).lstat()
    key = '%02x:%02x:%d' % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    return any(len(v) >= 8 and v[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and v[4] == str(pid) and v[5] == key
               for v in (line.split() for line in Path('/proc/locks').read_text().splitlines()))


def before_release(prepare, cfg, entry, config_path):
    base = prepare.base
    parent = base.identity(os.getpid())
    need(parent['argv'] == launch_command(config_path), 'Exact current SELF manager argv differs before release')
    need(entry['gpu'] == 3 and entry['expected'] == cfg['holders'][0] and entry['reserve'] == LEASE,
         'Current resource release scope differs')
    verify_lease(cfg['lease_snapshot_ns'][str(LEASE)])
    need(lease_value(entry['old']) == cfg['lease_snapshot_ns'][str(LEASE)], 'Original lease changed after current capture')
    need(flock_owned(parent['pid'], GPU_LOCK), 'Current parent native GPU FLOCK absent')
    lock = GPU_LOCK.lstat()
    need(stat.S_ISREG(lock.st_mode) and lock.st_uid == os.getuid() and lock.st_size == 0
         and fd_owned(parent['pid'], lock.st_dev, lock.st_ino), 'Current GPU lock FD identity differs')
    probe_lock(prepare.worker.OUT / 'reviewer.lock')
    current, opened = entry['reserve'].lstat(), os.fstat(entry['leasefd'])
    need(stat.S_ISREG(current.st_mode) and current.st_uid == os.getuid() and current.st_size == 0
         and (current.st_dev, current.st_ino) == entry['lease_inode'] == (opened.st_dev, opened.st_ino)
         and fd_owned(parent['pid'], current.st_dev, current.st_ino)
         and 0 <= time.time_ns() - current.st_mtime_ns <= 60 * 10**9, 'Current owned lease FD/inode/heartbeat differs')
    need(not select.select([entry['fd']], [], [], 0)[0], 'Current holder pidfd exited before release')
    need(base.headroom()['headroom'] >= base.required_budget(cfg) == BUDGET, 'Current headroom changed before release')
    base.check_holder(entry['expected'], base.identity(entry['expected']['pid']), 3, base.uuid_for(3), base.apps())


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--config')
    p.add_argument('--out')
    p.add_argument('--from-safe-config')
    p.add_argument('--from-safe-config-sha')
    p.add_argument('--config-sha')
    p.add_argument('--manager-sha')
    p.add_argument('--execution-sha', required=True)
    p.add_argument('--tests-sha', required=True)
    a = p.parse_args()
    need(a.execution_sha == content_sha(SELF) and a.tests_sha == content_sha(TEST), 'Execution/tests SHA differs')
    prepare = original()
    if a.check_plan or a.build:
        need(a.from_safe_config and a.from_safe_config_sha, 'Actual safe config path+SHA required; no mocklaunch')
        safe_ref = {'path': str(Path(a.from_safe_config).resolve()), 'sha256': a.from_safe_config_sha}
        if a.check_plan:
            _, _, proof, _, bindings = evidence(prepare, safe_ref, local_model=False)
            print(json.dumps({'check_plan': True, 'processed_source_count': proof['processed_source_count'],
                              'real_audit_receipt_count': proof['real_audit_receipt_count'], 'source_SHA_count': len(bindings),
                              'actual_resource_capture_performed': False, 'signals_sent': False,
                              'lease_modified': False, 'model_calls_performed': 0}), flush=True)
        else:
            need(a.out, 'New independent --out required')
            print(json.dumps(build(prepare, a.out, safe_ref)), flush=True)
        return
    need(a.manager_sha == BASE_SHA and a.config_sha and content_sha(a.config) == a.config_sha, 'Frozen base/config SHA differs')
    cfg = checked({'path': str(Path(a.config).resolve()), 'sha256': a.config_sha})
    base = prepare.base
    native_release = base.release_holder
    base.validate_pair = lambda value: admission(prepare, value, a.config, a.config_sha, a.manager_sha)
    def release(entry):
        before_release(prepare, cfg, entry, a.config)
        native_release(entry)
    base.release_holder = release
    # The frozen native parser receives ONLY its original three options. /proc
    # cmdline retains the explicit SELF/execution/tests SHA used at release.
    sys.argv = [str(SELF), '--config', a.config, '--config-sha', a.config_sha, '--manager-sha', a.manager_sha]
    base.main()


if __name__ == '__main__':
    main()
