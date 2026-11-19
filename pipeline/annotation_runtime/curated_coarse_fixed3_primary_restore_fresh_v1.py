"""Independent fresh-resource primary restore; frozen history is never TERM authority."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TEST = RT / 'test_curated_coarse_fixed3_primary_restore_fresh_v1.py'
V2 = RT / 'curated_coarse_fixed3_managed_primary_v2.py'
V2_SHA = '4a5444cac4e91f43cd2e6332b1d715efe2b6622a55b3de2f7cf523d6c6184206'
V2_TEST_SHA = '944c5036e82cac2dd029094bf05a9a126460dd9859e3ad3540b5456e80ca9a1f'
FINITE_SHA = 'f35c4d5142a5e152bcbfb19792f9724a9917d1f27e6cf24d6fd6af3078e45c95'
if hashlib.sha256(V2.read_bytes()).hexdigest() != V2_SHA:
    raise RuntimeError('frozen original v2 adapter changed')
spec = importlib.util.spec_from_file_location('fresh_restore_private_original_v2', V2)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
m, c = v.m, v.m.c
m.RESTORE_QUEUE = RT.parent / 'queues/curated_coarse_fixed3_primary_restore_fresh_v1'
m.RESTORE_LAUNCH = RT / 'curated_coarse_fixed3_primary_restore_fresh_launch_v1.json'
CAPTURE = RT / 'curated_coarse_fixed3_primary_restore_fresh_capture_v1.json'
PRIOR = RT / 'curated_coarse_fixed3_primary_restore_fresh_prior_v1.json'
_native_fresh = c.fresh_pair
_native_verify_restore = m.verify_restore_config


def historical_binding(contract):
    v.execution_bridge()
    if c.sha(v.TEST) != V2_TEST_SHA:
        raise RuntimeError('frozen original v2 tests changed')
    proof = c.checked({'path': str(v.PREFLIGHT), 'sha256': v.PREFLIGHT_SHA})
    bindings = {p: h for p, h in contract['bindings'].items() if p not in v.BRIDGE_KEYS}
    if (len(bindings) != 136 or bindings != proof['source_sha256']
            or proof['schema'] != 'fixed3_primary_cpu_preflight_v1'
            or proof['original_config_ref'] != {'path': str(m.OLD), 'sha256': m.OLD_SHA}
            or proof['jobs_manifest_ref'] != contract['manifest_ref']
            or proof['source_freeze_ref'] != contract['freeze_ref'] or proof['signals_sent'] is not False
            or type(proof['captured_unix_ns']) is not int or proof['captured_unix_ns'] <= 0):
        raise RuntimeError('historical original 136-source binding changed; not current authorization')
    return proof


def transactions(worker, contract):
    refs = []
    for hint, job in zip(contract['manifest']['jobs'], contract['jobs']):
        root = m.OUT / 'revisions' / m.REVISION / job['source_key']
        budget = worker.budget_path(job)
        if budget.exists():
            expected = dict(source_key=job['source_key'], original_result_sha256=job['original_result_sha256'],
                            global_revision=m.REVISION, extra_source_attempt=1, maximum_extra_source_attempts=1)
            if c.load(budget) != expected:
                raise RuntimeError('original finite canonical budget changed')
            refs.append(c.ref(budget))
        if not root.exists():
            continue
        if (not budget.exists() or c.load(root / 'canonical_budget_claim.json') != c.load(budget)
                or c.load(root / 'scope_job.json') != job
                or c.load(root / 'original_job.json') != c.checked(job['refs']['original_canonical_job'])):
            raise RuntimeError('finite original three-source partial transaction/budget changed')
        partial = root / 'partial.dispatch.json'
        if partial.exists():
            value = c.load(partial)
            if (value['status'] != 'partial_ambiguous_dispatch_spent' or value['current_job_ref'] != hint['job_ref']
                    or value['canonical_budget_claim'] != c.load(budget)
                    or any(value.get(k) is not False for k in worker.q.FLAGS)):
                raise RuntimeError('finite ambiguous failure transaction changed')
        attempts = list((root / 'windows').glob('*/attempt-*'))
        if any(p != root / 'windows' / job['target_window_id'] / 'attempt-000001' for p in attempts):
            raise RuntimeError('finite unexpected source attempt/window')
        if any(p.exists() for p in [root / 'result.json', *[a / n for a in attempts for n in ('receipt.json', 'binding.json')]]) and not (root / 'windows' / job['target_window_id'] / 'attempt-000001/request.json').exists():
            raise RuntimeError('finite partial transaction lacks original request')
        generation = root / 'windows' / job['target_window_id'] / 'attempt-000001/generation.json'
        if generation.exists() and c.load(generation).get('model_identity_sha256') != contract['consumer']['modelref']['model_identity_sha256']:
            raise RuntimeError('finite partial generation original model changed')
        refs.extend(c.ref(p) for p in sorted(root.rglob('*')) if p.is_file())
    m.verify_completed_transactions(worker, contract)
    return refs


def previous(base, contract=None, worker=None):
    if not m.QUEUE.is_dir() or not (m.QUEUE / 'resources_safe.json').is_file():
        raise RuntimeError('finite queue actual resources_safe mandatory; no original-oldqueue fallback')
    finite_ref = {'path': str(m.LAUNCH), 'sha256': FINITE_SHA}
    finite = c.checked(finite_ref)
    admission_ref = c.ref(m.QUEUE / 'admission.json')
    result = m.restored_evidence(base, m.QUEUE, admission_ref)
    worker = worker or m.worker_module()
    contract = contract or m.frozen(worker)
    proof = historical_binding(contract)
    m.verify_config(finite, contract)
    admission = c.checked(admission_ref)
    if c.load(m.QUEUE / 'resources_safe.json').get('phase') != finite['phase'] or len(admission['holders']) != 2:
        raise RuntimeError('finite safe phase/exact two-holder transaction changed')
    expected = {k: finite[k] for k in ('phase', 'targetcount', 'node', 'gpus', 'producer', 'plan', 'out', 'source_sha256')}
    expected.update(config_sha256=FINITE_SHA, manager_sha256=m.BASE_SHA,
                    model_identity_sha256=finite['modelref']['model_identity_sha256'])
    if any(admission.get(k) != value for k, value in expected.items()):
        raise RuntimeError('finite actual admission original config/source/model mismatch')
    manager = admission['manager_identity']
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(V2), '--config', str(m.LAUNCH), '--config-sha', FINITE_SHA, '--manager-sha', m.BASE_SHA]
    if manager['argv'] != argv or manager['uid'] != os.getuid():
        raise RuntimeError('finite actual managed identity/argv changed')
    original_ref, _ = m.primary_admission(m.original_config())
    if original_ref != proof['original_admission_ref']:
        raise RuntimeError('historical original primary admission changed')
    original = m.restored_evidence(base, Path(m.original_config()['queue']), original_ref)
    for gpu, holder, oldrow in zip(m.PAIR, admission['holders'], original['previous_pairs']):
        expected_holder = finite['holders'][gpu]
        prior_holder = oldrow['restored']['holder_identity']
        if (holder['gpu'] != gpu or holder['uuid'] != expected_holder['uuid']
                or any(holder['identity'][k] != expected_holder[k] or holder['identity'][k] != prior_holder[k] for k in base.IDENTITY_KEYS)):
            raise RuntimeError('finite original holder/primary restoration chain changed')
    live_ref = c.ref(m.QUEUE / 'live.json')
    live = c.checked(live_ref)
    child = live['child_identity']
    if (any(live['manager_identity'][k] != manager[k] for k in base.IDENTITY_KEYS)
            or live['argv'] != finite['argv'] or live['phase'] != finite['phase'] or live['targetcount'] != 3
            or child is None or child['ppid'] != manager['pid'] or child['argv'] != finite['argv']
            or child['uid'] != manager['uid'] or child['cgroup'] != manager['cgroup'] or child['CVD'] != ['CUDA_VISIBLE_DEVICES=0,1']):
        raise RuntimeError('finite actual live direct worker transaction changed')
    held_ref, worker_ref = c.ref(m.QUEUE / 'actual_leases_held.json'), c.ref(m.QUEUE / 'actual_worker_admission.json')
    held, actual = c.checked(held_ref), c.checked(worker_ref)
    if (held['schema'] != 'fixed3_actual_leases_held_v1' or held['config_ref'] != finite_ref or held['resource_admission_ref'] != admission_ref
            or any(held['manager_identity'][k] != manager[k] for k in base.IDENTITY_KEYS) or held['gpus'] != m.PAIR or held['node'] != 'N6'
            or actual['launch_config_ref'] != finite_ref or actual['resource_admission_ref'] != admission_ref
            or actual['caller_pid'] != child['pid'] or actual['model_identity_sha256'] != finite['modelref']['model_identity_sha256']
            or any(actual.get(k) != val for k, val in {'gpus': m.PAIR, 'tensor_parallel_size': 2, 'out': str(m.OUT), 'global_revision': m.REVISION, 'jobs_manifest_ref': contract['manifest_ref'], 'actual_held_leases_ref': held_ref}.items())
            or actual['source_job_refs'] != [h['job_ref'] for h in contract['manifest']['jobs']]
            or actual['original_source_job_refs'] != [j['refs']['original_canonical_job'] for j in contract['jobs']]):
        raise RuntimeError('finite actual held lease/three-source caller admission changed')
    refs = [finite_ref, admission_ref, live_ref, held_ref, worker_ref, proof['original_live_ref'], c.ref(v.PREFLIGHT), original_ref,
            original['resources_safe_ref'], *[r for row in original['previous_pairs'] for r in row['history_refs']]]
    for name in ('failure.json', 'worker_exit.json'):
        path = m.QUEUE / name
        if not path.exists():
            continue
        value = c.load(path)
        if (value.get('phase') != finite['phase'] or not (m.QUEUE / 'admission.json').stat().st_mtime_ns <= path.stat().st_mtime_ns <= (m.QUEUE / 'resources_safe.json').stat().st_mtime_ns):
            raise RuntimeError('finite terminal phase/order changed')
        required = {'partial_outputs_preserved': True, 'worker_started': True, 'unknown_jobs_not_signaled': True} if name == 'failure.json' else {'producer': finite['producer'], 'plan': finite['plan'], 'source_unchanged': True, 'input_count': 222939, 'targetcount': 3, 'semantic_approval': False}
        if any(value.get(k) != val for k, val in required.items()):
            raise RuntimeError('finite failure/worker_exit original transaction changed')
        refs.append(c.ref(path))
    if not any(Path(r['path']).name in ('failure.json', 'worker_exit.json') for r in refs):
        raise RuntimeError('finite actual failure or worker exit required before restore')
    for reference in refs:
        c.checked(reference)
    refs.extend(transactions(worker, contract))
    result.update(finite_launch_ref=finite_ref, historical_preflight_is_current_authority=False,
                  history_refs=refs, semantic_approval=False, primary_all_targets_completed_not_required=True)
    return result


def fresh_pair(base, prior):
    finite = c.checked(prior['finite_launch_ref'])
    fresh = _native_fresh(base, prior)
    for path, value in fresh['lease_snapshot_ns'].items():
        c.verify_lease_value(value)
        old = finite['lease_snapshot_ns'][path]
        c.verify_lease_value(old)
        if ((old is None) != (value is None) or old is not None and
                (any(old[k] != value[k] for k in c.LEASE_KEYS if k != 'st_ctime_ns') or value['st_ctime_ns'] < old['st_ctime_ns'])):
            raise RuntimeError('finite original leases not actually restored before fresh capture')
    fresh['schema'] = 'fixed3_primary_actual_current_restore_capture_v1'
    return fresh


def verify_restore(cfg, old):
    _native_verify_restore(cfg, old)
    if (cfg.get('fresh_resource_authority') != 'actual-current-after-finite-safe-v1'
            or cfg['holder_capture'] != c.ref(CAPTURE) or cfg['previous_resources_ref'] != c.ref(PRIOR)):
        raise RuntimeError('independent actual fresh resource authority missing')
    if c.checked(cfg['holder_capture']).get('schema') != 'fixed3_primary_actual_current_restore_capture_v1':
        raise RuntimeError('historical proof cannot replace actual current capture')
    for path in (SELF, TEST):
        if cfg['source_sha256'].get(str(path)) != c.sha(path):
            raise RuntimeError('independent restore execution/test source changed')


def build():
    m.check_node()
    if any(p.exists() for p in (m.RESTORE_QUEUE, m.RESTORE_LAUNCH, CAPTURE, PRIOR)):
        raise RuntimeError('independent restore scope immutable; no redispatch or overwrite')
    if not m.QUEUE.is_dir() or not (m.QUEUE / 'resources_safe.json').is_file():
        raise RuntimeError('finite queue actual resources_safe mandatory; no original-oldqueue fallback')
    base, worker = c.base_module(), m.worker_module()
    contract, old = m.frozen(worker), m.original_config()
    prior = previous(base, contract, worker)
    cfg = copy.deepcopy(old)
    cfg.update(queue=str(m.RESTORE_QUEUE), fixed3_primary_restore=True, original_primary_config_ref=c.ref(m.OLD),
               fixed3_launch_ref={'path': str(m.LAUNCH), 'sha256': FINITE_SHA}, execution_bridge=v.execution_bridge(),
               fresh_resource_authority='actual-current-after-finite-safe-v1')
    model = c.verify_local_model(dict(cfg, local_model_identity=c.checked(prior['finite_launch_ref'])['local_model_identity']), worker)
    bindings = dict(old['source_sha256'], **contract['bindings'])
    bindings.update(c.references(prior))
    bindings.update({str(SELF): c.sha(SELF), str(TEST): c.sha(TEST), str(m.OLD): m.OLD_SHA})
    for path, digest in bindings.items():
        if c.sha(path) != digest:
            raise RuntimeError('full source/model/history changed before actual fresh capture')
    fresh = fresh_pair(base, prior)
    c.save_once(PRIOR, prior)
    c.save_once(CAPTURE, fresh)
    bindings.update({str(PRIOR): c.sha(PRIOR), str(CAPTURE): c.sha(CAPTURE)})
    cfg.update(holders=fresh['holders'], lease_snapshot_ns=fresh['lease_snapshot_ns'], holder_capture=c.ref(CAPTURE),
               previous_resources_ref=c.ref(PRIOR), local_model_identity=model, source_sha256=bindings)
    verify_restore(cfg, old)
    c.save_once(m.RESTORE_LAUNCH, cfg)
    return dict(launch_ref=c.ref(m.RESTORE_LAUNCH), manager_ref=c.ref(SELF), signals_sent=False, model_loads=0,
                GPU_dispatch_performed=False, command=['/opt/venv_eval/bin/python', '-B', '-u', str(SELF), '--config',
                str(m.RESTORE_LAUNCH), '--config-sha', c.sha(m.RESTORE_LAUNCH), '--manager-sha', m.BASE_SHA])


m.primary_restore_previous, m.verify_restore_config, c.fresh_pair = previous, verify_restore, fresh_pair


def run_manager():
    if sys.argv.count('--config') != 1 or sys.argv[sys.argv.index('--config') + 1] != str(m.RESTORE_LAUNCH):
        raise RuntimeError('only independent primary restore config allowed')
    if c.load(m.RESTORE_LAUNCH).get('fixed3_primary_restore') is not True:
        raise RuntimeError('never dispatch a finite/relabelled worker from fresh restore')
    m.run_manager()


if __name__ == '__main__':
    if '--config' in sys.argv:
        run_manager()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--build-primary-restore', action='store_true', required=True)
        parser.parse_args()
        print(json.dumps(build(), ensure_ascii=False, allow_nan=False), flush=True)
