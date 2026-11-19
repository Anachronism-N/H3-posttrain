"""Derive worker admission from the real parent manager transaction, never booleans."""
import argparse
import os
from pathlib import Path
import stat
import time
import coarse6when_safe_common_v1 as c


def verify_actual_parent(cfg, base):
    if Path(cfg['queue']).resolve() != c.QUEUE or cfg['out'] != str(c.OUT) or cfg['node'] != 'N3' or cfg['gpus'] != c.PAIR:
        raise RuntimeError('exact physical pair/resource queue required')
    resource_ref = c.ref(c.QUEUE / 'admission.json')
    admission = c.checked(resource_ref)
    proof_ref = c.ref(c.QUEUE / 'actual_leases_held.json')
    proof = c.checked(proof_ref)
    if proof.get('schema') != 'coarse6_actual_manager_leases_held_v1' or proof.get('resource_admission_ref') != resource_ref or proof.get('config_ref') != c.ref(c.LAUNCH) or proof.get('gpus') != c.PAIR or proof.get('node') != 'N3':
        raise RuntimeError('real manager held-lease evidence absent/mismatched')
    if admission.get('config_sha256') != c.sha(c.LAUNCH) or admission.get('manager_sha256') != c.BASE_SHA or admission.get('source_sha256') != cfg['source_sha256'] or admission.get('producer') != cfg['producer'] or admission.get('plan') != cfg['plan'] or admission.get('model_identity_sha256') != cfg['modelref']['model_identity_sha256'] or admission.get('node') != 'N3' or admission.get('gpus') != c.PAIR or admission.get('out') != str(c.OUT) or admission.get('targetcount') != 2:
        raise RuntimeError('real parent resource admission binding changed')
    pid = os.getppid()
    parent = base.identity(pid)
    expected_argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(c.MANAGER), '--config', str(c.LAUNCH),
                     '--config-sha', c.sha(c.LAUNCH), '--manager-sha', c.BASE_SHA]
    if parent['pid'] != admission['manager_identity']['pid'] or parent['uid'] != os.getuid() or parent['argv'] != expected_argv or parent['cgroup'] != base.identity(os.getpid())['cgroup']:
        raise RuntimeError('thin driver is not the direct real manager child')
    for key in base.IDENTITY_KEYS:
        if parent[key] != admission['manager_identity'][key] or parent[key] != proof['manager_identity'][key]:
            raise RuntimeError('real manager identity changed: ' + key)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1' or os.environ.get('COARSE6_MANAGER_CONFIG') != str(c.LAUNCH):
        raise RuntimeError('actual worker environment differs from admitted pair')
    if not 0 <= time.time_ns() - proof['observed_unix_ns'] <= 120 * 10**9 or admission['budget']['headroom'] < 160 * 2**30:
        raise RuntimeError('held transaction evidence stale/underbudget')
    if len(proof['leases']) != 2 or [v['gpu'] for v in proof['leases']] != c.PAIR:
        raise RuntimeError('all real physical-pair leases required')
    for value in proof['leases']:
        path = Path('/tmp/od_reserve_%d' % value['gpu'])
        if value['path'] != str(path):
            raise RuntimeError('physical lease path mismatch')
        s = path.lstat()
        expected = value['stat']
        c.verify_lease_value(expected)
        if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size != 0 or (s.st_dev, s.st_ino) != (expected['st_dev'], expected['st_ino']) or not 0 <= time.time_ns() - s.st_mtime_ns <= 60 * 10**9:
            raise RuntimeError('real owned regular-empty active lease changed')
        if not c.fd_inode_owned(pid, s.st_dev, s.st_ino) or not c.proc_lock_owned(pid, '/tmp/ms12k_gpu_%d.lock' % value['gpu']):
            raise RuntimeError('parent does not actually hold all lock/lease descriptors')
    if base.apps():
        uuids = {h['uuid'] for h in admission['holders']}
        if any(app['uuid'] in uuids for app in base.apps()):
            raise RuntimeError('unexpected GPU app before model worker; no app touched')
    base.check_model(cfg)
    for path, digest in cfg['source_sha256'].items():
        if c.sha(path) != digest:
            raise RuntimeError('actual source freeze changed: ' + path)
    return resource_ref, proof_ref


def derive_worker_admission(cfg, base):
    resource_ref, proof_ref = verify_actual_parent(cfg, base)
    template, previous, consumer, manifest, jobs, bindings = c.frozen()
    if cfg['template_ref'] != c.ref(c.TEMPLATE) or cfg['modelref'] != consumer['modelref'] or any(cfg['source_sha256'].get(p) != h for p, h in bindings.items()):
        raise RuntimeError('worker immutable template/model/source config mismatch')
    c.require_unreserved(jobs)
    return {'schema': 'coarse6_actual_worker_admission_v1', 'template_ref': c.ref(c.TEMPLATE),
            'out': str(c.OUT), 'global_revision': c.REVISION, 'node': 'N3', 'gpus': c.PAIR,
            'jobs_manifest_ref': template['jobs_manifest_ref'], 'protocol_ref': template['protocol_ref'],
            'consumer_config_ref': template['consumer_config_ref'], 'worker_ref': template['worker_ref'],
            'modelref': cfg['modelref'], 'resource_admission_ref': resource_ref,
            'actual_held_leases_ref': proof_ref, 'launch_config_ref': c.ref(c.LAUNCH),
            'launch_authorized': True, 'fresh_pair_safety_verified': True,
            'resource_manager_verified_and_leases_held': True, 'verified_unix_ns': time.time_ns(),
            'semantic_approval': False, 'human_review_completed': False, 'formal_training': False,
            'training_eligible': False, 'independent_review_required': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', required=True)
    args = parser.parse_args()
    if Path(args.launch).resolve() != c.LAUNCH:
        raise RuntimeError('only immutable new launch allowed')
    cfg = c.load(c.LAUNCH)
    base = c.base_module()
    admission = derive_worker_admission(cfg, base)
    path = c.QUEUE / 'actual_worker_admission.json'
    c.save_once(path, admission)
    argv = ['/opt/venv_vllm/bin/python', '-B', '-u', str(c.WORKER), 'run', '--admission', str(path)]
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
