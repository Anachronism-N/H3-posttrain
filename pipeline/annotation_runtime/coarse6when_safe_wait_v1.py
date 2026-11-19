"""One managed CPU waiter; execs a fresh pair manager only after exact safe return."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import coarse6when_safe_common_v1 as c


def verify_scope(reference):
    scope = c.checked(reference)
    if scope.get('schema') != 'coarse6when_safe_scope_v1' or scope.get('launch_authorized') is not False:
        raise RuntimeError('source-only frozen nonauthorizing scope required')
    for path, digest in scope['source_sha256'].items():
        if c.sha(path) != digest:
            raise RuntimeError('waiting chain source changed: ' + path)
    if scope['previous_config_ref'] != {'path': str(c.PREVIOUS), 'sha256': c.PREVIOUS_SHA} or scope['template_ref'] != {'path': str(c.TEMPLATE), 'sha256': c.TEMPLATE_SHA}:
        raise RuntimeError('scope exact previous/template binding changed')
    return scope


def probe(base, previous, jobs):
    c.require_unreserved(jobs)
    evidence = c.previous_completed(previous)
    capture = c.fresh_pair(base, evidence)
    c.require_unreserved(jobs)
    return evidence, capture


def build_launch(scope_ref, scope, frozen, evidence, capture):
    template, previous, consumer, manifest, jobs, bindings = frozen
    c.require_unreserved(jobs)
    if c.LAUNCH.exists() or c.QUEUE.exists():
        raise RuntimeError('immutable launch/queue already exists; no retry/reset')
    evidence_path = c.RT / 'coarse6when_safe_previous_complete_v1.json'
    capture_path = c.RT / 'coarse6when_safe_fresh_pair_v1.json'
    c.save_once(evidence_path, evidence)
    c.save_once(capture_path, capture)
    bindings.update(scope['source_sha256'])
    bindings[scope_ref['path']] = scope_ref['sha256']
    for p in (evidence_path, capture_path):
        bindings[str(p)] = c.sha(p)
    cfg = {'node': 'N3', 'gpus': c.PAIR, 'phase': 'coarse6when_safe_n3g01_v1', 'targetcount': 2,
           'input_count': 222939, 'release_authorized': True, 'holders': capture['holders'],
           'lease_snapshot_ns': capture['lease_snapshot_ns'], 'holder_capture': c.ref(capture_path),
           'previous_complete_ref': c.ref(evidence_path), 'scope_ref': scope_ref,
           'global_revision': c.REVISION, 'template_ref': c.ref(c.TEMPLATE),
           'jobs_manifest_ref': template['jobs_manifest_ref'], 'protocol_ref': template['protocol_ref'],
           'modelref': consumer['modelref'], 'plan': consumer['plan'], 'producer': c.ref(c.DRIVER),
           'source_sha256': bindings, 'out': str(c.OUT), 'queue': str(c.QUEUE),
           'cwd': consumer['cwd'], 'reuse_existing_results': True,
           'env': {**template['env'], 'CUDA_VISIBLE_DEVICES': '0,1', 'COARSE6_MANAGER_CONFIG': str(c.LAUNCH)},
           'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(c.DRIVER), '--launch', str(c.LAUNCH)],
           'semantic_approval': False, 'training_eligible': False, 'independent_review_required': True}
    c.save_once(c.LAUNCH, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', required=True)
    parser.add_argument('--scope-sha', required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.wait == args.check or args.poll_seconds < 5:
        parser.error('choose exactly one of --wait/--check; poll >=5 seconds')
    if '28.216.19.213' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('execute the single CPU waiter locally on N3')
    reference = {'path': args.scope, 'sha256': args.scope_sha}
    scope = verify_scope(reference)
    frozen = c.frozen()
    base = c.base_module()
    lockfd = None
    if args.wait:
        lockfd = os.open('/tmp/coarse6when_safe_waiter_n3g01_v1.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        s = os.fstat(lockfd)
        if not c.stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size != 0:
            raise RuntimeError('unsafe waiter lock')
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_reason = None
    while True:
        try:
            verify_scope(reference)
            evidence, capture = probe(base, frozen[1], frozen[4])
            if args.wait:
                frozen = c.frozen()
                base.check_model({'node': 'N3', 'modelref': frozen[2]['modelref']})
                worker = c.module('coarse6_frozen_prelaunch_verify', c.WORKER)
                for job in frozen[4]:
                    worker.verify_job(job, probe=False)
                evidence, capture = probe(base, frozen[1], frozen[4])
        except Exception as error:
            reason = str(error)
            if reason != last_reason or args.check:
                print(json.dumps({'status': 'waiting', 'reason': reason, 'signals_sent': False,
                                  'gpu_started': False, 'launch_created': False, 'semantic_approval': False}), flush=True)
                last_reason = reason
            if args.check:
                return
            time.sleep(args.poll_seconds)
            continue
        if args.check:
            print(json.dumps({'status': 'ready_readonly_not_authorized', 'signals_sent': False,
                              'gpu_started': False, 'launch_created': False, 'terminal_count': evidence['terminal_count']}), flush=True)
            return
        build_launch(reference, scope, frozen, evidence, capture)
        argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(c.MANAGER), '--config', str(c.LAUNCH),
                '--config-sha', c.sha(c.LAUNCH), '--manager-sha', c.BASE_SHA]
        print(json.dumps({'status': 'exec_fresh_resource_manager', 'config_ref': c.ref(c.LAUNCH),
                          'signals_sent_by_waiter': False, 'gpu_started_by_waiter': False}), flush=True)
        os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
