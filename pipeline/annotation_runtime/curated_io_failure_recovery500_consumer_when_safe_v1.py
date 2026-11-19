"""CPU-only consumer waiter for the independent three-source I/O recovery.

Wait for the recovery producer's real exit and resource restoration, then admit
only N3 physical GPUs6/7 with fresh project holders. All signals remain solely
inside the verified consumer resource manager after exec.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

RT = Path(__file__).resolve().parent
OUT = RT.parent / 'curated222939_io_failure_recovery_v1'
PRODUCER_WAITER = RT / 'curated_io_failure_recovery500_when_safe_v1.py'
PRODUCER_WAITER_SHA = '7b81281aca399fa549c1a75d08931be55aa85d8b236fda7844915562edde23b8'
PRODUCER_POINTER = RT / 'curated_io_failure_recovery500_producer_when_safe_admission_v1.json'
CONSUMER_POINTER = RT / 'curated_io_failure_recovery500_consumer_when_safe_admission_v1.json'
BASE_MANAGER_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
EXPECTED_IDS = ['sSAKrWDrySCzGbar', 'OertkYlRRBiBIQbU', 'UbjwIZuNtSoLcUaG']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def producer_gate(cfg, support):
    if not PRODUCER_POINTER.exists():
        raise support.WaitSafe('Waiting recovery producer fresh-authorized config pointer')
    pointer = support.load(PRODUCER_POINTER)
    reference = pointer['config']
    if (pointer['source_prepared_ref'] != {'path': str(support.SOURCE), 'sha256': support.SOURCE_SHA}
            or Path(reference['path']).parent != RT
            or not Path(reference['path']).name.startswith('curated_io_failure_recovery500_n5g0_launch_')
            or support.base.sha(reference['path']) != reference['sha256']):
        raise RuntimeError('Recovery producer authorization provenance changed')
    producer = support.load(reference['path'])
    if (producer['mode'] != 'producer' or producer['node'] != 'N5' or producer['gpus'] != [0]
            or Path(producer['out']) != OUT or producer['targetcount'] != 3
            or producer['source_range'] != {'start': 500, 'end': 503}
            or producer['source_prepared_reference'] != pointer['source_prepared_ref']):
        raise RuntimeError('Recovery producer source/output binding changed')
    queue = Path(producer['queue'])
    paths = [queue / name for name in ('admission.json', 'worker_exit.json', 'resources_safe.json')]
    if not all(path.exists() for path in paths):
        raise support.WaitSafe('Waiting recovery producer actual exit and resources_safe')
    admission, finished, safe = [support.load(path) for path in paths]
    if (admission.get('config_sha256') != reference['sha256'] or admission.get('phase') != producer['phase']
            or admission.get('node') != 'N5' or admission.get('gpus') != [0]
            or admission.get('out') != str(OUT)):
        raise RuntimeError('Recovery producer admission binding changed')
    if (finished.get('exit_code') != 0 or finished.get('phase') != producer['phase']
            or finished.get('targetcount') != 3 or finished.get('input_count') != 222939
            or finished.get('source_selection') != producer['source_selection']
            or finished.get('producer') != producer['producer'] or finished.get('plan') != producer['plan']
            or finished.get('source_unchanged') is not True):
        raise support.WaitSafe('Recovery producer lacks successful exact three-source worker receipt')
    summaries = finished.get('summaries', {})
    if not any(value.get('summary', {}).get('processed_source_count') == 3
            and value.get('summary', {}).get('selected_source_count') == 3
            and value.get('selection', {}).get('selected_ids') == EXPECTED_IDS for value in summaries.values()):
        raise support.WaitSafe('Recovery producer exact selected-source summary absent')
    if (safe.get('phase') != producer['phase'] or not all(safe.get(key) is True
            for key in ('all_holders_safe', 'locks_released', 'leases_restored', 'unknown_jobs_not_signaled'))
            or paths[2].stat().st_mtime_ns < paths[1].stat().st_mtime_ns):
        raise support.WaitSafe('Recovery producer GPU holder/lease/lock restoration incomplete')
    references = [support.ref(PRODUCER_POINTER), support.ref(reference['path'])]
    references.extend(support.ref(path) for path in paths)
    rows = support.load(cfg['selected_sources']['path'])
    if [row['id'] for row in rows] != EXPECTED_IDS or [row['source_ordinal'] for row in rows] != [500, 501, 502]:
        raise RuntimeError('Recovery consumer selected paths changed')
    for row in rows:
        terminal = OUT / 'sources' / row['split'] / row['id'] / 'producer.terminal.json'
        if not terminal.exists():
            raise support.WaitSafe('Recovery producer terminal missing: ' + row['id'])
        value = support.load(terminal)
        if (value['id'] != row['id'] or value['split'] != row['split']
                or value['batch_id'] != row['batch_id'] or value['source_ordinal'] != row['source_ordinal']):
            raise RuntimeError('Recovery producer terminal binding changed')
        if value['status'] != 'ready':
            raise support.WaitSafe('Recovery producer has nonready source; preserve failed alternative: ' + row['id'])
        references.append(support.ref(terminal))
        for key in ('plan_ref', 'roi_ref'):
            artifact = value[key]
            if (not Path(artifact['path']).resolve().is_relative_to(OUT)
                    or support.base.sha(artifact['path']) != artifact['sha256']):
                raise RuntimeError('Recovery producer artifact changed/escaped isolated output')
            references.append(artifact)
    return references


def source_config(args, support):
    path = Path(args.source_prepared).resolve()
    manager = Path(args.manager).resolve()
    if (path.parent != RT or manager.parent != RT
            or support.base.sha(path) != args.source_prepared_sha
            or support.base.sha(manager) != args.manager_sha
            or support.base.sha(RT / 'resource_manager.py') != BASE_MANAGER_SHA):
        raise RuntimeError('Recovery consumer source-prepared/manager SHA/path changed')
    cfg = support.load(path)
    if (cfg['node'] != 'N3' or cfg['mode'] != 'consumer' or cfg['gpus'] != [6, 7]
            or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '6,7' or Path(cfg['out']) != OUT
            or cfg['targetcount'] != 3 or cfg['input_count'] != 222939
            or cfg['holders'] != [] or cfg['holder_capture'] is not None
            or cfg['release_authorized'] is not False or 'private_ready' in cfg
            or cfg['resource_manager_reference'] != {'path': str(manager), 'sha256': args.manager_sha}):
        raise RuntimeError('Only isolated three-source consumer placeholder N3G6/G7 admitted')
    for source, sha in cfg['source_sha256'].items():
        if support.base.sha(source) != sha:
            raise RuntimeError('Recovery consumer source/provenance changed: ' + source)
    return cfg


def authorize(args, cfg, capture, references, support, manager):
    if Path(cfg['queue']).exists() or CONSUMER_POINTER.exists():
        raise RuntimeError('Recovery consumer queue/admission already exists; refusing duplicate launch')
    token = uuid.uuid4().hex
    capture_path = RT / ('curated_io_failure_recovery500_n3g67_fresh_' + token + '.json')
    config_path = RT / ('curated_io_failure_recovery500_n3g67_launch_' + token + '.json')
    support.save_new(capture_path, capture)
    active = copy.deepcopy(cfg)
    active.update(holders=capture['holders'], holder_capture=support.ref(capture_path), release_authorized=True,
                  preparation_status='fresh_readonly_admitted_after_recovery_producer_safe',
                  resource_assignment_status='fresh_exact_N3_gpu6_gpu7_authorized',
                  source_prepared_reference={'path': str(Path(args.source_prepared).resolve()), 'sha256': args.source_prepared_sha})
    for reference in references + [active['source_prepared_reference'], support.ref(__file__),
                                  support.ref(PRODUCER_WAITER), support.ref(capture_path), support.ref(args.manager)]:
        active['source_sha256'][reference['path']] = reference['sha256']
    active['when_safe_producer_receipts'] = references
    support.save_new(config_path, active)
    manager.c.base.validate_pair(active)
    manager.c.base.check_sources(active, config_path, support.base.sha(config_path), BASE_MANAGER_SHA)
    manager.c.check_model(active)
    fresh = support.fresh_capture(active)
    if (any(any(current[key] != expected[key] for key in support.base.IDENTITY_KEYS + ('gpu', 'uuid'))
            for current, expected in zip(fresh['holders'], capture['holders']))
            or fresh['leases'] != capture['leases']):
        raise support.WaitSafe('Consumer fresh holder/lease changed before manager exec; no signals sent')
    value = {'status': 'fresh_readonly_admitted', 'config': support.ref(config_path),
             'source_prepared_ref': active['source_prepared_reference'], 'capture': support.ref(capture_path),
             'manager': support.ref(args.manager), 'base_manager_sha256': BASE_MANAGER_SHA,
             'out': str(OUT), 'queue': active['queue'], 'producer_receipts': references,
             'checked_unix_ns': time.time_ns(), 'signals_sent_by_waiter': False, 'gpu_started_by_waiter': False}
    support.save_new(CONSUMER_POINTER, value)
    print(json.dumps(value), flush=True)
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', args.manager,
            '--config', str(config_path), '--config-sha', support.base.sha(config_path), '--manager-sha', BASE_MANAGER_SHA]
    os.execv(argv[0], argv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-consumer', action='store_true')
    parser.add_argument('--readonly-now', action='store_true')
    parser.add_argument('--source-prepared', required=True)
    parser.add_argument('--source-prepared-sha', required=True)
    parser.add_argument('--manager', required=True)
    parser.add_argument('--manager-sha', required=True)
    parser.add_argument('--self-sha', required=True)
    args = parser.parse_args()
    if (hashlib.sha256(PRODUCER_WAITER.read_bytes()).hexdigest() != PRODUCER_WAITER_SHA
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.self_sha):
        raise RuntimeError('Frozen recovery wait support/self SHA changed')
    support = load_module('recovery_producer_wait_support_v1', PRODUCER_WAITER)
    cfg = source_config(args, support)
    manager = load_module('recovery_consumer_resource_manager_v1', args.manager)
    if not args.wait_consumer and not args.readonly_now:
        parser.error('Explicit --wait-consumer or --readonly-now required')
    with (RT / 'curated_io_failure_recovery500_consumer_cpu_waiter_v1.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_reason = None
        while True:
            try:
                references = producer_gate(cfg, support)
                capture = support.fresh_capture(cfg)
            except support.WaitSafe as error:
                reason = str(error)
                if reason != last_reason:
                    print(json.dumps({'status': 'waiting_recovery_consumer', 'reason': reason,
                          'checked_unix_ns': time.time_ns(), 'signals_sent': False, 'gpu_started': False}), flush=True)
                    last_reason = reason
                if args.readonly_now:
                    return
                time.sleep(30)
                continue
            if args.readonly_now:
                print(json.dumps({'status': 'readonly_ready', 'capture': capture, 'signals_sent': False, 'gpu_started': False}), flush=True)
                return
            try:
                authorize(args, cfg, capture, references, support, manager)
            except support.WaitSafe as error:
                print(json.dumps({'status': 'waiting_fresh_admission', 'reason': str(error),
                      'signals_sent': False, 'gpu_started': False}), flush=True)
                time.sleep(30)


if __name__ == '__main__':
    main()
