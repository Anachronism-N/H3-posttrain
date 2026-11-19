"""Exact isolated range admission; native verified lifecycle, no shared-output scan."""
import argparse
import importlib.util
import json
from pathlib import Path
import select
import subprocess
import sys
import time

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('curated_resource_original_v1', RT / 'curated_resource_manager.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('curated_collaborative21000_consumer_v2', RT / 'curated_collaborative21000_consumer_v2.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
c.ALLOWED = {'producer': {'N5': [(0,), (1,)]}, 'consumer': {'N3': [(6, 7)]}}
c.base.AUTHORIZED_PAIRS = {'N5': ((0,), (1,)), 'N3': ((6, 7),)}
original_validate = c.validate

PREVIOUS_PHASES = {
    ('N5', (0,)): 'curated_io_failure_recovery500_n5g0_v1',
    ('N5', (1,)): 'curated_collaborative20000_n5g1_v1',
    ('N3', (6, 7)): 'curated_io_failure_recovery500_consumer_n3g67_v1',
}


def completed_resource_gate(cfg):
    gate = cfg['completed_resource_gate']
    phase = PREVIOUS_PHASES[(cfg['node'], tuple(cfg['gpus']))]
    queue = RT.parent / 'queues' / phase
    if gate['queue'] != str(queue):
        raise RuntimeError('Resource authorization names wrong completed queue')
    references = gate['references']
    expected_paths = {str(queue / name) for name in ('admission.json', 'worker_exit.json', 'resources_safe.json')}
    expected_paths.update(str(queue / ('holder_restored_gpu%d.json' % gpu)) for gpu in cfg['gpus'])
    expected_paths.add(gate['launch_config']['path'])
    if {ref['path'] for ref in references} != expected_paths:
        raise RuntimeError('Completed resource authorization receipt inventory changed')
    for ref in references:
        if c.base.sha(ref['path']) != ref['sha256']:
            raise RuntimeError('Completed resource authorization SHA changed')
    if gate['launch_config'] not in references:
        raise RuntimeError('Completed launch config is not SHA bound')
    previous = json.loads(Path(gate['launch_config']['path']).read_text())
    admission = json.loads((queue / 'admission.json').read_text())
    finished = json.loads((queue / 'worker_exit.json').read_text())
    safe = json.loads((queue / 'resources_safe.json').read_text())
    if (previous['phase'] != phase or previous['node'] != cfg['node'] or previous['gpus'] != cfg['gpus']
            or previous['queue'] != str(queue) or admission['phase'] != phase
            or admission['config_sha256'] != gate['launch_config']['sha256']
            or admission['node'] != cfg['node'] or admission['gpus'] != cfg['gpus']
            or admission['out'] != previous['out'] or finished['phase'] != phase
            or finished['exit_code'] != 0 or finished['targetcount'] != previous['targetcount']
            or finished['input_count'] != 222939 or finished['source_unchanged'] is not True
            or safe['phase'] != phase or not all(safe.get(key) is True for key in
                ('all_holders_safe', 'locks_released', 'leases_restored', 'unknown_jobs_not_signaled'))
            or (queue / 'resources_safe.json').stat().st_mtime_ns < (queue / 'worker_exit.json').stat().st_mtime_ns):
        raise RuntimeError('Completed prior resource transaction is not safely authorized')
    for gpu, owner in zip(cfg['gpus'], cfg['holders']):
        restored = json.loads((queue / ('holder_restored_gpu%d.json' % gpu)).read_text())
        if restored['gpu'] != gpu or restored['uuid'] != owner['uuid']:
            raise RuntimeError('Restored prior GPU UUID differs from fresh capture')
        for key in c.base.IDENTITY_KEYS:
            if restored['holder_identity'][key] != owner[key]:
                raise RuntimeError('Fresh holder differs from actually restored prior holder: ' + key)


def readonly_lock_inventory(gpus):
    import os
    import stat
    text = Path('/proc/locks').read_text()
    result = []
    for gpu in gpus:
        path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
        if not path.exists():
            result.append({'gpu': gpu, 'path': str(path), 'exists': False, 'active_lock': False})
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size != 0:
            raise RuntimeError('Unsafe existing GPU lock metadata')
        key = '%02x:%02x:%d' % (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
        if any(key in line.split() for line in text.splitlines()):
            raise RuntimeError('Active kernel GPU lock: ' + str(path))
        result.append({'gpu': gpu, 'path': str(path), 'exists': True,
                       'stat': lease_value(info), 'active_lock': False})
    return result



def selected_rows(cfg):
    selection = {key: int(c.base.argv_option(cfg, '--' + key, default))
                 for key, default in (('shard', 0), ('nshard', 1), ('limit', 0))}
    expected = ({'shard': cfg['gpus'][0], 'nshard': 2, 'limit': 0}
                if cfg['mode'] == 'producer' else {'shard': 0, 'nshard': 1, 'limit': 0})
    reference = cfg['selected_sources']
    if (selection != expected or cfg['source_selection'] != selection
            or cfg['source_range'] != {'start': worker.START, 'end': worker.END}
            or cfg['input_count'] != 222939
            or cfg['targetcount'] != (5000 if cfg['mode'] == 'producer' else 10000)
            or c.base.sha(reference['path']) != reference['sha256']):
        raise RuntimeError('Explicit isolated range/count/selection changed')
    if cfg['mode'] == 'producer':
        if (int(c.base.argv_option(cfg, '--ordinal-start')) != worker.START
                or int(c.base.argv_option(cfg, '--ordinal-end')) != worker.END):
            raise RuntimeError('Producer ordinal argv changed')
    elif c.base.argv_option(cfg, '--selected-sources') != reference['path']:
        raise RuntimeError('Consumer explicit selected-sources argv changed')
    rows = worker.selected_rows(worker.load_pipeline(), cfg['manifest_summary']['path'])
    rows = [row for row in rows if row['source_ordinal'] % selection['nshard'] == selection['shard']]
    if json.loads(Path(reference['path']).read_text()) != rows:
        raise RuntimeError('Selected paths differ from original curated batches')
    return rows


def selected(cfg):
    return [row['id'] for row in selected_rows(cfg)], 222939


def lease_value(lease):
    return (None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid,
            lease.st_mode, lease.st_size, lease.st_mtime_ns])


def validate(cfg):
    original_validate(cfg)
    out, queue = Path(cfg['out']), Path(cfg['queue'])
    if (out != worker.OUT or out.resolve() != worker.OUT
            or c.base.argv_option(cfg, '--out') != str(worker.OUT)
            or queue.parent != RT.parent / 'queues'
            or not queue.name.startswith('curated_collaborative21000_')
            or 'private_ready' in cfg):
        raise RuntimeError('Independent sibling output/native ready queue binding changed')
    selected_rows(cfg)
    completed_resource_gate(cfg)
    capture = cfg['holder_capture']
    if c.base.sha(capture['path']) != capture['sha256']:
        raise RuntimeError('Fresh holder capture SHA changed')
    captured = json.loads(Path(capture['path']).read_text())
    age_ns = time.time_ns() - captured['captured_unix_ns']
    if (captured['holders'] != cfg['holders'] or captured['node'] != cfg['node']
            or captured['gpus'] != cfg['gpus'] or captured['mode'] != cfg['mode']
            or captured['signals_sent'] is not False or not 0 <= age_ns <= 1200 * 10**9):
        raise RuntimeError('Holder capture stale or outside exact authorization')
    for gpu, expected, owner in zip(cfg['gpus'], captured['leases'], cfg['holders']):
        if lease_value(c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != expected:
            raise RuntimeError('Exact nanosecond GPU lease changed before launch')
        c.base.check_holder(owner, c.base.identity(owner['pid']), gpu, c.base.uuid_for(gpu), c.base.apps())
    if cfg['mode'] == 'consumer':
        pipeline = worker.load_pipeline()
        _, remaining = argparse_consumer(cfg['argv'][4:])
        args = pipeline.parser().parse_args(remaining)
        configuration = worker.configuration(pipeline, args)
        _, _, visual = pipeline.modules(args.repo, args.ms_source)
        if visual.digest(configuration) != cfg['consumer_params_sha256']:
            raise RuntimeError('Native consumer parameter fingerprint changed')
        evidence = cfg['consumer_params_evidence']
        if (c.base.sha(evidence['path']) != evidence['sha256']
                or json.loads(Path(evidence['path']).read_text())['new_configuration'] != configuration):
            raise RuntimeError('Path-only parameter-change evidence changed')


def argparse_consumer(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--selected-sources', required=True)
    return parser.parse_known_args(argv)


def summaries(cfg, out):
    rows = selected_rows(cfg)
    results = {}
    filename = cfg['mode'] + '.terminal.json'
    for row in rows:
        path = out / 'sources' / row['split'] / row['id'] / filename
        value = json.loads(path.read_text())
        if (value['id'] != row['id'] or value['split'] != row['split']
                or value['batch_id'] != row['batch_id']):
            raise RuntimeError('Selected terminal source binding changed')
        if cfg['mode'] == 'producer':
            if value['status'] not in ('ready', 'failed') or value['source_ordinal'] != row['source_ordinal']:
                raise RuntimeError('Selected producer terminal status/ordinal changed')
            artifacts = [value[key] for key in ('plan_ref', 'roi_ref')] if value['status'] == 'ready' else []
        else:
            if (value.get('params_sha256') != cfg['consumer_params_sha256']
                    or value['status'] not in ('joint_candidate', 'incomplete_candidate', 'failed', 'producer_failed')):
                raise RuntimeError('Selected consumer terminal parameter/status changed')
            artifacts = [value['joint_candidate_ref']] if value.get('joint_candidate_ref') else []
        for artifact in artifacts:
            if c.base.sha(artifact['path']) != artifact['sha256']:
                raise RuntimeError('Selected output artifact changed')
        results[row['id']] = value
    markers = []
    pattern = 'p%d-*/completed.json' % cfg['source_selection']['shard'] if cfg['mode'] == 'producer' else 'v0-*/completed.json'
    for path in (out / 'runs').glob(pattern):
        value = json.loads(path.read_text())
        protocol = json.loads(path.with_name('protocol.json').read_text())
        if cfg['mode'] == 'producer':
            valid = (protocol.get('producer_io_publication_code') == cfg['producer']
                     and sum(value.get('counts', {}).values()) == cfg['targetcount'])
        else:
            valid = (value.get('selected_total') == 10000 and value.get('queue_sources_consumed') == 10000
                     and protocol.get('consumer_wrapper') == cfg['producer'])
        if valid:
            markers.append(path)
    if len(results) != cfg['targetcount'] or not markers:
        raise RuntimeError('Finite isolated completion receipt missing')
    return {str(markers[-1]): {'summary': {'stage': 'run', 'total_source_count': 222939,
            'selected_source_count': len(rows), 'processed_source_count': len(results),
            'plan_sha256': cfg['plan']['sha256']},
            'selection': {'selected_ids': [row['id'] for row in rows]},
            'source_failures_preserved': True, 'semantic_approval': False}}


c.selected = selected
c.base.validate_pair = validate
c.base.selected_sources = selected
c.base.stage_summaries = summaries


def capture_current(args):
    if tuple(args.gpus) not in c.ALLOWED.get(args.mode, {}).get(args.node, []):
        raise RuntimeError('Capture outside isolated exact GPU assignment')
    if c.HOSTS[args.node] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Capture host mismatch')
    owners, leases = [], []
    for gpu in args.gpus:
        uuid = c.base.uuid_for(gpu)
        active = [app for app in c.base.apps() if app['uuid'] == uuid]
        if len(active) != 1:
            raise RuntimeError('Exactly one authorized holder required')
        owner = c.base.identity(active[0]['pid'])
        owner.update(gpu=gpu, uuid=uuid)
        c.base.check_holder(owner, c.base.identity(owner['pid']), gpu, uuid, c.base.apps())
        leases.append(lease_value(c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))))
        owners.append(owner)
    value = {'node': args.node, 'mode': args.mode, 'gpus': args.gpus, 'holders': owners,
             'leases': leases, 'captured_unix_ns': time.time_ns(),
             'headroom': c.base.headroom(), 'signals_sent': False}
    with Path(args.out).open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({'path': args.out, 'sha256': c.base.sha(args.out), 'signals_sent': False}), flush=True)


def readonly_admission(args):
    cfg = json.loads(Path(args.config).read_text())
    validate(cfg)
    c.base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    c.check_model(cfg)
    locks = readonly_lock_inventory(cfg['gpus'])
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    if queue.exists() or (out.exists() and not out.is_dir()):
        raise RuntimeError('Fresh isolated queue/output admission failed')
    for owner in cfg['holders']:
        fd = c.base.os.pidfd_open(owner['pid'])
        try:
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('Exact expected holder already exited')
        finally:
            c.base.os.close(fd)
    budget = c.base.headroom()
    required = c.base.required_budget(cfg)
    if budget['headroom'] < required:
        raise RuntimeError('Readonly admission headroom below native required budget')
    value = {'status': 'readonly_admitted', 'node': cfg['node'], 'gpus': cfg['gpus'],
             'config': {'path': args.config, 'sha256': args.config_sha},
             'isolated_manager_sha256': c.base.sha(__file__), 'base_manager_sha256': args.manager_sha,
             'out': str(out), 'ready': str(out / 'ready.jsonl'),
             'source_range': cfg['source_range'], 'selected_count': cfg['targetcount'],
             'fresh_holder_capture': cfg['holder_capture'], 'completed_resource_gate': cfg['completed_resource_gate'],
             'readonly_locks': locks, 'budget': budget, 'required_bytes': required,
             'checked_unix_ns': time.time_ns(), 'signals_sent': False, 'gpu_started': False,
             'leases_created': False, 'queue_created': False, 'semantic_approval': False}
    with Path(args.report).open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    if '--capture-current' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--capture-current', action='store_true')
        parser.add_argument('--node', required=True)
        parser.add_argument('--gpus', nargs='+', type=int, required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--mode', required=True)
        capture_current(parser.parse_args())
    elif '--readonly-admission' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--readonly-admission', action='store_true')
        parser.add_argument('--config', required=True)
        parser.add_argument('--config-sha', required=True)
        parser.add_argument('--manager-sha', required=True)
        parser.add_argument('--report', required=True)
        readonly_admission(parser.parse_args())
    else:
        c.base.main()
