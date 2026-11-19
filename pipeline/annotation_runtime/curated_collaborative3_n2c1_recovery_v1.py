"""Fresh N2/GPU4,5 transaction; preserve native parameters and existing terminals."""
import argparse
import copy
import hashlib
import json
import os
import select
import shlex
import sys
from pathlib import Path
import curated_collaborative3_plan_v1 as p
import curated_collaborative3_build_v1 as builder
import curated_collaborative3_consumer_v1 as worker
import curated_collaborative3_resource_manager_v1 as manager

ENTRY = Path(__file__).resolve()
ORIGINAL = p.paths('n2c1')['launch']
ORIGINAL_SHA = '91e13942f44d96dc5c4a9507603a527048aabb8bf47fb05c12814249b13546bf'
PHASE = 'curated_collaborative3_n2c1_recovery_v1'
LAUNCH = p.RT / (PHASE + '_launch.json')
QUEUE = p.RT.parent / 'queues' / PHASE
base = manager.base


def native_digest(value):
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def prior_references():
    original = p.checked({'path': str(ORIGINAL), 'sha256': ORIGINAL_SHA})
    queue = Path(original['queue'])
    names = ['admission.json', 'failure.json', 'resources_safe.json',
             'holder_restored_gpu4.json', 'holder_restored_gpu5.json']
    return [p.ref(ORIGINAL)] + [p.ref(queue / name) for name in names]


def prior_gate(cfg):
    references = cfg['recovery_references']
    if references != prior_references():
        raise RuntimeError('Prior exact safe transaction references changed')
    original = p.checked(references[0])
    queue = Path(original['queue'])
    admission, failure, safe = [p.load(queue / name) for name in
                               ('admission.json', 'failure.json', 'resources_safe.json')]
    if (admission['config_sha256'] != ORIGINAL_SHA or admission['node'] != 'N2'
            or admission['gpus'] != [4, 5] or admission['out'] != cfg['out']
            or failure['phase'] != original['phase'] or failure['worker_started'] is not True
            or safe['phase'] != original['phase'] or any(safe.get(k) is not True for k in
                ('all_holders_safe', 'locks_released', 'leases_restored', 'unknown_jobs_not_signaled'))
            or (queue / 'resources_safe.json').stat().st_mtime_ns < (queue / 'failure.json').stat().st_mtime_ns):
        raise RuntimeError('Prior failed transaction not safely restored')
    for gpu, owner in zip(cfg['gpus'], cfg['holders']):
        restored = p.load(queue / ('holder_restored_gpu%d.json' % gpu))
        if restored['gpu'] != gpu or restored['uuid'] != owner['uuid'] or any(
                restored['holder_identity'][k] != owner[k] for k in base.IDENTITY_KEYS):
            raise RuntimeError('Fresh capture differs from prior restored exact holder')


def construct(capture_ref):
    original = p.checked({'path': str(ORIGINAL), 'sha256': ORIGINAL_SHA})
    capture = p.checked(capture_ref)
    import time
    if (capture['node'] != 'N2' or capture['mode'] != 'consumer' or capture['gpus'] != [4, 5]
            or capture['signals_sent'] is not False or len(capture['holders']) != 2
            or len(capture['leases']) != 2
            or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9):
        raise RuntimeError('Exact fresh N2 GPU4/5 capture required within 20 minutes')
    cfg = copy.deepcopy(original)
    cfg.update(phase=PHASE, queue=str(QUEUE), holders=capture['holders'], holder_capture=capture_ref,
               producer=p.ref(ENTRY), resource_manager=p.ref(ENTRY),
               native_consumer_params_sha256=native_digest(p.checked(cfg['consumer_params_evidence'])['new_configuration']),
               recovery_references=prior_references())
    cfg['argv'][3] = str(ENTRY)
    cfg['source_sha256'].pop(original['holder_capture']['path'])
    for reference in [p.ref(ENTRY), capture_ref, *cfg['recovery_references']]:
        cfg['source_sha256'][reference['path']] = reference['sha256']
    prior_gate(cfg)
    return cfg


def selected_rows(cfg):
    plan = p.verify_plan(p.checked(cfg['collab3_source_plan']), verify_sources=False)
    if (cfg['collab3_member'] != 'n2c1' or cfg['selected_sources'] != plan['selected_sources']['n2c1']
            or cfg['source_range'] != {'start': 32000, 'end': 62000}
            or cfg['source_selection'] != {'shard': 1, 'nshard': 9, 'limit': 0}
            or cfg['targetcount'] != 3333 or cfg['input_count'] != 222939):
        raise RuntimeError('Original exact N2 consumer assignment changed')
    rows = p.checked(cfg['selected_sources'])
    if len(rows) != 3333 or any(not 32000 <= r['source_ordinal'] < 62000
                              or r['source_ordinal'] % 9 != 1 for r in rows):
        raise RuntimeError('Exact selected source membership changed')
    return rows


def validate(cfg):
    manager.c.validate(cfg)
    if cfg != construct(cfg['holder_capture']):
        raise RuntimeError('Exact fresh recovery config differs from builder')
    if QUEUE.exists():
        raise RuntimeError('Existing fresh queue/repeated recovery refused')
    selected_rows(cfg)
    manager.verify_capture(cfg)
    manager.old.readonly_lock_inventory(cfg['gpus'])
    pipeline = p.pipeline()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--member', required=True)
    parser.add_argument('--selected-sources', required=True)
    _, remaining = parser.parse_known_args(cfg['argv'][4:])
    os.environ.update(cfg['env'])
    args = pipeline.parser().parse_args(remaining)
    configuration = worker.configuration(pipeline, args)
    evidence = p.checked(cfg['consumer_params_evidence'])['new_configuration']
    _, _, visual = pipeline.modules(args.repo, args.ms_source)
    if (configuration != evidence or builder.digest(configuration) != cfg['consumer_params_sha256']
            or visual.digest(configuration) != cfg['native_consumer_params_sha256']):
        raise RuntimeError('Native prompt/images/sampling/rewrite/gates changed')


def install_consumer(pipeline):
    source = worker.install(pipeline)
    anchor = '                hint = json.loads(line)'
    replacement = """                try:
                    hint = json.loads(line)
                except json.JSONDecodeError:
                    key = (str(queue), begin)
                    ready_failures[key] = ready_failures.get(key, 0) + 1
                    if ready_failures[key] >= 5:
                        raise SharedFailure('Ready JSON remained invalid after five reads: %s:%d' % key)
                    position = begin
                    positions[queue_index] = position
                    break"""
    if source.count(anchor) != 1 or source.count('    position = 0') != 1:
        raise RuntimeError('Native ready retry patch anchor changed')
    source = source.replace('    position = 0', '    ready_failures = {}\n    position = 0')
    source = source.replace(anchor, replacement)
    pipeline.CONSUMER_WRAPPER = str(ENTRY)
    exec(compile(source, str(ENTRY), 'exec'), pipeline.__dict__)
    return source


def summaries(cfg, out):
    native_cfg = {**cfg, 'consumer_params_sha256': cfg['native_consumer_params_sha256']}
    return manager.summaries(native_cfg, out)


def readonly_admission(args):
    if Path(args.config) != LAUNCH:
        raise RuntimeError('Only fresh immutable recovery launch admitted')
    cfg = p.load(args.config)
    validate(cfg)
    base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    manager.c.check_model(cfg)
    locks = manager.old.readonly_lock_inventory(cfg['gpus'])
    for owner in cfg['holders']:
        fd = os.pidfd_open(owner['pid'])
        try:
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('Exact restored holder exited before admission')
        finally:
            os.close(fd)
    budget = base.headroom()
    required = base.required_budget(cfg)
    if budget['headroom'] < required:
        raise RuntimeError('Fresh headroom below native 160GiB admission budget')
    report = {'status': 'readonly_admitted', 'node': 'N2', 'gpus': [4, 5],
              'config': p.ref(args.config), 'manager': p.ref(ENTRY), 'budget': budget,
              'required_bytes': required, 'reuse_existing_results': True, 'signals_sent': False,
              'gpu_started': False, 'leases_created': False, 'queue_created': False, 'model_calls': 0}
    p.write(args.report, report)
    return report


def build(capture):
    if QUEUE.exists() or LAUNCH.exists():
        raise RuntimeError('Repeated recovery/existing fresh queue or launch refused')
    cfg = construct(p.ref(capture))
    launch = p.write(LAUNCH, cfg)
    command = [cfg['argv'][0], '-B', '-u', str(ENTRY), '--config', launch['path'],
               '--config-sha', launch['sha256'], '--manager-sha', cfg['base_manager_sha256']]
    return {'launch': launch, 'command': shlex.join(command),
            'readonly_command': shlex.join(command + ['--readonly-admission', '--report',
                                                      str(p.RT / (PHASE + '_readonly.json'))]),
            'out': cfg['out'], 'targetcount': cfg['targetcount'], 'gpu_started': False, 'signals_sent': False}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'consumer':
        original_install = worker.install
        # The wrapper's original install is called by our install once, then restored.
        def install(pipeline):
            worker.install = original_install
            return install_consumer(pipeline)
        worker.install = install
        return worker.main(sys.argv[1:])
    manager.selected_rows = selected_rows
    base.validate_pair, base.stage_summaries = validate, summaries
    if '--build' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--build', action='store_true')
        parser.add_argument('--capture', required=True)
        return build(parser.parse_args().capture)
    if '--readonly-admission' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--readonly-admission', action='store_true')
        for name in ('config', 'config-sha', 'manager-sha', 'report'):
            parser.add_argument('--' + name, required=True)
        return readonly_admission(parser.parse_args())
    return base.main()


if __name__ == '__main__':
    result = main()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False), flush=True)
