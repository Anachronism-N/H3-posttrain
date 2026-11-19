"""Explicit N2/N4/N7 admission; original native TERM/pidfd/FLOCK/lease lifecycle."""
import argparse
import json
import os
import select
import sys
import time
from pathlib import Path
import curated_collaborative3_plan_v1 as p
import curated_collaborative3_build_v1 as builder
import curated_collaborative3_consumer_v1 as worker
import curated_collaborative21000_resource_manager_v2 as old

c, base = old.c, old.c.base
c.HOSTS.update(p.HOSTS)
c.ALLOWED = {mode: {node: [tuple(g) for n, m, g, _, _ in p.MEMBERS.values() if n == node and m == mode]
                   for node in p.HOSTS} for mode in ('producer', 'consumer')}
base.AUTHORIZED_PAIRS = {node: tuple(tuple(g) for n, _, g, _, _ in p.MEMBERS.values() if n == node)
                         for node in p.HOSTS}


def selected_rows(cfg):
    value = p.verify_plan(p.checked(cfg['collab3_source_plan']))
    label = cfg['collab3_member']
    if cfg['selected_sources'] != value['selected_sources'][label]:
        raise RuntimeError('Exact selected source reference changed')
    rows = p.checked(cfg['selected_sources'])
    if len(rows) != cfg['targetcount']:
        raise RuntimeError('Target count differs from selected membership')
    return rows


def selected(cfg):
    return [r['id'] for r in selected_rows(cfg)], 222939


def verify_capture(cfg):
    captured = p.checked(cfg['holder_capture'])
    age = time.time_ns() - captured['captured_unix_ns']
    if (not 0 <= age <= 1200 * 10**9 or captured['holders'] != cfg['holders']
            or captured['gpus'] != cfg['gpus'] or captured['node'] != cfg['node']
            or captured['mode'] != cfg['mode'] or captured['signals_sent'] is not False
            or len(captured['leases']) != len(cfg['gpus'])):
        raise RuntimeError('Fresh exact new-node holder capture changed')
    for gpu, lease, owner in zip(cfg['gpus'], captured['leases'], cfg['holders']):
        if (lease is not None and (not isinstance(lease, list) or len(lease) != 6
                                 or any(type(v) is not int for v in lease))):
            raise RuntimeError('Lease stat must contain exact integer nanoseconds')
        if old.lease_value(base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))) != lease:
            raise RuntimeError('Exact GPU lease changed after capture')
        base.check_holder(owner, base.identity(owner['pid']), gpu, base.uuid_for(gpu), base.apps())


def validate(cfg):
    c.validate(cfg)
    label = cfg['collab3_member']
    if label not in p.MEMBERS or cfg != builder.construct(label, cfg['holder_capture']):
        raise RuntimeError('Immutable exact new-node config differs from builder')
    if Path(cfg['queue']).exists():
        raise RuntimeError('Existing queue/repeated member dispatch refused')
    selected_rows(cfg)
    verify_capture(cfg)
    old.readonly_lock_inventory(cfg['gpus'])
    if cfg['mode'] == 'consumer':
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--member', required=True)
        parser.add_argument('--selected-sources', required=True)
        _, remaining = parser.parse_known_args(cfg['argv'][4:])
        pipeline = p.pipeline()
        os.environ.update(cfg['env'])
        args = pipeline.parser().parse_args(remaining)
        configuration = worker.configuration(pipeline, args)
        if (configuration != p.checked(cfg['consumer_params_evidence'])['new_configuration']
                or builder.digest(configuration) != cfg['consumer_params_sha256']):
            raise RuntimeError('Native prompt/images/sampling/rewrite/gate parameters changed')


def summaries(cfg, out):
    rows = selected_rows(cfg)
    results = {}
    for row in rows:
        path = out / 'sources' / row['split'] / row['id'] / (cfg['mode'] + '.terminal.json')
        terminal = p.load(path)
        if any(terminal[k] != row[k] for k in ('id', 'split', 'batch_id')):
            raise RuntimeError('Exact terminal source membership changed')
        if cfg['mode'] == 'producer':
            if terminal['status'] not in ('ready', 'failed') or terminal['source_ordinal'] != row['source_ordinal']:
                raise RuntimeError('Producer terminal ordinal/status differs')
            artifacts = [terminal[k] for k in ('plan_ref', 'roi_ref')] if terminal['status'] == 'ready' else []
        else:
            if (terminal.get('params_sha256') != cfg['consumer_params_sha256']
                    or terminal['status'] not in ('joint_candidate', 'incomplete_candidate', 'failed', 'producer_failed')):
                raise RuntimeError('Consumer terminal params/status differs')
            artifacts = [terminal['joint_candidate_ref']] if terminal.get('joint_candidate_ref') else []
        for artifact in artifacts:
            if p.sha(artifact['path']) != artifact['sha256']:
                raise RuntimeError('Exact terminal artifact SHA changed')
        results[row['id']] = p.ref(path)
    shard = cfg['source_selection']['shard']
    prefix = ('p' if cfg['mode'] == 'producer' else 'v') + str(shard) + '-*/completed.json'
    markers = []
    for path in (out / 'runs').glob(prefix):
        finished = p.load(path)
        protocol = p.load(path.with_name('protocol.json'))
        if cfg['mode'] == 'producer':
            valid = (protocol.get('collab3_producer_wrapper') == cfg['producer']
                     and protocol.get('collab3_member') == cfg['collab3_member']
                     and sum(finished.get('counts', {}).values()) == cfg['targetcount'])
        else:
            valid = (protocol.get('consumer_wrapper') == cfg['producer']
                     and finished.get('selected_total') == cfg['targetcount']
                     and finished.get('queue_sources_consumed') == cfg['targetcount'])
        if valid and protocol.get('collab3_source_plan') == cfg['collab3_source_plan']:
            markers.append(path)
    if not markers or len(results) != cfg['targetcount']:
        raise RuntimeError('Exact finite shard completion missing')
    return {str(markers[-1]): {'summary': {'stage': 'run', 'total_source_count': 222939,
            'selected_source_count': len(rows), 'processed_source_count': len(results),
            'plan_sha256': cfg['plan']['sha256']}, 'selection': {'selected_ids': [r['id'] for r in rows]},
            'exact_terminal_refs': results, 'source_failures_preserved': True, 'semantic_approval': False}}


def readonly_admission(args):
    cfg = p.load(args.config)
    if Path(args.config) != p.paths(cfg['collab3_member'])['launch']:
        raise RuntimeError('Only immutable member launch path allowed')
    validate(cfg)
    base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    c.check_model(cfg)
    locks = old.readonly_lock_inventory(cfg['gpus'])
    for owner in cfg['holders']:
        fd = os.pidfd_open(owner['pid'])
        try:
            if select.select([fd], [], [], 0)[0]:
                raise RuntimeError('Exact holder exited before admission')
        finally:
            os.close(fd)
    budget = base.headroom()
    required = base.required_budget(cfg)
    if budget['headroom'] < required:
        raise RuntimeError('Fresh headroom below native 160GiB admission budget')
    value = {'status': 'readonly_admitted', 'node': cfg['node'], 'gpus': cfg['gpus'],
             'source_plan': cfg['collab3_source_plan'], 'source_range': cfg['source_range'],
             'selected_count': cfg['targetcount'], 'config': p.ref(args.config),
             'manager': p.ref(__file__), 'base_manager_sha256': args.manager_sha,
             'readonly_locks': locks, 'budget': budget, 'required_bytes': required,
             'checked_unix_ns': time.time_ns(), 'signals_sent': False, 'gpu_started': False,
             'leases_created': False, 'queue_created': False, 'model_calls': 0}
    p.write(args.report, value)
    print(json.dumps(value), flush=True)


base.validate_pair, base.selected_sources, base.stage_summaries = validate, selected, summaries


if __name__ == '__main__':
    if '--capture-current' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--capture-current', action='store_true')
        parser.add_argument('--node', required=True)
        parser.add_argument('--gpus', nargs='+', type=int, required=True)
        parser.add_argument('--mode', required=True)
        parser.add_argument('--out', required=True)
        old.capture_current(parser.parse_args())
    elif '--readonly-admission' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--readonly-admission', action='store_true')
        parser.add_argument('--config', required=True)
        parser.add_argument('--config-sha', required=True)
        parser.add_argument('--manager-sha', required=True)
        parser.add_argument('--report', required=True)
        readonly_admission(parser.parse_args())
    else:
        base.main()
