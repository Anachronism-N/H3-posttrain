"""Independent fixed N2p0 admission using the unmodified native resource lifecycle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import caption_producer_cache_handoff_v1 as hand
import caption_producer_cached_worker_v1 as worker

base = hand.native_base()
# Import producer ROI verification independently; no original module/file mutation.
spec = importlib.util.spec_from_file_location('_caption_cache_original_curated', hand.RT / 'curated_resource_manager.py')
curated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curated)
model_globals = dict(curated.check_model.__globals__, base=base)
producer_model_check = types.FunctionType(curated.check_model.__code__, model_globals,
                                         curated.check_model.__name__, curated.check_model.__defaults__)


def selected(cfg):
    old = hand.load(cfg['old_config']['path'])
    rows = hand.validate_meta(hand.load(cfg['remaining_meta']['path']), hand.load(old['selected_sources']['path']))
    if len(rows) != cfg['targetcount'] or cfg['input_count'] != 222939:
        raise RuntimeError('Exact remaining count/full manifest input count changed')
    return [row['id'] for row in rows], 222939


def verify_capture(cfg, proof):
    captured = hand.checked(cfg['holder_capture'])
    age = time.time_ns() - captured['captured_unix_ns']
    if (not 0 <= age <= 1200 * 10**9 or captured['node'] != 'N2' or captured['gpus'] != [0]
            or captured['mode'] != 'producer' or captured['holders'] != cfg['holders']
            or captured['signals_sent'] is not False or len(captured['leases']) != 1):
        raise RuntimeError('Fresh exact restored-holder capture changed')
    owner, lease, _ = hand.safe_gate(base, hand.checked(cfg['old_config']), proof)
    if owner != cfg['holders'][0]:
        raise RuntimeError('New holder is not the actual latest original restored holder')
    lease_value = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns]
    if captured['leases'][0] != lease_value:
        raise RuntimeError('Exact native GPU lease changed since handoff capture')


def validate(cfg):
    rows, proof = worker.contract(cfg)
    hand.fixed_node(cfg)
    if (cfg.get('release_authorized') is not True or len(cfg['holders']) != 1
            or cfg['holders'][0]['gpu'] != 0 or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '0'
            or 'DWPOSE_ALLOW_CPU' in cfg['env'] or not Path(cfg['cwd']).is_absolute()
            or cfg['argv'] != hand.checked(cfg['old_config'])['argv'][:3] +
                [cfg['producer']['path'], '--config', str(hand.RT / (hand.PHASE + '_launch.json'))]
            or cfg['manifest_summary'] != hand.checked(cfg['old_config'])['manifest_summary']
            or cfg['original_full_manifest'] != hand.checked(cfg['old_config'])['original_full_manifest']
            or cfg['modelref'] != hand.checked(cfg['old_config'])['modelref']):
        raise RuntimeError('Actual fixed cached launch/ROI/manifest binding changed')
    if Path(cfg['queue']).exists() or Path(cfg['out']).exists():
        raise RuntimeError('Independent queue/output already claimed; repeated launch refused')
    if (type(cfg['targetcount']) is not int or cfg['targetcount'] <= 0
            or not cfg['phase'] or not all(c.isalnum() or c in '_-' for c in cfg['phase'])):
        raise RuntimeError('Invalid finite remaining stage count/phase')
    verify_capture(cfg, proof)
    available, blocked = hand.split_remaining(rows, hand.SOURCE_ROOT)
    if blocked or len(available) != len(rows):
        raise RuntimeError('Started source after handoff; source ownership not available')


def summaries(cfg, out):
    path = Path(out) / 'run.summary.json'
    value = hand.load(path)
    ids, total = selected(cfg)
    summary = value['summary']
    if (summary != {'stage': 'run', 'plan_sha256': cfg['plan']['sha256'],
                   'total_source_count': total, 'selected_source_count': cfg['targetcount'],
                   'processed_source_count': cfg['targetcount']}
            or value['selection']['selected_ids'] != ids or value['source_handoff'] != cfg['source_handoff']
            or value['cache_code'] != cfg['cache_code']):
        raise RuntimeError('Independent execution summary differs from exact remaining selection')
    run = Path(value['native_result']['run_root'])
    if run.parent != hand.SOURCE_ROOT / 'runs' or not run.name.startswith('p0-'):
        raise RuntimeError('Native producer run escaped shared source root')
    protocol = hand.load(run / 'protocol.json')
    if (protocol.get('caption_producer_cached_worker') != cfg['producer']
            or protocol.get('caption_producer_cache_code') != cfg['cache_code']
            or protocol.get('caption_producer_source_handoff') != cfg['source_handoff']
            or protocol.get('caption_producer_source_handoff_code') != cfg['handoff_code']
            or protocol.get('caption_producer_remaining_meta') != cfg['remaining_meta']
            or protocol.get('pipeline_code') != hand.ref(hand.RT / 'source_curated_producer_v1/curated_pipeline.py')
            or protocol.get('caption_producer_execution_root') != str(out)):
        raise RuntimeError('Real native protocol/cache/source handoff bindings changed')
    completed = hand.load(run / 'completed.json')
    if (completed['counts'] != value['native_result']['counts']
            or sum(completed['counts'].values()) != cfg['targetcount'] or completed['total'] != total
            or completed['counts'].get('resumed_terminal', 0)):
        raise RuntimeError('Native run not completed with exclusively unstarted sources')
    terminals = {}
    for row in hand.load(cfg['remaining_meta']['path']):
        terminal_path = hand.SOURCE_ROOT / 'sources' / row['split'] / row['id'] / 'producer.terminal.json'
        terminal = hand.load(terminal_path)
        if (any(terminal.get(k) != row[k] for k in hand.KEYS)
                or terminal['status'] not in ('ready', 'failed')):
            raise RuntimeError('Current remaining source terminal membership/status changed')
        if terminal['status'] == 'ready':
            expected = {'plan_ref': terminal_path.parent / 'bar_plan.json',
                        'roi_ref': hand.SOURCE_ROOT / 'roi' / row['id'] / 'roi_manifest.json'}
            for key in ('plan_ref', 'roi_ref'):
                artifact = terminal[key]
                if Path(artifact['path']) != expected[key].resolve():
                    raise RuntimeError('Current source terminal artifact escaped original source output')
                if hand.sha(artifact['path']) != artifact['sha256']:
                    raise RuntimeError('Current source terminal artifact changed')
        terminals[row['id']] = hand.ref(terminal_path)
    value['exact_terminal_refs'] = terminals
    value['native_protocol_ref'] = hand.ref(run / 'protocol.json')
    value['native_completed_ref'] = hand.ref(run / 'completed.json')
    return {str(path): value}


# Hooks live exclusively on this independently imported native base instance.
base.validate_pair = validate
base.selected_sources = selected
base.check_model = producer_model_check
base.stage_summaries = summaries


def readonly_admission(args):
    if hand.sha(args.config) != args.config_sha or args.manager_sha != hand.BASE_SHA:
        raise RuntimeError('Original native manager/config SHA mismatch')
    cfg = hand.load(args.config)
    if hand.ref(__file__) != cfg['resource_manager']:
        raise RuntimeError('Actual independent cached manager changed')
    validate(cfg)
    base.check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    base.check_model(cfg)
    budget = base.headroom()
    if budget['headroom'] < base.required_budget(cfg):
        raise RuntimeError('Fresh native 160GiB admission gate fails')
    value = {'status': 'readonly_admitted', 'node': 'N2', 'gpus': [0],
             'config': hand.ref(args.config), 'manager': hand.ref(__file__),
             'base_manager': hand.ref(base.__file__), 'remaining_meta': cfg['remaining_meta'],
             'source_handoff': cfg['source_handoff'], 'cache_code': cfg['cache_code'],
             'selected_count': cfg['targetcount'], 'input_count': 222939,
             'budget': budget, 'required_bytes': base.required_budget(cfg),
             'checked_unix_ns': time.time_ns(), 'signals_sent': False, 'gpu_started': False,
             'queue_created': False, 'leases_created': False, 'model_calls': 0}
    hand.write_once(args.report, value)
    return value


def main():
    if '--readonly-admission' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--readonly-admission', action='store_true')
        parser.add_argument('--config', required=True)
        parser.add_argument('--config-sha', required=True)
        parser.add_argument('--manager-sha', required=True)
        parser.add_argument('--report', required=True)
        print(json.dumps(readonly_admission(parser.parse_args()), ensure_ascii=False), flush=True)
    else:
        base.main()


if __name__ == '__main__':
    main()
