"""CPU plan validation and save-once N1 native-consumer cache handoff.

Never sends signals, starts a worker, edits leases, or writes a transition receipt.
The main coordinator must provide the exact completed restoration transition.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
MANAGER = RT / 'curated_consumer_cache_resource_manager_v1.py'


def load_manager(expected_sha):
    raw = MANAGER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('Cache manager SHA mismatch')
    spec = importlib.util.spec_from_file_location('prepared_cache_manager_v1', MANAGER)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(MANAGER), 'exec'), module.__dict__)
    return module


def build(args, manager):
    cfg = manager.plan_config(args.shard)
    manager.check_plan(cfg)
    manager.original_validate(cfg)
    out = Path(args.out).resolve()
    if (out.parent != RT or not out.name.startswith('curated_consumer_cache_s%d_' % args.shard)
            or out.suffix != '.json' or out.exists() or Path(cfg['queue']).exists()):
        raise RuntimeError('Unique new runtime launch and unused fixed cache queue required')
    transition_ref = {'path': str(Path(args.transition).resolve()), 'sha256': args.transition_sha}
    restores, references = manager.handoff_gate(cfg, transition_ref, require_parent=True)
    capture = manager.fresh_capture(cfg, restores)
    cfg.update(holders=capture['holders'], fresh_capture=capture,
               handoff={'transition_ref': transition_ref, 'receipt_refs': references})
    for reference in references:
        cfg['source_sha256'][reference['path']] = reference['sha256']
    # Recheck the complete sources/model before the final fresh admission capture.
    manager.check_plan(cfg)
    manager.validate(cfg)
    if Path(cfg['queue']).exists():
        raise RuntimeError('Fixed cache queue appeared; no launch written')
    with out.open('x') as stream:
        json.dump(cfg, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return {'status': 'fresh_launch_saved_once', 'config': manager.ref(out),
            'resource_manager_ref': cfg['resource_manager_ref'], 'preparation_ref': cfg['preparation_ref'],
            'queue': cfg['queue'], 'base_manager_sha256': manager.BASE_SHA,
            'signals_sent': False, 'gpu_started': False,
            'manager_argv': ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER),
                '--config', str(out), '--config-sha', manager.c.base.sha(out),
                '--manager-sha', manager.BASE_SHA]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--build', action='store_true')
    parser.add_argument('--shard', required=True, type=int, choices=(2, 3))
    parser.add_argument('--self-sha', required=True)
    parser.add_argument('--cache-manager-sha', required=True)
    parser.add_argument('--transition')
    parser.add_argument('--transition-sha')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.build and not all((args.transition, args.transition_sha, args.out)):
        parser.error('--build requires --transition, --transition-sha and --out')
    if args.check_plan and any((args.transition, args.transition_sha, args.out)):
        parser.error('--check-plan does not construct or write a launch')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.self_sha:
        raise RuntimeError('Preparation source SHA mismatch')
    manager = load_manager(args.cache_manager_sha)
    if args.check_plan:
        cfg = manager.plan_config(args.shard)
        manager.check_plan(cfg)
        value = {'status': 'cpu_plan_sources_model_verified', 'shard': args.shard,
                 'original_consumer_config_ref': cfg['original_consumer_config_ref'],
                 'resource_manager_ref': cfg['resource_manager_ref'], 'preparation_ref': cfg['preparation_ref'],
                 'queue': cfg['queue'], 'launch_written': False, 'signals_sent': False,
                 'gpu_started': False, 'gpu_lock_or_lease_accessed': False}
    else:
        value = build(args, manager)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'cache_prepare_refused', 'error': str(error),
                          'signals_sent': False, 'gpu_started': False}), file=sys.stderr, flush=True)
        sys.exit(2)
