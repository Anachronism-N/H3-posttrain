"""N1 shard3 prepare-ahead execution; reuse exact native cache handoff lifecycle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
AHEAD = RT / 'curated_consumer_prepare_ahead_v1.py'
CACHE_MANAGER = RT / 'curated_consumer_cache_resource_manager_v1.py'
CACHE_MANAGER_SHA = '31a7ee2e5868c485fee13341014e13f17d3d5a6bbfba044ef02cdb04e5cf0ca1'
PREPARE = RT / 'prepare_curated_consumer_cache_v1.py'
PREPARE_SHA = '6551215528c8d4dc51e5790a00f39bf91695e8ec341c1651abe3a643167257a5'


def module(name, path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('Frozen resource dependency differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), value.__dict__)
    return value


cache = module('prepare_ahead_native_cache_lifecycle', CACHE_MANAGER, CACHE_MANAGER_SHA)
builder = module('prepare_ahead_native_cache_builder', PREPARE, PREPARE_SHA)
original_plan = cache.plan_config
cache.MANAGER = cache.PREPARE = SELF


def plan_config(shard):
    if shard != 3:
        raise RuntimeError('Only original N1 shard3 GPU2,3 prepare-ahead admitted')
    cfg = original_plan(shard)
    cfg['argv'][3] = str(AHEAD)
    cfg['producer'] = cache.ref(AHEAD)
    cfg['phase'] = 'curated_consumer_prepare_ahead_s3_v1'
    cfg['queue'] = str(RT.parent / 'queues' / cfg['phase'])
    for path in (SELF, AHEAD, CACHE_MANAGER, PREPARE):
        cfg['source_sha256'][str(path)] = cache.c.base.sha(path)
    return cfg


def validate_semantics(cfg):
    if cfg['source_selection']['shard'] != 3:
        raise RuntimeError('Only original N1 shard3 admitted')
    old, reference = cache.original_config(3)
    if set(cfg) != set(old) | cache.NEW_FIELDS or cfg['original_consumer_config_ref'] != reference:
        raise RuntimeError('Exact original consumer fields/config required')
    for key in set(old) - cache.CHANGED:
        if cfg[key] != old[key]:
            raise RuntimeError('Original native semantic/scheduling contract changed: ' + key)
    argv = list(old['argv'])
    argv[3] = str(AHEAD)
    if (cfg['argv'] != argv or cfg['producer'] != cache.ref(AHEAD)
            or cfg['phase'] != 'curated_consumer_prepare_ahead_s3_v1'
            or cfg['queue'] != str(RT.parent / 'queues' / cfg['phase'])):
        raise RuntimeError('Only new prepare-ahead execution/isolated resource namespace may differ')
    for path, digest in old['source_sha256'].items():
        if cfg['source_sha256'].get(path) != digest:
            raise RuntimeError('Original source SHA must not be re-signed')
    required = {str(RT / name): digest for name, digest in cache.FROZEN.items()}
    required.update({str(CACHE_MANAGER): CACHE_MANAGER_SHA, str(PREPARE): PREPARE_SHA,
                     str(SELF): cache.c.base.sha(SELF), str(AHEAD): cache.c.base.sha(AHEAD)})
    for path, digest in required.items():
        if cfg['source_sha256'].get(path) != digest or cache.c.base.sha(path) != digest:
            raise RuntimeError('Frozen execution/source reference differs: ' + path)
    for key in ('resource_manager_ref', 'preparation_ref'):
        if cfg[key] != cache.ref(SELF):
            raise RuntimeError('Exact new lifecycle/build source reference required')


cache.plan_config = plan_config
cache.validate_semantics = validate_semantics


def main():
    if '--check-plan' in sys.argv or '--build' in sys.argv:
        parser = argparse.ArgumentParser()
        actions = parser.add_mutually_exclusive_group(required=True)
        actions.add_argument('--check-plan', action='store_true')
        actions.add_argument('--build', action='store_true')
        parser.add_argument('--self-sha', required=True)
        parser.add_argument('--ahead-sha', required=True)
        parser.add_argument('--transition')
        parser.add_argument('--transition-sha')
        parser.add_argument('--out')
        args = parser.parse_args()
        if cache.c.base.sha(SELF) != args.self_sha or cache.c.base.sha(AHEAD) != args.ahead_sha:
            raise RuntimeError('New execution/lifecycle SHA differs from caller freeze')
        if args.check_plan:
            if any((args.out, args.transition, args.transition_sha)):
                parser.error('Check-plan cannot construct a launch')
            cfg = plan_config(3)
            cache.check_plan(cfg)
            value = {'status': 'prepare_ahead_plan_verified', 'source_sha256': cfg['source_sha256'],
                     'gpus': cfg['gpus'], 'targetcount': cfg['targetcount'],
                     'signals_sent': False, 'GPU_started': False, 'queue_created': False}
        else:
            if not all((args.out, args.transition, args.transition_sha)):
                parser.error('Build requires exact safe transition and unique output')
            args.shard = 3
            value = builder.build(args, cache)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    else:
        cache.c.base.main()


if __name__ == '__main__':
    main()
