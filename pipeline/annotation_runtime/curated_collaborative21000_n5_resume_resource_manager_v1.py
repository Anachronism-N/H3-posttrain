"""Exact COLLAB2 N5 producer resume; unchanged native TERM/lock/lease lifecycle."""
import argparse
import sys
import prepare_curated_collaborative21000_n5_resume_v1 as c

old, base = c.old, c.base
original_snapshot = base.lease_snapshot
active_config = None


def verify_capture_leases(cfg):
    capture = c.finite.checked(cfg['holder_capture'])
    if (type(capture.get('captured_unix_ns')) is not int or len(capture.get('leases', [])) != 1
            or capture['holders'] != cfg['holders'] or capture['gpus'] != cfg['gpus']
            or capture['node'] != 'N5' or capture['mode'] != 'producer' or capture['signals_sent'] is not False):
        raise RuntimeError('exact fresh N5 producer capture required')
    for gpu, value in zip(cfg['gpus'], capture['leases']):
        if value is not None and (not isinstance(value, list) or len(value) != 6 or any(type(v) is not int for v in value)):
            raise RuntimeError('exact integer nanosecond lease stat required')
        current = original_snapshot(c.Path('/tmp/od_reserve_%d' % gpu))
        if old.lease_value(current) != value:
            raise RuntimeError('exact nanosecond lease changed since fresh resume capture')


def validate(cfg):
    global active_config
    c.verify_resume_config(cfg)
    verify_capture_leases(cfg)
    active_config = cfg
    old.validate(cfg)
    old.readonly_lock_inventory(cfg['gpus'])


def lease_snapshot(path):
    snapshot = original_snapshot(path)
    if active_config is None or str(path) != '/tmp/od_reserve_%d' % active_config['gpus'][0]:
        raise RuntimeError('lease snapshot outside exact active resume GPU')
    captured = c.finite.checked(active_config['holder_capture'])
    if old.lease_value(snapshot) != captured['leases'][0]:
        raise RuntimeError('exact integer nanosecond lease changed before native admission')
    return snapshot


old.completed_resource_gate = c.verify_resume_config
old.__file__ = str(c.MANAGER)
base.validate_pair = validate
base.lease_snapshot = lease_snapshot


if __name__ == '__main__':
    if '--readonly-admission' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--readonly-admission', action='store_true')
        parser.add_argument('--config', required=True)
        parser.add_argument('--config-sha', required=True)
        parser.add_argument('--manager-sha', required=True)
        parser.add_argument('--report', required=True)
        args = parser.parse_args()
        cfg = c.finite.load(args.config)
        if c.Path(args.config) != c.paths(cfg['gpus'][0])['launch']:
            raise RuntimeError('only immutable exact resume config path allowed')
        validate(cfg)
        old.readonly_admission(args)
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', required=True)
        args, _ = parser.parse_known_args()
        cfg = c.finite.load(args.config)
        if c.Path(args.config) != c.paths(cfg['gpus'][0])['launch']:
            raise RuntimeError('only immutable exact resume config path allowed')
        base.main()
