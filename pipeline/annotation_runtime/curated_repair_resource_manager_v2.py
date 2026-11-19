"""Explicit N6 physical0,1 resident repair admission with fresh holder rechecks."""
import importlib.util
import json
from pathlib import Path
import subprocess

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resident_repair_resource_v2', RT / 'resource_manager.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_validate = base.validate_pair
original_lease_snapshot = base.lease_snapshot
lease_expectations = None
LEASE_KEYS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def checked_lease_snapshot(path):
    old = original_lease_snapshot(path)
    if lease_expectations is None or str(path) not in lease_expectations:
        raise RuntimeError('missing fresh admission lease snapshot')
    expected = lease_expectations[str(path)]
    current = None if old is None else {key: getattr(old, key) for key in LEASE_KEYS}
    if current != expected:
        raise RuntimeError('lease differs from nanosecond admission snapshot')
    return old


def validate(cfg):
    global lease_expectations
    if cfg['node'] != 'N6' or cfg['gpus'] != [0, 1]:
        raise RuntimeError('resident repair only admits N6 physical0,1')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('resident repair executing on wrong physical node')
    original_validate(cfg)
    if cfg['phase'] != 'curated_repair_resident_v2':
        raise RuntimeError('repair v2 requires a separate resource transaction phase')
    if cfg.get('global_revision') != 'repair-20261025-v1':
        raise RuntimeError('repair v2 must reuse the existing attempt revision')
    if Path(cfg['out']).resolve() != RT.parent / 'curated222939_repair_v1':
        raise RuntimeError('repair v2 must retain the original attempt ledger')
    if Path(cfg['queue']).resolve() != RT.parent / 'queues' / 'curated_repair_resident_n6_g01_v2':
        raise RuntimeError('repair v2 requires a separate resource receipt queue')
    lease_expectations = cfg['lease_snapshot_ns']
    if set(lease_expectations) != {'/tmp/od_reserve_0', '/tmp/od_reserve_1'}:
        raise RuntimeError('exact N6 physical0,1 lease snapshots required')
    for expected in lease_expectations.values():
        if expected is not None and (set(expected) != set(LEASE_KEYS) or
                any(type(expected[key]) is not int for key in LEASE_KEYS)):
            raise RuntimeError('lease snapshot fields must be exact integer nanoseconds/stat values')
    if Path(cfg['producer']['path']).resolve() != RT / 'curated_repair_watch_v2.py':
        raise RuntimeError('resident repair requires independently frozen watcher v2')
    if base.argv_option(cfg, '--repair-source') != str(RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py'):
        raise RuntimeError('resident repair requires null-safe minimal frozen repair source')
    if '--recover-only' in cfg['argv']:
        raise RuntimeError('full resident manager must not dispatch CPU recovery mode')
    if '--watch' not in cfg['argv'] or '--run-job' in cfg['argv']:
        raise RuntimeError('resident queue watch required, never single RAQ run-job')
    if int(base.argv_option(cfg, '--total-source-count', '0')) != cfg['targetcount']:
        raise RuntimeError('resident full-source termination bound differs from admission')
    if base.argv_option(cfg, '--max-jobs', '0') != '0':
        raise RuntimeError('full curated watcher must not impose a source subset')
    if Path(base.argv_option(cfg, '--out')).resolve() != Path(cfg['out']).resolve():
        raise RuntimeError('resident attempt ledger differs from admitted output')
    if base.argv_option(cfg, '--global-revision') != cfg['global_revision']:
        raise RuntimeError('resident global revision differs from admission')


def summaries(cfg, out):
    path = out / 'resident.summary.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get('schema') != 'curated_repair_watch_v2' or value.get('global_revision') != cfg['global_revision']:
        raise RuntimeError('resident summary binding differs')
    if value.get('queue_drained') is not True:
        raise RuntimeError('resident worker exited before full collector drained')
    if value['source_count'] > cfg['targetcount']:
        raise RuntimeError('resident repair exceeds full curated count')
    return {str(path): {'summary': value, 'semantic_approval': False,
                        'independent_review_required': True}}


base.AUTHORIZED_PAIRS = {'N6': ((0, 1),)}
base.validate_pair = validate
base.lease_snapshot = checked_lease_snapshot
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
