"""New execution identity only; original N5G7 pidfd/lease/restore transaction."""
import argparse
import json
import os
from pathlib import Path
import select
import stat
import time

from main_review_watch_v8 import RT, ROOT, module, derive, need, checked, Preparation

OLD_MANAGER_SHA = '26ad88563d4875e22ac51ac5db7b5b0f28eac65ad0ef780c8770a1090c577b8a'
v7 = module('private_frozen_resource_v7_for_v8', RT / 'main_review_resource_manager_v7.py', OLD_MANAGER_SHA)
base = v7.base
BASE_MANAGER_SHA = v7.BASE_MANAGER_SHA
check_model = v7.check_model
_validate = derive(v7.validate, [
    ("'main_review_watch_v7.py'", "'main_review_watch_v8.py'"),
    ("cfg['phase'] != 'main_review_watch_v7'", "cfg['phase'] != 'main_review_watch_v8'"),
    ("startswith('main_review_watch_v7-')", "startswith('main_review_watch_v8-')")], vars(v7))
_summaries = derive(v7.summaries, [("'curated_independent_review_watch_v7'", "'curated_independent_review_watch_v8'")], vars(v7))


def validate(cfg):
    _validate(cfg)
    binding = checked(cfg['source_binding'])
    proof = binding.get('offset2053650')
    need(proof is not None, 'Frozen exact new incident preparation required')
    refs = [proof['snapshot_ref'], proof['collector_ref'], proof['raw_ref'], *proof['record_refs'], *proof['receipt_refs']]
    refs += [b['raw_prefix_ref'] for b in proof['mount_bindings'].values()]
    need(cfg['source_sha256'].get(str(RT / 'main_review_resource_manager_v8.py')) == base.sha(__file__), 'New execution manager SHA not frozen')
    need(all(cfg['source_sha256'].get(r['path']) == r['sha256'] for r in refs), 'Full immutable new proof refs must be source frozen')
    Preparation(proof, base.argv_option(cfg, '--results-root'), base.argv_option(cfg, '--manifest'))


def release_holder(entry):
    # Immediate owned fd/lease/path and lock proof; no new TTL or timestamp substitution.
    need(not select.select([entry['fd']], [], [], 0)[0], 'Owned holder pidfd exited before exact signal')
    st = os.fstat(entry['leasefd'])
    current = entry['reserve'].lstat()
    need((st.st_dev, st.st_ino) == entry['lease_inode'] == (current.st_dev, current.st_ino)
         and stat.S_ISREG(st.st_mode) and current.st_uid == st.st_uid == os.getuid()
         and current.st_size == st.st_size == 0 and current.st_mtime_ns == st.st_mtime_ns,
         'Current owned lease inode/FD/heartbeat differs before signal')
    need(0 <= time.time() - current.st_mtime <= 20, 'Actual owned heartbeat is not current before signal')
    lock = Path('/tmp/ms12k_gpu_7.lock')
    lock_st = lock.lstat()
    need(stat.S_ISREG(lock_st.st_mode) and lock_st.st_uid == os.getuid() and lock_st.st_size == 0, 'Current owned lock namespace differs')
    held = False
    for fd in Path('/proc/self/fd').iterdir():
        try:
            f = os.fstat(int(fd.name))
        except OSError:
            continue
        if (f.st_dev, f.st_ino) == (lock_st.st_dev, lock_st.st_ino):
            info = Path('/proc/self/fdinfo', fd.name).read_text()
            held |= any('FLOCK' in line and 'WRITE' in line for line in info.splitlines() if line.startswith('lock:'))
    need(held, 'Current owned FLOCK descriptor proof missing before signal')
    base.check_holder(entry['expected'], base.identity(entry['expected']['pid']), 7, base.uuid_for(7), base.apps())
    need(base.headroom()['headroom'] >= base.SAFETY + base.LOAD, 'Current headroom below original signal admission budget')
    _release(entry)


_release = base.release_holder
base.validate_pair = validate
base.check_model = check_model
base.stage_summaries = _summaries
base.release_holder = release_holder
base.AUTHORIZED_PAIRS = {'N5': ((7,),)}


def main():
    ap = argparse.ArgumentParser()
    for name in ('config', 'config-sha', 'manager-sha'):
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    need(args.manager_sha == BASE_MANAGER_SHA, 'Original base manager SHA required')
    cfg = json.loads(Path(args.config).read_text())
    need(cfg.get('release_authorized') is True, 'Main explicit current holder release authorization required')
    proof = json.loads(Path(cfg['source_only_preflight_receipt']).read_text())
    need(proof['schema'] == 'main_review_v8_source_only_v1'
         and proof['config'] == {'path': args.config, 'sha256': args.config_sha}
         and proof['wrapper_sha256'] == base.sha(__file__) and proof['base_manager_sha256'] == args.manager_sha
         and proof['holder_release_performed'] is False and proof['GPU_model_loaded'] is False,
         'Frozen real CPU source-only preflight required')
    base.main()


if __name__ == '__main__':
    main()
