"""Explicit v2 execution identity over frozen v1; no renewed preflight authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TEST = RT / 'test_curated_coarse_fixed3_managed_primary_v2.py'
V1 = RT / 'curated_coarse_fixed3_managed_primary_v1.py'
V1_TEST = RT / 'test_curated_coarse_fixed3_managed_primary_v1.py'
V1_SHA = 'ad19fe98c85dfaf053cdea9aa0f031b8b2e72cd7b3da28ec65b3384181df9348'
V1_TEST_SHA = 'd18d0b3c41af8dc494c143027f3c102f989610eb670c0e243484fd90c6792fde'
PREFLIGHT = RT / 'curated_coarse_fixed3_primary_preflight_v1.json'
PREFLIGHT_SHA = '8037b22c43c834b7ffacc04df34d8615403c32f563deb11599a2d1c1475fe270'
OLD_LAUNCH = RT / 'curated_coarse_fixed3_n6g01_launch_v1.json'
OLD_LAUNCH_SHA = '68c89e753511c57290bb3d03d30895d1fe5ad6e202d09f70d76f04fb2db34132'
OLD_QUEUE = RT.parent / 'queues/curated_coarse_fixed3_n6g01_v1'
PINNED = {str(V1): V1_SHA, str(V1_TEST): V1_TEST_SHA,
          str(PREFLIGHT): PREFLIGHT_SHA, str(OLD_LAUNCH): OLD_LAUNCH_SHA}

# This exact frozen-source transform is private, never edits/relabels v1 on disk.
# Only execution paths, the pre-fresh initializer, and safe restore routing change.
PATCHES = [
    ('test_curated_coarse_fixed3_managed_primary_v1.py', 'test_curated_coarse_fixed3_managed_primary_v2.py', 1),
    ('queues/curated_coarse_fixed3_n6g01_v1', 'queues/curated_coarse_fixed3_n6g01_v2', 1),
    ('curated_coarse_fixed3_n6g01_launch_v1.json', 'curated_coarse_fixed3_n6g01_launch_v2.json', 1),
    ('queues/curated_coarse_fixed3_primary_restore_v1', 'queues/curated_coarse_fixed3_primary_restore_v2', 1),
    ('curated_coarse_fixed3_primary_restore_launch_v1.json', 'curated_coarse_fixed3_primary_restore_launch_v2.json', 1),
    ("PHASE = 'curated_coarse_fixed3_n6g01_v1'", "PHASE = 'curated_coarse_fixed3_n6g01_v2'", 1),
    ('curated_coarse_fixed3_primary_restore_capture_v1.json', 'curated_coarse_fixed3_primary_restore_capture_v2.json', 1),
    ('curated_coarse_fixed3_n6_capture_v1.json', 'curated_coarse_fixed3_n6_capture_v2.json', 1),
    ('curated_coarse_fixed3_primary_restore_prior_v1.json', 'curated_coarse_fixed3_primary_restore_prior_v2.json', 1),
    ('curated_coarse_fixed3_n6_prior_v1.json', 'curated_coarse_fixed3_n6_prior_v2.json', 1),
    ("        fresh = c.fresh_pair(base, previous)\n        if fresh['lease_snapshot_ns']", "        active.update(cfg=cfg, contract=contract)\n        fresh = c.fresh_pair(base, previous)\n        if fresh['lease_snapshot_ns']", 1),
    ("previous = restored_evidence(base, QUEUE, c.ref(QUEUE / 'admission.json'))", "previous = primary_restore_previous(base)", 2),
    ('fixed3_launch_ref=c.ref(LAUNCH), previous_resources_ref=None)', 'fixed3_launch_ref=restore_launch_reference(), previous_resources_ref=None)', 1),
    ("cfg['fixed3_launch_ref'] != c.ref(LAUNCH)", "cfg['fixed3_launch_ref'] != restore_launch_reference()", 1),
    ("cfg['source_sha256'].update({str(LAUNCH): c.sha(LAUNCH), str(OLD): OLD_SHA})", "cfg['source_sha256'].update({restore_launch_reference()['path']: restore_launch_reference()['sha256'], str(OLD): OLD_SHA})", 1)]


def exact_bytes(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('pinned original execution/preflight SHA changed: ' + str(path))
    return data


def transformed_source():
    source = exact_bytes(V1, V1_SHA).decode()
    exact_bytes(V1_TEST, V1_TEST_SHA)
    for old, new, count in PATCHES:
        if source.count(old) != count:
            raise RuntimeError('exact private v2 source transform cardinality changed')
        source = source.replace(old, new)
    return source


source = transformed_source()
m = types.ModuleType('fixed3_explicit_v2_private_v1')
m.__file__ = str(SELF)
exec(compile(source, str(SELF) + ':private-frozen-v1', 'exec'), m.__dict__)
_original_frozen = m.frozen
_original_preflight_binding = m.preflight_binding
_original_verify_config = m.verify_config
_original_build = m.build
_original_verify_restore = m.verify_restore_config
BRIDGE_KEYS = {str(SELF), str(TEST), str(PREFLIGHT), str(OLD_LAUNCH)}


def execution_bridge():
    for path, digest in PINNED.items():
        exact_bytes(path, digest)
    return {'schema': 'fixed3_explicit_execution_bridge_v2',
            'v1_adapter_ref': {'path': str(V1), 'sha256': V1_SHA},
            'v1_test_ref': {'path': str(V1_TEST), 'sha256': V1_TEST_SHA},
            'original_preflight_ref': {'path': str(PREFLIGHT), 'sha256': PREFLIGHT_SHA},
            'failed_beforequeue_launch_ref': {'path': str(OLD_LAUNCH), 'sha256': OLD_LAUNCH_SHA},
            'v2_adapter_ref': m.c.ref(SELF), 'v2_test_ref': m.c.ref(TEST),
            'private_transform_sha256': hashlib.sha256(json.dumps(PATCHES, separators=(',', ':')).encode()).hexdigest(),
            'original_data_model_pid_history_ttl_not_reissued': True}


def frozen(worker, probe=True):
    contract = _original_frozen(worker, probe)
    contract['bindings'].update(PINNED)
    execution_bridge()
    for path, digest in contract['bindings'].items():
        if m.c.sha(path) != digest:
            raise RuntimeError('v2 actual frozen binding changed: ' + path)
    return contract


def preflight_binding(contract):
    execution_bridge()
    original_bindings = {p: h for p, h in contract['bindings'].items() if p not in BRIDGE_KEYS}
    proof = m.c.checked({'path': str(PREFLIGHT), 'sha256': PREFLIGHT_SHA})
    if original_bindings != proof['source_sha256']:
        raise RuntimeError('v2 bridge changed original 136 data/model/source bindings')
    translated = dict(contract, bindings=original_bindings)
    return _original_preflight_binding(translated)


def verify_config(cfg, contract):
    if cfg.get('execution_bridge') != execution_bridge():
        raise RuntimeError('explicit v2 code/config/preflight execution bridge missing or changed')
    if cfg.get('primary_preflight_ref') != {'path': str(PREFLIGHT), 'sha256': PREFLIGHT_SHA}:
        raise RuntimeError('original pinned preflight ref changed')
    _original_verify_config(cfg, contract)


def restore_launch_reference():
    return m.c.ref(m.LAUNCH) if m.LAUNCH.exists() else {'path': str(OLD_LAUNCH), 'sha256': OLD_LAUNCH_SHA}


def primary_restore_previous(base):
    # Once a v2 resource queue exists, its own actual resources_safe is mandatory.
    # A before-queue rejected v1/v2 has acquired nothing; original safe is authority.
    contract = frozen(m.worker_module())
    preflight_binding(contract)
    if m.QUEUE.exists():
        return m.restored_evidence(base, m.QUEUE, m.c.ref(m.QUEUE / 'admission.json'))
    if OLD_QUEUE.exists():
        raise RuntimeError('v1 resource queue exists; original-safe fallback prohibited')
    failed = m.c.checked({'path': str(OLD_LAUNCH), 'sha256': OLD_LAUNCH_SHA})
    if any(failed[k] != v for k, v in {'node': 'N6', 'gpus': [0, 1], 'global_revision': m.REVISION,
                                     'out': str(m.OUT)}.items()):
        raise RuntimeError('beforequeue failed launch scope changed')
    m.c.require_unreserved(contract['jobs'])
    old = m.original_config()
    proof = m.c.checked({'path': str(PREFLIGHT), 'sha256': PREFLIGHT_SHA})
    return m.restored_evidence(base, Path(old['queue']), proof['original_admission_ref'])


def verify_restore_config(cfg, old):
    if cfg.get('execution_bridge') != execution_bridge():
        raise RuntimeError('explicit v2 primary restoration bridge changed')
    _original_verify_restore(cfg, old)


def build(restore=False):
    # Original builder writes exclusive immutable files; bridge must be included
    # before that first write, never re-sign an already-created config/capture.
    original_save_once = m.c.save_once
    launch = m.RESTORE_LAUNCH if restore else m.LAUNCH

    def save_once(path, value):
        if Path(path) == launch:
            value = dict(value, execution_bridge=execution_bridge())
            value['source_sha256'].update(PINNED)
        return original_save_once(path, value)

    m.c.save_once = save_once
    try:
        return _original_build(restore)
    finally:
        m.c.save_once = original_save_once


m.frozen = frozen
m.preflight_binding = preflight_binding
m.verify_config = verify_config
m.verify_restore_config = verify_restore_config
m.primary_restore_previous = primary_restore_previous
m.restore_launch_reference = restore_launch_reference
m.build = build


def main():
    if '--config' in sys.argv:
        m.run_manager()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--build-from-original-safe', action='store_true')
    modes.add_argument('--build-primary-restore', action='store_true')
    modes.add_argument('--driver', action='store_true')
    parser.add_argument('--launch')
    args = parser.parse_args()
    if args.driver:
        if args.launch != str(m.LAUNCH):
            raise RuntimeError('exact v2 immutable managed driver launch required')
        m.run_driver()
    else:
        print(json.dumps(build(args.build_primary_restore), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
