"""One authorized N6 GPU2/3 config set, read-only capture then write-once metadata.

Does not create pool/queue/output, claim a source, load a model or signal a PID.
Historical origin receipt catalogs stay SHA-bound inside origin configs; only
source code, immutable plans/manifests/handoffs and native resource proofs are read.
"""
import copy
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_union_native_v4 as v


def main():
    prefix = 'caption_union_native_v4_n6pilot_20261026'
    planpath = v.RT / (prefix + '_merged_plan.json')
    manifestpath = v.RT / (prefix + '_manifest.json')
    configpath = v.RT / (prefix + '_n6g23.json')
    need = v.need
    need(all(not p.exists() and not p.is_symlink() for p in (planpath, manifestpath, configpath)), 'Pilot metadata exists; never overwrite')
    addresses = v.h.subprocess.check_output(['hostname', '-I'], text=True).split()
    need(v.HOSTS['N6'] in addresses, 'Builder must capture actual N6')
    queue = v.BAR / 'queues/caption_two_stage_local_n6_balanced_v1'
    safe_ref = v.poolmod.ref(queue / 'resources_safe.json')
    safe = v.checked(safe_ref)
    need(all(safe[k] is True for k in ('all_holders_safe', 'locks_released', 'leases_restored')), 'Previous native resources not safe')
    running, holders, holderrefs = v.base.apps(), [], []
    for gpu in (2, 3):
        reference = v.poolmod.ref(queue / ('holder_restored_gpu%d.json' % gpu))
        restored = v.checked(reference)
        current = v.base.identity(restored['holder_identity']['pid'])
        holder = dict(current, gpu=gpu, uuid=v.base.uuid_for(gpu))
        need(restored['uuid'] == holder['uuid'] and all(current[k] == restored['holder_identity'][k] for k in v.base.IDENTITY_KEYS), 'Fresh restored holder differs')
        v.base.check_holder(holder, current, gpu, holder['uuid'], running)
        need(not v.h.lock_snapshot(gpu)['held'], 'Fresh pair lock occupied')
        v.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
        holders.append(holder)
        holderrefs.append(reference)
    budget = v.base.headroom()
    free = v.shutil.disk_usage('/tmp').free
    need(budget['headroom'] >= 160 * 2**30 and free >= 8 * 2**30, '160GiB RAM/8GiB tmp boundary not available')
    refs = [v.poolmod.ref(v.RT / name) for name in ('caption_two_stage_local_pool2_n2g23_v1_launch.json', 'caption_two_stage_local_pool_n4g23_v1_launch.json')]
    readers = [v.read_origin(r) for r in refs]
    rows = {}
    bindings = {}
    for reference, reader in zip(refs, readers):
        for frozen in (reference, reader.cfg['plan'], reader.cfg['pool_manifest'], reader.cfg['source_handoff']):
            v.checked(frozen)
            bindings[frozen['path']] = frozen['sha256']
        for row in reader.rows.values():
            key = v.poolmod.source_key(row)
            need(key not in rows or rows[key] == row, 'Conflicting origin metadata')
            rows[key] = row
        for path, digest in reader.cfg['source_sha256'].items():
            if Path(path).suffix == '.py':
                need(path not in bindings or bindings[path] == digest, 'Conflicting source binding')
                need(v.base.sha(path) == digest, 'Frozen origin source changed: ' + path)
                bindings[path] = digest
    need(len(rows) == 13654, 'Actual deduplicated plan denominator is not 13654')
    origin = readers[0].cfg
    need(readers[1].cfg['modelref'] == origin['modelref'], 'Origin model identities differ')
    for module in list(sys.modules.values()) + [v.h, v.h.bench, v.h.c, v.mgr.old]:
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().parent == v.RT:
            path = str(Path(path).resolve())
            digest = v.base.sha(path)
            need(path not in bindings or bindings[path] == digest, 'Runtime source changed/conflicts')
            bindings[path] = digest
    for frozen in (safe_ref, *holderrefs, origin['modelref']):
        bindings[frozen['path']] = frozen['sha256']
    root = v.BAR / prefix
    phase = prefix + '__n6g23'
    env = copy.deepcopy(origin['env'])
    env['CUDA_VISIBLE_DEVICES'] = '2,3'
    cfg = dict(schema='caption_union_native_v4', node='N6', gpus=[2, 3], tensor_parallel_size=2,
               gpu_memory_utilization=.88, recipe=v.local.light.RECIPE, mode='consumer', release_authorized=True,
               env=env, cwd='/tmp', pool_root=str(root), pool_worker_id='n6g23', phase=phase,
               queue=str(v.BAR / 'queues' / phase), out=str(v.BAR / phase), draft_out=str(v.BAR / phase / 'archives'),
               stop_file=str(root / 'workers/n6g23/STOP'), config_path=str(configpath),
               argv=['/opt/venv_vllm/bin/python', '-B', str(v.SELF), 'worker', '--config', str(configpath)],
               producer=v.poolmod.ref(v.SELF), resource_manager_ref=v.poolmod.ref(v.SELF),
               origin_configs=[r['path'] for r in refs], origin_config_refs=refs, source_root=origin['source_root'],
               plan=dict(path=str(planpath), sha256=v.poolmod.digest(list(rows.values()))),
               union_manifest=dict(path=str(manifestpath), sha256='pending'), targetcount=len(rows), input_count=len(rows),
               holders=holders, holder_restored_refs=holderrefs, resource_previous_safe=safe_ref,
               modelref=origin['modelref'], source_sha256=bindings)
    need(all(not Path(cfg[k]).exists() for k in ('pool_root', 'queue', 'out')), 'Fresh pilot namespaces required')
    bindings[str(planpath)] = cfg['plan']['sha256']
    participant = dict(worker_id='n6g23', config_identity_sha256=v.config_identity(cfg),
                       **{k: cfg[k] for k in ('node', 'gpus', 'config_path', 'queue', 'out', 'draft_out')})
    manifest = dict(schema='caption_union_native_manifest_v4', participants=[participant],
                    **{k: cfg[k] for k in ('plan', 'origin_config_refs', 'source_root', 'pool_root', 'recipe')})
    cfg['union_manifest']['sha256'] = v.poolmod.digest(manifest)
    bindings[str(manifestpath)] = cfg['union_manifest']['sha256']
    # Exactly the three authorized new metadata files; no runtime directories.
    v.poolmod.exclusive(planpath, list(rows.values()))
    v.poolmod.exclusive(manifestpath, manifest)
    v.poolmod.exclusive(configpath, cfg)
    v.validate(cfg)
    print(json.dumps(dict(status='pilot_metadata_created_native_validate_passed', config=v.poolmod.ref(configpath),
                          producer=cfg['producer'], manager_sha256=v.BASE_SHA, N=len(rows), headroom=budget['headroom'],
                          tmp_free=free, holder_pids=[h['pid'] for h in holders], source_binding_count=len(bindings),
                          model_calls=0, signals_sent=False, claims_modified=False, queue_created=False), sort_keys=True))


if __name__ == '__main__':
    main()
