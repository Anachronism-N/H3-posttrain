"""One-shot N2 native collab3 slow 32B candidate consumers -> second immutable ready pool (pool2).
Only explicit --park-old-config may signal (pidfd TERM to the verified native manager, then confirmed GPU
remnants after child exit).  --check-old-config is read-only.  Prepare/validate never launch or reserve.
N7 consumers are deliberately NOT part of this group (fresh headroom below the native 160GiB gate).
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_pool_live_manager_v2 as manager  # process-local N2/N7 HOSTS for native check_node/validate
import caption_ready_pool_v1 as pool
import caption_local_rebalance_transition_v1 as rebalance

native = rebalance.native
m = native.m
need, load, ref, write_once = native.need, native.load, m.ref, pool.exclusive
BAR = RT.parent
SOURCE_ROOT = BAR / 'curated222939_collaborative_v3'
GROUP = 'caption_pool2_consumer_transition_v1'
POOL_ROOT = BAR / 'caption_two_stage_dynamic_pool_v2'
TEMPLATE = RT / 'caption_two_stage_local_pool_n4g67_v1_launch.json'
SPECS = (
    dict(old='curated_collaborative3_n2c0_v1', new='caption_two_stage_local_pool2_n2g23_v1', node='N2', gpus=[2, 3],
         shard=0, count=3333, sha='1acc445c08979f31b144c8f0b0c6f6442230b1e20f47b06a8154a2d468292152'),
    dict(old='curated_collaborative3_n2c1_recovery_v1', new='caption_two_stage_local_pool2_n2g45_v1', node='N2', gpus=[4, 5],
         shard=1, count=3333, sha='665b0084aca3510919d5b65e63c048b6ba977702d209c1babcb849ef55f02a07'),
    dict(old='curated_collaborative3_n2c2_v1', new='caption_two_stage_local_pool2_n2g67_v1', node='N2', gpus=[6, 7],
         shard=2, count=3333, sha='a18eb4ecec7c241be2ffa3af61f37f5850d532bcb06c471b79b0db413480b89b'),
)
# Live lanes whose plans must stay disjoint from the pool2 union (ownership never shared).
FOREIGN_CONFIGS = ('caption_two_stage_local_n4c3_balanced_v1_launch.json', 'caption_two_stage_local_n6_balanced_v1_launch.json',
                   'caption_two_stage_live_n5c3_v3_launch.json', 'curated_collaborative3_n7c7_v1_launch.json',
                   'curated_collaborative3_n7c8_v1_launch.json')
FOREIGN_PLANS = ('caption_pool_transition_v1_remaining.json',)
FROZEN = {
    'caption_local_rebalance_transition_v1.py': 'a2eb9f2492a064b9dac6c4763ac9d3bd5c1a30dbc67abfb8f8de44c12f466bbf',
    'caption_light_to_local_transition_v1.py': '0b53113c5e37f114a219ded9b1b49888d53139c5e9776fb4159ef7053c6cd5ee',
    'caption_local_live_manager_v1.py': '74dae116f4db431e6fed16fc86da0d8f41af1cbd9316f6052b54ef9433a14b85',
    'caption_pool_live_manager_v1.py': 'd4c1640b9cb8c58b74711e775fc5d5c22790f46ccbacf5109e798781d5b97bc2',
    'caption_ready_pool_v1.py': '6d5c8b0ad798c3cd61c70b1c7a828da0fac7f82fdee473593351e9db58aa7949',
    'resource_manager.py': manager.BASE_SHA,
}
BUDGET = dict(retired_candidate_consumers=3, new_pool_workers=3, new_copy_threads=3, copy_threads_per_instance=1,
              source_batch_size=3, max_inflight_batches=2, additional_models=0, n7_consumers_untouched=2)
ABSENCE_SCOPE = rebalance.ABSENCE_SCOPE


def artifact(suffix):
    return RT / (GROUP + suffix + '.json')


def cfg_path(spec, new=False):
    return RT / (spec['new' if new else 'old'] + '_launch.json')


def phase_path(spec, suffix):
    return RT / (spec['old'] + '_pool2_' + suffix + '.json')


def old_claim(spec):
    return RT / (spec['old'] + '_pool2_handoff_claim.json')


def frozen_refs():
    need(Path(m.base.__file__).resolve() == RT / 'resource_manager.py'
         and m.base.SAFETY + m.base.LOAD == native.MIN_HEADROOM == 160 * 2**30, 'Original lifecycle file/160GiB boundary differs')
    for name, digest in FROZEN.items():
        need(m.base.sha(RT / name) == digest, 'Frozen helper changed: ' + name)
    return [ref(SELF), ref(manager.SELF)] + [ref(RT / name) for name in FROZEN]


def check_old_identity(old, spec):
    path = cfg_path(spec)
    out = SOURCE_ROOT / 'consumers' / ('shard-%04d' % spec['shard'])
    producer = old['producer']['path']
    need(old['phase'] == spec['old'] and old['node'] == spec['node'] and old['gpus'] == spec['gpus']
         and old['mode'] == 'consumer' and old['release_authorized'] is True
         and old['targetcount'] == spec['count'] and old['source_selection'] == dict(shard=spec['shard'], nshard=9, limit=0)
         and old['out'] == str(out) and old['queue'] == str(BAR / 'queues' / spec['old'])
         and old['env']['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, spec['gpus']))
         and old['argv'][:5] == [native.PYTHON, '-B', '-u', producer, 'consumer']
         and m.base.sha(producer) == old['producer']['sha256']
         and m.base.sha(old['resource_manager']['path']) == old['resource_manager']['sha256']
         and old['base_manager_sha256'] == m.BASE_SHA, 'Fixed old consumer identity differs: ' + str(path))


def load_olds():
    frozen_refs()
    olds = []
    for spec in SPECS:
        path = cfg_path(spec)
        need(m.base.sha(path) == spec['sha'], 'Fixed old consumer config SHA differs')
        old = load(path)
        check_old_identity(old, spec)
        olds.append(old)
    need(all(old['modelref'] == olds[0]['modelref'] for old in olds), 'Old model refs differ')
    return olds


def select(path, new=False):
    path = Path(path).resolve()
    matches = [i for i, spec in enumerate(SPECS) if cfg_path(spec, new) == path]
    need(len(matches) == 1, 'Only the three fixed N2 consumer lanes are supported')
    return matches[0]


@contextmanager
def group_lock():
    with ExitStack() as stack:
        stack.enter_context(native.handoff_lock(GROUP))
        for spec in SPECS:
            stack.enter_context(native.handoff_lock(spec['old']))
        yield


def check_budget(value):
    need(all(value.get(k) == v for k, v in BUDGET.items()), 'Fixed pool2 budget differs')
    need(value['memory']['headroom'] >= native.MIN_HEADROOM and value['local_disk_free_bytes'] >= native.MIN_DISK,
         'Native 160GiB/8GiB gate failed')


def capacity():
    value = dict(native.capacity(), **BUDGET)
    check_budget(value)
    return value


def check_old_receipts(old, spec):
    """Exact native parent/child binding from the original admission/live receipts (read-only)."""
    path = cfg_path(spec)
    queue = Path(old['queue'])
    admission, live = load(queue / 'admission.json'), load(queue / 'live.json')
    digest = m.base.sha(path)
    need(admission['config_sha256'] == digest and admission['manager_sha256'] == m.BASE_SHA
         and admission['producer'] == old['producer'] and admission['plan'] == old['plan']
         and admission['out'] == old['out'] and admission['node'] == old['node'] and admission['gpus'] == old['gpus']
         and admission['phase'] == old['phase'] and admission['targetcount'] == old['targetcount']
         and admission['model_identity_sha256'] == old['modelref']['model_identity_sha256'], 'Old admission binding differs')
    parent, child = admission['manager_identity'], live['child_identity']
    keys = m.base.IDENTITY_KEYS
    need(child is not None and all(live['manager_identity'][k] == parent[k] for k in keys)
         and live['argv'] == old['argv'] and live['phase'] == old['phase'], 'Old live binding differs')
    need(parent['argv'] == [native.PYTHON, '-B', '-u', old['resource_manager']['path'], '--config', str(path),
                            '--config-sha', digest, '--manager-sha', m.BASE_SHA]
         and child['argv'] == old['argv'] and child['ppid'] == parent['pid']
         and parent['uid'] == child['uid'] == os.getuid() and parent['cgroup'] == child['cgroup']
         and child['CVD'] == ['CUDA_VISIBLE_DEVICES=' + old['env']['CUDA_VISIBLE_DEVICES']],
         'Old parent/worker native argv/UID/cgroup/PPID differs')
    holders = [dict(h['identity'], gpu=h['gpu'], uuid=h['uuid']) for h in admission['holders']]
    need([h['gpu'] for h in holders] == old['gpus'], 'Old admitted holder order differs')
    lane = dict(node=old['node'], gpus=old['gpus'], queue=old['queue'], holders=holders, env=old['env'])
    return parent, child, lane


def receipt_proofs(old, spec):
    queue = Path(old['queue'])
    return [native.file_identity(p) for p in (cfg_path(spec), Path(old['selected_sources']['path']),
                                              queue / 'admission.json', queue / 'live.json')]


def outputs_absent():
    need(not POOL_ROOT.exists(), 'New pool2 directory already exists')
    for suffix in ('_remaining', '_handoff', '_manifest', '_prepared'):
        need(not artifact(suffix).exists(), 'New immutable group artifact exists')
    for spec in SPECS:
        need(not cfg_path(spec, True).exists(), 'New launch config exists')
        need(not (RT / (spec['new'] + '_validated.json')).exists(), 'New validation receipt exists')
        for path in (BAR / spec['new'], BAR / (spec['new'] + '_execution'), BAR / 'queues' / spec['new']):
            need(not path.exists(), 'New output/queue exists')


def group_claim(create=False):
    need(not artifact('_failed').exists(), 'Group sealed after failure; no retry/resume')
    expected = dict(old_configs=[ref(cfg_path(s)) for s in SPECS], new_names=[s['new'] for s in SPECS],
                    pool_root=str(POOL_ROOT), code_refs=frozen_refs(), resource_budget=BUDGET,
                    original_config_resume_forbidden=True)
    path = artifact('_claim')
    if not path.exists():
        need(create, 'All three old consumers must be parked before prepare')
        outputs_absent()
        need(all(not old_claim(s).exists() for s in SPECS), 'Old consumer already claimed elsewhere')
        write_once(path, expected)
        for spec in SPECS:
            write_once(old_claim(spec), dict(old_config=ref(cfg_path(spec)), name=spec['new'], group_claim=ref(path),
                                             transition=ref(SELF), original_config_resume_forbidden=True))
    need(load(path) == expected, 'Group/config/helper/budget identity changed')
    group = ref(path)
    for spec in SPECS:
        need(load(old_claim(spec)) == dict(old_config=ref(cfg_path(spec)), name=spec['new'], group_claim=group,
             transition=ref(SELF), original_config_resume_forbidden=True), 'Old shared claim differs')
    return group


def fail(error):
    if not artifact('_failed').exists():
        write_once(artifact('_failed'), dict(reason=type(error).__name__ + ': ' + str(error),
                   original_config_resume_forbidden=True, no_automatic_retry=True, locks_leases_budgets_not_deleted=True))


def check_old(path):
    """Read-only preflight: identity, receipts, fresh capacity, exact GPU tree. No write, no signal."""
    i = select(path)
    olds = load_olds()
    old, spec = olds[i], SPECS[i]
    native.check_node(old)
    need(not (Path(old['queue']) / 'resources_safe.json').exists(), 'Old consumer already safe/exited')
    parent, child, lane = check_old_receipts(old, spec)
    proofs = receipt_proofs(old, spec)
    tree = native.capture_gpu_tree(lane, parent, child)
    budget = capacity()
    native.check_files(proofs)
    value = dict(event='pool2_old_checked', phase=old['phase'], manager_pid=parent['pid'], child_pid=child['pid'],
                 gpu_tree=[t['pid'] for t in tree], headroom_gib=round(budget['memory']['headroom'] / 2**30, 1),
                 tmp_free_gib=round(budget['local_disk_free_bytes'] / 2**30, 1), signals_sent=False, writes=False)
    print(json.dumps(value), flush=True)
    return value


def park(path):
    i = select(path)
    with group_lock():
        olds = load_olds()
        old, spec = olds[i], SPECS[i]
        native.check_node(old)
        # Everything before the attempt record is read-only; failures here neither signal nor seal.
        parent, child, lane = check_old_receipts(old, spec)
        proofs = receipt_proofs(old, spec)
        budget = capacity()
        attempt = phase_path(spec, 'park_claim')
        need(not attempt.exists(), 'Park already attempted; no retry')
        try:
            group = group_claim(create=True)
            write_once(attempt, dict(group_claim=group, old_config=ref(cfg_path(spec)), transition=ref(SELF)))
            budget = capacity()
            queue = Path(old['queue'])
            tree_path = phase_path(spec, 'old_tree')
            expected, safe = native.park_old(lane, parent, child, proofs, tree_path)
            need(safe['path'] == str(queue / 'resources_safe.json'), 'Historical safe cannot authorize handoff')
            owners = native.restored_owners(lane)
            native.old_tree_absent(expected)
            native.safe_parent_wait(lane, parent)
            proofs += [native.file_identity(p) for p in [tree_path, Path(safe['path'])]
                       + [queue / ('holder_restored_gpu%d.json' % gpu) for gpu in old['gpus']]]
            native.check_files(proofs)
            record = dict(group_claim=group, park_attempt=ref(attempt), old_config=ref(cfg_path(spec)), old_phase=old['phase'],
                          manager=parent, child=child, lane=lane, expected_old_tree=expected, old_worker_tree_absent=True,
                          old_worker_tree_absence_scope=ABSENCE_SCOPE, safe_old_manager_quiescent_wait_allowed=True,
                          resource_previous_safe=safe, owners=owners, proofs=proofs, resource_budget=budget)
            write_once(phase_path(spec, 'parked'), record)
            print(json.dumps(dict(event='pool2_old_parked', phase=old['phase'], no_new_model=True)), flush=True)
            return record
        except BaseException as error:
            fail(error)
            raise


def parked_records(olds, group):
    records = []
    for old, spec in zip(olds, SPECS):
        record = load(phase_path(spec, 'parked'))
        need(record['group_claim'] == group and record['park_attempt'] == ref(phase_path(spec, 'park_claim'))
             and record['old_config'] == ref(cfg_path(spec)) and record['old_phase'] == old['phase']
             and record['old_worker_tree_absent'] is True and record['old_worker_tree_absence_scope'] == ABSENCE_SCOPE
             and record['safe_old_manager_quiescent_wait_allowed'] is True, 'Parked record differs')
        native.check_files(record['proofs'])  # Same node (N2) for park/prepare/validate.
        need(native.safe_proof(Path(old['queue']) / 'resources_safe.json') == record['resource_previous_safe'],
             'Latest queue safe differs from park')
        need(len(record['owners']) == 2 and [o['gpu'] for o in record['owners']] == old['gpus']
             and len({o['pid'] for o in record['owners']}) == len({o['uuid'] for o in record['owners']}) == 2,
             'Parked holders differ')
        check_budget(record['resource_budget'])
        records.append(record)
    return records


def scan_ids(directory):
    if not directory.exists():
        return []
    need(directory.is_dir() and not directory.is_symlink(), 'Unsafe started-source directory')
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            need(entry.is_dir(follow_symlinks=False), 'Unsafe started-source entry')
            names.append(entry.name)
    return names


def jsonl_keys(path, allowed):
    keys = set()
    data = path.read_bytes()
    need(not data or data.endswith(b'\n'), 'Incomplete terminal journal after park: ' + str(path))
    for line in data.splitlines():
        if line.strip():
            row = json.loads(line)
            key = native.row_key(dict(id=row['id'], split=row['split']))
            need(key in allowed, 'Terminal outside old selected plan')
            keys.add(key)
    return keys


def consumer_exclusions(old, rows):
    """Started = any source/visual directory or any terminal journal row; nothing is ever reset."""
    out = Path(old['out'])
    by_id = {row['id']: native.row_key(row) for row in rows}
    allowed = set(by_id.values())
    started, terminal, proofs = set(), set(), []
    sources = out / 'sources'
    for split in scan_ids(sources):
        need(split in native.SPLITS, 'Unknown split in old consumer output')
        for sid in scan_ids(sources / split):
            key = native.row_key(dict(id=sid, split=split))
            need(key in allowed, 'Started source outside old selected plan')
            started.add(key)
    visual = out / 'visual'
    shard = 'shard-%04d' % old['source_selection']['shard']
    need(not visual.exists() or set(scan_ids(visual)) <= {shard}, 'Unknown visual shard in old consumer output')
    for sid in scan_ids(visual / shard / 'sources'):
        need(sid in by_id, 'Visual source outside old selected plan')
        started.add(by_id[sid])
    journals = [out / 'completed.jsonl'] + sorted((out / 'batches').glob('*/terminal_index.jsonl'))
    for path in journals:
        if path.exists():
            proofs.append(native.file_identity(path))
            terminal |= jsonl_keys(path, allowed)
    native.check_files(proofs)
    return started | terminal, dict(old_out=old['out'], started_source_keys=sorted(started),
                                     terminal_source_keys=sorted(terminal),
                                     started_without_terminal=sorted(started - terminal), budget_not_reset=True), proofs


def foreign_keys():
    keys = set()
    for name in FOREIGN_PLANS:
        keys |= {native.row_key(r) for r in load(RT / name)}
    for name in FOREIGN_CONFIGS:
        value = load(RT / name)
        reference = value.get('selected_sources') or value['plan']
        keys |= {native.row_key(dict(id=r['id'], split=r['split'])) for r in load(reference['path'])}
    return keys


def remaining_union(olds):
    union, owners, excluded, details, proofs = {}, [], set(), [], []
    for old, spec in zip(olds, SPECS):
        need(ref(old['selected_sources']['path']) == old['selected_sources'], 'Old selected sources changed')
        rows = load(old['selected_sources']['path'])
        keys = [native.row_key(row) for row in rows]
        need(rows and len(keys) == len(set(keys)) == old['targetcount'], 'Old selected keys/count differ')
        need(not set(keys) & set(union), 'Old consumer plans must be disjoint')
        denied, detail, lane_proofs = consumer_exclusions(old, rows)
        for row in rows:
            key = native.row_key(row)
            union[key] = row
            owners.append(dict(id=key[0], split=key[1], old_phase=old['phase'], worker_id=spec['new']))
        excluded |= denied
        details.append(dict(old_config=ref(cfg_path(spec)), details=detail, proofs=lane_proofs))
        proofs += lane_proofs
    need(len({k[0].casefold() for k in union}) == len(union), 'Portable unique IDs required')
    need(not set(union) & foreign_keys(), 'Pool2 union overlaps a live foreign lane/pool1 plan')
    rows = [row for key, row in union.items() if key not in excluded]
    need(rows, 'No unstarted sources remain')
    return rows, dict(original_union_count=len(union), excluded_source_keys=sorted(excluded), original_owner_map=owners,
                      per_lane_exclusions=details, all_unready_retained=True, budget_not_reset=True,
                      old_candidates_caches_claims_preserved=True, shared_eligibility_not_source_ownership=True), proofs


def code_bindings(records, proofs, paths):
    template = load(TEMPLATE)
    skip = {str(RT / 'caption_pool_live_manager_v1.py')}
    bindings = {}
    for path, digest in template['source_sha256'].items():
        if path.endswith('.py') and path not in skip:
            need(m.base.sha(path) == digest, 'Pool1 frozen code binding changed: ' + path)
            bindings[path] = digest
    for proof in proofs + [p for record in records for p in record['proofs']]:
        need(proof['path'] not in bindings or bindings[proof['path']] == proof['sha256'], 'Conflicting proof binding')
        bindings[proof['path']] = proof['sha256']
    for path in paths:
        reference = ref(path)
        need(bindings.get(reference['path'], reference['sha256']) == reference['sha256'], 'Immutable binding changed')
        bindings[reference['path']] = reference['sha256']
    return bindings


def build_configs(olds, records, bindings):
    template = load(TEMPLATE)
    manifest_path = artifact('_manifest')
    configs = []
    for old, spec, record in zip(olds, SPECS, records):
        env = dict(template['env'], CUDA_VISIBLE_DEVICES=','.join(map(str, spec['gpus'])))
        lane = dict(node=spec['node'], gpus=spec['gpus'], draft_out=old['out'], source_root=str(SOURCE_ROOT),
                    modelref=old['modelref'], env=env)
        cfg = native.build_config(lane, cfg_path(spec), spec['new'], record['owners'], artifact('_remaining'),
                                  artifact('_handoff'), record['resource_previous_safe'], bindings)
        cfg.update(producer=ref(manager.SELF), resource_manager_ref=ref(manager.SELF), pool_root=str(POOL_ROOT),
                   pool_manifest=dict(path=str(manifest_path), sha256='0' * 64), pool_worker_id=spec['new'],
                   pool_group_claim=ref(artifact('_claim')), pool_old_parked=ref(phase_path(spec, 'parked')))
        cfg['argv'] = [native.PYTHON, '-B', str(manager.SELF), 'worker', '--config', cfg['config_path']]
        cfg['source_sha256'].update({str(manager.SELF): cfg['producer']['sha256'],
                                    str(Path(pool.__file__).resolve()): m.base.sha(pool.__file__),
                                    str(manifest_path): '0' * 64})
        configs.append(cfg)
    return configs


def prepare():
    with group_lock():
        olds = load_olds()
        for old in olds:
            native.check_node(old)
        group = group_claim()
        need(not artifact('_prepare_claim').exists(), 'Prepare already attempted; no retry')
        try:
            outputs_absent()
            write_once(artifact('_prepare_claim'), dict(group_claim=group, transition=ref(SELF)))
            records = parked_records(olds, group)
            capacity()
            for old, record in zip(olds, records):
                native.old_tree_absent(record['expected_old_tree'])
                native.safe_parent_wait(record['lane'], record['manager'])
                need(native.restored_owners(record['lane']) == record['owners'], 'Fresh holders changed')
            rows, detail, progress_proofs = remaining_union(olds)
            write_once(artifact('_remaining'), rows)
            all_proofs = progress_proofs + [p for r in records for p in r['proofs']]
            write_once(artifact('_handoff'), dict(remaining_sources=rows, old_worker_tree_absent=True,
                original_config_resume_forbidden=True, old_worker_tree_absence_scope=ABSENCE_SCOPE,
                safe_old_manager_quiescent_wait_allowed=True, old_configs=[ref(cfg_path(s)) for s in SPECS],
                old_selected=[old['selected_sources'] for old in olds],
                old_native_parked=[ref(phase_path(s, 'parked')) for s in SPECS], old_proofs=all_proofs,
                resource_safety_proofs=[r['resource_previous_safe'] for r in records],
                remaining_plan=ref(artifact('_remaining')), resource_budget=BUDGET, group_claim=group, **detail))
            paths = [SELF, manager.SELF, Path(pool.__file__).resolve(), Path(old['modelref']['path'])]
            paths += [RT / name for name in FROZEN]
            paths += [artifact(s) for s in ('_claim', '_prepare_claim', '_remaining', '_handoff')]
            paths += [p for s in SPECS for p in (cfg_path(s), old_claim(s), phase_path(s, 'park_claim'), phase_path(s, 'parked'))]
            bindings = code_bindings(records, progress_proofs, paths)
            configs = build_configs(olds, records, bindings)
            participants = [dict(worker_id=c['pool_worker_id'], config_path=c['config_path'], node=c['node'], gpus=c['gpus'],
                                 config_identity_sha256=pool.config_identity(c)) for c in configs]
            write_once(artifact('_manifest'), dict(schema='caption_ready_pool_v1', plan=ref(artifact('_remaining')),
                       source_handoff=ref(artifact('_handoff')), source_root=str(SOURCE_ROOT),
                       pool_root=str(POOL_ROOT), participants=participants))
            for cfg in configs:
                cfg['pool_manifest'] = ref(artifact('_manifest'))
                cfg['source_sha256'][str(artifact('_manifest'))] = cfg['pool_manifest']['sha256']
                write_once(Path(cfg['config_path']), cfg)
                pool.check_manifest(cfg)
            native.check_files(all_proofs)
            write_once(artifact('_prepared'), dict(group_claim=group, transition=ref(SELF),
                       configs=[ref(cfg_path(s, True)) for s in SPECS], pool_manifest=ref(artifact('_manifest')),
                       handoff=ref(artifact('_handoff')), plan=ref(artifact('_remaining'))))
            argvs = [_validate(cfg_path(s, True), olds, group, records) for s in SPECS]
            print(json.dumps(dict(event='pool2_prepared', N=len(rows), original_union=detail['original_union_count'],
                                  excluded=len(detail['excluded_source_keys']), native_manager_argv=argvs)), flush=True)
            return configs
        except BaseException as error:
            fail(error)
            raise


def _validate(path, olds, group, records):
    i = select(path, new=True)
    old, spec, record = olds[i], SPECS[i], records[i]
    native.check_node(old)
    prepared = load(artifact('_prepared'))
    need(prepared['group_claim'] == group and prepared['transition'] == ref(SELF) and prepared['configs'][i] == ref(path)
         and prepared['pool_manifest'] == ref(artifact('_manifest')) and prepared['handoff'] == ref(artifact('_handoff'))
         and prepared['plan'] == ref(artifact('_remaining')), 'Prepared artifacts/config changed')
    native.old_tree_absent(record['expected_old_tree'])
    native.safe_parent_wait(record['lane'], record['manager'])
    need(native.restored_owners(record['lane']) == record['owners'], 'Latest sole restored holders differ')
    capacity()
    cfg = load(path)
    handoff = load(artifact('_handoff'))
    need(cfg['source_handoff'] == prepared['handoff'] and cfg['plan'] == prepared['plan']
         and cfg['pool_manifest'] == prepared['pool_manifest'] and cfg['pool_root'] == str(POOL_ROOT)
         and cfg['pool_worker_id'] == spec['new'] and cfg['pool_group_claim'] == group
         and cfg['pool_old_parked'] == ref(phase_path(spec, 'parked')) and cfg['previous_config'] == ref(cfg_path(spec))
         and cfg['old_out'] == old['out'] and cfg['resource_previous_safe'] == record['resource_previous_safe']
         and cfg['holders'] == record['owners'] and cfg['source_root'] == str(SOURCE_ROOT)
         and cfg['modelref'] == old['modelref'] and cfg['node'] == old['node'] and cfg['gpus'] == old['gpus']
         and cfg['env']['CUDA_VISIBLE_DEVICES'] == old['env']['CUDA_VISIBLE_DEVICES']
         and cfg['phase'] == spec['new'] and cfg['config_path'] == str(Path(path).resolve()) and cfg['cwd'] == str(RT)
         and cfg['draft_out'] == str(BAR / spec['new']) and cfg['out'] == str(BAR / (spec['new'] + '_execution'))
         and cfg['queue'] == str(BAR / 'queues' / spec['new'])
         and cfg['producer'] == cfg['resource_manager_ref'] == ref(manager.SELF)
         and cfg['argv'] == [native.PYTHON, '-B', str(manager.SELF), 'worker', '--config', str(path)]
         and cfg['targetcount'] == cfg['input_count'] == len(handoff['remaining_sources'])
         and cfg['release_authorized'] is True and cfg['reuse_existing_results'] is False
         and handoff['resource_budget'] == BUDGET and handoff['group_claim'] == group, 'New pool2 lane assignment differs')
    native.check_files(handoff['old_proofs'])
    manifest, rows = pool.check_manifest(cfg)
    need(rows == handoff['remaining_sources'] and handoff['remaining_plan'] == cfg['plan']
         and handoff['old_native_parked'] == [ref(phase_path(s, 'parked')) for s in SPECS]
         and len(manifest['participants']) == len(SPECS), 'Complete fixed union/participants required')
    for participant, assigned in zip(manifest['participants'], SPECS):
        need(participant['worker_id'] == assigned['new'] and participant['node'] == assigned['node']
             and participant['gpus'] == assigned['gpus'] and participant['config_path'] == str(cfg_path(assigned, True)),
             'Fixed pool2 participant differs')
    manager.validate(cfg)
    fn = manager.base.check_sources
    check = type(fn)(fn.__code__, dict(fn.__globals__, selected_sources=manager.selected), fn.__name__,
                     fn.__defaults__, fn.__closure__)
    check(cfg, str(path), m.base.sha(path), manager.BASE_SHA)
    argv = native.manager_argv(Path(path), manager.SELF, m.base.sha(path))
    receipt_path = RT / (spec['new'] + '_validated.json')
    receipt = dict(config=ref(path), group_claim=group, node=old['node'], transition=ref(SELF), handoff=prepared['handoff'],
                   pool_manifest=prepared['pool_manifest'], native_manager_argv=argv,
                   config_identity_sha256=pool.config_identity(cfg), fresh_native_capacity_passed=True,
                   current_sole_holders_locks_leases_parent_wait_checked=True)
    if receipt_path.exists():
        need(load(receipt_path) == receipt, 'Validated receipt changed')
    else:
        write_once(receipt_path, receipt)
    print(json.dumps(dict(event='pool2_native_validated', phase=cfg['phase'], native_manager_argv=argv)), flush=True)
    return argv


def validate_new(path):
    with group_lock():
        olds = load_olds()
        group = group_claim()
        return _validate(Path(path).resolve(), olds, group, parked_records(olds, group))


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check-old-config')
    actions.add_argument('--park-old-config')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--validate-new-config')
    args = parser.parse_args()
    if args.check_old_config:
        check_old(args.check_old_config)
    elif args.park_old_config:
        park(args.park_old_config)
    elif args.prepare:
        prepare()
    else:
        validate_new(args.validate_new_config)


if __name__ == '__main__':
    main()
