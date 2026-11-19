"""One-shot four native local lanes -> immutable shared ready pool.
Only explicit park can signal; prepare/validate never launch or reserve sources.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_local_rebalance_transition_v1 as rebalance
import caption_pool_live_manager_v1 as manager
import caption_ready_pool_v1 as pool

native = rebalance.native
m = native.m
need, load, ref, write_once = native.need, native.load, m.ref, pool.exclusive
BAR = RT.parent
GROUP = 'caption_pool_transition_v1'
POOL_ROOT = BAR / 'caption_two_stage_dynamic_pool_v1'
SPECS = (
    dict(old='caption_two_stage_local_n4_v1', new='caption_two_stage_local_pool_n4g67_v1',
         node='N4', gpus=[6, 7], count=949, sha='85eeeceaf4fe8697e633d7955a785d422c7b795711c0c687192b1ebd0413d400'),
    dict(old='caption_two_stage_local_n3_v1', new='caption_two_stage_local_pool_n3g01_v1',
         node='N3', gpus=[0, 1], count=948, sha='db0dd7746a7741abc7917df8d7e97ffddf12b4bc63bef41c704cf164879a8e2e'),
    dict(old='caption_two_stage_local_n4c5_v1', new='caption_two_stage_local_pool_n4g45_v1',
         node='N4', gpus=[4, 5], count=2767, sha='81200a8f7ea97331560e41cbf1f829357e8ff8aab5c706aa161c71e127e6c0f3'),
    dict(old='caption_two_stage_local_n4c4_v1', new='caption_two_stage_local_pool_n4g23_v1',
         node='N4', gpus=[2, 3], count=2736, sha='936a38911a2b85554ebafacebb8f8b78d7feabdcd37eed79810b8bbff6b42895'),
)
FROZEN = {
    'caption_local_rebalance_transition_v1.py': 'a2eb9f2492a064b9dac6c4763ac9d3bd5c1a30dbc67abfb8f8de44c12f466bbf',
    'caption_light_to_local_transition_v1.py': '0b53113c5e37f114a219ded9b1b49888d53139c5e9776fb4159ef7053c6cd5ee',
    'caption_light_to_local_transition_v2.py': 'ce6ef08c9173933133266a213c8521863c372acfa25b3bf71df97bc92182dd72',
    'caption_local_live_manager_v1.py': '74dae116f4db431e6fed16fc86da0d8f41af1cbd9316f6052b54ef9433a14b85',
    'caption_pool_live_manager_v1.py': 'd4c1640b9cb8c58b74711e775fc5d5c22790f46ccbacf5109e798781d5b97bc2',
    'caption_ready_pool_v1.py': '6d5c8b0ad798c3cd61c70b1c7a828da0fac7f82fdee473593351e9db58aa7949',
    'resource_manager.py': manager.BASE_SHA,
}
BUDGET = dict(existing_copy_threads=6, retired_copy_threads=4, new_copy_threads=4,
              total_copy_threads=6, total_n4_copy_threads=4, total_n3_copy_threads=1,
              total_n6_copy_threads=1, copy_threads_per_instance=1,
              source_batch_size=3, max_inflight_batches=2, additional_models=0)
ABSENCE_SCOPE = rebalance.ABSENCE_SCOPE


def artifact(suffix):
    return RT / (GROUP + suffix + '.json')


def cfg_path(spec, new=False):
    return RT / (spec['new' if new else 'old'] + '_launch.json')


def phase_path(spec, suffix):
    return RT / (spec['old'] + '_pool_' + suffix + '.json')


def old_claim(spec):
    return RT / (spec['old'] + '_light_to_local_handoff_claim.json')


def frozen_refs():
    need(Path(m.base.__file__).resolve() == RT / 'resource_manager.py'
         and m.base.SAFETY + m.base.LOAD == native.MIN_HEADROOM == 160 * 2**30,
         'Original lifecycle file/native 160GiB boundary differs')
    for name, digest in FROZEN.items():
        need(m.base.sha(RT / name) == digest, 'Frozen helper changed: ' + name)
    return [ref(SELF), ref(Path(rebalance.__file__).resolve())] + [ref(RT / name) for name in FROZEN]


def load_olds():
    frozen_refs()
    olds = []
    for spec in SPECS:
        path = cfg_path(spec)
        need(m.base.sha(path) == spec['sha'], 'Fixed old config SHA differs')
        old = load(path)
        producer = ref(RT / 'caption_local_live_manager_v1.py')
        need(old['config_path'] == str(path) and old['cwd'] == str(RT)
             and old['phase'] == spec['old'] and old['node'] == spec['node']
             and old['gpus'] == spec['gpus'] and old['targetcount'] == old['input_count'] == spec['count']
             and old['producer'] == old['resource_manager_ref'] == producer
             and old['argv'] == [native.PYTHON, '-B', producer['path'], 'worker', '--config', str(path)]
             and old['release_authorized'] is True
             and old['env']['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, spec['gpus']))
             and old['draft_out'] == str(BAR / spec['old'])
             and old['out'] == str(BAR / (spec['old'] + '_execution'))
             and old['queue'] == str(BAR / 'queues' / spec['old']), 'Fixed old lane identity differs')
        olds.append(old)
    need(all(old['source_root'] == str(BAR / 'curated222939_collaborative_v3')
             and old['modelref'] == olds[0]['modelref'] for old in olds), 'Old source root/model refs differ')
    return olds


def select(path, new=False):
    path = Path(path).resolve()
    matches = [i for i, spec in enumerate(SPECS) if cfg_path(spec, new) == path]
    need(len(matches) == 1, 'Only the four fixed local lanes are supported')
    return matches[0]


@contextmanager
def group_lock():
    with ExitStack() as stack:
        stack.enter_context(native.handoff_lock(GROUP))
        for spec in SPECS:
            stack.enter_context(native.handoff_lock(spec['old']))
        yield


def check_budget(value):
    need(all(value.get(k) == v for k, v in BUDGET.items()), 'Fixed replacement copy budget differs')
    need(value['memory']['headroom'] >= native.MIN_HEADROOM
         and value['local_disk_free_bytes'] >= native.MIN_DISK, 'Native 160GiB/8GiB gate failed')


def capacity():
    value = dict(native.capacity(), **BUDGET)
    check_budget(value)
    return value


def outputs_absent():
    need(not POOL_ROOT.exists(), 'New pool directory already exists')
    for suffix in ('_remaining', '_handoff', '_manifest', '_prepared'):
        need(not artifact(suffix).exists(), 'New immutable group artifact exists')
    for spec in SPECS:
        need(not cfg_path(spec, True).exists(), 'New launch config exists')
        need(not (RT / (spec['new'] + '_validated.json')).exists(), 'New validation receipt already exists')
        for path in (BAR / spec['new'], BAR / (spec['new'] + '_execution'), BAR / 'queues' / spec['new']):
            need(not path.exists(), 'New output/queue exists')


def group_claim(olds, create=False):
    need(not artifact('_failed').exists(), 'Group sealed after failure; no retry/resume')
    expected = dict(old_configs=[ref(cfg_path(s)) for s in SPECS],
                    new_names=[s['new'] for s in SPECS], pool_root=str(POOL_ROOT),
                    code_refs=frozen_refs(), resource_budget=BUDGET,
                    original_config_resume_forbidden=True)
    path = artifact('_claim')
    if not path.exists():
        need(create, 'All four old lanes must be parked before prepare')
        outputs_absent()
        need(all(not old_claim(s).exists() for s in SPECS), 'Old phase already claimed elsewhere')
        write_once(path, expected)
        for spec in SPECS:
            write_once(old_claim(spec), dict(old_config=ref(cfg_path(spec)), name=spec['new'],
                       group_claim=ref(path), transition=ref(SELF), original_config_resume_forbidden=True))
    need(load(path) == expected, 'Group/config/helper/budget identity changed')
    group = ref(path)
    for spec in SPECS:
        need(load(old_claim(spec)) == dict(old_config=ref(cfg_path(spec)), name=spec['new'],
             group_claim=group, transition=ref(SELF), original_config_resume_forbidden=True), 'Old shared claim differs')
    return group


def fail(error):
    if not artifact('_failed').exists():
        write_once(artifact('_failed'), dict(reason=type(error).__name__ + ': ' + str(error),
                   original_config_resume_forbidden=True, no_automatic_retry=True,
                   locks_leases_budgets_not_deleted=True))


def check_sources(cfg, path, selected):
    fn = manager.base.check_sources
    check = types.FunctionType(fn.__code__, dict(fn.__globals__, selected_sources=selected),
                               fn.__name__, fn.__defaults__, fn.__closure__)
    check(cfg, str(path), m.base.sha(path), manager.BASE_SHA)


def park(path):
    i = select(path)
    with group_lock():
        try:
            olds = load_olds()
            old, spec = olds[i], SPECS[i]
            scope = rebalance.lane_scope(old)
            scope['check_node'](old)
            group = group_claim(olds, create=True)
            attempt = phase_path(spec, 'park_claim')
            need(not attempt.exists(), 'Park already attempted; no retry')
            write_once(attempt, dict(group_claim=group, old_config=ref(cfg_path(spec)), transition=ref(SELF)))
            check_sources(old, cfg_path(spec), m.selected)
            budget = capacity()
            queue = Path(old['queue'])
            proofs = [native.file_identity(p) for p in (cfg_path(spec), Path(old['plan']['path']),
                      queue / 'admission.json', queue / 'live.json')]
            parent, child = scope['check_old_receipts'](old, cfg_path(spec), load(queue / 'admission.json'), load(queue / 'live.json'))
            tree_path = phase_path(spec, 'old_tree')
            expected, safe = scope['park_old'](old, parent, child, proofs, tree_path)
            need(safe['path'] == str(queue / 'resources_safe.json'), 'Historical predecessor safe cannot authorize handoff')
            owners = scope['restored_owners'](old)
            scope['old_tree_absent'](expected)
            scope['safe_parent_wait'](old, parent)
            proofs += [native.file_identity(p) for p in [tree_path, Path(safe['path'])]
                       + [queue / ('holder_restored_gpu%d.json' % gpu) for gpu in old['gpus']]]
            native.check_files(proofs)
            record = dict(group_claim=group, park_attempt=ref(attempt), old_config=ref(cfg_path(spec)), old_phase=old['phase'],
                          manager=parent, expected_old_tree=expected, old_worker_tree_absent=True,
                          old_worker_tree_absence_scope=ABSENCE_SCOPE, safe_old_manager_quiescent_wait_allowed=True,
                          resource_previous_safe=safe, owners=owners, proofs=proofs,
                          source_bindings=old['source_sha256'], resource_budget=budget)
            write_once(phase_path(spec, 'parked'), record)
            print(json.dumps(dict(event='pool_old_parked', phase=old['phase'], no_new_model=True)), flush=True)
            return record
        except BaseException as error:
            fail(error)
            raise


def parked_records(olds, group):
    records = []
    for old, spec in zip(olds, SPECS):
        record = load(phase_path(spec, 'parked'))
        need(record['group_claim'] == group and record['park_attempt'] == ref(phase_path(spec, 'park_claim'))
             and record['old_config'] == ref(cfg_path(spec))
             and record['old_phase'] == old['phase'] and record['old_worker_tree_absent'] is True
             and record['old_worker_tree_absence_scope'] == ABSENCE_SCOPE
             and record['safe_old_manager_quiescent_wait_allowed'] is True
             and record['source_bindings'] == old['source_sha256'], 'Four-lane parked record differs')
        rebalance.check_shared_files(record['proofs'])
        need(native.safe_proof(Path(old['queue']) / 'resources_safe.json') == record['resource_previous_safe'],
             'Latest current local queue safe differs from park')
        need(len(record['owners']) == 2 and [o['gpu'] for o in record['owners']] == old['gpus']
             and len({o['pid'] for o in record['owners']}) == len({o['uuid'] for o in record['owners']}) == 2,
             'Parked holders differ')
        check_budget(record['resource_budget'])
        records.append(record)
    return records


def remaining_union(olds):
    union, owners, excluded, details, proofs = {}, [], set(), [], []
    for old, spec in zip(olds, SPECS):
        rows = rebalance.plan_rows(old)
        keys = {native.row_key(row) for row in rows}
        need(not keys & set(union), 'Original four plans must be disjoint')
        denied, detail, lane_proofs = rebalance.local_exclusions(old, rows)
        for row in rows:
            key = native.row_key(row)
            union[key] = row
            owners.append(dict(id=key[0], split=key[1], old_phase=old['phase'],
                               old_config=ref(cfg_path(spec)), worker_id=spec['new']))
        excluded.update(denied)
        details.append(dict(old_config=ref(cfg_path(spec)), old_out=old['draft_out'], details=detail, proofs=lane_proofs))
        proofs.extend(lane_proofs)
    need(len(union) == sum(s['count'] for s in SPECS)
         and len({k[0].casefold() for k in union}) == len(union), 'Original union/count/unique ID differs')
    rows = [row for key, row in union.items() if key not in excluded]
    need(rows, 'No unstarted sources remain')
    return rows, dict(original_union_count=len(union), excluded_source_keys=sorted(excluded),
                     original_owner_map=owners, per_lane_exclusions=details,
                     all_unready_retained=True, budget_not_reset=True, old_inputs_tmp_preserved=True, shared_eligibility_not_source_ownership=True), proofs


def merge_bindings(records, proofs, paths):
    bindings = {}
    for record in records:
        for path, digest in record['source_bindings'].items():
            need(path not in bindings or bindings[path] == digest, 'Conflicting old source binding')
            bindings[path] = digest
    for proof in proofs:
        need(proof['path'] not in bindings or bindings[proof['path']] == proof['sha256'], 'Conflicting proof binding')
        bindings[proof['path']] = proof['sha256']
    for path in paths:
        reference = ref(path)
        need(reference['path'] not in bindings or bindings[reference['path']] == reference['sha256'],
             'Original immutable code/input binding changed')
        bindings[reference['path']] = reference['sha256']
    return bindings


def build_configs(olds, records, bindings):
    configs = []
    manifest_path = artifact('_manifest')
    for old, spec, record in zip(olds, SPECS, records):
        cfg = native.build_config(old, cfg_path(spec), spec['new'], record['owners'], artifact('_remaining'),
                                  artifact('_handoff'), record['resource_previous_safe'], bindings)
        cfg.update(producer=ref(manager.SELF), resource_manager_ref=ref(manager.SELF),
                   pool_root=str(POOL_ROOT), pool_manifest=dict(path=str(manifest_path), sha256='0' * 64),
                   pool_worker_id=spec['new'], pool_group_claim=ref(artifact('_claim')),
                   pool_old_parked=ref(phase_path(spec, 'parked')))
        cfg['argv'] = [native.PYTHON, '-B', str(manager.SELF), 'worker', '--config', cfg['config_path']]
        cfg['source_sha256'].update({str(manager.SELF): cfg['producer']['sha256'],
                                    str(Path(pool.__file__).resolve()): m.base.sha(pool.__file__),
                                    str(manifest_path): '0' * 64})
        configs.append(cfg)
    return configs


def local_indices(olds):
    ips = m.h.subprocess.check_output(['hostname', '-I'], text=True).split()
    nodes = {old['node'] for old in olds if m.HOSTS[old['node']] in ips}
    need(len(nodes) == 1, 'Prepare/validate must execute on exactly one old node')
    return [i for i, old in enumerate(olds) if old['node'] in nodes]


def prepare():
    with group_lock():
        try:
            olds = load_olds()
            local_ids = local_indices(olds)
            group = group_claim(olds)
            need(not artifact('_prepare_claim').exists(), 'Prepare already attempted; no retry')
            outputs_absent()
            write_once(artifact('_prepare_claim'), dict(group_claim=group, transition=ref(SELF)))
            records = parked_records(olds, group)
            capacity()
            for i in local_ids:
                scope = rebalance.lane_scope(olds[i])
                scope['old_tree_absent'](records[i]['expected_old_tree'])
                scope['safe_parent_wait'](olds[i], records[i]['manager'])
                need(scope['restored_owners'](olds[i]) == records[i]['owners'], 'Fresh local holders changed')
            rows, detail, progress_proofs = remaining_union(olds)
            all_proofs = progress_proofs + [proof for record in records for proof in record['proofs']]
            write_once(artifact('_remaining'), rows)
            write_once(artifact('_handoff'), dict(remaining_sources=rows, old_worker_tree_absent=True,
                original_config_resume_forbidden=True, old_worker_tree_absence_scope=ABSENCE_SCOPE,
                safe_old_manager_quiescent_wait_allowed=True, old_configs=[ref(cfg_path(s)) for s in SPECS],
                old_plans=[old['plan'] for old in olds], old_native_parked=[ref(phase_path(s, 'parked')) for s in SPECS],
                old_proofs=all_proofs, resource_safety_proofs=[r['resource_previous_safe'] for r in records],
                remaining_plan=ref(artifact('_remaining')), resource_budget=BUDGET, group_claim=group, **detail))
            paths = [SELF, Path(rebalance.__file__).resolve()] + [RT / name for name in FROZEN]
            paths += [RT / name for name in ('caption_local_source_bundle_v1.py', 'caption_two_stage_light_v1.py',
                      'caption_direct_facts_v1.py', 'visual_batch_fixed3_benchmark_v1.py',
                      'bench_manager_driver_v2.py', 'curated_resource_manager.py')]
            paths += [Path(old['modelref']['path']) for old in olds]
            paths += [artifact(s) for s in ('_claim', '_prepare_claim', '_remaining', '_handoff')]
            paths += [p for s in SPECS for p in (cfg_path(s), old_claim(s), phase_path(s, 'park_claim'), phase_path(s, 'parked'))]
            bindings = merge_bindings(records, all_proofs, paths)
            configs = build_configs(olds, records, bindings)
            participants = [dict(worker_id=cfg['pool_worker_id'], config_path=cfg['config_path'],
                                node=cfg['node'], gpus=cfg['gpus'], config_identity_sha256=pool.config_identity(cfg)) for cfg in configs]
            write_once(artifact('_manifest'), dict(schema='caption_ready_pool_v1', plan=ref(artifact('_remaining')),
                       source_handoff=ref(artifact('_handoff')), source_root=olds[0]['source_root'],
                       pool_root=str(POOL_ROOT), participants=participants))
            for cfg in configs:
                cfg['pool_manifest'] = ref(artifact('_manifest'))
                cfg['source_sha256'][str(artifact('_manifest'))] = cfg['pool_manifest']['sha256']
                write_once(Path(cfg['config_path']), cfg)
                pool.check_manifest(cfg)
            rebalance.check_shared_files(all_proofs)
            write_once(artifact('_prepared'), dict(group_claim=group, transition=ref(SELF),
                       configs=[ref(cfg_path(s, True)) for s in SPECS], pool_manifest=ref(artifact('_manifest')),
                       handoff=ref(artifact('_handoff')), plan=ref(artifact('_remaining'))))
            for i in local_ids:
                _validate(cfg_path(SPECS[i], True), olds, group, records)
            result = dict(event='pool_prepared', N=len(rows), local_node=olds[local_ids[0]]['node'],
                          all_nodes_validate_before_any_launch=True, other_node_validation_required=True,
                          other_node_configs=[str(cfg_path(SPECS[i], True)) for i in range(4) if i not in local_ids],
                          native_manager_argv=[native.manager_argv(cfg_path(s, True), manager.SELF,
                                               m.base.sha(cfg_path(s, True))) for s in SPECS])
            print(json.dumps(result), flush=True)
            return configs
        except BaseException as error:
            fail(error)
            raise


def _validate(path, olds, group, records):
    i = select(path, new=True)
    old, spec, record = olds[i], SPECS[i], records[i]
    scope = rebalance.lane_scope(old)
    scope['check_node'](old)
    prepared = load(artifact('_prepared'))
    need(prepared['group_claim'] == group and prepared['transition'] == ref(SELF)
         and prepared['configs'][i] == ref(path) and prepared['pool_manifest'] == ref(artifact('_manifest'))
         and prepared['handoff'] == ref(artifact('_handoff')) and prepared['plan'] == ref(artifact('_remaining')),
         'Prepared artifacts/config changed')
    scope['old_tree_absent'](record['expected_old_tree'])
    scope['safe_parent_wait'](old, record['manager'])
    need(scope['restored_owners'](old) == record['owners'], 'Latest sole restored holders differ')
    capacity()
    cfg = load(path)
    handoff = load(artifact('_handoff'))
    need(cfg['source_handoff'] == prepared['handoff'] and cfg['plan'] == prepared['plan']
         and cfg['pool_manifest'] == prepared['pool_manifest'] and cfg['pool_root'] == str(POOL_ROOT)
         and cfg['pool_worker_id'] == spec['new'] and cfg['pool_group_claim'] == group
         and cfg['pool_old_parked'] == ref(phase_path(spec, 'parked'))
         and cfg['previous_config'] == ref(cfg_path(spec)) and cfg['old_out'] == old['draft_out']
         and cfg['resource_previous_safe'] == record['resource_previous_safe'] and cfg['holders'] == record['owners']
         and cfg['source_root'] == old['source_root'] and cfg['modelref'] == old['modelref']
         and cfg['node'] == old['node'] and cfg['gpus'] == old['gpus'] and cfg['env'] == old['env']
         and cfg['phase'] == spec['new'] and cfg['config_path'] == str(Path(path).resolve()) and cfg['cwd'] == str(RT)
         and cfg['draft_out'] == str(BAR / spec['new']) and cfg['out'] == str(BAR / (spec['new'] + '_execution'))
         and cfg['queue'] == str(BAR / 'queues' / spec['new'])
         and cfg['producer'] == cfg['resource_manager_ref'] == ref(manager.SELF)
         and cfg['argv'] == [native.PYTHON, '-B', str(manager.SELF), 'worker', '--config', str(path)]
         and cfg['targetcount'] == cfg['input_count'] == len(handoff['remaining_sources'])
         and cfg['release_authorized'] is True and cfg['reuse_existing_results'] is False
         and handoff['resource_budget'] == BUDGET and handoff['group_claim'] == group, 'New fixed pool lane assignment differs')
    rebalance.check_shared_files(handoff['old_proofs'])
    manifest, rows = pool.check_manifest(cfg)
    need(rows == handoff['remaining_sources'] and handoff['remaining_plan'] == cfg['plan']
         and handoff['old_native_parked'] == [ref(phase_path(s, 'parked')) for s in SPECS]
         and len(manifest['participants']) == 4, 'Complete fixed union/four participants required')
    for participant, assigned in zip(manifest['participants'], SPECS):
        need(participant['worker_id'] == assigned['new'] and participant['node'] == assigned['node']
             and participant['gpus'] == assigned['gpus'] and participant['config_path'] == str(cfg_path(assigned, True)),
             'Fixed pool manifest participant differs')
    manager.validate(cfg)
    check_sources(cfg, Path(path), manager.selected)
    argv = native.manager_argv(Path(path), manager.SELF, m.base.sha(path))
    receipt_path = RT / (spec['new'] + '_validated.json')
    receipt = dict(config=ref(path), group_claim=group, node=old['node'], transition=ref(SELF),
                   handoff=prepared['handoff'], pool_manifest=prepared['pool_manifest'], native_manager_argv=argv,
                   config_identity_sha256=pool.config_identity(cfg), fresh_native_capacity_passed=True,
                   current_sole_holders_locks_leases_parent_wait_checked=True,
                   all_four_realnode_validations_required_before_any_launch=True)
    if receipt_path.exists():
        need(load(receipt_path) == receipt, 'Validated receipt changed')
    else:
        write_once(receipt_path, receipt)
    print(json.dumps(dict(event='pool_native_validated', phase=cfg['phase'], native_manager_argv=argv)), flush=True)
    return argv


def validate_new(path):
    i = select(path, new=True)
    with group_lock():
        try:
            olds = load_olds()
            rebalance.lane_scope(olds[i])['check_node'](olds[i])
            group = group_claim(olds)
            return _validate(Path(path).resolve(), olds, group, parked_records(olds, group))
        except BaseException as error:
            fail(error)
            raise


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--park-old-config')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--validate-new-config')
    args = parser.parse_args()
    if args.park_old_config:
        park(args.park_old_config)
    elif args.prepare:
        prepare()
    else:
        validate_new(args.validate_new_config)


if __name__ == '__main__':
    main()
