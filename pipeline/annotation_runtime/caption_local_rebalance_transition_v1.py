"""Two fixed native lanes: park on each host, prepare union, validate on each host.
No remote execution, model/media reads, retries, or changes to frozen modules.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
import caption_light_to_local_transition_v1 as native
import caption_light_to_local_transition_v2 as checked

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
m = native.m
need, load, write_once = native.need, native.load, native.write_once
GROUP = 'caption_local_rebalance_v1'
SPECS = (
    dict(old='caption_two_stage_live_n4c3_v3', new='caption_two_stage_local_n4c3_balanced_v1',
         config_sha='214a423c6145d8d1f41aa62d8dbfc18f3a44541d255c1762788e452197e7f3c2',
         producer='caption_two_stage_live_manager_v3.py', producer_sha='2076912d5b9b57c98e815affbbbe5f5d47e080e3493f9cdcef12071870b37687'),
    dict(old='caption_two_stage_local_n6_v1', new='caption_two_stage_local_n6_balanced_v1',
         config_sha='834ca59769429a82fe366602bd5640b987a77816956c641d67a33c80d7090cdc',
         producer='caption_local_live_manager_v1.py', producer_sha=checked.LOCAL_SHA),
)
CHECKED_SHA = 'ce6ef08c9173933133266a213c8521863c372acfa25b3bf71df97bc92182dd72'
BUDGET = dict(existing_copy_threads=5, retired_copy_threads=1, new_copy_threads=2,
    total_copy_threads=6, total_n4_copy_threads=4, total_n6_copy_threads=1,
    copy_threads_per_instance=1, source_batch_size=3, max_inflight_batches=2)
ABSENCE_SCOPE = 'recorded direct child and preconfirmed GPU model apps only; not all CPU descendants'


def artifact(suffix):
    return RT / (GROUP + suffix + '.json')


def old_path(spec):
    return RT / (spec['old'] + '_launch.json')


def load_pair():
    need(m.base.sha(native.SELF) == checked.FROZEN_SHA and m.base.sha(checked.WRAPPER) == CHECKED_SHA
         and m.base.sha(m.SELF) == checked.LOCAL_SHA, 'Frozen transition/local helpers changed')
    olds = []
    for spec in SPECS:
        path = old_path(spec)
        need(m.base.sha(path) == spec['config_sha'], 'Wrong frozen old config')
        old = load(path)
        need(old['config_path'] == str(path) and old['phase'] == spec['old']
             and old['producer'] == old['resource_manager_ref'] == m.ref(RT / spec['producer'])
             and old['producer']['sha256'] == spec['producer_sha'], 'Wrong old worker identity/path/SHA')
        olds.append(old)
    need(olds[0]['source_root'] == olds[1]['source_root'] and olds[0]['modelref'] == olds[1]['modelref'],
         'Two old source roots/model refs differ')
    return olds


def lane_scope(old):
    scope = dict(vars(native), LIGHT_MANAGER=Path(old['producer']['path']))
    for name, function in vars(native).items():
        if isinstance(function, types.FunctionType) and function.__globals__ is vars(native):
            scope[name] = types.FunctionType(function.__code__, scope, function.__name__,
                                            function.__defaults__, function.__closure__)
    generator = native.handoff_lock.__wrapped__
    scope['handoff_lock'] = contextmanager(types.FunctionType(
        generator.__code__, scope, generator.__name__, generator.__defaults__, generator.__closure__))
    return scope


def capacity():
    return dict(native.capacity(), **BUDGET)


def check_budget(budget):
    need(all(budget.get(key) == value for key, value in BUDGET.items()), 'Rebalance copy budget changed')
    need(budget['memory']['headroom'] >= native.MIN_HEADROOM
         and budget['local_disk_free_bytes'] >= native.MIN_DISK, 'Rebalance native capacity gate failed')


@contextmanager
def pair_lock():
    with ExitStack() as stack:
        stack.enter_context(native.handoff_lock(GROUP))
        for spec in SPECS:
            stack.enter_context(native.handoff_lock(spec['old']))
        yield


def new_outputs_absent():
    for spec in SPECS:
        for suffix in ('_launch.json', '_remaining.json'):
            need(not (RT / (spec['new'] + suffix)).exists(), 'New immutable artifact already exists')
        for path in (m.BAR / spec['new'], m.BAR / (spec['new'] + '_execution'), m.BAR / 'queues' / spec['new']):
            need(not path.exists(), 'New output/queue already exists')


def group_claim(olds, create=False):
    need(not artifact('_failed').exists(), 'Rebalance failed; no automatic retry/resume')
    path = artifact('_claim')
    expected = dict(old_configs=[m.ref(old_path(spec)) for spec in SPECS],
        new_names=[spec['new'] for spec in SPECS], transition=m.ref(SELF),
        frozen_transition=m.ref(native.SELF), resource_budget=BUDGET, original_config_resume_forbidden=True)
    if not path.exists():
        need(create, 'Both old lanes must be claimed/parked before prepare')
        new_outputs_absent()
        for spec in SPECS:
            need(not (RT / (spec['old'] + '_light_to_local_handoff_claim.json')).exists(),
                 'Old phase already claimed; no source budget replay')
        write_once(path, expected)
        for spec in SPECS:
            write_once(RT / (spec['old'] + '_light_to_local_handoff_claim.json'),
                dict(old_config=m.ref(old_path(spec)), name=spec['new'], group_claim=m.ref(path),
                     transition=m.ref(SELF), original_config_resume_forbidden=True))
    need(load(path) == expected, 'Rebalance claim/config/budget identity changed')
    for spec in SPECS:
        claim = load(RT / (spec['old'] + '_light_to_local_handoff_claim.json'))
        need(claim == dict(old_config=m.ref(old_path(spec)), name=spec['new'], group_claim=m.ref(path),
            transition=m.ref(SELF), original_config_resume_forbidden=True), 'Shared old phase claim differs')
    return m.ref(path)


def fail(error):
    write_once(artifact('_failed'), dict(reason=type(error).__name__ + ': ' + str(error),
        original_config_resume_forbidden=True, locks_leases_budgets_not_deleted=True, no_automatic_retry=True))


def select_spec(path, new=False):
    path = Path(path).resolve()
    matches = [i for i, spec in enumerate(SPECS)
               if path == RT / (spec['new' if new else 'old'] + '_launch.json')]
    need(len(matches) == 1, 'Only the two fixed rebalance configs are supported')
    return matches[0]


def park(path):
    index = select_spec(path)
    olds = load_pair()
    old, spec = olds[index], SPECS[index]
    scope = lane_scope(old)
    scope['check_node'](old)
    with pair_lock():
        group = group_claim(olds, create=True)
        attempt = RT / (spec['old'] + '_rebalance_park_claim.json')
        need(not attempt.exists(), 'Old park already attempted; no automatic retry')
        write_once(attempt, dict(group_claim=group, old_config=m.ref(path), transition=m.ref(SELF)))
        try:
            for cfg, item in zip(olds, SPECS):
                checked.check_native_sources(cfg, old_path(item))
            budget = capacity()
            check_budget(budget)
            queue = Path(old['queue'])
            proofs = [native.file_identity(old_path(spec)), native.file_identity(old['plan']['path']),
                      native.file_identity(queue / 'admission.json'), native.file_identity(queue / 'live.json')]
            parent, child = scope['check_old_receipts'](old, old_path(spec), load(queue / 'admission.json'), load(queue / 'live.json'))
            bindings = scope['code_bindings'](old)
            bindings.update({str(SELF): m.base.sha(SELF), str(checked.WRAPPER): CHECKED_SHA})
            tree_path = RT / (spec['new'] + '_old_tree.json')
            expected, safe = scope['park_old'](old, parent, child, proofs, tree_path)
            owners = scope['restored_owners'](old)
            scope['old_tree_absent'](expected)
            scope['safe_parent_wait'](old, parent)
            for proof_path in [tree_path, Path(safe['path'])] + [queue / ('holder_restored_gpu%d.json' % gpu) for gpu in old['gpus']]:
                proofs.append(native.file_identity(proof_path))
            native.check_files(proofs)
            record = dict(group_claim=group, old_config=m.ref(old_path(spec)), old_phase=old['phase'],
                manager=parent, expected_old_tree=expected, old_worker_tree_absent=True,
                old_worker_tree_absence_scope=ABSENCE_SCOPE, safe_old_manager_quiescent_wait_allowed=True,
                resource_previous_safe=safe, owners=owners, proofs=proofs, source_bindings=bindings, resource_budget=budget)
            write_once(RT / (spec['old'] + '_rebalance_parked.json'), record)
            print(json.dumps(dict(event='rebalance_native_parked', phase=old['phase'], record=record['old_config'], no_new_model=True)), flush=True)
            return record
        except BaseException as error:
            fail(error)
            raise


def plan_rows(old):
    need(m.ref(old['plan']['path']) == old['plan'], 'Original selected plan changed')
    rows = load(old['plan']['path'])
    keys = [native.row_key(row) for row in rows]
    need(rows and len(keys) == len(set(keys)) == old['targetcount'], 'Old selected plan keys/count differ')
    return rows


def local_exclusions(old, rows):
    root = Path(old['draft_out'])
    allowed = {native.row_key(row) for row in rows}
    progress_path = root / 'progress.json'
    proofs = [native.file_identity(progress_path)]
    progress = load(progress_path)
    complete = [native.row_key(dict(id=key[0], split=key[1])) for key in progress['completed_source_keys']]
    need(len(complete) == len(set(complete)) == progress['processed'] and set(complete) <= allowed
         and progress['selected'] == len(rows), 'Old local progress keys/count differ')
    reserved, committed, index = set(), set(), {}
    budgets = root / 'budgets'
    need(budgets.is_dir() and not budgets.is_symlink(), 'Missing/unsafe old local budget directory')
    with os.scandir(budgets) as entries:
        batches = list(entries)
    for entry in batches:
        need(entry.is_dir(follow_symlinks=False) and entry.name.startswith('batch-')
             and all(c.isascii() and (c.isalnum() or c in '_-') for c in entry.name), 'Unsafe old local budget entry')
        claim_path = budgets / entry.name / 'claim.json'
        need(not claim_path.is_symlink(), 'Unsafe old local claim')
        proofs.append(native.file_identity(claim_path))
        claim = load(claim_path)
        keys = [native.row_key(row) for row in claim['metas']]
        need(keys and len(keys) == len(set(keys)) and set(keys) <= allowed
             and claim.get('reserved_sources', len(keys)) == len(keys), 'Old local claim keys/count differ')
        reserved.update(keys)
        commit_path = root / (entry.name + '.committed.json')
        is_committed = commit_path.exists()
        if is_committed:
            proofs.append(native.file_identity(commit_path))
            commit = load(commit_path)
            need(commit['batch_id'] == entry.name and commit['reliable_writeback'] is True
                 and commit['archive_path'] == str(root / (entry.name + '.tar')), 'Old local commit receipt differs')
            committed.update(keys)
        index[entry.name] = dict(metas=claim['metas'], claim=m.ref(claim_path), committed=is_committed)
    details = dict(completed_source_keys=sorted(complete), reserved_source_keys=sorted(reserved),
        reserved_pending=sorted(reserved - set(complete)), reserved_uncommitted=sorted(reserved - committed),
        committed_without_progress=sorted(committed - set(complete)), budget_index=index,
        old_progress_identity=proofs[0], budget_not_reset=True, old_inputs_tmp_preserved=True)
    native.check_files(proofs)
    return set(complete) | reserved, details, proofs


def remaining_union(olds):
    plans = [plan_rows(old) for old in olds]
    _, excluded_live, directory_index = native.remaining_sources(olds[0])
    excluded_local, local_detail, proofs = local_exclusions(olds[1], plans[1])
    excluded = set(map(tuple, excluded_live)) | excluded_local
    union, owner = {}, {}
    for lane, rows in enumerate(plans):
        for row in rows:
            key = native.row_key(row)
            need(key not in union or union[key] == row, 'Conflicting union source metadata')
            union.setdefault(key, row)
            owner.setdefault(key, lane)
    need(len({key[0] for key in union}) == len(union), 'Native ready selector requires unique source IDs')
    remaining = [row for key, row in union.items() if key not in excluded]
    need(remaining, 'No unstarted union sources')
    details = dict(excluded_source_keys=sorted(excluded), live_directory_index=directory_index,
        old_local=local_detail, excluded_started_without_draft=sorted(
            set(map(tuple, native.exclusion_details(directory_index)['excluded_started_without_draft']))
            | set(map(tuple, local_detail['reserved_pending']))),
        reserved_pending=local_detail['reserved_pending'], original_union_count=len(union), budget_not_reset=True)
    return remaining, owner, details, proofs


def ready_index(root):
    root = Path(root)
    paths = [root / 'ready' / ('producer-%04d.jsonl' % i) for i in range(4)] if (root / 'ready').is_dir() else [root / 'ready.jsonl']
    ready, offsets = set(), []
    for path in paths:
        if not path.exists():
            continue
        data = path.read_bytes()
        end = data.rfind(b'\n') + 1
        prefix = data[:end]
        for line in prefix.splitlines():
            ready.add(native.row_key(m.local.light.schemas.strict(line)))
        offsets.append(dict(path=str(path), complete_prefix_bytes=end, sha256=hashlib.sha256(prefix).hexdigest()))
    return ready, offsets


def balanced_plans(remaining, owners, ready):
    ready_rows = [row for row in remaining if native.row_key(row) in ready]
    parts = [ready_rows[::2], ready_rows[1::2]]
    for row in remaining:
        key = native.row_key(row)
        if key not in ready:
            parts[owners[key]].append(row)
    keys = [{native.row_key(row) for row in part} for part in parts]
    need(all(parts) and not keys[0] & keys[1] and keys[0] | keys[1] == {native.row_key(row) for row in remaining},
         'Nonempty exclusive union partition required')
    return parts, [len(ready_rows[::2]), len(ready_rows[1::2])]


def check_shared_files(proofs):
    for proof in proofs:
        current = native.file_identity(proof['path'])
        need({k: v for k, v in current.items() if k != 'device'} ==
             {k: v for k, v in proof.items() if k != 'device'},
             'Shared proof/config/progress changed after capture')


def parked_records(olds, group):
    records = []
    for old, spec in zip(olds, SPECS):
        record = load(RT / (spec['old'] + '_rebalance_parked.json'))
        need(record['group_claim'] == group and record['old_config'] == m.ref(old_path(spec))
             and record['old_worker_tree_absent'] is True and record['old_worker_tree_absence_scope'] == ABSENCE_SCOPE,
             'Old native parked proof differs')
        check_shared_files(record['proofs'])
        need(native.safe_proof(Path(old['queue']) / 'resources_safe.json') == record['resource_previous_safe'],
             'Latest native safe differs from parked proof')
        check_budget(record['resource_budget'])
        records.append(record)
    return records


def prepare():
    olds = load_pair()
    with pair_lock():
        group = group_claim(olds)
        need(not artifact('_prepare_claim').exists(), 'Union prepare already attempted; no automatic retry')
        new_outputs_absent()
        write_once(artifact('_prepare_claim'), dict(group_claim=group, transition=m.ref(SELF)))
        try:
            records = parked_records(olds, group)
            remaining, owners, detail, progress_proofs = remaining_union(olds)
            ready, offsets = ready_index(olds[0]['source_root'])
            parts, ready_counts = balanced_plans(remaining, owners, ready)
            ready_path = artifact('_ready_snapshot')
            write_once(ready_path, dict(ready_source_keys=sorted(ready), complete_prefixes=offsets))
            plan_paths = [RT / (spec['new'] + '_remaining.json') for spec in SPECS]
            for path, rows in zip(plan_paths, parts):
                write_once(path, rows)
            handoff_path = artifact('_handoff')
            all_proofs = progress_proofs + [proof for record in records for proof in record['proofs']]
            write_once(handoff_path, dict(remaining_sources=remaining, old_worker_tree_absent=True,
                original_config_resume_forbidden=True, old_worker_tree_absence_scope=ABSENCE_SCOPE,
                safe_old_manager_quiescent_wait_allowed=True, old_configs=[m.ref(old_path(spec)) for spec in SPECS],
                old_plans=[old['plan'] for old in olds], old_native_parked=[m.ref(RT / (spec['old'] + '_rebalance_parked.json')) for spec in SPECS],
                old_proofs=all_proofs, resource_safety_proofs=[record['resource_previous_safe'] for record in records],
                remaining_plans=[m.ref(path) for path in plan_paths], ready_snapshot=m.ref(ready_path),
                assigned_ready_counts=ready_counts, all_unready_retained=True, unready_kept_with_original_owner=True,
                resource_budget=BUDGET, group_claim=group, **detail))
            bindings = {}
            for record in records:
                for path, digest in record['source_bindings'].items():
                    need(path not in bindings or bindings[path] == digest, 'Conflicting old source bindings')
                    bindings[path] = digest
            for proof in all_proofs:
                bindings[proof['path']] = proof['sha256']
            for path in [SELF, checked.WRAPPER, artifact('_claim'), artifact('_prepare_claim'), handoff_path, ready_path] + plan_paths + [RT / (spec['old'] + suffix) for spec in SPECS for suffix in ('_rebalance_parked.json', '_light_to_local_handoff_claim.json')]:
                bindings[str(path)] = m.base.sha(path)
            configs = []
            for old, spec, record, plan in zip(olds, SPECS, records, plan_paths):
                cfg = native.build_config(old, old_path(spec), spec['new'], record['owners'], plan,
                    handoff_path, record['resource_previous_safe'], bindings)
                cfg['rebalance_group_claim'] = group
                cfg['rebalance_old_parked'] = m.ref(RT / (spec['old'] + '_rebalance_parked.json'))
                write_once(Path(cfg['config_path']), cfg)
                checked.check_native_sources(cfg, Path(cfg['config_path']))
                configs.append(cfg)
            check_shared_files(all_proofs)
            # Hardware/physical host checks cannot be truthfully run for both hosts in one process.
            local_ip = m.h.subprocess.check_output(['hostname', '-I'], text=True).split()
            local_indices = [i for i, old in enumerate(olds) if m.HOSTS[old['node']] in local_ip]
            need(len(local_indices) == 1, 'Prepare must run on N4 or N6; no remote-host validation fiction')
            validate_new(configs[local_indices[0]]['config_path'])
            result = dict(event='rebalance_prepared', remaining=len(remaining), ready_counts=ready_counts,
                native_validation_required_on_each_host=True, both_hosts_validate_before_either_launch=True, native_manager_argv=[
                    native.manager_argv(Path(cfg['config_path']), m.SELF, m.base.sha(cfg['config_path'])) for cfg in configs])
            print(json.dumps(result), flush=True)
            return configs
        except BaseException as error:
            fail(error)
            raise


def validate_new(path):
    index = select_spec(path, new=True)
    olds = load_pair()
    old = olds[index]
    native.check_node(old)
    group = group_claim(olds)
    records = parked_records(olds, group)
    record = records[index]
    scope = lane_scope(old)
    scope['old_tree_absent'](record['expected_old_tree'])
    scope['safe_parent_wait'](old, record['manager'])
    check_budget(capacity())
    cfg = load(path)
    need(cfg['source_handoff'] == m.ref(artifact('_handoff')) and cfg['previous_config'] == m.ref(old_path(SPECS[index]))
         and cfg['resource_previous_safe'] == record['resource_previous_safe']
         and cfg['holders'] == record['owners'] and cfg['phase'] == SPECS[index]['new']
         and cfg['source_root'] == old['source_root'] and cfg['modelref'] == old['modelref'], 'New rebalance native config differs')
    handoff = load(cfg['source_handoff']['path'])
    need(handoff['resource_budget'] == BUDGET, 'New handoff copy budget differs')
    name = SPECS[index]['new']
    need(cfg['plan'] == handoff['remaining_plans'][index] and cfg['node'] == old['node']
         and cfg['gpus'] == old['gpus'] and cfg['env'] == old['env']
         and cfg['config_path'] == str(Path(path).resolve()) and cfg['draft_out'] == str(m.BAR / name)
         and cfg['out'] == str(m.BAR / (name + '_execution')) and cfg['queue'] == str(m.BAR / 'queues' / name),
         'New exclusive plan/node/GPU/output assignment differs')
    check_shared_files(handoff['old_proofs'])
    checked.check_native_sources(cfg, Path(path))
    m.validate(cfg)
    argv = native.manager_argv(Path(path), m.SELF, m.base.sha(path))
    print(json.dumps(dict(event='rebalance_native_validated', phase=cfg['phase'], native_manager_argv=argv)), flush=True)
    return argv


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
