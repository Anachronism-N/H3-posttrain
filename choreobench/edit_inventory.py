"""Audit an independent task/method/seed inventory. No models, scores or retries."""
import argparse
import copy
from pathlib import Path

from .editing import apply_task_patch, validate_edit_task
from .evaluation import evaluate_edit
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .schema import identifier, keys, text, validate_dataset, validate_plan


def _read(path, sources, role, expected=None):
    path = Path(path).resolve()
    value, sha = read_json_snapshot(path)
    if expected is not None and sha != expected:
        raise ValueError('Inventory source hash changed: ' + role)
    sources.append(dict(role=role, path=str(path), sha256=sha))
    return value, sha


def _reference(ref, base, sources, role):
    keys(ref, ('path', 'sha256'), where='inventory reference')
    text(ref['path'], 'reference.path')
    if not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64 or any(c not in '0123456789abcdef' for c in ref['sha256']):
        raise ValueError('Reference sha256 must be a lowercase byte hash')
    path = Path(ref['path']).expanduser()
    return _read(path if path.is_absolute() else base / path, sources, role, ref['sha256'])[0]


def audit_edit_inventory(inventory_path, attempts_path):
    """Keep task and seed results nested in songs, including every missing cell.

    The independent inventory binds original datasets, source plans and private
    tasks. A method's attempt list must never supply the task denominator.
    Semantic adjudication is deliberately not inferred from a valid patch.
    """
    sources = []
    inventory_path, attempts_path = Path(inventory_path).resolve(), Path(attempts_path).resolve()
    inventory, inventory_sha = _read(inventory_path, sources, 'inventory')
    keys(inventory, ('schema', 'inventory_id', 'methods', 'seeds', 'tasks'), where='edit inventory')
    if inventory['schema'] != 'choreobench.edit_inventory.v1':
        raise ValueError('Unsupported edit inventory schema')
    identifier(inventory['inventory_id'], 'inventory_id')
    methods, seeds = inventory['methods'], inventory['seeds']
    if not isinstance(methods, list) or not methods:
        raise ValueError('Inventory requires declared methods')
    for method in methods:
        identifier(method, 'method_id')
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate inventory method')
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Inventory requires unique integer seeds')
    if not isinstance(inventory['tasks'], list) or not inventory['tasks']:
        raise ValueError('Inventory requires independent tasks')
    tasks, songs, song_groupings = {}, {}, {}
    for entry in inventory['tasks']:
        keys(entry, ('task_id', 'dataset', 'source_plan', 'task'), where='inventory task')
        text(entry['task_id'], 'task_id')
        tid = entry['task_id']
        if tid in tasks:
            raise ValueError('Duplicate inventory task ID')
        dataset = _reference(entry['dataset'], inventory_path.parent, sources, tid + ':dataset')
        validate_dataset(dataset)
        before = _reference(entry['source_plan'], inventory_path.parent, sources, tid + ':source')
        task = _reference(entry['task'], inventory_path.parent, sources, tid + ':task')
        validate_edit_task(task, before)
        if task['task_id'] != tid:
            raise ValueError('Inventory task ID differs from bound original task')
        song = next((s for s in dataset['songs'] if s['song_id'] == task['song_id']), None)
        if song is None:
            raise ValueError('Task song is absent from its original dataset')
        validate_plan(before, song)
        signature = object_sha(song)
        if song['song_id'] in songs and songs[song['song_id']] != signature:
            raise ValueError('Same song ID has inconsistent original evidence')
        grouping_signature = object_sha(dataset['grouping'])
        if song['song_id'] in song_groupings and song_groupings[song['song_id']] != grouping_signature:
            raise ValueError('Same song ID has inconsistent dataset grouping declarations')
        songs[song['song_id']] = signature
        song_groupings[song['song_id']] = grouping_signature
        tasks[tid] = dict(task=task, before=before, song=song, dataset_sha256=entry['dataset']['sha256'],
            source_plan_sha256=object_sha(before), task_sha256=object_sha(task),
            grouping=copy.deepcopy(dataset['grouping']), synthetic=dataset['synthetic'])
    attempts, attempts_sha = _read(attempts_path, sources, 'attempts')
    keys(attempts, ('schema', 'inventory_sha256', 'attempts'), where='edit attempts')
    if attempts['schema'] != 'choreobench.edit_attempts.v1' or attempts['inventory_sha256'] != inventory_sha:
        raise ValueError('Attempts bind a different inventory byte snapshot')
    if not isinstance(attempts['attempts'], list):
        raise ValueError('Attempts must be a list')
    by_key = {}
    for row in attempts['attempts']:
        keys(row, ('task_id', 'method_id', 'seed', 'status', 'after', 'patch', 'error'), where='edit attempt')
        text(row['task_id'], 'attempt.task_id');identifier(row['method_id'], 'attempt.method_id')
        if type(row['seed']) is not int:
            raise ValueError('Attempt seed must be integer')
        key = (row['task_id'], row['method_id'], row['seed'])
        if row['task_id'] not in tasks or row['method_id'] not in methods or row['seed'] not in seeds or key in by_key:
            raise ValueError('Unknown or duplicate task/method/seed attempt')
        if row['status'] == 'failed':
            text(row['error'], 'attempt.error')
            if row['after'] is not None or row['patch'] is not None:
                raise ValueError('Failed attempt cannot publish a scored partial plan')
        elif row['status'] == 'returned':
            if row['error'] is not None or row['after'] is None or row['patch'] is None:
                raise ValueError('Returned attempt requires exact after/patch references and no error')
        else:
            raise ValueError('Attempt status must be returned/failed; omitted attempts stay missing')
        by_key[key] = row
    cells = []
    for tid, bound in tasks.items():
        for method in methods:
            for seed in seeds:
                row = by_key.get((tid, method, seed))
                cell = dict(task_id=tid, song_id=bound['song']['song_id'], method_id=method, seed=seed,
                    dataset_sha256=bound['dataset_sha256'], source_plan_sha256=bound['source_plan_sha256'],
                    task_sha256=bound['task_sha256'], status='missing_attempt', mechanical_valid=None,
                    empty_operations=None, no_op=None, literal_joint_success=None,
                    semantic_status='not_evaluated', joint_success=None, errors=[])
                if row is not None and row['status'] == 'failed':
                    cell.update(status='declared_failed', mechanical_valid=False, joint_success=False, errors=[row['error']])
                elif row is not None:
                    after = _reference(row['after'], attempts_path.parent, sources, tid + ':' + method + ':' + str(seed) + ':after')
                    patch = _reference(row['patch'], attempts_path.parent, sources, tid + ':' + method + ':' + str(seed) + ':patch')
                    cell['after_plan_sha256'] = object_sha(after)
                    try:
                        expected = apply_task_patch(bound['before'], patch, bound['task'])
                        if object_sha(expected) != object_sha(after):
                            raise ValueError('Returned plan differs from exact execution of its bound patch')
                    except (ValueError, TypeError, KeyError) as error:
                        cell.update(status='invalid_output', mechanical_valid=False, joint_success=False, errors=[str(error)])
                    else:
                        evaluation = evaluate_edit(bound['before'], after, bound['task'])
                        cell.update(mechanical_valid=True, empty_operations=not patch['operations'],
                            no_op=evaluation['no_op'], literal_joint_success=evaluation['literal_joint_success'],
                            semantic_status=evaluation['semantic_status'], evaluation=evaluation)
                        if evaluation['literal_joint_success'] is False:
                            cell.update(status='literal_fail', joint_success=False)
                        elif bound['task']['semantic_goals']:
                            cell.update(status='returned_pending_semantics')
                        elif bound['task']['goals']:
                            cell.update(status='literal_pass_semantics_not_requested', joint_success=True)
                        else:
                            cell.update(status='not_evaluable')
                cells.append(cell)
    song_rows = []
    for sid in songs:
        song_cells = [cell for cell in cells if cell['song_id'] == sid]
        song_rows.append(dict(song_id=sid, task_ids=[tid for tid, b in tasks.items() if b['song']['song_id'] == sid],
            grouping=copy.deepcopy(next(b['grouping'] for b in tasks.values() if b['song']['song_id'] == sid)),
            results={method: dict(expected_cells=sum(c['method_id'] == method for c in song_cells),
                status_counts={status: sum(c['method_id'] == method and c['status'] == status for c in song_cells)
                    for status in sorted({c['status'] for c in cells})},
                mechanically_valid=sum(c['method_id'] == method and c['mechanical_valid'] is True for c in song_cells),
                joint_successes=sum(c['method_id'] == method and c['joint_success'] is True for c in song_cells),
                joint_pending=sum(c['method_id'] == method and c['joint_success'] is None for c in song_cells)) for method in methods}))
    for ref in sources:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Edit inventory source changed while auditing')
    report = dict(schema='choreobench.edit_inventory_audit.v1', sources=sources,
        inventory_sha256=inventory_sha, attempts_sha256=attempts_sha, expected_cells=len(cells),
        recorded_attempts=len(by_key), task_count=len(tasks), song_count=len(songs), cells=cells, songs=song_rows,
        any_dataset_synthetic=any(bound["synthetic"] for bound in tasks.values()),
        model_execution_identity_verified=False,
        numeric_quality_or_ci=None, quality_verified=False, source_preregistration_or_failure_authentication=False,
        note='Independent inventory supplies the denominator. Task/seed cells remain nested within songs; no new independent samples or semantic successes from valid/no-op patches. No private targets enter a model.')
    report['record_sha256'] = object_sha(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--attempts', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit_edit_inventory(args.inventory, args.attempts)
    write_json(args.out, report)
    return report


if __name__ == '__main__':
    main()
