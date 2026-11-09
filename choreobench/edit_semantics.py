"""Independent per-goal edit judgments bound to anonymous original artifacts."""
import argparse
import copy
from pathlib import Path

from .edit_inventory import audit_edit_inventory
from .io import file_sha, object_sha, read_json_snapshot, write_json
from .reviews import import_reviews
from .schema import integer, keys, text


def _snapshot(path, role, sources):
    value, sha = read_json_snapshot(path)
    sources.append(dict(path=str(Path(path).resolve()), sha256=sha, role=role))
    return value, sha


def _unchanged(sources):
    for ref in sources:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Semantic review source changed: ' + ref['role'])


def _ranges(rows, samples):
    if not isinstance(rows, list):
        raise ValueError('Reviewed ranges must be a list')
    intervals = []
    for row in rows:
        keys(row, ('start', 'end'), where='reviewed range')
        integer(row['start'], 'range.start', 0);integer(row['end'], 'range.end', 1)
        if not row['start'] < row['end'] <= samples:
            raise ValueError('Reviewed range escapes original sample clock')
        intervals.append((row['start'], row['end']))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _required_range(before, after, task):
    permitted = set(task['allowed_event_ids']) | set(task['repair_event_ids'])
    ranges = []
    for plan in (before, after):
        events = plan['events']
        selected = [e for e in events if e['event_id'] in permitted]
        if not selected:
            ranges.append((0, plan['num_samples']))
            continue
        start = min(e['start'] for e in selected);end = max(e['end'] for e in selected)
        previous = [e for e in events if e['end'] <= start]
        following = [e for e in events if e['start'] >= end]
        if previous:
            start = min(e['start'] for e in previous if e['end'] == max(v['end'] for v in previous))
        if following:
            end = max(e['end'] for e in following if e['start'] == min(v['start'] for v in following))
        ranges.append((start, end))
    return [dict(start=min(a for a, b in ranges), end=max(b for a, b in ranges))]


def prepare_edit_semantics(inventory_path, attempts_path, public_path, bindings_path, mapping_path, output_dir):
    """Publish anonymous requirements, a private identity bridge and null template.

    Mapping is explicitly supplied, never inferred from song/method aliases.
    The evaluated goals are visible to reviewers only, not to generation.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    sources = []
    audit = audit_edit_inventory(inventory_path, attempts_path)
    sources.extend(audit['sources'])
    pack, pack_file_sha = _snapshot(public_path, 'public_pack', sources)
    bindings, bindings_sha = _snapshot(bindings_path, 'private_bindings', sources)
    import_reviews(pack, bindings, dict(schema='choreobench.reviews.v1', pack_sha256=object_sha(pack), reviews=[]))
    mapping, mapping_sha = _snapshot(mapping_path, 'explicit_mapping', sources)
    keys(mapping, ('schema', 'inventory_sha256', 'attempts_sha256', 'pack_sha256', 'cells'), where='semantic mapping')
    if (mapping['schema'] != 'choreobench.edit_semantic_mapping.v1'
            or mapping['inventory_sha256'] != audit['inventory_sha256']
            or mapping['attempts_sha256'] != audit['attempts_sha256'] or mapping['pack_sha256'] != object_sha(pack)):
        raise ValueError('Semantic mapping binds different original evidence')
    if not isinstance(mapping['cells'], list) or not mapping['cells']:
        raise ValueError('Explicit semantic mappings must be nonempty')
    cells = {(c['task_id'], c['method_id'], c['seed']): c for c in audit['cells']}
    private = {b['review_id']: b for b in bindings['items']}
    public = {p['review_id']: p for p in pack['items']}
    seen_reviews, seen_cells, requirements, bridge_cells = set(), set(), [], []
    for row in mapping['cells']:
        keys(row, ('review_id', 'task_id', 'method_id', 'seed'), where='semantic mapping cell')
        for name in ('review_id', 'task_id', 'method_id'):text(row[name], name)
        integer(row['seed'], 'mapping.seed', 0)
        key = (row['task_id'], row['method_id'], row['seed'])
        if row['review_id'] not in public or key not in cells or row['review_id'] in seen_reviews or key in seen_cells:
            raise ValueError('Unknown or duplicate mapping cell/review ID')
        seen_reviews.add(row['review_id']);seen_cells.add(key)
        cell = cells[key];original = private[row['review_id']]['original'];item = public[row['review_id']]
        if cell['mechanical_valid'] is not True or original.get('edit_task') is None or original.get('source_plan') is None:
            raise ValueError('Semantic mapping requires a mechanically valid bound edit item')
        task, before, after = original['edit_task'], original['source_plan'], original['plan']
        if (task['task_id'] != row['task_id'] or object_sha(task) != cell['task_sha256']
                or object_sha(before) != cell['source_plan_sha256'] or object_sha(after) != cell['after_plan_sha256']
                or original.get('source_binding', {}).get('dataset_sha256') != cell['dataset_sha256']):
            raise ValueError('Anonymous edit item differs from original task/before/after/dataset cell')
        requirement = dict(review_id=item['review_id'], item_sha256=item['item_sha256'],
            source_plan_sha256=object_sha(item['source_plan']), after_plan_sha256=object_sha(item['plan']),
            sample_rate=item['plan']['sample_rate'], num_samples=item['plan']['num_samples'],
            instruction=item['edit_task']['instruction'], required_review_ranges=_required_range(before, after, task),
            goals=[dict(goal_index=i, goal_sha256=object_sha(goal), description=goal)
                   for i, goal in enumerate(task['semantic_goals'])], boundary_review_required=True)
        requirements.append(requirement)
        bridge_cells.append(dict(row, task_sha256=cell['task_sha256'], before_sha256=cell['source_plan_sha256'],
            after_sha256=cell['after_plan_sha256'], requirement_sha256=object_sha(requirement)))
    if seen_reviews != set(public):
        raise ValueError('Explicit semantic mapping must cover every item in its selected edit review pack')
    requirements_doc = dict(schema='choreobench.edit_semantic_requirements.v1', pack_sha256=object_sha(pack), items=requirements,
        evaluation_only_not_generation_input=True, range_policy='Bounding edited/repair intervals plus nearest temporal neighbors in both before/after')
    template = dict(schema='choreobench.edit_semantic_verdicts.v1', requirements_sha256=object_sha(requirements_doc), records=[])
    bridge = dict(schema='choreobench.edit_semantic_bridge.v1', sources=sources, inventory_sha256=audit['inventory_sha256'],
        attempts_sha256=audit['attempts_sha256'], pack_file_sha256=pack_file_sha, bindings_file_sha256=bindings_sha,
        mapping_sha256=mapping_sha, requirements_sha256=object_sha(requirements_doc), cells=bridge_cells,
        no_model_calls=True, no_actual_reviewers_or_scores=True)
    bridge['record_sha256'] = object_sha(bridge)
    _unchanged(sources)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'requirements.json', requirements_doc)
    write_json(output/'bridge.private.json', bridge)
    write_json(output/'verdicts.none-submitted.json', template)
    draft = dict(template, records=[dict(review_id=r['review_id'], item_sha256=r['item_sha256'],
        source_plan_sha256=r['source_plan_sha256'], after_plan_sha256=r['after_plan_sha256'],
        reviewer_id='FILL_REVIEWER_ID', origin='human', model_id=None, reviewed_ranges=[],
        goals=[dict(goal_index=g['goal_index'], goal_sha256=g['goal_sha256'], status=None, evidence='') for g in r['goals']],
        boundary=dict(status=None, evidence=''), comments='') for r in requirements])
    write_json(output/'verdicts.template.json', draft)
    return dict(items=len(requirements), semantic_goal_cells=sum(len(v['goals']) for v in requirements),
        requirements_sha256=object_sha(requirements_doc), bridge_sha256=object_sha(bridge), new_model_calls=0, quality_verified=False)


def analyze_edit_semantics(inventory_path, attempts_path, requirements_path, bridge_path, verdicts_paths):
    """Return separate observer decisions, never majority/consensus or fake passes."""
    sources = []
    audit = audit_edit_inventory(inventory_path, attempts_path);sources.extend(audit['sources'])
    req, _ = _snapshot(requirements_path, 'requirements', sources)
    bridge, _ = _snapshot(bridge_path, 'identity_bridge', sources)
    keys(req, ('schema', 'pack_sha256', 'items', 'evaluation_only_not_generation_input', 'range_policy'), where='requirements')
    if (req['schema'] != 'choreobench.edit_semantic_requirements.v1'
            or bridge.get('schema') != 'choreobench.edit_semantic_bridge.v1'
            or bridge.get('record_sha256') != object_sha({k:v for k,v in bridge.items() if k!='record_sha256'})
            or bridge['requirements_sha256'] != object_sha(req)
            or bridge['inventory_sha256'] != audit['inventory_sha256'] or bridge['attempts_sha256'] != audit['attempts_sha256']):
        raise ValueError('Semantic bridge or requirements identity differs')
    _unchanged(bridge['sources']);sources.extend(bridge['sources'])
    # Revalidate original mapping against actual originals, not self-reported bridge hashes.
    refs = {r['role']: r['path'] for r in bridge['sources'] if r['role'] in ('public_pack','private_bindings','explicit_mapping')}
    pack, _ = _snapshot(refs['public_pack'], 'public_pack_recheck', sources)
    bindings, _ = _snapshot(refs['private_bindings'], 'bindings_recheck', sources)
    import_reviews(pack, bindings, dict(schema='choreobench.reviews.v1', pack_sha256=object_sha(pack), reviews=[]))
    by_review = {r['review_id']:r for r in req['items']}
    by_cell = {(c['task_id'],c['method_id'],c['seed']):c for c in audit['cells']}
    bound = {}
    originals = {b['review_id']:b['original'] for b in bindings['items']}
    public = {p['review_id']:p for p in pack['items']}
    mapping, _ = _snapshot(refs['explicit_mapping'], 'mapping_recheck', sources)
    if (req['pack_sha256'] != object_sha(pack) or len(by_review)!=len(req['items'])
            or set(by_review)!=set(public) or len(bridge['cells'])!=len(public)):
        raise ValueError('Requirements inventory changed')
    if [dict(review_id=b['review_id'],task_id=b['task_id'],method_id=b['method_id'],seed=b['seed']) for b in bridge['cells']] != mapping['cells']:
        raise ValueError('Private bridge mapping changed')
    for b in bridge['cells']:
        r = by_review[b['review_id']];c = by_cell[(b['task_id'],b['method_id'],b['seed'])]
        original, item = originals[b['review_id']], public[b['review_id']]
        task = original['edit_task']
        expected = dict(review_id=item['review_id'],item_sha256=item['item_sha256'],
            source_plan_sha256=object_sha(item['source_plan']),after_plan_sha256=object_sha(item['plan']),
            sample_rate=item['plan']['sample_rate'],num_samples=item['plan']['num_samples'],instruction=item['edit_task']['instruction'],
            required_review_ranges=_required_range(original['source_plan'], original['plan'], task),
            goals=[dict(goal_index=i,goal_sha256=object_sha(g),description=g) for i,g in enumerate(task['semantic_goals'])],
            boundary_review_required=True)
        if (r!=expected or b['requirement_sha256']!=object_sha(r) or object_sha(task)!=c['task_sha256']
                or object_sha(original['source_plan'])!=c['source_plan_sha256']
                or object_sha(original['plan'])!=c['after_plan_sha256'] or c['mechanical_valid'] is not True
                or b['task_sha256']!=c['task_sha256'] or b['before_sha256']!=c['source_plan_sha256']
                or b['after_sha256']!=c['after_plan_sha256']
                or original.get('source_binding',{}).get('dataset_sha256')!=c['dataset_sha256']):
            raise ValueError('Semantic requirement differs from original evidence')
        bound[b['review_id']] = c
    if isinstance(verdicts_paths,(str,Path)):verdicts_paths=[verdicts_paths]
    if not isinstance(verdicts_paths,(list,tuple)) or not verdicts_paths:
        raise ValueError('Provide original verdict files, including a records=[] file for zero submissions')
    decisions=[];seen=set();identities={}
    for index,path in enumerate(verdicts_paths):
        doc,_=_snapshot(path,'verdicts:'+str(index),sources)
        keys(doc,('schema','requirements_sha256','records'),where='verdict document')
        if doc['schema']!='choreobench.edit_semantic_verdicts.v1' or doc['requirements_sha256']!=object_sha(req) or not isinstance(doc['records'],list):
            raise ValueError('Verdicts bind different requirements')
        for row in doc['records']:
            keys(row,('review_id','item_sha256','source_plan_sha256','after_plan_sha256','reviewer_id','origin','model_id',
                      'reviewed_ranges','goals','boundary','comments'),where='semantic verdict record')
            text(row['review_id'],'review_id');text(row['reviewer_id'],'reviewer_id')
            rid=row['reviewer_id'];pair=(row['review_id'],rid)
            if row['review_id'] not in bound or rid=='FILL_REVIEWER_ID' or pair in seen:
                raise ValueError('Unknown/duplicate review or placeholder observer')
            seen.add(pair)
            if row['origin']=='human':
                if row['model_id'] is not None:raise ValueError('Human verdict cannot declare model_id')
            elif row['origin']=='model_judge':text(row['model_id'],'model_id')
            else:raise ValueError('Verdict origin must be human/model_judge')
            identity=(row['origin'],row['model_id'])
            if rid in identities and identities[rid]!=identity:raise ValueError('Observer identity/version changed')
            identities[rid]=identity
            r=by_review[row['review_id']];c=bound[row['review_id']]
            if any(row[k]!=r[k] for k in ('item_sha256','source_plan_sha256','after_plan_sha256')):
                raise ValueError('Verdict refers to different before/after/item')
            if not isinstance(row['comments'],str):raise ValueError('Comments must be text')
            merged=_ranges(row['reviewed_ranges'],r['num_samples'])
            covered=all(any(a<=v['start'] and v['end']<=b for a,b in merged) for v in r['required_review_ranges'])
            if not isinstance(row['goals'],list):raise ValueError('Goal verdicts must be a list')
            goals={g['goal_index']:g for g in r['goals']};observed={}
            def validate_verdict(v):
                if v['status'] not in ('pass','partial','fail','unknown'):raise ValueError('Invalid semantic status')
                text(v['evidence'],'verdict.evidence')
            for v in row['goals']:
                keys(v,('goal_index','goal_sha256','status','evidence'),where='goal verdict')
                integer(v['goal_index'],'goal_index',0);i=v['goal_index']
                if i not in goals or i in observed or v['goal_sha256']!=goals[i]['goal_sha256']:
                    raise ValueError('Unknown/duplicate/changed semantic goal')
                validate_verdict(v);observed[i]=v
            keys(row['boundary'],('status','evidence'),where='boundary verdict');validate_verdict(row['boundary'])
            known=[v['status'] for v in observed.values()]+[row['boundary']['status']]
            if c['literal_joint_success'] is False:
                status,success='mechanical_or_literal_fail',False
            elif 'fail' in known:
                status,success='declared_semantic_fail',False
            elif 'partial' in known:
                status,success='declared_partial',False
            elif not covered or set(observed)!=set(goals) or 'unknown' in known:
                status,success='pending_or_insufficient_review',None
            elif not goals and c['literal_joint_success'] is None:
                status,success='not_evaluable',None
            else:
                status,success='declared_joint_pass',True
            decisions.append(dict(task_id=c['task_id'],song_id=c['song_id'],method_id=c['method_id'],seed=c['seed'],
                review_id=row['review_id'],reviewer_id=rid,origin=row['origin'],model_id=row['model_id'],
                reviewed_range_union=[dict(start=a,end=b) for a,b in merged],required_range_covered=covered,
                missing_goal_indices=sorted(set(goals)-set(observed)),status=status,joint_success=success,original=copy.deepcopy(row)))
    _unchanged(sources)
    report=dict(schema='choreobench.edit_semantic_analysis.v1',sources=sources,inventory_sha256=audit['inventory_sha256'],
        requirements_sha256=object_sha(req),expected_inventory_cells=audit['expected_cells'],mapped_review_items=len(bound),
        mechanical_cells=audit['cells'],observer_decisions=decisions,
        review_items_without_any_verdict=[i for i in bound if not any(d['review_id']==i for d in decisions)],
        unrepresented_inventory_cells=[dict(task_id=c['task_id'],method_id=c['method_id'],seed=c['seed'],status=c['status'])
            for c in audit['cells'] if not any((b['task_id'],b['method_id'],b['seed'])==(c['task_id'],c['method_id'],c['seed']) for b in bridge['cells'])],
        consensus=None,quality_verified=False,actual_reviewers_listening_or_verdict_truth_authenticated=False,
        note='External declarations only; failures/partial dominate unresolved goals, passes require every goal, boundary and reviewed context. No automatic majority or grade-to-goal conversion; reviewer roster completion remains separate.')
    report['record_sha256']=object_sha(report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('inventory','attempts','public','private','mapping'):prepare.add_argument('--'+name,type=Path,required=True)
    prepare.add_argument('--out-dir',type=Path,required=True)
    analyze=sub.add_parser('analyze')
    for name in ('inventory','attempts','requirements','bridge','out'):analyze.add_argument('--'+name,type=Path,required=True)
    analyze.add_argument('--verdicts',type=Path,nargs='+',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        return prepare_edit_semantics(args.inventory,args.attempts,args.public,args.private,args.mapping,args.out_dir)
    report=analyze_edit_semantics(args.inventory,args.attempts,args.requirements,args.bridge,args.verdicts)
    write_json(args.out,report);return report


if __name__=='__main__':
    main()
