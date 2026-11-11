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
    joint_draft = copy.deepcopy(draft)
    joint_draft['schema'] = 'choreobench.edit_semantic_verdicts.v2'
    for row in joint_draft['records']:
        row['protection'] = dict(status=None, evidence='')
    write_json(output/'joint-verdicts.template.json', joint_draft)
    task_cells = {}
    for cell in audit['cells']:
        task_cells.setdefault(cell['task_id'], cell)
    authoring = dict(schema='choreobench.edit_task_authoring.v1', inventory_sha256=audit['inventory_sha256'],
        author_id='FILL_AUTHOR_ID', origin='human', model_id=None, frozen_before_execution=False,
        tasks=[dict(task_id=c['task_id'], task_sha256=c['task_sha256'], source_plan_sha256=c['source_plan_sha256'],
            category='unknown', validity='unknown', private_requirements_within_instruction=None, evidence='')
            for c in task_cells.values()])
    write_json(output/'task-authoring.template.private.json', authoring)
    return dict(items=len(requirements), semantic_goal_cells=sum(len(v['goals']) for v in requirements),
        requirements_sha256=object_sha(requirements_doc), bridge_sha256=object_sha(bridge), new_model_calls=0, quality_verified=False)


def analyze_edit_semantics(inventory_path, attempts_path, requirements_path, bridge_path, verdicts_paths, *,
                           task_authoring_path=None, reviewer_pair=None):
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
        if doc['schema'] not in ('choreobench.edit_semantic_verdicts.v1', 'choreobench.edit_semantic_verdicts.v2') or doc['requirements_sha256']!=object_sha(req) or not isinstance(doc['records'],list):
            raise ValueError('Verdicts bind different requirements')
        for row in doc['records']:
            fields = ('review_id','item_sha256','source_plan_sha256','after_plan_sha256','reviewer_id','origin','model_id',
                      'reviewed_ranges','goals','boundary','comments')
            if doc['schema']=='choreobench.edit_semantic_verdicts.v2':fields += ('protection',)
            keys(row,fields,where='semantic verdict record')
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
            protection = row.get('protection')
            if doc['schema']=='choreobench.edit_semantic_verdicts.v2':
                keys(protection, ('status', 'evidence'), where='protected semantics verdict')
                validate_verdict(protection)
            if success is False or protection is not None and protection['status'] in ('fail','partial'):
                e4_success = False
            elif success is True and goals and protection is not None and protection['status']=='pass':
                e4_success = True
            else:
                e4_success = None
            decisions.append(dict(task_id=c['task_id'],song_id=c['song_id'],method_id=c['method_id'],seed=c['seed'],
                review_id=row['review_id'],reviewer_id=rid,origin=row['origin'],model_id=row['model_id'],
                reviewed_range_union=[dict(start=a,end=b) for a,b in merged],required_range_covered=covered,
                missing_goal_indices=sorted(set(goals)-set(observed)),status=status,joint_success=success,
                e4_joint_success=e4_success, goal_inventory_empty=not goals,
                protection_status=None if protection is None else protection['status'],
                original=copy.deepcopy(row)))
    e4 = None
    if (task_authoring_path is None) != (reviewer_pair is None):
        raise ValueError('E4 joint evaluation requires both task authoring and a two-human reviewer pair')
    if task_authoring_path is not None:
        authoring, _ = _snapshot(task_authoring_path, 'task_authoring', sources)
        e4 = _analyze_e4(audit, bound, decisions, authoring, reviewer_pair)
    _unchanged(sources)
    report=dict(schema='choreobench.edit_semantic_analysis.v1',sources=sources,inventory_sha256=audit['inventory_sha256'],
        requirements_sha256=object_sha(req),expected_inventory_cells=audit['expected_cells'],mapped_review_items=len(bound),
        mechanical_cells=audit['cells'],observer_decisions=decisions,
        review_items_without_any_verdict=[i for i in bound if not any(d['review_id']==i for d in decisions)],
        unrepresented_inventory_cells=[dict(task_id=c['task_id'],method_id=c['method_id'],seed=c['seed'],status=c['status'])
            for c in audit['cells'] if not any((b['task_id'],b['method_id'],b['seed'])==(c['task_id'],c['method_id'],c['seed']) for b in bridge['cells'])],
        e4_joint_evaluation=e4, consensus=None,quality_verified=False,actual_reviewers_listening_or_verdict_truth_authenticated=False,
        note='External declarations only; failures/partial dominate unresolved goals, passes require every goal, boundary and reviewed context. No automatic majority or grade-to-goal conversion; reviewer roster completion remains separate.')
    report['record_sha256']=object_sha(report)
    return report


def _analyze_e4(audit, bound, decisions, authoring, reviewer_pair):
    """Use one author-validated denominator for both methods; no majority vote."""
    if (not isinstance(reviewer_pair, (list, tuple)) or len(reviewer_pair)!=2
            or len(set(reviewer_pair))!=2):
        raise ValueError('E4 requires two distinct declared human reviewers')
    for rid in reviewer_pair:
        text(rid, 'reviewer_id')
        if rid=='FILL_REVIEWER_ID':raise ValueError('Placeholder reviewer')
    keys(authoring, ('schema','inventory_sha256','author_id','origin','model_id','frozen_before_execution','tasks'),
         where='task authoring')
    if authoring['schema']!='choreobench.edit_task_authoring.v1' or authoring['inventory_sha256']!=audit['inventory_sha256']:
        raise ValueError('Task authoring binds another inventory')
    text(authoring['author_id'], 'author_id')
    if authoring['author_id']=='FILL_AUTHOR_ID':raise ValueError('Placeholder author')
    if authoring['origin']=='human':
        if authoring['model_id'] is not None:raise ValueError('Human author cannot declare model_id')
    elif authoring['origin']=='model_judge':text(authoring['model_id'], 'author model_id')
    else:raise ValueError('Author origin must be human/model_judge')
    if type(authoring['frozen_before_execution']) is not bool or not isinstance(authoring['tasks'],list):
        raise ValueError('Invalid task authoring declaration')
    tasks = {c['task_id']:c for c in audit['cells']}
    authored = {}
    for row in authoring['tasks']:
        keys(row, ('task_id','task_sha256','source_plan_sha256','category','validity',
                   'private_requirements_within_instruction','evidence'), where='authored task')
        text(row['task_id'], 'task_id');text(row['evidence'], 'authoring evidence')
        if row['task_id'] not in tasks or row['task_id'] in authored:
            raise ValueError('Unknown or duplicate authored task')
        cell = tasks[row['task_id']]
        if any(row[k]!=cell[k] for k in ('task_sha256','source_plan_sha256')):
            raise ValueError('Authoring differs from original task/source')
        if row['category'] not in ('change_required','already_satisfied','ambiguous_or_conflicting','source_unavailable','unknown'):
            raise ValueError('Invalid authored task category')
        if row['validity'] not in ('valid','invalid','unknown'):
            raise ValueError('Invalid task validity')
        scope = row['private_requirements_within_instruction']
        if scope is not None and type(scope) is not bool:raise ValueError('Instruction scope must be bool/null')
        if row['validity']=='valid' and (scope is not True or row['category'] in ('unknown','source_unavailable')):
            raise ValueError('Valid task requires known category and public-instruction scope confirmation')
        authored[row['task_id']] = row
    if set(authored)!=set(tasks):raise ValueError('Authoring must retain every registered task slot')
    for d in decisions:
        if d['reviewer_id'] in reviewer_pair and d['origin']!='human':
            raise ValueError('A model judge cannot fill the declared human pair')
    lookup = {(d['task_id'],d['method_id'],d['seed'],d['reviewer_id']):d for d in decisions}
    mapped = {(c['task_id'],c['method_id'],c['seed']):rid for rid,c in bound.items()}
    rows = []
    for cell in audit['cells']:
        key = (cell['task_id'],cell['method_id'],cell['seed'])
        task = authored[cell['task_id']]
        observations = [lookup.get(key+(rid,)) for rid in reviewer_pair]
        values = [None if d is None else d['e4_joint_success'] for d in observations]
        if cell['mechanical_valid'] is False or cell['literal_joint_success'] is False:
            status = 'failure'
        elif values==[True,True]:status='success'
        elif values==[False,False]:status='failure'
        else:status='unresolved'
        disagreements = []
        if all(d is not None for d in observations):
            a,b = observations
            for field in ('e4_joint_success','protection_status'):
                if a[field]!=b[field]:disagreements.append(field)
            if a['original']['boundary']['status']!=b['original']['boundary']['status']:
                disagreements.append('boundary')
            ga={g['goal_index']:g['status'] for g in a['original']['goals']}
            gb={g['goal_index']:g['status'] for g in b['original']['goals']}
            for i in sorted(set(ga)|set(gb)):
                if ga.get(i)!=gb.get(i):disagreements.append('goal:'+str(i))
        rows.append(dict(task_id=cell['task_id'],song_id=cell['song_id'],method_id=cell['method_id'],seed=cell['seed'],
            review_id=mapped.get(key),category=task['category'],task_validity=task['validity'],
            in_change_denominator=task['validity']=='valid' and task['category']=='change_required',
            mechanical_status=cell['status'],attempt_status=cell['attempt_status'],
            mechanical_valid=cell['mechanical_valid'],no_op=cell['no_op'],
            pair_status=status,pair_joint_success=values,disagreements=disagreements,
            missing_reviewers=[rid for rid,d in zip(reviewer_pair,observations) if d is None],
            unknown_verdict_reviewers=[rid for rid,d in zip(reviewer_pair,observations)
                if d is not None and (any(g['status']=='unknown' for g in d['original']['goals'])
                    or d['original']['boundary']['status']=='unknown' or d['protection_status']=='unknown')],
            insufficient_reviewers=[rid for rid,d in zip(reviewer_pair,observations)
                if d is not None and (not d['required_range_covered'] or d['missing_goal_indices'] or d['goal_inventory_empty'] or d['protection_status'] is None)]))
        row = rows[-1]
        if (cell['mechanical_valid'] is not False and cell['literal_joint_success'] is not False
                and (row['disagreements'] or row['missing_reviewers'] or row['unknown_verdict_reviewers'] or row['insufficient_reviewers'])):
            row['pair_status'] = 'unresolved'
    summaries = []
    methods = list(dict.fromkeys(c['method_id'] for c in audit['cells']))
    for method in methods:
        for seed in sorted({c['seed'] for c in audit['cells']}):
            selected = [r for r in rows if r['method_id']==method and r['seed']==seed and r['in_change_denominator']]
            n=len(selected);s=sum(r['pair_status']=='success' for r in selected);u=sum(r['pair_status']=='unresolved' for r in selected)
            summaries.append(dict(method_id=method,seed=seed,N=n,S=s,U=u,F=n-s-u,
                success_rate=None if not n else s/n,unresolved_rate=None if not n else u/n,
                success_rate_bounds=None if not n else [s/n,(s+u)/n],
                by_song=[dict(song_id=sid,N=sum(r['song_id']==sid for r in selected),
                    S=sum(r['song_id']==sid and r['pair_status']=='success' for r in selected),
                    U=sum(r['song_id']==sid and r['pair_status']=='unresolved' for r in selected))
                    for sid in dict.fromkeys(r['song_id'] for r in selected)]))
    return dict(schema='choreobench.edit_e4_joint_analysis.v1',reviewer_pair=list(reviewer_pair),
        authoring=copy.deepcopy(authoring),registered_tasks=len(tasks),registered_cells=len(rows),
        authoring_status_counts={v:sum(t['validity']==v for t in authored.values()) for v in ('valid','invalid','unknown')},
        authored_category_counts={v:sum(t['category']==v for t in authored.values()) for v in sorted({t['category'] for t in authored.values()})},
        cells=rows,change_required_summaries=summaries,
        controls=[r for r in rows if r['task_validity']=='valid' and r['category'] in ('already_satisfied','ambiguous_or_conflicting')],
        excluded_or_unresolved_authoring=[copy.deepcopy(t) for t in authored.values() if t['validity']!='valid'],
        human_authoring_and_preexecution_freeze_declared=authoring['origin']=='human' and authoring['frozen_before_execution'],
        author_identity_preexecution_timing_and_validity_authenticated=False,
        independent_human_truth_authenticated=False,formal_e4_completed=False,
        note='V2 explicit protected semantics plus every goal and boundary are required for E4 passes. '
             'Two specified humans must agree; unknown, missing and unresolved disagreements stay U. '
             'Mechanical/literal failures remain F without reviewers. Task authoring is declared, not authenticated. '
             'Control outcomes are separate, and no-op alone is not a reasonable-control judgment. '
             'Music adaptation and cost are separate; task/seed rows are nested in songs, not new independent songs.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    for name in ('inventory','attempts','public','private','mapping'):prepare.add_argument('--'+name,type=Path,required=True)
    prepare.add_argument('--out-dir',type=Path,required=True)
    analyze=sub.add_parser('analyze')
    for name in ('inventory','attempts','requirements','bridge','out'):analyze.add_argument('--'+name,type=Path,required=True)
    analyze.add_argument('--verdicts',type=Path,nargs='+',required=True)
    analyze.add_argument('--task-authoring',type=Path)
    analyze.add_argument('--reviewer-pair',nargs=2)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        return prepare_edit_semantics(args.inventory,args.attempts,args.public,args.private,args.mapping,args.out_dir)
    report=analyze_edit_semantics(args.inventory,args.attempts,args.requirements,args.bridge,args.verdicts,
        task_authoring_path=args.task_authoring,reviewer_pair=args.reviewer_pair)
    write_json(args.out,report);return report


if __name__=='__main__':
    main()
