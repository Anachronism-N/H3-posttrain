"""Declared reviewed-time denominators; no automatic quality or consensus."""
import argparse
import copy
from pathlib import Path

from .io import file_sha, object_sha, read_json_snapshot, write_json
from .review_issues import ISSUE_TYPES, validate_review_issues
from .schema import integer, keys, text


PROBLEM_TYPES = tuple(t for t in ISSUE_TYPES if t not in ('reasonable_recurrence', 'uncertain'))


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def _intersect(left, right):
    result, i, j = [], 0, 0
    left, right = _union(left), _union(right)
    while i < len(left) and j < len(right):
        a, b = left[i];c, d = right[j]
        if max(a, c) < min(b, d):
            result.append([max(a, c), min(b, d)])
        if b <= d:
            i += 1
        else:
            j += 1
    return result


def _measure(intervals):
    return sum(end - start for start, end in intervals)


def _snapshot(path, sources, role):
    value, sha = read_json_snapshot(path)
    sources.append(dict(role=role, path=str(Path(path).resolve()), sha256=sha))
    return value, sha


def analyze_review_ranges(public_path, issues_path, ranges_paths):
    """Compute per-observer, per-type |reviewed intersect declared issues|/|reviewed|.

    Explicit checked_types state what was inspected. A range record without an
    issue record, an absent/empty reviewed range or unchecked type remains NA.
    Uncertainty and reasonable recurrence are reported separately, not errors.
    Identity, listening and issue truth remain external declarations.
    """
    sources = []
    pack, _ = _snapshot(public_path, sources, 'public_pack')
    issues, issues_sha = _snapshot(issues_path, sources, 'issue_records')
    validate_review_issues(pack, issues)
    if isinstance(ranges_paths, (str, Path)):
        ranges_paths = [ranges_paths]
    if not isinstance(ranges_paths, (list, tuple)) or not ranges_paths:
        raise ValueError('Provide original reviewed-range files, including records=[] for no submissions')
    public = {item['review_id']: item for item in pack['items']}
    issue_rows = {(row['review_id'], row['reviewer_id']): row for row in issues['records']}
    identities = {}
    for row in issues['records']:
        identities[row['reviewer_id']] = (row['origin'], row['model_id'])
    scopes = {}
    for index, path in enumerate(ranges_paths):
        doc, _ = _snapshot(path, sources, 'reviewed_ranges:'+str(index))
        keys(doc, ('schema', 'pack_sha256', 'issues_file_sha256', 'records'), where='reviewed ranges document')
        if (doc['schema'] != 'choreobench.review_ranges.v1' or doc['pack_sha256'] != object_sha(pack)
                or doc['issues_file_sha256'] != issues_sha or not isinstance(doc['records'], list)):
            raise ValueError('Reviewed ranges bind different public pack or original issue-file bytes')
        for row in doc['records']:
            keys(row, ('review_id', 'item_sha256', 'reviewer_id', 'origin', 'model_id', 'audio_seen',
                       'checked_types', 'ranges'), where='reviewed scope record')
            text(row['review_id'], 'review_id');text(row['reviewer_id'], 'reviewer_id')
            rid = row['reviewer_id'];pair = (row['review_id'], rid)
            if row['review_id'] not in public or rid == 'FILL_REVIEWER_ID' or pair in scopes:
                raise ValueError('Unknown item, placeholder observer or duplicate reviewed scope')
            item = public[row['review_id']]
            if row['item_sha256'] != item['item_sha256']:
                raise ValueError('Reviewed scope item hash differs')
            if row['origin'] == 'human':
                if row['model_id'] is not None:
                    raise ValueError('Human scope cannot declare model_id')
            elif row['origin'] == 'model_judge':
                text(row['model_id'], 'model_id')
            else:
                raise ValueError('Scope origin must be human/model_judge')
            identity = (row['origin'], row['model_id'])
            if rid in identities and identities[rid] != identity:
                raise ValueError('Observer identity/model version differs across original files')
            identities[rid] = identity
            if type(row['audio_seen']) is not bool:
                raise ValueError('audio_seen must be a boolean declaration')
            observed = issue_rows.get(pair)
            if observed is not None and observed['audio_seen'] != row['audio_seen']:
                raise ValueError('Listening declarations differ across bound issue/scope records')
            types = row['checked_types']
            if (not isinstance(types, list) or not types or any(not isinstance(t, str) or t not in ISSUE_TYPES for t in types)
                    or len(set(types)) != len(types)):
                raise ValueError('checked_types must be nonempty unique known types')
            if 'music_mismatch' in types and not row['audio_seen']:
                raise ValueError('Checking music mismatch requires an audio_seen declaration')
            if not isinstance(row['ranges'], list):
                raise ValueError('Scope ranges must be a list')
            intervals = []
            for interval in row['ranges']:
                keys(interval, ('start', 'end'), where='reviewed interval')
                integer(interval['start'], 'range.start', 0);integer(interval['end'], 'range.end', 1)
                if not interval['start'] < interval['end'] <= item['plan']['num_samples']:
                    raise ValueError('Reviewed interval escapes canonical half-open clock')
                intervals.append((interval['start'], interval['end']))
            scopes[pair] = dict(original=copy.deepcopy(row), union=_union(intervals))
    rows = []
    for pair in sorted(set(scopes) | set(issue_rows)):
        observed, scope = issue_rows.get(pair), scopes.get(pair)
        item = public[pair[0]];rate = item['plan']['sample_rate'];total = item['plan']['num_samples']
        reviewed = scope['union'] if scope else []
        denominator = _measure(reviewed)
        checked = scope['original']['checked_types'] if scope else []
        records = observed['issues'] if observed else []
        ready = scope is not None and observed is not None and denominator > 0
        status = ('missing_scope' if scope is None else 'missing_issue_record' if observed is None else
                  'no_reviewed_time' if not denominator else 'declared_scope_and_observations')
        diagnostics = {}
        for kind in ISSUE_TYPES:
            relevant = [i for i in records if i['type'] == kind]
            affected = _intersect(reviewed, [(i['start'], i['end']) for i in relevant])
            available = ready and kind in checked
            diagnostics[kind] = dict(checked=kind in checked, declared_count=len(relevant),
                affected_samples=_measure(affected) if available else None,
                affected_seconds=_measure(affected)/rate if available else None,
                reviewed_fraction=_measure(affected)/denominator if available else None,
                affected_intervals=[dict(start=a, end=b) for a, b in affected] if available else None,
                reason=None if available else 'unchecked_type' if ready else status,
                classification='counterexample' if kind == 'reasonable_recurrence' else 'uncertainty' if kind == 'uncertain' else 'problem')
        major = [i for i in records if i['type'] in PROBLEM_TYPES and i['severity'] == 'major']
        major_ready = ready and set(PROBLEM_TYPES).issubset(checked)
        major_union = _intersect(reviewed, [(i['start'], i['end']) for i in major])
        major_by_type = {}
        for kind in PROBLEM_TYPES:
            affected = _intersect(reviewed, [(i['start'], i['end']) for i in major if i['type'] == kind])
            available = ready and kind in checked
            major_by_type[kind] = dict(affected_samples=_measure(affected) if available else None,
                reviewed_fraction=_measure(affected)/denominator if available else None)
        excluded = []
        for issue in records:
            inside = _measure(_intersect(reviewed, [(issue['start'], issue['end'])]))
            excluded.append(dict(issue_id=issue['issue_id'], type=issue['type'],
                inside_reviewed_samples=inside, outside_reviewed_samples=issue['end']-issue['start']-inside,
                type_not_declared_checked=issue['type'] not in checked))
        declaration = observed if observed is not None else scope['original']
        rows.append(dict(review_id=pair[0], reviewer_id=pair[1], origin=declaration['origin'], model_id=declaration['model_id'],
            item_sha256=item['item_sha256'], sample_rate=rate, num_samples=total, status=status,
            reviewed_intervals=[dict(start=a, end=b) for a, b in reviewed], reviewed_samples=denominator,
            reviewed_seconds=denominator/rate, reviewed_plan_fraction=denominator/total,
            original_scope=copy.deepcopy(scope['original']) if scope else None, original_issues=copy.deepcopy(observed),
            by_type=diagnostics, major_by_type=major_by_type,
            all_major_problem_union=dict(affected_samples=_measure(major_union) if major_ready else None,
                reviewed_fraction=_measure(major_union)/denominator if major_ready else None,
                at_least_one_major_problem=bool(major_union) if major_ready else None,
                reason=None if major_ready else 'all_problem_types_must_be_checked' if ready else status),
            issue_scope_accounting=excluded,
            declared_uncertainty_samples=_measure(_intersect(reviewed, [(i['start'], i['end']) for i in records if i['type']=='uncertain'])) if ready else None,
            quality_pass=None, reviewer_identity_listening_or_issue_truth_authenticated=False))
    for ref in sources:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Reviewed-range analysis source changed during processing')
    report = dict(schema='choreobench.review_range_analysis.v1', sources=sources, pack_sha256=object_sha(pack),
        public_items=len(public), rows=rows,
        items_without_any_scope_record=[rid for rid in public if not any(pair[0] == rid for pair in scopes)],
        items_without_any_issue_record=[rid for rid in public if not any(pair[0] == rid for pair in issue_rows)],
        consensus=None, scientific_quality_verified=False,
        note='Declared interval diagnostics per observer, not a quality score. Empty issues plus positive checked scope may mean zero declared issues, never verified quality. Missing/zero time is NA; uncertainty/recurrence are not problems. Do not union observers or treat repeated plans as independent songs. Roster/design completion remains separate.')
    report['record_sha256'] = object_sha(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--issues', type=Path, required=True)
    parser.add_argument('--ranges', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_review_ranges(args.public, args.issues, args.ranges)
    write_json(args.out, report)
    return report


if __name__ == '__main__':
    main()
