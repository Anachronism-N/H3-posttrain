"""Bound human-declared time observations; never infer scores or missing judgments."""
import argparse
import copy
import json
from pathlib import Path

from .io import file_sha, object_sha, read_json_snapshot, write_json
from .reviews import validate_review_pack
from .schema import identifier, integer, keys, text, validate_plan


ISSUE_TYPES = {
    'unintended_repeat': '非预期复读',
    'reasonable_recurrence': '合理主题回归（反例）',
    'ambiguous_action': '动作含糊',
    'music_mismatch': '音乐错配',
    'overdense_action': '动作过密',
    'abrupt_transition': '衔接突然',
    'uncertain': '无法判断',
}


def blank_issue_template(pack):
    validate_review_pack(pack)
    return dict(schema='choreobench.review_issues.v1', pack_sha256=object_sha(pack), records=[
        dict(review_id=item['review_id'], item_sha256=item['item_sha256'],
             reviewer_id='FILL_REVIEWER_ID', origin='human', model_id=None, audio_seen=False, issues=[])
        for item in pack['items']])


def validate_review_issues(pack, document):
    """Check exact public evidence/time references, not the truth of observations.

    start/end are integer canonical samples on the referenced Plan clock.
    All referenced output events must intersect that nonempty half-open range.
    No issue list (including an explicitly empty list) proves a plan is good.
    """
    validate_review_pack(pack)
    items = {item['review_id']: item for item in pack['items']}
    for item in items.values():
        song = dict(copy.deepcopy(item['song']), split='dev', song_group_id=None,
                    artist_group_id=None, near_duplicate_group_id=None)
        validate_plan(item['plan'], song)
    keys(document, ('schema', 'pack_sha256', 'records'), where='review issues')
    if document['schema'] != 'choreobench.review_issues.v1' or document['pack_sha256'] != object_sha(pack):
        raise ValueError('Review issues schema or public pack binding differs')
    if not isinstance(document['records'], list):
        raise ValueError('Issue records must be a list')
    seen, identities = set(), {}
    for row in document['records']:
        keys(row, ('review_id', 'item_sha256', 'reviewer_id', 'origin', 'model_id', 'audio_seen', 'issues'),
             where='issue record')
        text(row['review_id'], 'review_id')
        text(row['reviewer_id'], 'reviewer_id')
        if row['review_id'] not in items or row['reviewer_id'] == 'FILL_REVIEWER_ID':
            raise ValueError('Known review item and declared non-placeholder reviewer required')
        item = items[row['review_id']]
        if row['item_sha256'] != item['item_sha256']:
            raise ValueError('Issue item hash differs')
        if row['origin'] == 'human':
            if row['model_id'] is not None:
                raise ValueError('Human observations cannot declare model_id')
        elif row['origin'] == 'model_judge':
            text(row['model_id'], 'model_id')
        else:
            raise ValueError('Issue origin must be human/model_judge')
        identity = (row['origin'], row['model_id'])
        rid = row['reviewer_id']
        if rid in identities and identities[rid] != identity:
            raise ValueError('Observer origin/model identity changed')
        identities[rid] = identity
        pair = (row['review_id'], rid)
        if pair in seen:
            raise ValueError('Duplicate observer/item record')
        seen.add(pair)
        if type(row['audio_seen']) is not bool or not isinstance(row['issues'], list):
            raise ValueError('audio_seen must be boolean and issues must be a list')
        events = {event['event_id']: event for event in item['plan']['events']}
        issue_ids = set()
        for issue in row['issues']:
            keys(issue, ('issue_id', 'type', 'severity', 'start', 'end', 'event_ids', 'evidence', 'reason'),
                 where='time observation')
            identifier(issue['issue_id'], 'issue_id')
            if issue['issue_id'] in issue_ids:
                raise ValueError('Duplicate issue_id within observer/item')
            issue_ids.add(issue['issue_id'])
            if not isinstance(issue['type'], str) or issue['type'] not in ISSUE_TYPES:
                raise ValueError('Unknown observation type')
            permitted = (('not_applicable',) if issue['type'] == 'reasonable_recurrence' else
                         ('uncertain',) if issue['type'] == 'uncertain' else ('minor', 'major'))
            if issue['severity'] not in permitted:
                raise ValueError('Observation type/severity conflict')
            if issue['type'] == 'music_mismatch' and not row['audio_seen']:
                raise ValueError('Music mismatch requires an explicit audio_seen declaration')
            integer(issue['start'], 'issue.start', 0)
            integer(issue['end'], 'issue.end', 1)
            if not issue['start'] < issue['end'] <= item['plan']['num_samples']:
                raise ValueError('Issue interval escapes the canonical half-open audio clock')
            selected = issue['event_ids']
            if (not isinstance(selected, list) or not selected or any(not isinstance(e, str) for e in selected)
                    or len(set(selected)) != len(selected) or any(e not in events for e in selected)):
                raise ValueError('Issue requires unique known public output event IDs')
            if any(not (events[e]['start'] < issue['end'] and issue['start'] < events[e]['end']) for e in selected):
                raise ValueError('Each selected event must intersect the issue interval')
            text(issue['evidence'], 'issue.evidence')
            text(issue['reason'], 'issue.reason')
    return document


def check_review_issues(public_path, issues_path):
    pack, pack_sha = read_json_snapshot(public_path)
    document, issues_sha = read_json_snapshot(issues_path)
    validate_review_issues(pack, document)
    observed = {row['review_id'] for row in document['records']}
    for path, expected in ((public_path, pack_sha), (issues_path, issues_sha)):
        if file_sha(path) != expected:
            raise ValueError('Issue analysis source changed while validating')
    return dict(schema='choreobench.review_issues_check.v1', pack_sha256=object_sha(pack),
        sources=[dict(role='public_pack', sha256=pack_sha), dict(role='issue_records', sha256=issues_sha)],
        records=copy.deepcopy(document['records']), public_items=len(pack['items']),
        items_without_observer_record=[i['review_id'] for i in pack['items'] if i['review_id'] not in observed],
        declared_observations=sum(len(row['issues']) for row in document['records']),
        quality_verified=False, reviewer_identity_or_listening_authenticated=False,
        note='Only declarations and public references validated. Empty, absent or uncertain observations are not passes, scores or independent samples.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--issues', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = check_review_issues(args.public, args.issues)
    write_json(args.out, report)
    print(json.dumps(dict(declared_observations=report['declared_observations'], quality_verified=False)))
    return report


if __name__ == '__main__':
    main()
