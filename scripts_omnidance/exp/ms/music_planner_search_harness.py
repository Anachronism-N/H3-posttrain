"""Blind audio-plan review packages and source-paired planner search reports."""
import argparse
from collections import defaultdict
import hashlib
import hmac
import random
import secrets
from pathlib import Path

from scripts_omnidance.exp.ms.music_planner_search import (
    DIMENSIONS, _keys, _sha, object_sha, read_json, review_request, validate_pool,
    validate_reviews, write_json,
)


def blind_package(pool, policy, *, seed=42, secret=None):
    """The private mapping must never be sent to the reviewer."""
    request = review_request(pool, policy)
    if type(seed) is not int:
        raise ValueError('Blind shuffle seed must be an integer')
    secret = secrets.token_hex(32) if secret is None else secret
    _sha(secret, 'private blinding secret')
    items, bindings = [], []
    for candidate in request['candidates']:
        opaque = hmac.new(bytes.fromhex(secret), candidate['candidate_sha256'].encode(),
                          hashlib.sha256).hexdigest()[:32]
        items.append(dict(opaque_id=opaque, audio_ref=dict(path=candidate['audio_path'],
                                                         sha256=candidate['audio_sha256'],
                                                         window_seconds=[0., 5.0625]),
                          proposed_plan_text=candidate['raw_text']))
        bindings.append(dict(opaque_id=opaque, candidate_sha256=candidate['candidate_sha256'],
                             source_id=candidate['source_id'], audio_sha256=candidate['audio_sha256']))
    random.Random(seed).shuffle(items)
    public = dict(schema='music_planner_blind_request_v1', rubric=policy['rubric'],
                  dimensions=list(DIMENSIONS), score_range=[0, 5],
                  caption_protocol=dict(id='native-direct-action-fixed-two-windows-v1',
                                        windows_seconds=[[0., 2.], [2., 5.0625]], genre_header_required=False),
                  instructions=request['instructions'], candidates=items)
    public['request_sha256'] = object_sha(public)
    private = dict(schema='music_planner_blind_binding_v1', run_sha256=pool['run_sha256'],
                   regular_request_sha256=request['request_sha256'], policy_sha256=object_sha(policy),
                   public_request_sha256=public['request_sha256'], seed=seed, secret=secret,
                   bindings=bindings)
    private['binding_sha256'] = object_sha(private)
    return public, private


def validate_blind_package(pool, policy, public, private):
    _keys(private, ('schema', 'run_sha256', 'regular_request_sha256', 'policy_sha256',
                    'public_request_sha256', 'seed', 'secret', 'bindings', 'binding_sha256'), 'private binding')
    expected_public, expected_private = blind_package(pool, policy, seed=private['seed'], secret=private['secret'])
    if public != expected_public or private != expected_private:
        raise ValueError('Blind package/mapping differs from exact candidate/audio/policy binding')
    return {entry['opaque_id']: entry for entry in private['bindings']}


def import_blind_reviews(pool, policy, public, private, response):
    bindings = validate_blind_package(pool, policy, public, private)
    _keys(response, ('schema', 'request_sha256', 'judge', 'rubric', 'reviews'), 'blind response')
    if (response['schema'] != 'music_planner_blind_reviews_v1'
            or response['request_sha256'] != public['request_sha256']
            or response['judge'] != policy['judge'] or response['rubric'] != policy['rubric']):
        raise ValueError('Blind response request/judge/rubric identity differs')
    if not isinstance(response['reviews'], list):
        raise ValueError('Blind reviews must be a list')
    converted, seen = [], set()
    for review in response['reviews']:
        _keys(review, ('opaque_id', 'audio_sha256', 'original_audio_reviewed', 'verdict',
                       'scores', 'confidence', 'reason', 'allow_preference_use'), 'blind candidate review')
        opaque = review['opaque_id']
        if opaque not in bindings or opaque in seen:
            raise ValueError('Unknown or duplicate opaque review ID')
        binding = bindings[opaque]
        if review['audio_sha256'] != binding['audio_sha256']:
            raise ValueError('Blind review belongs to another audio')
        converted.append(dict({key: value for key, value in review.items() if key != 'opaque_id'},
                              candidate_sha256=binding['candidate_sha256'], source_id=binding['source_id']))
        seen.add(opaque)
    position = {item['candidate_sha256']: index for index, item in enumerate(pool['candidates'])}
    converted.sort(key=lambda row: position[row['candidate_sha256']])
    regular = dict(schema='music_planner_reviews_v1',
                   request_sha256=private['regular_request_sha256'], judge=policy['judge'],
                   rubric=policy['rubric'], reviews=converted)
    validate_reviews(pool, policy, regular)
    receipt = dict(schema='music_planner_blind_import_receipt_v1',
                   public_request_sha256=public['request_sha256'], binding_sha256=private['binding_sha256'],
                   response_sha256=object_sha(response), imported_reviews_sha256=object_sha(regular))
    return regular, receipt


def _independent_auditor(selection, audit):
    left, right = selection['judge'], audit['judge']
    if left['id'].strip().lower() == right['id'].strip().lower():
        raise ValueError('Audit must use a reviewer independent of selection')
    if left['kind'] == right['kind'] == 'audio_model':
        if any(left[key].strip().lower() == right[key].strip().lower()
               for key in ('model_id', 'base_model_id', 'model_sha256')):
            raise ValueError('Selection and audit model identities are not independent')
    if selection['rubric'] != audit['rubric']:
        raise ValueError('Paired audit must use the same fixed rubric')


def _bootstrap_mean(values, seed=42, repeats=2000):
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    return dict(method='source-paired percentile bootstrap', repeats=repeats, seed=seed,
                mean=sum(values) / len(values), low_95=samples[int(.025 * (repeats - 1))],
                high_95=samples[int(.975 * (repeats - 1))],
                small_sample_warning=len(values) < 30,
                interpretation='conditional audit scores for these sources, not proof of unseen-song generalization')


def search_report(pool, policy, reviews=None, *, audit_policy=None, audit_reviews=None,
                  bootstrap_seed=42):
    """Keep train/dev separate; same-review selection scores have no generalization CI."""
    validate_pool(pool)
    review_request(pool, policy)
    imported = {} if reviews is None else validate_reviews(pool, policy, reviews)[1]
    if (audit_policy is None) != (audit_reviews is None):
        raise ValueError('Independent audit requires both policy and reviews')
    audits = {}
    if audit_policy is not None:
        if reviews is None:
            raise ValueError('Audit report requires selection reviews')
        review_request(pool, audit_policy)
        _independent_auditor(policy, audit_policy)
        audits = validate_reviews(pool, audit_policy, audit_reviews)[1]
    grouped, records = defaultdict(list), []
    thresholds = policy['thresholds']
    for item in pool['candidates']:
        grouped[item['source']['id']].append(item)
    for sid, items in sorted(grouped.items()):
        baseline = next(item for item in items if item['index'] == 0)
        record = dict(source_id=sid, split=baseline['source']['split'], audio_sha256=baseline['source']['audio_sha256'],
                      baseline_candidate_sha256=baseline['candidate_sha256'],
                      baseline_kind='single_stochastic_sample_index0',
                      baseline_new_tokens=len(baseline['generation']['new_token_ids']),
                      search_candidates=len(items),
                      search_input_tokens=sum(item['generation']['input_tokens'] for item in items),
                      search_new_tokens=sum(len(item['generation']['new_token_ids']) for item in items),
                      search_max_new_token_budget=len(items) * pool['sampling']['max_new_tokens'],
                      distinct_candidate_texts=len({item['generation']['raw_text'] for item in items}),
                      format_eligible_candidates=sum(item['diagnostics']['format_eligible'] for item in items),
                      selected_candidate_sha256=None, selection_status='pending_independent_review',
                      descriptive_selection_score_delta=None, independent_audit_score_delta=None,
                      independent_audit_dimension_deltas=None)
        if all(item['candidate_sha256'] in imported for item in items):
            def score(item):
                return sum(imported[item['candidate_sha256']]['scores'].values()) / len(DIMENSIONS)
            approved = [item for item in items if item['diagnostics']['format_eligible']
                        and imported[item['candidate_sha256']]['verdict'] == 'approve'
                        and imported[item['candidate_sha256']]['confidence'] >= thresholds['min_confidence']
                        and min(imported[item['candidate_sha256']]['scores'].values()) >= thresholds['min_dimension_score']]
            approved.sort(key=lambda item: (-score(item), item['candidate_sha256']))
            if not approved:
                record['selection_status'] = 'no_approved_candidate'
            elif len(approved) > 1 and abs(score(approved[0]) - score(approved[1])) < 1e-12:
                record['selection_status'] = 'tie_no_unique_selection'
            else:
                selected = approved[0]
                digest, base_digest = selected['candidate_sha256'], baseline['candidate_sha256']
                record.update(selected_candidate_sha256=digest, selection_status='selected_for_review_analysis',
                              descriptive_selection_score_delta=score(selected) - score(baseline))
                if digest in audits and base_digest in audits:
                    chosen, base = audits[digest], audits[base_digest]
                    if (chosen['verdict'] != 'uncertain' and base['verdict'] != 'uncertain'
                            and min(chosen['confidence'], base['confidence']) >= audit_policy['thresholds']['min_confidence']):
                        deltas = {key: chosen['scores'][key] - base['scores'][key] for key in DIMENSIONS}
                        record.update(independent_audit_score_delta=sum(deltas.values()) / len(DIMENSIONS),
                                      independent_audit_dimension_deltas=deltas)
        records.append(record)
    splits = {}
    for split in ('train', 'dev'):
        subset = [record for record in records if record['split'] == split]
        apparent = [r['descriptive_selection_score_delta'] for r in subset
                    if r['descriptive_selection_score_delta'] is not None]
        paired = [r['independent_audit_score_delta'] for r in subset
                  if r['independent_audit_score_delta'] is not None]
        splits[split] = dict(sources=len(subset), selected_sources=len(apparent),
                             candidate_calls=sum(r['search_candidates'] for r in subset),
                             search_new_tokens=sum(r['search_new_tokens'] for r in subset),
                             baseline_new_tokens=sum(r['baseline_new_tokens'] for r in subset),
                             descriptive_same_review_mean_delta=sum(apparent) / len(apparent) if apparent else None,
                             same_review_generalization_ci=None,
                             independently_audited_pairs=len(paired),
                             independent_audit_source_bootstrap=_bootstrap_mean(paired, bootstrap_seed))
    return dict(schema='music_planner_search_report_v1', run_sha256=pool['run_sha256'],
                caption_protocol=dict(id='native-direct-action-fixed-two-windows-v1',
                                      windows_seconds=[[0., 2.], [2., 5.0625]], genre_header_required=False),
                policy_sha256=object_sha(policy), reviews_sha256=object_sha(reviews) if reviews else None,
                audit_policy_sha256=object_sha(audit_policy) if audit_policy else None,
                audit_reviews_sha256=object_sha(audit_reviews) if audit_reviews else None,
                baseline_kind='single_stochastic_sample_index0_not_fair_greedy', splits=splits, sources=records,
                group_isolation_verified=all(proof['status'] == 'verified'
                                             for proof in pool['isolation']['verification'].values()),
                review_blinding_status='verify the separate blind-import receipt; not asserted from regular reviews alone',
                cost_scope='generation tokens and calls only; judge/processor time and hardware costs excluded',
                quality_interpretation=('Same-review deltas describe selection on the scoring sample; they are '
                                        'optimistic and do not establish generalization. Only a separate fixed auditor '
                                        'provides paired audit intervals. Audio/text review cannot certify execution.'),
                format_and_length_are_diagnostics_not_music_rewards=True,
                test_used=False, training_export_performed=False)


def _fresh_outputs(*paths):
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved) or any(path.exists() for path in resolved):
        raise ValueError('Output paths must be distinct and fresh; no partial overwrite of blind mappings')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    blind = sub.add_parser('blind-request')
    restore = sub.add_parser('import-blind')
    report = sub.add_parser('report')
    for command in (blind, restore, report):
        command.add_argument('--pool', required=True)
        command.add_argument('--policy', required=True)
    blind.add_argument('--seed', type=int, default=42)
    blind.add_argument('--public-out', required=True)
    blind.add_argument('--private-out', required=True)
    for name in ('public', 'private', 'response', 'out', 'receipt-out'):
        restore.add_argument('--' + name, required=True)
    report.add_argument('--reviews')
    report.add_argument('--audit-policy')
    report.add_argument('--audit-reviews')
    report.add_argument('--seed', type=int, default=42)
    report.add_argument('--out', required=True)
    args = parser.parse_args()
    pool, policy = read_json(args.pool), read_json(args.policy)
    if args.command == 'blind-request':
        _fresh_outputs(args.public_out, args.private_out)
        public, private = blind_package(pool, policy, seed=args.seed)
        write_json(args.public_out, public)
        write_json(args.private_out, private)
    elif args.command == 'import-blind':
        _fresh_outputs(args.out, args.receipt_out)
        reviews, receipt = import_blind_reviews(pool, policy, read_json(args.public),
                                                read_json(args.private), read_json(args.response))
        write_json(args.out, reviews)
        write_json(args.receipt_out, receipt)
    else:
        result = search_report(pool, policy, read_json(args.reviews) if args.reviews else None,
                               audit_policy=read_json(args.audit_policy) if args.audit_policy else None,
                               audit_reviews=read_json(args.audit_reviews) if args.audit_reviews else None,
                               bootstrap_seed=args.seed)
        write_json(args.out, result)


if __name__ == '__main__':
    main()
