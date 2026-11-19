"""One to five changed supported clause-salvage editions; independent source-video audit."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

RT = Path(__file__).resolve().parent
ROOT = RT.parent
CANDIDATES = ROOT / 'curated222939_clause_salvage_v1'
OUT = ROOT / 'curated_clause_salvage_reviews_pilot5_v1'
EDITION = 'clause-salvage-20261025-v1'
PHASE = 'curated_clause_salvage_review_pilot5_v1'
MANIFEST = ROOT / 'curated222939_inputs_v1/manifest.summary.json'
MANIFEST_SHA = 'a1eb129cf8557b4adeb4e11e00ace334ecb1575e43a587d665e33f09410e2bfd'
SUPPORT = RT / 'curated_review_extension_watch_v1.py'
SUPPORT_SHA = '410cd5384a36664bedd2c6eebd4bd4ffe5024653510219307cdaa8e75bc72f23'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def need(ok, message):
    if not ok:
        raise ValueError(message)


class NotReady(RuntimeError):
    pass


def support():
    need(sha(SUPPORT) == SUPPORT_SHA, 'Frozen original review support changed')
    spec = importlib.util.spec_from_file_location('clause_pilot5_support', SUPPORT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bind_notice(notice, manifest, w):
    key = notice['source_key']
    need(re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key), 'Unsafe clause source key')
    need(notice['global_revision'] == EDITION and notice.get('risk_review_case') is True,
         'Explicit new clause edition/risk case required')
    reference = notice['joint_candidate_ref']
    need(notice['result_ref'] == reference
         and Path(reference['path']).resolve() == CANDIDATES / 'revisions' / EDITION / key / 'result.json',
         'Only exact new clause JOINT result is reviewable')
    candidate = w.checked(reference)
    bid, sid = candidate['batch_id'], candidate['id']
    original = manifest.members(bid).get(sid)
    need(original is not None and notice['id'] == sid and key == original['split'] + '/' + sid
         and notice['split'] == original['split'] and notice['batch_id'] == bid,
         'Clause source belongs to a different fixed main manifest member')
    need(candidate['global_revision'] == EDITION and candidate.get('risk_review_case') is True
         and candidate.get('style_label_changed') is False and candidate.get('model_calls') == 0
         and candidate.get('attempt_budget_consumed') is False,
         'Zero-VLM clause derivation with unchanged style required')
    repair_root = ROOT / 'curated222939_repair_v1'
    oldrepair_ref = candidate['original_repair_joint_result_ref']
    need(Path(oldrepair_ref['path']).resolve() == repair_root / 'joint/revisions/repair-20261025-v1' / key / 'result.json',
         'Original real main repair JOINT lineage required')
    oldrepair = w.checked(oldrepair_ref)
    main_ref = oldrepair['original_result_ref']
    need(Path(main_ref['path']).resolve() == ROOT / 'curated222939_production_v1/sources' / key /
         'joint-results' / (main_ref['sha256'] + '.json'), 'Original main content-addressed JOINT lineage required')
    main = w.checked(main_ref)
    need(candidate['original_repair_result_ref'] == oldrepair['repair_result_ref'], 'Original visual repair reference changed')
    repair_visual = w.checked(candidate['original_repair_result_ref'])
    need(all(oldrepair.get(k) == original[k] and main.get(k) == original[k]
             for k in ('id', 'split', 'batch_id', 'video_path', 'audio_path')),
         'Original main/repair media/source differs')
    need(candidate['style_result'] == main['style_result']
         and candidate['label']['dance_style'] == main['label']['dance_style'], 'Original style changed')
    visual_ref = candidate['visual_result_ref']
    need(notice['visual_result_ref'] == visual_ref and Path(visual_ref['path']).resolve() ==
         CANDIDATES / 'revisions' / EDITION / key / 'visual_result.json', 'Exact offline visual facts reference required')
    visual = w.checked(visual_ref)
    need(candidate['label']['timed_motion_script'] == visual['final_timed_caption']
         and visual['final']['derivation'] == 'verbatim_admitted_fact_concatenation',
         'Timed label differs from verbatim admitted facts')
    windows, facts = visual['final']['json']['windows'], visual['facts']
    for window in windows:
        local = [f for f in facts if f['window_id'] == window['window_id']]
        need(window['caption'] == ' '.join(f['raw_text'] for f in local)
             and window['used_fact_ids'] == [f['fact_id'] for f in local]
             and window['used_boundary_ids'] == [], 'Fact clause text changed or inferred boundary added')
    for field in ('raw_rgb_manifest', 'hand_roi'):
        need(candidate.get(field) == main.get(field), 'Original native evidence lineage changed: ' + field)
        if candidate.get(field, {}).get('path'):
            need(sha(candidate[field]['path']) == candidate[field]['sha256'], 'Original evidence SHA changed')
    row = w.bound_row(original, candidate, reference, edition=EDITION,
        origin_root=ROOT / 'curated222939_production_v1',
        stream_path=CANDIDATES / 'review_required.jsonl', original_joint_ref=main_ref)
    refs = dict(joint_label=reference, original_joint_label=main_ref,
        original_repair_joint_label=oldrepair_ref, original_repair_result=candidate['original_repair_result_ref'],
        clause_visual_facts=visual_ref, original_visual_result=candidate['original_visual_result_ref'],
        original_worker_terminal=candidate['original_worker_terminal_ref'], original_collector_job=candidate['original_job_ref'])
    row['source_refs']['clause_salvage_review_lineage'] = copy.deepcopy(refs)
    delta = review_delta(visual, repair_visual, oldrepair)
    need(notice['need_real_review'] is delta['review_required']
         and candidate['need_real_review'] is delta['review_required']
         and notice['review_reason'] == candidate['review_reason'] == delta['reason']
         and notice['pure_rewrite_failure'] is candidate['pure_rewrite_failure'] is delta['pure_rewrite_failure']
         and notice['new_safe_facts'] == candidate['new_safe_facts'] == delta['added_supported_body_facts']
         and notice['new_safe_fact_ids'] == candidate['new_safe_fact_ids'] ==
             [f['fact_id'] for f in delta['added_supported_body_facts']]
         and notice['new_safe_fact_refs'] == [dict(visual_result_ref=visual_ref, fact_id=f['fact_id'])
             for f in delta['added_supported_body_facts']], 'Producer review eligibility differs from verified facts')
    row['review_provenance']['review_delta'] = delta
    roi = row.get('hand_roi') or {}
    row['review_provenance'].update(original_batch_source_count=manifest.descriptors[bid]['count'],
        original_repair_joint_result_ref=oldrepair_ref,
        original_repair_result_ref=candidate['original_repair_result_ref'],
        original_visual_result_ref=candidate['original_visual_result_ref'], clause_visual_result_ref=visual_ref,
        every_selected_case_is_risk_reviewed=True, fixed_pilot_case_not_population_sample=True,
        original_review_scores_reused=False, partial_unknown_never_training_eligible=True)
    if roi.get('available') is True:
        row['review_provenance']['original_roi_root'] = roi['path']
    return row



def review_delta(visual, repair_visual, oldrepair):
    facts, old = visual['facts'], repair_visual['facts']
    added = [f for f in facts if f not in old]
    shape = r'\b(fists?|palms?|fingers?|thumbs?|knuckles?|peace sign|v[- ]sign|hand[- ]?shapes?)\b'
    for fact in added:
        need(not re.search(shape, fact['raw_text'], re.I), 'New named hand shapes must remain metadata, not facts')
    body = {'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso', 'head', 'hips', 'body', 'feet', 'left_foot', 'right_foot'}
    safe = [f for f in added if isinstance(f.get('parts'), list) and set(f['parts']) & body
            and len(set(f.get('frame_indices', []))) >= 2 and str(f['raw_text']).strip()]
    pure = bool(not added and facts and repair_visual['final']['status'] == 'failed'
                and all(o['status'] in ('candidate', 'abstain') for o in repair_visual['window_observations'])
                and all(o['status'] in ('candidate', 'abstain') for o in visual['window_observations'])
                and visual['final_timed_caption'].strip()
                and visual['final_timed_caption'] != oldrepair['label']['timed_motion_script'])
    return dict(review_required=bool(safe or pure),
        reason='new_safe_coarse_body_facts' if safe else 'pure_rewrite_failure_original_admitted_facts' if pure
            else 'no_new_supported_facts',
        added_supported_body_facts=copy.deepcopy(safe), added_fact_count=len(added),
        pure_rewrite_failure=pure, no_semantic_approval=True)


def ready_records(selection, selection_sha):
    need(Path(selection).resolve().parent == RT and sha(selection) == selection_sha,
         'Exact frozen one-to-five-source selection SHA/path required')
    chosen = load(selection)
    need(chosen['schema'] == PHASE and chosen['global_revision'] == EDITION
         and chosen['original_manifest_ref'] == dict(path=str(MANIFEST), sha256=MANIFEST_SHA)
         and chosen['total_source_count'] == 222939 and chosen['expected_batch_count'] == 448
         and 1 <= chosen['selected_source_count'] <= chosen['candidate_source_count'] <= 5
         and len(chosen['cases']) == chosen['candidate_source_count']
         and len({n['source_key'] for n in chosen['cases']}) == chosen['candidate_source_count'],
         'One-to-five fixed changed cases and full222939/448 denominator required')
    need(sha(MANIFEST) == MANIFEST_SHA, 'Original full curated manifest changed')
    w = support()
    manifest = w.Manifest(MANIFEST)
    records, references = [], [ref(selection), ref(MANIFEST)]
    for case in chosen['cases']:
        row = bind_notice(case['notice'], manifest, w)
        need(case['source_key'] == row['split'] + '/' + row['id']
             and case['label_result_ref'] == row['review_provenance']['label_result_ref']
             and case['input_row_sha256'] == canonical_sha(row), 'Fixed case or label edition changed')
        required = row['review_provenance']['review_delta']['review_required']
        need(case['review_required'] is required and case['review_reason'] ==
             row['review_provenance']['review_delta']['reason'], 'Frozen changed-facts review eligibility changed')
        if required:
            records.append(row)
        references += case['frozen_refs']
        for reference in case['frozen_refs']:
            need(sha(reference['path']) == reference['sha256'], 'Frozen case source/label SHA changed')
    need(len(records) == chosen['selected_source_count'], 'Actual changed-facts review count changed')
    return records, references


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def review_records(records, out, reviewer, review, selected_ref):
    need(1 <= len(records) <= 5 and len({r['id'] for r in records}) == len(records),
         'Finite one-to-five actual changed-source scope required')
    need(all(r['review_provenance']['review_delta']['review_required'] for r in records),
         'Sources without new supported facts or pure rewrite failure must not call VLM')
    w = support()
    summaries, audits = [], {}
    def audit(row, root, **kwargs):
        result = reviewer(row, root, **kwargs)
        audits[review.batch.digest(row)] = copy.deepcopy(result)
        return result
    for row in records:
        bid = row['split'] + '-' + hashlib.sha256(row['id'].encode()).hexdigest()[:24]
        value = review.review_batch_ready([row], out / EDITION, audit, batch_id=bid,
            expected_source_ids=[row['id']], sample_count=1, risk_count=0, expansion_counts=())
        value = w.gate_summary(value, edition=EDITION, provenance=row['review_provenance'])
        value.update(original_batch_id=row['batch_id'],
            original_batch_source_count=row['review_provenance']['original_batch_source_count'],
            original_batch_review_complete=False, total_source_count=222939, expected_batch_count=448,
            fixed_pilot_case_not_population_sample=True, every_selected_case_is_risk_reviewed=True,
            selected_manifest_ref=selected_ref)
        w.publish(out / EDITION / bid, value, [row], review, audits)
        summaries.append(value)
    value = dict(schema=PHASE, global_revision=EDITION, label_edition=EDITION,
        total_source_count=222939, expected_batch_count=448, selected_source_count=len(records),
        selected_source_ids=[r['id'] for r in records], processed_source_count=len(summaries),
        sampled_sources_reviewed=sum(s['sampled_reviewed_denominator'] for s in summaries),
        AI_error_candidate_count=sum(s['confirmed_error_count'] for s in summaries),
        label_incomplete_source_ids=[r['id'] for r in records if not review.label_complete(r)],
        original_batch_ids=[r['batch_id'] for r in records],
        review_record_refs=[s['review_records_ref'] for s in summaries],
        selected_manifest_ref=selected_ref, original_manifest_ref=dict(path=str(MANIFEST), sha256=MANIFEST_SHA),
        original_batch_review_complete=False, all_batches_reviewed=False,
        fixed_pilot_case_not_population_sample=True, statistical_population_error_estimate=False,
        source_records_consumed=True, original_review_scores_reused=False,
        human_resolution_stream=None, **w.FLAGS)
    selected = load(selected_ref['path'])
    skipped = [dict(source_key=c['source_key'], source_id=c['notice']['id'],
        original_batch_id=c['notice']['batch_id'], label_result_ref=c['label_result_ref'],
        status='no_new_supported_facts', independent_review_status='pending',
        review_performed=False, semantic_approval=False, training_eligible=False,
        human_review_completed=False) for c in selected['cases'] if not c['review_required']]
    value.update(candidate_source_count=selected['candidate_source_count'],
                 unreviewed_candidates=skipped, no_new_supported_facts_count=len(skipped))
    w.atomic(out / 'unreviewed_candidates.json', dict(items=skipped, **w.FLAGS))
    w.atomic(out / 'run.summary.json', value)
    w.atomic(out / 'progress.json', value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--selection-sha', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-reference', required=True)
    p.add_argument('--annotator-model-reference', required=True)
    p.add_argument('--ms-source', required=True)
    a = p.parse_args()
    need(Path(a.out).resolve() == OUT and Path(a.ms_source).resolve() == RT / 'source_curated_review_v2',
         'Only independent pilot5 output/frozen reviewer allowed')
    rows, _ = ready_records(a.selection, a.selection_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'reviewer.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, a.ms_source)
        import ms_batch_review as review
        local = review.LocalReviewer(a.model, OUT, model_reference=a.model_reference,
            annotator_model_reference=a.annotator_model_reference,
            ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
        print(json.dumps(review_records(rows, OUT, w_reviewer(local), review, ref(a.selection)),
                         ensure_ascii=False), flush=True)


def w_reviewer(local):
    return support().EditionReviewer(local)


if __name__ == '__main__':
    main()
