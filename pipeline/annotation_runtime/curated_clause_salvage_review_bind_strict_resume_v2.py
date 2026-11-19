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
    for field in ('original_visual_result_ref', 'original_worker_terminal_ref', 'original_job_ref'):
        w.checked(candidate[field])
    for k in ('id', 'split', 'batch_id', 'video_path', 'audio_path'):
        need(candidate[k] == original[k], 'Clause candidate source/media binding changed')
    producer_fields = ('need_real_review', 'review_reason', 'pure_rewrite_failure', 'new_safe_facts', 'new_safe_fact_ids')
    need(all(notice[k] == candidate[k] == visual[k] for k in producer_fields),
         'Producer notice/candidate/visual original declaration changed')
    need(notice['new_safe_fact_refs'] == [dict(visual_result_ref=visual_ref, fact_id=fid)
         for fid in notice['new_safe_fact_ids']], 'Producer fact references changed')
    delta = review_delta(visual, repair_visual, oldrepair, w)
    row['review_provenance'].update(original_notice_sha256=canonical_sha(notice),
        original_notice_flags={k: copy.deepcopy(notice[k]) for k in producer_fields},
        independent_review_eligibility=copy.deepcopy(delta))
    expected = dict(need_real_review=delta['review_required'], review_reason=delta['reason'],
        pure_rewrite_failure=delta['pure_rewrite_failure'], new_safe_facts=delta['added_supported_body_facts'],
        new_safe_fact_ids=[f['fact_id'] for f in delta['added_supported_body_facts']])
    differing = [k for k in producer_fields if notice[k] != expected[k]]
    if differing and not delta['review_required'] and delta['pending_status'] == 'no_new_supported_facts':
        delta['pending_status'] = 'source_eligibility_disagreement'
    row['review_provenance']['independent_review_eligibility'] = copy.deepcopy(delta)
    row['review_provenance']['eligibility_disagreement'] = (dict(
        schema='clause_source_eligibility_disagreement_v2', source_key=key,
        reason='producer_independent_coarse_eligibility_differs', differing_fields=differing,
        oldflag=notice['need_real_review'], newflag=delta['review_required'],
        original_notice_sha256=canonical_sha(notice), original_candidate_ref=copy.deepcopy(reference),
        original_visual_ref=copy.deepcopy(visual_ref), original_declaration={k:copy.deepcopy(notice[k]) for k in producer_fields},
        independent_review_eligibility=copy.deepcopy(delta), review_performed=False,
        semantic_approval=False, human_review_completed=False, training_eligible=False) if differing else None)
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


def claimed_part_groups(text):
    """Conservative lexical support requirements, not semantic truth scoring."""
    noun_to_kind = {"hand": "hand", "hands": "hand", "finger": "hand", "fingers": "hand",
        "palm": "hand", "palms": "hand", "wrist": "hand", "wrists": "hand", "fist": "hand", "fists": "hand",
        "thumb": "hand", "thumbs": "hand", "foot": "foot", "feet": "foot", "toe": "foot", "toes": "foot",
        "heel": "foot", "heels": "foot", "arm": "arm", "arms": "arm", "elbow": "arm", "elbows": "arm",
        "shoulder": "arm", "shoulders": "arm", "leg": "leg", "legs": "leg", "knee": "leg", "knees": "leg",
        "thigh": "leg", "thighs": "leg"}
    pattern = r"\b(?:(left|right)\s+)?(" + "|".join(sorted(noun_to_kind, key=len, reverse=True)) + r")\b"
    mentions = [(m.group(1), m.group(2), noun_to_kind[m.group(2)]) for m in re.finditer(pattern, text.lower())]
    groups = []
    for side, noun, kind in mentions:
        if side:
            groups.append({side + "_" + kind})
        elif noun in ("hands", "feet", "arms", "legs"):
            explicit = {s + "_" + kind for s, _, k in mentions if s and k == kind}
            if explicit and not re.search(r"\b(?:both|two)\s+" + noun + r"\b", text, re.I):
                groups.append(explicit)
            else:
                groups.extend(({"left_" + kind}, {"right_" + kind}))
        else:
            groups.append({"left_" + kind, "right_" + kind})
    for words, part in ((r"head|neck", "head"), (r"torso|hips?|waist|pelvis", "torso")):
        if re.search(r"\b(?:" + words + r")\b", text, re.I):
            groups.append({part})
    return groups


DETAIL = re.compile(r'\b(?:fists?|palms?|fingers?|thumbs?|digits?|knuckles?|peace sign|v[- ]sign|OK sign|heart sign|hand[- ]?shapes?|open[- ]hand|closed[- ]hand)\b|\b(?:open|closed|relaxed)\s+(?:hand\s+)?(?:posture|shape)|\bhand\s+(?:posture|shape)', re.I)
UNCERTAIN = re.compile(r"\b(?:uncertain|unclear|unconfirmed|possibly|perhaps|maybe|ambiguous|indistinct|cannot|not|no)\b|\b\w+n['’]t\b", re.I)
BODY_PARTS = {'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso', 'head', 'left_foot', 'right_foot', 'body', 'hips', 'feet'}
COARSE_PARTS = {'left_arm', 'right_arm', 'left_leg', 'right_leg', 'torso', 'head', 'left_hand', 'right_hand', 'left_foot', 'right_foot', 'body', 'hips', 'feet'}


def named_hand_detail(text):
    if DETAIL.search(text):
        return True
    return bool(re.search(r'\b(?:hands?|palms?|fingers?|thumbs?)\b', text, re.I)
        and re.search(r'\b(?:V(?:[- ](?:shape|sign))?|OK|heart)\b', text, re.I))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fact_check(fact, observation, w):
    reasons = []
    value = observation.get('json') or {}
    event = next((e for e in value.get('events', []) if
        observation['window']['window_id'] + ':' + e['id'] == fact['fact_id']), None)
    if observation['status'] not in ('candidate', 'abstain') or event is None:
        need(False, 'New fact absent from checked visual observation')
    if not (fact['raw_text'] == event['description'] and fact['parts'] == event['parts']
            and fact['frame_indices'] == event['frame_indices'] and fact['source_clause'] == event['source_clause']):
        need(False, 'New fact differs from checked visual event')
    text, clause = fact['raw_text'], fact['source_clause']
    if not str(text).strip() or clause.get('raw_source_clause') != text or text not in clause.get('original_description', ''):
        need(False, 'New fact raw text differs from original verbatim clause')
    if DETAIL.search(text):
        reasons.append('coarse_only_no_named_hand_shape_or_finger_detail')
    if UNCERTAIN.search(text):
        reasons.append('uncertain_or_negated_text_not_affirmative_motion')
    parts = set(fact.get('parts', []))
    if not parts or not parts <= COARSE_PARTS:
        need(False, 'New fact has undeclared or invalid parts')
    if any(not (group & parts) for group in claimed_part_groups(text)):
        need(False, 'New fact text names undeclared event parts')
    frames = set(fact.get('frame_indices', []))
    support = {p['part']:set(p['frame_indices']) for p in value.get('part_support', [])}
    if len(frames) < 2 or any(len(frames & support.get(part,set())) < 2 for part in parts):
        need(False, 'New fact required part lacks own two visible event frames')
    evidence = observation.get('source_evidence')
    if not evidence:
        need(False, 'New fact missing checked original raw evidence')
    if evidence:
        refs = evidence['source_artifact_refs']
        for reference in refs.values():
            need(sha(reference['path']) == reference['sha256'], 'Original raw evidence SHA changed')
        receipt, binding = w.checked(refs['receipt.json']), w.checked(refs['binding.json'])
        need(receipt['binding'] == binding == evidence['source_binding'], 'Original receipt/window binding changed')
        need(binding['window'] == observation['window'], 'Original raw evidence window changed')
        folder = Path(refs['receipt.json']['path']).parent.resolve()
        for name, expected in receipt['files_sha256'].items():
            path = (folder / name).resolve()
            need(path.parent == folder and sha(path) == expected, 'Original receipt artifact SHA changed')
        raw = w.checked(refs['response.raw.txt'])
        original = next((e for e in raw.get('events', []) if e.get('id') == clause['raw_event_id']), None)
        if original is None or original.get('description') != clause['original_description']:
            need(False, 'New fact clause differs from original raw event')
        elif not parts <= set(original.get('parts',[])) or not frames <= set(original.get('frame_indices',[])):
            need(False, 'New fact uses undeclared original parts or foreign event frames')
    if not parts & BODY_PARTS:
        reasons.append('hand_only_motion_outside_current_coarse_body_review_scope')
    return reasons


def review_delta(visual, repair_visual, oldrepair, w):
    facts, old = visual['facts'], repair_visual['facts']
    need(all(f in facts for f in old), 'Original admitted fact changed or disappeared')
    added = [f for f in facts if f not in old]
    observations = {o['window']['window_id']:o for o in visual['window_observations']}
    safe, rejected = [], []
    for fact in added:
        observation = observations.get(fact['window_id'])
        need(observation is not None, 'New fact missing checked visual window')
        reasons = fact_check(fact, observation, w)
        if reasons:
            rejected.append(dict(fact_id=fact['fact_id'], reasons=reasons, fact=copy.deepcopy(fact)))
        else:
            safe.append(fact)
    windows = visual['final']['json']['windows']
    pure = bool(not added and facts and repair_visual['final']['status'] == 'failed'
        and all(o['status'] in ('candidate', 'abstain') for o in repair_visual['window_observations'])
        and all(o['status'] in ('candidate', 'abstain') for o in visual['window_observations'])
        and visual['final']['status'] == 'candidate' and windows
        and all(w['caption'].strip() and w['used_fact_ids'] for w in windows)
        and not any(named_hand_detail(f['raw_text']) or UNCERTAIN.search(f['raw_text']) for f in facts)
        and visual['final_timed_caption'] != oldrepair['label']['timed_motion_script'])
    if pure:
        pure = all(not fact_check(fact, observations[fact['window_id']], w) for fact in facts)
    unsafe_detail = any(named_hand_detail(f['raw_text']) for f in added)
    required = bool(safe or pure) and not unsafe_detail
    pending_status = 'declared_handshape_unsafe' if unsafe_detail else None if required else 'no_new_supported_facts'
    return dict(review_required=required, pending_status=pending_status,
        reason='new_safe_coarse_body_facts' if safe else 'pure_rewrite_failure_original_admitted_facts' if pure else 'no_new_supported_facts',
        added_supported_body_facts=copy.deepcopy(safe), added_fact_count=len(added),
        rejected_new_facts=rejected, pure_rewrite_failure=pure, no_semantic_approval=True,
        independent_coarse_only=True, original_facts_or_flags_modified=False)
