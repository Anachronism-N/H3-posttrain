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


class SourceEvidenceViolation(ValueError):
    """Only checked original-event part/frame containment failures are isolatable."""
    def __init__(self, violations, delta=None):
        self.details = dict(schema='clause_source_evidence_violation_v4', violations=copy.deepcopy(violations))
        self.delta = delta
        self.row = None
        super().__init__('New fact uses undeclared original parts or foreign event frames')


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
    original_visual = w.checked(candidate['original_visual_result_ref'])
    source_error = None
    try:
        delta = review_delta(visual, repair_visual, oldrepair, w, original_visual=original_visual)
    except SourceEvidenceViolation as error:
        source_error, delta = error, error.delta
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
    if source_error is not None:
        for reference in immutable_references([row['source_refs'], row['review_provenance'], visual]):
            need(sha(reference['path']) == reference['sha256'], 'Checked isolation source lineage SHA changed')
        source_error.details.update(source_key=key, source_id=sid, split=original['split'], batch_id=bid,
            original_notice_sha256=canonical_sha(notice), original_candidate_ref=copy.deepcopy(reference),
            original_visual_ref=copy.deepcopy(visual_ref), producer_code_refs=copy.deepcopy(visual['offline_code_refs']))
        row['review_provenance']['source_evidence_violation'] = copy.deepcopy(source_error.details)
        source_error.row = row
        raise source_error
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


def fact_check(fact, observation, w, *, original_visual=None, source_visual=None):
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
    if original_visual is not None:
        original_attempt_check(fact, observation, original_visual, w)
    else:
        evidence = observation.get('source_evidence')
        if not evidence:
            need(False, 'New fact missing checked original raw evidence')
        if evidence:
            original, verified_refs = checked_repair_raw_evidence(fact, observation, source_visual, w)
            if not parts <= set(original['parts']) or not frames <= set(original['frame_indices']):
                raise SourceEvidenceViolation([dict(fact_id=fact['fact_id'], window_id=fact['window_id'],
                    original_event_id=clause['raw_event_id'], fact=copy.deepcopy(fact),
                    original_raw_event=copy.deepcopy(original), original_parts=copy.deepcopy(original['parts']),
                    candidate_parts=copy.deepcopy(fact['parts']), original_frame_indices=copy.deepcopy(original['frame_indices']),
                    candidate_frame_indices=copy.deepcopy(fact['frame_indices']),
                    undeclared_parts=sorted(parts - set(original['parts'])),
                    foreign_frame_indices=sorted(frames - set(original['frame_indices'])),
                    source_evidence=copy.deepcopy(evidence), checked_original_artifact_refs=verified_refs)])
    if not parts & BODY_PARTS:
        reasons.append('hand_only_motion_outside_current_coarse_body_review_scope')
    return reasons



def immutable_references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str) and 'byte_offset' not in value:
            yield value
        else:
            for item in value.values():
                yield from immutable_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from immutable_references(item)


def checked_repair_raw_evidence(fact, observation, visual, w):
    """Hash raw receipts, lineage and current original media before source isolation."""
    need(visual is not None, 'Checked source visual required for new raw evidence')
    evidence = observation['source_evidence']
    need(evidence['status'] == 'complete_original_model_raw_verified'
         and evidence['actual_frame_SHA_verified'] is True, 'Original source evidence not complete')
    refs = evidence['source_artifact_refs']
    required = {'receipt.json', 'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'final.json'}
    need(required <= set(refs), 'Original receipt incomplete artifact references')
    folder = Path(refs['receipt.json']['path']).parent.resolve()
    expected_folder = ROOT / 'curated222939_repair_v1/revisions/repair-20261025-v1' / visual['source_key'] / 'windows' / observation['window']['window_id']
    need(folder == expected_folder, 'Original repair receipt/source/window path changed')
    verified = {}
    for name, reference in refs.items():
        need(Path(reference['path']).resolve() == folder / name
             and sha(reference['path']) == reference['sha256'], 'Original raw evidence SHA/path changed')
        verified[name] = copy.deepcopy(reference)
    receipt, binding = w.checked(refs['receipt.json']), w.checked(refs['binding.json'])
    need(receipt['status'] == 'failed' and receipt.get('semantic_approval') is False
         and receipt.get('human_review_completed') is False
         and receipt['binding'] == binding == evidence['source_binding']
         and binding['window'] == observation['window']
         and {k:v for k,v in binding.items() if k not in ('window','frames_sha256','repair_revision')} == visual['binding']
         and binding['repair_revision'] == 'repair-20261025-v1', 'Original receipt/window/source binding changed')
    need(required - {'receipt.json'} <= set(receipt['files_sha256']), 'Original receipt incomplete artifact SHA inventory')
    for name, expected in receipt['files_sha256'].items():
        path = (folder / name).resolve()
        need(path.parent == folder and sha(path) == expected
             and refs.get(name) == dict(path=str(path), sha256=expected), 'Original receipt artifact SHA changed')
    generation, request = w.checked(refs['generation.json']), w.checked(refs['request.json'])
    final, gpu_load = w.checked(refs['final.json']), w.checked(evidence['original_gpu_load_receipt_ref'])
    need(final['status'] == 'failed' and generation.get('model_call_performed') is True
         and generation['raw_response_sha256'] == refs['response.raw.txt']['sha256']
         and generation['request_sha256'] == refs['request.json']['sha256']
         and generation['model_identity_sha256'] == gpu_load['model_identity_sha256'],
         'Original generation/raw/request/model binding changed')
    provenance = request['repair_provenance']
    job = w.checked(provenance['job_ref'])
    need(job['source_key'] == visual['source_key'] and job['id'] == visual['id']
         and job['source_binding'] == binding['source_binding']
         and provenance['global_revision'] == binding['repair_revision']
         and provenance['automatic_reference_fill'] is False, 'Original request/job/source binding changed')
    verified.update(original_gpu_load_receipt=copy.deepcopy(evidence['original_gpu_load_receipt_ref']),
        original_dispatch_job=copy.deepcopy(provenance['job_ref']), raw_rgb_manifest=copy.deepcopy(visual['raw_rgb_manifest']))
    rgb = w.checked(visual['raw_rgb_manifest'])
    source_root = Path(visual['raw_rgb_manifest']['path']).parent.parent.resolve()
    need(rgb['binding'] == visual['binding'], 'Original union RGB source binding changed')
    manifest_ref = ref(source_root / 'windows' / observation['window']['window_id'] / 'manifest.json')
    manifest = w.checked(manifest_ref)
    need(manifest['window'] == observation['window'] and manifest['binding'] == visual['binding']
         and manifest['union_manifest_sha256'] == visual['raw_rgb_manifest']['sha256']
         and hashlib.sha256((json.dumps(manifest['frames'], ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode()).hexdigest() == binding['frames_sha256'], 'Original window/frame manifest changed')
    verified['original_window_manifest'] = manifest_ref
    for kind in ('video', 'audio'):
        asset = rgb['source_identity']['files'][kind + '_path']
        need(asset['path'] == visual['manifest_record'][kind + '_path']
             and sha(asset['path']) == asset['sha256'] == binding['source_binding'][kind + '_sha256'],
             'Original actual source media SHA changed')
        verified['original_' + kind] = copy.deepcopy(asset)
    for frame in manifest['frames']:
        path = (source_root / 'rgb' / frame['file']).resolve()
        need(path.is_relative_to(source_root / 'rgb') and sha(path) == frame['sha256'], 'Original RGB frame SHA changed')
        verified['frame_' + str(frame['index'])] = dict(path=str(path), sha256=frame['sha256'])
    for group in manifest.get('hand_roi_evidence', {}).values():
        if isinstance(group, dict):
            for crop in group.get('crops', []):
                need(sha(crop['path']) == crop['sha256'], 'Original native ROI SHA changed')
                verified['roi_' + canonical_sha(crop)] = copy.deepcopy(crop)
    for name, reference in visual['offline_code_refs'].items():
        need(sha(reference['path']) == reference['sha256'], 'Original clause producer code SHA changed')
        verified['producer_' + name] = copy.deepcopy(reference)
    raw = w.checked(refs['response.raw.txt'])
    clause = fact['source_clause']
    event = next((e for e in raw['events'] if e['id'] == clause['raw_event_id']), None)
    need(event is not None and event['description'] == clause['original_description']
         and clause['source_text_is_verbatim_substring'] is True, 'New fact clause differs from original raw event')
    clauses = re.split(r'(?<=[.!?;])\s+', event['description'])
    index = clause['raw_clause_index']
    need(isinstance(index, int) and 0 <= index < len(clauses)
         and clauses[index] == fact['raw_text'] == clause['raw_source_clause'], 'Original raw clause index/verbatim changed')
    sampled = {f['index'] for f in manifest['frames']}
    need(set(event['frame_indices']) <= sampled, 'Original raw event has non-original sampled frames')
    need(sha(refs['response.raw.txt']['path']) == refs['response.raw.txt']['sha256'], 'Original raw changed during validation')
    return event, verified


class MissingOriginalEvidence(RuntimeError):
    def __init__(self, path):
        self.path = str(path)
        super().__init__('Missing original evidence: ' + self.path)


def original_attempt_references(observation):
    attempt = Path(observation['attempt_path']).resolve()
    return [ref(attempt / name) for name in ('receipt.json', 'request.json', 'binding.json',
        'response.raw.txt', 'generation.json', 'response.json', 'final.json')] + [ref(attempt.parent / 'manifest.json')]


def original_attempt_check(fact, observation, original_visual, w):
    path = observation.get('attempt_path')
    if not path:
        raise MissingOriginalEvidence('original_observation.attempt_path')
    attempt = Path(path).resolve()
    source_root = attempt.parents[2]
    need(source_root.parent.parent.parent == ROOT / 'curated222939_production_v1/visual'
         and source_root.name == original_visual['id']
         and attempt.parent.name == observation['window']['window_id']
         and attempt.parent.parent.name == 'windows' and re.fullmatch(r'attempt-[0-9]+', attempt.name),
         'Original attempt path/source/window changed')
    required = ('receipt.json', 'request.json', 'binding.json', 'response.raw.txt',
        'generation.json', 'response.json', 'final.json')
    for name in required:
        if not (attempt / name).is_file():
            raise MissingOriginalEvidence(attempt / name)
    manifest_path = attempt.parent / 'manifest.json'
    if not manifest_path.is_file():
        raise MissingOriginalEvidence(manifest_path)
    receipt = load(attempt / 'receipt.json')
    need(set(required[1:]) <= set(receipt['files_sha256']), 'Original receipt incomplete artifact SHA inventory')
    for name, expected in receipt['files_sha256'].items():
        artifact = (attempt / name).resolve()
        need(artifact.parent == attempt, 'Original artifact escaped attempt')
        if not artifact.is_file():
            raise MissingOriginalEvidence(artifact)
        need(sha(artifact) == expected, 'Original receipt artifact SHA changed')
    binding, manifest = load(attempt / 'binding.json'), load(manifest_path)
    need(receipt['status'] == 'candidate' and receipt.get('semantic_approval') is False
         and receipt.get('human_review_completed') is False
         and receipt['binding'] == binding
         and binding['window'] == manifest['window'] == observation['window']
         and manifest['binding'] == original_visual['binding']
         and {k:v for k,v in binding.items() if k not in ('window','frames_sha256')} == original_visual['binding']
         and binding['frames_sha256'] == hashlib.sha256((json.dumps(manifest['frames'], ensure_ascii=False, indent=2, allow_nan=False) + chr(10)).encode()).hexdigest(),
         'Original receipt/window/frame/source binding changed')
    need(manifest['union_manifest_sha256'] == original_visual['raw_rgb_manifest']['sha256'],
         'Original window RGB manifest lineage changed')
    w.checked(original_visual['raw_rgb_manifest'])
    source = binding['source_binding']
    need(source['id'] == original_visual['id'], 'Original attempt source ID changed')
    for kind in ('video', 'audio'):
        need(sha(original_visual['manifest_record'][kind + '_path']) == source[kind + '_sha256'],
             'Original source media SHA changed')
    for frame in manifest['frames']:
        artifact = (source_root / 'rgb' / frame['file']).resolve()
        need(artifact.is_relative_to(source_root / 'rgb'), 'Original frame path escaped RGB root')
        if not artifact.is_file():
            raise MissingOriginalEvidence(artifact)
        need(sha(artifact) == frame['sha256'], 'Original RGB frame SHA changed')
    for group in manifest.get('hand_roi_evidence', {}).values():
        if isinstance(group, dict):
            for crop in group.get('crops', []):
                need(sha(crop['path']) == crop['sha256'], 'Original native ROI SHA changed')
    raw, response = load(attempt / 'response.raw.txt'), load(attempt / 'response.json')
    generation, final = load(attempt / 'generation.json'), load(attempt / 'final.json')
    need(response == observation['json'] and final['status'] == 'candidate'
         and final['json'] == response
         and generation['raw_response_sha256'] == sha(attempt / 'response.raw.txt')
         and generation['request_sha256'] == sha(attempt / 'request.json')
         and generation.get('model_call_performed') is True,
         'Original response/generation/filtered observation changed')
    clause = fact['source_clause']
    event = next((e for e in raw['events'] if e['id'] == clause['raw_event_id']), None)
    need(event is not None and event['description'] == clause['original_description']
         and clause.get('source_text_is_verbatim_substring') is True,
         'Original pure clause/event text changed')
    clauses = re.split(r'(?<=[.!?;])\s+', event['description'])
    index = clause['raw_clause_index']
    need(isinstance(index, int) and 0 <= index < len(clauses)
         and clauses[index] == fact['raw_text'] == clause['raw_source_clause'],
         'Original raw clause index/verbatim changed')
    parts, frames = set(fact['parts']), set(fact['frame_indices'])
    support = {p['part']:set(p['frame_indices']) for p in raw['part_support']}
    sampled = {f['index'] for f in manifest['frames']}
    need(parts <= set(event['parts']) and frames <= set(event['frame_indices']) <= sampled
         and all(len(frames & set(event['frame_indices']) & support.get(part,set())) >= 2 for part in parts),
         'Original pure event/own-part two-frame evidence changed')


def review_delta(visual, repair_visual, oldrepair, w, *, original_visual=None):
    facts, old = visual['facts'], repair_visual['facts']
    need(all(f in facts for f in old), 'Original admitted fact changed or disappeared')
    added = [f for f in facts if f not in old]
    observations = {o['window']['window_id']:o for o in visual['window_observations']}
    safe, rejected, violations = [], [], []
    for fact in added:
        observation = observations.get(fact['window_id'])
        need(observation is not None, 'New fact missing checked visual window')
        try:
            reasons = fact_check(fact, observation, w, source_visual=visual)
        except SourceEvidenceViolation as error:
            violations.extend(error.details['violations'])
            continue
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
    original_attempt_refs, missing = [], []
    if pure:
        need(original_visual is not None and facts == old == original_visual['facts'],
             'Pure facts differ from checked original repair/visual facts')
        original_observations = {o['window']['window_id']:o for o in original_visual['window_observations']}
        reasons = []
        for fact in facts:
            observation = observations[fact['window_id']]
            need(observation == original_observations.get(fact['window_id']),
                 'Pure original window observation changed')
            try:
                reasons += fact_check(fact, observation, w, original_visual=original_visual)
                original_attempt_refs += original_attempt_references(observation)
            except MissingOriginalEvidence as error:
                missing.append(dict(fact_id=fact['fact_id'], window_id=fact['window_id'],
                    missing_path=error.path, original_attempt_path=observation.get('attempt_path')))
        pure = not reasons and not missing
    unsafe_detail = any(named_hand_detail(f['raw_text']) for f in added)
    required = bool(safe or pure) and not unsafe_detail
    pending_status = ('source_pending_missing_original_evidence' if missing else
        'declared_handshape_unsafe' if unsafe_detail else None if required else 'no_new_supported_facts')
    delta = dict(review_required=required, pending_status=pending_status,
        reason='new_safe_coarse_body_facts' if safe else 'pure_rewrite_failure_original_admitted_facts' if pure else 'no_new_supported_facts',
        added_supported_body_facts=copy.deepcopy(safe), added_fact_count=len(added),
        rejected_new_facts=rejected, pure_rewrite_failure=pure, no_semantic_approval=True,
        original_attempt_refs=original_attempt_refs, missing_original_evidence=missing,
        independent_coarse_only=True, original_facts_or_flags_modified=False)

    if violations:
        delta.update(review_required=False, pending_status='source_evidence_violation',
            reason='checked_original_event_part_frame_scope_violation', pure_rewrite_failure=False,
            no_semantic_approval=True, review_performed=False, model_calls_performed=0)
        raise SourceEvidenceViolation(violations, delta)
    return delta
