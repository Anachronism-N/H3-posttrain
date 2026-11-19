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
    """Only reproducible frozen-filter mismatches with checked artifacts are isolatable."""
    def __init__(self, violations, delta=None):
        self.details = dict(schema='clause_source_evidence_violation_v5', violations=copy.deepcopy(violations))
        self.delta = delta
        self.row = None
        super().__init__('Fact differs from exact frozen original filter replay')


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
    need(candidate['original_visual_result_ref'] == oldrepair['original_visual_result_ref'] == main['visual_result_ref']
         and Path(candidate['original_repair_result_ref']['path']).resolve() == repair_root / 'revisions/repair-20261025-v1' / key / 'result.json'
         and candidate['original_job_ref'] == oldrepair['original_job_ref']
         and Path(candidate['original_job_ref']['path']).resolve() == repair_root / 'jobs' / (key + '.json')
         and Path(candidate['original_worker_terminal_ref']['path']).resolve() == repair_root / 'worker-terminals' / (key + '.json'),
         'Exact main visual/primary repair/collector/terminal reference lineage required')
    terminal = w.checked(candidate['original_worker_terminal_ref'])
    collector = w.checked(candidate['original_job_ref'])
    need(terminal['source_key'] == collector['source_key'] == visual['source_key'] == key
         and terminal['id'] == collector['id'] == visual['id'] == repair_visual['id'] == original_visual['id'] == sid
         and terminal['joint_result_ref'] == candidate['original_repair_joint_result_ref']
         and terminal['original_job_ref'] == candidate['original_job_ref']
         and collector['original_result_ref'] == main_ref
         and collector['refs']['visual_result'] == candidate['original_visual_result_ref']
         and terminal['global_revision'] == collector['global_revision'] == repair_visual['global_revision'] == 'repair-20261025-v1'
         and visual['binding'] == main['source_binding'] == original_visual['binding']
         and repair_visual['original_source_binding'] == collector['source_binding'] == visual['binding']['source_binding']
         and all(visual[field] == candidate[field] for field in ('original_job_ref','original_worker_terminal_ref',
             'original_visual_result_ref','original_repair_result_ref','original_repair_joint_result_ref')),
         'Shared source/repair/media evidence lineage changed')
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
        for checked_ref in immutable_references([row['source_refs'], row['review_provenance'], visual]):
            need(sha(checked_ref['path']) == checked_ref['sha256'], 'Checked isolation source lineage SHA changed')
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


DETAIL = re.compile(r'\b(?:fists?|palms?|fingers?|thumbs?|digits?|knuckles?|peace sign|v[- ]sign|OK sign|heart sign|hand[- ]?shapes?|open[- ]hand|closed[- ]hand|grips?|grasps?|grasping)\b|\b(?:open|closed|relaxed)\s+(?:hand\s+)?(?:posture|shape)|\bhand\s+(?:posture|shape)', re.I)
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


REPLAY_CODE_SHA = {
    "source_repair_gpu_v3/curated_repair_queue_v1.py": "0ff1f304db79170095c97cc44e3d6e7dff38a4fd658691ad502ba49ae5c60548",
    "source_curated_consumer_v1/curated_pipeline.py": "d5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc",
    "source_curated_consumer_v1/ms_bar_annotation.py": "df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e",
    "source_curated_consumer_v1/ms_visual_batch.py": "547b005a779357e25845e01a048d2eb6a79d41d01a9d534f4ebc901405a816e1",
    "source_curated_consumer_v1/ms_caption_rewrite.py": "6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232",
    "source_curated_consumer_v1/ms_visual_observe.py": "b83a048d885edb4f6ae4dc41bb5aeaabe24b8646f764bfea9308164423cf2e6d",
    "source_curated_consumer_v1/ms_optimize.py": "d519390c8588a1019674fa15e9de932b89504171ae682e6c0c1cf75ae2378350",
    "source_curated_consumer_v1/ms_landmarks.py": "570a9be78a20e2926b05a01f774d7b43c756db92fd06b9c51c46b2adc0a0b515",
    "source_curated_consumer_v1/ms_hand_roi.py": "fa8d77d9212772642b584858bb1a01a330ba3efbca7e291f9baa4e3546599941",
    "source_curated_consumer_v1/dwpose_full_extract.py": "040b4030b511bbbb097aab6826a8addf4b48baf5965083d9d197671696458d3a",
    "source_curated_consumer_v1/ms_dance_style.py": "e38032fd009894b9b4747999685ece4d720b91fbb5a62354bb73e4447b4fc0e1",
    "source_curated_consumer_v1/vlm_label.py": "a71cc6732347e7a18b4f60618c01321bbf2460a51cd9f66363ab4a35eb6801dc",
    "source_curated_consumer_v1/motionscript_v2/__init__.py": "2fa82e2455f8045f4219cc090d0aa63d5e070541f9a4e69d7cdcb6bce0b078e6",
    "source_curated_consumer_v1/motionscript_v2/coordinates.py": "324e9045a6c9a2c46fa0e2c7c238fc154f525162506512276f0b0c1c9a903c49",
    "source_curated_consumer_v1/motionscript_v2/chunks.py": "68350fed466f8f066e76c128d6971e7830e2e3e96d75e4c180389e4fb3277d4d",
    "source_curated_consumer_v1/motionscript_v2/landmarks.py": "b54741e91b7e9c640f18657e976f9e73487bee5039c8c23df322dfe1de5aa051",
    "source_curated_consumer_v1/motionscript_v2/events.py": "87d6b7c72e83e8095cf7361ed6e3b37402b9903c5e481c357f6ad6b22d69d985",
    "source_curated_consumer_v1/motionscript_v2/landmark_events.py": "25d81a3d33e316a163276342e54c2f82131da2daa2719a5eced46769f49ed632",
    "source_curated_consumer_v1/motionscript_v2/selection.py": "e03771a3beb09c31fae8c4af89bb50a2eb12ebd5516d1e11e84cc3fef3f0864e"
}
_FROZEN = None


def frozen_protocol():
    global _FROZEN
    for name, expected in REPLAY_CODE_SHA.items():
        need(sha(RT / name) == expected, 'Frozen original replay code SHA changed: ' + name)
    if _FROZEN is None:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(RT / 'source_curated_consumer_v1'))
        import ms_bar_annotation as bar
        need(sha(bar.__file__) == REPLAY_CODE_SHA['source_curated_consumer_v1/ms_bar_annotation.py'],
             'Imported filter is not the original frozen implementation')
        for dep in (bar.batch, bar.core, sys.modules['ms_visual_observe']):
            need(sha(dep.__file__) == REPLAY_CODE_SHA['source_curated_consumer_v1/' + Path(dep.__file__).name],
                 'Imported original replay dependency differs')
        spec = importlib.util.spec_from_file_location('salvage_v5_original_repair',
            RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
        queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(queue)
        _FROZEN = (bar, queue)
    _FROZEN[1].sha = sha
    return _FROZEN


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


class MissingOriginalEvidence(RuntimeError):
    def __init__(self, path):
        self.path = str(path)
        super().__init__('Missing original evidence: ' + self.path)


def required_ref(path):
    if not Path(path).is_file():
        raise MissingOriginalEvidence(path)
    return ref(path)


def original_attempt_references(observation):
    attempt = Path(observation['attempt_path']).resolve()
    return [required_ref(attempt / name) for name in ('receipt.json', 'request.json', 'binding.json',
        'response.raw.txt', 'generation.json', 'response.json', 'final.json')] + [required_ref(attempt.parent / 'manifest.json')]


def prepared_context(visual, w):
    bar, queue = frozen_protocol()
    key = visual['source_key']
    need(re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key), 'Unsafe checked visual source key')
    reference = visual['original_job_ref']
    need(Path(reference['path']).resolve() == ROOT / 'curated222939_repair_v1/jobs' / (key + '.json'),
         'Exact original collector job required')
    job = w.checked(reference)
    need(job['source_key'] == key and job['id'] == visual['id']
         and job['global_revision'] == 'repair-20261025-v1'
         and job['source_binding'] == visual['binding']['source_binding']
         and Path(job['production']).resolve() == ROOT / 'curated222939_production_v1',
         'Original collector source/revision/production binding changed')
    root = Path(job['visual_root']).resolve()
    need(root.parent.name == 'sources' and root.name == visual['id']
         and re.fullmatch(r'shard-[0-9]{4}', root.parent.parent.name)
         and root.parent.parent.parent == ROOT / 'curated222939_production_v1/visual',
         'Exact original main visual source root required')
    need(Path(job['refs']['plan_ref']['path']).resolve() == ROOT / 'curated222939_production_v1/sources' / key / 'bar_plan.json'
         and Path(job['refs']['prepared']['path']).resolve() == root / 'prepared.json'
         and Path(job['refs']['prepared_receipt']['path']).resolve() == root / 'prepared.receipt.json'
         and job['refs']['raw_rgb_manifest'] == visual['raw_rgb_manifest']
         and job['refs']['visual_result'] == visual['original_visual_result_ref'],
         'Original plan/prepared/RGB/visual lineage changed')
    queue.verify_job(job)
    plan = w.checked(job['refs']['plan_ref'])
    prepared = queue.verify_prepared_content(root, plan, bar, '/opt/ms_tools/ffprobe')
    need(prepared == w.checked(job['refs']['prepared']) and prepared['binding'] == visual['binding']
         and bar.batch.digest(visual['manifest_record']) == visual['binding']['manifest_record_sha256'],
         'Checked original prepared/source record differs')
    references = [reference] + list(job['refs'].values())
    references += [required_ref(root / name) for name in ('plan.json', 'plan.line.raw.json', 'binding.json')]
    references += [dict(path=str(RT / name), sha256=expected) for name, expected in REPLAY_CODE_SHA.items()]
    for reference in immutable_references(prepared):
        need(sha(reference['path']) == reference['sha256'], 'Original support mask/source reference SHA changed')
        references.append(copy.deepcopy(reference))
    return bar, queue, job, root, prepared, references


def check_window_assets(visual, observation, root, prepared, w):
    wid = observation['window']['window_id']
    manifest_ref = required_ref(root / 'windows' / wid / 'manifest.json')
    manifest = w.checked(manifest_ref)
    expected = next((m for m in prepared['windows'] if m['window']['window_id'] == wid), None)
    bar, _ = frozen_protocol()
    need(expected is not None and manifest == {**expected, 'binding':visual['binding'],
             'union_manifest_sha256':prepared['union_manifest_sha256'], 'raw2d_mask':prepared['raw2d_mask'], **bar.FLAGS}
         and manifest['window'] == observation['window']
         and manifest['binding'] == visual['binding']
         and manifest['union_manifest_sha256'] == visual['raw_rgb_manifest']['sha256'],
         'Original window/prepared/union RGB binding changed')
    references = [manifest_ref, copy.deepcopy(visual['raw_rgb_manifest'])]
    for frame in manifest['frames']:
        path = (root / 'rgb' / frame['file']).resolve()
        need(path.is_relative_to(root / 'rgb') and sha(path) == frame['sha256'], 'Original actual RGB frame SHA changed')
        references.append(dict(path=str(path), sha256=frame['sha256']))
    for group in manifest.get('hand_roi_evidence', {}).values():
        if isinstance(group, dict):
            for crop in group.get('crops', []):
                need(sha(crop['path']) == crop['sha256'], 'Original native ROI SHA changed')
                references.append(dict(path=crop['path'], sha256=crop['sha256']))
    return expected, references


def replay_fact(fact, observation, raw, manifest, forbidden, bar, verified_refs, *, isolatable):
    replay = bar.filter_supported_observation(raw, manifest, forbidden)
    event = next((event for event in replay['events'] if manifest['window']['window_id'] + ':' + event['id'] == fact['fact_id']), None)
    actual = next((event for event in observation['json']['events'] if manifest['window']['window_id'] + ':' + event['id'] == fact['fact_id']), None)
    expected = dict(id=fact['fact_id'].split(':', 1)[1], description=fact['raw_text'],
        parts=fact['parts'], frame_indices=fact['frame_indices'], source_clause=fact['source_clause'])
    mismatch = event != expected or actual != expected or raw['part_support'] != observation['json']['part_support']
    if mismatch:
        violation = dict(fact_id=fact['fact_id'], window_id=fact['window_id'], fact=copy.deepcopy(fact),
            reason='exact_frozen_filter_replay_differs', original_replayed_event=copy.deepcopy(event),
            checked_observation_event=copy.deepcopy(actual),
            checked_original_artifact_refs=copy.deepcopy(verified_refs),
            original_raw_event=next((copy.deepcopy(e) for e in raw['events'] if e.get('id') == fact['source_clause']['raw_event_id']), None))
        if isolatable:
            raise SourceEvidenceViolation([violation])
        need(False, 'Original admitted fact differs from exact frozen filter replay')
    return replay


def original_attempt_check(fact, observation, original_visual, w, *, source_visual=None):
    need(source_visual is not None, 'Checked salvage lineage required for original main replay')
    path = observation.get('attempt_path')
    if not path:
        raise MissingOriginalEvidence('original_observation.attempt_path')
    bar, queue, job, root, prepared, refs = prepared_context(source_visual, w)
    attempt = Path(path).resolve()
    need(attempt.parent == root / 'windows' / observation['window']['window_id']
         and re.fullmatch(r'attempt-[0-9]+', attempt.name), 'Original attempt path/source/window changed')
    need(original_visual == w.checked(job['refs']['visual_result'])
         and observation in original_visual['window_observations'], 'Original main observation lineage changed')
    manifest, assets = check_window_assets(source_visual, observation, root, prepared, w)
    refs += assets + original_attempt_references(observation)
    receipt = load(attempt / 'receipt.json')
    names = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
    need(names <= set(receipt['files_sha256']), 'Original receipt incomplete artifact SHA inventory')
    for name, expected in receipt['files_sha256'].items():
        artifact = (attempt / name).resolve()
        need(artifact.parent == attempt and sha(artifact) == expected, 'Original receipt artifact SHA changed')
    binding = load(attempt / 'binding.json')
    need(receipt['status'] == 'candidate' and receipt.get('semantic_approval') is False
         and receipt.get('human_review_completed') is False and receipt['binding'] == binding
         and binding == {**original_visual['binding'], 'window': observation['window'], 'frames_sha256': bar.batch.digest(manifest['frames'])},
         'Original receipt/window/frame/source binding changed')
    raw = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
    response, final = load(attempt / 'response.json'), load(attempt / 'final.json')
    generation = bar.core.validate_generation(attempt, load(attempt / 'request.json'))
    model = w.checked(job['refs']['model_identity_ref'])
    need(response == observation['json'] and final['status'] == 'candidate' and final['json'] == response
         and generation.get('model_call_performed') is True and generation['model_identity_sha256'] == model['sha256'],
         'Original response/generation/model identity changed')
    replay_fact(fact, observation, raw, manifest, {p:set(v) for p,v in prepared['forbidden'].items()}, bar, refs, isolatable=False)
    return refs


def primary_repair_attempt_check(fact, observation, visual, repair_visual, w, *, salvaged):
    bar, queue, collector, root, prepared, refs = prepared_context(visual, w)
    wid, key = observation['window']['window_id'], visual['source_key']
    folder = ROOT / 'curated222939_repair_v1/revisions/repair-20261025-v1' / key / 'windows' / wid
    evidence = observation.get('source_evidence') if salvaged else None
    if salvaged:
        need(evidence is not None and evidence['status'] == 'complete_original_model_raw_verified'
             and evidence['actual_frame_SHA_verified'] is True, 'Complete original salvage raw evidence required')
        artifacts = evidence['source_artifact_refs']
        need(Path(artifacts['receipt.json']['path']).resolve().parent == folder, 'Original repair receipt/source/window path changed')
    else:
        need(observation.get('attempt_path') and Path(observation['attempt_path']).resolve() == folder
             and observation in repair_visual['window_observations'], 'Exact existing primary repair observation required')
        artifacts = {name:required_ref(folder / name) for name in ('receipt.json', 'request.json', 'binding.json',
            'response.raw.txt', 'generation.json', 'response.json', 'final.json')}
    required = {'receipt.json', 'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'final.json'}
    need(required <= set(artifacts), 'Original repair six artifact references incomplete')
    for name, reference in artifacts.items():
        need(Path(reference['path']).resolve() == folder / name and sha(reference['path']) == reference['sha256'],
             'Original repair artifact SHA/path changed')
    refs += list(artifacts.values())
    receipt, binding = w.checked(artifacts['receipt.json']), w.checked(artifacts['binding.json'])
    expected_status = 'failed' if salvaged else 'candidate'
    manifest, assets = check_window_assets(visual, observation, root, prepared, w)
    refs += assets
    expected_binding = {**visual['binding'], 'repair_revision':'repair-20261025-v1',
        'window':observation['window'], 'frames_sha256':bar.batch.digest(manifest['frames'])}
    need(receipt['status'] == expected_status and receipt.get('semantic_approval') is False
         and receipt.get('human_review_completed') is False and receipt['binding'] == binding == expected_binding,
         'Original primary repair receipt/source/window/frame binding changed')
    if evidence is not None:
        need(binding == evidence['source_binding'], 'Original salvage evidence binding changed')
    need(required - {'receipt.json'} <= set(receipt['files_sha256']), 'Original repair receipt incomplete SHA inventory')
    if not salvaged:
        need('response.json' in receipt['files_sha256'], 'Successful primary repair response SHA missing')
    for name, expected in receipt['files_sha256'].items():
        path = (folder / name).resolve()
        need(path.parent == folder and sha(path) == expected and artifacts.get(name) == dict(path=str(path), sha256=expected),
             'Original repair receipt artifact SHA changed')
    request, generation = w.checked(artifacts['request.json']), w.checked(artifacts['generation.json'])
    final = w.checked(artifacts['final.json'])
    need(final['status'] == expected_status, 'Original primary repair terminal status changed')
    bar.core.validate_generation(folder, request)
    load_ref = (evidence['original_gpu_load_receipt_ref'] if evidence is not None else
        required_ref(ROOT / 'curated222939_repair_v1/gpu-runtime/repair-20261025-v1/load.receipt.json'))
    need(Path(load_ref['path']).resolve() == ROOT / 'curated222939_repair_v1/gpu-runtime/repair-20261025-v1/load.receipt.json',
         'Original repair model load receipt path changed')
    gpu_load = w.checked(load_ref)
    need(generation.get('model_call_performed') is True
         and generation['raw_response_sha256'] == artifacts['response.raw.txt']['sha256']
         and generation['request_sha256'] == artifacts['request.json']['sha256']
         and generation['model_identity_sha256'] == gpu_load['model_identity_sha256'], 'Original repair generation/raw/request/model changed')
    provenance = request['repair_provenance']
    dispatch_ref = provenance['job_ref']
    need(Path(dispatch_ref['path']).resolve() == ROOT / 'curated222939_repair_v1/dispatch-jobs' / (key + '.json')
         and dispatch_ref == repair_visual['original_job_ref'], 'Exact primary repair dispatch job reference required')
    dispatch = w.checked(dispatch_ref)
    need(dispatch['source_key'] == key and dispatch['id'] == visual['id']
         and dispatch['source_binding'] == binding['source_binding']
         and dispatch['collector_job_ref'] == visual['original_job_ref']
         and w.checked(dispatch['collector_job_ref']) == collector
         and dispatch['refs'] == collector['refs']
         and dispatch['original_result_ref'] == collector['original_result_ref'] == repair_visual['original_result_ref']
         and dispatch['global_revision'] == provenance['global_revision'] == binding['repair_revision'] == repair_visual['global_revision']
         and repair_visual['original_source_binding'] == binding['source_binding']
         and provenance['automatic_reference_fill'] is False, 'Original primary repair dispatch/collector/revision binding changed')
    refs += [load_ref, dispatch_ref]
    for reference in immutable_references(dispatch):
        need(sha(reference['path']) == reference['sha256'], 'Original primary repair job lineage SHA changed')
        refs.append(copy.deepcopy(reference))
    raw = bar.core.strict_json(Path(artifacts['response.raw.txt']['path']).read_bytes())
    forbidden = {p:set(v) for p,v in prepared['forbidden'].items()}
    if not salvaged:
        replay = queue.validate_reobserve(raw, manifest, forbidden, bar)
        need(replay == observation['json'] == w.checked(artifacts['response.json']) == final['json'],
             'Existing primary repair observation differs from original validator replay')
    replay_fact(fact, observation, raw, manifest, forbidden, bar, refs, isolatable=salvaged)
    for reference in refs:
        need(sha(reference['path']) == reference['sha256'], 'Original checked evidence changed during replay')
    return refs


def fact_check(fact, observation, w, *, original_visual=None, source_visual=None, repair_visual=None):
    event = next((e for e in (observation.get('json') or {}).get('events', [])
        if observation['window']['window_id'] + ':' + e['id'] == fact['fact_id']), None)
    need(observation['status'] in ('candidate', 'abstain') and event is not None, 'Fact absent from checked observation')
    need(fact['window_id'] == observation['window']['window_id'] and
         event == dict(id=fact['fact_id'].split(':',1)[1], description=fact['raw_text'], parts=fact['parts'],
            frame_indices=fact['frame_indices'], source_clause=fact['source_clause']), 'Fact differs from checked visual event')
    if original_visual is not None:
        refs = original_attempt_check(fact, observation, original_visual, w, source_visual=source_visual)
    else:
        need(source_visual is not None and repair_visual is not None, 'Checked source/primary repair lineage required')
        refs = primary_repair_attempt_check(fact, observation, source_visual, repair_visual, w,
            salvaged=observation.get('source_evidence') is not None)
    text, parts = fact['raw_text'], set(fact['parts'])
    reasons = []
    if named_hand_detail(text):
        reasons.append('coarse_only_no_named_hand_shape_or_finger_detail')
    if UNCERTAIN.search(text):
        reasons.append('uncertain_or_negated_text_not_affirmative_motion')
    if not parts & BODY_PARTS:
        reasons.append('hand_only_motion_outside_current_coarse_body_review_scope')
    return reasons, refs


def review_delta(visual, repair_visual, oldrepair, w, *, original_visual=None):
    facts, old = visual['facts'], repair_visual['facts']
    need(all(f in facts for f in old), 'Original admitted repair fact changed or disappeared')
    added = [f for f in facts if f not in old]
    observations = {o['window']['window_id']:o for o in visual['window_observations']}
    safe, rejected, violations, references, missing = [], [], [], [], []
    for fact in added:
        observation = observations.get(fact['window_id'])
        need(observation is not None, 'New fact missing checked visual window')
        try:
            reasons, refs = fact_check(fact, observation, w, source_visual=visual, repair_visual=repair_visual)
            references += refs
        except SourceEvidenceViolation as error:
            violations += error.details['violations']
            continue
        except MissingOriginalEvidence as error:
            missing.append(dict(fact_id=fact['fact_id'], window_id=fact['window_id'], missing_path=error.path))
            continue
        if reasons:
            rejected.append(dict(fact_id=fact['fact_id'], reasons=reasons, fact=copy.deepcopy(fact),
                raw_text_preserved_verbatim=True, stored_as_metadata_not_coarse_motion=True))
        else:
            safe.append(fact)
    windows = visual['final']['json']['windows']
    pure = bool(not added and facts and facts == old and repair_visual['final']['status'] == 'failed'
        and all(o['status'] in ('candidate','abstain') for o in repair_visual['window_observations'])
        and all(o['status'] in ('candidate','abstain') for o in visual['window_observations'])
        and visual['final']['status'] == 'candidate' and windows
        and all(window['caption'].strip() and window['used_fact_ids'] for window in windows)
        and visual['final_timed_caption'] != oldrepair['label']['timed_motion_script'])
    if pure:
        need(original_visual is not None, 'Checked original main visual required for pure lineage')
        prior_observations = {o['window']['window_id']:o for o in repair_visual['window_observations']}
        main_observations = {o['window']['window_id']:o for o in original_visual['window_observations']}
        reasons = []
        for fact in facts:
            observation = observations[fact['window_id']]
            need(observation == prior_observations.get(fact['window_id']), 'Pure existing repair observation changed')
            try:
                if observation == main_observations.get(fact['window_id']):
                    local_reasons, refs = fact_check(fact, observation, w, original_visual=original_visual, source_visual=visual)
                else:
                    local_reasons, refs = fact_check(fact, observation, w, source_visual=visual, repair_visual=repair_visual)
                reasons += local_reasons
                references += refs
                if local_reasons:
                    rejected.append(dict(fact_id=fact['fact_id'], reasons=local_reasons, fact=copy.deepcopy(fact),
                        raw_text_preserved_verbatim=True, stored_as_metadata_not_coarse_motion=True))
            except MissingOriginalEvidence as error:
                missing.append(dict(fact_id=fact['fact_id'], window_id=fact['window_id'], missing_path=error.path,
                    original_attempt_path=observation.get('attempt_path')))
        pure = not reasons and not missing
    unsafe_detail = any('coarse_only_no_named_hand_shape_or_finger_detail' in item['reasons'] for item in rejected)
    required = bool(safe or pure) and not unsafe_detail and not missing and not violations
    status = ('source_evidence_violation' if violations else 'source_pending_missing_original_evidence' if missing else
        'declared_handshape_unsafe' if unsafe_detail else None if required else 'no_new_supported_facts')
    delta = dict(review_required=required, pending_status=status,
        reason='new_safe_coarse_body_facts' if safe else 'pure_rewrite_failure_original_admitted_facts' if pure else 'no_new_supported_facts',
        added_supported_body_facts=copy.deepcopy(safe), added_fact_count=len(added), rejected_new_facts=rejected,
        pure_rewrite_failure=pure, no_semantic_approval=True, original_attempt_refs=references,
        missing_original_evidence=missing, independent_coarse_only=True, original_facts_or_flags_modified=False,
        original_filter_replayed=True, prior_repair_facts_not_required_equal_main=True)
    if violations:
        raise SourceEvidenceViolation(violations, delta)
    return delta
