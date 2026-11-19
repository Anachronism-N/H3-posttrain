"""CPU preparation of three empty coarse windows; execution owns no GPU resources."""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
PRIMARY = BAR / 'curated222939_repair_v1'
OUT = BAR / 'curated_coarse_empty_window_reobserve_v1'
REVISION = 'empty-coarse-reobserve-20261025-v1'
SCHEMA = 'curated_coarse_empty_window_reobserve_v1'
TARGETS = {
    'AuuitdcBJyGOnWaH': ('w00000', '14ebe57c4ba38cdd44aad322d6acf1685a9783e6d067d1a23ef8ff4ab602acef',
        '074872bd175ca0321b7fe8a1f3dfb2e358b337dcad5a1a8716c09907a4a64db0',
        [0, 2, 4, 8, 10, 30, 32, 35], 3,
        '3ce2b99ee3efd8eeb3c64ae9429215a98cf9ba0f6ad2df5df824db65c852a10f'),
    'AzPomlvsANsevIFb': ('w00001', '201a675ffb6c3698f38b0984d1e2b93103da6b4df48dbeb1cfcdcecbd049b14f',
        '68ff95ea8c3717ebcd6af3c8808ab16abd936904023aff25abb8382d4ae168b8',
        [38, 40, 42, 44, 46, 48, 50, 52], 0,
        '5d5377981a68ab8f5ec0110b0aef46fb3d095e88a44583992aa0dbe2ba166ea1'),
    'DxNCjtNsxcazNpGb': ('w00001', 'c51a7e8102d5f4110675f994b1646f0955d827199af5bee5c972cf7b4930a586',
        '9269a38ad8e879f3cf7b69c0d3be514fe152e8e203d212c76f9e4033dd782496',
        [50, 52, 54, 56, 58, 60, 74, 80], 3,
        '7cbcee07076ca05d8532490256a231fcfb2f0604953f28a8f6cc854e28e90ca8')}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = module('empty_coarse_protocol', RT / 'curated_io_visual_repair_coarse6_v2.py')
q, bar, pipeline = base.q, base.bar, base.pipeline
COARSE_PARTS = base.COARSE_PARTS
COARSE_RULES = base.COARSE_RULES
FINE_HAND = re.compile(
    r'\b(?:\w*(?:hand|wrist|finger|palm|fist|grip|thumb|knuckle|digit)\w*|openness|'
    r'v(?:[- ]?(?:shape|sign))?|ok|heart|feet|foot|toes?|heels?|ankles?)\b|手|拳|掌|拇|指尖|爱心', re.I)
FILLER = re.compile(r'\b(?:stationary|stillness|static|unknown|uncertain|unclear|unobservable)\b|'
    r'\b(?:stands? still|remains? still|does not move|no (?:motion|movement)|is visible)\b', re.I)


def worker_args():
    return base.worker_args()


def verify_args(args):
    if vars(args) != vars(worker_args()):
        raise ValueError('Native worker configuration/model/sampling/image budget must remain unchanged')


def protocol():
    args = worker_args()
    return {'schema': SCHEMA, 'global_revision': REVISION,
        'allowed_event_parts': list(COARSE_PARTS), 'part_support_parts': list(bar.PARTS),
        'coarse_rules_sha256': q.hashlib.sha256(COARSE_RULES.encode()).hexdigest(),
        'worker_args': vars(args), 'consumer_config_ref': q.identity(RT / 'curated_consumer_s0_v1_template.json'),
        'dependency_refs': [q.identity(base.__file__), q.identity(q.__file__),
                            q.identity(bar.__file__), q.identity(pipeline.__file__)],
        'source_population': [{'id': sid, 'window_id': value[0]} for sid, value in TARGETS.items()],
        'source_population_is_not_full_500_batch': True, 'maximum_extra_source_attempts': 1,
        'maximum_visual_calls_per_source': 1, 'maximum_total_visual_calls': 3,
        'maximum_text_model_calls': 0, 'maximum_images_per_request': 16,
        'primary_budget_root': str(PRIMARY), 'output_root': str(OUT),
        'no_reference_fill': True, 'requires_caller_current_leases_and_job_admission': True,
        'GPU_dispatch_performed': False, **q.FLAGS}


def original_source(sid):
    if sid not in TARGETS:
        raise ValueError('Source outside the fixed three-source preparation')
    ref = {'path': str(PRIMARY / 'jobs/train' / (sid + '.json')), 'sha256': TARGETS[sid][1]}
    original = q.checked(ref)
    terminal_ref = q.identity(PRIMARY / 'worker-terminals/train' / (sid + '.json'))
    terminal = q.checked(terminal_ref)
    if (terminal.get('status') != 'deferred' or terminal.get('schema') != 'curated_repair_watch_v2'
            or terminal.get('original_job_ref') != ref or terminal.get('global_revision') != original['global_revision']
            or terminal.get('source_key') != q.safe_key(original) or terminal.get('id') != sid
            or terminal.get('unresolved_stages') != ['reobserve']
            or original['stages'] != ['reobserve'] or original.get('legacy_visual_demo')
            or Path(original['queue_root']).resolve() != PRIMARY):
        raise q.SourceChanged('Original canonical source/deferred terminal changed')
    return ref, original, terminal_ref


def build_job(sid, protocol_ref):
    ref, original, terminal_ref = original_source(sid)
    wid = TARGETS[sid][0]
    diagnostic = next(d for d in original['window_diagnostics'] if d['window_id'] == wid)
    if (diagnostic['body_status'] != 'present' or diagnostic['status'] != 'abstain'
            or diagnostic['accepted_event_count'] != 0):
        raise q.SourceChanged('Only original valid-body abstaining empty windows may be targeted')
    job = copy.deepcopy(original)
    job['refs'].update(original_canonical_job=ref, original_deferred_terminal=terminal_ref,
                       coarse_protocol=protocol_ref)
    job.update(schema=SCHEMA, global_revision=REVISION, output_root=str(OUT),
        primary_budget_root=str(PRIMARY), allowed_event_parts=list(COARSE_PARTS),
        maximum_reobserve_requests=1, maximum_rewrite_requests=0,
        maximum_total_visual_calls=3, maximum_text_model_calls=0,
        target_window_id=wid, target_manifest_ref=diagnostic['manifest_ref'],
        original_source_window_diagnostics=copy.deepcopy(original['window_diagnostics']),
        text_rewrite_policy='checked_verbatim_only_untargeted_caption_and_refs_frozen',
        ledger_bindings={'original_canonical_job_ref': ref,
                        'original_deferred_terminal_ref': terminal_ref,
                        'original_result_sha256': original['original_result_sha256']})
    return job


def verify_window(job, prepared):
    sid, wid = job['id'], job['target_window_id']
    expected = TARGETS[sid]
    root = Path(job['visual_root'])
    ref = job['target_manifest_ref']
    if (wid != expected[0] or Path(ref['path']).resolve() != root / 'windows' / wid / 'manifest.json'
            or ref['sha256'] != expected[2]):
        raise q.SourceChanged('Original target manifest path/SHA changed')
    manifest = q.checked(ref)
    original = next(m for m in prepared['windows'] if m['window']['window_id'] == wid)
    if (any(manifest.get(k) != v for k, v in original.items())
            or manifest['binding'] != prepared['binding']
            or manifest['raw2d_mask'] != prepared['raw2d_mask']
            or manifest['union_manifest_sha256'] != prepared['union_manifest_sha256']
            or [f['index'] for f in manifest['frames']] != expected[3]):
        raise q.SourceChanged('Original prepared/target window/frame/mask binding changed')
    start, end = manifest['window']['start'], manifest['window']['end']
    for frame in manifest['frames']:
        i = frame['index']
        if (type(i) is not int or not start <= prepared['probe']['times'][i] < end
                or not math.isclose(frame['time'], prepared['probe']['times'][i], rel_tol=0, abs_tol=1e-8)
                or not math.isclose(frame['source_timestamp'], prepared['probe']['source_timestamps'][i], rel_tol=0, abs_tol=1e-8)):
            raise q.SourceChanged('Original actual PTS/half-open window changed')
        q.verify_bytes({'path': str(root / 'rgb' / frame['file']), 'sha256': frame['sha256']})
    inventory = prepared['hand_roi']
    all_crops = q.load(inventory['path'])['crops'] if inventory.get('available') else []
    total = 0
    for part in ('left_hand', 'right_hand'):
        entry = (manifest.get('hand_roi_evidence') or {}).get(part, {})
        crops = entry.get('crops', [])
        if entry.get('frame_indices', []) != sorted(set(c['frame_index'] for c in crops)):
            raise q.SourceChanged('Original ROI frame inventory changed')
        for crop in crops:
            if crop not in all_crops or crop['part'] != part or crop['frame_index'] not in expected[3]:
                raise q.SourceChanged('Original native ROI request changed')
            q.verify_bytes({'path': crop['path'], 'sha256': crop['sha256']})
        total += len(crops)
    if total != expected[4] or len(manifest['frames']) + total > 16:
        raise q.SourceChanged('Original body/ROI image budget changed')
    return manifest


def verify_job(job, probe=False):
    sid = job['id']
    if sid not in TARGETS or job != build_job(sid, job['refs']['coarse_protocol']):
        raise q.SourceChanged('Independent job differs from frozen complete canonical source job')
    if (Path(job['refs']['coarse_protocol']['path']).resolve() != OUT / 'protocol.json'
            or q.checked(job['refs']['coarse_protocol']) != protocol()):
        raise q.SourceChanged('Independent protocol/config/dependency binding changed')
    q.verify_job(job)
    original = q.checked(job['refs']['visual_result'])
    joint = q.checked(job['refs']['joint_result'])
    producer = q.checked(job['refs']['producer_terminal'])
    consumer = q.checked(job['refs']['consumer_terminal'])
    plan = q.checked(job['refs']['plan_ref'])
    binding = original['binding']
    if (plan['binding'] != producer['source_binding'] or plan['binding'] != job['source_binding']
            or producer['plan_ref'] != job['refs']['plan_ref']
            or consumer['joint_candidate_ref'] != job['original_result_ref']
            or job['original_result_ref'] != job['refs']['joint_result']
            or job['original_result_sha256'] != job['original_result_ref']['sha256']
            or binding['source_binding'] != job['source_binding'] or joint['source_binding'] != binding
            or binding['plan_sha256'] != job['refs']['plan_ref']['sha256']
            or bar.batch.digest(plan) != binding['full_source_row_sha256']
            or bar.batch.digest(plan['manifest_record']) != binding['manifest_record_sha256']
            or joint['source_refs'] != plan['manifest_record']['source_refs']
            or joint['visual_result_ref'] != job['refs']['visual_result']):
        raise q.SourceChanged('Original plan/producer/consumer/visual/joint provenance changed')
    for value in (original, joint, producer, consumer, plan, plan['manifest_record']):
        if any(value[k] != job[k] for k in ('id', 'split', 'batch_id') if k in value):
            raise q.SourceChanged('Original full source identity changed')
    if job['refs'].get('style_result') and q.checked(job['refs']['style_result']) != joint['style_result']:
        raise q.SourceChanged('Original style reference changed')
    prepared = q.checked(job['refs']['prepared'])
    if probe:
        q.verify_original_csv(plan['source_refs']['original_csv'], 'train')
        q.verify_references_by_content(plan['source_refs'])
        prepared = q.verify_prepared_content(job['visual_root'], plan, bar, worker_args().ffprobe)
    if prepared['binding'] != binding:
        raise q.SourceChanged('Prepared/current visual binding changed')
    windows = [m['window'] for m in prepared['windows']]
    if ([o['window'] for o in original['window_observations']] != windows
            or original['facts'] != bar.facts_from_observations(windows, original['window_observations'])):
        raise q.SourceChanged('Original window order/success facts changed')
    observation = next(o for o in original['window_observations'] if o['window']['window_id'] == job['target_window_id'])
    if observation['status'] != 'abstain' or observation['json']['events']:
        raise q.SourceChanged('Target is no longer an original empty abstaining window')
    manifest = verify_window(job, prepared)
    request_path = Path(observation['attempt_path']) / 'request.json'
    if q.sha(request_path) != TARGETS[sid][5]:
        raise q.SourceChanged('Original target request SHA changed')
    native = bar.visual_request(worker_args(), Path(job['visual_root']) / 'rgb', manifest,
                                {p: set(v) for p, v in prepared['forbidden'].items()})
    if native != q.load(request_path):
        raise q.SourceChanged('Original native RGB/ROI/sampling request differs from reconstructed request')
    return original, joint, prepared, plan


def validate_coarse(value, manifest, forbidden):
    bar.validate_observation(value, manifest, forbidden)
    if [p['part'] for p in value['part_support']] != list(bar.PARTS):
        raise ValueError('Ten independently judged support entries must retain strict schema order')
    if len(value['uncertainties']) > 20 or any(len(v) > 600 for v in value['uncertainties']):
        raise ValueError('Native uncertainty bounds exceeded')
    if any(len(p['reason']) > 600 for p in value['part_support']):
        raise ValueError('Native support reason bounds exceeded')
    for event in value['events']:
        text = event['description'].strip()
        if any(p not in COARSE_PARTS for p in event['parts']):
            raise ValueError('Only six coarse parts may be claimed')
        if FINE_HAND.search(text):
            raise ValueError('Fine hand/foot details are uncertainties metadata only')
        if FILLER.search(text) or not bar.claimed_part_groups(text):
            raise ValueError('No unknown/static/vague filler as an action event')
        if (len(text) > 1600 or not text.endswith(('.', '!', '?', '。')) or ';' in text
                or len(re.split(r'(?<=[.!?。])\s+', text)) != 1):
            raise ValueError('Exactly one bounded complete coarse body-phase sentence required')
    filtered = bar.filter_supported_observation(value, manifest, forbidden)
    if any(e['parts'] != old['parts'] for e, old in zip(filtered['events'], value['events'])):
        raise ValueError('No synthesized/repaired part references permitted')
    return filtered


def make_request(job, job_ref, prepared, args):
    verify_args(args)
    manifest = verify_window(job, prepared)
    request = bar.visual_request(args, Path(job['visual_root']) / 'rgb', manifest,
                                 {p: set(v) for p, v in prepared['forbidden'].items()})
    header = json.loads(request['messages'][1]['content'][0]['text'])
    header['native_hand_ROI_crop_frames'] = header.pop('hand_ROI_support')
    header['part_support_parts'] = header.pop('parts')
    header['allowed_event_parts'] = list(COARSE_PARTS)
    header.pop('hand_detail_unknown_unless_actual_fingers_distinguishable_in_two_frames')
    header.update(hand_detail_is_unknown_metadata_only=True, ROI_inventory_is_not_hand_visibility_frame_set=True)
    request['messages'][1]['content'][0]['text'] = json.dumps(header, ensure_ascii=False, allow_nan=False)
    parts = request['json_schema']['properties']['events']['items']['properties']['parts']
    parts['items']['enum'] = list(COARSE_PARTS)
    parts['maxItems'] = 6
    request['messages'][0]['content'] += '\n' + COARSE_RULES
    request['prompt_sha256'] = q.hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest()
    request['repair_provenance'] = {'schema': SCHEMA, 'global_revision': REVISION,
        'current_job_ref': job_ref, 'original_canonical_job_ref': job['refs']['original_canonical_job'],
        'original_deferred_terminal_ref': job['refs']['original_deferred_terminal'],
        'original_visual_result_ref': job['refs']['visual_result'], 'original_result_ref': job['original_result_ref'],
        'prepared_ref': job['refs']['prepared'], 'manifest_ref': job['target_manifest_ref'],
        'original_request_sha256': TARGETS[job['id']][5], 'ledger_bindings': job['ledger_bindings'],
        'maximum_visual_calls_per_source': 1, 'maximum_total_visual_calls': 3,
        'maximum_text_model_calls': 0, 'no_reference_fill': True}
    return request


def budget_path(job):
    if Path(job['primary_budget_root']).resolve() != PRIMARY or Path(job['queue_root']).resolve() != PRIMARY:
        raise ValueError('Canonical original primary budget root is required')
    return PRIMARY / 'attempt-budget' / (q.safe_key(job) + '.json')


def check_budget(job):
    path = budget_path(job)
    if not path.exists():
        return None
    claim = q.load(path)
    if (claim.get('source_key') != job['source_key']
            or claim.get('original_result_sha256') != job['original_result_sha256']
            or claim.get('extra_source_attempt') != 1 or claim.get('maximum_extra_source_attempts') != 1):
        raise q.SourceChanged('Canonical reserved budget identity/result SHA/maximum changed')
    return q.identity(path)


def source_freeze():
    return {'schema': SCHEMA, 'global_revision': REVISION, 'worker_ref': q.identity(__file__),
        'test_ref': q.identity(RT / 'test_curated_coarse_empty_window_reobserve_v1.py'),
        'source_sha256': {**bar.code_identity(), str(Path(base.__file__).resolve()): q.sha(base.__file__),
                          str(Path(q.__file__).resolve()): q.sha(q.__file__),
                          str(Path(pipeline.__file__).resolve()): q.sha(pipeline.__file__)},
        'protocol_ref': q.identity(OUT / 'protocol.json'),
        'jobs_manifest_ref': q.identity(OUT / 'jobs.manifest.json'),
        'prepare_only_argv': ['/opt/venv_eval/bin/python', '-B', str(Path(__file__).resolve()), '--prepare-only'],
        'env': {'MS_VISUAL_CPU_SITEPATH': '/opt/venv_eval/lib/python3.11/site-packages'},
        'caller_runtime_current_manager_lease_verifier_required': True,
        'GPU_resource_prepared': False, 'launch_authorized': False, 'GPU_dispatch_performed': False, **q.FLAGS}


def caller_admission(runtime, job_ref):
    """The caller validates its live manager leases; this module never acquires resources."""
    verifier = getattr(runtime, 'verify_current_admission', None)
    if not callable(runtime) or not callable(verifier):
        raise ValueError('Caller-owned runtime with current manager admission verification is required')
    manifest_ref = q.identity(OUT / 'jobs.manifest.json')
    manifest = q.checked(manifest_ref)
    refs = [v['job_ref'] for v in manifest['jobs']]
    if (len(refs) != 3 or job_ref not in refs or manifest['global_revision'] != REVISION
            or {q.checked(r)['id'] for r in refs} != set(TARGETS)):
        raise q.SourceChanged('Caller must admit exactly the frozen three-source job manifest')
    admission = verifier(refs)
    if (not isinstance(admission, dict) or admission.get('jobs_manifest_ref') != manifest_ref
            or admission.get('source_job_refs') != refs or admission.get('out') != str(OUT)
            or admission.get('global_revision') != REVISION or admission.get('caller_pid') != os.getpid()
            or admission.get('tensor_parallel_size') != 2 or not admission.get('resource_admission_ref')
            or len(admission.get('current_lease_refs', [])) != 2
            or len(admission.get('gpus', [])) != 2 or len(set(admission['gpus'])) != 2
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, admission['gpus']))):
        raise ValueError('Caller admission must bind original job refs/current leases/live TP2 environment')
    q.verify_bytes(admission['resource_admission_ref'])
    for ref in admission['current_lease_refs']:
        q.verify_bytes(ref)
    if len({r['path'] for r in admission['current_lease_refs']}) != 2:
        raise ValueError('Two distinct current manager lease refs are required')
    return admission


def claim_once(job, job_ref, runtime):
    verify_job(job, probe=True)
    if q.checked(job_ref) != job:
        raise q.SourceChanged('Independent job reference changed')
    if check_budget(job):
        raise ValueError('Already reserved across editions; no GPU call/redispatch')
    caller_admission(runtime, job_ref)
    runtime()  # A shared model load failure does not reserve any source.
    caller_admission(runtime, job_ref)
    verify_job(job, probe=True)
    path = budget_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (PRIMARY / 'empty-coarse-attempt.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if check_budget(job):
            raise ValueError('Source concurrently reserved; no redispatch')
        claim = q.reserve_attempt(job, PRIMARY, REVISION)
    return claim


def package(job, observations, original, old_joint, prepared, plan):
    windows = [m['window'] for m in prepared['windows']]
    wid = job['target_window_id']
    if (len(observations) != len(original['window_observations'])
            or [o['window'] for o in observations] != windows):
        raise q.SourceChanged('Original complete window population/order changed')
    for observation, old in zip(observations, original['window_observations']):
        if old['window']['window_id'] != wid and observation != old:
            raise q.SourceChanged('Untargeted original observation changed')
    target = next(o for o in observations if o['window']['window_id'] == wid)
    if target['status'] in ('candidate', 'abstain'):
        value = target['json']
        strict = {k: copy.deepcopy(value[k]) for k in ('events', 'part_support', 'uncertainties', 'abstain')}
        strict['events'] = [{k: copy.deepcopy(e[k]) for k in ('id', 'description', 'parts', 'frame_indices')}
                            for e in strict['events']]
        validated = validate_coarse(strict, verify_window(job, prepared), prepared['forbidden'])
        if validated['events'] != value['events'] or validated['part_support'] != value['part_support']:
            raise q.SourceChanged('Target parts/refs/source clauses were fabricated after validation')
    facts = bar.facts_from_observations(windows, observations)
    for window in windows:
        if window['window_id'] != wid:
            if ([f for f in facts if f['window_id'] == window['window_id']]
                    != [f for f in original['facts'] if f['window_id'] == window['window_id']]):
                raise q.SourceChanged('Untargeted original facts/IDs/references changed')
    old_final = original['final'].get('json') or {}
    old_items = {v['window_id']: v for v in old_final.get('windows', [])}
    body = {'windows': [], 'issues': copy.deepcopy(old_final.get('issues', []))}
    for window in windows:
        key = window['window_id']
        local = [f for f in facts if f['window_id'] == key]
        item = {'window_id': key, 'caption': ' '.join(f['raw_text'] for f in local),
                'used_fact_ids': [f['fact_id'] for f in local], 'used_boundary_ids': []}
        if key != wid and key in old_items:
            item = copy.deepcopy(old_items[key])
        body['windows'].append(item)
    boundaries = [r for o in original.get('boundary_observations', [])
                  if o['status'] in ('candidate', 'unknown') for r in o['json']['relations']]
    bar.check_supported_rewrite(body, windows, facts, boundaries)
    old_body = (original['final'].get('json') or {}).get('windows', [])
    if any(new != old for new, old in zip(body['windows'], old_body) if old['window_id'] != wid):
        raise q.SourceChanged('Untargeted original caption/used IDs changed')
    failed = any(o['status'] not in ('candidate', 'abstain') for o in observations)
    full = not failed and all(w['caption'].strip() for w in body['windows'])
    visual = copy.deepcopy(original)
    visual.update(window_observations=copy.deepcopy(observations), facts=facts,
                  original_final=copy.deepcopy(original['final']))
    visual['final'] = {'status': 'candidate' if full else 'failed' if failed else 'abstain',
        'json': body, 'model_call_performed': False, 'model_rewrite_performed': False,
        'derivation': 'verbatim_new_coarse_facts_with_original_untargeted_caption_refs_frozen', **q.FLAGS}
    visual['status'] = visual['final']['status']
    visual['final_timed_caption'] = bar.core.timed_text({'windows': windows}, body['windows'])
    visual['coverage'] = {**original.get('coverage', {}), 'full_window_caption': full,
        'admitted_fact_count': len(facts), 'window_status_counts':
        {s: sum(o['status'] == s for o in observations) for s in {o['status'] for o in observations}}}
    joint = pipeline.joint_candidate(plan['manifest_record'], visual, copy.deepcopy(old_joint['style_result']),
                                     old_joint.get('annotator_model_sha256'))
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in old_joint:
            joint[name] = copy.deepcopy(old_joint[name])
    if joint['style_result'] != old_joint['style_result'] or joint['dance_style_label'] != old_joint['dance_style_label']:
        raise q.SourceChanged('Original style must remain byte-for-byte identical')
    if not full and (joint['joint_complete'] or joint['target_text']):
        raise q.SourceChanged('Failed/abstaining empty window cannot become complete text')
    joint.update(status='joint_repair_candidate' if joint['joint_complete'] else 'incomplete_repair_candidate',
                 labels_complete=bool(joint['joint_complete']), training_approved=False)
    visual.update(q.FLAGS)
    joint.update(q.FLAGS)
    return visual, joint


def execute(job_ref, args, runtime):
    """Later main-only managed TP2 entry; never called by prepare-only or these tests."""
    job = q.checked(job_ref)
    if Path(job_ref['path']).resolve() != OUT / 'jobs' / (q.safe_key(job) + '.json'):
        raise ValueError('Execution job must be bound to its independent output queue')
    verify_args(args)
    if q.load(OUT / 'source.freeze.json') != source_freeze():
        raise q.SourceChanged('Frozen worker/test/native source dependencies changed')
    original, old_joint, prepared, plan = verify_job(job, probe=True)
    if check_budget(job):
        return {'id': job['id'], 'status': 'pending_budget_already_reserved',
                'reason': 'canonical_source_attempt_already_reserved_no_redispatch', **q.FLAGS}
    root = OUT / 'revisions' / REVISION / job['source_key']
    if root.exists():
        raise q.SourceChanged('Independent transaction already exists without original canonical budget')
    request = make_request(job, job_ref, prepared, args)
    preview_ref = q.identity(OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
    if request != q.checked(preview_ref):
        raise q.SourceChanged('Frozen CPU request preview changed')
    manifest = verify_window(job, prepared)
    item = {'root': root / 'windows' / job['target_window_id'] / 'attempt-000001', 'request': request,
        'validator': lambda v: validate_coarse(v, manifest, prepared['forbidden']),
        'binding': {**original['binding'], 'repair_revision': REVISION, 'current_job_ref': job_ref,
            'original_canonical_job_ref': job['refs']['original_canonical_job'],
            'ledger_bindings': job['ledger_bindings'], 'window': manifest['window'],
            'frames_sha256': bar.batch.digest(manifest['frames']), 'request_preview_ref': preview_ref}}
    claim = claim_once(job, job_ref, runtime)
    q.save_once(root / 'original_job.json', job)
    q.save_once(root / 'canonical_budget_claim.json', claim)
    def admitted_runtime():
        caller_admission(runtime, job_ref)
        return runtime()
    result = bar.transactions([item], admitted_runtime, args)[str(item['root'])]
    if result['status'] == 'candidate' and result['json']['abstain']:
        result = {**result, 'status': 'abstain'}
    observations = copy.deepcopy(original['window_observations'])
    index = next(i for i, o in enumerate(observations) if o['window']['window_id'] == job['target_window_id'])
    observations[index] = {**result, 'window': observations[index]['window'], 'attempt_path': str(item['root'])}
    visual, joint = package(job, observations, original, old_joint, prepared, plan)
    provenance = {'schema': SCHEMA, 'global_revision': REVISION, 'original_job_ref': job_ref,
        'original_result_ref': job['original_result_ref'], 'original_visual_result_ref': job['refs']['visual_result'],
        'original_source_binding': job['source_binding'], 'ledger_bindings': job['ledger_bindings'],
        'primary_budget_ref': q.identity(budget_path(job)), 'extra_source_attempts': 1,
        'maximum_extra_source_attempts': 1, 'new_visual_requests': 1, 'new_text_model_requests': 0,
        'maximum_total_visual_calls': 3, 'original_raw_or_terminal_modified': False,
        'style_label_changed': False, 'repair_completed': False,
        'independent_review_status': 'pending_independent_vlm_review', 'new_label_review_required': True, **q.FLAGS}
    visual.update(provenance)
    joint.update(provenance)
    verify_job(job, probe=True)
    q.save_once(root / 'visual_result.json', visual)
    joint['visual_result_ref'] = q.identity(root / 'visual_result.json')
    q.save_once(root / 'result.json', joint)
    terminal = {'schema': SCHEMA, 'id': job['id'], 'source_key': job['source_key'],
        'status': joint['status'], 'global_revision': REVISION, 'original_job_ref': job_ref,
        'joint_complete': joint['joint_complete'], 'joint_result_ref': q.identity(root / 'result.json'), **q.FLAGS}
    if (item['root'] / 'dispatch_error.json').exists():
        terminal['shared_dispatch_failure'] = True
    q.save_once(OUT / 'worker-terminals' / (job['source_key'] + '.json'), terminal)
    if terminal.get('shared_dispatch_failure'):
        raise RuntimeError('Shared generation failed; persisted partial result; stop before reserving another source')
    return terminal


def prepare():
    q.save_once(OUT / 'protocol.json', protocol())
    pref = q.identity(OUT / 'protocol.json')
    jobs, checks = [], []
    for sid in TARGETS:
        job = build_job(sid, pref)
        original, joint, prepared, plan = verify_job(job, probe=True)
        path = OUT / 'jobs' / (job['source_key'] + '.json')
        q.save_once(path, job)
        ref = q.identity(path)
        jobs.append({'source_key': job['source_key'], 'job_ref': ref})
        request = make_request(job, ref, prepared, worker_args())
        manifest = verify_window(job, prepared)
        pp = OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json'
        q.save_once(pp, request)
        native_images = [c for c in request['messages'][1]['content'] if c['type'] == 'image_url']
        images = [{'sha256': q.hashlib.sha256(base64.b64decode(c['image_url']['url'].split(',', 1)[1], validate=True)).hexdigest(),
                   'bytes': len(base64.b64decode(c['image_url']['url'].split(',', 1)[1], validate=True))} for c in native_images]
        if len(images) != len(TARGETS[sid][3]) + TARGETS[sid][4]:
            raise q.SourceChanged('Preview body/ROI original request image budget changed')
        checks.append({'id': sid, 'window_id': job['target_window_id'], 'original_canonical_job_ref': job['refs']['original_canonical_job'],
            'original_deferred_terminal_ref': job['refs']['original_deferred_terminal'], 'original_result_ref': job['original_result_ref'],
            'original_prepared_ref': job['refs']['prepared'], 'target_manifest_ref': job['target_manifest_ref'],
            'body_frame_indices': TARGETS[sid][3], 'body_frames': copy.deepcopy(manifest['frames']),
            'ROI_images': TARGETS[sid][4], 'actual_image_count': len(images), 'image_bytes_and_sha256': images,
            'actual_video_pts_sha256': plan['binding']['actual_video_pts_sha256'],
            'sample_fps': prepared['sampling']['sample_fps'] if 'sampling' in prepared else worker_args().sample_fps,
            'request_preview_ref': q.identity(pp), 'request_digest': q.digest(request),
            'schema_digest': q.digest(request['json_schema']), 'prompt_sha256': request['prompt_sha256'],
            'source_refs': copy.deepcopy(plan['source_refs']), 'original_style_result_digest': q.digest(joint['style_result']),
            'original_success_facts_digest': q.digest(original['facts']),
            'original_untargeted_observations_digest': q.digest([o for o in original['window_observations']
                if o['window']['window_id'] != job['target_window_id']]),
            'primary_budget_reserved_ref': check_budget(job), 'GPU_dispatch_performed': False})
    q.save_once(OUT / 'jobs.manifest.json', {'schema': SCHEMA, 'global_revision': REVISION,
        'source_count': 3, 'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0,
        'population_scope': 'only_three_original_deferred_empty_windows_not_full_batch',
        'jobs': jobs, 'GPU_dispatch_performed': False, **q.FLAGS})
    receipt = {'schema': SCHEMA, 'global_revision': REVISION, 'checks': checks, 'source_count': 3,
        'model_loads': 0, 'visual_model_calls': 0, 'text_model_calls': 0, 'budget_reservations_performed': 0,
        'source_CPU_prepared': True, 'GPU_resource_prepared': False, 'launch_authorized': False,
        'planned_maximum_total_visual_calls': 3, 'GPU_dispatch_performed': False, **q.FLAGS}
    # Reservation state is live, so only the immutable evidence subset is write-once.
    q.save_once(OUT / 'cpu.prepare.evidence.json', {**receipt, 'checks': [
        {k: v for k, v in c.items() if k != 'primary_budget_reserved_ref'} for c in checks]})
    q.save_once(OUT / 'source.freeze.json', source_freeze())
    return {**receipt, 'source_freeze_ref': q.identity(OUT / 'source.freeze.json'),
            'pending_budget_source_ids': [c['id'] for c in checks if c['primary_budget_reserved_ref']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
