"""Fixed-three new body observations; no resource acquisition or text model calls."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
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
OUT = BAR / 'curated_coarse_reobserve_fixed3_v1'
REVISION = 'coarse-reobserve-fixed3-20261025-v1'
ORIGINAL_REVISION = 'repair-20261025-v1'
SCHEMA = 'curated_coarse_reobserve_fixed3_v1'
TARGETS = {
    'mrDPKCMnBHBbcIpO': ('w00000', '6589819d03c31920f14076dea69225f5bc5fbc8146f745464e9e793ba3555709',
        '1737b1a11a2c59f7ce6ef76b6d3df4b251627f52c37e9acc9a542d96bbcad38c',
        [0, 2, 4, 6, 8, 14, 28, 32], 5, '206795ed3f17dda3579280ceeb23f82c7cca0a90fde222bff17d26883b85c211'),
    'EcdimtpCvgeGIGNw': ('w00001', 'f7db2a755aa59d3785ba89946bf727da4b214d38507eceb732d371e25d353313',
        '2f7846b2c1007cc206365a5bf9cd38ea0452af1bd5194b64568dcbc36db2fa96',
        [25, 29, 33, 36, 37, 39, 68, 80], 2, 'a947f588c4e3b2b0b374b709fdd2536b3296a15ebb49d369a0c3ffd8a39b2d37'),
    'FSjBuXNyaWYFrpXj': ('w00003', '1a0acd51e20dad4d2280003469bdbd68c529681c0338be574769c6dee72c03aa',
        '69848b59d0767d2f7f090d422c6413abcc12ecc4f700d7e44ce85caa336396f4',
        [56, 58, 60, 64, 72, 74, 76, 80], 6, 'a42391b13217ef65be8836b6a48cbfb4364225f8482db5c7d1b28ac3a1e93fe8')}
PINNED = {
    'source_curated_consumer_v1/ms_bar_annotation.py': 'df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e',
    'source_curated_consumer_v1/ms_caption_rewrite.py': '6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232',
    'source_curated_consumer_v1/ms_visual_batch.py': '547b005a779357e25845e01a048d2eb6a79d41d01a9d534f4ebc901405a816e1',
    'source_curated_consumer_v1/ms_landmarks.py': '570a9be78a20e2926b05a01f774d7b43c756db92fd06b9c51c46b2adc0a0b515',
    'source_curated_consumer_v1/ms_visual_observe.py': 'b83a048d885edb4f6ae4dc41bb5aeaabe24b8646f764bfea9308164423cf2e6d',
    'source_curated_consumer_v1/ms_optimize.py': 'd519390c8588a1019674fa15e9de932b89504171ae682e6c0c1cf75ae2378350',
    'source_curated_consumer_v1/ms_hand_roi.py': 'fa8d77d9212772642b584858bb1a01a330ba3efbca7e291f9baa4e3546599941',
    'source_curated_consumer_v1/dwpose_full_extract.py': '040b4030b511bbbb097aab6826a8addf4b48baf5965083d9d197671696458d3a',
    'source_curated_consumer_v1/curated_pipeline.py': 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc',
    'source_repair_gpu_v3/curated_repair_queue_v1.py': '0ff1f304db79170095c97cc44e3d6e7dff38a4fd658691ad502ba49ae5c60548',
    'curated_io_visual_repair_coarse6_v2.py': 'ad8d98cc6b90afd6c6f2c639f273961c045c1df7211e9c84ed8cef9821fa06c5',
    'curated_repair_watch_v2.py': 'f46950a5ea6d0b6d406bd297b990f4c7ca9d8a4d45da3739873e1ae0e806b077',
    'curated_consumer_s0_v1_template.json': '7e72d21429d47d9de689c57671ec09ea7d48f6e1ae2b4f48ccafe9cdfa3e849c'}


def verify_native_sources():
    for name, digest in PINNED.items():
        if hashlib.sha256((RT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Original frozen native source changed: ' + name)


verify_native_sources()
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = module('fixed3_original_coarse_protocol', RT / 'curated_io_visual_repair_coarse6_v2.py')
q, bar, pipeline = base.q, base.bar, base.pipeline
COARSE_PARTS = base.COARSE_PARTS
COARSE_RULES = """
EXPLICIT OVERRIDE FOR THIS NEW SIX-PART OBSERVATION: ignore any older SYSTEM
requirement to narrate hand/finger shape, hand position, wrist, contact or grip.
This is a new observation of the same supplied pixels, not a rewrite or deletion
of old event text. The available_crop_inventory_not_visibility_or_handtruth
header is ONLY the unchanged native crop INVENTORY; it is NOT visibility,
complete visible body-frame support, hand truth, or automatic support evidence.
Return all TEN part_support rows in original schema order, independently judged.
Events may name ONLY head, torso, left_arm, right_arm, left_leg, right_leg.
Each event is ONE complete body-phase sentence with at least TWO distinct
actual BODY frames per claimed part intersecting that part's own visible support.
A native crop is not another temporal observation. Do not copy old supports,
add parts, widen frame refs, borrow arm support as hand truth, or fabricate IDs.
Fine hand/wrist/finger/palm/fist/shape/orientation/contact/grip/openness and foot
fine details belong ONLY in uncertainties metadata, NEVER event descriptions.
Opening/spreading/raising independently visible ARMS is allowed without any
unknown hand detail attached. Do not suppress supported body motion for lack
of fine-hand evidence. No static, visibility-only, unknown or vague filler.
Only RGB pixels support action claims. If no supported body phase, abstain.
"""
FINE_HAND = re.compile(
    r'\b(?:\w*(?:hand|wrist|finger|palm|fist|grip|thumb|knuckle|digit)\w*|openness|'
    r'v(?:[- ]?(?:shape|sign))?|ok|heart|contact|shape|touch(?:es|ing)?|feet|foot|toes?|heels?|ankles?)\b|手|拳|掌|拇|指尖|爱心', re.I)
FILLER = re.compile(r'\b(?:stationary|stillness|static|motionless|visible|visibility|unknown|uncertain|unclear|unobservable|possibly|maybe|apparently)\b|'
    r'\b(?:stands? still|remains? still|does not move|no (?:motion|movement)|is visible)\b', re.I)


def worker_args():
    return base.worker_args()


def verify_args(args):
    if vars(args) != vars(worker_args()):
        raise ValueError('Native worker configuration/model/sampling/image budget must remain unchanged')



def original_source(sid):
    if sid not in TARGETS:
        raise ValueError('Source outside fixed three-source scope')
    ref = {'path': str(PRIMARY / 'jobs/train' / (sid + '.json')), 'sha256': TARGETS[sid][1]}
    original = q.checked(ref)
    stages = ['reobserve', 'rewrite'] if sid == 'FSjBuXNyaWYFrpXj' else ['reobserve']
    if (original['id'] != sid or original['source_key'] != q.safe_key(original)
            or original['global_revision'] != ORIGINAL_REVISION or original['stages'] != stages
            or original.get('legacy_visual_demo') or Path(original['queue_root']).resolve() != PRIMARY
            or original['maximum_extra_source_attempts'] != 1
            or original['maximum_reobserve_requests'] < 1):
        raise q.SourceChanged('Original canonical fixed-three identity/stages/budget changed')
    return ref, original


def model_reference_record(args):
    ref = q.identity(args.model_reference)
    record = q.checked(ref)
    identity = record.get('model', record.get('model_identity', record))
    if (identity.get('full_content_hash_computed_once_per_process') is not True
            or Path(identity['root']).resolve() != Path(args.model).resolve()):
        raise q.SourceChanged('Native frozen model reference differs')
    content = {name: {k: value[k] for k in ('path', 'sha256')} for name, value in identity['files'].items()}
    if bar.batch.digest({'root': identity['root'], 'files': content}) != identity['sha256']:
        raise q.SourceChanged('Native model reference content digest differs')
    return ref, identity['sha256']


def protocol():
    verify_native_sources()
    args = worker_args()
    model_ref, model_sha = model_reference_record(args)
    return {'schema': SCHEMA, 'global_revision': REVISION, 'original_global_revision': ORIGINAL_REVISION,
        'allowed_event_parts': list(COARSE_PARTS), 'part_support_parts': list(bar.PARTS),
        'coarse_rules_sha256': hashlib.sha256(COARSE_RULES.encode()).hexdigest(),
        'worker_args': vars(args), 'runtime_model_reference': model_ref, 'model_identity_sha256': model_sha,
        'native_source_sha256': {str(RT / k): v for k, v in PINNED.items()},
        'source_population': [{'id': sid, 'window_id': v[0], 'original_job_sha256': v[1]} for sid, v in TARGETS.items()],
        'original_source_selector': {'shard': args.shard, 'nshard': args.nshard},
        'resource_scope_selector': {'shard': 0, 'nshard': 1},
        'maximum_extra_source_attempts': 1, 'maximum_visual_calls_per_source': 1,
        'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0,
        'primary_budget_root': str(PRIMARY), 'output_root': str(OUT),
        'caller_owned_current_admission_required': True, 'GPU_dispatch_performed': False,
        'no_reference_fill': True, 'no_old_ready_or_cursor_access': True, **q.FLAGS}


def build_job(sid, protocol_ref):
    ref, original = original_source(sid)
    wid = TARGETS[sid][0]
    diagnostic = next(d for d in original['window_diagnostics'] if d['window_id'] == wid)
    if diagnostic['body_status'] != 'present' or diagnostic['status'] != 'abstain' or diagnostic['accepted_event_count'] != 0:
        raise q.SourceChanged('Target must be an original valid-body empty abstaining window')
    job = copy.deepcopy(original)
    job['refs'].update(original_canonical_job=ref, coarse_protocol=protocol_ref)
    job.update(schema=SCHEMA, execution_revision=REVISION, output_root=str(OUT), primary_budget_root=str(PRIMARY),
        target_window_id=wid, target_manifest_ref=diagnostic['manifest_ref'], allowed_event_parts=list(COARSE_PARTS),
        execution_plan={'visual_model_requests': 1, 'text_model_requests': 0,
                        'original_maximum_reobserve_requests': original['maximum_reobserve_requests'],
                        'original_maximum_rewrite_requests': original['maximum_rewrite_requests'],
                        'text_budget_change': 'original_text_model_1_to_0_checked_existing_text_only'},
        resource_scope_selector={'shard': 0, 'nshard': 1},
        ledger_bindings={'original_canonical_job_ref': ref, 'original_global_revision': ORIGINAL_REVISION,
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
    raw_events = {event['id']: event for event in value['events']}
    for event in filtered['events']:
        clause = event['source_clause']
        raw = raw_events[clause['raw_event_id']]
        if (clause['raw_clause_index'] != 0 or event['description'] != raw['description']
                or clause['raw_source_clause'] != raw['description']
                or clause['original_description'] != raw['description']):
            raise ValueError('Only whole verbatim raw body-phase sentences are admissible')
        if event['frame_indices'] != raw['frame_indices'] or not set(event['parts']) <= set(raw['parts']):
            raise ValueError('No added parts or changed raw frame references permitted')
    return filtered


def make_request(job, job_ref, prepared, args):
    verify_args(args)
    manifest = verify_window(job, prepared)
    request = bar.visual_request(args, Path(job['visual_root']) / 'rgb', manifest,
                                 {p: set(v) for p, v in prepared['forbidden'].items()})
    native_content = copy.deepcopy(request['messages'][1]['content'][1:])
    header = json.loads(request['messages'][1]['content'][0]['text'])
    header['available_crop_inventory_not_visibility_or_handtruth'] = header.pop('hand_ROI_support')
    header['part_support_parts'] = header.pop('parts')
    header['allowed_event_parts'] = list(COARSE_PARTS)
    header.pop('hand_detail_unknown_unless_actual_fingers_distinguishable_in_two_frames')
    header['fine_hand_detail_uncertainties_only'] = True
    request['messages'][1]['content'][0]['text'] = json.dumps(header, ensure_ascii=False, allow_nan=False)
    parts = request['json_schema']['properties']['events']['items']['properties']['parts']
    parts['items']['enum'] = list(COARSE_PARTS)
    parts['maxItems'] = 6
    request['messages'][0]['content'] += '\n' + COARSE_RULES
    request['prompt_sha256'] = hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest()
    request['repair_provenance'] = {'schema': SCHEMA, 'execution_revision': REVISION,
        'current_job_ref': job_ref, 'original_canonical_job_ref': job['refs']['original_canonical_job'],
        'original_visual_result_ref': job['refs']['visual_result'], 'original_result_ref': job['original_result_ref'],
        'prepared_ref': job['refs']['prepared'], 'manifest_ref': job['target_manifest_ref'],
        'original_request_sha256': TARGETS[job['id']][5], 'ledger_bindings': job['ledger_bindings'],
        'maximum_visual_calls_per_source': 1, 'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0,
        'original_RGB_ROI_content_preserved': True, 'no_reference_fill': True}
    if request['messages'][1]['content'][1:] != native_content:
        raise q.SourceChanged('Native RGB/ROI metadata/order/bytes were changed')
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


def package(job, observations, original, old_joint, prepared, plan, raw_target=None):
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
        if raw_target is None:
            raise ValueError('Packaging requires the original strict raw target, not a derived event-ID second pass')
        validated = validate_coarse(raw_target, verify_window(job, prepared), prepared['forbidden'])
        if validated != target['json']:
            raise q.SourceChanged('Target normalized JSON differs from exact original raw replay')
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
        item = {'window_id': key, 'caption': ' '.join(f['text'] for f in local),
                'used_fact_ids': [f['fact_id'] for f in local], 'used_boundary_ids': []}
        if key != wid:
            if old_items:
                if key not in old_items:
                    raise q.SourceChanged('Missing original final caption for non-target window')
                item = copy.deepcopy(old_items[key])
            elif any(f['text'] != bar.core.neutral_fact_text(f['raw_text']) for f in local):
                raise q.SourceChanged('Existing no-final fallback must use exact native neutral fact text')
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
        'derivation': 'existing_neutral_fact_text_no_original_final_caption' if not old_items else 'new_body_facts_original_non_target_final_fields_frozen', **q.FLAGS}
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
    if joint['source_refs'] != old_joint['source_refs'] or joint['source_binding'] != old_joint['source_binding']:
        raise q.SourceChanged('Original joint source references/binding changed')
    visual.update(q.FLAGS)
    joint.update(q.FLAGS)
    return visual, joint



def scope_manifest():
    value = q.load(OUT / 'jobs.manifest.json')
    refs = [item['job_ref'] for item in value['jobs']]
    if (value['schema'] != SCHEMA or value['global_revision'] != REVISION
            or value['source_count'] != 3 or len(refs) != 3
            or [q.checked(ref)['id'] for ref in refs] != list(TARGETS)
            or len({q.checked(ref)['source_key'] for ref in refs}) != 3):
        raise q.SourceChanged('Exactly the frozen three ordered source job refs are required')
    for ref in refs:
        job = q.checked(ref)
        if Path(ref['path']).resolve() != OUT / 'jobs' / (job['source_key'] + '.json'):
            raise q.SourceChanged('Independent source job path changed')
    return value, refs


def caller_admission(runtime, refs):
    verifier = getattr(runtime, 'verify_current_admission', None)
    if not callable(runtime) or not callable(verifier):
        raise ValueError('Caller-owned runtime/current admission callback required; default execution denied')
    manifest_ref = q.identity(OUT / 'jobs.manifest.json')
    admission = verifier(refs)
    originals = [q.checked(ref)['refs']['original_canonical_job'] for ref in refs]
    if (not isinstance(admission, dict) or admission.get('jobs_manifest_ref') != manifest_ref
            or admission.get('source_job_refs') != refs or admission.get('original_source_job_refs') != originals
            or admission.get('out') != str(OUT) or admission.get('global_revision') != REVISION
            or admission.get('caller_pid') != os.getpid() or admission.get('tensor_parallel_size') != 2
            or admission.get('gpus') != [0, 1] or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1'
            or not admission.get('resource_admission_ref') or len(admission.get('current_lease_refs', [])) != 2):
        raise ValueError('Current caller-owned N6 g0/1 TP2 admission must bind fixed3/original refs and live leases')
    q.verify_bytes(admission['resource_admission_ref'])
    for ref in admission['current_lease_refs']:
        q.verify_bytes(ref)
    if len({ref['path'] for ref in admission['current_lease_refs']}) != 2:
        raise ValueError('Two distinct current caller manager leases required')
    return admission


def source_freeze():
    return {'schema': SCHEMA, 'global_revision': REVISION, 'worker_ref': q.identity(__file__),
        'test_ref': q.identity(RT / 'test_curated_coarse_reobserve_fixed3_v1.py'),
        'native_source_sha256': protocol()['native_source_sha256'],
        'protocol_ref': q.identity(OUT / 'protocol.json'),
        'jobs_manifest_ref': q.identity(OUT / 'jobs.manifest.json'),
        'GPU_resource_prepared': False, 'GPU_dispatch_performed': False,
        'launch_authorized': False, 'caller_owned_current_admission_required': True, **q.FLAGS}


def transaction_proof(item, result, model_sha):
    root = item['root']
    receipt = q.load(root / 'receipt.json')
    six = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
    inventory = receipt['files_sha256']
    if set(inventory) != {name for name in six if (root / name).exists()}:
        raise q.SourceChanged('Original full transaction receipt inventory differs')
    for name, digest in inventory.items():
        q.verify_bytes({'path': str(root / name), 'sha256': digest})
    if q.load(root / 'request.json') != item['request'] or q.load(root / 'binding.json') != item['binding']:
        raise q.SourceChanged('Actual transaction request/binding differs')
    if (receipt['binding'] != item['binding'] or receipt['status'] != result['status']
            or q.load(root / 'final.json') != result
            or any(receipt.get(key) is not False or result.get(key) is not False for key in bar.core.FLAGS)):
        raise q.SourceChanged('Original transaction receipt/final status/approval differs')
    raw = None
    if (root / 'generation.json').exists() and q.load(root / 'generation.json').get('model_identity_sha256') != model_sha:
        raise q.SourceChanged('Actual failed/successful generation model identity differs')
    if result['status'] == 'candidate':
        if set(inventory) != six or q.load(root / 'response.json') != result['json']:
            raise q.SourceChanged('Successful original transaction requires all six complete artifacts')
        generation = bar.core.validate_generation(root, item['request'])
        if generation['model_identity_sha256'] != model_sha:
            raise q.SourceChanged('Actual generation model identity differs from native frozen model reference')
        raw = bar.core.strict_json((root / 'response.raw.txt').read_bytes())
    return raw, q.identity(root / 'receipt.json')


def persist_result(job, job_ref, root, item, result, original, old_joint, prepared, plan, claim, model_sha):
    raw, receipt_ref = transaction_proof(item, result, model_sha)
    if result['status'] == 'candidate' and result['json']['abstain']:
        result = {**result, 'status': 'abstain'}
    observations = copy.deepcopy(original['window_observations'])
    index = next(i for i, o in enumerate(observations) if o['window']['window_id'] == job['target_window_id'])
    observations[index] = {**result, 'window': observations[index]['window'], 'attempt_path': str(item['root'])}
    try:
        visual, joint = package(job, observations, original, old_joint, prepared, plan, raw)
    except ValueError as error:
        if isinstance(error, q.SourceChanged):
            raise
        # Preserve the failed checked package, without claiming other captions or facts were fixed.
        visual, joint = copy.deepcopy(original), copy.deepcopy(old_joint)
        visual.update(window_observations=observations, original_final=copy.deepcopy(original['final']),
                      facts=bar.facts_from_observations([m['window'] for m in prepared['windows']], observations),
                      final={'status': 'failed', 'error': str(error), 'model_call_performed': False, **q.FLAGS},
                      status='failed', final_timed_caption='')
        joint = pipeline.joint_candidate(plan['manifest_record'], visual, copy.deepcopy(old_joint['style_result']),
                                         old_joint.get('annotator_model_sha256'))
        joint.update(status='incomplete_repair_candidate', labels_complete=False)
        for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
            if name in old_joint:
                joint[name] = copy.deepcopy(old_joint[name])
    provenance = {'schema': SCHEMA, 'global_revision': REVISION, 'execution_revision': REVISION,
        'original_job_ref': job['refs']['original_canonical_job'], 'current_job_ref': job_ref,
        'original_result_ref': job['original_result_ref'], 'original_visual_result_ref': job['refs']['visual_result'],
        'original_source_binding': job['source_binding'], 'ledger_bindings': job['ledger_bindings'],
        'original_stages': copy.deepcopy(job['stages']), 'original_failure_metadata': copy.deepcopy(job['failures']),
        'execution_plan': copy.deepcopy(job['execution_plan']), 'primary_budget_ref': q.identity(budget_path(job)),
        'canonical_budget_claim': claim, 'transaction_receipt_ref': receipt_ref,
        'native_source_sha256': protocol()['native_source_sha256'],
        'extra_source_attempts': 1, 'maximum_extra_source_attempts': 1,
        'new_visual_requests': 1, 'new_text_model_requests': 0, 'maximum_total_visual_calls': 3,
        'target_window_success': observations[index]['status'] == 'candidate' and bool(observations[index]['json']['events']),
        'original_raw_or_terminal_modified': False, 'style_label_changed': False, 'repair_completed': False,
        'independent_review_status': 'pending_independent_vlm_review', 'new_label_review_required': True,
        'existing_facts_independently_reviewed': False, 'training_approved': False, **q.FLAGS}
    visual.update(provenance)
    joint.update(provenance)
    verify_job(job, probe=True)
    if q.checked(provenance['primary_budget_ref']) != claim:
        raise q.SourceChanged('Canonical budget claim changed after dispatch')
    q.save_once(root / 'visual_result.json', visual)
    joint['visual_result_ref'] = q.identity(root / 'visual_result.json')
    q.save_once(root / 'result.json', joint)
    terminal = {'schema': SCHEMA, 'id': job['id'], 'source_key': job['source_key'],
        'status': joint['status'], 'global_revision': REVISION, 'current_job_ref': job_ref,
        'original_job_ref': job['refs']['original_canonical_job'], 'target_window_success': provenance['target_window_success'],
        'joint_complete': joint['joint_complete'], 'joint_result_ref': q.identity(root / 'result.json'),
        'shared_dispatch_failure': (item['root'] / 'dispatch_error.json').exists(), **q.FLAGS}
    q.save_once(OUT / 'worker-terminals' / (job['source_key'] + '.json'), terminal)
    return terminal


def execute_scope(args, runtime=None):
    """Main-owned managed caller only. Load once, then claim original source budgets."""
    verify_args(args)
    if not callable(runtime) or not callable(getattr(runtime, 'verify_current_admission', None)):
        raise ValueError('Caller-owned admission callback required; default execution denied')
    if q.load(OUT / 'source.freeze.json') != source_freeze():
        raise q.SourceChanged('Frozen private worker/tests/native sources changed')
    manifest, refs = scope_manifest()
    rows, pending = [], []
    for ref in refs:
        job = q.checked(ref)
        evidence = verify_job(job, probe=True)
        if check_budget(job):
            rows.append({'id': job['id'], 'status': 'pending_budget_already_reserved', 'new_visual_requests': 0})
        else:
            pending.append((ref, job, evidence))
    if not pending:
        return {'schema': SCHEMA, 'sources': rows, 'new_visual_requests': 0, 'new_text_model_requests': 0, **q.FLAGS}
    caller_admission(runtime, refs)
    loaded = runtime()  # Shared load failures happen before any source attempt is reserved.
    model_sha = protocol()['model_identity_sha256']
    if not isinstance(loaded, tuple) or len(loaded) != 2 or getattr(loaded[0], '_ms_bar_model_sha256', None) != model_sha:
        raise ValueError('Loaded actual native model identity differs; no budget reservation')
    caller_admission(runtime, refs)
    count = 0
    for position, (ref, job, evidence) in enumerate(pending):
        original, old_joint, prepared, plan = verify_job(job, probe=True)
        caller_admission(runtime, refs)
        if check_budget(job):
            rows.append({'id': job['id'], 'status': 'pending_budget_already_reserved', 'new_visual_requests': 0})
            continue
        root = OUT / 'revisions' / REVISION / job['source_key']
        if root.exists():
            raise q.SourceChanged('Independent result exists without canonical source budget; no redispatch')
        request = make_request(job, ref, prepared, args)
        preview_ref = q.identity(OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json')
        if request != q.checked(preview_ref):
            raise q.SourceChanged('Frozen request preview changed')
        manifest = verify_window(job, prepared)
        item = {'root': root / 'windows' / job['target_window_id'] / 'attempt-000001', 'request': request,
            'validator': lambda value, m=manifest, f=prepared['forbidden']: validate_coarse(value, m, f),
            'binding': {**original['binding'], 'repair_revision': REVISION, 'current_job_ref': ref,
                'original_canonical_job_ref': job['refs']['original_canonical_job'], 'ledger_bindings': job['ledger_bindings'],
                'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames']),
                'request_preview_ref': preview_ref}}
        canonical = q.checked(job['refs']['original_canonical_job'])
        if count >= 3:
            raise ValueError('Fixed-three visual budget exhausted')
        try:
            claim = q.reserve_attempt(canonical, PRIMARY, REVISION)
        except ValueError as error:
            if isinstance(error, q.SourceChanged) or not check_budget(job):
                raise
            rows.append({'id': job['id'], 'status': 'pending_budget_already_reserved', 'new_visual_requests': 0})
            continue
        q.save_once(root / 'original_job.json', canonical)
        q.save_once(root / 'scope_job.json', job)
        q.save_once(root / 'canonical_budget_claim.json', claim)
        def admitted_runtime():
            caller_admission(runtime, refs)
            verify_job(job, probe=True)
            if q.checked(ref) != job:
                raise q.SourceChanged('Current source job ref changed before dispatch')
            return loaded
        try:
            result = bar.transactions([item], admitted_runtime, args)[str(item['root'])]
        except Exception as error:
            # An admitted/reserved ambiguous call is spent; never retry it or reserve later sources.
            q.save_once(root / 'partial.dispatch.json', {'status': 'partial_ambiguous_dispatch_spent',
                'error': str(error), 'canonical_budget_claim': claim, 'current_job_ref': ref, **q.FLAGS})
            rows.append({'id': job['id'], 'status': 'partial_ambiguous_dispatch_spent', 'new_visual_requests': 1})
            rows.extend({'id': j['id'], 'status': 'not_reserved_after_shared_failure', 'new_visual_requests': 0}
                        for _, j, _ in pending[position + 1:])
            count += 1
            break
        count += 1
        terminal = persist_result(job, ref, root, item, result, original, old_joint, prepared, plan, claim, model_sha)
        rows.append(terminal)
        if terminal['shared_dispatch_failure']:
            rows.extend({'id': j['id'], 'status': 'not_reserved_after_shared_failure', 'new_visual_requests': 0}
                        for _, j, _ in pending[position + 1:])
            break
    summary = {'schema': SCHEMA, 'global_revision': REVISION, 'sources': rows,
        'new_visual_requests': count, 'new_text_model_requests': 0,
        'target_window_success_count': sum(bool(row.get('target_window_success')) for row in rows),
        'joint_complete_count': sum(bool(row.get('joint_complete')) for row in rows),
        'original_population_completed': False, 'all_sources_complete_guaranteed': False, **q.FLAGS}
    q.save_once(OUT / 'runtime' / (str(os.getpid()) + '.json'), summary)
    return summary


def prepare():
    """Freeze three local source/input/request refs only; no runtime or weight scans."""
    before = {sid: check_budget({**original_source(sid)[1], 'primary_budget_root': str(PRIMARY)}) for sid in TARGETS}
    q.save_once(OUT / 'protocol.json', protocol())
    pref = q.identity(OUT / 'protocol.json')
    jobs, checks = [], []
    for sid, expected in TARGETS.items():
        job = build_job(sid, pref)
        original, joint, prepared, plan = verify_job(job, probe=True)
        path = OUT / 'jobs' / (job['source_key'] + '.json')
        q.save_once(path, job)
        ref = q.identity(path)
        jobs.append({'source_key': job['source_key'], 'job_ref': ref})
        request = make_request(job, ref, prepared, worker_args())
        native = bar.visual_request(worker_args(), Path(job['visual_root']) / 'rgb', verify_window(job, prepared),
                                    {p: set(v) for p, v in prepared['forbidden'].items()})
        if native['messages'][1]['content'][1:] != request['messages'][1]['content'][1:] or native['sampling'] != request['sampling']:
            raise q.SourceChanged('Native image content/metadata/order/sampling must be unchanged')
        pp = OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json'
        q.save_once(pp, request)
        images = [base64.b64decode(item['image_url']['url'].split(',', 1)[1], validate=True)
                  for item in request['messages'][1]['content'] if item['type'] == 'image_url']
        if len(images) != len(expected[3]) + expected[4]:
            raise q.SourceChanged('Frozen native image count differs')
        checks.append({'id': sid, 'target_window_id': expected[0], 'original_job_ref': job['refs']['original_canonical_job'],
            'target_manifest_ref': job['target_manifest_ref'], 'original_request_sha256': expected[5],
            'body_frame_indices': expected[3], 'actual_image_count': len(images),
            'image_bytes_sha256': [{'bytes': len(image), 'sha256': hashlib.sha256(image).hexdigest()} for image in images],
            'body_frames': copy.deepcopy(verify_window(job, prepared)['frames']),
            'request_preview_ref': q.identity(pp), 'original_stages': job['stages'], 'execution_plan': job['execution_plan'],
            'original_non_target_observations_digest': q.digest([o for o in original['window_observations'] if o['window']['window_id'] != expected[0]]),
            'original_facts_digest': q.digest(original['facts']), 'original_style_digest': q.digest(joint['style_result']),
            'native_content1_unchanged': True, 'native_sampling_unchanged': True})
    q.save_once(OUT / 'jobs.manifest.json', {'schema': SCHEMA, 'global_revision': REVISION,
        'source_count': 3, 'maximum_total_visual_calls': 3, 'maximum_text_model_calls': 0,
        'jobs': jobs, 'not_original_ready_queue': True, 'GPU_dispatch_performed': False, **q.FLAGS})
    after = {sid: check_budget({**original_source(sid)[1], 'primary_budget_root': str(PRIMARY)}) for sid in TARGETS}
    if before != after:
        raise q.SourceChanged('Canonical budget before/after SHA changed during CPU-only preparation')
    evidence = {'schema': SCHEMA, 'global_revision': REVISION, 'checks': checks,
        'model_loads': 0, 'visual_model_calls': 0, 'text_model_calls': 0, 'budget_reservations_performed': 0,
        'budget_beforeafter_unchanged': True, 'source_count': 3, 'launch_authorized': False,
        'GPU_resource_prepared': False, 'GPU_dispatch_performed': False, **q.FLAGS}
    q.save_once(OUT / 'cpu.prepare.evidence.json', evidence)
    q.save_once(OUT / 'source.freeze.json', source_freeze())
    return {**evidence, 'budget_before': before, 'budget_after': after,
            'source_freeze_ref': q.identity(OUT / 'source.freeze.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        parser.error('--run denied: main must call execute_scope(args, caller_owned_admitted_runtime)')
    print(json.dumps(prepare(), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
