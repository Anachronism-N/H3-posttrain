"""Independent two-source coarse observation; primary source budget is never bypassed."""
from __future__ import annotations

import argparse
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
ROOT = RT.parent
OUT = ROOT / 'curated222939_io_visual_repair_coarse6_v2'
PRIMARY = ROOT / 'curated222939_repair_v1'
RECOVERY = ROOT / 'curated222939_io_failure_recovery_v1'
REVISION = 'io-coarse6-reobserve-20261025-v2'
SCHEMA = 'curated_io_visual_repair_coarse6_v2'
TARGETS = {
    'OertkYlRRBiBIQbU': {
        'w00000': ('f25a4468bc7c93d395d959a85259adf0eb1efb7e583a071de87b8171b784c366', [0,2,4,6,8,10,24,32], 6),
        'w00001': ('27da0ed77f712d38bc66db31a62c0b2e65b59c0f003bbd64188967e11382cc95', [33,37,38,42,56,64,76,80], 6)},
    'UbjwIZuNtSoLcUaG': {
        'w00001': ('b4a91ad2de8b6dc5b9caa6aa61b4c01f91875f3adbcb4e71aefc1c03f255ea60', [46,48,50,58,66,74,76,80], 5)}}
COARSE_PARTS = ('head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
COARSE_RULES = """
This is ONE NEW STRICT SIX-PART coarse-body observation, not an edit of any
old response. Only head, torso, left_arm, right_arm, left_leg, right_leg are
allowed in events.parts and event descriptions. Keep all TEN part_support
entries in schema order, independently judged from original body frames.
The native_hand_ROI_crop_frames header is only crop INVENTORY, NOT a complete
list of hand-visible frames and NOT a source of automatic part_support.
Each event must be ONE complete sentence describing ONE visually supported
body phase using only those six coarse parts. Each claimed part needs an
intersection of at least TWO distinct actual BODY frames with event references.
A native crop is not a second time observation. No mechanical ID copying.
ALL hand/wrist/finger/palm/fist/grip/hand-contact/hand-shape/openness/orientation
or V/OK/heart-sign details are UNKNOWN metadata only in uncertainties, NEVER
in event descriptions, even if apparently visible. Describe independently
certain arms, torso, legs or head without attaching any such detail.
Opening or spreading the ARMS is allowed when supported; unknown hand detail
must not suppress that independently visible arm movement. Do not create
static/stillness filler for unknown motion. Only RGB supports actions, not
source metadata. Return the supplied strict JSON with no new fields or IDs.
"""
FINE_HAND = re.compile(r'\b(?:v(?:[- ]?(?:shape|sign))?|ok|heart|hands?|wrists?|fists?|grip\w*|thumb\w*|finger\w*|palms?|knuckles?|digits?)\b|拳|手指|掌心|掌朝|拇指', re.I)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


q = module('io_visual_q', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
sys.path.insert(0, str(RT / 'source_curated_consumer_v1'))
import ms_bar_annotation as bar
import curated_pipeline as pipeline


def worker_args():
    config = q.load(RT / 'curated_consumer_s0_v1_template.json')
    args = pipeline.parser().parse_args(config['argv'][4:])
    args.ffprobe = '/opt/ms_tools/ffprobe'
    args.ffmpeg = '/opt/ms_tools/ffmpeg'
    args.retry_failed = False
    args.rewrite_reference_repair = 'off'
    args.batch_size = 2
    if args.tp != 2 or args.stage != 'run':
        raise ValueError('Require native 32B TP2 run configuration')
    return args


def validate_coarse(value, manifest, forbidden):
    # Strict validation precedes clause filtering: unknown IDs/support never get repaired.
    bar.validate_observation(value, manifest, forbidden)
    for event in value['events']:
        if any(part not in COARSE_PARTS for part in event['parts']):
            raise ValueError('Events may claim only the six coarse body parts')
        text = event['description']
        if FINE_HAND.search(text):
            raise ValueError('Fine hand detail prohibited outside uncertainties')
        if len([s for s in re.split(r'(?<=[.!?;])\s+', text.strip()) if s]) != 1 or ';' in text:
            raise ValueError('Exactly one complete body-phase sentence per event')
    return bar.filter_supported_observation(value, manifest, forbidden)


def verify_window(job, wid, prepared):
    sid = job['id']
    if sid not in TARGETS or wid not in TARGETS[sid]:
        raise ValueError('Source/window not in immutable two-source target set')
    expected_sha, expected_frames, expected_roi = TARGETS[sid][wid]
    ref = job['target_manifest_refs'][wid]
    old_root = Path(job['visual_root'])
    if (Path(ref['path']).resolve() != old_root / 'windows' / wid / 'manifest.json'
            or ref['sha256'] != expected_sha):
        raise q.SourceChanged('Target manifest path/SHA differs from original evidence')
    manifest = q.checked(ref)
    original = next(m for m in prepared['windows'] if m['window']['window_id'] == wid)
    if (any(manifest.get(k) != v for k, v in original.items())
            or manifest['binding'] != prepared['binding']
            or manifest['union_manifest_sha256'] != prepared['union_manifest_sha256']
            or manifest['raw2d_mask'] != prepared['raw2d_mask']):
        raise q.SourceChanged('Window/prepared binding differs')
    frames = manifest['frames']
    if [f['index'] for f in frames] != expected_frames:
        raise q.SourceChanged('Original target body frame selection changed')
    start, end = manifest['window']['start'], manifest['window']['end']
    for f in frames:
        i = f['index']
        if (type(i) is not int or not start <= prepared['probe']['times'][i] < end
                or not math.isclose(f['time'], prepared['probe']['times'][i], abs_tol=1e-8)
                or not math.isclose(f['source_timestamp'], prepared['probe']['source_timestamps'][i], abs_tol=1e-8)):
            raise q.SourceChanged('Original native PTS/half-open window differs')
        q.verify_bytes({'path': str(old_root / 'rgb' / f['file']), 'sha256': f['sha256']})
    inventory = prepared['hand_roi']
    all_crops = q.load(inventory['path'])['crops'] if inventory.get('available') else []
    hand = manifest.get('hand_roi_evidence') or {}
    count = 0
    for part in ('left_hand', 'right_hand'):
        entry = hand.get(part, {})
        crops = entry.get('crops', [])
        if entry.get('frame_indices', []) != sorted(set(c['frame_index'] for c in crops)):
            raise q.SourceChanged('ROI inventory frame list differs')
        for c in crops:
            if c not in all_crops or c['part'] != part or c['frame_index'] not in expected_frames:
                raise ValueError('Unknown or changed ROI crop')
            q.verify_bytes({'path': c['path'], 'sha256': c['sha256']})
        count += len(crops)
    if count != expected_roi or len(frames) + count > 16:
        raise ValueError('Original ROI/image budget changed')
    return manifest


def verify_job(job, probe=False):
    sid = job['id']
    if sid not in TARGETS or job['schema'] != SCHEMA or job['global_revision'] != REVISION:
        raise ValueError('Unknown source/schema/edition')
    if (q.safe_key(job) != job['source_key'] or job['split'] != 'train'
            or Path(job['queue_root']).resolve() != PRIMARY or Path(job['primary_budget_root']).resolve() != PRIMARY
            or Path(job['output_root']).resolve() != OUT
            or job['allowed_event_parts'] != list(COARSE_PARTS)
            or job['stages'] != ['reobserve'] or job['maximum_extra_source_attempts'] != 1
            or job['maximum_reobserve_requests'] != len(TARGETS[sid])
            or job['maximum_rewrite_requests'] != 0
            or set(job['target_manifest_refs']) != set(TARGETS[sid])):
        raise ValueError('Independent job namespace/target/budget changed')
    q.verify_job(job)
    protocol = q.checked(job['refs']['coarse6_protocol'])
    if protocol['schema'] != SCHEMA or protocol['allowed_event_parts'] != list(COARSE_PARTS):
        raise q.SourceChanged('Frozen coarse6 protocol changed')
    primary = q.checked(job['refs']['original_primary_media_job'])
    failed = q.checked(job['refs']['original_failed_terminal'])
    terminal = q.checked(job['refs']['consumer_terminal'])
    producer = q.checked(job['refs']['producer_terminal'])
    joint = q.checked(job['refs']['joint_result'])
    original = q.checked(job['refs']['visual_result'])
    plan = q.checked(job['refs']['plan_ref'])
    binding = original['binding']
    if (primary['stage'] != 'media' or primary['visual_root'] is not None
            or primary['original_result_ref'] != job['refs']['original_failed_terminal']
            or primary['id'] != sid or failed['id'] != sid or primary['source_key'] != job['source_key']
            or job['original_result_ref'] != job['refs']['joint_result']
            or job['original_result_sha256'] != job['original_result_ref']['sha256']
            or terminal['joint_candidate_ref'] != job['original_result_ref']
            or joint['visual_result_ref'] != job['refs']['visual_result']
            or plan['binding'] != producer['source_binding'] or plan['binding'] != job['source_binding']
            or binding['source_binding'] != job['source_binding'] or joint['source_binding'] != binding
            or bar.batch.digest(plan) != binding['full_source_row_sha256']
            or bar.batch.digest(plan['manifest_record']) != binding['manifest_record_sha256']
            or joint['source_refs'] != plan['manifest_record']['source_refs']):
        raise q.SourceChanged('Original primary failed terminal/current recovery joint/plan provenance differs')
    for value in (producer, joint, original, plan, plan['manifest_record']):
        if value['id'] != sid:
            raise q.SourceChanged('Wrong source ID in immutable provenance')
    prepared = q.checked(job['refs']['prepared'])
    if probe:
        q.verify_original_csv(plan['source_refs']['original_csv'], 'train')
        prepared = q.verify_prepared_content(job['visual_root'], plan, bar, '/opt/ms_tools/ffprobe')
    if prepared['binding'] != binding:
        raise q.SourceChanged('Prepared/current visual binding differs')
    windows = [m['window'] for m in prepared['windows']]
    if original['facts'] != bar.facts_from_observations(windows, original['window_observations']):
        raise q.SourceChanged('Original admitted facts changed')
    for o in original['window_observations']:
        if o['window']['window_id'] in TARGETS[sid] and (o['status'] != 'abstain' or o['json']['events']):
            raise q.SourceChanged('Only original abstaining empty windows may be targeted')
    for wid in TARGETS[sid]:
        verify_window(job, wid, prepared)
    expected = {
        'originalprimarymediajobref': job['refs']['original_primary_media_job'],
        'originalfailedterminalSHA': job['refs']['original_failed_terminal']['sha256'],
        'currentrecoveryjointSHA': job['original_result_sha256']}
    if job['ledger_bindings'] != expected:
        raise q.SourceChanged('Dedicated ledger mapping differs')
    return original, joint, prepared, plan


def make_request(job, wid, manifest, prepared, args, job_ref):
    forbidden = {part: set(indices) for part, indices in prepared['forbidden'].items()}
    request = bar.visual_request(args, Path(job['visual_root']) / 'rgb', manifest, forbidden)
    header = json.loads(request['messages'][1]['content'][0]['text'])
    header['native_hand_ROI_crop_frames'] = header.pop('hand_ROI_support')
    header['part_support_parts'] = header.pop('parts')
    header['allowed_event_parts'] = list(COARSE_PARTS)
    header.pop('hand_detail_unknown_unless_actual_fingers_distinguishable_in_two_frames')
    header['hand_detail_is_unknown_metadata_only'] = True
    header['ROI_inventory_is_not_hand_visibility_frame_set'] = True
    request['messages'][1]['content'][0]['text'] = json.dumps(header, ensure_ascii=False, allow_nan=False)
    event_parts = request['json_schema']['properties']['events']['items']['properties']['parts']
    event_parts['items']['enum'] = list(COARSE_PARTS)
    event_parts['maxItems'] = len(COARSE_PARTS)
    request['messages'][0]['content'] += '\n' + COARSE_RULES
    request['prompt_sha256'] = q.hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest()
    request['repair_provenance'] = {
        'schema': SCHEMA, 'current_job_ref': job_ref, 'global_revision': REVISION,
        'current_job_source_binding': job['source_binding'], 'ledger_bindings': job['ledger_bindings'],
        'original_recovery_terminal_ref': job['refs']['consumer_terminal'],
        'original_recovery_joint_ref': job['original_result_ref'],
        'original_visual_result_ref': job['refs']['visual_result'],
        'prepared_ref': job['refs']['prepared'], 'raw_rgb_manifest_ref': job['refs']['raw_rgb_manifest'],
        'current_video_ref': prepared['source_identity']['files']['video_path'],
        'ROI_manifest_file_ref': job['refs']['hand_roi'], 'manifest_ref': job['target_manifest_refs'][wid],
        'coarse_body_only': True, 'allowed_event_parts': list(COARSE_PARTS),
        'protocol_ref': job['refs']['coarse6_protocol'], 'ROI_support_is_inventory_not_visibility': True,
        'maximum_total_images': 16, 'maximum_visual_calls_for_source': len(TARGETS[job['id']]),
        'maximum_text_calls_for_source': 0, 'automatic_reference_fill': False}
    return request


def reserve_primary_attempt(job, job_ref, runtime, primary_root=PRIMARY, verifier=verify_job):
    if Path(primary_root).resolve() != Path(job['primary_budget_root']).resolve():
        raise ValueError('Dedicated wrapper must use explicitly bound original primary budget root')
    verifier(job)
    if q.checked(job_ref) != job:
        raise q.SourceChanged('Reserved independent job ref changed')
    path = Path(primary_root) / 'attempt-budget' / (job['source_key'] + '.json')
    if path.exists():
        raise ValueError('Extra source attempt already reserved; no call/redispatch across revisions')
    runtime()  # Shared load failure must not consume either source budget.
    verifier(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = {'schema': SCHEMA, 'source_key': job['source_key'],
        'original_result_sha256': job['original_result_sha256'], 'global_revision': REVISION,
        'extra_source_attempt': 1, 'maximum_extra_source_attempts': 1,
        'independent_job_ref': job_ref, 'ledger_bindings': job['ledger_bindings'],
        'maximum_targeted_visual_calls': len(TARGETS[job['id']]), 'maximum_text_calls': 0,
        'runtime_loaded_before_reservation': True}
    try:
        with path.open('xb') as stream:
            stream.write(q.encoded(claim))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ValueError('Extra source attempt concurrently reserved; no redispatch')
    return claim


def concatenate(windows, facts, original, targeted):
    body = {'windows': [], 'issues': []}
    old = {w['window_id']: w for w in (original['final'].get('json') or {}).get('windows', [])}
    for window in windows:
        wid = window['window_id']
        local = [f for f in facts if f['window_id'] == wid]
        item = {'window_id': wid, 'caption': ' '.join(f['raw_text'] for f in local),
                'used_fact_ids': [f['fact_id'] for f in local], 'used_boundary_ids': []}
        if wid not in targeted and wid in old:
            item = copy.deepcopy(old[wid])
        body['windows'].append(item)
    bar.check_supported_rewrite(body, windows, facts, [])
    return body


def package(job, job_ref, observations, original, old_joint, prepared, plan, claim):
    windows = [m['window'] for m in prepared['windows']]
    facts = bar.facts_from_observations(windows, observations)
    targeted = set(TARGETS[job['id']])
    for o, old in zip(observations, original['window_observations']):
        if old['window']['window_id'] not in targeted and o != old:
            raise q.SourceChanged('Untargeted original observation changed')
    for wid in set(w['window_id'] for w in windows) - targeted:
        if [f for f in facts if f['window_id'] == wid] != [f for f in original['facts'] if f['window_id'] == wid]:
            raise q.SourceChanged('Untargeted original facts/references changed')
    body = concatenate(windows, facts, original, targeted)
    failed = any(o['status'] not in ('candidate', 'abstain') for o in observations)
    full = not failed and all(w['caption'].strip() for w in body['windows'])
    visual = copy.deepcopy(original)
    visual.update(window_observations=observations, facts=facts, original_final=copy.deepcopy(original['final']))
    visual['final'] = {'status': 'failed' if failed else 'candidate' if full else 'abstain',
        'json': body, 'model_call_performed': False, 'model_rewrite_performed': False,
        'derivation': 'checked_verbatim_facts_with_untargeted_original_caption_and_refs_frozen', **q.FLAGS}
    visual['status'] = visual['final']['status']
    visual['final_timed_caption'] = bar.core.timed_text({'windows': windows}, body['windows'])
    visual['coverage'] = {
        **original.get('coverage', {}), 'full_window_caption': full, 'admitted_fact_count': len(facts),
        'window_status_counts': {s: sum(o['status'] == s for o in observations) for s in {o['status'] for o in observations}}}
    root = OUT / 'revisions' / REVISION / job['source_key']
    provenance = {'schema': SCHEMA, 'global_revision': REVISION, 'original_job_ref': job_ref,
        'original_primary_media_job_ref': job['refs']['original_primary_media_job'],
        'original_failed_terminal_ref': job['refs']['original_failed_terminal'],
        'original_result_ref': job['original_result_ref'], 'original_visual_result_ref': job['refs']['visual_result'],
        'original_source_binding': job['source_binding'], 'ledger_bindings': job['ledger_bindings'],
        'primary_budget_ref': q.identity(PRIMARY / 'attempt-budget' / (job['source_key'] + '.json')),
        'extra_source_attempts': 1, 'maximum_extra_source_attempts': 1,
        'new_visual_requests': len(targeted), 'new_text_model_requests': 0,
        'maximum_total_calls_across_two_sources': 5, 'actual_planned_total_calls': 3,
        'original_raw_or_terminal_modified': False, 'primary_media_stage_changed': False,
        'style_label_changed': False, 'repair_completed': False,
        'independent_review_status': 'pending_independent_vlm_review', 'new_label_review_required': True,
        'risk_review_case': True, **q.FLAGS}
    visual.update(provenance)
    q.save_once(root / 'visual_result.json', visual)
    visual['result_ref'] = q.identity(root / 'visual_result.json')
    style = copy.deepcopy(old_joint['style_result'])
    joint = pipeline.joint_candidate(plan['manifest_record'], visual, style, old_joint.get('annotator_model_sha256'))
    joint.update(provenance)
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in old_joint:
            joint[name] = copy.deepcopy(old_joint[name])
    joint['status'] = 'joint_repair_candidate' if joint['joint_complete'] else 'incomplete_repair_candidate'
    joint['labels_complete'] = bool(joint['joint_complete'])
    joint['training_approved'] = False
    if joint['style_result'] != old_joint['style_result'] or joint['dance_style_label'] != old_joint['dance_style_label']:
        raise q.SourceChanged('Original resolved/candidate style changed')
    if not full and (joint['joint_complete'] or joint['target_text']):
        raise q.SourceChanged('Partial/empty observation cannot become complete training text')
    verify_job(job)
    q.save_once(root / 'result.json', joint)
    return q.identity(root / 'result.json'), joint


def execute(job_ref, args, runtime, recover_only=False):
    job = q.checked(job_ref)
    if Path(job_ref['path']).resolve() != OUT / 'jobs' / (q.safe_key(job) + '.json'):
        raise ValueError('Independent job must remain in its own queue')
    original, old_joint, prepared, plan = verify_job(job, probe=True)
    root = OUT / 'revisions' / REVISION / job['source_key']
    budget = PRIMARY / 'attempt-budget' / (job['source_key'] + '.json')
    if budget.exists():
        claim = q.load(budget)
        if (claim.get('schema') != SCHEMA or claim.get('independent_job_ref') != job_ref
                or claim['original_result_sha256'] != job['original_result_sha256']
                or claim['global_revision'] != REVISION or claim['ledger_bindings'] != job['ledger_bindings']):
            raise ValueError('Primary source attempt already reserved by another immutable job')
        if not (root / 'result.json').exists():
            return {'id': job['id'], 'status': 'interrupted_attempt',
                'reason': 'primary_budget_reserved_without_final_no_call_no_redispatch', **q.FLAGS}
        result = q.load(root / 'result.json')
        if result['original_job_ref'] != job_ref or result['primary_budget_ref'] != q.identity(budget):
            raise q.SourceChanged('Existing final has mismatched primary budget/job')
        return {'id': job['id'], 'status': result['status'], 'joint_result_ref': q.identity(root / 'result.json'), **q.FLAGS}
    if recover_only:
        raise ValueError('CPU recovery cannot dispatch an unreserved source')
    observations = copy.deepcopy(original['window_observations'])
    items, positions = [], []
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    for i, observation in enumerate(observations):
        wid = observation['window']['window_id']
        if wid not in TARGETS[job['id']]:
            continue
        manifest = verify_window(job, wid, prepared)
        item_root = root / 'windows' / wid / 'attempt-000001'
        if item_root.exists():
            raise q.SourceChanged('Unreserved independent transaction already exists')
        items.append({'root': item_root, 'request': make_request(job, wid, manifest, prepared, args, job_ref),
            'validator': lambda value, m=manifest: validate_coarse(value, m, forbidden),
            'binding': {**original['binding'], 'repair_revision': REVISION,
                'current_job_source_binding': job['source_binding'], 'ledger_bindings': job['ledger_bindings'],
                'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames'])}})
        positions.append(i)
    claim = reserve_primary_attempt(job, job_ref, runtime)
    q.save_once(root / 'original_job.json', job)
    results = bar.transactions(items, runtime, args)
    for index, item in zip(positions, items):
        result = results[str(item['root'])]
        if result['status'] == 'candidate' and result['json']['abstain']:
            result = {**result, 'status': 'abstain'}
        observations[index] = {**result, 'window': observations[index]['window'], 'attempt_path': str(item['root'])}
    ref, joint = package(job, job_ref, observations, original, old_joint, prepared, plan, claim)
    terminal = {'schema': SCHEMA, 'id': job['id'], 'source_key': job['source_key'],
        'batch_id': job['batch_id'], 'status': joint['status'], 'global_revision': REVISION,
        'joint_complete': joint['joint_complete'], 'joint_result_ref': ref, 'original_job_ref': job_ref, **q.FLAGS}
    q.save_once(OUT / 'worker-terminals' / (job['source_key'] + '.json'), terminal)
    return terminal


def prepare():
    args = worker_args()
    protocol = {'schema': SCHEMA, 'global_revision': REVISION,
        'allowed_event_parts': list(COARSE_PARTS), 'part_support_parts': list(bar.PARTS),
        'system_addition_sha256': q.hashlib.sha256(COARSE_RULES.encode()).hexdigest(),
        'strict_original_raw_validation_before_filter': True,
        'crop_inventory_header': 'native_hand_ROI_crop_frames',
        'hand_detail_unknown_metadata_only': True, 'maximum_total_images': 16,
        'maximum_extra_source_attempts': 1, 'planned_targeted_visual_calls': 3,
        'planned_text_model_calls': 0, 'maximum_total_calls': 5,
        'tensor_parallel_size': args.tp, 'batch_size': args.batch_size,
        'max_num_seqs': args.max_num_seqs, 'max_model_len': args.max_model_len,
        'max_pixels': args.max_pixels, 'gpu_memory_utilization': args.gpu_memory_utilization,
        'visual_max_tokens': args.visual_max_tokens, 'text_max_tokens': args.text_max_tokens,
        'retry_failed': False, 'rewrite_reference_repair': 'off',
        'primary_budget_root': str(PRIMARY), 'independent_output_root': str(OUT),
        'GPU_dispatch_performed': False, **q.FLAGS}
    q.save_once(OUT / 'protocol.coarse6_v2.json', protocol)
    jobs, checks = [], []
    for sid in TARGETS:
        terminal = q.load(RECOVERY / 'sources/train' / sid / 'consumer.terminal.json')
        job = q.classify(terminal, RECOVERY)
        primary_ref = q.identity(PRIMARY / 'jobs/train' / (sid + '.json'))
        primary = q.checked(primary_ref)
        job['refs']['original_primary_media_job'] = primary_ref
        job['refs']['original_failed_terminal'] = primary['original_result_ref']
        job['refs']['coarse6_protocol'] = q.identity(OUT / 'protocol.coarse6_v2.json')
        job.update(schema=SCHEMA, global_revision=REVISION, queue_root=str(PRIMARY), output_root=str(OUT), allowed_event_parts=list(COARSE_PARTS),
            primary_budget_root=str(PRIMARY), stages=['reobserve'], stage='reobserve',
            failures=[{'stage': 'reobserve', 'window_id': wid, 'reason': 'new_coarse_body_only_observation'} for wid in TARGETS[sid]],
            maximum_reobserve_requests=len(TARGETS[sid]), maximum_rewrite_requests=0,
            maximum_total_calls_across_two_sources=5, planned_total_calls_across_two_sources=3,
            target_manifest_refs={wid: q.identity(Path(job['visual_root']) / 'windows' / wid / 'manifest.json') for wid in TARGETS[sid]},
            ledger_bindings={'originalprimarymediajobref': primary_ref,
                'originalfailedterminalSHA': primary['original_result_ref']['sha256'],
                'currentrecoveryjointSHA': job['original_result_sha256']},
            primary_stage_frozen=True, primary_job_preserved=True, text_rewrite_policy='deterministic_verbatim_only')
        original, joint, prepared, plan = verify_job(job, probe=True)
        path = OUT / 'jobs' / (job['source_key'] + '.json')
        q.save_once(path, job)
        ref = q.identity(path)
        jobs.append({'source_key': job['source_key'], 'job_ref': ref, 'global_revision': REVISION})
        for wid in TARGETS[sid]:
            manifest = verify_window(job, wid, prepared)
            request = make_request(job, wid, manifest, prepared, args, ref)
            images = sum(v['type'] == 'image_url' for v in request['messages'][1]['content'])
            q.save_once(OUT / 'request-previews' / job['source_key'] / wid / 'request.json', request)
            checks.append({'id': sid, 'window_id': wid, 'manifest_ref': job['target_manifest_refs'][wid],
                'body_indices': [f['index'] for f in manifest['frames']], 'actual_images': images,
                'schema_sha256': q.digest(request['json_schema']),
                'request_sha256': q.digest(request), 'prompt_sha256': request['prompt_sha256'],
                'request_ref': q.identity(OUT / 'request-previews' / job['source_key'] / wid / 'request.json'),
                'original_native_PTS_SHA_verified': True})
    q.save_once(OUT / 'jobs.manifest.json', {'schema': SCHEMA, 'global_revision': REVISION, 'jobs': jobs})
    q.save_once(OUT / 'cpu.prepare.receipt.json', {'schema': SCHEMA, 'checks': checks,
        'source_count': 2, 'targeted_visual_calls': 3, 'text_model_calls': 0,
        'primary_budget_files_absent': {sid: not (PRIMARY / 'attempt-budget/train' / (sid + '.json')).exists() for sid in TARGETS},
        'GPU_dispatch_performed': False, **q.FLAGS})
    template = {'schema': SCHEMA, 'mode': 'independent_coarse_visual_repair',
        'node': None, 'gpus': [], 'required_GPU_count': 2, 'tensor_parallel_size': 2,
        'holders': [], 'release_authorized': False, 'launch_authorized': False,
        'requires_fresh_pair_admission': True, 'model_staging_authorized': False,
        'source_prepared': False, 'source_CPU_prepared': True, 'GPU_resource_prepared': False,
        'protocol_ref': q.identity(OUT / 'protocol.coarse6_v2.json'),
        'protected_live_pairs': ['N3G0/G1', 'N3G6/G7', 'N6G0/G1'],
        'out': str(OUT), 'global_revision': REVISION,
        'jobs_manifest_ref': q.identity(OUT / 'jobs.manifest.json'),
        'worker_ref': q.identity(__file__), 'source_sha256': {**bar.code_identity(), str(Path(pipeline.__file__).resolve()): q.sha(pipeline.__file__)},
        'repair_queue_ref': q.identity(q.__file__),
        'consumer_config_ref': q.identity(RT / 'curated_consumer_s0_v1_template.json'),
        'env': {'OMP_NUM_THREADS': '2', 'MS_VISUAL_CPU_SITEPATH': '/opt/venv_eval/lib/python3.11/site-packages',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
        'argv': ['/opt/venv_vllm/bin/python', '-B', '-u', str(Path(__file__).resolve()),
            'run', '--admission', 'REQUIRES_MAIN_FRESH_PAIR_ADMISSION'], **q.FLAGS}
    q.save_once(RT / 'curated_io_visual_repair_coarse6_safe_template_v2.json', template)
    return {'jobs': jobs, 'checks': checks, 'GPU_dispatch_performed': False}


def run(admission_path, recover_only=False):
    admission = q.load(admission_path)
    template = q.load(RT / 'curated_io_visual_repair_coarse6_safe_template_v2.json')
    q.verify_bytes(template['worker_ref'])
    q.verify_bytes(template['jobs_manifest_ref'])
    q.verify_bytes(template['repair_queue_ref'])
    q.verify_bytes(template['consumer_config_ref'])
    for path, sha in template['source_sha256'].items():
        q.verify_bytes({'path': path, 'sha256': sha})
    if (admission.get('template_ref') != q.identity(RT / 'curated_io_visual_repair_coarse6_safe_template_v2.json')
            or admission.get('out') != str(OUT) or admission.get('global_revision') != REVISION):
        raise ValueError('Fresh admission must bind immutable safe template/output/edition')
    if not recover_only:
        if (admission.get('launch_authorized') is not True or admission.get('fresh_pair_safety_verified') is not True
                or len(admission.get('gpus', [])) != 2 or len(set(admission['gpus'])) != 2
                or not admission.get('resource_manager_verified_and_leases_held')
                or not admission.get('resource_admission_ref')
                or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(str(i) for i in admission['gpus'])):
            raise ValueError('No current resource-manager pair admission/held leases; GPU launch forbidden')
        q.verify_bytes(admission['resource_admission_ref'])
    OUT.mkdir(parents=True, exist_ok=True)
    args = worker_args()
    runtime_root = OUT / 'gpu-runtime' / REVISION
    with (OUT / 'independent-worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if recover_only:
            def runtime():
                raise RuntimeError('GPU call forbidden in CPU-only recovery')
        else:
            runtime_root.mkdir(parents=True, exist_ok=True)
            runtime = bar.runtime_loader(args, runtime_root, bar.code_identity())
        manifest = q.checked(template['jobs_manifest_ref'])
        for hint in manifest['jobs']:
            terminal = execute(hint['job_ref'], args, runtime, recover_only)
            q.save_once(OUT / 'announced' / (hint['source_key'] + '.json'), terminal)
            if terminal.get('joint_result_ref'):
                q.save_once(OUT / 'joint/review-required' / (hint['source_key'] + '.json'), {
                    **terminal, 'result_ref': terminal['joint_result_ref'],
                    'independent_review_status': 'pending_independent_vlm_review', 'new_label_review_required': True})
            print(json.dumps(terminal, ensure_ascii=False), flush=True)
            if any((OUT / 'revisions' / REVISION / hint['source_key']).rglob('dispatch_error.json')):
                raise RuntimeError('Shared generation failure: stop before next source reservation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'recover'))
    parser.add_argument('--admission')
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(), ensure_ascii=False))
    else:
        if not args.admission:
            parser.error('Require explicit immutable fresh-resource admission')
        run(args.admission, args.mode == 'recover')


if __name__ == '__main__':
    main()
