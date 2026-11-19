#!/usr/bin/env python3
"""Offline exact-candidate delivery; never loads a model or changes live outputs."""
import argparse
import base64
from collections import Counter
import copy
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
IDS = ('AuuitdcBJyGOnWaH', 'AzPomlvsANsevIFb', 'DxNCjtNsxcazNpGb')
REVISION = 'coarse-filter-recovery-20261025-v1'
PINS = {
    'coarse3_completed_review_recovery_v1/summary.json': '147be057a30544fa424b9db0ace1361b49d6837fb894bbf46da3b9f610fea95e',
    'coarse3_completed_review_recovery_v1/review.records.json': '4ee8dd82cb936b246f7ce52ea9a421f1de706e980c3f079ab1ff0936c8c96865',
    'curated_coarse_empty_filter_recovery_v1/summary.json': 'd6d4d5f4931f84ce86b7e145604cd5eb75c86ffb028e7e962913321e3c5f5374',
    'curated222939_inputs_v1/manifest.summary.json': 'a1eb129cf8557b4adeb4e11e00ace334ecb1575e43a587d665e33f09410e2bfd',
}
FLAGS = {k: False for k in ('semantic_approval', 'human_review_completed', 'formal_approval',
    'formal_training', 'training_eligible', 'eligible_for_training', 'training_approved',
    'human_approved', 'full_pool_updated', 'training_entry_updated', 'full_batch_semantic_pass_claimed')}
SIX = {'binding.json', 'request.json', 'response.raw.txt', 'response.json', 'generation.json', 'final.json'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(raw(value))


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path.read_bytes())}


def checked(reference):
    path = Path(reference['path'])
    content = path.read_bytes()
    need(sha(content) == reference['sha256'], 'SHA changed: ' + str(path))
    return json.loads(content)


def key(row):
    return row['split'] + '/' + row.get('source_id', row.get('id'))


def pinned(name):
    return checked({'path': str(ROOT / name), 'sha256': PINS[name]})


def immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        need(path.read_bytes() == content, 'Immutable output collision: ' + str(path))
    return ref(path)


def check_image(path, expected_sha):
    from PIL import Image
    content = Path(path).read_bytes()
    need(sha(content) == expected_sha, 'Native RGB/ROI SHA changed')
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        return image.size


def verify(record, manifest, recovery):
    sid = record['source_id']
    candidate_ref = record['candidate_ref']
    canonical = ROOT / 'curated_coarse_empty_filter_recovery_v1/revisions' / REVISION / 'train' / sid / 'result.json'
    need(Path(candidate_ref['path']) == canonical, 'Exact candidate revision/path required')
    candidate = checked(candidate_ref)
    need(key(candidate) == 'train/' + sid and candidate['global_revision'] == REVISION
         and candidate['joint_complete'] is True and candidate['all_windows_nonempty'] is True,
         'Only exact complete recovered candidate admitted')
    need(any(s['result_ref'] == candidate_ref and s['source_key'] == key(candidate) for s in recovery['sources']),
         'Recovery summary does not bind candidate')
    need(record['status'] == 'pass' and record['review_dispatches'] == record['actual_VLM_calls'] == 1
         and record['recovery_model_calls'] == record['recovery_new_model_calls'] == record['recovery_new_budget_reservations'] == 0,
         'Only original completed one-call pass accepted')
    need(all(record.get(k) is False for k in FLAGS if k in record)
         and all(candidate.get(k) is not True for k in FLAGS), 'No human or training approval allowed')
    membership = record['original_membership']
    descriptor = next(d for d in manifest['batches'] if d['batch_id'] == membership['batch_id'])
    membership_ref = {'path': descriptor['path'], 'sha256': descriptor['sha256']}
    need(membership['membership_ref'] == membership_ref, 'Original authoritative membership differs')
    content = Path(membership_ref['path']).read_bytes()
    need(sha(content) == membership_ref['sha256'], 'Membership SHA changed')
    members = [json.loads(line) for line in content.splitlines() if line.strip()]
    need(len(members) == descriptor['count'] == membership['original_member_count'], 'Original batch denominator differs')
    member = next(r for r in members if key(r) == key(candidate))
    need(candidate['batch_id'] == member['batch_id'] and candidate['source_refs'] == member['source_refs']
         and membership['original_csv_ref'] == member['source_refs']['csv'], 'Candidate original CSV/row/batch differs')
    evidence = checked(record['evidence_ref'])
    receipt = checked(record['review_receipt'])
    transaction = Path(record['review_receipt']['path']).parent
    need(receipt['status'] == 'candidate' and set(receipt['files_sha256']) == SIX, 'Native six-file transaction required')
    artifacts = {n: {'path': str(transaction / n), 'sha256': h} for n, h in receipt['files_sha256'].items()}
    for reference in artifacts.values():
        need(sha(Path(reference['path']).read_bytes()) == reference['sha256'], 'Review artifact SHA changed')
    binding = checked(artifacts['binding.json'])
    need(binding == receipt['binding'] and binding['source_id'] == sid and binding['split'] == candidate['split']
         and binding['label_sha256'] == digest(candidate['label'])
         and binding['input_row_sha256'] == record['input_row_sha256']
         and binding['evidence_sha256'] == record['evidence_ref']['sha256']
         and binding['checkpoint_content_independence_verified'] is True
         and binding['hand_roi_ref'] == candidate['hand_roi'], 'Exact semantic/evidence/independence binding differs')
    need(evidence['binding'] == {k: binding[k] for k in evidence['binding']}, 'Evidence binding differs')
    need(any(r['path'] == candidate_ref['path'] and r['sha256'] == candidate_ref['sha256'] for r in binding['source_refs']),
         'Receipt not bound to candidate revision')
    for reference in binding['source_refs']:
        need(sha(Path(reference['path']).read_bytes()) == reference['sha256'], 'Original review source reference changed')
    row = {'id': sid, 'source_id': sid, 'split': candidate['split'], 'video_path': candidate['video_path'],
        'audio_path': candidate['audio_path'], 'label': copy.deepcopy(candidate['label']), 'label_status': 'completed',
        'joint_complete': True, 'all_windows_nonempty': True, 'source_hint': record['source_hint'],
        'origin_root': record['origin_root'], 'hand_roi': candidate['hand_roi'],
        'annotator_model_sha256': candidate['annotator_model_sha256'],
        'sourceRefs': {'candidate': candidate_ref, 'terminal': record['terminal_ref'],
            'original_job': record['source_job_ref'], 'population': ref(ROOT / 'coarse3_filter_recovery_independent_reviews_v1/frozen.population.json'),
            'recovery_proof': record['recovery_proof_ref'], 'recovery_code': record['recovery_code_ref'],
            'recovery_test': record['recovery_test_ref']},
        'explicit_review_binding': {'schema': 'coarse3_filter_recovery_independent_review_v1', 'label_scope': record['label_scope'],
            'candidate_sha256': digest(candidate), 'label_sha256': digest(candidate['label']),
            'target_text_sha256': sha(candidate['target_text'].encode()), 'old_errors_are_not_prior': True,
            'original_membership': {k: membership[k] for k in ('batch_id', 'split', 'source_id',
                'original_member_count', 'membership_ref', 'membership_ids_sha256', 'original_csv_ref')}}}
    original_row = json.loads((transaction.parent / 'input.row.json').read_bytes())
    need(row == original_row and digest(row) == record['input_row_sha256'],
         'Exact original review input-row serialization differs')
    request, generation = checked(artifacts['request.json']), checked(artifacts['generation.json'])
    claim = json.loads(request['messages'][1]['content'][0]['text'])
    frames = evidence['frames']
    indices = [f['index'] for f in frames]
    need(claim['label_to_test'] == candidate['label'] and claim['allowed_source_frame_indices'] == indices
         and indices == binding['frame_indices'] and len(frames) == min(16, len(evidence['probe']['times'])),
         'Review request exact label/native sample indices differ')
    need(generation['request_sha256'] == artifacts['request.json']['sha256']
         and generation['raw_response_sha256'] == artifacts['response.raw.txt']['sha256']
         and generation['structured_output_schema_sha256'] == digest(request['json_schema'])
         and generation['model_identity_sha256'] == binding['model_sha256']
         and generation['model_call_performed'] is True and generation['finish_reason'] == 'stop'
         and 0 < generation['completion_tokens'] < generation['max_tokens'], 'Review generation incomplete/unbound')
    final, response = checked(artifacts['final.json']), checked(artifacts['response.json'])
    response_raw = checked(artifacts['response.raw.txt'])
    need(final['status'] == 'candidate' and final['json'] == response and response['status'] == record['status']
         and all(response.get(k) == v for k, v in response_raw.items())
         and response['errors'] == record['errors'] and response['uncertainties'] == record['uncertainties'],
         'Original raw/final/recovered review differ')
    for document in (receipt, final):
        need(all(document.get(k) is False for k in ('semantic_approval', 'human_review_completed',
             'formal_approval', 'formal_training', 'training_entry_updated')), 'Review cannot grant approval')
    model, code = checked(record['actual_reviewer_model_ref']), checked(record['actual_reviewer_code_ref'])
    need(model['sha256'] == binding['model_sha256'] and code['sha256'] == binding['code_sha256'], 'Historical model/code identity differs')
    source = evidence['source_identity']['files']
    for field in ('video_path', 'audio_path'):
        reference = source[field]
        need(reference['path'] == candidate[field] and sha(Path(reference['path']).read_bytes()) == reference['sha256'],
             'Actual original media bytes changed')
    need(source['video_path']['sha256'] == binding['video_sha256'], 'Source video binding differs')
    probe = json.loads(subprocess.check_output(['/opt/ms_tools/ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', candidate['video_path']]))['streams'][0]
    native_size = (probe['width'], probe['height'])
    hashes, images = [], []
    roi = checked(candidate['hand_roi'])
    for frame in frames:
        path = transaction.parent / frame['file']
        need(check_image(path, frame['sha256']) == native_size, 'Review body RGB is not native video size')
        hashes.append(frame['sha256'])
        images.append({'path': str(path), 'sha256': frame['sha256'], 'index': frame['index'], 'time': frame['time'], 'kind': 'native_source_RGB'})
        for crop in evidence['hand_rois']:
            if crop['frame_index'] != frame['index']:
                continue
            need(crop in roi['crops'] and crop['resized'] is False and crop['rotated'] is False
                 and crop['mirrored'] is False and crop['anatomical_side_guaranteed'] is False
                 and tuple(crop['source_frame_size']) == native_size, 'ROI not original same-frame native crop')
            need(check_image(crop['path'], crop['sha256']) == tuple(crop['crop_pixel_size']), 'ROI native pixel size differs')
            hashes.append(crop['sha256'])
            images.append({'path': crop['path'], 'sha256': crop['sha256'], 'index': crop['frame_index'],
                'time': crop['time'], 'kind': 'native_hand_ROI_unverified_side', 'pixel_size': crop['crop_pixel_size']})
    requested = [sha(base64.b64decode(c['image_url']['url'].split(',', 1)[1], validate=True))
        for c in request['messages'][1]['content'] if c['type'] == 'image_url']
    need(requested == hashes, 'Review request native RGB/ROI bytes differ')
    review = {'status': 'pass', 'independent': True, 'candidate_ref': candidate_ref,
        'global_revision': REVISION, 'semantic_label_sha256': digest(candidate['label']),
        'input_row_sha256': record['input_row_sha256'], 'review_receipt': record['review_receipt'],
        'review_record_ref': ref(ROOT / 'coarse3_completed_review_recovery_v1/sources' / (sid + '.json')),
        'evidence_ref': record['evidence_ref'], 'reviewer_model_ref': record['actual_reviewer_model_ref'],
        'historical_code_inventory_ref': record['actual_reviewer_code_ref'], 'original_actual_VLM_calls': 1,
        'delivery_model_calls': 0, 'native_source_frames': len(frames), 'native_hand_rois': len(evidence['hand_rois']), **FLAGS}
    need(checked(review['review_record_ref']) == record, 'Source record differs from immutable recovered record array')
    return member, candidate, review, images


def derive(progress, rows, verified, manifest):
    need(len({key(r) for r in rows}) == len(rows), 'Duplicate base source keys')
    result = {key(r): copy.deepcopy(r) for r in rows}
    increments = Counter(verified_candidates=3, overlapping_source_unique=0, new_source_unique=0,
                         newly_selected_revision_unique=0, net_complete_source_unique=0)
    for member, candidate, review, _ in verified:
        identity = key(member)
        prior = result.get(identity)
        increments['overlapping_source_unique'] += prior is not None
        increments['new_source_unique'] += prior is None
        alternative = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
            'origin': 'coarse3_completed_review_recovery', 'kind': 'complete_candidate', 'revision': True,
            'global_revision': REVISION, 'joint_candidate_ref': review['candidate_ref'],
            'label': copy.deepcopy(candidate['label']), 'semantic_label_sha256': review['semantic_label_sha256'],
            'source_binding': candidate['source_binding'], 'reviews': [review], 'reviewed': True, 'sampled_pass': True,
            'pendinghuman': False, 'individual_review_status': 'reviewed_exact_candidate_revision_and_payload', **FLAGS}
        if prior is None:
            prior = {**{k: member[k] for k in ('source_id', 'split', 'batch_id', 'source_ordinal', 'source_refs')},
                'state': 'missing', 'selection': None, 'alternatives': [], 'ready_alternatives': [], 'failed_attempts': [],
                'pendinghuman': False, 'overlay_kind': 'candidate_not_training_approved', **FLAGS}
            result[identity] = prior
        old_complete = prior['state'] == 'complete_candidate'
        prior['pre_coarse3_state'] = prior['state']
        prior['pre_coarse3_selection'] = copy.deepcopy(prior['selection'])
        matching = [a for a in prior['alternatives'] if a.get('joint_candidate_ref') == review['candidate_ref']]
        need(not matching, 'Already integrated candidate requires a distinct original base snapshot')
        prior['alternatives'].append(alternative)
        if not old_complete:
            prior['selection'], prior['state'] = alternative, 'complete_candidate'
            increments['newly_selected_revision_unique'] += 1
            increments['net_complete_source_unique'] += 1
        prior.update(coarse3_reviewed_revision_present=True, approval_pending=True,
                     prior_alternatives_errors_and_reviews_preserved=True, **FLAGS)
    states = ('complete_candidate', 'partial_candidate', 'ready_not_joint', 'terminal_without_result', 'failed', 'missing')
    counts = Counter({s: 0 for s in states})
    split_counts = {s: Counter({state: 0 for state in states}) for s in manifest['counts']}
    batches = {d['batch_id']: Counter({s: 0 for s in states}) for d in manifest['batches']}
    for row in result.values():
        need(row['state'] in states, 'Unknown derived source state')
        counts[row['state']] += 1
        split_counts[row['split']][row['state']] += 1
        batches[row['batch_id']][row['state']] += 1
    counts['missing'] += manifest['total'] - len(result)
    for split, expected in manifest['counts'].items():
        split_counts[split]['missing'] += expected - sum(split_counts[split].values())
    descriptors = []
    for descriptor in manifest['batches']:
        c = batches[descriptor['batch_id']]
        c['missing'] += descriptor['count'] - sum(c.values())
        need(sum(c.values()) == descriptor['count'], 'Batch denominator mismatch')
        descriptors.append({'batch_id': descriptor['batch_id'], 'split': descriptor['split'],
            'expected': descriptor['count'], 'counts': dict(c), 'all_joint_complete': c['complete_candidate'] == descriptor['count'],
            'coarse3_subset_approves_original_batch': False})
    need(sum(counts.values()) == manifest['total'], 'Source denominator mismatch')
    output = copy.deepcopy(progress)
    output.update(schema='coarse3_exact_review_derived_overlay_v1', base_snapshot_only=True, latest_live_base_claimed=False,
        base_before_coarse3_counts=copy.deepcopy(progress['counts']), counts={**progress['counts'], **dict(counts)},
        split_counts={s: dict(c) for s, c in split_counts.items()}, batches=descriptors,
        coarse3_incremental_counts=dict(increments), coarse3_original_actual_VLM_calls=3,
        coarse3_recovery_model_calls=0, delivery_model_calls=0, new_budget_reservations=0,
        all_individually_reviewed_source_unique=sum(any(a.get('reviewed') for a in r['alternatives']) for r in result.values()),
        all_individually_sampled_pass_source_unique=sum(any(a.get('sampled_pass') for a in r['alternatives']) for r in result.values()),
        selected_individually_reviewed_source_unique=sum(bool((r.get('selection') or {}).get('reviewed')) for r in result.values()),
        selected_individually_sampled_pass_source_unique=sum(bool((r.get('selection') or {}).get('sampled_pass')) for r in result.values()),
        all_pendinghuman_source_unique=sum(bool(r.get('pendinghuman')) for r in result.values()),
        original_base_review_counts_retained_unchanged=True, approval_pending=True,
        selection_policy='preserve_preexisting_complete_revision; select_exact_coarse3_if_incomplete; retain_all_prior_scores_on_original_labels',
        all_batches_reviewed=False, **FLAGS)
    return output, sorted(result.values(), key=lambda r: r['source_ordinal'])


def page(verified):
    esc = lambda value: html.escape(str(value), quote=True)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>coarse3 精确候选审核</title>',
        '<style>body{font:16px sans-serif;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.frames{display:flex;flex-wrap:wrap;gap:12px}img{max-width:240px;height:auto}figure{margin:0}video{max-height:500px;max-width:100%}</style>',
        '<h1>coarse3：3 条确切 revision 的独立 AI 抽样 pass</h1>',
        '<p>原真实 8B 审核 3 次；恢复与本次接入均 0 模型调用。非全批次通过、非人工批准、非训练准入。稀疏原RGB不是连续视频；手ROI无解剖左右保证，不添加原生细节。旧评分未借给新label。</p>',
        '<p><a href="./">返回原首页</a></p>']
    for member, candidate, review, images in verified:
        body.append('<section><h2>' + esc(key(member)) + '</h2><pre>' + esc(candidate['target_text']) + '</pre>')
        body.append('<video controls preload="metadata" src="media?' + esc(urlencode({'p': candidate['video_path']})) + '"></video>')
        body.append('<details><summary>确切候选和评分绑定</summary><pre>' + esc(json.dumps(review, ensure_ascii=False, indent=2)) + '</pre></details><div class="frames">')
        for image in images:
            body.append('<figure><a href="media?' + esc(urlencode({'p': image['path']})) + '"><img loading="lazy" src="media?' + esc(urlencode({'p': image['path']})) + '" alt="' + esc(image['kind']) + '"></a><figcaption>'
                + esc(str(image['index']) + ' · ' + str(image['time']) + 's · ' + image['kind']) + '</figcaption></figure>')
        body.append('</div></section>')
    return ''.join(body).encode()


def run(args):
    base = Path(args.base_root).resolve()
    out = Path(args.out).resolve()
    need(out.parent == ROOT and out.name == 'coarse3_delivery_derived_v1', 'Use the isolated delivery namespace only')
    summary = pinned('coarse3_completed_review_recovery_v1/summary.json')
    records = pinned('coarse3_completed_review_recovery_v1/review.records.json')
    recovery = pinned('curated_coarse_empty_filter_recovery_v1/summary.json')
    manifest = pinned('curated222939_inputs_v1/manifest.summary.json')
    need(summary['statuses'] == {'pass': 3} and summary['original_model_calls'] == summary['actual_VLM_calls'] == 3
         and summary['recovery_model_calls'] == 0 and tuple(r['source_id'] for r in records) == IDS, 'Fixed three original results required')
    verified = [verify(r, manifest, recovery) for r in records]
    progress_path, rows_path = base / 'progress.json', base / 'candidates.jsonl'
    p_raw, r_raw = progress_path.read_bytes(), rows_path.read_bytes()
    progress = json.loads(p_raw)
    need(progress['candidate_export_sha256'] == sha(r_raw), 'Live base snapshot not internally stable; retry later')
    need(progress['authoritative_manifest']['sha256'] == PINS['curated222939_inputs_v1/manifest.summary.json'], 'Different base population')
    need(progress_path.read_bytes() == p_raw and rows_path.read_bytes() == r_raw, 'Live base changed while reading; retry later')
    rows = [json.loads(line) for line in r_raw.splitlines() if line.strip()]
    base_counts = Counter(r['state'] for r in rows)
    base_counts['missing'] += manifest['total'] - len(rows)
    need(all(progress['counts'][s] == n for s, n in base_counts.items()), 'Base source denominator differs')
    derived, result = derive(progress, rows, verified, manifest)
    binding = {'base_progress_ref': {'path': str(progress_path), 'sha256': sha(p_raw)},
        'base_candidates_ref': {'path': str(rows_path), 'sha256': sha(r_raw)},
        'fixed_inputs': [{'path': str(ROOT / n), 'sha256': h} for n, h in PINS.items()], 'adapter_ref': ref(__file__)}
    snapshot = out / 'snapshots' / digest(binding)
    immutable(snapshot / 'base.progress.json', p_raw)
    immutable(snapshot / 'base.candidates.jsonl', r_raw)
    rows_encoded = b''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode() + b'\n' for r in result)
    derived.update(candidate_export_sha256=sha(rows_encoded), frozen_delivery_binding=binding,
        frozen_base_progress_ref=ref(snapshot / 'base.progress.json'), frozen_base_candidates_ref=ref(snapshot / 'base.candidates.jsonl'))
    refs = [immutable(snapshot / 'candidates.jsonl', rows_encoded), immutable(snapshot / 'progress.json', raw(derived)),
        immutable(snapshot / 'review.records.json', raw([r for _, _, r, _ in verified])),
        immutable(snapshot / 'review.html', page(verified))]
    report = {'schema': 'coarse3_delivery_cpu_verification_v1', 'snapshot_root': str(snapshot),
        'source_count': len(result), 'expected_sources': manifest['total'], 'counts': derived['counts'],
        'coarse3_incremental_counts': derived['coarse3_incremental_counts'],
        'exact_reviewed_revision_count': 3, 'original_actual_VLM_calls': 3, 'recovery_model_calls': 0,
        'delivery_model_calls': 0, 'new_budget_reservations': 0,
        'native_source_frames': sum(r['native_source_frames'] for _, _, r, _ in verified),
        'native_hand_rois': sum(r['native_hand_rois'] for _, _, r, _ in verified),
        'all_individually_reviewed_source_unique': derived['all_individually_reviewed_source_unique'],
        'all_individually_sampled_pass_source_unique': derived['all_individually_sampled_pass_source_unique'],
        'artifacts': refs, 'web_status': 'offline_page_ready_existing_server_has_no_static_route_no_frozen_code_changed',
        'required_web_integration': 'Add an explicitly SHA-bound fixed review.html route only in an approved non-frozen replacement server; use existing relative media endpoint. Do not loosen media allowlist.',
        'live_worker_overlay_webupdater_modified': False, **FLAGS}
    report_ref = immutable(snapshot / 'verification.json', raw(report))
    return {**report, 'verification_ref': report_ref}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-root', default=str(ROOT / 'curated222939_overlay_with_oldbar_v1'))
    parser.add_argument('--out', default=str(ROOT / 'coarse3_delivery_derived_v1'))
    parser.add_argument('--once', action='store_true', required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False), flush=True)
