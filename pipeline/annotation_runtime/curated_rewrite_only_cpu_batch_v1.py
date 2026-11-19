"""Single-pass CPU alternatives from exact original rewrite-only evidence; never dispatch."""
import argparse
import copy
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
QUEUE = BAR / 'curated222939_repair_v1'
PRODUCTION = BAR / 'curated222939_production_v1'
OUT = BAR / 'curated_rewrite_only_cpu_candidates_v1'
REVISION = 'rewrite-only-cpu-20261025-v1'
SCHEMA = 'curated_rewrite_only_cpu_batch_v1'
HELPERS = {
    'curated_rewrite_facts_offline_v1.py': '75e9f95ba24f2213ef49708f330bb188e71e09dab319b5c66b7fc84c369d105f',
    'curated_clause_salvage_review_bind_v5.py': 'caf717320525f8e51ce69893e5de2723ce4396a7267e66bcd435d57fef0b4d67',
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


off = module('rewrite_only_offline', RT / 'curated_rewrite_facts_offline_v1.py')
v5 = module('rewrite_only_frozen_bind', RT / 'curated_clause_salvage_review_bind_v5.py')
q, w, bar = off.q, off.w, off.bar
_BASE_SHA = q.sha
_CACHE = {}


class Pending(ValueError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


def require(ok, reason, detail):
    if not ok:
        raise Pending(reason, detail)


def register_shared(reference):
    path = str(Path(reference['path']).resolve())
    previous = _CACHE.get(path)
    if previous is not None:
        if previous['sha256'] != reference['sha256']:
            raise q.SourceChanged('Registered immutable plan/style expected SHA changed')
        cached_sha(path)
        return
    before = q.original_csv_stat(path)
    if _BASE_SHA(path) != reference['sha256'] or q.original_csv_stat(path) != before:
        raise q.SourceChanged('Immutable plan/style first stable SHA differs')
    _CACHE[path] = dict(sha256=reference['sha256'], stat=before)


def cached_sha(path):
    key = str(Path(path).resolve())
    cached = _CACHE.get(key)
    if cached is None:
        return _BASE_SHA(path)
    if q.original_csv_stat(key) != cached['stat']:
        raise q.SourceChanged('Registered immutable plan/style stat changed')
    return cached['sha256']


# This is a private CPU import of the unchanged queue; media/evidence are never cached.
q.sha = cached_sha


def frozen():
    for name, expected in HELPERS.items():
        if _BASE_SHA(RT / name) != expected:
            raise q.SourceChanged('Frozen CPU helper changed: ' + name)
    frozen_bar, _ = v5.frozen_protocol()
    require(frozen_bar is bar, 'frozen_protocol_mismatch', 'Original filter import differs')


def scope(job, visual):
    require(not job.get('legacy_visual_demo') and job['id'] != w.RAQ_ID
            and job.get('stages') and not set(job['stages']) - {'rewrite', 'styleunknown'}
            and all(f['stage'] in ('rewrite', 'styleunknown') for f in job['failures']),
            'excluded_non_rewrite_scope', 'Reobserve/media/RAQ/legacy source cannot be rewritten offline')
    require(visual.get('final', {}).get('status') == 'failed',
            'not_actual_failed_final', 'Original immutable final must actually have failed')
    windows = [m['window'] for m in q.checked(job['refs']['prepared'])['windows']]
    observations, facts = visual['window_observations'], visual['facts']
    require(bool(windows) and len(windows) == len(observations)
            and all(o['window'] == win and o['status'] == 'candidate'
                    and o.get('json', {}).get('events') for o, win in zip(observations, windows))
            and len(visual['original_windows']) == len(windows),
            'incomplete_original_windows', 'All original windows require candidate/nonempty motion evidence')
    require(facts == bar.facts_from_observations(windows, observations),
            'facts_observation_mismatch', 'Original facts/parts/raw_text/frame refs differ from original observations')
    require(all(any(f['window_id'] == win['window_id'] for f in facts) for win in windows),
            'incomplete_original_windows', 'Every original window must have existing facts')
    for fact in facts:
        require(isinstance(fact['raw_text'], str) and fact['raw_text'].strip()
                and bool(fact['parts']) and set(fact['parts']) <= v5.COARSE_PARTS,
                'non_coarse_source_fact', 'Every existing fact must be nonempty coarse body evidence')
        # Whole-source exclusion: never delete words, split commas, or synthesize replacements.
        source = fact['source_clause']
        texts = [fact['raw_text'], source['raw_source_clause']]
        require(not any(v5.named_hand_detail(text) for text in texts),
                'excluded_fine_hand_source', 'Original admitted fact/source contains hand-shape detail')
        require(not any(v5.UNCERTAIN.search(text) for text in texts),
                'pending_uncertain_source', 'Original admitted fact/source contains uncertain/negative detail')
    return windows


def receipt_inventory(attempt, status):
    receipt_ref = q.identity(attempt / 'receipt.json')
    receipt = q.checked(receipt_ref)
    names = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'final.json'}
    if status == 'candidate':
        names.add('response.json')
    require(names <= set(receipt['files_sha256']), 'missing_original_evidence', 'Original receipt lacks six artifact SHA inventory')
    refs = [receipt_ref]
    for name, expected in sorted(receipt['files_sha256'].items()):
        artifact = (attempt / name).resolve()
        require(artifact.parent == attempt, 'unsafe_original_path', 'Original receipt artifact escapes attempt')
        reference = dict(path=str(artifact), sha256=expected)
        q.verify_bytes(reference)
        refs.append(reference)
    require(receipt['status'] == status and receipt.get('semantic_approval') is False
            and receipt.get('human_review_completed') is False,
            'receipt_status_mismatch', 'Original receipt status/weak-candidate flags changed')
    require(receipt['binding'] == q.load(attempt / 'binding.json'),
            'receipt_binding_mismatch', 'Original receipt/binding differs')
    return receipt, refs


def replay_window(job, original, observation, context, root, prepared):
    wid = observation['window']['window_id']
    require(re.fullmatch(r'w[0-9]+', wid) is not None, 'unsafe_original_path', 'Unsafe original window ID')
    attempt = Path(observation['attempt_path']).resolve()
    require(attempt.parent == root / 'windows' / wid and re.fullmatch(r'attempt-[0-9]+', attempt.name),
            'unsafe_original_path', 'Original attempt path/source/window differs')
    manifest, assets = v5.check_window_assets(context, observation, root, prepared, q)
    receipt, references = receipt_inventory(attempt, 'candidate')
    references += assets
    binding = q.load(attempt / 'binding.json')
    require(binding == {**original['binding'], 'window': observation['window'],
                        'frames_sha256': bar.batch.digest(manifest['frames'])},
            'receipt_binding_mismatch', 'Original window/frame/source receipt binding changed')
    raw = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
    response, final = q.load(attempt / 'response.json'), q.load(attempt / 'final.json')
    request = q.load(attempt / 'request.json')
    generation = bar.core.validate_generation(attempt, request)
    model = q.checked(job['refs']['model_identity_ref'])
    require(response == observation['json'] and final['status'] == 'candidate' and final['json'] == response
            and generation.get('model_call_performed') is True and generation['model_identity_sha256'] == model['sha256']
            and Path(request['model']).resolve() == Path(model['root']).resolve(),
            'original_generation_mismatch', 'Original response/generation/model identity or path differs')
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    replay = bar.filter_supported_observation(raw, manifest, forbidden)
    require(replay == response, 'frozen_filter_replay_mismatch', 'Full original filter replay differs from admitted observation')
    facts = [f for f in original['facts'] if f['window_id'] == wid]
    for fact in facts:
        v5.replay_fact(fact, observation, raw, manifest, forbidden, bar, references, isolatable=False)
        supports = {p['part']: set(p['frame_indices']) for p in replay['part_support']}
        actual = {f['index'] for f in manifest['frames']}
        require(set(fact['frame_indices']) <= actual
                and all(len(set(fact['frame_indices']) & supports.get(p, set())) >= 2 for p in fact['parts']),
                'part_actual_frame_deficit', 'Every required part needs its own two distinct actual frames')
    for reference in references:
        q.verify_bytes(reference)
    return dict(window_id=wid, original_attempt_path=str(attempt), fact_ids=[f['fact_id'] for f in facts],
                original_attempt_refs=references, complete_original_filter_replayed=True,
                per_part_two_distinct_actual_frames_checked=True)


def check_style(job, style, plan, root, prepared):
    require(style.get('status') in ('resolved', 'candidate') and style.get('dance_style') not in (None, '', 'unknown')
            and style['id'] == job['id'], 'pending_style_unknown', 'Original style must already be resolved/candidate')
    metadata = style['metadata']
    q.verify_references_by_content(metadata.get('source_refs', {}))
    require(metadata['video_sha256'] == plan['binding']['video_sha256'],
            'style_source_mismatch', 'Original style video SHA differs')
    if style['status'] == 'candidate':
        attempt = Path(metadata['request_path']).resolve().parent
        require(attempt.parent == root / 'style' and re.fullmatch(r'attempt-[0-9]+', attempt.name),
                'unsafe_original_path', 'Original style attempt escapes source')
        receipt, refs = receipt_inventory(attempt, 'candidate')
        for name in ('generation.json', 'receipt.json'):
            reference = metadata[name]
            q.verify_bytes(reference)
            require(Path(reference['path']).resolve() == attempt / name and reference['value'] == q.load(attempt / name),
                    'style_source_mismatch', 'Original style embedded receipt/generation differs')
        model = q.checked(job['refs']['model_identity_ref'])
        request = q.load(attempt / 'request.json')
        generation = bar.core.validate_generation(attempt, request)
        require(generation['model_identity_sha256'] == model['sha256']
                and Path(request['model']).resolve() == Path(model['root']).resolve(),
                'style_source_mismatch', 'Original style model binding differs')
        evidence = metadata['RGB_evidence']
        require(receipt['binding']['RGB_evidence'] == evidence
                and evidence['video_sha256'] == plan['binding']['video_sha256']
                and Path(evidence['rgb_root']).resolve() == root / 'rgb',
                'style_source_mismatch', 'Original style actual RGB source differs')
        union = q.checked(job['refs']['raw_rgb_manifest'])
        require(all(frame in union['frames'] for frame in evidence['frames']),
                'style_source_mismatch', 'Original style RGB not in actual union manifest')
        import ms_dance_style as style_protocol
        raw = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
        response = style_protocol.check_style(raw, [f['index'] for f in evidence['frames']])
        final = q.load(attempt / 'final.json')
        require(response == q.load(attempt / 'response.json') == final['json'] == metadata['transaction']['json']
                and response['dance_style'] == style['dance_style'],
                'style_source_mismatch', 'Original style full response replay differs')
        return refs
    return list(v5.immutable_references(metadata.get('source_refs', {})))


def derive(job_ref, out):
    frozen()
    job = q.checked(job_ref)
    key = q.safe_key(job)
    require(job['source_key'] == key and Path(job_ref['path']).resolve() == QUEUE / 'jobs' / (key + '.json')
            and Path(job['production']).resolve() == PRODUCTION and Path(job['queue_root']).resolve() == QUEUE
            and job['global_revision'] == 'repair-20261025-v1',
            'unsafe_original_path', 'Exact canonical original source/queue/revision required')
    original = q.checked(job['refs']['visual_result'])
    windows = scope(job, original)
    for name in ('plan_ref', 'style_result'):
        if job['refs'].get(name):
            register_shared(job['refs'][name])
    plan = q.checked(job['refs']['plan_ref'])
    register_shared(plan['source_refs']['manifest'])
    # Original helper preserves all facts and raw_text and performs original source/job bindings.
    checked_job, row, style, visual, provenance, joint_original = off.candidate(job_ref, REVISION)
    require(checked_job == job, 'source_changed', 'Original job changed during CPU derivation')
    root = Path(job['visual_root']).resolve()
    require(root.parent.name == 'sources' and root.name == job['id']
            and re.fullmatch(r'shard-[0-9]{4}', root.parent.parent.name)
            and root.parent.parent.parent == PRODUCTION / 'visual'
            and Path(job['refs']['plan_ref']['path']).resolve() == PRODUCTION / 'sources' / key / 'bar_plan.json'
            and Path(job['refs']['prepared']['path']).resolve() == root / 'prepared.json'
            and Path(job['refs']['prepared_receipt']['path']).resolve() == root / 'prepared.receipt.json'
            and job['refs']['raw_rgb_manifest'] == original['raw_rgb_manifest'],
            'unsafe_original_path', 'Exact original prepared/plan/RGB lineage required')
    prepared = q.verify_prepared_content(root, plan, bar, '/opt/ms_tools/ffprobe')
    require(prepared == q.checked(job['refs']['prepared']) and prepared['binding'] == original['binding'],
            'prepared_binding_mismatch', 'Original prepared actual media/frame/ROI differs')
    q.verify_references_by_content(prepared)
    style_refs = check_style(job, style, plan, root, prepared)
    context = {**original, 'source_key': key, 'original_job_ref': job_ref,
               'original_visual_result_ref': job['refs']['visual_result']}
    evidence = [replay_window(job, original, observation, context, root, prepared)
                for observation in original['window_observations']]
    final_attempt = Path(original['final']['attempt_path']).resolve()
    require(final_attempt.parent == root / 'rewrite' and re.fullmatch(r'attempt-[0-9]+', final_attempt.name),
            'unsafe_original_path', 'Original failed rewrite attempt path differs')
    _, failed_refs = receipt_inventory(final_attempt, 'failed')
    require(q.load(final_attempt / 'final.json') == {k:v for k,v in original['final'].items() if k != 'attempt_path'},
            'not_actual_failed_final', 'Original actual failed final receipt differs')
    out = Path(out).resolve()
    for protected in (PRODUCTION, QUEUE, RT):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Output must be isolated from production, queue and runtime')
    destination = out / 'revisions' / REVISION / key
    refs = [copy.deepcopy(job_ref)] + list(copy.deepcopy(job['refs']).values())
    refs += [q.identity(root / name) for name in ('plan.json', 'plan.line.raw.json', 'binding.json')]
    refs += list(v5.immutable_references(prepared)) + style_refs + failed_refs
    union = q.checked(job['refs']['raw_rgb_manifest'])
    refs += [dict(path=str((root / 'rgb' / frame['file']).resolve()), sha256=frame['sha256'])
             for frame in union['frames']]
    if prepared['hand_roi'].get('available'):
        roi_path = Path(prepared['hand_roi']['path']).resolve()
        refs += [q.identity(roi_path.with_name(name)) for name in ('source_manifest.json', 'extract_receipt.json')]
        refs += [dict(path=crop['path'], sha256=crop['sha256']) for crop in q.load(roi_path)['crops']]
    refs += [dict(path=str(RT / name), sha256=expected) for name, expected in {**HELPERS, **v5.REPLAY_CODE_SHA}.items()]
    audit = dict(schema=SCHEMA, source_key=key, id=job['id'], split=job['split'], batch_id=job['batch_id'],
                 original_job_ref=job_ref, original_source_refs=refs, original_window_evidence=evidence,
                 facts_raw_text_unchanged=True, facts_and_parts_full_original_replay=True,
                 actual_media_SHA_PTS_RGB_ROI_checked=True, admitted_claim_fine_or_uncertain_exclusion=True,
                 unadmitted_context_detail_does_not_suppress_admitted_coarse_facts=True,
                 model_calls=0, extra_source_attempts=0, attempt_budget_consumed=False,
                 primary_worker_reservation_modified=False, independent_review_required=True, **q.FLAGS)
    q.verify_job(job)
    for reference in refs:
        q.verify_bytes(reference)
    q.save_once(destination / 'evidence.manifest.json', audit)
    evidence_ref = q.identity(destination / 'evidence.manifest.json')
    provenance.update(schema=SCHEMA, model_calls=0, alternative_to_primary_repair=True,
                      source_dedup_key=key, original_batch_id=job['batch_id'], original_population_quality_approved=False,
                      primary_worker_reservation_modified=False, original_evidence_manifest_ref=evidence_ref)
    visual.update(provenance)
    q.save_once(destination / 'visual_result.json', visual)
    visual['result_ref'] = q.identity(destination / 'visual_result.json')
    joint = off.pipeline.joint_candidate(row, visual, style, joint_original.get('annotator_model_sha256'))
    joint.update(provenance)
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in joint_original:
            joint[name] = copy.deepcopy(joint_original[name])
    require(joint['joint_complete'] and joint['all_windows_nonempty'],
            'incomplete_joint_candidate', 'Verbatim original facts/style cannot produce full joint alternative')
    joint['status'] = 'rewrite_only_cpu_joint_candidate'
    q.save_once(destination / 'result.json', joint)
    return dict(source_key=key, id=job['id'], split=job['split'], batch_id=job['batch_id'],
                status=joint['status'], result_ref=q.identity(destination / 'result.json'),
                visual_result_ref=visual['result_ref'], original_job_ref=job_ref,
                original_visual_result_ref=job['refs']['visual_result'], original_result_ref=job['original_result_ref'],
                evidence_manifest_ref=evidence_ref, joint_complete=True, independent_review_status='pending_independent_vlm_review',
                risk_review_case=True, model_calls=0, attempt_budget_consumed=False, **q.FLAGS)


def stable_prefix(limit):
    require(type(limit) is int and 1 <= limit <= 500, 'unsafe_limit', 'Only a bounded 1..500 stable ready prefix is allowed')
    path = QUEUE / 'ready.jsonl'
    with path.open('rb') as stream:
        before = stream.fileno()
        import os
        initial = os.fstat(before)
        lines = []
        for _ in range(limit):
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                break
            lines.append(line)
        prefix = b''.join(lines)
        stream.seek(0)
        require(stream.read(len(prefix)) == prefix and os.fstat(before).st_ino == initial.st_ino
                and path.stat().st_ino == initial.st_ino and path.stat().st_size >= len(prefix),
                'ready_prefix_changed', 'Original finite ready prefix changed/replaced')
    return [json.loads(line) for line in lines], dict(path=str(path), prefix_bytes=len(prefix),
        prefix_lines=len(lines), prefix_sha256=hashlib.sha256(prefix).hexdigest())


def once(limit=500, ids=None, out=OUT):
    out = Path(out).resolve()
    for protected in (PRODUCTION, QUEUE, RT):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Output must be isolated from production, queue and runtime')
    notices, prefix = stable_prefix(limit)
    selected = set(ids or [])
    for sid in selected:
        require(re.fullmatch(r'(?:train/|val/|test/)?[A-Za-z0-9][A-Za-z0-9_-]*', sid),
                'unsafe_source_id', 'Unsafe requested source ID')
    results, pending, seen = [], [], {}
    for notice in notices:
        key = notice['source_key']
        require(re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key), 'unsafe_source_id', 'Unsafe ready source key')
        if selected and key not in selected and key.split('/')[1] not in selected:
            continue
        previous = seen.get(key)
        if previous is not None:
            if previous != notice['job_ref']:
                raise q.SourceChanged('Duplicate ready source has different original job SHA')
            continue
        seen[key] = notice['job_ref']
        try:
            job = q.checked(notice['job_ref'])
            require(job['source_key'] == key and notice['stages'] == job['stages'],
                    'ready_job_mismatch', 'Ready/job source/stage binding differs')
            require(not set(job['stages']) - {'rewrite', 'styleunknown'},
                    'excluded_non_rewrite_scope', 'Original source needs reobserve/media work')
            results.append(derive(notice['job_ref'], out))
        except (Pending, q.SourceChanged, v5.MissingOriginalEvidence, ValueError, KeyError, OSError) as error:
            reason = (error.reason if isinstance(error, Pending) else 'source_changed' if isinstance(error, q.SourceChanged)
                      else 'missing_original_evidence' if isinstance(error, (v5.MissingOriginalEvidence, FileNotFoundError))
                      else 'original_protocol_replay_failed')
            pending.append(dict(source_key=key, original_job_ref=notice['job_ref'], status='pending', reason=reason,
                                detail=str(error), model_calls=0, attempt_budget_consumed=False, **q.FLAGS))
    report = dict(schema=SCHEMA, global_revision=REVISION, original_ready_prefix=prefix,
        requested_ids=sorted(selected),
        requested_ids_not_in_prefix=sorted(sid for sid in selected
            if sid not in seen and not any(key.split('/')[1] == sid for key in seen)),
        source_dedup_count=len(seen), joint_complete_count=len(results),
        model_calls=0, extra_source_attempts=0, attempt_budget_consumed=False, GPU_dispatch_performed=False,
        primary_worker_reservation_modified=False, original_population_quality_approved=False,
        pending_reason_counts=dict(Counter(row['reason'] for row in pending)), candidates=results, pending=pending,
        independent_review_required=True, **q.FLAGS)
    destination = Path(out).resolve() / 'revisions' / REVISION
    # The manifest itself is immutable; changed selection/output requires another isolated --out.
    q.save_once(destination / 'review-required.manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=500, help='Original ready stable prefix, at most 500 rows')
    parser.add_argument('--ids', nargs='+', help='Explicit source IDs or split/ID; restricted to the prefix')
    parser.add_argument('--once', action='store_true', help='Single pass (also the default); no watch deployment')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    report = once(args.limit, args.ids, args.out)
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidates', 'pending')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
