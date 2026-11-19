"""CPU-only independent clause salvage from explicitly indexed immutable repair terminals."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SCHEMA = 'curated_clause_salvage_offline_v1'
DEFAULT_REVISION = 'clause-salvage-20261025-v1'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


offline = load_module('clause_salvage_facts', RT / 'curated_rewrite_facts_offline_v1.py')
scope = load_module('clause_salvage_scope', RT / 'source_repair_scope_v3/raq_scope_revalidate.py')
q, bar, pipeline, watch = offline.q, offline.bar, offline.pipeline, offline.w
FLAGS = {**q.FLAGS, **scope.FLAGS, 'model_calls': 0, 'model_call_performed': False,
         'VLM_dispatch_performed': False, 'model_rewrite_performed': False,
         'extra_source_attempts': 0, 'attempt_budget_consumed': False,
         'repair_completed': False, 'original_raw_or_terminal_modified': False}


def safe_revision(revision):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe independent revision')


def isolated(out, repair, production=None):
    out = Path(out).resolve()
    for source in (repair, production):
        if source is not None:
            source = Path(source).resolve()
            if out == source or out.is_relative_to(source) or source.is_relative_to(out):
                raise ValueError('Clause output must be independent of repair and production')
    return out


def metadata_clauses(value):
    notes = []
    for event in value.get('events', []):
        if not isinstance(event, dict) or not isinstance(event.get('description'), str):
            continue
        for ci, clause in enumerate(re.split(r'(?<=[.!?;])\s+', event['description'])):
            if scope.internal_detail(clause):
                notes.append({'raw_event_id': event.get('id'), 'raw_clause_index': ci,
                    'raw_source_clause': clause, 'original_description': event['description'],
                    'original_parts': copy.deepcopy(event.get('parts')),
                    'original_frame_indices': copy.deepcopy(event.get('frame_indices')),
                    'reason': 'uncertain_or_negated_detail_not_an_affirmative_motion_fact',
                    'text_preserved_verbatim': True, 'stored_as_metadata_not_motion_fact': True})
    return notes


def salvage_clauses(value, manifest, forbidden):
    # The frozen filter alone owns splitting, parts, IDs, and event frame references.
    result = bar.filter_supported_observation(value, manifest, forbidden)
    notes, retained = metadata_clauses(value), []
    for event in result['events']:
        if scope.internal_detail(event['description']):
            continue
        detail = (scope.SHAPE.search(event['description']) or
                  re.search(r'\b(?:fingers?|thumbs?|digits?|knuckles?)\b', event['description'], re.I))
        error = 'coarse_motion_only_finger_detail_not_reapproved' if detail else None
        if error:
            raw_event = next(raw for raw in value['events']
                             if isinstance(raw, dict) and raw.get('id') == event['source_clause']['raw_event_id'])
            note = {**copy.deepcopy(event['source_clause']), 'clause_event_id': event['id'],
                'original_parts': copy.deepcopy(raw_event['parts']),
                'filtered_clause_parts': copy.deepcopy(event['parts']),
                'original_frame_indices': copy.deepcopy(raw_event['frame_indices']),
                'reason': error, 'strict_named_shape_gate': True,
                'text_preserved_verbatim': True, 'stored_as_metadata_not_motion_fact': True}
            notes.append(note)
            result['support_filter_audit']['rejected_events'].append(copy.deepcopy(note))
            continue
        retained.append(event)
    retained_sources = {(event['source_clause']['raw_event_id'], event['source_clause']['raw_clause_index'])
                        for event in retained}
    noted_sources = {(note['raw_event_id'], note['raw_clause_index']) for note in notes}
    for event in value['events']:
        if not isinstance(event, dict) or not isinstance(event.get('description'), str):
            continue
        for ci, clause in enumerate(re.split(r'(?<=[.!?;])\s+', event['description'])):
            source = (event.get('id'), ci)
            if scope.SHAPE.search(clause) and source not in retained_sources and source not in noted_sources:
                notes.append({'raw_event_id': event.get('id'), 'raw_clause_index': ci,
                    'raw_source_clause': clause, 'original_description': event['description'],
                    'original_parts': copy.deepcopy(event.get('parts')),
                    'original_frame_indices': copy.deepcopy(event.get('frame_indices')),
                    'reason': 'original_support_filter_did_not_admit_named_shape_clause',
                    'text_preserved_verbatim': True, 'stored_as_metadata_not_motion_fact': True})
    result.update(events=retained, abstain=not retained)
    result['support_filter_audit']['admitted_event_count'] = len(retained)
    result['scope_filter_audit'] = {'rejected_or_internal_clauses': notes,
        'uncertainty_tokens_removed_to_strengthen_claim': False,
        'affirmative_text_modified': False, 'frame_references_added': False,
        'comma_splitting_added': False, 'actual_part_support_gate_unchanged': True,
        'native_named_shape_gate_preserved': True, 'filter_is_not_semantic_truth_review': True}
    for note in notes:
        if note['raw_source_clause'] not in result['uncertainties']:
            result['uncertainties'].append(note['raw_source_clause'])
    return result


def checked_attempt(attempt, job, manifest, binding, load_ref):
    attempt = Path(attempt).resolve()
    names = ('request.json', 'response.raw.txt', 'generation.json', 'receipt.json')
    if any(not (attempt / name).is_file() for name in names):
        return None, {'status': 'not_salvaged', 'reason': 'missing_complete_original_model_four_artifacts'}
    refs = {name: q.identity(attempt / name) for name in names}
    receipt, request = q.checked(refs['receipt.json']), q.checked(refs['request.json'])
    files = receipt.get('files_sha256', {})
    if not {'request.json', 'response.raw.txt', 'generation.json', 'binding.json', 'final.json'} <= set(files):
        return None, {'status': 'not_salvaged', 'reason': 'incomplete_original_receipt', 'source_artifact_refs': refs}
    for name, expected in files.items():
        path = (attempt / name).resolve()
        if path.parent != attempt:
            raise q.SourceChanged('Original receipt artifact escapes attempt')
        q.verify_bytes({'path': str(path), 'sha256': expected})
        refs[name] = {'path': str(path), 'sha256': expected}
    expected_binding = {**binding, 'repair_revision': job['global_revision'],
                        'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames'])}
    if receipt.get('binding') != expected_binding or q.load(attempt / 'binding.json') != expected_binding:
        raise q.SourceChanged('Original model receipt source/window/actual frames binding differs')
    if receipt.get('status') != 'failed' or q.load(attempt / 'final.json').get('status') != 'failed':
        raise q.SourceChanged('Failed observation is not bound to an actual failed terminal transaction')
    provenance = request.get('repair_provenance', {})
    if (q.checked(provenance.get('job_ref', {})) != job or
            provenance.get('global_revision') != job['global_revision'] or
            provenance.get('automatic_reference_fill') is not False):
        raise q.SourceChanged('Original request checked job/revision binding differs')
    load = q.checked(load_ref)
    try:
        generation = bar.core.validate_generation(attempt, request)
    except ValueError as error:
        # SHA differences are never converted into an admitted candidate.
        generation = q.checked(refs['generation.json'])
        if (generation.get('raw_response_sha256') != refs['response.raw.txt']['sha256'] or
                generation.get('request_sha256') != refs['request.json']['sha256']):
            raise q.SourceChanged('Original generation raw/request SHA differs') from error
        return None, {'status': 'not_salvaged', 'reason': str(error), 'source_artifact_refs': refs}
    if generation.get('model_identity_sha256') != load['model_identity_sha256']:
        raise q.SourceChanged('Original model generation/load identity differs')
    try:
        value = bar.core.strict_json((attempt / 'response.raw.txt').read_bytes())
    except (ValueError, UnicodeError) as error:
        return None, {'status': 'not_salvaged', 'reason': 'complete_raw_not_strict_json: ' + str(error),
                      'source_artifact_refs': refs}
    for ref in refs.values():
        q.verify_bytes(ref)
    return value, {'status': 'complete_original_model_raw_verified', 'source_artifact_refs': refs,
                   'original_gpu_load_receipt_ref': load_ref,
                   'source_binding': expected_binding, 'actual_frame_SHA_verified': True}


def derive(terminal_ref, repair_out, revision, out=None, ffprobe='ffprobe'):
    safe_revision(revision)
    repair_out = Path(repair_out).resolve()
    terminal = q.checked(terminal_ref)
    key = q.safe_key({'id': terminal['id'], 'split': terminal['source_key'].split('/')[0]})
    if (terminal['source_key'] != key or
            Path(terminal_ref['path']).resolve() != repair_out / 'worker-terminals' / (key + '.json')):
        raise q.SourceChanged('Explicit worker terminal source/path binding differs')
    job_ref = terminal['original_job_ref']
    if Path(job_ref['path']).resolve() != repair_out / 'jobs' / (key + '.json'):
        raise q.SourceChanged('Worker original collector job path differs')
    collector = q.checked(job_ref)
    if collector['source_key'] != key or collector['global_revision'] != terminal['global_revision']:
        raise q.SourceChanged('Worker/collector source edition binding differs')
    if out is not None:
        out = isolated(out, repair_out, collector['production'])
        if revision == terminal['global_revision']:
            raise ValueError('Independent clause revision cannot impersonate the running repair edition')
    source_provenance = {'schema': SCHEMA, 'id': terminal['id'], 'source_key': key,
        'global_revision': revision, 'original_worker_terminal_ref': terminal_ref,
        'original_job_ref': job_ref, 'original_result_ref': collector['original_result_ref'],
        'original_repair_revision': terminal['global_revision'],
        'original_worker_status': terminal['status'], 'original_source_binding': collector['source_binding'],
        'independent_review_status': 'pending_independent_vlm_review',
        'risk_review_case': True, 'style_label_changed': False, **FLAGS}
    if not terminal.get('joint_result_ref'):
        skipped = {**source_provenance, 'status': terminal['status'],
                   'original_terminal': terminal, 'joint_complete': False,
                   'reason': 'no_complete_real_repair_result_available'}
        if out is not None:
            path = out / 'revisions' / revision / key / 'terminal.json'
            q.save_once(path, skipped)
            return q.identity(path), skipped
        return None, skipped
    repair_joint_ref = terminal['joint_result_ref']
    expected_joint = repair_out / 'joint/revisions' / terminal['global_revision'] / key / 'result.json'
    if Path(repair_joint_ref['path']).resolve() != expected_joint:
        raise q.SourceChanged('Worker joint result path/edition differs')
    repair_joint = q.checked(repair_joint_ref)
    repair_ref = repair_joint['repair_result_ref']
    repair_root = repair_out / 'revisions' / terminal['global_revision'] / key
    if Path(repair_ref['path']).resolve() != repair_root / 'result.json':
        raise q.SourceChanged('Real repair result path/edition differs')
    repair = q.checked(repair_ref)
    job = q.checked(repair['original_job_ref'])
    if (q.checked(q.identity(repair_root / 'original_job.json')) != job or
            job['source_key'] != key or q.safe_key(job) != key or
            Path(job['queue_root']).resolve() != repair_out or
            Path(repair['original_job_ref']['path']).resolve() != repair_out / 'dispatch-jobs' / (key + '.json') or
            job['global_revision'] != terminal['global_revision'] or
            repair['global_revision'] != terminal['global_revision'] or
            repair['original_source_binding'] != collector['source_binding'] or
            repair['original_result_ref'] != collector['original_result_ref'] or
            job['original_result_ref'] != collector['original_result_ref'] or
            repair_joint['original_job_ref'] != job_ref or
            repair_joint['original_source_binding'] != collector['source_binding']):
        raise q.SourceChanged('Real repair checked job/collector/result binding differs')
    if job.get('legacy_visual_demo') or job['id'] == watch.RAQ_ID:
        raise ValueError('Legacy/RAQ demo excluded from indexed clause salvage')
    q.verify_job(collector)
    q.verify_job(job)
    original = q.checked(job['refs']['visual_result'])
    original_joint = q.checked(job['refs']['joint_result'])
    plan = q.checked(job['refs']['plan_ref'])
    row = watch.checked_manifest(job, q.checked(job['refs']['producer_terminal']), q)
    binding = original['binding']
    if (bar.batch.digest(row) != binding['manifest_record_sha256'] or
            bar.batch.digest(plan) != binding['full_source_row_sha256'] or
            binding['plan_sha256'] != job['refs']['plan_ref']['sha256'] or
            binding['source_binding'] != job['source_binding'] or
            original_joint['source_binding'] != binding or original_joint['source_refs'] != row['source_refs']):
        raise q.SourceChanged('Original full manifest/plan/visual/joint SHA binding differs')
    prepared = q.verify_prepared_content(job['visual_root'], plan, bar, ffprobe)
    if prepared != q.checked(job['refs']['prepared']) or prepared['binding'] != binding:
        raise q.SourceChanged('Checked original prepared result binding differs')
    load_ref = q.identity(repair_out / 'gpu-runtime' / terminal['global_revision'] / 'load.receipt.json')
    load = q.checked(load_ref)
    if bar.code_identity() != load['code_sha256']:
        raise q.SourceChanged('Frozen bar code differs from actual GPU load receipt')
    for path, expected in load['code_sha256'].items():
        q.verify_bytes({'path': path, 'sha256': expected})
    windows = [m['window'] for m in prepared['windows']]
    observations = copy.deepcopy(repair['window_observations'])
    if (len(observations) != len(windows) or
            any(o['window'] != window for o, window in zip(observations, windows)) or
            repair['facts'] != bar.facts_from_observations(windows, observations)):
        raise q.SourceChanged('Real repair observations/window/facts binding differs')
    original_observations = original['window_observations']
    if len(original_observations) != len(windows):
        raise q.SourceChanged('Original observation window count differs')
    targeted = {f['window_id'] for f in job['failures'] if f['stage'] == 'reobserve'}
    for old, current in zip(original_observations, observations):
        if (old['status'] in ('candidate', 'abstain') and
                old['window']['window_id'] not in targeted and current != old):
            raise q.SourceChanged('Untargeted successful observation was modified by repair')
    budget_path = repair_out / 'attempt-budget' / (key + '.json')
    budget_ref = q.identity(budget_path)
    budget = q.checked(budget_ref)
    if (budget['source_key'] != key or budget['global_revision'] != terminal['global_revision'] or
            budget['original_result_sha256'] != job['original_result_sha256']):
        raise q.SourceChanged('Real repair budget binding differs')
    audit, evidence_refs = [], []
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    for i, (observation, manifest) in enumerate(zip(observations, prepared['windows'])):
        if observation['status'] != 'failed':
            continue
        attempt = repair_root / 'windows' / windows[i]['window_id']
        if not observation.get('attempt_path') or Path(observation['attempt_path']).resolve() != attempt:
            audit.append({'window_id': windows[i]['window_id'], 'status': observation['status'],
                          'reason': 'no_real_completed_repair_window_attempt'})
            continue
        raw, evidence = checked_attempt(attempt, job, manifest, binding, load_ref)
        evidence_refs.extend(evidence.get('source_artifact_refs', {}).values())
        if raw is None:
            audit.append({'window_id': windows[i]['window_id'], **evidence})
            continue
        try:
            admitted = salvage_clauses(raw, manifest, forbidden)
        except (ValueError, TypeError, KeyError) as error:
            audit.append({'window_id': windows[i]['window_id'], **evidence,
                          'status': 'not_salvaged', 'reason': 'original_filter_rejected_structure: ' + str(error)})
            continue
        observations[i] = {'status': 'abstain' if admitted['abstain'] else 'candidate', 'json': admitted,
            'window': windows[i], 'format_valid': True, 'offline_derivation': SCHEMA,
            'original_observation': copy.deepcopy(observation), 'source_evidence': evidence, **FLAGS}
        audit.append({'window_id': windows[i]['window_id'], **evidence,
            'status': observations[i]['status'], 'original_status': observation['status'],
            'admitted_event_count': len(admitted['events']),
            'scope_filter_audit': copy.deepcopy(admitted['scope_filter_audit']),
            'support_filter_audit': copy.deepcopy(admitted['support_filter_audit'])})
    facts = bar.facts_from_observations(windows, observations)
    for old in repair['facts']:
        if old not in facts:
            raise q.SourceChanged('An existing successful admitted fact changed or disappeared')
    # Never invent or strengthen an old relation across windows; captions cite no boundaries.
    boundaries = copy.deepcopy(repair['boundary_observations'])
    body = {'windows': [{'window_id': window['window_id'],
        'caption': ' '.join(f['raw_text'] for f in facts if f['window_id'] == window['window_id']),
        'used_fact_ids': [f['fact_id'] for f in facts if f['window_id'] == window['window_id']],
        'used_boundary_ids': []} for window in windows], 'issues': []}
    error = None
    try:
        bar.check_supported_rewrite(body, windows, facts, [])
    except ValueError as exc:
        error = str(exc)
    resolved = all(o['status'] in ('candidate', 'abstain') for o in observations)
    status = 'candidate' if resolved and error is None and facts else 'abstain' if resolved and error is None else 'failed'
    final = {'status': status, 'json': body, 'derivation': 'verbatim_admitted_fact_concatenation',
             'format_valid': error is None, 'semantic_validation_completed': False, **FLAGS}
    if error or not resolved:
        final['error'] = error or 'existing_unrecoverable_failed_or_pending_observation'
    full = status == 'candidate' and bool(windows) and all(w['caption'].strip() for w in body['windows'])
    new_facts = [f for f in facts if f not in repair['facts']]
    pure_rewrite = (not new_facts and resolved and bool(facts) and
                    repair['final']['status'] == 'failed' and
                    all(o['status'] in ('candidate', 'abstain') for o in repair['window_observations']))
    need_review = bool(new_facts or pure_rewrite)
    source_provenance.update({'need_real_review': need_review,
        'review_reason': 'new_safe_coarse_body_facts' if new_facts else
                         'pure_rewrite_failure_original_admitted_facts' if pure_rewrite else 'no_new_supported_facts',
        'pure_rewrite_failure': pure_rewrite,
        'new_safe_facts': copy.deepcopy(new_facts),
        'new_safe_fact_ids': [f['fact_id'] for f in new_facts],
        'original_repair_result_ref': repair_ref,
        'original_repair_joint_result_ref': repair_joint_ref,
        'original_visual_result_ref': job['refs']['visual_result'],
        'original_final': copy.deepcopy(repair['final']), 'original_attempt_budget_ref': budget_ref,
        'original_gpu_load_receipt_ref': load_ref, 'window_derivation_audit': audit,
        'admitted_fact_count': len(facts), 'salvaged_window_count': sum(a['status'] in ('candidate', 'abstain') for a in audit),
        'unresolved_window_ids': [o['window']['window_id'] for o in observations if o['status'] not in ('candidate', 'abstain')],
        'boundary_observations_preserved_without_new_inference': True,
        'offline_code_refs': {name: q.identity(path) for name, path in
            (('salvage', Path(__file__)), ('scope', Path(scope.__file__)), ('facts', Path(offline.__file__)),
             ('queue', Path(q.__file__)), ('pipeline', Path(pipeline.__file__)))}})
    visual = copy.deepcopy(original)
    visual.update(window_observations=observations, facts=facts, boundary_observations=boundaries,
                  final=final, status=status, global_revision=revision,
                  original_final=copy.deepcopy(repair['final']),
                  final_timed_caption=bar.core.timed_text({'windows': windows}, body['windows']),
                  coverage={'window_count': len(windows), 'window_status_counts': dict(Counter(o['status'] for o in observations)),
                            'admitted_fact_count': len(facts), 'full_window_caption': full,
                            'boundary_failure_count': sum(b['status'] == 'failed' for b in boundaries)})
    visual.update(source_provenance)
    style = copy.deepcopy(original_joint['style_result'])
    if job['refs'].get('style_result') and q.checked(job['refs']['style_result']) != style:
        raise q.SourceChanged('Original checked style differs')
    q.verify_job(job)
    q.checked(repair_ref)
    q.checked(terminal_ref)
    q.checked(budget_ref)
    for ref in evidence_refs:
        q.verify_bytes(ref)
    if out is not None:
        root = out / 'revisions' / revision / key
        q.save_once(root / 'visual_result.json', visual)
        visual['result_ref'] = q.identity(root / 'visual_result.json')
    joint = pipeline.joint_candidate(row, visual, style, original_joint.get('annotator_model_sha256'))
    joint.update(source_provenance)
    joint['status'] = 'clause_salvage_joint_candidate' if joint['joint_complete'] else 'clause_salvage_incomplete_candidate'
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in original_joint:
            joint[name] = copy.deepcopy(original_joint[name])
    if out is not None:
        q.save_once(root / 'result.json', joint)
        return q.identity(root / 'result.json'), joint
    return None, joint


def manifest_row(ref, result):
    return {'schema': SCHEMA, 'id': result['id'], 'source_key': result['source_key'],
        'split': result['split'], 'batch_id': result['batch_id'],
        'global_revision': result['global_revision'], 'status': result['status'],
        'original_job_ref': result['original_job_ref'],
        'original_worker_terminal_ref': result['original_worker_terminal_ref'],
        'result_ref': ref, 'joint_candidate_ref': ref,
        'visual_result_ref': result['visual_result_ref'], 'joint_complete': result['joint_complete'],
        'need_real_review': result['need_real_review'], 'review_reason': result['review_reason'],
        'pure_rewrite_failure': result['pure_rewrite_failure'],
        'new_safe_facts': copy.deepcopy(result['new_safe_facts']),
        'new_safe_fact_ids': result['new_safe_fact_ids'],
        'new_safe_fact_refs': [{'visual_result_ref': result['visual_result_ref'], 'fact_id': fid}
                              for fid in result['new_safe_fact_ids']],
        'review_interface': 'explicit_independent_manifest_v1',
        'independent_review_status': 'pending_independent_vlm_review',
        'risk_review_case': True, **FLAGS}


def append_checkpointed(path, value, offset):
    data = q.encoded(value)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if size < offset:
            raise q.SourceChanged('Independent review manifest truncated')
        if size > offset:
            stream.seek(offset)
            if stream.read() != data:
                raise q.SourceChanged('Independent review manifest uncheckpointed bytes differ')
        else:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    return offset + len(data)


def consume_once(repair_out, out, revision, limit=0, source_keys=(), ffprobe='ffprobe'):
    safe_revision(revision)
    repair_out, out = Path(repair_out).resolve(), isolated(out, repair_out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'consumer.state.json'
    with (out / 'consumer.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = q.load(state_path) if state_path.exists() else {
            'schema': SCHEMA, 'repair_root': str(repair_out), 'global_revision': revision,
            'seen': {}, 'counts': {}, 'review_manifest_offset': 0, **FLAGS}
        if state['repair_root'] != str(repair_out) or state['global_revision'] != revision:
            raise q.SourceChanged('Independent consumer source/revision differs')
        worker_path = repair_out / 'worker.state.json'
        if not worker_path.is_file():
            return state, 0
        indexed = q.load(worker_path).get('seen', {})
        keys = list(source_keys) if source_keys else list(indexed)
        added = 0
        for key in keys:
            if key not in indexed:
                continue
            if not re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key):
                raise q.SourceChanged('Unsafe source key in explicit terminal index')
            terminal_path = repair_out / 'worker-terminals' / (key + '.json')
            terminal_ref = q.identity(terminal_path)
            terminal = q.checked(terminal_ref)
            if terminal['original_job_ref']['sha256'] != indexed[key]:
                raise q.SourceChanged('Worker seen job SHA differs from real terminal')
            if key in state['seen']:
                if state['seen'][key]['terminal_ref'] != terminal_ref:
                    raise q.SourceChanged('Already-derived source terminal changed')
                continue
            if limit and added >= limit:
                break
            ref, result = derive(terminal_ref, repair_out, revision, out, ffprobe)
            if result.get('visual_result_ref'):
                state['review_manifest_offset'] = append_checkpointed(
                    out / 'review_required.jsonl', manifest_row(ref, result), state['review_manifest_offset'])
            state['seen'][key] = {'terminal_ref': terminal_ref, 'result_ref': ref, 'status': result['status']}
            state['counts'][result['status']] = state['counts'].get(result['status'], 0) + 1
            q.replace_state(state_path, state)
            added += 1
            print(json.dumps({'source_key': key, 'status': result['status'], 'joint_complete': result['joint_complete'],
                              'salvaged_window_count': result.get('salvaged_window_count', 0), 'result_ref': ref,
                              'model_calls': 0, 'attempt_budget_consumed': False}, ensure_ascii=False), flush=True)
        if not state_path.exists():
            q.replace_state(state_path, state)
        return state, added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-out', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--global-revision', default=DEFAULT_REVISION)
    parser.add_argument('--source-key', action='append', default=[])
    parser.add_argument('--limit-per-poll', type=int, default=5)
    parser.add_argument('--max-sources', type=int, default=0)
    parser.add_argument('--ffprobe', default='ffprobe')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=20)
    args = parser.parse_args()
    if args.limit_per_poll < 0 or args.max_sources < 0 or args.poll_seconds <= 0:
        parser.error('nonnegative bounds and positive poll-seconds required')
    total = 0
    while True:
        limit = args.limit_per_poll
        if args.max_sources:
            remaining = args.max_sources - total
            if remaining <= 0:
                break
            limit = min(limit, remaining) if limit else remaining
        state, added = consume_once(args.repair_out, args.out, args.global_revision, limit, args.source_key, args.ffprobe)
        total += added
        print(json.dumps({'new_sources': added, 'total_derived': len(state['seen']), 'counts': state['counts'],
                          'review_manifest': str(Path(args.out).resolve() / 'review_required.jsonl'), **FLAGS}), flush=True)
        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
