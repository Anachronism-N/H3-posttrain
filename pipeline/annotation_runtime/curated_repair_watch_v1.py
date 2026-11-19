"""Resident bounded visual repair; joint revisions never replace production terminals."""
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

SCHEMA = 'curated_repair_watch_v1'
RAQ_ID = 'RAQoxqRxCfMvTHSx'


def technical_window(diagnostic):
    if diagnostic.get('reason') == 'no_model_attempt_or_insufficient_actual_frames':
        return False
    if diagnostic.get('body_status') in ('absent', 'empty', 'malformed'):
        return True
    if any(d.get('technical_reference_mismatch_possible') for d in diagnostic.get('reference_deficits', [])):
        return True
    return diagnostic.get('status') == 'failed' and not diagnostic.get('reference_deficits')


def dispatch_job(job, original_ref):
    if job.get('legacy_visual_demo') or job['id'] == RAQ_ID or 'media' in job['stages']:
        return None
    diagnostics = {d['window_id']: d for d in job['window_diagnostics']}
    failures = [f for f in job['failures'] if f['stage'] == 'rewrite' or
                f['stage'] == 'reobserve' and technical_window(diagnostics[f['window_id']])]
    if not failures:
        return None
    result = copy.deepcopy(job)
    result['collector_job_ref'] = original_ref
    result['failures'] = failures
    result['stages'] = [s for s in ('reobserve', 'rewrite', 'styleunknown')
                        if s == 'styleunknown' and s in job['stages'] or any(f['stage'] == s for f in failures)]
    result['stage'] = result['stages'][0]
    result['maximum_reobserve_requests'] = sum(f['stage'] == 'reobserve' for f in failures)
    result['maximum_rewrite_requests'] = 1
    result['untargeted_evidence_insufficiency_not_repaired'] = [f for f in job['failures']
        if f['stage'] == 'reobserve' and f not in failures]
    return result


def package_joint(job, collector_ref, repair, out, revision, q, pipeline, bar):
    original_joint = q.checked(job['refs']['joint_result'])
    original_visual = q.checked(job['refs']['visual_result'])
    producer = q.checked(job['refs']['producer_terminal'])
    style = copy.deepcopy(original_joint['style_result'])
    if job['refs'].get('style_result') and q.checked(job['refs']['style_result']) != style:
        raise q.SourceChanged('Original joint/style reference mismatch')
    if repair['id'] != job['id'] or repair['original_result_ref'] != job['original_result_ref']:
        raise q.SourceChanged('Repair original result binding differs')
    windows = [o['window'] for o in repair['window_observations']]
    visual = copy.deepcopy(original_visual)
    visual.update({k: copy.deepcopy(repair[k]) for k in
                   ('window_observations', 'facts', 'boundary_observations', 'final')})
    visual['status'] = repair['final']['status']
    visual['final_timed_caption'] = bar.core.timed_text({'windows': windows},
        repair['final']['json']['windows']) if 'json' in repair['final'] else ''
    visual['coverage'] = {
        'window_count': len(windows),
        'window_status_counts': dict(Counter(o['status'] for o in repair['window_observations'])),
        'admitted_fact_count': len(repair['facts']),
        'full_window_caption': bool(repair['all_windows_nonempty']),
        'boundary_failure_count': sum(b['status'] == 'failed' for b in repair['boundary_observations'])}
    visual['original_repair_result_ref'] = q.identity(out / 'revisions' / revision / job['source_key'] / 'result.json')
    root = out / 'joint' / 'revisions' / revision / job['source_key']
    q.save_once(root / 'visual_result.json', visual)
    visual['result_ref'] = q.identity(root / 'visual_result.json')
    joint = pipeline.joint_candidate(producer['manifest_record'], visual, style,
                                     original_joint.get('annotator_model_sha256'))
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in original_joint:
            joint[name] = copy.deepcopy(original_joint[name])
    joint.update({'schema': SCHEMA, 'status': 'joint_repair_candidate' if joint['joint_complete'] else 'incomplete_repair_candidate',
        'global_revision': revision, 'original_job_ref': collector_ref,
        'original_result_ref': job['original_result_ref'],
        'original_visual_result_ref': job['refs']['visual_result'],
        'repair_result_ref': visual['original_repair_result_ref'],
        'original_source_binding': job['source_binding'],
        'extra_source_attempts': 1, 'maximum_extra_source_attempts': 1,
        'style_label_changed': False, 'repair_completed': False,
        'original_raw_or_terminal_modified': False,
        'independent_review_status': 'pending_independent_vlm_review',
        'risk_review_case': True, 'unresolved_stages': ['styleunknown'] if 'styleunknown' in job['stages'] else [],
        **q.FLAGS})
    q.verify_job(job)
    q.save_once(root / 'result.json', joint)
    return q.identity(root / 'result.json'), joint


def announce(out, key, terminal, q):
    notice = out / 'worker-announced' / (key + '.json')
    if notice.exists():
        if q.load(notice) != terminal:
            raise q.SourceChanged('Worker announcement changed')
        return
    if terminal.get('joint_result_ref'):
        (out / 'joint').mkdir(parents=True, exist_ok=True)
        q.append(out / 'joint' / 'review-required.jsonl', {
            'schema': SCHEMA, 'id': terminal['id'], 'source_key': key,
            'batch_id': terminal['batch_id'], 'global_revision': terminal['global_revision'],
            'result_ref': terminal['joint_result_ref'], 'joint_result_ref': terminal['joint_result_ref'],
            'original_job_ref': terminal['original_job_ref'],
            'independent_review_status': 'pending_independent_vlm_review',
            'risk_review_case': True, 'repair_completed': False, **q.FLAGS})
    else:
        q.append(out / 'deferred.jsonl', terminal)
    q.save_once(notice, terminal)


def consume_once(out, revision, q, pipeline, bar, runtime, worker, limit=0):
    out = Path(out).resolve()
    stream_path = out / 'ready.jsonl'
    state_path = out / 'worker.state.json'
    state = q.load(state_path) if state_path.exists() else {
        'schema': SCHEMA, 'input_path': str(stream_path), 'offset': 0, 'seen': {}, 'counts': {}}
    if not stream_path.exists():
        return state, 0
    st = stream_path.stat()
    identity = [st.st_dev, st.st_ino]
    if state['input_path'] != str(stream_path) or st.st_size < state['offset'] or (
            state.get('input_identity', identity) != identity):
        raise q.SourceChanged('Ready stream replaced/truncated')
    state['input_identity'] = identity
    processed = 0
    with stream_path.open('rb') as stream:
        stream.seek(state['offset'])
        while not limit or processed < limit:
            begin = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                break
            hint = json.loads(line)
            key = hint['source_key']
            if not re.fullmatch(r'(?:train|val|test)/[A-Za-z0-9][A-Za-z0-9_-]*', key):
                raise ValueError('Unsafe ready source key')
            job_ref = hint['job_ref']
            if Path(job_ref['path']).resolve() != out / 'jobs' / (key + '.json'):
                raise ValueError('Ready job path differs from queue-local source key')
            job = q.checked(job_ref)
            if q.safe_key(job) != key or Path(job['queue_root']).resolve() != out:
                raise ValueError('Ready job key/queue binding differs')
            previous = state['seen'].get(key)
            if previous and previous != job_ref['sha256']:
                raise q.SourceChanged('Ready source original job changed')
            terminal_path = out / 'worker-terminals' / (key + '.json')
            if not previous:
                if terminal_path.exists():
                    terminal = q.load(terminal_path)
                    if terminal['original_job_ref'] != job_ref:
                        raise q.SourceChanged('Resumed worker terminal job differs')
                else:
                    selected = dispatch_job(job, job_ref)
                    terminal = {'schema': SCHEMA, 'id': job['id'], 'source_key': key,
                        'batch_id': job.get('batch_id'), 'original_job_ref': job_ref,
                        'global_revision': revision, 'repair_completed': False, **q.FLAGS}
                    budget = out / 'attempt-budget' / (key + '.json')
                    if selected is None:
                        terminal.update(status='deferred', reason='excluded_legacy_RAQ_or_media_style_or_evidence_insufficiency',
                                        unresolved_stages=job['stages'])
                    elif budget.exists() and q.load(budget)['original_result_sha256'] != job['original_result_sha256']:
                        raise q.SourceChanged('Reserved source original SHA changed')
                    elif budget.exists() and q.load(budget)['global_revision'] != revision:
                        terminal.update(status='budget_already_reserved', reason='no_redispatch_across_revisions')
                    else:
                        dispatch_path = out / 'dispatch-jobs' / (key + '.json')
                        q.save_once(dispatch_path, selected)
                        result_path = out / 'revisions' / revision / key / 'result.json'
                        if result_path.exists():
                            repair = q.load(result_path)
                        elif budget.exists():
                            terminal.update(status='interrupted_attempt', reason='budget_reserved_without_final_result_no_redispatch')
                            repair = None
                        else:
                            repair = q.repair_visual(str(dispatch_path), out, revision, bar, runtime, worker)
                        if repair is not None:
                            joint_ref, joint = package_joint(selected, job_ref, repair, out, revision, q, pipeline, bar)
                            terminal.update(status=joint['status'], joint_complete=joint['joint_complete'],
                                            joint_result_ref=joint_ref, extra_source_attempts=1)
                            repair_root = out / 'revisions' / revision / key
                            if any(repair_root.rglob('dispatch_error.json')):
                                terminal['shared_dispatch_failure'] = True
                    q.save_once(terminal_path, terminal)
                announce(out, key, terminal, q)
                state['seen'][key] = job_ref['sha256']
                status = terminal['status']
                state['counts'][status] = state['counts'].get(status, 0) + 1
                processed += 1
                print(json.dumps({'source_key': key, 'status': status,
                                  'joint_result_ref': terminal.get('joint_result_ref')}, ensure_ascii=False), flush=True)
            state['offset'] = stream.tell()
            q.replace_state(state_path, state)
            if not previous and terminal.get('shared_dispatch_failure'):
                raise RuntimeError('Generation dispatch failed; stop before consuming another source budget')
    if not state_path.exists():
        q.replace_state(state_path, state)
    return state, processed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--global-revision', required=True)
    parser.add_argument('--repair-source', required=True)
    parser.add_argument('--ms-source', required=True)
    parser.add_argument('--consumer-config', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-jobs', type=int, default=0)
    parser.add_argument('--total-source-count', type=int, default=0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.max_jobs < 0 or args.total_source_count < 0:
        parser.error('positive poll-seconds and nonnegative job/source bounds required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.global_revision):
        parser.error('unsafe revision')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('frozen_curated_repair', Path(args.repair_source).resolve())
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    sys.path.insert(0, str(Path(args.ms_source).resolve()))
    import curated_pipeline as pipeline
    import ms_bar_annotation as bar
    if Path(bar.__file__).resolve().parent != Path(args.ms_source).resolve():
        parser.error('imported bar differs from explicitly frozen source')
    config = q.load(args.consumer_config)
    worker = pipeline.parser().parse_args(config['argv'][4:])
    if worker.stage != 'run':
        parser.error('consumer config must select run stage')
    worker.retry_failed = False
    worker.rewrite_reference_repair = 'off'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime_root = out / 'gpu-runtime' / args.global_revision
    runtime_root.mkdir(parents=True, exist_ok=True)
    with (out / 'resident-worker.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = bar.runtime_loader(worker, runtime_root, bar.code_identity())
        processed = 0
        while True:
            state, count = consume_once(out, args.global_revision, q, pipeline, bar, runtime, worker,
                                        max(0, args.max_jobs - processed) if args.max_jobs else 0)
            processed += count
            collector = out / 'collector.state.json'
            complete = False
            if args.total_source_count and collector.exists():
                sources_seen = len(q.load(collector)['seen'])
                if sources_seen > args.total_source_count:
                    raise q.SourceChanged('Collector exceeds frozen full source count')
                ready = out / 'ready.jsonl'
                complete = sources_seen == args.total_source_count and (not ready.exists() or state['offset'] == ready.stat().st_size)
            summary = {'schema': SCHEMA, 'global_revision': args.global_revision,
                'source_count': len(state['seen']), 'counts': state['counts'],
                'queue_drained': complete, 'runtime_reused_across_sources': True,
                'joint_review_required_path': str(out / 'joint' / 'review-required.jsonl'),
                'visual_only_diagnostic_path': str(out / 'review-required.jsonl'), **q.FLAGS}
            q.replace_state(out / 'resident.summary.json', summary)
            if not args.watch or complete or args.max_jobs and processed >= args.max_jobs:
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
