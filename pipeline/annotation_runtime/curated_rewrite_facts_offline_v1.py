"""Opt-in offline candidates: concatenate checked admitted facts verbatim, never dispatch VLM."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import sys

RT = Path(__file__).resolve().parent
SCHEMA = 'curated_rewrite_facts_offline_v1'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


q = load_module('offline_facts_q', RT / 'source_repair_gpu_v3/curated_repair_queue_v1.py')
w = load_module('offline_facts_watch', RT / 'curated_repair_watch_v2.py')
sys.dont_write_bytecode = True
sys.path.insert(0, str(RT / 'source_curated_consumer_v1'))
import curated_pipeline as pipeline
import ms_bar_annotation as bar


def concatenate_facts(windows, facts):
    rows = []
    for window in windows:
        local = [fact for fact in facts if fact['window_id'] == window['window_id']]
        rows.append({'window_id': window['window_id'],
                     'caption': ' '.join(fact['raw_text'] for fact in local),
                     'used_fact_ids': [fact['fact_id'] for fact in local], 'used_boundary_ids': []})
    result = {'windows': rows, 'issues': []}
    bar.check_supported_rewrite(result, windows, facts, [])
    return result


def candidate(job_ref, revision):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', revision):
        raise ValueError('Unsafe offline revision')
    job = q.checked(job_ref)
    key = q.safe_key(job)
    if (job['source_key'] != key or Path(job_ref['path']).resolve() !=
            Path(job['queue_root']).resolve() / 'jobs' / (key + '.json')):
        raise q.SourceChanged('Offline job key/path binding differs')
    if (job.get('legacy_visual_demo') or job['id'] == w.RAQ_ID or
            'rewrite' not in job['stages'] or set(job['stages']) - {'rewrite', 'styleunknown'} or
            any(f['stage'] != 'rewrite' for f in job['failures'])):
        raise ValueError('Offline mode only admits rewrite-only failures; reobserve/media/RAQ excluded')
    q.verify_job(job)
    original = q.checked(job['refs']['visual_result'])
    original_joint = q.checked(job['refs']['joint_result'])
    producer = q.checked(job['refs']['producer_terminal'])
    row = w.checked_manifest(job, producer, q)
    plan = q.checked(job['refs']['plan_ref'])
    binding = original['binding']
    if (bar.batch.digest(row) != binding['manifest_record_sha256'] or
            bar.batch.digest(plan) != binding['full_source_row_sha256'] or
            binding['plan_sha256'] != job['refs']['plan_ref']['sha256'] or
            binding['source_binding'] != job['source_binding'] or
            original_joint['source_binding'] != binding or original_joint['source_refs'] != row['source_refs']):
        raise q.SourceChanged('Offline checked original source SHA/refs binding differs')
    if original['final']['status'] != 'failed':
        raise ValueError('Offline mode requires an actual immutable failed final')
    prepared = q.checked(job['refs']['prepared'])
    if prepared['binding'] != binding:
        raise q.SourceChanged('Offline prepared/original visual binding differs')
    windows = [manifest['window'] for manifest in prepared['windows']]
    observations = original['window_observations']
    if (len(observations) != len(windows) or
            any(o['window'] != window or o['status'] not in ('candidate', 'abstain')
                for o, window in zip(observations, windows))):
        raise ValueError('Offline mode requires all existing observations terminal and admitted')
    facts = original['facts']
    if facts != bar.facts_from_observations(windows, observations):
        raise q.SourceChanged('Original facts differ from checked admitted observations')
    body = concatenate_facts(windows, facts)
    visual = copy.deepcopy(original)
    visual['original_final'] = copy.deepcopy(original['final'])
    visual['final'] = {'status': 'candidate', 'json': body, 'model_rewrite_performed': False,
                       'derivation': 'verbatim_admitted_fact_concatenation',
                       'format_valid': True, 'semantic_validation_completed': False, **q.FLAGS}
    visual['status'] = 'candidate'
    visual['final_timed_caption'] = bar.core.timed_text({'windows': windows}, body['windows'])
    full = bool(windows) and all(item['caption'].strip() for item in body['windows'])
    visual['coverage'] = {**original.get('coverage', {}), 'full_window_caption': full}
    style = copy.deepcopy(original_joint['style_result'])
    if job['refs'].get('style_result') and q.checked(job['refs']['style_result']) != style:
        raise q.SourceChanged('Offline original style reference differs')
    provenance = {'schema': SCHEMA, 'id': job['id'], 'source_key': key, 'global_revision': revision,
                  'original_job_ref': job_ref, 'original_result_ref': job['original_result_ref'],
                  'original_visual_result_ref': job['refs']['visual_result'],
                  'original_source_binding': job['source_binding'], 'original_final': copy.deepcopy(original['final']),
                  'model_rewrite_performed': False, 'VLM_dispatch_performed': False,
                  'extra_source_attempts': 0, 'attempt_budget_consumed': False,
                  'repair_completed': False, 'independent_review_status': 'pending_independent_vlm_review',
                  'risk_review_case': True, 'style_label_changed': False,
                  'original_raw_or_terminal_modified': False, **q.FLAGS}
    return job, row, style, visual, provenance, original_joint


def derive(job_ref, revision, out=None):
    job, row, style, visual, provenance, original_joint = candidate(job_ref, revision)
    if out is not None:
        out = Path(out).resolve()
        for source in (Path(job['production']).resolve(), Path(job['queue_root']).resolve()):
            if out == source or out.is_relative_to(source) or source.is_relative_to(out):
                raise ValueError('Offline output must be separately isolated from production and repair queue')
        root = out / 'revisions' / revision / job['source_key']
        q.save_once(root / 'visual_result.json', visual)
        visual['result_ref'] = q.identity(root / 'visual_result.json')
    joint = pipeline.joint_candidate(row, visual, style, original_joint.get('annotator_model_sha256'))
    joint.update(provenance)
    joint['status'] = 'offline_joint_candidate' if joint['joint_complete'] else 'offline_incomplete_candidate'
    for name in ('source_identity', 'actual_video_probe', 'style_result_ref'):
        if name in original_joint:
            joint[name] = copy.deepcopy(original_joint[name])
    q.verify_job(job)
    if out is not None:
        q.save_once(root / 'result.json', joint)
        return q.identity(root / 'result.json'), joint
    return None, joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--job-sha256', required=True)
    parser.add_argument('--global-revision', required=True)
    parser.add_argument('--out', help='Separate opt-in output; omit for memory-only inspection')
    args = parser.parse_args()
    ref, joint = derive({'path': str(Path(args.job).resolve()), 'sha256': args.job_sha256},
                        args.global_revision, args.out)
    print(json.dumps({'result_ref': ref, 'candidate': joint}, ensure_ascii=False))


if __name__ == '__main__':
    main()
