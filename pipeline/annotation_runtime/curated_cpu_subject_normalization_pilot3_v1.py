"""Exactly three zero-model subject-only alternatives; original raw facts remain unchanged."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

RT = Path(__file__).resolve().parent
BASE = RT / 'curated_rewrite_only_cpu_batch_v1.py'
BASE_SHA = '0f8c684daa401c8a9a7deac3f1b6f86c166c45a8cd9a610a5ca8b1b10fe5290f'
OUT = RT.parent / 'curated_cpu_subject_normalization_pilot3_v1'
IDS = ('ZgWjGkWgXwKWZgxD', 'qjqnEQREEGkWSpWk', 'nIDvZTxoDtTHBdMO')
REVISION = 'rewrite-only-neutral-subject-20261025-v1'


def subject_text(text):
    text = re.sub(r'<s/he>|\b(?:he|she)\b', 'the dancer', text, flags=re.I)
    text = re.sub(r'\b(?:his|her|hers)\b', "the dancer's", text, flags=re.I)
    return re.sub(r'\bhim\b', 'the dancer', text, flags=re.I)


def run():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Frozen original zero-model source validator changed')
    spec = importlib.util.spec_from_file_location('cpu_subject_only_private_replay', BASE)
    base = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(BASE), 'exec'), base.__dict__)
    base.REVISION = REVISION
    original_concat, original_candidate = base.off.concatenate_facts, base.off.candidate

    def concatenate(windows, facts):
        for fact in facts:
            normalized = subject_text(fact['raw_text'])
            if normalized != fact['text'] or fact['raw_text'] != fact['source_clause']['raw_source_clause']:
                raise ValueError('Only existing exact subject-normalized text with unchanged raw evidence is allowed')
        body = {'windows': [], 'issues': []}
        for window in windows:
            local = [f for f in facts if f['window_id'] == window['window_id']]
            body['windows'].append({'window_id': window['window_id'],
                'caption': ' '.join(subject_text(f['raw_text']) for f in local),
                'used_fact_ids': [f['fact_id'] for f in local], 'used_boundary_ids': []})
        base.bar.check_supported_rewrite(body, windows, facts, [])
        return body

    def candidate(job_ref, revision):
        result = original_candidate(job_ref, revision)
        job, row, style, visual, provenance, old_joint = result
        visual['final']['derivation'] = 'existing_native_subject_only_normalization_of_admitted_facts'
        provenance.update(subject_only_normalization=True, raw_text_and_source_clause_unchanged=True,
            new_motion_or_timing_claims_added=False,
            subject_normalization_execution_ref=base.q.identity(__file__))
        visual.update(provenance)
        return job, row, style, visual, provenance, old_joint

    base.off.concatenate_facts, base.off.candidate = concatenate, candidate
    records = []
    try:
        for sid in IDS:
            job_ref = base.q.identity(base.QUEUE / 'jobs/train' / (sid + '.json'))
            job = base.q.checked(job_ref)
            before = base.q.checked(job['refs']['visual_result'])
            budget = base.QUEUE / 'attempt-budget/train' / (sid + '.json')
            budget_before = base.q.identity(budget) if budget.exists() else None
            records.append(base.derive(job_ref, OUT))
            candidate_path = Path(records[-1]['result_ref']['path'])
            value = base.q.checked(records[-1]['result_ref'])
            visual = base.q.checked(records[-1]['visual_result_ref'])
            if (visual['facts'] != before['facts'] or visual['window_observations'] != before['window_observations']
                    or visual['original_final'] != before['final']
                    or (base.q.identity(budget) if budget.exists() else None) != budget_before):
                raise ValueError('Original raw facts/observations/failed-final/budget changed')
            if not value['joint_complete'] or value['style_result'] != base.q.checked(job['refs']['joint_result'])['style_result']:
                raise ValueError('Complete candidate or original unchanged style binding differs')
    finally:
        base.off.concatenate_facts, base.off.candidate = original_concat, original_candidate
    summary = {'schema': 'cpu_subject_normalization_pilot3_v1', 'source_count': 3, 'candidates': records,
               'model_calls': 0, 'new_budget_reservations': 0, 'subject_only_normalization': True,
               'original_motion_raw_text_and_source_refs_preserved': True,
               'independent_review_required': True, **base.q.FLAGS}
    base.q.save_once(OUT / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False), flush=True)
