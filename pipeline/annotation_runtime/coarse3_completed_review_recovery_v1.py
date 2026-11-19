"""Recover three completed review transactions without model loading or dispatch."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

RT = Path(__file__).resolve().parent
OUT = RT.parent / 'coarse3_completed_review_recovery_v1'
PINS = {
    'coarse3_independent_review_v1.py': 'd1a924d5fbac924f37fcb1a149e5e1929c37419009a4b7e73d691a49e5009f1d',
    'coarse3_review_mount_bridge_v1.py': 'd30adeeeaf38c16900d61a59f0a95a552945036b800fbca5a83af3cfae4300be',
}
PREVIOUS_SHA = '8d07eb22484fdf70e183bb1466a6c65b01fa6248a7ddabea8765523601a97438'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(name, path):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == PINS[Path(path).name], 'Frozen recovery dependency changed')
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), m.__dict__)
    return m


def run():
    m = load_module('completed_three_frozen_review', RT / 'coarse3_independent_review_v1.py')
    bridge = load_module('completed_three_content_mount_bridge', RT / 'coarse3_review_mount_bridge_v1.py')
    lib = m.libraries()
    q, batch = lib.q, lib.batch
    launch = q.checked({'path': str(RT / 'coarse3_n6_launch_zero_dispatch_resume_main_v1.json'), 'sha256': PREVIOUS_SHA})
    plan = q.checked(launch['plan'])
    previous_summary_ref = q.identity(m.OUT / 'independent.summary.json')
    previous_ledger_ref = q.identity(m.OUT / 'independent.ledger.jsonl')
    previous_stage_ref = q.identity(m.OUT / 'finite.stage.summary.json')
    summary = q.checked(previous_summary_ref)
    pending = [json.loads(line) for line in Path(previous_ledger_ref['path']).read_bytes().splitlines() if line.strip()]
    need(summary['review_dispatches'] == 3 and summary['statuses'] == {'pending': 3}
         and len(pending) == 3 and [r['source_id'] for r in pending] == list(m.IDS)
         and all(r['review_dispatches'] == 1 and r['status'] == 'pending'
                 and r['error'].startswith('ValueError: source asset changed: ') for r in pending),
         'Only exact three already-dispatched post-generation asset failures can be recovered')
    model_ref = q.identity(m.OUT / 'reviewer.model.identity.json')
    code_ref = q.identity(m.OUT / 'reviewer.code.identity.json')
    model, code = q.checked(model_ref), q.checked(code_ref)
    stage = q.checked(launch['modelref'])
    need(model['sha256'] == stage['model_identity']['sha256'] == launch['modelref']['model_identity_sha256'],
         'Actual reviewer checkpoint differs from staged admission')
    need(model['root'] == stage['model_identity']['root']
         and {k:v['sha256'] for k,v in model['files'].items()}
         == {k:v['sha256'] for k,v in stage['model_identity']['files'].items()},
         'Actual reviewer full weight-content inventory changed')
    content = {k:{'path':v['path'], 'sha256':v['sha256']} for k,v in code['files'].items()}
    need(batch.digest({'root': code['root'], 'files': content}) == code['sha256'], 'Recorded execution code digest differs')
    code_pairs = []
    for value in code['files'].values():
        current = batch.file_identity(Path(value['path']))
        _, pairs = bridge.bridge_references([value], [current])
        code_pairs.extend(pairs)
    teacher_ref = q.identity(m.TEACHER)
    teacher, original_teacher_reference = lib.audit.read_model_reference(m.TEACHER, permit_attested=True)
    need(original_teacher_reference['path'] == teacher_ref['path']
         and original_teacher_reference['sha256'] == teacher_ref['sha256'],
         'Original attested teacher reference content changed')
    reviewer = SimpleNamespace(model=model['root'], code=code, identity=lambda: model,
        _annotator_identity=teacher, _annotator_reference_ref=original_teacher_reference)
    original_teacher = q.checked({'path': str(RT / 'curated_coarse_empty_n5g01_launch_v1.json'),
        'sha256': '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806'})
    manifest = q.checked(original_teacher['jobs_manifest_ref'])
    population_ref = q.identity(m.OUT / 'frozen.population.json')
    results = []
    for source, notice, previous in zip(plan['sources'], manifest['jobs'], pending):
        candidate, verified = m.validate_candidate(lib, original_teacher, notice, ffprobe='/opt/ms_tools/ffprobe')
        row = m.review_row(lib, candidate, verified, population_ref)
        root = Path(source['root'])
        need(root == m.OUT / 'sources' / ('train-' + row['id']) / batch.digest(row)[:24],
             'Original runtime input-row serialization/source root changed')
        evidence = q.load(root / 'evidence.json')
        m.verify_evidence(lib, evidence, row, root, verified['probe'])
        asset_pairs = []
        with lib.review_library():
            for value in evidence['source_identity']['files'].values():
                if value is None:
                    continue
                current = batch.file_identity(Path(value['path']))
                _, pairs = bridge.bridge_references([value], [current])
                asset_pairs.extend(pairs)
            for value in evidence['binding']['source_refs']:
                current = batch.file_identity(Path(value['path']))
                _, pairs = bridge.bridge_references([value], [current])
                asset_pairs.extend(pairs)
            claim_ref = q.identity(m.OUT / 'review-attempts' / (row['split'] + '-' + row['id'] + '.json'))
            claim = q.checked(claim_ref)
            need(claim['source_id'] == row['id'] and claim['input_row_sha256'] == batch.digest(row)
                 and claim['root'] == str(root) and claim['maximum_review_calls_per_source'] == 1
                 and claim['maximum_total_review_calls'] == 3, 'Original one-call claim differs')
            value, receipt_ref = lib.old.verify_review_receipt(root, row, evidence, reviewer)
            need(value['status'] in ('pass', 'confirmed_error', 'uncertain'), 'No complete original review result')
        record = {**copy.deepcopy(previous), **value, 'review_receipt': receipt_ref,
            'review_dispatches': 1, 'actual_VLM_calls': 1, 'recovery_model_calls': 0,
            'recovery_derivation': 'completed_transaction_native_replay_content_verified_mount_device_only',
            'original_pending_record_preserved': True, 'media_and_reference_mount_pairs': asset_pairs,
            'original_summary_ref': previous_summary_ref, 'original_ledger_ref': previous_ledger_ref,
            'original_stage_ref': previous_stage_ref, 'original_call_claim_ref': claim_ref,
            'actual_reviewer_model_ref': model_ref, 'actual_reviewer_code_ref': code_ref,
            'recovery_execution_ref': q.identity(__file__), **m.NO_APPROVAL}
        record.pop('error', None)
        q.save_once(OUT / 'sources' / (row['id'] + '.json'), record)
        results.append(record)
    need(q.identity(m.OUT / 'independent.summary.json') == previous_summary_ref
         and q.identity(m.OUT / 'independent.ledger.jsonl') == previous_ledger_ref,
         'Original diagnostic results changed during recovery')
    report = m.aggregate(results, prepare_only=False, population_ref=population_ref,
                         reviewer=SimpleNamespace(_loaded=True))
    report.update(schema='coarse3_completed_review_recovery_v1', recovery_model_calls=0,
        original_model_calls=3, original_diagnostic_results_preserved=True,
        original_summary_ref=previous_summary_ref, original_ledger_ref=previous_ledger_ref,
        original_stage_ref=previous_stage_ref, actual_reviewer_model_ref=model_ref,
        actual_reviewer_code_ref=code_ref, code_content_mount_pairs=code_pairs,
        completed_original_transaction_refs=[r['review_receipt'] for r in results])
    q.save_once(OUT / 'summary.json', report)
    q.save_once(OUT / 'review.records.json', results)
    return {k:v for k,v in report.items() if k != 'code_content_mount_pairs'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False), flush=True)
