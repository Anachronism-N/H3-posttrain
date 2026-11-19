"""Build only next-range files; no GPU launch, signal, lease, or old-output writes."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
OUT = RT.parent / 'curated222939_collaborative_v2'
PREFIX = 'curated_collaborative21000'
GROUPS = ('n5g0', 'n5g1', 'n3g67')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return ref(path)


def previous_gate(group, capture):
    phases = {
        'n5g0': 'curated_io_failure_recovery500_n5g0_v1',
        'n5g1': 'curated_collaborative20000_n5g1_v1',
        'n3g67': 'curated_io_failure_recovery500_consumer_n3g67_v1',
    }
    queue = RT.parent / 'queues' / phases[group]
    if group == 'n5g1':
        launch = ref(RT / 'curated_collaborative20000_n5g1_launch_v1.json')
    else:
        stage = 'producer' if group == 'n5g0' else 'consumer'
        pointer = load(RT / ('curated_io_failure_recovery500_%s_when_safe_admission_v1.json' % stage))
        launch = pointer['config']
        if sha(launch['path']) != launch['sha256']:
            raise RuntimeError('Completed recovery launch pointer changed')
    paths = [queue / name for name in ('admission.json', 'worker_exit.json', 'resources_safe.json')]
    paths += [queue / ('holder_restored_gpu%d.json' % gpu) for gpu in capture['gpus']]
    return {'queue': str(queue), 'launch_config': launch,
            'references': [launch] + [ref(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', default='v2')
    args = parser.parse_args()
    if not args.generation or not all(c.isalnum() or c in '_-' for c in args.generation):
        raise ValueError('Unsafe generation namespace')
    if OUT.exists():
        raise RuntimeError('Unexpected next-range output already exists; do not remove or reuse blindly')
    manager_path = RT / (PREFIX + '_resource_manager_v2.py')
    spec = importlib.util.spec_from_file_location('next_range_manager', manager_path)
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    worker = manager.worker
    pipeline = worker.load_pipeline()
    templates = {group: RT / ('curated_collaborative20000_%s_launch_v1.json' % ('n3g01' if group == 'n3g67' else group)) for group in GROUPS}
    originals = {group: load(path) for group, path in templates.items()}
    for cfg in originals.values():
        for path, expected in cfg['source_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError('Frozen old source changed: ' + path)
        if sha(cfg['producer']['path']) != cfg['producer']['sha256']:
            raise RuntimeError('Frozen worker changed')
    summary_ref = originals['n3g67']['manifest_summary']
    summary = load(summary_ref['path'])
    if sha(summary_ref['path']) != summary_ref['sha256']:
        raise RuntimeError('Original manifest summary changed')
    rows = worker.selected_rows(pipeline, summary_ref['path'])
    batches = [b for b in summary['batches'] if worker.START <= b['first_source_ordinal'] < worker.END]
    expected_batches = ['train-b%06d' % i for i in range(42, 62)]
    if ([b['batch_id'] for b in batches] != expected_batches or len(batches) != 20
            or any(b['count'] != 500 or b['split'] != 'train' for b in batches)
            or summary['manifest'] != originals['n3g67']['original_full_manifest']):
        raise RuntimeError('Must retain exactly 20 original 500-source train batches42:62')
    previous_rows = load(originals['n3g67']['selected_sources']['path'])
    if set(r['id'].casefold() for r in previous_rows) & set(r['id'].casefold() for r in rows):
        raise RuntimeError('Next-range selection overlaps previous collaborative selection')
    selected = {}
    captures = {}
    for group in GROUPS:
        parity = int(group[-1]) if group != 'n3g67' else None
        chosen = rows if parity is None else [r for r in rows if r['source_ordinal'] % 2 == parity]
        selected_path = RT / ('%s_%s_selected_%s.json' % (PREFIX, group, args.generation))
        if selected_path.exists():
            if load(selected_path) != chosen:
                raise RuntimeError('Existing next-range selection metadata differs; do not overwrite')
            selected[group] = ref(selected_path)
        else:
            selected[group] = write(selected_path, chosen)
        path = RT / ('%s_%s_holder_lease_capture_%s.json' % (PREFIX, group, args.generation))
        captures[group] = (ref(path), load(path))
        captured = captures[group][1]
        expected_gpus = [6, 7] if group == 'n3g67' else [parity]
        if (captured['gpus'] != expected_gpus or captured['signals_sent'] is not False
                or not 0 <= time.time_ns() - captured['captured_unix_ns'] <= 1200 * 10**9):
            raise RuntimeError('Fresh exact capture unavailable')
    consumer_cfg = copy.deepcopy(originals['n3g67'])
    consumer_cfg['argv'][3] = str(RT / (PREFIX + '_consumer_v2.py'))
    for option, value in (('--out', str(OUT)), ('--selected-sources', selected['n3g67']['path'])):
        consumer_cfg['argv'][consumer_cfg['argv'].index(option) + 1] = value
    _, remaining = manager.argparse_consumer(consumer_cfg['argv'][4:])
    worker_args = pipeline.parser().parse_args(remaining)
    new_configuration = worker.configuration(pipeline, worker_args)
    _, _, visual = pipeline.modules(worker_args.repo, worker_args.ms_source)
    old_evidence = load(originals['n3g67']['consumer_params_evidence']['path'])
    previous_configuration = copy.deepcopy(new_configuration)
    previous_configuration['hand_roi_root'] = old_evidence['new_configuration']['hand_roi_root']
    if previous_configuration != old_evidence['new_configuration']:
        raise RuntimeError('Previous native argv parameters differ from frozen evidence')
    changes = [key for key in set(previous_configuration) | set(new_configuration)
               if previous_configuration.get(key) != new_configuration.get(key)]
    if changes != ['hand_roi_root'] or visual.digest(previous_configuration) != originals['n3g67']['consumer_params_sha256']:
        raise RuntimeError('Only native hand_roi_root parameter may change')
    evidence = write(RT / ('%s_params_evidence_%s.json' % (PREFIX, args.generation)), {
        'previous_configuration': previous_configuration, 'original_configuration': old_evidence['original_configuration'],
        'new_configuration': new_configuration, 'previous_params_sha256': visual.digest(previous_configuration),
        'new_params_sha256': visual.digest(new_configuration), 'changed_fields': changes,
        'bar_parameters_and_source_unchanged_except_hand_roi_root': True, 'full_params_sha256_equal': False,
        'reason': 'Independent next-range sibling output changes native hand ROI root only',
        'previous_evidence_ref': originals['n3g67']['consumer_params_evidence'],
        'consumer_base_sha256': worker.BASE_SHA, 'semantic_approval': False})
    review_manifest = write(RT / ('%s_auto_review_manifest_%s.json' % (PREFIX, args.generation)), {
        'schema': 'curated_next_range_original_batch_review_manifest_v2', 'original_summary': summary_ref,
        'original_full_manifest': summary['manifest'], 'source_denominator': 222939, 'split_denominators': summary['counts'],
        'source_range': {'start': worker.START, 'end': worker.END}, 'selected_source_count': 10000,
        'selected_sources': selected['n3g67'], 'selected_batches': batches, 'selected_batch_count': 20,
        'results_root': str(OUT), 'consumer_stream': str(OUT / 'completed.jsonl'),
        'review_out': str(OUT / 'independent_review_v2'), 'overlay_out': str(OUT / 'review_overlay_v2'),
        'sample_count': 25, 'risk_count': 5, 'sample_unit': 'unchanged original 500-source batch',
        'random_sample_membership_fixed': True, 'risk_additions_not_in_random_denominator': True,
        'label_ready_gate': 'original batch membership retained; complete joint labels only; missing/incomplete/failed kept pending',
        'batch_runner_api': 'ms_batch_review.review_batch_ready(sample_count=25,risk_count=5,expected_source_ids=original500)',
        'later_cli_parameters': ['--manifest', '--results-root', '--out', '--selected-batch-ids', '--consumer-stream', '--overlay-out'],
        'old_hardcoded_watch_attached': False, 'current_v1_audit_covers_this_out': False,
        'reviewer_started': False, 'human_approval_automatically_granted': False, 'full_curated_complete': False})
    configs = {}
    for group in GROUPS:
        cfg = copy.deepcopy(consumer_cfg if group == 'n3g67' else originals[group])
        capture_ref, captured = captures[group]
        cfg.update(phase='%s_%s_%s' % (PREFIX, group, args.generation), holders=captured['holders'],
                   gpus=captured['gpus'], targetcount=10000 if group == 'n3g67' else 5000,
                   out=str(OUT), queue=str(RT.parent / 'queues' / ('%s_%s_%s' % (PREFIX, group, args.generation))),
                   holder_capture=capture_ref, selected_sources=selected[group],
                   source_range={'start': worker.START, 'end': worker.END},
                   template_reference=ref(templates[group]), resource_manager=ref(manager_path),
                   completed_resource_gate=previous_gate(group, captured))
        cfg['env']['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, captured['gpus']))
        cfg['argv'][cfg['argv'].index('--out') + 1] = str(OUT)
        if group == 'n3g67':
            cfg['producer'] = ref(RT / (PREFIX + '_consumer_v2.py'))
            cfg['consumer_params_sha256'] = visual.digest(new_configuration)
            cfg['consumer_params_evidence'] = evidence
        else:
            for option, value in (('--ordinal-start', str(worker.START)), ('--ordinal-end', str(worker.END))):
                cfg['argv'][cfg['argv'].index(option) + 1] = value
        bindings = {path: digest for path, digest in cfg['source_sha256'].items()
                    if 'curated_collaborative20000_' not in Path(path).name
                    and Path(path).name not in ('train-b000040.manifest.jsonl', 'train-b000041.manifest.jsonl')}
        refs = [ref(manager_path), ref(RT / (PREFIX + '_consumer_v2.py')), selected[group], capture_ref,
                cfg['template_reference'], review_manifest]
        if group == 'n3g67':
            refs.append(evidence)
        refs.extend({'path': b['path'], 'sha256': b['sha256']} for b in batches)
        refs.extend(cfg['completed_resource_gate']['references'])
        for reference in refs:
            bindings[reference['path']] = reference['sha256']
        cfg['source_sha256'] = bindings
        if Path(cfg['queue']).exists():
            raise RuntimeError('Unexpected next-range queue already exists; do not delete')
        manager.selected_rows(cfg)
        manager.completed_resource_gate(cfg)
        configs[group] = write(RT / ('%s_%s_launch_%s.json' % (PREFIX, group, args.generation)), cfg)
    commands = {}
    for group, config in configs.items():
        commands[group] = shlex.join(['/opt/venv_eval/bin/python', '-B', '-u', str(manager_path),
                                    '--config', config['path'], '--config-sha', config['sha256'],
                                    '--manager-sha', originals[group]['base_manager_sha256']])
    receipt = write(RT / ('%s_build_receipt_%s.json' % (PREFIX, args.generation)), {
        'configs': configs, 'manager': ref(manager_path), 'worker': ref(RT / (PREFIX + '_consumer_v2.py')),
        'auto_review_manifest': review_manifest, 'params_evidence': evidence, 'main_commands': commands,
        'source_range': {'start': worker.START, 'end': worker.END}, 'selected_count': 10000,
        'original_batch_count': 20, 'selection_overlaps_previous': False, 'signals_sent': False,
        'gpu_started': False, 'leases_created': False, 'queue_created': False, 'full_curated_complete': False})
    print(json.dumps({'build_receipt': receipt, 'configs': configs, 'main_commands': commands}, ensure_ascii=False))


if __name__ == '__main__':
    main()
