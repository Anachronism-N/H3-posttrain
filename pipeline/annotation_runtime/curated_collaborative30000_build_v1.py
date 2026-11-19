"""Build immutable launch configs from native templates and fresh readonly captures."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import time

RT = Path(__file__).resolve().parent
NAME = 'curated_collaborative30000'
spec = importlib.util.spec_from_file_location(NAME + '_worker', RT / (NAME + '_consumer_v1.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    if Path(path).exists():
        if json.loads(Path(path).read_text()) != value:
            raise RuntimeError('Existing immutable build artifact differs: ' + str(path))
        return ref(path)
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return ref(path)


def option(argv, name, value):
    if argv.count(name) != 1:
        raise RuntimeError('Unexpected native argv option: ' + name)
    argv[argv.index(name) + 1] = str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=['n5g0', 'n5g1', 'n3g67'], required=True)
    parser.add_argument('--capture', required=True)
    args = parser.parse_args()
    consumer = args.label == 'n3g67'
    old_label = 'n3g01' if consumer else args.label
    template = RT / ('curated_collaborative20000_' + old_label + '_launch_v1.json')
    cfg = json.loads(template.read_text())
    os.environ.update(cfg['env'])
    capture = json.loads(Path(args.capture).read_text())
    gpus = [6, 7] if consumer else [int(args.label[-1])]
    if (capture['node'] != cfg['node'] or capture['mode'] != cfg['mode']
            or capture['gpus'] != gpus or capture['signals_sent'] is not False
            or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9):
        raise RuntimeError('Capture not fresh or mismatched')
    cfg.update(phase=NAME + '_' + args.label + '_v1', gpus=gpus, holders=capture['holders'],
               holder_capture=ref(args.capture), out=str(worker.OUT),
               queue=str(RT.parent / 'queues' / (NAME + '_' + args.label + '_v1')),
               source_range={'start': worker.START, 'end': worker.END},
               targetcount=2000 if consumer else 1000, semantic_approval=False,
               template_reference=ref(template))
    cfg['env']['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    option(cfg['argv'], '--out', worker.OUT)
    pipeline = worker.load_pipeline()
    rows = worker.selected_rows(pipeline, cfg['manifest_summary']['path'])
    summary = pipeline.load(cfg['manifest_summary']['path'])
    batches = [b for b in summary['batches'] if worker.START <= b['first_source_ordinal'] < worker.END]
    if ([b['batch_id'] for b in batches] != ['train-b000060', 'train-b000061', 'train-b000062', 'train-b000063']
            or any(b['count'] != 500 or b['split'] != 'train' for b in batches)):
        raise RuntimeError('Original four 500-source train batches changed')
    if not consumer:
        rows = [r for r in rows if r['source_ordinal'] % 2 == gpus[0]]
    if len(rows) != cfg['targetcount']:
        raise RuntimeError('Explicit selected budget changed')
    cfg['selected_sources'] = write(RT / (NAME + '_' + args.label + '_selected_v1.json'), rows)
    cfg['selected_original_batches'] = batches
    cfg['source_sha256'] = {p: s for p, s in cfg['source_sha256'].items()
                           if 'curated_collaborative20000' not in p and '.manifest.jsonl' not in p}
    if consumer:
        cfg['producer'] = ref(RT / (NAME + '_consumer_v1.py'))
        cfg['argv'][3] = cfg['producer']['path']
        option(cfg['argv'], '--selected-sources', cfg['selected_sources']['path'])
        extra_parser = argparse.ArgumentParser(add_help=False)
        extra_parser.add_argument('--selected-sources', required=True)
        _, remaining = extra_parser.parse_known_args(cfg['argv'][4:])
        native_args = pipeline.parser().parse_args(remaining)
        configuration = worker.configuration(pipeline, native_args)
        evidence_old = json.loads((RT / 'curated_collaborative20000_params_evidence_v1.json').read_text())
        original = evidence_old['new_configuration']
        changed = [k for k in original if original[k] != configuration[k]]
        if set(original) != set(configuration) or changed != ['hand_roi_root']:
            raise RuntimeError('Native parameters differ beyond isolated hand ROI root')
        _, _, visual = pipeline.modules(native_args.repo, native_args.ms_source)
        cfg['consumer_params_sha256'] = visual.digest(configuration)
        cfg['consumer_params_evidence'] = write(RT / (NAME + '_params_evidence_v1.json'), {
            'original_configuration': original, 'new_configuration': configuration,
            'original_params_sha256': visual.digest(original), 'new_params_sha256': visual.digest(configuration),
            'changed_fields': changed, 'bar_parameters_and_source_unchanged_except_hand_roi_root': True,
            'consumer_base_sha256': worker.BASE_SHA, 'semantic_approval': False, 'formal_training': False})
    else:
        option(cfg['argv'], '--ordinal-start', worker.START)
        option(cfg['argv'], '--ordinal-end', worker.END)
        if cfg['producer']['path'] != str(RT / 'curated_producer_io_v3.py'):
            raise RuntimeError('Only frozen native IO v3 producer admitted')
    cfg['resource_manager'] = ref(RT / (NAME + '_resource_manager_v1.py'))
    cfg['base_manager_sha256'] = sha(RT / 'resource_manager.py')
    refs = [cfg['selected_sources'], cfg['holder_capture'], cfg['resource_manager'], cfg['template_reference'],
            ref(RT / (NAME + '_consumer_v1.py')), ref(Path(__file__))]
    if consumer:
        refs.append(cfg['consumer_params_evidence'])
    for r in refs:
        cfg['source_sha256'][r['path']] = r['sha256']
    for b in batches:
        cfg['source_sha256'][b['path']] = b['sha256']
    launch = write(RT / (NAME + '_' + args.label + '_launch_v1.json'), cfg)
    command = [cfg['argv'][0], '-B', '-u', cfg['resource_manager']['path'], '--config', launch['path'],
               '--config-sha', launch['sha256'], '--manager-sha', cfg['base_manager_sha256']]
    result = {'launch': launch, 'manager': cfg['resource_manager'], 'command': shlex.join(command),
              'node': cfg['node'], 'gpus': gpus, 'selected_count': len(rows), 'source_range': cfg['source_range'],
              'out': cfg['out'], 'queue': cfg['queue'], 'gpu_started': False, 'signals_sent': False}
    write(RT / (NAME + '_' + args.label + '_build_receipt_v1.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
