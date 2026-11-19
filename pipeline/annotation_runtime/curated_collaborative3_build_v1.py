"""CPU source-plan preparation, same-byte ROI binding, immutable member launch builder."""
import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
import curated_collaborative3_plan_v1 as p

MANAGER = p.RT / 'curated_collaborative3_resource_manager_v1.py'
EVIDENCE = p.RT / 'curated_collaborative3_params_evidence_v1.json'
ROI_TEMPLATE = p.RT / 'curated_roi_local_n5_v1.json'
TEMPLATES = {'producer': p.RT / 'curated_collaborative21000_n5g0_launch_v2.json',
             'consumer': p.RT / 'curated_collaborative21000_n3g67_launch_v2.json'}


def option(argv, name, value):
    if argv.count(name) != 1:
        raise RuntimeError('Unique frozen argv option required: ' + name)
    argv[argv.index(name) + 1] = str(value)


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()


def snapshot(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns, 'device': s.st_dev, 'inode': s.st_ino}


def bind_roi(node):
    if node not in ('N2', 'N7') or p.HOSTS[node] not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('ROI reference may be bound only on its exact new producer node')
    value = p.load(ROI_TEMPLATE)
    for f in value['files'].values():
        before = snapshot(f['path'])
        if p.sha(f['path']) != f['sha256'] or snapshot(f['path']) != before:
            raise RuntimeError('ROI byte identity differs from frozen original reference')
        f['stat'] = before
    value['sha256'] = digest(value['files'])
    return p.write(p.RT / ('curated_collaborative3_roi_' + node + '_v1.json'), value, identical=True)


def prepare_evidence():
    old = p.load(p.RT / 'curated_collaborative21000_params_evidence_v2.json')['new_configuration']
    new = copy.deepcopy(old)
    new['hand_roi_root'] = str(p.OUT / 'roi')
    return p.write(EVIDENCE, {'original_configuration': old, 'new_configuration': new,
                  'changed_fields': ['hand_roi_root'], 'original_params_sha256': digest(old),
                  'new_params_sha256': digest(new), 'consumer_base_sha256': p.BASE_SHA,
                  'bar_parameters_and_source_unchanged_except_hand_roi_root': True,
                  'semantic_approval': False, 'formal_training': False}, identical=True)


def construct(label, capture_ref):
    value = p.verify_plan(p.load(p.PLAN), verify_sources=False)
    node, mode, gpus, shard, nshard = p.MEMBERS[label]
    capture = p.checked(capture_ref)
    if (capture['node'] != node or capture['mode'] != mode or capture['gpus'] != gpus
            or capture['signals_sent'] is not False or len(capture['holders']) != len(gpus)
            or len(capture['leases']) != len(gpus)
            or not 0 <= time.time_ns() - capture['captured_unix_ns'] <= 1200 * 10**9):
        raise RuntimeError('Exact new-node capture must be fresh within 20 minutes')
    template = TEMPLATES[mode]
    cfg = p.load(template)
    cfg.update(node=node, mode=mode, phase='curated_collaborative3_' + label + '_v1',
               gpus=gpus, holders=capture['holders'], holder_capture=capture_ref,
               out=str(p.paths(label)['out']), queue=str(p.paths(label)['queue']),
               source_range=value['source_range'], selected_sources=value['selected_sources'][label],
               targetcount=len(p.checked(value['selected_sources'][label])),
               source_selection={'shard': shard, 'nshard': nshard, 'limit': 0},
               collab3_member=label, collab3_source_plan=p.ref(p.PLAN),
               template_reference=p.ref(template), resource_manager=p.ref(MANAGER),
               base_manager_sha256=p.sha(p.RT / 'resource_manager.py'),
               selected_original_batches=value['selected_original_batches'],
               reuse_existing_results=True, semantic_approval=False)
    cfg.pop('completed_resource_gate', None)
    cfg.pop('previous_generation_gate', None)
    cfg['producer'] = p.ref(p.RT / ('curated_collaborative3_' + mode + '_v1.py'))
    cfg['argv'][3] = cfg['producer']['path']
    cfg['argv'][5:5] = ['--member', label]
    cfg['env']['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    option(cfg['argv'], '--out', cfg['out'])
    option(cfg['argv'], '--shard', shard)
    option(cfg['argv'], '--nshard', nshard)
    # Retain only native code/input bindings, never prior transaction or active queue receipts.
    source = p.load(template)['source_sha256']
    bindings = {name: sha for name, sha in source.items()
                if ('/source_curated_' in name or '/source_hand_roi_v3/' in name
                    or '/DWPoseProcess/dwpose/' in name or name.endswith('/resource_manager.py')
                    or name.endswith('/curated_resource_manager.py') or name == str(p.MANIFEST)
                    or name.endswith('/curated_producer_io_v2.py') or name.endswith('/curated_producer_io_v3.py'))}
    if mode == 'consumer':
        option(cfg['argv'], '--selected-sources', cfg['selected_sources']['path'])
        cfg['consumer_params_evidence'] = p.ref(EVIDENCE)
        cfg['consumer_params_sha256'] = p.load(EVIDENCE)['new_params_sha256']
    else:
        roi_path = p.RT / ('curated_collaborative3_roi_' + node + '_v1.json')
        roi = p.load(roi_path)
        original = p.load(ROI_TEMPLATE)
        if (set(roi) != set(original) or any(roi[k] != original[k] for k in roi if k not in ('files', 'sha256'))
                or set(roi['files']) != set(original['files'])
                or any((f['path'], f['sha256']) != (original['files'][k]['path'], original['files'][k]['sha256'])
                       for k, f in roi['files'].items()) or roi['sha256'] != digest(roi['files'])):
            raise RuntimeError('ROI reference changes beyond same-byte node-local stat rebinding')
        cfg['modelref'] = {**p.ref(roi_path), 'model_identity_sha256': roi['sha256']}
        option(cfg['argv'], '--roi-model-reference', roi_path)
        option(cfg['argv'], '--ordinal-start', p.START)
        option(cfg['argv'], '--ordinal-end', p.END)
    references = [cfg['producer'], cfg['selected_sources'], capture_ref, cfg['resource_manager'],
                  cfg['collab3_source_plan'], cfg['template_reference'], cfg['modelref'],
                  p.ref(p.RT / 'curated_collaborative3_plan_v1.py'), p.ref(__file__),
                  p.ref(p.RT / 'curated_collaborative21000_resource_manager_v2.py'),
                  p.ref(p.RT / 'curated_collaborative21000_consumer_v2.py'),
                  p.ref(p.RT / 'curated_producer_io_v4.py'),
                  p.ref(p.RT / 'curated_fast_source_v1.py'),
                  p.ref(p.RT / 'curated_request_image_cache_v1.py'),
                  p.ref(p.RT / 'curated_transaction_cache_v1.py')]
    if mode == 'consumer':
        references.append(cfg['consumer_params_evidence'])
    for r in references:
        bindings[r['path']] = r['sha256']
    for batch in cfg['selected_original_batches']:
        bindings[batch['path']] = batch['sha256']
    cfg['source_sha256'] = bindings
    return cfg


def build(label, capture):
    cfg = construct(label, p.ref(capture))
    launch_path = p.paths(label)['launch']
    if p.paths(label)['queue'].exists() or launch_path.exists():
        raise RuntimeError('Repeated member dispatch/existing managed queue refused')
    launch = p.write(launch_path, cfg)
    command = [cfg['argv'][0], '-B', '-u', str(MANAGER), '--config', launch['path'],
               '--config-sha', launch['sha256'], '--manager-sha', cfg['base_manager_sha256']]
    return {'launch': launch, 'node': cfg['node'], 'gpus': cfg['gpus'], 'source_range': cfg['source_range'],
            'targetcount': cfg['targetcount'], 'command': shlex.join(command),
            'readonly_command': shlex.join(command + ['--readonly-admission', '--report', str(launch_path).replace('_launch.json', '_readonly.json')]),
            'gpu_started': False, 'signals_sent': False, 'leases_created': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check-plan', action='store_true')
    parser.add_argument('--bind-roi', choices=['N2', 'N7'])
    parser.add_argument('--member', choices=list(p.MEMBERS))
    parser.add_argument('--capture')
    args = parser.parse_args()
    if args.cpu_check_plan:
        plan_ref = p.prepare_plan()
        evidence = prepare_evidence()
        import curated_collaborative3_consumer_v1 as consumer
        consumer.install(p.pipeline())
        print(json.dumps({'plan': plan_ref, 'evidence': evidence, 'members': list(p.MEMBERS),
                          'input_count': 222939, 'selected_count': 30000, 'source_range': {'start': p.START, 'end': p.END},
                          'model_calls': 0, 'signals_sent': False, 'leases_created': False}), flush=True)
    elif args.bind_roi:
        print(json.dumps(bind_roi(args.bind_roi)), flush=True)
    else:
        if not args.member or not args.capture:
            parser.error('--member and --capture required')
        print(json.dumps(build(args.member, args.capture)), flush=True)


if __name__ == '__main__':
    main()
