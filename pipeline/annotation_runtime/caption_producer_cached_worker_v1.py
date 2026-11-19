"""Fixed N2p0 native IOv4 worker; opt-in source cache and sole ready writer."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import caption_producer_cache_handoff_v1 as hand


def option(args, name):
    if args.count(name) != 1:
        raise RuntimeError('Unique explicit native option required: ' + name)
    index = args.index(name)
    if index + 1 == len(args):
        raise RuntimeError('Missing native option value: ' + name)
    return args[index + 1]


def contract(cfg):
    old = hand.checked(cfg['old_config'])
    if cfg['old_config'] != {'path': str(hand.OLD_CONFIG), 'sha256': hand.OLD_SHA}:
        raise RuntimeError('Only frozen old N2p0 config accepted')
    if (cfg['node'] != 'N2' or cfg['gpus'] != [0] or cfg['mode'] != 'producer'
            or cfg['source_root'] != str(hand.SOURCE_ROOT) or cfg['input_count'] != 222939
            or cfg['env']['CUDA_VISIBLE_DEVICES'] != '0'
            or cfg['phase'] != hand.PHASE or cfg['out'] != str(hand.B / hand.PHASE)
            or cfg['queue'] != str(hand.B / 'queues' / hand.PHASE)
            or cfg['source_selection'] != {'shard': 0, 'nshard': 4, 'limit': 0}
            or cfg['old_config_resume_forbidden'] is not True):
        raise RuntimeError('Fixed cached producer node/GPU/source scope changed')
    proof = hand.checked(cfg['source_handoff'])
    if (proof['schema'] != 'caption_producer_cache_handoff_v1'
            or proof['old_config'] != cfg['old_config'] or proof['source_root'] != cfg['source_root']
            or proof['remaining_meta'] != cfg['remaining_meta'] or proof['ids_file'] != cfg['ids_file']
            or proof['source_owner_absence'] is not True or proof['old_gpu_tree_absent'] is not True
            or proof['sole_ready_writer'] != 'n2p0' or proof['old_config_resume_forbidden'] is not True):
        raise RuntimeError('Exact source handoff proof changed')
    original = hand.checked(old['selected_sources'])
    if len(original) != 7500 or proof['old_selected'] != old['selected_sources']:
        raise RuntimeError('Frozen original 7500-source selection changed')
    hand.validate_meta(original, original)
    rows = hand.validate_meta(hand.checked(cfg['remaining_meta']), original)
    excluded = hand.validate_meta(proof['excluded_started_meta'], original)
    excluded_ids = {r['id'] for r in excluded}
    if rows != [r for r in original if r['id'] not in excluded_ids]:
        raise RuntimeError('Remaining must be exactly original 7500 minus all started sources')
    old_admission = hand.checked(proof['old_admission'])
    old_live = hand.checked(proof['old_live'])
    if (old_admission['config_sha256'] != hand.OLD_SHA
            or old_admission['manager_sha256'] != hand.BASE_SHA
            or old_admission['manager_identity'] != proof['old_parent']
            or old_live['child_identity'] != proof['old_child'] or old_live['argv'] != old['argv']
            or old_live['manager_identity'] != proof['old_parent']
            or proof['old_child']['ppid'] != proof['old_parent']['pid']
            or proof['old_child']['pid'] not in proof['old_tree_pids']):
        raise RuntimeError('Original admitted owner/tree proof changed')
    if not rows or len(rows) != cfg['targetcount'] or cfg['plan'] != cfg['remaining_meta']:
        raise RuntimeError('Nonempty exact remaining metadata/count required')
    if hand.sha(cfg['ids_file']['path']) != cfg['ids_file']['sha256']:
        raise RuntimeError('Native IDs binding changed')
    if Path(cfg['ids_file']['path']).read_text().splitlines() != [r['id'] for r in rows]:
        raise RuntimeError('Native ID list differs from remaining manifest order')
    if cfg['native_argv'] != hand.native_args(old, cfg['ids_file']['path']):
        raise RuntimeError('Native pipeline/source/ROI/BeatThis/sampling args changed')
    if (cfg['producer'] != hand.ref(__file__)
            or cfg['cache_code'] != hand.ref(hand.RT / 'caption_producer_stage_cache_v1.py')
            or cfg['resource_manager'] != hand.ref(hand.RT / 'caption_producer_cached_manager_v1.py')
            or cfg['handoff_code'] != hand.ref(hand.RT / 'caption_producer_cache_handoff_v1.py')):
        raise RuntimeError('Actual cached worker/manager/core code changed')
    claim = hand.checked(proof['claim'])
    if claim['old_config'] != cfg['old_config'] or claim['no_retry_after_failure'] is not True:
        raise RuntimeError('One-time handoff claim changed')
    hand.checked(proof['old_resource_safe'])
    hand.checked(proof['restored_holder'])
    hand.verify_prefix(proof['old_ready_prefix'])
    return rows, proof


def parent_gate(base, cfg, config_path, proof):
    hand.fixed_node(cfg)
    hand.old_owner_absence(base, proof)
    queue = Path(cfg['queue'])
    admission = hand.load(queue / 'admission.json')
    parent = base.identity(os.getppid())
    hand.exact_identity(base, admission['manager_identity'], parent)
    expected_argv = cfg['argv'][:3] + [cfg['resource_manager']['path'], '--config', str(config_path),
                                      '--config-sha', hand.sha(config_path), '--manager-sha', hand.BASE_SHA]
    if (parent['argv'] != expected_argv or admission['node'] != 'N2' or admission['gpus'] != [0]
            or admission['config_sha256'] != hand.sha(config_path) or admission['manager_sha256'] != hand.BASE_SHA
            or admission['producer'] != cfg['producer'] or admission['plan'] != cfg['plan']
            or admission['out'] != cfg['out'] or admission['targetcount'] != cfg['targetcount']):
        raise RuntimeError('Exact native admission parent/argv/source binding mismatch')
    current = base.identity(os.getpid())
    if (current['ppid'] != parent['pid'] or current['argv'] != cfg['argv']
            or current['uid'] != parent['uid'] or current['cgroup'] != parent['cgroup']
            or current['CVD'] != ['CUDA_VISIBLE_DEVICES=0'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'):
        raise RuntimeError('Exact cached child identity/CVD/parent chain changed')
    hand.lock_state(base, parent['pid'])
    lease = Path('/tmp/od_reserve_0').lstat()
    import stat
    if not stat.S_ISREG(lease.st_mode) or lease.st_uid != os.getuid() or lease.st_size or time.time()-lease.st_mtime > 30:
        raise RuntimeError('Native parent lease missing/stale')
    if base.uuid_for(0) != proof['gpu_uuid']:
        raise RuntimeError('Actual N2G0 UUID changed')
    if [a for a in base.apps() if a['uuid'] == proof['gpu_uuid']]:
        raise RuntimeError('GPU must be empty before native worker model load')
    deadline = time.monotonic() + 5
    while not (queue / 'live.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Native parent live receipt missing')
        time.sleep(.05)
    live = hand.load(queue / 'live.json')
    hand.exact_identity(base, live['child_identity'], base.identity(os.getpid()))
    hand.exact_identity(base, live['manager_identity'], parent)
    if live['argv'] != cfg['argv']:
        raise RuntimeError('Native child live argv mismatch')
    return {'parent': parent, 'child': current, 'gpu_uuid': proof['gpu_uuid']}


def run_cached(cfg, rows, proof, io4, cache_class, emit, execution):
    native = io4.base.load_pipeline
    expected = {r['id']: r for r in rows}
    native_run = [None]
    base = hand.native_base()
    with ExitStack() as stack:
        stack.callback(setattr, io4.base, 'load_pipeline', native)
        stack.callback(setattr, io4.base, 'publish_roi', io4.base.publish_roi)
        stack.callback(setattr, io4.base, 'LocalROI', io4.base.LocalROI)

        def loader(path=io4.base.FROZEN):
            pipeline = native(path)
            bar, roi, visual = pipeline.modules(option(cfg['native_argv'], '--repo'))
            original_append, original_save, original_modules = pipeline.append, pipeline.save, pipeline.modules
            stack.callback(setattr, pipeline, 'append', original_append)
            stack.callback(setattr, pipeline, 'save', original_save)
            stack.callback(setattr, pipeline, 'modules', original_modules)
            pipeline.modules = lambda repo: (bar, roi, visual)
            root = Path(cfg['source_root'])
            ready = root / 'ready' / 'producer-0000.jsonl'

            def routed_append(path, value):
                if Path(path) == root / 'ready.jsonl':
                    metadata = expected.get(value.get('id'))
                    if metadata is None or any(value.get(k) != metadata[k] for k in hand.KEYS):
                        raise RuntimeError('Ready publication outside exact remaining original source')
                    path = ready
                return original_append(path, value)

            def bound_save(path, value):
                path = Path(path)
                if path.name == 'protocol.json':
                    if path.parent.parent != root / 'runs' or native_run[0] is not None or path.exists():
                        raise RuntimeError('Only a new independent native producer run protocol may be written')
                    native_run[0] = str(path.parent)
                    value = {**value, 'caption_producer_cache_code': cfg['cache_code'],
                             'caption_producer_cached_worker': cfg['producer'],
                             'caption_producer_source_handoff': cfg['source_handoff'],
                             'caption_producer_source_handoff_code': cfg['handoff_code'],
                             'caption_producer_remaining_meta': cfg['remaining_meta'],
                             'caption_producer_execution_root': str(execution)}
                    hand.write_once(execution / 'native_run.json', {'run_root': str(path.parent),
                                    'remaining_meta': cfg['remaining_meta'], 'source_handoff': cfg['source_handoff']})
                return original_save(path, value)

            pipeline.append, pipeline.save = routed_append, bound_save
            # IO4 has set the actual IO2 publisher before invoking this loader.
            if io4.base.publish_roi is not io4.publish_roi:
                raise RuntimeError('Actual IO2 callee is not the native IO4 publisher')
            captured_run = pipeline.run_producer
            stack.callback(setattr, pipeline, 'run_producer', captured_run)

            def cached_run(args):
                actual_bar, local_roi, actual_visual = pipeline.modules(args.repo)
                cache = cache_class(pipeline, actual_bar, local_roi.roi, actual_visual, root, emit)
                with cache.installed(publishers=(io4.base,)):
                    return cache.run_wrapper(captured_run)(args)

            pipeline.run_producer = cached_run
            native_load_model = roi.load_cuda_model
            stack.callback(setattr, roi, 'load_cuda_model', native_load_model)

            def load_model(*args, **kwargs):
                model = native_load_model(*args, **kwargs)
                running = [a for a in base.apps() if a['uuid'] == proof['gpu_uuid']]
                if len(running) != 1 or running[0]['pid'] != os.getpid():
                    raise RuntimeError('Native ROI model GPU app is not the sole admitted child')
                return model

            roi.load_cuda_model = load_model
            return pipeline

        io4.base.load_pipeline = loader
        result = io4.main(cfg['native_argv'])
    if not native_run[0] or result.get('run_root') != native_run[0]:
        raise RuntimeError('Native cached run result binding missing')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    extra, remaining = parser.parse_known_args(argv)
    config_path = Path(extra.config)
    cfg = hand.load(config_path)
    rows, proof = contract(cfg)
    if remaining and remaining != cfg['native_argv']:
        raise RuntimeError('Optional original producer args must exactly match bound native args')
    if str(config_path) != str(hand.RT / (hand.PHASE + '_launch.json')):
        raise RuntimeError('Only immutable fixed launch config is accepted')
    base = hand.native_base()
    owner = parent_gate(base, cfg, config_path, proof)
    available, blocked = hand.split_remaining(rows, hand.SOURCE_ROOT)
    if blocked or len(available) != len(rows):
        raise RuntimeError('Remaining source has started since handoff; no replay permitted')
    execution = Path(cfg['out'])
    if not execution.is_dir() or execution.is_symlink():
        raise RuntimeError('Native parent-owned execution directory missing or unsafe')
    hand.write_once(execution / 'producer_owner_claim.json', {'config': hand.ref(config_path), **owner,
                    'ready_writer': str(hand.SOURCE_ROOT / 'ready' / 'producer-0000.jsonl'),
                    'old_config_resume_forbidden': True, 'no_retry_after_failure': True})
    import curated_producer_io_v4 as io4
    from caption_producer_stage_cache_v1 import NativeStageCache
    with (execution / 'source_stage_timings.jsonl').open('x') as stream:
        def emit(record):
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            stream.flush()
        result = run_cached(cfg, rows, proof, io4, NativeStageCache, emit, execution)
    summary = {'summary': {'stage': 'run', 'plan_sha256': cfg['plan']['sha256'],
                          'total_source_count': 222939, 'selected_source_count': len(rows),
                          'processed_source_count': sum(result['counts'].values())},
               'selection': {'selected_ids': [r['id'] for r in rows]}, 'native_result': result,
               'source_handoff': cfg['source_handoff'], 'cache_code': cfg['cache_code'],
               'semantic_approval': False, 'source_failures_preserved': True}
    hand.write_once(execution / 'run.summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
