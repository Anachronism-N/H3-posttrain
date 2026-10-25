"""N2/GPU0-only H3 fixed3 comparison; independently frozen refs, no deployment.
Config: node/host/gpus/release_authorized, queue/out, source_sha256, protocol/final_selection,
environment_refs [{path,sha256}], families keyed by three native planner names.
Each family: training_launch/full_complete/resources_safe/checkpoint_selection/
checkpoint_marker/fingerprint/validation_candidates (four ordered epoch refs), generation_complete/generation_resources_safe/generation_exit/
manifest/raw/h3_records refs. Capture prints read-only holder; run additionally
requires holder_capture {path,sha256}. weight_pre_audit optionally names cache/video
model_sources JSON refs; otherwise real run uses renderer weight_sources full SHA.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import sys

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
RT, HANDOFF = EXP / 'runtime', EXP / 'eval_handoff_clean_v11'
RENDER = RT / 'h3_renderer_v10'
EVAL = RT / 'planner_eval_manager_v3.py'
PINNED = {str(EVAL): 'c387edfd246fd25ab3ac73d00bffe2f41d5f1450cba15df9509b53192ac63c9d',
          str(RENDER / 'h3_video_infer.py'): '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8',
          str(RENDER / 'h3_video_only.py'): '89439d733d62f522bb3dd993bc456541b3ca15ead34ff2e19f86657a471c5055',
          str(RENDER / 'upstream_python_manifest.json'): '799333b231a5b6a46f217a7bed47318ec2fe528d4dd6a21378b050efbe4fa091',
          str(HANDOFF / 'protocol.json'): 'f86d5d15b25a5d10f2870b369e68a9784dbb172e2746e04ebc86aae2a9ed276b'}
for filename in ('h3_video_infer.py', 'h3_video_only.py'):
    PINNED[str(ROOT / 'H3-posttrain/scripts_omnidance/exp/ms' / filename)] = PINNED[str(RENDER / filename)]
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
GEOMETRY = dict(height=1280, width=704, num_frames=124, fps=24)
WEIGHTS = Path('/dev/shm/h3/Ref2VA')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap():
    import hashlib
    if hashlib.sha256(EVAL.read_bytes()).hexdigest() != PINNED[str(EVAL)]:
        raise RuntimeError('resource verifier source SHA changed')
    audit = load('h3_render_eval_audit', EVAL)
    for file in ('h3_video_infer.py', 'h3_video_only.py'):
        if audit.sha(audit.regular(RENDER / file)) != PINNED[str(RENDER / file)]:
            raise RuntimeError('renderer source changed before import')
    return audit, *audit.load_helpers(), load('h3_render_infer', RENDER / 'h3_video_infer.py')


def require(value, expected, reason):
    if any(value.get(k) != v or type(value.get(k)) is not type(v) for k, v in expected.items()):
        raise RuntimeError(reason)


def image_geometry(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.size != (704, 1280):
            raise RuntimeError('actual reference image geometry differs')


def safe_resources(ref, count, audit):
    value = audit.ref_json(ref)
    require(value, dict(all_holders_safe=True, locks_released=True, leases_restored=True,
            managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=count), 'resources unsafe')


def check_sources(cfg, path, digest, own_sha, audit):
    bindings = cfg['source_sha256']
    required = {**PINNED, **audit.PINNED, str(Path(__file__).resolve()): own_sha}
    if audit.sha(audit.regular(path)) != digest:
        raise RuntimeError('launch config changed')
    for name, frozen in required.items():
        if bindings.get(name) != frozen:
            raise RuntimeError('missing/conflicting frozen source: ' + name)
    if not cfg.get('environment_refs'):
        raise RuntimeError('prepared same-package environment refs required')
    for ref in cfg['environment_refs']:
        if bindings.get(ref['path']) != ref['sha256']:
            raise RuntimeError('environment ref is not independently source-frozen')
    upstream = audit.ref_json(dict(path=str(RENDER / 'upstream_python_manifest.json'),
                                  sha256=PINNED[str(RENDER / 'upstream_python_manifest.json')]))
    if len(upstream['files']) != 288 or len({r['path'] for r in upstream['files']}) != 288:
        raise RuntimeError('complete 288-file upstream inventory required')
    for name, frozen in {**bindings, **{r['path']: r['sha256'] for r in upstream['files']}}.items():
        if audit.sha(audit.regular(name)) != frozen:
            raise RuntimeError('source/environment/input SHA changed: ' + name)


def prerequisites(cfg, audit, manager, fixed, ev, h3):
    require(cfg, dict(schema='music-planner-h3-render-manager-v1', node='N2', host='28.216.17.70',
                      gpus=[0], release_authorized=True), 'only authorized N2 GPU0 render')
    if any(type(g) is not int for g in cfg['gpus']):
        raise RuntimeError('physical GPU indices must be integers')
    manager.stage.validate_scope('N2', 'music_flamingo', cfg['host'])
    if cfg['protocol'] != dict(path=str(HANDOFF / 'protocol.json'), sha256=PINNED[str(HANDOFF / 'protocol.json')]):
        raise RuntimeError('fixed protocol ref required')
    protocol = audit.ref_json(cfg['protocol'])
    if [r['id'] for r in protocol['fixed3']] != IDS or set(cfg['families']) != {v[1] for v in audit.SCOPE.values()}:
        raise RuntimeError('exact three cases and three families required')
    if cfg['final_selection']['path'] != str(HANDOFF / 'final_selection.json'):
        raise RuntimeError('actual shared final selection required')
    final = audit.ref_json(cfg['final_selection'])
    require(final, dict(schema='music-planner-final-selection-v1', test_used=False, selection_frozen_before_test=True,
                        protocol=cfg['protocol'], data=protocol['data'], decoding=audit.DECODE), 'final selection differs')
    if set(final['families']) != set(cfg['families']):
        raise RuntimeError('final selection family membership differs')
    records = {}
    for node, (host, family) in audit.SCOPE.items():
        row = cfg['families'][family]
        for key in ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection', 'checkpoint_marker', 'fingerprint'):
            if row[key] != final['families'][family][key]:
                raise RuntimeError('generation training/selection differs from shared final freeze')
        scope = dict(row, node=node, host=host, family=family, protocol=cfg['protocol'], job='fixed3',
                     out=str(Path(row['manifest']['path']).parent))
        exact = dict(training_launch=RT / (family + '_launch_clean_v11.json'),
                     full_complete=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json',
                     resources_safe=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json',
                     checkpoint_selection=EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json')
        if any(row[k]['path'] != str(v) for k, v in exact.items()):
            raise RuntimeError('not actual completed training/best refs')
        _, result = audit.check_completed(scope, manager, local=False)
        checkpoint, identity, selection = audit.selected_checkpoint(scope, result, protocol, fixed, ev, full_hash=False)
        if len(row['validation_candidates']) != 4 or selection['validation_candidates'] != row['validation_candidates']:
            raise RuntimeError('actual best candidates differ from independently frozen epoch markers/validation')
        generation_queue = Path(row['generation_complete']['path']).parent
        if (Path(row['generation_complete']['path']).name != 'eval.complete.json'
                or row['generation_resources_safe']['path'] != str(generation_queue / 'resources_safe.json')
                or row['generation_exit']['path'] != str(generation_queue / 'fixed3.exit.json')
                or not generation_queue.is_relative_to(HANDOFF)
                or Path(row['manifest']['path']).name != 'manifest.json'
                or row['raw']['path'] != str(Path(scope['out']) / 'raw.jsonl')
                or row['h3_records']['path'] != str(Path(scope['out']) / 'h3_records.json')):
            raise RuntimeError('generation receipts do not share actual handoff paths')
        complete = audit.ref_json(row['generation_complete'])
        safe_resources(row['generation_resources_safe'], 1, audit)
        require(audit.ref_json(row['generation_exit']), dict(exit_code=0), 'fixed3 actual worker exit is not zero')
        require(complete, dict(technical_collection_complete=True, complete_native_GPU_outputs_verified=True,
                              human_quality_accepted=False), 'fixed3 generation not complete')
        if complete != audit.check_worker_outputs(scope, checkpoint, identity, protocol, ev):
            raise RuntimeError('generation completion does not bind actual manifest/raw')
        manifest = audit.ref_json(row['manifest'])
        raw_path = audit.regular(row['raw']['path'])
        if (complete['manifest'] != row['manifest'] or complete['raw'] != row['raw']
                or audit.sha(raw_path) != row['raw']['sha256'] or manifest['h3_records'] != row['h3_records']):
            raise RuntimeError('manifest/raw/H3-record frozen binding mismatch')
        require(manifest, dict(complete=True, h3_handoff_approved=True,
                              pretrained_GPU_generation_executed=True), 'actual native GPU generation missing')
        if len(audit.ref_json(row['h3_records'])['records']) != 3:
            raise RuntimeError('H3-record JSON must contain exactly three cases')
        records[family] = h3.captions(Path(row['h3_records']['path']), 3)
        raw = [json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
        if len(records[family]) != 3 or [r['id'] for r in records[family]] != IDS:
            raise RuntimeError('not exact three complete H3 captions')
        for caption, native, case in zip(records[family], raw, protocol['fixed3']):
            if caption['motion_caption'] != native['raw_text'] or caption['image_path'] != case['image_path']:
                raise RuntimeError('raw motion caption/image changed or cleaned')
            image_geometry(caption['image_path'])
            for key in ('image_ref', 'audio_ref'):
                if native[key] != case[key] or audit.sha(audit.regular(case[key]['path'])) != case[key]['sha256']:
                    raise RuntimeError('fixed3 image/audio SHA mismatch')
    return records


def check_stage(out, mode, records, caption_ref, models, audit, h3, cache=None):
    report = h3.read_json(out / 'run_complete.json')
    require(report, dict(protocol=h3.PROTOCOL, mode=mode, complete=True, completed_ids=IDS,
                         geometry=GEOMETRY, steps=50, seed=42, cfg=1., flow_shift=12.,
                         lora_source=None, old_y_reused=False), 'bad actual renderer stage metadata')
    if report['model_sources'] != models or report['caption_source'] != caption_ref:
        raise RuntimeError('renderer model/caption source changed')
    for key, file in (('source_code', 'h3_video_infer.py'), ('wrapper_source', 'h3_video_only.py')):
        if report[key] != h3.source(RENDER / file):
            raise RuntimeError('renderer code differs')
    for index, record in enumerate(records, 1):
        case = h3.read_json(out / (record['id'] + '.audit.json'))
        require(case, dict(protocol=h3.PROTOCOL, id=record['id'], mode=mode, complete=True,
                           motion_caption=record['motion_caption'], caption_version=record['caption_version']), 'bad case audit')
        if (case['sources'] != dict(caption_json=caption_ref, image=h3.source(record['image_path']))
                or case['condition_cache'] != h3.source((cache or out) / (record['id'] + '.condition.pt'))):
            raise RuntimeError('case caption/image/cache sources differ')
        require(case['packed'], dict(audio_pos_numel=0, tag2_count=0), 'audio packed into H3')
        guard = case['audio_guard']
        require(guard, dict(audio_vae_is_none=True, scheduler_audio_is_none=True), 'audio H3 modules used')
        if (any(guard['calls'].values()) or case.get('review_music_consumed_by_h3', False)
                or not case['prompt'].endswith('[Shot 1] ' + record['motion_caption'].strip())):
            raise RuntimeError('caption wrapper or audio exclusion differs')
        calls = guard['observed_video_calls']
        if mode == 'video_generation':
            require(calls, dict(text_encoder=0, dit=50 * index, video_vae_decode=index, video_vae_encode=0), 'not actual 50-step decoded H3 video')
            require(case['sampling'], dict(video_scheduler_updates=50, target_audio_generated=False), 'sampling differs')
            if case['video'] != h3.source(out / (record['id'] + '.mp4')) or case['media'] != h3.verify_media(out / (record['id'] + '.mp4'), GEOMETRY):
                raise RuntimeError('actual all-frame media decode differs')
        elif calls['text_encoder'] != index or calls['video_vae_encode'] < index:
            raise RuntimeError('cache not generated with native TC/image VAE')
    if report['audio_guard'] != guard:
        raise RuntimeError('run/case final native module evidence differs')
    return report


def manage(cfg, path, digest, own_sha, deps):
    audit, manager, fixed, ev, h3 = deps
    def check():
        check_sources(cfg, path, digest, own_sha, audit)
        return prerequisites(cfg, audit, manager, fixed, ev, h3)
    records = check()
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    for p in (queue, out):
        if not p.is_absolute() or p != p.resolve() or not p.is_relative_to(HANDOFF) or p == HANDOFF or os.path.lexists(p):
            raise RuntimeError('fresh canonical handoff output required; no overwrite')
    if queue == out or queue.is_relative_to(out) or out.is_relative_to(queue):
        raise RuntimeError('queue/output overlap')
    capture = audit.ref_json(cfg['holder_capture'])
    require(capture, dict(signals_sent=False, other_seven_GPUs_untouched=True), 'unapproved holder capture')
    expected = capture['holder']
    if expected['gpu'] != 0:
        raise RuntimeError('captured holder must be GPU0')
    models = {}
    for mode, cache_only in (('cache', True), ('video', False)):
        models[mode] = h3.weight_sources(WEIGHTS, cache_only, not cache_only)
        if cfg.get('weight_pre_audit') and models[mode] != audit.ref_json(cfg['weight_pre_audit'][mode]):
            raise RuntimeError('current full H3 bytes differ from independently preaudited weights')
    vae = lambda m: [r for r in m['files'] if Path(r['path']).is_relative_to(WEIGHTS / 'video_vae')]
    if vae(models['cache']) != vae(models['video']):
        raise RuntimeError('cache/video VAE full SHA differs')
    queue.mkdir(parents=True, exist_ok=False)
    logfd = os.open(queue / 'manager.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(logfd, 1); os.dup2(logfd, 2); os.close(logfd)
    save = lambda name, value: manager.save_receipt(queue, name, value, 'h3_render')
    locks, entries, holders, owned = [], [], [], dict(child=None, started=False)
    signal.signal(signal.SIGTERM, manager.interrupted); signal.signal(signal.SIGINT, manager.interrupted)
    try:
        check()
        lockfd = audit.open_gpu0_lock(); locks.append(lockfd)
        reserve = Path('/tmp/od_reserve_0')
        old = manager.lease_snapshot(reserve)
        if (old.st_mtime_ns if old else None) != capture['previous_lease_mtime_ns']:
            raise RuntimeError('lease changed since independent holder capture')
        fd = os.pidfd_open(expected['pid'])
        e = dict(gpu=0, uuid=expected['uuid'], fd=fd, expected=expected, reserve=reserve,
                 old=old, leasefd=None, released=False, queue=queue)
        entries.append(e)
        if select.select([fd], [], [], 0)[0]:
            raise RuntimeError('captured holder exited')
        e['identity'] = manager.identity(expected['pid'])
        manager.check_holder(expected, e['identity'], 0, manager.uuid_for(0), manager.apps())
        proc = Path('/proc') / str(expected['pid'])
        e['original_env'] = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if x)
        e['original_cwd'] = os.readlink(proc / 'cwd')
        manager.configure_local_restore(e, cfg)
        check()
        if manager.headroom()['headroom'] < manager.required_budget(cfg):
            raise RuntimeError('insufficient fresh memory headroom')
        manager.acquire_lease(e)
        check()
        if manager.headroom()['headroom'] < manager.required_budget(cfg):
            raise RuntimeError('fresh headroom changed before holder release')
        audit.check_gpu0_lock(lockfd)
        manager.check_holder(expected, manager.identity(expected['pid']), 0, manager.uuid_for(0), manager.apps())
        if select.select([fd], [], [], 0)[0]:
            raise RuntimeError('exact holder exited before authorized pidfd signal')
        manager.release_holder(e)
        if not select.select([fd], [], [], 30)[0]:
            raise RuntimeError('exact holder release pending')
        manager.wait_gpu_cleanup(entries, queue, 'initial_GPU0_cleanup')
        out.mkdir(parents=True, exist_ok=False)
        worker = dict(cfg, cwd=str(RENDER), env=dict(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'))
        for family, captions in records.items():
            family_out = out / family; family_out.mkdir()
            cache, video = family_out / 'cache', family_out / 'video'
            ref = cfg['families'][family]['h3_records']
            for kind, destination in (('cache', cache), ('video', video)):
                check(); audit.check_gpu0_lock(lockfd)
                argv = ['/opt/venv_h3/bin/python', '-B', '-u', str(RENDER / 'h3_video_infer.py'),
                        '--caption-json', ref['path'], '--weights-root', str(WEIGHTS), '--out', str(destination),
                        '--limit', '3', '--steps', '50', '--seed', '42', '--cfg', '1', '--flow-shift', '12']
                argv += ['--cache-only'] if kind == 'cache' else ['--condition-cache', str(cache)]
                phase = family + '.' + kind
                if manager.run_child(worker, argv, phase, entries, queue, save, owned) != 0:
                    raise RuntimeError('renderer nonzero exit')
                manager.wait_gpu_cleanup(entries, queue, phase + '_GPU0_cleanup')
                check()
                report = check_stage(destination, 'condition_cache_only' if kind == 'cache' else 'video_generation',
                                     captions, h3.source(ref['path']), models[kind], audit, h3, cache if kind == 'video' else None)
                audit.check_gpu0_lock(lockfd)
                save(phase + '.complete.json', dict(report=h3.source(destination / 'run_complete.json'), completed_ids=report['completed_ids']))
        save('h3.compare.complete.json', dict(technical_video_complete=True, families=list(records), completed_ids=IDS,
             model_sources=models, human_quality_accepted=False, human_quality=False, winner=None,
             config_sha256=digest, manager_sha256=own_sha, other_seven_GPUs_untouched=True))
    except BaseException as error:
        save('failure.json', dict(error=repr(error), partial_outputs_preserved=True,
                                 worker_started=owned['started'], unknown_jobs_not_signaled=True))
    finally:
        manager.finish_resources(entries, locks, queue, save, holders, owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--manager-sha', required=True)
    args = parser.parse_args()
    deps = bootstrap()
    audit, manager, fixed, ev, h3 = deps
    cfg = audit.ref_json(dict(path=str(args.config), sha256=args.config_sha))
    check_sources(cfg, args.config, args.config_sha, args.manager_sha, audit)
    prerequisites(cfg, audit, manager, fixed, ev, h3)
    if args.action == 'capture':
        print(json.dumps(audit.capture_gpu0(dict(node='N2', family='music_flamingo', host=cfg['host']), manager), indent=2))
    else:
        manage(cfg, args.config, args.config_sha, args.manager_sha, deps)


if __name__ == '__main__':
    main()
