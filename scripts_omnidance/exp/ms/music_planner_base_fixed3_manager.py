"""Exclusive once-only GPU0 pure-base fixed3 manager (capture never sends signals).

Config schema music-planner-base-fixed3-manager-v1: node/host/family, gpus=[0],
release_authorized=true, queue/out exact base_fixed3_compare_v3 family children,
protocol/final_selection/trained_compare frozen {path,sha256}, same-family
training_launch/full_complete/resources_safe/checkpoint_selection/checkpoint_marker
and fingerprint from trained_compare, expected_base_audit frozen ref to the actual
family fixed3 generation queue integrity_receipt.json, source_sha256 including both new scripts
and pinned helpers, holder from independent capture, and predecessor {complete,
safe, exits:[{path,sha256}]} required at run time. No auto-freeze/retry/test ledger.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
RT = EXP / 'runtime'
HANDOFF = EXP / 'eval_handoff_clean_v11'
OUTPUT = HANDOFF / 'base_fixed3_compare_v3'
AUDIT_SOURCE = RT / 'planner_eval_manager_v3.py'
AUDIT_SHA = 'c387edfd246fd25ab3ac73d00bffe2f41d5f1450cba15df9509b53192ac63c9d'
PRODUCER = Path(__file__).resolve().with_name('music_planner_base_fixed3.py')
SCOPE = {'N2': ('28.216.17.70', 'music_flamingo'),
         'N4': ('28.216.19.143', 'omni_captioner'),
         'N7': ('28.216.18.144', 'moss')}
PYTHON = '/opt/venv_music_planner_v1/bin/python'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers(cfg):
    if sha(AUDIT_SOURCE) != AUDIT_SHA:
        raise RuntimeError('frozen audit helper SHA drift')
    expected = cfg['source_sha256'].get(str(PRODUCER))
    if not expected or sha(PRODUCER) != expected:
        raise RuntimeError('independently pinned new producer required')
    audit = load_module('base_fixed3_frozen_audit', AUDIT_SOURCE)
    audit.regular(AUDIT_SOURCE)
    audit.regular(PRODUCER)
    producer = load_module('base_fixed3_producer', PRODUCER)
    manager, fixed, ev = audit.load_helpers()
    return audit, manager, fixed, ev, producer


def validate_scope(cfg):
    if (cfg.get('schema') != 'music-planner-base-fixed3-manager-v1'
            or SCOPE.get(cfg.get('node')) != (cfg.get('host'), cfg.get('family'))
            or cfg.get('gpus') != [0] or any(type(g) is not int for g in cfg['gpus'])
            or cfg.get('release_authorized') is not True):
        raise RuntimeError('only authorized N2/Flamingo N4/Captioner N7/MOSS GPU0 allowed')
    family = cfg['family']
    for key, expected in [('queue', OUTPUT / (family + '_queue')), ('out', OUTPUT / (family + '_out'))]:
        path = Path(cfg[key])
        if path != expected or path != path.resolve() or path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
            raise RuntimeError('fresh canonical exact family output/queue required; no overwrite')
    if cfg.get('human_quality_accepted') is True or cfg.get('test_used') is True:
        raise RuntimeError('base fixed3 cannot claim test or quality acceptance')
    if 'holder' in cfg and cfg['holder'].get('gpu') != 0:
        raise RuntimeError('holder must identify GPU0')
    expected_refs = dict(training_launch=RT / (family + '_launch_clean_v11.json'),
        full_complete=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json',
        resources_safe=EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json',
        checkpoint_selection=EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json')
    for key, path in expected_refs.items():
        if cfg[key]['path'] != str(path):
            raise RuntimeError('wrong training receipt path: ' + key)


def check_sources(cfg, path, digest, own_sha, audit, producer):
    if sha(audit.regular(path)) != digest or sha(audit.regular(__file__)) != own_sha:
        raise RuntimeError('independently pinned manager/config drift')
    required = {**audit.PINNED, **producer.PINNED,
                str(AUDIT_SOURCE): AUDIT_SHA, str(Path(__file__).resolve()): own_sha}
    if str(PRODUCER) not in cfg['source_sha256']:
        raise RuntimeError('producer source pin absent')
    for source, expected in required.items():
        if cfg['source_sha256'].get(source) != expected:
            raise RuntimeError('missing/conflicting helper pin: ' + source)
    producer.check_sources(cfg, path, digest, cfg['source_sha256'][str(PRODUCER)])


def predecessor_paths(cfg):
    if cfg['node'] == 'N2':
        queue = HANDOFF / 'h3_compare_v1/queue'
        return queue / 'h3.compare.complete.json', queue / 'resources_safe.json', [
            queue / (family + '.' + kind + '.exit.json')
            for family in ('music_flamingo', 'moss', 'omni_captioner') for kind in ('cache', 'video')], 1
    queue = HANDOFF / 'parallel_final_test_v2' / (cfg['family'] + '_test_queue')
    return queue / 'eval.complete.json', queue / 'resources_safe.json', [queue / 'test.exit.json'], 8


def check_predecessor(cfg, producer):
    required_complete, required_safe, required_exits, count = predecessor_paths(cfg)
    refs = cfg.get('predecessor', {})
    if (refs.get('complete', {}).get('path') != str(required_complete)
            or refs.get('safe', {}).get('path') != str(required_safe)
            or {r.get('path') for r in refs.get('exits', [])} != {str(p) for p in required_exits}
            or len(refs.get('exits', [])) != len(required_exits)):
        raise RuntimeError('exact related current predecessor complete/safe/exit0 required')
    complete, safe = producer.ref_json(refs['complete']), producer.ref_json(refs['safe'])
    if required_complete.parent.joinpath('failure.json').exists():
        raise RuntimeError('predecessor failure receipt present')
    if (any(safe.get(k) is not True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
                                             'managed_parent_wait', 'unknown_jobs_not_signaled'))
            or safe.get('admitted_gpu_count') != count):
        raise RuntimeError('predecessor resources not proven safe')
    for ref in refs['exits']:
        value = producer.ref_json(ref)
        if type(value.get('exit_code')) is not int or value['exit_code'] != 0:
            raise RuntimeError('predecessor real worker exit0 missing')
    if cfg['node'] == 'N2':
        if (complete.get('technical_video_complete') is not True
                or set(complete.get('families', [])) != set(producer.FAMILIES)
                or complete.get('completed_ids') != producer.IDS
                or complete.get('human_quality_accepted') is True):
            raise RuntimeError('all fixed3 H3 stages must finish before N2 base')
    elif (complete.get('technical_collection_complete') is not True
            or complete.get('complete_native_GPU_outputs_verified') is not True
            or complete.get('source_count') != 500 or complete.get('worker_exit_code') != 0
            or complete.get('final_selection') != cfg['final_selection']):
        raise RuntimeError('related family final test must finish before base')
    return refs


def claim_path(cfg):
    return HANDOFF / ('base_fixed3_once_v3_' + cfg['family'] + '.json')


def claim_once(cfg, producer):
    path = claim_path(cfg)
    producer.new_json(path, dict(schema='music-planner-base-fixed3-once-v1', family=cfg['family'],
        node=cfg['node'], out=cfg['out'], queue=cfg['queue'], protocol=cfg['protocol'],
        trained_compare=cfg['trained_compare'], pid=os.getpid(), claimed_unix=time.time(),
        retained_on_failure=True, automatic_retry=False, test_once_ledger_untouched=True))
    return path


def start_heartbeat(entries, manager):
    stop, errors = threading.Event(), []
    def heartbeat():
        while not stop.wait(5):
            try:
                manager.refresh_leases(entries)
            except BaseException as exc:
                errors.append(exc)
                break
    thread = threading.Thread(target=heartbeat, name='owned-base-fixed3-lease-heartbeat', daemon=True)
    thread.start()
    return stop, errors, thread


def stop_heartbeat(watch):
    if watch is not None:
        watch[0].set()
        watch[2].join()


def check_outputs(cfg, data, producer):
    out = Path(cfg['out'])
    manifest = producer.ref_json(producer.file_ref(out / 'manifest.json'))
    raw = [json.loads(line) for line in producer.regular(out / 'raw.jsonl').read_text().splitlines() if line.strip()]
    if (manifest.get('schema') != 'music-planner-base-fixed3-v1' or manifest.get('family') != cfg['family']
            or manifest.get('complete') is not True or manifest.get('technical_collection_complete') is not True
            or manifest.get('source_ids') != producer.IDS or manifest.get('device') != 'cuda:0'
            or manifest.get('physical_gpus') != [0] or manifest.get('decoding') != producer.DECODE
            or manifest.get('code') != producer.file_ref(PRODUCER)
            or manifest.get('protocol') != producer.verify_ref(cfg['protocol'])
            or manifest.get('trained_reference') != data['trained_refs']
            or manifest.get('run_config') != data['run_config'] or manifest.get('max_length') != data['max_length']
            or manifest.get('initialization') != 'official_pure_base_no_checkpoint_no_lora'
            or any(manifest.get(k) is not False for k in ('checkpoint_loaded', 'lora_initialized',
                'modules_to_save_initialized', 'test_used', 'video_generated', 'human_quality_accepted',
                'temporal_event_GT_available', 'temporal_accuracy_metrics_computed'))
            or manifest.get('applied_generation_patch_ref') != data['expected_patch']
            or manifest.get('raw') != producer.file_ref(out / 'raw.jsonl')
            or [r['id'] for r in raw] != producer.IDS):
        raise RuntimeError('pure-base actual GPU/native output evidence invalid')
    exit_ref = producer.file_ref(Path(cfg['queue']) / 'base_fixed3.exit.json')
    actual_exit = producer.ref_json(exit_ref)
    if type(actual_exit.get('exit_code')) is not int or actual_exit['exit_code'] != 0:
        raise RuntimeError('real base_fixed3 worker exit0 required')
    for row, record, trained in zip(data['protocol']['fixed3'], raw, data['trained_raw']):
        if record.get('audio_ref') != row['audio_ref'] or record.get('image_ref') != row['image_ref']:
            raise RuntimeError('base raw audio binding differs')
        result = record.get('native_output')
        stop = producer.validate_result(result)
        if (result.get('input_tokens') != trained['native_output']['input_tokens']
                or result.get('eos_token_ids') != trained['native_output']['eos_token_ids']):
            raise RuntimeError('base/trained effective EOS or native input length differs')
        count, finite = record.get('actual_forward_count'), record.get('finite_native_logits_forward_count')
        if (record.get('failure') is not None or record.get('raw_text') != result['raw_text']
                or record.get('new_token_ids') != result['new_token_ids']
                or record.get('eos_token_ids') != result['eos_token_ids'] or record.get('stop_reason') != stop
                or record.get('generation_budget') != 2048 or record.get('generated_tokens') != len(result['new_token_ids'])
                or type(count) is not int or count < 1 or type(finite) is not int or finite != count
                or record.get('native_logits_all_finite') is not True):
            raise RuntimeError('native pre-argmax finite/token evidence invalid')
    return dict(manifest=producer.file_ref(out / 'manifest.json'), raw=producer.file_ref(out / 'raw.jsonl'),
        source_ids=producer.IDS, family=cfg['family'], technical_collection_complete=True,
        complete_native_GPU_outputs_verified=True, worker_exit_code=actual_exit['exit_code'], worker_exit=exit_ref,
        diagnostic_count=sum(bool(r.get('quality_diagnostics')) for r in raw),
        diagnostics_do_not_trigger_retry=True, human_quality_accepted=False,
        temporal_event_accuracy_not_measured=True, automatic_retry=False)


def check_base_expected(actual, expected):
    def contents(value):
        return {r['path']: (r['sha256'], r['size_bytes']) for r in value['files']}
    if (actual.get('current_full_content_SHA_and_official_digest_verified') is not True
            or expected.get('current_full_content_SHA_and_official_digest_verified') is not True
            or contents(actual) != contents(expected)
            or actual['identity']['official_stage']['revision'] != expected['identity']['official_stage']['revision']):
        raise RuntimeError('current full base SHA differs from independently frozen prior official base identity')


def manage(cfg, path, digest, own_sha, deps):
    audit, manager, fixed, ev, producer = deps
    validate_scope(cfg)
    check_sources(cfg, path, digest, own_sha, audit, producer)
    manager.stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    check_predecessor(cfg, producer)
    launch, result = audit.check_completed(cfg, manager)
    protocol = audit.check_protocol(cfg, ev)
    checkpoint, identity, selected = audit.selected_checkpoint(cfg, result, protocol, fixed, ev, full_hash=False)
    data = producer.bindings(cfg)
    if claim_path(cfg).exists():
        raise FileExistsError('base family permanently claimed; no rerun')
    model = manager.stage.MODEL_BASE / cfg['family']
    base = audit.audit_base(model, cfg['family'], identity, fixed, ev, manager)
    check_base_expected(base, data['expected_base'])
    fresh = audit.capture_gpu0(cfg, manager)
    manager.check_holder(cfg['holder'], fresh['holder'], 0, fresh['holder']['uuid'], manager.apps())
    # O_EXCL family claim precedes all releases. It survives every later failure.
    claim = claim_once(cfg, producer)
    queue = Path(cfg['queue'])
    queue.mkdir(parents=True, exist_ok=False)
    logfd = os.open(queue / 'manager.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.close(logfd)
    def save(name, value):
        manager.save_receipt(queue, name, value, 'planner_base_fixed3')
    locks, entries, holders = [], [], []
    owned, watch = dict(child=None, started=False), None
    def check():
        if watch is not None and watch[1]:
            raise RuntimeError('owned lease heartbeat failed') from watch[1][0]
        check_sources(cfg, path, digest, own_sha, audit, producer)
        check_predecessor(cfg, producer)
        audit.check_completed(cfg, manager)
        for ref in selected['validation_candidates'] + [identity['run_config']]:
            producer.verify_ref(ref)
        fixed.checkpoint_selection_identity(cfg['checkpoint_selection']['path'], cfg['checkpoint_selection']['sha256'],
            cfg['checkpoint_marker']['sha256'], identity, cfg['family'], ev)
        # Current SHA audit happened once above; subsequent stat checks only guard that result.
        _, official_digest, _, _, official_files = manager.stage.read_download(launch['modelref'], cfg['family'])
        marker = producer.ref_json(base['identity']['official_stage']['receipt'])
        manager.stage.check_staged(marker, official_digest, cfg['family'], official_files, model)
        for item in base['files']:
            if manager.stage.file_stat(Path(item['path'])) != item['stat']:
                raise RuntimeError('base changed after current full content SHA')
        producer.verify_ref(cfg['expected_base_audit'])
        for ref in data['processor_refs']:
            producer.verify_ref(ref)
        producer.check_processor_inventory(data['export'], data['processor_refs'])
        for ref in (data['trained_refs'][k] for k in ('manifest', 'raw', 'generation_complete', 'generation_resources_safe', 'generation_exit')):
            producer.verify_ref(ref)
        if watch is not None:
            if watch[1]:
                raise RuntimeError('owned lease heartbeat failed during source check') from watch[1][0]
            manager.refresh_leases(entries)
    save('integrity_receipt.json', dict(schema='music-planner-base-fixed3-audit-v1', family=cfg['family'],
        model=str(model), config_sha256=digest, manager_sha256=own_sha, protocol=cfg['protocol'],
        base=base, selected_checkpoint_reference_only=True, checkpoint_identity=identity,
        trained_compare=cfg['trained_compare'], expected_base_audit=cfg['expected_base_audit'],
        predecessor=cfg['predecessor'], once_claim=str(claim),
        human_quality_accepted=False, real_GPU_generation_verified=False))
    signal.signal(signal.SIGTERM, manager.interrupted)
    signal.signal(signal.SIGINT, manager.interrupted)
    try:
        check()
        lockfd = audit.open_gpu0_lock()
        locks.append(lockfd)
        expected = cfg['holder']
        reserve = Path('/tmp/od_reserve_0')
        old = manager.lease_snapshot(reserve)
        fd = os.pidfd_open(expected['pid'])
        e = dict(gpu=0, uuid=expected['uuid'], fd=fd, expected=expected, reserve=reserve,
                 old=old, leasefd=None, released=False, original_env=None, original_cwd=None, queue=queue)
        entries.append(e)
        if select.select([fd], [], [], 0)[0]:
            raise RuntimeError('exact authorized holder exited')
        e['identity'] = manager.identity(expected['pid'])
        manager.check_holder(expected, e['identity'], 0, manager.uuid_for(0), manager.apps())
        proc = Path('/proc') / str(expected['pid'])
        e['original_env'] = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if x)
        e['original_cwd'] = os.readlink(proc / 'cwd')
        manager.configure_local_restore(e, cfg)
        check()
        if manager.headroom()['headroom'] < manager.required_budget(cfg):
            raise RuntimeError('fresh headroom below native safety admission')
        manager.acquire_lease(e)
        watch = start_heartbeat(entries, manager)
        if select.select([fd], [], [], 0)[0]:
            raise RuntimeError('holder exited before signal')
        manager.check_holder(expected, manager.identity(expected['pid']), 0, manager.uuid_for(0), manager.apps())
        check()
        audit.check_gpu0_lock(lockfd)
        manager.release_holder(e)
        if not select.select([fd], [], [], 30)[0]:
            raise RuntimeError('authorized holder did not exit')
        manager.wait_gpu_cleanup(entries, queue, 'waiting_GPU0_cleanup')
        check()
        integrity = producer.file_ref(queue / 'integrity_receipt.json')
        argv = [PYTHON, '-u', str(PRODUCER), 'generate', '--config', str(path), '--config-sha', digest,
                '--producer-sha', cfg['source_sha256'][str(PRODUCER)],
                '--audit-path', integrity['path'], '--audit-sha', integrity['sha256']]
        worker = dict(cfg, cwd=str(PRODUCER.parent), env=dict(launch['env'], CUDA_VISIBLE_DEVICES='0'))
        save('admission.json', dict(gpus=[0], holder=e['identity'], argv=argv,
            source_sha256=cfg['source_sha256'], config_sha256=digest, other_seven_GPUs_untouched=True))
        check()
        returncode = manager.run_child(worker, argv, 'base_fixed3', entries, queue, save, owned)
        if type(returncode) is not int or returncode != 0:
            raise RuntimeError('base_fixed3 actual worker return code is not exit0: ' + repr(returncode))
        check()
        save('base_fixed3.complete.json', check_outputs(cfg, data, producer))
    except BaseException as exc:
        save('failure.json', dict(error=repr(exc), worker_started=owned['started'],
            raw_and_failed_outputs_preserved=True, unknown_jobs_not_signaled=True,
            family_once_claim_retained=True, automatic_retry=False))
    finally:
        stop_heartbeat(watch)
        manager.finish_resources(entries, locks, queue, save, holders, owned)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'capture', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--manager-sha', required=True)
    args = parser.parse_args(argv)
    # Pin the caller config before reading or importing caller-selected sources.
    if sha(args.config) != args.config_sha or sha(__file__) != args.manager_sha:
        raise RuntimeError('manager/config SHA drift')
    cfg = json.loads(args.config.read_text())
    validate_scope(cfg)
    deps = load_helpers(cfg)
    audit, manager, fixed, ev, producer = deps
    check_sources(cfg, args.config, args.config_sha, args.manager_sha, audit, producer)
    if args.action == 'run':
        manage(cfg, args.config, args.config_sha, args.manager_sha, deps)
    else:
        audit.check_completed(cfg, manager, local=args.action == 'capture')
        audit.check_protocol(cfg, ev)
        producer.bindings(cfg)
        if args.action == 'capture':
            check_predecessor(cfg, producer)
            print(json.dumps(audit.capture_gpu0(cfg, manager), indent=2))
        else:
            print(json.dumps(dict(CPU_only=True, signals_sent=False, approved_for_gpu=False,
                family=cfg['family'], source_ids=producer.IDS, predecessor_checked=False)))


if __name__ == '__main__':
    main()
