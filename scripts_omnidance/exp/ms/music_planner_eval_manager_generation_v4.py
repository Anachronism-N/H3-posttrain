"""Post-completion GPU0-only handoff. No deployment, training, or automatic test retry.

Launch config schema: music-planner-eval-manager-v1; node/family/host, gpus=[0],
release_authorized=true, job=val_panel|fixed3|test, queue/out (fresh handoff children),
fingerprint, holder (from capture), source_sha256, and independently frozen refs
{path, sha256} named training_launch, full_complete, resources_safe, protocol,
checkpoint_selection (actual best.json), checkpoint_marker. Test additionally needs
final_selection (shared final_selection.json). No current-value auto-freeze exists.
Capture uses the same config except holder is not required; writes only stdout.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
RT = EXP / 'runtime'
FROZEN = RT / 'source_v11'
DATA = EXP.parent / 'music_planner_fixed2s_baseline_20261025_clean_v1'
HANDOFF = EXP / 'eval_handoff_clean_v11'
MANAGER = RT / 'planner_node_manager_clean_v11.py'
STAGE = RT / 'stage_models.py'
HOLDER_SOURCE = ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py'
GPU0_LOCK = Path('/tmp/ms12k_gpu_0.lock')
PINNED = {
    str(HOLDER_SOURCE): '814d796dec843d9a3e255fd7085d9ab6d105275905cfa604e03b474c922f1c66',
    str(MANAGER): '9dc57d2af5026ad73ff8397d588b347c38d021475854fdf50e38d8b89b0ea6bf',
    str(STAGE): 'c40037d2da67848b888c1dd729d72743fcdc17001d0fb803fb37e47e6f8b907d',
    str(FROZEN / 'music_planner_train.py'): '9583342d51c3f9b08a1c07f57958c102313364e24b9ed92620e49901fc20fae3',
    str(FROZEN / 'music_planner_adapters.py'): '713dc3eb89408396bbae35a5e2957cb65b212a1b55276a728779ac2acf5fa4d1',
    str(FROZEN / 'music_planner_eval.py'): '55ca82f4ec117ffa05087e7dba54c1787de15ce6522ab0cd69e95f8f1270bea6',
    str(FROZEN / 'music_planner_fixed3_eval.py'): '3a0f69cb46df350282a1ea4bc454490936544d678f5471446e4c92addee8058c',
}
MOSS_GENERATION_PATCH = RT / 'eval_generation_patches_v1/music_planner_moss_generation_patch.py'
MOSS_GENERATION_PATCH_SHA = 'c7b2ed3c57b1d7fdb9f19d33bfb35f0015e417f48f7981f6482f7002b16e6dc0'
PINNED[str(MOSS_GENERATION_PATCH)] = MOSS_GENERATION_PATCH_SHA

SCOPE = {'N2': ('28.216.17.70', 'music_flamingo'),
         'N4': ('28.216.19.143', 'omni_captioner'),
         'N7': ('28.216.18.144', 'moss')}
DECODE = dict(seed=42, do_sample=False, max_new_tokens=2048,
              native_training_prefix_unchanged=True, extra_instruction=False,
              raw_output_cleaning=False, GT_length_truncation=False, added_penalties=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('absolute nonsymlink path required: ' + str(path))
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError('regular file required: ' + str(path))
    return path


def ref_json(ref):
    path = regular(ref['path'])
    if sha(path) != ref['sha256']:
        raise RuntimeError('independently frozen receipt SHA changed: ' + str(path))
    return json.loads(path.read_text())


def new_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_helpers():
    # Hash before executing any imported code, including the stage import side effect.
    for path, expected in PINNED.items():
        if sha(regular(path)) != expected:
            raise RuntimeError('frozen helper SHA changed: ' + path)
    sys.path.insert(0, str(RT))
    if 'stage_models' in sys.modules:
        if Path(sys.modules['stage_models'].__file__).resolve() != STAGE:
            raise RuntimeError('foreign stage import')
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    manager = load('eval_resource_manager_v10', MANAGER)
    fixed = load('eval_fixed3_v10', FROZEN / 'music_planner_fixed3_eval.py')
    return manager, fixed, fixed.frozen_eval()


def validate_scope(cfg):
    if SCOPE.get(cfg['node']) != (cfg['host'], cfg['family']):
        raise RuntimeError('wrong node/family/host')
    if cfg.get('schema') != 'music-planner-eval-manager-v1':
        raise RuntimeError('wrong eval manager schema')
    if cfg.get('gpus') != [0] or any(type(g) is not int for g in cfg['gpus']):
        raise RuntimeError('only physical GPU0 is authorized')
    if cfg.get('release_authorized') is not True:
        raise RuntimeError('explicit GPU0 holder release authorization required')
    if cfg.get('job') not in ('val_panel', 'fixed3', 'test'):
        raise RuntimeError('unknown eval job')
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    for path in (queue, out):
        if (not path.is_absolute() or path != path.resolve() or path == HANDOFF
                or not path.is_relative_to(HANDOFF) or path.exists() or path.is_symlink()):
            raise RuntimeError('fresh canonical handoff child required')
    if queue == out or queue.is_relative_to(out) or out.is_relative_to(queue):
        raise RuntimeError('queue/output overlap')
    family = cfg['family']
    exact = {'training_launch': RT / (family + '_launch_clean_v11.json'),
             'full_complete': EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json',
             'resources_safe': EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json',
             'protocol': HANDOFF / 'protocol.json',
             'checkpoint_selection': EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json'}
    for key, path in exact.items():
        if cfg[key]['path'] != str(path):
            raise RuntimeError('wrong frozen path: ' + key)
    if 'holder' in cfg and cfg['holder'].get('gpu') != 0:
        raise RuntimeError('only exact GPU0 holder identity is admitted')
    if cfg.get('human_quality_accepted') is True:
        raise RuntimeError('CPU/resource handoff cannot claim human acceptance')


def check_sources(cfg, config_path, config_sha, manager_sha):
    if sha(regular(__file__)) != manager_sha or sha(regular(config_path)) != config_sha:
        raise RuntimeError('new manager/config SHA changed')
    bindings = cfg['source_sha256']
    for path, expected in {**PINNED, str(Path(__file__).resolve()): manager_sha}.items():
        if bindings.get(path) != expected:
            raise RuntimeError('missing/conflicting frozen code SHA: ' + path)
    for path, expected in bindings.items():
        if sha(regular(path)) != expected:
            raise RuntimeError('source/input SHA changed: ' + path)
    for key in ('training_launch', 'full_complete', 'resources_safe', 'protocol',
                'checkpoint_selection', 'checkpoint_marker'):
        ref_json(cfg[key])


def check_completed(cfg, manager, local=True):
    launch = ref_json(cfg['training_launch'])
    if (launch['node'], launch['family'], launch['host']) != (cfg['node'], cfg['family'], cfg['host']):
        raise RuntimeError('training launch scope mismatch')
    if local:
        manager.validate_config(launch)
    else:
        for phase in ('smoke', 'resume_smoke', 'full'):
            manager.validate_training_argv(launch, phase)
        if launch['gpus'] != list(range(8)) or launch['env']['CUDA_VISIBLE_DEVICES'] != '0,1,2,3,4,5,6,7':
            raise RuntimeError('remote training recipe differs')
    manager.check_sources(launch, cfg['training_launch']['path'], cfg['training_launch']['sha256'], PINNED[str(MANAGER)])
    full = ref_json(cfg['full_complete'])
    safe = ref_json(cfg['resources_safe'])
    if (any(safe.get(k) is not True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
                                             'managed_parent_wait', 'unknown_jobs_not_signaled'))
            or safe.get('admitted_gpu_count') != 8):
        raise RuntimeError('training resources not safely restored')
    result = full['result']
    if full['result_path'] != launch['full_result'] or sha(regular(full['result_path'])) != full['result_sha256']:
        raise RuntimeError('full.complete does not bind actual full result')
    if json.loads(Path(full['result_path']).read_text()) != result:
        raise RuntimeError('full result payload changed')
    required = dict(schema='music-planner-outputs-v1', family=cfg['family'], mode='full',
                    world_size=8, test_selection_performed=False, optimizer_updates=670,
                    epoch_budget=2, epochs_completed=2, source_exposure_budget=21426,
                    source_exposures=21426, full_budget_completed=True, checkpoint_complete=True,
                    train_count=10713, val_count=490, global_batch=32, starting_update=3,
                    invocation_optimizer_updates=667, initialization='official_base_then_matching_checkpoint',
                    resume_verified=True, verified_new_process=True, resumed_next_update_performed=True)
    if any(result.get(k) != v or type(result.get(k)) is not type(v) for k, v in required.items()):
        raise RuntimeError('real same-family 670/two-epoch/21426 completion missing')
    if result.get('fingerprint') != cfg['fingerprint']:
        raise RuntimeError('full completion fingerprint mismatch')
    for split in ('train', 'val'):
        if result['source_sha256'][split] != launch['dataset'][split]['sha256']:
            raise RuntimeError('full result dataset differs')
    if any(result.get('cursor', {}).get(k) != v for k, v in
           dict(epoch=2, position=0, update=670, source_exposures=21426).items()):
        raise RuntimeError('full result exact cursor mismatch')
    marker_path = Path(result['checkpoint']['path'])
    if marker_path != Path(launch['full_result']).parent / 'checkpoint-0000670/trainer_state.json':
        raise RuntimeError('full completion checkpoint path mismatch')
    marker = ref_json(result['checkpoint'])
    if (marker.get('complete') is not True or marker.get('world_size') != 8 or marker.get('zero3') is not True
            or marker.get('cursor') != result['cursor'] or marker.get('fingerprint') != cfg['fingerprint']
            or marker.get('files') != result.get('checkpoint_files')):
        raise RuntimeError('full checkpoint marker differs from actual training receipt')
    config = ref_json(result['run_config'])
    if result['run_config']['path'] != str(Path(launch['full_result']).parent / 'run_config.json'):
        raise RuntimeError('training run config path mismatch')
    if config.get('fingerprint') != cfg['fingerprint']:
        raise RuntimeError('actual training run config fingerprint differs')
    return launch, result


def checkpoint_snapshot(path):
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise RuntimeError('checkpoint tree symlink forbidden')
    values = {}
    for item in path.rglob('*'):
        if item.is_file():
            st = regular(item).stat()
            values[str(item)] = dict(size=st.st_size, device=st.st_dev, inode=st.st_ino,
                                    mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
    return values


def selected_checkpoint(cfg, result, protocol, fixed, ev, full_hash=True):
    selection = ref_json(cfg['checkpoint_selection'])
    path = Path(selection['checkpoint'])
    train = Path(cfg['checkpoint_selection']['path']).parent
    if path not in {train / 'checkpoint-0000335', train / 'checkpoint-0000670'}:
        raise RuntimeError('best must be actual 335/670 complete full-val checkpoint')
    if cfg['checkpoint_marker']['path'] != str(path / 'trainer_state.json'):
        raise RuntimeError('independent marker path differs from best')
    marker = ref_json(cfg['checkpoint_marker'])
    if marker.get('world_size') != 8 or marker.get('zero3') is not True:
        raise RuntimeError('selected checkpoint must be eight-rank ZeRO3')
    before = checkpoint_snapshot(path) if full_hash else None
    identity = ev.checkpoint_identity(path, cfg['family'], DATA)
    if identity['fingerprint'] != cfg['fingerprint'] or identity['run_config'] != result['run_config']:
        raise RuntimeError('selected checkpoint does not bind the actual full training run')
    source = fixed.checkpoint_source_identity(identity, protocol)
    receipt = fixed.checkpoint_selection_identity(cfg['checkpoint_selection']['path'],
        cfg['checkpoint_selection']['sha256'], cfg['checkpoint_marker']['sha256'], identity, cfg['family'], ev)
    validation = json.loads((path / 'validation.json').read_text())
    if (validation.get('source_count') != 490 or selection.get('source_count') != 490
            or identity['cursor']['update'] not in (335, 670)):
        raise RuntimeError('selection is not actual within-family full val490')
    candidates = []
    losses = []
    for update in (335, 670):
        candidate = train / ('checkpoint-%07d' % update)
        candidate_marker = json.loads(regular(candidate / 'trainer_state.json').read_text())
        candidate_val = json.loads(regular(candidate / 'validation.json').read_text())
        nll = candidate_val.get('nll')
        if (candidate_marker.get('complete') is not True or candidate_marker.get('fingerprint') != cfg['fingerprint']
                or candidate_marker.get('world_size') != 8 or candidate_marker.get('zero3') is not True
                or candidate_marker.get('cursor', {}).get('update') != update
                or candidate_val.get('source_count') != 490 or candidate_val.get('test_used') is not False
                or candidate_val.get('comparison_scope') != 'within-family-tokenizer-only'
                or type(candidate_val.get('assistant_tokens')) is not int or candidate_val['assistant_tokens'] <= 0
                or type(nll) not in (float, int) or not math.isfinite(nll) or nll < 0):
            raise RuntimeError('both actual epoch checkpoints require complete within-family fullval490')
        losses.append(nll)
        candidates.extend([ev.file_ref(candidate / 'trainer_state.json'), ev.file_ref(candidate / 'validation.json')])
    expected_best = train / ('checkpoint-%07d' % (335 if losses[0] <= losses[1] else 670))
    if path != expected_best:
        raise RuntimeError('best.json is not the actual within-family epoch-validation minimum')
    full = fixed.checkpoint_content_identity(path, identity, ev) if full_hash else None
    snapshot = checkpoint_snapshot(path) if full_hash else None
    if full_hash and snapshot != before:
        raise RuntimeError('checkpoint changed during content audit')
    return path, identity, dict(selection=receipt, source=source, content=full, stat_snapshot=snapshot,
                                validation_candidates=candidates)


def check_protocol(cfg, ev):
    protocol = ref_json(cfg['protocol'])
    if (protocol.get('schema') != 'music_planner_fixed3_protocol_v1'
            or protocol.get('decoding') != DECODE or protocol['data']['path'] != str(DATA)):
        raise RuntimeError('wrong fixed clean data/audio-only decoding protocol')
    refs = protocol['code'] + protocol['historical_refs'] + list(protocol['data']['splits'].values())
    refs += [protocol['data']['summary'], protocol['training_templates'], protocol['val32']['selection_ref']]
    for ref in refs:
        regular(ref['path'])
        if ev.file_ref(ref['path']) != ref:
            raise RuntimeError('protocol source/data changed: ' + ref['path'])
    for name, expected in PINNED.items():
        if name in (str(MANAGER), str(HOLDER_SOURCE), str(MOSS_GENERATION_PATCH)):
            continue
        if {r['path']: r['sha256'] for r in protocol['code']}.get(name) != expected:
            raise RuntimeError('protocol frozen code mismatch')
    return protocol


def audit_base(model, family, identity, fixed, ev, manager):
    base = fixed.base_stage_identity(model, family, identity, ev)
    stage = manager.stage
    marker = json.loads(Path(base['official_stage']['receipt']['path']).read_text())
    _, digest, _, _, files = stage.read_download(
        dict(path=marker['download_receipt'], sha256=marker['download_receipt_sha256']), family)
    stage.check_staged(marker, digest, family, files, model)
    if model.is_symlink() or any(p.is_symlink() for p in model.rglob('*')):
        raise RuntimeError('base tree symlink forbidden')
    audited = []
    for name, official in sorted(files.items()):
        path = regular(model / name)
        before = stage.file_stat(path)
        content, git = hashlib.sha256(), hashlib.sha1()
        git.update(('blob %d\0' % before['size']).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(stage.BLOCK), b''):
                content.update(block)
                git.update(block)
            stage.drop_cache(stream)
        actual = content.hexdigest()
        if (stage.file_stat(path) != before or actual != marker['files'][name]['sha256']
                or (official['sha256'] and actual != official['sha256'])
                or (official['git_blob'] and git.hexdigest() != official['git_blob'])):
            raise RuntimeError('current official local base bytes differ: ' + name)
        audited.append(dict(path=str(path), sha256=actual, size_bytes=before['size'], stat=before))
    stage.check_staged(marker, digest, family, files, model)
    return dict(identity=base, files=audited, current_full_content_SHA_and_official_digest_verified=True,
                local_weight_hash_passes_this_audit=1, old_stage_receipt_modified=False)


def check_gpu0_lock(fd):
    path = regular(GPU0_LOCK)
    current, opened = path.lstat(), os.fstat(fd)
    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_size
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
        raise RuntimeError('existing GPU0 lock replaced/type/owner/content changed')


def open_gpu0_lock():
    fd = os.open(regular(GPU0_LOCK), os.O_RDWR | os.O_NOFOLLOW)
    try:
        check_gpu0_lock(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_gpu0_lock(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def capture_gpu0(cfg, manager):
    manager.stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    uuid = manager.uuid_for(0)
    running = manager.apps()
    candidates = [r for r in running if r['uuid'] == uuid]
    if len(candidates) != 1:
        raise RuntimeError('GPU0 must have exactly one authorized project holder')
    current = manager.identity(candidates[0]['pid'])
    item = dict(current, gpu=0, uuid=uuid)
    manager.check_holder(item, current, 0, uuid, running)
    fd = open_gpu0_lock()
    os.close(fd)
    lease = manager.lease_snapshot(Path('/tmp/od_reserve_0'))
    manager.check_holder(item, manager.identity(item['pid']), 0, manager.uuid_for(0), manager.apps())
    return dict(holder=item, previous_lease_mtime_ns=lease.st_mtime_ns if lease else None,
                captured_unix=time.time(), signals_sent=False, other_seven_GPUs_untouched=True)


def check_final_selection(cfg, manager, fixed, ev, protocol):
    ref = cfg['final_selection']
    if ref['path'] != str(HANDOFF / 'final_selection.json'):
        raise RuntimeError('test requires the real shared final_selection.json')
    final = ref_json(ref)
    if (final.get('schema') != 'music-planner-final-selection-v1' or final.get('test_used') is not False
            or final.get('selection_frozen_before_test') is not True or final.get('decoding') != DECODE
            or final.get('data') != protocol['data'] or final.get('protocol') != cfg['protocol']
            or set(final.get('families', {})) != {v[1] for v in SCOPE.values()}):
        raise RuntimeError('shared three-family test selection is not frozen and identical')
    for node, (host, family) in SCOPE.items():
        row = final['families'][family]
        scope = dict(cfg, **row, node=node, host=host, family=family)
        # Remote training receipts are shared; never invoke remote/local host admission here.
        launch = ref_json(scope['training_launch'])
        if (launch['node'], launch['host'], launch['family']) != (node, host, family):
            raise RuntimeError('final selection launch scope mismatch')
        for key, expected in (('training_launch', RT / (family + '_launch_clean_v11.json')),
                ('full_complete', EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json'),
                ('resources_safe', EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json'),
                ('checkpoint_selection', EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json')):
            if scope[key]['path'] != str(expected):
                raise RuntimeError('final selection wrong family receipt path')
        launch, result = check_completed(scope, manager, local=False)
        for split in ('train', 'val'):
            if launch['dataset'][split]['sha256'] != protocol['data']['splits'][split]['sha256']:
                raise RuntimeError('final selection sameDATA mismatch')
        selected_checkpoint(scope, result, protocol, fixed, ev, full_hash=False)
        if family == cfg['family']:
            for key in ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection',
                        'checkpoint_marker', 'fingerprint'):
                if row[key] != cfg[key]:
                    raise RuntimeError('test config differs from shared own-family selection')
    return final


def claim_test(cfg):
    # Global family key, not output/run key. Keep forever, including worker failure.
    path = HANDOFF / ('test_once_' + cfg['family'] + '.json')
    new_json(path, dict(family=cfg['family'], final_selection=cfg['final_selection'],
                       checkpoint_selection=cfg['checkpoint_selection'], checkpoint_marker=cfg['checkpoint_marker'],
                       out=cfg['out'], claimed_unix=time.time(), pid=os.getpid(),
                       retry_requires_explicit_diagnosis_and_recorded_authorization=True,
                       automatic_retry=False, test_used_for_selection=False))
    return path


def native_worker_argv(cfg, checkpoint, protocol):
    common = ['--family', cfg['family'], '--model', str(Path('/opt/music_planner_models') / cfg['family']),
              '--checkpoint', str(checkpoint)]
    python = '/opt/venv_music_planner_v1/bin/python'
    if cfg['job'] == 'fixed3':
        return [python, '-u', str(FROZEN / 'music_planner_fixed3_eval.py'), 'generate', *common,
                '--checkpoint-selection', cfg['checkpoint_selection']['path'],
                '--checkpoint-selection-sha', cfg['checkpoint_selection']['sha256'],
                '--checkpoint-marker-sha', cfg['checkpoint_marker']['sha256'],
                '--device', 'cuda:0', '--out', cfg['out']]
    argv = [python, '-u', str(FROZEN / 'music_planner_eval.py'), *common, '--data', str(DATA),
            '--mode', 'test' if cfg['job'] == 'test' else 'val', '--limit', '500' if cfg['job'] == 'test' else '32',
            '--seed', '42', '--max-new-tokens', '2048', '--device', 'cuda', '--out', cfg['out']]
    if cfg['job'] == 'val_panel':
        argv += ['--selection', protocol['val32']['selection_ref']['path'], '--counterfactual']
    return argv


def worker_argv(cfg, checkpoint, protocol):
    original = native_worker_argv(cfg, checkpoint, protocol)
    if cfg['family'] != 'moss':
        return original
    prefix = [original[0], '-B', '-u', str(MOSS_GENERATION_PATCH),
        '--frozen-source', str(FROZEN), '--entry', 'fixed3' if cfg['job'] == 'fixed3' else 'eval',
        '--expected-adapter-sha256', PINNED[str(FROZEN / 'music_planner_adapters.py')],
        '--expected-eval-sha256', PINNED[str(FROZEN / 'music_planner_eval.py')],
        '--expected-patch-sha256', MOSS_GENERATION_PATCH_SHA]
    if cfg['job'] == 'fixed3':
        prefix += ['--expected-fixed3-sha256', PINNED[str(FROZEN / 'music_planner_fixed3_eval.py')]]
    return prefix + ['--'] + original[3:]


def verify_generation_patch(cfg, manifest, records, ev):
    if cfg['family'] != 'moss':
        return
    expected = ev.file_ref(MOSS_GENERATION_PATCH)
    if (expected['sha256'] != MOSS_GENERATION_PATCH_SHA or
            manifest.get('requested_generation_patch_ref') != expected or
            manifest.get('applied_generation_patch_ref') != expected):
        raise RuntimeError('MOSS eval-only generation patch not explicitly identified')
    path = Path(cfg['out']) / 'generation_patch.json'
    if manifest.get('generation_patch_receipt_ref') != ev.file_ref(path):
        raise RuntimeError('MOSS generation patch receipt bytes differ')
    receipt = json.loads(path.read_text())
    if (receipt.get('schema') != 'moss_generation_patch_receipt_v1' or
            receipt.get('applied_generation_patch_ref') != expected or
            receipt.get('original_training_identity_preserved') is not True or
            receipt.get('cache_disabled') is not False or
            receipt.get('frozen_source_refs', {}).get('adapter') != ev.file_ref(FROZEN / 'music_planner_adapters.py') or
            receipt.get('frozen_source_refs', {}).get('eval') != ev.file_ref(FROZEN / 'music_planner_eval.py') or
            any(r.get('native_output', {}).get('applied_generation_patch_ref') != expected for r in records)):
        raise RuntimeError('MOSS actual native decode patch provenance mismatch')


def check_worker_outputs(cfg, checkpoint, identity, protocol, ev):
    out = Path(cfg['out'])
    manifest = json.loads(regular(out / 'manifest.json').read_text())
    raw = [json.loads(line) for line in regular(out / 'raw.jsonl').read_text().splitlines() if line.strip()]
    count = 3 if cfg['job'] == 'fixed3' else 500 if cfg['job'] == 'test' else 96
    ids = [r['id'] for r in protocol['fixed3']] if cfg['job'] == 'fixed3' else protocol['val32']['ids']
    if cfg['job'] == 'test':
        splits, _ = ev.validate_data(DATA)
        ids = [r['id'] for r in ev.select_rows(splits['test'], 'test', limit=500, seed=42)]
    expected = [(id_, condition) for id_ in ids for condition in
                (('original', 'mismatched', 'silence') if cfg['job'] == 'val_panel' else ('original',))]
    actual = [(r['id'], r.get('condition', 'original')) for r in raw]
    if (manifest.get('complete') is not True or manifest.get('family') != cfg['family']
            or manifest.get('checkpoint_identity') != identity or len(raw) != count or actual != expected
            or manifest.get('human_quality_accepted') is True):
        raise RuntimeError('worker manifest/membership/checkpoint not a real complete handoff')
    producer = FROZEN / ('music_planner_fixed3_eval.py' if cfg['job'] == 'fixed3' else 'music_planner_eval.py')
    if manifest.get('code') != ev.file_ref(producer):
        raise RuntimeError('worker producer source identity mismatch')
    if cfg['job'] == 'fixed3':
        if (manifest.get('protocol') != ev.file_ref(cfg['protocol']['path'])
                or manifest.get('checkpoint_selection', {}).get('receipt') != ev.file_ref(cfg['checkpoint_selection']['path'])
                or manifest.get('pretrained_GPU_generation_executed') is not True
                or manifest.get('h3_handoff_approved') is not True or manifest.get('device') != 'cuda:0'
                or manifest.get('decoding') != DECODE):
            raise RuntimeError('fixed3 not a completed GPU native handoff')
    else:
        decoding = manifest['decoding']
        if (manifest.get('adapter_code') != ev.file_ref(FROZEN / 'music_planner_adapters.py')
                or manifest.get('data_summary') != protocol['data']['summary']
                or manifest.get('split') != protocol['data']['splits']['test' if cfg['job'] == 'test' else 'val']
                or manifest.get('test_used_for_checkpoint_selection') is not False
                or manifest.get('device') != 'cuda' or manifest['checkpoint'] != str(checkpoint)
                or manifest.get('mode') != ('test' if cfg['job'] == 'test' else 'val')
                or any(decoding.get(k) != DECODE[k] for k in
                       ('seed', 'do_sample', 'max_new_tokens', 'added_penalties', 'GT_length_truncation', 'raw_output_cleaning'))):
            raise RuntimeError('val/test worker device/protocol mismatch')
    diagnostics = []
    stop_counts = dict(budget_truncation=0, unknown_stop=0, empty_raw=0)
    stop_reason_counts = dict(eos=0, native_stop=0, budget_truncation=0, unknown_stop=0)
    for record in raw:
        result = record.get('native_output')
        tokens = record.get('new_token_ids')
        eos = record.get('eos_token_ids')
        text = record.get('raw_text')
        if (record.get('failure') is not None or not isinstance(result, dict)
                or not isinstance(text, str) or result.get('raw_text') != text
                or not isinstance(tokens, list) or not isinstance(eos, list) or not 0 < len(tokens) <= 2048
                or any(type(t) is not int or t < 0 for t in tokens + eos)
                or result.get('new_token_ids') != tokens or result.get('eos_token_ids') != eos
                or record.get('generation_budget') != 2048 or record.get('generated_tokens') != len(tokens)):
            raise RuntimeError('native interface failure/invalid tokens/budget is a technical collection failure; raw retained')
        stop = ev.classify_stop(tokens, eos, 2048, result.get('native_stop', False))
        if record.get('stop_reason') != stop or ('budget_truncation' in record
                and record['budget_truncation'] is not (stop == 'budget_truncation')):
            raise RuntimeError('reported stop evidence differs from actual tokens; raw retained')
        stop_reason_counts[stop] += 1
        reasons = []
        if stop in ('budget_truncation', 'unknown_stop'):
            reasons.append(stop)
            stop_counts[stop] += 1
        if not text.strip():
            reasons.append('empty_raw')
            stop_counts['empty_raw'] += 1
        if cfg['job'] == 'fixed3' and (stop not in ('eos', 'native_stop') or not text.strip()):
            raise RuntimeError('fixed3 incomplete/empty generation is not an effective H3 handoff; raw retained')
        if reasons:
            diagnostics.append(dict(id=record['id'], condition=record.get('condition', 'original'),
                                    stop_reason=stop, reasons=reasons))
    verify_generation_patch(cfg, manifest, raw, ev)
    return dict(manifest=ev.file_ref(out / 'manifest.json'), raw=ev.file_ref(out / 'raw.jsonl'),
                technical_collection_complete=True, complete_native_GPU_outputs_verified=True,
                diagnostic_count=len(diagnostics), diagnostic_counts=stop_counts,
                stop_reason_counts=stop_reason_counts, empty_raw_count=stop_counts['empty_raw'],
                budget_truncation_count=stop_counts['budget_truncation'],
                diagnostics=diagnostics, diagnostics_do_not_trigger_retry=True,
                quality_accepted=False, human_quality_accepted=False)


def manage_eval(cfg, config_path, config_sha, manager_sha):
    validate_scope(cfg)
    manager, fixed, ev = load_helpers()
    manager.stage.validate_scope(cfg['node'], cfg['family'], cfg['host'])
    check_sources(cfg, config_path, config_sha, manager_sha)
    launch, result = check_completed(cfg, manager)
    protocol = check_protocol(cfg, ev)
    checkpoint, identity, selected = selected_checkpoint(cfg, result, protocol, fixed, ev)
    if cfg['job'] == 'test':
        check_final_selection(cfg, manager, fixed, ev, protocol)
        if (HANDOFF / ('test_once_' + cfg['family'] + '.json')).exists():
            raise FileExistsError('family test already claimed; no holder will be released')
    model = manager.stage.MODEL_BASE / cfg['family']
    # Large local model reads occur only here, after real completion, never during preparation.
    base = audit_base(model, cfg['family'], identity, fixed, ev, manager)
    fresh = capture_gpu0(cfg, manager)
    manager.check_holder(cfg['holder'], fresh['holder'], 0, fresh['holder']['uuid'], manager.apps())
    queue = Path(cfg['queue'])
    queue.mkdir(parents=True, exist_ok=False)
    logfd = os.open(queue / 'manager.stdout.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    os.close(logfd)
    def save(name, value):
        manager.save_receipt(queue, name, value, 'planner_eval')
    def check():
        check_sources(cfg, config_path, config_sha, manager_sha)
        check_completed(cfg, manager)
        fixed.checkpoint_selection_identity(cfg['checkpoint_selection']['path'],
            cfg['checkpoint_selection']['sha256'], cfg['checkpoint_marker']['sha256'], identity, cfg['family'], ev)
        for ref in selected['validation_candidates']:
            if ev.file_ref(ref['path']) != ref:
                raise RuntimeError('epoch validation/marker changed after selection audit')
        if ev.file_ref(identity['run_config']['path']) != identity['run_config']:
            raise RuntimeError('training run config changed after audit')
        if checkpoint_snapshot(checkpoint) != selected['stat_snapshot']:
            raise RuntimeError('checkpoint exact inventory/stat changed after content audit')
        if cfg['job'] == 'test':
            check_final_selection(cfg, manager, fixed, ev, protocol)
        _, digest, _, _, files = manager.stage.read_download(launch['modelref'], cfg['family'])
        marker = json.loads(Path(base['identity']['official_stage']['receipt']['path']).read_text())
        manager.stage.check_staged(marker, digest, cfg['family'], files, model)
        for ref in base['files']:
            if manager.stage.file_stat(Path(ref['path'])) != ref['stat']:
                raise RuntimeError('base file stat changed after content audit')
    save('integrity_receipt.json', dict(config_sha256=config_sha, manager_sha256=manager_sha,
        protocol=cfg['protocol'], training_launch=cfg['training_launch'], full_complete=cfg['full_complete'],
        resources_safe=cfg['resources_safe'], checkpoint_identity=identity, checkpoint=selected,
        base=base, human_quality_accepted=False, real_GPU_eval_verified=False))
    locks, entries, holders = [], [], []
    owned = dict(child=None, started=False)
    signal.signal(signal.SIGTERM, manager.interrupted)
    signal.signal(signal.SIGINT, manager.interrupted)
    try:
        check()
        lockfd = open_gpu0_lock()
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
            raise RuntimeError('fresh memory headroom below frozen safety admission')
        manager.acquire_lease(e)
        if select.select([fd], [], [], 0)[0]:
            raise RuntimeError('exact holder exited before signal')
        manager.check_holder(expected, manager.identity(expected['pid']), 0, manager.uuid_for(0), manager.apps())
        check()
        check_gpu0_lock(lockfd)
        manager.release_holder(e)
        if not select.select([fd], [], [], 30)[0]:
            raise RuntimeError('authorized GPU0 holder did not exit')
        manager.wait_gpu_cleanup(entries, queue, 'waiting_GPU0_cleanup')
        check()
        if cfg['job'] == 'test':
            save('test_claim.json', dict(path=str(claim_test(cfg)), final_selection=cfg['final_selection']))
        argv = worker_argv(cfg, checkpoint, protocol)
        worker = dict(cfg, cwd=str(FROZEN), env=dict(launch['env'], CUDA_VISIBLE_DEVICES='0'))
        save('admission.json', dict(gpus=[0], holder=e['identity'], argv=argv, source_sha256=cfg['source_sha256'],
                                   other_seven_GPUs_untouched=True, config_sha256=config_sha))
        check()  # Last pre-job source/config/receipt/selection/stat gate.
        manager.run_child(worker, argv, cfg['job'], entries, queue, save, owned)
        check()
        save('eval.complete.json', check_worker_outputs(cfg, checkpoint, identity, protocol, ev))
    except BaseException as error:
        save('failure.json', dict(error=repr(error), worker_started=owned['started'],
             raw_and_failed_outputs_preserved=True, unknown_jobs_not_signaled=True, automatic_test_retry=False))
    finally:
        # Preserves signal ignoring, exact holder recheck, leases, unlock, and restored-holder parent lifetime.
        manager.finish_resources(entries, locks, queue, save, holders, owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--manager-sha', required=True)
    args = parser.parse_args()
    cfg = ref_json(dict(path=str(args.config), sha256=args.config_sha))
    validate_scope(cfg)
    check_sources(cfg, args.config, args.config_sha, args.manager_sha)
    if args.action == 'run':
        manage_eval(cfg, args.config, args.config_sha, args.manager_sha)
    else:
        manager, _, _ = load_helpers()
        check_completed(cfg, manager)
        print(json.dumps(capture_gpu0(cfg, manager), indent=2))


if __name__ == '__main__':
    main()
