"""Eight independent native test shards; serial CPU loading, concurrent GPU generation.

Only the new parent manager may create --dispatch. No dist collective, sampling,
retry, teacher likelihood, cleaning, or checkpoint selection occurs here.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
WORLD = 8
TEST_COUNT = 500
DECODE = dict(seed=42, do_sample=False, max_new_tokens=2048,
              native_training_prefix_unchanged=True, extra_instruction=False,
              raw_output_cleaning=False, GT_length_truncation=False, added_penalties=False)
PATCH_ENTRIES = {'moss': 'apply_moss_generation_patch',
                 'omni_captioner': 'apply_generation_eos_patch'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('absolute nonsymlink file required: ' + str(path))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('regular file required: ' + str(path))
    return path


def file_ref(path):
    path = regular(path)
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def verify_ref(ref):
    actual = file_ref(ref['path'])
    if any(actual.get(k) != v for k, v in ref.items() if k in ('path', 'sha256', 'size_bytes')):
        raise RuntimeError('explicit frozen SHA changed: ' + ref['path'])
    return actual


def new_json(path, value):
    # Publish only complete JSON; linking is atomic and never overwrites a receipt.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def partition(rows, rank, world=WORLD):
    if type(rank) is not int or not 0 <= rank < WORLD or type(world) is not int or world != WORLD:
        raise RuntimeError('exact eight-rank local partition required')
    ids = [row['id'] for row in rows]
    if len(rows) != TEST_COUNT or len(set(ids)) != TEST_COUNT:
        raise RuntimeError('unchanged unique test500 required')
    return [(index, row) for index, row in enumerate(rows) if index % WORLD == rank]


def pid_start(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19])


def process_alive(receipt):
    try:
        return pid_start(receipt['pid']) == receipt['pid_start_ticks']
    except (FileNotFoundError, ProcessLookupError):
        return False


def identity_fields(dispatch, rank):
    return dict(invocation_nonce=dispatch['invocation_nonce'], rank=rank, world_size=WORLD,
                source_ids_sha256=dispatch['source_ids_sha256'], family=dispatch['family'],
                source_refs=dispatch['source_refs'], checkpoint_identity=dispatch['checkpoint_identity'],
                decoding=DECODE)


def check_identity(receipt, dispatch, rank):
    if any(receipt.get(k) != value for k, value in identity_fields(dispatch, rank).items()):
        raise RuntimeError('wrong invocation/rank/source identity receipt')
    if type(receipt.get('pid')) is not int or type(receipt.get('pid_start_ticks')) is not int:
        raise RuntimeError('missing actual process identity')


def read_receipt(path):
    return json.loads(regular(path).read_text())


def check_failed_ranks(out, dispatch):
    for rank in range(WORLD):
        path = out / ('exit-rank%d.json' % rank)
        if path.exists():
            receipt = read_receipt(path)
            check_identity(receipt, dispatch, rank)
            if receipt.get('exit_code') != 0:
                raise RuntimeError('a real rank failed; no retry or further load')


def wait_previous_loaded(out, dispatch, rank, clock=time.time, sleep=time.sleep):
    if rank == 0:
        return None
    deadline = dispatch['created_unix'] + dispatch['loading_timeout_seconds']
    previous = rank - 1
    while True:
        check_failed_ranks(out, dispatch)
        start_path = out / ('start-rank%d.json' % previous)
        loaded_path = out / ('loaded-rank%d.json' % previous)
        if start_path.exists():
            start = read_receipt(start_path)
            check_identity(start, dispatch, previous)
            if loaded_path.exists():
                loaded = read_receipt(loaded_path)
                check_identity(loaded, dispatch, previous)
                if (loaded['pid'], loaded['pid_start_ticks']) != (start['pid'], start['pid_start_ticks']):
                    raise RuntimeError('previous loaded receipt process changed')
                if loaded.get('model_parameters_on_assigned_GPU') is not True or loaded.get('device') != 'cuda:%d' % previous:
                    raise RuntimeError('previous model has not actually moved to its GPU')
                if not process_alive(start):
                    exit_path = out / ('exit-rank%d.json' % previous)
                    if not exit_path.exists() or read_receipt(exit_path).get('exit_code') != 0:
                        raise RuntimeError('previous loader exited without real successful completion')
                return loaded
            if not process_alive(start):
                raise RuntimeError('previous loader died before GPU-ready receipt')
        elif loaded_path.exists():
            raise RuntimeError('GPU-ready receipt lacks actual start receipt')
        if clock() >= deadline:
            raise RuntimeError('bounded serial CPU loading deadline expired')
        sleep(1)


def check_inventory(dispatch):
    for path, expected in dispatch['checkpoint_stat_snapshot'].items():
        if file_stat(regular(path)) != expected:
            raise RuntimeError('selected checkpoint stat changed after single full-content audit')
    checkpoint = Path(dispatch['checkpoint'])
    inventory = {str(p) for p in checkpoint.rglob('*') if p.is_file()}
    if inventory != set(dispatch['checkpoint_stat_snapshot']) or any(p.is_symlink() for p in checkpoint.rglob('*')):
        raise RuntimeError('selected checkpoint inventory changed')
    for ref in dispatch['base_audit']['files']:
        if file_stat(regular(ref['path'])) != ref['stat']:
            raise RuntimeError('official base stat changed after single full-content audit')
    if dispatch['base_audit'].get('current_full_content_SHA_and_official_digest_verified') is not True:
        raise RuntimeError('current official base full-content audit required')


def file_stat(path):
    st = Path(path).stat()
    return dict(size=st.st_size, device=st.st_dev, inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def load_module(name, ref):
    verify_ref(ref)
    if name in sys.modules:
        raise RuntimeError('isolated frozen-source import required: ' + name)
    spec = importlib.util.spec_from_file_location(name, ref['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_dispatch(dispatch, producer_sha):
    if (dispatch.get('schema') != 'music-planner-parallel-test-dispatch-v1'
            or dispatch.get('world_size') != WORLD or type(dispatch.get('world_size')) is not int
            or dispatch.get('family') not in PATCH_ENTRIES or dispatch.get('decoding') != DECODE
            or dispatch.get('test_used_for_checkpoint_selection') is not False
            or dispatch.get('source_ids_sha256') != ids_sha(dispatch['source_ids'])
            or len(dispatch['source_ids']) != TEST_COUNT or len(set(dispatch['source_ids'])) != TEST_COUNT):
        raise RuntimeError('wrong immutable parallel test500 contract')
    sources = dispatch['source_refs']
    if (sources['producer']['path'] != str(Path(__file__).resolve())
            or sources['producer']['sha256'] != producer_sha
            or sha(regular(__file__)) != producer_sha):
        raise RuntimeError('actual new producer source SHA mismatch')
    # Verify every executable source before importing torch or the trained adapter.
    for ref in sources.values():
        verify_ref(ref)
    if dispatch['generation_patch_entrypoint'] != PATCH_ENTRIES[dispatch['family']]:
        raise RuntimeError('wrong explicitly frozen generation patch entrypoint')
    config = json.loads(regular(dispatch['config']['path']).read_text())
    verify_ref(dispatch['config'])
    for key in ('family', 'fingerprint', 'generation_patch_entrypoint'):
        if config[key] != dispatch[key]:
            raise RuntimeError('dispatch differs from frozen parent config')
    for key in ('protocol', 'final_selection', 'checkpoint_selection', 'checkpoint_marker'):
        verify_ref(dispatch[key])
        if config[key] != dispatch[key]:
            raise RuntimeError('dispatch selection receipt differs from parent config')
    trained = dispatch['checkpoint_identity']['training_config']
    if (trained.get('adapter_source_sha256') != sources['frozen_adapter_source']['sha256']
            or trained.get('fingerprint') != dispatch['fingerprint']
            or trained.get('family') != dispatch['family']):
        raise RuntimeError('no migration of original trained adapter/fingerprint allowed')
    for ref in dispatch['data_refs'].values():
        verify_ref(ref)
    check_inventory(dispatch)


def validate_record(record, ev):
    result, tokens, eos = record.get('native_output'), record.get('new_token_ids'), record.get('eos_token_ids')
    if record.get('failure') is not None:
        raise RuntimeError('native technical interface failure; preserve original record')
    if (not isinstance(result, dict) or not isinstance(record.get('raw_text'), str)
            or not isinstance(tokens, list) or not isinstance(eos, list) or not 0 < len(tokens) <= 2048
            or any(type(t) is not int or t < 0 for t in tokens + eos)
            or result.get('raw_text') != record['raw_text'] or result.get('new_token_ids') != tokens
            or result.get('eos_token_ids') != eos or record.get('generation_budget') != 2048
            or record.get('generated_tokens') != len(tokens)
            or record.get('condition') != 'original' or record.get('audio_transform') != 'none'
            or record.get('audio_source_id') != record.get('id')
            or record.get('audio_source_group_id') != record.get('source_group_id')
            or record.get('audio_ref', {}).get('sha256') != record.get('audio_sha256')
            or 'teacher_forced' in record or 'teacher_forced_failure' in record):
        raise RuntimeError('invalid native token/text/budget/audio-only test evidence')
    if (type(record.get('actual_forward_count')) is not int or record['actual_forward_count'] <= 0
            or type(record.get('finite_native_logits_forward_count')) is not int
            or record['finite_native_logits_forward_count'] != record['actual_forward_count']
            or record.get('native_logits_all_finite') is not True):
        raise RuntimeError('missing/nonfinite native logits evidence is a technical collection failure')
    stop = ev.classify_stop(tokens, eos, 2048, result.get('native_stop', False))
    if record.get('stop_reason') != stop:
        raise RuntimeError('native stop classification differs from actual token evidence')
    return [reason for reason, present in (
        ('budget_truncation', stop == 'budget_truncation'), ('unknown_stop', stop == 'unknown_stop'),
        ('empty_raw', not record['raw_text'].strip())) if present]


def collect_shards(out, dispatch, ev, require_rank0_exit=True):
    records, byte_lines, receipts, diagnostics = {}, {}, [], []
    pids = set()
    for rank in range(WORLD):
        manifest_path = out / ('rank%d' % rank) / 'manifest.json'
        manifest = read_receipt(manifest_path)
        check_identity(manifest, dispatch, rank)
        if manifest['pid'] in pids:
            raise RuntimeError('ranks cannot reuse one process')
        pids.add(manifest['pid'])
        ready = read_receipt(out / ('loaded-rank%d.json' % rank))
        check_identity(ready, dispatch, rank)
        loaded_path = out / ('rank%d' % rank) / 'native_source_loaded.json'
        loaded = read_receipt(loaded_path)
        check_identity(loaded, dispatch, rank)
        started = read_receipt(out / ('start-rank%d.json' % rank))
        check_identity(started, dispatch, rank)
        for evidence in (started, ready, loaded):
            if (evidence['pid'], evidence['pid_start_ticks']) != (manifest['pid'], manifest['pid_start_ticks']):
                raise RuntimeError('rank actual loading/generation process identity differs')
        if (ready.get('model_parameters_on_assigned_GPU') is not True
                or ready.get('device') != 'cuda:%d' % rank or loaded.get('actual_imported_source_refs') != dispatch['source_refs']
                or manifest.get('device') != 'cuda:%d' % rank or manifest.get('complete') is not True
                or manifest.get('pretrained_GPU_generation_executed') is not True
                or manifest.get('human_quality_accepted') is not False
                or manifest.get('generation_patch') != loaded.get('generation_patch')
                or ready.get('generation_patch') != loaded.get('generation_patch')):
            raise RuntimeError('eight actual GPU loading/native generation receipts required')
        patch = loaded['generation_patch']
        if patch.get('requested_ref') != dispatch['source_refs']['generation_patch']:
            raise RuntimeError('requested patch provenance differs')
        expected_applied = patch.get('applied_ref')
        if expected_applied is not None and expected_applied != dispatch['source_refs']['generation_patch']:
            raise RuntimeError('actual patch differs from explicitly frozen patch')
        if dispatch['family'] == 'moss' and expected_applied is None:
            raise RuntimeError('MOSS native cache compatibility patch not applied')
        if rank or require_rank0_exit:
            exited = read_receipt(out / ('exit-rank%d.json' % rank))
            check_identity(exited, dispatch, rank)
            if exited.get('exit_code') != 0 or (exited['pid'], exited['pid_start_ticks']) != (manifest['pid'], manifest['pid_start_ticks']):
                raise RuntimeError('rank execution did not finish successfully')
        expected_indices = list(range(rank, TEST_COUNT, WORLD))
        raw_path = out / ('rank%d' % rank) / 'raw.jsonl'
        rank_records = []
        with regular(raw_path).open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n') or not line.strip():
                    raise RuntimeError('partial/empty raw line')
                record = json.loads(line)
                index = record.get('global_index')
                if type(index) is not int or index in records or not 0 <= index < TEST_COUNT:
                    raise RuntimeError('duplicate/extra global test index')
                if (index % WORLD != rank or record.get('rank') != rank or record.get('device') != 'cuda:%d' % rank
                        or record.get('id') != dispatch['source_ids'][index]
                        or record.get('invocation_nonce') != dispatch['invocation_nonce']
                        or record.get('source_refs') != dispatch['source_refs']
                        or record.get('data_refs') != dispatch['data_refs']
                        or record.get('generation_patch') != patch
                        or record.get('eos_token_ids') != patch.get('native_eos_token_ids')):
                    raise RuntimeError('raw membership/GPU/source/patch mismatch')
                if type(record.get('actual_forward_count')) is not int or record['actual_forward_count'] <= 0:
                    raise RuntimeError('no actual generation forward observed')
                native_key = 'applied_generation_patch_ref' if dispatch['family'] == 'moss' else 'applied_generation_eos_patch_ref'
                if record.get('native_output', {}).get(native_key) != expected_applied:
                    raise RuntimeError('native output patch stamp differs from actual loaded patch')
                reasons = validate_record(record, ev)
                if reasons:
                    diagnostics.append(dict(id=record['id'], global_index=index, reasons=reasons, stop_reason=record['stop_reason']))
                rank_records.append(index)
                records[index], byte_lines[index] = record, line
        if (rank_records != expected_indices or manifest.get('global_indices') != expected_indices
                or manifest.get('source_ids') != [dispatch['source_ids'][i] for i in expected_indices]
                or manifest.get('attempted') != len(expected_indices)
                or manifest.get('actual_forward_count') != sum(records[i]['actual_forward_count'] for i in expected_indices)
                or manifest.get('finite_native_logits_forward_count') != manifest.get('actual_forward_count')
                or manifest.get('native_logits_all_finite') is not True):
            raise RuntimeError('rank shard missing/duplicate/reordered IDs or forward count')
        if manifest.get('raw') != file_ref(raw_path):
            raise RuntimeError('rank raw bytes changed after completion')
        receipts.append(dict(rank=rank, manifest=file_ref(manifest_path), raw=file_ref(raw_path),
                             native_source_loaded=file_ref(loaded_path), device='cuda:%d' % rank))
    if set(records) != set(range(TEST_COUNT)):
        raise RuntimeError('not exactly one original output for each global test500 ID')
    return [records[i] for i in range(TEST_COUNT)], [byte_lines[i] for i in range(TEST_COUNT)], receipts, diagnostics


def aggregate(out, dispatch, ev):
    records, byte_lines, receipts, diagnostics = collect_shards(out, dispatch, ev, require_rank0_exit=False)
    # Membership/technical checks precede the single exclusive combined-raw write.
    with (out / 'raw.jsonl').open('xb') as stream:
        for line in byte_lines:
            stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    new_json(out / 'summary.json', ev.summarize(records))
    manifest = dict(schema='music-planner-parallel-native-test-v1', mode='finaltest_parallel',
        family=dispatch['family'], world_size=WORLD, rank_count=WORLD, complete=True,
        invocation_nonce=dispatch['invocation_nonce'], source_ids=dispatch['source_ids'],
        source_ids_sha256=dispatch['source_ids_sha256'], source_refs=dispatch['source_refs'],
        code=dispatch['source_refs']['producer'], frozen_eval_source=dispatch['source_refs']['frozen_eval_source'],
        frozen_adapter_source=dispatch['source_refs']['frozen_adapter_source'],
        checkpoint=dispatch['checkpoint'], checkpoint_identity=dispatch['checkpoint_identity'],
        config=dispatch['config'], model=dispatch['model'], base_identity=dispatch['base_audit']['identity'],
        current_official_base_full_content_SHA_verified=True,
        requested_generation_patch_ref=dispatch['source_refs']['generation_patch'],
        generation_patch_by_rank={str(rank): read_receipt(out / ('loaded-rank%d.json' % rank))['generation_patch'] for rank in range(WORLD)},
        data_refs=dispatch['data_refs'], protocol=dispatch['protocol'], final_selection=dispatch['final_selection'],
        checkpoint_selection=dispatch['checkpoint_selection'], checkpoint_marker=dispatch['checkpoint_marker'],
        decoding=DECODE, teacher_forced=False, counterfactual=False, rank_receipts=receipts,
        case_GPU_assignment=[dict(id=r['id'], global_index=r['global_index'], rank=r['rank'], device=r['device']) for r in records],
        actual_forward_count=sum(r['actual_forward_count'] for r in records),
        finite_native_logits_forward_count=sum(r['finite_native_logits_forward_count'] for r in records),
        native_logits_all_finite=True, attempted=TEST_COUNT,
        actual_generated_tokens=sum(r['generated_tokens'] for r in records),
        raw=file_ref(out / 'raw.jsonl'), summary=file_ref(out / 'summary.json'),
        diagnostic_count=len(diagnostics), diagnostics=diagnostics, diagnostics_do_not_trigger_retry=True,
        pretrained_GPU_generation_executed=True, technical_collection_complete=True,
        test_used_for_checkpoint_selection=False, automatic_retry=False,
        automatic_quality_ranking=False, winner=None, human_quality_accepted=False)
    new_json(out / 'manifest.json', manifest)
    return manifest


def await_other_completions(out, dispatch):
    deadline = dispatch['created_unix'] + dispatch['completion_timeout_seconds']
    while True:
        check_failed_ranks(out, dispatch)
        done = True
        for rank in range(1, WORLD):
            start_path = out / ('start-rank%d.json' % rank)
            exit_path = out / ('exit-rank%d.json' % rank)
            if not exit_path.exists():
                done = False
                if start_path.exists():
                    start = read_receipt(start_path)
                    check_identity(start, dispatch, rank)
                    if not process_alive(start):
                        raise RuntimeError('rank died before durable completion')
        if done:
            return
        if time.time() >= deadline:
            raise RuntimeError('bounded rank completion deadline expired')
        time.sleep(1)


def run_rank(dispatch, producer_sha):
    verify_dispatch(dispatch, producer_sha)
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != WORLD or int(os.environ['RANK']) != rank or not 0 <= rank < WORLD:
        raise RuntimeError('single-node torchrun LOCAL_RANK0..7 only')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise RuntimeError('exact eight physical GPUs required')
    out = Path(dispatch['out'])
    if not out.is_dir() or out.is_symlink():
        raise RuntimeError('parent-owned fresh invocation directory required')
    identity = dict(identity_fields(dispatch, rank), pid=os.getpid(), pid_start_ticks=pid_start(os.getpid()))
    rank_out = out / ('rank%d' % rank)
    rank_out.mkdir(exist_ok=False)
    new_json(out / ('start-rank%d.json' % rank), identity)
    try:
        ev = load_module('_parallel_frozen_eval', dispatch['source_refs']['frozen_eval_source'])
        splits, _ = ev.validate_data(Path(dispatch['data']))
        rows = ev.select_rows(splits['test'], 'test', limit=TEST_COUNT, seed=42)
        if [r['id'] for r in rows] != dispatch['source_ids']:
            raise RuntimeError('global SHA-sorted seed42 test rows differ from frozen dispatch')
        shard = partition(rows, rank, world)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD:
            raise RuntimeError('real eight-visible-device CUDA required')
        torch.cuda.set_device(rank)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        adapters = load_module('music_planner_adapters', dispatch['source_refs']['frozen_adapter_source'])
        patch = load_module('_parallel_generation_patch', dispatch['source_refs']['generation_patch'])
        previous = wait_previous_loaded(out, dispatch, rank)
        check_inventory(dispatch)
        remaining = dispatch['created_unix'] + dispatch['loading_timeout_seconds'] - time.time()
        if remaining <= 0:
            raise RuntimeError('bounded model loading deadline expired before loading')
        def loading_timed_out(signum, frame):
            raise TimeoutError('bounded model loading deadline expired')
        old_alarm = signal.signal(signal.SIGALRM, loading_timed_out)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            adapter = adapters.load_planner_adapter(family=dispatch['family'], model_path=Path(dispatch['model']),
                checkpoint=Path(dispatch['checkpoint']), device='cuda:%d' % rank,
                max_length=dispatch['checkpoint_identity']['training_config']['max_length'])
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        if time.time() >= dispatch['created_unix'] + dispatch['loading_timeout_seconds']:
            raise RuntimeError('model loading returned after bounded deadline')
        parameters = list(adapter.model.parameters())
        if not parameters or any(p.device.type != 'cuda' or p.device.index != rank for p in parameters):
            raise RuntimeError('loaded model parameters not on assigned real GPU')
        del parameters
        applied = getattr(patch, dispatch['generation_patch_entrypoint'])(adapter)
        requested = dispatch['source_refs']['generation_patch']
        if applied is not None and applied != requested:
            raise RuntimeError('applied patch source identity differs')
        if dispatch['family'] == 'moss' and applied is None:
            raise RuntimeError('required inference-only MOSS cache patch was not applied')
        patch_evidence = dict(requested_ref=requested, applied_ref=applied,
            entrypoint=dispatch['generation_patch_entrypoint'],
            details=getattr(adapter, 'generation_eos_patch_details', None),
            native_eos_token_ids=adapter.eos_token_ids(), cache_disabled=False,
            trained_prefix_weights_unchanged=True, raw_output_cleaning=False)
        if dispatch['family'] == 'omni_captioner' and applied is not None:
            if patch_evidence['details'].get('explicit_eos_token_ids') != [151645]:
                raise RuntimeError('Captioner fallback must be native trained im_end151645')
        check_inventory(dispatch)
        gc.collect()
        torch.cuda.synchronize(rank)
        loaded = dict(identity, device='cuda:%d' % rank, model_parameters_on_assigned_GPU=True,
            generation_patch=patch_evidence, previous_loaded_ref=file_ref(out / ('loaded-rank%d.json' % (rank-1))) if previous else None,
            GPU_name=torch.cuda.get_device_name(rank), allocated_bytes=torch.cuda.memory_allocated(rank),
            loaded_unix=time.time())
        actual_refs = dict(dispatch['source_refs'])
        actual_refs['producer'] = file_ref(__file__)
        actual_refs['frozen_eval_source'] = file_ref(ev.__file__)
        actual_refs['frozen_adapter_source'] = file_ref(adapters.__file__)
        actual_refs['generation_patch'] = file_ref(patch.__file__)
        if actual_refs != dispatch['source_refs']:
            raise RuntimeError('actual imported sources differ from pinned dispatch')
        new_json(rank_out / 'native_source_loaded.json', dict(loaded, actual_imported_source_refs=actual_refs))
        new_json(out / ('loaded-rank%d.json' % rank), loaded)
        new_json(rank_out / 'ready.json', dict(loaded, global_indices=[i for i, _ in shard]))
        forward_counter, finite_counter = [0], [0]
        def observed_forward(module, args):
            forward_counter[0] += 1
        def checked_forward_logits(module, args, output):
            logits = getattr(output, 'logits', None)
            if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
                raise RuntimeError('native model forward did not return actual nonempty Tensor logits')
            if logits.device.type != 'cuda' or logits.device.index != rank:
                raise RuntimeError('native model logits not on the assigned actual GPU')
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError('nonfinite native logits detected before generation argmax')
            finite_counter[0] += 1
        native_model = adapter.model.get_base_model() if hasattr(adapter.model, 'get_base_model') else adapter.model
        hook = native_model.register_forward_pre_hook(observed_forward)
        logits_hook = native_model.register_forward_hook(checked_forward_logits)
        records = []
        try:
            with (rank_out / 'raw.jsonl').open('x', encoding='utf-8') as stream:
                for index, source in shard:
                    check_failed_ranks(out, dispatch)
                    before, finite_before = forward_counter[0], finite_counter[0]
                    record = ev.evaluate_one(adapter, source, 'original', source, 2048, seed=42, teacher_forced=False)
                    record.update(global_index=index, rank=rank, device='cuda:%d' % rank,
                        invocation_nonce=dispatch['invocation_nonce'], source_refs=dispatch['source_refs'],
                        data_refs=dispatch['data_refs'], generation_patch=patch_evidence,
                        actual_forward_count=forward_counter[0]-before,
                        finite_native_logits_forward_count=finite_counter[0]-finite_before,
                        native_logits_all_finite=(forward_counter[0]-before > 0
                            and finite_counter[0]-finite_before == forward_counter[0]-before))
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    validate_record(record, ev)
                    if record['actual_forward_count'] <= 0:
                        raise RuntimeError('generation did not execute an observed native model forward')
        finally:
            hook.remove()
            logits_hook.remove()
        check_inventory(dispatch)
        new_json(rank_out / 'manifest.json', dict(identity, device='cuda:%d' % rank,
            mode='finaltest_parallel_shard', source_ids=[source['id'] for _, source in shard],
            global_indices=[i for i, _ in shard], generation_patch=patch_evidence,
            actual_forward_count=forward_counter[0], finite_native_logits_forward_count=finite_counter[0],
            native_logits_all_finite=finite_counter[0] == forward_counter[0] and forward_counter[0] > 0,
            attempted=len(records), raw=file_ref(rank_out / 'raw.jsonl'),
            complete=True, pretrained_GPU_generation_executed=True, human_quality_accepted=False))
        if rank == 0:
            await_other_completions(out, dispatch)
            aggregate(out, dispatch, ev)
        new_json(out / ('exit-rank%d.json' % rank), dict(identity, exit_code=0,
            execution_completed=True, OS_process_exit_verified_by_parent_torchrun=False))
    except BaseException as error:
        exit_path = out / ('exit-rank%d.json' % rank)
        if not exit_path.exists():
            new_json(exit_path, dict(identity, exit_code=1, error=repr(error),
                raw_preserved=True, automatic_retry=False, complete=False))
        if rank == 0 and not (out / 'manifest.json').exists():
            new_json(out / 'manifest.json', dict(identity, schema='music-planner-parallel-native-test-v1',
                mode='finaltest_parallel', code=dispatch['source_refs']['producer'], complete=False,
                error=repr(error), raw_shards_preserved=True, automatic_retry=False, human_quality_accepted=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', required=True, type=Path)
    parser.add_argument('--dispatch-sha', required=True)
    parser.add_argument('--producer-sha', required=True)
    args = parser.parse_args()
    verify_ref(dict(path=str(args.dispatch), sha256=args.dispatch_sha))
    if sha(regular(__file__)) != args.producer_sha:
        raise RuntimeError('new worker source SHA changed before any import')
    dispatch = read_receipt(args.dispatch)
    def interrupted(signum, frame):
        raise InterruptedError('managed torchrun worker signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_rank(dispatch, args.producer_sha)


if __name__ == '__main__':
    main()
