"""Same-native pure official-base fixed3 collection; no training, test500 or video."""
import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import stat
import sys
import time
import wave

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
RT = EXP / 'runtime'
HANDOFF = EXP / 'eval_handoff_clean_v11'
OUTPUT = HANDOFF / 'base_fixed3_compare_v1'
COMPARE = HANDOFF / 'h3_compare_v1/run_config_v3.json'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
FAMILIES = ('moss', 'music_flamingo', 'omni_captioner')
SCOPE = {'N2': ('28.216.17.70', 'music_flamingo'),
         'N4': ('28.216.19.143', 'omni_captioner'),
         'N7': ('28.216.18.144', 'moss')}
DECODE = dict(seed=42, do_sample=False, max_new_tokens=2048,
              native_training_prefix_unchanged=True, extra_instruction=False,
              raw_output_cleaning=False, GT_length_truncation=False, added_penalties=False)
NATIVE_DECODE = dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
                     no_repeat_ngram_size=0, encoder_repetition_penalty=1.0)
ADAPTER = RT / 'source_v11/music_planner_adapters.py'
EVALUATOR = RT / 'source_v11/music_planner_eval.py'
COMPARE_SHA = '55f7fd1c75b996a43aacfdbbb275a7ffd0cdf813991680f66c9311b4ceb5a332'
PROCESSOR_SUFFIXES = ('.json', '.txt', '.model', '.py', '.jinja')
MOSS_PATCH = RT / 'eval_generation_patches_v2/music_planner_moss_inference_compat_v2.py'
EOS_PATCH = RT / 'eval_generation_patches_v1/music_planner_generation_eos_patch.py'
POSITION_PATCH = RT / 'eval_generation_patches_v2/music_planner_moss_position_buffer_patch.py'
PINNED = {
    str(EVALUATOR): '55ca82f4ec117ffa05087e7dba54c1787de15ce6522ab0cd69e95f8f1270bea6',
    str(ADAPTER): '713dc3eb89408396bbae35a5e2957cb65b212a1b55276a728779ac2acf5fa4d1',
    str(MOSS_PATCH): '043efb725ca75331c315a64cff8c64eef0f63a67c1f55afb26b0ccf2082c4908',
    str(EOS_PATCH): '6b2cf7e4ad59acd546aa398a0b6cd3807318e674be22c852e65be73360b53bf4',
    str(POSITION_PATCH): '0077f742bb98f64763a38eb082ec228e99b1168ad8090813abb45af90dfcba33',
}
PROTOCOL_SHA = 'f86d5d15b25a5d10f2870b369e68a9784dbb172e2746e04ebc86aae2a9ed276b'
FINAL_SHA = '7aa6d2459ba659c514ebf69dc688aacaa802b8fb1f3d9e01559afda2935895dd'


def regular(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('absolute nonsymlink path required: ' + str(path))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('regular file required: ' + str(path))
    return path


def file_ref(path):
    path = regular(path)
    before = file_stat(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    if file_stat(path) != before:
        raise RuntimeError('file changed during SHA audit: ' + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=before['size'])


def file_stat(path):
    st = regular(path).stat()
    return dict(size=st.st_size, device=st.st_dev, inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def verify_ref(ref):
    actual = file_ref(ref['path'])
    if any(actual.get(k) != v for k, v in ref.items()):
        raise RuntimeError('frozen input changed: ' + ref['path'])
    return actual


def ref_json(ref):
    verify_ref(ref)
    return json.loads(Path(ref['path']).read_text())


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_sources(cfg, config_path, config_sha, producer_sha):
    if file_ref(config_path)['sha256'] != config_sha or file_ref(__file__)['sha256'] != producer_sha:
        raise RuntimeError('config/producer SHA changed')
    required = dict(PINNED, **{str(Path(__file__).resolve()): producer_sha})
    for path, digest in required.items():
        if cfg['source_sha256'].get(path) != digest:
            raise RuntimeError('missing/conflicting source pin: ' + path)
    for path, digest in cfg['source_sha256'].items():
        if file_ref(path)['sha256'] != digest:
            raise RuntimeError('source SHA drift: ' + path)


def check_rows(rows, trained):
    if [r['id'] for r in rows] != IDS or [r['id'] for r in trained] != IDS:
        raise RuntimeError('exact fixed3 membership/order required')
    for row, old in zip(rows, trained):
        if (row.get('audio_window_seconds') != [0., 5.0625]
                or row.get('audio_path') != row['audio_ref']['path']
                or row.get('audio_sha256') != row['audio_ref']['sha256']
                or old.get('audio_ref') != row['audio_ref']
                or old.get('image_ref') != row['image_ref']):
            raise RuntimeError('reference full-audio binding differs')
        verify_ref(row['audio_ref'])
        with wave.open(row['audio_path'], 'rb') as stream:
            rate, count = stream.getframerate(), stream.getnframes()
            if stream.getnchannels() != 1 or abs(count / rate - 5.0625) > 1 / rate:
                raise RuntimeError('original complete mono 5.0625s required')
            if len(stream.readframes(count)) != count * stream.getsampwidth():
                raise RuntimeError('truncated audio')
        verify_ref(row['image_ref'])
    if len({r['audio_sha256'] for r in rows}) != 3:
        raise RuntimeError('fixed3 audio duplicates')


def check_trained_record(record):
    result = record.get('native_output', {})
    if (record.get('failure') is not None or result.get('raw_text') != record.get('raw_text')
            or result.get('new_token_ids') != record.get('new_token_ids')
            or result.get('eos_token_ids') != record.get('eos_token_ids')
            or record.get('generation_budget') != 2048 or result.get('seed') != 42
            or result.get('decoding') != NATIVE_DECODE
            or result.get('audio_input') != 'full-waveform' or result.get('silence_applied') is not False
            or result.get('processor_sample_rate') != 16000 or result.get('n_samples') != 81000
            or type(result.get('input_tokens')) is not int or result['input_tokens'] < 1):
        raise RuntimeError('trained native audio/greedy reference invalid')


def bindings(cfg):
    if cfg.get('family') not in FAMILIES:
        raise RuntimeError('unknown family')
    for key, path, digest in [('protocol', HANDOFF / 'protocol.json', PROTOCOL_SHA),
                              ('final_selection', HANDOFF / 'final_selection.json', FINAL_SHA)]:
        if cfg[key] != dict(path=str(path), sha256=digest):
            raise RuntimeError('wrong independently frozen ' + key)
    if cfg['trained_compare'] != dict(path=str(COMPARE), sha256=COMPARE_SHA):
        raise RuntimeError('wrong trained compare source')
    protocol, final, compare = (ref_json(cfg[k]) for k in ('protocol', 'final_selection', 'trained_compare'))
    if protocol.get('schema') != 'music_planner_fixed3_protocol_v1' or protocol.get('decoding') != DECODE:
        raise RuntimeError('wrong fixed3 native protocol')
    if (final.get('selection_frozen_before_test') is not True or final.get('test_used') is not False
            or final.get('protocol') != cfg['protocol'] or final.get('decoding') != DECODE):
        raise RuntimeError('wrong frozen selection')
    family = cfg['family']
    refs = compare['families'][family]
    for key in ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection', 'checkpoint_marker', 'fingerprint'):
        if cfg[key] != refs[key] or cfg[key] != final['families'][family][key]:
            raise RuntimeError('trained/final/current family identity differs: ' + key)
    for key in ('generation_complete', 'generation_resources_safe', 'generation_exit', 'manifest', 'raw'):
        verify_ref(refs[key])
    safe, complete, exit_ = (ref_json(refs[k]) for k in ('generation_resources_safe', 'generation_complete', 'generation_exit'))
    if (any(safe.get(k) is not True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
                                             'managed_parent_wait', 'unknown_jobs_not_signaled'))
            or safe.get('admitted_gpu_count') != 1 or exit_.get('exit_code') != 0
            or complete.get('complete_native_GPU_outputs_verified') is not True
            or complete.get('manifest') != refs['manifest'] or complete.get('raw') != refs['raw']):
        raise RuntimeError('trained generation not safely complete')
    expected_audit_path = Path(refs['generation_complete']['path']).parent / 'integrity_receipt.json'
    if cfg['expected_base_audit']['path'] != str(expected_audit_path):
        raise RuntimeError('expected base identity must use actual same-family full-SHA generation audit')
    expected_base = ref_json(cfg['expected_base_audit'])['base']
    if expected_base.get('current_full_content_SHA_and_official_digest_verified') is not True or not expected_base.get('files'):
        raise RuntimeError('expected base audit is not a real full content SHA receipt')
    manifest = ref_json(refs['manifest'])
    ci = manifest['checkpoint_identity']
    if (manifest.get('complete') is not True or manifest.get('family') != family
            or manifest.get('protocol') != file_ref(cfg['protocol']['path'])
            or manifest.get('source_ids') != IDS or manifest.get('device') != 'cuda:0'
            or manifest.get('decoding') != DECODE or manifest.get('test_used') is not False
            or manifest.get('pretrained_GPU_generation_executed') is not True
            or ci.get('trainer_state') != verify_ref(cfg['checkpoint_marker'])
            or ci.get('fingerprint') != cfg['fingerprint']):
        raise RuntimeError('trained fixed3 manifest invalid')
    run = ref_json(ci['run_config'])
    if run != ci['training_config'] or run.get('fingerprint') != cfg['fingerprint']:
        raise RuntimeError('trained max_length/run config binding differs')
    if type(run.get('max_length')) is not int or run['max_length'] <= 2048:
        raise RuntimeError('invalid trained context')
    raw = [json.loads(line) for line in Path(refs['raw']['path']).read_text().splitlines() if line.strip()]
    check_rows(protocol['fixed3'], raw)
    for record in raw:
        check_trained_record(record)
    applied = manifest.get('applied_generation_patch_ref') if family == 'moss' else manifest.get('applied_generation_eos_patch_ref')
    expected = file_ref(MOSS_PATCH if family == 'moss' else EOS_PATCH) if family != 'music_flamingo' else None
    if applied != expected:
        raise RuntimeError('trained generation interface patch differs')
    marker = ref_json(cfg['checkpoint_marker'])
    export = Path(cfg['checkpoint_marker']['path']).parent / 'adapter'
    processor_refs = []
    for ref in marker.get('files', []):
        path = Path(ref['path'])
        if path.is_relative_to(export) and path.suffix in PROCESSOR_SUFFIXES:
            processor_refs.append(verify_ref(ref))
    check_processor_inventory(export, processor_refs)
    if not processor_refs:
        raise RuntimeError('export processor static content proof absent')
    return dict(protocol=protocol, trained_refs=refs, trained_manifest=manifest, trained_raw=raw,
                run_config=ci['run_config'], max_length=run['max_length'], export=export,
                processor_refs=processor_refs, expected_patch=expected, expected_base=expected_base)


def check_processor_inventory(export, refs):
    actual = {str(path) for path in export.rglob('*') if path.is_file() and path.suffix in PROCESSOR_SUFFIXES}
    if any(path.is_symlink() for path in export.rglob('*')) or actual != {r['path'] for r in refs}:
        raise RuntimeError('export processor resource inventory differs from trained marker')


def tokenizer_identity(adapter):
    tok = adapter.tokenizer
    value = dict(vocab=tok.get_vocab(), added_vocab=tok.get_added_vocab(),
                 model_max_length=tok.model_max_length, all_special_ids=tok.all_special_ids,
                 special_tokens_map={k: str(v) for k, v in tok.special_tokens_map.items()},
                 tokenizer_chat_template=getattr(tok, 'chat_template', None),
                 processor_chat_template=getattr(adapter.processor, 'chat_template', None),
                 prompt=adapter._prompt(), assistant_end=tok.encode('<|im_end|>', add_special_tokens=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compare_prefixes(base, reference, rows, trained, torch):
    if tokenizer_identity(base) != tokenizer_identity(reference):
        raise RuntimeError('base/export tokenizer, chat template or native prompt differs')
    evidence = []
    for row, old in zip(rows, trained):
        a, b = base.prefix(row), reference.prefix(row)
        if set(a) != set(b):
            raise RuntimeError('base/export processor prefix keys differ')
        tensors = {}
        for key in a:
            if not isinstance(a[key], torch.Tensor) or not isinstance(b[key], torch.Tensor):
                raise RuntimeError('native prefix is not a tensor batch')
            if a[key].dtype != b[key].dtype or a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
                raise RuntimeError('base/export native audio prefix differs: ' + key)
            if a[key].is_floating_point() and not torch.isfinite(a[key]).all().item():
                raise RuntimeError('nonfinite native audio features')
            tensors[key] = dict(shape=list(a[key].shape), dtype=str(a[key].dtype),
                sha256=hashlib.sha256(a[key].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        ids = a['input_ids'][0].tolist()
        if len(ids) != old['native_output']['input_tokens'] or len(ids) + 2048 > base.max_length:
            raise RuntimeError('native trained prefix length/context differs')
        evidence.append(dict(id=row['id'], audio_ref=row['audio_ref'], input_token_ids=ids,
                             tensors=tensors, base_export_exact_equal=True,
                             trained_historical_prefix_ids_available=False,
                             trained_historical_input_length_equal=True))
    return dict(tokenizer_identity_sha256=tokenizer_identity(base), records=evidence,
                exported_processor_used_for_comparison_only=True, base_processor_replaced=False,
                verification_scope='replayed-frozen-processor-prefix-not-recorded-historical-prefix',
                trained_historical_prefix_evidence='Audio binding and length only; historical full prefix tensors were not saved')


def pure_base(adapters, family, model, max_length, torch):
    messages = []
    class LoadingObserver(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())
    observer = LoadingObserver()
    loggers = (logging.getLogger(), logging.getLogger('transformers'))
    for logger in loggers:
        logger.addHandler(observer)
    try:
        adapter = adapters.load_planner_adapter(family, model, checkpoint=None, lora_rank=None,
                                               device='cuda:0', dtype=torch.bfloat16, max_length=max_length)
    finally:
        for logger in loggers:
            logger.removeHandler(observer)
    missing = [text for text in messages if re.search(
        r'newly initialized|not initialized|uninitialized|missing(?:.*(?:key|weight|parameter))', text, re.I)]
    if missing:
        raise RuntimeError('native HF learned missing/uninitialized loading warning: ' + '\n'.join(missing))
    adapter.loading_evidence = dict(loading_info_not_recorded=True, learned_missing_warning_observed=False,
        observed_HF_log_messages=list(dict.fromkeys(messages)),
        exact_per_parameter_restore_not_verified=True)
    if (adapter.family != family or adapter.trainable_info or hasattr(adapter.model, 'peft_config')
            or any('lora_' in n or 'modules_to_save' in n for n, _ in adapter.model.named_parameters())):
        raise RuntimeError('pure base cannot contain LoRA/modules_to_save/PEFT')
    parameters = list(adapter.model.parameters())
    if not parameters or any(p.device != torch.device('cuda:0') for p in parameters):
        raise RuntimeError('all base parameters must be on explicit CUDA0; no CPU fallback')
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in parameters):
        raise RuntimeError('base parameters must use the native BF16 load')
    adapter.model.eval().requires_grad_(False)
    return adapter


def finite_guard(adapter, torch):
    counts = dict(actual_forward_count=0, finite_native_logits_forward_count=0)
    def hook(module, inputs, output):
        counts['actual_forward_count'] += 1
        logits = getattr(output, 'logits', None)
        if not isinstance(logits, torch.Tensor) or not torch.isfinite(logits).all().item():
            raise RuntimeError('native pre-argmax logits missing or NaN/Inf')
        counts['finite_native_logits_forward_count'] += 1
    return adapter.model.register_forward_hook(hook), counts


def classify_stop(tokens, eos, native_stop=False):
    if tokens and tokens[-1] in eos:
        return 'eos'
    if native_stop:
        return 'native_stop'
    return 'budget_truncation' if len(tokens) == 2048 else 'unknown_stop'


def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get('raw_text'), str):
        raise RuntimeError('native output must contain raw text')
    tokens, eos = result.get('new_token_ids'), result.get('eos_token_ids')
    if (not isinstance(tokens, list) or not isinstance(eos, list) or not eos or len(tokens) > 2048
            or any(type(t) is not int or t < 0 for t in tokens + eos)):
        raise RuntimeError('native token interface invalid')
    if (type(result.get('input_tokens')) is not int or result['input_tokens'] < 1
            or result.get('seed') != 42 or result.get('decoding') != NATIVE_DECODE
            or result.get('audio_input') != 'full-waveform' or result.get('silence_applied') is not False
            or result.get('processor_sample_rate') != 16000 or result.get('n_samples') != 81000):
        raise RuntimeError('native full-audio/greedy protocol changed')
    return classify_stop(tokens, eos, result.get('native_stop', False))


def frozen_windows(text):
    if file_ref(EVALUATOR)['sha256'] != PINNED[str(EVALUATOR)]:
        raise RuntimeError('frozen window parser source drift')
    return load_module('base_fixed3_frozen_parser', EVALUATOR).parse_windows(text)


def native_record(adapter, row, torch, trained_result=None):
    record = dict(id=row['id'], audio_ref=row['audio_ref'], image_ref=row['image_ref'],
                  generation_budget=2048, raw_text='', failure=None)
    start = time.monotonic()
    handle, counts = finite_guard(adapter, torch)
    try:
        verify_ref(row['audio_ref'])
        result = adapter.generate(dict(id=row['id'], audio_path=row['audio_path'],
            audio_sha256=row['audio_sha256'], audio_window_seconds=[0., 5.0625]),
            max_new_tokens=2048, silence=False, seed=42)
        record['native_output'] = result
        record['raw_text'] = result.get('raw_text')
        stop = validate_result(result)
        if trained_result is not None and (result.get('input_tokens') != trained_result['input_tokens']
                or result.get('eos_token_ids') != trained_result['eos_token_ids']):
            raise RuntimeError('base/trained effective EOS or native input length differs')
        if counts['actual_forward_count'] <= 0 or counts['actual_forward_count'] != counts['finite_native_logits_forward_count']:
            raise RuntimeError('no real finite native generation forwards')
        record.update(new_token_ids=result['new_token_ids'], eos_token_ids=result['eos_token_ids'],
                      stop_reason=stop, generated_tokens=len(result['new_token_ids']),
                      budget_truncation=stop == 'budget_truncation', ended_with_eos=stop == 'eos')
        text = result['raw_text']
        windows = frozen_windows(text)
        dual = windows['matches_fixed_windows'] and windows['nonempty_window_text']
        record['window_format'] = windows
        record['matches_fixed_windows'] = windows['matches_fixed_windows']
        record['nonempty_window_text'] = windows['nonempty_window_text']
        reasons = []
        if not text.strip():
            reasons.append('empty_raw')
        if stop in ('budget_truncation', 'unknown_stop'):
            reasons.append(stop)
        if not dual:
            reasons.append('no_exact_dual_window_format')
        record['quality_diagnostics'] = reasons
        record['dual_window_format'] = dual
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        handle.remove()
    record.update(counts, native_logits_all_finite=counts['actual_forward_count'] > 0 and
                  counts['actual_forward_count'] == counts['finite_native_logits_forward_count'],
                  elapsed_seconds=time.monotonic() - start)
    return record


def check_audit(ref, cfg, config_sha, model):
    if ref['path'] != str(Path(cfg['queue']) / 'integrity_receipt.json'):
        raise RuntimeError('only exact manager queue integrity receipt is accepted')
    audit = ref_json(ref)
    manager_source = str(Path(__file__).resolve().with_name('music_planner_base_fixed3_manager.py'))
    if (audit.get('schema') != 'music-planner-base-fixed3-audit-v1'
            or audit.get('manager_sha256') != cfg['source_sha256'].get(manager_source)
            or audit.get('config_sha256') != config_sha or audit.get('family') != cfg['family']
            or audit.get('model') != str(model) or audit.get('protocol') != cfg['protocol']
            or audit.get('base', {}).get('current_full_content_SHA_and_official_digest_verified') is not True
            or audit['base'].get('local_weight_hash_passes_this_audit') != 1):
        raise RuntimeError('real current full base content audit is required; snapshot is not SHA')
    files = audit['base'].get('files', [])
    if not files or model.is_symlink() or any(item.is_symlink() for item in model.rglob('*')):
        raise RuntimeError('nonempty nonsymlink current audited base inventory required')
    audited_paths = {item['path'] for item in files}
    current_paths = {str(item) for item in model.rglob('*') if item.is_file()}
    if len(audited_paths) != len(files) or current_paths != audited_paths:
        raise RuntimeError('base exact inventory changed after full SHA audit')
    for item in files:
        if file_stat(item['path']) != item['stat']:
            raise RuntimeError('base changed after current full SHA audit')
    return audit


def generate(args, cfg):
    check_sources(cfg, args.config, args.config_sha, args.producer_sha)
    if (cfg.get('schema') != 'music-planner-base-fixed3-manager-v1'
            or SCOPE.get(cfg.get('node')) != (cfg.get('host'), cfg.get('family'))
            or cfg.get('gpus') != [0] or type(cfg['gpus'][0]) is not int
            or cfg.get('release_authorized') is not True):
        raise RuntimeError('explicit same-family node and physical GPU0 authorization required')
    data = bindings(cfg)
    out, model = Path(cfg['out']), Path('/opt/music_planner_models') / cfg['family']
    if out != OUTPUT / (cfg['family'] + '_out') or out.exists() or out != out.resolve():
        raise RuntimeError('only fresh canonical family base_fixed3 output allowed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('exact single physical CUDA_VISIBLE_DEVICES=0 required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one real CUDA device required; no CPU/auto fallback')
    audit = check_audit(args.audit, cfg, args.config_sha, model)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    adapters = load_module('base_fixed3_native_adapters', ADAPTER)
    adapter = pure_base(adapters, cfg['family'], model, data['max_length'], torch)
    processor = adapters.load_native_processor(cfg['family'], data['export'])
    reference = adapters.PlannerAdapter(cfg['family'], None, processor, max_length=data['max_length'])
    prefix = compare_prefixes(adapter, reference, data['protocol']['fixed3'], data['trained_raw'], torch)
    applied = None
    if cfg['family'] == 'moss':
        applied = load_module('base_fixed3_moss_patch', MOSS_PATCH).apply_moss_generation_patch(adapter)
    elif cfg['family'] == 'omni_captioner':
        applied = load_module('base_fixed3_eos_patch', EOS_PATCH).apply_generation_eos_patch(adapter)
    if applied != data['expected_patch']:
        raise RuntimeError('base/trained actual inference interface patch differs')
    check_audit(args.audit, cfg, args.config_sha, model)
    check_sources(cfg, args.config, args.config_sha, args.producer_sha)
    out.mkdir(exist_ok=False)
    manifest = dict(schema='music-planner-base-fixed3-v1', family=cfg['family'], device='cuda:0',
        physical_gpus=[0], protocol=verify_ref(cfg['protocol']), source_ids=IDS, decoding=DECODE,
        code=file_ref(__file__), trained_reference=data['trained_refs'], run_config=data['run_config'],
        max_length=data['max_length'], initialization='official_pure_base_no_checkpoint_no_lora',
        checkpoint_loaded=False, lora_initialized=False, modules_to_save_initialized=False,
        base_audit_ref=args.audit, prefix_equivalence=prefix, processor_refs=data['processor_refs'],
        applied_generation_patch_ref=applied, position_buffer_receipt=getattr(adapter, 'position_buffer_patch_receipt', None),
        test_used=False, video_generated=False, human_quality_accepted=False,
        temporal_event_GT_available=False, temporal_accuracy_metrics_computed=False,
        loading_evidence=getattr(adapter, 'loading_evidence', None),
        finite_validation_scope='base-only-observational-pre-argmax-logits-not-training-improvement')
    records = []
    with (out / 'raw.jsonl').open('x', encoding='utf-8') as stream:
        for row, trained in zip(data['protocol']['fixed3'], data['trained_raw']):
            record = native_record(adapter, row, torch, trained['native_output'])
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    manifest.update(complete=len(records) == 3 and not any(r['failure'] for r in records),
                    technical_collection_complete=len(records) == 3 and not any(r['failure'] for r in records),
                    attempted=len(records), raw=file_ref(out / 'raw.jsonl'),
                    diagnostics_do_not_trigger_retry=True,
                    trained_historical_finite_logits_evidence_not_invented=True,
                    quality_diagnostic_count=sum(bool(r.get('quality_diagnostics')) for r in records))
    new_json(out / 'manifest.json', manifest)
    if not manifest['complete']:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'generate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--producer-sha', required=True)
    parser.add_argument('--audit-path', type=Path)
    parser.add_argument('--audit-sha')
    args = parser.parse_args(argv)
    cfg = ref_json(dict(path=str(args.config), sha256=args.config_sha))
    check_sources(cfg, args.config, args.config_sha, args.producer_sha)
    if args.action == 'validate':
        data = bindings(cfg)
        print(json.dumps(dict(family=cfg['family'], ids=IDS, max_length=data['max_length'],
                             CPU_only=True, signals_sent=False, approved_for_gpu=False)))
    else:
        if args.audit_path is None or args.audit_sha is None:
            parser.error('generate requires the manager current full-SHA audit receipt')
        args.audit = dict(path=str(args.audit_path), sha256=args.audit_sha)
        generate(args, cfg)


if __name__ == '__main__':
    main()
