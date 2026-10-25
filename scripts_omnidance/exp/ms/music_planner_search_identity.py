"""Audit local fair/direct-action planner artifacts before candidate collection."""
import hashlib
import ast
import json
import math
from pathlib import Path, PurePosixPath

from scripts_omnidance.exp.ms.music_planner_search import file_sha, object_sha, read_json

# Reject other experimental protocols even if their family happens to match.
FAIR_CONFIG_FIELDS = set(('family model data epochs total_updates source_exposure_budget '
    'global_batch micro accum world_size seed lora_rank lora_alpha lora_lr projector_lr '
    'grad_clip warmup_updates max_length backend base_dtype load_dtype compute_dtype '
    'trainable_dtype adapter_source_sha256 optimizer gradient_checkpointing '
    'checkpointing_use_reentrant parameter_groups trainable_signature trainer_sha256 '
    'zero3_parameter_lifecycle smoke_validation_before_save fingerprint trainable_parameters').split())
DIRECT_SCHEMAS = {f'music-planner-captioner-ablation-v3/direct-caption/{policy}'
                  for policy in ('weak-caption', 'reviewed-shared')}
DIRECT_FIELDS = set(('schema mode require_shared_annotations loss_chunk_size model family data epochs '
    'accum seed rank alpha dtype lora_lr projector_lr warmup_updates grad_clip max_length segment_weights '
    'objective gradient_checkpointing parameter_groups parameter_signature code_sha256 fingerprint').split())


def training_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = file_sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Artifact changed during content audit')
    return dict(path=str(path), sha256=digest, size_bytes=after.st_size)


def inventory(path, recursive=False):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError('Require an existing local artifact directory')
    files = sorted(p for p in (path.rglob('*') if recursive else path.iterdir()) if p.is_file())
    if not files:
        raise ValueError('Empty local artifact directory')
    return [dict(ref(p), relative_path=p.relative_to(path).as_posix()) for p in files]


def audit_model_artifacts(family, model_path, checkpoint=None, training_source_dir=None):
    """Names are annotations; actual content and trained portable identities are evidence."""
    from scripts_omnidance.exp.ms import music_planner_adapters as adapters
    from scripts_omnidance.exp.ms.music_planner_eval import verify_base_identity
    base = Path(model_path).resolve()
    base_files = inventory(base)
    if not (base / 'config.json').is_file() or not any(
            item['relative_path'].endswith(('.safetensors', '.bin')) for item in base_files):
        raise ValueError('Base requires config and actual local weight files')
    receipt = dict(schema='music_planner_local_artifact_identity_v1', family=family,
                   mode='native_base_only', protocol='native-direct-action-fixed-two-windows-v1',
                   base_files=base_files, checkpoint=None,
                   inference_adapter_source=ref(adapters.__file__))
    if checkpoint is None:
        return receipt
    checkpoint = Path(checkpoint).resolve()
    # An arbitrary PEFT folder is insufficient: require the complete fair portable checkpoint.
    marker = read_json(checkpoint / 'trainer_state.json')
    config_path = checkpoint.parent / 'run_config.json'
    config = read_json(config_path)
    direct = config.get('schema') in DIRECT_SCHEMAS
    excluded = ('fingerprint',) if direct else ('fingerprint', 'trainable_parameters')
    canonical = {key: value for key, value in config.items() if key not in excluded}
    fingerprint = training_sha(canonical)
    weights = config.get('segment_weights')
    valid_direct = (family == 'omni_captioner' and config.get('mode') == 'direct-caption'
                    and config.get('require_shared_annotations') is config['schema'].endswith('/reviewed-shared')
                    and config.get('data', {}).get('annotation_schema') == config['schema']
                    and config.get('data', {}).get('provenance_checked') is config['require_shared_annotations']
                    and config.get('objective') == 'weighted-mean-of-segment-mean-causal-NLL-EOS-in-caption'
                    and isinstance(weights, list) and len(weights) == 3
                    and all(type(v) in (int, float) and math.isfinite(v) for v in weights)
                    and weights[0] == weights[1] == 0 and weights[2] > 0) if direct else False
    if (set(config) - (DIRECT_FIELDS if direct else FAIR_CONFIG_FIELDS) or config.get('family') != family
            or config.get('fingerprint') != fingerprint or marker.get('fingerprint') != fingerprint
            or (not valid_direct if direct else config.get('optimizer') != 'AdamW-FP32-full-states')
            or config.get('data', {}).get('test_used') is not False
            or config.get('data', {}).get('length_policy') !=
                'one-full-5.0625s-audio-two-text-windows-no-truncation-v1'):
        raise ValueError('Require the original fair/direct-action training config; other output protocols unsupported')
    if marker.get('schema') != 'music-planner-train-v1' or marker.get('complete') is not True:
        raise ValueError('Checkpoint is not a completed fair portable checkpoint')
    cursor = marker.get('cursor', {})
    if (any(type(cursor.get(key)) is not int or cursor[key] < 0
            for key in ('epoch', 'position', 'update', 'source_exposures'))
            or type(marker.get('world_size')) is not int or marker['world_size'] < 1
            or marker['world_size'] != config.get('world_size', 1) or type(marker.get('zero3')) is not bool):
        raise ValueError('Portable checkpoint completion cursor/world identity is invalid')
    if direct and marker['zero3']:
        raise ValueError('Captioner direct-v3 portable protocol is single-device, without ZeRO3')
    signature = config.get('parameter_signature' if direct else 'trainable_signature')
    if marker.get('trainable_signature') != signature or not signature:
        raise ValueError('Checkpoint/training trainable signatures differ')
    source_dir = Path(training_source_dir).resolve() if training_source_dir else Path(adapters.__file__).resolve().parent
    sources = {}
    expected_sources = config.get('code_sha256') if direct else {
        'music_planner_adapters.py': config.get('adapter_source_sha256'),
        'music_planner_train.py': config.get('trainer_sha256')}
    if direct and (not isinstance(expected_sources, dict) or set(expected_sources) != {
            'music_planner_cot.py', 'music_planner_segment_loss.py', 'music_planner_train.py',
            'music_planner_adapters.py', 'music_planner_eval.py'}):
        raise ValueError('Captioner direct checkpoint must pin its exact five training sources')
    for name, expected_sha in expected_sources.items():
        sources[name] = ref(source_dir / name)
        if sources[name]['sha256'] != expected_sha:
            raise ValueError('Training source identity differs; provide the exact frozen --training-source-dir')
    if receipt['inference_adapter_source']['sha256'] != expected_sources['music_planner_adapters.py']:
        raise ValueError('Actual inference adapter source differs from the trained native adapter; no implicit migration')
    trained_base = verify_base_identity(base, config['model'])
    files = marker.get('files')
    if not isinstance(files, list):
        raise ValueError('Portable checkpoint lacks recorded file integrity inventory')
    trainable_refs = [item for item in files if str(item.get('path', '')).replace('\\', '/').endswith('/trainable.pt')]
    if len(trainable_refs) != 1:
        raise ValueError('Portable checkpoint must identify exactly one trainable.pt')
    original_root = PurePosixPath(str(trainable_refs[0]['path']).replace('\\', '/')).parent
    declared = {}
    for item in files:
        original = PurePosixPath(str(item.get('path', '')).replace('\\', '/'))
        try:
            relative = original.relative_to(original_root).as_posix()
        except ValueError as error:
            raise ValueError('Checkpoint file inventory escapes original checkpoint root') from error
        if '..' in PurePosixPath(relative).parts or relative in declared:
            raise ValueError('Invalid or duplicate checkpoint inventory member')
        if relative == 'trainable.pt' or relative.startswith('adapter/'):
            declared[relative] = item
    export = inventory(checkpoint / 'adapter', recursive=True)
    consumed = {'adapter/' + item['relative_path']: item for item in export}
    consumed['trainable.pt'] = ref(checkpoint / 'trainable.pt')
    if set(consumed) != set(declared):
        raise ValueError('Actual portable adapter export differs from recorded files')
    for name, actual in consumed.items():
        if any(actual[key] != declared[name].get(key) for key in ('sha256', 'size_bytes')):
            raise ValueError('Portable checkpoint/adapter export bytes changed: ' + name)
    if consumed['trainable.pt']['sha256'] != marker.get('trainable_sha256'):
        raise ValueError('Portable trainable SHA differs from complete marker')
    metadata = read_json(checkpoint / 'adapter' / 'planner_adapter.json')
    if (metadata.get('family') != family or metadata.get('prefix_policy') != 'native-audio-only-no-target-in-prefix'
            or metadata.get('max_length') != config.get('max_length')
            or metadata.get('time_markers') is not (family == 'moss')):
        raise ValueError('Portable adapter native prefix/family/context metadata differs')
    if direct:
        exported_mode = read_json(checkpoint / 'adapter' / 'short_plan_adapter.json')
        assignments = ast.parse((source_dir / 'music_planner_cot.py').read_text(encoding='utf-8'))
        instruction = next((ast.literal_eval(node.value) for node in assignments.body
                            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                            and target.id == 'DIRECT_INSTRUCTION' for target in node.targets)), None)
        if (not isinstance(instruction, str) or exported_mode.get('schema') != config['schema']
                or exported_mode.get('mode') != 'direct-caption' or exported_mode.get('family') != 'omni_captioner'
                or exported_mode.get('require_shared_annotations') is not config['require_shared_annotations']
                or exported_mode.get('max_length') != config['max_length']
                or exported_mode.get('captioner_audio_only') is not True
                or exported_mode.get('instruction_sha256') != training_sha(instruction)):
            raise ValueError('Captioner direct export protocol/native-audio metadata differs')
    required = ('adapter_config.json', 'planner_adapter.json', 'tokenizer_config.json')
    if any(not (checkpoint / 'adapter' / name).is_file() for name in required) or not any(
            (checkpoint / 'adapter' / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        raise ValueError('Portable adapter lacks actual weights/config/native processor export')
    receipt.update(mode='fair_portable_checkpoint', checkpoint=dict(
        root=str(checkpoint), trainer_state=ref(checkpoint / 'trainer_state.json'), run_config=ref(config_path),
        training_fingerprint=fingerprint, training_sources=sources, consumed_files=consumed,
        training_signature=signature, adapter_metadata=metadata,
        training_protocol=config.get('schema', 'original-fair-direct-action'),
        trained_base_verification=trained_base,
        training_base_weight_content_sha_verified=False,
        limitation='Historical training metadata pins weight membership/size, not every base weight content SHA'))
    return receipt


def verify_loaded_portable(adapter, receipt):
    """Validate the actual loaded learned tensors against portable training tensors."""
    import torch
    if adapter.family != receipt['family']:
        raise ValueError('Loaded adapter family differs from audited artifacts')
    if receipt['mode'] == 'native_base_only':
        if getattr(adapter.model, 'peft_config', None):
            raise ValueError('Base-only collection unexpectedly loaded an adapter')
        return dict(schema='music_planner_loaded_artifact_verification_v1', mode=receipt['mode'],
                    artifact_identity_sha256=object_sha(receipt), loaded_portable_state_verified=False)
    checkpoint = receipt['checkpoint']
    # Rehash consumed files after loading to close accidental preflight/load drift.
    for item in checkpoint['consumed_files'].values():
        if ref(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Portable file changed while loading')
    saved = torch.load(checkpoint['consumed_files']['trainable.pt']['path'], map_location='cpu', weights_only=True)
    signature = checkpoint['training_signature']
    metadata = checkpoint['adapter_metadata']['trainable_info']['trainable']
    if not isinstance(saved, dict) or set(saved) != set(signature) or set(saved) != {p['name'] for p in metadata}:
        raise ValueError('Portable learned tensor keys differ from trained/exported scope')
    parameters = dict(adapter.model.named_parameters())
    if {name for name in parameters if 'lora_' in name or '.modules_to_save.' in name} != set(saved):
        raise ValueError('Actual loaded adapter learned scope differs from portable training scope')
    for name, value in saved.items():
        actual, spec = parameters.get(name), signature[name]
        if (not isinstance(value, torch.Tensor) or actual is None or list(value.shape) != spec['shape']
                or str(value.dtype) != spec['dtype'] or actual.dtype != value.dtype
                or not torch.isfinite(value).all().item() or not torch.equal(actual.detach().cpu(), value)):
            raise ValueError('Actual loaded adapter differs from portable learned tensor: ' + name)
    return dict(schema='music_planner_loaded_artifact_verification_v1', mode=receipt['mode'],
                artifact_identity_sha256=object_sha(receipt), loaded_portable_state_verified=True)
