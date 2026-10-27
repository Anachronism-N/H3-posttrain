"""MOSS v2: select a trained style, prefill its assistant header, generate actions.

This is a single-device inference entry, never a training launcher. The default
uses an input-only JEV forward followed by a separate, full native MOSS generation
prefill containing the selected assistant header. No KV cache is shared between
these forwards. Explicit --selected-genre is a controlled intervention, not a
claim that an untrained head has predicted the style.

Example: python -m scripts_omnidance.exp.ms.music_planner_moss_generate generate
  --model /models/moss --checkpoint /runs/moss/checkpoint-0000100 --row audio.json
  --output result.json --device cuda:0 --max-new-tokens 1024

audio.json contains id, source_group_id, audio_path, audio_sha256,
audio_duration_seconds and audio_window_seconds=[0,duration]. Target/genre/window
labels, if present in a caller's row, are never read or forwarded. Output preserves
every returned token and special token. A failed stop/format check has caption=null;
it is never repaired or trimmed into a successful example.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from . import music_planner_caption_format as caption_format
from . import music_planner_jev as jev
from . import music_planner_moss_generation_patch as cache_patch
from . import music_planner_moss_position_buffer_patch as position_patch
from . import music_planner_generation_eos_patch as eos_patch
from . import music_planner_train as trainer


CONTRACT_SCHEMA = 'music-planner-caption-contract-v2'
PROTOCOL_SCHEMA = 'music-planner-conditioned-caption-v1'
TOKEN_PROTOCOL = 'header-then-actions-then-native-im-end-v1'
SOURCE_KEYS = ('id', 'source_group_id', 'audio_path', 'audio_sha256',
               'audio_duration_seconds', 'audio_window_seconds')
HEX = re.compile(r'^[0-9a-f]{64}$')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path), size_bytes=path.stat().st_size)


def _recorded_root(marker):
    portable = [ref for ref in marker.get('files', []) if isinstance(ref, dict) and
                isinstance(ref.get('path'), str) and ref['path'].replace('\\', '/').endswith('/trainable.pt')]
    if len(portable) != 1 or portable[0].get('sha256') != marker.get('trainable_sha256'):
        raise ValueError('Checkpoint needs one recorded portable state root')
    return portable[0]['path'].replace('\\', '/')[:-len('/trainable.pt')]


def _member(checkpoint, reference, recorded_root):
    """Resolve recorded checkpoint members after relocating the complete folder."""
    recorded = reference.get('path')
    if not isinstance(recorded, str):
        raise ValueError('Checkpoint member has no recorded path')
    normalized = recorded.replace('\\', '/')
    if not normalized.startswith(recorded_root + '/'):
        raise ValueError('Checkpoint member is outside its recorded directory')
    relative = normalized[len(recorded_root) + 1:]
    if not relative or any(part in ('..', '.', '') or ':' in part for part in relative.split('/')):
        raise ValueError('Checkpoint member escapes its directory')
    resolved = (checkpoint / relative).resolve()
    if checkpoint not in resolved.parents:
        raise ValueError('Checkpoint member resolves outside the checkpoint')
    return relative, resolved


def verify_checkpoint(checkpoint):
    """One-time CLI verification, including all recorded distributed file hashes."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    marker = _read(checkpoint / 'trainer_state.json')
    config = _read(checkpoint.parent / 'run_config.json')
    canonical = {key: value for key, value in config.items() if key not in ('fingerprint', 'trainable_parameters')}
    fingerprint = trainer.object_sha(canonical)
    contract = config.get('data', {}).get('contract')
    cursor = marker.get('cursor', {})
    if (marker.get('schema') != 'music-planner-train-v1' or marker.get('complete') is not True or
            marker.get('fingerprint') != fingerprint or config.get('fingerprint') != fingerprint or
            config.get('family') != 'moss' or config.get('data', {}).get('test_used') is not False or
            not isinstance(contract, dict) or contract.get('schema') != CONTRACT_SCHEMA or
            type(cursor.get('update')) is not int or cursor['update'] <= 0 or
            type(marker.get('world_size')) is not int or marker['world_size'] <= 0 or
            marker['world_size'] != config.get('world_size') or type(marker.get('zero3')) is not bool):
        raise ValueError('Require a complete, updated MOSS checkpoint trained with caption contract v2')
    refs = marker.get('files')
    if not isinstance(refs, list) or not refs:
        raise ValueError('Complete checkpoint has no recorded file inventory')
    names = trainer.checkpoint_member_refs(checkpoint, marker, world=marker['world_size'], zero3=marker['zero3'])
    if (not {'trainable.pt', 'adapter/dance_caption_protocol.json'}.issubset(names) or
            marker.get('trainable_sha256') != _sha(checkpoint / 'trainable.pt')):
        raise ValueError('Checkpoint portable weights or conditioned-caption protocol is missing')
    return dict(checkpoint_path=str(checkpoint), marker=marker, config=config,
                marker_ref=_ref(checkpoint / 'trainer_state.json'),
                run_config_ref=_ref(checkpoint.parent / 'run_config.json'),
                all_recorded_checkpoint_bytes_verified=True)


def _protocol(adapter):
    if getattr(adapter, 'family', None) != 'moss':
        raise ValueError('Conditioned generation supports native MOSS only')
    contract = getattr(adapter, 'contract', None)
    if not isinstance(contract, dict) or contract.get('schema') != CONTRACT_SCHEMA:
        raise ValueError('Only trained caption contract v2 is supported; old protocols are not inferred')
    classes = jev.validate_classes(getattr(adapter, 'genre_classes', None))
    spec = getattr(adapter, 'dance_format', None)
    if spec != contract.get('dance_caption_format'):
        raise ValueError('Adapter dance format differs from its training contract')
    headers = {genre: adapter.tokenizer.encode(caption_format.format_header(genre, classes, spec),
                                              add_special_tokens=False) for genre in classes}
    expected = dict(schema=PROTOCOL_SCHEMA, family='moss', classes=classes,
                    classes_sha256=jev.classes_identity(classes), dance_format=spec,
                    header_token_ids=headers, token_protocol=TOKEN_PROTOCOL)
    policy = contract.get('genre_supervision')
    if policy is not None:
        if (policy != dict(unknown_label='unknown', unknown_policy='mask', mask_unknown_header=True) or
                getattr(adapter, 'genre_supervision', None) != policy or policy['unknown_label'] not in classes or
                len(classes) < 3):
            raise ValueError('Unknown abstention policy differs from the trained caption protocol')
        expected['genre_supervision'] = policy
    elif getattr(adapter, 'genre_supervision', None) is not None:
        raise ValueError('Adapter declares an unknown policy absent from its trained contract')
    if (getattr(adapter, 'token_protocol', None) != TOKEN_PROTOCOL or
            getattr(adapter, 'header_token_ids', None) != headers or
            getattr(adapter, 'dance_caption_protocol', None) != expected):
        raise ValueError('Training/inference header IDs, vocabulary, or token protocol differ')
    identity = getattr(adapter, 'training_protocol_identity', None)
    if not isinstance(identity, dict) or not identity.get('checkpoint_path'):
        raise ValueError('Restore this complete checkpoint protocol before conditioned inference')
    checkpoint = Path(identity['checkpoint_path']).resolve()
    marker_path = checkpoint / 'trainer_state.json'
    marker = _read(marker_path)
    config = _read(checkpoint.parent / 'run_config.json')
    canonical = {key: value for key, value in config.items() if key not in ('fingerprint', 'trainable_parameters')}
    if (identity.get('family') != 'moss' or identity.get('marker_sha256') != _sha(marker_path) or
            marker.get('schema') != 'music-planner-train-v1' or marker.get('complete') is not True or
            not HEX.fullmatch(str(identity.get('fingerprint', ''))) or
            identity.get('fingerprint') != marker.get('fingerprint') or
            trainer.object_sha(canonical) != marker.get('fingerprint') or
            config.get('fingerprint') != marker.get('fingerprint') or config.get('family') != 'moss' or
            config.get('data', {}).get('contract') != contract or config.get('max_length') != adapter.max_length or
            type(adapter.max_length) is not int or adapter.max_length <= 0 or
            identity.get('trainable_sha256') != marker.get('trainable_sha256') or
            identity.get('cursor') != marker.get('cursor') or
            type(marker.get('cursor', {}).get('update')) is not int or marker['cursor']['update'] <= 0):
        raise ValueError('Loaded protocol checkpoint marker/fingerprint/cursor identity differs')
    if _read(checkpoint / 'adapter' / 'dance_caption_protocol.json') != expected:
        raise ValueError('Loaded protocol differs from the checkpoint export')
    root = _recorded_root(marker)
    refs = [reference for reference in marker.get('files', [])
            if _member(checkpoint, reference, root)[0] == 'adapter/dance_caption_protocol.json']
    if len(refs) != 1 or refs[0]['sha256'] != _sha(checkpoint / 'adapter' / 'dance_caption_protocol.json'):
        raise ValueError('Conditioned-caption export is not bound to the complete checkpoint')
    return classes, spec, headers, identity


def install_runtime_patches(adapter, official_model_path):
    """Install portable, eval-only fixes BEFORE even the input-only head forward."""
    adapter.position_buffer_patch_receipt = position_patch.restore_moss_position_buffer(adapter, official_model_path)
    cache_patch.apply_moss_generation_patch(adapter)
    eos_patch.apply_generation_eos_patch(adapter)
    return _require_runtime(adapter)


def _require_runtime(adapter):
    model = adapter.model.get_base_model() if hasattr(adapter.model, 'get_base_model') else adapter.model
    if (getattr(model.config, 'model_type', None) != 'moss_music' or
            getattr(adapter, 'applied_generation_patch_ref', None) != cache_patch.generation_patch_ref() or
            getattr(getattr(model, 'prepare_inputs_for_generation', None), '__func__', None)
            is not cache_patch._prepare_inputs_for_generation or
            vars(model).get('_planner_generation_audio_token_id') != adapter.processor.audio_token_id):
        raise ValueError('Native MOSS cached-decode compatibility patch is required')
    receipt = getattr(adapter, 'position_buffer_patch_receipt', None)
    if (not isinstance(receipt, dict) or
            receipt.get('applied_position_buffer_patch_ref') != position_patch.position_buffer_patch_ref() or
            receipt.get('parameters_changed') is not False or receipt.get('persistent') is not False or
            receipt.get('training_identity_preserved') is not True):
        raise ValueError('Verified native MOSS position-buffer restoration is required')
    owner = model.audio_encoder.embed_positions
    buffer = owner._buffers.get('inv_timescales')
    if (not isinstance(buffer, torch.Tensor) or buffer.shape != (640,) or buffer.dtype != torch.float32 or
            buffer.requires_grad or buffer.is_meta or not torch.isfinite(buffer).all() or not (buffer > 0).all() or
            'inv_timescales' not in owner._non_persistent_buffers_set or
            position_patch._tensor_ref(buffer) != receipt.get('canonical_tensor_ref')):
        raise ValueError('Current MOSS position buffer differs from the restored native canonical tensor')
    eos = adapter.eos_token_ids()
    terminal = adapter.tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if (len(terminal) != 1 or terminal[0] not in eos or
            any(type(token) is not int or token < 0 for token in eos)):
        raise ValueError('Native assistant terminator must be a single supported EOS token')
    native_eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    if not eos_patch._valid_eos(native_eos) and (
            getattr(adapter, 'applied_generation_eos_patch_ref', None) != eos_patch.generation_eos_patch_ref() or
            vars(model).get('_planner_generation_eos_patch_ref') != eos_patch.generation_eos_patch_ref()):
        raise ValueError('Missing native EOS requires the verified EOS bridge')
    return dict(cache_patch=cache_patch.generation_patch_ref(), position_patch=position_patch.position_buffer_patch_ref(),
                eos_patch=getattr(adapter, 'applied_generation_eos_patch_ref', None),
                eos_token_ids=eos, assistant_terminator_id=terminal[0], model=model)


def _audio_source(row, contract):
    # Deliberately never enumerate/get target_text, genre, target_windows, or any
    # other annotation. A caller may pass a full row; only these input fields live.
    source = {key: row[key] for key in SOURCE_KEYS}
    for key in ('id', 'source_group_id', 'audio_path'):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ValueError('Nonempty audio source identity/path required: ' + key)
    duration = source['audio_duration_seconds']
    if (not HEX.fullmatch(str(source['audio_sha256'])) or type(duration) not in (int, float) or
            not math.isfinite(duration) or not contract['audio_min_seconds'] <= duration <= contract['audio_max_seconds'] or
            source['audio_window_seconds'] != [0., duration]):
        raise ValueError('Require declared finite, complete-source audio identity/duration/window')
    return source


def _prefill(adapter, source, header_ids):
    batch = adapter.prefix(source)
    ids, mask = batch.get('input_ids'), batch.get('attention_mask')
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long or
            not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or not mask.bool().all() or
            'labels' in batch or 'position_ids' in batch or 'jev_prefix_end' in batch or
            any('pixel_values' in key or 'grid_thw' in key for key in batch)):
        raise ValueError('Require one exact, unpadded native audio-only input prefix')
    audio_mask = batch.get('audio_input_mask')
    if (not isinstance(audio_mask, torch.Tensor) or audio_mask.dtype != torch.bool or audio_mask.shape != ids.shape or
            not torch.equal(audio_mask, ids.eq(adapter.processor.audio_token_id))):
        raise ValueError('Native MOSS audio mask must exactly match its original audio positions')
    forced = ids.new_tensor([header_ids])
    result = dict(batch, input_ids=torch.cat((ids, forced), 1),
                  attention_mask=torch.cat((mask, torch.ones_like(forced, dtype=mask.dtype)), 1),
                  audio_input_mask=torch.cat((audio_mask, torch.zeros_like(forced, dtype=torch.bool)), 1))
    for key in ('token_type_ids', 'mm_token_type_ids'):
        if key in batch:
            if batch[key].shape != ids.shape:
                raise ValueError('Native token type mask is not aligned')
            result[key] = torch.cat((batch[key], torch.zeros_like(forced, dtype=batch[key].dtype)), 1)
    return result, ids


def _decode(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@torch.no_grad()
def generate_conditioned(adapter, row, max_new_tokens=1024, selected_genre=None, min_style_confidence=None):
    """Return auditable raw output; metadata errors reject before any forward.

    Callers loading adapters themselves must load the matching native PEFT bytes,
    call restore_adapter_protocol, and install_runtime_patches before this function.
    The protocol helper verifies portable bindings, not the caller's actual base
    weights. The CLI below additionally verifies base inventory when available.
    """
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    classes, spec, headers, identity = _protocol(adapter)
    runtime = _require_runtime(adapter)
    source = _audio_source(row, adapter.contract)
    policy = adapter.contract.get('genre_supervision')
    unknown = policy['unknown_label'] if policy is not None else None
    probability_classes = [name for name in classes if name != unknown] if policy is not None else list(classes)
    forced = selected_genre is not None
    if forced and (not isinstance(selected_genre, str) or selected_genre not in classes):
        raise ValueError('Selected genre must belong to the trained vocabulary')
    if not forced and identity['cursor'].get('jev_head_exercised') is not True:
        raise ValueError('Default style prediction requires checkpoint proof that the JEV head was exercised')
    if min_style_confidence is not None and (
            type(min_style_confidence) not in (int, float) or not math.isfinite(min_style_confidence) or
            not 0 <= min_style_confidence <= 1 or policy is None or forced):
        raise ValueError('Explicit confidence threshold requires the trained unknown policy, a finite [0,1] value, and head selection')
    if not forced:
        capture = vars(adapter.model).get('_planner_jev_capture')
        if capture is None or capture.metadata.get('classes') != probability_classes:
            raise ValueError('JEV head vocabulary differs from the trained semantic classes (unknown is not a class)')
    result = dict(schema='music-planner-moss-conditioned-generation-v1', family='moss',
                  checkpoint_fingerprint=identity['fingerprint'], checkpoint_marker_sha256=identity['marker_sha256'],
                  source_id=source['id'], audio_sha256=source['audio_sha256'],
                  selection_mode='explicit-style-intervention' if forced else 'trained-jev-argmax',
                  selected_genre=selected_genre, probabilities=None, probability_classes=None, calibrated=False,
                  selected_genre_is_unknown=unknown is not None and selected_genre == unknown,
                  min_style_confidence=min_style_confidence, style_confidence=None,
                  style_selection_reason=('explicit-unknown-abstention' if selected_genre == unknown and unknown is not None
                                          else 'explicit-style-intervention' if forced else None),
                  confidence_threshold_source='explicit-user-parameter' if min_style_confidence is not None else None,
                  genre_supervision=policy,
                  caption=None, caption_valid=False, validation_errors=[], raw_new_token_ids=None,
                  raw_sequence_token_ids=None, raw_text=None, action_text=None, stop_reason='not_generated',
                  max_new_tokens=max_new_tokens, forced_header_text=None, forced_header_token_ids=None,
                  forced_header_tokens=0, new_tokens=None,
                  native_prefix_token_ids=None, expected_prefills=1 if forced else 2,
                  actual_style_forward_count=0, actual_generation_forward_count=0,
                  kv_cache_shared_between_selection_and_generation=False, target_consumed=False,
                  native_runtime_patches={key: value for key, value in runtime.items() if key != 'model'},
                  inference_base_identity=getattr(adapter, 'inference_base_identity', None),
                  decoding=dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
                                no_repeat_ngram_size=0, encoder_repetition_penalty=1.0, use_cache=True))
    native = runtime['model']
    counts = {'style': 0, 'generation': 0}
    phase = ['style']
    validator = position_patch.install_moss_generation_logits_validator(adapter)
    def observe(module, inputs, output):
        counts[phase[0]] += 1
    observer = native.register_forward_hook(observe)
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        if not forced:
            prediction = jev.predict_jev(adapter, source)
            probabilities = prediction['probabilities']
            if (prediction.get('target_consumed') is not False or prediction.get('classes') != probability_classes or
                    not isinstance(probabilities, torch.Tensor) or probabilities.shape != (1, len(probability_classes)) or
                    not torch.isfinite(probabilities).all() or (probabilities < 0).any() or
                    not torch.allclose(probabilities.sum(-1), torch.ones(1, device=probabilities.device), atol=1e-6)):
                raise ValueError('Input-only JEV probability/vocabulary evidence differs')
            if not counts['style']:
                raise ValueError('No actual native forward occurred for style prediction')
            confidence = float(probabilities.max().item())
            selected_genre = probability_classes[int(probabilities.argmax(-1).item())]
            reason = 'argmax-without-confidence-threshold' if min_style_confidence is None else 'explicit-confidence-threshold-met'
            if min_style_confidence is not None and confidence < min_style_confidence:
                selected_genre = unknown
                reason = 'below-explicit-confidence-threshold-abstention'
                result['selection_mode'] = 'confidence-threshold-abstention'
            result['selected_genre'] = selected_genre
            result['probabilities'] = probabilities.detach().cpu().tolist()[0]
            result['probability_classes'] = list(probability_classes)
            result['selected_genre_is_unknown'] = unknown is not None and selected_genre == unknown
            result['style_confidence'] = confidence
            result['style_selection_reason'] = reason
        header = caption_format.format_header(selected_genre, classes, spec)
        header_ids = headers[selected_genre]
        if (not header_ids or any(type(token) is not int or token < 0 or
                token in runtime['eos_token_ids'] or token == adapter.processor.audio_token_id for token in header_ids) or
                _decode(adapter.tokenizer, header_ids) != header):
            raise ValueError('Forced header must reproduce the exact trained plain-text header tokens')
        batch, prefix_ids = _prefill(adapter, source, header_ids)
        result.update(forced_header_text=header, forced_header_token_ids=list(header_ids),
                      forced_header_tokens=len(header_ids),
                      native_prefix_token_ids=prefix_ids[0].cpu().tolist(), input_tokens=batch['input_ids'].shape[1])
        if batch['input_ids'].shape[1] + max_new_tokens > adapter.max_length:
            raise ValueError('Native prefix plus forced header plus action budget exceeds trained context; no truncation')
        phase[0] = 'generation'
        parameter = next(adapter.model.parameters())
        autocast = torch.autocast(parameter.device.type, dtype=torch.bfloat16) if parameter.dtype == torch.bfloat16 else nullcontext()
        with autocast:
            output = adapter.model.generate(**adapter._move(batch), **result['decoding'],
                max_new_tokens=max_new_tokens, eos_token_id=list(runtime['eos_token_ids']))
        sequences = output.sequences if hasattr(output, 'sequences') else output
        if isinstance(sequences, tuple) and sequences:
            sequences = sequences[0]
        if isinstance(sequences, torch.Tensor):
            result['raw_sequence_token_ids'] = sequences.detach().cpu().tolist()
        if (not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1 or
                sequences.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Native MOSS generate did not return one integer token sequence')
        offset = batch['input_ids'].shape[1]
        new_ids = sequences[0, offset:].detach().cpu().tolist()
        result['raw_new_token_ids'] = new_ids
        result['new_tokens'] = len(new_ids)
        result['raw_text'] = _decode(adapter.tokenizer, new_ids)
        if not torch.equal(sequences[:, :offset].cpu(), batch['input_ids'].cpu()):
            raise ValueError('Native generation did not preserve the exact prefix and forced header')
        first_eos = next((index for index, token in enumerate(new_ids) if token in runtime['eos_token_ids']), None)
        result['first_eos_index'] = first_eos
        result['stop_reason'] = ('eos' if first_eos == len(new_ids) - 1 and first_eos is not None else
                                 'tokens_after_eos' if first_eos is not None else
                                 'max_new_tokens' if len(new_ids) == max_new_tokens else 'unknown_stop')
        if not counts['generation']:
            raise ValueError('No actual native generation forward observed')
        if not new_ids or len(new_ids) > max_new_tokens:
            raise ValueError('Empty or over-budget native continuation')
        if first_eos != len(new_ids) - 1 or new_ids[-1] != runtime['assistant_terminator_id']:
            raise ValueError('Continuation must end at its FIRST native assistant EOS; raw tokens retained')
        actions = _decode(adapter.tokenizer, new_ids[:-1])
        result['action_text'] = actions
        parsed = caption_format.parse_caption(header + actions, classes, spec, source['audio_duration_seconds'])
        if parsed['genre'] != selected_genre or not parsed['action_text'].strip():
            raise ValueError('Generated actions/genre differ from the forced trained protocol')
        result.update(caption=header + actions, caption_valid=True, parsed=parsed)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        result['validation_errors'].append(type(error).__name__ + ': ' + str(error))
        result['caption'] = None
        result['caption_valid'] = False
    finally:
        observer.remove()
        validator.remove()
        adapter.model.train(was_training)
        result['actual_style_forward_count'] = counts['style']
        result['actual_generation_forward_count'] = counts['generation']
    return result


def load_conditioned_adapter(model_path, checkpoint, device='cpu', dtype=torch.bfloat16):
    """Load portable PEFT, restore the exact v2 protocol/head, then native patches."""
    from .music_planner_adapters import load_planner_adapter
    from .music_planner_schema import restore_adapter_protocol
    from .music_planner_cluster import model_inventory
    verified = verify_checkpoint(checkpoint)
    config = verified['config']
    model_path = Path(model_path).expanduser().resolve()
    expected = config.get('model', {})
    expected_sha = expected.get('content_inventory_sha256')
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or not HEX.fullmatch(expected_sha):
            raise ValueError('Training base content inventory SHA is malformed')
        inventory = model_inventory(model_path)
        if inventory['inventory_sha256'] != expected_sha:
            raise ValueError('Actual inference base model contents differ from the trained inventory')
        base_identity = dict(content_verified=True, inventory_sha256=expected_sha, identity_method='full-local-content-inventory')
    else:
        if expected.get('config_sha256') != _sha(model_path / 'config.json'):
            raise ValueError('Legacy base config SHA differs')
        weights = expected.get('weights')
        if not isinstance(weights, list) or not weights:
            raise ValueError('Legacy training identity has no declared base weights')
        for weight in weights:
            path = model_path / weight['name']
            if path.parent != model_path or not path.is_file() or path.stat().st_size != weight['size']:
                raise ValueError('Legacy declared base weight name/size differs')
        base_identity = dict(content_verified=False, identity_method='legacy-config-SHA-and-weight-size-only',
                             limitation='Training recorded no base content inventory; weight bytes are not proven')
    native = load_planner_adapter('moss', model_path, checkpoint=checkpoint, device=device, dtype=dtype,
                                  max_length=config['max_length'])
    adapter = restore_adapter_protocol(native, checkpoint)
    adapter.inference_base_identity = base_identity
    _protocol(adapter)
    install_runtime_patches(adapter, model_path)
    adapter.conditioned_checkpoint_verification = {key: value for key, value in verified.items() if key not in ('marker', 'config')}
    return adapter


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = result.add_subparsers(dest='action', required=True)
    generate = actions.add_parser('generate')
    generate.add_argument('--model', type=Path, required=True)
    generate.add_argument('--checkpoint', type=Path, required=True)
    generate.add_argument('--row', type=Path, required=True)
    generate.add_argument('--output', type=Path, required=True)
    generate.add_argument('--device', default='cpu')
    generate.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    generate.add_argument('--max-new-tokens', type=int, default=1024)
    generate.add_argument('--selected-genre')
    generate.add_argument('--min-style-confidence', type=float,
                          help='Explicit uncalibrated max-P threshold; only trained unknown-policy profiles can abstain')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise ValueError('Output exists; preserve the previous raw generation and choose a fresh output')
    row = _read(args.row)
    adapter = load_conditioned_adapter(args.model, args.checkpoint, device=args.device,
                                        dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float32)
    result = generate_conditioned(adapter, row, max_new_tokens=args.max_new_tokens, selected_genre=args.selected_genre,
                                  min_style_confidence=args.min_style_confidence)
    result['checkpoint_verification'] = adapter.conditioned_checkpoint_verification
    result['inference_source_ref'] = _ref(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()), caption_valid=result['caption_valid'],
                          actual_training_started=False), ensure_ascii=False))
    return 0 if result['caption_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
