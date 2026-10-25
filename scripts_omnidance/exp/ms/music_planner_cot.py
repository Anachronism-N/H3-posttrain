"""Independent, explicitly reviewed short-plan SFT experiment; never edits fair SFT."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts_omnidance.exp.ms import music_planner_train as base
from scripts_omnidance.exp.ms import music_planner_eval as evaluation
from scripts_omnidance.exp.ms import music_planner_generation_eos_patch as eos_patch
from scripts_omnidance.exp.ms import music_planner_moss_generation_patch as moss_patch
from scripts_omnidance.exp.ms import music_planner_moss_position_buffer_patch as position_patch
from scripts_omnidance.exp.ms import music_planner_segment_loss as efficient_loss
from scripts_omnidance.exp.ms.music_planner_adapters import (
    FAMILIES, PlannerAdapter, load_planner_adapter,
)

SCHEMA = 'music-planner-captioner-ablation-v3'
MODES = ('short-plan', 'direct-caption')
SEGMENTS = ('music_evidence', 'short_plan', 'caption')
MUSIC_SOURCES = ('human_audio_review', 'human_reviewed_audio_teacher', 'human_reviewed_mir')
PLAN_SOURCES = ('human_choreography_plan', 'human_reviewed_teacher_plan')
INSTRUCTION = (
    'Listen to the full music. First state concise music-only evidence in '
    '<music_evidence>...</music_evidence>, then a short choreography plan in '
    '<short_plan>...</short_plan>, then the final natural-language choreography in '
    '<caption>...</caption>. Distinguish audible evidence from creative movement '
    'choices. The final caption must cover [0-2s] and [2-5.062s]. '
    'Do not invent musical observations or infer a unique observed dance from music.'
)
DIRECT_INSTRUCTION = (
    'Listen to the full music and describe dance choreography in two time windows: '
    '[0-2s] and [2-5.062s]. Output only the final two time-marked natural-language '
    'choreography descriptions, without music analysis, planning traces or XML tags.'
)


def mode_schema(mode, require_shared_annotations=False):
    if mode not in MODES:
        raise ValueError('Unknown reviewed planner mode: ' + str(mode))
    policy = 'reviewed-shared' if mode == 'short-plan' or require_shared_annotations else 'weak-caption'
    return SCHEMA + '/' + mode + '/' + policy


def effective_weights(mode, values=None):
    mode_schema(mode)
    weights = tuple(values if values is not None else ((.2, .3, 1.) if mode == 'short-plan' else (0., 0., 1.)))
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Three finite nonnegative weights with a positive sum are required')
    if mode == 'direct-caption' and (weights[0] or weights[1] or not weights[2]):
        raise ValueError('Direct-caption mode requires weights 0 0 positive-caption-weight')
    return weights


def implementation_sources():
    paths = (__file__, base.__file__, evaluation.__file__,
             __import__(PlannerAdapter.__module__, fromlist=['']).__file__, efficient_loss.__file__)
    return {Path(path).name: base.file_sha(path) for path in paths}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing nonempty ' + name)
    if '<|' in value or '<audio>' in value or any(
            '<' + tag in value or '</' + tag in value for tag in SEGMENTS):
        raise ValueError('Reserved assistant/segment tag in ' + name)
    return value.strip()


def validate_annotations(row, require_review=True):
    """Validate declared provenance; this does not perform or claim semantic review."""
    try:
        json.dumps(row, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError('Annotations must contain finite JSON values') from error
    caption = _text(row.get('target_text'), 'target_text')
    timing = evaluation.parse_windows(caption)
    if not timing['matches_fixed_windows'] or not timing['nonempty_window_text']:
        raise ValueError('Target caption must contain the complete two fixed time windows')
    if not require_review:
        return row
    for name, sources in (('music_evidence', MUSIC_SOURCES), ('short_plan', PLAN_SOURCES)):
        annotation = row.get(name)
        if not isinstance(annotation, dict):
            raise ValueError('Missing reviewed ' + name + '; no automatic ground truth is fabricated')
        _text(annotation.get('text'), name)
        provenance = annotation.get('provenance', {})
        if (not isinstance(provenance, dict) or provenance.get('source') not in sources or
                provenance.get('human_approved') is not True or
                not isinstance(provenance.get('reviewer'), str) or not provenance['reviewer'].strip() or
                not isinstance(provenance.get('version'), str) or not provenance['version'].strip()):
            raise ValueError(name + ' requires an allowed source, explicit human approval, reviewer and version')
        if provenance.get('audio_sha256') != row.get('audio_sha256'):
            raise ValueError(name + ' is not bound to this audio SHA256')
        if name == 'music_evidence':
            if provenance.get('input_modalities') != ['audio']:
                raise ValueError('music_evidence must declare audio-only observation, without video/motion targets')
        elif provenance.get('target_caption_sha256') != hashlib.sha256(caption.encode('utf-8')).hexdigest():
            raise ValueError('short_plan approval is not bound to the final target caption')
    return row


def check_data(train_path, val_path, mode='short-plan', require_shared_annotations=False):
    train, val, metadata = base.check_data(train_path, val_path)
    required = mode == 'short-plan' or require_shared_annotations
    for row in train + val:
        validate_annotations(row, required)
    metadata = dict(metadata, annotation_schema=mode_schema(mode, required), test_used=False,
                    provenance_checked=required, semantic_approval=False)
    return train, val, metadata


def serialize_target(row, mode='short-plan', require_shared_annotations=False):
    validate_annotations(row, mode == 'short-plan' or require_shared_annotations)
    mode_schema(mode)
    if mode == 'direct-caption':
        text = row['target_text'].strip()
        return text, [(0, len(text))]
    values = (row['music_evidence']['text'], row['short_plan']['text'], row['target_text'])
    text, bounds = '', []
    for name, value in zip(SEGMENTS, values):
        start = len(text)
        text += '<' + name + '>\n' + value.strip() + '\n</' + name + '>'
        if name != SEGMENTS[-1]:
            text += '\n'
        bounds.append((start, len(text)))
    return text, bounds


def tokenized_target(tokenizer, row, eos_ids, mode='short-plan', require_shared_annotations=False):
    """Tokenize once; offsets, not separately encoded substrings, define ownership.

    A token crossing a segment boundary belongs to the segment containing its
    final character. This deterministic rule includes tags/whitespace in loss.
    """
    text, bounds = serialize_target(row, mode, require_shared_annotations)
    active = (0, 1, 2) if mode == 'short-plan' else (2,)
    if mode == 'direct-caption':
        ids = tokenizer.encode(text, add_special_tokens=False)
        terminator = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        if (not ids or len(terminator) != 1 or terminator[0] not in eos_ids or
                any(token in eos_ids for token in ids)):
            raise ValueError('Require nonempty direct caption and exactly one appended native assistant EOS')
        return ids + terminator, [2] * (len(ids) + 1), dict(text=text, boundary_crossing_tokens=0)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    except (NotImplementedError, TypeError, KeyError) as error:
        raise ValueError('Short-plan loss requires a tokenizer with exact character offset mapping') from error
    if ids != tokenizer.encode(text, add_special_tokens=False) or len(ids) != len(offsets):
        raise ValueError('Tokenizer offsets and full-target encoding disagree')
    ownership, crossing = [], 0
    for left, right in offsets:
        if not (0 <= left < right <= len(text)):
            raise ValueError('Invalid/empty target token character offsets')
        local = next((i for i, (start, end) in enumerate(bounds) if start <= right - 1 < end), None)
        if local is None:
            raise ValueError('Target token lies outside all segments')
        crossing += int(left < bounds[local][0])
        ownership.append(active[local])
    if set(ownership) != set(active):
        raise ValueError('Every target segment must contain at least one supervised token')
    terminator = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if len(terminator) != 1 or terminator[0] not in eos_ids or any(i in eos_ids for i in ids):
        raise ValueError('Require exactly one native assistant EOS, appended only after the complete target')
    return ids + terminator, ownership + [2], dict(text=text, boundary_crossing_tokens=crossing)


def segment_nll(logits, labels, segment_ids, weights=None, mode='short-plan', chunk_size=0):
    """Weighted mean of the three segment means, with causal shift and EOS."""
    weights = effective_weights(mode, weights)
    active_segments = (0, 1, 2) if mode == 'short-plan' else (2,)
    if labels.shape != segment_ids.shape or logits.shape[:2] != labels.shape:
        raise ValueError('Segment labels/logits shape mismatch')
    shifted_labels, shifted_segments = labels[:, 1:], segment_ids[:, 1:]
    active = shifted_labels.ne(-100)
    if not torch.equal(active, shifted_segments.ge(0)) or bool((shifted_segments > 2).any()):
        raise ValueError('Assistant labels and segment ownership must match exactly')
    sums, counts = efficient_loss.assistant_segment_sums(logits, shifted_labels, shifted_segments, chunk_size)
    if any(int(counts[index]) == 0 for index in active_segments) or any(
            int(counts[index]) for index in range(3) if index not in active_segments):
        raise ValueError('Segment membership differs from the selected ablation protocol')
    loss = sum(weights[index] * sums[index] / counts[index] for index in active_segments) / sum(weights)
    return loss, sums, counts


def extract_caption(raw):
    import re
    pattern = (r'\s*<music_evidence>\s*(.*?)\s*</music_evidence>\s*'
               r'<short_plan>\s*(.*?)\s*</short_plan>\s*<caption>\s*(.*?)\s*</caption>\s*')
    match = re.fullmatch(pattern, raw, flags=re.DOTALL)
    if match is None or any(not value.strip() for value in match.groups()):
        raise ValueError('Generated output lacks a complete, ordered short-plan response')
    values = [_text(value, name) for name, value in zip(SEGMENTS, match.groups())]
    return dict(zip(SEGMENTS, values))


def validate_completion_tokens(generated, budget):
    ids, eos = generated.get('new_token_ids'), generated.get('eos_token_ids')
    if (type(budget) is not int or budget <= 0 or not isinstance(ids, list) or not ids or
            len(ids) > budget or any(type(value) is not int or value < 0 for value in ids) or
            not isinstance(eos, list) or not eos or any(type(value) is not int or value < 0 for value in eos)):
        raise ValueError('Invalid raw generation token IDs or generation budget')
    if generated.get('stop_reason') != 'eos' or ids[-1] not in eos or any(value in eos for value in ids[:-1]):
        raise ValueError('Generation must finish at its first native EOS, without a trailing suffix')


class ShortPlanAdapter(PlannerAdapter):
    def __init__(self, *args, mode='short-plan', require_shared_annotations=False, **kwargs):
        mode_schema(mode)
        self.mode = mode
        self.require_shared_annotations = mode == 'short-plan' or require_shared_annotations
        super().__init__(*args, **kwargs)

    def _prompt(self):
        if self.family == 'omni_captioner':
            # Captioner remains native audio-only; target format is learned via SFT.
            return super()._prompt()
        instruction = INSTRUCTION if self.mode == 'short-plan' else DIRECT_INSTRUCTION
        if self.family == 'moss':
            return self.processor._build_default_prompt(instruction, has_audio=True)
        return self.processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'audio', 'audio': 'native-waveform'}, {'type': 'text', 'text': instruction}]}],
            tokenize=False, add_generation_prompt=True)

    def collate(self, rows):
        if len(rows) != 1:
            raise ValueError('Short-plan native audio packing requires microbatch=1')
        row = rows[0]
        ids, segments, metadata = tokenized_target(self.tokenizer, row, self.eos_token_ids(),
            self.mode, self.require_shared_annotations)
        target, _ = serialize_target(row, self.mode, self.require_shared_annotations)
        batch = super().collate([dict(row, target_text=target)])
        prefix_length = batch['input_ids'].shape[1] - len(ids)
        if batch['labels'][0, prefix_length:].tolist() != ids:
            raise ValueError('Native assistant encoding differs from single-pass segment tokenization')
        batch['segment_ids'] = torch.tensor([[-1] * prefix_length + segments], dtype=torch.long)
        self.last_segment_metadata = metadata
        return batch

    def save_pretrained(self, path, state_dict=None):
        super().save_pretrained(path, state_dict=state_dict)
        with (Path(path) / 'short_plan_adapter.json').open('w', encoding='utf-8') as stream:
            json.dump(dict(schema=mode_schema(self.mode, self.require_shared_annotations), mode=self.mode,
                           require_shared_annotations=self.require_shared_annotations,
                           family=self.family, max_length=self.max_length,
                           instruction_sha256=base.object_sha(INSTRUCTION if self.mode == 'short-plan' else DIRECT_INSTRUCTION),
                           captioner_audio_only=True), stream, ensure_ascii=False, indent=2)

    def generate_caption(self, row, **kwargs):
        # Deliberately select only audio input keys; no gold evidence/plan/caption.
        audio = {key: row[key] for key in ('audio_path', 'audio_sha256', 'audio_window_seconds')}
        # Use the instance method so upstream eval-only cache/EOS wrappers execute.
        generated = self.generate(audio, **kwargs)
        generated['mode'] = self.mode
        generated['generation_compatibility'] = getattr(self, 'generation_compatibility_provenance', None)
        try:
            validate_completion_tokens(generated, kwargs.get('max_new_tokens', 2048))
            extracted = (extract_caption(generated['raw_text']) if self.mode == 'short-plan'
                         else dict(caption=_text(generated['raw_text'], 'caption')))
            timing = evaluation.parse_windows(extracted['caption'])
            if not timing['matches_fixed_windows'] or not timing['nonempty_window_text']:
                raise ValueError('Final caption does not cover the complete fixed time windows')
            return dict(generated, **extracted, temporal=timing, caption_valid=True)
        except ValueError as error:
            return dict(generated, caption=None, caption_valid=False, extraction_error=str(error))


def apply_generation_compatibility(adapter, training_fingerprint=None, official_model_path=None):
    """Keep eval-only patch receipts outside the immutable training fingerprint."""
    requested = dict(eos=eos_patch.generation_eos_patch_ref())
    applied = dict(eos=None, moss_cache=None, moss_position=None)
    position_receipt = None
    if adapter.family == 'moss':
        if official_model_path is None:
            raise ValueError('MOSS generation compatibility requires the official model directory')
        requested['moss_position'] = position_patch.position_buffer_patch_ref()
        position_receipt = position_patch.restore_moss_position_buffer(adapter, official_model_path)
        applied['moss_position'] = position_receipt['applied_position_buffer_patch_ref']
        requested['moss_cache'] = moss_patch.generation_patch_ref()
        applied['moss_cache'] = moss_patch.apply_moss_generation_patch(adapter)
    applied['eos'] = eos_patch.apply_generation_eos_patch(adapter)
    receipt = dict(schema='reviewed-planner-eval-compatibility-v1', requested_sources=requested,
        applied_sources=applied, training_fingerprint=training_fingerprint,
        training_fingerprint_mutated=False, weights_mutated=False, prefix_mutated=False,
        raw_output_cleaning=False, scope='eval-only-native-MOSS-position-cache-and-missing-native-EOS',
        position_buffer_receipt=position_receipt,
        eos_details=getattr(adapter, 'generation_eos_patch_details', None))
    adapter.generation_compatibility_provenance = receipt
    return receipt


def load_short_plan_adapter(family, model_path, checkpoint=None, mode='short-plan',
                            generation_compatibility=True, require_shared_annotations=False, **kwargs):
    required = mode == 'short-plan' or require_shared_annotations
    schema = mode_schema(mode, required)
    training_fingerprint = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not (checkpoint / 'trainer_state.json').is_file():
            raise ValueError('Short-plan inference requires the full experiment checkpoint and parent run_config')
        identity = evaluation.checkpoint_identity(checkpoint)
        config = json.loads((checkpoint.parent / 'run_config.json').read_text(encoding='utf-8'))
        canonical = {key: value for key, value in config.items() if key != 'fingerprint'}
        if (config.get('schema') != schema or config.get('mode') != mode or
                config.get('require_shared_annotations') != required or config.get('family') != family or
                config.get('fingerprint') != identity['fingerprint'] or
                base.object_sha(canonical) != identity['fingerprint'] or
                config.get('data', {}).get('test_used') is not False):
            raise ValueError('Short-plan checkpoint training config/fingerprint differs')
        training_fingerprint = identity['fingerprint']
        if config.get('code_sha256') != implementation_sources():
            raise ValueError('Short-plan checkpoint source version differs; use its original experiment code')
        evaluation.verify_base_identity(model_path, config['model'])
        directory = Path(checkpoint) / 'adapter'
        marker = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
        exported = {}
        anchor = '/' + checkpoint.name + '/adapter/'
        for reference in marker.get('files', []):
            original = '/' + reference['path'].replace('\\', '/')
            if anchor in original:
                relative = original.rsplit(anchor, 1)[1]
                if '..' in relative.split('/') or relative in exported:
                    raise ValueError('Invalid checkpoint export file manifest')
                exported[relative] = reference['sha256']
        actual = {p.relative_to(directory).as_posix(): base.file_sha(p)
                  for p in directory.rglob('*') if p.is_file()}
        if not exported or actual != exported:
            raise ValueError('Portable short-plan adapter export bytes/membership changed')
        metadata = json.loads((directory / 'short_plan_adapter.json').read_text(encoding='utf-8'))
        if (metadata.get('schema') != schema or metadata.get('mode') != mode or
                metadata.get('require_shared_annotations') != required or metadata.get('family') != family or
                metadata.get('instruction_sha256') != base.object_sha(INSTRUCTION if mode == 'short-plan' else DIRECT_INSTRUCTION) or
                metadata.get('max_length') != kwargs.get('max_length', 8192)):
            raise ValueError('Short-plan checkpoint format/family/context differs')
    native = load_planner_adapter(family, model_path, checkpoint=checkpoint, **kwargs)
    adapter = ShortPlanAdapter(family, native.model, native.processor, max_length=native.max_length,
        mode=mode, require_shared_annotations=required)
    adapter.trainable_info = native.trainable_info
    if checkpoint is not None and generation_compatibility:
        apply_generation_compatibility(adapter, training_fingerprint, model_path)
    return adapter


def _forward(model, batch, device, dtype):
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in batch.items() if key not in ('labels', 'segment_ids')}
    context = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
    with context:
        return model(**inputs).logits


def evaluate(model, adapter, rows, cache, device, dtype, weights, loss_chunk_size=0):
    if not rows:
        raise ValueError('Validation requires at least one reviewed source')
    totals, counts = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.long)
    source_objective = 0.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                cache.verify(row)
                batch = adapter.collate([row])
                logits = _forward(model, batch, device, dtype)
                objective, sums, number = segment_nll(logits, batch['labels'].to(device),
                    batch['segment_ids'].to(device), weights, adapter.mode, loss_chunk_size)
                if not bool(torch.isfinite(objective)) or not bool(torch.isfinite(sums).all()):
                    raise FloatingPointError('Nonfinite short-plan validation NLL')
                source_objective += float(objective)
                totals += sums.detach().cpu().double()
                counts += number.cpu()
    finally:
        model.train(was_training)
    means = [float(total / count) if int(count) else None for total, count in zip(totals, counts)]
    return dict(mode=adapter.mode, segment_nll=dict(zip(SEGMENTS, means)),
                segment_tokens=dict(zip(SEGMENTS, counts.tolist())),
                weighted_segment_nll=source_objective / len(rows),
                objective_aggregation='mean-of-per-source-weighted-segment-means',
                caption_nll=float(means[2]), sources=len(rows), test_used=False)


def train(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This independent experiment currently supports one device; do not launch torchrun')
    weights = effective_weights(args.mode, args.segment_weights)
    if (any(not math.isfinite(value) or value <= 0 for value in
            (args.lora_lr, args.projector_lr, args.grad_clip)) or args.warmup_updates < 0):
        raise ValueError('Finite positive learning rates/gradient clip and nonnegative warmup required')
    required = args.mode == 'short-plan' or args.require_shared_annotations
    if args.resume:
        previous_path = args.out / 'run_config.json'
        if args.resume.parent.resolve() != args.out.resolve() or not previous_path.is_file():
            raise ValueError('Resume requires the original checkpoint run output directory')
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        if previous.get('schema') != mode_schema(args.mode, required):
            raise ValueError('Short-plan resume config/data/source fingerprint differs')
    train_rows, val_rows, data = check_data(args.train, args.val, args.mode, required)
    if args.check_data_only:
        return dict(schema=mode_schema(args.mode, required), mode=args.mode, segment_weights=weights,
                    data=data, actual_training_started=False, test_used=False)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    if device.type == 'cpu' and dtype != torch.float32:
        raise ValueError('CPU experiment uses explicit float32; select --dtype float32')
    if args.out.exists() and not args.resume:
        raise ValueError('Output exists; resume explicitly or choose a new output')
    identity = base.local_model_identity(args.model, args.weights_manifest)
    torch.manual_seed(args.seed)
    adapter = load_short_plan_adapter(args.family, args.model, device=device, dtype=dtype,
        max_length=args.max_length, lora_rank=args.rank, lora_alpha=args.alpha, mode=args.mode,
        require_shared_annotations=required)
    model = adapter.model
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    groups, names = base.optimizer_groups(model, args.lora_lr, args.projector_lr, base_dtype=dtype)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step + 1) / max(1, args.warmup_updates)))
    config = dict(schema=mode_schema(args.mode, required), mode=args.mode,
        require_shared_annotations=required, loss_chunk_size=args.loss_chunk_size,
        model=identity, family=args.family, data=data, epochs=args.epochs,
        accum=args.accum, seed=args.seed, rank=args.rank, alpha=args.alpha, dtype=args.dtype,
        lora_lr=args.lora_lr, projector_lr=args.projector_lr, warmup_updates=args.warmup_updates,
        grad_clip=args.grad_clip, max_length=args.max_length, segment_weights=weights,
        objective='weighted-mean-of-segment-mean-causal-NLL-EOS-in-caption',
        gradient_checkpointing=args.gradient_checkpointing, parameter_groups=names,
        parameter_signature=base.parameter_signature(model),
        code_sha256=implementation_sources())
    fingerprint = base.object_sha(config)
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / 'run_config.json'
    if args.resume:
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous.get('fingerprint') != fingerprint:
            raise ValueError('Short-plan resume config/data/source fingerprint differs')
    else:
        base.atomic_json(config_path, dict(config, fingerprint=fingerprint))
    state = dict(epoch=0, position=0, update=0, source_exposures=0, best_val_nll=None,
                 supervised_segment_tokens=[0, 0, 0])
    if args.resume:
        state = base.load_checkpoint(args.resume, model, optimizer, scheduler, fingerprint)
    base.validate_cursor(state, len(train_rows), args.epochs, args.accum)
    starting_update = state['update']
    cache = base.AudioIdentityCache(args.out / 'audio_identity_cache.json')
    checkpoint = args.resume
    model.train()
    while state['epoch'] < args.epochs and (not args.max_updates or state['update'] < args.max_updates):
        order = base.epoch_order(len(train_rows), args.seed, state['epoch'])
        selected = order[state['position']:state['position'] + args.accum]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        update_segment_tokens = torch.zeros(3, dtype=torch.long)
        for index in selected:
            row = train_rows[index]
            cache.verify(row)
            batch = adapter.collate([row])
            logits = _forward(model, batch, device, dtype)
            loss, _, segment_counts = segment_nll(logits, batch['labels'].to(device),
                batch['segment_ids'].to(device), weights, args.mode, args.loss_chunk_size)
            update_segment_tokens += segment_counts.detach().cpu()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite short-plan training loss')
            (loss / len(selected)).backward()
            losses.append(float(loss.detach()))
        for kind, parameters in ((group['name'], group['params']) for group in optimizer.param_groups):
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError('Absent/nonfinite gradients in ' + kind)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        state['update'] += 1
        state['position'] += len(selected)
        state['source_exposures'] += len(selected)
        state['supervised_segment_tokens'] = [old + int(new) for old, new in
            zip(state['supervised_segment_tokens'], update_segment_tokens)]
        epoch_end = state['position'] == len(train_rows)
        validation = None
        improved = False
        if epoch_end:
            state['epoch'] += 1
            state['position'] = 0
            validation = evaluate(model, adapter, val_rows, cache, device, dtype, weights, args.loss_chunk_size)
            score = validation['caption_nll']
            improved = state['best_val_nll'] is None or score < state['best_val_nll']
            state['best_val_nll'] = score if state['best_val_nll'] is None else min(score, state['best_val_nll'])
        final_update = (state['epoch'] == args.epochs or
                        bool(args.max_updates and state['update'] >= args.max_updates))
        if epoch_end or final_update or (args.save_every and state['update'] % args.save_every == 0):
            cache.flush()
            checkpoint = args.out / ('checkpoint-%07d' % state['update'])
            base.save_checkpoint(checkpoint, model, optimizer, scheduler, state, fingerprint, adapter=adapter)
            if validation is not None:
                base.atomic_json(checkpoint / 'validation.json', validation)
                if improved:
                    base.atomic_json(args.out / 'best.json', dict(checkpoint=str(checkpoint.resolve()),
                        selection_metric='val_caption_segment_nll', **validation))
            base.atomic_json(args.out / 'latest.json', dict(checkpoint=str(checkpoint.resolve()), cursor=state))
        print(json.dumps(dict(update=state['update'], loss=sum(losses) / len(losses),
            gradient_norm=float(grad_norm), source_exposures=state['source_exposures'],
            supervised_segment_tokens=dict(zip(SEGMENTS, update_segment_tokens.tolist())),
            supervised_tokens_total=int(update_segment_tokens.sum()),
            cumulative_supervised_segment_tokens=dict(zip(SEGMENTS, state['supervised_segment_tokens'])),
            test_used=False)), flush=True)
    if checkpoint is None:
        raise ValueError('No checkpoint produced')
    if args.verify_reload:
        expected = base.trainable_state(model)
        restored = base.load_checkpoint(checkpoint, model, optimizer, scheduler, fingerprint)
        if restored != state:
            raise ValueError('Reloaded cursor differs')
        base.assert_portable_equal(expected, base.trainable_state(model))
    result = dict(schema=mode_schema(args.mode, required), mode=args.mode,
        require_shared_annotations=required, fingerprint=fingerprint, cursor=state,
        checkpoint=str(Path(checkpoint).resolve()), actual_training_started=state['update'] > starting_update,
        invocation_optimizer_updates=state['update'] - starting_update,
        supervised_segment_tokens=dict(zip(SEGMENTS, state['supervised_segment_tokens'])),
        supervised_tokens_total=sum(state['supervised_segment_tokens']),
        full_budget_completed=state['epoch'] == args.epochs, quality_accepted=False,
        reload_verified=bool(args.verify_reload), test_used=False,
        scope='planner-only single-device experiment; not the frozen fair baseline')
    base.atomic_json(args.out / 'outputs.json', result)
    return result


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--train', type=Path, required=True)
    training.add_argument('--val', type=Path, required=True)
    training.add_argument('--out', type=Path, required=True)
    training.add_argument('--epochs', type=int, default=1)
    training.add_argument('--max-updates', type=int, default=0)
    training.add_argument('--save-every', type=int, default=100)
    training.add_argument('--accum', type=int, default=1)
    training.add_argument('--rank', type=int, default=32)
    training.add_argument('--alpha', type=int, default=64)
    training.add_argument('--lora-lr', type=float, default=1e-5)
    training.add_argument('--projector-lr', type=float, default=5e-5)
    training.add_argument('--warmup-updates', type=int, default=10)
    training.add_argument('--grad-clip', type=float, default=1.)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--segment-weights', type=float, nargs=3, default=None,
                          metavar=('MUSIC', 'PLAN', 'CAPTION'))
    training.add_argument('--resume', type=Path)
    training.add_argument('--weights-manifest', type=Path)
    training.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    training.add_argument('--verify-reload', action='store_true')
    training.add_argument('--check-data-only', action='store_true')
    training.add_argument('--loss-chunk-size', type=int, default=0,
                          help='Positive value uses checkpointed assistant-only FP32 CE chunks; 0 uses one assistant CE')
    generation = commands.add_parser('generate')
    generation.add_argument('--checkpoint', type=Path, required=True)
    generation.add_argument('--input', type=Path, required=True, help='Audio-only JSONL; no gold annotations needed')
    generation.add_argument('--out', type=Path, required=True)
    generation.add_argument('--max-new-tokens', type=int, default=2048)
    generation.add_argument('--generation-compatibility', action=argparse.BooleanOptionalAction, default=True,
                            help='Eval-only native MOSS cache/missing-EOS bridges, with source receipts')
    for command in (training, generation):
        command.add_argument('--mode', choices=MODES, default='short-plan')
        command.add_argument('--require-shared-annotations', action='store_true',
                             help='Direct mode also requires reviewed music evidence and plan on the same manifests')
        command.add_argument('--family', choices=FAMILIES, required=True)
        command.add_argument('--model', type=Path, required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
        command.add_argument('--max-length', type=int, default=8192)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_length < 1:
        raise ValueError('Positive context length required')
    if args.command == 'train':
        if min(args.epochs, args.accum, args.rank, args.alpha) < 1 or min(args.max_updates, args.save_every, args.loss_chunk_size) < 0:
            raise ValueError('Positive epochs/accum/rank/alpha and nonnegative update limit required')
        return train(args)
    if args.out.exists():
        raise ValueError('Generation output exists; choose a new output')
    adapter = load_short_plan_adapter(args.family, args.model, checkpoint=args.checkpoint,
        device=args.device, dtype=getattr(torch, args.dtype), max_length=args.max_length,
        mode=args.mode, generation_compatibility=args.generation_compatibility,
        require_shared_annotations=args.require_shared_annotations)
    records = []
    with args.input.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get('split') not in ('train', 'val'):
                    raise ValueError('Experimental generation accepts train/val only; test is final-only')
                audio_path = Path(row['audio_path']).expanduser()
                if not audio_path.is_absolute():
                    row['audio_path'] = str((args.input.parent / audio_path).resolve())
                records.append(dict(id=row['id'], split=row['split'], test_used=False,
                    **adapter.generate_caption(row, max_new_tokens=args.max_new_tokens)))
    with args.out.open('x', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return records


if __name__ == '__main__':
    result = main()
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
