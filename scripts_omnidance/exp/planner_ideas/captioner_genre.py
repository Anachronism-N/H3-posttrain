"""Captioner joint style/action SFT with assistant-side header conditioning.

The user prefix remains the native audio-only template. Header and action are
encoded separately by an explicit protocol, identically in SFT and inference.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from scripts_omnidance.exp.ms import music_planner_train as training
from scripts_omnidance.exp.ms import music_planner_eval as evaluation
from scripts_omnidance.exp.ms.music_planner_adapters import PlannerAdapter, load_planner_adapter
from scripts_omnidance.exp.ms import music_planner_adapters as native_adapters
from scripts_omnidance.exp.ms import music_planner_generation_eos_patch as eos_patch
from scripts_omnidance.exp.planner_ideas import genre_decision as genre

PROTOCOL = 'captioner-assistant-style-header-v1'
TOKEN_PROTOCOL = 'separately-encode-header-action-then-native-eos-v1'
FAMILY = 'omni_captioner'


def object_sha(value):
    return training.object_sha(value)


def source_identity():
    paths = (__file__, training.__file__, evaluation.__file__, native_adapters.__file__, genre.__file__)
    return {Path(path).name: genre.sha256(path) for path in paths}


def header_text(style):
    return 'Dance style: ' + style + '.\n'


def model_identity(path):
    """Hash actual base weight contents once per invocation, not just a model alias."""
    result = training.local_model_identity(path)
    result['weight_content_sha256'] = {member['name']: genre.sha256(Path(path) / member['name'])
                                       for member in result['weights']}
    result['identity_method'] = 'base-config-processor-and-complete-weight-content-SHA256'
    return result


def verify_model(path, expected):
    evaluation.verify_base_identity(path, expected)
    hashes = expected.get('weight_content_sha256')
    if not isinstance(hashes, dict) or set(hashes) != {r['name'] for r in expected['weights']}:
        raise ValueError('Joint run must bind complete base weight content identities')
    if any(genre.sha256(Path(path) / name) != digest for name, digest in hashes.items()):
        raise ValueError('Joint base weight content identity differs')


def read_labels(rows, path, classes, split):
    labels = {}
    for label in genre.read_jsonl(path):
        if label.get('split') != split or label.get('id') in labels:
            raise ValueError('Pure split and unique ids required in genre labels')
        if (label.get('genre') not in classes or 'genre_distribution' in label or
                any(not isinstance(label.get(key), str) or not label[key].strip()
                    for key in ('label_provenance', 'label_version'))):
            raise ValueError('Joint SFT needs independent hard genre labels with provenance/version')
        labels[label['id']] = label
    if set(labels) != {row['id'] for row in rows}:
        raise ValueError('Genre labels must match every action source exactly; no caption-derived guesses')
    result = []
    for row in rows:
        label = labels[row['id']]
        for key in ('source_group_id', 'audio_sha256'):
            if row[key] != label.get(key):
                raise ValueError('Genre/action source binding differs: ' + key)
        result.append(dict(row, selected_style_label=label['genre']))
    return result


def check_data(args, verify_audio=False):
    classes = genre.validate_classes(json.loads(args.classes.read_text(encoding='utf-8')))
    train, val, data = training.check_data(args.train, args.val)
    rows = {name: read_labels(values, getattr(args, name + '_labels'), classes, name)
            for name, values in (('train', train), ('val', val))}
    if verify_audio:
        for values in rows.values():
            for row in values:
                if genre.sha256(row['audio_path']) != row['audio_sha256']:
                    raise ValueError('Audio file differs from declared identity')
    isolation = genre.check_isolation(rows)
    data = dict(data, genre_label_files={name: genre.sha256(getattr(args, name + '_labels'))
                                       for name in rows}, classes=classes, classes_sha256=object_sha(classes),
                isolation=isolation, audio_bytes_checked=verify_audio,
                song_near_duplicate_status='unknown-unless-independently-audited',
                genre_label_semantic_review_performed_by_program=False)
    return rows['train'], rows['val'], classes, data


def append_ids(batch, ids):
    batch = dict(batch)
    addition = torch.tensor([ids], dtype=batch['input_ids'].dtype, device=batch['input_ids'].device)
    batch['input_ids'] = torch.cat((batch['input_ids'], addition), -1)
    batch['attention_mask'] = torch.cat((batch['attention_mask'], torch.ones_like(addition)), -1)
    for key in ('audio_input_mask', 'token_type_ids', 'mm_token_type_ids'):
        if key in batch:
            batch[key] = torch.cat((batch[key], torch.zeros_like(addition, dtype=batch[key].dtype)), -1)
    return batch


class JointCaptionerAdapter(PlannerAdapter):
    def __init__(self, native, classes):
        if native.family != FAMILY:
            raise ValueError('Assistant style-header protocol is restricted to native omni_captioner')
        super().__init__(FAMILY, native.model, native.processor, native.max_length)
        self.classes = genre.validate_classes(classes)
        self.trainable_info = native.trainable_info
        self._audio_identities = native._audio_identities
        self.header_ids = {name: self.tokenizer.encode(header_text(name), add_special_tokens=False)
                           for name in classes}
        for name, ids in self.header_ids.items():
            if (not ids or any(i in self.eos_token_ids() for i in ids) or
                    self.tokenizer.decode(ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False) != header_text(name)):
                raise ValueError('Style header must round-trip exactly without EOS')

    def collate(self, rows):
        if len(rows) != 1:
            raise ValueError('Joint Captioner supports microbatch=1')
        row = rows[0]
        style = row.get('selected_style_label')
        if style not in self.header_ids:
            raise ValueError('Missing independently labelled style for joint SFT')
        action = row['target_text']
        timing = evaluation.parse_windows(action)
        if (not timing['matches_fixed_windows'] or not timing['nonempty_window_text'] or
                '<|' in action or '<audio>' in action or action.startswith('Dance style:')):
            raise ValueError('Joint target must be plain complete action caption, without a style header')
        batch = self.prefix(genre.audio_only_row(row))
        prefix_length = batch['input_ids'].shape[1]
        header = self.header_ids[style]
        # Explicit segmentation is the protocol; never replace by encode(header+action).
        action_ids = self._assistant_ids(action)[0].tolist()
        target = header + action_ids
        if prefix_length + len(target) > self.max_length:
            raise ValueError('Joint target exceeds context; truncation forbidden')
        batch = append_ids(batch, target)
        batch['labels'] = torch.tensor([[-100] * prefix_length + target], dtype=torch.long)
        batch['style_action_segments'] = torch.tensor([[-1] * prefix_length +
            [0] * len(header) + [1] * len(action_ids)], dtype=torch.long)
        return batch

    def save_pretrained(self, path, state_dict=None):
        super().save_pretrained(path, state_dict)
        genre.write_json(Path(path) / 'captioner_genre_adapter.json', dict(
            protocol=PROTOCOL, token_protocol=TOKEN_PROTOCOL, family=FAMILY,
            classes=self.classes, classes_sha256=object_sha(self.classes),
            header_token_ids=self.header_ids, user_prefix='native-audio-only', max_length=self.max_length))

    @torch.no_grad()
    def generate(self, row, selected_style=None, max_new_tokens=1024):
        if not getattr(self, 'trained_joint_checkpoint', None):
            raise ValueError('Generation requires a validated checkpoint with actual joint style/action SFT')
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError('Positive generation budget required')
        if selected_style is not None and selected_style not in self.header_ids:
            raise ValueError('Requested style outside trained dictionary')
        batch = self.prefix(genre.audio_only_row(row))
        native_length = batch['input_ids'].shape[1]
        header = [] if selected_style is None else self.header_ids[selected_style]
        batch = append_ids(batch, header)
        if batch['input_ids'].shape[1] + max_new_tokens > self.max_length:
            raise ValueError('Joint generation exceeds context; truncation forbidden')
        prefix_ids = batch['input_ids']
        self.model.eval()
        parameter = next(self.model.parameters())
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            output = self.model.generate(**self._move(batch), do_sample=False, num_beams=1,
                repetition_penalty=1., no_repeat_ngram_size=0, encoder_repetition_penalty=1.,
                max_new_tokens=max_new_tokens, use_cache=True)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        if isinstance(sequences, tuple):
            sequences = sequences[0]
        if (sequences.ndim != 2 or sequences.shape[0] != 1 or
                not torch.equal(sequences[:, :prefix_ids.shape[1]].cpu(), prefix_ids.cpu())):
            raise ValueError('Joint generation changed its exact native+assistant prefix')
        ids = sequences[0, prefix_ids.shape[1]:].cpu().tolist()
        eos = self.eos_token_ids()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        full_ids = header + ids
        result = dict(protocol=PROTOCOL, token_protocol=TOKEN_PROTOCOL,
            raw_continuation_text=raw, raw_full_assistant_text=self.tokenizer.decode(full_ids,
                skip_special_tokens=True, clean_up_tokenization_spaces=False),
            raw_with_special_tokens=self.tokenizer.decode(full_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False), new_token_ids=ids, eos_token_ids=eos,
            assistant_header_prefill_ids=header, native_input_tokens=native_length,
            forced_header_token_ids=header, raw_new_token_ids=ids, raw_tail_text=raw,
            full_response_text=self.tokenizer.decode(full_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), style_header_generated_by_joint_model=not bool(header),
            input_tokens=prefix_ids.shape[1], max_new_tokens=max_new_tokens,
            stop_reason=evaluation.classify_stop(ids, eos, max_new_tokens),
            decoding=dict(do_sample=False, num_beams=1, repetition_penalty=1., no_repeat_ngram_size=0),
            route='forced-assistant-header' if header else 'joint-self-style', **self._last_audio_info)
        try:
            if not ids or ids[-1] not in eos or any(token in eos for token in ids[:-1]):
                raise ValueError('Generation must finish at its first native EOS')
            matching = [name for name, tokens in self.header_ids.items() if full_ids[:len(tokens)] == tokens]
            if len(matching) != 1:
                raise ValueError('Generated style header does not follow the trained token protocol/dictionary')
            style = matching[0]
            action_ids = full_ids[len(self.header_ids[style]):]
            action = self.tokenizer.decode(action_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            timing = evaluation.parse_windows(action)
            if (not timing['matches_fixed_windows'] or not timing['nonempty_window_text'] or
                    '<|' in action or '<audio>' in action):
                raise ValueError('Generated action caption lacks complete nonempty fixed windows')
            result.update(caption_valid=True, extraction_error=None, generated_style=style,
                          action_caption=action, caption_text=header_text(style) + action, temporal=timing)
        except ValueError as error:
            result.update(caption_valid=False, extraction_error=str(error), generated_style=None,
                          action_caption=None, caption_text=None)
        return result


def style_action_loss(logits, labels, segments, weights=(.25, 1.)):
    if (len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights) or
            logits.shape[:2] != labels.shape or labels.shape != segments.shape):
        raise ValueError('Positive style/action weights and aligned causal labels required')
    target, sections = labels[:, 1:], segments[:, 1:]
    if not torch.equal(target.ne(-100), sections.ge(0)) or (sections > 1).any():
        raise ValueError('Style/action ownership differs from supervised labels')
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         target.reshape(-1), ignore_index=-100, reduction='none').reshape_as(target)
    sums, counts = [], []
    for index in range(2):
        valid = sections.eq(index)
        sums.append((ce * valid).sum())
        counts.append(valid.sum())
    if any(int(number) <= 0 for number in counts):
        raise ValueError('Both style and action must receive supervised tokens')
    loss = sum(w * total / n for w, total, n in zip(weights, sums, counts)) / sum(weights)
    return loss, torch.stack(sums), torch.stack(counts)


def _portable_artifacts(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if (not (checkpoint / 'trainer_state.json').is_file() or
            not (checkpoint.parent / 'run_config.json').is_file()):
        raise ValueError('Require a complete planner checkpoint and parent run_config, not a bare adapter')
    identity = evaluation.checkpoint_identity(checkpoint)
    marker = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    expected_adapter = {}
    anchor = '/' + checkpoint.name + '/'
    for reference in marker.get('files', []):
        original = '/' + reference['path'].replace('\\', '/')
        if anchor not in original:
            raise ValueError('Checkpoint file identity is outside its original checkpoint directory')
        relative = original.rsplit(anchor, 1)[1]
        if '..' in relative.split('/'):
            raise ValueError('Checkpoint member escapes its directory')
        # Initializers can be 8-rank ZeRO3 runs: consume portable weights, not optimizer shards.
        if relative != 'trainable.pt' and not relative.startswith('adapter/'):
            continue
        if genre.sha256(checkpoint / relative) != reference['sha256']:
            raise ValueError('Checkpoint member contents changed')
        if relative.startswith('adapter/'):
            expected_adapter[relative[8:]] = reference['sha256']
    actual = {p.relative_to(checkpoint / 'adapter').as_posix(): genre.sha256(p)
              for p in (checkpoint / 'adapter').rglob('*') if p.is_file()}
    if not expected_adapter or expected_adapter != actual:
        raise ValueError('Checkpoint adapter members/content changed')
    return identity, marker, actual


def validate_joint_checkpoint(path, model, classes=None):
    path = Path(path).resolve()
    identity, marker, members = _portable_artifacts(path)
    if marker.get('world_size') != 1 or marker.get('zero3') is not False:
        raise ValueError('Joint protocol currently requires single-device portable checkpoints')
    config = json.loads((path.parent / 'run_config.json').read_text(encoding='utf-8'))
    canonical = {k: v for k, v in config.items() if k != 'fingerprint'}
    if (config.get('protocol') != PROTOCOL or config.get('token_protocol') != TOKEN_PROTOCOL or
            config.get('family') != FAMILY or object_sha(canonical) != identity['fingerprint'] or
            config.get('fingerprint') != identity['fingerprint'] or
            config.get('sources') != source_identity() or config.get('data', {}).get('test_used') is not False):
        raise ValueError('Not a matching trained joint Captioner protocol/source/config checkpoint')
    trained_classes = genre.validate_classes(config['classes'])
    if classes is not None and classes != trained_classes:
        raise ValueError('Joint trained style dictionary differs')
    counts = marker['cursor'].get('conditioning_supervised_tokens', [])
    if marker['cursor'].get('update', 0) < 1 or len(counts) != 2 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError('No actual style+action SFT updates in this checkpoint')
    verify_model(model, config['model'])
    metadata = json.loads((path / 'adapter/captioner_genre_adapter.json').read_text(encoding='utf-8'))
    if (metadata.get('protocol') != PROTOCOL or metadata.get('token_protocol') != TOKEN_PROTOCOL or
            metadata.get('classes') != trained_classes or metadata.get('classes_sha256') != object_sha(trained_classes) or
            metadata.get('family') != FAMILY or metadata.get('max_length') != config['max_length']):
        raise ValueError('Joint adapter protocol/style dictionary differs')
    descriptor = dict(protocol=PROTOCOL, fingerprint=identity['fingerprint'],
        trainer_state_sha256=genre.sha256(path / 'trainer_state.json'),
        trainable_sha256=marker['trainable_sha256'], adapter_members=members,
        base_weight_content_sha256=config['model']['weight_content_sha256'], classes_sha256=object_sha(trained_classes))
    return config, metadata, descriptor


def load_joint(model_path, checkpoint, *, device='cpu', dtype=torch.float32, generation=False, expected_head=None):
    config, metadata, descriptor = validate_joint_checkpoint(checkpoint, model_path)
    if expected_head is not None and (expected_head['extractor'] != extractor_identity(descriptor) or
                                     expected_head['classes'] != config['classes']):
        raise ValueError('Genre head must use native input-only features from this exact trained joint checkpoint/dictionary')
    native = load_planner_adapter(FAMILY, model_path, checkpoint=checkpoint, device=device,
                                  dtype=dtype, max_length=config['max_length'])
    adapter = JointCaptionerAdapter(native, config['classes'])
    if metadata['header_token_ids'] != adapter.header_ids:
        raise ValueError('Restored tokenizer changes the exact trained header token sequence')
    adapter.trained_joint_checkpoint = descriptor
    receipt = None
    if generation:
        applied = eos_patch.apply_generation_eos_patch(adapter)
        receipt = dict(requested_eos_patch_ref=eos_patch.generation_eos_patch_ref(),
                       applied_eos_patch_ref=applied, details=getattr(adapter, 'generation_eos_patch_details', None),
                       weights_mutated=False, user_prefix_mutated=False)
    return adapter, config, descriptor, receipt


def validate_initialization(path, model_path, rank, alpha):
    identity, marker, members = _portable_artifacts(path)
    config = json.loads((Path(path).resolve().parent / 'run_config.json').read_text(encoding='utf-8'))
    canonical = {k: v for k, v in config.items() if k not in ('fingerprint', 'trainable_parameters')}
    baseline = (config.get('protocol') is None and config.get('mode') is None and
                isinstance(config.get('adapter_source_sha256'), str) and
                isinstance(config.get('trainer_sha256'), str))
    direct_schemas = {
        'music-planner-reviewed-ablation-v2/direct-caption': None,
        'music-planner-captioner-ablation-v3/direct-caption/weak-caption': False,
        'music-planner-captioner-ablation-v3/direct-caption/reviewed-shared': True,
    }
    direct = (config.get('schema') in direct_schemas and config.get('mode') == 'direct-caption' and
              config.get('protocol') is None)
    if direct:
        expected_review = direct_schemas[config['schema']]
        direct_metadata = json.loads((Path(path) / 'adapter/short_plan_adapter.json').read_text(encoding='utf-8'))
        if (direct_metadata.get('schema') != config['schema'] or direct_metadata.get('mode') != 'direct-caption' or
                direct_metadata.get('family') != FAMILY or
                (expected_review is not None and (config.get('require_shared_annotations') is not expected_review or
                 direct_metadata.get('require_shared_annotations') is not expected_review))):
            raise ValueError('Direct-caption initializer protocol metadata differs')
    if (config.get('family') != FAMILY or marker['cursor'].get('update', 0) < 1 or
            not (baseline or direct) or config.get('data', {}).get('test_used') is not False or
            config.get('fingerprint') != identity['fingerprint'] or object_sha(canonical) != identity['fingerprint']):
        raise ValueError('Initialization must be a completed non-test Captioner fair/direct-action checkpoint')
    evaluation.verify_base_identity(model_path, config['model'])
    meta = json.loads((Path(path) / 'adapter/planner_adapter.json').read_text(encoding='utf-8'))
    if meta.get('family') != FAMILY:
        raise ValueError('Initialization adapter is not Captioner')
    info = meta.get('trainable_info', {})
    if info.get('rank') != rank or info.get('alpha') != alpha:
        raise ValueError('Initialization LoRA rank/alpha must match the explicit run arguments')
    declared = {row['name']: dict(shape=row['shape'], dtype=row['dtype'])
                for row in info.get('trainable', [])}
    if not declared or declared != marker['trainable_signature']:
        raise ValueError('Initialization declared trainable metadata differs from checkpoint signature')
    return dict(fingerprint=identity['fingerprint'], trainer_state_sha256=genre.sha256(Path(path) / 'trainer_state.json'),
                trainable_sha256=marker['trainable_sha256'], adapter_members=members), marker


def _forward(adapter, batch, dtype):
    inputs = adapter._move({k: v for k, v in batch.items() if k != 'style_action_segments'})
    device = next(adapter.model.parameters()).device
    context = torch.autocast(device.type, dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()
    with context:
        return adapter.model(**inputs).logits


def evaluate(adapter, rows, cache, weights, dtype):
    was_training = adapter.model.training
    adapter.model.eval()
    sums, counts, objectives = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.long), []
    try:
        with torch.no_grad():
            for row in rows:
                cache.verify(row)
                batch = adapter.collate([row])
                logits = _forward(adapter, batch, dtype)
                loss, totals, number = style_action_loss(logits, batch['labels'].to(logits.device),
                    batch['style_action_segments'].to(logits.device), weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite joint validation loss')
                sums += totals.cpu().double()
                counts += number.cpu()
                objectives.append(float(loss))
    finally:
        adapter.model.train(was_training)
    return dict(style_nll=float(sums[0] / counts[0]), action_nll=float(sums[1] / counts[1]),
                source_mean_objective=sum(objectives) / len(objectives), supervised_tokens=counts.tolist(), test_used=False)


def train(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Joint Captioner training supports one device, not torchrun/DDP')
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError('--init-checkpoint initializes weights; --resume restores a run; choose one')
    if (min(args.epochs, args.accum, args.rank, args.alpha) < 1 or min(args.max_updates, args.save_every) < 0 or
            any(not math.isfinite(value) or value <= 0 for value in
                (args.lr, args.projector_lr, args.style_weight, args.action_weight, args.grad_clip))):
        raise ValueError('Positive training configuration required')
    if args.out.exists() and args.resume is None:
        raise ValueError('Output exists; use a new run or explicit resume')
    rows, val, classes, data = check_data(args)
    identity = model_identity(args.model)
    initialization, init_marker = None, None
    if args.init_checkpoint:
        initialization, init_marker = validate_initialization(args.init_checkpoint, args.model, args.rank, args.alpha)
    if args.resume:
        original = json.loads((args.out / 'run_config.json').read_text(encoding='utf-8'))
        initialization = original.get('initialization')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dtype = getattr(torch, args.dtype)
    if torch.device(args.device).type == 'cpu' and dtype != torch.float32:
        raise ValueError('CPU joint training requires --dtype float32')
    if args.init_checkpoint:
        native = load_planner_adapter(FAMILY, args.model, checkpoint=args.init_checkpoint,
            device=args.device, dtype=dtype, max_length=args.max_length)
        signature = init_marker['trainable_signature']
        if not set(signature) <= {name for name, _ in native.model.named_parameters()}:
            raise ValueError('Initialization trainable names differ from native model')
        for name, parameter in native.model.named_parameters():
            parameter.requires_grad_(name in signature)
            if name in signature:
                native_adapters._cast_parameter_dtype(parameter, torch.float32)
        if training.parameter_signature(native.model) != signature:
            raise ValueError('Initialization trainable signature differs')
    else:
        native = load_planner_adapter(FAMILY, args.model, device=args.device, dtype=dtype,
            max_length=args.max_length, lora_rank=args.rank, lora_alpha=args.alpha)
    adapter = JointCaptionerAdapter(native, classes)
    adapter.model.config.use_cache = False
    if args.gradient_checkpointing:
        adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    groups, names = training.optimizer_groups(adapter.model, args.lr, args.projector_lr, base_dtype=dtype)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    config = dict(protocol=PROTOCOL, token_protocol=TOKEN_PROTOCOL, family=FAMILY, classes=classes,
        model=identity, initialization=initialization, data=data, max_length=args.max_length,
        epochs=args.epochs, accum=args.accum, rank=args.rank, alpha=args.alpha,
        lr=args.lr, projector_lr=args.projector_lr, style_weight=args.style_weight,
        action_weight=args.action_weight, seed=args.seed, dtype=args.dtype, grad_clip=args.grad_clip,
        gradient_checkpointing=args.gradient_checkpointing, parameter_groups=names,
        sources=source_identity())
    fingerprint = object_sha(config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if original.get('fingerprint') != fingerprint:
            raise ValueError('Joint resume data/model/classes/source/config identity differs')
    else:
        training.atomic_json(args.out / 'run_config.json', dict(config, fingerprint=fingerprint))
    state = dict(epoch=0, position=0, update=0, source_exposures=0, best_val_nll=None,
                 conditioning_supervised_tokens=[0, 0])
    if args.resume:
        state = training.load_checkpoint(args.resume, adapter.model, optimizer, scheduler, fingerprint)
    training.validate_cursor(state, len(rows), args.epochs, args.accum)
    starting_update = state['update']
    cache = training.AudioIdentityCache(args.out / 'audio_identity_cache.json')
    checkpoint = args.resume
    weights = (args.style_weight, args.action_weight)
    adapter.model.train()
    while state['epoch'] < args.epochs and (not args.max_updates or state['update'] < args.max_updates):
        order = training.epoch_order(len(rows), args.seed, state['epoch'])
        selected = order[state['position']:state['position'] + args.accum]
        optimizer.zero_grad(set_to_none=True)
        number = torch.zeros(2, dtype=torch.long)
        for index in selected:
            cache.verify(rows[index])
            batch = adapter.collate([rows[index]])
            logits = _forward(adapter, batch, dtype)
            loss, _, tokens = style_action_loss(logits, batch['labels'].to(logits.device),
                batch['style_action_segments'].to(logits.device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint SFT objective')
            (loss / len(selected)).backward()
            number += tokens.cpu()
        for group in optimizer.param_groups:
            grads = [p.grad for p in group['params'] if p.grad is not None]
            if not grads or not all(torch.isfinite(value).all() for value in grads):
                raise ValueError('Missing/nonfinite joint conditioning gradients')
        torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        state['position'] += len(selected)
        state['source_exposures'] += len(selected)
        state['update'] += 1
        state['conditioning_supervised_tokens'] = [old + int(new) for old, new in
            zip(state['conditioning_supervised_tokens'], number)]
        end = state['position'] == len(rows)
        if end:
            state['epoch'] += 1
            state['position'] = 0
        final = state['epoch'] == args.epochs or bool(args.max_updates and state['update'] >= args.max_updates)
        if end or final or (args.save_every and state['update'] % args.save_every == 0):
            report = evaluate(adapter, val, cache, weights, dtype)
            state['best_val_nll'] = report['action_nll'] if state['best_val_nll'] is None else min(
                state['best_val_nll'], report['action_nll'])
            cache.flush()
            checkpoint = args.out / ('checkpoint-%07d' % state['update'])
            training.save_checkpoint(checkpoint, adapter.model, optimizer, scheduler, state, fingerprint, adapter=adapter)
            training.atomic_json(checkpoint / 'validation.json', report)
        print(json.dumps(dict(update=state['update'], supervised_style_action_tokens=number.tolist(), test_used=False)), flush=True)
    if checkpoint is None:
        raise ValueError('No trained joint checkpoint produced')
    result = dict(protocol=PROTOCOL, checkpoint=str(Path(checkpoint).resolve()), fingerprint=fingerprint,
        cursor=state, invocation_optimizer_updates=state['update'] - starting_update,
        full_budget_completed=state['epoch'] == args.epochs, quality_accepted=False, test_used=False)
    training.atomic_json(args.out / 'outputs.json', result)
    return result


def extractor_identity(descriptor):
    return dict(schema=genre.CACHE_SCHEMA, family=FAMILY,
                model_identity='joint-captioner-sha256:' + object_sha(descriptor),
                joint_checkpoint=descriptor, feature_export_protocol=PROTOCOL,
                prefix_policy='native-input-only-no-answer-or-genre',
                code_sha256=genre.sha256(__file__), adapter_code_sha256=genre.adapter_code_sha())


def read_sources(path):
    rows = genre.read_jsonl(path)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Unique nonempty source manifest required')
    result = []
    for row in rows:
        source = genre.audio_only_row(row)
        if row.get('split') not in ('train', 'val', 'cal', 'predict', 'test'):
            raise ValueError('Source needs explicit train/val/cal/predict/test split')
        if not Path(source['audio_path']).is_absolute():
            source['audio_path'] = str((Path(path).resolve().parent / source['audio_path']).resolve())
        result.append(dict(source, split=row['split']))
    return result


def check_joint_holdout(rows, config):
    for row in rows:
        if row['split'] not in ('val', 'cal', 'test'):
            continue
        excluded = ('train',) if row['split'] == 'val' else ('train', 'val')
        for name in excluded:
            previous = config['data']['isolation'][name]
            for key in ('id', 'source_group_id', 'audio_sha256'):
                if row[key] in previous[key]:
                    raise ValueError(row['split'] + ' source overlaps joint SFT ' + name + ': ' + key)


def export_features(args):
    genre.require_fresh_output(args.out_dir)
    rows = read_sources(args.input)
    adapter, config, descriptor, _ = load_joint(args.model, args.checkpoint, device=args.device,
                                          dtype=getattr(torch, args.dtype))
    check_joint_holdout(rows, config)
    identity = extractor_identity(descriptor)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        features, info = genre.export_prefill_features(adapter, row)
        name = hashlib.sha256(row['id'].encode()).hexdigest() + '.npz'
        path = args.out_dir / name
        np.savez_compressed(path, features=features.numpy(), mask=np.ones(len(features), dtype=np.bool_))
        records.append(dict(id=row['id'], source_group_id=row['source_group_id'], split=row['split'],
            audio_sha256=row['audio_sha256'], feature_path=name, feature_sha256=genre.sha256(path),
            extractor=identity, extraction=info))
    genre.write_jsonl_fresh(args.out_dir / 'features.jsonl', records)


def generate(args):
    genre.require_fresh_output(args.out)
    rows = read_sources(args.input)
    head, saved = genre.load_head(args.head)
    genre.validate_inference_rows(rows, saved, args.final_test)
    adapter, config, descriptor, patch_receipt = load_joint(args.model, args.checkpoint,
        device=args.device, dtype=getattr(torch, args.dtype), generation=True, expected_head=saved)
    check_joint_holdout(rows, config)
    records = []
    head_sha, input_sha = genre.sha256(args.head), genre.sha256(args.input)
    for row in rows:
        features, info = genre.export_prefill_features(adapter, row)
        decision = genre.decide(head, features[None], torch.ones(1, len(features), dtype=torch.bool),
                                saved['classes'], saved['temperature'], **genre.routing_options(args))[0]
        output = adapter.generate(row, selected_style=decision['selected_genre'], max_new_tokens=args.max_new_tokens)
        route = dict(decision['routing'], route=output['route'],
                     native_unconditioned_fallback=False, semantics='same-joint-model-style-generation')
        records.append(dict(id=row['id'], split=row['split'], test_used=args.final_test,
            joint_checkpoint=descriptor, head_sha256=head_sha, input_sha256=input_sha,
            generation_patches=patch_receipt, extraction=info, **dict(decision, routing=route), generation=output,
            caption_text=output['caption_text'], caption_valid=output['caption_valid'],
            style_agreement=dict(head_predicted=decision['predicted_genre'],
                joint_output=output.get('generated_style'),
                forced_by_head=decision['selected_genre'] is not None,
                independent_style_prediction=decision['selected_genre'] is None,
                match=output.get('generated_style') == decision['predicted_genre'] if output['caption_valid'] else None,
                semantic_quality_approved=False)))
    genre.write_jsonl_fresh(args.out, records)
    return records


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('check', 'train'):
        sub = commands.add_parser(name)
        for option in ('train', 'val', 'train-labels', 'val-labels', 'classes'):
            sub.add_argument('--' + option, type=Path, required=True)
        if name == 'check':
            continue
        sub.add_argument('--model', type=Path, required=True)
        sub.add_argument('--out', type=Path, required=True)
        sub.add_argument('--init-checkpoint', type=Path)
        sub.add_argument('--resume', type=Path)
        sub.add_argument('--epochs', type=int, default=1)
        sub.add_argument('--accum', type=int, default=1)
        sub.add_argument('--rank', type=int, default=32)
        sub.add_argument('--alpha', type=int, default=64)
        sub.add_argument('--lr', type=float, default=1e-5)
        sub.add_argument('--projector-lr', type=float, default=5e-5)
        sub.add_argument('--style-weight', type=float, default=.25)
        sub.add_argument('--action-weight', type=float, default=1.)
        sub.add_argument('--grad-clip', type=float, default=1.)
        sub.add_argument('--max-updates', type=int, default=0)
        sub.add_argument('--save-every', type=int, default=100)
        sub.add_argument('--max-length', type=int, default=8192)
        sub.add_argument('--seed', type=int, default=42)
        sub.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    for name in ('export-features', 'generate'):
        sub = commands.add_parser(name)
        sub.add_argument('--model', type=Path, required=True)
        sub.add_argument('--checkpoint', type=Path, required=True)
        sub.add_argument('--input', type=Path, required=True)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
        if name == 'export-features':
            sub.add_argument('--out-dir', type=Path, required=True)
        else:
            sub.add_argument('--head', type=Path, required=True)
            sub.add_argument('--out', type=Path, required=True)
            sub.add_argument('--max-new-tokens', type=int, default=1024)
            sub.add_argument('--final-test', action='store_true')
            genre.add_routing_arguments(sub)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'check':
        _, _, _, data = check_data(args, verify_audio=True)
        return dict(protocol=PROTOCOL, data=data, actual_training_started=False)
    if args.command == 'train':
        return train(args)
    if args.command == 'export-features':
        return export_features(args)
    return generate(args)


if __name__ == '__main__':
    result = main()
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
