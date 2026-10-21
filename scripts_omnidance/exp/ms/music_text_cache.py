"""Cache original contextual text-token targets without video/audio VAE computation."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import torch

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')


def contextual_dependencies():
    # The pure-text path must never import the image/presentation pipeline.
    global Image, MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PromptEmbedder
    global MiniMaxH3Unit_ReferenceEncoder, prefix_length, tensor_sha256, write_new, neutralize_gender_pronouns
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PromptEmbedder,
        MiniMaxH3Unit_ReferenceEncoder,
    )
    from music_text_condition import prefix_length, tensor_sha256, write_new, neutralize_gender_pronouns


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_prompt(entry, neutral_teacher=False):
    text = entry['motion_original']
    genre = entry['dims']['genre']
    if not text.strip() or not genre.strip() or genre == 'generation_holdout':
        raise ValueError('Target requires original motion and a declared dance genre')
    prompt = f'Dance style: {genre}. ' + text
    return neutralize_gender_pronouns(prompt)[0] if neutral_teacher else prompt


def pure_neutralize_gender_pronouns(text):
    # Keep the existing transform exactly, without importing its multimodal module.
    possessive_nouns = {'hand', 'hands', 'arm', 'arms', 'elbow', 'elbows', 'shoulder', 'shoulders',
                        'foot', 'feet', 'leg', 'legs', 'knee', 'knees', 'ankle', 'ankles',
                        'hip', 'hips', 'torso', 'body', 'head', 'neck', 'face', 'waist', 'chest',
                        'left', 'right', 'both', 'back', 'pose', 'position', 'stance', 'weight'}
    changes = []

    def replace(match):
        original = match.group()
        word = original.lower()
        following = re.match(r'\s+([A-Za-z]+)', text[match.end():])
        possessive = word == 'his' or (word == 'her' and following is not None
                                       and following[1].lower() in possessive_nouns)
        replacement = "the dancer's" if possessive else 'the dancer'
        if original[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        changes.append(dict(start=match.start(), end=match.end(), original=original, replacement=replacement))
        return replacement

    normalized = re.sub(r'\b(?:he|she|his|her|him|himself|herself)\b', replace, text, flags=re.IGNORECASE)
    return normalized, changes


def atomic_write_new(path, value, binary=False):
    """Publish only a fully fsynced record, atomically and without replacing any output."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb' if binary else 'w', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            if binary:
                torch.save(value, stream)
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pure_text_inputs(caps, split, neutral_teacher=False):
    if not isinstance(caps, dict) or not isinstance(split, dict):
        raise ValueError('Caps and split must be JSON objects')
    train, val = split.get('train'), split.get('val')
    if not isinstance(train, list) or not isinstance(val, list) or not train:
        raise ValueError('Require explicit nonempty train list and val list')
    ids = train + val
    if any(not isinstance(sid, str) or not sid or Path(sid).name != sid or sid in ('.', '..') for sid in ids):
        raise ValueError('Invalid sample ID')
    if len(ids) != len(set(ids)):
        raise ValueError('Split duplicates or train/validation overlap')
    prompts, replacements = {}, {}
    for sid in ids:
        entry = caps.get(sid)
        if (not isinstance(entry, dict) or not isinstance(entry.get('motion_original'), str)
                or not isinstance(entry.get('dims'), dict) or not isinstance(entry['dims'].get('genre'), str)):
            raise ValueError('Require dims.genre and original MotionScript text: ' + sid)
        original = target_prompt(entry)
        prompt, changes = pure_neutralize_gender_pronouns(original) if neutral_teacher else (original, [])
        prompts[sid], replacements[sid] = prompt, changes
    return train, val, prompts, replacements


def pure_token_length_report(tokenizer, prompts, train, val):
    from text_embed import raw_tensor_sha256
    ids = train + val
    encoded = tokenizer([prompts[sid] for sid in ids], add_special_tokens=False,
                        truncation=False, padding=False)['input_ids']
    if len(encoded) != len(ids) or any(not row for row in encoded):
        raise ValueError('Invalid full-split tokenization')
    token_ids = {sid: torch.tensor(row, dtype=torch.long) for sid, row in zip(ids, encoded)}
    lengths = {sid: len(row) for sid, row in token_ids.items()}
    train_max = max(lengths[sid] for sid in train)
    val_max = max((lengths[sid] for sid in val), default=None)
    report = dict(lengths=lengths, token_ids_sha256={sid: raw_tensor_sha256(row) for sid, row in token_ids.items()},
                  qmax=train_max, train_max=train_max, val_max=val_max,
                  training_query_max=train_max, training_mean_length=sum(lengths[sid] for sid in train) / len(train),
                  validation_max_length=val_max, validation_does_not_set_query_max=True,
                  full_token_check=True, token_check_train_count=len(train), token_check_val_count=len(val),
                  tokenizer_padding_side='right', add_special_tokens=False, truncation=False)
    return report, token_ids


def pure_text_main():
    from text_embed import load_pure_text_tokenizer, load_pure_text_lm, pure_text_token_batch, raw_tensor_sha256
    parser = argparse.ArgumentParser(description='Pure text-only H3 raw token teacher; no image or presentation prefix')
    parser.add_argument('--pure_text', action='store_true', required=True)
    parser.add_argument('--caps', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--w', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--neutral_teacher', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='CPU real-token full-split check; --out is a new JSON file')
    args = parser.parse_args()
    if args.nshard <= 0 or not 0 <= args.shard < args.nshard or args.batch_size <= 0:
        parser.error('Invalid shard or batch size')
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new output path and existing parent; no resume/overwrite')
    caps_sha, split_sha = sha256(args.caps), sha256(args.split)
    caps, split = json.loads(args.caps.read_text()), json.loads(args.split.read_text())
    train, val, prompts, replacements = pure_text_inputs(caps, split, args.neutral_teacher)
    tokenizer = load_pure_text_tokenizer(args.w)
    length_report, source_ids = pure_token_length_report(tokenizer, prompts, train, val)
    processor = args.w / 'processor'
    tokenizer_paths = [processor / name for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                                                    'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.json')]
    tokenizer_shas = {path.name: sha256(path) for path in tokenizer_paths if path.is_file()}
    if 'tokenizer.json' not in tokenizer_shas or 'tokenizer_config.json' not in tokenizer_shas:
        raise ValueError('Missing native tokenizer metadata')
    ids = (train + val)[args.shard::args.nshard]
    source = dict(kind='pure_text_h3_raw_tokens', text_only=True, image_conditioned=False,
                  prefix_length=0, interface='raw_h3_last_hidden_5120_pure_text', final_norm='Identity',
                  caption_fields=['dims.genre', 'motion_original'], special_template=False,
                  caps_sha256=caps_sha, split_sha256=split_sha,
                  tokenizer_metadata_file_sha256=tokenizer_shas, tokenizer_root=str(processor),
                  original_full_train=len(train), original_full_val=len(val),
                  train_ids=train, val_ids=val, ids=ids, expected_ids=ids,
                  shard=args.shard, nshard=args.nshard, batch_size=args.batch_size,
                  teacher_transform='gender_pronouns_to_dancer_v1' if args.neutral_teacher else 'none',
                  normalization='none', pooling='none', dtype='torch.bfloat16', hidden_width=5120,
                  **length_report)
    if sha256(args.caps) != caps_sha or sha256(args.split) != split_sha:
        raise RuntimeError('Caps/split changed during token preflight')
    if args.preflight:
        report = dict(source, kind='CPU_pure_text_preflight_not_teacher_encoding', complete=True,
                      actual_lm_forward=False, GPU_teacher_cache_pending=True, image_io=False)
        atomic_write_new(args.out, report)
        print(json.dumps({key: value for key, value in report.items() if key not in ('lengths', 'token_ids_sha256', 'ids', 'expected_ids', 'train_ids', 'val_ids')}), flush=True)
        return
    if not ids:
        parser.error('Empty shard; choose fewer shards')
    args.out.mkdir()
    lm, model_provenance = load_pure_text_lm(args.w)
    source.update(model_provenance)
    if any(sha256(processor / name) != digest for name, digest in tokenizer_shas.items()):
        raise RuntimeError('Tokenizer metadata changed during LM load')
    atomic_write_new(args.out / 'plan.json', source)
    protocol_sha = sha256(args.out / 'plan.json')
    rows, started = [], time.monotonic()
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start:start + args.batch_size]
        encoded = pure_text_token_batch(lm, tokenizer, [prompts[sid] for sid in batch])
        if len(encoded) != len(batch):
            raise ValueError('LM batch result count mismatch')
        for sid, record in zip(batch, encoded):
            if not torch.equal(record['source_token_ids'], source_ids[sid]):
                raise ValueError('Batched valid token IDs differ from full CPU token check: ' + sid)
            metadata = dict(sample_id=sid, partition='train' if sid in train else 'val',
                            kind='pure_text_h3_raw_tokens', text_only=True, image_conditioned=False,
                            prefix_length=0, tail_length=record['teacher_length'], teacher_length=record['teacher_length'],
                            prompt=prompts[sid], original_prompt=target_prompt(caps[sid]),
                            teacher_replacements=replacements[sid], caption_sha256=hashlib.sha256(prompts[sid].encode()).hexdigest(),
                            source_token_ids_sha256=raw_tensor_sha256(record['source_token_ids']),
                            token_ids_sha256=raw_tensor_sha256(record['source_token_ids']), tail_sha256=raw_tensor_sha256(record['tail']),
                            source_protocol_path=str(args.out / 'plan.json'), source_protocol_sha256=protocol_sha,
                            caps_sha256=caps_sha, split_sha256=split_sha,
                            model_weight_content_sha256=source['model_weight_content_sha256'],
                            tokenizer_metadata_file_sha256=tokenizer_shas, final_norm='Identity', normalization='none', pooling='none')
            path = args.out / (sid + '.pt')
            atomic_write_new(path, dict(metadata=metadata, **record), binary=True)
            rows.append(dict(sample_id=sid, path=str(path), file_sha256=sha256(path),
                             tail_length=record['teacher_length'], teacher_length=record['teacher_length'], prefix_length=0))
            print(json.dumps(dict(event='pure_text_target_cached', sample_id=sid, index=len(rows), expected=len(ids),
                                  tokens=record['teacher_length'])), flush=True)
    if ([row['sample_id'] for row in rows] != ids or set(path.stem for path in args.out.glob('*.pt')) != set(ids)
            or sha256(args.caps) != caps_sha or sha256(args.split) != split_sha
            or sha256(args.out / 'plan.json') != protocol_sha
            or any(sha256(processor / name) != digest for name, digest in tokenizer_shas.items())
            or any(sha256(row['path']) != row['file_sha256'] for row in rows)):
        raise RuntimeError('Final exact shard-ID/source/file integrity check failed')
    report = dict(complete=True, expected=len(ids), expected_ids=ids, rows=rows, source=source,
                  training_query_max=length_report['training_query_max'], validation_max_length=length_report['val_max'],
                  qmax=length_report['qmax'], train_max=length_report['train_max'], val_max=length_report['val_max'],
                  train_count=sum(sid in train for sid in ids), val_count=sum(sid in val for sid in ids),
                  source_protocol_sha256=protocol_sha, seconds=time.monotonic() - started,
                  exact_expected_full_shard_ids=True, actual_lm_forward=True, image_io=False)
    atomic_write_new(args.out / 'finished.json', report)
    print(json.dumps(dict(event='pure_text_finished', complete=True, expected=len(ids))), flush=True)


def audit_student_context_main():
    from music_text_condition import difference
    parser = argparse.ArgumentParser(description='Actual empty-prefix/context parity against cached teacher supervision')
    parser.add_argument('--audit_student_context', action='store_true', required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--targets', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--w', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new audit directory with existing parent')
    split = json.loads(args.split.read_text())
    names = split['train'][:3] + split['val'][:1]
    records = {}
    for directory in args.targets:
        finished = json.loads((directory / 'finished.json').read_text())
        if not finished['complete'] or finished['source']['split_sha256'] != sha256(args.split):
            raise ValueError('Target protocol/split mismatch')
        for row in finished['rows']:
            if row['sample_id'] in names:
                if sha256(row['path']) != row['file_sha256']:
                    raise ValueError('Teacher target changed')
                records[row['sample_id']] = torch.load(row['path'], map_location='cpu', weights_only=True)
    if set(records) != set(names):
        raise ValueError('Missing selected train/val teacher records')
    weights = sorted(glob.glob(str(args.w / 'text_encoder/model*.safetensors')))
    if not weights:
        raise FileNotFoundError(args.w / 'text_encoder')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cuda', onload_dtype=torch.bfloat16,
                onload_device='cuda', preparing_dtype=torch.bfloat16, preparing_device='cuda',
                computation_dtype=torch.bfloat16, computation_device='cuda')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=weights, **vram)],
        processor_config=ModelConfig(path=str(args.w / 'processor')))
    pipe.text_encoder.requires_grad_(False).eval()
    prompt_unit, reference_unit = MiniMaxH3Unit_PromptEmbedder(), MiniMaxH3Unit_ReferenceEncoder()
    args.out.mkdir()
    rows = []
    with torch.no_grad():
        for sid in names:
            target = records[sid]
            metadata = target['metadata']
            image_path = Path(metadata['image_path'])
            if sha256(image_path) != metadata['image_sha256']:
                raise ValueError('Training reference image changed')
            with Image.open(image_path) as image:
                width, height = reference_unit._resolve_reference_image_shape(pipe, *image.size, 2048)
                prepared = image.convert('RGB').resize((width, height), Image.LANCZOS)
            ref_blocks = [dict(kind='image', prepared_image=prepared, ref_audio_t=0),
                          dict(kind='audio', ref_audio_t=203)]
            input_ids, tags = prompt_unit.preprocess_ref_blocks(pipe, ' ', ref_blocks)[:2]
            lp = prefix_length(pipe.tokenizer, ' ', input_ids)
            if (lp != metadata['prefix_length'] or not torch.equal(input_ids[:lp].cpu(), target['token_ids'][:lp])
                    or not torch.equal(tags[:lp].cpu(), target['prefix_tags'])):
                raise ValueError('Student reference structure differs from cached teacher prefix')
            encoded = prompt_unit.process(pipe, ' ', ref_blocks=ref_blocks)
            context = encoded['prompt_embeds'][lp - 1].cpu().clone()
            diff = difference(context, target['prefix_context'])
            if diff['relative_rms'] > 0.002:
                raise ValueError('Empty student context differs from cached teacher: ' + sid)
            # Preserve only frozen structural evidence; no caption is read by the student forward.
            write_new(args.out / (sid + '.pt'), dict(prefix=encoded['prompt_embeds'][:lp].cpu(),
                       prefix_tags=encoded['text_token_tags'][:lp].cpu(), context=context,
                       cached_context=target['prefix_context'], sample_id=sid), binary=True)
            rows.append(dict(sample_id=sid, partition='train' if sid in split['train'] else 'val',
                             prefix_length=lp, context_difference=diff,
                             prefix_ids_tags_exact=True, student_semantic_prompt=' '))
            print(json.dumps(dict(event='student_context_audit', **rows[-1])), flush=True)
    report = dict(complete=True, rows=rows, selected_train=3, selected_validation=1,
                  actual_frozen_tc_encoding=True, student_reads_caption=False,
                  split_sha256=sha256(args.split), scope='Four selected references, not all2000 context numerical comparisons')
    write_new(args.out / 'finished.json', report)
    print(json.dumps(dict(event='student_context_audit_finished', **report)), flush=True)


def main():
    if '--pure_text' in sys.argv:
        return pure_text_main()
    contextual_dependencies()
    if '--audit_student_context' in sys.argv:
        return audit_student_context_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--neutral_teacher', action='store_true', help='Only normalize teacher gender pronouns, preserve original actions')
    parser.add_argument('--preflight', action='store_true', help='CPU-only image presence and token lengths; --out is a new JSON report')
    parser.add_argument('--max_items', type=int, default=0, help='Explicit smoke subset, never called the full split')
    parser.add_argument('--w', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshard or args.max_items < 0:
        parser.error('Invalid shard or subset size')
    if args.out.exists() or not args.out.parent.is_dir() or not args.image_root.is_dir():
        parser.error('Require a new output directory and existing output/image parents')
    caps = json.loads(args.caps.read_text())
    split = json.loads(args.split.read_text())
    train, val = split['train'], split['val']
    if len(train) != len(set(train)) or len(val) != len(set(val)) or set(train) & set(val):
        raise ValueError('Split duplicates or train/validation overlap')
    ids = train + val
    if args.max_items:
        ids = ids[:args.max_items]
    ids = ids[args.shard::args.nshard]
    for sid in ids:
        if Path(sid).name != sid or sid in ('.', '..'):
            raise ValueError('Invalid sample ID')
        target_prompt(caps[sid], args.neutral_teacher)
        if not (args.image_root / (sid + '.png')).is_file():
            raise FileNotFoundError(args.image_root / (sid + '.png'))
    if args.preflight:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.w / 'processor'), local_files_only=True)
        lengths = {sid: len(tokenizer(target_prompt(caps[sid], args.neutral_teacher), add_special_tokens=False)['input_ids']) for sid in ids}
        train_lengths = [lengths[sid] for sid in ids if sid in set(train)]
        report = dict(kind='CPU_preflight_not_teacher_encoding', complete=True,
                      caps_sha256=sha256(args.caps), split_sha256=sha256(args.split),
                      image_root=str(args.image_root), expected=len(ids), all_reference_images_present=True,
                      lengths=lengths, training_query_max=max(train_lengths),
                      training_mean_length=sum(train_lengths) / len(train_lengths),
                      all_lengths_max=max(lengths.values()), validation_does_not_set_query_max=True)
        write_new(args.out, report)
        print(json.dumps({key: value for key, value in report.items() if key != 'lengths'}), flush=True)
        return
    args.out.mkdir()
    source = dict(caps_sha256=sha256(args.caps), split_sha256=sha256(args.split),
                  image_root=str(args.image_root), original_full_train=len(train), original_full_val=len(val),
                  ids=ids, shard=args.shard, nshard=args.nshard, explicit_subset=args.max_items,
                  interface='raw_h3_last_hidden_5120_tail_replacement',
                  reference_image_short_edge=2048, structural_audio_label=True,
                  semantic_student_prompt=' ', final_norm='Identity', caption_fields=['dims.genre', 'motion_original'],
                  teacher_transform='gender_pronouns_to_dancer_v1' if args.neutral_teacher else 'none')
    weights = sorted(glob.glob(str(args.w / 'text_encoder/model*.safetensors')))
    if not weights:
        raise FileNotFoundError(args.w / 'text_encoder')
    # Fingerprint metadata explicitly; this is not mislabeled as a model content hash.
    source['model_files_metadata'] = [dict(path=path, size=Path(path).stat().st_size,
                                         mtime_ns=Path(path).stat().st_mtime_ns) for path in weights]
    write_new(args.out / 'plan.json', source)
    protocol_sha256 = sha256(args.out / 'plan.json')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cuda', onload_dtype=torch.bfloat16,
                onload_device='cuda', preparing_dtype=torch.bfloat16, preparing_device='cuda',
                computation_dtype=torch.bfloat16, computation_device='cuda')
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=weights, **vram)],
        processor_config=ModelConfig(path=str(args.w / 'processor')))
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    prompt_unit, reference_unit = MiniMaxH3Unit_PromptEmbedder(), MiniMaxH3Unit_ReferenceEncoder()
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for index, sid in enumerate(ids):
            image_path = args.image_root / (sid + '.png')
            before = image_path.stat()
            with Image.open(image_path) as image:
                w, h = reference_unit._resolve_reference_image_shape(pipe, *image.size, 2048)
                prepared = image.convert('RGB').resize((w, h), Image.LANCZOS)
            ref_blocks = [dict(kind='image', prepared_image=prepared, ref_audio_t=0),
                          dict(kind='audio', ref_audio_t=203)]
            prompt = target_prompt(caps[sid], args.neutral_teacher)
            ids_tensor, tags = prompt_unit.preprocess_ref_blocks(pipe, prompt, ref_blocks)[:2]
            lp = prefix_length(pipe.tokenizer, prompt, ids_tensor)
            encoded = prompt_unit.process(pipe, prompt, ref_blocks=ref_blocks)
            hidden = encoded['prompt_embeds']
            if hidden.shape != (len(ids_tensor), 5120) or not torch.isfinite(hidden).all():
                raise ValueError('Invalid native raw text sequence: ' + sid)
            if not torch.equal(encoded['text_token_tags'].cpu(), tags):
                raise ValueError('Presentation tags changed')
            tail, context = hidden[lp:].cpu().clone(), hidden[lp - 1].cpu().clone()
            image_sha = sha256(image_path)
            after = image_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Reference image changed: ' + sid)
            metadata = dict(sample_id=sid, image_path=str(image_path), image_sha256=image_sha,
                            prompt=prompt, original_prompt=target_prompt(caps[sid]),
                            teacher_replacements=neutralize_gender_pronouns(target_prompt(caps[sid]))[1] if args.neutral_teacher else [],
                            caption_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            prefix_length=lp, tail_length=len(tail), prepared_image_size=[w, h],
                            tail_sha256=tensor_sha256(tail), context_sha256=tensor_sha256(context),
                            source_protocol_path=str(args.out / 'plan.json'),
                            source_protocol_sha256=protocol_sha256)
            path = args.out / (sid + '.pt')
            write_new(path, dict(metadata=metadata, tail=tail, prefix_context=context,
                                 token_ids=ids_tensor.cpu(), prefix_tags=tags[:lp]), binary=True)
            rows.append(dict(sample_id=sid, path=str(path), file_sha256=sha256(path),
                             tail_length=len(tail), prefix_length=lp))
            print(json.dumps(dict(event='target_cached', sample_id=sid, index=index + 1,
                                  expected=len(ids), tokens=len(tail))), flush=True)
    train_rows = [row for row in rows if row['sample_id'] in set(train)]
    report = dict(complete=len(rows) == len(ids), expected=len(ids), rows=rows,
                  training_query_max=max((row['tail_length'] for row in train_rows), default=None),
                  train_count=len(train_rows), val_count=len(rows) - len(train_rows),
                  seconds=time.monotonic() - started, source=source)
    write_new(args.out / 'finished.json', report)
    print(json.dumps(dict(event='finished', complete=report['complete'], expected=len(ids))), flush=True)


if __name__ == '__main__':
    main()
