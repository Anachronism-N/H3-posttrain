"""Cache original contextual text-token targets without video/audio VAE computation."""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
sys.path.insert(0, str(ROOT / 'DiffSynth-Studio-upstream-20260930'))
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PromptEmbedder,
    MiniMaxH3Unit_ReferenceEncoder,
)
from music_text_condition import prefix_length, tensor_sha256, write_new


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_prompt(entry):
    text = entry['motion_original']
    genre = entry['dims']['genre']
    if not text.strip() or not genre.strip() or genre == 'generation_holdout':
        raise ValueError('Target requires original motion and a declared dance genre')
    return f'Dance style: {genre}. ' + text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caps', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
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
        target_prompt(caps[sid])
        if not (args.image_root / (sid + '.png')).is_file():
            raise FileNotFoundError(args.image_root / (sid + '.png'))
    if args.preflight:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.w / 'processor'), local_files_only=True)
        lengths = {sid: len(tokenizer(target_prompt(caps[sid]), add_special_tokens=False)['input_ids']) for sid in ids}
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
                  semantic_student_prompt=' ', final_norm='Identity', caption_fields=['dims.genre', 'motion_original'])
    weights = sorted(glob.glob(str(args.w / 'text_encoder/model*.safetensors')))
    if not weights:
        raise FileNotFoundError(args.w / 'text_encoder')
    # Fingerprint metadata explicitly; this is not mislabeled as a model content hash.
    source['model_files_metadata'] = [dict(path=path, size=Path(path).stat().st_size,
                                         mtime_ns=Path(path).stat().st_mtime_ns) for path in weights]
    write_new(args.out / 'plan.json', source)
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
            prompt = target_prompt(caps[sid])
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
                            prompt=prompt, caption_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            prefix_length=lp, tail_length=len(tail), prepared_image_size=[w, h],
                            tail_sha256=tensor_sha256(tail), context_sha256=tensor_sha256(context),
                            source_protocol=source)
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
