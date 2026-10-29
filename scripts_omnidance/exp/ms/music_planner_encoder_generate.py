"""Free-caption MOSS inference for native/frozen-MusicFM encoder experiments.

The complete style header and action text are model-generated. The shared
parser validates the unmodified output; no gold header or target is supplied.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch

from . import music_planner_caption_format as caption_format
from . import music_planner_moss_generate as moss


@torch.no_grad()
def generate_free(adapter, row, max_new_tokens=1024):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    classes, spec, _, identity = moss._protocol(adapter)
    runtime = moss._require_runtime(adapter)
    source = moss._audio_source(row, adapter.contract)
    batch, ids = moss._prefill(adapter, source, [])
    # _prefill accepts a zero-length forced header and preserves the native IDs.
    if batch['input_ids'].shape[1] + max_new_tokens > adapter.max_length:
        raise ValueError('Native prefix plus generation budget exceeds trained context; no truncation')
    result = dict(schema='music-planner-moss-free-caption-v1', family='moss',
        checkpoint_fingerprint=identity['fingerprint'], checkpoint_marker_sha256=identity['marker_sha256'],
        source_id=source['id'], audio_sha256=source['audio_sha256'], target_consumed=False,
        selection_mode='model-generated-header', forced_header_tokens=0,
        caption=None, caption_valid=False, validation_errors=[], raw_sequence_token_ids=None,
        raw_new_token_ids=None, raw_text=None, stop_reason='not_generated',
        max_new_tokens=max_new_tokens, actual_generation_forward_count=0,
        native_prefix_token_ids=ids[0].cpu().tolist(),
        musicfm_cache_identity=getattr(getattr(adapter, 'musicfm_store', None), 'identity', None),
        inference_base_identity=getattr(adapter, 'inference_base_identity', None),
        fusion_installation=getattr(adapter, 'fusion_metadata', None),
        decoding=dict(do_sample=False, num_beams=1, repetition_penalty=1.0,
                      no_repeat_ngram_size=0, encoder_repetition_penalty=1.0, use_cache=True))
    counts = [0]
    def observe(module, inputs, output):
        counts[0] += 1
    observer = runtime['model'].register_forward_hook(observe)
    validator = moss.position_patch.install_moss_generation_logits_validator(adapter)
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        parameter = next(adapter.model.parameters())
        autocast = (torch.autocast(parameter.device.type, dtype=torch.bfloat16)
                    if parameter.dtype == torch.bfloat16 else nullcontext())
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
            raise ValueError('Native MOSS must return one integer token sequence')
        offset = ids.shape[1]
        tokens = sequences[0, offset:].detach().cpu().tolist()
        result.update(raw_new_token_ids=tokens, raw_text=moss._decode(adapter.tokenizer, tokens), new_tokens=len(tokens))
        if not torch.equal(sequences[:, :offset].cpu(), ids.cpu()):
            raise ValueError('Native generation changed the exact input prefix')
        first_eos = next((i for i, token in enumerate(tokens) if token in runtime['eos_token_ids']), None)
        result['first_eos_index'] = first_eos
        result['stop_reason'] = ('eos' if first_eos is not None and first_eos == len(tokens) - 1 else
            'tokens_after_eos' if first_eos is not None else
            'max_new_tokens' if len(tokens) == max_new_tokens else 'unknown_stop')
        if not counts[0]:
            raise ValueError('No actual native generation forward observed')
        if not tokens or len(tokens) > max_new_tokens:
            raise ValueError('Empty or over-budget continuation')
        if first_eos != len(tokens) - 1 or tokens[-1] != runtime['assistant_terminator_id']:
            raise ValueError('Output must stop at its FIRST native assistant EOS')
        text = moss._decode(adapter.tokenizer, tokens[:-1])
        parsed = caption_format.parse_caption(text, classes, spec, source['audio_duration_seconds'])
        result.update(caption=text, caption_valid=True, parsed=parsed)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        result['validation_errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        observer.remove()
        validator.remove()
        adapter.model.train(was_training)
        result['actual_generation_forward_count'] = counts[0]
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--musicfm-cache', type=Path, help='Inference feature index; may include new audio with the SAME extractor identity')
    result.add_argument('--row', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--device', default='cpu')
    result.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    result.add_argument('--max-new-tokens', type=int, default=1024)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh output to preserve raw evidence')
    adapter = moss.load_conditioned_adapter(args.model, args.checkpoint, device=args.device,
        dtype=torch.bfloat16 if args.dtype == 'bf16' else torch.float32, musicfm_cache=args.musicfm_cache)
    row = json.loads(args.row.read_text(encoding='utf-8-sig'))
    result = generate_free(adapter, row, max_new_tokens=args.max_new_tokens)
    result['checkpoint_verification'] = getattr(adapter, 'conditioned_checkpoint_verification', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()), caption_valid=result['caption_valid'])))
    return 0 if result['caption_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
