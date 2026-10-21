"""冻结 H3 文本 LM（5120d, final norm=Identity），提供纯文本 raw token helper。
原有 CLI 仍将 caption 正文的 last_hidden_state mean-pool，用于属性评测。
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

U = '/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930'


def raw_tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    # Hash in bounded views rather than allocating a second full embedding tensor.
    data = memoryview(value.view(torch.uint8).numpy()).cast('B')
    for offset in range(0, len(data), 4 * 1024 * 1024):
        digest.update(data[offset:offset + 4 * 1024 * 1024])
    return digest.hexdigest()


def load_pure_text_tokenizer(weights_root):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(weights_root) / 'processor'), local_files_only=True)
    tokenizer.padding_side = 'right'
    return tokenizer


def load_pure_text_lm(weights_root, device='cuda', fingerprint=True):
    """Load only the retained native H3 LM tensors, without materializing vision weights."""
    from safetensors import safe_open
    sys.path.insert(0, U)
    from diffsynth.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
    paths = sorted((Path(weights_root) / 'text_encoder').glob('model*.safetensors'))
    if not paths:
        raise FileNotFoundError(Path(weights_root) / 'text_encoder')
    with torch.device('meta'):
        encoder = MiniMaxH3TextEncoder()
    lm = encoder.model.language_model
    if not isinstance(lm.norm, torch.nn.Identity):
        raise ValueError('Native H3 final norm must be Identity')
    expected = set(lm.state_dict())
    prefix = 'model.language_model.'
    state, skipped, tensor_hashes = {}, [], {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as checkpoint:
            for name in checkpoint.keys():
                if not name.startswith(prefix):
                    continue
                key = name[len(prefix):]
                if key not in expected:
                    skipped.append(name)
                    continue
                if key in state:
                    raise ValueError('Duplicate LM tensor: ' + name)
                tensor = checkpoint.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError('Expected native BF16 LM weights: ' + name)
                state[key] = tensor
                if fingerprint:
                    tensor_hashes[key] = raw_tensor_sha256(tensor)
    lm.load_state_dict(state, strict=True, assign=True)
    del state, encoder
    lm.rotary_emb = type(lm.rotary_emb)(config=lm.config, device='cpu')
    if any(value.is_meta for value in list(lm.parameters()) + list(lm.buffers())):
        raise ValueError('Unmaterialized LM parameters or rotary buffers')
    # The content digest describes exactly the state assigned above, not file metadata.
    content_hash = hashlib.sha256(json.dumps(tensor_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest() if fingerprint else None
    provenance = dict(model_weight_content_sha256=content_hash,
                      model_weight_tensor_sha256=tensor_hashes,
                      model_weight_hash_scope='actual_loaded_H3_50_layer_language_model_state_BF16',
                      loaded_tensor_count=len(expected), skipped_language_model_tensor_names=sorted(skipped),
                      final_norm='Identity', retained_layers=lm.config.num_hidden_layers,
                      strict_assign=True, rotary_buffer_rebuilt=True)
    return lm.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval(), provenance


def pure_text_token_batch(lm, tokenizer, texts, device='cuda'):
    """Return variable-length raw hidden rows and their exact non-padding source IDs."""
    if tokenizer.padding_side != 'right' or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Require nonempty texts and right padding')
    tokens = tokenizer(texts, add_special_tokens=False, truncation=False,
                       padding=True, return_tensors='pt')
    ids, mask = tokens['input_ids'], tokens['attention_mask']
    if ids.ndim != 2 or mask.shape != ids.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Invalid token IDs or attention mask')
    if (mask.sum(1) == 0).any() or (mask[:, 1:] > mask[:, :-1]).any():
        raise ValueError('Empty or non-right-padded token sequence')
    with torch.inference_mode():
        hidden = lm(input_ids=ids.to(device), attention_mask=mask.to(device),
                    use_cache=False).last_hidden_state
    if hidden.shape != (*ids.shape, 5120) or not torch.isfinite(hidden).all():
        raise ValueError('Expected finite raw H3 [batch,tokens,5120] last_hidden_state')
    if hidden.dtype != torch.bfloat16:
        raise ValueError('Expected native BF16 raw hidden tokens')
    rows = []
    for index in range(len(texts)):
        valid = mask[index].bool()
        tail = hidden[index, valid.to(hidden.device)].detach().cpu().clone()
        source_ids = ids[index, valid].detach().cpu().clone()
        rows.append(dict(tail=tail, source_token_ids=source_ids, teacher_length=len(source_ids)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--caps')
    source.add_argument('--ms_dir', help='Original MotionScript per-clip JSON directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--w', default='/dev/shm/h3/Ref2VA')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--key', default='caption')
    ap.add_argument('--dance_only', action='store_true', help='Align dance genre and original motion, excluding music BPM descriptions')
    args = ap.parse_args()
    if args.ms_dir:
        captions = {}
        for path in sorted(glob.glob(os.path.join(args.ms_dir, '*.json'))):
            item = json.loads(Path(path).read_text())
            if item.get('ms'):
                text = ' '.join(sentence.strip() for sentence in item['ms'] if sentence.strip())
                if text:
                    captions[item['sid']] = {args.key: text}
    else:
        captions = json.loads(Path(args.caps).read_text())
    sids = sorted(captions)[args.shard::args.nshard]
    tokenizer = load_pure_text_tokenizer(args.w)
    lm, _ = load_pure_text_lm(args.w, fingerprint=False)
    output = {}
    with torch.inference_mode():
        for index in range(0, len(sids), args.bs):
            batch = sids[index:index + args.bs]
            texts = ([f"Dance style: {captions[sid]['dims']['genre']}. " + captions[sid]['motion_original'] for sid in batch]
                     if args.dance_only else [captions[sid][args.key] for sid in batch])
            tokens = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
            hidden = lm(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], use_cache=False).last_hidden_state.float()
            mask = tokens['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            for sid, value in zip(batch, pooled.cpu()):
                output[sid] = value.half()
            if index % (args.bs * 50) == 0:
                print(index, len(sids), flush=True)
    torch.save(output, args.out)
    print('DONE', len(output), flush=True)


if __name__ == '__main__':
    main()
