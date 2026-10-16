"""冻结 H3 文本编码器（Qwen3-VL LM, 5120d, final norm=Identity，与 H3 推理一致）对 caption 编码。
caption 正文单独编码（不套 H3 模板，模板对所有样本相同、只会稀释差异），取 last_hidden_state 做 mean-pool。
--key 可选 caption 以外的字段（如单维度短语），供属性评测使用。
用法: python text_embed.py --caps captions_v1.json --out text_emb_v1.pt [--shard i --nshard n]
"""
import argparse, glob, json, os, sys
import torch
from safetensors.torch import load_file

U = '/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930'
sys.path.insert(0, U)
from diffsynth.models.minimax_h3_text_encoder import MiniMaxH3TextEncoder  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
source = ap.add_mutually_exclusive_group(required=True)
source.add_argument('--caps'); source.add_argument('--ms_dir', help='Original MotionScript per-clip JSON directory')
ap.add_argument('--out', required=True)
ap.add_argument('--w', default='/dev/shm/h3/Ref2VA'); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--key', default='caption')
a = ap.parse_args()
if a.ms_dir:
    C = {}
    for path in sorted(glob.glob(os.path.join(a.ms_dir, '*.json'))):
        item = json.load(open(path))
        if item.get('ms'):
            text = ' '.join(sentence.strip() for sentence in item['ms'] if sentence.strip())
            if text:
                C[item['sid']] = {a.key: text}
else:
    C = json.load(open(a.caps))
sids = sorted(C)[a.shard::a.nshard]
tok = AutoTokenizer.from_pretrained(f'{a.w}/processor')
with torch.device('meta'):
    enc = MiniMaxH3TextEncoder()
sd = {}
for f in sorted(glob.glob(f'{a.w}/text_encoder/model*.safetensors')):
    sd.update({k: v for k, v in load_file(f).items() if k.startswith('model.language_model.')})
lm = enc.model.language_model
res = lm.load_state_dict({k[len('model.language_model.'):]: v for k, v in sd.items()}, strict=False, assign=True)
assert not res.missing_keys, res.missing_keys[:5]
del sd
lm.rotary_emb = type(lm.rotary_emb)(config=lm.config)  # meta 上建的非持久 buffer(inv_freq) 需重建
assert not [n for n, b in lm.named_buffers() if b.is_meta] and not [n for n, p_ in lm.named_parameters() if p_.is_meta]
lm = lm.to('cuda', torch.bfloat16).eval()
out = {}
with torch.inference_mode():
    for i in range(0, len(sids), a.bs):
        b = sids[i:i + a.bs]
        texts = [C[s][a.key] for s in b]
        t = tok(texts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
        h = lm(input_ids=t['input_ids'], attention_mask=t['attention_mask'], use_cache=False).last_hidden_state.float()
        m = t['attention_mask'].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
        for s, v in zip(b, e.cpu()):
            out[s] = v.half()
        if i % (a.bs * 50) == 0: print(i, len(sids), flush=True)
torch.save(out, a.out)
print('DONE', len(out), flush=True)
