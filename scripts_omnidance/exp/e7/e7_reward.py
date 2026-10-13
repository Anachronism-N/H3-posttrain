"""E7-v1 step 2: score rollout branches. reward = TextAlign(motion caption clauses) + identity hinge.
TA: Qwen3-VL-8B P(Yes) averaged over the full motion caption's clauses (same scorer as the eval, 8 frames).
ID: DINOv2 cosine to the reference image (mean over 8 frames); hinge penalty 2*min(0, id-0.80).
python e7_reward.py --gen <dir> --out <dir>/rewards_<shard>.jsonl --shard i --nshard n   (venv_eval, 1 GPU/shard)
"""
import argparse, csv, glob, json, os, re, sys

import decord
import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--gen", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--vlm", default="/dev/shm/od/vlm/Qwen3-VL-8B-Instruct")
p.add_argument("--contrast", action="store_true", help="v2: reward uses TA(own) - TA(other sample's motion caption)")
p.add_argument("--pool", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E7/pool_caps.csv")
a = p.parse_args()

from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16).cuda().eval()
vproc = AutoProcessor.from_pretrained(a.vlm)
vlm = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm, torch_dtype=torch.bfloat16).cuda().eval()
tok = vproc.tokenizer
YES = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")]
NO = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")]


@torch.no_grad()
def dino_feat(imgs):
    x = dproc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
    return torch.nn.functional.normalize(dino(pixel_values=x).pooler_output.float(), dim=-1)


@torch.no_grad()
def text_align(frames, caption):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if len(s.strip()) > 15][:8] or [caption[:300]]
    sc = []
    for c in sents:
        msgs = [{"role": "user", "content": [{"type": "video", "video": frames},
                {"type": "text", "text": f"Does the dancer's movement in this video match the following description? \"{c}\" Answer Yes or No."}]}]
        text = vproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = vproc(text=[text], videos=[frames], return_tensors="pt", do_resize=True).to("cuda")
        lg = vlm(**inp).logits[0, -1].float()
        sc.append(torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item())
    return float(np.mean(sc))


import csv as _csv
_csv.field_size_limit(10 ** 9)
_pool = list(_csv.DictReader(open(a.pool)))
_pid = {r["sample_id"]: i for i, r in enumerate(_pool)}


def other_caption(sid):
    return _pool[(_pid.get(sid, 0) + 97) % len(_pool)]["motion_caption"]


groups = []
for f in sorted(glob.glob(f"{a.gen}/groups_*.jsonl")):
    groups += [json.loads(l) for l in open(f)]
groups = groups[a.shard::a.nshard]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sid"] for l in open(a.out)}
fo = open(a.out, "a")
for g in groups:
    if g["sid"] in done:
        continue
    ref = Image.open(g["image_path"]).convert("RGB")
    res = []
    for sp in g["specs"]:
        vr = decord.VideoReader(f"{a.gen}/{g['sid']}_k{sp['k']}.mp4")
        idx = np.linspace(0, len(vr) - 1, 8).astype(int)
        fr = vr.get_batch(list(idx)).asnumpy()
        H, W = fr.shape[1:3]
        pil = [Image.fromarray(x) for x in fr]
        small = [im.resize((W * 448 // H // 28 * 28 or 224, 448)) for im in pil]
        fe = dino_feat([ref.resize((W, H))] + pil)
        idd = float((fe[1:] @ fe[0]).mean())
        ta = text_align(small, g["motion_caption"])
        if a.contrast:
            ta_o = text_align(small, other_caption(g["sid"]))
            res.append(dict(k=sp["k"], ta=ta, ta_other=ta_o, id=idd, reward=(ta - ta_o) + 2.0 * min(0.0, idd - 0.80)))
        else:
            res.append(dict(k=sp["k"], ta=ta, id=idd, reward=ta + 2.0 * min(0.0, idd - 0.80)))
    fo.write(json.dumps(dict(sid=g["sid"], round=g["round"], res=res)) + "\n")
    fo.flush()
    print(f"[e7rew] {g['sid']} {[round(r['reward'], 3) for r in res]}", flush=True)
print("[e7rew] done", flush=True)
