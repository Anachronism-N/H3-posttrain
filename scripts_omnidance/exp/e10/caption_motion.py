"""E10: rewrite captions into motion-centric atomic sentences with Qwen3-VL-8B (watching the video).
python caption_motion.py --csv <csv with sample_id,video> --out <out.csv> [--shard i --nshard n]
"""
import argparse, csv, os, re
import numpy as np, torch, decord
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--vlm", default="/dev/shm/od/vlm/Qwen3-VL-8B-Instruct")
p.add_argument("--max_dur", type=float, default=81 / 16)
a = p.parse_args()
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.csv)))[a.shard::a.nshard]
done = set()
if os.path.exists(a.out):
    done = {r["sample_id"] for r in csv.DictReader(open(a.out))}
rows = [r for r in rows if r["sample_id"] not in done]
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
proc = AutoProcessor.from_pretrained(a.vlm)
vlm = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm, dtype=torch.bfloat16).cuda().eval()
INSTR = (
    "Watch this dance clip carefully and describe ONLY what the dancer's body actually does, in temporal order. "
    "Write 5 to 7 short sentences. Each sentence states one specific movement you can see: name the body part(s), "
    "the direction and the manner of the movement, and when it happens (beginning, middle or end of the clip). "
    "Be faithful to this particular clip; do not use generic or stock phrases, and do not mention anything you cannot see. "
    "Do NOT describe clothing, appearance, background, camera, emotion, style or quality. Output the sentences only."
)
new = not os.path.exists(a.out)
fo = open(a.out, "a", newline="")
w = csv.DictWriter(fo, fieldnames=["sample_id", "motion_caption"])
if new:
    w.writeheader()
for r in rows:
    vr = decord.VideoReader(r["video"])
    fps = float(vr.get_avg_fps()) or 16.0
    n = min(len(vr), int(round(a.max_dur * fps)))
    idx = np.linspace(0, n - 1, 16).astype(int)
    fr = vr.get_batch(list(idx)).asnumpy()
    H, W = fr.shape[1:3]
    frames = [Image.fromarray(f).resize((max(28, W * 448 // H // 28 * 28), 448)) for f in fr]
    msgs = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": INSTR}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], videos=[frames], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=220, do_sample=False)
    cap = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    cap = re.sub(r"\s+", " ", cap)
    w.writerow(dict(sample_id=r["sample_id"], motion_caption=cap)); fo.flush()
    print(r["sample_id"], cap[:160], flush=True)
