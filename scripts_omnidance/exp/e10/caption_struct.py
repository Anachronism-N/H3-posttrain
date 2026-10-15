"""caption_struct.py (10-15): structured dance caption for the music->H3-text encoder target.
Qwen3-VL-8B watches the clip (16 frames over the first 124/24 s, same window H3 trains on) and outputs JSON:
{"genre": one of GENRES, "segments": [{"t": "0-1.7s", "intensity": "calm|medium|energetic", "moves": "<= 12 words"} x3],
 "overall": "calm|medium|energetic"}
The three segments are the same 1.7 s thirds used by music_caption.py "seg". Free-form fields are kept short so that the
resulting caption text matches the template-body format used at inference.
python caption_struct.py --csv <csv sample_id,video> --out <out.jsonl> [--shard i --nshard n]
"""
import argparse, csv, json, os, re
import numpy as np, torch, decord
from PIL import Image

GENRES = ["hip-hop", "k-pop", "jazz", "contemporary", "chinese classical", "latin", "ballet", "freestyle/other"]
p = argparse.ArgumentParser()
p.add_argument("--csv", required=True); p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--vlm", default="/dev/shm/od/vlm/Qwen3-VL-8B-Instruct")
p.add_argument("--dur", type=float, default=124 / 24)
a = p.parse_args()
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(a.csv)))[a.shard::a.nshard]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sample_id"] for l in open(a.out) if l.strip()}
rows = [r for r in rows if r["sample_id"] not in done]
print(f"[cap] shard {a.shard}: {len(rows)} todo", flush=True)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
proc = AutoProcessor.from_pretrained(a.vlm)
vlm = Qwen3VLForConditionalGeneration.from_pretrained(a.vlm, dtype=torch.bfloat16).cuda().eval()
e = a.dur / 3
INSTR = (
    f"Watch this dance clip. It is split into three equal parts: part 1 = 0-{e:.1f}s, part 2 = {e:.1f}-{2 * e:.1f}s, "
    f"part 3 = {2 * e:.1f}-{a.dur:.1f}s. Answer with ONE JSON object and nothing else, with keys:\n"
    f'"genre": the dance style, exactly one of {json.dumps(GENRES)};\n'
    '"segments": a list of 3 objects, one per part, each {"intensity": "calm" or "medium" or "energetic" (how big and fast '
    'the dancer\'s movements are in that part), "moves": at most 12 words naming the main body movements seen in that part};\n'
    '"overall": "calm" or "medium" or "energetic" for the whole clip.\n'
    "Judge intensity only from the dancer's body motion (amplitude, speed, jumps, kicks), not from clothing, camera or music."
)
fo = open(a.out, "a")
for r in rows:
    try:
        vr = decord.VideoReader(r["video"]); fps = float(vr.get_avg_fps()) or 24.0
        n = min(len(vr), int(round(a.dur * fps))); idx = np.linspace(0, n - 1, 16).astype(int)
        fr = vr.get_batch(list(idx)).asnumpy(); H, W = fr.shape[1:3]
        frames = [Image.fromarray(f).resize((max(28, W * 448 // H // 28 * 28), 448)) for f in fr]
        msgs = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": INSTR}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], videos=[frames], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=260, do_sample=False)
        raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        m = re.search(r"\{.*\}", raw, re.S); d = json.loads(m.group(0)) if m else None
        ok = (isinstance(d, dict) and d.get("genre") in GENRES and isinstance(d.get("segments"), list) and len(d["segments"]) == 3
              and all(s.get("intensity") in ("calm", "medium", "energetic") for s in d["segments"])
              and d.get("overall") in ("calm", "medium", "energetic"))
        fo.write(json.dumps({"sample_id": r["sample_id"], "ok": bool(ok), "cap": d if ok else None, "raw": None if ok else raw}) + "\n"); fo.flush()
        print(r["sample_id"], "ok" if ok else "BAD", json.dumps(d)[:200] if ok else raw[:120], flush=True)
    except Exception as ex:  # unreadable video: record and continue
        fo.write(json.dumps({"sample_id": r["sample_id"], "ok": False, "cap": None, "raw": repr(ex)[:200]}) + "\n"); fo.flush()
